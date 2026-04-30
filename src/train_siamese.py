import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import sys
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Calculate absolute path to the project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
import importlib

import utils
import Data_handler
import architectures
import model_interpertation

importlib.reload(utils)
importlib.reload(Data_handler)
importlib.reload(architectures)
importlib.reload(model_interpertation)

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from utils import load_config
from Data_handler import prepare_agn_data, SyntheticSiameseDataset
from architectures import SpectraNet, BinaryFocalLossWithLogits, SiameseSpectraNet
from model_interpertation import run_gradcam_analysis, plot_attention_map, plot_transformer_attention

import torch.nn.functional as F


def morphological_continuum_subtraction(x, window_size=151):
    # (Use the exact same function from your original train.py to ensure consistency)
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
    x_flattened = x - continuum
    mean = x_flattened.mean(dim=-1, keepdim=True)
    std = x_flattened.std(dim=-1, keepdim=True)
    return (x_flattened - mean) / (std + 1e-8)

def train_one_epoch_siamese(model, dataloader, criterion, optimizer, device, threshold):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch_x1, batch_x2, batch_y in dataloader:
        batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)

    

        # 2. Forward pass through the Siamese Network
        logits = model(batch_x1, batch_x2)
        
        # 3. Calculate Focal Loss
        loss = criterion(logits, batch_y)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 4. Track metrics (using standard 0.5 threshold for training monitoring)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(batch_y.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate_one_epoch_siamese(model, dataloader, criterion, device,threshold):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in dataloader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)

            logits = model(batch_x1, batch_x2)
            loss = criterion(logits, batch_y)

            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1



def train_siamese():

    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_dir = os.path.join(BASE_DIR, 'models/siamese_network/')
    os.makedirs(models_dir, exist_ok=True)
    threshold = float(config['siamese_training']['decision_threshold'])
    
    # 2. Data
    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df_clean = df_clean.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])

    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df_clean.columns if c not in meta_cols]

    df_train, df_temp = train_test_split(df_clean, test_size=0.3, stratify=df_clean['agn_type'], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['agn_type'], random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 3. Create Datasets
    train_dataset = SyntheticSiameseDataset(df_train, flux_cols, epoch_size=10000)
    val_dataset = SyntheticSiameseDataset(df_val, flux_cols, epoch_size=2000)
    test_dataset = SyntheticSiameseDataset(df_test, flux_cols, epoch_size=2000)

    # 4. Loaders
    train_loader = DataLoader(train_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False)


    print("Loading Pretrained SpectraNet Backbone...")
    base_model = SpectraNet(config)
    base_model_path = os.path.join(BASE_DIR, 'models/working_classifier_backbone/', 'best_spectranet_working_model.pth')
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    
    model = SiameseSpectraNet(base_model, freeze_backbone=True).to(device)


    criterion = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0,reduction='mean')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=float(config['siamese_training']['learning_rate']), 
                            weight_decay=float(config['siamese_training']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
 
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_val_loss = float('inf')
    model_save_path = os.path.join(models_dir, 'best_siamese_net.pth')


    for epoch in range(config['siamese_training']['num_epochs']):

        t_loss,t_f1 = train_one_epoch_siamese(model, train_loader, criterion, optimizer, device,threshold)
        v_loss,v_f1 = validate_one_epoch_siamese(model, val_loader, criterion, device,threshold)

        # Track history
        history['train_loss'].append(t_loss)
        history['train_f1'].append(t_f1)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(v_f1)

        print(f"Epoch {epoch+1}/{config['siamese_training']['num_epochs']}: Train Loss {t_loss:.4f}, Train F1 {t_f1:.4f}, Val Loss {v_loss:.4f}, Val F1 {v_f1:.4f}")


        scheduler.step(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[Epoch {epoch+1}] New best model saved!")

    # ======================== POST-TRAINING VISUALIZATION ========================
    print("\n=== Generating post-training plots ===")

    # ---------- 1. Loss & F1 Macro Score vs Epochs ----------
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss subplot
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_title('Siamese Network — Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # F1 subplot
    ax2.plot(epochs_range, history['train_f1'], label='Train F1 (macro)', linewidth=2)
    ax2.plot(epochs_range, history['val_f1'], label='Val F1 (macro)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('F1 Macro', fontsize=13)
    ax2.set_title('Siamese Network — F1 Macro vs Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    loss_f1_path = os.path.join(models_dir, 'siamese_loss_f1_curves.png')
    plt.savefig(loss_f1_path, dpi=300)
    print(f"Saved loss/F1 curves to {loss_f1_path}")
    plt.show()

    # ---------- 2. Confusion Matrix on Test Set ----------
    print("Evaluating on test set for confusion matrix...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in test_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            logits = model(batch_x1, batch_x2)
            probs = torch.sigmoid(logits)
            preds = (probs >= config['siamese_training']['decision_threshold']).float()
            all_test_preds.extend(preds.cpu().numpy().flatten())
            all_test_targets.extend(batch_y.cpu().numpy().flatten())

    cm = confusion_matrix(all_test_targets, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Static (0)', 'Change (1)'])
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Siamese Network — Test Confusion Matrix', fontsize=14, fontweight='bold')
    cm_path = os.path.join(models_dir, 'siamese_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"Saved confusion matrix to {cm_path}")
    plt.show()

    test_f1 = f1_score(all_test_targets, all_test_preds, average='macro')
    print(f"Test F1 (macro): {test_f1:.4f}")

    # ---------- 3. Attention Map on a Random Test Pair ----------
    print("Plotting attention map on a random test pair...")

    # Get the backbone model for attention map visualization
    backbone = model.feature_extractor  # this is the pretrained SpectraNet backbone's feature_extractor
    base_model_for_attn = SpectraNet(config).to(device)
    base_model_for_attn.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model_for_attn.eval()

    # Grab wavelengths from flux columns
    wavelengths = np.array(flux_cols, dtype=float)

    # Pick a random pair from the test dataset
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample_x1, sample_x2, sample_label = test_dataset[rand_idx]
    sample_x1 = sample_x1.to(device)  # Shape: [1, 1024]
    sample_x2 = sample_x2.to(device)  # Shape: [1, 1024]
    pair_label = int(sample_label.item())
    
    # Get the siamese prediction for this pair
    with torch.no_grad():
        pair_logits = model(sample_x1, sample_x2)
        pair_prob = torch.sigmoid(pair_logits).item()
        pair_pred = 1 if pair_prob >= config['siamese_training']['decision_threshold'] else 0

    # Plot attention maps for both spectra in the pair (stacked vertically)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, spec_proc, spec_label_str in [
        (ax_top, sample_x1_proc, 'Spectrum 1'),
        (ax_bot, sample_x2_proc, 'Spectrum 2')
    ]:
        # Run GradCAM on the backbone to get the attention/importance map
        from model_interpertation import GradCAM
        target_layer = base_model_for_attn.feature_extractor[2]
        grad_cam = GradCAM(base_model_for_attn, target_layer)

        # Use target_class=1 (Type 2) to highlight regions the backbone cares about
        cam, pred_prob = grad_cam(spec_proc, target_class=1)

        spec_array = spec_proc.squeeze().cpu().numpy()

        # Plot the spectrum
        ax.plot(wavelengths, spec_array, color='black', linewidth=1.2, label='Spectrum')
        # Overlay Grad-CAM heatmap
        sc = ax.scatter(wavelengths, spec_array, c=cam, cmap='jet', alpha=0.6, s=15, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Grad-CAM Importance', rotation=270, labelpad=15)

        ax.set_ylabel('Normalized Flux', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_title(f'{spec_label_str} — Backbone Pred Prob: {pred_prob:.3f}', fontsize=13, fontweight='bold')

    ax_bot.set_xlabel('Rest Wavelength (Å)', fontsize=12)

    pred_str = "Change" if pair_pred == 1 else "Static"
    true_str = "Change" if pair_label == 1 else "Static"
    fig.suptitle(
        f'Siamese Pair Attention Map — Pred: {pred_str} (prob={pair_prob:.3f}) | True: {true_str}',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    attn_path = os.path.join(models_dir, 'siamese_pair_attention_map.png')
    plt.savefig(attn_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention map to {attn_path}")
    plt.show()

    print("\n=== All post-training visualizations complete ===")

if __name__ == "__main__":
    train_siamese()