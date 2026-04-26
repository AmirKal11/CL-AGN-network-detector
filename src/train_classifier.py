import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

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
from Data_handler import prepare_agn_data
from architectures import SpectraNet, BinaryFocalLossWithLogits
from model_interpertation import run_signed_gradcam_subplot_analysis, plot_attention_map, plot_transformer_attention

import torch.nn.functional as F


def morphological_continuum_subtraction(x, window_size=151, clip_max=4.0, taper_len=5):
    """
    Acts as a lightweight spectral decomposition by estimating and subtracting 
    the continuum envelope using morphological pooling.
    x: [Batch, 1, Sequence_Length]
    """
    # 1. Pad the sequence to handle edge artifacts smoothly
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    
    # 2. Extract the continuum envelope using a wide average pool
    # This acts as a low-pass filter, ignoring sharp narrow lines and following the slope
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
    
    # 3. Subtract the continuum from the original flux
    x_flattened = x - continuum
    
    # 4. Standardize the result to ensure numeric stability for the CNN
    mean = x_flattened.mean(dim=-1, keepdim=True)
    std = x_flattened.std(dim=-1, keepdim=True)
    x_normalized = (x_flattened - mean) / (std + 1e-8)
    
    # 5. DECAPITATE NARROW LINES (The Grad-CAM Fix)
    # This prevents the network from using the [O III] / Narrow Balmer ratio to cheat
    x_clipped = torch.clamp(x_normalized, min=-10.0, max=clip_max)
    
    # 6. EDGE TAPERING (The Limits Fix)
    # Surgically fade the first and last 5 pixels to destroy reflection artifacts
    seq_len = x.shape[-1]
    taper = torch.ones(seq_len, device=x.device)
    
    # Create a linear fade from 0 to 1
    fade = torch.linspace(0, 1, taper_len, device=x.device)
    
    # Apply to left and right edges
    taper[:taper_len] = fade
    taper[-taper_len:] = torch.flip(fade, dims=[0])
    
    # Reshape so it broadcasts perfectly over (Batch, Channels, SeqLen)
    taper = taper.view(1, 1, -1)
    
    # Apply the taper to the clipped tensor
    x_final = x_clipped * taper

    return x_final


def extract_spectral_derivative(x):
    """
    Computes the first derivative of the spectrum to remove global continuum trends.
    x: [Batch, 1, 1024]
    """
    # Calculate differences between adjacent pixels
    diff = torch.diff(x, dim=-1)
    
    # Pad the left edge by 1 to maintain the 1024 sequence length
    # Using 'replicate' preserves the boundary value without introducing artificial spikes
    diff_compressed = torch.sign(diff) * torch.log1p(torch.abs(diff))
    
    return diff_compressed


def train_one_epoch(model, dataloader, criterion_class, criterion_redshift, optimizer, device, alpha,lambda_z = 0.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch_X, batch_y, batch_z in dataloader:
        batch_X, batch_y, batch_z = batch_X.to(device), batch_y.to(device), batch_z.to(device)
        optimizer.zero_grad()
        
        # Forward pass returns TWO outputs
        class_outputs, redshift_outputs = model(batch_X, alpha=alpha)
        
        batch_y = batch_y.view_as(class_outputs)
        batch_z = batch_z.view_as(redshift_outputs)
        
        loss_class = criterion_class(class_outputs, batch_y)
        loss_redshift = criterion_redshift(redshift_outputs, batch_z)
        
        total_loss = loss_class + lambda_z * loss_redshift
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * batch_X.size(0)
        
        probs = torch.sigmoid(class_outputs)
        preds = (probs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
    return running_loss / len(dataloader.dataset), f1_score(all_labels, all_preds, average='macro')

def validate_one_epoch(model, dataloader, criterion_class, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y, _ in dataloader: # Note the underscore for z
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            class_outputs, _ = model(batch_X, alpha=0.0) 
            
            batch_y = batch_y.view_as(class_outputs)
            loss = criterion_class(class_outputs, batch_y)
            
            running_loss += loss.item() * batch_X.size(0)
            probs = torch.sigmoid(class_outputs)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            
    return running_loss / len(dataloader.dataset), f1_score(all_labels, all_preds, average='macro')

def train_model():
    # 1. Setup
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Data
    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df_clean = df_clean.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])
    
    if config['training'].get('redshift_overlap', False):
        df_clean = df_clean[(df_clean['z'] > 0.15) & (df_clean['z'] < 0.22)].copy()
        print('Using redshift overlap data')
    else:
        print('Using full data')

    train_loader, val_loader, test_loader, pos_weight = prepare_agn_data(df_clean, batch_size=config['training']['batch_size'])
    
    # 3. Model & Optimizer
    model = SpectraNet(config).to(device)
    criterion_class = BinaryFocalLossWithLogits(alpha=0.84, gamma=2.0)
    criterion_redshift = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 4. Training
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_val_loss = float('inf')
    model_save_path = os.path.join(models_dir, 'best_spectranet.pth')

    for epoch in range(config['training']['num_epochs']):
        p = epoch / config['training']['num_epochs']
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 # DANN schedule
        
        warmup_epochs = 5
        max_lambda = 0.0 # Your target adversarial weight
        
        if epoch < warmup_epochs:
            current_lambda_z = 0.0
        else:
            current_lambda_z = max_lambda * alpha

        t_loss, t_f1 = train_one_epoch(model, train_loader, criterion_class, criterion_redshift, optimizer, device, alpha,current_lambda_z)
        v_loss, v_f1 = validate_one_epoch(model, val_loader, criterion_class, device)
        
        scheduler.step(v_loss)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)
            
        history['train_loss'].append(t_loss); history['train_f1'].append(t_f1)
        history['val_loss'].append(v_loss); history['val_f1'].append(v_f1)
        print(f"Epoch {epoch+1} | Val F1: {v_f1:.4f} | Alpha: {alpha:.2f}")

    # 5. Plotting (Enhanced)
    epochs_range = range(1, config['training']['num_epochs'] + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='#1f77b4', lw=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='--', lw=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # F1 Score Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_f1'], label='Training F1', color='#2ca02c', lw=2)
    plt.plot(epochs_range, history['val_f1'], label='Validation F1', color='#d62728', linestyle='--', lw=2)
    plt.title('Training and Validation F1 Score', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Macro F1 Score', fontsize=12)
    plt.ylim(0, 1) # F1 is always between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'training_history.png'), dpi=300)
    plt.show()
    plt.close()

    # 6. Evaluation
    print("\n=== Evaluating on Test Data ===")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    all_test_preds, all_test_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y, _ in test_loader:
            batch_X = batch_X.to(device)
            
            # 1. APPLY THE FILTER DURING EVALUATION
            #batch_X = morphological_continuum_subtraction(batch_X)
            
            outputs, _ = model(batch_X, alpha=0.0) 
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_test_preds.extend(preds.cpu().numpy().flatten())
            all_test_labels.extend(batch_y.numpy().flatten())
            
    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T1', 'T2'])
    disp.plot(cmap='Blues'); plt.savefig(os.path.join(models_dir, 'cm.png')); plt.show()

    # 7. Attention Visualization
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    wavelengths = np.array([c for c in df_clean.columns if c not in meta_cols]).astype(float)
    
    spec1, spec2 = None, None
    for batch_X, batch_y, _ in test_loader: 
        batch_X = batch_X.to(device)
        
        # 2. APPLY THE FILTER FOR VISUALIZATION
        #batch_X = morphological_continuum_subtraction(batch_X)
        
        for i in range(len(batch_y)):
            if batch_y[i] == 0 and spec1 is None: 
                spec1 = batch_X[i:i+1] # Now contains flattened flux
            if batch_y[i] == 1 and spec2 is None: 
                spec2 = batch_X[i:i+1] # Now contains flattened flux
                
        if spec1 is not None and spec2 is not None: break
            
    plot_transformer_attention(model, spec1, spec2, wavelengths)
    run_signed_gradcam_subplot_analysis(n_per_class=50)


if __name__ == "__main__":
    train_model()