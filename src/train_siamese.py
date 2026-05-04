import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import json
import sys
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    fbeta_score,
)


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

from utils import load_config
from Data_handler import prepare_agn_data, SyntheticSiameseDataset
from architectures import SpectraNet, BinaryFocalLossWithLogits, SiameseSpectraNet
#from model_interpertation import run_gradcam_analysis, plot_attention_map, plot_transformer_attention

import torch.nn.functional as F

def seed_everything(seed=42):
    """
    Makes the experiment more reproducible.
    This fixes Python random, NumPy random, PyTorch initialization,
    dropout randomness, and most CUDA randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Could not fully enforce deterministic algorithms: {e}")


def evaluate_siamese_threshold_sweep(
    model,
    dataloader,
    device,
    thresholds=None,
    beta=0.5,
    min_recall=0.10,
    max_fpr=0.01,
):
    """
    Selects the best threshold for a high-purity CL-AGN search.

    Positive class:
        1 = change / CL-AGN-like

    Rule:
        First keep only thresholds satisfying:
            recall_change >= min_recall
            false_positive_rate <= max_fpr

        Among valid thresholds, choose the one with:
            highest F-beta score, usually F0.5 for purity
            then highest precision_change
            then lowest false_positive_rate
            then highest recall_change

        If no threshold satisfies the constraints, fall back to the best
        F-beta threshold over all thresholds and mark used_fallback=True.
    """
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in dataloader:
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            logits = model(batch_x1, batch_x2)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets).astype(int)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        num_predicted_positive = int(preds.sum())
        predicted_positive_fraction = num_predicted_positive / len(preds)

        tn, fp, fn, tp = confusion_matrix(
            all_targets,
            preds,
            labels=[0, 1],
        ).ravel()

        precision_change = precision_score(
            all_targets,
            preds,
            pos_label=1,
            zero_division=0,
        )

        recall_change = recall_score(
            all_targets,
            preds,
            pos_label=1,
            zero_division=0,
        )

        fbeta = fbeta_score(
            all_targets,
            preds,
            beta=beta,
            pos_label=1,
            zero_division=0,
        )

        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)

        results.append({
            "threshold": float(threshold),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision_change": float(precision_change),
            "recall_change": float(recall_change),
            "fbeta": float(fbeta),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "num_predicted_positive": int(num_predicted_positive),
            "predicted_positive_fraction": float(predicted_positive_fraction),
        })

    valid = [
        r for r in results
        if r["recall_change"] >= min_recall
        and r["false_positive_rate"] <= max_fpr
    ]

    used_fallback = len(valid) == 0

    if used_fallback:
        print(
            f"No threshold satisfied min_recall={min_recall} "
            f"and max_fpr={max_fpr}. "
            "Falling back to best F-beta over all thresholds."
        )
        candidates = results
    else:
        candidates = valid

    best = max(
        candidates,
        key=lambda r: (
            r["fbeta"],
            r["precision_change"],
            -r["false_positive_rate"],
            r["recall_change"],
        )
    )

    best["used_fallback"] = used_fallback

    return best, results


def train_one_epoch_siamese(model, dataloader, criterion, optimizer, device, threshold):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch_x1, batch_x2, batch_y in dataloader:
        batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)

        optimizer.zero_grad()
        
        logits = model(batch_x1, batch_x2)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        
        all_preds.extend(preds.detach().cpu().numpy().flatten())
        all_targets.extend(batch_y.detach().cpu().numpy().flatten())

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
            
            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_targets.extend(batch_y.detach().cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1

def evaluate_masked_siamese():
    print("\n=== Evaluating on test set with MASKED lines ===")
    
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    seed = int(config['siamese_training'].get('seed', 42))
    seed_everything(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_dir = os.path.join(BASE_DIR, 'models/siamese_network/')
    
    eval_change_pair_prob = float(config["siamese_training"].get("eval_change_pair_prob", 0.02))

    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df_clean = df_clean.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])

    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df_clean.columns if c not in meta_cols]

    _, df_temp = train_test_split(df_clean, test_size=0.3, stratify=df_clean['agn_type'], random_state=seed)
    _, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['agn_type'], random_state=seed)
    df_test = df_test.reset_index(drop=True)

    print("Loading Model...")
    base_model = SpectraNet(config)
    base_model_path = os.path.join(BASE_DIR, 'models/selected_backbone/', 'best_spectranet.pth')
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    model = SiameseSpectraNet(base_model, freeze_backbone=True).to(device)
    
    model_save_path = os.path.join(models_dir, 'best_siamese_net.pth')
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"]
    model.eval()

    masked_test_dataset = SyntheticSiameseDataset(
        df_test, 
        flux_cols, 
        epoch_size=10000, 
        mode='test',
        change_pair_prob=eval_change_pair_prob,
        mask_lines=True
    )
    masked_test_loader = DataLoader(masked_test_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False)

    all_masked_preds, all_masked_targets = [], []
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in masked_test_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            logits = model(batch_x1, batch_x2)
            probs = torch.sigmoid(logits)
            preds = (probs >= best_threshold).float()
            all_masked_preds.extend(preds.cpu().numpy().flatten())
            all_masked_targets.extend(batch_y.cpu().numpy().flatten())

    cm_masked = confusion_matrix(all_masked_targets, all_masked_preds)
    tn, fp, fn, tp = cm_masked.ravel()
    
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_masked, display_labels=['Static (0)', 'Change (1)'])
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Masked Siamese Network — Test Confusion Matrix', fontsize=14, fontweight='bold')
    cm_path = os.path.join(models_dir, 'masked_siamese_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"Saved masked confusion matrix to {cm_path}")
    plt.show()
    
    test_precision_masked = precision_score(all_masked_targets, all_masked_preds, pos_label=1, zero_division=0)
    test_recall_masked = recall_score(all_masked_targets, all_masked_preds, pos_label=1, zero_division=0)
    test_f0_5_masked = fbeta_score(all_masked_targets, all_masked_preds, beta=0.5, pos_label=1, zero_division=0)
    test_fpr_masked = fp / (fp + tn + 1e-8)
    test_fnr_masked = fn / (fn + tp + 1e-8)

    print(f"Masked Test precision(change): {test_precision_masked:.4f}")
    print(f"Masked Test recall(change): {test_recall_masked:.4f}")
    print(f"Masked Test F0.5(change): {test_f0_5_masked:.4f}")
    print(f"Masked Test FPR: {test_fpr_masked:.4f}")
    print(f"Masked Test FNR: {test_fnr_masked:.4f}")
    print(f"Masked TP={tp}, FP={fp}, TN={tn}, FN={fn}")


def train_siamese():

    config = load_config(os.path.join(BASE_DIR, 'config.yml'))

    seed = int(config['siamese_training'].get('seed', 42))
    seed_everything(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_dir = os.path.join(BASE_DIR, 'models/siamese_network/')
    os.makedirs(models_dir, exist_ok=True)
    threshold = float(config['siamese_training']['decision_threshold'])
    
    train_change_pair_prob = float(
        config["siamese_training"].get("train_change_pair_prob", 0.15)
    )

    eval_change_pair_prob = float(
        config["siamese_training"].get("eval_change_pair_prob", 0.02)
    )



    # 2. Data
    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df_clean = df_clean.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])

    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df_clean.columns if c not in meta_cols]

    df_train, df_temp = train_test_split(df_clean, test_size=0.3, stratify=df_clean['agn_type'], random_state=seed)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['agn_type'], random_state=seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 3. Create Datasets
    train_dataset = SyntheticSiameseDataset(
        df_train, 
        flux_cols, 
        epoch_size=20000, 
        mode = 'train',
        change_pair_prob=train_change_pair_prob
    )
    val_dataset = SyntheticSiameseDataset(
        df_val, 
        flux_cols, 
        epoch_size=10000, 
        mode = 'val',
        change_pair_prob=eval_change_pair_prob
    )
    test_dataset = SyntheticSiameseDataset(
        df_test, 
        flux_cols, 
        epoch_size=10000, 
        mode = 'test',
        change_pair_prob=eval_change_pair_prob
    )

    # 4. Loaders
    train_loader = DataLoader(train_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['siamese_training']['batch_size'], shuffle=False)


    print("Loading Pretrained SpectraNet Backbone...")
    base_model = SpectraNet(config)
    base_model_path = os.path.join(BASE_DIR, 'models/selected_backbone/', 'best_spectranet.pth')
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    
    model = SiameseSpectraNet(base_model, freeze_backbone=True).to(device)


    criterion = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0,reduction='mean')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=float(config['siamese_training']['learning_rate']), 
                            weight_decay=float(config['siamese_training']['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) 
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_score = (-1.0, -1.0, -1.0,-1.0)
    best_threshold = None
    best_threshold_metrics = None
    model_save_path = os.path.join(models_dir, 'best_siamese_net.pth')

    max_fpr = float(config['siamese_training'].get('max_fpr', 0.05))
    min_recall = float(config["siamese_training"].get("min_recall", 0.5))


    for epoch in range(config['siamese_training']['num_epochs']):

        t_loss,t_f1 = train_one_epoch_siamese(model, train_loader, criterion, optimizer, device,threshold)
        v_loss,v_f1 = validate_one_epoch_siamese(model, val_loader, criterion, device,threshold)


        best_epoch_threshold, threshold_results = evaluate_siamese_threshold_sweep(
        model=model,
        dataloader=val_loader,
        device=device,
        beta=0.5,
        min_recall=min_recall,
        max_fpr=max_fpr,
    )

        current_score = (
            best_epoch_threshold["fbeta"],
            best_epoch_threshold["precision_change"],
            -best_epoch_threshold["false_positive_rate"],
            best_epoch_threshold["recall_change"],
        )

        # Track history
        history['train_loss'].append(t_loss)
        history['train_f1'].append(t_f1)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(v_f1)

        scheduler_score = best_epoch_threshold["fbeta"]
        scheduler.step(scheduler_score)
        
        print(
        f"Epoch {epoch+1}/{config['siamese_training']['num_epochs']} | "
        f"Train Loss: {t_loss:.4f} | Train F1: {t_f1:.4f} | "
        f"Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f} | "
        f"Thr: {best_epoch_threshold['threshold']:.2f} | "
        f"Selected: {best_epoch_threshold['num_predicted_positive']} "
        f"({100 * best_epoch_threshold['predicted_positive_fraction']:.2f}%) | "
        f"Prec: {best_epoch_threshold['precision_change']:.4f} | "
        f"Rec: {best_epoch_threshold['recall_change']:.4f} | "
        f"FPR: {best_epoch_threshold['false_positive_rate']:.4f} | "
        f"F0.5: {best_epoch_threshold['fbeta']:.4f} | "
        f"TP/FP: {best_epoch_threshold['tp']}/{best_epoch_threshold['fp']}"
        )

        if current_score > best_score:
            best_score = current_score
            best_threshold = best_epoch_threshold["threshold"]
            best_threshold_metrics = best_epoch_threshold
        
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_threshold": best_threshold,
                    "best_threshold_metrics": best_threshold_metrics,
                    "best_score": best_score,
                    "max_fpr": max_fpr,
                },
                model_save_path,
            )
            summary_path = os.path.join(models_dir, "best_siamese_threshold_summary.json")

            summary = {
                "best_epoch": epoch + 1,
                "best_threshold": float(best_threshold),
                "best_threshold_metrics": best_threshold_metrics,
                "best_score": {
                    "fbeta": float(best_score[0]),
                    "precision_change": float(best_score[1]),
                    "negative_fpr": float(best_score[2]),
                    "fpr": float(-best_score[2]),
                    "recall_change": float(best_score[3]),
                },
                "min_recall": float(min_recall),
                "max_fpr": float(max_fpr),
                "model_path": model_save_path,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
                    
            print(
                f"[Epoch {epoch+1}] New best model saved! "
                f"Precision={best_score[0]:.4f}, "
                f"FPR={-best_score[1]:.4f}, "
                f"Recall={best_score[2]:.4f}, "
                f"F0.5={best_score[3]:.4f}, "
                f"threshold={best_threshold:.2f}, "
                f"max_fpr={max_fpr:.4f}"
                )
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
    
    checkpoint = torch.load(model_save_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"]
    print(f"Loaded best threshold: {best_threshold:.2f}")
    print("Best validation threshold metrics:")
    print(checkpoint["best_threshold_metrics"])
    model.eval()

    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in test_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            logits = model(batch_x1, batch_x2)
            probs = torch.sigmoid(logits)
            preds = (probs >= best_threshold).float()
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
    
    tn, fp, fn, tp = cm.ravel()
    test_selected = int(tp + fp)
    test_selected_fraction = test_selected / len(all_test_preds)

    test_precision_change = precision_score(
        all_test_targets,
        all_test_preds,
        pos_label=1,
        zero_division=0,
    )

    test_recall_change = recall_score(
        all_test_targets,
        all_test_preds,
        pos_label=1,
        zero_division=0,
    )

    test_f0_5 = fbeta_score(
        all_test_targets,
        all_test_preds,
        beta=0.5,
        pos_label=1,
        zero_division=0,
    )

    test_fpr = fp / (fp + tn + 1e-8)
    test_fnr = fn / (fn + tp + 1e-8)

    print(f"Test F1 macro: {test_f1:.4f}")
    print(f"Test precision(change): {test_precision_change:.4f}")
    print(f"Test recall(change): {test_recall_change:.4f}")
    print(f"Test F0.5(change): {test_f0_5:.4f}")
    print(f"Test FPR: {test_fpr:.4f}")
    print(f"Test FNR: {test_fnr:.4f}")
    print(f"Test selected candidates: {test_selected} ({100 * test_selected_fraction:.2f}%)")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # ---------- 3. Attention Map on a Random Test Pair ----------
    #print("Plotting attention map on a random test pair...")

    # Get the backbone model for attention map visualization
    #backbone = model.feature_extractor  # this is the pretrained SpectraNet backbone's feature_extractor
    #base_model_for_attn = SpectraNet(config).to(device)
    #base_model_for_attn.load_state_dict(torch.load(base_model_path, map_location=device))
    #base_model_for_attn.eval()

    # Grab wavelengths from flux columns
    #wavelengths = np.array(flux_cols, dtype=float)

    # Pick a random pair from the test dataset
    #rand_idx = random.randint(0, len(test_dataset) - 1)
    #sample_x1, sample_x2, sample_label = test_dataset[rand_idx]
    #sample_x1 = sample_x1.to(device)  # Shape: [1, 1024]
    #sample_x2 = sample_x2.to(device)  # Shape: [1, 1024]
    #pair_label = int(sample_label.item())
    
    # Get the siamese prediction for this pair
    #with torch.no_grad():
    #    pair_logits = model(sample_x1, sample_x2)
    #    pair_prob = torch.sigmoid(pair_logits).item()
    #    pair_pred = 1 if pair_prob >= config['siamese_training']['decision_threshold'] else 0

    # Plot attention maps for both spectra in the pair (stacked vertically)
    #fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    #for ax, spec_proc, spec_label_str in [
    #    (ax_top, sample_x1_proc, 'Spectrum 1'),
    #    (ax_bot, sample_x2_proc, 'Spectrum 2')
   # ]:
        # Run GradCAM on the backbone to get the attention/importance map
        #from model_interpertation import GradCAM
        #target_layer = base_model_for_attn.feature_extractor[2]
        #grad_cam = GradCAM(base_model_for_attn, target_layer)

        # Use target_class=1 (Type 2) to highlight regions the backbone cares about
        #cam, pred_prob = grad_cam(spec_proc, target_class=1)

        #spec_array = spec_proc.squeeze().cpu().numpy()

        # Plot the spectrum
        #ax.plot(wavelengths, spec_array, color='black', linewidth=1.2, label='Spectrum')
        # Overlay Grad-CAM heatmap
        #sc = ax.scatter(wavelengths, spec_array, c=cam, cmap='jet', alpha=0.6, s=15, zorder=2)
        #cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        #cbar.set_label('Grad-CAM Importance', rotation=270, labelpad=15)

        #ax.set_ylabel('Normalized Flux', fontsize=12)
        #ax.grid(True, linestyle='--', alpha=0.5)
        #ax.legend(loc='upper right')
        #ax.set_title(f'{spec_label_str} — Backbone Pred Prob: {pred_prob:.3f}', fontsize=13, fontweight='bold')

    #ax_bot.set_xlabel('Rest Wavelength (Å)', fontsize=12)

    #pred_str = "Change" if pair_pred == 1 else "Static"
    #true_str = "Change" if pair_label == 1 else "Static"
    #fig.suptitle(
        #f'Siamese Pair Attention Map — Pred: {pred_str} (prob={pair_prob:.3f}) | True: {true_str}',
        #fontsize=15, fontweight='bold', y=1.02
    #)

    #plt.tight_layout()
    #attn_path = os.path.join(models_dir, 'siamese_pair_attention_map.png')
    #plt.savefig(attn_path, dpi=300, bbox_inches='tight')
    #print(f"Saved attention map to {attn_path}")
    #plt.show()

    # ---------- 4. Evaluate with Masked Lines ----------
    

    print("\n=== All post-training visualizations complete ===")

if __name__ == "__main__":
    evaluate_masked_siamese()
    #train_siamese()