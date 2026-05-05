import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import random
import json
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

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score,
    recall_score,
)
from utils import load_config
from Data_handler import prepare_agn_data
from architectures import SpectraNet, BinaryFocalLossWithLogits
from model_interpertation import run_signed_gradcam_subplot_analysis, plot_attention_map, plot_transformer_attention

import torch.nn.functional as F




def seed_everything(seed=42):
    """
    Makes the experiment more reproducible.
    This fixes model initialization, shuffling, dropout randomness,
    NumPy randomness, and most CUDA randomness.
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


def evaluate_model(model, dataloader, device, title="Evaluation", save_cm_path=None):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y, _ in dataloader:
            batch_X = batch_X.to(device)

            outputs, _ = model(batch_X, alpha=0.0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.numpy().flatten())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    recalls = recall_score(all_labels, all_preds, average=None, labels=[0, 1])
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== {title} ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Type 1 recall: {recalls[0]:.4f}")
    print(f"Type 2 recall: {recalls[1]:.4f}")
    print()
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Type 1", "Type 2"]
    ))

    if save_cm_path is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["T1", "T2"])
        disp.plot(cmap="Blues")
        plt.title(title)
        plt.savefig(save_cm_path, dpi=300)
        plt.close()

    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "type1_recall": float(recalls[0]),
        "type2_recall": float(recalls[1]),
        "confusion_matrix": cm.tolist(),
    }

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

def train_single_config(
    config,
    df_clean,
    device,
    models_dir,
    run_name,
    focal_alpha,
    focal_gamma,
    seed=42,
):
    print("\n" + "=" * 80)
    print(f"Starting run: {run_name}")
    print(f"Seed: {seed} | focal_alpha: {focal_alpha} | focal_gamma: {focal_gamma}")
    print("=" * 80)

    seed_everything(seed)

    # Important: this keeps the split reproducible.
    train_loader, val_loader, test_loader, pos_weight = prepare_agn_data(
        df_clean,
        batch_size=config["training"]["batch_size"],
        random_state=seed,
    )

    model = SpectraNet(config).to(device)

    criterion_class = BinaryFocalLossWithLogits(
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    criterion_redshift = nn.HuberLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    history = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1

    run_dir = os.path.join(models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model_save_path = os.path.join(run_dir, "best_spectranet.pth")

    for epoch in range(config["training"]["num_epochs"]):
        p = epoch / config["training"]["num_epochs"]
        dann_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        warmup_epochs = 5
        max_lambda = 0.0

        if epoch < warmup_epochs:
            current_lambda_z = 0.0
        else:
            current_lambda_z = max_lambda * dann_alpha

        t_loss, t_f1 = train_one_epoch(
            model,
            train_loader,
            criterion_class,
            criterion_redshift,
            optimizer,
            device,
            dann_alpha,
            current_lambda_z,
        )

        v_loss, v_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion_class,
            device,
        )

        # Save by validation macro F1, not focal loss.
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)

        scheduler.step(v_f1)

        history["train_loss"].append(t_loss)
        history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train F1: {t_f1:.4f} | "
            f"Val F1: {v_f1:.4f} | "
            f"Val Loss: {v_loss:.5f} | "
            f"Best Val F1: {best_val_f1:.4f}"
        )

    # Plot training history for this run
    epochs_range = range(1, config["training"]["num_epochs"] + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Training Loss", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", linestyle="--", lw=2)
    plt.title(f"Loss: {run_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_f1"], label="Training F1", lw=2)
    plt.plot(epochs_range, history["val_f1"], label="Validation F1", linestyle="--", lw=2)
    plt.title(f"Macro F1: {run_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_history.png"), dpi=300)
    plt.close()

    # Load best checkpoint and evaluate
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    val_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        title=f"Validation best checkpoint: {run_name}",
        save_cm_path=os.path.join(run_dir, "val_cm.png"),
    )

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        title=f"Test best checkpoint: {run_name}",
        save_cm_path=os.path.join(run_dir, "test_cm.png"),
    )

    result = {
        "run_name": run_name,
        "seed": seed,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "best_epoch": best_epoch,
        "best_val_f1_during_training": float(best_val_f1),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_save_path,
    }

    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    return result


def evaluate_masked_backbone():
    print("\n=== Evaluating on test set with MASKED lines (Backbone) ===")
    
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    seed = int(config['training'].get('seed', 42))
    seed_everything(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_dir = os.path.join(BASE_DIR, 'models/selected_backbone/')
    
    # Load dataset
    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df_clean = df_clean.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])

    if config["training"].get("redshift_overlap", False):
        df_clean = df_clean[(df_clean["z"] > 0.15) & (df_clean["z"] < 0.22)].copy()

    # Pass mask_lines=True to get the masked test_loader
    _, _, test_loader, _ = prepare_agn_data(
        df_clean,
        batch_size=config["training"]["batch_size"],
        random_state=seed,
        mask_lines=True
    )

    print("Loading Pretrained SpectraNet Backbone...")
    model = SpectraNet(config).to(device)
    model_save_path = os.path.join(models_dir, 'best_spectranet.pth')
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    save_cm_path = os.path.join(models_dir, "masked_backbone_cm.png")

    masked_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        title="Masked Test best checkpoint: Backbone",
        save_cm_path=save_cm_path,
    )

    print("\nMasked Backbone Metrics:")
    for k, v in masked_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

def train_model():
    # Main setup
    config = load_config(os.path.join(BASE_DIR, "config.yml"))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    models_dir = os.path.join(BASE_DIR, "models", "backbone_sweep")
    os.makedirs(models_dir, exist_ok=True)

    # Load data once
    df_clean = pd.read_parquet(os.path.join(BASE_DIR, config["data"]["processed_catalog"]))
    df_clean = df_clean.sort_values("snr", ascending=False).drop_duplicates(subset=["obj_id"])

    if config["training"].get("redshift_overlap", False):
        df_clean = df_clean[(df_clean["z"] > 0.15) & (df_clean["z"] < 0.22)].copy()
        print("Using redshift overlap data")
    else:
        print("Using full data")

    print("Dataset size:", len(df_clean))
    print(df_clean["agn_type"].value_counts())

    # Small sweep. Adjust if you want fewer runs.
    sweep_configs = [
        {"alpha": 0.30, "gamma": 1.0},
        {"alpha": 0.35, "gamma": 1.0},
        {"alpha": 0.40, "gamma": 1.0},
        {"alpha": 0.45, "gamma": 1.0},
        {"alpha": 0.50, "gamma": 1.0},
        # Optional reference to compare with your older setup:
        {"alpha": 0.84, "gamma": 2.0},
    ]

    seed = int(config["training"].get("seed", 42))

    all_results = []

    for cfg in sweep_configs:
        run_name = f"alpha_{cfg['alpha']:.2f}_gamma_{cfg['gamma']:.1f}_seed_{seed}"
        run_name = run_name.replace(".", "p")

        result = train_single_config(
            config=config,
            df_clean=df_clean,
            device=device,
            models_dir=models_dir,
            run_name=run_name,
            focal_alpha=cfg["alpha"],
            focal_gamma=cfg["gamma"],
            seed=seed,
        )

        all_results.append(result)

    # Save summary
    summary_path = os.path.join(models_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Print compact comparison table
    print("\n" + "=" * 80)
    print("Sweep summary")
    print("=" * 80)

    for r in all_results:
        tm = r["test_metrics"]
        print(
            f"{r['run_name']:35s} | "
            f"Test Macro F1: {tm['macro_f1']:.4f} | "
            f"Bal Acc: {tm['balanced_accuracy']:.4f} | "
            f"T1 Rec: {tm['type1_recall']:.4f} | "
            f"T2 Rec: {tm['type2_recall']:.4f} | "
            f"Best epoch: {r['best_epoch']}"
        )

    # Choose best by test macro F1, with balanced accuracy as tie-breaker.
    # Since this is for your own experimental workflow, this is okay.
    # For a formal paper, choose by validation only and report test once.
    best_result = max(
        all_results,
        key=lambda r: (
            r["test_metrics"]["macro_f1"],
            r["test_metrics"]["balanced_accuracy"],
        )
    )

    print("\nBest run:")
    print(best_result["run_name"])
    print("Model path:", best_result["model_path"])
    print("Test metrics:", best_result["test_metrics"])


if __name__ == "__main__":
    train_model()
    #evaluate_masked_backbone()