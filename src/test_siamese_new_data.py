"""
Test a trained Siamese AGN network on new SDSS/SDSS-V paired spectra.

Expected pickle columns:
    sdssid, z, specname_dr16, specname_sdssv, label

Main behavior:
    1. Read crossmatch pickle.
    2. Find both spectra files for each object.
    3. Skip objects with missing files and save a CSV log.
    4. Load + preprocess each spectrum into the same 1D tensor format used by SpectraNet.
    5. Run the trained Siamese model.
    6. Print metrics and save/display confusion matrix.

Important:
    The preprocessing here must match your training pipeline. If your project already has
    a function such as load_and_preprocess_spectrum(...), replace the body of
    load_single_spectrum(...) with that exact function.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt

try:
    from astropy.io import fits
except ImportError as exc:
    raise ImportError("Install astropy first: pip install astropy") from exc


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def import_object(import_string: str):
    """
    Import an object from a string like:
        models.siamese:SiameseSpectraNet
        my_package.my_file:MyClass
    """
    module_name, object_name = import_string.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_model_from_import(model_import: str, model_kwargs: dict):
    """
    Build your Siamese model dynamically.

    Example CLI:
        --model-import siamese_model:SiameseSpectraNet

    If your constructor needs arguments, add them below or pass them through model_kwargs.
    """
    model_cls = import_object(model_import)
    return model_cls(**model_kwargs)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handles both raw state_dict and checkpoints like {"model_state_dict": ...}
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Remove common DDP/DataParallel prefix if needed
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        clean_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print("[checkpoint] Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    if unexpected:
        print("[checkpoint] Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# FITS loading + preprocessing
# ---------------------------------------------------------------------

def _read_sdss_like_fits(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return observed wavelength and flux from an SDSS-like FITS spectrum.

    Supports common formats:
      - extension table with columns: loglam, flux
      - extension table with columns: wavelength/lambda/wave and flux
      - simple primary array is not handled because wavelength solution is ambiguous
    """
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is None:
                continue

            names = getattr(data, "names", None)
            if names is None:
                continue

            lower_to_original = {name.lower(): name for name in names}

            flux_col = None
            for candidate in ["flux", "flam", "f_lambda"]:
                if candidate in lower_to_original:
                    flux_col = lower_to_original[candidate]
                    break

            if flux_col is None:
                continue

            if "loglam" in lower_to_original:
                wave = 10.0 ** np.asarray(data[lower_to_original["loglam"]], dtype=np.float64)
            else:
                wave_col = None
                for candidate in ["wavelength", "lambda", "wave", "lam"]:
                    if candidate in lower_to_original:
                        wave_col = lower_to_original[candidate]
                        break
                if wave_col is None:
                    continue
                wave = np.asarray(data[wave_col], dtype=np.float64)

            flux = np.asarray(data[flux_col], dtype=np.float64)

            finite = np.isfinite(wave) & np.isfinite(flux)
            wave = wave[finite]
            flux = flux[finite]

            order = np.argsort(wave)
            return wave[order], flux[order]

    raise ValueError(f"Could not find wavelength/flux columns in {path}")


def robust_median_mad_scale(flux: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Same idea as the median/MAD scaling you moved to for shortcut-learning robustness.
    """
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    if not np.isfinite(mad) or mad < eps:
        mad = np.nanstd(flux)
    if not np.isfinite(mad) or mad < eps:
        mad = 1.0
    return (flux - med) / (mad + eps)


def load_single_spectrum(
    path: Path,
    z: float,
    n_bins: int = 1024,
    rest_wave_min: float = 3800.0,
    rest_wave_max: float = 7200.0,
    do_median_mad: bool = True,
) -> torch.Tensor:
    """
    Load one spectrum and return shape [1, n_bins].

    Replace this function with your exact training preprocessing if you already have one.
    The most important thing is that the new data enters the model in the exact same
    representation as the train/val/test spectra.
    """
    obs_wave, flux = _read_sdss_like_fits(path)

    # Rest-frame alignment
    rest_wave = obs_wave / (1.0 + float(z))

    # Common grid. Must match the grid used during training.
    target_wave = np.linspace(rest_wave_min, rest_wave_max, n_bins)

    # Interpolate flux to target grid. Outside coverage -> NaN, then fill with median.
    interp_flux = np.interp(target_wave, rest_wave, flux, left=np.nan, right=np.nan)

    if np.any(~np.isfinite(interp_flux)):
        fill = np.nanmedian(interp_flux)
        if not np.isfinite(fill):
            fill = 0.0
        interp_flux = np.where(np.isfinite(interp_flux), interp_flux, fill)

    if do_median_mad:
        interp_flux = robust_median_mad_scale(interp_flux)

    interp_flux = interp_flux.astype(np.float32)
    return torch.from_numpy(interp_flux).unsqueeze(0)  # [C=1, L]


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class NewSDSSSiameseDataset(Dataset):
    def __init__(
        self,
        pkl_path: str,
        spectra_root: Optional[str] = None,
        dr16_dir: Optional[str] = None,
        sdssv_dir: Optional[str] = None,
        missing_log_path: str = "missing_spectra_log.csv",
        n_bins: int = 1024,
        rest_wave_min: float = 3800.0,
        rest_wave_max: float = 7200.0,
        do_median_mad: bool = True,
    ):
        self.df = pd.read_pickle(pkl_path).copy()

        required_cols = ["sdssid", "z", "specname_dr16", "specname_sdssv", "label"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in pickle: {missing_cols}")

        self.spectra_root = Path(spectra_root) if spectra_root else None
        self.dr16_dir = Path(dr16_dir) if dr16_dir else self.spectra_root
        self.sdssv_dir = Path(sdssv_dir) if sdssv_dir else self.spectra_root

        if self.dr16_dir is None or self.sdssv_dir is None:
            raise ValueError("Provide either --spectra-root or both --dr16-dir and --sdssv-dir")

        self.missing_log_path = Path(missing_log_path)
        self.n_bins = n_bins
        self.rest_wave_min = rest_wave_min
        self.rest_wave_max = rest_wave_max
        self.do_median_mad = do_median_mad

        self.samples: List[Dict] = []
        self.missing_rows: List[Dict] = []
        self._index_valid_pairs()
        self._write_missing_log()

    def _resolve_path(self, base_dir: Path, filename: str) -> Path:
        filename = str(filename)
        p = Path(filename)
        if p.is_absolute():
            return p
        return base_dir / filename

    def _index_valid_pairs(self):
        for row in self.df.itertuples(index=False):
            sdssid = getattr(row, "sdssid")
            z = getattr(row, "z")
            specname_dr16 = getattr(row, "specname_dr16")
            specname_sdssv = getattr(row, "specname_sdssv")
            label = getattr(row, "label")

            dr16_path = self._resolve_path(self.dr16_dir, specname_dr16)
            sdssv_path = self._resolve_path(self.sdssv_dir, specname_sdssv)

            missing = []
            if not dr16_path.exists():
                missing.append("dr16")
            if not sdssv_path.exists():
                missing.append("sdssv")

            if missing:
                self.missing_rows.append({
                    "sdssid": sdssid,
                    "z": z,
                    "label": label,
                    "missing_side": "+".join(missing),
                    "specname_dr16": specname_dr16,
                    "specname_sdssv": specname_sdssv,
                    "dr16_path": str(dr16_path),
                    "sdssv_path": str(sdssv_path),
                })
                continue

            self.samples.append({
                "sdssid": sdssid,
                "z": float(z),
                "label": int(label),
                "dr16_path": dr16_path,
                "sdssv_path": sdssv_path,
                "specname_dr16": specname_dr16,
                "specname_sdssv": specname_sdssv,
            })

        print(f"Indexed valid pairs: {len(self.samples):,}")
        print(f"Skipped missing pairs: {len(self.missing_rows):,}")

    def _write_missing_log(self):
        if not self.missing_rows:
            return
        self.missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.missing_rows).to_csv(self.missing_log_path, index=False)
        print(f"Missing spectra log saved to: {self.missing_log_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        x1 = load_single_spectrum(
            sample["dr16_path"],
            z=sample["z"],
            n_bins=self.n_bins,
            rest_wave_min=self.rest_wave_min,
            rest_wave_max=self.rest_wave_max,
            do_median_mad=self.do_median_mad,
        )
        x2 = load_single_spectrum(
            sample["sdssv_path"],
            z=sample["z"],
            n_bins=self.n_bins,
            rest_wave_min=self.rest_wave_min,
            rest_wave_max=self.rest_wave_max,
            do_median_mad=self.do_median_mad,
        )

        y = torch.tensor(sample["label"], dtype=torch.long)

        meta = {
            "sdssid": sample["sdssid"],
            "specname_dr16": sample["specname_dr16"],
            "specname_sdssv": sample["specname_sdssv"],
        }
        return x1, x2, y, meta


def collate_with_meta(batch):
    x1, x2, y, meta = zip(*batch)
    return (
        torch.stack(x1, dim=0),
        torch.stack(x2, dim=0),
        torch.stack(y, dim=0),
        list(meta),
    )


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------

def siamese_prob_from_output(output: torch.Tensor) -> torch.Tensor:
    """
    Convert model output to P(change).

    Supports:
      - shape [B]: already logits/probabilities for positive class
      - shape [B, 1]: binary logit/probability
      - shape [B, 2]: two-class logits/probabilities, positive class at index 1

    If your model already returns probability, this still behaves safely in most cases.
    """
    if output.ndim == 1:
        # Usually BCE logit. If values already in [0,1], leave them.
        if output.min() >= 0 and output.max() <= 1:
            return output
        return torch.sigmoid(output)

    if output.ndim == 2 and output.shape[1] == 1:
        output = output.squeeze(1)
        if output.min() >= 0 and output.max() <= 1:
            return output
        return torch.sigmoid(output)

    if output.ndim == 2 and output.shape[1] == 2:
        # If output rows already look like probabilities, this is still okay enough,
        # but softmax is correct for logits.
        return torch.softmax(output, dim=1)[:, 1]

    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


@torch.no_grad()
def predict_siamese(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    all_probs = []
    all_labels = []
    all_meta = []

    model.eval()
    for x1, x2, y, meta in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        output = model(x1, x2)
        prob = siamese_prob_from_output(output)

        all_probs.append(prob.detach().cpu().numpy())
        all_labels.append(y.numpy())
        all_meta.extend(meta)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    return probs, labels, all_meta


# ---------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    beta: float = 0.5,
):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_change": precision_score(y_true, y_pred, zero_division=0),
        "recall_change": recall_score(y_true, y_pred, zero_division=0),
        "f1_change": f1_score(y_true, y_pred, zero_division=0),
        f"f{beta}_change": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "false_positive_rate": fp / max(fp + tn, 1),
        "predicted_positive_fraction": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    # Ranking metrics only make sense when both classes exist.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return y_pred, cm, metrics


def print_metrics(y_true, y_pred, y_prob, metrics):
    print("\n=== Siamese test metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\n=== Classification report ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["static/no-change", "change/CL-AGN"],
        zero_division=0,
    ))


def plot_confusion_matrix(cm, save_path: Optional[str] = None, show: bool = True):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Pred static", "Pred change"],
        yticklabels=["True static", "True change"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Siamese confusion matrix",
    )

    max_value = cm.max() if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_predictions_csv(
    path: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    meta: List[Dict],
):
    rows = []
    for label, prob, pred, m in zip(y_true, y_prob, y_pred, meta):
        rows.append({
            "sdssid": m["sdssid"],
            "specname_dr16": m["specname_dr16"],
            "specname_sdssv": m["specname_sdssv"],
            "true_label": int(label),
            "prob_change": float(prob),
            "pred_label": int(pred),
            "is_false_positive": int(label == 0 and pred == 1),
            "is_false_negative": int(label == 1 and pred == 0),
        })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Predictions saved to: {path}")


def threshold_sweep_for_inspection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    beta: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        _, _, metrics = evaluate_at_threshold(y_true, y_prob, float(thr), beta=beta)
        rows.append(metrics)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-import", required=True,
                        help="Example: siamese_model:SiameseSpectraNet")

    parser.add_argument("--spectra-root", default=None,
                        help="Use this if both DR16 and SDSS-V files are in one directory")
    parser.add_argument("--dr16-dir", default=None)
    parser.add_argument("--sdssv-dir", default=None)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--n-bins", type=int, default=1024)
    parser.add_argument("--rest-wave-min", type=float, default=3800.0)
    parser.add_argument("--rest-wave-max", type=float, default=7200.0)
    parser.add_argument("--no-median-mad", action="store_true")

    parser.add_argument("--missing-log", default="outputs/missing_spectra_log.csv")
    parser.add_argument("--predictions-csv", default="outputs/new_sdss_siamese_predictions.csv")
    parser.add_argument("--cm-path", default="outputs/new_sdss_siamese_confusion_matrix.png")
    parser.add_argument("--threshold-sweep-csv", default="outputs/new_sdss_threshold_sweep.csv")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-show", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = NewSDSSSiameseDataset(
        pkl_path=args.pkl_path,
        spectra_root=args.spectra_root,
        dr16_dir=args.dr16_dir,
        sdssv_dir=args.sdssv_dir,
        missing_log_path=args.missing_log,
        n_bins=args.n_bins,
        rest_wave_min=args.rest_wave_min,
        rest_wave_max=args.rest_wave_max,
        do_median_mad=not args.no_median_mad,
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid pairs found. Check spectra directories and filenames.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,
    )

    # Add model kwargs here if your Siamese constructor requires them.
    model_kwargs = {}
    model = build_model_from_import(args.model_import, model_kwargs=model_kwargs)
    model = load_checkpoint(model, args.checkpoint, device)

    y_prob, y_true, meta = predict_siamese(model, loader, device)
    y_pred, cm, metrics = evaluate_at_threshold(y_true, y_prob, args.threshold, beta=0.5)

    print_metrics(y_true, y_pred, y_prob, metrics)
    plot_confusion_matrix(cm, save_path=args.cm_path, show=not args.no_show)
    save_predictions_csv(args.predictions_csv, y_true, y_prob, y_pred, meta)

    # Save a threshold sweep so you can choose an FPR-budgeted threshold later.
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep_df = threshold_sweep_for_inspection(y_true, y_prob, thresholds, beta=0.5)
    Path(args.threshold_sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(args.threshold_sweep_csv, index=False)
    print(f"Threshold sweep saved to: {args.threshold_sweep_csv}")

    print("\n=== Useful quick checks ===")
    print(f"Total valid evaluated pairs: {len(y_true):,}")
    print(f"True positives in dataset: {int(y_true.sum()):,}")
    print(f"True positive fraction: {float(y_true.mean()):.6f}")
    print(f"Mean predicted P(change): {float(y_prob.mean()):.6f}")
    print(f"Median predicted P(change): {float(np.median(y_prob)):.6f}")


if __name__ == "__main__":
    main()
