"""
gradcam_pairs.py
================
GradCAM visualization for SiameseChangeNet pair predictions.

Hooks the last conv feature map (encoder.feature_extractor output, [B,256,512])
and back-propagates the classification logit to get per-wavelength importance.
No weights are modified — only a forward+backward pass is run.

Usage:
    conda run -n astro_dl python src/gradcam_pairs.py --config config_v2.yml
    conda run -n astro_dl python src/gradcam_pairs.py --config config_v2.yml --n_tp 3 --n_fp 2 --n_tn 2 --n_fn 2
"""

from __future__ import annotations

import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import load_config
from datasets_v2 import load_or_build_pair_arrays, split_indices, _two_channel
from architectures_v2 import SiameseChangeNet


# ── helpers ───────────────────────────────────────────────────────────────────

def pick_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_norm_stats(path):
    import json
    with open(path) as f:
        return json.load(f)


def compute_input_gradients(model, x1, x2, device):
    """
    Signed per-pixel gradient of the classification logit w.r.t. the
    ch0 (MAD-normalised) channel of each input spectrum.

    Returns (grad1, grad2): each is a [4096] numpy array (signed, raw gradient).
    No weights are modified — .grad on params is cleared after use.
    """
    x1_t = torch.from_numpy(x1).unsqueeze(0).float().to(device).requires_grad_(True)
    x2_t = torch.from_numpy(x2).unsqueeze(0).float().to(device).requires_grad_(True)

    model.zero_grad()
    logit = model(x1_t, x2_t)
    logit.backward()

    grad1 = x1_t.grad[0, 0].cpu().numpy()   # ch0 gradient, shape [4096]
    grad2 = x2_t.grad[0, 0].cpu().numpy()

    model.zero_grad()
    return grad1, grad2


def plot_pair(pair_idx, arrays, model, device, wave, channel1_scale,
              label, prob, threshold, tag, out_path):
    """
    2-panel plot, one per epoch. Each panel:
      - Black spectrum line (MAD-normalised ch0)
      - Scatter of (wavelength, flux) coloured by signed input gradient
      - Diverging RdBu_r colorbar on the right
    """
    raw1 = arrays["raw1"][pair_idx];  raw2 = arrays["raw2"][pair_idx]
    mad1 = arrays["mad1"][pair_idx];  mad2 = arrays["mad2"][pair_idx]
    v1   = arrays["valid1"][pair_idx].astype(bool)
    v2   = arrays["valid2"][pair_idx].astype(bool)
    o1   = float(arrays["oiii1"][pair_idx]); o2 = float(arrays["oiii2"][pair_idx])
    r1   = bool(arrays["rel1"][pair_idx]);   r2 = bool(arrays["rel2"][pair_idx])

    x1 = _two_channel(raw1, mad1, o1, r1, channel1_scale)
    x2 = _two_channel(raw2, mad2, o2, r2, channel1_scale)

    grad1, grad2 = compute_input_gradients(model, x1, x2, device)

    def blankout(arr, v): a = arr.copy().astype(float); a[~v] = np.nan; return a

    both_valid = v1 & v2
    cov_idx    = np.where(both_valid)[0]
    x_lo = wave[cov_idx[0]]  - 50
    x_hi = wave[cov_idx[-1]] + 50

    emission_lines = {"Hβ": 4861, "[OIII]": 5007, "Hα": 6563,
                      "MgII": 2798, "CIV": 1549}

    verdict = "POSITIVE" if label == 1 else "NEGATIVE"
    correct = (prob >= threshold) == (label == 1)
    outcome = ("TP" if correct else "FN") if label == 1 else ("TN" if correct else "FP")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    epoch_labels = ["Epoch 1 (DR16)", "Epoch 2"]

    for ax, flux_raw, grad, valid, epoch_lbl in [
        (axes[0], mad1, grad1, v1, epoch_labels[0]),
        (axes[1], mad2, grad2, v2, epoch_labels[1]),
    ]:
        flux  = blankout(flux_raw, valid)
        yvals = flux[np.isfinite(flux)]
        y_lo  = np.percentile(yvals, 1)  - 0.3 if len(yvals) else -2
        y_hi  = np.percentile(yvals, 99) + 0.3 if len(yvals) else  2

        # Smooth gradient (sigma ~20 px ≈ ~30 Å at rest-frame resolution)
        grad_smooth = gaussian_filter1d(grad, sigma=20)
        # Zero out uncovered pixels
        grad_smooth[~valid] = 0.0

        # Symmetric colormap limits (99th percentile of abs smoothed gradient)
        vlim = np.percentile(np.abs(grad_smooth[valid]), 99) if valid.any() else 0.1
        vlim = max(vlim, 1e-8)

        norm_cm = plt.Normalize(vmin=-vlim, vmax=vlim)
        cmap    = plt.get_cmap("RdBu_r")

        # Build LineCollection: each segment coloured by smoothed gradient
        flux_plot = flux.copy()
        flux_plot[~valid] = np.nan
        points   = np.array([wave, flux_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Black spectrum underneath
        ax.plot(wave, flux_plot, lw=0.8, color="black", zorder=2, label="Spectrum")
        # Coloured gradient overlay on top
        lc = LineCollection(segments, cmap=cmap, norm=norm_cm,
                            linewidth=1.5, zorder=3, alpha=0.85)
        lc.set_array(grad_smooth)
        ax.add_collection(lc)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("Grad-CAM Importance", fontsize=8, rotation=270, labelpad=12)
        cb.ax.tick_params(labelsize=7)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("Normalised Flux", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2, ls="--")

        # Emission line markers — anchored to axis fraction
        y0, y1 = ax.get_ylim()
        for name, wl in emission_lines.items():
            if x_lo <= wl <= x_hi:
                ax.axvline(wl, color="0.5", ls="--", lw=0.8, alpha=0.7)
                ax.text(wl + 25, y0 + 0.85 * (y1 - y0), name,
                        fontsize=8, color="0.3", rotation=90, va="top", clip_on=True)

        ax.set_title(
            f"Grad-CAM ({epoch_lbl})  |  Pred: {'CL-AGN' if prob >= threshold else 'No Change'} "
            f"(Prob: {prob:.3f})  |  Actual: {'CL-AGN' if label == 1 else 'No Change'}  [{outcome}]",
            fontsize=9, pad=4,
        )

    axes[1].set_xlabel("Rest-frame Wavelength (Å)", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",  default="config_v2.yml")
    parser.add_argument("--n_tp",   type=int, default=2, help="# TP examples")
    parser.add_argument("--n_fp",   type=int, default=2, help="# FP examples")
    parser.add_argument("--n_tn",   type=int, default=2, help="# TN examples")
    parser.add_argument("--n_fn",   type=int, default=2, help="# FN examples")
    parser.add_argument("--seed",   type=int, default=7)
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else \
               os.path.join(BASE_DIR, args.config)
    cfg    = load_config(cfg_path)
    paths  = cfg["paths"]
    pp     = cfg["preprocessing"]
    s      = cfg["siamese"]
    device = pick_device()
    rng    = np.random.default_rng(args.seed)

    print(f"[gradcam] device: {device}")

    # ── Paths ────────────────────────────────────────────────────────────
    def abs(p): return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
    ckpt_path    = abs(paths["siamese_checkpoint"])
    stats_path   = abs(paths["norm_stats"])
    pkl_path     = abs(paths["clagn_test_pickle"])
    spectra_dir  = abs(paths["clagn_test_spectra_dir"])
    out_dir      = abs(paths["out_dir"])
    eval_cache   = os.path.join(out_dir, "eval_pair_arrays_cache.npz")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model (eval mode, no grad on params) ─────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SiameseChangeNet(
        in_channels=2, dropout=s["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Frozen: only activations (not params) will accumulate grad during GradCAM
    for p in model.parameters():
        p.requires_grad_(False)
    # GradCAM needs grad on intermediate activations (not params) — enable it
    # on the feature_extractor output by setting requires_grad on its output
    # (handled automatically by PyTorch autograd once we call .backward())

    stats = load_norm_stats(stats_path)
    channel1_scale = float(stats.get("channel1_scale", 1.0))
    saved_threshold = float(ckpt.get("best_threshold", s["decision_threshold"]))
    print(f"[gradcam] channel1_scale={channel1_scale:.6f}  threshold={saved_threshold:.2f}")

    # ── Load pair arrays (uses cache if available) ────────────────────
    print(f"[gradcam] loading pair arrays (cache={eval_cache})")
    arrays = load_or_build_pair_arrays(
        pkl_path, spectra_dir,
        cache_path=eval_cache,
        oiii_snr_min=pp["oiii_snr_min"],
        split_filter="test",
    )
    n = len(arrays["y"])
    print(f"[gradcam] {n} test pairs")

    # ── Wave grid ────────────────────────────────────────────────────
    from data_preprocessing import MASTER_GRID
    wave = MASTER_GRID

    # ── Classify all pairs to get probs ──────────────────────────────
    from datasets_v2 import RealPairDataset
    import pandas as pd

    df_test = pd.read_pickle(pkl_path)
    if "split" in df_test.columns:
        df_test = df_test[df_test["split"] == "test"].reset_index(drop=True)

    probs = []
    with torch.no_grad():
        for i in range(n):
            r1  = arrays["raw1"][i]; r2  = arrays["raw2"][i]
            m1  = arrays["mad1"][i]; m2  = arrays["mad2"][i]
            o1  = float(arrays["oiii1"][i]); o2 = float(arrays["oiii2"][i])
            rl1 = bool(arrays["rel1"][i]);   rl2 = bool(arrays["rel2"][i])
            x1  = torch.from_numpy(_two_channel(r1, m1, o1, rl1, channel1_scale)).unsqueeze(0).float().to(device)
            x2  = torch.from_numpy(_two_channel(r2, m2, o2, rl2, channel1_scale)).unsqueeze(0).float().to(device)
            logit = model(x1, x2)
            probs.append(torch.sigmoid(logit).item())

    probs  = np.array(probs)
    labels = arrays["y"]
    preds  = (probs >= saved_threshold).astype(int)

    tp_idx = np.where((preds == 1) & (labels == 1))[0]
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    tn_idx = np.where((preds == 0) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]

    print(f"[gradcam] TP={len(tp_idx)} FP={len(fp_idx)} TN={len(tn_idx)} FN={len(fn_idx)}")

    def pick(idx_arr, n_req):
        if len(idx_arr) == 0: return []
        chosen = rng.choice(idx_arr, size=min(n_req, len(idx_arr)), replace=False)
        return chosen.tolist()

    to_plot = (
        [(i, "tp") for i in pick(tp_idx, args.n_tp)] +
        [(i, "fp") for i in pick(fp_idx, args.n_fp)] +
        [(i, "tn") for i in pick(tn_idx, args.n_tn)] +
        [(i, "fn") for i in pick(fn_idx, args.n_fn)]
    )

    # GradCAM needs grad on activations even when params are frozen.
    # Enable grad computation for this block only.
    for i, tag in to_plot:
        out_path = os.path.join(out_dir, f"gradcam_{tag}_{i:04d}.png")
        with torch.enable_grad():
            plot_pair(
                pair_idx=i,
                arrays=arrays,
                model=model,
                device=device,
                wave=wave,
                channel1_scale=channel1_scale,
                label=int(labels[i]),
                prob=float(probs[i]),
                threshold=saved_threshold,
                tag=tag,
                out_path=out_path,
            )

    print(f"[gradcam] done — {len(to_plot)} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
