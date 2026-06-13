"""
plot_ssl_reconstruction.py
==========================
Plot one DR16 spectrum: input with masked spans (top) and reconstruction (bottom).
X-axis is cropped to the spectrum's covered region.

Usage:
    conda run -n astro_dl python src/plot_ssl_reconstruction.py
    conda run -n astro_dl python src/plot_ssl_reconstruction.py --seed 7
    conda run -n astro_dl python src/plot_ssl_reconstruction.py --idx 42
"""

import argparse, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Change this to pick a different spectrum ──────────────────────────────────
DEFAULT_SEED = 500   # controls which DR16 spectrum is randomly selected
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import load_config
from datasets_v2 import SSLSpectraDataset
from architectures_v2 import MaskedSpectraAutoencoder, apply_span_mask


def pick_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",     type=int, default=None)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--channel", type=int, default=None, choices=[0, 1],
                        help="Which channel to plot (default: both → two files)")
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    channels = [args.channel] if args.channel is not None else [0, 1]

    cfg    = load_config(os.path.join(BASE_DIR, "config_v2.yml"))
    paths  = cfg["paths"]
    s      = cfg["ssl"]
    device = pick_device()
    rng    = np.random.default_rng(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────
    parquets = [os.path.join(BASE_DIR, q) if not os.path.isabs(q) else q
                for q in paths["ssl_parquets"]]
    parquets = [q for q in parquets if os.path.exists(q)]
    dataset  = SSLSpectraDataset(
        parquets, channel1_scale=None,
        oiii_snr_min=cfg["preprocessing"]["oiii_snr_min"],
    )

    # ── Pick a DR16 spectrum ──────────────────────────────────────────────
    surveys  = np.asarray(dataset.meta["survey"].values)
    dr16_idx = np.where(surveys == "dr16")[0]
    idx = args.idx if args.idx is not None else int(rng.choice(dr16_idx))

    z_val   = float(dataset.meta["z"].iloc[idx])
    survey  = dataset.meta["survey"].iloc[idx]
    fname   = dataset.meta.get("filename", dataset.meta.index).iloc[idx]
    wave    = dataset.wave

    print(f"[plot] idx={idx}  survey={survey}  z={z_val:.4f}  "
          f"file={os.path.basename(str(fname))}")

    # ── 2-channel input + mask ────────────────────────────────────────────
    x, valid = dataset[idx]
    x_t     = x.unsqueeze(0).to(device)
    valid_t = valid.unsqueeze(0).to(device)

    torch.manual_seed(args.seed)
    x_masked_t, span_mask_t = apply_span_mask(
        x_t, valid_t,
        mask_ratio=s["mask_ratio"],
        min_span=s["min_span"],
        max_span=s["max_span"],
    )
    span_mask = span_mask_t[0].cpu().numpy().astype(bool)
    valid_np  = valid.numpy().astype(bool)

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = paths["ssl_checkpoint"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = MaskedSpectraAutoencoder(in_channels=2).to(device)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()

    with torch.no_grad():
        recon_t = model(x_masked_t)

    # Blank out-of-coverage pixels
    def cov(arr):
        a = arr.copy().astype(float)
        a[~valid_np] = np.nan
        return a

    # ── Crop x-axis to covered region ────────────────────────────────────
    cov_idx = np.where(valid_np)[0]
    x_lo = wave[cov_idx[0]]  - 50
    x_hi = wave[cov_idx[-1]] + 50

    BLUE  = "#1f4e79"
    RED   = "#c0392b"
    GRAY  = "#cccccc"
    emission_lines = {"Hβ": 4861, "[OIII]": 5007, "Hα": 6563}

    def shade(ax):
        in_span = False
        for i, m in enumerate(span_mask):
            if m and not in_span:
                s0 = wave[i]; in_span = True
            elif not m and in_span:
                ax.axvspan(s0, wave[i-1], color=GRAY, alpha=0.6, lw=0)
                in_span = False
        if in_span:
            ax.axvspan(s0, wave[-1], color=GRAY, alpha=0.6, lw=0)

    def add_lines(ax, ydata):
        ymax = np.nanmax(ydata) if not np.all(np.isnan(ydata)) else 1
        for name, wl in emission_lines.items():
            if x_lo <= wl <= x_hi:
                ax.axvline(wl, color="0.5", ls="--", lw=0.8, alpha=0.7)
                ax.text(wl + 15, ymax * 0.88, name, fontsize=7.5,
                        color="0.3", rotation=90, va="top")

    from matplotlib.patches import Patch

    ch_labels = [
        "Channel 0 — MAD-normalised flux (arcsinh)",
        "Channel 1 — OIII-normalised flux (arcsinh)",
    ]
    ch_suffixes = ["ch0", "ch1"]

    out_dir = os.path.join(BASE_DIR, paths["out_dir"])
    saved = []

    for ch in channels:
        input_ch = x[ch].numpy()
        recon_ch = recon_t[0, ch].cpu().numpy()
        y_input  = cov(input_ch)
        y_recon  = cov(recon_ch)

        vm = valid_np & span_mask
        mse_masked = float(np.mean((recon_ch[vm] - input_ch[vm]) ** 2)) if vm.any() else np.nan

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                 gridspec_kw={"hspace": 0.05})
        fig.suptitle(
            f"{ch_labels[ch]}\n"
            f"SSL reconstruction — DR16  |  z={z_val:.4f}  |  "
            f"file: {os.path.basename(str(fname))}",
            fontsize=10,
        )

        # Top: original
        ax = axes[0]
        shade(ax)
        ax.plot(wave, y_input, lw=0.7, color=BLUE)
        add_lines(ax, y_input)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("flux", fontsize=9)
        ax.set_title("original", fontsize=9, pad=3)
        ax.legend(handles=[Patch(facecolor=GRAY, alpha=0.7, label="masked (zeroed input)")],
                  fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
        ax.axhline(0, color="0.8", lw=0.5)

        # Bottom: reconstruction
        ax = axes[1]
        shade(ax)
        ax.plot(wave, y_recon, lw=0.8, color=RED)
        add_lines(ax, y_recon)
        ax.set_ylabel("flux", fontsize=9)
        ax.set_xlabel("rest-frame wavelength [Å]", fontsize=9)
        ax.set_title(f"reconstruction  (masked-region MSE = {mse_masked:.4f})",
                     fontsize=9, pad=3)
        ax.grid(alpha=0.2)
        ax.axhline(0, color="0.8", lw=0.5)

        if args.out and len(channels) == 1:
            out_path = args.out
        else:
            out_path = os.path.join(out_dir, f"ssl_reconstruction_{ch_suffixes[ch]}.png")

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[plot] ch{ch} → {out_path}")


if __name__ == "__main__":
    main()
