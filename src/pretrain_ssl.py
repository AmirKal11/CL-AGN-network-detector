"""
pretrain_ssl.py  --  STAGE 1
============================
Self-supervised masked-reconstruction pretraining of the 2-channel encoder on
the pooled, unlabelled multi-survey spectra.

No labels are used. For each spectrum, random wavelength spans are blanked and
the network must reconstruct them; the encoder learns line shapes, continuum
and broad-vs-narrow structure from far more data than the labelled set.

Run from the project root.

Phase A -- full retrain on the pooled parquets (DR7+DESI + DR16+SDSS-V):
    python src/pretrain_ssl.py

Phase B -- continual pretraining on top of an existing encoder, with
50/50 replay of old (DR7+DESI) and new (DR16+SDSS-V) spectra to prevent
catastrophic forgetting:
    python src/pretrain_ssl.py \\
        --resume-from models/clagn_v2/ssl_encoder_dr7desi.pth \\
        --replay --lr 1e-4 --num-epochs 20 \\
        --output-ckpt models/clagn_v2/ssl_encoder_continual.pth

Produces (under models/clagn_v2/):
    ssl_encoder.pth      (or --output-ckpt) encoder weights -> Stage 2
    norm_stats.json      channel-1 [OIII] scale -> reused by Stage 2
    ssl_loss_curve.png   reconstruction-loss curve
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import load_config
from datasets_v2 import SSLSpectraDataset
from architectures_v2 import MaskedSpectraAutoencoder, apply_span_mask
from preprocessing_oiii import save_norm_stats


def masked_mse(recon, target, span_mask, valid=None):
    """Reconstruction MSE over span-masked AND covered positions.

    recon/target [B,C,L]; span_mask/valid [B,L] bool. Restricting to covered
    pixels stops the model from being rewarded for predicting the 0.0
    out-of-coverage sentinel, which would otherwise dominate the loss for
    high-redshift spectra (a large fraction of the wide grid is uncovered).
    """
    m = span_mask
    if valid is not None:
        m = m & valid
    m = m.unsqueeze(1).float()                         # [B,1,L]
    se = (recon - target) ** 2 * m
    return se.sum() / (m.sum() * recon.shape[1] + 1e-8)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_input_sample(dataset, out_dir):
    """Sanity check before training: plot one random spectrum and the two
    processed channels the network will actually see.

    Three stacked panels -- the spectrum as stored in the parquet, then
    channel 0 and channel 1 after the arcsinh-compressed 2-channel build.
    Saved to ssl_input_sample.png so obvious preprocessing breakage (a dead
    channel, all-zero rows, runaway dynamic range) is caught here rather than
    after a multi-hour run.
    """
    idx = int(np.random.default_rng().integers(len(dataset)))
    raw = np.asarray(dataset.flux[idx], dtype=np.float32)   # channel 0, pre-arcsinh
    x, valid = dataset[idx]                                 # x [2,L], arcsinh-compressed
    x = x.numpy()
    valid = valid.numpy().astype(bool)
    wave = dataset.wave
    reliable = bool(dataset.oiii_reliable[idx])
    oiii = float(dataset.oiii_flux[idx])

    # Blank out-of-coverage pixels (the 0.0 sentinel) so the panels show real
    # data instead of the wide grid's zero padding.
    def covered(arr):
        a = np.asarray(arr, dtype=np.float32).copy()
        a[~valid] = np.nan
        return a

    panels = [
        (covered(raw),  "Input spectrum -- MAD-normalised full flux "
                        "(as stored in the parquet)"),
        (covered(x[0]), "Channel 0 -- MAD-normalised flux, arcsinh-compressed"),
        (covered(x[1]), "Channel 1 -- [OIII]-normalised flux, arcsinh-compressed"
                        + ("" if reliable else
                           "  (weak [OIII]: fallen back to channel 0)")),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (y, title) in zip(axes, panels):
        ax.plot(wave, y, lw=0.6, color="#1f4e79")
        ax.axhline(0.0, color="0.7", lw=0.6)
        if wave[0] <= 5007.0 <= wave[-1]:
            ax.axvline(5007.0, color="#c0392b", ls="--", lw=0.8, alpha=0.7,
                       label="[OIII] 5007")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("flux")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("rest-frame wavelength [A]")
    fig.suptitle(f"SSL input sanity check  --  spectrum #{idx} / {len(dataset):,}"
                 f"   ([OIII] flux={oiii:.3g}, reliable={reliable})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(out_dir, "ssl_input_sample.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ssl] input sanity plot -> {out_path}")


# Survey strings considered "old" (already pretrained on) for replay mixing.
# Anything not in this set is considered "new" (added in the extension parquet).
OLD_SURVEYS = {"sdss_dr7", "desi"}


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--resume-from", default=None,
                   help="Path to an existing SSL checkpoint. If given, the "
                        "encoder + decoder weights are loaded from it before "
                        "training continues. The optimizer is re-initialised "
                        "(with the lr below) so this is continual pretraining, "
                        "not a vanilla resume.")
    p.add_argument("--replay", action="store_true",
                   help="Use a WeightedRandomSampler that targets ~50%% old "
                        "(sdss_dr7+desi) and ~50%% new (everything else) "
                        "spectra per batch. Prevents catastrophic forgetting "
                        "during continual pretraining. Requires both pools "
                        "present in paths.ssl_parquets.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the config's ssl.learning_rate. Use a "
                        "smaller value (e.g. 1e-4) for continual pretraining.")
    p.add_argument("--num-epochs", type=int, default=None,
                   help="Override the config's ssl.num_epochs (e.g. 15-20 "
                        "for continual pretraining).")
    p.add_argument("--output-ckpt", default=None,
                   help="Override paths.ssl_checkpoint. Use this in Phase B "
                        "so the original encoder isn't overwritten.")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(os.path.join(BASE_DIR, "config_v2.yml"))
    paths, s = cfg["paths"], cfg["ssl"]

    # CLI overrides
    if args.lr is not None:
        s = dict(s); s["learning_rate"] = float(args.lr)
    if args.num_epochs is not None:
        s = dict(s); s["num_epochs"] = int(args.num_epochs)

    torch.manual_seed(s["seed"])
    np.random.seed(s["seed"])
    device = pick_device()
    print(f"[ssl] device: {device}")
    if args.resume_from:
        print(f"[ssl] CONTINUAL pretraining: resume from {args.resume_from}")
    if args.replay:
        print(f"[ssl] REPLAY enabled (50/50 old:new sampler)")
    print(f"[ssl] lr={s['learning_rate']:.2e}  num_epochs={s['num_epochs']}")

    # ---- data -----------------------------------------------------------
    parquets = [os.path.join(BASE_DIR, q) for q in paths["ssl_parquets"]]
    parquets = [q for q in parquets if os.path.exists(q)]
    if not parquets:
        raise FileNotFoundError("No SSL parquet found. Check paths.ssl_parquets "
                                "in config_v2.yml")
    print(f"[ssl] pooling {len(parquets)} parquet catalog(s)")

    dataset = SSLSpectraDataset(
        parquets,
        channel1_scale=None,                 # calibrated from this pool
        oiii_snr_min=cfg["preprocessing"]["oiii_snr_min"],
        max_rows=s.get("max_rows"),
    )

    # Persist the channel-1 [OIII] scale so Stage 2 uses the identical value.
    save_norm_stats(
        os.path.join(BASE_DIR, paths["norm_stats"]),
        dataset.channel1_scale,
        extra={"oiii_snr_min": cfg["preprocessing"]["oiii_snr_min"],
               "n_spectra": int(len(dataset))},
    )

    n_val = max(1, int(s["val_frac"] * len(dataset)))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(s["seed"])
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"[ssl] train {n_train:,}  val {n_val:,}")

    # ---- replay sampler (50/50 old:new) for continual pretraining -----
    if args.replay:
        # SSLSpectraDataset carries .meta with a 'survey' column for every row.
        surveys = np.asarray(dataset.meta["survey"].values)
        is_old_full = np.isin(surveys, list(OLD_SURVEYS))
        # train_set is a torch.utils.data.Subset -- map subset indices back.
        train_idx = np.asarray(train_set.indices)
        is_old_train = is_old_full[train_idx]
        n_old = int(is_old_train.sum())
        n_new = int((~is_old_train).sum())
        if n_old == 0 or n_new == 0:
            print(f"[ssl]   replay requested but pool has only one source "
                  f"(old={n_old}, new={n_new}); falling back to shuffle")
            train_loader = DataLoader(
                train_set, batch_size=s["batch_size"], shuffle=True,
                num_workers=s["num_workers"], drop_last=True,
            )
        else:
            w_old = 0.5 / n_old
            w_new = 0.5 / n_new
            weights = np.where(is_old_train, w_old, w_new).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights),
                num_samples=len(train_idx),    # epoch size = dataset size
                replacement=True,
            )
            print(f"[ssl]   replay sampler: old(n={n_old}) old-weight={w_old:.2e}  "
                  f"new(n={n_new}) new-weight={w_new:.2e}  "
                  f"(expect ~{s['batch_size']//2} of each per batch)")
            train_loader = DataLoader(
                train_set, batch_size=s["batch_size"], sampler=sampler,
                num_workers=s["num_workers"], drop_last=True,
            )
    else:
        train_loader = DataLoader(
            train_set, batch_size=s["batch_size"], shuffle=True,
            num_workers=s["num_workers"], drop_last=True,
        )
    val_loader = DataLoader(val_set, batch_size=s["batch_size"], shuffle=False,
                            num_workers=s["num_workers"])

    # ---- model ----------------------------------------------------------
    model = MaskedSpectraAutoencoder(in_channels=2).to(device)

    # Phase B: warm-start from an existing checkpoint.
    if args.resume_from:
        resume_path = (args.resume_from if os.path.isabs(args.resume_from)
                       else os.path.join(BASE_DIR, args.resume_from))
        if not os.path.exists(resume_path):
            sys.exit(f"[ssl] --resume-from path not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        missing_enc = model.encoder.load_state_dict(
            ckpt["encoder_state_dict"], strict=False,
        )
        missing_dec = model.decoder.load_state_dict(
            ckpt["decoder_state_dict"], strict=False,
        ) if "decoder_state_dict" in ckpt else None
        print(f"[ssl] loaded encoder from {resume_path}  "
              f"(missing={list(missing_enc.missing_keys)[:3]}...)")
        if missing_dec is not None:
            print(f"[ssl] loaded decoder from {resume_path}  "
                  f"(missing={list(missing_dec.missing_keys)[:3]}...)")
        else:
            print(f"[ssl] WARNING: checkpoint has no decoder_state_dict; "
                  f"decoder starts random")

    opt = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"],
                            weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["num_epochs"])

    out_dir = os.path.join(BASE_DIR, paths["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = (
        args.output_ckpt if args.output_ckpt is not None
        else paths["ssl_checkpoint"]
    )
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)
    print(f"[ssl] checkpoint will be written to: {ckpt_path}")

    # ---- pre-training input sanity check --------------------------------
    plot_input_sample(dataset, out_dir)

    history = {"train": [], "val": []}
    best_val = float("inf")

    # ---- training loop --------------------------------------------------
    for epoch in range(s["num_epochs"]):
        t0 = time.time()
        model.train()
        running, nb = 0.0, 0
        for x, valid in train_loader:
            x = x.to(device)
            valid = valid.to(device)
            x_masked, span = apply_span_mask(x, valid, s["mask_ratio"],
                                             s["min_span"], s["max_span"])
            recon = model(x_masked)
            loss = masked_mse(recon, x, span, valid)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(nb, 1)

        model.eval()
        vrun, vnb = 0.0, 0
        with torch.no_grad():
            for x, valid in val_loader:
                x = x.to(device)
                valid = valid.to(device)
                x_masked, span = apply_span_mask(x, valid, s["mask_ratio"],
                                                 s["min_span"], s["max_span"])
                recon = model(x_masked)
                vrun += masked_mse(recon, x, span, valid).item()
                vnb += 1
        val_loss = vrun / max(vnb, 1)
        sched.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"[ssl] epoch {epoch + 1:3d}/{s['num_epochs']}  "
              f"train {train_loss:.5f}  val {val_loss:.5f}  "
              f"({time.time() - t0:.0f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"encoder_state_dict": model.encoder.state_dict(),
                        "decoder_state_dict": model.decoder.state_dict(),
                        "channel1_scale": dataset.channel1_scale,
                        "epoch": epoch + 1,
                        "val_loss": val_loss}, ckpt_path)
            print(f"[ssl]   saved best encoder -> {ckpt_path}")

    # ---- loss curve -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("masked reconstruction MSE")
    plt.title("Stage 1 -- self-supervised pretraining")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ssl_loss_curve.png"), dpi=150)
    print(f"[ssl] done. best val loss = {best_val:.5f}")
    print(f"[ssl] encoder ready for Stage 2: {ckpt_path}")


if __name__ == "__main__":
    main()
