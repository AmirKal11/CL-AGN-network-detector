"""
predict.py — CL-AGN inference on spectrum pairs
=================================================
Given a directory of FITS files and a CSV that pairs them, runs the
continuum_subtracted_full_dr7 Siamese model and writes a ranked CSV of
predicted CL-AGN probabilities.

Input CSV columns
-----------------
Required:
    file1       basename (or relative path) of the first-epoch FITS file
    file2       basename (or relative path) of the second-epoch FITS file

Optional (passed through to output; used for FITS fallback if absent):
    z1, z2      redshifts of each epoch (falls back to FITS header)
    z           shared redshift (used for both epochs if z1/z2 absent)
    ra, dec     sky coordinates
    object_id   identifier for the object
    <any other> passed through unchanged

Output CSV columns
------------------
    All input columns +
    prob            P(CL-AGN) from the Siamese head
    label           1 if prob >= threshold, else 0
    (rows are sorted by prob descending)

Usage
-----
    python src/predict.py \\
        --spectra-dir  data/spectra/ \\
        --pairs-csv    data/pairs.csv \\
        --output       results/predictions.csv \\
        [--model-dir   models/continuum_subtracted_full_dr7] \\
        [--threshold   0.547] \\
        [--batch-size  512] \\
        [--device      mps|cuda|cpu]

Performance
-----------
Spectra are processed in parallel (joblib) and cached so each FITS file is
read at most once even if it appears in multiple pairs. On Apple Silicon with
MPS the model inference is fast; CPU fallback is used automatically when MPS
is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure src/ is importable when called from the project root
# ---------------------------------------------------------------------------
SRC = Path(__file__).parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_preprocessing import process_single_spectrum          # noqa: E402
from preprocessing_oiii import (                                # noqa: E402
    continuum_subtract,
    mad_normalize,
    build_two_channel,
    valid_from_flux,
    load_norm_stats,
)
from architectures_v2 import SiameseChangeNet                   # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "models" / "continuum_subtracted_full_dr7"
DEFAULT_THRESHOLD = 0.547   # val-set F2 optimum of the best run
OIII_SNR_MIN = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(spec_dir: Path, filename: str) -> Path | None:
    """Find a FITS file: try as-is, then under spec_dir."""
    p = Path(filename)
    if p.is_file():
        return p
    q = spec_dir / p.name
    if q.is_file():
        return q
    q2 = spec_dir / filename
    if q2.is_file():
        return q2
    return None


def _process_one(fits_path: Path, z: float | None) -> np.ndarray | None:
    """
    FITS → rest-frame grid → continuum-subtract → MAD-norm → float32[4096].
    Returns None on any failure.
    """
    result = process_single_spectrum(str(fits_path), z=z)
    if result is None:
        return None
    raw = result["flux_array"]                     # float32 [4096], raw physical flux
    valid = valid_from_flux(raw)
    cs = continuum_subtract(raw, valid=valid)
    madnorm, _ = mad_normalize(cs, valid=valid)
    return madnorm                                 # float32 [4096]


def _build_channel(madnorm: np.ndarray, channel1_scale: float) -> np.ndarray:
    """MAD-norm spectrum → 2-channel tensor [2, 4096]."""
    valid = valid_from_flux(madnorm)
    x, _ = build_two_channel(
        madnorm,
        channel1_scale=channel1_scale,
        oiii_snr_min=OIII_SNR_MIN,
        valid=valid,
    )
    return x    # float32 [2, 4096]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(
    spectra_dir: str | Path,
    pairs_csv: str | Path,
    output: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 512,
    device: str | None = None,
    n_jobs: int = -2,
) -> pd.DataFrame:
    """
    Run CL-AGN inference on all pairs in pairs_csv and write predictions.csv.

    Returns the output DataFrame (sorted by prob descending).
    """
    spectra_dir = Path(spectra_dir)
    model_dir   = Path(model_dir)
    pairs_csv   = Path(pairs_csv)
    output      = Path(output)

    # ---- device ------------------------------------------------------------
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dev = torch.device(device)
    print(f"[predict] device: {dev}")

    # ---- norm stats --------------------------------------------------------
    norm_stats_path = model_dir / "norm_stats.json"
    ns = load_norm_stats(str(norm_stats_path))
    channel1_scale = float(ns.get("channel1_scale", 1.0))
    print(f"[predict] channel1_scale = {channel1_scale:.6f}")

    # ---- model -------------------------------------------------------------
    ckpt_path = model_dir / "siamese_changenet.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = SiameseChangeNet(encoder_freeze=False)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(dev).eval()
    print(f"[predict] loaded model from {ckpt_path}")

    # ---- pairs CSV ---------------------------------------------------------
    df = pd.read_csv(pairs_csv)
    required = {"file1", "file2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    print(f"[predict] {len(df):,} pairs to score")

    # ---- resolve FITS paths ------------------------------------------------
    df["_path1"] = df["file1"].apply(lambda f: _resolve_path(spectra_dir, f))
    df["_path2"] = df["file2"].apply(lambda f: _resolve_path(spectra_dir, f))

    n_missing = df["_path1"].isna().sum() + df["_path2"].isna().sum()
    if n_missing:
        print(f"[predict] WARNING: {n_missing} spectrum file(s) not found — "
              "those pairs will be skipped (prob=NaN)")

    # ---- deduplicate: process each unique FITS only once -------------------
    unique_specs: dict[str, tuple[Path | None, float | None]] = {}
    for _, row in df.iterrows():
        for col, zcol in [("_path1", "z1"), ("_path2", "z2")]:
            p = row[col]
            if p is None:
                continue
            key = str(p)
            if key not in unique_specs:
                z = None
                for zc in (zcol, "z"):
                    if zc in row and pd.notna(row.get(zc)):
                        try:
                            z = float(row[zc])
                            break
                        except (ValueError, TypeError):
                            pass
                unique_specs[key] = (p, z)

    print(f"[predict] processing {len(unique_specs):,} unique spectra "
          f"(n_jobs={n_jobs}) ...")

    # parallel FITS read + continuum-subtract + MAD-norm
    keys   = list(unique_specs.keys())
    paths  = [unique_specs[k][0] for k in keys]
    zvals  = [unique_specs[k][1] for k in keys]

    madnorms_raw = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_one)(p, z)
        for p, z in tqdm(zip(paths, zvals), total=len(keys),
                         desc="reading FITS", unit="spec")
    )
    madnorm_cache: dict[str, np.ndarray | None] = dict(zip(keys, madnorms_raw))

    # build 2-channel tensors (fast, no I/O — keep sequential)
    print("[predict] building 2-channel inputs ...")
    x_cache: dict[str, np.ndarray | None] = {}
    for key, madnorm in madnorm_cache.items():
        if madnorm is None:
            x_cache[key] = None
        else:
            x_cache[key] = _build_channel(madnorm, channel1_scale)

    # ---- batch inference ---------------------------------------------------
    print(f"[predict] running inference (batch_size={batch_size}) ...")
    probs = []

    x1_buf, x2_buf, idx_buf = [], [], []

    def _flush():
        if not x1_buf:
            return
        t1 = torch.from_numpy(np.stack(x1_buf)).to(dev)
        t2 = torch.from_numpy(np.stack(x2_buf)).to(dev)
        with torch.no_grad():
            logits = model(t1, t2).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
        for i, idx in enumerate(idx_buf):
            probs[idx] = float(p[i])
        x1_buf.clear(); x2_buf.clear(); idx_buf.clear()

    probs = [float("nan")] * len(df)
    for row_idx, row in tqdm(df.iterrows(), total=len(df),
                              desc="scoring pairs", unit="pair"):
        p1 = row["_path1"]
        p2 = row["_path2"]
        if p1 is None or p2 is None:
            continue
        x1 = x_cache.get(str(p1))
        x2 = x_cache.get(str(p2))
        if x1 is None or x2 is None:
            continue

        x1_buf.append(x1)
        x2_buf.append(x2)
        idx_buf.append(row_idx)

        if len(x1_buf) >= batch_size:
            _flush()

    _flush()

    # ---- assemble output ---------------------------------------------------
    out = df.drop(columns=["_path1", "_path2"]).copy()
    out["prob"]  = probs
    out["label"] = (out["prob"] >= threshold).astype(int)
    out = out.sort_values("prob", ascending=False).reset_index(drop=True)

    n_scored  = out["prob"].notna().sum()
    n_skipped = len(out) - n_scored
    n_pos     = int((out["label"] == 1).sum())
    print(f"\n[predict] done — {n_scored:,} pairs scored, "
          f"{n_skipped:,} skipped (missing/failed), "
          f"{n_pos:,} predicted CL-AGN (threshold={threshold})")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[predict] results → {output}")
    return out


# ---------------------------------------------------------------------------
# IDE / notebook config — edit these paths when running directly in an IDE.
# When running from the terminal these are ignored (CLI args take over).
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[1]   # repo root, auto-detected

IDE_CONFIG = dict(
    spectra_dir = PROJECT_ROOT / "data" / "spectra",
    pairs_csv   = PROJECT_ROOT / "data" / "pairs.csv",
    output      = PROJECT_ROOT / "results" / "predictions.csv",
    model_dir   = DEFAULT_MODEL_DIR,
    threshold   = DEFAULT_THRESHOLD,
    batch_size  = 512,
    device      = None,    # None = auto (mps → cuda → cpu)
    n_jobs      = -2,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Score spectrum pairs for CL-AGN transitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spectra-dir",  required=True,
                   help="Directory containing FITS files.")
    p.add_argument("--pairs-csv",    required=True,
                   help="CSV with columns: file1, file2 [, z1, z2, ra, dec, ...].")
    p.add_argument("--output",       default="results/predictions.csv",
                   help="Output CSV path.")
    p.add_argument("--model-dir",    default=str(DEFAULT_MODEL_DIR),
                   help="Model directory (must contain siamese_changenet.pth "
                        "and norm_stats.json).")
    p.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD,
                   help="Classification threshold (prob >= threshold → label=1).")
    p.add_argument("--batch-size",   type=int,   default=512,
                   help="Inference batch size.")
    p.add_argument("--device",       default=None,
                   choices=["mps", "cuda", "cpu"],
                   help="Compute device (auto-detected if omitted).")
    p.add_argument("--n-jobs",       type=int,   default=-2,
                   help="Parallel workers for FITS reading (-2 = all CPUs minus 1).")
    return p.parse_args()


if __name__ == "__main__":
    # Running from terminal → use CLI args.
    # Running from an IDE (no sys.argv) → use IDE_CONFIG above.
    import sys
    if len(sys.argv) > 1:
        args = _parse_args()
        predict(
            spectra_dir = args.spectra_dir,
            pairs_csv   = args.pairs_csv,
            output      = args.output,
            model_dir   = args.model_dir,
            threshold   = args.threshold,
            batch_size  = args.batch_size,
            device      = args.device,
            n_jobs      = args.n_jobs,
        )
    else:
        predict(**IDE_CONFIG)
