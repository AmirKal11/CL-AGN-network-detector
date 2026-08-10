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
        --spectra-dir  data/sdssv_desi_crossmatch/ \\
        --pairs-csv    data/sdssv_desi_crossmatch.csv \\
        --output       results/predictions.csv \\
        [--model-dir   models/continuum_subtracted_full_dr7] \\
        [--threshold   0.547] \\
        [--batch-size  512] \\
        [--device      mps|cuda|cpu] \\
        [--plot-dir    data/plots]

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    print("[predict] WARNING: plotly not installed — "
          "interactive HTML plots unavailable. "
          "Install with: pip install plotly")

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
    measure_oiii_flux,
    load_norm_stats,
    MASTER_GRID,
)
# Single source of truth for the 2-channel build: the SAME function the SSL and
# Siamese datasets use (datasets_v2.SSLSpectraDataset.__getitem__ /
# CLAGNPairDataset.__getitem__). Do not re-implement it here -- an earlier
# duplicate drifted and silently fed the model a different ch1.
from datasets_v2 import _two_channel                            # noqa: E402
from architectures_v2 import SiameseChangeNet                   # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# FINAL model (PR-AUC 0.832, recall 88.6% @ FPR 2.4%). Previously defaulted to
# models/weighted_loss_per_Z — a REJECTED ablation — which contradicted the
# --model-dir example in this file's own usage string and silently ran
# inference at that run's threshold (0.657) instead of the deployed 0.547.
DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "models" / "continuum_subtracted_full_dr7"
# Fallback only: the real value is read from ckpt["best_threshold"] (line 224).
# 0.547 = max recall s.t. FPR <= 5% on the SDSS-V validation subset.
DEFAULT_THRESHOLD = 0.547  # best_threshold saved in siamese_changenet.pth
OIII_SNR_MIN = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASENAME_INDEX: dict[Path, dict[str, Path]] = {}


def _basename_index(spec_dir: Path) -> dict[str, Path]:
    """Lazy recursive basename -> path index for spec_dir (built once, cached)."""
    idx = _BASENAME_INDEX.get(spec_dir)
    if idx is None:
        idx = {}
        for ext in ("*.fits", "*.fits.gz", "*.fit"):
            for f in spec_dir.rglob(ext):
                idx.setdefault(f.name, f)
        _BASENAME_INDEX[spec_dir] = idx
    return idx


def _resolve_path(spec_dir: Path, filename: str) -> Path | None:
    """
    Find a FITS file, searching in order:
      1. As an absolute/relative path (as-is)
      2. spec_dir / <path as stored in the CSV>   ← pair CSVs store sub-paths
         (e.g. 'desi/clagn_desi_dr16_sample_paper2/sdss/302_1584_52943.fits')
      3. spec_dir / basename, spec_dir.parent / basename
      4. recursive basename search under spec_dir

    Mirrors datasets_v2._resolve, which the pair pipeline uses: without step 2
    a CSV that stores relative sub-paths resolves nothing at all.
    """
    p = Path(filename)
    if p.is_file():
        return p

    for search_dir in (spec_dir, spec_dir.parent):
        candidate = search_dir / p                     # as-stored relative path
        if candidate.is_file():
            return candidate
    name = p.name
    for search_dir in (spec_dir, spec_dir.parent):
        candidate = search_dir / name                  # bare basename
        if candidate.is_file():
            return candidate
    return _basename_index(spec_dir).get(name)         # recursive fallback


def _process_one(fits_path: Path, z: float | None) -> tuple[
    "np.ndarray | None", "np.ndarray | None", "np.ndarray | None", "float | None"
]:
    """
    FITS → rest-frame grid → continuum-subtract → MAD-norm → float32[4096].

    Mirrors datasets_v2.load_or_build_pair_arrays step for step: same
    processing chain, and crucially the same validity mask -- the coverage
    mask computed by process_single_spectrum (isfinite on the interpolated
    grid), NOT valid_from_flux(flux != 0). See the note in
    process_single_spectrum for why the two differ.

    Returns (madnorm, raw_flux, valid, snr).
      madnorm  : MAD-normalised continuum-subtracted flux  (ch0 input)
      raw_flux : raw physical flux, zero-filled gaps       (ch1 OIII anchor)
      valid    : per-pixel coverage mask                   (bool [4096])
      snr      : median S/N from FITS header, or None
    All four are None on any failure.
    """
    result = process_single_spectrum(str(fits_path), z=z)
    if result is None:
        return None, None, None, None
    raw = result["flux_array"]                     # float32 [4096], raw physical flux
    snr = result.get("snr")                        # float | None
    valid = np.asarray(result["valid"], dtype=bool)
    cs = continuum_subtract(raw, valid=valid)
    madnorm, _ = mad_normalize(cs, valid=valid)
    return madnorm, raw, valid, snr


def _oiii_of(raw_flux: np.ndarray, valid: np.ndarray) -> tuple[float, float, bool]:
    """[O III] on RAW physical flux with the coverage mask — as the pair pipeline does."""
    oiii_flux, oiii_snr = measure_oiii_flux(raw_flux, valid=valid)
    return float(oiii_flux), float(oiii_snr), bool(oiii_snr >= OIII_SNR_MIN
                                                   and oiii_flux > 1e-6)


def _build_channel(raw_flux: np.ndarray, madnorm: np.ndarray,
                   valid: np.ndarray, channel1_scale: float) -> np.ndarray:
    """(raw_flux, madnorm, valid) → 2-channel tensor [2, 4096] via datasets_v2._two_channel."""
    oiii_flux, _, oiii_reliable = _oiii_of(raw_flux, valid)
    return _two_channel(raw_flux, madnorm, oiii_flux, oiii_reliable, channel1_scale)


def _build_channel_with_info(
    raw_flux: np.ndarray, madnorm: np.ndarray, valid: np.ndarray, channel1_scale: float
) -> tuple["np.ndarray", dict]:
    """Same as _build_channel but also returns the OIII diagnostic info dict."""
    oiii_flux, oiii_snr, oiii_reliable = _oiii_of(raw_flux, valid)
    x = _two_channel(raw_flux, madnorm, oiii_flux, oiii_reliable, channel1_scale)
    info = {
        "oiii_flux": oiii_flux,
        "oiii_snr": oiii_snr,
        "oiii_reliable": oiii_reliable,
    }
    return x, info    # float32 [2, 4096], {oiii_flux, oiii_snr, oiii_reliable}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(
    spectra_dir: str | Path,
    pairs_csv: str | Path,
    output: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    threshold: float | None = None,   # None = load best_threshold from checkpoint
    batch_size: int = 512,
    device: str | None = None,
    n_jobs: int = -2,
    min_snr_pairs: float | None = 4.0,   # minimum SNR for both epochs (None = no cut)
    max_dz: float | None = 0.05,         # maximum |z1-z2| (None = no cut)
    max_z: float | None = 0.8,           # maximum redshift (None = no cut)
) -> pd.DataFrame:
    """
    Run CL-AGN inference on all pairs in pairs_csv and write predictions.csv.

    threshold=None (default) loads best_threshold directly from the checkpoint,
    which is the exact value optimised during training.  Pass an explicit float
    to override.

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

    # ---- norm stats (fallback only — checkpoint is primary) ----------------
    norm_stats_path = model_dir / "norm_stats.json"
    # (channel1_scale is read below after the checkpoint is loaded)
    _ = norm_stats_path  # resolved later

    # ---- model + checkpoint (load once, extract all metadata) --------------
    ckpt_path = model_dir / "siamese_changenet.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    # Use threshold saved in checkpoint when caller passes None
    if threshold is None:
        threshold = float(ckpt.get("best_threshold", DEFAULT_THRESHOLD))
        print(f"[predict] threshold loaded from checkpoint: {threshold:.6f}")
    else:
        print(f"[predict] threshold (caller-supplied): {threshold:.6f}")

    # channel1_scale: prefer checkpoint value, fall back to norm_stats.json
    channel1_scale = float(ckpt.get("channel1_scale", 0.0))
    if channel1_scale == 0.0:
        norm_stats_path = model_dir / "norm_stats.json"
        ns = load_norm_stats(str(norm_stats_path))
        channel1_scale = float(ns.get("channel1_scale", 1.0))
    print(f"[predict] channel1_scale = {channel1_scale:.6f}")

    model = SiameseChangeNet()
    model.load_state_dict(state, strict=False)
    model.to(dev).eval()
    print(f"[predict] loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})"
          f"  val_AUC={ckpt.get('val_auc', float('nan')):.4f}")

    # ---- pairs CSV ---------------------------------------------------------
    df = pd.read_csv(pairs_csv)

    # ---- auto-remap crossmatch-style column names --------------------------
    # Transparently accept both file1/file2 (canonical) and the column names
    # produced directly by the DR16×DR20 crossmatch step, so no external
    # renaming script is needed.
    _CROSSMATCH_REMAP = {
        "spec_file_dr16": "file1",
        "spec_file_dr20": "file2",
        "ra_dr16":        "ra",
        "dec_dr16":       "dec",
        "z_dr16":         "z1",
        "z_dr20":         "z2",
        "mjd_dr16":       "mjd1",
        "mjd_dr20":       "mjd2",
        "snr_dr16":       "snr1",
        "snr_dr20":       "snr2",
    }
    _remap = {k: v for k, v in _CROSSMATCH_REMAP.items()
              if k in df.columns and v not in df.columns}
    if _remap:
        print(f"[predict] remapping columns: {_remap}")
        df = df.rename(columns=_remap)

    required = {"file1", "file2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    print(f"[predict] {len(df):,} pairs loaded")

    # ---- quality pre-filter (applied before any FITS I/O) ------------------
    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    # SNR cut: require both epochs above min_snr_pairs
    if min_snr_pairs is not None:
        for scol in ("snr1", "snr2"):
            if scol in df.columns:
                mask &= df[scol].fillna(0) >= min_snr_pairs
            else:
                print(f"[predict] WARNING: '{scol}' column not found — SNR cut skipped for this epoch")

    # Redshift consistency cut: |z1 - z2| < max_dz
    if max_dz is not None and "z1" in df.columns and "z2" in df.columns:
        dz = (df["z1"] - df["z2"]).abs()
        mask &= dz < max_dz

    # Maximum redshift cut (to prevent domain shift on high-z targets)
    if max_z is not None:
        for zcol in ("z1", "z2"):
            if zcol in df.columns:
                mask &= df[zcol].fillna(0) <= max_z

    df = df[mask].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[predict] quality filter: {n_dropped:,} pairs dropped "
              f"({len(df):,} remaining) — "
              f"SNR≥{min_snr_pairs}, |Δz|<{max_dz}, z≤{max_z}")
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

    raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_one)(p, z)
        for p, z in tqdm(zip(paths, zvals), total=len(keys),
                         desc="reading FITS", unit="spec")
    )
    # Unzip (madnorm, raw_flux, valid, snr) 4-tuples
    if raw_results:
        madnorms_raw, raws_raw, valids_raw, snrs_raw = zip(*raw_results)
    else:
        madnorms_raw, raws_raw, valids_raw, snrs_raw = [], [], [], []
    madnorm_cache: dict[str, np.ndarray | None] = dict(zip(keys, madnorms_raw))
    raw_cache:     dict[str, np.ndarray | None] = dict(zip(keys, raws_raw))
    valid_cache:   dict[str, np.ndarray | None] = dict(zip(keys, valids_raw))
    snr_cache:     dict[str, float | None]      = dict(zip(keys, snrs_raw))

    # build 2-channel tensors (fast, no I/O — keep sequential)
    print("[predict] building 2-channel inputs ...")
    x_cache: dict[str, np.ndarray | None] = {}
    for key, madnorm in madnorm_cache.items():
        raw_flux = raw_cache.get(key)
        valid    = valid_cache.get(key)
        if madnorm is None or raw_flux is None or valid is None:
            x_cache[key] = None
        else:
            x_cache[key] = _build_channel(raw_flux, madnorm, valid, channel1_scale)

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

    # ---- SNR columns -------------------------------------------------------
    # Prefer FITS-derived SNR from the cache; fall back to any snr1/snr2
    # already present in the input CSV (e.g. from the crossmatch step).
    snr1_fits = [
        snr_cache.get(str(row["_path1"])) if row["_path1"] is not None else None
        for _, row in df.iterrows()
    ]
    snr2_fits = [
        snr_cache.get(str(row["_path2"])) if row["_path2"] is not None else None
        for _, row in df.iterrows()
    ]

    def _merge_snr(fits_vals, csv_col):
        """Use FITS-derived value when available, else keep existing CSV value."""
        csv_vals = out[csv_col].tolist() if csv_col in out.columns else [None] * len(out)
        return [
            fv if fv is not None else cv
            for fv, cv in zip(fits_vals, csv_vals)
        ]

    out["snr1"] = _merge_snr(snr1_fits, "snr1")
    out["snr2"] = _merge_snr(snr2_fits, "snr2")

    n_snr1 = sum(v is not None and not (isinstance(v, float) and np.isnan(v))
                 for v in out["snr1"])
    n_snr2 = sum(v is not None and not (isinstance(v, float) and np.isnan(v))
                 for v in out["snr2"])
    print(f"[predict] SNR available — snr1: {n_snr1}/{len(out)} | snr2: {n_snr2}/{len(out)}")

    # ---- reorder: put snr1/snr2 just before prob --------------------------
    cols = [c for c in out.columns if c not in ("snr1", "snr2", "prob", "label")]
    out  = out[cols + ["snr1", "snr2", "prob", "label"]]

    out = out.sort_values("prob", ascending=False).reset_index(drop=True)

    n_scored  = out["prob"].notna().sum()
    n_skipped = len(out) - n_scored
    n_pos     = int((out["label"] == 1).sum())
    print(f"[predict] done — {n_scored:,} pairs scored, "
          f"{n_skipped:,} skipped (missing/failed), "
          f"{n_pos:,} predicted CL-AGN (threshold={threshold})")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[predict] results → {output}")
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _read_raw_spectrum(fits_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read (wavelength_Å, flux) from a FITS file.

    Handles two formats:
      • SDSS-V  — HDU[1] BinTable with LOGLAM (log10 Å) + FLUX columns
      • DESI    — HDU["SPECTRUM"] BinTable with WAVELENGTH (Å) + FLUX columns

    Returns (wave, flux) as float64 arrays, or None on failure.
    """
    try:
        with fits.open(str(fits_path)) as hdul:
            # Try named SPECTRUM extension first (DESI custom format)
            if "SPECTRUM" in hdul:
                data = hdul["SPECTRUM"].data
            else:
                data = hdul[1].data

            flux = np.asarray(data["FLUX"], dtype=np.float64)
            names_lower = [n.lower() for n in data.names]

            if "loglam" in names_lower:
                wave = 10.0 ** np.asarray(data["LOGLAM"], dtype=np.float64)
            elif "wavelength" in names_lower:
                wave = np.asarray(data["WAVELENGTH"], dtype=np.float64)
            else:
                hdr  = hdul[1].header
                wave = hdr["CRVAL1"] + np.arange(len(flux)) * hdr["CDELT1"]

            # Try reading ivar for error shading
            ivar = None
            if "ivar" in names_lower:
                ivar_raw = np.asarray(data["IVAR"], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma = np.where(ivar_raw > 0, 1.0 / np.sqrt(ivar_raw), np.nan)
                ivar = sigma

            return wave, flux, ivar
    except Exception:
        return None, None, None


def plot_cl_agn_pairs(
    predictions: pd.DataFrame | str | Path,
    spectra_dir: str | Path,
    plots_dir: str | Path = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_prob: float | None = None,
    min_snr: float | None = None,
) -> int:
    """
    Plot all predicted CL-AGN pairs and save one PNG per pair.

    Both spectra (SDSS-V epoch 1 and DESI epoch 2) are overlaid on the same
    rest-frame wavelength axis.  Dashed vertical lines mark the rest-frame
    positions of the main broad AGN emission lines (Mg II, Hb, Ha).

    Parameters
    ----------
    predictions : DataFrame or path to the predictions CSV produced by predict().
    spectra_dir : Directory containing the FITS files.
    plots_dir   : Output directory.  Defaults to <project_root>/data/plots/.
    threshold   : Minimum prob to plot (used when min_prob is None).
    min_prob    : Explicit minimum P(CL-AGN) threshold.  Overrides `threshold`
                  and always filters by prob (ignoring any pre-computed label
                  column).  Set to e.g. 0.8 to plot only high-confidence pairs.
    min_snr     : If set, skip pairs where either epoch has a median per-pixel
                  SNR (= median(flux * sqrt(ivar))) below this value.  Requires
                  IVAR data in the FITS file; pairs without IVAR are kept.

    Returns
    -------
    Number of plots saved.
    """
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_csv(predictions)

    spectra_dir = Path(spectra_dir)
    if plots_dir is None:
        plots_dir = Path(__file__).parents[1] / "data" / "plots_continuum_norm"
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine positives — min_prob always filters by prob directly;
    # threshold is the fallback when min_prob is not supplied.
    _prob_cut = min_prob if min_prob is not None else threshold
    if "prob" in predictions.columns:
        positives = predictions[predictions["prob"] >= _prob_cut].copy()
    else:
        raise ValueError("predictions must contain a 'prob' column.")
    print(f"[plot] {len(positives)} pairs with prob >= {_prob_cut}")

    # Deduplicate: one DESI spectrum = one unique object.
    # The crossmatch can pair N SDSS-V epochs with the same DESI target,
    # so keep only the highest-probability pair per unique file2.
    if "prob" in positives.columns and "file2" in positives.columns:
        before = len(positives)
        positives = (
            positives
            .sort_values("prob", ascending=False)
            .drop_duplicates(subset="file2", keep="first")
            .reset_index(drop=True)
        )
        print(f"[plot] deduplicated {before} → {len(positives)} unique objects "
              f"(best pair per DESI spectrum kept)")

    print(f"[plot] {len(positives)} CL-AGN candidates to plot -> {plots_dir}")


    # -- AGN broad emission lines (rest-frame A) ----------------------------
    EMISSION_LINES = {
        "Mg II":       2799,
        "Hβ":          4861,
        "[O III]":     5007,
        "Hα":          6563,
    }

    # -- Colours -----------------------------------------------------------
    COLOR_SDSSV = "#1565C0"   # deep blue
    COLOR_DESI  = "#C62828"   # deep red
    COLOR_LINES = "#888888"   # grey for line markers

    def _extract_mjd(filename: str, fits_path: "Path | None") -> str:
        """
        Return a short MJD string for use in legend labels.

        Strategy:
          1. Parse from SDSS-V filename: spec-<plate>-<MJD>-<fiber>.fits
          2. Fall back to FITS header keys MJD, MJD-OBS, or MJDOBS.
          3. Return '?' if nothing is found.
        """
        stem = Path(filename).stem          # e.g. spec-015000-59146-4375786564
        parts = stem.split("-")
        # SDSS-V convention: spec-<plate>-<MJD>-<fiber>
        if len(parts) >= 3 and parts[0].lower() == "spec":
            try:
                return str(int(parts[2]))
            except ValueError:
                pass
        # Fallback: check FITS header
        if fits_path is not None:
            try:
                with fits.open(str(fits_path)) as hdul:
                    hdr = hdul[0].header
                    for key in ("MJD", "MJD-OBS", "MJDOBS"):
                        val = hdr.get(key)
                        if val is not None:
                            return str(int(float(val)))
            except Exception:
                pass
        return "?"

    def _compute_snr(flux: np.ndarray | None,
                     sigma: np.ndarray | None) -> float:
        """
        Compute median per-pixel SNR from raw flux and sigma (= 1/sqrt(ivar)).

        Returns NaN when IVAR data is unavailable (sigma is None), so the
        pair is NOT rejected on an SNR cut — only pairs with measured low SNR
        are filtered out.
        """
        if flux is None or sigma is None:
            return float("nan")
        valid = (sigma > 0) & np.isfinite(sigma) & np.isfinite(flux)
        if valid.sum() < 10:
            return float("nan")
        return float(np.median(np.abs(flux[valid]) / sigma[valid]))

    saved   = 0
    skipped_snr = 0
    skipped_z   = 0

    for _, row in tqdm(positives.iterrows(), total=len(positives),
                       desc="Plotting pairs", unit="pair"):
        file1 = str(row.get("file1", ""))
        file2 = str(row.get("file2", ""))

        path1 = _resolve_path(spectra_dir, file1)
        path2 = _resolve_path(spectra_dir, file2)

        ra   = row.get("ra",  float("nan"))
        dec  = row.get("dec", float("nan"))
        z1   = row.get("z1",  row.get("z",  float("nan")))
        z2   = row.get("z2",  row.get("z",  float("nan")))
        prob = row.get("prob", float("nan"))

        # -- Redshift filter ------------------------------------------------
        if np.isfinite(z1) and np.isfinite(z2) and abs(z1 - z2) > 0.05:
            skipped_z += 1
            continue

        # Prefer MJD values from predictions/pairs DataFrame if present.
        # Check both the canonical renamed columns (mjd1/mjd2) and the raw
        # crossmatch column names (mjd_dr16/mjd_dr20) as fallbacks.
        mjd1_val = row.get("mjd1")
        if pd.isna(mjd1_val):
            mjd1_val = row.get("mjd_dr16")
        mjd2_val = row.get("mjd2")
        if pd.isna(mjd2_val):
            # support both old (mjd_sdssv) and new (mjd_dr20) column names
            mjd2_val = row.get("mjd_dr20") or row.get("mjd_sdssv")
        mjd1 = str(int(float(mjd1_val))) if pd.notna(mjd1_val) else _extract_mjd(file1, path1)
        mjd2 = str(int(float(mjd2_val))) if pd.notna(mjd2_val) else _extract_mjd(file2, path2)

        tag       = f"ra{ra:.4f}_dec{dec:+.4f}".replace("+", "p").replace("-", "m")

        # -- read raw spectra -----------------------------------------------
        w1, f1, e1 = (None, None, None) if path1 is None else _read_raw_spectrum(path1)
        w2, f2, e2 = (None, None, None) if path2 is None else _read_raw_spectrum(path2)

        # -- SNR filter (applied before drawing) ----------------------------
        if min_snr is not None:
            snr1 = _compute_snr(f1, e1)
            snr2 = _compute_snr(f2, e2)
            # Reject only when SNR is *measured* and below the cut.
            # NaN (= IVAR absent) is treated as passing.
            if (np.isfinite(snr1) and snr1 < min_snr) or \
               (np.isfinite(snr2) and snr2 < min_snr):
                skipped_snr += 1
                continue

        def _box_smooth(flux, wave, window_aa=5.0):
            """Box-smooth flux over ~window_aa Angstroms, adapted to pixel scale."""
            if wave is None or len(wave) < 2:
                return flux
            dpix = float(np.abs(np.median(np.diff(wave))))
            n = max(1, int(round(window_aa / dpix))) if dpix > 0 else 1
            if n <= 1:
                return flux
            kernel = np.ones(n) / n
            smoothed = np.convolve(flux, kernel, mode="same")
            # convolve introduces edge artefacts; restore raw flux at borders
            half = n // 2
            smoothed[:half]  = flux[:half]
            smoothed[-half:] = flux[-half:]
            return smoothed

        def _normalize_at_5100(wave_rest, flux):
            """
            Fit the local continuum at 5100 Å and return
            (normalized_flux, f_cont_5100).

            normalized_flux = flux / f_cont_5100  (≈ 1 at 5100 Å)
            f_cont_5100     = fitted continuum value in original flux units

            Returns (flux, None) on any failure so callers can still plot.
            """
            lam_ref = 5100.0
            # 5080–5350 Å: avoids Hβ (4861), [OIII] 4959/5007, and He I 5876
            lam_lo, lam_hi = 5080.0, 5350.0

            # Keep ALL finite pixels (including negative noise excursions):
            # these flux-calibrated, sky-subtracted spectra legitimately go
            # negative on faint epochs, and a `flux > 0` cut keeps only upward
            # noise → biases the continuum estimate high exactly where it hurts.
            mask = (
                (wave_rest >= lam_lo) &
                (wave_rest <= lam_hi) &
                np.isfinite(flux)
            )
            if mask.sum() < 10:
                return flux, None    # too few pixels

            w = wave_rest[mask]
            f = flux[mask]

            # --- linear fit -----------------------------------------------
            c_lin      = np.polyfit(w, f, deg=1)
            f_lin      = np.polyval(c_lin, w)
            rms_lin    = float(np.sqrt(np.mean((f - f_lin) ** 2)))
            f_cont_lin = float(np.polyval(c_lin, lam_ref))

            # --- power-law fit (linear in log-log) --------------------------
            # Only positive pixels have a real logarithm; restrict the log-log
            # fit to them (the linear fit above already uses all finite pixels).
            try:
                pos = f > 0
                if int(pos.sum()) >= 10:
                    log_w      = np.log10(w[pos])
                    log_f      = np.log10(f[pos])
                    c_pl       = np.polyfit(log_w, log_f, deg=1)
                    f_pl       = 10.0 ** np.polyval(c_pl, log_w)
                    rms_pl     = float(np.sqrt(np.mean((f[pos] - f_pl) ** 2)))
                    f_cont_pl  = float(10.0 ** np.polyval(c_pl, np.log10(lam_ref)))
                else:
                    rms_pl, f_cont_pl = np.inf, np.nan
            except Exception:
                rms_pl, f_cont_pl = np.inf, np.nan

            # --- pick better model -----------------------------------------
            if np.isfinite(f_cont_pl) and f_cont_pl > 0 and rms_pl <= rms_lin:
                f_cont_5100 = f_cont_pl
            elif np.isfinite(f_cont_lin) and f_cont_lin > 0:
                f_cont_5100 = f_cont_lin
            else:
                return flux, None    # both degenerate

            return flux / f_cont_5100, f_cont_5100

        def _compute_oiii_flux(wave_rest, flux):
            """
            Compute the integrated [O III] 5007 flux by fitting a local continuum
            using windows before Hbeta and after [O III].
            """
            # Continuum windows: before Hb (4861) and after [O III] (5007)
            mask_cont = (
                ((wave_rest >= 4750.0) & (wave_rest <= 4820.0)) |
                ((wave_rest >= 5080.0) & (wave_rest <= 5150.0))
            ) & np.isfinite(flux)
            
            if mask_cont.sum() < 5:
                return None
                
            w_c = wave_rest[mask_cont]
            f_c = flux[mask_cont]
            
            # Linear continuum fit
            try:
                c_lin = np.polyfit(w_c, f_c, deg=1)
            except Exception:
                return None
                
            # [O III] integration window
            mask_oiii = (wave_rest >= 4990.0) & (wave_rest <= 5025.0) & np.isfinite(flux)
            if mask_oiii.sum() < 3:
                return None
                
            w_oiii = wave_rest[mask_oiii]
            f_oiii = flux[mask_oiii]
            
            cont_oiii = np.polyval(c_lin, w_oiii)
            # Multiply by the rest-frame pixel width so this is a true integrated
            # flux (erg s^-1 cm^-2), not a bare sum of flux densities.
            dlam = float(np.median(np.diff(wave_rest)))
            integrated_flux = np.sum(f_oiii - cont_oiii) * dlam
            
            return float(integrated_flux) if integrated_flux > 0 else None

        # ------------------------------------------------------------------
        # Prepare rest-frame, clipped, smoothed flux arrays
        # ------------------------------------------------------------------
        def _prep(wave, flux, z):
            """Rest-frame + clip + smooth.  Returns (wave_rest, flux_out, f5100, f_oiii)."""
            if wave is None or len(wave) == 0:
                return None, None, None, None
            z_val = float(z) if pd.notna(z) else 0.0
            wr = wave / (1.0 + z_val)
            # Measure f5100 and [O III] on the UNCLIPPED flux so the narrow
            # [O III] 5007 peak is not truncated and its integrated flux is accurate.
            _, f5100 = _normalize_at_5100(wr, flux)
            f_oiii = _compute_oiii_flux(wr, flux)
            # Display clip: use p5/p99.5 so bright emission lines survive but
            # edge-noise spikes (often large negative values at the blue/red
            # coverage boundaries) are suppressed.
            p5, p995 = np.nanpercentile(flux, [5, 99.5])
            _rng = max(p995 - p5, 1e-8)
            lo_clip = max(p5 - 0.10 * _rng, -0.30 * max(p995, 0.0))
            hi_clip = p995 + 0.50 * _rng        # permissive upper: keeps bright lines
            fc = np.clip(flux, lo_clip, hi_clip)
            fs = _box_smooth(fc, wr, window_aa=5.0)
            return wr, fs, f5100, f_oiii

        wr1, fs1, f5100_1, oiii_1 = _prep(w1, f1, z1)
        wr2, fs2, f5100_2, oiii_2 = _prep(w2, f2, z2)

        # ------------------------------------------------------------------
        # Scale-factor grid for the SDSS-V slider
        # ------------------------------------------------------------------
        # Each epoch is divided by its OWN reference level (f5100 mark =
        # 1 / f_cont(5100)), so the marked position drives that epoch's
        # continuum to unity independently of the other epoch.
        if f5100_1 and f5100_1 > 0:
            f5100_ratio = float(1.0 / f5100_1)
        else:
            f5100_ratio = 1.0

        if oiii_1 and oiii_2 and oiii_1 > 0 and oiii_2 > 0:
            # [O III] line-flux ratio: scale SDSS-V onto DESI's [O III] (DESI is
            # the reference epoch). Dimensionless (flux/flux) -> a throughput
            # correction, since intrinsic [O III] is constant between epochs.
            oiii_ratio = float(oiii_2 / oiii_1)
        else:
            oiii_ratio = None

        # ------------------------------------------------------------------
        # Compute calibration scale factors
        # ------------------------------------------------------------------
        # f5100 normalisation: divide each spectrum by its own continuum at 5100Å
        # so both sit at ~1 in flux-density units at 5100Å.
        f5100_sdss = f5100_1 if (f5100_1 and f5100_1 > 0) else None
        f5100_desi = f5100_2 if (f5100_2 and f5100_2 > 0) else None

        # [O III] cross-calibration: scale SDSS-V so its [O III] matches DESI's.
        if oiii_1 and oiii_2 and oiii_1 > 0 and oiii_2 > 0:
            oiii_ratio = float(oiii_2 / oiii_1)   # multiply SDSS-V by this
        else:
            oiii_ratio = None

        # ------------------------------------------------------------------
        # Build 3-panel matplotlib figure
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        COLOR_SDSSV = "#1565C0"
        COLOR_DESI  = "#C62828"
        COLOR_LINES = "#aaaaaa"

        # Determine shared x-range from whichever spectra are available
        x_lo = max(2600.0, min(
            ([float(wr1[0])]  if wr1 is not None else []) +
            ([float(wr2[0])]  if wr2 is not None else []) or [2600.0]
        ))
        x_hi = max(
            ([float(wr1[-1])] if wr1 is not None else []) +
            ([float(wr2[-1])] if wr2 is not None else []) or [9000.0]
        )

        def _draw_lines(ax):
            """Draw vertical emission-line markers on an Axes."""
            for lname, lwave in EMISSION_LINES.items():
                if x_lo <= lwave <= x_hi:
                    ax.axvline(lwave, color=COLOR_LINES, lw=0.8, ls="--", zorder=0)
                    ax.text(lwave, 1.0, lname, transform=ax.get_xaxis_transform(),
                            fontsize=7, color=COLOR_LINES, rotation=90,
                            ha="right", va="bottom")

        def _decorate(ax, title_str):
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("Rest Wavelength [Å]", fontsize=9)
            ax.set_ylabel(r"Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=9)
            ax.set_title(title_str, fontsize=9, pad=4)
            ax.legend(fontsize=8, loc="upper right")
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.tick_params(which="both", direction="in", top=True, right=True)
            _draw_lines(ax)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True,
                                 gridspec_kw={"hspace": 0.38})

        # ------------------------------------------------------------------
        # Shared y-limit helper for all panels
        # ------------------------------------------------------------------
        def _auto_ylim(arrays, plo=5, phi=99, upper_margin=0.15):
            """
            Compute tight ylim from the union of all plotted arrays.
              Upper: phi-th percentile + upper_margin × range  (keeps bright lines)
              Lower: max(plo-th percentile, -0.30 × upper)     (floors huge negatives)
            """
            data = np.concatenate(
                [np.asarray(a, dtype=float).ravel() for a in arrays]
            )
            data = data[np.isfinite(data)]
            if not data.size:
                return None
            lo, hi = np.percentile(data, [plo, phi])
            rng    = hi - lo
            y_hi   = hi + upper_margin * rng
            y_lo   = max(lo - 0.10 * rng,          # a little breathing room below
                         -0.30 * max(y_hi, 1e-6))  # floor: don't show huge negatives
            return y_lo, y_hi

        # ── Panel 1: Raw ──────────────────────────────────────────────────
        ax = axes[0]
        p1_data = []
        if wr2 is not None:
            ax.plot(wr2, fs2, color=COLOR_DESI,  lw=0.9, label=f"DESI  MJD={mjd2}  z={z2:.4f}")
            p1_data.append(fs2)
        if wr1 is not None:
            ax.plot(wr1, fs1, color=COLOR_SDSSV, lw=0.9, label=f"SDSS-V  MJD={mjd1}  z={z1:.4f}")
            p1_data.append(fs1)
        _decorate(ax, "Raw spectra (×1.0)")
        lims1 = _auto_ylim(p1_data)
        if lims1:
            axes[0].set_ylim(*lims1)

        # ── Panel 2: [O III]-calibrate SDSS-V first, then ÷f₅₁₀₀ for both ─
        # Step 1: scale SDSS-V so its [O III] matches DESI's (flux cross-calibration)
        # Step 2: divide each epoch by its own f5100 continuum → both sit at ~1
        ax = axes[1]
        sf2 = oiii_ratio if oiii_ratio else 1.0          # OIII scale for SDSS-V
        fs1_oiii = fs1 * sf2 if wr1 is not None else None # SDSS-V after OIII calibration
        # f5100 of the OIII-scaled SDSS-V spectrum; reuse the raw f5100 value
        # (scaling by a constant doesn't change the spectral shape, only the
        # normalisation, so the new f5100 = old f5100 × sf2)
        f5100_sdss_oiii = f5100_sdss * sf2 if f5100_sdss else None
        desi_f5100_str2  = f"÷{f5100_desi:.3g}" if f5100_desi else "N/A"
        sdss_oiii_str    = (f"×{sf2:.3g} then ÷{f5100_sdss_oiii:.3g}"
                            if f5100_sdss_oiii else "N/A")
        p2_data = []
        if wr2 is not None:
            y2 = fs2 / f5100_desi if f5100_desi else fs2
            ax.plot(wr2, y2, color=COLOR_DESI, lw=0.9,
                    label=f"DESI  MJD={mjd2}  (f₅₁₀₀ {desi_f5100_str2})")
            p2_data.append(y2)
        if wr1 is not None and fs1_oiii is not None:
            y1 = fs1_oiii / f5100_sdss_oiii if f5100_sdss_oiii else fs1_oiii
            ax.plot(wr1, y1, color=COLOR_SDSSV, lw=0.9,
                    label=f"SDSS-V  MJD={mjd1}  ([O III] {sdss_oiii_str})")
            p2_data.append(y1)
        _decorate(ax, "[O III]-calibrated SDSS-V, both ÷ f₅₁₀₀ (flux-ratio comparison)")
        lims2 = _auto_ylim(p2_data)
        if lims2:
            axes[1].set_ylim(*lims2)

        # ── Panel 3: [O III]-calibrated only (no f₅₁₀₀ division) ─────────
        ax = axes[2]
        oiii_str = f"×{oiii_ratio:.3g}" if oiii_ratio else "N/A"
        p3_data = []
        if wr2 is not None:
            ax.plot(wr2, fs2, color=COLOR_DESI, lw=0.9, label=f"DESI  MJD={mjd2}  (reference)")
            p3_data.append(fs2)
        if wr1 is not None:
            sf3 = oiii_ratio if oiii_ratio else 1.0
            y3  = fs1 * sf3
            ax.plot(wr1, y3, color=COLOR_SDSSV, lw=0.9,
                    label=f"SDSS-V  MJD={mjd1}  ({oiii_str} — [O III] matched to DR16)")
            p3_data.append(y3)
        _decorate(ax, f"[O III]-calibrated SDSS-V, raw flux ({oiii_str})")
        lims3 = _auto_ylim(p3_data)
        if lims3:
            axes[2].set_ylim(*lims3)

        # ── Shared super-title ────────────────────────────────────────────
        prob_str = f"{prob:.3f}" if pd.notna(prob) else "N/A"
        fig.suptitle(
            f"CL-AGN candidate  |  P = {prob_str}  |  "
            f"RA = {ra:.5f}   Dec = {dec:+.5f}  "
            f"z₁ = {z1:.4f}   z₂ = {z2:.4f}\n"
            f"{Path(file1).stem}  →  {Path(file2).stem}",
            fontsize=10, y=0.995,
        )

        png_path = plots_dir / f"{tag}.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    if skipped_z > 0:

        print(f"[plot] skipped {skipped_z} pairs with |z1 - z2| > 0.05")
    if min_snr is not None:
        print(f"[plot] skipped {skipped_snr} pairs with SNR < {min_snr} in at least one epoch")
    print(f"[plot] saved {saved} plots to {plots_dir}")
    return saved



# ---------------------------------------------------------------------------
# Processed-channel diagnostic plots
# ---------------------------------------------------------------------------

def plot_processed_channels(
    predictions: "pd.DataFrame | str | Path",
    spectra_dir: "str | Path",
    plots_dir: "str | Path",
    model_dir: "str | Path" = DEFAULT_MODEL_DIR,
    min_prob: float = 0.8,
) -> int:
    """
    For each CL-AGN candidate with prob >= min_prob, save one figure per survey
    epoch showing the fully processed 2-channel input fed to the network.

    Figure layout (one file per epoch per candidate):
      Top panel    — Channel 0: arcsinh( MAD-normalised continuum-subtracted )
      Bottom panel — Channel 1: arcsinh( [O III]-normalised )  *or*  copy of
                     ch0 when [O III] SNR is below OIII_SNR_MIN (fallback).

    The x-axis is the fixed rest-frame MASTER_GRID (3000–10400 Å, 4096 px);
    only pixels with real spectral coverage deviate from zero.

    Files are written as::

        <plots_dir>/<tag>_sdssv_channels.png
        <plots_dir>/<tag>_dr16_channels.png

    Parameters
    ----------
    predictions : DataFrame or path to the predictions CSV produced by predict().
    spectra_dir : Directory containing the FITS files.
    plots_dir   : Output directory for the channel plots.
    model_dir   : Model directory (for channel1_scale).
    min_prob    : Only plot candidates with prob >= this value.

    Returns
    -------
    Number of figures saved.
    """
    import matplotlib.ticker as mticker

    if isinstance(predictions, (str, Path)):
        predictions = pd.read_csv(predictions)

    spectra_dir = Path(spectra_dir)
    plots_dir   = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir   = Path(model_dir)

    # ---- load channel1_scale -----------------------------------------------
    ckpt_path = model_dir / "siamese_changenet.pth"
    if ckpt_path.exists():
        ckpt           = torch.load(str(ckpt_path), map_location="cpu")
        channel1_scale = float(ckpt.get("channel1_scale", 0.0))
    else:
        channel1_scale = 0.0
    if channel1_scale == 0.0:
        ns             = load_norm_stats(str(model_dir / "norm_stats.json"))
        channel1_scale = float(ns.get("channel1_scale", 1.0))
    print(f"[plot_channels] channel1_scale = {channel1_scale:.6f}")

    # ---- filter candidates -------------------------------------------------
    candidates = predictions[predictions["prob"] >= min_prob].copy()
    if "prob" in candidates.columns:
        candidates = candidates.sort_values("prob", ascending=False)
    print(f"[plot_channels] {len(candidates)} candidates with prob >= {min_prob}")

    # ---- emission-line markers (rest-frame Å) ------------------------------
    EMISSION_LINES = {
        "Mg II":   2799,
        "Hβ":      4861,
        "[O III]": 5007,
        "Hα":      6563,
    }
    COLOR_LINES = "#aaaaaa"

    saved = 0
    for _, row in tqdm(candidates.iterrows(), total=len(candidates),
                       desc="Plotting channels", unit="pair"):
        file1 = str(row.get("file1", ""))
        file2 = str(row.get("file2", ""))
        path1 = _resolve_path(spectra_dir, file1)
        path2 = _resolve_path(spectra_dir, file2)

        z1   = row.get("z1",  row.get("z",  None))
        z2   = row.get("z2",  row.get("z",  None))
        prob = row.get("prob", float("nan"))
        ra   = row.get("ra",  float("nan"))
        dec  = row.get("dec", float("nan"))
        tag  = f"ra{ra:.4f}_dec{dec:+.4f}".replace("+", "p").replace("-", "m")

        for survey, fpath, z, stem, color in [
            ("SDSS-V", path1, z1, Path(file1).stem, "#1565C0"),
            ("SDSS-DR16",   path2, z2, Path(file2).stem, "#C62828"),
        ]:
            if fpath is None:
                print(f"[plot_channels] {survey}: file not found — skipping")
                continue

            madnorm, raw_flux, valid, _ = _process_one(fpath, z)
            if madnorm is None or raw_flux is None or valid is None:
                print(f"[plot_channels] {survey}: preprocessing failed — skipping")
                continue

            x, info        = _build_channel_with_info(raw_flux, madnorm, valid, channel1_scale)
            ch0            = x[0]                           # arcsinh MAD-norm
            ch1            = x[1]                           # arcsinh OIII-norm / fallback
            oiii_reliable  = info.get("oiii_reliable", False)
            oiii_snr_val   = info.get("oiii_snr",     0.0)
            oiii_flux_val  = info.get("oiii_flux",    0.0)

            z_val = float(z) if z is not None and pd.notna(z) else float("nan")

            # ── Raw rest-frame spectrum from FITS ───────────────────────────
            wave_obs, flux_raw, sigma_raw = _read_raw_spectrum(fpath)
            if wave_obs is not None and len(wave_obs) > 0:
                wave_rest_raw = wave_obs / (1.0 + z_val) if np.isfinite(z_val) else wave_obs
                # tight clip: 1st–99th percentile; display window adds 5% padding
                p1, p99     = np.nanpercentile(flux_raw, [1, 99])
                raw_range   = p99 - p1
                raw_margin  = 0.05 * raw_range
                flux_disp   = np.clip(flux_raw, p1, p99)
                raw_ylim    = (p1 - raw_margin, p99 + raw_margin)
            else:
                wave_rest_raw = flux_disp = sigma_raw = None

            fig, (ax_raw, ax0, ax1) = plt.subplots(
                3, 1, figsize=(14, 10), sharex=True,
                gridspec_kw={"hspace": 0.40},
            )

            def _draw_lines(ax):
                for lname, lwave in EMISSION_LINES.items():
                    if MASTER_GRID[0] <= lwave <= MASTER_GRID[-1]:
                        ax.axvline(lwave, color=COLOR_LINES, lw=0.8, ls="--", zorder=0)
                        ax.text(lwave, 1.0, lname,
                                transform=ax.get_xaxis_transform(),
                                fontsize=7, color=COLOR_LINES,
                                rotation=90, ha="right", va="bottom")

            # ── Panel 0: raw rest-frame flux ────────────────────────────────
            if wave_rest_raw is not None:
                ax_raw.plot(wave_rest_raw, flux_disp, color=color, lw=0.7)
                if sigma_raw is not None:
                    ax_raw.fill_between(
                        wave_rest_raw,
                        flux_disp - sigma_raw,
                        flux_disp + sigma_raw,
                        color=color, alpha=0.18, linewidth=0,
                    )
                if raw_ylim is not None:
                    ax_raw.set_ylim(*raw_ylim)
            ax_raw.set_xlim(MASTER_GRID[0], MASTER_GRID[-1])
            ax_raw.set_ylabel(
                r"Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=9
            )
            ax_raw.set_title("Raw rest-frame spectrum", fontsize=9, pad=4)
            ax_raw.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax_raw.tick_params(which="both", direction="in", top=True, right=True)
            _draw_lines(ax_raw)

            # ── Panel 1: Channel 0 ──────────────────────────────────────────
            ax0.plot(MASTER_GRID, ch0, color=color, lw=0.7)
            ax0.set_xlim(MASTER_GRID[0], MASTER_GRID[-1])
            ax0.set_ylabel("arcsinh(MAD-norm)", fontsize=9)
            ax0.set_title(
                "Channel 0 — MAD-normalised continuum-subtracted",
                fontsize=9, pad=4,
            )
            ax0.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax0.tick_params(which="both", direction="in", top=True, right=True)
            _draw_lines(ax0)

            # ── Panel 2: Channel 1 ──────────────────────────────────────────
            if oiii_reliable:
                ch1_title  = "Channel 1 — [O III]-normalised"
                ch1_legend = (
                    f"[O III] SNR = {oiii_snr_val:.1f}   "
                    f"flux = {oiii_flux_val:.3g}"
                )
                ch1_ylabel = "arcsinh([O III]-norm)"
            else:
                ch1_title  = (
                    f"Channel 1 — fallback = ch0  "
                    f"([O III] SNR = {oiii_snr_val:.1f} < {OIII_SNR_MIN})"
                )
                ch1_legend = "[O III] unreliable — channel 1 copied from channel 0"
                ch1_ylabel = "arcsinh(MAD-norm)  [ch0 copy]"

            ax1.plot(MASTER_GRID, ch1, color=color, lw=0.7, label=ch1_legend)
            ax1.set_xlim(MASTER_GRID[0], MASTER_GRID[-1])
            ax1.set_xlabel("Rest Wavelength [Å]", fontsize=9)
            ax1.set_ylabel(ch1_ylabel, fontsize=9)
            ax1.set_title(ch1_title, fontsize=9, pad=4)
            ax1.legend(fontsize=8, loc="upper right")
            ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax1.tick_params(which="both", direction="in", top=True, right=True)
            _draw_lines(ax1)

            # ── Super-title ─────────────────────────────────────────────────
            prob_str = f"{prob:.3f}" if pd.notna(prob) else "N/A"
            fig.suptitle(
                f"{survey} processed channels  |  P = {prob_str}  |  "
                f"RA = {ra:.5f}   Dec = {dec:+.5f}   z = {z_val:.4f}\n"
                f"{stem}",
                fontsize=10, y=0.999,
            )

            fname = (
                f"{tag}_{survey.lower().replace('-', '').replace(' ', '')}"
                "_channels.png"
            )
            fig.savefig(str(plots_dir / fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved += 1

    print(f"[plot_channels] saved {saved} figures to {plots_dir}")
    return saved


# ---------------------------------------------------------------------------
# IDE / notebook config — edit these paths when running directly in an IDE.
# When running from the terminal these are ignored (CLI args take over).
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[1]   # repo root, auto-detected

IDE_CONFIG = dict(
    # ── DR16 × DR20 BHM crossmatch run ──────────────────────────────────────
    # pairs_csv must have columns 'file1' (DR16 spec) and 'file2' (DR20 spec).
    # Rename from the crossmatch output before passing here:
    #   pairs = pd.read_parquet("data/dr16_dr20bhm_crossmatch_3arcsec.parquet")
    #   pairs = pairs.rename(columns={"spec_file_dr16": "file1",
    #                                  "spec_file_dr20": "file2",
    #                                  "ra_dr16": "ra", "dec_dr16": "dec",
    #                                  "z_dr16": "z1", "z_dr20": "z2",
    #                                  "mjd_dr16": "mjd1", "mjd_dr20": "mjd2"})
    #   pairs.to_csv("data/dr16_dr20bhm_pairs_for_predict.csv", index=False)
    spectra_dir        = PROJECT_ROOT / "data" / "sdssv_dr16_crossmatch",
    pairs_csv          = PROJECT_ROOT / "data" / "sdssv_dr16_crossmatch_unique.csv",
    output             = PROJECT_ROOT / "results" / "predictions_dr16_dr20bhm_regular_loss_fixed_ch1.csv",
    model_dir          = DEFAULT_MODEL_DIR,
    threshold          = None,    # None = auto-load best_threshold from checkpoint
    batch_size         = 512,
    device             = None,    # None = auto (mps → cuda → cpu)
    n_jobs             = -2,
    min_snr_pairs      = 4.0,     # both epochs must have SNR ≥ this
    max_dz             = 0.05,    # |z1 - z2| must be < this
    max_z              = 0.8,     # skip pairs with redshift > 0.8
    plot_dir           = None,    # None = skip raw-spectrum plots (CSV only)
    plot_only          = False,   # False = run inference first
    min_prob           = None,    # None = fallback to classification threshold
    min_snr            = None,    # None = no SNR filter on raw-spectrum plots
    # ---- processed-channel diagnostic plots --------------------------------
    plot_channels      = False,   # True = also save 2-channel diagnostic plots
    plot_channels_only = False,  # True = skip inference AND raw-spectrum plots; channels only
    plot_channels_dir  = PROJECT_ROOT / "results" / "sdss_crossmatch_agn_only",
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
    p.add_argument("--output",       default="results/predictions_sdssv_desi.csv",
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
    p.add_argument("--plot-dir",     default=None,
                   help="If set, save spectral plots for all predicted CL-AGN "
                        "pairs to this directory.")
    p.add_argument("--plot-only",    action="store_true",
                   help="Skip inference entirely. Load predictions from --output "
                        "and run only the plotting step (requires --plot-dir).")
    return p.parse_args()


if __name__ == "__main__":
    # Running from terminal → use CLI args.
    # Running from an IDE (no sys.argv) → use IDE_CONFIG above.
    import sys
    if len(sys.argv) > 1:
        args = _parse_args()

        if args.plot_only:
            # ---- plot-only mode: skip inference, replot from saved CSV ----
            if args.plot_dir is None:
                raise SystemExit("--plot-only requires --plot-dir to be set.")
            plot_cl_agn_pairs(
                predictions = args.output,          # read existing predictions CSV
                spectra_dir = args.spectra_dir,
                plots_dir   = args.plot_dir,
                threshold   = args.threshold or DEFAULT_THRESHOLD,
            )
        else:
            # ---- normal mode: run inference then optionally plot ----------
            results = predict(
                spectra_dir = args.spectra_dir,
                pairs_csv   = args.pairs_csv,
                output      = args.output,
                model_dir   = args.model_dir,
                threshold   = args.threshold if args.threshold != DEFAULT_THRESHOLD else None,
                batch_size  = args.batch_size,
                device      = args.device,
                n_jobs      = args.n_jobs,
            )
            if args.plot_dir is not None:
                plot_cl_agn_pairs(
                    predictions = results,
                    spectra_dir = args.spectra_dir,
                    plots_dir   = args.plot_dir,
                    threshold   = args.threshold,
                )
    else:
        # IDE mode — run predict then optionally plot (or plot-only)
        plot_dir           = IDE_CONFIG.pop("plot_dir",           None)
        plot_only          = IDE_CONFIG.pop("plot_only",          False)
        min_prob           = IDE_CONFIG.pop("min_prob",           None)
        min_snr            = IDE_CONFIG.pop("min_snr",            None)
        plot_channels      = IDE_CONFIG.pop("plot_channels",      False)
        plot_channels_only = IDE_CONFIG.pop("plot_channels_only", False)
        plot_channels_dir  = IDE_CONFIG.pop("plot_channels_dir",  None)

        if plot_channels_only:
            # ---- channel-plots-only mode: skip inference and raw plots -----
            if plot_channels_dir is None:
                raise SystemExit(
                    "IDE_CONFIG: plot_channels_only=True requires "
                    "plot_channels_dir to be set."
                )
            plot_processed_channels(
                predictions = IDE_CONFIG["output"],
                spectra_dir = IDE_CONFIG["spectra_dir"],
                plots_dir   = plot_channels_dir,
                model_dir   = IDE_CONFIG.get("model_dir", DEFAULT_MODEL_DIR),
                min_prob    = min_prob or DEFAULT_THRESHOLD,
            )
        elif plot_only:
            if plot_dir is None:
                raise SystemExit("IDE_CONFIG: plot_only=True requires plot_dir to be set.")
            plot_cl_agn_pairs(
                predictions = IDE_CONFIG["output"],
                spectra_dir = IDE_CONFIG["spectra_dir"],
                plots_dir   = plot_dir,
                threshold   = IDE_CONFIG.get("threshold") or DEFAULT_THRESHOLD,
                min_prob    = min_prob,
                min_snr     = min_snr,
            )
            if plot_channels and plot_channels_dir is not None:
                plot_processed_channels(
                    predictions = IDE_CONFIG["output"],
                    spectra_dir = IDE_CONFIG["spectra_dir"],
                    plots_dir   = plot_channels_dir,
                    model_dir   = IDE_CONFIG.get("model_dir", DEFAULT_MODEL_DIR),
                    min_prob    = min_prob or DEFAULT_THRESHOLD,
                )
        else:
            results = predict(**IDE_CONFIG)
            if plot_dir is not None:
                plot_cl_agn_pairs(
                    predictions = results,
                    spectra_dir = IDE_CONFIG["spectra_dir"],
                    plots_dir   = plot_dir,
                    threshold   = IDE_CONFIG.get("threshold") or DEFAULT_THRESHOLD,
                    min_prob    = min_prob,
                    min_snr     = min_snr,
                )
            if plot_channels and plot_channels_dir is not None:
                plot_processed_channels(
                    predictions = results,
                    spectra_dir = IDE_CONFIG["spectra_dir"],
                    plots_dir   = plot_channels_dir,
                    model_dir   = IDE_CONFIG.get("model_dir", DEFAULT_MODEL_DIR),
                    min_prob    = min_prob or DEFAULT_THRESHOLD,
                )
