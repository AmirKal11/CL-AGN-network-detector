"""
data_preprocessing.py  (v2 — production)
==========================================
Orchestration layer: metadata CSV + flat FITS directory → Parquet.

Expected layout
---------------
    <data_dir>/
        spec-0276-51909-0006.fits
        desi-spec-39633285443860992.fits
        ...                              # all spectra, flat
    metadata.csv                         # one row per spectrum

Required CSV columns
--------------------
    spec_filename   basename of the FITS file
    object_id       cross-epoch identifier (same value for both epochs of a pair)
    survey          e.g. sdss_dr7 | dr16 | sdss_v | desi
    agn_type        type1 | type2
    z               redshift (preferred over FITS header; FITS used as fallback)

Optional CSV columns (passed through to the parquet when present)
-----------------------------------------------------------------
    ra, dec         sky coordinates
    snr             median SNR (if absent, extracted from FITS)
    <any other>     added as metadata columns in the output parquet

CLI
---
    python src/data_preprocessing.py \\
        --data-dir  data/spectra/ \\
        --csv       data/metadata.csv \\
        --output    data/processed.parquet \\
        [--min-snr 5.0] [--max-zeros-pct 0.8] \\
        [--no-subtract-continuum] [--n-jobs -2]

Low-level processing (sky removal, grid resampling, continuum subtraction) is
delegated to the existing helpers defined below; only orchestration and
path-handling live here.
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Master wavelength grid (shared with preprocessing_oiii.MASTER_GRID)
# ---------------------------------------------------------------------------
GRID_MIN, GRID_MAX, GRID_N = 3000.0, 10400.0, 4096
MASTER_GRID = np.linspace(GRID_MIN, GRID_MAX, GRID_N)


# ===========================================================================
# Low-level helpers (unchanged from v1; kept here so the module is self-
# contained when used as a library without the old import chain)
# ===========================================================================

def remove_sky_line(
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    line_center: float = 5577.3,
    window: float = 20.0,
    threshold: float = 4.0,
) -> np.ndarray:
    """
    Interpolate over the [O I] 5577 Å night-sky residual when it spikes
    more than `threshold` × local σ above the surrounding continuum.
    """
    mask_line = (wave_obs > line_center - window / 2) & (wave_obs < line_center + window / 2)
    if not np.any(mask_line):
        return flux_obs

    mask_cont = (
        ((wave_obs > line_center - window * 1.5) & (wave_obs <= line_center - window / 2))
        | ((wave_obs >= line_center + window / 2) & (wave_obs < line_center + window * 1.5))
    )
    if not np.any(mask_cont):
        return flux_obs

    local_med = np.nanmedian(flux_obs[mask_cont])
    local_std = np.nanstd(flux_obs[mask_cont])
    line_max = np.nanmax(flux_obs[mask_line])

    if local_std > 0 and line_max > local_med + threshold * local_std:
        x_cont = wave_obs[mask_cont]
        y_cont = flux_obs[mask_cont]
        if len(x_cont) > 1:
            f = interp1d(x_cont, y_cont, kind="linear", bounds_error=False, fill_value="extrapolate")
            flux_cleaned = flux_obs.copy()
            flux_cleaned[mask_line] = f(wave_obs[mask_line])
            return flux_cleaned

    return flux_obs


def _get_redshift(hdul) -> float | None:
    """Extract redshift from the SPECOBJ extension, or return None."""
    try:
        if "SPECOBJ" in hdul:
            return float(hdul["SPECOBJ"].data["Z"][0])
    except Exception:
        pass
    return None


def _get_snr(hdul) -> float | None:
    """
    Extract median SNR from SPECOBJ.SN_MEDIAN_ALL, or compute it from
    flux × sqrt(ivar) if the extension is absent.
    """
    try:
        if "SPECOBJ" in hdul:
            data = hdul["SPECOBJ"].data
            if "SN_MEDIAN_ALL" in data.names:
                return float(data["SN_MEDIAN_ALL"][0])
    except Exception:
        pass

    try:
        data = hdul[1].data
        names_lower = [n.lower() for n in data.names] if hasattr(data, "names") else []
        if "flux" in names_lower:
            flux = data["flux"]
            if "ivar" in names_lower:
                ivar = data["ivar"]
                valid = (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
                if np.any(valid):
                    return float(np.nanmedian(flux[valid] * np.sqrt(ivar[valid])))
            valid = np.isfinite(flux)
            if np.any(valid):
                std_f = np.nanstd(flux[valid])
                if std_f > 0:
                    return float(np.nanmean(flux[valid]) / std_f)
    except Exception:
        pass
    return None


@torch.no_grad()
def morphological_continuum_subtraction(
    x: torch.Tensor,
    window_size: int = 173,
    taper_len: int = 5,
    apply_mad_scaling: bool = False,
    valid_mask: torch.Tensor | None = None,
    subtract_continuum: bool = True,
) -> torch.Tensor:
    """
    Wide moving-average continuum removal + optional MAD scaling + edge taper. Currently dead

    x shape: [B, 1, L]
    """
    pad = window_size // 2
    vm = valid_mask.to(dtype=x.dtype) if valid_mask is not None else None

    if subtract_continuum:
        if vm is not None:
            x_pad = F.pad(x * vm, (pad, pad), mode="reflect")
            v_pad = F.pad(vm, (pad, pad), mode="reflect")
            num = F.avg_pool1d(x_pad, kernel_size=window_size, stride=1)
            den = F.avg_pool1d(v_pad, kernel_size=window_size, stride=1)
            continuum = num / (den + 1e-8)
        else:
            continuum = F.avg_pool1d(F.pad(x, (pad, pad), mode="reflect"), kernel_size=window_size, stride=1)
        x_flat = x - continuum
    else:
        x_flat = x

    if apply_mad_scaling:
        x_proc = torch.zeros_like(x_flat)
        for b in range(x_flat.shape[0]):
            if vm is not None:
                m = vm[b, 0] > 0.5
                if int(m.sum()) < 2:
                    continue
                vals = x_flat[b, 0][m]
            else:
                vals = x_flat[b, 0]
            median = vals.median()
            mad = (vals - median).abs().median()
            x_proc[b, 0] = (x_flat[b, 0] - median) / (mad * 1.4826 + 1e-8)
    else:
        x_proc = x_flat

    if vm is not None:
        x_proc = x_proc * vm

    seq_len = x.shape[-1]
    taper = torch.ones(seq_len, device=x.device)
    if taper_len > 0:
        fade = torch.linspace(0.0, 1.0, taper_len, device=x.device)
        taper[:taper_len] = fade
        taper[-taper_len:] = torch.flip(fade, dims=[0])
    return x_proc * taper.view(1, 1, -1)


# ===========================================================================
# Per-spectrum processing
# ===========================================================================

def process_single_spectrum(
    fits_path: str | Path,
    z: float | None,
    master_grid: np.ndarray = MASTER_GRID,
) -> dict | None:
    """
    FITS → rest-frame grid → raw interpolated flux.

    Returns the physical flux on the master grid with no normalisation.
    Processing chain: sky-line removal → de-redshift → resample → zero-fill
    out-of-coverage pixels. All normalisation (continuum subtraction, MAD,
    OIII) is deferred to the dataset/training layer.

    Parameters
    ----------
    fits_path   Path to the FITS file.
    z           Redshift (from the metadata CSV). Falls back to the FITS
                SPECOBJ header if None or NaN.
    master_grid Rest-frame wavelength grid to interpolate onto.

    Returns
    -------
    dict with keys: flux_array (np.ndarray[float32], shape [L]),
                    valid_frac (float), snr (float | None), z (float)
    None on any failure.
    """
    fits_path = str(fits_path)
    try:
        with fits.open(fits_path) as hdul:
            # --- redshift ---
            if z is None or (isinstance(z, float) and np.isnan(z)):
                z = _get_redshift(hdul)
            if z is None:
                return None

            # --- SNR ---
            snr = _get_snr(hdul)

            # --- flux + wavelength ---
            data = hdul[1].data
            flux_obs = np.asarray(data["flux"], dtype=np.float64)
            names_lower = [n.lower() for n in data.names] if hasattr(data, "names") else []

            if "loglam" in names_lower:
                wave_obs = 10.0 ** np.asarray(data["loglam"], dtype=np.float64)
            elif "wavelength" in names_lower:
                wave_obs = np.asarray(data["wavelength"], dtype=np.float64)
            else:
                hdr = hdul[1].header
                wave_obs = hdr["CRVAL1"] + np.arange(len(flux_obs)) * hdr["CDELT1"]

            # --- sky-line removal ---
            flux_obs = remove_sky_line(wave_obs, flux_obs)

            # --- rest-frame shift (flux-conserving) ---
            wave_rest = wave_obs / (1.0 + z)
            flux_rest = flux_obs * (1.0 + z)

            # --- resample onto master grid ---
            f_interp = interp1d(wave_rest, flux_rest, bounds_error=False, fill_value=np.nan)
            grid_flux = f_interp(master_grid)

            valid = np.isfinite(grid_flux)
            if int(valid.sum()) < 50:
                return None

            grid_flux = np.nan_to_num(grid_flux, nan=0.0)

            # Raw physical flux; out-of-coverage pixels zeroed.
            # No normalisation -- deferred to dataset/training layer.
            flux_array = np.where(valid, grid_flux, 0.0).astype(np.float32)

            return {
                "flux_array": flux_array,
                # Coverage mask as computed here (isfinite on the interpolated
                # grid) -- the SAME convention datasets_v2.fits_to_flat stores
                # in the pair cache. Callers must use this rather than
                # reconstructing it with valid_from_flux(flux != 0), because
                # SDSS-V reductions write exactly 0.0 for masked pixels that
                # ARE inside coverage; the two conventions disagree there and
                # a zero pixel landing in an [O III] band shifts the measured
                # flux (and hence the channel-1 scale).
                "valid": valid.astype(bool),
                "valid_frac": float(valid.mean()),
                "snr": snr,
                "z": float(z),
            }

    except Exception as exc:
        print(f"[skip] {os.path.basename(fits_path)}: {exc}")
        return None


# ===========================================================================
# Parquet builder
# ===========================================================================

def _process_row(
    fits_path: str,
    row: dict,
    master_grid: np.ndarray,
) -> dict | None:
    """Process one CSV row + its FITS file into a parquet-ready dict."""
    z = row.get("z", None)
    if z is not None:
        try:
            z = float(z)
            if np.isnan(z):
                z = None
        except (ValueError, TypeError):
            z = None

    result = process_single_spectrum(
        fits_path,
        z=z,
        master_grid=master_grid,
    )
    if result is None:
        return None

    out = {k: v for k, v in row.items() if k != "z"}  # row metadata (no duplicate z)
    out["z"] = result["z"]                             # canonical redshift
    out["valid_frac"] = result["valid_frac"]

    # SNR: prefer CSV value if present; fall back to FITS-derived
    if "snr" not in out or pd.isna(out.get("snr")):
        out["snr"] = result["snr"]

    out["flux_array"] = result["flux_array"]
    return out


def build_parquet(
    data_dir: str | Path,
    csv_path: str | Path,
    output: str | Path,
    master_grid: np.ndarray = MASTER_GRID,
    min_snr: float = 4.0,
    max_zeros_pct: float = 0.8,
    n_jobs: int = -2,
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Read `csv_path`, find each FITS in `data_dir`, process all spectra in
    parallel, filter, and write a parquet to `output`.

    The parquet schema
    ------------------
    Metadata columns first (all CSV columns + valid_frac + snr + z),
    followed by 4096 float32 flux columns named by their rest-frame wavelength
    (as strings, e.g. "3001.8").

    Flux values are raw physical flux: sky-removed, de-redshifted, resampled
    onto the master grid, out-of-coverage pixels zero-filled. No continuum
    subtraction or normalisation is applied; that is deferred to the
    dataset/training layer so that OIII can be measured on physical flux.

    Parameters
    ----------
    data_dir        Directory (searched non-recursively) containing FITS files.
    csv_path        Metadata CSV — one row per spectrum.
    output          Destination parquet path (parent dirs created if needed).
    min_snr         Drop spectra below this SNR.
    max_zeros_pct   Drop spectra where more than this fraction of grid pixels
                    are zero (i.e. outside the spectrograph coverage).
    n_jobs          joblib parallelism (-1 = all cores, -2 = all but one).
    recursive       If True, search data_dir recursively for FITS files
                    (useful when FITS are organized in subdirectories).

    Returns
    -------
    The cleaned DataFrame (also written to `output`).
    """
    data_dir = Path(data_dir)
    csv_path = Path(csv_path)
    output = Path(output)

    # --- load metadata ---
    meta = pd.read_csv(csv_path)
    required = {"spec_filename", "object_id", "survey", "agn_type", "z"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata CSV is missing required columns: {missing}")

    # build filename → row lookup
    rows = {row["spec_filename"]: row.to_dict() for _, row in meta.iterrows()}

    # --- resolve FITS paths ---
    pattern = "**/*.fits" if recursive else "*.fits"
    fits_on_disk = {Path(f).name: str(f) for f in data_dir.glob(pattern)}
    tasks = []
    not_found = []
    for fname, row in rows.items():
        if fname in fits_on_disk:
            tasks.append((fits_on_disk[fname], row))
        else:
            not_found.append(fname)

    if not_found:
        print(f"[warn] {len(not_found)} CSV entries have no matching FITS in {data_dir}")
    print(f"Processing {len(tasks)} spectra (parallel, n_jobs={n_jobs})…")

    # --- parallel processing ---
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_row)(fits_path, row, master_grid)
        for fits_path, row in tasks
    )
    results = [r for r in results if r is not None]
    print(f"  succeeded: {len(results)} / {len(tasks)}")

    if not results:
        raise RuntimeError("No spectra processed successfully.")

    # --- assemble DataFrame ---
    flux_matrix = np.stack([r.pop("flux_array") for r in results])  # [N, 4096]
    meta_df = pd.DataFrame(results)
    flux_df = pd.DataFrame(flux_matrix, columns=master_grid.astype(str))
    df = pd.concat([meta_df.reset_index(drop=True), flux_df], axis=1)
    df.columns = df.columns.astype(str)

    # --- quality filtering ---
    n_before = len(df)

    if "snr" in df.columns:
        snr_vals = pd.to_numeric(df["snr"], errors="coerce")
        mask_snr = snr_vals >= min_snr
    else:
        mask_snr = pd.Series(True, index=df.index)

    zero_frac = (flux_matrix == 0.0).mean(axis=1)
    mask_cov = zero_frac <= max_zeros_pct

    mask_finite = np.isfinite(flux_matrix).any(axis=1)

    df = df[mask_snr & mask_cov & mask_finite].copy()

    print(
        f"Quality filter: kept {len(df)} / {n_before}  "
        f"(dropped {(~mask_snr).sum()} low-SNR, "
        f"{(~mask_cov).sum()} low-coverage, "
        f"{(~mask_finite).sum()} non-finite)"
    )

    # --- write ---
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Saved → {output}  ({len(df)} spectra, {len(df.columns)} columns)")
    return df


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a spectral parquet from a flat FITS directory + metadata CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing FITS files (flat).")
    p.add_argument("--csv", required=True,
                   help="Metadata CSV (one row per spectrum).")
    p.add_argument("--output", required=True,
                   help="Output parquet path.")
    p.add_argument("--min-snr", type=float, default=5.0,
                   help="Minimum median SNR to keep a spectrum.")
    p.add_argument("--max-zeros-pct", type=float, default=0.8,
                   help="Maximum fraction of zero-filled grid pixels (coverage filter).")
    p.add_argument("--recursive", action="store_true",
                   help="Search data-dir recursively for FITS (use when FITS are in subdirs).")
    p.add_argument("--n-jobs", type=int, default=-2,
                   help="joblib parallelism (-1 all cores, -2 all but one).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_parquet(
        data_dir=args.data_dir,
        csv_path=args.csv,
        output=args.output,
        min_snr=args.min_snr,
        max_zeros_pct=args.max_zeros_pct,
        recursive=args.recursive,
        n_jobs=args.n_jobs,
    )
