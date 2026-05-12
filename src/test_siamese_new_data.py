"""
Evaluate the trained Siamese CL-AGN network on real paired SDSS / SDSS-V spectra.

This version intentionally reuses the SAME preprocessing logic as the project:
    - remove_sky_line(...) from data_preprocessing.py
    - morphological_continuum_subtraction(...) from data_preprocessing.py
    - master_grid = np.linspace(4575, 6699, 1024)
    - rest-frame correction: wave_rest = wave_obs / (1 + z), flux_rest = flux_obs * (1 + z)
    - median fill for NaNs before continuum subtraction

Why not use SyntheticSiameseDataset directly?
    SyntheticSiameseDataset dynamically creates artificial Type1-Type2 / Type1-Type1 pairs
    from a processed catalog. Here we already have real object pairs from a pickle, so we only
    reuse its tensor format and the trained model/evaluation assumptions.

Expected pickle columns:
    sdssid, z, specname_dr16, specname_sdssv, label

Positive class:
    label = 1 means change / CL-AGN-like
    label = 0 means static / no-change
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from astropy.io import fits
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
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

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
# Put this script in your project scripts/ directory, or pass --project-root.
# The script adds <project_root>/src to sys.path so these imports match training.


def add_project_to_path(project_root: str):
    project_root = os.path.abspath(project_root)
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    return project_root


# These are imported after add_project_to_path(...) inside main().
SpectraNet = None
SiameseSpectraNet = None
load_config = None
remove_sky_line = None
morphological_continuum_subtraction = None


# ---------------------------------------------------------------------
# Exact training-style preprocessing for one real spectrum
# ---------------------------------------------------------------------

def read_sdss_like_flux_wave(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read observed-frame wavelength and flux from an SDSS-like FITS file.

    This follows the same assumptions as process_single_spectrum(...) in
    data_preprocessing.py:
        - extension 1 contains the spectrum
        - use flux column
        - wavelength from loglam, wavelength, or CRVAL1/CDELT1 fallback
    """
    with fits.open(file_path, memmap=False) as hdul:
        data = hdul[1].data
        flux_obs = np.asarray(data["flux"], dtype=np.float32)

        names_lower = [str(n).lower() for n in data.names]
        if "loglam" in names_lower:
            wave_obs = 10 ** np.asarray(data["loglam"], dtype=np.float64)
        elif "wavelength" in names_lower:
            wave_obs = np.asarray(data["wavelength"], dtype=np.float64)
        else:
            header = hdul[0].header
            if "COEFF0" in header:
                wave_obs = 10 ** (header["COEFF0"] + np.arange(len(flux_obs)) * header["COEFF1"])
            else:
                header1 = hdul[1].header
                wave_obs = header1["CRVAL1"] + np.arange(len(flux_obs)) * header1["CDELT1"]
            wave_obs = np.asarray(wave_obs, dtype=np.float64)

    finite = np.isfinite(wave_obs) & np.isfinite(flux_obs)
    wave_obs = wave_obs[finite]
    flux_obs = flux_obs[finite]

    order = np.argsort(wave_obs)
    return wave_obs[order], flux_obs[order]


def preprocess_single_spectrum_like_training(
    file_path: Path,
    z: float,
    master_grid: np.ndarray,
    sky_line_center: float = 5577.3,
    continuum_window_size: int = 151,
    taper_len: int = 5,
    clip_max: float = 4.0,
) -> np.ndarray:
    """
    Reproduce the core preprocessing from process_single_spectrum(...), but use
    the redshift from the crossmatch pickle instead of requiring SPECOBJ metadata.

    Training logic being preserved:
        1. read observed wave/flux
        2. remove prominent 5577 Å sky-line residual if it spikes
        3. rest-frame correction
        4. interpolate to master_grid
        5. fill NaNs with median in raw space
        6. apply morphological_continuum_subtraction(...)
        7. return flat 1024 processed flux
    """
    if z is None or not np.isfinite(float(z)):
        raise ValueError(f"Invalid redshift z={z} for {file_path}")

    wave_obs, flux_obs = read_sdss_like_flux_wave(file_path)

    # Same artifact-removal function used by training preprocessing.
    flux_obs = remove_sky_line(
        wave_obs,
        flux_obs,
        line_center=sky_line_center,
    )

    # Same rest-frame convention as data_preprocessing.process_single_spectrum.
    wave_rest = wave_obs / (1.0 + float(z))
    flux_rest = flux_obs * (1.0 + float(z))

    # Same interpolation/fill strategy.
    f_interp = interp1d(
        wave_rest,
        flux_rest,
        bounds_error=False,
        fill_value=np.nan,
    )
    interpolated_flux = f_interp(master_grid)

    valid_median = np.nanmedian(interpolated_flux)
    if not np.isfinite(valid_median):
        raise ValueError(f"No valid wavelength overlap after interpolation for {file_path}")

    interpolated_flux = np.nan_to_num(interpolated_flux, nan=valid_median)

    # Same tensor shape as training preprocessing: [Batch, Channel, Length].
    tensor_flux = torch.tensor(interpolated_flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    processed_tensor = morphological_continuum_subtraction(
        tensor_flux,
        window_size=continuum_window_size,
        taper_len=taper_len,
        clip_max=clip_max,
    )

    processed_flux = processed_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    return processed_flux

# ============================================================
# MAD / broad-line diagnostic utilities
# ============================================================

import torch.nn.functional as F


def compute_flat_before_mad(interpolated_flux, window_size=151):
    """
    Reproduce the continuum subtraction part of morphological_continuum_subtraction,
    but stop BEFORE median/MAD normalization.

    Input
    -----
    interpolated_flux : array-like, shape [1024]
        Rest-frame interpolated flux on master_grid.

    Returns
    -------
    flat : np.ndarray, shape [1024]
        Continuum-subtracted flux before MAD normalization.
    continuum : np.ndarray, shape [1024]
        Estimated smooth continuum.
    """
    x = torch.tensor(interpolated_flux, dtype=torch.float32).view(1, 1, -1)

    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode="reflect")
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)

    flat = x - continuum

    return (
        flat.squeeze().detach().cpu().numpy(),
        continuum.squeeze().detach().cpu().numpy(),
    )


def independent_mad_normalize(flat, taper_len=5, eps=1e-8):
    """
    Same per-spectrum MAD normalization as your current preprocessing,
    applied to a continuum-subtracted spectrum.
    """
    flat_t = torch.tensor(flat, dtype=torch.float32).view(1, 1, -1)

    median = flat_t.median(dim=-1, keepdim=True).values
    mad = (flat_t - median).abs().median(dim=-1, keepdim=True).values

    x_norm = (flat_t - median) / (mad * 1.4826 + eps)

    seq_len = flat_t.shape[-1]
    taper = torch.ones(seq_len, device=flat_t.device)
    fade = torch.linspace(0, 1, taper_len, device=flat_t.device)
    taper[:taper_len] = fade
    taper[-taper_len:] = torch.flip(fade, dims=[0])
    taper = taper.view(1, 1, -1)

    x_final = x_norm * taper

    return {
        "normalized": x_final.squeeze().detach().cpu().numpy(),
        "median": float(median.item()),
        "mad": float(mad.item()),
    }


def pair_shared_mad_normalize(flat1, flat2, taper_len=5, eps=1e-8):
    """
    Normalize both epochs using ONE shared median/MAD.

    This preserves between-epoch amplitude differences better than independent MAD.
    This is diagnostic only unless the model is retrained with this normalization.
    """
    x1 = torch.tensor(flat1, dtype=torch.float32).view(1, 1, -1)
    x2 = torch.tensor(flat2, dtype=torch.float32).view(1, 1, -1)

    combined = torch.cat([x1, x2], dim=-1)

    shared_median = combined.median(dim=-1, keepdim=True).values
    shared_mad = (combined - shared_median).abs().median(dim=-1, keepdim=True).values

    x1_norm = (x1 - shared_median) / (shared_mad * 1.4826 + eps)
    x2_norm = (x2 - shared_median) / (shared_mad * 1.4826 + eps)

    seq_len = x1.shape[-1]
    taper = torch.ones(seq_len, device=x1.device)
    fade = torch.linspace(0, 1, taper_len, device=x1.device)
    taper[:taper_len] = fade
    taper[-taper_len:] = torch.flip(fade, dims=[0])
    taper = taper.view(1, 1, -1)

    x1_norm = x1_norm * taper
    x2_norm = x2_norm * taper

    return {
        "x1_normalized": x1_norm.squeeze().detach().cpu().numpy(),
        "x2_normalized": x2_norm.squeeze().detach().cpu().numpy(),
        "shared_median": float(shared_median.item()),
        "shared_mad": float(shared_mad.item()),
    }


def interpolate_raw_spectrum_to_grid(file_path, z, master_grid):
    """
    Same early-stage preprocessing as the test script:
    FITS -> remove sky line -> rest frame -> interpolate -> median fill.
    """
    wave_obs, flux_obs = read_sdss_like_flux_wave(Path(file_path))

    flux_obs = remove_sky_line(
        wave_obs,
        flux_obs,
        line_center=5577.3,
    )

    wave_rest = wave_obs / (1.0 + float(z))
    flux_rest = flux_obs * (1.0 + float(z))

    f_interp = interp1d(
        wave_rest,
        flux_rest,
        bounds_error=False,
        fill_value=np.nan,
    )

    interpolated_flux = f_interp(master_grid)

    valid_median = np.nanmedian(interpolated_flux)
    if not np.isfinite(valid_median):
        raise ValueError(f"No valid overlap for {file_path}")

    interpolated_flux = np.nan_to_num(interpolated_flux, nan=valid_median)
    return interpolated_flux.astype(np.float32)


def paper_style_line_flux(flux, wave, line_name):
    """
    Paper-inspired local-continuum line-flux measurement.

    Uses the same rest-frame windows from the paper for Hbeta and Halpha.
    This is a simple diagnostic, not full PyQSOFit decomposition.
    """
    if line_name.lower() in ["hb", "hbeta", "hβ"]:
        blue_band = (4670.0, 4730.0)
        red_band = (5080.0, 5120.0)
        line_band = (4797.8, 4927.5)

    elif line_name.lower() in ["ha", "halpha", "hα"]:
        blue_band = (6150.0, 6250.0)
        red_band = (6950.0, 7150.0)
        line_band = (6477.1, 6652.1)

    else:
        raise ValueError(f"Unsupported line_name={line_name}")

    wave = np.asarray(wave)
    flux = np.asarray(flux)

    blue_mask = (wave >= blue_band[0]) & (wave <= blue_band[1])
    red_mask = (wave >= red_band[0]) & (wave <= red_band[1])
    line_mask = (wave >= line_band[0]) & (wave <= line_band[1])

    if blue_mask.sum() < 2 or red_mask.sum() < 2 or line_mask.sum() < 2:
        return np.nan

    blue_flux = np.nanmedian(flux[blue_mask])
    red_flux = np.nanmedian(flux[red_mask])

    blue_center = 0.5 * (blue_band[0] + blue_band[1])
    red_center = 0.5 * (red_band[0] + red_band[1])

    continuum = np.interp(
        wave[line_mask],
        [blue_center, red_center],
        [blue_flux, red_flux],
    )

    line_flux_density = flux[line_mask] - continuum

    return float(np.trapz(line_flux_density, wave[line_mask]))


def broad_wing_score(x, wave, line_name):
    """
    Broad-wing score designed to avoid being dominated by narrow cores.

    Uses mean absolute continuum-subtracted flux in broad side windows.
    This is intentionally simple and diagnostic.
    """
    wave = np.asarray(wave)
    x = np.asarray(x)

    if line_name.lower() in ["hb", "hbeta", "hβ"]:
        # Avoid the narrow Hbeta core around 4861.
        mask = (
            ((wave >= 4797.8) & (wave <= 4840.0)) |
            ((wave >= 4885.0) & (wave <= 4927.5))
        )

    elif line_name.lower() in ["ha", "halpha", "hα"]:
        # Avoid the narrow Halpha/[NII] central blend as much as possible.
        mask = (
            ((wave >= 6477.1) & (wave <= 6535.0)) |
            ((wave >= 6600.0) & (wave <= 6652.1))
        )

    else:
        raise ValueError(f"Unsupported line_name={line_name}")

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(x[mask])))


def region_peak_stats(x, wave, line_name):
    """
    Peak-above-median diagnostic in broad line regions.
    Useful for checking MAD amplification/inversion.
    """
    wave = np.asarray(wave)
    x = np.asarray(x)

    if line_name.lower() in ["hb", "hbeta", "hβ"]:
        mask = (wave >= 4700.0) & (wave <= 5100.0)
    elif line_name.lower() in ["ha", "halpha", "hα"]:
        mask = (wave >= 6300.0) & (wave <= 6699.0)
    else:
        raise ValueError(f"Unsupported line_name={line_name}")

    med = np.nanmedian(x)
    peak = np.nanmax(x[mask] - med)

    return float(peak)


def diagnose_pair_mad_effect(row, master_grid=None, make_plot=True, output_dir="outputs/mad_diagnostics"):
    """
    Diagnose one real pair.

    row should come from real_sdssv_siamese_predictions.csv and contain:
        sdssid, z, prob_change, true_label, dr16_path, sdssv_path

    Returns a dictionary of diagnostic measurements.
    """
    if master_grid is None:
        master_grid = np.linspace(4575, 6699, 1024)

    sdssid = row["sdssid"]
    z = float(row["z"])
    prob_change = float(row["prob_change"])
    true_label = int(row["true_label"])

    dr16_path = row["dr16_path"]
    sdssv_path = row["sdssv_path"]

    # 1. Interpolated raw-space spectra
    interp_dr16 = interpolate_raw_spectrum_to_grid(dr16_path, z, master_grid)
    interp_sdssv = interpolate_raw_spectrum_to_grid(sdssv_path, z, master_grid)

    # 2. Continuum-subtracted spectra before MAD
    flat_dr16, cont_dr16 = compute_flat_before_mad(interp_dr16)
    flat_sdssv, cont_sdssv = compute_flat_before_mad(interp_sdssv)

    # 3. Current independent MAD normalization
    ind_dr16 = independent_mad_normalize(flat_dr16)
    ind_sdssv = independent_mad_normalize(flat_sdssv)

    xind_dr16 = ind_dr16["normalized"]
    xind_sdssv = ind_sdssv["normalized"]

    # 4. Diagnostic pair-shared MAD normalization
    pair_norm = pair_shared_mad_normalize(flat_dr16, flat_sdssv)
    xpair_dr16 = pair_norm["x1_normalized"]
    xpair_sdssv = pair_norm["x2_normalized"]

    # 5. Paper-style line fluxes on interpolated raw spectra
    hb_flux_dr16 = paper_style_line_flux(interp_dr16, master_grid, "hb")
    hb_flux_sdssv = paper_style_line_flux(interp_sdssv, master_grid, "hb")
    ha_flux_dr16 = paper_style_line_flux(interp_dr16, master_grid, "ha")
    ha_flux_sdssv = paper_style_line_flux(interp_sdssv, master_grid, "ha")

    def safe_ratio(a, b):
        if not np.isfinite(a) or not np.isfinite(b):
            return np.nan
        denom = min(abs(a), abs(b))
        numer = max(abs(a), abs(b))
        return float(numer / (denom + 1e-8))

    # 6. Broad-wing scores before/current/pair normalization
    result = {
        "sdssid": sdssid,
        "z": z,
        "true_label": true_label,
        "prob_change": prob_change,

        "dr16_ind_median": ind_dr16["median"],
        "dr16_ind_mad": ind_dr16["mad"],
        "sdssv_ind_median": ind_sdssv["median"],
        "sdssv_ind_mad": ind_sdssv["mad"],
        "mad_ratio_max_over_min": safe_ratio(ind_dr16["mad"], ind_sdssv["mad"]),

        "pair_shared_median": pair_norm["shared_median"],
        "pair_shared_mad": pair_norm["shared_mad"],

        "hb_paper_flux_dr16": hb_flux_dr16,
        "hb_paper_flux_sdssv": hb_flux_sdssv,
        "hb_paper_flux_ratio": safe_ratio(hb_flux_dr16, hb_flux_sdssv),

        "ha_paper_flux_dr16": ha_flux_dr16,
        "ha_paper_flux_sdssv": ha_flux_sdssv,
        "ha_paper_flux_ratio": safe_ratio(ha_flux_dr16, ha_flux_sdssv),

        "hb_flat_wing_dr16": broad_wing_score(flat_dr16, master_grid, "hb"),
        "hb_flat_wing_sdssv": broad_wing_score(flat_sdssv, master_grid, "hb"),
        "ha_flat_wing_dr16": broad_wing_score(flat_dr16, master_grid, "ha"),
        "ha_flat_wing_sdssv": broad_wing_score(flat_sdssv, master_grid, "ha"),

        "hb_ind_wing_dr16": broad_wing_score(xind_dr16, master_grid, "hb"),
        "hb_ind_wing_sdssv": broad_wing_score(xind_sdssv, master_grid, "hb"),
        "ha_ind_wing_dr16": broad_wing_score(xind_dr16, master_grid, "ha"),
        "ha_ind_wing_sdssv": broad_wing_score(xind_sdssv, master_grid, "ha"),

        "hb_pair_wing_dr16": broad_wing_score(xpair_dr16, master_grid, "hb"),
        "hb_pair_wing_sdssv": broad_wing_score(xpair_sdssv, master_grid, "hb"),
        "ha_pair_wing_dr16": broad_wing_score(xpair_dr16, master_grid, "ha"),
        "ha_pair_wing_sdssv": broad_wing_score(xpair_sdssv, master_grid, "ha"),

        "hb_flat_peak_dr16": region_peak_stats(flat_dr16, master_grid, "hb"),
        "hb_flat_peak_sdssv": region_peak_stats(flat_sdssv, master_grid, "hb"),
        "ha_flat_peak_dr16": region_peak_stats(flat_dr16, master_grid, "ha"),
        "ha_flat_peak_sdssv": region_peak_stats(flat_sdssv, master_grid, "ha"),

        "hb_ind_peak_dr16": region_peak_stats(xind_dr16, master_grid, "hb"),
        "hb_ind_peak_sdssv": region_peak_stats(xind_sdssv, master_grid, "hb"),
        "ha_ind_peak_dr16": region_peak_stats(xind_dr16, master_grid, "ha"),
        "ha_ind_peak_sdssv": region_peak_stats(xind_sdssv, master_grid, "ha"),
    }

    if make_plot:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        axes[0].plot(master_grid, interp_dr16, label="DR16")
        axes[0].plot(master_grid, interp_sdssv, label="SDSS-V")
        axes[0].set_title(f"Raw interpolated flux | sdssid={sdssid} | true={true_label} | P(change)={prob_change:.4f}")
        axes[0].legend()

        axes[1].plot(master_grid, flat_dr16, label="DR16")
        axes[1].plot(master_grid, flat_sdssv, label="SDSS-V")
        axes[1].set_title("Continuum-subtracted flux before MAD")
        axes[1].legend()

        axes[2].plot(master_grid, xind_dr16, label="DR16")
        axes[2].plot(master_grid, xind_sdssv, label="SDSS-V")
        axes[2].set_title("Current independent MAD normalization")
        axes[2].legend()

        axes[3].plot(master_grid, xpair_dr16, label="DR16")
        axes[3].plot(master_grid, xpair_sdssv, label="SDSS-V")
        axes[3].set_title("Diagnostic pair-shared MAD normalization")
        axes[3].legend()

        for ax in axes:
            ax.axvspan(4797.8, 4927.5, alpha=0.15, label=None)
            ax.axvspan(6477.1, 6652.1, alpha=0.15, label=None)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Rest wavelength [Å]")

        plt.tight_layout()

        out_path = output_dir / f"mad_diagnostic_sdssid_{sdssid}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        result["plot_path"] = str(out_path)

    return result


def run_mad_diagnosis_on_predictions(
    predictions_csv="outputs/real_sdssv_siamese_predictions.csv",
    output_csv="outputs/mad_diagnostic_results.csv",
    n_false_negatives=20,
    n_true_positives=10,
    n_false_positives=10,
    n_top_candidates=10,
):
    """
    Run the diagnostic on representative groups:
        - worst false negatives: true CL-AGN with lowest P(change)
        - true positives: true CL-AGN with highest P(change)
        - false positives: static objects with highest P(change)
        - top candidates overall
    """
    pred = pd.read_csv(predictions_csv)

    groups = []

    false_neg = (
        pred[(pred["true_label"] == 1) & (pred["pred_label"] == 0)]
        .sort_values("prob_change", ascending=True)
        .head(n_false_negatives)
        .copy()
    )
    false_neg["diagnostic_group"] = "false_negative_lowest_prob"
    groups.append(false_neg)

    true_pos_like = (
        pred[pred["true_label"] == 1]
        .sort_values("prob_change", ascending=False)
        .head(n_true_positives)
        .copy()
    )
    true_pos_like["diagnostic_group"] = "true_change_highest_prob"
    groups.append(true_pos_like)

    false_pos = (
        pred[pred["true_label"] == 0]
        .sort_values("prob_change", ascending=False)
        .head(n_false_positives)
        .copy()
    )
    false_pos["diagnostic_group"] = "static_highest_prob"
    groups.append(false_pos)

    top_candidates = (
        pred.sort_values("prob_change", ascending=False)
        .head(n_top_candidates)
        .copy()
    )
    top_candidates["diagnostic_group"] = "top_candidates_overall"
    groups.append(top_candidates)

    selected = pd.concat(groups, ignore_index=True)
    selected = selected.drop_duplicates(subset=["sdssid", "specname_dr16", "specname_sdssv"])

    print(f"Running MAD diagnostics on {len(selected)} selected pairs...")

    results = []
    for i, row in selected.iterrows():
        print(
            f"[{i + 1}/{len(selected)}] "
            f"sdssid={row['sdssid']} | true={row['true_label']} | "
            f"P(change)={row['prob_change']:.4f}"
        )

        try:
            res = diagnose_pair_mad_effect(row, make_plot=True)
            res["diagnostic_group"] = row["diagnostic_group"]
            results.append(res)
        except Exception as exc:
            print(f"FAILED sdssid={row.get('sdssid')}: {exc}")
            results.append({
                "sdssid": row.get("sdssid"),
                "diagnostic_group": row.get("diagnostic_group"),
                "error": str(exc),
            })

    out = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"\nSaved MAD diagnostic results to: {output_csv}")
    return out

# ---------------------------------------------------------------------
# Real-pair dataset
# ---------------------------------------------------------------------

class RealSDSSVSiameseDataset(Dataset):
    """
    Dataset for real paired spectra from the crossmatch pickle.

    This is not SyntheticSiameseDataset because we do not want to generate pairs.
    The pairs already exist in the pickle. But each returned item has the same
    tensor format as the Siamese training dataset:
        x1: [1, 1024]
        x2: [1, 1024]
        y:  [1]
    """

    def __init__(
        self,
        pkl_path: str,
        spectra_root: Optional[str] = None,
        dr16_dir: Optional[str] = None,
        sdssv_dir: Optional[str] = None,
        missing_log_path: str = "outputs/missing_or_failed_spectra_log.csv",
        master_grid: Optional[np.ndarray] = None,
        require_label: bool = True,
        cache_preprocessed: bool = True,
    ):
        self.df = pd.read_pickle(pkl_path).copy()
        self.spectra_root = Path(spectra_root) if spectra_root else None
        self.dr16_dir = Path(dr16_dir) if dr16_dir else self.spectra_root
        self.sdssv_dir = Path(sdssv_dir) if sdssv_dir else self.spectra_root
        self.missing_log_path = Path(missing_log_path)
        self.master_grid = master_grid if master_grid is not None else np.linspace(4575, 6699, 1024)
        self.require_label = require_label
        self.cache_preprocessed = cache_preprocessed
        self._cache: Dict[Tuple[str, float], torch.Tensor] = {}

        required_cols = ["sdssid", "z", "specname_dr16", "specname_sdssv"]
        if self.require_label:
            required_cols.append("label")

        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in pickle: {missing_cols}")

        if self.dr16_dir is None or self.sdssv_dir is None:
            raise ValueError("Provide either --spectra-root or both --dr16-dir and --sdssv-dir")

        self.samples: List[Dict] = []
        self.skipped_rows: List[Dict] = []
        self._index_existing_pairs()
        self._write_skip_log()

    def _resolve_path(self, base_dir: Path, filename: str) -> Path:
        p = Path(str(filename))
        if p.is_absolute():
            return p
        return base_dir / p

    def _index_existing_pairs(self):
        for row in self.df.itertuples(index=False):
            sdssid = getattr(row, "sdssid")
            z = getattr(row, "z")
            specname_dr16 = getattr(row, "specname_dr16")
            specname_sdssv = getattr(row, "specname_sdssv")
            label = getattr(row, "label") if self.require_label else -1

            dr16_path = self._resolve_path(self.dr16_dir, specname_dr16)
            sdssv_path = self._resolve_path(self.sdssv_dir, specname_sdssv)

            missing = []
            if not dr16_path.exists():
                missing.append("dr16")
            if not sdssv_path.exists():
                missing.append("sdssv")

            if missing:
                self.skipped_rows.append({
                    "sdssid": sdssid,
                    "z": z,
                    "label": label,
                    "skip_reason": "missing_file",
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
                "label": float(label),
                "dr16_path": dr16_path,
                "sdssv_path": sdssv_path,
                "specname_dr16": specname_dr16,
                "specname_sdssv": specname_sdssv,
            })

        print(f"Existing file pairs: {len(self.samples):,}")
        print(f"Skipped missing file pairs: {len(self.skipped_rows):,}")

    def _write_skip_log(self):
        if not self.skipped_rows:
            return
        self.missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.skipped_rows).to_csv(self.missing_log_path, index=False)
        print(f"Missing/failed spectra log saved to: {self.missing_log_path}")

    def __len__(self):
        return len(self.samples)

    def _load_processed_tensor(self, path: Path, z: float) -> torch.Tensor:
        key = (str(path), float(z))
        if self.cache_preprocessed and key in self._cache:
            return self._cache[key].clone()

        processed = preprocess_single_spectrum_like_training(
            file_path=path,
            z=z,
            master_grid=self.master_grid,
        )
        x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)  # [1, 1024]

        if self.cache_preprocessed:
            self._cache[key] = x.clone()
        return x

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            x1 = self._load_processed_tensor(s["dr16_path"], s["z"])
            x2 = self._load_processed_tensor(s["sdssv_path"], s["z"])
        except Exception as exc:
            print(f"Skipping sdssid={s['sdssid']} due to error: {exc}")
            return None

        y = torch.tensor([s["label"]], dtype=torch.float32)
        meta = {
            "sdssid": s["sdssid"],
            "z": s["z"],
            "specname_dr16": s["specname_dr16"],
            "specname_sdssv": s["specname_sdssv"],
            "dr16_path": str(s["dr16_path"]),
            "sdssv_path": str(s["sdssv_path"]),
        }
        return x1, x2, y, meta

    def validate_all_preprocessing(self):
        """
        Optional slow pass before inference.
        Removes pairs that exist on disk but fail FITS/preprocessing checks.
        """
        valid_samples = []
        for i, s in enumerate(self.samples):
            try:
                _ = self._load_processed_tensor(s["dr16_path"], s["z"])
                _ = self._load_processed_tensor(s["sdssv_path"], s["z"])
                valid_samples.append(s)
            except Exception as exc:
                self.skipped_rows.append({
                    "sdssid": s["sdssid"],
                    "z": s["z"],
                    "label": s["label"],
                    "skip_reason": "preprocessing_failed",
                    "error": str(exc),
                    "specname_dr16": s["specname_dr16"],
                    "specname_sdssv": s["specname_sdssv"],
                    "dr16_path": str(s["dr16_path"]),
                    "sdssv_path": str(s["sdssv_path"]),
                })

            if (i + 1) % 250 == 0:
                print(f"Validated preprocessing for {i + 1:,}/{len(self.samples):,} pairs")

        self.samples = valid_samples
        self._write_skip_log()
        print(f"Pairs after preprocessing validation: {len(self.samples):,}")


# ---------------------------------------------------------------------
# Collation / inference / metrics
# ---------------------------------------------------------------------

def collate_with_meta(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None
    x1, x2, y, meta = zip(*batch)
    return torch.stack(x1), torch.stack(x2), torch.stack(y), list(meta)


@torch.no_grad()
def predict_real_pairs(model, dataloader, device):
    model.eval()
    all_probs, all_targets, all_meta = [], [], []

    for batch_x1, batch_x2, batch_y, meta in dataloader:
        if batch_x1 is None:
            continue
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        logits = model(batch_x1, batch_x2)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.detach().cpu().numpy().flatten())
        all_targets.extend(batch_y.detach().cpu().numpy().flatten())
        all_meta.extend(meta)

    return np.array(all_probs), np.array(all_targets).astype(int), all_meta


def compute_metrics(y_true, y_prob, threshold: float, beta: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_change": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_change": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_change": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fbeta_change": float(fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn + 1e-8)),
        "false_negative_rate": float(fn / (fn + tp + 1e-8)),
        "num_predicted_positive": int(tp + fp),
        "predicted_positive_fraction": float((tp + fp) / len(y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return y_pred, cm, metrics


def threshold_sweep(y_true, y_prob, beta=0.5):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        _, _, m = compute_metrics(y_true, y_prob, threshold=float(threshold), beta=beta)
        rows.append(m)
    return pd.DataFrame(rows)


def save_predictions(path, y_true, y_prob, y_pred, meta):
    rows = []
    for yt, yp, pred, m in zip(y_true, y_prob, y_pred, meta):
        rows.append({
            **m,
            "true_label": int(yt),
            "prob_change": float(yp),
            "pred_label": int(pred),
            "is_false_positive": int(yt == 0 and pred == 1),
            "is_false_negative": int(yt == 1 and pred == 0),
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved predictions to: {path}")


def plot_and_save_cm(cm, path, show=True):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Static (0)", "Change (1)"],
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Real SDSS/SDSS-V Siamese Test — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    print(f"Saved confusion matrix to: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# Model loading exactly like training
# ---------------------------------------------------------------------

def load_trained_siamese(project_root, config, checkpoint_path, backbone_path, device):
    print("Loading pretrained SpectraNet backbone...")
    base_model = SpectraNet(config)
    base_model.load_state_dict(torch.load(backbone_path, map_location=device))

    model = SiameseSpectraNet(base_model, freeze_backbone=True).to(device)

    print("Loading trained Siamese checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_threshold = checkpoint.get("best_threshold", 0.5)
    best_threshold_metrics = checkpoint.get("best_threshold_metrics", None)

    print(f"Loaded checkpoint threshold: {best_threshold:.4f}")
    if best_threshold_metrics is not None:
        print("Checkpoint validation threshold metrics:")
        print(best_threshold_metrics)

    return model, float(best_threshold), checkpoint


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/Users/amir/Documents/Deep learning/cl-agn classifier")
    p.add_argument("--pkl-path", default="data/dr16_sdssv_crossmatch_lowz.pkl")

    p.add_argument("--spectra-root", default="data/dr16_sdssv_crossmatch")
    p.add_argument("--dr16-dir", default=None)
    p.add_argument("--sdssv-dir", default=None)

    p.add_argument("--checkpoint", default="models/siamese_network/best_siamese_net.pth")
    p.add_argument("--backbone", default="models/selected_backbone/best_spectranet.pth")
    p.add_argument("--threshold", type=float, default=None, help="Defaults to checkpoint best_threshold")

    p.add_argument("--batch-size", type=int, default=None, help="Defaults to config siamese batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--validate-first", action="store_true", help="Slow pass that logs FITS/preprocessing failures before inference")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-show", action="store_true")

    p.add_argument("--missing-log", default="outputs/real_sdssv_missing_or_failed_log.csv")
    p.add_argument("--predictions-csv", default="outputs/real_sdssv_siamese_predictions.csv")
    p.add_argument("--threshold-sweep-csv", default="outputs/real_sdssv_threshold_sweep.csv")
    p.add_argument("--cm-path", default="outputs/real_sdssv_confusion_matrix.png")
    p.add_argument("--metrics-json", default="outputs/real_sdssv_metrics.json")

    return p.parse_args()


def main():
    global SpectraNet, SiameseSpectraNet, load_config
    global remove_sky_line, morphological_continuum_subtraction

    args = parse_args()
    project_root = add_project_to_path(args.project_root)

    from utils import load_config as _load_config
    from architectures import SpectraNet as _SpectraNet, SiameseSpectraNet as _SiameseSpectraNet
    from data_preprocessing import (
        remove_sky_line as _remove_sky_line,
        morphological_continuum_subtraction as _morphological_continuum_subtraction,
    )

    load_config = _load_config
    SpectraNet = _SpectraNet
    SiameseSpectraNet = _SiameseSpectraNet
    remove_sky_line = _remove_sky_line
    morphological_continuum_subtraction = _morphological_continuum_subtraction

    config = load_config(os.path.join(project_root, "config.yml"))

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or os.path.join(project_root, "models/siamese_network/best_siamese_net.pth")
    backbone_path = args.backbone or os.path.join(project_root, "models/selected_backbone/best_spectranet.pth")

    # Exact grid from run_preprocessing(...).
    master_grid = np.linspace(4575, 6699, 1024)

    dataset = RealSDSSVSiameseDataset(
        pkl_path=args.pkl_path,
        spectra_root=args.spectra_root,
        dr16_dir=args.dr16_dir,
        sdssv_dir=args.sdssv_dir,
        missing_log_path=args.missing_log,
        master_grid=master_grid,
        require_label=True,
        cache_preprocessed=not args.no_cache,
    )

    if args.validate_first:
        dataset.validate_all_preprocessing()

    if len(dataset) == 0:
        raise RuntimeError("No valid pairs remain after file/preprocessing checks.")

    batch_size = args.batch_size or int(config["siamese_training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_meta,
        pin_memory=(device.type == "cuda"),
    )

    model, checkpoint_threshold, checkpoint = load_trained_siamese(
        project_root=project_root,
        config=config,
        checkpoint_path=checkpoint_path,
        backbone_path=backbone_path,
        device=device,
    )

    threshold = checkpoint_threshold if args.threshold is None else args.threshold
    print(f"Using evaluation threshold: {threshold:.4f}")

    y_prob, y_true, meta = predict_real_pairs(model, loader, device)
    y_pred, cm, metrics = compute_metrics(y_true, y_prob, threshold=threshold, beta=0.5)

    print("\n=== Real SDSS/SDSS-V Siamese metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Classification report ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Static (0)", "Change (1)"],
        zero_division=0,
    ))

    save_predictions(args.predictions_csv, y_true, y_prob, y_pred, meta)
    diag = run_mad_diagnosis_on_predictions(
    predictions_csv="outputs/real_sdssv_siamese_predictions.csv",
    output_csv="outputs/mad_diagnostic_results.csv",
    n_false_negatives=20,
    n_true_positives=10,
    n_false_positives=10,
    n_top_candidates=10,
)
    plot_and_save_cm(cm, args.cm_path, show=not args.no_show)

    sweep_df = threshold_sweep(y_true, y_prob, beta=0.5)
    Path(args.threshold_sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(args.threshold_sweep_csv, index=False)
    print(f"Saved threshold sweep to: {args.threshold_sweep_csv}")

    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {args.metrics_json}")

    print("\n=== Quick sanity checks ===")
    print(f"Evaluated pairs: {len(y_true):,}")
    print(f"True change count: {int(y_true.sum()):,}")
    print(f"True change fraction: {float(y_true.mean()):.6f}")
    print(f"Mean P(change): {float(y_prob.mean()):.6f}")
    print(f"Median P(change): {float(np.median(y_prob)):.6f}")


if __name__ == "__main__":
    main()
