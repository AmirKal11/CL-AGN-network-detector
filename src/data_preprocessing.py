import pandas as pd
import numpy as np
import astropy 
from astropy.table import Table, hstack
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import glob
import argparse
import torch
import torch.nn.functional as F
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


OIII_META_COLS = [
    "OIII_FLUX",
    "OIII_FLUX_ERR",
    "OIII_LUM",
    "OIII_LUM_ERR",
    "LOG_OIII_LUM",
    "OIII_SNR",
    "OIII_VALID",
    "OIII_QUALITY_FLAG",
    "OIII_SIGMA",
    "OIII_VOFF",
    "OIII_CHI2",
    "OIII_FLUX_CGS",
    "OIII_FLUX_SOURCE",
    "OIII_USABLE",
]

def add_common_oiii_flux_columns(df):
    """
    Create a common OIII flux column for both catalogues.

    Type 2:
        MPA-JHU gives OIII_FLUX, usually in units of 1e-17 erg/s/cm^2.
        Therefore:
            OIII_FLUX_CGS = OIII_FLUX * 1e-17

    Type 1:
        Shen gives OIII luminosity.
        Therefore:
            OIII_FLUX_CGS = OIII_LUM / (4*pi*d_L^2)

    Creates:
        OIII_FLUX_CGS
        OIII_FLUX_SOURCE
        OIII_USABLE
    """
    df = df.copy()

    if "OIII_FLUX_CGS" not in df.columns:
        df["OIII_FLUX_CGS"] = np.nan

    df["OIII_FLUX_SOURCE"] = "missing"

    # Make sure OIII_LUM exists from LOG_OIII_LUM if needed.
    if "OIII_LUM" not in df.columns:
        df["OIII_LUM"] = np.nan

    if "LOG_OIII_LUM" in df.columns:
        valid_log_lum = (
            df["OIII_LUM"].isna()
            & df["LOG_OIII_LUM"].notna()
            & np.isfinite(df["LOG_OIII_LUM"].astype(float))
        )
        df.loc[valid_log_lum, "OIII_LUM"] = (
            10 ** df.loc[valid_log_lum, "LOG_OIII_LUM"].astype(float)
        )

    # ------------------------------------------------------------
    # Type 2 / MPA-JHU flux-based OIII
    # ------------------------------------------------------------
    if "OIII_FLUX" in df.columns:
        valid_flux = (
            df["OIII_FLUX"].notna()
            & np.isfinite(df["OIII_FLUX"].astype(float))
            & (df["OIII_FLUX"].astype(float) > 0)
        )

        df.loc[valid_flux, "OIII_FLUX_CGS"] = (
            df.loc[valid_flux, "OIII_FLUX"].astype(float) * 1e-17
        )
        df.loc[valid_flux, "OIII_FLUX_SOURCE"] = "catalog_flux"

    # ------------------------------------------------------------
    # Type 1 / Shen luminosity-based OIII
    # Only fill rows that do not already have flux.
    # ------------------------------------------------------------
    z_col = "z" if "z" in df.columns else None

    if z_col is not None and "OIII_LUM" in df.columns:
        valid_lum = (
            df["OIII_FLUX_CGS"].isna()
            & df[z_col].notna()
            & df["OIII_LUM"].notna()
            & np.isfinite(df[z_col].astype(float))
            & np.isfinite(df["OIII_LUM"].astype(float))
            & (df[z_col].astype(float) > 0)
            & (df["OIII_LUM"].astype(float) > 0)
        )

        if valid_lum.any():
            z = df.loc[valid_lum, z_col].astype(float).values
            lum = df.loc[valid_lum, "OIII_LUM"].astype(float).values

            d_l_cm = COSMO.luminosity_distance(z).to(u.cm).value
            flux_from_lum = lum / (4.0 * np.pi * d_l_cm**2)

            df.loc[valid_lum, "OIII_FLUX_CGS"] = flux_from_lum
            df.loc[valid_lum, "OIII_FLUX_SOURCE"] = "luminosity_converted"

    df["OIII_USABLE"] = (
        df["OIII_FLUX_CGS"].notna()
        & np.isfinite(df["OIII_FLUX_CGS"].astype(float))
        & (df["OIII_FLUX_CGS"].astype(float) > 0)
    )

    return df

def build_download_filename(row):
    """
    Build the actual downloaded SDSS filename:
        spec-PPPP-MMMMM-FFFF.fits

    Handles both Type 1 and Type 2 CSV naming conventions.
    """

    plate = int(row["PLATE"])

    if "mjd" in row:
        mjd = int(row["mjd"])
    elif "MJD" in row:
        mjd = int(row["MJD"])
    elif "MJD_class_table" in row:
        mjd = int(row["MJD_class_table"])
    else:
        raise KeyError("Could not find MJD column.")

    if "fiber" in row:
        fiber = int(row["fiber"])
    elif "FIBER" in row:
        fiber = int(row["FIBER"])
    elif "FIBERID_class_table" in row:
        fiber = int(row["FIBERID_class_table"])
    elif "FIBERID" in row:
        fiber = int(row["FIBERID"])
    else:
        raise KeyError("Could not find fiber column.")

    return f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"


def load_candidate_metadata(csv_path):
    """
    Load Type 1 / Type 2 candidate CSV and create a dictionary:
        downloaded spectrum filename -> metadata dict

    This lets process_single_spectrum save OIII columns into the final parquet.
    """

    if csv_path is None or not os.path.exists(csv_path):
        print(f"Warning: candidate metadata CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    df = add_common_oiii_flux_columns(df)

    df["download_filename"] = df.apply(build_download_filename, axis=1)

    meta_cols = ["download_filename"]

    for col in OIII_META_COLS:
        if col in df.columns:
            meta_cols.append(col)

    # Useful extra columns if available
    for col in ["SDSS_NAME", "RA", "DEC", "PLATE", "mjd", "MJD", "fiber", "FIBER"]:
        if col in df.columns and col not in meta_cols:
            meta_cols.append(col)

    meta_df = df[meta_cols].copy()

    metadata = {}
    for _, row in meta_df.iterrows():
        fname = row["download_filename"]
        metadata[fname] = row.drop(labels=["download_filename"]).to_dict()

    print(f"Loaded metadata for {len(metadata)} spectra from {csv_path}")
    return metadata


def morphological_continuum_subtraction(
    x,
    window_size=151,
    clip_max=4.0,
    taper_len=5,
    apply_mad_scaling=False,
):
    """
    Lightweight continuum removal using wide average pooling.

    New behavior for OIII experiment:
        - By default, this function does NOT apply independent MAD scaling.
        - It only subtracts the smooth continuum and applies edge tapering.
        - OIII columns are saved as metadata for later pair matching / calibration.

    x shape: [Batch, 1, Sequence_Length]
    """

    # 1. Pad the sequence to handle edge artifacts smoothly
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode="reflect")

    # 2. Estimate smooth continuum
    continuum = F.avg_pool1d(
        x_padded,
        kernel_size=window_size,
        stride=1,
    )

    # 3. Subtract continuum
    x_flattened = x - continuum

    # 4. OLD MAD SCALING — disabled by default for OIII/difference-spectrum work.
    # This was useful for single-spectrum Type1/Type2 classification, but it removes
    # per-spectrum amplitude scale and can hurt same-object CL-AGN comparisons.
    if apply_mad_scaling:
        median = x_flattened.median(dim=-1, keepdim=True).values
        mad = (x_flattened - median).abs().median(dim=-1, keepdim=True).values
        x_processed = (x_flattened - median) / (mad * 1.4826 + 1e-8)
    else:
        x_processed = x_flattened

    # 5. Optional clipping — also disabled unless you explicitly turn it back on.
    # x_processed = torch.clamp(x_processed, min=-10.0, max=clip_max)

    # 6. Edge tapering
    seq_len = x.shape[-1]
    taper = torch.ones(seq_len, device=x.device)

    fade = torch.linspace(0, 1, taper_len, device=x.device)
    taper[:taper_len] = fade
    taper[-taper_len:] = torch.flip(fade, dims=[0])

    taper = taper.view(1, 1, -1)
    x_final = x_processed * taper

    return x_final




def standardize_flux(flux_array):
    """Standardizes a flux array by mean and standard deviation."""
    mean = np.nanmean(flux_array)
    std = np.nanstd(flux_array)
    # Adding a small epsilon to avoid division by zero
    normalized_flux = (flux_array - mean) / (std + 1e-8)
    return normalized_flux

def get_redshift(hdul):
    """Extracts redshift from the 'SPECOBJ' extension of the HDUList."""
    try:
        if 'SPECOBJ' in hdul:
            data = hdul['SPECOBJ'].data
            return data['Z'][0]
        else:
            return None
    except Exception as e:
        return None

def get_snr(hdul):
    """Extracts the median SNR from the 'SPECOBJ' extension."""
    try:
        if 'SPECOBJ' in hdul:
            data = hdul['SPECOBJ'].data
            return data['SN_MEDIAN_ALL'][0] 
        return None
    except Exception:
        return None

def remove_sky_line(wave_obs, flux_obs, line_center=5577.3, window=20.0, threshold=4.0):
    """
    Removes sky line residuals (e.g., 5577 A [O I]) by checking for a sudden peak.
    If a peak > threshold * local_std is found, it linearly interpolates over the region.
    """
    mask_line = (wave_obs > line_center - window/2) & (wave_obs < line_center + window/2)
    if not np.any(mask_line):
        return flux_obs
        
    mask_cont = ((wave_obs > line_center - window*1.5) & (wave_obs <= line_center - window/2)) | \
                ((wave_obs >= line_center + window/2) & (wave_obs < line_center + window*1.5))
                
    if not np.any(mask_cont):
        return flux_obs
        
    local_med = np.nanmedian(flux_obs[mask_cont])
    local_std = np.nanstd(flux_obs[mask_cont])
    line_max = np.nanmax(flux_obs[mask_line])
    
    if local_std > 0 and line_max > local_med + threshold * local_std:
        x_cont = wave_obs[mask_cont]
        y_cont = flux_obs[mask_cont]
        if len(x_cont) > 1:
            f = interp1d(x_cont, y_cont, kind='linear', bounds_error=False, fill_value='extrapolate')
            flux_cleaned = flux_obs.copy()
            flux_cleaned[mask_line] = f(wave_obs[mask_line])
            return flux_cleaned
            
    return flux_obs

def process_single_spectrum(
    file_path,
    agn_type,
    master_grid,
    metadata_lookup=None,
    apply_mad_scaling=False,
):
    try:
        with fits.open(file_path) as hdul:
            z = get_redshift(hdul)
            snr = get_snr(hdul)
            
            if z is None or snr is None: return None
            
            # SDSS SPEC_ID or fallback to filename
            obj_id = hdul[0].header.get('SPEC_ID', os.path.basename(file_path))
            
            # Extension 1 contains the 'COADD' spectrum in SDSS
            data = hdul[1].data
            flux_obs = data['flux']
            
            # Handle wavelengths: check for 'loglam' (common in SDSS) or 'wavelength'
            if 'loglam' in data.names:
                wave_obs = 10**data['loglam']
            elif 'wavelength' in data.names:
                wave_obs = data['wavelength']
            else:
                header = hdul[1].header
                wave_obs = header['CRVAL1'] + np.arange(len(flux_obs)) * header['CDELT1']
            
            # Remove prominent night sky line at 5577A if it spikes
            flux_obs = remove_sky_line(wave_obs, flux_obs, line_center=5577.3)
            
            # 3. Rest-frame correction
            wave_rest = wave_obs / (1 + z)
            flux_rest = flux_obs * (1 + z)

            # 4. Interpolate to fixed grid FIRST using NaN for out-of-bounds
            f_interp = interp1d(wave_rest, flux_rest, bounds_error=False, fill_value=np.nan)
            interpolated_flux = f_interp(master_grid)
            
            # 5. Handle NaNs in RAW space using the median (prevents the rolling average from crashing)
            valid_median = np.nanmedian(interpolated_flux)
            interpolated_flux = np.nan_to_num(interpolated_flux, nan=valid_median)
            
            # 6. Convert to PyTorch Tensor and add Batch/Channel dimensions -> shape [1, 1, 1024]
            tensor_flux = torch.tensor(interpolated_flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 7. Apply Morphological Continuum Subtraction 
            # (This safely handles the subtraction, the standardization, the clipping, and the tapering all at once!)
            processed_tensor = morphological_continuum_subtraction(
                tensor_flux,
                window_size=151,
                taper_len=5,
                clip_max=4.0,
                apply_mad_scaling=apply_mad_scaling,
            )
            
            # 8. Squeeze it back down to a flat 1D NumPy array for your Parquet file
            processed_flux = processed_tensor.squeeze().numpy()
            filename = os.path.basename(file_path)

            result = {
                "filename": filename,
                "obj_id": obj_id,
                "agn_type": agn_type,
                "z": z,
                "flux_array": processed_flux,
                "snr": snr,
            }

            # Attach OIII / catalogue metadata if available.
            if metadata_lookup is not None and filename in metadata_lookup:
                for key, value in metadata_lookup[filename].items():
                    result[key] = value
            else:
                for col in OIII_META_COLS:
                    result[col] = np.nan

            return result
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def build_agn_catalog(
    type1_path,
    type2_path,
    master_grid,
    type1_metadata_csv=None,
    type2_metadata_csv=None,
    apply_mad_scaling=False,
):
    type1_metadata = load_candidate_metadata(type1_metadata_csv)
    type2_metadata = load_candidate_metadata(type2_metadata_csv)

    files_type1 = [
        (f, 1, type1_metadata)
        for f in glob.glob(os.path.join(type1_path, "*.fits"))
    ]
    files_type2 = [
        (f, 2, type2_metadata)
        for f in glob.glob(os.path.join(type2_path, "*.fits"))
    ]

    all_tasks = files_type1 + files_type2
    print(f"Processing {len(all_tasks)} spectra...")
    
    results = Parallel(n_jobs=-2)(
    delayed(process_single_spectrum)(
        f_path,
        a_type,
        master_grid,
        metadata_lookup,
        apply_mad_scaling,
    )
    for f_path, a_type, metadata_lookup in all_tasks
)
    
    # Filter out None results from failed loads
    results = [r for r in results if r is not None]
            
    # Assemble Metadata
    meta_keys = [
        "filename",
        "obj_id",
        "agn_type",
        "z",
        "snr",
    ] + OIII_META_COLS

    # Include only keys that exist in at least one result.
    # This prevents errors if some optional OIII columns are missing.
    all_result_keys = set()
    for r in results:
        all_result_keys.update(r.keys())

    meta_keys = [k for k in meta_keys if k in all_result_keys]

    meta_df = pd.DataFrame([
        {key: r.get(key, np.nan) for key in meta_keys}
        for r in results
    ])
    
    # Assemble Fluxes (Matrix construction is faster than row-by-row append)
    flux_matrix = np.array([r['flux_array'] for r in results])
    flux_df = pd.DataFrame(flux_matrix, columns=master_grid)
    # Merge horizontally
    final_df = pd.concat([meta_df, flux_df], axis=1)
    final_df.columns = final_df.columns.astype(str)
    return final_df


def clean_dataset(
    df,
    max_zeros_pct=0.5,
    min_snr=5.0,
    max_flux_outlier=None,
    max_neg_flux=None,
    require_valid_oiii=False,
):
    """
    Cleans the dataset.

    For the OIII/no-MAD experiment:
        - OIII metadata columns are excluded from the flux matrix.
        - max_flux_outlier and max_neg_flux are optional because the spectra
          are no longer MAD-scaled.
        - require_valid_oiii can be used to keep only spectra with good OIII flags.
    """

    meta_cols = [
        "filename",
        "obj_id",
        "agn_type",
        "z",
        "snr",
    ] + OIII_META_COLS

    # Only wavelength columns should be treated as flux pixels.
    flux_cols = [c for c in df.columns if c not in meta_cols]

    flux_mat = df[flux_cols].values.astype(float)

    # 1. Filter bad coverage
    zeros_pct = (flux_mat == 0.0).mean(axis=1)
    valid_coverage = zeros_pct <= max_zeros_pct

    # 2. Optional positive outlier filtering.
    # Disabled by default because non-MAD-scaled fluxes do not have the old scale.
    if max_flux_outlier is not None:
        max_flux = np.nanmax(flux_mat, axis=1)
        valid_outlier = max_flux <= max_flux_outlier
    else:
        valid_outlier = np.ones(len(df), dtype=bool)

    # 3. Filter low SNR
    valid_snr = df["snr"] >= min_snr

    # 4. Optional negative outlier filtering.
    if max_neg_flux is not None:
        min_flux = np.nanmin(flux_mat, axis=1)
        valid_neg_flux = min_flux >= -max_neg_flux
    else:
        valid_neg_flux = np.ones(len(df), dtype=bool)

    if require_valid_oiii:
        if "OIII_VALID" in df.columns:
            valid_oiii = (
                df["OIII_VALID"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .values
            )
        elif "OIII_USABLE" in df.columns:
            valid_oiii = (
                df["OIII_USABLE"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .values
            )
        elif "OIII_FLUX_CGS" in df.columns:
            valid_oiii = (
                df["OIII_FLUX_CGS"].notna()
                & np.isfinite(df["OIII_FLUX_CGS"].astype(float))
                & (df["OIII_FLUX_CGS"].astype(float) > 0)
            ).values
        else:
            print("Warning: require_valid_oiii=True, but neither OIII_USABLE nor OIII_FLUX_CGS exists.")
            valid_oiii = np.zeros(len(df), dtype=bool)
    else:
        valid_oiii = np.ones(len(df), dtype=bool)


    # 6. Drop rows with all-NaN or non-finite flux values
    valid_finite = np.isfinite(flux_mat).any(axis=1)

    good_mask = (
        valid_coverage
        & valid_outlier
        & valid_snr
        & valid_neg_flux
        & valid_oiii
        & valid_finite
    )

    df_clean = df[good_mask].copy()

    print(f"Original spectra: {len(df)}")
    print(f"Dropped due to coverage:       {(~valid_coverage).sum()}")

    if max_flux_outlier is not None:
        print(f"Dropped due to pos outliers:   {(~valid_outlier).sum()}")
    else:
        print("Dropped due to pos outliers:   skipped")

    print(f"Dropped due to low SNR:        {(~valid_snr).sum()}")

    if max_neg_flux is not None:
        print(f"Dropped due to neg flux:       {(~valid_neg_flux).sum()}")
    else:
        print("Dropped due to neg flux:       skipped")

    if require_valid_oiii:
        print(f"Dropped due to invalid OIII:   {(~valid_oiii).sum()}")

    print(f"Dropped due to non-finite flux: {(~valid_finite).sum()}")
    print(f"Remaining clean spectra:       {len(df_clean)}")

    return df_clean
    """
    Cleans the dataset by removing spectra with low coverage, poor SNR,
    extreme positive outliers, or extreme negative flux values.
    
    Parameters
    ----------
    max_neg_flux : float
        After z-normalization, if any flux value in a spectrum is below
        -max_neg_flux (i.e. has a negative dip larger than this threshold),
        the entire spectrum is discarded.
    """
    meta_cols = ['filename', 'obj_id', 'agn_type', 'z', 'snr']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    flux_mat = df[flux_cols].values
    
    # 1. Filter bad coverage
    zeros_pct = (flux_mat == 0.0).mean(axis=1)
    valid_coverage = zeros_pct <= max_zeros_pct
    
    # 2. Filter extreme positive outliers
    max_flux = flux_mat.max(axis=1)
    valid_outlier = max_flux <= max_flux_outlier
    
    # 3. Filter low SNR
    valid_snr = df['snr'] >= min_snr
    
    # 4. Filter extreme negative flux after z-normalization
    #    A spectrum is bad if min(flux) < -max_neg_flux
    min_flux = flux_mat.min(axis=1)
    valid_neg_flux = min_flux >= -max_neg_flux
    
    # Combine masks
    good_mask = valid_coverage & valid_outlier & valid_snr & valid_neg_flux
    df_clean = df[good_mask].copy()
    
    print(f"Original spectra: {len(df)}")
    print(f"Dropped due to coverage:       {(~valid_coverage).sum()}")
    print(f"Dropped due to pos outliers:   {(~valid_outlier).sum()}")
    print(f"Dropped due to low SNR:        {(~valid_snr).sum()}")
    print(f"Dropped due to neg flux (<-{max_neg_flux}): {(~valid_neg_flux).sum()}")
    print(f"Remaining clean spectra:       {len(df_clean)}")
    
    return df_clean

def run_preprocessing(
    mode="full",
    existing_parquet="data/O3_normalized_network/processed_agn_catalog_cut.parquet",
    output="data/O3_normalized_network/processed_agn_OIII_ready.parquet",
    type1_metadata_csv="data/O3_normalized_network/type1_candidates.csv",
    type2_metadata_csv="data/O3_normalized_network/type2_candidates.csv",
    apply_mad_scaling=False,
    require_valid_oiii=False,
):
    """
    Main preprocessing pipeline.
    
    Parameters
    ----------
    mode : str
        'full'     - process original type1/ and type2/ directories.
        'new_only' - process type1_new/ and type2_new/ only.
        'merge'    - process new dirs and merge with existing parquet.
    existing_parquet : str
        Path to existing parquet file (used only in 'merge' mode).
    output : str
        Path to save the cleaned output parquet.
    """
    master_grid = np.linspace(4575, 6699, 1024)
    
    if mode == 'full':
        print("=== Processing original type1/type2 directories ===")
        df = build_agn_catalog(
            type1_path="data/O3_normalized_network/Type1/",
            type2_path="data/O3_normalized_network/Type2/",
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
        )
    elif mode == 'new_only':
        print("=== Processing NEW type1_new/type2_new directories ===")
        df = build_agn_catalog(
            type1_path="data/O3_normalized_network/Type1/",
            type2_path="data/O3_normalized_network/Type2/",
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
        )
    elif mode == 'merge':
        print("=== Processing NEW data and merging with existing parquet ===")
        df = build_agn_catalog(
            type1_path="data/O3_normalized_network/Type1/",
            type2_path="data/O3_normalized_network/Type2/",
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
        )
        print(f"\nNew spectra processed: {len(df_new)}")
        
        print(f"Loading existing parquet: {existing_parquet}")
        df_existing = pd.read_parquet(existing_parquet)
        print(f"Existing spectra: {len(df_existing)}")
        
        # Merge: drop duplicates based on filename to avoid re-adding existing spectra
        df = pd.concat([df_existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset='filename', keep='first')
        print(f"Combined (deduplicated): {len(df)}")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: 'full', 'new_only', 'merge'.")
    
    print("\nCleaning dataset...")
    df_clean = clean_dataset(
    df,
        max_zeros_pct=0.5,
        min_snr=5.0,
        max_flux_outlier=None,
        max_neg_flux=None,
        require_valid_oiii=require_valid_oiii,
    )

    df_clean.to_parquet(output)
    print(f'\nSaved cleaned df with {len(df_clean)} spectra to {output}')
    return df_clean


# --- EXECUTION ---
if __name__ == "__main__":
    run_preprocessing(
        mode="new_only",
        output="data/O3_normalized_network/processed_agn_new_OIII_ready_no_MAD.parquet",
        type1_metadata_csv="data/O3_normalized_network/type1_candidates.csv",
        type2_metadata_csv="data/O3_normalized_network/type2_candidates.csv",
        apply_mad_scaling=False,
        require_valid_oiii=True,
    )
