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


# OIII functions and columns removed to support generic metadata.

def build_download_filename(row):
    """
    Build the actual downloaded SDSS filename:
        spec-PPPP-MMMMM-FFFF.fits

    Handles both Type 1 and Type 2 CSV naming conventions.
    """

    plate = int(row.get("PLATE", 0))

    if "TARGETID" in row:
        return f"desi-spec-{int(row['TARGETID'])}.fits"

    if "mjd" in row:
        mjd = int(row["mjd"])
    elif "MJD" in row:
        mjd = int(row["MJD"])
    elif "MJD_class_table" in row:
        mjd = int(row["MJD_class_table"])
    else:
        mjd = 0

    if "fiber" in row:
        fiber = int(row["fiber"])
    elif "FIBER" in row:
        fiber = int(row["FIBER"])
    elif "FIBERID_class_table" in row:
        fiber = int(row["FIBERID_class_table"])
    elif "FIBERID" in row:
        fiber = int(row["FIBERID"])
    else:
        fiber = 0

    if plate > 0:
        return f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"
    return "unknown.fits"


def load_candidate_metadata(csv_path):
    """
    Load candidate CSV and create a dictionary:
        downloaded spectrum filename -> metadata dict
    """
    if csv_path is None or not os.path.exists(csv_path):
        print(f"Warning: candidate metadata CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    df["download_filename"] = df.apply(build_download_filename, axis=1)

    # Only keep essential metadata to prevent bloat
    keep_cols = ["TARGETID", "TARGET_RA", "TARGET_DEC", "RA", "DEC", "SDSS_NAME", "Z", "z"]

    metadata = {}
    for _, row in df.iterrows():
        fname = row["download_filename"]
        
        row_meta = {}
        for col in keep_cols:
            if col in df.columns:
                row_meta[col] = row[col]
                
        metadata[fname] = row_meta

    print(f"Loaded metadata for {len(metadata)} spectra from {csv_path}")
    return metadata


@torch.no_grad()
def morphological_continuum_subtraction(
    x,
    window_size=173,
    clip_max=4.0,
    taper_len=5,
    apply_mad_scaling=False,
    valid_mask=None,
    subtract_continuum=True,
):
    """
    Lightweight continuum removal using wide average pooling.

    New behavior for OIII experiment:
        - By default, this function does NOT apply independent MAD scaling.
        - It only subtracts the smooth continuum and applies edge tapering.
        - OIII columns are saved as metadata for later pair matching / calibration.

    subtract_continuum=False keeps the full continuum (v2 default, set by
    build_unified_ssl_parquet); True (the default here) preserves v1 behaviour.

    x shape: [Batch, 1, Sequence_Length]
    """

    # 1. Pad the sequence to handle edge artifacts smoothly
    pad = window_size // 2

    # The validity mask is needed below by MAD scaling and the zero-fill step
    # whether or not the continuum is removed, so resolve it up front.
    vm = valid_mask.to(dtype=x.dtype) if valid_mask is not None else None

    # 2-3. Estimate and subtract the smooth continuum. v1 anti-shortcut
    #      representation; v2 sets subtract_continuum=False and keeps the full
    #      continuum (the SSL encoder learns more from a full spectrum than
    #      from a near-noise residual, and the cross-object classification
    #      shortcut does not apply to SSL or same-object change detection).
    #      When subtracting and a mask is supplied the continuum is a MASKED
    #      moving average over covered pixels only, so the zero-filled
    #      out-of-coverage region does not drag it toward zero near the edges.
    if subtract_continuum:
        if valid_mask is not None:
            x_pad = F.pad(x * vm, (pad, pad), mode="reflect")
            v_pad = F.pad(vm, (pad, pad), mode="reflect")
            num = F.avg_pool1d(x_pad, kernel_size=window_size, stride=1)
            den = F.avg_pool1d(v_pad, kernel_size=window_size, stride=1)
            continuum = num / (den + 1e-8)
        else:
            x_padded = F.pad(x, (pad, pad), mode="reflect")
            continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
        x_flattened = x - continuum
    else:
        x_flattened = x

    # 4. OLD MAD SCALING — disabled by default for OIII/difference-spectrum work.
    # This was useful for single-spectrum Type1/Type2 classification, but it removes
    # per-spectrum amplitude scale and can hurt same-object CL-AGN comparisons.
    # When enabled, the median/MAD are taken over covered pixels only.
    if apply_mad_scaling:
        if valid_mask is not None:
            x_processed = torch.zeros_like(x_flattened)
            for b in range(x_flattened.shape[0]):
                m = vm[b, 0] > 0.5
                if int(m.sum()) < 2:
                    continue
                vals = x_flattened[b, 0][m]
                median = vals.median()
                mad = (vals - median).abs().median()
                x_processed[b, 0] = (x_flattened[b, 0] - median) / (mad * 1.4826 + 1e-8)
        else:
            median = x_flattened.median(dim=-1, keepdim=True).values
            mad = (x_flattened - median).abs().median(dim=-1, keepdim=True).values
            x_processed = (x_flattened - median) / (mad * 1.4826 + 1e-8)
    else:
        x_processed = x_flattened

    # 4b. Zero out non-covered pixels so they carry the 0.0 "missing" sentinel.
    if valid_mask is not None:
        x_processed = x_processed * vm

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
            if 'SN_MEDIAN_ALL' in data.names:
                return data['SN_MEDIAN_ALL'][0] 
    except Exception:
        pass
    
    # Calculate manually if not in SPECOBJ
    try:
        data = hdul[1].data
        names_lower = [n.lower() for n in data.names] if hasattr(data, 'names') else []
        if 'flux' in names_lower:
            flux = data['flux']
            if 'ivar' in names_lower:
                ivar = data['ivar']
                valid = (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
                if np.any(valid):
                    snr_arr = flux[valid] * np.sqrt(ivar[valid])
                    return np.nanmedian(snr_arr)
            valid = np.isfinite(flux)
            if np.any(valid):
                std_f = np.nanstd(flux[valid])
                if std_f > 0:
                    return np.nanmean(flux[valid]) / std_f
    except Exception:
        pass
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
    file_metadata=None,
    apply_mad_scaling=False,
    continuum_window_size=173,
    subtract_continuum=True,
):
    try:
        with fits.open(file_path) as hdul:
            z = None
            if file_metadata:
                z = file_metadata.get("Z", file_metadata.get("z", None))
                
            if z is None or pd.isna(z):
                z = get_redshift(hdul)
                
            snr = get_snr(hdul)
            
            if z is None or pd.isna(z) or snr is None: return None
            
            # SDSS SPEC_ID or fallback to filename
            obj_id = hdul[0].header.get('SPEC_ID', os.path.basename(file_path))
            
            # Extension 1 contains the 'COADD' spectrum in SDSS
            data = hdul[1].data
            flux_obs = data['flux']
            
            names_lower = [n.lower() for n in data.names] if hasattr(data, 'names') else []
            
            # Handle wavelengths: check for 'loglam' (common in SDSS) or 'wavelength'
            if 'loglam' in names_lower:
                wave_obs = 10**data['loglam']
            elif 'wavelength' in names_lower:
                wave_obs = data['wavelength']
            else:
                header = hdul[1].header
                wave_obs = header['CRVAL1'] + np.arange(len(flux_obs)) * header['CDELT1']
            
            # Remove prominent night sky line at 5577A if it spikes
            flux_obs = remove_sky_line(wave_obs, flux_obs, line_center=5577.3)
            
            # 3. Rest-frame correction
            wave_rest = wave_obs / (1 + z)
            flux_rest = flux_obs * (1 + z)

            # 4. Interpolate to the fixed grid; NaN outside the observed range.
            f_interp = interp1d(wave_rest, flux_rest, bounds_error=False, fill_value=np.nan)
            interpolated_flux = f_interp(master_grid)

            # 5. Validity mask + zero-fill (SpectraNet convention): pixels the
            #    spectrograph never covered, once shifted to rest frame, are
            #    flagged and set to 0.0 -- NOT median-filled, which would paint
            #    a fake continuum the network could learn as a redshift cue.
            valid = np.isfinite(interpolated_flux)
            if int(valid.sum()) < 50:
                return None
            interpolated_flux = np.nan_to_num(interpolated_flux, nan=0.0)

            # 6. Convert to PyTorch Tensors -> shape [1, 1, L]
            tensor_flux = torch.tensor(interpolated_flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            valid_tensor = torch.tensor(valid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # 7. Morphological continuum subtraction over covered pixels only.
            #    Non-covered pixels stay exactly 0.0 in the output.
            processed_tensor = morphological_continuum_subtraction(
                tensor_flux,
                window_size=continuum_window_size,
                taper_len=5,
                clip_max=4.0,
                apply_mad_scaling=apply_mad_scaling,
                valid_mask=valid_tensor,
                subtract_continuum=subtract_continuum,
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

            # Attach catalogue metadata if available. The redshift key is
            # SKIPPED here: result["z"] above is already the canonical
            # redshift (taken from this same metadata or the FITS header).
            # Copying the CSV's redshift back in would create a second,
            # survey-dependent column -- DESI catalogs name it `Z`, SDSS
            # catalogs name it `z` -- so the pooled parquet would carry both
            # a `z` and a `Z`. Skipping any `z`/`Z` key guarantees exactly
            # one redshift column, `z`, identical for both surveys.
            if file_metadata:
                for key, value in file_metadata.items():
                    if str(key).lower() == "z":
                        continue
                    result[key] = value

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
    continuum_window_size=173,
    restrict_to_metadata=False,
    subtract_continuum=True,
):
    """
    restrict_to_metadata : bool
        False (default) -- every *.fits file in type1_path/type2_path is
        processed; the metadata CSV only attaches extra columns.
        True -- the metadata CSV acts as a KEEP-LIST: a FITS file is processed
        only if its basename appears in the CSV. This is how the DESI
        SNR-uniform subset (desi_type1_type2_snr_uniform.csv) trims the ~204k
        downloaded DESI spectra down to the redshift-balanced pool without
        moving or deleting a single file.
    """
    type1_metadata = load_candidate_metadata(type1_metadata_csv)
    type2_metadata = load_candidate_metadata(type2_metadata_csv)

    files_type1 = [
        (f, 1, type1_metadata.get(os.path.basename(f), {}))
        for f in glob.glob(os.path.join(type1_path, "*.fits"))
        if (not restrict_to_metadata) or os.path.basename(f) in type1_metadata
    ]
    files_type2 = [
        (f, 2, type2_metadata.get(os.path.basename(f), {}))
        for f in glob.glob(os.path.join(type2_path, "*.fits"))
        if (not restrict_to_metadata) or os.path.basename(f) in type2_metadata
    ]

    if restrict_to_metadata:
        print(f"restrict_to_metadata=True: keep-list filter kept "
              f"{len(files_type1)} type1 + {len(files_type2)} type2 "
              f"FITS files out of all on disk")

    all_tasks = files_type1 + files_type2
    print(f"Processing {len(all_tasks)} spectra...")
    
    results = Parallel(n_jobs=-2)(
        delayed(process_single_spectrum)(
            f_path,
            a_type,
            master_grid,
            file_meta,
            apply_mad_scaling,
            continuum_window_size,
            subtract_continuum,
        )
        for f_path, a_type, file_meta in all_tasks
    )
    
    # Filter out None results from failed loads
    results = [r for r in results if r is not None]
            
    # Assemble Metadata
    all_result_keys = set()
    for r in results:
        all_result_keys.update(r.keys())

    meta_keys = [k for k in all_result_keys if k != 'flux_array']

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
    max_zeros_pct=0.8,
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

    # Dynamically identify flux columns vs metadata columns
    # Flux columns are the wavelength grids (floats converted to strings)
    flux_cols = [c for c in df.columns if str(c).replace('.', '', 1).isdigit()]
    meta_cols = [c for c in df.columns if c not in flux_cols]

    flux_mat = df[flux_cols].values.astype(float)

    # 1. Filter bad coverage.
    # On the wide 3000-10400 A grid a zero pixel means "out of the
    # spectrograph's coverage" -- a high-redshift spectrum legitimately fills
    # only part of the grid. max_zeros_pct=0.8 keeps anything covering >=20% of
    # the grid (~1480 A), which still spans the Hbeta/[OIII] CL diagnostics;
    # it must NOT be set near the old 0.5, which would discard every z>~0.4
    # spectrum and defeat the point of the wide grid.
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
        print("Warning: require_valid_oiii=True is ignored because OIII checking is removed.")
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
    type1_path="data/O3_normalized_network/Type1/",
    type2_path="data/O3_normalized_network/Type2/",
    apply_mad_scaling=False,
    require_valid_oiii=False,
    continuum_window_size=173,
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
    # Wide rest-frame grid (see preprocessing_oiii.MASTER_GRID): 3000-10400 A,
    # 4096 px. Spectra cover only a redshift-dependent sub-range of it; the rest
    # is zero-filled and flagged. This is what lets every redshift be used.
    master_grid = np.linspace(3000, 10400, 4096)
    
    if mode == 'full':
        print("=== Processing original type1/type2 directories ===")
        df = build_agn_catalog(
            type1_path=type1_path,
            type2_path=type2_path,
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
            continuum_window_size=continuum_window_size,
        )
    elif mode == 'new_only':
        print("=== Processing NEW type1_new/type2_new directories ===")
        df = build_agn_catalog(
            type1_path=type1_path,
            type2_path=type2_path,
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
            continuum_window_size=continuum_window_size,
        )
    elif mode == 'merge':
        print("=== Processing NEW data and merging with existing parquet ===")
        df_new = build_agn_catalog(
            type1_path=type1_path,
            type2_path=type2_path,
            master_grid=master_grid,
            type1_metadata_csv=type1_metadata_csv,
            type2_metadata_csv=type2_metadata_csv,
            apply_mad_scaling=apply_mad_scaling,
            continuum_window_size=continuum_window_size,
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
        max_zeros_pct=0.8,
        min_snr=5.0,
        max_flux_outlier=None,
        max_neg_flux=None,
        require_valid_oiii=require_valid_oiii,
    )

    df_clean.to_parquet(output)
    print(f'\nSaved cleaned df with {len(df_clean)} spectra to {output}')
    return df_clean


def build_unified_ssl_parquet(
    dr7_type1_path="data/full_data/type1/",
    dr7_type2_path="data/full_data/type2/",
    dr7_type1_metadata_csv="data/full_data/type1_candidates.csv",
    dr7_type2_metadata_csv="data/full_data/type2_candidates.csv",
    desi_type1_path="data/full_data/desi_spectra/type1/",
    desi_type2_path="data/full_data/desi_spectra/type2/",
    desi_metadata_csv="data/full_data/desi_type1_type2_snr_uniform.csv",
    output="data/ssl_unified_dr7_desi.parquet",
    apply_mad_scaling=True,
    subtract_continuum=False,
    continuum_window_size=173,
    max_zeros_pct=0.8,
    min_snr=8.0,
):
    """
    Build ONE unified self-supervised parquet pooling SDSS-DR7 + DESI spectra.

    Stage-1 SSL pretraining (pretrain_ssl.py) learns from unlabelled spectra
    pooled across surveys. This builds that pool into a single parquet, so
    SSLSpectraDataset reads one file instead of one parquet per survey.

    Both surveys are processed onto the identical wide master grid
    (3000-10400 A, 4096 px), so their 4096 flux columns align exactly; a
    `survey` column ('sdss_dr7' / 'desi') tags every row. Each survey is built
    and cleaned separately -- so the coverage / SNR drop report is per-survey
    -- and the two cleaned frames are then concatenated.

    subtract_continuum=False (v2 default): the full continuum is kept; rows are
    MAD-scaled full spectra (over covered pixels only), i.e. channel 0 directly
    for SSLSpectraDataset. apply_mad_scaling=True applies that MAD scaling.
    min_snr is the spectrum-quality cut; raised from the old 5.0 so the SSL
    pool is cleaner -- tune it from the per-survey cleaning report.
    """
    master_grid = np.linspace(3000, 10400, 4096)

    def _one_survey(name, t1_path, t2_path, t1_csv, t2_csv, restrict=False):
        print(f"\n{'=' * 60}\n=== {name}: building catalog ===\n{'=' * 60}")
        df = build_agn_catalog(
            type1_path=t1_path,
            type2_path=t2_path,
            master_grid=master_grid,
            type1_metadata_csv=t1_csv,
            type2_metadata_csv=t2_csv,
            apply_mad_scaling=apply_mad_scaling,
            continuum_window_size=continuum_window_size,
            restrict_to_metadata=restrict,
            subtract_continuum=subtract_continuum,
        )
        print(f"\n{name}: {len(df)} spectra processed; cleaning...")
        df = clean_dataset(
            df,
            max_zeros_pct=max_zeros_pct,
            min_snr=min_snr,
            max_flux_outlier=None,
            max_neg_flux=None,
            require_valid_oiii=False,
        )
        df["survey"] = name
        return df

    # DR7: every FITS file in the dirs is wanted -> no keep-list filter.
    # DESI: the dirs hold ~204k spectra but only the SNR-uniform subset
    # (desi_type1_type2_snr_uniform.csv) should enter the pool -> restrict.
    df_dr7 = _one_survey("sdss_dr7", dr7_type1_path, dr7_type2_path,
                         dr7_type1_metadata_csv, dr7_type2_metadata_csv)
    df_desi = _one_survey("desi", desi_type1_path, desi_type2_path,
                          desi_metadata_csv, desi_metadata_csv, restrict=True)

    # ---- reconcile to ONE identical schema across surveys -------------
    # Both frames share the 4096 flux columns (identical master grid), the
    # core columns filename/obj_id/agn_type/z/snr/survey, and RA/DEC. DESI
    # RA/DEC come from desi_type1_type2_snr_uniform.csv, which renames the
    # catalog's TARGET_RA/TARGET_DEC to RA/DEC so they line up with the SDSS
    # columns. The only columns that still differ are survey-private
    # identifiers -- TARGETID (DESI) and SDSS_NAME (DR7) -- and those are
    # dropped, so the pooled parquet has a single consistent column set with
    # exactly ONE redshift column `z` valid for both surveys. Per-object
    # identity is still preserved by `filename` (desi-spec-<TARGETID>.fits /
    # spec-<plate>-<mjd>-<fiber>.fits).
    common = [c for c in df_dr7.columns if c in set(df_desi.columns)]
    drop_dr7 = [c for c in df_dr7.columns if c not in common]
    drop_desi = [c for c in df_desi.columns if c not in common]
    if drop_dr7 or drop_desi:
        print("\nUnifying schema -- dropping survey-specific columns:")
        print(f"  DR7  only : {drop_dr7}")
        print(f"  DESI only : {drop_desi}")
    df_dr7 = df_dr7[common]
    df_desi = df_desi[common]

    # Pool the surveys. Columns are now identical in name and order, so the
    # concat is a clean vertical stack with no NaN-padded columns.
    df = pd.concat([df_dr7, df_desi], ignore_index=True)
    df.columns = df.columns.astype(str)

    # Schema sanity check: no column should be entirely NaN (that would mean a
    # column slipped through that only one survey populates).
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        print(f"WARNING: {len(all_nan)} all-NaN column(s) after pooling: "
              f"{all_nan[:8]}")

    print(f"\n{'=' * 60}")
    print(f"Unified SSL pool: {len(df)} spectra")
    print(f"  by survey : {df['survey'].value_counts().to_dict()}")
    print(f"{'=' * 60}")

    df.to_parquet(output)
    print(f"Saved unified SSL parquet -> {output}")
    return df


# --- EXECUTION ---
if __name__ == "__main__":
    # Build the unified Stage-1 (SSL) pretraining pool: SDSS-DR7 + DESI in ONE
    # parquet, on the wide 3000-10400 A / 4096-px grid. Point config_v2.yml
    # paths.ssl_parquets at the output below.
    build_unified_ssl_parquet(
        dr7_type1_path="data/full_data/type1/",
        dr7_type2_path="data/full_data/type2/",
        dr7_type1_metadata_csv="data/full_data/type1_candidates.csv",
        dr7_type2_metadata_csv="data/full_data/type2_candidates.csv",
        desi_type1_path="data/full_data/desi_spectra/type1/",
        desi_type2_path="data/full_data/desi_spectra/type2/",
        desi_metadata_csv="data/full_data/desi_type1_type2_snr_uniform.csv",
        output="data/ssl_unified_dr7_desi.parquet",
        apply_mad_scaling=True,
        subtract_continuum=False,   # v2: keep the full continuum
        min_snr=8.0,                # spectrum-quality cut (was 5.0); tune from
                                    # the per-survey cleaning report
    )
