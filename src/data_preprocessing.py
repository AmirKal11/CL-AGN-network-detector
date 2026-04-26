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


def morphological_continuum_subtraction(x, window_size=151, clip_max=4.0, taper_len=5):
    """
    Acts as a lightweight spectral decomposition by estimating and subtracting 
    the continuum envelope using morphological pooling.
    x: [Batch, 1, Sequence_Length]
    """
    # 1. Pad the sequence to handle edge artifacts smoothly
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    
    # 2. Extract the continuum envelope using a wide average pool
    # This acts as a low-pass filter, ignoring sharp narrow lines and following the slope
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
    
    # 3. Subtract the continuum from the original flux
    x_flattened = x - continuum
    
    # 4. Standardize the result to ensure numeric stability for the CNN
    mean = x_flattened.mean(dim=-1, keepdim=True)
    std = x_flattened.std(dim=-1, keepdim=True)
    x_normalized = (x_flattened - mean) / (std + 1e-8)
    
    # 5. DECAPITATE NARROW LINES (The Grad-CAM Fix)
    # This prevents the network from using the [O III] / Narrow Balmer ratio to cheat
    #x_clipped = torch.clamp(x_normalized, min=-10.0, max=clip_max)
    
    # 6. EDGE TAPERING (The Limits Fix)
    # Surgically fade the first and last 5 pixels to destroy reflection artifacts
    seq_len = x.shape[-1]
    taper = torch.ones(seq_len, device=x.device)
    
    # Create a linear fade from 0 to 1
    fade = torch.linspace(0, 1, taper_len, device=x.device)
    
    # Apply to left and right edges
    taper[:taper_len] = fade
    taper[-taper_len:] = torch.flip(fade, dims=[0])
    
    # Reshape so it broadcasts perfectly over (Batch, Channels, SeqLen)
    taper = taper.view(1, 1, -1)
    
    # Apply the taper to the clipped tensor
    x_final = x_normalized * taper

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


def process_single_spectrum(file_path, agn_type, master_grid):
    """
    Opens a FITS file, extracts metadata, corrects to rest-frame, 
    standardizes, and resamples to the master grid.
    """
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
                tensor_flux, window_size=151, taper_len=5, clip_max=4.0
            )
            
            # 8. Squeeze it back down to a flat 1D NumPy array for your Parquet file
            processed_flux = processed_tensor.squeeze().numpy()

            return {
                "filename": os.path.basename(file_path), 
                "obj_id": obj_id,
                "agn_type": agn_type, 
                "z": z, 
                "flux_array": processed_flux,
                "snr": snr
            }
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def build_agn_catalog(type1_path, type2_path, master_grid):
    """Gathers files from both directories and processes them into one DataFrame."""
    files_type1 = [(f, 1) for f in glob.glob(os.path.join(type1_path, "*.fits"))]
    files_type2 = [(f, 2) for f in glob.glob(os.path.join(type2_path, "*.fits"))]
    
    all_tasks = files_type1 + files_type2
    print(f"Processing {len(all_tasks)} spectra...")
    
    # Use Parallel for speed (n_jobs=-1 uses all cores)
    results = Parallel(n_jobs=-2)(
        delayed(process_single_spectrum)(f_path, a_type, master_grid) 
        for f_path, a_type in all_tasks
    )
    
    # Filter out None results from failed loads
    results = [r for r in results if r is not None]
            
    # Assemble Metadata
    meta_df = pd.DataFrame([
        {'filename': r['filename'], 'obj_id': r['obj_id'], 'agn_type': r['agn_type'], 'z': r['z'], 'snr': r['snr']} 
        for r in results
    ])
    
    # Assemble Fluxes (Matrix construction is faster than row-by-row append)
    flux_matrix = np.array([r['flux_array'] for r in results])
    flux_df = pd.DataFrame(flux_matrix, columns=master_grid)
    # Merge horizontally
    final_df = pd.concat([meta_df, flux_df], axis=1)
    final_df.columns = final_df.columns.astype(str)
    return final_df


def clean_dataset(df, max_zeros_pct=0.5, min_snr=5.0, max_flux_outlier=30.0, max_neg_flux=5.0):
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

def run_preprocessing(mode='full', existing_parquet='data/processed_agn_catalog_cut.parquet',
                      output='data/processed_agn_catalog_cut.parquet'):
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
            type1_path="data/type1/",
            type2_path="data/type2/",
            master_grid=master_grid
        )
    elif mode == 'new_only':
        print("=== Processing NEW type1_new/type2_new directories ===")
        df = build_agn_catalog(
            type1_path="data/type1_new/",
            type2_path="data/type2_new/",
            master_grid=master_grid
        )
    elif mode == 'merge':
        print("=== Processing NEW data and merging with existing parquet ===")
        df_new = build_agn_catalog(
            type1_path="data/type1_new/",
            type2_path="data/type2_new/",
            master_grid=master_grid
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
    df_clean = clean_dataset(df, max_zeros_pct=0.5, min_snr=5.0, 
                             max_flux_outlier=30.0, max_neg_flux=5.0)

    df_clean.to_parquet(output)
    print(f'\nSaved cleaned df with {len(df_clean)} spectra to {output}')
    return df_clean


# --- EXECUTION ---
if __name__ == "__main__":
    run_preprocessing(mode='new_only', output='data/processed_agn_new.parquet')
