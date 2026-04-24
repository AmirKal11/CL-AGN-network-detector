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
            
            # 5. Standardize the interpolated flux
            processed_flux = standardize_flux(interpolated_flux)
        
            # 6. Fill out-of-bounds areas with 0.0 (which is now exactly the mean)
            processed_flux = np.nan_to_num(processed_flux, nan=0.0).astype(np.float32)
            
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


def clean_dataset(df, max_zeros_pct=0.5, min_snr=5.0, max_flux_outlier=30.0):
    """
    Cleans the dataset by removing spectra with low coverage, poor SNR, or extreme outliers.
    """
    meta_cols = ['filename', 'obj_id', 'agn_type', 'z', 'snr']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    flux_mat = df[flux_cols].values
    
    # 1. Filter bad coverage
    zeros_pct = (flux_mat == 0.0).mean(axis=1)
    valid_coverage = zeros_pct <= max_zeros_pct
    
    # 2. Filter extreme outliers
    max_flux = flux_mat.max(axis=1)
    valid_outlier = max_flux <= max_flux_outlier
    
    # 3. Filter low SNR
    valid_snr = df['snr'] >= min_snr
    
    # Combine masks
    good_mask = valid_coverage & valid_outlier & valid_snr
    df_clean = df[good_mask].copy()
    
    print(f"Original spectra: {len(df)}")
    print(f"Dropped due to coverage: {(~valid_coverage).sum()}")
    print(f"Dropped due to outliers: {(~valid_outlier).sum()}")
    print(f"Dropped due to low SNR:  {(~valid_snr).sum()}")
    print(f"Remaining clean spectra: {len(df_clean)}")
    
    return df_clean

# --- EXECUTION ---
if __name__ == "__main__":
    master_grid = np.linspace(4575,6699,1024) 

    df = build_agn_catalog(
        type1_path = "data/type1/", 
        type2_path = "data/type2/",
        master_grid = master_grid
    )
    
    print("\nCleaning dataset...")
    # HERE is where the dataset is cleaned!
    df_clean = clean_dataset(df, max_zeros_pct=0.5, min_snr=5.0, max_flux_outlier=30.0)

    # Note: explicitly saving to data/ folder so it's easily found by the notebook
    df_clean.to_parquet('data/processed_agn_catalog_cut.parquet') 
    print(f'\nSaved cleaned df with {len(df_clean)} spectra to data/processed_agn_catalog_cut.parquet')
