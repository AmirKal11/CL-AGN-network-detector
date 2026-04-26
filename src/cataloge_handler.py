import pandas as pd
import numpy as np
import astropy 
from astropy.table import Table, hstack
from astropy.io import fits
import matplotlib.pyplot as plt
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


BASE_DIR = 'data'
TYPE1_DIR = os.path.join(BASE_DIR, 'type1_new')
TYPE2_DIR = os.path.join(BASE_DIR, 'type2_new')

MAX_WORKERS = 10


# Ensure both specific subdirectories exist
for folder in [TYPE1_DIR, TYPE2_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_targets_by_type():
    """Reads the CSVs and returns two separate lists of (plate, mjd, fiber)."""
    type1_targets = []
    type2_targets = []
    
    # Process Type 1
    try:
        df1 = pd.read_csv('data/type1_candidates_fixed.csv')
        # Dropping duplicates within the file itself
        df1_unique = df1.drop_duplicates(subset=['PLATE', 'mjd', 'fiber'])
        for _, row in df1_unique.iterrows():
            type1_targets.append((int(row['PLATE']), int(row['mjd']), int(row['fiber'])))
    except Exception as e:
        print(f"Error reading Type 1 file: {e}")

    # Process Type 2
    try:
        df2 = pd.read_csv('data/type2_candidates_fixed.csv')
        # Dropping duplicates within the file itself
        df2_unique = df2.drop_duplicates(subset=['PLATE', 'MJD_class_table', 'FIBERID_class_table'])
        for _, row in df2_unique.iterrows():
            type2_targets.append((int(row['PLATE']), int(row['MJD_class_table']), int(row['FIBERID_class_table'])))
    except Exception as e:
        print(f"Error reading Type 2 file: {e}")

    return type1_targets, type2_targets

def download_spectrum(target_info):
    """
    Unpacks target data and directory info to download the file.
    target_info is a tuple: (plate, mjd, fiber, save_path)
    """
    plate, mjd, fiber, save_path = target_info
    
    url = (f"https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/"
           f"{plate:04d}/spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits")
    
    filename = f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"
    filepath = os.path.join(save_path, filename)

    # Skip logic: checks the specific subdirectory
    if os.path.exists(filepath):
        return "skipped"

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return "downloaded"
        else:
            return "failed"
    except:
        return "error"

def run_download_batch(targets, save_path, label):
    """Helper to run a batch of downloads with a progress bar."""
    if not targets:
        return
    
    print(f"\nProcessing {label} targets...")
    # Prepare the list with the destination path included
    task_list = [(p, m, f, save_path) for p, m, f in targets]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_spectrum, task_list), 
                            total=len(task_list), 
                            unit="spec"))
    
    stats = pd.Series(results).value_counts()
    print(f"Summary for {label}: {stats.get('downloaded', 0)} new, {stats.get('skipped', 0)} skipped.")

def main():
    t1_list, t2_list = get_targets_by_type()
    
    # Download Type 1 into data/type1
    run_download_batch(t1_list, TYPE1_DIR, "Type 1")
    
    # Download Type 2 into data/type2
    run_download_batch(t2_list, TYPE2_DIR, "Type 2")

def merge_gal_catalog(class_path, info_path, SNR_threshold=5):
    class_table = Table.read(class_path)
    info_table = Table.read(info_path)
    
    combined_table = hstack([class_table, info_table], table_names=['class_table', 'info_table'])

    names_merged = [name for name in combined_table.colnames if len(combined_table[name].shape) <= 1]    
    merged_df = combined_table[names_merged].to_pandas()

    # Clean up the string columns 
    # Astropy FITS tables load strings as byte arrays (e.g., b'GALAXY   '). 
    # We must decode them so pandas can read them normally.
    string_columns = ['SPECTROTYPE', 'SUBCLASS', 'TARGETTYPE']
    for col in string_columns:
        if col in merged_df.columns:
            # Check if the first item is bytes
            if isinstance(merged_df[col].dropna().iloc[0], bytes):
                merged_df[col] = merged_df[col].str.decode('utf-8').str.strip()
            else:
                merged_df[col] = merged_df[col].str.strip()

    # --- NEW: 2. Apply Strict SDSS & MPA-JHU Filtering ---
    # Fill NaN values in SUBCLASS with an empty string so string matching doesn't crash
    merged_df['SUBCLASS'] = merged_df['SUBCLASS'].fillna('')

    mask = (
        (merged_df['I_CLASS'] == 4) &                          # MPA-JHU confirms it is a BPT AGN
        (merged_df['Z_WARNING'] == 0) &                        # SDSS Pipeline had ZERO extraction/fit errors
        (merged_df['SN_MEDIAN'] >= SNR_threshold) &            # General spectrum quality check
        (merged_df['SPECTROTYPE'] == 'GALAXY') &               # Ensure it wasn't misclassified as a STAR
        (~merged_df['SUBCLASS'].str.contains('BROADLINE'))     # Explicitly reject Type 1 leaks in the Type 2 catalog
    )

    df_filtered = merged_df[mask].copy()

    print(f"Filtered Type 2 Catalog: Retained {len(df_filtered)} highly confident candidates.")
    
    return df_filtered
    


def filter_SNR(df, hb_col, ha_col, SNR_threshold):
    cols = [hb_col, ha_col]
    
    # Condition: If a line is present (SNR > 0), it MUST be above the threshold.
    # We allow SNR <= 0 (meaning the line is not in the spectrum/not detected)
    mask = True
    for col in cols:
        mask &= ((df[col] <= 0) | (df[col] >= SNR_threshold))
    
    # Additionally, ensure at least one of these lines is actually detected above threshold
    any_line_detected = False
    for col in cols:
        any_line_detected |= (df[col] >= SNR_threshold)
        
    df_filtered = df[mask & any_line_detected].copy()
    return df_filtered

def extract_shen_type1_catalog(catalog_path, z_threshold=0.4, SNR_threshold=5.0):
    """
    Specifically tailored to filter the Shen 2011 DR7 Quasar Catalog (dr7_bh_Nov19_2013.fits).
    """
    print(f"Loading Shen 2011 Quasar Catalog from {catalog_path}...")
    catalog = Table.read(catalog_path)
    
    # Filter multidimensional arrays
    names = [name for name in catalog.colnames if len(catalog[name].shape) <= 1]
    df = catalog[names].to_pandas()

    # 1. Decode Byte Strings (SDSS_NAME)
    if 'SDSS_NAME' in df.columns and isinstance(df['SDSS_NAME'].dropna().iloc[0], bytes):
        df['SDSS_NAME'] = df['SDSS_NAME'].str.decode('utf-8').str.strip()

    # 2. The Physics & Quality Filters
    # SN_RATIO is the Shen catalog's median SNR per pixel
    mask = ((df['BAL_FLAG'] == 0) & (df['REDSHIFT'] < z_threshold))
    
    df_filtered = df[mask].copy()


    df_filtered = filter_SNR(
        df_filtered, 
        'LINE_MED_SN_HB', 
        'LINE_MED_SN_HA', 
        SNR_threshold
    )

    # 3. Construct the legacy SDSS filename (spSpec-MMMMM-PPPP-FFF.fit)
    # Shen uses 'FIBERID', not 'FIBER'
    df_filtered['filename'] = (
        "spSpec-" + 
        df_filtered['MJD'].astype(int).astype(str).str.zfill(5) + "-" + 
        df_filtered['PLATE'].astype(int).astype(str).str.zfill(4) + "-" + 
        df_filtered['FIBER'].astype(int).astype(str).str.zfill(3) + ".fit"
    )

    # 4. Standardize column names to match your pipeline
    cols_to_keep = ['filename', 'SDSS_NAME', 'RA', 'DEC', 'REDSHIFT', 'MJD', 'PLATE', 'FIBER', 'LINE_MED_SN_HB']
    df_final = df_filtered[cols_to_keep].rename(columns={
        'REDSHIFT': 'z', 
        'MJD': 'mjd', 
        'FIBER': 'fiber',
        'LINE_MED_SN_HB': 'snr'
    })

    print(f"Filtered Type 1 Catalog: Retained {len(df_final)} clean, low-z broad-line candidates.")
    return df_final


def find_candidates(catalog_input,**kwargs):
    z_threshold = kwargs.get('z_threshold',None)
    object_class = kwargs.get('object_class',None)
    SNR_threshold = kwargs.get('SNR_threshold',None)
    
    if isinstance(catalog_input, Table):
        catalog = catalog_input
    else:
        catalog = Table.read(catalog_input)

    # Filter out multidimensional columns that pandas cannot handle
    names = [name for name in catalog.colnames if len(catalog[name].shape) <= 1]
    df = catalog[names].to_pandas()
    
    # Start with all objects where BAL_FLAG is 0
    df_filtered = df[(df['BAL_FLAG'] == 0)].copy()
    
    # Identify the correct redshift column
    z_col = 'REDSHIFT' if 'REDSHIFT' in df_filtered.columns else 'z'
    
    if z_threshold is not None:
        df_filtered = df_filtered[(df_filtered[z_col] < z_threshold)].copy()
    
    if SNR_threshold is not None:
        # Corrected C4 to CIV based on catalog columns
        df_filtered = filter_SNR(df_filtered,'LINE_MED_SN_CIV','LINE_MED_SN_MGII','LINE_MED_SN_HB','LINE_MED_SN_HA',SNR_threshold)

    # Use MJD if SMJD is missing
    mjd_col = 'SMJD' if 'SMJD' in df_filtered.columns else 'MJD'
    
    df_filtered['filename'] = (
    "spSpec-" + 
    df_filtered[mjd_col].astype(int).astype(str).str.zfill(5) + "-" + 
    df_filtered['PLATE'].astype(int).astype(str).str.zfill(4) + "-" + 
    df_filtered['FIBER'].astype(int).astype(str).str.zfill(3) + ".fit"
    )
    

    if not df_filtered.empty and isinstance(df_filtered['SDSS_NAME'].iloc[0], bytes):
        df_filtered['SDSS_NAME'] = df_filtered['SDSS_NAME'].str.decode('utf-8')

    # Available columns list for dynamic selection
    available_cols = df_filtered.columns.tolist()
    
    # Base columns to keep (SDSS_NAME usually acts as the primary object identifier)
    cols_to_keep = ['filename', 'SDSS_NAME', 'RA', 'DEC', z_col, mjd_col, 'PLATE', 'FIBER']
    
    # Optionally add other specific object IDs or fields if they exist
    for opt_col in ['FIELD', 'OBJID', 'SPECOBJID']:
        if opt_col in available_cols:
            cols_to_keep.append(opt_col)
        
    df_final = df_filtered[cols_to_keep].rename(columns={mjd_col: 'mjd'})
    return df_final


if __name__ == "__main__":

    main()
    
    
    #type1_candidates = extract_shen_type1_catalog('/Users/amir/Documents/Deep learning/cl-agn classifier/cataloges/dr7_bh_Nov19_2013.fits.gz', SNR_threshold=5)
    #type1_candidates.to_csv('/Users/amir/Documents/Deep learning/cl-agn classifier/data/type1_candidates_fixed.csv', index=False)    
    
    #type1= pd.read_csv('/Users/amir/Documents/Deep learning/cl-agn classifier/data/type1_candidates_fixed.csv')
    #type2= pd.read_csv('/Users/amir/Documents/Deep learning/cl-agn classifier/data/type2_candidates_fixed.csv')

