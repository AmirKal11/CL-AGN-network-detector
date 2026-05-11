import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import glob



def get_test_subsample(full_df, cl_agn_candidates):

    # 1. Clean and filter the full DataFrame (assumed to be variable 'df')
    full_df['class'] = full_df['class'].str.strip()
    full_df['target'] = full_df['target'].str.strip()

    # Get the set of candidate IDs to ensure their inclusion and for labeling
    candidate_ids = set(cl_agn_candidates['sdssid'].unique())

    # Class, target, redshift, and candidate filters
    is_qso = full_df['class'] == 'QSO'
    is_bhm = full_df['target'].str.contains('bhm', case=False, na=False)
    is_nan = full_df['target'].isna()
    is_candidate = full_df['sdssid'].isin(candidate_ids)
    is_low_z = full_df['z'] < 0.4

    # Combined filter: Keep candidates OR standard QSO/BHM targets, and apply the z < 0.4 cut
    df_filtered = full_df[(is_candidate | (is_qso & (is_bhm | is_nan))) & is_low_z].copy()

    # 2. Get common objects present in both surveys
    ids_DR16 = set(df_filtered[df_filtered['survey'] == 'dr16']['sdssid'])
    ids_SDSSV = set(df_filtered[df_filtered['survey'] == 'sdssv']['sdssid'])
    common_ids = ids_DR16.intersection(ids_SDSSV)

    df_common = df_filtered[df_filtered['sdssid'].isin(common_ids)].copy()

    # 3. Pull min and max MJD per object to get exactly one DR16 and one SDSS-V row per object
    min_indices = df_common.groupby('sdssid')['mjd'].idxmin()
    max_indices = df_common.groupby('sdssid')['mjd'].idxmax()

    keep_indices = pd.concat([min_indices, max_indices]).unique()
    df_objects_clean = df_common.loc[keep_indices].copy()

    # 4. Separate by survey to merge them side-by-side
    df_dr16 = df_objects_clean[df_objects_clean['survey'] == 'dr16']
    df_sdssv = df_objects_clean[df_objects_clean['survey'] == 'sdssv']

    # 5. Merge into a single row per object/pair
    df_pairs = pd.merge(
        df_dr16[['sdssid', 'z', 'ra', 'dec', 'mjd', 'specname']],
        df_sdssv[['sdssid', 'mjd', 'specname']],
        on='sdssid',
        suffixes=('_dr16', '_sdssv')
    )

    # 6. Create the 'label' column: 1 if part of cl_agn_candidates, 0 otherwise
    df_pairs['label'] = df_pairs['sdssid'].isin(candidate_ids).astype(int)

    # Optional: Calculate the time delta between observations
    df_pairs['mjd_diff'] = df_pairs['mjd_sdssv'] - df_pairs['mjd_dr16']
    df_pairs['dt_years'] = df_pairs['mjd_diff'] / 365.25

    # Print diagnostics
    print("=== Pairs DataFrame (df_pairs) created ===")
    print(f"Total unique objects / pairs: {len(df_pairs)}")
    print(f"Columns in df_pairs:          {list(df_pairs.columns)}")
    print(f"\nLabel distribution in df_pairs:\n{df_pairs['label'].value_counts()}")

    # Display the final DataFrame
    display(df_pairs.head())
    return df_pairs

def check_missing_candidate_spectra(df_pairs, directory='../data/dr16_sdssv_crossmatch/'):
    """
    Checks if both DR16 and SDSS-V spectra exist for all CL-AGN candidates.
    Returns a DataFrame containing the sdssid of candidates with missing spectra,
    and which spectrum is missing ('dr16', 'sdssv', or 'both').
    """
    # Filter only the CL-AGN candidates
    candidates = df_pairs[df_pairs['label'] == 1]
    
    missing_list = []
    
    for _, row in candidates.iterrows():
        sdssid = row['sdssid']
        spec_dr16 = row['specname_dr16']
        spec_sdssv = row['specname_sdssv']
        
        dr16_path = os.path.join(directory, spec_dr16)
        sdssv_path = os.path.join(directory, spec_sdssv)
        
        dr16_exists = os.path.exists(dr16_path)
        sdssv_exists = os.path.exists(sdssv_path)
        
        if not dr16_exists and not sdssv_exists:
            missing_list.append({'sdssid': sdssid, 'missing': 'both', 'spec_dr16': spec_dr16, 'spec_sdssv': spec_sdssv})
        elif not dr16_exists:
            missing_list.append({'sdssid': sdssid, 'missing': 'dr16', 'spec_dr16': spec_dr16, 'spec_sdssv': spec_sdssv})
        elif not sdssv_exists:
            missing_list.append({'sdssid': sdssid, 'missing': 'sdssv', 'spec_dr16': spec_dr16, 'spec_sdssv': spec_sdssv})
            
    missing_df = pd.DataFrame(missing_list)
    
    if not missing_df.empty:
        print(f"Found {len(missing_df)} candidates with missing spectra.")
        try:
            display(missing_df)
        except NameError:
            print(missing_df)
    else:
        print("Excellent! All candidate spectra (both DR16 and SDSS-V) are present in the directory.")
        
    return missing_df

def download_missing_candidate_spectra(df_missing, directory='../data/dr16_sdssv_crossmatch/'):
    """
    Takes the output of check_missing_candidate_spectra and downloads the missing
    spectra using the existing download_dr16 and download_SDSS_file functions.
    Returns a DataFrame containing the candidates and spectra that failed to download.
    """
    failed_downloads = []
    
    dr16_rows = df_missing[df_missing['missing'].isin(['dr16', 'both'])]
    sdssv_rows = df_missing[df_missing['missing'].isin(['sdssv', 'both'])]
    
    # Download missing DR16 spectra using the existing parallel function
    if not dr16_rows.empty:
        dr16_to_download = dr16_rows['spec_dr16'].tolist()
        print(f"Downloading {len(dr16_to_download)} missing DR16 spectra...")
        # download_dr16 handles checking if files exist and parallel execution
        download_dr16(dr16_to_download, directory)
            
    # Download missing SDSS-V spectra
    if not sdssv_rows.empty:
        print(f"Downloading {len(sdssv_rows)} missing SDSS-V spectra...")
        for _, row in tqdm(sdssv_rows.iterrows(), total=len(sdssv_rows), desc="SDSS-V Downloads"):
            spec = row['spec_sdssv']
            try:
                download_SDSS_file(spec, directory)
            except Exception as e:
                failed_downloads.append({
                    'sdssid': row['sdssid'], 
                    'spectrum': spec, 
                    'survey': 'sdssv', 
                    'error': str(e)
                })

    # Verify if DR16 downloads were successful
    if not dr16_rows.empty:
        for _, row in dr16_rows.iterrows():
            spec = row['spec_dr16']
            if not os.path.exists(os.path.join(directory, spec)):
                failed_downloads.append({
                    'sdssid': row['sdssid'], 
                    'spectrum': spec, 
                    'survey': 'dr16', 
                    'error': 'Failed during download_dr16 execution'
                })
                
    failed_df = pd.DataFrame(failed_downloads)
    
    if not failed_df.empty:
        print(f"\nFailed to download {len(failed_df)} spectra.")
        try:
            display(failed_df)
        except NameError:
            print(failed_df)
    else:
        print("\nSuccess! All missing spectra were downloaded properly.")
        
    return failed_df

def extract_mjd_and_field(spec_name: str) -> tuple:
    """
    Extracts the field and MJD from an SDSS spec name.
    Expects format: spec-{field}-{mjd}-{fiberid/targetid}
    """
    # Remove file extension if it exists (e.g., .fits)
    clean_name = spec_name.split('.')[0]
    
    # Split the string by hyphens
    parts = clean_name.split('-')
    
    # Extract and convert to integers
    # parts[1] is the field, parts[2] is the MJD
    field = int(parts[1])
    mjd = int(parts[2])
    
    return field, mjd



def download_dr16(names, save_path):

    download = True
    max_workers = 10
    folder_path = save_path
    os.makedirs(folder_path, exist_ok=True)

    # 1. Identify what we already have
    # We use a set for O(1) lookup speed
    existing_files = {os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.fits'))}

    # 2. Identify what needs to be downloaded
    # specnames_to_download from your dataframe
    all_specs = names
    to_download = [s for s in all_specs if s not in existing_files]

    print(f"Total requested: {len(all_specs)}")
    print(f"Already exists:  {len(existing_files)}")
    print(f"Will download:   {len(to_download)}")

    def download_one(session, folder_path, specname, url_candidates):
        path = os.path.join(folder_path, specname)
        last_err = None
        
        for url in url_candidates:
            try:
                with session.get(url, stream=True, timeout=30) as r:
                    if r.status_code == 200:
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=32768): # 32KB
                                if chunk:
                                    f.write(chunk)
                        return specname, True, None
                    else:
                        last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
                
        return specname, False, last_err

    if download and len(to_download) > 0:
        # Build task list
        tasks = []
        for specname in to_download:
            # Parsing filename: fiber_plate_mjd.fits
            parts = specname.replace('.fits', '').split('_')
            fiber = parts[0]
            plate = parts[1]
            mjd = parts[2]

            if int(plate) > 3523:
                urls = [f'http://dr16.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber.zfill(4)}.fits']
            else:
                urls = [
                    f'http://dr16.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber.zfill(4)}.fits',
                    f'http://dr16.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber.zfill(4)}.fits',
                    f'http://dr16.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber.zfill(4)}.fits',
                ]
            tasks.append((specname, urls))

        # Parallel Execution
        with requests.Session() as session:
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=retry)
            session.mount("http://", adapter)
            
            results_success = 0
            results_failed = []

            # progress bar context
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(download_one, session, folder_path, spec, urls): spec for spec, urls in tasks}
                
                # wrap as_completed in tqdm for a status bar
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Spectra"):
                    spec, success, err = fut.result()
                    if success:
                        results_success += 1
                    else:
                        results_failed.append(f"{spec}: {err}")

        print(f"\nSummary: {results_success} downloaded, {len(results_failed)} failed.")
        if results_failed:
            print("First few errors:")
            for error in results_failed[:5]:
                print(f"  - {error}")
    elif len(to_download) == 0:
        print("All files already exist. Nothing to do.")



def download_SDSS_file(fits_name, save_directory, ver='v6_2_1'):
   
    field, mjd = extract_mjd_and_field(fits_name)
    
    path_to_save = os.path.join(save_directory, fits_name)
    
    field_formatted = str(field).zfill(6)
    url = f'https://data.sdss5.org/sas/sdsswork/bhm/boss/spectro/redux/{ver}/spectra/epoch/lite/{field_formatted[:3]}XXX/{field_formatted}/{mjd}/{fits_name}'
    r = requests.get(url, auth=('sdss5', 'panoPtic-5'))
    if r.status_code == 200:
        open(path_to_save, 'wb').write(r.content)
        print(f"Successfully downloaded: {fits_name}")
    else:
        raise Exception(f"Failed to download {fits_name}. HTTP Status: {r.status_code}. URL: {url}")



