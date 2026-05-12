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
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


BASE_DIR = 'data/O3_normalized_network/'
TYPE1_DIR = os.path.join(BASE_DIR, 'type1')
TYPE2_DIR = os.path.join(BASE_DIR, 'type2')
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
MAX_WORKERS = 10


# Ensure both specific subdirectories exist
for folder in [TYPE1_DIR, TYPE2_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def add_oiii_flux_from_luminosity(
    df,
    z_col="z",
    lum_col="OIII_LUM",
    lum_err_col="OIII_LUM_ERR",
):
    """
    Convert OIII luminosity to flux using:

        F = L / (4*pi*d_L^2)

    If OIII_LUM_ERR exists, convert it to flux error using the same factor.
    """
    df = df.copy()

    if "OIII_FLUX" not in df.columns:
        df["OIII_FLUX"] = np.nan

    if "OIII_FLUX_ERR" not in df.columns:
        df["OIII_FLUX_ERR"] = np.nan

    valid = (
        df[z_col].notna()
        & df[lum_col].notna()
        & (df[z_col] > 0)
        & (df[lum_col] > 0)
    )

    if valid.any():
        z = df.loc[valid, z_col].astype(float).values
        lum = df.loc[valid, lum_col].astype(float).values

        d_l_cm = COSMO.luminosity_distance(z).to(u.cm).value
        flux = (lum / (4.0 * np.pi * d_l_cm**2)) / 1e-17

        df.loc[valid, "OIII_FLUX"] = flux

    if lum_err_col in df.columns:
        valid_err = (
            df[z_col].notna()
            & df[lum_err_col].notna()
            & (df[z_col] > 0)
            & (df[lum_err_col] > 0)
        )

        if valid_err.any():
            z = df.loc[valid_err, z_col].astype(float).values
            lum_err = df.loc[valid_err, lum_err_col].astype(float).values

            d_l_cm = COSMO.luminosity_distance(z).to(u.cm).value
            flux_err = (lum_err / (4.0 * np.pi * d_l_cm**2)) / 1e-17

            df.loc[valid_err, "OIII_FLUX_ERR"] = flux_err

    valid_snr = (
        df["OIII_FLUX"].notna()
        & df["OIII_FLUX_ERR"].notna()
        & (df["OIII_FLUX_ERR"] > 0)
    )

    if "OIII_SNR" not in df.columns:
        df["OIII_SNR"] = np.nan

    df.loc[valid_snr, "OIII_SNR"] = (
        df.loc[valid_snr, "OIII_FLUX"].astype(float)
        / df.loc[valid_snr, "OIII_FLUX_ERR"].astype(float)
    )

    return df

def decode_string_columns(df):
    """
    Decode FITS byte-string columns into normal Python strings.
    """
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], bytes):
                df[col] = df[col].str.decode("utf-8").str.strip()
    return df


def add_oiii_luminosity_from_flux(
    df,
    z_col="z",
    flux_col="OIII_FLUX",
    flux_err_col="OIII_FLUX_ERR",
    flux_unit=1e-17,
):
    """
    Convert [O III] flux to luminosity using:
        L = 4*pi*d_L^2*F

    Assumes OIII_FLUX is in units of flux_unit erg/s/cm^2.
    For MPA-JHU line catalogues, flux_unit=1e-17 is usually the correct scale.
    """
    df = df.copy()

    df["OIII_LUM"] = np.nan
    df["OIII_LUM_ERR"] = np.nan
    df["LOG_OIII_LUM"] = np.nan
    df["OIII_SNR"] = np.nan

    valid = (
        df[z_col].notna()
        & df[flux_col].notna()
        & (df[z_col] > 0)
        & (df[flux_col] > 0)
    )

    if valid.any():
        z = df.loc[valid, z_col].astype(float).values
        flux = df.loc[valid, flux_col].astype(float).values * flux_unit

        d_l_cm = COSMO.luminosity_distance(z).to(u.cm).value
        lum = 4.0 * np.pi * d_l_cm**2 * flux

        df.loc[valid, "OIII_LUM"] = lum
        df.loc[valid, "LOG_OIII_LUM"] = np.log10(lum)

        if flux_err_col in df.columns:
            valid_err = valid & df[flux_err_col].notna() & (df[flux_err_col] > 0)
            if valid_err.any():
                z_err = df.loc[valid_err, z_col].astype(float).values
                flux_err = df.loc[valid_err, flux_err_col].astype(float).values * flux_unit
                d_l_cm_err = COSMO.luminosity_distance(z_err).to(u.cm).value

                lum_err = 4.0 * np.pi * d_l_cm_err**2 * flux_err
                df.loc[valid_err, "OIII_LUM_ERR"] = lum_err

                df.loc[valid_err, "OIII_SNR"] = (
                    df.loc[valid_err, flux_col].astype(float)
                    / df.loc[valid_err, flux_err_col].astype(float)
                )

    return df

def add_oiii_quality_flags(
    df,
    flux_col="OIII_FLUX",
    flux_err_col="OIII_FLUX_ERR",
    lum_col="OIII_LUM",
    lum_err_col="OIII_LUM_ERR",
    snr_col="OIII_SNR",
    sigma_col="OIII_SIGMA",
    voff_col="OIII_VOFF",
    chi2_col="OIII_CHI2",
    min_snr=5.0,
    min_lum=1e38,
    max_lum=1e45,
    min_sigma=30.0,
    max_sigma=1000.0,
    max_abs_voff=1000.0,
    max_chi2=None,
):
    """
    Add quality flags for OIII flux/luminosity.

    Creates:
        OIII_VALID : bool
        OIII_QUALITY_FLAG : string explaining rejection reason

    Notes
    -----
    min_lum/max_lum are intentionally loose. They mainly catch broken values.
    max_chi2 is optional because the interpretation of catalogue chi2 depends
    on the catalogue fitting convention.
    """
    df = df.copy()

    if "OIII_VALID" not in df.columns:
        df["OIII_VALID"] = True

    df["OIII_QUALITY_FLAG"] = "valid"

    def reject(mask, reason):
        df.loc[mask & df["OIII_VALID"], "OIII_VALID"] = False
        df.loc[mask, "OIII_QUALITY_FLAG"] = reason

    # Flux checks, if flux exists.
    if flux_col in df.columns:
        reject(df[flux_col].isna(), "missing_flux")
        reject(df[flux_col] <= 0, "non_positive_flux")

    if flux_err_col in df.columns:
        reject(df[flux_err_col].isna(), "missing_flux_err")
        reject(df[flux_err_col] <= 0, "non_positive_flux_err")

    # SNR. If OIII_SNR is missing but flux/error exist, compute it.
    if snr_col not in df.columns:
        df[snr_col] = np.nan

    if (
        flux_col in df.columns
        and flux_err_col in df.columns
    ):
        valid_err = (
            df[flux_col].notna()
            & df[flux_err_col].notna()
            & (df[flux_err_col] > 0)
        )
        df.loc[valid_err, snr_col] = (
            df.loc[valid_err, flux_col].astype(float)
            / df.loc[valid_err, flux_err_col].astype(float)
        )

    reject(df[snr_col].isna(), "missing_snr")
    reject(df[snr_col] < min_snr, "low_snr")

    # Luminosity checks.
    if lum_col in df.columns:
        reject(df[lum_col].isna(), "missing_lum")
        reject(df[lum_col] <= 0, "non_positive_lum")
        reject(df[lum_col] < min_lum, "too_low_lum")
        reject(df[lum_col] > max_lum, "too_high_lum")

    if lum_err_col in df.columns:
        # Do not reject if luminosity error is missing for Shen catalogue,
        # because Shen may provide luminosity but not luminosity uncertainty.
        has_lum_err_info = df[lum_err_col].notna()
        reject(has_lum_err_info & (df[lum_err_col] <= 0), "non_positive_lum_err")

    # Fit quality / kinematic sanity checks.
    if sigma_col in df.columns:
        has_sigma = df[sigma_col].notna()
        reject(has_sigma & (df[sigma_col] <= min_sigma), "sigma_too_low")
        reject(has_sigma & (df[sigma_col] > max_sigma), "sigma_too_high")

    if voff_col in df.columns:
        has_voff = df[voff_col].notna()
        reject(has_voff & (np.abs(df[voff_col]) > max_abs_voff), "voff_too_large")

    if max_chi2 is not None and chi2_col in df.columns:
        has_chi2 = df[chi2_col].notna()
        reject(has_chi2 & (df[chi2_col] > max_chi2), "chi2_too_high")

    return df

def standardize_shen_oiii_columns(df):
    """
    Standardize [O III] columns from the Shen DR7 quasar catalogue.

    Shen gives [O III] 5007 luminosity as log10(L / erg/s),
    and its uncertainty as an uncertainty in log luminosity.

    We store:
      LOG_OIII_LUM      = catalogue log luminosity
      LOG_OIII_LUM_ERR  = catalogue log-luminosity uncertainty
      OIII_LUM          = linear luminosity, erg/s
      OIII_LUM_ERR      = linear luminosity error, erg/s
    """
    df = df.copy()

    print("OIII-related Shen columns:")
    print([c for c in df.columns if "OIII" in c.upper()])

    rename_candidates = {
        # [O III] 5007 log luminosity
        "LOGL_OIII_5007": "LOG_OIII_LUM",
        "LOGL_OIII": "LOG_OIII_LUM",
        "LOGL_NAR_OIII_5007": "LOG_OIII_LUM",
        "LOGL_NAR_OIII": "LOG_OIII_LUM",

        # [O III] 5007 log luminosity error
        "LOGL_OIII_5007_ERR": "LOG_OIII_LUM_ERR",
        "LOGL_OIII_ERR": "LOG_OIII_LUM_ERR",
        "ERR_LOGL_OIII_5007": "LOG_OIII_LUM_ERR",
        "ERR_LOGL_OIII": "LOG_OIII_LUM_ERR",
        "LOGL_NAR_OIII_5007_ERR": "LOG_OIII_LUM_ERR",
        "ERR_LOGL_NAR_OIII_5007": "LOG_OIII_LUM_ERR",
        "ERR_LOGL_NAR_OIII": "LOG_OIII_LUM_ERR",

        # Possible linear luminosity names, less likely for Shen
        "OIII_5007_LUM": "OIII_LUM",
        "OIII_LUM": "OIII_LUM",
        "OIII_5007_LUM_ERR": "OIII_LUM_ERR",
        "OIII_LUM_ERR": "OIII_LUM_ERR",
        "ERR_OIII_5007_LUM": "OIII_LUM_ERR",
        "ERR_OIII_LUM": "OIII_LUM_ERR",

        # Other useful measurements
        "OIII_5007_EW": "OIII_EW",
        "EW_OIII_5007": "OIII_EW",
        "OIII_5007_FWHM": "OIII_FWHM",
        "FWHM_OIII_5007": "OIII_FWHM",
    }

    for old, new in rename_candidates.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Always create standardized columns, so downstream code never crashes
    for col in [
        "LOG_OIII_LUM",
        "LOG_OIII_LUM_ERR",
        "OIII_LUM",
        "OIII_LUM_ERR",
        "OIII_FLUX",
        "OIII_FLUX_ERR",
        "OIII_SNR",
        "OIII_SIGMA",
        "OIII_VOFF",
        "OIII_CHI2",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Convert catalogue log luminosity to linear luminosity
    valid_log_lum = (
        df["LOG_OIII_LUM"].notna()
        & np.isfinite(df["LOG_OIII_LUM"].astype(float))
        & (df["LOG_OIII_LUM"].astype(float) > 0)
    )

    df.loc[valid_log_lum, "OIII_LUM"] = (
        10 ** df.loc[valid_log_lum, "LOG_OIII_LUM"].astype(float)
    )

    # If catalogue has linear luminosity but not log luminosity
    valid_lum = (
        df["LOG_OIII_LUM"].isna()
        & df["OIII_LUM"].notna()
        & np.isfinite(df["OIII_LUM"].astype(float))
        & (df["OIII_LUM"].astype(float) > 0)
    )

    df.loc[valid_lum, "LOG_OIII_LUM"] = np.log10(
        df.loc[valid_lum, "OIII_LUM"].astype(float)
    )

    # Convert catalogue log-luminosity error to linear luminosity error
    valid_log_err = (
        df["LOG_OIII_LUM_ERR"].notna()
        & df["OIII_LUM"].notna()
        & np.isfinite(df["LOG_OIII_LUM_ERR"].astype(float))
        & np.isfinite(df["OIII_LUM"].astype(float))
        & (df["LOG_OIII_LUM_ERR"].astype(float) > 0)
        & (df["OIII_LUM"].astype(float) > 0)
    )

    df.loc[valid_log_err, "OIII_LUM_ERR"] = (
        np.log(10)
        * df.loc[valid_log_err, "OIII_LUM"].astype(float)
        * df.loc[valid_log_err, "LOG_OIII_LUM_ERR"].astype(float)
    )

    # If somehow catalogue has linear luminosity error but not log error
    valid_lum_err = (
        df["LOG_OIII_LUM_ERR"].isna()
        & df["OIII_LUM_ERR"].notna()
        & df["OIII_LUM"].notna()
        & np.isfinite(df["OIII_LUM_ERR"].astype(float))
        & np.isfinite(df["OIII_LUM"].astype(float))
        & (df["OIII_LUM_ERR"].astype(float) > 0)
        & (df["OIII_LUM"].astype(float) > 0)
    )

    df.loc[valid_lum_err, "LOG_OIII_LUM_ERR"] = (
        df.loc[valid_lum_err, "OIII_LUM_ERR"].astype(float)
        / (np.log(10) * df.loc[valid_lum_err, "OIII_LUM"].astype(float))
    )

    print("Standardized OIII non-null counts:")
    print(df[[
        "LOG_OIII_LUM",
        "LOG_OIII_LUM_ERR",
        "OIII_LUM",
        "OIII_LUM_ERR",
    ]].notna().sum())

    return df



def get_targets_by_type():
    """Reads the CSVs and returns two separate lists of (plate, mjd, fiber)."""
    type1_targets = []
    type2_targets = []
    
    # Process Type 1
    try:
        df1 = pd.read_csv(os.path.join(BASE_DIR, "type1_candidates.csv"))

        # Dropping duplicates within the file itself
        df1_unique = df1.drop_duplicates(subset=['PLATE', 'mjd', 'fiber'])
        for _, row in df1_unique.iterrows():
            type1_targets.append((int(row['PLATE']), int(row['mjd']), int(row['fiber'])))
    except Exception as e:
        print(f"Error reading Type 1 file: {e}")

    # Process Type 2
    try:
        df2 = pd.read_csv(os.path.join(BASE_DIR, "type2_candidates.csv"))
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
def merge_gal_catalog(class_path, info_path, line_path=None, SNR_threshold=5):
    class_table = Table.read(class_path)
    info_table = Table.read(info_path)

    tables = [class_table, info_table]
    table_names = ["class_table", "info_table"]

    if line_path is not None:
        line_table = Table.read(line_path)
        tables.append(line_table)
        table_names.append("line_table")

    combined_table = hstack(tables, table_names=table_names)

    names_merged = [
        name for name in combined_table.colnames
        if len(combined_table[name].shape) <= 1
    ]

    merged_df = combined_table[names_merged].to_pandas()
    merged_df = decode_string_columns(merged_df)

    # Fill NaN values in SUBCLASS with an empty string so string matching does not crash.
    merged_df["SUBCLASS"] = merged_df["SUBCLASS"].fillna("")

    mask = (
        (merged_df["I_CLASS"] == 4)
        & (merged_df["Z_WARNING"] == 0)
        & (merged_df["SN_MEDIAN"] >= SNR_threshold)
        & (merged_df["SPECTROTYPE"] == "GALAXY")
        & (~merged_df["SUBCLASS"].str.contains("BROADLINE"))
    )

    df_filtered = merged_df[mask].copy()

    # Standardize redshift name for downstream code.
    if "Z" in df_filtered.columns and "z" not in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={"Z": "z"})

    # Add OIII luminosity from MPA-JHU OIII flux.
    required_oiii_cols = ["OIII_FLUX", "OIII_FLUX_ERR"]
    if all(col in df_filtered.columns for col in required_oiii_cols):
        df_filtered = add_oiii_luminosity_from_flux(
            df_filtered,
            z_col="z",
            flux_col="OIII_FLUX",
            flux_err_col="OIII_FLUX_ERR",
            flux_unit=1e-17,
        )
    else:
        print("Warning: OIII_FLUX / OIII_FLUX_ERR not found in merged Type 2 catalogue.")
        for col in ["OIII_FLUX", "OIII_FLUX_ERR", "OIII_LUM", "OIII_LUM_ERR", "LOG_OIII_LUM", "OIII_SNR"]:
            if col not in df_filtered.columns:
                df_filtered[col] = np.nan
    df_filtered = add_oiii_quality_flags(
    df_filtered,
    flux_col="OIII_FLUX",
    flux_err_col="OIII_FLUX_ERR",
    lum_col="OIII_LUM",
    lum_err_col="OIII_LUM_ERR",
    snr_col="OIII_SNR",
    sigma_col="OIII_SIGMA",
    voff_col="OIII_VOFF",
    chi2_col="OIII_CHI2",
    min_snr=5.0,
    min_lum=1e38,
    max_lum=1e45,
    min_sigma=30.0,
    max_sigma=1000.0,
    max_abs_voff=1000.0,
    max_chi2=None,
)
    # Keep OIII fit-quality columns if they exist; otherwise create them.
    for col in ["OIII_SIGMA", "OIII_VOFF", "OIII_CHI2"]:
        if col not in df_filtered.columns:
            df_filtered[col] = np.nan

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
    df_filtered = df_filtered.rename(columns={"REDSHIFT": "z"})
    
    df_filtered = standardize_shen_oiii_columns(df_filtered)
    
    df_filtered = add_oiii_flux_from_luminosity(
        df_filtered,
        z_col="z",
        lum_col="OIII_LUM",
        lum_err_col="OIII_LUM_ERR",
    )

    df_filtered = add_oiii_quality_flags(
        df_filtered,
        flux_col="OIII_FLUX",
        flux_err_col="OIII_FLUX_ERR",
        lum_col="OIII_LUM",
        lum_err_col="OIII_LUM_ERR",
        snr_col="OIII_SNR",
        sigma_col="OIII_SIGMA",
        voff_col="OIII_VOFF",
        chi2_col="OIII_CHI2",
        min_snr=5.0,
        min_lum=1e38,
        max_lum=1e45,
        min_sigma=30.0,
        max_sigma=1000.0,
        max_abs_voff=1000.0,
        max_chi2=None,
    )

    oiii_cols = [
    "OIII_FLUX",
    "OIII_FLUX_ERR",
    "OIII_LUM",
    "OIII_LUM_ERR",
    "LOG_OIII_LUM",
    "LOG_OIII_LUM_ERR",
    "OIII_SNR",
    "OIII_SIGMA",
    "OIII_VOFF",
    "OIII_CHI2",
    "OIII_VALID",
    "OIII_QUALITY_FLAG",
    ]

    for col in oiii_cols:
        if col not in df_filtered.columns:
            df_filtered[col] = np.nan

    cols_to_keep = [
        "filename",
        "SDSS_NAME",
        "RA",
        "DEC",
        "z",
        "MJD",
        "PLATE",
        "FIBER",
        "LINE_MED_SN_HB",
    ] + oiii_cols
    df_final = df_filtered[cols_to_keep].rename(columns={
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
        df_filtered = filter_SNR(df_filtered,'LINE_MED_SN_HB','LINE_MED_SN_HA',SNR_threshold)

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

    
    
   
    type2_candidates = merge_gal_catalog(
        class_path="/Users/amir/Documents/Deep learning/cl-agn classifier/cataloges/gal_iclass_table_dr7_v5_2.fits",
        info_path="/Users/amir/Documents/Deep learning/cl-agn classifier/cataloges/gal_info_dr7_v5_2.fit.gz",
        line_path="/Users/amir/Documents/Deep learning/cl-agn classifier/cataloges/gal_line_dr7_v5_2.fit.gz",
        SNR_threshold=5,
    )

    type2_candidates.to_csv(
        "/Users/amir/Documents/Deep learning/cl-agn classifier/data/O3_normalized_network/type2_candidates.csv",
        index=False,
    )


    type1_candidates = extract_shen_type1_catalog('/Users/amir/Documents/Deep learning/cl-agn classifier/cataloges/dr7_bh_Nov19_2013.fits.gz', SNR_threshold=5)
    type1_candidates.to_csv('/Users/amir/Documents/Deep learning/cl-agn classifier/data/O3_normalized_network/type1_candidates.csv', index=False)    
    
    main()