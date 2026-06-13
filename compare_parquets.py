import pandas as pd, numpy as np

OLD = "/Users/amir/Documents/Deep learning/cl-agn-classifier/data_trained/OIII_bug/ssl_all_spectra.parquet"
NEW = "/Users/amir/Documents/Deep learning/cl-agn-classifier/data_trained/ssl_all_spectra.parquet"

META = ["filename", "survey", "agn_type", "z", "snr", "obj_id", "valid_frac"]

old = pd.read_parquet(OLD)
new = pd.read_parquet(NEW)

print(f"Old shape: {old.shape}  |  New shape: {new.shape}")
print(f"\nOld surveys:\n{old['survey'].value_counts().to_string()}")
print(f"\nNew surveys:\n{new['survey'].value_counts().to_string()}")

# --- Same rows? ---
old_s = old["filename"].sort_values().reset_index(drop=True)
new_s = new["filename"].sort_values().reset_index(drop=True)
same_files = old_s.equals(new_s)
print(f"\nSame filenames (same rows): {same_files}")
if not same_files:
    only_old = set(old_s) - set(new_s)
    only_new = set(new_s) - set(old_s)
    print(f"  Only in old: {len(only_old)}  |  Only in new: {len(only_new)}")

# --- Flux comparison (should differ because OIII bug is fixed) ---
old_num = set(old.select_dtypes(include=[np.number]).columns)
new_num = set(new.select_dtypes(include=[np.number]).columns)
flux_cols = sorted(old_num & new_num)
print(f"\nNumeric columns in old but not new: {old_num - new_num}")
print(f"Numeric columns in new but not old: {new_num - old_num}")
print(f"Comparing {len(flux_cols)} shared numeric columns")
old_flux = old.sort_values("filename")[flux_cols].values.astype(np.float32)
new_flux = new.sort_values("filename")[flux_cols].values.astype(np.float32)

identical_flux = np.allclose(old_flux, new_flux, equal_nan=True)
print(f"\nFlux arrays identical: {identical_flux}  (expected False — OIII bug fixed)")
diff = np.abs(old_flux - new_flux)
print(f"Mean |flux diff|:          {np.nanmean(diff):.6f}")
print(f"Max  |flux diff|:          {np.nanmax(diff):.6f}")
print(f"Fraction of pixels changed: {(diff > 1e-6).mean():.4f}")
