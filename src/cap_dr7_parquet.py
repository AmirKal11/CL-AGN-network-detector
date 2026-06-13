"""
cap_dr7_parquet.py
==================
Cap DR7 rows in the uncapped SSL parquet to DR7_CAP per agn_type,
then save a new parquet ready for SSL pretraining.

Usage:
    conda run -n astro_dl python src/cap_dr7_parquet.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

SOURCE  = "/Users/amir/Documents/Deep learning/cl-agn-classifier/data/ssl_all_spectra_uncapped_DR7.parquet"
OUTPUT  = "/Users/amir/Documents/Deep learning/cl-agn-classifier/data/ssl_all_spectra_dr7capped.parquet"
DR7_CAP = 12_000   # per agn_type
SEED    = 42

rng = np.random.default_rng(SEED)

print(f"[cap] Reading {SOURCE} …")
df = pd.read_parquet(SOURCE)
print(f"[cap] {len(df):,} total rows")
print(df.groupby(["survey", "agn_type"]).size().to_string())

dr7  = df[df["survey"] == "sdss_dr7"]
rest = df[df["survey"] != "sdss_dr7"]

print(f"\n[cap] DR7 agn_type breakdown:\n{dr7['agn_type'].value_counts().to_string()}")

# Normalise labels: treat "1"/"type-1"/"type1" all as "type1", same for type2
def normalise_type(t):
    t = str(t).strip().lower().replace("-", "").replace(" ", "")
    if t in ("1", "type1"):   return "type1"
    if t in ("2", "type2"):   return "type2"
    return t

dr7 = dr7.copy()
dr7["agn_type_norm"] = dr7["agn_type"].map(normalise_type)
print(f"\n[cap] DR7 normalised agn_type:\n{dr7['agn_type_norm'].value_counts().to_string()}")

capped_parts = []
for atype in ["type1", "type2"]:
    grp = dr7[dr7["agn_type_norm"] == atype].drop(columns=["agn_type_norm"])
    if len(grp) > DR7_CAP:
        grp = grp.iloc[rng.choice(len(grp), DR7_CAP, replace=False)]
        print(f"[cap] sdss_dr7 / {atype}: capped to {DR7_CAP:,}")
    else:
        print(f"[cap] sdss_dr7 / {atype}: {len(grp):,} (under cap, kept all)")
    capped_parts.append(grp)

out = pd.concat([rest] + capped_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"\n[cap] Final: {len(out):,} rows")
print(out.groupby(["survey", "agn_type"]).size().to_string())

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUTPUT, index=False)
print(f"\n[cap] Saved → {OUTPUT}")
