#!/usr/bin/env python3
"""
build_ssl_parquets_legacy.py
============================
Rebuild the SSL parquets with raw (un-normalised) flux from the legacy FITS
data.  Run ONCE in astro_dl before pretrain_ssl.py; overwrites the old
MAD-scaled parquets in <legacy-dir>/data_v4/.

Usage (from the cl-agn-classifier repo root):
    conda run -n astro_dl python src/build_ssl_parquets_legacy.py \\
        --legacy-dir "/Users/amir/Documents/Deep learning/cl-agn classifier -Legacy version"

Parquets built
--------------
data_v4/ssl_dr7_types.parquet
    DR7 type-1 + type-2  (~36 k spectra, uncapped)

data_v4/ssl_unified_dr7capped_desi.parquet
    DR7 capped at DR7_CAP per type + all DESI from snr_uniform CSV  (~34 k)

data_v4/ssl_dr16_sdssv_extension.parquet
    DR16 + SDSS-V sampled from the crossmatch pickle, limited to FITS that
    are already on disk  (up to EXT_CAP per survey)

Note: ssl_type2.parquet is NOT rebuilt — SDSS-V type-2 FITS are unavailable
(sdssv_type2_spectra/ is empty).  Remove it from config_v2.yml ssl_parquets.
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Add src/ to path so we can import data_preprocessing
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from data_preprocessing import process_single_spectrum, MASTER_GRID

# ---------------------------------------------------------------------------
# Tunable constants (mirror legacy build scripts)
# ---------------------------------------------------------------------------
DR7_CAP     = 12_000   # spectra per type (type1 / type2) in the capped parquet
EXT_CAP     = 20_000   # max DR16 or SDSS-V spectra for the extension parquet
MIN_SNR     = 4.0
MAX_ZEROS   = 0.80
N_JOBS      = -2       # joblib: all-but-one core

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def _process_one(fits_path: str, meta: dict):
    """Process one FITS → parquet-ready dict, or None on failure."""
    result = process_single_spectrum(fits_path, z=meta.get("z"))
    if result is None:
        return None
    out = {**meta}
    out["z"] = result["z"]
    out["valid_frac"] = result["valid_frac"]
    if pd.isna(out.get("snr")):
        out["snr"] = result["snr"]
    out["flux_array"] = result["flux_array"]
    return out


def build_from_tasks(
    tasks: list[tuple[str, dict]],
    output: str | Path,
    min_snr: float = MIN_SNR,
    max_zeros_pct: float = MAX_ZEROS,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    """
    Process a list of (fits_path, meta_dict) tasks in parallel and save a
    raw-flux parquet.  meta_dict should contain at least: z, survey, agn_type,
    obj_id.  Optional: snr.
    """
    output = Path(output)
    print(f"\n[build] {output.name}: {len(tasks):,} tasks …")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(p, m) for p, m in tasks
    )
    results = [r for r in results if r is not None]
    print(f"[build]   succeeded: {len(results):,} / {len(tasks):,}")
    if not results:
        raise RuntimeError(f"No spectra processed for {output.name}")

    flux_matrix = np.stack([r.pop("flux_array") for r in results])
    meta_df = pd.DataFrame(results)
    flux_df = pd.DataFrame(flux_matrix, columns=MASTER_GRID.astype(str))
    df = pd.concat([meta_df.reset_index(drop=True), flux_df], axis=1)
    df.columns = df.columns.astype(str)

    # --- quality filter ---
    n_before = len(df)
    snr_vals = pd.to_numeric(df.get("snr", pd.Series(np.nan, index=df.index)),
                              errors="coerce")
    mask_snr = snr_vals.isna() | (snr_vals >= min_snr)  # NaN = no cutoff
    zero_frac = (flux_matrix == 0.0).mean(axis=1)
    mask_cov = zero_frac <= max_zeros_pct
    df = df[mask_snr & mask_cov].copy()
    print(f"[build]   quality filter: kept {len(df):,} / {n_before:,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"[build]   → {output}  ({len(df):,} spectra)")
    survey_counts = df["survey"].value_counts().to_dict() if "survey" in df.columns else {}
    print(f"[build]   survey breakdown: {survey_counts}")
    return df


# ---------------------------------------------------------------------------
# Per-parquet task builders
# ---------------------------------------------------------------------------
def tasks_dr7(legacy_dir: str, cap: int | None = None,
              rng_seed: int = 42) -> list[tuple[str, dict]]:
    """
    DR7 type-1 and type-2 spectra from data/full_data/.

    type-1: spec_filename constructed from PLATE / mjd / fiber columns.
    type-2: spec_filename constructed from PLATE / MJD_class_table / FIBERID_class_table.
    """
    full_data = os.path.join(legacy_dir, "data", "full_data")
    t1_csv = os.path.join(full_data, "type1_candidates.csv")
    t2_csv = os.path.join(full_data, "type2_candidates.csv")
    t1_dir = os.path.join(full_data, "type1")
    t2_dir = os.path.join(full_data, "type2")

    rng = np.random.default_rng(rng_seed)
    tasks = []

    # ---- type 1 ----
    t1 = pd.read_csv(t1_csv, low_memory=False)
    t1["_fname"] = t1.apply(
        lambda r: f"spec-{int(r.PLATE):04d}-{int(r.mjd)}-{int(r.fiber):04d}.fits",
        axis=1,
    )
    on_disk = set(os.listdir(t1_dir))
    t1 = t1[t1["_fname"].isin(on_disk)].copy()
    if cap and len(t1) > cap:
        t1 = t1.iloc[rng.choice(len(t1), cap, replace=False)]
    for _, row in t1.iterrows():
        tasks.append((
            os.path.join(t1_dir, row["_fname"]),
            {"z": float(row["z"]), "survey": "sdss_dr7", "agn_type": "type1",
             "obj_id": str(row.get("SDSS_NAME", "")),
             "snr": float(row["snr"]) if "snr" in row and pd.notna(row["snr"]) else np.nan},
        ))
    print(f"[dr7]  type1 on disk: {len(t1):,}")

    # ---- type 2 ----
    t2 = pd.read_csv(t2_csv, low_memory=False)
    t2["_fname"] = t2.apply(
        lambda r: f"spec-{int(r.PLATE):04d}-{int(r.MJD_class_table)}-{int(r.FIBERID_class_table):04d}.fits",
        axis=1,
    )
    on_disk2 = set(os.listdir(t2_dir))
    t2 = t2[t2["_fname"].isin(on_disk2)].copy()
    if cap and len(t2) > cap:
        t2 = t2.iloc[rng.choice(len(t2), cap, replace=False)]
    for _, row in t2.iterrows():
        snr_val = row["SN_MEDIAN"] if "SN_MEDIAN" in row.index else np.nan
        tasks.append((
            os.path.join(t2_dir, row["_fname"]),
            {"z": float(row["z"]), "survey": "sdss_dr7", "agn_type": "type2",
             "obj_id": str(int(row["SPECOBJID"])) if "SPECOBJID" in row.index else "",
             "snr": float(snr_val) if pd.notna(snr_val) else np.nan},
        ))
    print(f"[dr7]  type2 on disk: {len(t2):,}")

    return tasks


def tasks_desi(legacy_dir: str, rng_seed: int = 42) -> list[tuple[str, dict]]:
    """
    DESI spectra from the SNR-uniform CSV (data/full_data/desi_type1_type2_snr_uniform.csv).
    Remaps old path column → actual FITS path under the legacy dir.
    Excludes any DESI spectra that appear in the held-out test pkl (specname_sdssv
    column) so test-set DESI objects don't leak into SSL pretraining.
    """
    full_data = os.path.join(legacy_dir, "data", "full_data")
    csv = os.path.join(full_data, "desi_type1_type2_snr_uniform.csv")
    desi_root = os.path.join(full_data, "desi_spectra")
    test_pkl = os.path.join(legacy_dir, "data_v4", "clagn_test.pkl")

    # Build exclusion set: DESI basenames from the test pkl's specname_sdssv column
    excluded_fnames: set[str] = set()
    if os.path.exists(test_pkl):
        with open(test_pkl, "rb") as f:
            test_df = pickle.load(f)
        for col in ("specname_dr16", "specname_sdssv"):
            if col in test_df.columns:
                excluded_fnames.update(
                    test_df[col].dropna().map(lambda x: os.path.basename(str(x)))
                )
        print(f"[desi] excluding {len(excluded_fnames):,} test-set specnames")

    df = pd.read_csv(csv, low_memory=False)
    tasks = []
    missing = excluded = 0
    for _, row in df.iterrows():
        # remap old base path to current legacy dir
        fname = os.path.basename(str(row["path"]))
        if fname in excluded_fnames:
            excluded += 1
            continue
        agn_t = int(row["agn_type"])
        subdir = "type1" if agn_t == 1 else "type2"
        fits_path = os.path.join(desi_root, subdir, fname)
        if not os.path.exists(fits_path):
            missing += 1
            continue
        tasks.append((
            fits_path,
            {"z": float(row["Z"]), "survey": "desi",
             "agn_type": "type1" if agn_t == 1 else "type2",
             "obj_id": str(int(row["TARGETID"])),
             "snr": float(row["snr"]) if pd.notna(row.get("snr")) else np.nan},
        ))
    print(f"[desi] from snr_uniform CSV: {len(tasks):,} on disk, "
          f"{excluded:,} excluded (test), {missing:,} missing")
    return tasks


def tasks_extension(legacy_dir: str, cap_per_survey: int = EXT_CAP,
                    rng_seed: int = 42) -> list[tuple[str, dict]]:
    """
    DR16 + SDSS-V from the crossmatch pickle (data/dr16-sdssv_crossmatch.pkl).
    Searches data_v4/dr16_sdssv_crossmatch/ then data/dr16_sdssv_crossmatch/.
    Excludes objects in the held-out test pickle if it exists.
    """
    pkl_path = os.path.join(legacy_dir, "data", "dr16-sdssv_crossmatch.pkl")
    spectra_roots = [
        os.path.join(legacy_dir, "data_v4", "dr16_sdssv_crossmatch"),
        os.path.join(legacy_dir, "data", "dr16_sdssv_crossmatch"),
    ]
    test_pkl = os.path.join(legacy_dir, "data_v4", "clagn_test.pkl")

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    # exclude held-out test objects
    if os.path.exists(test_pkl):
        with open(test_pkl, "rb") as f:
            test_df = pickle.load(f)
        test_ids = set()
        for col in ("specname_dr16", "specname_sdssv"):
            if col in test_df.columns:
                test_ids.update(test_df[col].dropna().map(os.path.basename).tolist())
        n_before = len(df)
        df = df[~df["specname"].map(os.path.basename).isin(test_ids)]
        print(f"[ext]  excluded {n_before - len(df):,} test objects")

    rng = np.random.default_rng(rng_seed)

    def _resolve(specname):
        basename = os.path.basename(str(specname))
        for root in spectra_roots:
            p = os.path.join(root, basename)
            if os.path.exists(p):
                return p
        return None

    tasks = []
    for survey in ("dr16", "sdssv"):
        sub = df[df["survey"] == survey].copy()
        sub["_path"] = sub["specname"].map(_resolve)
        sub = sub[sub["_path"].notna()]
        if cap_per_survey and len(sub) > cap_per_survey:
            sub = sub.iloc[rng.choice(len(sub), cap_per_survey, replace=False)]
        print(f"[ext]  {survey}: {len(sub):,} FITS on disk")
        for _, row in sub.iterrows():
            tasks.append((
                row["_path"],
                {"z": float(row["z"]), "survey": survey, "agn_type": "type1",
                 "obj_id": str(int(row["sdssid"])) if pd.notna(row.get("sdssid")) else "",
                 "snr": np.nan},
            ))
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rebuild SSL parquets with raw flux from legacy FITS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--legacy-dir",
        default="/Users/amir/Documents/Deep learning/cl-agn classifier -Legacy version",
        help="Root of the legacy project directory.",
    )
    parser.add_argument("--dr7-cap", type=int, default=DR7_CAP,
                        help="Max DR7 spectra per type in the capped parquet (0=no cap).")
    parser.add_argument("--ext-cap", type=int, default=EXT_CAP,
                        help="Max DR16/SDSS-V spectra for the extension parquet (0=no cap).")
    parser.add_argument("--min-snr", type=float, default=MIN_SNR)
    parser.add_argument("--max-zeros-pct", type=float, default=MAX_ZEROS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument(
        "--output", default=None,
        help="Output parquet path. Defaults to <legacy-dir>/data_v4/ssl_all_spectra.parquet",
    )
    args = parser.parse_args()

    legacy = args.legacy_dir
    out_dir = Path(legacy) / "data_v4"
    output = Path(args.output) if args.output else out_dir / "ssl_all_spectra.parquet"

    kwargs = dict(min_snr=args.min_snr, max_zeros_pct=args.max_zeros_pct,
                  n_jobs=args.n_jobs)

    # Collect all tasks from every source in one pass
    cap_dr7 = args.dr7_cap if args.dr7_cap > 0 else None
    cap_ext = args.ext_cap if args.ext_cap > 0 else None

    print("[main] Collecting tasks from all sources …")
    all_tasks = (
        tasks_dr7(legacy, cap=cap_dr7)
        + tasks_desi(legacy)
        + tasks_extension(legacy, cap_per_survey=cap_ext or 10**9)
    )
    print(f"[main] Total tasks: {len(all_tasks):,}")

    build_from_tasks(all_tasks, output, **kwargs)


if __name__ == "__main__":
    main()
