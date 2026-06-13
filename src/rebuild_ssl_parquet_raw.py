#!/usr/bin/env python3
"""
rebuild_ssl_parquet_raw.py
==========================
Re-process the EXACT same spectra as the last training run (OIII_bug) but
with the fixed data_preprocessing.py that outputs raw (un-normalised) flux.

This lets you compare OIII-fixed vs OIII-bug performance on identical data.

Usage (from cl-agn-classifier repo root):
    conda run -n astro_dl python src/rebuild_ssl_parquet_raw.py

Input:  data_trained/OIII_bug/ssl_all_spectra.parquet  (metadata + old flux)
Output: <legacy-dir>/data_v4/ssl_all_spectra.parquet   (same rows, raw flux)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from data_preprocessing import process_single_spectrum, MASTER_GRID

LEGACY_DIR     = "/Users/amir/Documents/Deep learning/cl-agn classifier -Legacy version"
SOURCE_PARQUET = "/Users/amir/Documents/Deep learning/cl-agn-classifier/data_trained/OIII_bug/ssl_all_spectra.parquet"
OUTPUT_PARQUET = os.path.join(LEGACY_DIR, "data_v4", "ssl_all_spectra.parquet")

SPECTRA_ROOTS = [
    os.path.join(LEGACY_DIR, "data_v4"),
    os.path.join(LEGACY_DIR, "data"),
]
META_COLS = ["filename", "obj_id", "agn_type", "survey", "z", "snr"]
N_JOBS = 4   # conservative: each worker loads astropy+scipy; more cores → OOM on MPS


def build_fits_index(roots: list[str]) -> dict[str, str]:
    """Recursively scan roots → basename: full_path index (data_v4 wins)."""
    print("[index] Scanning FITS files …", flush=True)
    index: dict[str, str] = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith((".fits", ".fit")) and fn not in index:
                    index[fn] = os.path.join(dirpath, fn)
    print(f"[index] {len(index):,} unique FITS basenames found", flush=True)
    return index


# Worker function — receives a fully-resolved path so no global state needed
def _process_one(fits_path: str, meta: dict) -> dict | None:
    result = process_single_spectrum(fits_path, z=meta.get("z"))
    if result is None:
        return None
    return {
        "filename":   meta["filename"],
        "obj_id":     meta.get("obj_id", ""),
        "agn_type":   meta.get("agn_type", ""),
        "survey":     meta.get("survey", ""),
        "z":          result["z"],
        "snr":        result["snr"],
        "valid_frac": result["valid_frac"],
        "flux_array": result["flux_array"],
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source",  default=SOURCE_PARQUET)
    parser.add_argument("--output",  default=OUTPUT_PARQUET)
    parser.add_argument("--n-jobs",  type=int, default=N_JOBS,
                        help="Parallel workers. Keep <=6 on MPS to avoid OOM.")
    args = parser.parse_args()

    # ── 1. Read metadata (no flux columns) ───────────────────────────────
    print(f"[rebuild] Reading metadata from:\n  {args.source}")
    all_cols = pd.read_parquet(args.source, columns=["filename"]).columns  # probe
    cols = [c for c in META_COLS if c in
            pd.read_parquet(args.source, columns=META_COLS).columns]
    meta = pd.read_parquet(args.source, columns=cols)
    print(f"[rebuild] {len(meta):,} spectra | columns: {cols}")
    print(f"[rebuild] Survey counts:\n{meta['survey'].value_counts().to_string()}")

    # ── 2. Resolve paths in the main process ─────────────────────────────
    index = build_fits_index(SPECTRA_ROOTS)
    tasks = []
    missing_pre = []
    for row in meta.to_dict("records"):
        path = index.get(os.path.basename(row["filename"]))
        if path is None:
            missing_pre.append(row["filename"])
        else:
            tasks.append((path, row))

    print(f"[rebuild] {len(tasks):,} FITS resolved | {len(missing_pre):,} not found on disk")
    if missing_pre:
        print(f"[rebuild] First 5 missing: {missing_pre[:5]}")

    # ── 3. Process in parallel (workers receive full paths, no globals) ───
    print(f"\n[rebuild] Processing {len(tasks):,} spectra (n_jobs={args.n_jobs}) …")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_process_one)(p, m) for p, m in tasks
    )
    results = [r for r in results if r is not None]
    print(f"[rebuild] Succeeded: {len(results):,} / {len(tasks):,}")

    # ── 4. Assemble and save parquet ──────────────────────────────────────
    flux_matrix = np.stack([r.pop("flux_array") for r in results])
    meta_df = pd.DataFrame(results)
    flux_df  = pd.DataFrame(flux_matrix, columns=MASTER_GRID.astype(str))
    df = pd.concat([meta_df.reset_index(drop=True), flux_df], axis=1)
    df.columns = df.columns.astype(str)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n[rebuild] → {args.output}  ({len(df):,} spectra)")
    print(f"[rebuild] Survey breakdown:\n{df['survey'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
