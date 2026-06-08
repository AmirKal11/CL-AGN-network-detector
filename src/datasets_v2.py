"""
datasets_v2.py
==============
Datasets for the redesigned CL-AGN pipeline.

SSLSpectraDataset
    Stage 1. Unlabelled single spectra pooled from one or more processed
    parquet catalogs (SDSS-DR7 + DESI). Emits 2-channel tensors; masking is
    applied in the training loop.

RealPairDataset
    Stage 2. Real same-object epoch pairs (DR16 spectrum + SDSS-V spectrum)
    from the crossmatch pickle. This replaces SyntheticSiameseDataset, whose
    cross-object pairs were the core bug. Optionally turns static pairs into
    within-object synthetic positives during training.

Helper functions
    load_or_build_pair_arrays  preprocess every FITS pair once, cache to .npz
    split_indices              object-disjoint, label-stratified train/val/test

All spectra end up as the identical 2-channel [MAD-norm, OIII-norm]
representation on the wide rest-frame grid (see preprocessing_oiii.MASTER_GRID).
Pixels outside a spectrum's observed coverage are zero-filled and carried as a
per-pixel validity mask, so spectra at any redshift can be used -- there is no
longer a z < 0.4 restriction.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing_oiii import (
    MASTER_GRID,
    CONTINUUM_WINDOW,
    continuum_subtract,
    mad_normalize,
    measure_oiii_flux,
    make_synthetic_change,
    valid_from_flux,
)


# ----------------------------------------------------------------------
# Column / FITS helpers
# ----------------------------------------------------------------------
def _is_wavelength_col(name) -> bool:
    """True if a column name is a float (a wavelength), i.e. a flux column."""
    try:
        float(name)
        return True
    except (ValueError, TypeError):
        return False


def read_fits_flux_wave(path):
    """
    Read observed-frame wavelength + flux from an SDSS/DESI-style FITS file.

    Mirrors the extraction logic of data_preprocessing.process_single_spectrum
    (extension 1; flux column; wavelength from loglam / wavelength / CRVAL1).
    """
    from astropy.io import fits  # imported lazily so the module loads w/o astropy

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        flux = np.asarray(data["flux"], dtype=np.float64)
        names = [str(n).lower() for n in data.names]
        if "loglam" in names:
            wave = 10.0 ** np.asarray(data["loglam"], dtype=np.float64)
        elif "wavelength" in names:
            wave = np.asarray(data["wavelength"], dtype=np.float64)
        else:
            hdr = hdul[1].header
            wave = hdr["CRVAL1"] + np.arange(len(flux)) * hdr["CDELT1"]
            wave = np.asarray(wave, dtype=np.float64)

    finite = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[finite], flux[finite]
    order = np.argsort(wave)
    return wave[order], flux[order]


def fits_to_flat(path, z, subtract_continuum=False):
    """
    FITS file -> rest-frame flux on MASTER_GRID, plus the per-pixel validity
    mask.

    Same preprocessing chain as data_preprocessing.process_single_spectrum:
    sky-line removal -> rest-frame correction -> interpolation to the master
    grid -> validity mask + zero-fill. With subtract_continuum=True the smooth
    continuum is then removed over covered pixels only; v2 leaves it
    subtract_continuum=False so the result matches the full-spectrum SSL
    parquet (data_preprocessing.build_unified_ssl_parquet also has it False).
    No MAD here -- MAD is applied separately so the [OIII] channel can be
    derived consistently.

    Returns
    -------
    flat  : np.ndarray float32 [L]  flux on the grid, 0.0 where invalid
    valid : np.ndarray bool    [L]  True where the spectrum has real coverage
    """
    from scipy.interpolate import interp1d
    from data_preprocessing import remove_sky_line

    wave_obs, flux_obs = read_fits_flux_wave(path)
    flux_obs = remove_sky_line(wave_obs, flux_obs, line_center=5577.3)

    wave_rest = wave_obs / (1.0 + float(z))
    flux_rest = flux_obs * (1.0 + float(z))

    f_interp = interp1d(wave_rest, flux_rest, bounds_error=False, fill_value=np.nan)
    interp = f_interp(MASTER_GRID)

    valid = np.isfinite(interp)
    if int(valid.sum()) < 50:
        raise ValueError(f"insufficient wavelength overlap on master grid: {path}")
    interp = np.nan_to_num(interp, nan=0.0)

    if subtract_continuum:
        flat = continuum_subtract(interp, valid=valid,
                                  window_size=CONTINUUM_WINDOW)
    else:
        # v2: keep the full continuum so the pair path matches the SSL
        # parquet representation. MAD normalisation happens later.
        flat = np.where(valid, interp, 0.0)
    return flat.astype(np.float32), valid.astype(bool)


def _two_channel(madnorm, oiii_flux, oiii_reliable, channel1_scale):
    """
    Assemble a [2, L] tensor-ready array from a MAD-normalised spectrum.

    Both channels are arcsinh-compressed. MAD-normalised spectra have a very
    heavy dynamic range -- noise is O(1) but line peaks reach tens to hundreds,
    and the [OIII]-divided channel 1 can be larger still for weak-[OIII]
    objects. Left raw, that tail makes the reconstruction loss explode and the
    training oscillate. arcsinh is linear near zero (noise is barely touched)
    and logarithmic for large values (peaks are compressed), and it is
    monotonic + sign-preserving, so relative line amplitudes -- the CL-AGN
    signal -- are preserved.
    """
    ch0 = madnorm
    if oiii_reliable:
        ch1 = (madnorm / oiii_flux) / channel1_scale
    else:
        ch1 = madnorm  # graceful fallback -> channel 1 == channel 0
    x = np.stack([ch0, ch1], axis=0)
    return np.arcsinh(x).astype(np.float32)


# ----------------------------------------------------------------------
# Stage 1: self-supervised single-spectrum dataset
# ----------------------------------------------------------------------
class SSLSpectraDataset(Dataset):
    """
    Unlabelled spectra pooled from processed parquet catalogs.

    Each parquet is expected to be MAD-scaled full spectra on the wide master
    grid (i.e. the output of data_preprocessing.build_unified_ssl_parquet,
    which sets subtract_continuum=False, apply_mad_scaling=True) -- so a
    parquet row is channel 0 directly.
    Out-of-coverage pixels are exactly 0.0; the per-pixel validity mask is
    recovered from that sentinel via preprocessing_oiii.valid_from_flux.

    Parameters
    ----------
    parquet_paths : list[str]   e.g. [DR7 parquet, DESI parquet]
    channel1_scale : float|None  if None it is calibrated from this pool
    max_rows : int|None          optional cap (debugging / low memory)

    __getitem__ returns (x [2, L] float tensor, valid [L] bool tensor);
    span-masking happens in the training loop via architectures_v2.apply_span_mask.
    """

    def __init__(self, parquet_paths, channel1_scale=None,
                 oiii_snr_min=4.0, max_rows=None, verbose=True):
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
        self.oiii_snr_min = float(oiii_snr_min)

        # Canonical flux columns from the first parquet (same grid everywhere).
        first = pd.read_parquet(parquet_paths[0])
        wl_cols = sorted([c for c in first.columns if _is_wavelength_col(c)],
                         key=lambda c: float(c))
        if len(wl_cols) != len(MASTER_GRID):
            print(f"[SSLSpectraDataset] WARNING: {len(wl_cols)} flux columns "
                  f"(expected {len(MASTER_GRID)})")
        self.wave = np.array([float(c) for c in wl_cols], dtype=np.float64)

        # Per-spectrum metadata kept alongside self.flux so downstream tools
        # (sanity plots, the reconstruction test, future raw-FITS lookups)
        # can look up parquet info for any row. Absent columns are filled
        # with NaN so this works on parquets that don't have every field.
        meta_cols = ["filename", "obj_id", "agn_type", "z", "snr", "survey"]

        blocks = []
        meta_blocks = []
        for p in parquet_paths:
            df = first if p == parquet_paths[0] else pd.read_parquet(p)
            missing = [c for c in wl_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{p} is missing {len(missing)} flux columns "
                                 f"-- parquets are not on the same grid")
            mat = df[wl_cols].to_numpy(dtype=np.float32)
            blocks.append(mat)
            meta_blocks.append(pd.DataFrame(
                {c: (df[c].to_numpy() if c in df.columns
                     else np.full(len(df), np.nan)) for c in meta_cols}
            ))
            if verbose:
                print(f"[SSLSpectraDataset] {os.path.basename(p)}: "
                      f"{mat.shape[0]:,} spectra")
            del df
        del first

        self.flux = np.concatenate(blocks, axis=0)
        del blocks
        self.meta = pd.concat(meta_blocks, ignore_index=True)
        del meta_blocks
        if max_rows is not None and len(self.flux) > max_rows:
            sel = np.random.default_rng(0).choice(len(self.flux), max_rows,
                                                  replace=False)
            self.flux = self.flux[sel]
            self.meta = self.meta.iloc[sel].reset_index(drop=True)
        if verbose:
            print(f"[SSLSpectraDataset] total pool: {len(self.flux):,} spectra")

        # Measure [OIII] once per spectrum.
        n = len(self.flux)
        self.oiii_flux = np.zeros(n, dtype=np.float32)
        reliable = np.zeros(n, dtype=bool)
        for i in range(n):
            vi = valid_from_flux(self.flux[i])
            f, s = measure_oiii_flux(self.flux[i], self.wave, valid=vi)
            self.oiii_flux[i] = f
            reliable[i] = (s >= self.oiii_snr_min) and (f > 1e-6)
        self.oiii_reliable = reliable

        if channel1_scale is None:
            rel = self.oiii_flux[reliable]
            self.channel1_scale = (float(1.0 / np.median(rel))
                                   if len(rel) >= 10 else 1.0)
        else:
            self.channel1_scale = float(channel1_scale)
        if verbose:
            print(f"[SSLSpectraDataset] [OIII] reliable: "
                  f"{reliable.sum():,}/{n:,}  channel1_scale={self.channel1_scale:.4g}")

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, i):
        row = self.flux[i]
        valid = valid_from_flux(row)
        x = _two_channel(row, self.oiii_flux[i],
                         bool(self.oiii_reliable[i]), self.channel1_scale)
        return torch.from_numpy(x), torch.from_numpy(valid)


# ----------------------------------------------------------------------
# Stage 2: real same-object pair preprocessing + dataset
# ----------------------------------------------------------------------
# v4: spectra are split across data_v4/ (new) and data/ (existing). Resolve by
# searching the given dir, then both roots, so pair_spectra_dir need not be exact.
_V4_ROOTS = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
             for d in ("data_v4", "data")]


def _resolve(spectra_dir, name):
    p = str(name)
    if os.path.isabs(p):
        return p
    cand = os.path.join(spectra_dir, p)
    if os.path.exists(cand):
        return cand
    for r in _V4_ROOTS:
        c = os.path.join(r, p)
        if os.path.exists(c):
            return c
    return cand


def load_or_build_pair_arrays(pkl_path, spectra_dir, cache_path=None,
                              oiii_snr_min=4.0, subtract_continuum=False,
                              verbose=True, split_filter=None):
    """
    Preprocess every real epoch pair once and return parallel arrays.

    Heavy FITS work is done a single time and cached to .npz, so the
    train/val/test datasets are cheap index views over the same arrays.
    """
    arrays = None
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cached_ok = ("valid1" in d.files
                     and d["mad1"].ndim == 2
                     and d["mad1"].shape[1] == len(MASTER_GRID)
                     and "repr_continuum_subtracted" in d.files
                     and bool(d["repr_continuum_subtracted"])
                     == bool(subtract_continuum))
        if cached_ok:
            if verbose:
                print(f"[pairs] loading cached preprocessing from {cache_path}")
            arrays = {k: d[k] for k in d.files}
        elif verbose:
            print(f"[pairs] cache {cache_path} is stale (grid, format or continuum mode changed) -- rebuilding")

    if arrays is None:
        df = pd.read_pickle(pkl_path)
        required = ["z", "specname_dr16", "specname_sdssv", "label"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"crossmatch pickle missing column '{c}'")
        has_id = "sdssid" in df.columns
        has_survey = "survey" in df.columns

        mad1, mad2 = [], []
        valid1, valid2 = [], []
        oiii1, oiii2, rel1, rel2 = [], [], [], []
        ys, ids, surveys = [], [], []
        n_fail = 0

        rows = list(df.itertuples(index=False))
        for k, row in enumerate(rows):
            z = float(getattr(row, "z"))
            p1 = _resolve(spectra_dir, getattr(row, "specname_dr16"))
            p2 = _resolve(spectra_dir, getattr(row, "specname_sdssv"))
            try:
                f1, v1 = fits_to_flat(p1, z, subtract_continuum=subtract_continuum)
                f2, v2 = fits_to_flat(p2, z, subtract_continuum=subtract_continuum)
                m1, _ = mad_normalize(f1, valid=v1)
                m2, _ = mad_normalize(f2, valid=v2)
                o1, s1 = measure_oiii_flux(m1, valid=v1)
                o2, s2 = measure_oiii_flux(m2, valid=v2)
            except Exception as exc:
                n_fail += 1
                if verbose and n_fail <= 5:
                    print(f"[pairs] skip row {k}: {exc}")
                continue

            mad1.append(m1)
            mad2.append(m2)
            valid1.append(v1)
            valid2.append(v2)
            oiii1.append(o1)
            oiii2.append(o2)
            rel1.append((s1 >= oiii_snr_min) and (o1 > 1e-6))
            rel2.append((s2 >= oiii_snr_min) and (o2 > 1e-6))
            ys.append(int(getattr(row, "label")))
            ids.append(getattr(row, "sdssid") if has_id else k)
            surveys.append(str(getattr(row, "survey")) if has_survey else "unknown")

            if verbose and (k + 1) % 500 == 0:
                print(f"[pairs] preprocessed {k + 1:,}/{len(rows):,}")

        arrays = {
            "mad1": np.asarray(mad1, dtype=np.float32),
            "mad2": np.asarray(mad2, dtype=np.float32),
            "valid1": np.asarray(valid1, dtype=bool),
            "valid2": np.asarray(valid2, dtype=bool),
            "oiii1": np.asarray(oiii1, dtype=np.float32),
            "oiii2": np.asarray(oiii2, dtype=np.float32),
            "rel1": np.asarray(rel1, dtype=bool),
            "rel2": np.asarray(rel2, dtype=bool),
            "y": np.asarray(ys, dtype=np.int64),
            "sdssid": np.asarray(ids),
            "survey": np.asarray(surveys),
            "repr_continuum_subtracted": np.asarray(bool(subtract_continuum)),
        }
        if verbose:
            n = len(arrays["y"])
            print(f"[pairs] preprocessed {n:,} pairs OK, {n_fail:,} failed/missing")
            print(f"[pairs] labels: static={int((arrays['y'] == 0).sum()):,}  "
                  f"CL-AGN={int((arrays['y'] == 1).sum()):,}")
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            np.savez(cache_path, **arrays)
            if verbose:
                print(f"[pairs] cached preprocessing -> {cache_path}")

    if split_filter is not None:
        df = pd.read_pickle(pkl_path)
        if "split" in df.columns:
            keep_ids = set(df[df["split"] == split_filter]["sdssid"].values)
            keep_mask = np.array([sid in keep_ids for sid in arrays["sdssid"]])
            if verbose:
                print(f"[pairs] filtered by split '{split_filter}': kept {keep_mask.sum()} of {len(keep_mask)}")
            for k in arrays:
                if k != "repr_continuum_subtracted" and len(np.shape(arrays[k])) > 0:
                    arrays[k] = arrays[k][keep_mask]
    return arrays


def split_indices(y, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Object-disjoint, label-stratified train/val/test split.

    Each pair is a distinct object, so a row-level split is automatically
    object-disjoint. Stratifying on the label keeps a few of the 54 real
    CL-AGN in every split.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        rng.shuffle(ci)
        n = len(ci)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test.extend(ci[:n_test])
        val.extend(ci[n_test:n_test + n_val])
        train.extend(ci[n_test + n_val:])
    return (np.array(sorted(train), dtype=int),
            np.array(sorted(val), dtype=int),
            np.array(sorted(test), dtype=int))


class RealPairDataset(Dataset):
    """
    Real same-object epoch pairs, as 2-channel tensors.

    Parameters
    ----------
    arrays : dict           output of load_or_build_pair_arrays
    indices : array         which rows of `arrays` this split uses
    channel1_scale : float  the [OIII] scale calibrated in Stage 1
    mode : 'train'|'val'|'test'
    synthetic_prob : float  (train only) probability that a static pair is
                            turned into a within-object synthetic positive
    seed : int

    __getitem__ returns (x1 [2,L], x2 [2,L], y [1]).
    Validation/test always return real pairs with real labels.
    """

    def __init__(self, arrays, indices, channel1_scale, mode="train",
                 synthetic_prob=0.35, seed=42):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train/val/test, got {mode}")
        self.a = arrays
        self.indices = np.asarray(indices, dtype=int)
        self.channel1_scale = float(channel1_scale)
        self.mode = mode
        self.synthetic_prob = float(synthetic_prob) if mode == "train" else 0.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def label_counts(self):
        y = self.a["y"][self.indices]
        return {"static": int((y == 0).sum()), "clagn": int((y == 1).sum())}

    def __getitem__(self, i):
        row = self.indices[i]
        m1 = self.a["mad1"][row].copy()
        m2 = self.a["mad2"][row].copy()
        v1 = self.a["valid1"][row]
        v2 = self.a["valid2"][row]
        o1, r1 = float(self.a["oiii1"][row]), bool(self.a["rel1"][row])
        o2, r2 = float(self.a["oiii2"][row]), bool(self.a["rel2"][row])
        y = int(self.a["y"][row])

        # Within-object synthetic positive: suppress broad-line wings on one
        # epoch. The [OIII] window is untouched, so o1/o2/r1/r2 stay valid.
        # On the wide grid make_synthetic_change only suppresses broad lines
        # that are within coverage; if neither is, meta["changed"] is False
        # and the pair correctly stays labelled static.
        if self.mode == "train" and y == 0 and \
                self.rng.random() < self.synthetic_prob:
            if self.rng.random() < 0.5:
                m1, meta = make_synthetic_change(m1, self.rng, valid=v1)
            else:
                m2, meta = make_synthetic_change(m2, self.rng, valid=v2)
            if meta["changed"]:
                y = 1

        x1 = _two_channel(m1, o1, r1, self.channel1_scale)
        x2 = _two_channel(m2, o2, r2, self.channel1_scale)
        return (torch.from_numpy(x1),
                torch.from_numpy(x2),
                torch.tensor([y], dtype=torch.float32))
