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


def fits_to_flat(path, z):
    """
    FITS file -> raw rest-frame flux on MASTER_GRID, plus the per-pixel
    validity mask.

    Processing chain: sky-line removal -> rest-frame correction ->
    interpolation to the master grid -> zero-fill out-of-coverage pixels.
    No continuum subtraction or normalisation -- those are deferred so that
    OIII can be measured on the physical flux before MAD normalisation.

    Returns
    -------
    flat  : np.ndarray float32 [L]  raw flux on the grid, 0.0 where invalid
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

    flat = np.where(valid, interp, 0.0)
    return flat.astype(np.float32), valid.astype(bool)


def _two_channel(raw_flux, madnorm, oiii_flux, oiii_reliable, channel1_scale):
    """
    Assemble a [2, L] tensor-ready array.

    ch0 = MAD-normalised flux (robust shape encoding)
    ch1 = [OIII]-normalised raw flux (cross-epoch amplitude anchor)

    OIII flux must be measured on the raw physical spectrum (before MAD
    normalisation) so the amplitude anchor retains its physical meaning.
    Dividing raw_flux by oiii_flux puts both epochs on a common scale;
    channel1_scale normalises so the typical ch1 magnitude matches ch0.

    Both channels are arcsinh-compressed to tame the heavy dynamic-range tail
    (line peaks, and the OIII-divided channel for weak-OIII objects) while
    preserving sign and relative line amplitudes -- the CL-AGN signal.
    """
    ch0 = madnorm
    if oiii_reliable:
        ch1 = raw_flux / (oiii_flux * channel1_scale)
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

    Each parquet is expected to contain raw physical flux on the wide master
    grid (the output of data_preprocessing.build_parquet): sky-removed,
    de-redshifted, resampled, out-of-coverage pixels zero-filled. No
    normalisation in the parquet -- continuum subtraction and MAD normalisation
    happen in __getitem__; OIII is pre-computed in __init__ on raw flux.
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

    # Survey loss weights: upweight deployment surveys (sdssv, dr16) relative to
    # the large DR7/DESI pretraining pool to reduce the domain gap.
    SURVEY_LOSS_WEIGHTS: dict[str, float] = {
        "sdssv":    3.0,
        "dr16":     3.0,
        "sdss_dr7": 1.0,
        "desi":     1.0,
    }

    def __getitem__(self, i):
        raw = self.flux[i]                           # raw physical flux from parquet
        valid = valid_from_flux(raw)
        cs = continuum_subtract(raw, valid=valid)    # continuum-subtracted (shape)
        madnorm, _ = mad_normalize(cs, valid=valid)  # MAD-normalised (ch0)
        x = _two_channel(raw, madnorm, self.oiii_flux[i],
                         bool(self.oiii_reliable[i]), self.channel1_scale)
        survey = str(self.meta.iloc[i].get("survey", "")) if hasattr(self.meta.iloc[i], "get") else str(self.meta["survey"].iloc[i])
        w = self.SURVEY_LOSS_WEIGHTS.get(survey, 1.0)
        return torch.from_numpy(x), torch.from_numpy(valid), torch.tensor(w, dtype=torch.float32)


# ----------------------------------------------------------------------
# Stage 2: real same-object pair preprocessing + dataset
# ----------------------------------------------------------------------
# v4: spectra are split across data_v4/ (new) and data/ (existing). Resolve by
# searching the given dir, then sibling dirs of spectra_dir, then project roots.
_V4_ROOTS = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
             for d in ("data_v4", "data")]


def _resolve(spectra_dir, name):
    p = str(name)
    if os.path.isabs(p):
        return p
    cand = os.path.join(spectra_dir, p)
    if os.path.exists(cand):
        return cand
    # Check sibling directories of spectra_dir (e.g. data/ when spectra_dir is data_v4/)
    parent = os.path.dirname(os.path.abspath(spectra_dir))
    try:
        siblings = [os.path.join(parent, d) for d in os.listdir(parent)
                    if os.path.isdir(os.path.join(parent, d))]
    except OSError:
        siblings = []
    for sib in siblings:
        c = os.path.join(sib, p)
        if os.path.exists(c):
            return c
    # Project-relative fallback
    for r in _V4_ROOTS:
        c = os.path.join(r, p)
        if os.path.exists(c):
            return c
    return cand


def load_or_build_pair_arrays(pkl_path, spectra_dir, cache_path=None,
                              oiii_snr_min=4.0, verbose=True,
                              split_filter=None):
    """
    Preprocess every real epoch pair once and return parallel arrays.

    Heavy FITS work is done a single time and cached to .npz, so the
    train/val/test datasets are cheap index views over the same arrays.

    Processing order per pair:
        1. fits_to_flat -> raw physical flux f1, f2
        2. measure_oiii_flux(f1/f2) -> OIII on raw (physically meaningful)
        3. continuum_subtract(f1/f2) -> remove smooth continuum
        4. mad_normalize(cs1/cs2) -> robust shape normalisation

    Both raw and MAD-normalised arrays are stored so _two_channel can use raw
    for ch1 (OIII-normalised amplitude) and mad for ch0 (shape).
    """
    arrays = None
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cached_ok = ("valid1" in d.files
                     and "raw1" in d.files
                     and d["mad1"].ndim == 2
                     and d["mad1"].shape[1] == len(MASTER_GRID))
        if cached_ok:
            if verbose:
                print(f"[pairs] loading cached preprocessing from {cache_path}")
            arrays = {k: d[k] for k in d.files}
        elif verbose:
            print(f"[pairs] cache {cache_path} is stale (grid or format changed) -- rebuilding")

    if arrays is None:
        df = pd.read_pickle(pkl_path)
        required = ["z", "specname_dr16", "specname_sdssv", "label"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"crossmatch pickle missing column '{c}'")
        has_id = "sdssid" in df.columns
        has_survey = "survey" in df.columns

        raw1, raw2 = [], []
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
                f1, v1 = fits_to_flat(p1, z)
                f2, v2 = fits_to_flat(p2, z)
                # Measure OIII on raw physical flux BEFORE any normalisation,
                # so the amplitude anchor retains its physical meaning.
                o1, s1 = measure_oiii_flux(f1, valid=v1)
                o2, s2 = measure_oiii_flux(f2, valid=v2)
                # Continuum subtraction then MAD normalisation for ch0 shape.
                cs1 = continuum_subtract(f1, valid=v1)
                cs2 = continuum_subtract(f2, valid=v2)
                m1, _ = mad_normalize(cs1, valid=v1)
                m2, _ = mad_normalize(cs2, valid=v2)
            except Exception as exc:
                n_fail += 1
                if verbose and n_fail <= 5:
                    print(f"[pairs] skip row {k}: {exc}")
                continue

            raw1.append(f1)
            raw2.append(f2)
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
            "raw1": np.asarray(raw1, dtype=np.float32),
            "raw2": np.asarray(raw2, dtype=np.float32),
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
        f1 = self.a["raw1"][row]               # raw physical flux (for ch1)
        f2 = self.a["raw2"][row]
        m1 = self.a["mad1"][row].copy()        # MAD-normalised (for ch0)
        m2 = self.a["mad2"][row].copy()
        v1 = self.a["valid1"][row]
        v2 = self.a["valid2"][row]
        o1, rel1 = float(self.a["oiii1"][row]), bool(self.a["rel1"][row])
        o2, rel2 = float(self.a["oiii2"][row]), bool(self.a["rel2"][row])
        y = int(self.a["y"][row])

        # Within-object synthetic positive: suppress broad-line wings on one
        # epoch's MAD-normalised spectrum. OIII and raw arrays are untouched,
        # so the amplitude anchor (ch1) correctly reflects the un-changed epoch.
        # synthetic_prob is 0.0 by default (disabled); this path is dead code
        # unless explicitly enabled.
        if self.mode == "train" and y == 0 and \
                self.rng.random() < self.synthetic_prob:
            if self.rng.random() < 0.5:
                m1, meta = make_synthetic_change(m1, self.rng, valid=v1)
            else:
                m2, meta = make_synthetic_change(m2, self.rng, valid=v2)
            if meta["changed"]:
                y = 1

        x1 = _two_channel(f1, m1, o1, rel1, self.channel1_scale)
        x2 = _two_channel(f2, m2, o2, rel2, self.channel1_scale)
        return (torch.from_numpy(x1),
                torch.from_numpy(x2),
                torch.tensor([y], dtype=torch.float32))
