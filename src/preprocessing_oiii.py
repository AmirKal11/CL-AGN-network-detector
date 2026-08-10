"""
preprocessing_oiii.py
=====================
Two-channel spectral representation for the CL-AGN change detector (v2 pipeline).

    Channel 0 : MAD-normalised flux, full continuum retained  -> robust SHAPE
    Channel 1 : [OIII] 5007-normalised flux                   -> cross-epoch AMPLITUDE

The continuum is deliberately KEPT (v2). It was subtracted in v1 only to stop a
single-spectrum Type 1/2 classifier shortcutting on continuum colour; that
shortcut does not exist for the self-supervised encoder (no class to cheat on)
or for the same-object change detector, and masked-reconstruction pretraining
learns far more from a full spectrum than from a near-noise residual.

Why two channels
----------------
Independent per-spectrum MAD normalisation rescales every epoch to unit robust
scale, so a real flux change between two epochs is partly normalised away.
[OIII] 5007 is emitted by the narrow-line region, which is light-years across,
so its flux is physically constant between epochs (years apart). Dividing each
epoch by its own [OIII] flux therefore:
    * puts both epochs on a common, physically meaningful amplitude scale,
    * preserves a real broad-line / continuum change as a genuine difference,
    * corrects throughput differences between SDSS and SDSS-V / DESI.

The MAD channel is kept as a robust, well-conditioned fallback for spectra whose
[OIII] is too weak or noisy to anchor on.

This module is intentionally pure NumPy so it can be unit-tested without torch.
The FITS-reading / interpolation stage (which needs astropy + scipy) lives in
datasets_v2.py.
"""

from __future__ import annotations

import json
import os

import numpy as np

# ----------------------------------------------------------------------
# Fixed grid and line windows  (rest-frame, Angstrom)
# ----------------------------------------------------------------------
# Identical grid to data_preprocessing.run_preprocessing(...).
# Wide rest-frame grid (SpectraNet / AppleCiDEr II convention): every SDSS /
# BOSS / DESI spectrum, at any redshift, covers some contiguous sub-range of
# it; pixels outside a given spectrum's observed coverage are zero-filled and
# flagged via a validity mask. This removes the old z < 0.4 restriction, which
# only existed because the previous narrow grid (4575-6699 A) had to be fully
# covered by every input spectrum.
GRID_MIN, GRID_MAX, GRID_N = 3000.0, 10400.0, 4096
MASTER_GRID = np.linspace(GRID_MIN, GRID_MAX, GRID_N)

# [OIII] 5007 measurement bands (rest-frame Angstrom).
# A local continuum is interpolated from the two side bands and subtracted
# before integrating the line band. This removes any broad-Hbeta pedestal
# sitting underneath [OIII], so the [OIII] anchor stays insensitive to the
# broad-line state (which is exactly the thing that changes in a CL-AGN).
OIII_LINE_BAND = (4996.0, 5018.0)   # [OIII] 5007 core; tolerates ~+/-10 A z error
OIII_BLUE_BAND = (4970.0, 4990.0)   # clear of [OIII] 4959 and the line core
OIII_RED_BAND = (5020.0, 5045.0)    # clear of the [OIII] 5007 red wing

# Broad-line WING regions (deliberately avoid the narrow cores).
# Same windows as broad_wing_score() in test_siamese_new_data.py.
BROAD_WING_REGIONS = {
    "hb": [(4797.8, 4840.0), (4885.0, 4927.5)],
    "ha": [(6477.1, 6535.0), (6600.0, 6652.1)],
}

# Mostly line-free band, used to estimate the per-spectrum noise level.
NOISE_BAND = (5150.0, 5700.0)

# Continuum-subtraction window (matches morphological_continuum_subtraction).
# ~313 A physical scale: 173 px on the 1.81 A/px wide grid (was 151 px on the
# old 2.07 A/px grid). Kept odd for a symmetric moving average.
CONTINUUM_WINDOW = 173


# ----------------------------------------------------------------------
# Low-level numeric helpers
# ----------------------------------------------------------------------
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Reflect-padded moving average; mirrors F.avg_pool1d(..., 'reflect')."""
    x = np.asarray(x, dtype=np.float64)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")
    csum = np.cumsum(np.insert(xp, 0, 0.0))
    ma = (csum[win:] - csum[:-win]) / float(win)
    return ma[: len(x)]


def _masked_moving_average(x: np.ndarray, valid: np.ndarray,
                           win: int) -> np.ndarray:
    """Reflect-padded moving average over covered pixels only.

    continuum[i] = mean of x over the in-window pixels that are valid. This
    mirrors the masked avg-pool in
    data_preprocessing.morphological_continuum_subtraction, so the SSL-parquet
    path and the FITS-pair path stay numerically identical.
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.float64)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xv = np.pad(x * valid, pad, mode="reflect")
    vv = np.pad(valid, pad, mode="reflect")
    cs_x = np.cumsum(np.insert(xv, 0, 0.0))
    cs_v = np.cumsum(np.insert(vv, 0, 0.0))
    num = cs_x[win:] - cs_x[:-win]
    den = cs_v[win:] - cs_v[:-win]
    ma = num / np.maximum(den, 1e-8)
    return ma[: len(x)]


def _edge_taper(x: np.ndarray, taper_len: int = 5) -> np.ndarray:
    """Linearly fade the first/last `taper_len` samples to zero at the edges."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = len(x)
    if taper_len <= 0 or n < 2 * taper_len:
        return x
    fade = np.linspace(0.0, 1.0, taper_len)
    x[:taper_len] *= fade
    x[-taper_len:] *= fade[::-1]
    return x


def continuum_subtract(flux: np.ndarray, valid: np.ndarray | None = None,
                       window_size: int = CONTINUUM_WINDOW) -> np.ndarray:
    """
    Continuum removal by wide moving-average subtraction.

    If `valid` is given, the moving-average continuum is estimated from covered
    pixels only (a masked moving average), so the zero-filled out-of-coverage
    regions do not drag the continuum toward zero near the coverage edges.
    Out-of-coverage pixels are returned as exactly 0.0 (the "missing" sentinel).
    """
    flux = np.asarray(flux, dtype=np.float64)
    if valid is None:
        continuum = _moving_average(flux, window_size)
        return (flux - continuum).astype(np.float32)
    valid = np.asarray(valid, dtype=bool)
    continuum = _masked_moving_average(flux, valid, window_size)
    out = flux - continuum
    out[~valid] = 0.0
    return out.astype(np.float32)


def mad_normalize(flat: np.ndarray, valid: np.ndarray | None = None,
                  taper_len: int = 5, eps: float = 1e-8):
    """
    Per-spectrum MAD normalisation of a continuum-subtracted spectrum.

        (flat - median) / (MAD * 1.4826)  then edge taper.

    If `valid` is given, the median and MAD are computed from covered pixels
    only -- otherwise the zero-filled out-of-coverage pixels (a large fraction
    of a high-redshift spectrum on the wide grid) would dominate the robust
    statistics. Out-of-coverage pixels are returned as exactly 0.0.

    Returns
    -------
    madnorm : np.ndarray  float32, shape [L]
    mad     : float       the raw MAD scale (diagnostic)
    """
    flat = np.asarray(flat, dtype=np.float64)
    if valid is None:
        ref = flat
    else:
        valid = np.asarray(valid, dtype=bool)
        ref = flat[valid]
        if ref.size < 2:
            return np.zeros_like(flat, dtype=np.float32), 0.0
    median = np.median(ref)
    mad = np.median(np.abs(ref - median))
    norm = (flat - median) / (mad * 1.4826 + eps)
    if valid is not None:
        norm[~valid] = 0.0
    norm = _edge_taper(norm, taper_len)
    return norm.astype(np.float32), float(mad)


# ----------------------------------------------------------------------
# [OIII] 5007 measurement
# ----------------------------------------------------------------------
def measure_oiii_flux(spec: np.ndarray, wave: np.ndarray = MASTER_GRID,
                      valid: np.ndarray | None = None):
    """
    Local-continuum-subtracted [OIII] 5007 flux. Works on either a
    continuum-subtracted or a full (continuum-retained) spectrum.

    A straight line is interpolated between the median flux of the blue and red
    side bands and subtracted from the line band before integrating. This is a
    simple band-photometry measurement (the same approach as the project's
    paper_style_line_flux) -- NOT a spectral decomposition / line fit -- so it
    stays inside the "no PyQSOFit" rule, while removing any broad-Hbeta pedestal
    that would otherwise make the [OIII] anchor drift with broad-line state.

    Parameters
    ----------
    spec : array [L]   flux on MASTER_GRID -- continuum-subtracted or full.
    wave : array [L]   rest-frame wavelength grid.

    Returns
    -------
    flux : float   local-continuum-subtracted integrated [OIII] flux.
    snr  : float   peak-over-noise ratio of the [OIII] line above local cont.
    """
    spec = np.asarray(spec, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)

    lm = (wave >= OIII_LINE_BAND[0]) & (wave <= OIII_LINE_BAND[1])
    bm = (wave >= OIII_BLUE_BAND[0]) & (wave <= OIII_BLUE_BAND[1])
    rm = (wave >= OIII_RED_BAND[0]) & (wave <= OIII_RED_BAND[1])
    nm = (wave >= NOISE_BAND[0]) & (wave <= NOISE_BAND[1])
    # Restrict every band to pixels with real coverage: on the wide grid a
    # band can fall partly or wholly inside the zero-filled out-of-coverage
    # region, where it carries no [OIII] information.
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        lm &= valid
        bm &= valid
        rm &= valid
        nm &= valid
    if lm.sum() < 2 or bm.sum() < 1 or rm.sum() < 1 or nm.sum() < 8:
        return 0.0, 0.0

    blue = np.median(spec[bm])
    red = np.median(spec[rm])
    blue_c = 0.5 * (OIII_BLUE_BAND[0] + OIII_BLUE_BAND[1])
    red_c = 0.5 * (OIII_RED_BAND[0] + OIII_RED_BAND[1])
    local_cont = np.interp(wave[lm], [blue_c, red_c], [blue, red])

    line = spec[lm] - local_cont
    dlam = float(np.mean(np.diff(wave)))      # uniform grid -> constant spacing
    flux = float(np.sum(line) * dlam)         # integral, no deprecated np.trapz

    # Pixel-noise from the line-free band. On a full (continuum-retained)
    # spectrum the 550 A band carries the continuum slope, which would inflate
    # a raw MAD; remove a smooth trend first so the MAD reflects pixel noise.
    # On a continuum-subtracted spectrum the trend is ~0, so this is a no-op.
    nseg = spec[nm]
    if nseg.size >= 64:
        resid = nseg - _moving_average(nseg, 31)
    else:
        resid = nseg - np.median(nseg)
    noise = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-8
    peak = float(np.max(line))
    snr = peak / noise
    return flux, snr



# ----------------------------------------------------------------------
# Channel-1 scale calibration
# ----------------------------------------------------------------------
def estimate_channel1_scale(madnorm_list, wave: np.ndarray = MASTER_GRID,
                            oiii_snr_min: float = 4.0) -> float:
    """
    Estimate a single global channel-1 scale so that channel 1 has roughly the
    same magnitude as channel 0.

    channel1_scale = 1 / median(reliable [OIII] fluxes)
        => for a typical object channel1 ~ channel0.

    Run this once over the pre-training (single-spectrum) set.
    """
    fluxes = []
    for m in madnorm_list:
        f, s = measure_oiii_flux(m, wave, valid=valid_from_flux(m))
        if s >= oiii_snr_min and f > 1e-6:
            fluxes.append(f)
    if len(fluxes) < 10:
        print("[preprocessing_oiii] WARNING: <10 reliable [OIII] spectra; "
              "channel1_scale defaults to 1.0")
        return 1.0
    return float(1.0 / np.median(fluxes))


def save_norm_stats(path: str, channel1_scale: float, extra: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"channel1_scale": float(channel1_scale)}
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[preprocessing_oiii] saved norm stats -> {path}")


def load_norm_stats(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[preprocessing_oiii] WARNING: {path} not found; "
              "channel1_scale defaults to 1.0")
        return {"channel1_scale": 1.0}
    with open(path) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------
# Coverage / validity helper
# ----------------------------------------------------------------------
def valid_from_flux(flux: np.ndarray) -> np.ndarray:
    """
    Reconstruct the per-pixel validity mask from a processed spectrum.

    Out-of-coverage pixels are written as exactly 0.0 by the preprocessing
    chain (zero-fill sentinel); continuum-subtracted, MAD-normalised real flux
    is a float that is essentially never exactly 0.0. This is the same
    convention clean_dataset() uses to measure per-spectrum coverage, so a
    parquet row needs no separate stored mask.
    """
    return np.asarray(flux) != 0.0


# ----------------------------------------------------------------------
# Synthetic within-object change generation (Stage-2 positive augmentation)
# ----------------------------------------------------------------------
def _line_covered(line: str, wave: np.ndarray,
                  valid: np.ndarray | None, min_frac: float = 0.8) -> bool:
    """True if a broad line's wing windows are (mostly) within real coverage."""
    if valid is None:
        return True
    valid = np.asarray(valid, dtype=bool)
    wave = np.asarray(wave, dtype=np.float64)
    pix = []
    for (w0, w1) in BROAD_WING_REGIONS[line]:
        idx = np.where((wave >= w0) & (wave <= w1))[0]
        if idx.size:
            pix.append(valid[idx])
    if not pix:
        return False
    pix = np.concatenate(pix)
    return pix.size > 0 and float(pix.mean()) >= min_frac
def suppress_broad_lines(
    spec: np.ndarray,
    wave: np.ndarray = MASTER_GRID,
    lines=("hb", "ha"),
    factor: float = 0.0,
    edge_taper: int = 4,
) -> np.ndarray:
    """
    Multiply the broad-line WING regions of `spec` by `factor` (0 = full
    turn-off, 1 = unchanged), with a short cosine-free linear blend at the
    window edges to avoid hard discontinuities.

    The narrow cores and the [OIII] window are left untouched, so this models a
    broad-line Type-1 -> Type-1.9/2 change while keeping host, narrow lines and
    continuum fixed. Applying it to a single line (`lines=("hb",)`) produces a
    partial transition.
    """
    spec = np.asarray(spec, dtype=np.float32)
    wave = np.asarray(wave, dtype=np.float64)
    out = spec.copy()

    for ln in lines:
        for (w0, w1) in BROAD_WING_REGIONS[ln]:
            idx = np.where((wave >= w0) & (wave <= w1))[0]
            n = idx.size
            if n == 0:
                continue
            mult = np.full(n, float(factor), dtype=np.float32)
            t = min(edge_taper, n // 2)
            if t > 0:
                ramp = np.linspace(1.0, float(factor), t + 1)[1:]
                mult[:t] = ramp
                mult[-t:] = ramp[::-1]
            out[idx] = out[idx] * mult
    return out


def make_synthetic_change(
    madnorm: np.ndarray,
    rng: np.random.Generator,
    wave: np.ndarray = MASTER_GRID,
    min_factor: float = 0.0,
    max_factor: float = 0.45,
    valid: np.ndarray | None = None,
):
    """
    Turn one epoch (a MAD-normalised spectrum) into a synthetic "changed" epoch
    by suppressing broad-line wings.

    Randomly chooses Hbeta-only, Halpha-only or both, so the model is trained on
    full AND partial transitions. On the wide grid a high-redshift spectrum may
    not cover Halpha (or, rarely, Hbeta); only broad lines whose wing windows
    are actually within coverage are eligible. If neither line is covered the
    epoch is returned untouched and meta["changed"] is False, so the caller
    must not relabel it as a positive.

    Returns
    -------
    changed_madnorm : np.ndarray   float32, shape [L]
    meta            : dict         {lines, factor, changed}
    """
    covered = [ln for ln in ("hb", "ha") if _line_covered(ln, wave, valid)]
    if not covered:
        return (np.asarray(madnorm, dtype=np.float32),
                {"lines": "", "factor": 1.0, "changed": False})
    if len(covered) == 2:
        choice = int(rng.integers(0, 3))
        lines = {0: ("hb",), 1: ("ha",), 2: ("hb", "ha")}[choice]
    else:
        lines = (covered[0],)
    factor = float(rng.uniform(min_factor, max_factor))
    changed = suppress_broad_lines(madnorm, wave, lines=lines, factor=factor)
    return changed, {"lines": "+".join(lines), "factor": factor, "changed": True}
