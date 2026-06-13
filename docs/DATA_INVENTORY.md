# CL-AGN v4 — training data inventory (per phase / survey / class)

Status: **✓ built** · **▶ run pending** · **⬇ download** · **~est/target**.
z-cap 0.9 everywhere. EVQ (88) excluded from every set.

**Directory layout (v4).** Built artifacts + new spectra → `data_v4/`; raw inputs +
existing spectra stay read-only in `data/`. Paths are centralized in `src/paths_v4.py`
(single source of truth); spectra are resolved by searching `data_v4/` then `data/`
(`datasets_v2._resolve`), so nothing needs to be copied/moved. New download dirs:
`data_v4/dr16_spectra/` (DR16/BOSS), `data_v4/sdssv_type2_spectra/` (SDSS-V type-2);
SDSS-V negative late epochs → `data_v4/dr16_sdssv_crossmatch/`.

**Type labels (no fitting, rule #1) now come from spAll `SUBCLASS` on every survey:**
- SDSS-V: `spAll-lite-v6_2_1-epoch.fits` → `extract_sdssv_spall.py` → `sdssv_spall_obs.pkl`
- DR16/eBOSS: `spAll-v5_13_0.fits` → `extract_dr16_spall.py` → `dr16_spall_obs.pkl`
- DESI: `desi_type1_type2.csv` `AGN_TYPE`
Mapping: `type2 = GALAXY & SUBCLASS=='AGN'` · `type1 = QSO, or GALAXY & SUBCLASS in {BROADLINE, AGN BROADLINE}`.
Quality: ZWARNING==0, SN_MEDIAN_ALL≥6, SPECPRIMARY, z<0.9 (DR16 also PLATEQUALITY=='good').

---

## Stage 2 — Siamese pairs (labeled, two-epoch, same object)

### Positives (CL-AGN transitions) — unchanged

| Survey pair | Source | Train | Test | Total | early-epoch instrument | Status |
|---|---|---|---|---|---|---|
| DR16 × SDSS-V (lowz) | `dr16_sdssv_crossmatch_lowz.pkl` | 81 | 20 | full range | BOSS/eBOSS (mostly) | ✓ |
| DR16 × DESI (paper2) | `paper2_{train,test}_pairs.pkl` | 434 | **15** | train uncapped; **test z<0.4** | BOSS/eBOSS (mostly) | ✓ |
| **Total** | | **515** | **35** | | | |

> **DESI z<0.4 cap (test only).** Same-object SDSS×DESI non-CL-AGN exist only at z<0.4,
> so there are no high-z DESI negatives to z-match the high-z paper2 positives. To avoid
> a redshift shortcut, the held-out **test** DESI arm is capped to z<0.4 (30→15 positives);
> **train keeps all 434** (extra real transitions; the SDSS-V arm, which has high-z
> negatives, still forces real high-z learning). DESI eval is therefore z<0.4 only.

### Negatives (stable, **type-verified both epochs**, instrument-matched) — 20:1 per survey

| Survey pair | Builder | Epochs (both typed via SUBCLASS) | Instrument vs pos | Pool | Status |
|---|---|---|---|---|---|
| DR16 × SDSS-V | `build_sdssv_negatives.py` | DR16 spAll **×** SDSS-V spAll, same type | **matched** (BOSS×SDSS-V) ✓ | TBD | ▶ on DR16+SDSS-V spAll |
| DR16 × DESI | `build_desi_negatives.py` | DR16 spAll **×** DESI `AGN_TYPE`, same type | **matched** (BOSS×DESI) ✓ | TBD | ▶ on DR16 spAll |

> **Both negative builders now use SDSS-DR16 (BOSS/eBOSS) as the SDSS epoch** —
> type-verified via spAll `SUBCLASS`, instrument-matched to the positives. **No DR7
> in any labeled pair** (DR7 is encoder-only). The earlier DR7-vs-BOSS confound is closed.

Carve target (20:1 per survey, train+test): SDSS-V ~1,620+400 (need pool ≥2,020) ·
DESI ~8,680+600 (need ≥9,280). Carve is z-matched; type-2 fraction ≈ pool's.

### Stage-2 totals (targets)

| | Train | Test |
|---|---|---|
| Positives | 515 | 50 |
| Negatives | ~10,300 | ~1,000 |
| **Total pairs** | **~10,815** | **~1,050** |

Per-survey test: SDSS-V 20 pos + ~400 neg · DESI 30 pos + ~600 neg.

---

## Stage 1 — SSL encoder pool (unlabeled, single spectra)

| Parquet | Survey | Type | v3 count | v4 status |
|---|---|---|---|---|
| `ssl_unified_dr7capped_desi` | SDSS-DR7 | type-1 / QSO | 12,000 | ▶ re-cut z<0.9 |
| `ssl_unified_dr7capped_desi` | DESI | QSO+GALAXY (t1+t2) | 12,021 | ▶ re-cut z<0.9 |
| `ssl_dr16_sdssv_extension` | DR16 | type-1 / QSO only | 10,434 | ▶ re-cut z<0.9 |
| `ssl_dr16_sdssv_extension` | SDSS-V | type-1 / QSO only | 7,957 | ▶ re-cut z<0.9 |
| **`ssl_type2`** (NEW) | **DR16 + SDSS-V** | **type-2** (spAll SUBCLASS=='AGN') | — | ▶ DR16 ~9k + SDSS-V ~5k |
| **`ssl_dr7_types`** (NEW) | **SDSS-DR7** | **type-1 + type-2** (legacy, uncapped) | — | ▶ ~18k t1 + ~18k t2 (on disk) |
| **Total** | | | ~42,412 | + ~14k type-2 + ~36k DR7 typed |

> **DR7 typed AGN added to the encoder** (`build_ssl_dr7_parquet.py`): ~18k type-1 + ~18k
> type-2 legacy DR7 spectra (on disk). DR7 stays **encoder-only** (never in pairs); the
> checkpoint selection (`ssl.select_survey=[sdssv,dr16]`) keeps DR7 out of the selection
> metric so it boosts type coverage without re-biasing the chosen encoder. This ~triples
> type-2 in the encoder (the frozen encoder is where type-2 separability is decided).
> Note: some DR7 type-1 QSO overlap the capped DR7 in `ssl_unified_dr7capped_desi` (minor
> double-count; aligned with the uncapped choice — can dedup by making that builder DESI-only).

Now the encoder gets type-2 in **both** instruments it must encode (DR16/eBOSS + SDSS-V),
matching the negative/positive epochs. (DR7 type-2 in `full_data/type2` remains an
optional add; not currently pooled.)

---

## What needs to be downloaded

| # | What | Count | Into | Source list |
|---|---|---|---|---|
| 1 | SDSS-V negatives: DR16 **and** SDSS-V epoch FITS | TBD | `dr16_spectra/` + `dr16_sdssv_crossmatch/` | `sdssv_dr16_negatives_to_download.csv` |
| 2 | DESI negative early-epoch FITS (DR16/BOSS) | TBD | `dr16_spectra/` | `desi_dr16_negatives_to_download.csv` |
| 3 | type-2 SSL FITS (DR16 + SDSS-V) | TBD (~14k) | `dr16_spectra/` + `sdssv_type2_spectra/` | `ssl_type2_to_download.csv` |

**On disk already:** legacy type1/2 (`full_data/{type1,type2}`, 36k), DESI spectra
(~151k incl. 93k type-2), DR7/DESI/DR16-QSO/SDSS-V-QSO SSL spectra.

**One-time prerequisites (not downloads):**
`extract_dr16_spall.py` (15.8 GB, columns only) → `dr16_spall_obs.pkl`;
`extract_sdssv_spall.py` (9.6 GB) → `sdssv_spall_obs.pkl`.
These two feed the negatives AND the type-2 SSL. Retire `merged.pkl`,
`extract_sdssv_observations.py`, `select_sdssv_type2_spall.py`, `build_ssl_sdssv_type2_parquet.py` (deprecated).

---

## Build order
```
extract_dr16_spall.py  +  extract_sdssv_spall.py        # spAll -> *_spall_obs.pkl
build_sdssv_negatives.py                                 # DR16×SDSS-V, type-verified  (+dl #1)
build_desi_negatives.py                                  # DR16×DESI, type-verified    (+dl #2)
build_phase2_pickle.py                                   # 20:1 carve, both-survey test
build_ssl_type2_parquet.py                               # DR16+SDSS-V type-2 SSL        (+dl #3)
prepare_clean_ssl_dr7desi.py / build_ssl_extension_parquet.py   # re-cut z<0.9
pretrain_ssl.py -> train_siamese_v2.py -> eval_clagn_test.py
```

## Open / TBD
1. Pool sizes after the spAll rebuilds (SDSS-V ≥2,020, DESI ≥9,280) — both depend on
   how many AGN have a DR16 counterpart; check the printed match counts.
2. v4 SSL recounts (z<0.9); type-2 yields (lower SN_MEDIAN_ALL to ~4 if thin).
3. Minor residual: a few *positives* have genuinely DR7-era early epochs (real CL-AGN
   data, can't reselect) — dominant instrument for both classes is now BOSS/eBOSS.
4. Optional: type-stratified negative carve; strict z<0.9 on positives.
