# CL-AGN pipeline — file-by-file, raw data → training (v4)

Reading guide for reviewing the code end to end. Each entry: **[TAG]** role —
`inputs → outputs` — *what to scrutinize*.

Tags: **RUN** (run for v4, in order) · **RUN-once** (upstream/given — review only) ·
**LIB** (imported, not run) · **CFG** (config).

**Layout:** raw inputs + existing spectra are read-only in `data/`; everything the v4
pipeline builds + new spectra downloads go to `data_v4/`. Paths are centralized in
`src/paths_v4.py`; spectra are resolved by searching `data_v4/` then `data/`. Run from
the project root in conda `astro_dl`. (Assistant syntax-checks only; you run.)

**Priority:** the science target is the **SDSS-V × DR16** arm. DESI is auxiliary —
included only for its large CL-AGN positive count; judge the model on the SDSS-V arm.

---

## Stage 0 — Upstream / given (REVIEW only, already built)

1. **[RUN-once] `Main_analysis.ipynb`** — SDSS-V×DR16 base. `merged.pkl →
   dr16-sdssv_crossmatch.pkl` (174k). *Scrutinize:* `is_qso` + `is_bhm|is_nan` then
   DR16∩SDSS-V — the QSO-both + BHM cut that also gates the lowz positives.
2. **[RUN-once] `cataloge_handler.py`** — SDSS-epoch type catalogs
   `full_data/type{1,2}_candidates.csv` (legacy/DR7-era; now only used for the encoder,
   not pairs).
3. **[LIB+once] `data_preprocessing.py`** — `process_single_spectrum` (rest-frame/MAD/
   arcsinh grid 3000–10400 Å/4096 px) + `build_unified_ssl_parquet` → `ssl_unified_dr7_desi.parquet`.
4. **[RUN-once] `vac_filter.py`** — VAC quality filter → `ssl_unified_dr7_desi_filtered.parquet`.
5. **Given:** `cl_agn_list_dr16.pkl` (116 CL-AGN + 88 EVQ — the exclusion list),
   `dr16_sdssv_crossmatch_lowz.pkl` (101 lowz positives), `paper2_{master,train,test}_pairs.pkl`
   (464 DESI CL-AGN positives), `full_data/desi_type1_type2.csv` (DESI AGN_TYPE VAC),
   the two spAll catalogs (in the Thesis dir).

---

## Stage 1 — spAll type extracts (RUN, one-time; the prerequisite for everything v4)

6. **[RUN] `extract_dr16_spall.py`** — DR16/eBOSS AGN from `spAll-v5_13_0.fits` (15.8 GB,
   columns only). type via SUBCLASS (type2=GALAXY&AGN; type1=QSO or galaxy-broadline),
   cuts PLATEQUALITY='good'+ZWARNING=0+SN≥6+SPECPRIMARY+z<0.9. → `data_v4/dr16_spall_obs.pkl`.
7. **[RUN] `extract_sdssv_spall.py`** — same for SDSS-V `spAll-lite…` → `data_v4/sdssv_spall_obs.pkl`.
   These two feed BOTH the negatives and the type-2 encoder pool. *Watch:* type1/type2 counts.

---

## Stage 2 — Negatives (RUN) — DR16/BOSS × {SDSS-V | DESI}, type-verified both epochs

Every pair is **SDSS-DR16 (BOSS/eBOSS) × {SDSS-V|DESI}** — instrument-matched to the
positives, no DR7 in pairs. Both exclude CL-AGN+EVQ via `cl_agn_list_dr16.pkl`.

8. **[RUN] `build_sdssv_negatives.py`** — `dr16_spall_obs × sdssv_spall_obs`, same object
   + same type (stable), excl CL-AGN/EVQ+Paper-2. → `data_v4/sdssv_dr16_negatives.pkl`
   (+ download CSV). *Watch:* type1/type2 counts (type-2 ~tens — two-epoch type-2 is scarce).
9. **[RUN] `build_desi_negatives.py`** — `dr16_spall_obs × DESI AGN_TYPE VAC`, same object
   + same type. → `data_v4/desi_dr16_negatives.pkl` (+ download CSV). *Note:* the
   same-object SDSS×DESI overlap is **z<0.4 only** (handled by the test cap in step 10).

→ **Download** the negative FITS the two CSVs list: DR16/BOSS early epochs →
`data_v4/dr16_spectra/`; SDSS-V late epochs → `data_v4/dr16_sdssv_crossmatch/`. (DESI
late epochs already on disk.) Do this **before** step 10 — the carve uses on-disk pairs only.

> `build_neg_expansion_pickle.py`, the legacy `type{1,2}_candidates`-based negatives,
> `extract_sdssv_observations.py`/`merged.pkl` are all SUPERSEDED (DR7-era → confound).

---

## Stage 3 — Phase-2 pairs (RUN) — assembles train + defines the held-out test

10. **[RUN] `build_phase2_pickle.py`** — merges lowz + paper2 positives with the two clean
    negative pools, carves **20:1 per survey** (z-matched), builds the both-survey test,
    runs the train⟂test 1″ object-disjoint leak guard.
    → `data_v4/dr16_sdssv_phase2_train.pkl` + `data_v4/clagn_test.pkl`.
    *v4 specifics:* `Z_MAX=0.9`; **`Z_MAX_DESI=0.4`** — DESI **test** capped to z<0.4
    (no high-z DESI negatives → would be a redshift shortcut) while DESI **train** keeps
    all 434 paper2 positives; `_load_*_negatives` filter to on-disk + warn if 0 type-2.
    **Must run before the SSL builders** (they exclude `clagn_test.pkl`; they fail loudly if missing).

---

## Stage 4 — SSL encoder pool (RUN, after step 10) — three parquets, all test-excluded

11. **[RUN] `prepare_clean_ssl_dr7desi.py`** — DR7+DESI base re-cut: z<0.9, drop ALL test
    objects (DR7 *and* DESI rows). → `data_v4/ssl_unified_dr7capped_desi.parquet`. (DR7 is
    encoder-only.)
12. **[RUN] `build_ssl_extension_parquet.py`** — DR16+SDSS-V **QSO/type-1** spectra, z<0.9,
    test-excluded. → `data_v4/ssl_dr16_sdssv_extension.parquet`.
13. **[RUN] `build_ssl_type2_parquet.py`** — DR16 **and** SDSS-V **type-2** spectra
    (from `dr16_spall_obs` + `sdssv_spall_obs`, `sdssv_type=='type2'`), test/CL-AGN-excluded.
    → `data_v4/ssl_type2.parquet` (+ download CSV). → **download** the type-2 FITS
    (`data_v4/dr16_spectra/` + `data_v4/sdssv_type2_spectra/`) and **re-run**.
    (Supersedes `build_ssl_sdssv_type2_parquet.py` / `select_sdssv_type2_spall.py`.)

All three are listed in `config_v2.yml → ssl_parquets` (pointing at `data_v4/`).

---

## Stage 5 — Shared library / config (LIB/CFG — review, not run)

14. **[LIB] `paths_v4.py`** — single source of truth: `DATA_RAW`/`DATA_OUT`, spAll paths,
    every built-output path, 2-root spectrum resolver (`spec_exists`/`spec_path`).
15. **[LIB] `datasets_v2.py`** — `load_or_build_pair_arrays` (FITS→arrays, carries `survey`),
    `RealPairDataset`, `SSLSpectraDataset`, `split_indices`, `_resolve` (searches data_v4/ then data/).
16. **[LIB] `preprocessing_oiii.py`** — `MASTER_GRID`, MAD/arcsinh, [OIII] channel + fallback.
17. **[LIB] `architectures_v2.py`** — `MaskedSpectraAutoencoder`, encoder (CNN+transformer→512),
    `SiameseChangeNet` (symmetric fusion).
18. **[LIB] `utils.py`** — `load_config`, `save/load_norm_stats`, `pick_device`.
19. **[CFG] `config_v2.yml`** — built paths → `data_v4/`; out_dir `models/clagn_v4`;
    F2 (`fbeta_beta 2.0`, `min_recall 0.30`, `max_fpr 0.05`); `sampler_pos_rate 0.3`;
    `source_balanced true`; `ssl.select_survey [sdssv,dr16]`; `select_survey_stage2 sdssv`.

---

## Stage 6 — Training (after all data built)
```bash
rm -f models/clagn_v4/*cache*.npz
python src/pretrain_ssl.py        # checkpoints on SDSS-V+DR16 recon val
python src/train_siamese_v2.py    # source-balanced; selects on SDSS-V val
python src/eval_clagn_test.py     # per-survey + per-object prob dump
```

## Condensed run order
```
extract_dr16_spall.py + extract_sdssv_spall.py
  → build_sdssv_negatives.py + build_desi_negatives.py → [download neg FITS]
  → build_phase2_pickle.py
  → prepare_clean_ssl_dr7desi.py + build_ssl_extension_parquet.py
  → build_ssl_type2_parquet.py → [download type-2 FITS] → re-run
  → pretrain_ssl.py → train_siamese_v2.py → eval_clagn_test.py
```

## Cuts worth watching while reviewing
- **z<0.9** everywhere; **DESI test z<0.4** (no high-z DESI negatives). Mirror z<0.9 at deployment.
- **QSO-both + BHM** in `Main_analysis.ipynb` gates the lowz positives (biases against
  QSO→galaxy transitions); the type-2 SSL pool + spAll-typed negatives mitigate the
  encoder side, but the positive selection bias remains.
- **SN_MEDIAN_ALL ≥ 6** on the spAll extracts (lower to ~4 if type-2 too thin — already tried; type-2 stays scarce by data availability, not by cut).
- **Two-epoch type-2 negatives are ~tens total** — a hard data limit; the encoder gets type-2 via `ssl_type2`, but type-2 *negatives* are intrinsically scarce.
