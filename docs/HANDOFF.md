# CL-AGN Spectral Change Classifier — Handoff (continue this work)

## Goal
Build a deep-learning network that, given TWO spectra of the SAME astronomical
object at two epochs, decides whether it underwent a changing-look AGN
transition (Type 1 ↔ Type 2 — broad emission lines appearing/disappearing).
Side project; the deliverable is a credible end-to-end pipeline + a portfolio
writeup, not a published paper.

## Workspace
- Code + data: `/Users/amir/Documents/Deep learning/cl-agn classifier/`
- Env: conda `astro_dl`, Python 3.10, Apple Silicon (MPS device).
- **The user runs all training/preprocessing scripts.** The assistant cannot
  run torch / astropy / scipy / pandas in its sandbox; use Read/Write/Edit
  on actual files and `python -m py_compile` for syntax checks. File deletion
  is not permitted from the sandbox; ask the user to `rm` if needed.

## Hard rules (do not violate)
1. **No spectral-decomposition algorithms** (no pyqsofit etc.). The network
   replaces that kind of fitting. Local side-band continuum subtraction
   inside `measure_oiii_flux` is the only continuum estimation allowed.
2. **Unlabeled data is the project's strength** — keep using it for SSL.
3. **Never build training pairs across different objects** — pairs must be
   same-object, two epochs.
4. **Never tune thresholds on the held-out CL-AGN test set.** The eval script
   reports a "post-hoc tuned-on-test" upper bound for reference, but the
   headline numbers are always at the train-time saved threshold.

## Architecture (v2, two stages)
- **Stage 1 (SSL)** — masked-autoencoder pretraining of a 2-channel encoder.
  Class: `architectures_v2.MaskedSpectraAutoencoder`. Currently trained on a
  pool of SDSS-DR7 + DESI (~44k spectra). The plan below adds DR16 + SDSS-V.
- **Stage 2 (Siamese)** — `SiameseChangeNet` (shared SSL encoder + symmetric
  sum/abs-diff/product head) fine-tuned on real same-object epoch pairs.
  **The encoder is FROZEN** (`encoder_freeze: true`); only the head is
  trained. This is the linear-probe regime that fits ~470 positives + 16k
  negatives without overfitting and without amplifying survey-correlated
  features.
- **Input convention**: `x` has shape `[B, 2, 4096]`:
  - ch0 = MAD-normalised flux (full continuum retained — v2 design)
  - ch1 = [OIII] 5007-normalised flux
  - Both arcsinh-compressed.
- **Grid**: rest-frame 3000–10400 Å, 4096 px. Pixels outside coverage are
  zero-filled (0.0 sentinel) and recovered as a per-pixel validity mask
  via `preprocessing_oiii.valid_from_flux`.

## Current status (training + held-out eval complete)

- ✅ **Stage 1 SSL trained** on `data/ssl_unified_dr7_desi_filtered.parquet`
  (~44k spectra). Best val MSE = 0.128. Checkpoint at
  `models/clagn_v2/ssl_encoder.pth`.
- ✅ **Stage 2 Siamese trained** with the design below. Best val F0.5 ≈
  0.85, precision ≈ 0.93, recall ≈ 0.55. Saved threshold = 0.80.
  Checkpoint at `models/clagn_v2/siamese_changenet.pth`.
- ✅ **Held-out evaluation on `data/clagn_test.pkl`** (50 pos + 350 neg):
  - **Overall**: F0.5=0.80, **precision=1.000**, recall=0.44, AUC=0.85, F1=0.61
    (TP/FP/TN/FN = 22/0/350/28)
  - **post-hoc tuned-on-test (leaky, informational only)**: thr=0.55,
    F0.5=0.91, precision=0.97, recall=0.72
  - **per-source breakdown is the headline finding**:
    - paper2 (DESI×DR16): precision=1.000, recall=**0.667** (20/30) → F0.5=0.91
    - lowz   (SDSS-V×DR16): precision=1.000, recall=**0.100** (2/20) → F0.5=0.36
    - **6.7× recall gap** between Paper-2 and lowz at the saved threshold.
  - **per-z TPR** (positives only) increases with z: 50% at z<0.22 vs 83%
    at z>0.69. Confounded with source (lowz only contributes at z≤0.4).

## The headline finding to address next

The lowz vs Paper-2 recall gap is real and has three plausible drivers,
in (probably) descending order of contribution:

1. **Training distribution imbalance.** Training pool is ~434 Paper-2
   positives vs ~34 lowz positives = 13:1. The head learned Paper-2-style
   transitions much better simply because it saw them more often.
2. **Survey OOD in the encoder.** The SSL encoder has only seen SDSS-DR7
   and DESI. It has **never** seen SDSS-V (the second epoch of lowz pairs)
   or DR16 (the first epoch of every Stage-2 pair, where BOSS/eBOSS plates
   ≥3523 weren't in DR7). Paper-2 has ≥1 in-distribution epoch (DESI);
   lowz has zero. Even with the encoder frozen, the OOD features it
   produces for SDSS-V are noisier than the in-distribution DESI features
   it produces for Paper-2.
3. **Paper-2 transitions may be intrinsically stronger.** Guo+ selected
   the most dramatic CL-AGN; lowz might be weaker changes. Unverified.

## Planned next experiments (Phase A + Phase B)

To address the OOD half of the gap, the encoder needs to see DR16 +
SDSS-V during SSL. Two paths, both fully scripted:

### Phase A — Full SSL retrain on DR7+DESI+DR16+SDSS-V
1. `python src/build_ssl_extension_parquet.py`
   — samples 10k DR16 + 10k SDSS-V from the 174k crossmatch (z-stratified,
   excludes held-out test objects), writes `data/ssl_dr16_sdssv_extension.parquet`
   (~17–19k surviving after `clean_dataset(min_snr=8.0)`).
2. `cp models/clagn_v2/ssl_encoder.pth models/clagn_v2/ssl_encoder_dr7desi.pth`
   — back up the current encoder (Phase B needs the original).
3. `python src/pretrain_ssl.py`
   — pools both parquets (~62k total), full SSL retrain from scratch.
   Overwrites `ssl_encoder.pth`.
4. `rm models/clagn_v2/pair_arrays_cache.npz models/clagn_v2/eval_pair_arrays_cache.npz`
   — stale caches use the old norm_stats.
5. `python src/train_siamese_v2.py && python src/eval_clagn_test.py`

### Phase B — Continual SSL with 50/50 replay (the "extension" experiment)
Demonstrates that adding a new survey doesn't require a full retrain.

```
python src/pretrain_ssl.py \
    --resume-from models/clagn_v2/ssl_encoder_dr7desi.pth \
    --replay \
    --lr 1e-4 \
    --num-epochs 20 \
    --output-ckpt models/clagn_v2/ssl_encoder_continual.pth
```

- `--resume-from`: loads encoder + decoder weights from the backup.
- `--replay`: WeightedRandomSampler with target 50% old (DR7+DESI) /
  50% new (DR16+SDSS-V) per batch — prevents catastrophic forgetting.
- Lower LR (1e-4 vs 3e-4) and fewer epochs (20 vs 50). ~2–3h total.
- Writes to a separate checkpoint so Phase A isn't overwritten.

Evaluation: temporarily point `paths.ssl_checkpoint` at
`ssl_encoder_continual.pth`, re-run Siamese + eval (with cache deleted).

### Three-way comparison (the writeup table)
| Model | SSL pool | Test F0.5 | paper2 recall | lowz recall |
|---|---|---|---|---|
| Current | DR7+DESI (44k) | 0.80 | 0.67 | 0.10 |
| Phase A | DR7+DESI+DR16+SDSS-V (~62k) | ? | ? | ? |
| Phase B | Continual on top of current | ? | ? | ? |

- A vs current: does adding the missing surveys close the lowz gap?
- B vs A: does continual match a full retrain at a fraction of the cost?

Both questions are first-class findings either way they go.

## Data pipeline — run order and what each script does

```
# One-time, data preparation
python src/build_paper2_master.py        # parse .mrt files -> paper2_master.pkl
python src/build_paper2_test_pickle.py   # split Paper-2 into ~434 train + ~30 test
python src/build_phase2_pickle.py        # merge pool: dr16_sdssv_phase2_train.pkl
                                         #            + clagn_test.pkl
python src/download_phase2_spectra.py    # fetches missing DR16 + SDSS-V FITS

# Optional one-time: SSL extension parquet (Phase A/B prerequisite)
python src/build_ssl_extension_parquet.py

# Per-experiment, after any data or SSL change
rm models/clagn_v2/pair_arrays_cache.npz models/clagn_v2/eval_pair_arrays_cache.npz
python src/pretrain_ssl.py               # Phase A; or with --resume-from for Phase B
python src/train_siamese_v2.py
python src/eval_clagn_test.py
```

### Pickle schemas (what each file contains)

| Pickle | Rows | Schema highlights | Purpose |
|---|---|---|---|
| `data/dr16-sdssv_crossmatch.pkl` | 174k (long, 2 rows / sdssid) | sdssid, ra, dec, z, zwarning, class, mjd, specname, survey ∈ {dr16, sdssv} | Source of expansion negatives + the DR16 / SDSS-V half of the SSL extension |
| `data/dr16_sdssv_crossmatch_lowz.pkl` | 4,288 (wide) | sdssid, z, ra, dec, mjd_*, specname_*, label (54 pos / 4234 neg) | Original z≤0.4 hand-curated training set; 54 real CL-AGN positives |
| `data/paper2_master.pkl` | 561 pairs | obj_position, ra, dec, z, transition, plate/fiber/mjd, targetid, specname_*, sdss/desi_on_disk | All Paper-2 (DESI×DR16) pairs after SDSS↔DESI 1″ join |
| `data/paper2_train_pairs.pkl` | ~520 pairs | trainer schema + obj_position, transition | Paper-2 training positives |
| `data/paper2_test_pairs.pkl` | ~30 pairs | same | Paper-2 held-out positives (30 SDSS objects, transition-stratified) |
| `data/dr16_sdssv_phase2_train.pkl` | ~16,716 | sdssid, ra, dec, z, mjd_*, specname_*, label | **Stage-2 training pickle** — 468 pos (34 lowz + 434 paper2) + 16,248 neg |
| `data/clagn_test.pkl` | 400 | trainer schema + `source` ∈ {lowz, paper2, phase2_neg} | **Held-out test** — 50 pos (20 lowz + 30 paper2) + 350 z-matched neg |

### Path convention (subdir-prefixed specnames)
Specnames in `dr16_sdssv_phase2_train.pkl` and `clagn_test.pkl` carry a
subdir prefix relative to `data/`:
- `dr16_sdssv_crossmatch/0001_0454_51908.fits` for SDSS-V × DR16 pairs
- `desi/clagn_desi_dr16_sample_paper2/sdss/344_4296_55499.fits`,
  `desi/clagn_desi_dr16_sample_paper2/desi/desi-spec-39627820967665156.fits`
  for Paper-2 pairs
With `paths.pair_spectra_dir: data`, the trainer's `_resolve` finds both
sources from one root.

## Stage-2 trainer details (`train_siamese_v2.py`)

- **Frozen encoder**: `encoder_freeze: true`. All encoder params have
  `requires_grad=False` and are excluded from the optimizer.
  `model.encoder.eval()` is held even inside the training loop (no
  dropout/BN updates).
- **WeightedRandomSampler**: `sampler_pos_rate: 0.2` → each batch is
  ~20% positives. Reasoning: at 2.8% positive rate with batch=64, a
  natural sample averages ~1.8 positives per batch — too few. 0.2 gives
  ~13 positives per batch, much better gradient signal, and the calibration
  drift vs the deployment prior (~1–3%) is manageable.
- **No synthetic positives**: `synthetic_prob: 0.0`. Real positives only.
  (Previously we considered synthetic-only training via
  `make_synthetic_change`; abandoned in favour of real labels once we had
  ~500 of them.)
- **F0.5 threshold sweep** (mirrors v1 `train_siamese.py` exactly):
  - Every epoch, sweep 19 thresholds in [0.05, 0.95]
  - Filter to recall ≥ `min_recall=0.10` AND fpr ≤ `max_fpr=0.01`
  - Fall back to argmax-F0.5 over all thresholds if nothing passes (prints `FALLBACK`)
  - Tie-break: `(fbeta, precision, -fpr, recall)` descending
- **Checkpoint selection**: same tuple comparison as v1. Checkpoint saves
  `best_threshold` and `best_threshold_metrics` for the eval script.
- **Focal loss**: `BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)` —
  unchanged from v1.

## Stage-2 eval (`eval_clagn_test.py`)

- Loads `models/clagn_v2/siamese_changenet.pth`, reads its `best_threshold`.
- Preprocesses `clagn_test.pkl` into a **separate** cache
  (`eval_pair_arrays_cache.npz`) so it doesn't clobber the training cache.
- Reports overall metrics (accuracy, precision, recall/sensitivity, NPV,
  specificity, FPR, FNR, F0.5, F1, AUC) at the saved threshold and the
  post-hoc tuned-on-test threshold.
- Per-source breakdown (paper2 / lowz / phase2_neg) — **the survey-pair
  diagnostic**.
- Per-z-bin TPR.
- PR curve + confusion matrix → `models/clagn_v2/eval_clagn_test.png`.
- Example pair plot (raw FITS + processed channel 0 for one TP and one TN)
  → `models/clagn_v2/eval_pair_examples.png`.
- JSON summary → `models/clagn_v2/eval_clagn_test.json`.

## Key files

**v2 pipeline (active)**
- `src/architectures_v2.py` — SpectraEncoder, SpectraDecoder,
  MaskedSpectraAutoencoder, SiameseChangeNet, apply_span_mask,
  load_encoder_into
- `src/datasets_v2.py` — SSLSpectraDataset (with `self.meta`),
  RealPairDataset, load_or_build_pair_arrays, split_indices, fits_to_flat,
  read_fits_flux_wave
- `src/preprocessing_oiii.py` — MASTER_GRID, continuum_subtract,
  mad_normalize, measure_oiii_flux, valid_from_flux,
  make_synthetic_change, suppress_broad_lines, save_/load_norm_stats
- `src/data_preprocessing.py` — build_unified_ssl_parquet, build_agn_catalog,
  process_single_spectrum, clean_dataset, morphological_continuum_subtraction
- `src/pretrain_ssl.py` — Stage 1 entry. **Patched** with `--resume-from`,
  `--replay` (50/50 sampler), `--lr`, `--num-epochs`, `--output-ckpt`.
- `src/train_siamese_v2.py` — Stage 2 entry. **Patched** with frozen-encoder
  support, WeightedRandomSampler, and the F0.5 threshold sweep + tuple
  tie-break.
- `src/eval_clagn_test.py` — held-out evaluation (overall + per-source +
  per-z + PR curve + example pair plots + JSON summary).
- `config_v2.yml` — paths, ssl, siamese, preprocessing sections. Now has
  `encoder_freeze`, `sampler_pos_rate`, `fbeta_beta`, `min_recall`,
  `max_fpr`, `clagn_test_pickle`/`clagn_test_spectra_dir`.
- `src/utils.py` — load_config
- `src/smoke_test.py` — t_encoder, t_masked_autoencoder, t_siamese, etc.

**New data-prep scripts (added this iteration)**
- `src/build_paper2_master.py` — parse the two `.mrt` tables, RA/Dec join,
  produce `data/paper2_master.pkl`.
- `src/build_paper2_test_pickle.py` — split Paper-2 by `obj_position` into
  train (~434) + test (~30), stratified by `transition`. Subdir-prefixed
  paths.
- `src/build_phase2_pickle.py` — pivot the 174k crossmatch to wide, apply
  cuts (z≤0.85, zwarning=0, QSO×QSO), drop Paper-2 overlap (RA/Dec 1″),
  drop lowz overlap (sdssid), z-match a 350-pair neg test, merge Paper-2
  training positives + lowz training positives, write
  `dr16_sdssv_phase2_train.pkl` + `clagn_test.pkl`.
- `src/download_phase2_spectra.py` — driver for `download_dr16` +
  `download_SDSS_file` in `download_spectra.py`. No-args mode reads paths
  from config_v2.yml and downloads both training and test pickles' FITS
  into `data/dr16_sdssv_crossmatch/`.
- `src/build_ssl_extension_parquet.py` — pull 10k DR16 + 10k SDSS-V from
  the 174k crossmatch (z-stratified, test set excluded), preprocess via
  the existing `process_single_spectrum`, write
  `data/ssl_dr16_sdssv_extension.parquet`. Phase A/B prerequisite.

**Patched existing scripts**
- `src/download_spectra.py` — `download_dr16` now zero-pads plate to 4
  digits in the URL (fixed a 404 bug on SDSS legacy plates < 1000).
  Recovered 97 missing Paper-2 SDSS spectra.

**Legacy / dead code (do not resurrect)**
- `src/architectures.py` — only `SpectraBlock`, `TransformerStage`,
  `PositionalEncoding1D`, and `BinaryFocalLossWithLogits` are reused.
- `src/Data_handler.py`, `src/train_siamese.py`, `src/test_siamese_new_data.py`,
  `src/model_interpertation.py`, `src/train_desi_backbone.py`.
- The v1 `train_siamese.py` had a fundamental cross-object-pairs bug; do
  NOT use it as a baseline. Its evaluation logic (F0.5 threshold sweep
  with tuple tie-break) has been ported into `train_siamese_v2.py`.

## What to avoid (lessons from this iteration)

- **Don't tune the threshold on `clagn_test.pkl`.** The eval script reports
  a "post-hoc tuned-on-test" upper bound — that number is data-leaky and
  always belongs in parentheses, never as the headline.
- **Don't forget to delete `pair_arrays_cache.npz` and
  `eval_pair_arrays_cache.npz`** when the training pickle, SSL encoder, or
  norm_stats change. The cache will silently keep stale preprocessed data.
- **Don't "fix" the lowz vs Paper-2 gap by simply lowering the saved
  threshold.** Moving the threshold raises both recalls but the *ratio*
  is what tells the survey story. The gap is a distribution + OOD problem,
  not a threshold problem.
- **Don't add spectral-decomposition libraries** (pyqsofit etc.).
- **Don't build cross-object training pairs.**
- **Don't try to run torch / astropy / scipy / pandas in the assistant
  sandbox** — only the user can. Use `python -m py_compile` for syntax
  checks.
- **Don't assume the SSL pool is "complete".** It currently only covers
  DR7 + DESI. DR16 and SDSS-V are out-of-distribution for the encoder and
  this matters for the lowz failure mode.

## Open questions for the next chat

1. **Phase A vs Phase B outcome.** Once both runs complete, the three-way
   comparison table is the centrepiece of the writeup. Both directions
   (A helps a lot / A doesn't help / B matches A / B doesn't match A)
   are publishable findings.
2. **If the lowz gap doesn't close after Phase A**, the next lever is
   per-source balanced positive sampling in `WeightedRandomSampler` —
   currently 20% pos / 80% neg per batch, but within positives the 13:1
   Paper-2:lowz ratio still stands. A small patch can balance positives
   across sources too.
3. **Per-source-per-z breakdown** in `eval_clagn_test.py` is not yet
   implemented (only overall per-z and per-source separately). Worth
   adding once Phase A results are in — disentangles z from source.
4. **`compare_ssl_checkpoints.py`** — a small helper that takes a list
   of SSL checkpoints, runs the Siamese + eval pipeline on each, and
   produces the comparison table directly. Was proposed but not written.

## VERY NEXT STEP

Run Phase A:

```bash
cd "/Users/amir/Documents/Deep learning/cl-agn classifier"

# 1. Build the SSL extension parquet (~10 min)
python src/build_ssl_extension_parquet.py

# 2. Back up the current encoder so Phase B has the original
cp models/clagn_v2/ssl_encoder.pth models/clagn_v2/ssl_encoder_dr7desi.pth

# 3. Full SSL retrain on the pooled parquet (~5-7h)
python src/pretrain_ssl.py

# 4. Clean stale caches
rm -f models/clagn_v2/pair_arrays_cache.npz models/clagn_v2/eval_pair_arrays_cache.npz

# 5. Re-run Stage 2 + eval
python src/train_siamese_v2.py
python src/eval_clagn_test.py
```

Watch the eval output's per-source table. If lowz recall climbs from
0.10 toward Paper-2's 0.67, Phase A worked. If it stays low, Phase A
didn't fix the gap — proceed to per-source balanced positive sampling
or skip directly to Phase B for the comparison.
