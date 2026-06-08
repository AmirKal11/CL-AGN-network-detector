# HANDOFF — CL-AGN Classifier (v4 complete: confound-free, recall-first)

> Current rolling handoff. Supersedes the earlier v3 content and the parallel
> `HANDOFF_v4.md` (which can be deleted). v4 is built, trained, and evaluated.

## TL;DR — where we are

A two-stage detector for **changing-look AGN (CL-AGN)**: given two spectra of the *same*
object at two epochs, decide whether it underwent a Type 1 ↔ Type 2 transition.
Stage 1 = self-supervised masked-autoencoder encoder on unlabeled spectra; Stage 2 =
frozen-encoder Siamese head on real same-object pairs.

**v4 is finished.** It rebuilt the entire labeled set to be confound-free, retyped every
survey from external spAll `SUBCLASS` / DESI `AGN_TYPE` (no fitting), moved all built
artifacts to `data_v4/`, added DR7 type-1+type-2 to the encoder, and reframed the task as a
**recall-first candidate ranker**. SSL → Siamese → eval have all run.

**Headline result (held-out test, 35 CL-AGN + 700 controls):** SDSS-V (`lowz`) recall
**0.05 → 0.75** at a 3% false-positive rate (≈0.95 at a looser threshold), **PR-AUC 0.83**,
**top-150 ranked candidates recover 97%** of all positives, per-z recall flat (no z/survey
shortcut), type-2 negatives not false-flagged.

> **What's left (optional):** set the deployment threshold to the real N/B budget; the small
> refinements in "Open questions". The clean portfolio repo README (§7) is already updated
> with the v4 story + figures.

---

## PRIORITY
Science target = the **SDSS-V × DR16** arm (lowz positives + SDSS-V negatives). **DESI is
auxiliary** — included only for its large CL-AGN positive count, and capped to z<0.4 in the
test. Judge the model on the SDSS-V arm. The model is a **candidate ranker**: the threshold
is a deployment parameter (≈B/N), never a fixed purity-first cut.

## Directories
- **Code (active):** `/Users/amir/Documents/Deep learning/cl-agn classifier -Legacy version/`
- **Raw inputs + existing spectra (read-only):** `data/`
- **All v4-built artifacts + new spectra downloads:** `data_v4/`
- **spAll catalogs:** `/Users/amir/Documents/Msc/Thesis/data/` (`spAll-v5_13_0.fits` DR16,
  `spAll-lite-v6_2_1-epoch.fits` SDSS-V)
- **Clean portfolio repo (README updated with v4 §7 + figures):**
  `/Users/amir/Documents/Deep learning/cl-agn-classifier/` (GitHub: AmirKal11/CL-AGN-network-detector)
- Paths centralized in `src/paths_v4.py`; spectra resolve by searching `data_v4/` then `data/`.

Env: conda `astro_dl`, Python 3.10, Apple Silicon (MPS).

## Hard rules (do not violate)
1. **No spectral decomposition / line-fitting.** Type labels are EXTERNAL (spAll `SUBCLASS`,
   DESI `AGN_TYPE`), never our own fitting. [OIII] side-band subtraction is the only continuum step.
2. **Same-object, two-epoch pairs only.**
3. **Never tune the decision threshold on the held-out test** (eval's "tuned-on-test" line is leaky/diagnostic).
4. **Exclude EVERY held-out test object from the SSL pool** (all parquets, both epochs).
5. **Every labeled pair is SDSS-DR16 (BOSS/eBOSS) × {SDSS-V | DESI}. DR7 is encoder-only,
   never in a pair** (avoids a DR7-vs-BOSS instrument shortcut).

## Architecture (recap)
Rest-frame grid 3000–10400 Å / 4096 px; out-of-coverage zero-filled + per-pixel mask. Input
`[B,2,4096]`: ch0 MAD-normalized flux, ch1 [OIII]-5007-normalized (MAD fallback), both
arcsinh. Encoder = multiscale 1-D CNN + transformer → 512-d, **frozen** in Stage 2. Siamese
fusion symmetric `[e1+e2,|e1−e2|,e1·e2]` (epoch-order invariant). **Recall-first:** focal loss
trains the head; **checkpoint selected on the mean of per-survey PR-AUC** (SDSS-V & DESI equal
weight); **deployment threshold = max recall at FPR≤`max_fpr`(≈B/N=0.05) on SDSS-V**.
`sampler_pos_rate 0.3`, `source_balanced true`. Encoder checkpoint selected on SDSS-V+DR16
reconstruction val. (`_pr_auc`/`_recall_at_fpr` live in `train_siamese_v2.py`.)

## What v4 changed (vs v2/v3)
1. **Survey-pairing confound removed.** v2/v3 had ~no DESI negatives → "DESI present →
   positive" shortcut. v4 builds type-verified negatives for both arms at an equal neg:pos
   ratio per survey (`NEG_POS_RATIO=20`), so survey ⊥ label.
2. **Instrument confound removed.** Negatives are DR16/BOSS × {SDSS-V|DESI} (matching the
   positives), not DR7-era. DR7 → encoder only.
3. **Redshift confound removed.** Same-object SDSS×DESI non-CL-AGN exist only at z<0.4 →
   DESI **test** capped to z<0.4; DESI **train** positives kept (SDSS-V arm carries high-z).
4. **External-catalog typing** (`extract_dr16_spall.py`, `extract_sdssv_spall.py`).
5. **Recall-first reframe** (F2 → mean per-survey PR-AUC selection + recall@budget threshold).
6. **Type-2 encoder coverage** (`ssl_type2` DR16+SDSS-V + `ssl_dr7_types` DR7 t1+t2; ~9k → ~27k type-2).
7. **`data_v4/` reorg + `paths_v4.py` + leak guards** (SSL builders refuse to run without the
   test; train⟂test 1″ object-disjoint guard in `build_phase2_pickle`).

## Current data state (built & verified)
**Stage-2 train** `data_v4/dr16_sdssv_phase2_train.pkl` — 3,969 pairs: 515 pos (DESI 434,
SDSS-V 81) + 3,454 neg (DESI 2,027, SDSS-V 1,427); pos rate ~13%; negatives type-1 3,148 /
type-2 306 (DESI 292, SDSS-V 14). DESI train positives uncapped; DESI negs z<0.4.

**Held-out test** `data_v4/clagn_test.pkl` — 735 pairs: 35 pos (lowz 20 full-range, paper2 15
at z<0.4) + 700 neg (sdssv_neg 400, desi_neg 300); test type-2 negs 65.

**SSL pool** (`config_v2.yml → ssl_parquets`, 4 parquets): `ssl_unified_dr7capped_desi`
(DR7+DESI) · `ssl_dr16_sdssv_extension` (DR16+SDSS-V QSO/type-1) · `ssl_type2` (DR16+SDSS-V
type-2) · `ssl_dr7_types` (DR7 type-1+type-2, ~36k). ~80k total, ~27k type-2.

Catalog: `cl_agn_list_dr16.pkl` = 116 CL-AGN + 88 EVQ (EVQ excluded everywhere). lowz pos 101;
paper2 pos 464.

## Results (held-out test)
| Model / setting | SDSS-V (`lowz`) recall | DESI (`paper2`) recall | PR-AUC | note |
|---|---|---|---|---|
| v2 (old test) | 0.20 | 0.73 | — | leaky/confounded |
| v3 (new test) | 0.05 | 0.63 | — | survey shortcut |
| **v4 @ thr 0.46** | **0.75** (15/20) | 0.87 (z<0.4) | **0.83** | FPR 3% (SDSS-V) |
| **v4, top-150 inspected** | **0.95** (19/20) | 1.00 (15/15) | 0.83 | 97% of all pos |

Per-z recall flat (0.75–1.0 across bins → no z-shortcut); false positives do NOT cluster at
the top of the ranking → type-2 negatives not false-flagged (the DR7-type-2 encoder boost
worked; synthetic type-2 negatives NOT needed). Each version used its own rebuilt test, so the
SDSS-V jump is the confound-fix + recall-first signal, not a like-for-like leaderboard.
Figures: `data_v4/analysis/{siamese,test}_redshift.png`; clean-repo `figures/10_v4_results.png`.

## Run order (all done; here for reproduction)
```
extract_dr16_spall.py + extract_sdssv_spall.py            # spAll -> *_spall_obs.pkl
build_sdssv_negatives.py + build_desi_negatives.py        # DR16×{SDSS-V|DESI}, type-verified
download_v4_negatives.py                                  # ~7.5k DR16 + ~5.5k SDSS-V FITS
build_phase2_pickle.py                                    # 20:1 carve, DESI test z<0.4, leak guard
build_ssl_dr7_parquet.py                                  # DR7 type1+type2 (on disk)
build_ssl_type2_parquet.py  (+download type-2 FITS, re-run)
prepare_clean_ssl_dr7desi.py + build_ssl_extension_parquet.py
pretrain_ssl.py -> train_siamese_v2.py -> eval_clagn_test.py
analyze_v4_data.py                                        # redshift histograms per phase
```

## Key files (v4)
Extracts: `extract_dr16_spall.py`, `extract_sdssv_spall.py`. Negatives:
`build_sdssv_negatives.py`, `build_desi_negatives.py`. Downloads: `download_v4_negatives.py`.
Pairs/test: `build_phase2_pickle.py`. SSL: `prepare_clean_ssl_dr7desi.py`,
`build_ssl_extension_parquet.py`, `build_ssl_type2_parquet.py`, `build_ssl_dr7_parquet.py`.
Train/eval: `pretrain_ssl.py`, `train_siamese_v2.py`, `eval_clagn_test.py`. Infra:
`paths_v4.py`, `datasets_v2.py`, `config_v2.yml`. Analysis: `analyze_v4_data.py`.
Docs: `PIPELINE.md`, `DATA_INVENTORY.md`. **Deprecated:** `extract_sdssv_observations.py`,
`select_sdssv_type2_spall.py`, `build_ssl_sdssv_type2_parquet.py`, `build_neg_expansion_pickle.py`,
legacy `type{1,2}_candidates` as pairs, `merged.pkl`.

## Workflow constraints / gotchas
- **User runs all torch/astropy/downloads.** Sandbox can't run those and **can't read parquet**
  (no pyarrow) — `analyze_v4_data.py` skips the SSL figure there; run it on your machine.
- **IDE↔disk sync issue is real:** it silently reverted the `train_siamese_v2.py` PR-AUC edits
  once (a stale editor buffer overwrote them). Reload/close files in the IDE or run from
  terminal; re-check disk if results look like an old version.
- **Run order matters:** download neg FITS → `build_phase2_pickle` → SSL builders → train.
  SSL builders fail loudly if `clagn_test.pkl` is missing.
- **SSL pretraining ~2× longer** (pool doubled with DR7 typed).
- Minor: DR7-QSO double-count between `ssl_unified_dr7capped_desi` and `ssl_dr7_types` (aligned
  with the uncapped choice; dedup by making `prepare_clean` DESI-only if wanted).

## Open questions / next levers
1. **Deployment threshold:** set FPR ≈ B/N (B≤1000 over N tens of thousands → ~0.02–0.05) on
   new data; 0.46 gives 0.75 recall, looser → ~0.95. Pick on the budget, never on the test.
2. **The one hard-missed lowz** (prob 0.076, z 0.264) — likely low-SNR/subtle; not worth chasing.
3. **Type-2 negatives** stay scarce (306 train) but the encoder handles type-2 — revisit only
   if a future eval shows type-2 false positives.
4. **Cleanups:** DR7-QSO dedup; group-aware Stage-2 train/val split (paper2 multi-exposure
   objects can span train/val — affects val selection only, not the test); optional DESI-val
   z<0.4 cap (needs z added to the pair arrays + cache rebuild) to fully de-confound selection.
5. Copy `clagn_v4` checkpoints into the clean repo `models/` if you want the README reference to resolve.
