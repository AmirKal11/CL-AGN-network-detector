# Model provenance

What each directory under `models/` is, where its numbers come from, and what
cannot be reconstructed from this repository.

All metrics below are **held-out test-set** values, read directly from each
directory's `eval_clagn_test.json` (735 pairs: 35 positive, 700 negative).

---

## The final model

**`models/continuum_subtracted_full_dr7`** — this is the deployed model and the
source of every headline number in the README and the deck.

| Quantity | Value | Source |
|---|---|---|
| PR-AUC (test) | 0.832 | `eval_clagn_test.json` → `pr_auc` |
| ROC-AUC (test) | 0.984 | `eval_clagn_test.json` → `auc` |
| Recall (test) | 88.6% — 31/35 | `saved_metrics.recall` |
| FPR (test) | 2.4% — 17/700 | `saved_metrics.fpr` |
| Deployed threshold | 0.5465555787086487 | `siamese_changenet.pth` → `best_threshold` |
| `channel1_scale` | 0.007764926149907767 | `norm_stats.json` |
| SSL pool size | 82,006 spectra | `norm_stats.json` → `n_spectra` |
| Best epoch | 31 (of 40) | `siamese_changenet.pth` → `epoch` |

### It was renamed

This directory was formerly **`models/fixed_OIII`**. The rename is recorded
nowhere in the code — the only surviving evidence is inside its own eval output:

```
models/continuum_subtracted_full_dr7/eval_clagn_test.json
  "checkpoint": "models/fixed_OIII/siamese_changenet.pth"
```

Re-running `eval_clagn_test.py` will overwrite that JSON and destroy the last
trace of the old name. `models/raw_continuum_dr7_capped` was likewise renamed
from `models/raw_continuum`.

### The name carries information the config does not

Continuum subtraction — the single best preprocessing choice — is **not a
`config_v2.yml` setting**. It is a CLI flag on a different script
(`data_preprocessing.py --no-subtract-continuum`) applied when the parquet is
built, so it is baked into the data file. Nothing inside the checkpoint or the
config records whether ch0 was continuum-subtracted. **The directory name is the
only record.**

---

## Directory index

| Directory | Status | PR-AUC | Recall | FPR | Threshold |
|---|---|---|---|---|---|
| `continuum_subtracted_full_dr7` | **FINAL** (was `fixed_OIII`) | 0.832 | 88.6% | 2.4% | 0.547 |
| `weighted_loss_per_Z` | rejected — see below | 0.875 | 71.4% | 0.3% | 0.657 |
| `sdssv_weighted` | ablation | 0.842 | 62.9% | 0.4% | 0.445 |
| `raw_continuum_dr7_capped` | ablation (was `raw_continuum`) | 0.812 | 80.0% | 2.6% | 0.565 |
| `raw_continuum_full_dr7` | ablation | 0.790 | 77.1% | 2.4% | 0.431 |

### Why `weighted_loss_per_Z` was rejected despite the best PR-AUC

It scores the highest test PR-AUC (0.875) and the lowest FPR (0.3%), but its
recall is 17 points worse (71.4% vs 88.6%) — it misses 10 of 35 real
transitions where the final model misses 4. The deployment goal is recall-first
under an inspection budget, not purity.

The deciding evidence was a two-component Gaussian mixture fit to scores on
real unlabelled SDSS-V×DR16 pairs: this run assigns 42% of objects to the
"changed" component at p ≈ 0.50 — physically implausible against expected
CL-AGN rates of a few percent — versus 13% at p ≈ 0.73 for the final model.
Better test metrics, worse separability on real data.

---

## Number collisions — do not mix these up

Two nearly identical numbers belong to different models and different splits:

| Number | What it is |
|---|---|
| **0.8775** | `val_pr_auc_mean` of the **final** model — *validation*, mean per-survey |
| **0.8750** | `pr_auc` of the **rejected** `weighted_loss_per_Z` — *test* |

The README's "0.832 → 0.875" comparison refers to the second. Always name the
split when quoting a 0.87-something figure.

The final model's per-survey validation PR-AUC breakdown is
`{desi: 0.963, sdssv: 0.792}` — the mean of those two is what selected the
checkpoint, not the test number.

---

## Threshold provenance — small-sample warning

`siamese_changenet.pth` → `best_threshold_metrics` records the SDSS-V
validation subset the operating point was fitted on:

```
tp: 12   fn: 1   fp: 8   tn: 183
recall 0.923   precision 0.600   fpr 0.0419
```

**That is 13 positive pairs and 191 negatives.** The deployed threshold of
0.547 rests on 13 positive examples; the 88.6% test recall rests on 35. Both
the operating point and the recall estimate are small-sample fragile — the
argument for shipping a ranked candidate list rather than a hard cut.

---

## What cannot be reconstructed from this repo

Neither checkpoint stores a config snapshot, and `config_v2.yml` was last
edited for the rejected `weighted_loss_per_Z` run.

**Recoverable from the checkpoints:**

| Field | Where |
|---|---|
| `channel1_scale`, threshold, `best_threshold_metrics` | `siamese_changenet.pth` |
| `selection_metric`, `op_survey`, `max_fpr` | `siamese_changenet.pth` |
| `select_surveys = [sdssv, dr16]` | `ssl_encoder.pth` |
| architecture (via `state_dict` shapes) | both |

**Recoverable from nothing:** `mask_ratio`, `min_span`, `max_span`, all
learning rates, batch sizes, `dropout`, `focal_alpha`/`focal_gamma`,
`sampler_pos_rate`, `source_balanced`, `val_frac`, the seed, and the
train/val split.

`min_recall` and `fbeta_beta` *are* in the checkpoint metadata, but they were
dead at training time too (see below) — their presence proves nothing about
how the model was selected.

### Stage 1 cannot currently be re-run at all

`survey_loss_weights` in `config_v2.yml` has **zero readers**. The per-survey
weighting that actually executes is the rejected per-z-bin scheme, hardcoded as
class constants at `datasets_v2.py:281-286`, applied unconditionally in
`_build_sample_weights()` and consumed by the SSL loss at `pretrain_ssl.py:414`.

Running `pretrain_ssl.py` today reproduces `weighted_loss_per_Z`, regardless of
the YAML. Turning it off requires editing `datasets_v2.py`.

---

## Dead configuration keys

Verified by grep against `src/`. These are set in `config_v2.yml` and read by
nothing that affects a result:

| Key | Status |
|---|---|
| `fbeta_beta` | read at `train_siamese_v2.py:394`, written to ckpt metadata only |
| `min_recall` | read at line 395, marked "informational", enforced nowhere |
| `select_survey_stage2` | 0 readers — `op_survey = "sdssv"` is a literal at line 405 |
| `survey_loss_weights` | 0 readers — see above |
| `decision_threshold` | unused by trainer; downstream `best_threshold` fallback only |

`max_fpr` is the only live threshold-selection knob.

### The threshold is not chosen by an F-beta sweep

`_threshold_sweep()` (`train_siamese_v2.py:159`) implements the F₂ rule the
config comments describe, and is **never called during training**. It is
diagnostic-only, invoked from `eval_clagn_test.py`.

The deployed threshold comes from `_recall_at_fpr()` (line 136): maximum recall
subject to FPR ≤ `max_fpr`, on the SDSS-V validation subset only (line 465).

Checkpoint selection is a separate decision: maximum **mean per-survey PR-AUC**,
each survey weighted equally (lines 459, 490).

---

## The `out_dir` footgun

The five `paths:` keys are overloaded:

- `pretrain_ssl.py`, `train_siamese_v2.py` → **write** destination
- `eval_clagn_test.py:261`, `gradcam_pairs.py:201` → **read** location

There is no separate "which model to evaluate" key. `config_v2.yml` now points
these at the final model so eval and Grad-CAM load the right checkpoints —
which means **running either training script as-is will overwrite the final
model.** Point all five at a new directory before training.

The clean fix (not applied, to avoid pre-interview churn) is to split the keys
into `model_dir` (read) and `out_dir` (write) across the four scripts.
