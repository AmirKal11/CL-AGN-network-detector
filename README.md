# Detecting Changing-Look AGN with Self-Supervised Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![Self-Supervised](https://img.shields.io/badge/Self--Supervised-Masked%20Autoencoder-5C2D91)
![Siamese](https://img.shields.io/badge/Architecture-Siamese%20%2B%20Transformer-0A7E8C)
![Domain](https://img.shields.io/badge/Domain-Astrophysics%20%2F%20Spectroscopy-1f6feb)

A two-stage deep-learning pipeline that detects a rare astrophysical **state-transition** from pairs of telescope spectra from different surveys — built around **self-supervised pretraining, used for spectra reconstruction**, a **frozen-encoder Siamese head**, and a dynamic per-survey evaluation to detect shortcut learning / data imbalance. Trained on ~85k spectra and reached **PR-AUC 0.832** and **ROC-AUC 0.984**.



---

## Scientific introduction and background

Some galaxies' supermassive black holes visibly **change state over just a few years** — a *changing-look AGN* (CL-AGN). These events are rare (currently estimated to be around 1% to 5% of AGN in samples) and scientifically valuable. We can identify one by comparing two spectra (intensity vs. wavelength) of the **same object** taken years apart: emission features appear or vanish. This is called a transition between Type 1 to Type 2 AGN (or vice versa). They are traditionally confirmed by manually fitting the spectra with different existing models and comparing the derived properties. **The goal of this project: train a neural network to flag the transition directly from a pair of spectra, replacing the fitting step which is time consuming and error-prone.**

As a machine-learning problem, that's **binary change-detection on pairs (static vs cl-agn)** under three conditions that make it genuinely hard:

- **Heavy class imbalance** — only a few hundred confirmed positives, against tens of thousands of negatives.
- **Heavy domain shift** — data comes from four different instruments/surveys, each calibrated differently and introducing unique artifacts and noise patterns.
- **A self-imposed hard constraint** — no line-fitting / spectral-decomposition allowed anywhere; the network has to *learn* the physics that fitting would otherwise hand it.

---



## Repository layout

```
src/
  pretrain_ssl.py           # Stage 1 — self-supervised pretraining
  train_siamese_v2.py       # Stage 2 — frozen-encoder Siamese, recall-first selection
  eval_clagn_test.py        # held-out eval: per-source / per-redshift / per-object ranking
  gradcam_pairs.py          # input-gradient visualisation: TP/FP/TN/FN pair plots
  plot_ssl_reconstruction.py# SSL reconstruction diagnostic (both channels)
  architectures_v2.py       # SpectraEncoder, MaskedSpectraAutoencoder, SiameseChangeNet
  datasets_v2.py            # SSL + real-pair datasets, pair-array cache
  preprocessing_oiii.py     # 2-channel representation, line anchor, masking, rest-frame grid
  data_preprocessing.py     # FITS → arrays, sky-line removal, SNR cut, continuum
  (v1 backbone — see TECHNICAL.md §3)
config_v2.yml               # pipeline config (paths + hyperparameters)
models/fixed_OIII/          # current best checkpoint + eval artifacts + interpretability plots
docs/                       # data inventory + project handoff
TECHNICAL.md                # full methods note (physics, experiments, ablations, references)
```




## Headline results

| Metric | Value |
|---|---|
| **PR-AUC** | **0.832** |
| **ROC-AUC** | **0.984** |
| **Recall** at operating threshold | **88.6%** (31 / 35 confirmed CL-AGN) |
| **FPR** at operating threshold | **2.4%** (17 / 700 non-CL-AGN flagged) |

Threshold (0.547) was selected on the validation set by maximising **F₂** subject to FPR ≤ 5%, then applied to the held-out test set without modification. This allowed us to find as many positive targets as possible while maintaining a low false-positive rate, which can be later manually inspected.

<p align="center">
  <img src="models/continuum_subtracted_full_DR7/eval_clagn_test.png" alt="PR curve and confusion matrix — fixed OIII model" width="800"/>
</p>

A separate architecture validation uses the same encoder backbone trained on a supervised Type-1/Type-2 classification task. This network scores **99.5% accuracy** on full spectra, but collapses to **39%** (below the majority-class baseline) when emission lines are masked out — confirming the architecture is capable of learning physically meaningful spectral features rather than calibration artifacts. While this classifier was not used in the final pipeline, it validates that the encoder architecture has the capacity to learn the right signal.

<p align="center">
  <img src="figures/AGN_classifier_cm_unmasked.png" alt="Type 1/2 classifier — unmasked (99.5% accuracy)" width="45%"/>
  &nbsp;&nbsp;
  <img src="figures/AGN_classifier_cm_masked.png" alt="Type 1/2 classifier — emission lines masked (39% accuracy)" width="45%"/>
</p>
<p align="center"><em>Left: full spectra — near-perfect classification. Right: emission lines masked — collapses below majority-class baseline, confirming the architecture learns emission line features, not calibration artifacts.</em></p>

### Data representation & training

Each spectrum pair is converted to a **2-channel, 4096-pixel rest-frame array** before any learning:

- **Channel 0** — continuum-subtracted flux, robustly normalised by the median absolute deviation (MAD). This isolates the emission line shapes, making them comparable across objects and epochs regardless of flux calibration differences between instruments.
- **Channel 1** — The same original flux is divided by the amplitude of the [O III] 5007 Å  emission line, measured on the raw flux before MAD normalisation. Because [O III] is assumed to remain constant during CL-AGN transitions, this channel normalises different flux calibration between different instruments.

Both channels are arcsinh-compressed to handle the dynamic range of emission-line spikes.

The **Siamese head** is trained with **binary focal loss** (α = 0.5, γ = 2) to down-weight the large number of easy negatives in the heavily imbalanced dataset (≈ 1:7 positive ratio after oversampling). The batch sampler targets **30% positives per batch** (oversampled), and the AdamW head learning rate is 1 × 10⁻³ over 40 epochs with cosine annealing. The encoder is frozen throughout Stage 2, so only the ≈ 500k-parameter MLP head is updated. Checkpoint selection uses **F₂** (which weights recall twice as heavily as precision) — matching the deployment goal of surfacing a short candidate list for human inspection rather than maximising purity.

### Ablations

**Continuum subtraction in ch0.** Two runs were compared — identical architecture, data, and hyperparameters, differing only in whether the smooth continuum was subtracted from ch0 before MAD normalisation:

| Configuration | Recall | FPR | PR-AUC |
|---|---|---|---|
| Continuum subtracted + MAD (full DR7) | **88.6%** (31/35) | 2.4% | **0.832** |
| Raw flux + MAD only (full DR7) | 80.0% (28/35) | 2.6% | 0.812 |
| Continuum subtracted + SDSS-V weighted SSL loss (full DR7) | TBD | TBD | TBD |

Continuum subtraction isolates emission line shapes from the slowly-varying flux baseline, giving the encoder a cleaner signal to reconstruct and the Siamese head a sharper change feature to detect. The 8-point recall difference confirms it is a meaningful preprocessing choice for this task.

---

## Technical approach

![Two-stage architecture: SSL pretraining, frozen encoder, Siamese change head](figures/architecture.svg)

**Stage 1 — self-supervised encoder.** Random spans of unlabeled spectra are masked and reconstructed (a 1-D masked autoencoder). Label-free, so there's no class to shortcut on. The decoder is discarded; the **512-dim encoder** is kept.

**Stage 2 — frozen-encoder Siamese head.** The encoder is frozen and applied to both epochs; only a small MLP change-head is trained on real same-object pairs. Freezing is what lets a few hundred positives be fit without overfitting or amplifying instrument-correlated features.

**Input representation.** Each spectrum becomes a 2-channel, 4096-px rest-frame array: a robustly-normalized flux channel plus a channel anchored to a physically *constant* emission line (so a real cross-epoch change survives normalization instead of being divided away). Both arcsinh-compressed.

### Why self-supervised pretraining?

The central bottleneck is label scarcity: a few hundred confirmed CL-AGN against tens of thousands of negatives, with no way to cheaply generate more positives — these are rare real events. Even Type1 / Type2 labels are scarce and depend on catalogues created by other researchers.
 The solution is to separate *representation learning* from *classification*: a masked autoencoder pretrained on ~80k unlabeled spectra learns a general spectral encoder without ever seeing a label, then a small Siamese head trained on the scarce labeled pairs handles the actual detection. The encoder learns from the data that's abundant; the classifier only needs to learn from what's rare. 

## Repository layout

```
src/
  predict.py                # ← inference on new data (see below)
  pretrain_ssl.py           # Stage 1 — self-supervised pretraining
  train_siamese_v2.py       # Stage 2 — frozen-encoder Siamese, recall-first selection
  eval_clagn_test.py        # held-out eval: per-source / per-redshift / per-object ranking
  gradcam_pairs.py          # input-gradient visualisation: TP/FP/TN/FN pair plots
  plot_ssl_reconstruction.py# SSL reconstruction diagnostic (both channels)
  architectures_v2.py       # SpectraEncoder, MaskedSpectraAutoencoder, SiameseChangeNet
  datasets_v2.py            # SSL + real-pair datasets, pair-array cache
  preprocessing_oiii.py     # 2-channel representation, line anchor, masking, rest-frame grid
  data_preprocessing.py     # FITS → arrays, sky-line removal, SNR cut, continuum
  (v1 backbone — see TECHNICAL.md §3)
config_v2.yml               # pipeline config (paths + hyperparameters)
models/
  continuum_subtracted_full_dr7/  # best model (PR-AUC 0.832) — checkpoints + eval artifacts
  raw_continuum_full_dr7/         # ablation: no continuum subtraction (PR-AUC 0.812)
  raw_continuum_dr7_capped/       # ablation: capped DR7 SSL pool
docs/                       # data inventory + project handoff
TECHNICAL.md                # full methods note (physics, experiments, ablations, references)
```

The survey-specific catalog-building and download code is not included; how the training inputs were constructed is documented in [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md).

### Running inference on new data

Prepare a CSV with one row per same-object spectrum pair:

| `file1` | `file2` | `ra` | `dec` | `z` |
|---|---|---|---|---|
| spec-epoch1.fits | spec-epoch2.fits | 185.3 | 22.1 | 0.21 |

`file1`/`file2` are FITS basenames (or paths relative to `--spectra-dir`). `ra`, `dec`, `z` are optional — all input columns pass through to the output. Per-epoch redshifts can be given as `z1`/`z2`; if absent the FITS header is used as fallback.

```bash
python src/predict.py \
    --spectra-dir  data/spectra/ \
    --pairs-csv    data/my_pairs.csv \
    --output       results/predictions.csv
```

The output CSV adds `prob` (P(CL-AGN)) and `label` (1 if `prob ≥ 0.547`) to every input row, sorted by probability descending. Each unique FITS file is read only once regardless of how many pairs it appears in, keeping runtime practical for tens of thousands of pairs.

### Training from scratch

```bash
python src/pretrain_ssl.py                    # Stage 1
python src/train_siamese_v2.py               # Stage 2
python src/eval_clagn_test.py                # held-out evaluation
python src/gradcam_pairs.py --config config_v2.yml   # interpretability plots
```

---

## Example outputs

**SSL reconstruction (Stage 1).** The masked autoencoder learns to reconstruct randomly-masked spans — shown here for both input channels (MAD-normalised flux and OIII-anchored amplitude):

<p align="center">
  <img src="models/continuum_subtracted_full_dr7/ssl_reconstruction_ch1.png" alt="SSL reconstruction — channel 0" width="700"/>
</p>

**Siamese training curve (Stage 2):**

<p align="center">
  <img src="models/continuum_subtracted_full_dr7/siamese_loss_curve.png" alt="Siamese training / validation loss" width="500"/>
</p>

**Input-gradient interpretability.** Each spectrum is coloured by the signed gradient of the CL-AGN logit with respect to the flux at that wavelength. Red regions pushed the prediction toward CL-AGN; blue pushed against it. Example true positive and false positive — in the FP case the model focuses on Hα at the red edge, where coverage drops and broad-wing structure is hard to rule out:

<p align="center">
  <img src="models/continuum_subtracted_full_dr7/gradcam_tp_0403.png" alt="Gradient map — true positive" width="700"/>
</p>
<p align="center">
  <img src="models/continuum_subtracted_full_dr7/gradcam_fp_0489.png" alt="Gradient map — false positive (sdssv_neg, ambiguous label)" width="700"/>
</p>

---
