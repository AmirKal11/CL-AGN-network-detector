# HANDOFF — DESI × SDSS-V blind crossmatch run (inference, vetting, issues)

> Self-contained handover for continuing in a fresh chat. Covers the work done
> applying the trained CL-AGN detector to the **first DESI × SDSS-V crossmatch**,
> the problems found, the `predict.py` changes made, and the decision to pivot to
> SDSS-V × DR16. For the model/training background see `docs/HANDOFF.md` (v4) and
> the README/TECHNICAL writeups.

---

## 1. Project in one paragraph

Two-stage detector for **changing-look AGN (CL-AGN)**: given two spectra of the
*same* object at two epochs, decide whether it underwent a Type 1 ↔ Type 2
transition (broad emission lines appear/disappear). **Stage 1** = self-supervised
masked-autoencoder encoder on ~80k unlabeled spectra (decoder discarded).
**Stage 2** = frozen encoder + small Siamese head on real same-object pairs.
Input `[B,2,4096]` on a rest-frame grid 3000–10400 Å: **ch0** = MAD-normalised
flux (this checkpoint: continuum-subtracted variant), **ch1** = [O III]-5007
anchored flux; both arcsinh-compressed. **Hard rules:** no spectral-decomposition/
line-fitting in the network; same-object two-epoch pairs only; threshold is a
deployment parameter (recall-first ranker), never tuned on the held-out test.
Model in use: `models/continuum_subtracted_full_dr7/` (`channel1_scale = 0.00776`,
i.e. `1/median([O III] flux)` — confirmed not the 1.0 fallback, so ch1 is sound).

## 2. What this run is

Ran inference (`src/predict.py`) on the **first known DESI × SDSS-V crossmatch**:
**~78,940 usable pairs** (78,966 rows, 26 NaN prob). Output:
`results/predictions_sdssv_desi.csv` (and a channel-0-only variant, see §5).
**This survey pair is out-of-distribution:** the Stage-2 head was trained on
DR16×{SDSS-V|DESI}; it never saw DESI×SDSS-V pairs, and there are **no known
CL-AGN in this crossmatch** (genuinely new territory — no ground truth).

## 3. Headline problem: the score ranking is not trustworthy here (OOD)

- At threshold 0.5, **21.1k / 78k flagged positive (~27%)** — physically
  impossible (true CL-AGN rate ≪1%). The threshold is meaningless on this arm;
  the model is a ranker and 0.5 carries no calibrated meaning OOD.
- **Score distribution is bimodal.** A 2-component Gaussian mixture in logit
  space gives ~70% "stable" (p≈0.11) + ~30% "changed" (p≈0.77, ~23k objects).
  The high mode is **"changed-looking," NOT CL-AGN** — it's dominated by ordinary
  variability + artifacts + OOD inflation. CL-AGN are a tiny sub-population inside
  it, so **no threshold on p alone separates them**.
- **Redshift-driven OOD inflation.** Score correlates with redshift (rank corr
  ≈ +0.30). The sharp high-p spike (~1.3%, prob>0.95) is almost entirely at
  **z ≳ 0.4** (clean prob>0.95 fraction: 0.1% at z 0.2–0.4 → 1.5% at z 0.6–0.8).
  Higher z → less rest-frame coverage on the grid + lines near the window edge →
  encoder more OOD → spurious change. (Independently reproduces the v4 decision to
  cap the DESI arm at z<0.4.) **Do not blanket-discard high-z** — real turn-offs
  live there too; gate on quality instead.
- **Data-quality contaminants:** 26 NaN prob; 37 rows with z1=0; ~1% with
  |z1−z2| ≥ 0.05 (catastrophic-z / bad pairs) — the latter are ~12× enriched
  among prob>0.9. Pre-filter all three.

## 4. False-positive modes identified by visual vetting

The high scores are overwhelmingly **data-quality driven**, not physics:
1. **Low-SNR / dead-epoch fakes** — one epoch near-noise; huge `|e1−e2|`. (But
   note: a faint noisy epoch can also be the *real* faded state of a turn-off, so
   low SNR alone is ambiguous — needs the other epoch to judge.)
2. **Cosmic-ray / sky-line artifacts** — narrow single-epoch spikes at non-line
   wavelengths drive high scores. Need a despiking gate.
3. **[O III] mis-measurement / calibration inconsistency** — see §6.
4. **Ordinary variability** — Type 1 in *both* epochs, continuum up/down ~1.5×,
   no broad-line type change. Real change, but **not** CL-AGN.

## 5. Channel-0-only experiment (key negative result)

To test whether ch1/[O III] drives the inflation, forced the existing [O III]
fallback for all spectra (`OIII_SNR_MIN → inf` ⇒ `ch1 = ch0`, an in-distribution
mode the net saw in training). Output: `results/predictions_sdssv_desi_channel0.csv`.
**Result: inflation did NOT decrease — it slightly worsened** (changed-mode mass
29.5% → 35.8%; flags@0.5 29.0% → 31.3%; frac>0.9 4.8% → 6.2%). **Conclusion: the
inflation lives in channel-0 / the continuum representation and the OOD encoder,
NOT in the [O III] channel.** Tweaking channels or thresholds will not fix it.

## 6. The calibration / [O III] issue (important)

Cross-instrument flux scaling between DESI and SDSS-V is the crux of vetting.
- **Physics:** [O III] 5007 (narrow-line region) is constant between epochs, so
  the **ratio of [O III] line fluxes = the relative throughput/calibration
  factor** (van Groningen–Wanders scaling). Scale one epoch onto the other by that
  dimensionless ratio to remove throughput; residual continuum/broad-line change =
  the real signal.
- **Dimensional point (resolved):** dividing a flux *density* by an *integrated*
  [O III] flux gives Å⁻¹ (wrong). The correct operation is the dimensionless
  **line-flux ratio** as a multiplicative scale factor. The network's ch1 is fine
  because `channel1_scale = 1/median(F)` turns `madnorm/F` into the dimensionless
  ratio `madnorm × median(F)/F` (and MAD cancels → throughput-independent).
- **Vetting consistency check (developed this run):** the continuum (f5100) ratio
  and the [O III] ratio between epochs **should agree** (at z~0.5–0.7, [O III] and
  5100 are adjacent in observed wavelength → same throughput). When they disagree
  in magnitude or **direction**, the [O III] is mis-measured → object is
  **indeterminate**, not a candidate. Real well-calibrated objects show the two
  ratios agreeing; broken ones (e.g. [O III] ratio 5.4× vs continuum 1.26×, or
  opposite directions) are calibration/measurement failures, often sky-line
  contamination at the [O III] observed wavelength.
- **Architecture link:** the Siamese fusion `[e1+e2, |e1−e2|, e1·e2]` happens in
  the **512-d embedding space**; each spectrum is encoded *independently* (frozen
  SSL encoder); the model does **no inter-epoch pixel scaling** — it relies on each
  epoch's ch1 [O III] anchoring (global-median reference) to put both on a common
  scale before encoding. So a mis-measured / low-SNR [O III] corrupts ch1 → shifts
  that embedding → spurious `|e1−e2|` → confident false positive.

## 7. `predict.py` changes made this run (state on disk)

Plotting now produces a 3-panel diagnostic (raw / ÷own f5100 / SDSS-V scaled onto
DESI by [O III] ratio) and an interactive plotly slider version. Normalization
logic was corrected:
- **f5100 (each-by-own, dimensionless, → unity at 5100):**
  `f5100_ratio = 1/f5100_1` (SDSS mark), `desi_f5100_ratio = 1/f5100_2` (DESI mark).
  Both sliders carry an f5100 mark; set **both** to compare.
- **[O III] (inter-epoch line-flux ratio, SDSS scaled onto DESI):**
  `oiii_ratio = oiii_2/oiii_1` (SDSS mark); `desi_oiii_ratio = None` (DESI is the
  reference — no [O III] mark, but its slider is kept for the f5100 mark). Use
  single-sided: move the SDSS slider to its [O III] mark, leave DESI at ×1.
- **`_compute_oiii_flux` now multiplies by dλ** (`np.sum(...) * dlam`) → true
  integrated flux (consistent with the model's `measure_oiii_flux`; dλ differs
  SDSS vs DESI by ~2× because of log vs linear grids, so it matters).
- Title shows `f5100 norm: SDSS=×.., DESI=×..` and `[O III]: SDSS=×.. (onto DESI)`.

**Bugs flagged but NOT fixed (intentionally left):**
- `flux > 0` mask in `_normalize_at_5100` and `_compute_oiii_flux` biases
  faint-epoch continua **upward** (keeps only positive noise) → unreliable f5100
  on faint epochs. Fix: all-pixel / inverse-variance-weighted fit.
- Pre-clip in `_prep` (clip to p98+50%) **truncates the [O III] peak** before
  integration → underestimates [O III]. Fix: measure [O III] before clipping.
- (Optional) `np.trapezoid(y, x)` is cleaner than median-dλ on the log grid; model
  uses `sum × dλ` deliberately to avoid the numpy-2.0 `np.trapz` deprecation.

## 8. Result so far

After pre-filtering + visual + consistency vetting: **~5 "reliable" CL-AGN
candidates** from DESI×SDSS-V. Treat as a **lower bound, not a census**
(completeness unknown — no injection-recovery; OOD ranking unreliable). Most
inspected high-p objects were rejected (artifacts / calibration / ordinary
variability). Apply the strict CL-AGN bar to the 5: a **broad line must appear or
vanish** (type change), not just brighten/dim, with self-consistent calibration.

## 9. Decisions & next steps

**Decision: pivot to SDSS-V × DR16 (in-domain) as the primary track.** The model
is calibrated there (threshold meaningful), and both are SDSS-family spectrographs
so cross-instrument calibration issues should be **much milder** — verify this
explicitly; if clean, it confirms the DESI-arm problems are cross-instrument domain
shift (the project's thesis), a citable contrast.

Order of work:
1. **Lock down the 5 DESI×SDSS-V candidates** (don't mine that arm for more):
   ZTF/WISE photometric-variability cross-check (independent confirmation of a real
   continuum change), characterize turn-on vs turn-off, z, MJD baseline.
2. **Run SDSS-V × DR16 inference**, apply the gate (below), and compare its score
   distribution / calibration behavior to the DESI arm.
3. **Build the quality + consistency gate** (highest-leverage fix, no retraining):
   - per-epoch SNR **and** [O III] SNR; require comfortably above threshold;
   - signal-content check (channels carry structure above the noise floor);
   - multi-line consistency: [O III] vs [O II] 3727 / [N II] / [S II] / host
     continuum agree on one scale factor → trust; [O III] outlier → mis-measured;
   - despiking: reject narrow single-epoch spikes with no counterpart in the other
     epoch; optional spectrum-vs-photometry SED check.
   - **Gate on self-consistency / external calibration, NOT on novelty** (a real
     dramatic CL-AGN is also an outlier — don't reject it for being unusual).
4. **Durable fix for the DESI arm (only if pursuing discoveries there):** build
   type-verified DESI×SDSS-V **negatives** (same object, same external type both
   epochs — no positives needed) and **re-fit the frozen-encoder head** so it learns
   the DESI-vs-SDSS-V instrument baseline; this is what would de-inflate the OOD
   ranking. Optional learned detector: train a small head on synthetic
   miscalibration to flag/abstain on bad spectra.

## 10. Tooling / environment notes

- Code: `/Users/amir/Documents/Deep learning/cl-agn-classifier/` (env: conda
  `astro_dl`, Python 3.10, Apple Silicon/MPS). Data/checkpoints partly under the
  `... -Legacy version/data_v4/` tree per `config_v2.yml` / `paths_v4.py`.
- **Sandbox can't run torch/astropy/plotly/parquet — the user runs all training,
  preprocessing, inference, and plotting.** Code edits are validated with
  `python -m py_compile` only.
- **IDE↔disk sync issue is real** — reload files in the IDE before running so a
  stale buffer doesn't overwrite edits.
- Helper: a 2-component logit-space Gaussian-mixture script (with restarts) exists
  for analysing any probability column and plotting the stable/changed
  decomposition (used for §3, §5).
