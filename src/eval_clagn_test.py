"""
eval_clagn_test.py
==================
Held-out evaluation for the Stage-2 SiameseChangeNet.

Loads:
    - The best Siamese checkpoint produced by train_siamese_v2.py
      (carries best_threshold + best_threshold_metrics).
    - data/clagn_test.pkl -- ~50 positives + ~350 z-matched negatives,
      with a `source` column distinguishing paper2 / lowz / phase2_neg.

Reports:
    - Overall F0.5 / precision / recall / FPR / AUC / TP-FP-TN-FN at the
      training-time saved threshold.
    - Full threshold sweep on the held-out test, with the gate-+-tie-break
      logic applied for reference -- this is the post-hoc "optimum if we
      had tuned on test" number, useful for upper-bounding the gap.
    - **Per-source breakdown** (paper2 vs lowz vs phase2_neg): surfaces
      the survey-pair confound. If Paper-2 TPR >> lowz TPR, the model
      probably shortcuts on second-epoch survey identity.
    - **Per-z-bin TPR**: shows whether performance degrades at high z
      (Paper-2 is z<=0.85; phase-2 expansion is sparse below z<0.3).
    - Confusion matrix + PR curve plot at models/clagn_v2/eval_clagn_test.png
    - JSON summary at models/clagn_v2/eval_clagn_test.json

Run:
    python src/eval_clagn_test.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import load_config
from datasets_v2 import load_or_build_pair_arrays, RealPairDataset, read_fits_flux_wave
from architectures_v2 import SiameseChangeNet
from preprocessing_oiii import load_norm_stats, MASTER_GRID
from pretrain_ssl import pick_device
from train_siamese_v2 import _threshold_sweep, _auc


def _metrics_at_threshold(probs: np.ndarray,
                          labels: np.ndarray,
                          threshold: float,
                          beta: float = 0.5) -> dict:
    """
    Full set of binary-classification metrics at a single threshold.

    Returns: threshold, tp/fp/tn/fn, precision, recall (=TPR/sensitivity),
    specificity (=TNR), npv, accuracy, fpr, fnr, fbeta, f1.
    """
    labels = labels.astype(np.int64)
    preds = (probs >= float(threshold)).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n = max(tp + fp + tn + fn, 1)
    n_pos = max(tp + fn, 1)
    n_neg = max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / n_pos                       # recall / sensitivity / TPR
    spec = tn / n_neg                       # specificity / TNR
    npv  = tn / max(tn + fn, 1)
    acc  = (tp + tn) / n
    fpr  = fp / n_neg
    fnr  = fn / n_pos
    denom = (beta * beta * prec) + rec
    fbeta = ((1.0 + beta * beta) * prec * rec / denom) if denom > 0 else 0.0
    f1_denom = prec + rec
    f1 = (2.0 * prec * rec / f1_denom) if f1_denom > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_pos": int(tp + fn), "n_neg": int(fp + tn), "n": n,
        "precision":    float(prec),
        "recall":       float(rec),
        "specificity":  float(spec),
        "npv":          float(npv),
        "accuracy":     float(acc),
        "fpr":          float(fpr),
        "fnr":          float(fnr),
        "fbeta":        float(fbeta),
        "f1":           float(f1),
    }


def _pr_curve(probs: np.ndarray, labels: np.ndarray):
    """Sklearn precision-recall + AUC if available, otherwise sweep."""
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        prec, rec, thr = precision_recall_curve(labels, probs)
        ap = float(average_precision_score(labels, probs))
        return prec, rec, thr, ap
    except Exception:
        return None, None, None, float("nan")


def _resolve_path(spectra_dir: str, name) -> str:
    p = str(name)
    return p if os.path.isabs(p) else os.path.join(spectra_dir, p)


def _pick_examples(probs, labels, saved_threshold, df_aligned):
    """
    Pick one positive and one negative example for visualisation.

    Prefer correct predictions (TP and TN) so the figure shows the
    representative behaviour. Falls back to any positive / negative if
    no correct example exists.
    """
    preds = (probs >= float(saved_threshold)).astype(np.int64)
    pos_mask = labels == 1
    neg_mask = labels == 0

    # TP candidates -- prefer the most confident one
    tp_idx_arr = np.where(pos_mask & (preds == 1))[0]
    if len(tp_idx_arr) > 0:
        pos_idx = int(tp_idx_arr[np.argmax(probs[tp_idx_arr])])
        pos_label = "TP"
    else:
        # any positive
        cand = np.where(pos_mask)[0]
        pos_idx = int(cand[np.argmax(probs[cand])]) if len(cand) else -1
        pos_label = "FN" if pos_idx >= 0 else "n/a"

    # TN candidates -- prefer the most confidently-negative
    tn_idx_arr = np.where(neg_mask & (preds == 0))[0]
    if len(tn_idx_arr) > 0:
        neg_idx = int(tn_idx_arr[np.argmin(probs[tn_idx_arr])])
        neg_label = "TN"
    else:
        cand = np.where(neg_mask)[0]
        neg_idx = int(cand[np.argmin(probs[cand])]) if len(cand) else -1
        neg_label = "FP" if neg_idx >= 0 else "n/a"

    return (pos_idx, pos_label), (neg_idx, neg_label)


def _plot_example_pair(ax_raw, ax_proc, arr_i, df_row, arrays, spectra_dir,
                       title_prefix, prob, threshold):
    """
    Render two panels for one example pair:
        ax_raw  -- raw observed-frame flux for both epochs
        ax_proc -- processed (MAD-normalised, rest-frame) flux on the
                   master grid for both epochs
    """
    # ---- raw ---------------------------------------------------------
    spec1 = _resolve_path(spectra_dir, df_row["specname_dr16"])
    spec2 = _resolve_path(spectra_dir, df_row["specname_sdssv"])
    raw_ok = True
    try:
        w1, f1 = read_fits_flux_wave(spec1)
        w2, f2 = read_fits_flux_wave(spec2)
    except Exception as exc:
        raw_ok = False
        ax_raw.text(0.5, 0.5, f"raw read failed\n{exc}",
                    ha="center", va="center", transform=ax_raw.transAxes,
                    fontsize=9, color="#888888")
    if raw_ok:
        ax_raw.plot(w1, f1, color="#1f4e79", lw=0.8, alpha=0.8,
                    label=f"epoch 1 (MJD {int(df_row.get('mjd_dr16', 0))})")
        ax_raw.plot(w2, f2, color="#c0392b", lw=0.8, alpha=0.8,
                    label=f"epoch 2 (MJD {int(df_row.get('mjd_sdssv', 0))})")
        ax_raw.set_xlabel(r"observed wavelength ($\AA$)")
        ax_raw.set_ylabel("flux  (raw FITS units)")
        ax_raw.legend(fontsize=8, loc="upper right")
        ax_raw.grid(alpha=0.3)

    z = float(df_row["z"])
    src = str(df_row.get("source", "?"))
    ax_raw.set_title(
        f"{title_prefix}  raw   z={z:.3f}  src={src}  "
        f"p={prob:.3f}  thr={threshold:.2f}",
        fontsize=10,
    )

    # ---- processed ---------------------------------------------------
    mad1 = arrays["mad1"][arr_i]
    mad2 = arrays["mad2"][arr_i]
    v1 = arrays["valid1"][arr_i]
    v2 = arrays["valid2"][arr_i]
    grid = np.asarray(MASTER_GRID)
    m1 = np.where(v1, mad1, np.nan)
    m2 = np.where(v2, mad2, np.nan)
    ax_proc.plot(grid, m1, color="#1f4e79", lw=0.8, alpha=0.85,
                 label="epoch 1 MAD-norm")
    ax_proc.plot(grid, m2, color="#c0392b", lw=0.8, alpha=0.85,
                 label="epoch 2 MAD-norm")
    # Mark covered (valid) overlap region
    both_valid = v1 & v2
    if both_valid.any():
        lo = float(grid[np.argmax(both_valid)])
        hi = float(grid[len(grid) - 1 - np.argmax(both_valid[::-1])])
        ax_proc.axvspan(lo, hi, color="#27ae60", alpha=0.05,
                        label="both epochs covered")
    ax_proc.set_xlabel(r"rest-frame wavelength ($\AA$)")
    ax_proc.set_ylabel("MAD-norm flux  (model input ch0)")
    ax_proc.legend(fontsize=8, loc="upper right")
    ax_proc.grid(alpha=0.3)
    ax_proc.set_title(f"{title_prefix}  processed (rest-frame, MAD-norm)",
                      fontsize=10)


def _per_z_bin_tpr(probs, labels, z, threshold, n_bins=5):
    edges = np.linspace(float(np.min(z)), float(np.max(z)) + 1e-9, n_bins + 1)
    out = []
    preds = (probs >= float(threshold)).astype(np.int64)
    for b in range(n_bins):
        m = (z >= edges[b]) & (z < edges[b + 1])
        pos_m = m & (labels == 1)
        n_pos = int(pos_m.sum())
        n_tp  = int((pos_m & (preds == 1)).sum())
        out.append({
            "z_lo": float(edges[b]),
            "z_hi": float(edges[b + 1]),
            "n_pos": n_pos,
            "tp": n_tp,
            "tpr": (n_tp / n_pos) if n_pos > 0 else float("nan"),
        })
    return out


def main():
    cfg = load_config(os.path.join(BASE_DIR, "config_v2.yml"))
    paths = cfg["paths"]
    s = cfg["siamese"]
    pp = cfg["preprocessing"]

    device = pick_device()
    print(f"[eval] device: {device}")

    # ---- inputs --------------------------------------------------------
    ckpt_path = os.path.join(BASE_DIR, paths["siamese_checkpoint"])
    stats_path = os.path.join(BASE_DIR, paths["norm_stats"])
    pkl_path = os.path.join(BASE_DIR, paths["clagn_test_pickle"])
    spectra_dir = os.path.join(BASE_DIR, paths["clagn_test_spectra_dir"])
    out_dir = os.path.join(BASE_DIR, paths["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        sys.exit(f"[eval] checkpoint not found: {ckpt_path}")
    if not os.path.exists(pkl_path):
        sys.exit(f"[eval] test pickle not found: {pkl_path}")

    # ---- read test pickle for the source / metadata columns ------------
    df = pd.read_pickle(pkl_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[eval] test pickle: {len(df):,} rows  "
          f"(pos={int((df.label == 1).sum())}, "
          f"neg={int((df.label == 0).sum())})")
    if "source" in df.columns:
        print(f"[eval] sources: {df.source.value_counts().to_dict()}")

    # ---- preprocess pairs (separate cache so the training cache isn't
    # ----  overwritten by this test pickle's pair arrays) ---------------
    eval_cache = os.path.join(out_dir, "eval_pair_arrays_cache.npz")
    stats = load_norm_stats(stats_path)
    channel1_scale = float(stats.get("channel1_scale", 1.0))
    print(f"[eval] channel1_scale = {channel1_scale:.4g}")
    print(f"[eval] loading/building pair arrays  cache={eval_cache}")
    arrays = load_or_build_pair_arrays(
        pkl_path,
        spectra_dir,
        cache_path=eval_cache,
        oiii_snr_min=pp["oiii_snr_min"],
        subtract_continuum=False,
        split_filter="test"
    )
    n = len(arrays["y"])
    print(f"[eval] preprocessed {n:,} pairs  "
          f"(pos={int((arrays['y'] == 1).sum())}, "
          f"neg={int((arrays['y'] == 0).sum())})")

    # Map array rows back to the source rows so per-source / per-z works
    # even after failed rows are dropped during preprocessing.
    arr_sdssid = arrays["sdssid"]
    # Use astype to handle pandas Int64 / numpy int comparisons cleanly.
    df_idx_by_id = {int(v): i for i, v in enumerate(df["sdssid"].values)}
    df_idx = np.array(
        [df_idx_by_id.get(int(sid), -1) for sid in arr_sdssid],
        dtype=np.int64,
    )
    ok = df_idx >= 0
    if (~ok).any():
        print(f"[eval] WARNING: {int((~ok).sum())} preprocessed rows have "
              f"no matching sdssid in the test pickle and are dropped")
    df_idx = df_idx[ok]
    df_aligned = df.iloc[df_idx].reset_index(drop=True)

    # ---- build a no-shuffle DataLoader and score every pair ------------
    all_idx = np.where(ok)[0]
    eval_set = RealPairDataset(
        arrays, all_idx, channel1_scale,
        mode="val",       # disables synthetic augmentation
        seed=s["seed"],
    )
    loader = DataLoader(
        eval_set, batch_size=s["batch_size"],
        shuffle=False, num_workers=s["num_workers"],
    )

    # ---- model ---------------------------------------------------------
    train_pos_rate = float((arrays["y"] == 1).mean())
    model = SiameseChangeNet(
        in_channels=2,
        dropout=s["dropout"],
        prior_pos=max(train_pos_rate, 0.01),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    saved_threshold = float(ckpt.get(
        "best_threshold", s.get("decision_threshold", 0.5)
    ))
    saved_metrics = ckpt.get("best_threshold_metrics", None)
    fbeta_beta = float(ckpt.get("fbeta_beta", s.get("fbeta_beta", 0.5)))
    min_recall = float(ckpt.get("min_recall", s.get("min_recall", 0.10)))
    max_fpr    = float(ckpt.get("max_fpr",    s.get("max_fpr",    0.01)))
    print(f"[eval] checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"saved threshold={saved_threshold:.3f}  "
          f"fbeta_beta={fbeta_beta}  min_recall={min_recall}  max_fpr={max_fpr}")
    if saved_metrics is not None:
        print(f"[eval] checkpoint val metrics at saved threshold: "
              f"F{fbeta_beta}={saved_metrics.get('fbeta', float('nan')):.4f}  "
              f"prec={saved_metrics.get('precision', float('nan')):.4f}  "
              f"rec={saved_metrics.get('recall', float('nan')):.4f}")

    # ---- inference -----------------------------------------------------
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
            logits = model(x1, x2)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel().astype(int)
    assert len(probs) == len(df_aligned), \
        f"prob/label length mismatch: {len(probs)} vs {len(df_aligned)}"

    # ---- per-object probability dump (deployment ranking view) --------
    # The model is a candidate ranker: in deployment you inspect the top-K by
    # probability. This dump shows the probability every object got (esp. the
    # CL-AGN positives) and how recall grows with the inspection budget, so the
    # operating threshold can be chosen at deployment -- NOT tuned on this test.
    dump = df_aligned.copy()
    dump["prob"] = probs
    dump["label"] = labels
    keep = [c for c in ["sdssid", "source", "z", "ra", "dec",
                        "specname_dr16", "specname_sdssv"] if c in dump.columns]
    dump = dump[keep + ["label", "prob"]].sort_values(
        "prob", ascending=False).reset_index(drop=True)
    dump["rank"] = np.arange(1, len(dump) + 1)
    probs_csv = os.path.join(out_dir, "eval_per_object_probs.csv")
    dump.to_csv(probs_csv, index=False)
    print(f"\n[eval] wrote per-object probabilities -> {probs_csv}")

    n_pos_total = int((dump["label"] == 1).sum())
    has_src = "source" in dump.columns
    pos = dump[dump["label"] == 1]
    print(f"[eval] CL-AGN positives -- probability & rank (of {len(dump)}):")
    for _, r in pos.iterrows():
        src = f"  {str(r['source']):>10s}" if has_src else ""
        print(f"   rank {int(r['rank']):4d}/{len(dump)}  prob={r['prob']:.3f}"
              f"  z={r['z']:.3f}{src}")
    print("[eval] recall vs inspection budget (top-K by probability):")
    for K in [50, 100, 150, 200, 300, 500]:
        if K <= len(dump):
            tp = int(dump["label"].values[:K].sum())
            line = (f"   top {K:4d}: {tp:3d}/{n_pos_total} positives "
                    f"(recall {tp / max(n_pos_total, 1):.2f})")
            if has_src:
                for s in sorted(dump["source"].dropna().unique()):
                    if s == "phase2_neg":
                        continue
                    msk = dump["source"].values[:K] == s
                    tp_s = int(dump["label"].values[:K][msk].sum())
                    tot_s = int((pos["source"] == s).sum())
                    line += f"  {s}={tp_s}/{tot_s}"
            print(line)

    # ---- overall metrics ----------------------------------------------
    saved = _metrics_at_threshold(probs, labels, saved_threshold, fbeta_beta)
    auc = _auc(probs, labels)
    print()
    print(f"[eval] ============ OVERALL  @ saved threshold "
          f"{saved_threshold:.3f}  ============")
    print(f"  N         = {saved['n']:6d}  "
          f"(pos={saved['n_pos']}, neg={saved['n_neg']})")
    print(f"  TP / FP   = {saved['tp']:6d} / {saved['fp']:6d}")
    print(f"  TN / FN   = {saved['tn']:6d} / {saved['fn']:6d}")
    print(f"  accuracy    = {saved['accuracy']:.4f}")
    print(f"  precision   = {saved['precision']:.4f}")
    print(f"  recall      = {saved['recall']:.4f}   (=TPR/sensitivity)")
    print(f"  specificity = {saved['specificity']:.4f}   (=TNR)")
    print(f"  NPV         = {saved['npv']:.4f}")
    print(f"  FPR         = {saved['fpr']:.4f}")
    print(f"  FNR         = {saved['fnr']:.4f}")
    print(f"  F{fbeta_beta}        = {saved['fbeta']:.4f}   "
          f"(primary -- prioritises precision)")
    print(f"  F1          = {saved['f1']:.4f}")
    print(f"  AUC         = {auc:.4f}")

    # ---- post-hoc upper bound: tune on test (informational) -----------
    chosen, sweep, used_fallback = _threshold_sweep(
        probs, labels,
        beta=fbeta_beta, min_recall=min_recall, max_fpr=max_fpr,
    )
    print(f"\n[eval] post-hoc tuned-on-test (UPPER BOUND, leaky):  "
          f"thr={chosen['threshold']:.2f}  "
          f"F{fbeta_beta}={chosen['fbeta']:.4f}  "
          f"prec={chosen['precision']:.4f}  rec={chosen['recall']:.4f}  "
          f"{'FALLBACK' if used_fallback else ''}")

    # ---- per-source breakdown -----------------------------------------
    per_source = {}
    if "source" in df_aligned.columns:
        print("\n[eval] per-source @ saved threshold:")
        for src in sorted(df_aligned["source"].dropna().unique()):
            m = (df_aligned["source"].values == src)
            sub = _metrics_at_threshold(
                probs[m], labels[m], saved_threshold, fbeta_beta,
            )
            per_source[src] = sub
            n_pos = int((labels[m] == 1).sum())
            n_neg = int((labels[m] == 0).sum())
            print(f"  {src:>10s}  n_pos={n_pos:4d}  n_neg={n_neg:4d}  "
                  f"prec={sub['precision']:.4f}  rec={sub['recall']:.4f}  "
                  f"F{fbeta_beta}={sub['fbeta']:.4f}  "
                  f"TP/FP={sub['tp']}/{sub['fp']}")

    # ---- per-z TPR (positives only) -----------------------------------
    per_z = _per_z_bin_tpr(probs, labels, df_aligned["z"].values,
                           saved_threshold, n_bins=5)
    print("\n[eval] per-z-bin TPR (positives only) @ saved threshold:")
    for b in per_z:
        print(f"  z {b['z_lo']:.2f}-{b['z_hi']:.2f}  "
              f"n_pos={b['n_pos']:3d}  tp={b['tp']:3d}  "
              f"tpr={b['tpr']:.3f}")

    # ---- PR curve + plot ----------------------------------------------
    pr_p, pr_r, pr_t, ap = _pr_curve(probs, labels)
    print(f"\n[eval] PR-AUC (average precision) = {ap:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # PR curve
    ax = axes[0]
    if pr_p is not None:
        ax.plot(pr_r, pr_p, color="#1f4e79", lw=2)
        ax.scatter([saved["recall"]], [saved["precision"]],
                   color="#c0392b", s=70, zorder=5,
                   label=f"saved thr={saved_threshold:.2f}")
        ax.scatter([chosen["recall"]], [chosen["precision"]],
                   color="#27ae60", s=70, marker="s", zorder=5,
                   label=f"tuned-on-test thr={chosen['threshold']:.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
    ax.set_title(f"PR curve  (AP={ap:.3f})")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # Confusion matrix
    ax = axes[1]
    cm = np.array([[saved["tn"], saved["fp"]],
                   [saved["fn"], saved["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"])
    ax.set_yticklabels(["true neg", "true pos"])
    ax.set_title(f"Confusion @ thr={saved_threshold:.2f}")

    fig.tight_layout()
    png_path = os.path.join(out_dir, "eval_clagn_test.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\n[eval] wrote plot -> {png_path}")

    # ---- example pair plots: one positive (TP if available) and one
    # ---- negative (TN if available), each with raw + processed views.
    (pos_idx, pos_label), (neg_idx, neg_label) = _pick_examples(
        probs, labels, saved_threshold, df_aligned,
    )
    if pos_idx >= 0 and neg_idx >= 0:
        # probs/df_aligned are indexed 0..K-1 (only rows that survived the
        # sdssid match). arrays is the original preprocessed pool; the
        # mapping from probs-index -> arrays-index is np.where(ok)[0].
        ok_idx = np.where(ok)[0]
        arr_pos = int(ok_idx[pos_idx])
        arr_neg = int(ok_idx[neg_idx])
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        _plot_example_pair(
            axes[0, 0], axes[0, 1],
            arr_i=arr_pos,
            df_row=df_aligned.iloc[pos_idx],
            arrays=arrays,
            spectra_dir=spectra_dir,
            title_prefix=f"POSITIVE ({pos_label})",
            prob=float(probs[pos_idx]),
            threshold=saved_threshold,
        )
        _plot_example_pair(
            axes[1, 0], axes[1, 1],
            arr_i=arr_neg,
            df_row=df_aligned.iloc[neg_idx],
            arrays=arrays,
            spectra_dir=spectra_dir,
            title_prefix=f"NEGATIVE ({neg_label})",
            prob=float(probs[neg_idx]),
            threshold=saved_threshold,
        )
        fig.suptitle("Example pairs: raw FITS (left) vs. model input ch0 (right)",
                     fontsize=12, y=0.995)
        fig.tight_layout()
        examples_png = os.path.join(out_dir, "eval_pair_examples.png")
        fig.savefig(examples_png, dpi=140)
        plt.close(fig)
        print(f"[eval] wrote example pairs -> {examples_png}")
    else:
        print("[eval] could not pick example pairs (need at least one "
              "positive and one negative in test).")

    # ---- JSON dump ----------------------------------------------------
    summary = {
        "checkpoint":       os.path.relpath(ckpt_path, BASE_DIR),
        "n_pairs":          int(len(probs)),
        "n_pos":            int((labels == 1).sum()),
        "n_neg":            int((labels == 0).sum()),
        "saved_threshold":  saved_threshold,
        "saved_metrics":    saved,
        "tuned_on_test_threshold": chosen["threshold"],
        "tuned_on_test_metrics":   chosen,
        "tuned_on_test_used_fallback": bool(used_fallback),
        "auc":              float(auc),
        "pr_auc":           float(ap),
        "per_source":       per_source,
        "per_z":            per_z,
        "fbeta_beta":       fbeta_beta,
        "min_recall":       min_recall,
        "max_fpr":          max_fpr,
    }
    json_path = os.path.join(out_dir, "eval_clagn_test.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval] wrote summary -> {json_path}")


if __name__ == "__main__":
    main()
