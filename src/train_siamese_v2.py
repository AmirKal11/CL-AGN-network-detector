"""
train_siamese_v2.py  --  STAGE 2
================================
Fine-tune the SSL-pretrained encoder + a fresh symmetric change head on real
same-object epoch pairs (DR16 + SDSS-V crossmatch). For each pair (x1, x2),
the model predicts whether the object underwent a CL-AGN transition.

Components (all v2):
    SiameseChangeNet           shared SSL encoder + sum/abs-diff/prod head
    load_encoder_into          load SSL encoder weights into the Siamese
    RealPairDataset            real same-object pairs (no cross-object pairs)
                               with optional within-object synthetic positives
    BinaryFocalLossWithLogits  focal loss for the heavy class imbalance
    load_or_build_pair_arrays  preprocess all pairs once -> cache .npz

Why two learning rates: the encoder is already trained (val recon MSE ~0.13),
so it gets a low LR to fine-tune carefully without forgetting; the head is
fresh and gets a higher LR. This is the standard recipe for transferred
backbones.

Run from the project root:
    python src/train_siamese_v2.py

Produces (under models/clagn_v2/):
    siamese_changenet.pth    best checkpoint (selected by F0.5 sweep --
                             see _threshold_sweep + tuple tie-break)
    siamese_loss_curve.png   focal loss + F0.5/prec/rec/FPR/threshold

Model selection (mirrors v1 train_siamese.py):
    Each epoch the validation probabilities are scored across thresholds
    in [0.05, 0.95]. The chosen threshold is the one that:
        1. has recall >= min_recall  AND  fpr <= max_fpr     (purity gate)
        2. among survivors, maximises (F0.5, precision, -fpr, recall)
    If no threshold satisfies the purity gate, the epoch falls back to
    the best F0.5 over all thresholds. The checkpoint saved is the one
    that achieved the best (F0.5, precision, -fpr, recall) tuple across
    all epochs.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import load_config
from datasets_v2 import (
    load_or_build_pair_arrays,
    split_indices,
    RealPairDataset,
)
from architectures_v2 import (
    SiameseChangeNet,
    BinaryFocalLossWithLogits,
    load_encoder_into,
)
from preprocessing_oiii import load_norm_stats
from pretrain_ssl import pick_device


# ----------------------------------------------------------------------
# Metrics helpers
# ----------------------------------------------------------------------
def _binary_metrics(probs: np.ndarray, labels: np.ndarray,
                    threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n = max(tp + fp + tn + fn, 1)
    return {
        "acc":  (tp + tn) / n,
        "prec": tp / max(tp + fp, 1),
        "rec":  tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via sklearn if available, else Mann-Whitney-U fallback."""
    labels = labels.astype(np.int64)
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        # Wilcoxon rank-sum estimator -- ties handled by average rank.
        all_p = np.concatenate([pos, neg])
        order = all_p.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(all_p) + 1, dtype=np.float64)
        # Average ranks for ties
        # (small dataset -> good enough without tie averaging)
        sum_pos = ranks[: len(pos)].sum()
        return float((sum_pos - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))


def _pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under precision-recall). Threshold-free, recall-first
    ranking metric used for model selection."""
    labels = labels.astype(np.int64)
    npos = int((labels == 1).sum())
    if npos == 0 or npos == len(labels):
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, probs))
    except Exception:
        order = np.argsort(-probs)
        y = labels[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / npos
        rec_prev = np.concatenate([[0.0], rec[:-1]])
        return float(np.sum((rec - rec_prev) * prec))


def _recall_at_fpr(probs: np.ndarray, labels: np.ndarray, max_fpr: float) -> dict:
    """Deployment operating point: threshold giving MAX recall s.t. FPR <= max_fpr
    (the inspection-budget ceiling). Tie-break: lower FPR."""
    labels = labels.astype(np.int64)
    npos = max(int((labels == 1).sum()), 1)
    nneg = max(int((labels == 0).sum()), 1)
    best = {"threshold": 1.0, "recall": 0.0, "precision": 0.0, "fpr": 0.0,
            "tp": 0, "fp": 0, "tn": nneg, "fn": npos}
    for thr in np.unique(np.concatenate([probs, [1.0]])):
        pred = probs >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fpr = fp / nneg
        if fpr > max_fpr:
            continue
        rec = tp / npos
        if (rec > best["recall"]) or (rec == best["recall"] and fpr < best["fpr"]):
            best = {"threshold": float(thr), "recall": rec,
                    "precision": tp / max(tp + fp, 1), "fpr": fpr,
                    "tp": tp, "fp": fp, "tn": nneg - fp, "fn": npos - tp}
    return best


def _threshold_sweep(probs: np.ndarray,
                     labels: np.ndarray,
                     beta: float,
                     min_recall: float,
                     max_fpr: float,
                     thresholds: np.ndarray | None = None):
    """
    Ported from train_siamese.py / evaluate_siamese_threshold_sweep.

    For each threshold in [0.05, 0.95] (19 steps by default), compute
    precision, recall, F-beta, FPR, and the confusion matrix on the
    positive class.

    Filter to thresholds satisfying both:
        recall >= min_recall    AND    fpr <= max_fpr
    If no threshold qualifies, fall back to all thresholds and flag
    ``used_fallback=True``.

    Tie-break (descending) order:
        F-beta, precision, -fpr, recall

    Returns
    -------
    chosen : dict   the winning threshold and its metrics
    results : list  per-threshold metrics for plotting / debugging
    used_fallback : bool
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    labels = labels.astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    results = []
    for thr in thresholds:
        preds = (probs >= float(thr)).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(n_pos, 1)
        fpr  = fp / max(n_neg, 1)
        denom = (beta * beta * prec) + rec
        fbeta = ((1.0 + beta * beta) * prec * rec / denom) if denom > 0 else 0.0
        results.append({
            "threshold": float(thr),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(prec), "recall": float(rec),
            "fbeta": float(fbeta), "fpr": float(fpr),
        })

    valid = [r for r in results
             if r["recall"] >= min_recall and r["fpr"] <= max_fpr]
    used_fallback = False
    if not valid:
        valid = results
        used_fallback = True
    valid.sort(
        key=lambda r: (r["fbeta"], r["precision"], -r["fpr"], r["recall"]),
        reverse=True,
    )
    return valid[0], results, used_fallback


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    cfg = load_config(os.path.join(BASE_DIR, "config_v2.yml"))
    paths = cfg["paths"]
    s = cfg["siamese"]
    pp = cfg["preprocessing"]

    torch.manual_seed(s["seed"])
    np.random.seed(s["seed"])
    device = pick_device()
    print(f"[siamese] device: {device}")

    # ---- norm stats (channel1_scale calibrated during SSL) -------------
    stats_path = os.path.join(BASE_DIR, paths["norm_stats"])
    stats = load_norm_stats(stats_path)
    channel1_scale = float(stats.get("channel1_scale", 1.0))
    print(f"[siamese] channel1_scale = {channel1_scale:.4g} (from {stats_path})")

    # ---- pair arrays (cached) ------------------------------------------
    pkl_path = os.path.join(BASE_DIR, paths["pair_pickle"])
    spectra_dir = os.path.join(BASE_DIR, paths["pair_spectra_dir"])
    cache_path = os.path.join(BASE_DIR, paths["pair_cache"])

    if not os.path.exists(pkl_path):
        sys.exit(f"[siamese] pair pickle not found: {pkl_path}")
    if not os.path.isdir(spectra_dir):
        sys.exit(f"[siamese] pair spectra dir not found: {spectra_dir}")

    print(f"[siamese] loading or building pair arrays from {pkl_path}")
    arrays = load_or_build_pair_arrays(
        pkl_path,
        spectra_dir,
        cache_path=cache_path,
        oiii_snr_min=pp["oiii_snr_min"],
        split_filter="train"
    )
    n_pairs = len(arrays["y"])
    n_pos = int((arrays["y"] == 1).sum())
    n_neg = int((arrays["y"] == 0).sum())
    print(f"[siamese] {n_pairs:,} pairs preprocessed  "
          f"(static={n_neg:,}, CL-AGN={n_pos:,})")

    # ---- object-disjoint, label-stratified split -----------------------
    train_idx, val_idx, test_idx = split_indices(
        arrays["y"],
        val_frac=s["val_frac"],
        test_frac=s["test_frac"],
        seed=s["seed"],
    )
    print(f"[siamese] split: train {len(train_idx):,}  val {len(val_idx):,}  "
          f"test {len(test_idx):,}")

    train_set = RealPairDataset(
        arrays, train_idx, channel1_scale,
        mode="train",
        synthetic_prob=s["synthetic_prob"],
        seed=s["seed"],
    )
    val_set = RealPairDataset(
        arrays, val_idx, channel1_scale,
        mode="val",
        seed=s["seed"],
    )
    print(f"[siamese] train labels (pre-synthetic): {train_set.label_counts()}")
    print(f"[siamese] val labels:                   {val_set.label_counts()}")

    # ---- WeightedRandomSampler so each batch is ~sampler_pos_rate positive
    # Option C: oversample positives to target a configurable batch rate
    # without throwing away any negatives. sampler_pos_rate = 0 disables it.
    sampler_pos_rate = float(s.get("sampler_pos_rate", 0.0))
    source_balanced = bool(s.get("source_balanced", False))   # v4
    if sampler_pos_rate > 0.0:
        y_train = arrays["y"][train_idx]
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        weights = np.empty(len(y_train), dtype=np.float64)
        # negatives share (1 - pos_rate) uniformly
        weights[y_train == 0] = (1.0 - sampler_pos_rate) / n_neg
        surv_train = (arrays["survey"][train_idx]
                      if source_balanced and "survey" in arrays else None)
        if surv_train is not None and len(set(surv_train[y_train == 1])) > 1:
            # v4 source-balanced: split the positive mass EQUALLY across the
            # surveys present in positives, so the head sees the scarce SDSS-V
            # positives as often as the abundant DESI ones (attacks the
            # 81-vs-434 imbalance that biased the head toward DESI).
            uniq = sorted(set(surv_train[y_train == 1].tolist()))
            for sv in uniq:
                m = (y_train == 1) & (surv_train == sv)
                n_sv = max(int(m.sum()), 1)
                weights[m] = sampler_pos_rate / (len(uniq) * n_sv)
            print(f"[siamese] source-balanced positives across {uniq} "
                  f"(equal mass per survey)")
        else:
            weights[y_train == 1] = sampler_pos_rate / n_pos
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights),
            num_samples=len(y_train),    # one "epoch" still = dataset size
            replacement=True,
        )
        print(f"[siamese] WeightedRandomSampler enabled  "
              f"(target pos rate per batch = {sampler_pos_rate:.0%}; "
              f"n_pos={n_pos}, n_neg={n_neg})")
        train_loader = DataLoader(
            train_set, batch_size=s["batch_size"], sampler=sampler,
            num_workers=s["num_workers"], drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=s["batch_size"],
            shuffle=True, num_workers=s["num_workers"], drop_last=False,
        )
    val_loader = DataLoader(
        val_set, batch_size=s["batch_size"],
        shuffle=False, num_workers=s["num_workers"],
    )

    # ---- model ---------------------------------------------------------
    # Initialise the head bias with the observed positive rate so the model
    # starts calibrated under heavy class imbalance.
    train_pos_rate = float((arrays["y"][train_idx] == 1).mean())
    print(f"[siamese] train positive rate (real only) = {train_pos_rate:.3f}")
    model = SiameseChangeNet(
        in_channels=2,
        dropout=s["dropout"],
        prior_pos=max(train_pos_rate, 0.01),
    )
    ssl_ckpt = os.path.join(BASE_DIR, paths["ssl_checkpoint"])
    if not os.path.exists(ssl_ckpt):
        sys.exit(f"[siamese] SSL checkpoint not found: {ssl_ckpt}\n"
                 "  Run pretrain_ssl.py first.")
    load_encoder_into(model, ssl_ckpt, device=device)
    model = model.to(device)

    # ---- optionally freeze the encoder (linear-probe regime) -----------
    encoder_freeze = bool(s.get("encoder_freeze", True))
    enc_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    if encoder_freeze:
        for p in enc_params:
            p.requires_grad = False
        model.encoder.eval()  # also disables encoder dropout/BN train mode
        n_enc = sum(p.numel() for p in enc_params)
        n_head = sum(p.numel() for p in head_params)
        print(f"[siamese] encoder FROZEN  "
              f"(enc params {n_enc:,} -- no grad; head params {n_head:,})")
        opt = torch.optim.AdamW(
            [{"params": head_params, "lr": s["head_lr"]}],
            weight_decay=s["weight_decay"],
        )
    else:
        opt = torch.optim.AdamW([
            {"params": enc_params,  "lr": s["encoder_lr"]},
            {"params": head_params, "lr": s["head_lr"]},
        ], weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["num_epochs"])

    criterion = BinaryFocalLossWithLogits(
        alpha=s["focal_alpha"], gamma=s["focal_gamma"],
    )

    # ---- output paths --------------------------------------------------
    out_dir = os.path.join(BASE_DIR, paths["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(BASE_DIR, paths["siamese_checkpoint"])

    # ---- recall-first selection config --------------------------------
    # Candidate ranker: CHECKPOINT is selected on the MEAN of per-survey PR-AUC
    # (each survey weighted equally so the larger DESI count can't drown SDSS-V),
    # and the saved deployment threshold is max recall at FPR<=max_fpr on SDSS-V.
    fbeta_beta = float(s.get("fbeta_beta", 2.0))   # kept in ckpt metadata only
    min_recall = float(s.get("min_recall", 0.10))  # informational
    max_fpr    = float(s.get("max_fpr", 0.05))     # B/N inspection-budget ceiling

    val_survey = np.asarray(arrays["survey"][val_idx]) if "survey" in arrays else None
    y_val = arrays["y"][val_idx]
    if val_survey is not None:
        sel_surveys = [sv for sv in np.unique(val_survey)
                       if int((y_val[val_survey == sv] == 1).sum()) >= 1]
    else:
        sel_surveys = []
    op_survey = "sdssv" if (val_survey is not None and "sdssv" in sel_surveys) else None
    print(f"[siamese] selection = mean PR-AUC over surveys "
          f"{sel_surveys or ['(full val)']}; operating point on "
          f"{op_survey or 'full val'} @FPR<={max_fpr}")

    history = {
        "train_loss": [], "val_loss": [], "val_auc": [], "val_prauc": [],
        "val_prec": [], "val_rec": [], "val_fpr": [], "val_threshold": [],
    }
    best_score = -1.0          # best mean per-survey PR-AUC seen so far
    best_threshold = None
    best_threshold_metrics = None

    # ---- training loop -------------------------------------------------
    for epoch in range(s["num_epochs"]):
        t0 = time.time()
        model.train()
        if encoder_freeze:
            # Keep encoder deterministic (no dropout / BN updates) when frozen.
            model.encoder.eval()
        running, nb = 0.0, 0
        for x1, x2, y in train_loader:
            x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
            logits = model(x1, x2)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(nb, 1)

        # validation
        model.eval()
        vrun, vnb = 0.0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
                logits = model(x1, x2)
                vrun += criterion(logits, y).item()
                vnb += 1
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(y.cpu().numpy())
        val_loss = vrun / max(vnb, 1)
        probs = np.concatenate(all_probs).ravel()
        labels = np.concatenate(all_labels).ravel().astype(int)
        # v4: per-survey PR-AUC -> mean (selection); operating point on SDSS-V.
        val_auc = _auc(probs, labels)
        if sel_surveys:
            prauc_by = {sv: _pr_auc(probs[val_survey == sv], labels[val_survey == sv])
                        for sv in sel_surveys}
            vals = [v for v in prauc_by.values() if v == v]      # drop NaN
            mean_prauc = float(np.mean(vals)) if vals else float("nan")
        else:
            prauc_by = {"all": _pr_auc(probs, labels)}
            mean_prauc = prauc_by["all"]
        if op_survey is not None:
            m = val_survey == op_survey
            op = _recall_at_fpr(probs[m], labels[m], max_fpr)
        else:
            op = _recall_at_fpr(probs, labels, max_fpr)
        sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_prauc"].append(mean_prauc)
        history["val_prec"].append(op["precision"])
        history["val_rec"].append(op["recall"])
        history["val_fpr"].append(op["fpr"])
        history["val_threshold"].append(op["threshold"])

        by_str = " ".join(f"{sv}={prauc_by[sv]:.2f}" for sv in prauc_by)
        print(
            f"[siamese] ep {epoch + 1:3d}/{s['num_epochs']}  "
            f"train {train_loss:.4f}  val {val_loss:.4f}  "
            f"meanPR-AUC {mean_prauc:.3f} ({by_str})  AUC {val_auc:.3f}  | "
            f"op[{op_survey or 'all'}]@FPR<={max_fpr}: thr {op['threshold']:.2f} "
            f"rec {op['recall']:.3f} prec {op['precision']:.3f} fpr {op['fpr']:.3f}  "
            f"[{time.time() - t0:.0f}s]"
        )

        # Model selection: highest MEAN per-survey PR-AUC (all surveys equal weight).
        if mean_prauc > best_score:
            best_score = mean_prauc
            best_threshold = op["threshold"]      # max recall at FPR<=max_fpr (budget)
            best_threshold_metrics = op
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_threshold": best_threshold,
                "best_threshold_metrics": best_threshold_metrics,
                "val_pr_auc_mean": mean_prauc,
                "val_pr_auc_by_survey": {k: float(v) for k, v in prauc_by.items()},
                "val_auc": val_auc,
                "val_loss": val_loss,
                "channel1_scale": channel1_scale,
                "selection_metric": "mean_per_survey_pr_auc",
                "op_survey": op_survey or "all",
                "max_fpr": max_fpr,
                "min_recall": min_recall,
                "fbeta_beta": fbeta_beta,
            }, ckpt_path)
            print(f"[siamese]   saved best (mean PR-AUC={mean_prauc:.4f}) -> "
                  f"{ckpt_path}  thr={best_threshold:.2f} rec={op['recall']:.3f} "
                  f"fpr={op['fpr']:.3f}")

    # ---- loss + metric curves -----------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_ylabel("focal loss")
    axes[0].set_title("Stage 2 -- Siamese (mean per-survey PR-AUC selection)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_prauc"], label="val mean PR-AUC (selection)",
                 color="#1f4e79", lw=2)
    axes[1].plot(history["val_rec"],   label=f"val recall @FPR<={max_fpr}",
                 color="#c0392b", lw=1, ls="--")
    axes[1].plot(history["val_prec"],  label="val precision @op",
                 color="#27ae60", lw=1, ls="--")
    axes[1].plot(history["val_fpr"],   label="val FPR @op",
                 color="#888888", lw=1, ls=":")
    axes[1].plot(history["val_auc"],   label="val ROC-AUC (info)",
                 color="#888888", lw=1, alpha=0.6)
    axes[1].plot(history["val_threshold"], label="op threshold",
                 color="#000000", lw=1, alpha=0.4)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric / threshold")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower right", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "siamese_loss_curve.png"), dpi=150)
    plt.close(fig)

    print(f"[siamese] done.")
    print(f"[siamese]   best mean per-survey PR-AUC = {best_score:.4f}")
    print(f"[siamese]   deployment threshold (recall@FPR<={max_fpr}, SDSS-V) "
          f"= {best_threshold:.2f}")
    if best_threshold_metrics:
        print(f"[siamese]   op metrics = rec={best_threshold_metrics['recall']:.4f}  "
              f"prec={best_threshold_metrics['precision']:.4f}  "
              f"fpr={best_threshold_metrics['fpr']:.4f}")
    print(f"[siamese] checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
