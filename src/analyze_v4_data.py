"""
analyze_v4_data.py
==================
Data analysis for the three v4 phases: SSL pool, Siamese training pairs, and the
held-out test. Produces redshift histograms (marked by survey / label / AGN type)
and prints summary tables. Figures -> data_v4/analysis/.

Runs anywhere; if pyarrow/fastparquet is missing it skips the SSL parquet figure
(prints a note) and still does Siamese + Test from the pickles.

Run:
    python src/analyze_v4_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths_v4 as P

OUT = os.path.join(P.DATA_OUT, "analysis")
os.makedirs(OUT, exist_ok=True)
EDGES = np.linspace(0.0, 0.9, 19)            # 0.05-wide z bins to 0.9
SURVEY_C = {"sdss_dr7": "#9467bd", "desi": "#1f77b4", "dr16": "#2ca02c",
            "sdssv": "#d62728"}


def _early_type_lookup() -> dict:
    """early-epoch specname -> sdss_type, from the two negative pools."""
    m = {}
    if os.path.exists(P.SDSSV_NEG):
        s = pd.read_pickle(P.SDSSV_NEG)
        m.update(dict(zip(s["specname_dr16"], s["sdss_type"])))
    if os.path.exists(P.DESI_NEG):
        d = pd.read_pickle(P.DESI_NEG)
        m.update(dict(zip(d["specname_sdss"], d["sdss_type"])))
    return m


# ===================== Phase 1: SSL ==================================
def analyze_ssl():
    parqs = [("DR7+DESI", P.SSL_DR7CAPPED), ("DR16+SDSS-V QSO", P.SSL_EXTENSION),
             ("type-2 (DR16+SDSS-V)", P.SSL_TYPE2),
             ("DR7 typed (t1+t2)", os.path.join(P.DATA_OUT, "ssl_dr7_types.parquet"))]
    frames = []
    for name, path in parqs:
        if not os.path.exists(path):
            print(f"[ssl] missing {os.path.basename(path)} -- skip"); continue
        try:
            df = pd.read_parquet(path, columns=["z", "survey", "agn_type"])
        except Exception:
            try:
                df = pd.read_parquet(path, columns=["z", "survey"])
            except Exception as e:
                print(f"[ssl] cannot read parquet ({str(e)[:60]}...) -- "
                      "install pyarrow to get the SSL figure. Skipping SSL.")
                return
            df["agn_type"] = "type2" if name.startswith("type-2") else "unknown"
        df["pool"] = name
        frames.append(df)
    if not frames:
        print("[ssl] no SSL parquets found"); return
    ssl = pd.concat(frames, ignore_index=True)
    print("\n[ssl] SSL pool by survey:")
    print(ssl.groupby("survey").size())
    print("[ssl] type-2 pool size:", int((ssl.pool.str.startswith("type-2")).sum()))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for sv, sub in ssl.groupby("survey"):
        ax[0].hist(sub.z.clip(0, 0.9), bins=EDGES, histtype="step", lw=2,
                   label=f"{sv} ({len(sub):,})", color=SURVEY_C.get(sv))
    ax[0].set_title("SSL pool — redshift by survey"); ax[0].set_xlabel("z")
    ax[0].set_ylabel("spectra"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    t2 = ssl[ssl.agn_type == "type2"]
    other = ssl[ssl.agn_type != "type2"]
    ax[1].hist(other.z.clip(0, 0.9), bins=EDGES, color="#bbbbbb",
               label=f"type-1 / mixed ({len(other):,})")
    ax[1].hist(t2.z.clip(0, 0.9), bins=EDGES, color="#d62728", alpha=0.8,
               label=f"type-2 AGN ({len(t2):,})")
    ax[1].set_title("SSL pool — type-2 contribution"); ax[1].set_xlabel("z")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ssl_redshift.png"), dpi=140)
    plt.close(fig); print(f"[ssl] wrote {OUT}/ssl_redshift.png")


# ===================== Phase 2: Siamese train =======================
def analyze_siamese():
    tr = pd.read_pickle(P.PHASE2_TRAIN)
    look = _early_type_lookup()
    tr["agn_type"] = tr["specname_dr16"].map(look)   # negs get type1/type2; pos -> NaN
    print("\n[siamese] train by survey x label:")
    print(tr.groupby(["survey", "label"]).size())
    print("[siamese] negatives by AGN type:")
    print(tr[tr.label == 0].groupby(["survey", "agn_type"]).size())

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    # (a) pos vs neg
    ax[0].hist(tr.loc[tr.label == 0, "z"].clip(0, 0.9), bins=EDGES, color="#888",
               alpha=0.6, label=f"negatives ({int((tr.label==0).sum()):,})")
    ax[0].hist(tr.loc[tr.label == 1, "z"].clip(0, 0.9), bins=EDGES, color="#27ae60",
               alpha=0.7, label=f"positives ({int((tr.label==1).sum()):,})")
    ax[0].set_title("Siamese train — positive vs negative")
    ax[0].set_xlabel("z"); ax[0].set_ylabel("pairs"); ax[0].legend(); ax[0].grid(alpha=0.3)
    # (b) survey x label
    for (sv, lab), sub in tr.groupby(["survey", "label"]):
        ls = "-" if lab == 1 else "--"
        ax[1].hist(sub.z.clip(0, 0.9), bins=EDGES, histtype="step", lw=2, ls=ls,
                   color=SURVEY_C.get(sv),
                   label=f"{sv} {'pos' if lab else 'neg'} ({len(sub):,})")
    ax[1].set_title("Siamese train — survey × label"); ax[1].set_xlabel("z")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # (c) negatives by AGN type
    neg = tr[tr.label == 0]
    for t, c in [("type1", "#1f77b4"), ("type2", "#d62728")]:
        s = neg[neg.agn_type == t]
        ax[2].hist(s.z.clip(0, 0.9), bins=EDGES, histtype="step", lw=2, color=c,
                   label=f"{t} neg ({len(s):,})")
    ax[2].set_title("Siamese train — negatives by AGN type"); ax[2].set_xlabel("z")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "siamese_redshift.png"), dpi=140)
    plt.close(fig); print(f"[siamese] wrote {OUT}/siamese_redshift.png")


# ===================== Phase 3: Test ================================
def analyze_test():
    te = pd.read_pickle(P.CLAGN_TEST)
    look = _early_type_lookup()
    te["agn_type"] = te["specname_dr16"].map(look)
    print("\n[test] by source:")
    print(te.groupby("source").size())
    print("[test] negatives by AGN type:")
    print(te[te.label == 0].groupby(["survey", "agn_type"]).size())

    SRC_C = {"lowz": "#27ae60", "paper2": "#2ca02c", "sdssv_neg": "#d62728",
             "desi_neg": "#1f77b4"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for src, sub in te.groupby("source"):
        ax[0].hist(sub.z.clip(0, 0.9), bins=EDGES, histtype="step", lw=2,
                   color=SRC_C.get(src), label=f"{src} ({len(sub)})")
    ax[0].set_title("Test — redshift by source"); ax[0].set_xlabel("z")
    ax[0].set_ylabel("pairs"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # per-survey pos vs neg
    for (sv, lab), sub in te.groupby(["survey", "label"]):
        ls = "-" if lab == 1 else "--"
        ax[1].hist(sub.z.clip(0, 0.9), bins=EDGES, histtype="step", lw=2, ls=ls,
                   color=SURVEY_C.get(sv),
                   label=f"{sv} {'pos' if lab else 'neg'} ({len(sub)})")
    ax[1].set_title("Test — survey × label"); ax[1].set_xlabel("z")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "test_redshift.png"), dpi=140)
    plt.close(fig); print(f"[test] wrote {OUT}/test_redshift.png")


if __name__ == "__main__":
    analyze_ssl()
    analyze_siamese()
    analyze_test()
    print(f"\n[done] figures in {OUT}")
