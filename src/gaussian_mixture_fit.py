"""
Two-component Gaussian mixture (in logit space) for a column of classifier
probabilities. Fits by EM, reports principled thresholds, and plots the
decomposition over probability space.

Usage:
    python prob_mixture.py predictions.csv --col prob
    python prob_mixture.py predictions.csv --col prob --out fit.png --restarts 10

Notes:
  * The "high" component is whichever mode has the larger mean -- for a CL-AGN
    ranker this is the "changed-looking" population, NOT pure CL-AGN. The
    thresholds/incompleteness therefore refer to that broad class.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import erf


# ---------- core math ----------------------------------------------------------
def _npdf(x, m, s):
    """Gaussian density N(x; m, s)."""
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _ncdf(x, m, s):
    """Gaussian CDF via erf (no scipy needed)."""
    return 0.5 * (1.0 + erf((x - m) / (s * np.sqrt(2.0))))


def fit_em(l, mu, sd, pi, n_iter=500, tol=1e-8):
    """One EM run on logit data `l` from the given init. Returns (mu, sd, pi, ll)."""
    N = len(l)
    prev_ll = -np.inf
    for _ in range(n_iter):
        # E-step: responsibilities (component 1 = high)
        a0 = pi[0] * _npdf(l, mu[0], sd[0])
        a1 = pi[1] * _npdf(l, mu[1], sd[1])
        denom = a0 + a1 + 1e-300
        R = a1 / denom                       # responsibility of HIGH comp
        # log-likelihood (for convergence + restart selection)
        ll = np.log(denom).sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
        # M-step: responsibility-weighted MLEs
        n1 = R.sum()
        n0 = N - n1
        mu = np.array([((1 - R) * l).sum() / n0, (R * l).sum() / n1])
        sd = np.array([
            np.sqrt(((1 - R) * (l - mu[0]) ** 2).sum() / n0),
            np.sqrt((R * (l - mu[1]) ** 2).sum() / n1),
        ])
        sd = np.clip(sd, 1e-3, None)         # guard against a component collapsing
        pi = np.array([n0 / N, n1 / N])
    return mu, sd, pi, prev_ll


def fit_mixture(p, restarts=10, eps=1e-4, seed=0):
    """
    Fit a 2-comp Gaussian mixture in logit space with multiple random restarts.
    Returns a dict of fitted params (component 0 = low mean, 1 = high mean).
    """
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    l = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    rng = np.random.default_rng(seed)

    best = None
    for r in range(restarts):
        if r == 0:                           # one sensible deterministic init
            mu0 = np.array([np.percentile(l, 25), np.percentile(l, 90)])
        else:                                # random inits for the rest
            mu0 = np.sort(rng.choice(l, size=2, replace=False))
        sd0 = np.array([l.std() / 2, l.std() / 2])
        pi0 = np.array([0.7, 0.3])
        mu, sd, pi, ll = fit_em(l, mu0.copy(), sd0.copy(), pi0.copy())
        if best is None or ll > best["ll"]:
            best = dict(mu=mu, sd=sd, pi=pi, ll=ll)

    # order components so index 0 = lower mean ("stable"), 1 = higher ("changed")
    order = np.argsort(best["mu"])
    mu, sd, pi = best["mu"][order], best["sd"][order], best["pi"][order]
    return dict(mu=mu, sd=sd, pi=pi, ll=best["ll"], logit=l, N=len(l))


# ---------- thresholds & summaries --------------------------------------------
def crossover(fit):
    """Logit (and prob) where P(high|score)=0.5."""
    mu, sd, pi = fit["mu"], fit["sd"], fit["pi"]
    grid = np.linspace(-12, 12, 20000)
    a0 = pi[0] * _npdf(grid, mu[0], sd[0])
    a1 = pi[1] * _npdf(grid, mu[1], sd[1])
    R = a1 / (a0 + a1 + 1e-300)
    lx = grid[np.argmin(np.abs(R - 0.5))]
    return lx, 1 / (1 + np.exp(-lx))


def lfdr_threshold(fit, alpha):
    """Smallest logit where local FDR = P(low|score) < alpha. Returns (logit, prob) or None."""
    mu, sd, pi = fit["mu"], fit["sd"], fit["pi"]
    grid = np.linspace(-12, 12, 20000)
    a0 = pi[0] * _npdf(grid, mu[0], sd[0])
    a1 = pi[1] * _npdf(grid, mu[1], sd[1])
    lf = a0 / (a0 + a1 + 1e-300)             # P(belongs to LOW | score)
    hit = np.where(lf < alpha)[0]
    if len(hit) == 0:
        return None
    lx = grid[hit[0]]
    return lx, 1 / (1 + np.exp(-lx))


def incompleteness(fit, prob_cut):
    """Estimated # (and fraction) of the HIGH component falling below prob_cut."""
    mu, sd, pi, N = fit["mu"], fit["sd"], fit["pi"], fit["N"]
    lcut = np.log(prob_cut / (1 - prob_cut))
    frac_below = _ncdf(lcut, mu[1], sd[1])
    total_high = pi[1] * N
    return frac_below * total_high, total_high, frac_below


def report(fit):
    mu, sd, pi, N = fit["mu"], fit["sd"], fit["pi"], fit["N"]
    p_of = lambda x: 1 / (1 + np.exp(-x))
    print(f"N = {N}")
    print(f"LOW  comp: pi={pi[0]:.3f}  mu_logit={mu[0]:+.2f} (p={p_of(mu[0]):.3f})  sd={sd[0]:.2f}")
    print(f"HIGH comp: pi={pi[1]:.3f}  mu_logit={mu[1]:+.2f} (p={p_of(mu[1]):.3f})  sd={sd[1]:.2f}")
    print(f"HIGH component mass: {pi[1]*N:.0f} objects ({pi[1]*100:.1f}%)")
    lx, px = crossover(fit)
    print(f"\ncrossover  P(high)=0.5 : prob>{px:.3f}  -> {(fit['logit']>lx).sum()} flagged")
    for a in (0.5, 0.2, 0.1):
        t = lfdr_threshold(fit, a)
        if t:
            print(f"local-FDR<{a:<4}      : prob>{t[1]:.3f}  -> {(fit['logit']>t[0]).sum()} flagged")
    print("\nincompleteness of the HIGH (changed) component:")
    for pc in (0.8, 0.7, 0.6, 0.5):
        below, tot, frac = incompleteness(fit, pc)
        print(f"  cut p>{pc}: ~{below:.0f}/{tot:.0f} below cut ({frac*100:.0f}% of changed-mode missed)")


# ---------- plot ---------------------------------------------------------------
def plot(fit, out="mixture_fit.png", prob_cut=0.8, title=None):
    mu, sd, pi = fit["mu"], fit["sd"], fit["pi"]
    p = 1 / (1 + np.exp(-fit["logit"]))
    pp = np.linspace(1e-3, 1 - 1e-3, 600)
    lp = np.log(pp / (1 - pp))
    jac = 1 / (pp * (1 - pp))                # change-of-variables: density over p
    c0 = pi[0] * _npdf(lp, mu[0], sd[0]) * jac
    c1 = pi[1] * _npdf(lp, mu[1], sd[1]) * jac

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(p, bins=100, density=True, color="#ccc", alpha=0.8, label="all scores")
    ax.plot(pp, c0, "b-", lw=2, label=f'"stable"  {pi[0]*100:.0f}% (p≈{1/(1+np.exp(-mu[0])):.2f})')
    ax.plot(pp, c1, "r-", lw=2, label=f'"changed" {pi[1]*100:.0f}% (p≈{1/(1+np.exp(-mu[1])):.2f})')
    ax.plot(pp, c0 + c1, "k--", lw=1, alpha=0.6, label="mixture")
    px = crossover(fit)[1]
    ax.axvline(px, color="purple", ls=":", lw=1.5, label=f"crossover p={px:.2f}")
    ax.axvline(prob_cut, color="green", ls=":", lw=1.5, label=f"cut p={prob_cut}")
    ax.set_xlabel("probability"); ax.set_ylabel("density")
    ax.set_ylim(0, np.percentile(np.r_[c0, c1], 99) * 1.5)
    ax.set_title(title or "Two-component mixture (logit-space EM)")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


# ---------- IDE / CLI Config --------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dynamic check for prediction files
_pred_file = Path('/Users/amir/Documents/Deep learning/cl-agn-classifier/results/predictions_dr16_dr20bhm_regular_loss_fixed_pred.csv')
if not _pred_file.exists():
    _pred_file = PROJECT_ROOT / "results" / "predictions.csv"

IDE_CONFIG = dict(
    csv = str(_pred_file),
    col = "prob",
    out = str('/Users/amir/Documents/Deep learning/cl-agn-classifier/results/mixture_fit_dr16_dr20bhm_regular_loss_fixed_pred.png'),
    restarts = 10,
    cut = 0.8,
)


def main():
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("csv")
        ap.add_argument("--col", default="prob", help="probability column name")
        ap.add_argument("--out", default="mixture_fit.png")
        ap.add_argument("--restarts", type=int, default=10)
        ap.add_argument("--cut", type=float, default=0.8, help="reference cut to draw/report")
        args = ap.parse_args()

        csv_path = args.csv
        col = args.col
        out = args.out
        restarts = args.restarts
        cut = args.cut
    else:
        print("[gaussian_mixture_fit] No CLI arguments provided, using IDE_CONFIG:")
        for k, v in IDE_CONFIG.items():
            print(f"  {k} = {v}")
        csv_path = IDE_CONFIG["csv"]
        col = IDE_CONFIG["col"]
        out = IDE_CONFIG["out"]
        restarts = IDE_CONFIG["restarts"]
        cut = IDE_CONFIG["cut"] 

    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at '{csv_path}'. Please check your path.")
        sys.exit(1)

    p = pd.read_csv(csv_path)[col].values
    fit = fit_mixture(p, restarts=restarts)
    report(fit)
    plot(fit, out=out, prob_cut=cut, title=None)


if __name__ == "__main__":
    main()