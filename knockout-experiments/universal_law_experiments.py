#!/usr/bin/env python3
"""
Universal Law Experiments for Explanation Reliability
=====================================================

Experiment 1: Test whether f(r_eff / P) predicts the fraction of unreliable
              feature-importance pairs across datasets (Path 1).

Experiment 2: Test whether the enrichment knee point converges to a universal
              constant across datasets (Path 6).
"""

import json
import warnings
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris,
    fetch_california_housing, load_diabetes as load_sk_diabetes,
    fetch_openml, make_classification,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy import stats

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def effective_rank(X, threshold=0.90):
    """Number of PCA components explaining `threshold` fraction of variance."""
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    r_eff = int(np.searchsorted(cumvar, threshold) + 1)
    return min(r_eff, X.shape[1])


def is_classification(y):
    """Heuristic: classification if <=20 unique values."""
    return len(np.unique(y)) <= 20


def bootstrap_importances(X, y, n_boot=30, random_state=42):
    """Train n_boot XGBoost models on bootstrap samples, return importances matrix."""
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    imp_matrix = []
    clf_task = is_classification(y)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_b, y_b = X[idx], y[idx]
        if clf_task:
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=random_state + i, verbosity=0,
                use_label_encoder=False, eval_metric="logloss",
            )
        else:
            model = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=random_state + i, verbosity=0,
            )
        model.fit(X_b, y_b)
        imp_matrix.append(model.feature_importances_)
    return np.array(imp_matrix)  # shape (n_boot, P)


def pairwise_snr(imp_matrix):
    """Compute SNR for all feature pairs: |mean(d_ij)| / std(d_ij).
    Returns array of SNR values and count of unreliable pairs (SNR < 0.5)."""
    P = imp_matrix.shape[1]
    snrs = []
    for i in range(P):
        for j in range(i + 1, P):
            diff = imp_matrix[:, i] - imp_matrix[:, j]
            mu = np.mean(diff)
            sigma = np.std(diff, ddof=1)
            snr = abs(mu) / sigma if sigma > 1e-12 else np.inf
            snrs.append(snr)
    snrs = np.array(snrs)
    n_pairs = len(snrs)
    n_unreliable = np.sum(snrs < 0.5)
    frac_unreliable = n_unreliable / n_pairs if n_pairs > 0 else 0.0
    return snrs, frac_unreliable


# ── Build dataset list ───────────────────────────────────────────────────────

def build_datasets():
    datasets = []
    # sklearn built-ins
    datasets.append(("Breast Cancer", *load_breast_cancer(return_X_y=True)))
    datasets.append(("Wine", *load_wine(return_X_y=True)))
    datasets.append(("CalHousing", *fetch_california_housing(return_X_y=True)))
    datasets.append(("Iris", *load_iris(return_X_y=True)))
    datasets.append(("SK-Diabetes", *load_sk_diabetes(return_X_y=True)))

    # OpenML
    for name, oml in [("Diabetes", "diabetes"), ("Heart", "heart-statlog")]:
        try:
            X, y = fetch_openml(oml, version=1, return_X_y=True, as_frame=False, parser="auto")
            y = LabelEncoder().fit_transform(y.astype(str))
            datasets.append((name, X, y))
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    # Synthetic with varying r/P
    for r_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        n_inform = int(50 * r_frac)
        n_redund = int(50 * (1 - r_frac) * 0.5)
        X, y = make_classification(
            n_features=50, n_informative=max(2, n_inform),
            n_redundant=max(0, n_redund), n_clusters_per_class=1,
            random_state=42, n_samples=1000,
        )
        datasets.append((f"Synth r/P~{r_frac}", X, y))

    return datasets


# ── EXPERIMENT 1: f(r/P) universal law ───────────────────────────────────────

def experiment1(datasets):
    print("=" * 70)
    print("EXPERIMENT 1: Universal Law f(r/P)")
    print("=" * 70)

    results = []
    for name, X, y in datasets:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Drop NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        P = X.shape[1]
        r_eff = effective_rank(X)
        ratio = r_eff / P

        print(f"\n  {name}: N={X.shape[0]}, P={P}, r_eff={r_eff}, r/P={ratio:.3f}")

        imp_matrix = bootstrap_importances(X, y, n_boot=30)
        snrs, frac_unreliable = pairwise_snr(imp_matrix)

        print(f"    fraction unreliable (SNR<0.5): {frac_unreliable:.3f}")

        results.append({
            "dataset": name,
            "N": int(X.shape[0]),
            "P": P,
            "r_eff": r_eff,
            "r_over_P": round(ratio, 4),
            "frac_unreliable": round(frac_unreliable, 4),
            "median_snr": round(float(np.median(snrs)), 4),
        })

    # Correlation analysis
    ratios = np.array([r["r_over_P"] for r in results])
    fracs = np.array([r["frac_unreliable"] for r in results])
    slope, intercept, r_value, p_value, se = stats.linregress(ratios, fracs)
    r_squared = r_value ** 2

    # Also try Spearman (monotonic, not necessarily linear)
    rho_spearman, p_spearman = stats.spearmanr(ratios, fracs)

    summary = {
        "pearson_r": round(r_value, 4),
        "R_squared": round(r_squared, 4),
        "p_value": round(p_value, 6),
        "spearman_rho": round(rho_spearman, 4),
        "spearman_p": round(p_spearman, 6),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
    }

    print(f"\n  ── Correlation summary ──")
    print(f"    Pearson r   = {r_value:.4f}, R² = {r_squared:.4f}, p = {p_value:.4g}")
    print(f"    Spearman rho= {rho_spearman:.4f}, p = {p_spearman:.4g}")
    print(f"    Regression: frac_unreliable = {slope:.4f} * (r/P) + {intercept:.4f}")

    return results, summary, (ratios, fracs, slope, intercept)


# ── EXPERIMENT 2: Universal Knee Point ───────────────────────────────────────

def enrichment_tradeoff(imp_matrix, thresholds):
    """At each threshold, compute (frac_retained, stability).
    Stability = mean pairwise Spearman rho among bootstrap rankings
    restricted to features whose mean importance > threshold."""
    from scipy.stats import spearmanr as sp_rho

    P = imp_matrix.shape[1]
    mean_imp = imp_matrix.mean(axis=0)

    results = []
    for thr in thresholds:
        mask = mean_imp >= thr
        n_retained = mask.sum()
        frac_retained = n_retained / P

        if n_retained < 2:
            results.append((thr, frac_retained, np.nan))
            continue

        # Stability: mean pairwise Spearman among bootstraps for retained features
        sub = imp_matrix[:, mask]
        n_boot = sub.shape[0]
        rhos = []
        for i in range(n_boot):
            for j in range(i + 1, n_boot):
                rho_val, _ = sp_rho(sub[i], sub[j])
                rhos.append(rho_val)
        stability = np.mean(rhos) if rhos else np.nan
        results.append((thr, frac_retained, stability))

    return results


def find_knee(tradeoff_results):
    """Find the knee point: threshold where d(stability)/d(threshold) is steepest."""
    valid = [(t, fr, s) for t, fr, s in tradeoff_results if not np.isnan(s) and fr > 0]
    if len(valid) < 3:
        return np.nan

    thresholds = np.array([v[0] for v in valid])
    stabilities = np.array([v[2] for v in valid])

    # Numerical first derivative (stability w.r.t. threshold)
    d_stab = np.diff(stabilities)
    d_thr = np.diff(thresholds)
    deriv = d_stab / np.where(np.abs(d_thr) < 1e-12, 1e-12, d_thr)

    # Knee = where the derivative is maximal (steepest *improvement* in stability)
    idx = np.argmax(deriv)
    knee_threshold = (thresholds[idx] + thresholds[idx + 1]) / 2.0
    return knee_threshold


def experiment2(datasets_subset):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Universal Knee-Point Constant")
    print("=" * 70)

    thresholds = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50])

    results = []
    tradeoff_data = {}  # for plotting

    for name, X, y in datasets_subset:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        print(f"\n  {name}: training 100 bootstrap models...")
        imp_matrix = bootstrap_importances(X, y, n_boot=100)

        tradeoff = enrichment_tradeoff(imp_matrix, thresholds)
        knee = find_knee(tradeoff)

        tradeoff_data[name] = tradeoff

        print(f"    Knee-point threshold: {knee:.4f}")
        results.append({
            "dataset": name,
            "knee_threshold": round(float(knee), 4) if not np.isnan(knee) else None,
            "tradeoff": [(round(t, 4), round(fr, 4), round(s, 4) if not np.isnan(s) else None)
                         for t, fr, s in tradeoff],
        })

    knee_vals = [r["knee_threshold"] for r in results if r["knee_threshold"] is not None]
    if knee_vals:
        knee_mean = np.mean(knee_vals)
        knee_std = np.std(knee_vals)
        knee_range = (min(knee_vals), max(knee_vals))
        knee_cv = knee_std / knee_mean if knee_mean > 0 else np.inf
    else:
        knee_mean = knee_std = knee_cv = np.nan
        knee_range = (np.nan, np.nan)

    summary = {
        "knee_values": knee_vals,
        "knee_mean": round(float(knee_mean), 4),
        "knee_std": round(float(knee_std), 4),
        "knee_range": [round(float(knee_range[0]), 4), round(float(knee_range[1]), 4)],
        "knee_cv": round(float(knee_cv), 4),
        "converges": bool(knee_cv < 0.5),  # CV < 50% as rough convergence criterion
    }

    print(f"\n  ── Knee-point summary ──")
    print(f"    Values: {[round(k, 4) for k in knee_vals]}")
    print(f"    Mean = {knee_mean:.4f}, Std = {knee_std:.4f}, CV = {knee_cv:.4f}")
    print(f"    Range = [{knee_range[0]:.4f}, {knee_range[1]:.4f}]")
    print(f"    Converges (CV<0.5)? {summary['converges']}")

    return results, summary, tradeoff_data


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(exp1_plot, exp2_tradeoff, exp2_summary, exp1_summary):
    ratios, fracs, slope, intercept = exp1_plot

    with PdfPages(str(FIG_DIR / "universal_law.pdf")) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # ── Panel A: f(r/P) scatter ──
        ax = axes[0]
        ax.scatter(ratios, fracs, s=60, edgecolors="black", zorder=3, alpha=0.8)
        x_fit = np.linspace(ratios.min() - 0.05, ratios.max() + 0.05, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "r--", lw=2, label=(
            f"OLS: y = {slope:.2f}x + {intercept:.2f}\n"
            f"R² = {exp1_summary['R_squared']:.3f}, p = {exp1_summary['p_value']:.2g}"
        ))
        ax.set_xlabel("Effective rank ratio  r_eff / P", fontsize=12)
        ax.set_ylabel("Fraction unreliable pairs (SNR < 0.5)", fontsize=12)
        ax.set_title("A) Universal Law: f(r/P) predicts unreliability", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── Panel B: Knee-point comparison ──
        ax = axes[1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(exp2_tradeoff)))
        for idx, (name, tradeoff) in enumerate(exp2_tradeoff.items()):
            thrs = [t for t, fr, s in tradeoff if not np.isnan(s)]
            stabs = [s for t, fr, s in tradeoff if not np.isnan(s)]
            ax.plot(thrs, stabs, "o-", color=colors[idx], label=name, markersize=5)

        # Mark knee points
        knee_vals = exp2_summary["knee_values"]
        if knee_vals:
            for kv in knee_vals:
                ax.axvline(kv, color="gray", alpha=0.3, ls=":")
            ax.axvline(exp2_summary["knee_mean"], color="red", ls="--", lw=2,
                       label=f"Mean knee = {exp2_summary['knee_mean']:.3f} "
                             f"(CV={exp2_summary['knee_cv']:.2f})")

        ax.set_xlabel("Importance threshold", fontsize=12)
        ax.set_ylabel("Stability (mean pairwise Spearman)", fontsize=12)
        ax.set_title("B) Enrichment knee point across datasets", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n  Figure saved: {FIG_DIR / 'universal_law.pdf'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building datasets...")
    datasets = build_datasets()
    print(f"  {len(datasets)} datasets loaded.\n")

    # Experiment 1: all datasets
    exp1_results, exp1_summary, exp1_plot = experiment1(datasets)

    # Experiment 2: subset of 5 real datasets
    exp2_names = {"Breast Cancer", "Wine", "CalHousing", "Diabetes", "Heart"}
    exp2_datasets = [(n, X, y) for n, X, y in datasets if n in exp2_names]
    # Fallback: if some OpenML failed, use what we have
    if len(exp2_datasets) < 3:
        exp2_datasets = [(n, X, y) for n, X, y in datasets if "Synth" not in n][:5]

    exp2_results, exp2_summary, exp2_tradeoff = experiment2(exp2_datasets)

    # ── Save results ──
    output = {
        "experiment1": {
            "description": "Universal law f(r/P): does effective-rank ratio predict unreliability?",
            "per_dataset": exp1_results,
            "summary": exp1_summary,
        },
        "experiment2": {
            "description": "Universal knee-point constant in enrichment tradeoff",
            "per_dataset": exp2_results,
            "summary": exp2_summary,
        },
    }

    out_path = OUT_DIR / "results_universal_law.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ── Plot ──
    make_figure(exp1_plot, exp2_tradeoff, exp2_summary, exp1_summary)

    # ── Final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"\n  (1) f(r/P) universal law:")
    print(f"      R² = {exp1_summary['R_squared']:.4f}")
    print(f"      Evidence for universal law (R²>0.7)? "
          f"{'YES' if exp1_summary['R_squared'] > 0.7 else 'NO'}")
    print(f"      Spearman rho = {exp1_summary['spearman_rho']:.4f} "
          f"(p={exp1_summary['spearman_p']:.4g})")

    print(f"\n  (2) Universal knee-point constant:")
    print(f"      Knee values: {[round(k, 4) for k in exp2_summary['knee_values']]}")
    print(f"      Mean = {exp2_summary['knee_mean']:.4f}, "
          f"Std = {exp2_summary['knee_std']:.4f}")
    print(f"      Range = {exp2_summary['knee_range']}")
    print(f"      CV = {exp2_summary['knee_cv']:.4f}")
    print(f"      Converges to constant (CV<0.5)? "
          f"{'YES' if exp2_summary['converges'] else 'NO'}")

    return output


if __name__ == "__main__":
    main()
