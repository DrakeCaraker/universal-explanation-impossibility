"""
SNR Calibration: Theory vs Empirical Flip Rates
================================================
Calibrates the theoretical flip rate formula Φ(-SNR) against empirical data
across multiple datasets. Bridges the gap between the exact theory
(flip rate = 1/2 at SNR=0) and practice.

Usage: python paper/scripts/snr_calibration.py
Output: paper/figures/snr_calibration.pdf
        paper/results_snr_calibration.txt
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import os
warnings.filterwarnings("ignore")

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from sklearn.datasets import (
    load_breast_cancer, load_wine, load_diabetes,
    load_digits, fetch_california_housing, load_iris
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, pearsonr
import xgboost as xgb
import shap
import itertools
import os

# --- Publication-quality matplotlib settings ---
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["axes.labelsize"] = 13
rcParams["axes.titlesize"] = 14
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["legend.fontsize"] = 10
rcParams["figure.dpi"] = 150

# --- Config ---
N_MODELS = 50
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    verbosity=0,
)
RHO_THRESHOLD = 0.3
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results_snr_calibration.txt")

DATASET_STYLES = {
    "Breast Cancer": {"color": "#e41a1c", "marker": "o"},
    "Wine":          {"color": "#377eb8", "marker": "s"},
    "Diabetes":      {"color": "#4daf4a", "marker": "^"},
    "Digits (0v1)":  {"color": "#984ea3", "marker": "D"},
    "CA Housing":    {"color": "#ff7f00", "marker": "P"},
    "Iris (bin)":    {"color": "#a65628", "marker": "X"},
}


def load_datasets():
    """Load and prepare all six datasets as (X, y, name) tuples."""
    datasets = []

    # 1. Breast Cancer (30 features, binary)
    bc = load_breast_cancer()
    datasets.append((bc.data, bc.target.astype(float), "Breast Cancer"))

    # 2. Wine (13 features, binary: class 0 vs rest)
    wine = load_wine()
    y_wine = (wine.target == 0).astype(float)
    datasets.append((wine.data, y_wine, "Wine"))

    # 3. Diabetes (10 features, regression → binary via median split)
    diab = load_diabetes()
    y_diab = (diab.target > np.median(diab.target)).astype(float)
    datasets.append((diab.data, y_diab, "Diabetes"))

    # 4. Digits binary: 0 vs 1 (64 features, subsample to 500 rows)
    dig = load_digits()
    mask = (dig.target == 0) | (dig.target == 1)
    X_dig, y_dig = dig.data[mask], (dig.target[mask] == 1).astype(float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_dig), size=min(500, len(X_dig)), replace=False)
    datasets.append((X_dig[idx], y_dig[idx], "Digits (0v1)"))

    # 5. California Housing (8 features, subsample to 5000 rows, binary via median)
    cal = fetch_california_housing()
    rng2 = np.random.default_rng(1)
    idx2 = rng2.choice(len(cal.data), size=5000, replace=False)
    X_cal = cal.data[idx2]
    y_cal = (cal.target[idx2] > np.median(cal.target[idx2])).astype(float)
    datasets.append((X_cal, y_cal, "CA Housing"))

    # 6. Iris binary: setosa vs non-setosa (4 features)
    iris = load_iris()
    y_iris = (iris.target == 0).astype(float)
    datasets.append((iris.data, y_iris, "Iris (bin)"))

    return datasets


def compute_snr_fliprate(X_test, shap_values_all):
    """
    For each feature pair (j,k) with |ρ| > threshold:
      SNR  = |mean_Δ| / std_Δ   where Δ_m = mean_i(SHAP_j(m,i) - SHAP_k(m,i))
      flip = min(#j>k, #k>j) / N_MODELS

    shap_values_all: array of shape (N_MODELS, n_test, n_features)
    Returns: list of (snr, flip_rate) tuples
    """
    n_models, n_test, n_features = shap_values_all.shape
    results = []

    # Per-model mean SHAP for each feature: shape (N_MODELS, n_features)
    mean_shap = shap_values_all.mean(axis=1)  # (N_MODELS, n_features)

    # Correlation matrix on test set (use first model's SHAP as representative data)
    # We use the raw features for collinearity filter
    corr_mat = np.corrcoef(X_test.T)

    for j, k in itertools.combinations(range(n_features), 2):
        rho = corr_mat[j, k]
        if abs(rho) <= RHO_THRESHOLD:
            continue

        # Per-model mean SHAP difference: shape (N_MODELS,)
        delta = mean_shap[:, j] - mean_shap[:, k]

        mean_delta = delta.mean()
        std_delta = delta.std(ddof=1)

        if std_delta < 1e-12:
            continue

        snr = abs(mean_delta) / std_delta

        # Flip rate: fraction of models where j and k are "swapped"
        j_beats_k = np.sum(mean_shap[:, j] > mean_shap[:, k])
        k_beats_j = n_models - j_beats_k
        flip_rate = min(j_beats_k, k_beats_j) / n_models

        results.append((snr, flip_rate))

    return results


def process_dataset(X, y, name):
    """Train N_MODELS XGBoost models, compute SHAP, return (snr, flip) pairs."""
    print(f"\n--- {name} | n={len(X)}, p={X.shape[1]} ---")

    scaler = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    shap_all = []
    for seed in range(N_MODELS):
        if (seed + 1) % 10 == 0:
            print(f"  model {seed+1}/{N_MODELS}", flush=True)

        params = dict(XGB_PARAMS, random_state=seed)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te)
        # xgboost binary returns shape (n_test, n_features) or list
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        shap_all.append(sv)

    shap_all = np.array(shap_all)  # (N_MODELS, n_test, n_features)
    print(f"  SHAP array: {shap_all.shape}")

    pairs = compute_snr_fliprate(X_te, shap_all)
    print(f"  Pairs with |ρ|>{RHO_THRESHOLD}: {len(pairs)}")
    return pairs


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    datasets = load_datasets()

    all_snr = []
    all_flip = []
    all_labels = []
    per_dataset = {}

    for X, y, name in datasets:
        pairs = process_dataset(X, y, name)
        if pairs:
            snr_vals, flip_vals = zip(*pairs)
            all_snr.extend(snr_vals)
            all_flip.extend(flip_vals)
            all_labels.extend([name] * len(pairs))
            per_dataset[name] = (list(snr_vals), list(flip_vals))

    all_snr = np.array(all_snr)
    all_flip = np.array(all_flip)

    # --- Theoretical curve ---
    snr_curve = np.linspace(0, max(all_snr.max(), 3.5), 300)
    theory_curve = norm.cdf(-snr_curve)  # Φ(-SNR)

    # --- Fit statistics ---
    theory_at_data = norm.cdf(-all_snr)
    corr, p_corr = pearsonr(theory_at_data, all_flip)
    ss_res = np.sum((all_flip - theory_at_data) ** 2)
    ss_tot = np.sum((all_flip - all_flip.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(all_flip - theory_at_data))

    # Fraction of pairs with SNR > 1.96 that have flip < 5%
    mask_high_snr = all_snr > 1.96
    n_high = mask_high_snr.sum()
    n_low_flip = (all_flip[mask_high_snr] < 0.05).sum() if n_high > 0 else 0
    frac_low_flip = n_low_flip / n_high if n_high > 0 else float("nan")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for name, (snr_vals, flip_vals) in per_dataset.items():
        style = DATASET_STYLES[name]
        ax.scatter(
            snr_vals, flip_vals,
            color=style["color"], marker=style["marker"],
            s=18, alpha=0.45, linewidths=0.3, edgecolors="none",
            label=name, zorder=3,
        )

    # Theoretical curve
    ax.plot(
        snr_curve, theory_curve,
        color="black", linewidth=2.2, label=r"Theory: $\Phi(-\mathrm{SNR})$", zorder=5
    )

    # Vertical dashed lines
    ax.axvline(1.28, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8, zorder=4)
    ax.axvline(1.96, color="#222222", linestyle="--", linewidth=1.2, alpha=0.8, zorder=4)
    ax.text(1.28 + 0.08, 0.45, "SNR=1.28\n(10\% flip)", fontsize=7.5, color="#555555", va="top")
    ax.text(1.96 + 0.08, 0.40, "SNR=1.96\n(2.5\% flip)", fontsize=7.5, color="#222222", va="top")

    ax.set_xlabel(r"Signal-to-noise ratio $|\Delta|/\sigma$")
    ax.set_ylabel("Pairwise flip rate")
    ax.set_title("Universal SNR Calibration: Theory vs Empirical")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 0.55)

    # Stats annotation
    stats_text = f"$n={len(all_snr)}$ pairs across 6 datasets"
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes, fontsize=9,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.78),
        framealpha=0.9, edgecolor="#cccccc",
        ncol=1,
    )
    ax.grid(True, alpha=0.2, linewidth=0.3, color='#cccccc')
    fig.tight_layout()

    out_pdf = os.path.join(FIGURES_DIR, "snr_calibration.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nFigure saved: {out_pdf}")

    # --- Results text file ---
    lines = ["SNR Calibration Results", "=" * 50, ""]
    lines.append(f"Total (SNR, flip_rate) pairs: {len(all_snr)}")
    lines.append("")
    lines.append("Points per dataset:")
    for name, (sv, _) in per_dataset.items():
        lines.append(f"  {name:20s}: {len(sv)}")
    lines.append("")
    lines.append(f"Overall Pearson correlation (Φ(-SNR) vs flip): {corr:.4f}  (p={p_corr:.2e})")
    lines.append(f"R² of theoretical curve:                         {r2:.4f}")
    lines.append(f"Mean absolute error:                              {mae:.4f}")
    lines.append("")
    lines.append(f"Pairs with SNR > 1.96: {n_high}")
    lines.append(f"  Of those with flip rate < 5%: {n_low_flip} ({100*frac_low_flip:.1f}%)")
    lines.append("")

    # SNR percentile breakdown
    lines.append("Empirical vs theoretical flip rates by SNR bin:")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.28), (1.28, 1.96), (1.96, 3.0), (3.0, np.inf)]
    for lo, hi in bins:
        mask = (all_snr >= lo) & (all_snr < hi)
        if mask.sum() == 0:
            continue
        emp_mean = all_flip[mask].mean()
        mid = (lo + hi) / 2 if hi < np.inf else lo + 1.0
        th_mean = norm.cdf(-mid)
        label = f"[{lo:.2f}, {hi:.2f})" if hi < np.inf else f"[{lo:.2f}, ∞)"
        lines.append(
            f"  SNR {label:18s}  n={mask.sum():4d}  emp={emp_mean:.3f}  theory={th_mean:.3f}"
        )

    results_text = "\n".join(lines)
    with open(RESULTS_PATH, "w") as f:
        f.write(results_text)
    print(f"Results saved: {RESULTS_PATH}")
    print()
    print(results_text)


if __name__ == "__main__":
    main()
