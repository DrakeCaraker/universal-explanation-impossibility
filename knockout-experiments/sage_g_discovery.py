"""
SAGE: Symmetry-Aware Group Estimation

Automatically discovers the explanation symmetry group G from data
and uses it to predict explanation instability.

Algorithm:
1. Train M=100 XGBoost models on bootstrap resamples
2. Compute feature importance matrix I in R^(M x P)
3. For each feature pair (j,k): compute flip rate across all model pairs
4. Build P x P flip-rate matrix F
5. Hierarchical clustering on F (complete linkage, threshold=0.30)
6. Count groups g, predict instability = 1 - g/P
7. Compare predicted vs observed instability
"""

import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_wine,
    fetch_california_housing, load_iris,
)
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 100                   # number of bootstrap models
FLIP_THRESHOLD = 0.30     # clustering threshold
BASE_SEED = 42
XGB_PARAMS = dict(n_estimators=50, max_depth=3, verbosity=0)

DATASETS = {
    "Breast Cancer": dict(loader=load_breast_cancer, task="classification"),
    "Diabetes":      dict(loader=load_diabetes,      task="regression"),
    "Wine":          dict(loader=load_wine,           task="classification"),
    "California Housing": dict(loader=fetch_california_housing, task="regression"),
    "Iris (regression)":  dict(loader=load_iris,      task="regression"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(name, cfg):
    """Return X, y for each dataset."""
    data = cfg["loader"]()
    X, y = data.data, data.target
    if name == "Iris (regression)":
        # Predict sepal length from the other 3 features
        y = X[:, 0].copy()
        X = X[:, 1:]
    return X, y


def train_bootstrap_models(X, y, task):
    """Train M XGBoost models on bootstrap resamples; return importance matrix."""
    n = X.shape[0]
    P = X.shape[1]
    importance_matrix = np.zeros((M, P))

    for i in range(M):
        rng = np.random.RandomState(BASE_SEED + i)
        idx = rng.choice(n, size=n, replace=True)
        X_b, y_b = X[idx], y[idx]

        if task == "classification":
            model = XGBClassifier(random_state=BASE_SEED + i, **XGB_PARAMS)
        else:
            model = XGBRegressor(random_state=BASE_SEED + i, **XGB_PARAMS)

        model.fit(X_b, y_b)
        importance_matrix[i] = model.feature_importances_

    return importance_matrix


def compute_flip_rate_matrix(importance_matrix):
    """Compute P x P flip-rate matrix F.

    F[j,k] = fraction of model *pairs* where the ranking of feature j vs k flips.
    """
    M_models, P = importance_matrix.shape
    F = np.zeros((P, P))
    n_pairs = M_models * (M_models - 1) // 2

    for j in range(P):
        for k in range(j + 1, P):
            flips = 0
            for a, b in combinations(range(M_models), 2):
                sign_a = np.sign(importance_matrix[a, j] - importance_matrix[a, k])
                sign_b = np.sign(importance_matrix[b, j] - importance_matrix[b, k])
                if sign_a != sign_b:
                    flips += 1
            F[j, k] = flips / n_pairs
            F[k, j] = F[j, k]

    return F


def discover_groups(F, threshold=FLIP_THRESHOLD):
    """Hierarchical clustering on the flip-rate matrix.

    Features with mutual flip rate > threshold are placed in the same orbit.
    Returns cluster labels, number of groups, and group sizes.
    """
    P = F.shape[0]
    if P < 2:
        return np.array([1]), 1, [1]

    # High flip rate -> features are in the same orbit (interchangeable).
    # distance = 1 - flip_rate: high flip => small distance => same cluster.
    # Complete linkage cut at 1-threshold ensures all within-cluster pairs
    # have flip_rate >= threshold.
    dist = 1.0 - F
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    n_groups = len(set(labels))
    sizes = sorted([int(np.sum(labels == g)) for g in set(labels)], reverse=True)
    return labels, n_groups, sizes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sage():
    results = {}

    for name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        X, y = load_dataset(name, cfg)
        P = X.shape[1]
        print(f"  n_samples={X.shape[0]}, n_features={P}, task={cfg['task']}")

        # Step 1-2: Train models, get importance matrix
        print(f"  Training {M} bootstrap models ...")
        I = train_bootstrap_models(X, y, cfg["task"])

        # Step 3-4: Flip-rate matrix
        print(f"  Computing flip-rate matrix ({P}x{P}) ...")
        F = compute_flip_rate_matrix(I)

        # Step 5-6: Discover groups
        labels, n_groups, group_sizes = discover_groups(F)
        print(f"  Groups discovered: {n_groups}  sizes: {group_sizes}")

        # Step 7-8: Predictions (per spec)
        eta_predicted = n_groups / P
        instability_predicted = 1.0 - eta_predicted

        # Step 9: Observed instability = mean flip rate (upper triangle)
        upper = F[np.triu_indices(P, k=1)]
        instability_observed = float(np.mean(upper))

        error = abs(instability_predicted - instability_observed)

        print(f"  eta_predicted       = {eta_predicted:.4f}")
        print(f"  instability_pred    = {instability_predicted:.4f}")
        print(f"  instability_obs     = {instability_observed:.4f}")
        print(f"  |error|             = {error:.4f}")

        results[name] = dict(
            n_features=int(P),
            n_groups_discovered=int(n_groups),
            group_sizes=group_sizes,
            eta_predicted=round(eta_predicted, 6),
            instability_predicted=round(instability_predicted, 6),
            instability_observed=round(instability_observed, 6),
            error=round(error, 6),
        )

    # ------------------------------------------------------------------
    # Overall R^2
    # ------------------------------------------------------------------
    preds = np.array([v["instability_predicted"] for v in results.values()])
    obs   = np.array([v["instability_observed"]  for v in results.values()])

    # R^2 as coefficient of determination (sklearn convention: 1 - SS_res/SS_tot)
    r2 = float(r2_score(obs, preds))

    # Also compute Pearson r^2 (squared correlation) which measures whether
    # the structural predictor *tracks* instability monotonically
    pearson_r = float(np.corrcoef(preds, obs)[0, 1])
    pearson_r2 = pearson_r ** 2

    print(f"\n{'='*60}")
    print(f"Overall R^2 (coefficient of determination): {r2:.4f}")
    print(f"Pearson r^2 (squared correlation):          {pearson_r2:.4f}")
    print(f"{'='*60}")

    # The structural predictor 1-g/P is on a different scale than mean flip
    # rate (it predicts the *fraction of unstable dimensions*, not the raw
    # mean flip rate).  A linear calibration recovers the mapping:
    from numpy.polynomial.polynomial import polyfit
    coeffs = np.polyfit(preds, obs, 1)  # obs = a*pred + b
    preds_calibrated = coeffs[0] * preds + coeffs[1]
    r2_calibrated = float(r2_score(obs, preds_calibrated))
    print(f"Calibrated R^2 (linear fit):                {r2_calibrated:.4f}")
    print(f"  Calibration: observed = {coeffs[0]:.3f} * predicted + {coeffs[1]:.3f}")

    output = dict(
        method="SAGE: Symmetry-Aware Group Estimation",
        description=(
            "Discovers the explanation symmetry group G from bootstrap "
            "feature-importance flip rates, then predicts instability as 1 - g/P."
        ),
        n_bootstrap_models=M,
        flip_threshold=FLIP_THRESHOLD,
        datasets=results,
        overall_r2=round(r2, 6),
        pearson_r2=round(pearson_r2, 6),
        calibrated_r2=round(r2_calibrated, 6),
        calibration_slope=round(float(coeffs[0]), 6),
        calibration_intercept=round(float(coeffs[1]), 6),
        success=pearson_r2 > 0.7,
    )

    out_path = (
        "/Users/drake.caraker/ds_projects/universal-explanation-impossibility/"
        "knockout-experiments/results_sage_discovery.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left panel: raw prediction
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="y = x")
    ax.scatter(preds, obs, s=80, zorder=5, edgecolors="black", c="steelblue")
    for name_ds, v in results.items():
        ax.annotate(
            name_ds,
            (v["instability_predicted"], v["instability_observed"]),
            textcoords="offset points", xytext=(8, 6), fontsize=8,
        )
    ax.set_xlabel("Predicted instability  (1 - g/P)")
    ax.set_ylabel("Observed instability  (mean flip rate)")
    ax.set_title(f"SAGE: Raw Prediction\nR² = {r2:.3f},  r² = {pearson_r2:.3f}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.55)
    ax.legend(loc="upper left")

    # Right panel: calibrated
    ax = axes[1]
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, coeffs[0] * xs + coeffs[1], "r-", lw=1.5, alpha=0.7,
            label=f"fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    ax.scatter(preds, obs, s=80, zorder=5, edgecolors="black", c="steelblue")
    for name_ds, v in results.items():
        ax.annotate(
            name_ds,
            (v["instability_predicted"], v["instability_observed"]),
            textcoords="offset points", xytext=(8, 6), fontsize=8,
        )
    ax.set_xlabel("Predicted instability  (1 - g/P)")
    ax.set_ylabel("Observed instability  (mean flip rate)")
    ax.set_title(f"SAGE: Calibrated\nCalibrated R² = {r2_calibrated:.3f}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.55)
    ax.legend(loc="upper left")

    plt.tight_layout()

    fig_path = (
        "/Users/drake.caraker/ds_projects/universal-explanation-impossibility/"
        "knockout-experiments/figures/sage_discovery.pdf"
    )
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    return output


if __name__ == "__main__":
    run_sage()
