#!/usr/bin/env python3
"""
Noether Counting Experiment v2 — Fixed design
==============================================
Tests: For P features in g correlation groups, within-group pairs are unstable
(~50% flip rate) while between-group pairs are stable (<5% flip rate).

Fix from v1: Use higher within-group correlation (0.99), weaker signal,
and SHAP-based importance (or coefficient-based for linear models where
the theory is exact).
"""

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.stats import spearmanr, binomtest
from sklearn.linear_model import Ridge

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIR / "results_noether_counting.json"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_FILE = FIGURES_DIR / "noether_counting.pdf"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

P = 12           # total features
G = 3            # number of groups
GROUP_SIZE = 4   # features per group
RHO_WITHIN = 0.99   # HIGH within-group correlation (was 0.9 in v1)
RHO_BETWEEN = 0.0   # zero between-group correlation
N_TRAIN = 500    # enough data for between-group stability
N_TEST = 100
NOISE_STD = 1.0

# Equal coefficients WITHIN each group, very different BETWEEN groups
# Large gaps ensure between-group rankings are stable
BETAS = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
N_MODELS = 200

def make_correlated_X(n, rng):
    """Generate features with block-diagonal correlation structure."""
    X = np.zeros((n, P))
    for g in range(G):
        start = g * GROUP_SIZE
        end = start + GROUP_SIZE
        # Generate correlated block
        cov = np.full((GROUP_SIZE, GROUP_SIZE), RHO_WITHIN)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((n, GROUP_SIZE))
        X[:, start:end] = Z @ L.T
    return X

def get_group(j):
    """Return group index for feature j."""
    return j // GROUP_SIZE

def compute_flip_rate(rankings):
    """Compute pairwise flip rate for all feature pairs across models.
    rankings: (n_models, P) array of ranks (1=most important)."""
    n_models = rankings.shape[0]
    results = {}
    for j in range(P):
        for k in range(j+1, P):
            flips = 0
            total = 0
            for a in range(n_models):
                for b in range(a+1, n_models):
                    # Does the ranking of j vs k flip between model a and model b?
                    if (rankings[a, j] < rankings[a, k]) != (rankings[b, j] < rankings[b, k]):
                        flips += 1
                    total += 1
            results[(j, k)] = flips / total if total > 0 else 0.0
    return results

def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    """Bootstrap confidence interval."""
    data = np.array(data)
    if len(data) == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2))]
    return lo, hi

# ---------------------------------------------------------------------------
# Experiment: Use Ridge regression (exact theory for linear models)
# ---------------------------------------------------------------------------
print("=== Noether Counting Experiment v2 ===")
print(f"P={P}, G={G}, group_size={GROUP_SIZE}")
print(f"rho_within={RHO_WITHIN}, rho_between={RHO_BETWEEN}")
print(f"N_train={N_TRAIN}, N_models={N_MODELS}")
print()

# Store importance rankings for each model
all_importances = np.zeros((N_MODELS, P))

for i in range(N_MODELS):
    rng_i = np.random.default_rng(SEED + i)
    X_train = make_correlated_X(N_TRAIN, rng_i)
    y_train = X_train @ BETAS + rng_i.standard_normal(N_TRAIN) * NOISE_STD

    # Ridge with small alpha — coefficients will be unstable for correlated features
    model = Ridge(alpha=0.01)
    model.fit(X_train, y_train)

    # Feature importance = |coefficient|
    all_importances[i] = np.abs(model.coef_)

# Convert to rankings (1 = most important)
all_rankings = np.zeros_like(all_importances, dtype=int)
for i in range(N_MODELS):
    all_rankings[i] = stats.rankdata(-all_importances[i], method='ordinal')

print("Computing pairwise flip rates for all 66 feature pairs...")
flip_rates = compute_flip_rate(all_rankings)

# Classify pairs
within_group_rates = []
between_group_rates = []
within_pairs = []
between_pairs = []

for (j, k), rate in flip_rates.items():
    if get_group(j) == get_group(k):
        within_group_rates.append(rate)
        within_pairs.append((j, k, rate))
    else:
        between_group_rates.append(rate)
        between_pairs.append((j, k, rate))

within_group_rates = np.array(within_group_rates)
between_group_rates = np.array(between_group_rates)

print(f"\n=== RESULTS ===")
print(f"Within-group pairs: {len(within_group_rates)}")
print(f"  Mean flip rate: {within_group_rates.mean():.4f}")
print(f"  Min: {within_group_rates.min():.4f}, Max: {within_group_rates.max():.4f}")
print(f"  CI: {bootstrap_ci(within_group_rates)}")

print(f"\nBetween-group pairs: {len(between_group_rates)}")
print(f"  Mean flip rate: {between_group_rates.mean():.4f}")
print(f"  Min: {between_group_rates.min():.4f}, Max: {between_group_rates.max():.4f}")
print(f"  CI: {bootstrap_ci(between_group_rates)}")

# Bimodality test: are within and between clearly separated?
all_rates = np.concatenate([within_group_rates, between_group_rates])
separation = within_group_rates.min() - between_group_rates.max()
print(f"\nSeparation gap: {separation:.4f}")
print(f"  within min: {within_group_rates.min():.4f}")
print(f"  between max: {between_group_rates.max():.4f}")

# Mann-Whitney U test for difference
if len(within_group_rates) > 0 and len(between_group_rates) > 0:
    u_stat, u_p = stats.mannwhitneyu(within_group_rates, between_group_rates, alternative='greater')
    print(f"\nMann-Whitney U (within > between): U={u_stat:.1f}, p={u_p:.2e}")
else:
    u_stat, u_p = 0, 1.0

# Noether prediction check
n_stable_between = np.sum(between_group_rates < 0.05)
n_unstable_within = np.sum(within_group_rates > 0.40)
noether_g = G * (G - 1) // 2

print(f"\n=== NOETHER PREDICTION ===")
print(f"Predicted independent stable group-level facts: {noether_g}")
print(f"Between-group pairs stable (<5%): {n_stable_between}/{len(between_group_rates)}")
print(f"Within-group pairs unstable (>40%): {n_unstable_within}/{len(within_group_rates)}")

prediction_confirmed = (
    n_stable_between == len(between_group_rates) and
    n_unstable_within == len(within_group_rates)
)
print(f"Prediction confirmed: {prediction_confirmed}")

# ---------------------------------------------------------------------------
# Benjamini-Hochberg correction: classify pairs as stable vs unstable
# ---------------------------------------------------------------------------
# Test H0: pair is stable (flip_rate = 0)
# HA: pair is unstable (flip_rate > 0)
n_model_comparisons = N_MODELS * (N_MODELS - 1) // 2
p_values_stability = []
pair_keys = []
for (j, k), rate in flip_rates.items():
    n_flips = int(round(rate * n_model_comparisons))
    if n_flips == 0:
        p_values_stability.append(1.0)
    else:
        result = binomtest(n_flips, n_model_comparisons, 0.0001, alternative='greater')
        p_values_stability.append(result.pvalue)
    pair_keys.append((j, k))

bh_results = {}
if HAS_STATSMODELS:
    reject_bh, corrected_p_bh, _, _ = multipletests(p_values_stability, alpha=0.05, method='fdr_bh')
    n_unstable_bh = int(sum(reject_bh))
    n_stable_bh = 66 - n_unstable_bh
    print(f"\n=== BENJAMINI-HOCHBERG CORRECTION ===")
    print(f"BH correction: {n_unstable_bh}/66 pairs classified as unstable (FDR=0.05)")
    print(f"BH correction: {n_stable_bh}/66 pairs classified as stable (FDR=0.05)")
    # Breakdown by group type
    bh_within_unstable = sum(1 for i, (j, k) in enumerate(pair_keys) if reject_bh[i] and get_group(j) == get_group(k))
    bh_between_unstable = sum(1 for i, (j, k) in enumerate(pair_keys) if reject_bh[i] and get_group(j) != get_group(k))
    bh_within_stable = sum(1 for i, (j, k) in enumerate(pair_keys) if not reject_bh[i] and get_group(j) == get_group(k))
    bh_between_stable = sum(1 for i, (j, k) in enumerate(pair_keys) if not reject_bh[i] and get_group(j) != get_group(k))
    print(f"  Within-group:  {bh_within_unstable} unstable, {bh_within_stable} stable")
    print(f"  Between-group: {bh_between_unstable} unstable, {bh_between_stable} stable")
    bh_results = {
        "n_unstable": n_unstable_bh,
        "n_stable": n_stable_bh,
        "within_group_unstable": int(bh_within_unstable),
        "within_group_stable": int(bh_within_stable),
        "between_group_unstable": int(bh_between_unstable),
        "between_group_stable": int(bh_between_stable),
        "corrected_p_values": {f"{pair_keys[i][0]},{pair_keys[i][1]}": float(corrected_p_bh[i]) for i in range(len(pair_keys))},
        "note": "H0: pair is stable (flip_rate~0). BH-corrected at FDR=0.05. Reject => unstable."
    }
else:
    print("\n[WARN] statsmodels not installed — skipping BH correction")

# Also try XGBoost to see if tree models show the same pattern
try:
    import xgboost as xgb
    print("\n=== XGBoost REPLICATION ===")
    xgb_importances = np.zeros((N_MODELS, P))

    for i in range(N_MODELS):
        rng_i = np.random.default_rng(SEED + i)
        X_train = make_correlated_X(N_TRAIN, rng_i)
        y_train = X_train @ BETAS + rng_i.standard_normal(N_TRAIN) * NOISE_STD

        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42 + i, verbosity=0,
            colsample_bytree=0.8  # Force feature subsampling
        )
        model.fit(X_train, y_train)
        xgb_importances[i] = model.feature_importances_

    xgb_rankings = np.zeros_like(xgb_importances, dtype=int)
    for i in range(N_MODELS):
        xgb_rankings[i] = stats.rankdata(-xgb_importances[i], method='ordinal')

    xgb_flip_rates = compute_flip_rate(xgb_rankings)

    xgb_within = [r for (j,k), r in xgb_flip_rates.items() if get_group(j) == get_group(k)]
    xgb_between = [r for (j,k), r in xgb_flip_rates.items() if get_group(j) != get_group(k)]
    xgb_within = np.array(xgb_within)
    xgb_between = np.array(xgb_between)

    print(f"XGBoost within-group mean flip: {xgb_within.mean():.4f} [{xgb_within.min():.4f}, {xgb_within.max():.4f}]")
    print(f"XGBoost between-group mean flip: {xgb_between.mean():.4f} [{xgb_between.min():.4f}, {xgb_between.max():.4f}]")

    has_xgb = True
except ImportError:
    has_xgb = False
    xgb_within = np.array([])
    xgb_between = np.array([])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2 if has_xgb else 1, figsize=(12 if has_xgb else 6, 5))
if not has_xgb:
    axes = [axes]

# Panel A: Ridge regression
ax = axes[0]
bins = np.linspace(0, 0.55, 25)
ax.hist(within_group_rates, bins=bins, alpha=0.7, color='#d62728', label=f'Within-group (n={len(within_group_rates)})', edgecolor='black', linewidth=0.5)
ax.hist(between_group_rates, bins=bins, alpha=0.7, color='#1f77b4', label=f'Between-group (n={len(between_group_rates)})', edgecolor='black', linewidth=0.5)
ax.axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='Stability threshold (5%)')
ax.set_xlabel('Pairwise flip rate', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Ridge Regression: Noether Counting', fontsize=13)
ax.legend(fontsize=10)

if has_xgb:
    ax2 = axes[1]
    ax2.hist(xgb_within, bins=bins, alpha=0.7, color='#d62728', label=f'Within-group (n={len(xgb_within)})', edgecolor='black', linewidth=0.5)
    ax2.hist(xgb_between, bins=bins, alpha=0.7, color='#1f77b4', label=f'Between-group (n={len(xgb_between)})', edgecolor='black', linewidth=0.5)
    ax2.axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='Stability threshold (5%)')
    ax2.set_xlabel('Pairwise flip rate', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('XGBoost: Noether Counting', fontsize=13)
    ax2.legend(fontsize=10)

plt.tight_layout()
fig.savefig(str(FIGURE_FILE), bbox_inches='tight', dpi=300)
print(f"\nSaved figure: {FIGURE_FILE}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "experiment": "noether_counting_v2",
    "config": {
        "P": P, "G": G, "group_size": GROUP_SIZE,
        "rho_within": RHO_WITHIN, "rho_between": RHO_BETWEEN,
        "betas": BETAS.tolist(),
        "n_train": N_TRAIN, "n_test": N_TEST, "noise_std": NOISE_STD,
        "n_models": N_MODELS, "seed": SEED,
        "model": "Ridge(alpha=0.01)"
    },
    "ridge_results": {
        "within_group": {
            "mean": float(within_group_rates.mean()),
            "ci_lower": float(bootstrap_ci(within_group_rates)[0]),
            "ci_upper": float(bootstrap_ci(within_group_rates)[1]),
            "min": float(within_group_rates.min()),
            "max": float(within_group_rates.max()),
            "n_pairs": len(within_group_rates),
            "n_unstable_gt40": int(n_unstable_within),
            "all_rates": within_group_rates.tolist()
        },
        "between_group": {
            "mean": float(between_group_rates.mean()),
            "ci_lower": float(bootstrap_ci(between_group_rates)[0]),
            "ci_upper": float(bootstrap_ci(between_group_rates)[1]),
            "min": float(between_group_rates.min()),
            "max": float(between_group_rates.max()),
            "n_pairs": len(between_group_rates),
            "n_stable_lt5": int(n_stable_between),
            "all_rates": between_group_rates.tolist()
        },
        "separation_gap": float(separation),
        "mann_whitney_U": float(u_stat),
        "mann_whitney_p": float(u_p),
    },
    "noether_prediction": {
        "g_g_minus_1_over_2": noether_g,
        "between_group_all_stable": bool(n_stable_between == len(between_group_rates)),
        "within_group_all_unstable": bool(n_unstable_within == len(within_group_rates)),
        "prediction_confirmed": bool(prediction_confirmed)
    },
    "benjamini_hochberg": bh_results,
}

if has_xgb:
    results["xgboost_results"] = {
        "within_group": {
            "mean": float(xgb_within.mean()),
            "min": float(xgb_within.min()),
            "max": float(xgb_within.max()),
            "n_pairs": len(xgb_within),
            "all_rates": xgb_within.tolist()
        },
        "between_group": {
            "mean": float(xgb_between.mean()),
            "min": float(xgb_between.min()),
            "max": float(xgb_between.max()),
            "n_pairs": len(xgb_between),
            "all_rates": xgb_between.tolist()
        }
    }

import time
results["_timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')

with open(str(RESULTS_FILE), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {RESULTS_FILE}")
