#!/usr/bin/env python3
"""
Noether Counting Sensitivity Analysis
======================================
Does the bimodal gap between within-group and between-group flip rates
survive at moderate correlations (rho=0.5, 0.7, 0.85)?

Design: P=12 features, g=3 groups of 4, beta=[5,5,5,5, 2,2,2,2, 0.5,0.5,0.5,0.5]
N_train=500, noise=1.0, 200 Ridge(alpha=0.01) models per rho value.
Permutation test (5000 permutations) for statistical rigor.
"""

import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import Ridge

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIR / "results_noether_sensitivity.json"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_FILE = FIGURES_DIR / "noether_sensitivity.pdf"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
P = 12
G = 3
GROUP_SIZE = 4
RHO_BETWEEN = 0.0
N_TRAIN = 500
NOISE_STD = 1.0
BETAS = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
N_MODELS = 200
N_PERM = 5000

RHO_VALUES = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def make_correlated_X(n, rho_within, rng):
    """Generate features with block-diagonal correlation structure."""
    X = np.zeros((n, P))
    for g in range(G):
        start = g * GROUP_SIZE
        end = start + GROUP_SIZE
        cov = np.full((GROUP_SIZE, GROUP_SIZE), rho_within)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((n, GROUP_SIZE))
        X[:, start:end] = Z @ L.T
    return X


def get_group(j):
    return j // GROUP_SIZE


def compute_flip_rate(rankings):
    """Compute pairwise flip rate for all feature pairs across models.
    rankings: (n_models, P) array of ranks (1=most important).
    Returns dict (j,k) -> flip_rate for all j<k."""
    n_models = rankings.shape[0]
    results = {}
    for j in range(P):
        for k in range(j + 1, P):
            flips = 0
            total = 0
            for a in range(n_models):
                for b in range(a + 1, n_models):
                    if (rankings[a, j] < rankings[a, k]) != (rankings[b, j] < rankings[b, k]):
                        flips += 1
                    total += 1
            results[(j, k)] = flips / total if total > 0 else 0.0
    return results


def classify_pairs(flip_rates):
    """Split flip rates into within-group and between-group."""
    within = []
    between = []
    for (j, k), rate in flip_rates.items():
        if get_group(j) == get_group(k):
            within.append(rate)
        else:
            between.append(rate)
    return np.array(within), np.array(between)


def permutation_test(within_rates, between_rates, n_perm, rng):
    """Permutation test: is the gap significant?
    Permute group labels and recompute gap n_perm times."""
    observed_gap = np.mean(within_rates) - np.mean(between_rates)
    all_rates = np.concatenate([within_rates, between_rates])
    n_within = len(within_rates)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(all_rates))
        perm_within = all_rates[perm[:n_within]]
        perm_between = all_rates[perm[n_within:]]
        perm_gap = np.mean(perm_within) - np.mean(perm_between)
        if perm_gap >= observed_gap:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)  # +1 for continuity correction
    return p_value


def bootstrap_ci(data, rng, n_boot=2000, alpha=0.05):
    """Bootstrap confidence interval for the mean."""
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
# Main experiment
# ---------------------------------------------------------------------------
print("=" * 70)
print("Noether Counting Sensitivity Analysis")
print("=" * 70)
print(f"P={P}, G={G}, group_size={GROUP_SIZE}, N_train={N_TRAIN}")
print(f"N_models={N_MODELS}, noise={NOISE_STD}")
print(f"betas={BETAS.tolist()}")
print(f"rho_values={RHO_VALUES}")
print(f"Permutation test: {N_PERM} permutations")
print()

results_all = []

for rho in RHO_VALUES:
    print(f"--- rho_within = {rho:.2f} ---")

    # Train models and collect importances
    all_importances = np.zeros((N_MODELS, P))
    for i in range(N_MODELS):
        rng_i = np.random.default_rng(SEED + i)
        X_train = make_correlated_X(N_TRAIN, rho, rng_i)
        y_train = X_train @ BETAS + rng_i.standard_normal(N_TRAIN) * NOISE_STD
        model = Ridge(alpha=0.01)
        model.fit(X_train, y_train)
        all_importances[i] = np.abs(model.coef_)

    # Convert to rankings
    all_rankings = np.zeros_like(all_importances, dtype=int)
    for i in range(N_MODELS):
        all_rankings[i] = stats.rankdata(-all_importances[i], method='ordinal')

    # Compute flip rates
    flip_rates = compute_flip_rate(all_rankings)
    within_rates, between_rates = classify_pairs(flip_rates)

    mean_within = float(np.mean(within_rates))
    mean_between = float(np.mean(between_rates))
    gap = mean_within - mean_between

    # Bootstrap CIs on the gap
    rng_boot = np.random.default_rng(SEED + 9999)
    # Bootstrap the gap directly
    n_within = len(within_rates)
    all_rates_arr = np.concatenate([within_rates, between_rates])
    gap_boots = []
    for _ in range(2000):
        w_samp = rng_boot.choice(within_rates, size=len(within_rates), replace=True)
        b_samp = rng_boot.choice(between_rates, size=len(between_rates), replace=True)
        gap_boots.append(np.mean(w_samp) - np.mean(b_samp))
    gap_boots = sorted(gap_boots)
    gap_ci_lo = gap_boots[int(2000 * 0.025)]
    gap_ci_hi = gap_boots[int(2000 * 0.975)]

    # Permutation test
    rng_perm = np.random.default_rng(SEED + 7777)
    perm_p = permutation_test(within_rates, between_rates, N_PERM, rng_perm)

    print(f"  Within-group mean flip: {mean_within:.4f}")
    print(f"  Between-group mean flip: {mean_between:.4f}")
    print(f"  Gap: {gap:.4f}  [{gap_ci_lo:.4f}, {gap_ci_hi:.4f}]")
    print(f"  Permutation p-value: {perm_p:.6f}")

    # Benjamini-Hochberg correction for this rho value
    n_model_comparisons = N_MODELS * (N_MODELS - 1) // 2
    p_values_stability = []
    pair_keys_rho = []
    for (j, k), rate in flip_rates.items():
        n_flips = int(round(rate * n_model_comparisons))
        if n_flips == 0:
            p_values_stability.append(1.0)
        else:
            result = binomtest(n_flips, n_model_comparisons, 0.0001, alternative='greater')
            p_values_stability.append(result.pvalue)
        pair_keys_rho.append((j, k))

    bh_result_rho = {}
    if HAS_STATSMODELS:
        reject_bh, corrected_p_bh, _, _ = multipletests(p_values_stability, alpha=0.05, method='fdr_bh')
        n_unstable_bh = int(sum(reject_bh))
        bh_within_unstable = sum(1 for i, (j, k) in enumerate(pair_keys_rho) if reject_bh[i] and get_group(j) == get_group(k))
        bh_between_unstable = sum(1 for i, (j, k) in enumerate(pair_keys_rho) if reject_bh[i] and get_group(j) != get_group(k))
        print(f"  BH correction: {n_unstable_bh}/66 unstable (within: {bh_within_unstable}, between: {bh_between_unstable})")
        bh_result_rho = {
            "n_unstable": n_unstable_bh,
            "n_stable": 66 - n_unstable_bh,
            "within_group_unstable": int(bh_within_unstable),
            "between_group_unstable": int(bh_between_unstable),
            "note": "H0: pair is stable (flip_rate~0). BH-corrected at FDR=0.05."
        }

    print()

    results_all.append({
        "rho": rho,
        "mean_within": mean_within,
        "mean_between": mean_between,
        "gap": gap,
        "gap_ci_lo": gap_ci_lo,
        "gap_ci_hi": gap_ci_hi,
        "perm_p": perm_p,
        "within_rates": within_rates.tolist(),
        "between_rates": between_rates.tolist(),
        "benjamini_hochberg": bh_result_rho,
    })


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"{'rho':>6}  {'within':>8}  {'between':>8}  {'gap':>8}  {'95% CI':>20}  {'perm_p':>10}")
print("-" * 70)
for r in results_all:
    ci_str = f"[{r['gap_ci_lo']:.4f}, {r['gap_ci_hi']:.4f}]"
    print(f"{r['rho']:6.2f}  {r['mean_within']:8.4f}  {r['mean_between']:8.4f}  {r['gap']:8.4f}  {ci_str:>20}  {r['perm_p']:10.6f}")
print("=" * 70)

# Identify threshold
for r in results_all:
    if r['gap'] < 0.10:
        print(f"\n** Gap drops below 10pp at rho = {r['rho']:.2f} (gap = {r['gap']:.4f}) **")
        break
else:
    print("\n** Gap stays >= 10pp for all tested rho values **")


# ---------------------------------------------------------------------------
# Figure: gap vs rho with error bars
# ---------------------------------------------------------------------------
rhos = [r['rho'] for r in results_all]
gaps = [r['gap'] for r in results_all]
ci_lo = [r['gap_ci_lo'] for r in results_all]
ci_hi = [r['gap_ci_hi'] for r in results_all]
yerr_lo = [g - lo for g, lo in zip(gaps, ci_lo)]
yerr_hi = [hi - g for g, hi in zip(gaps, ci_hi)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(rhos, gaps, yerr=[yerr_lo, yerr_hi],
            fmt='o-', color='#2c3e50', capsize=5, capthick=1.5,
            markersize=8, linewidth=2, ecolor='#7f8c8d', elinewidth=1.5,
            label='Gap (within - between)')
ax.axhline(0.10, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5,
           label='10pp threshold')
ax.axhline(0.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax.set_xlabel(r'Within-group correlation $\rho$', fontsize=13)
ax.set_ylabel('Flip rate gap (within - between)', fontsize=13)
ax.set_title('Noether Counting Sensitivity: Bimodal Gap vs. Correlation', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0.45, 1.02)
ax.set_ylim(bottom=-0.02)
ax.tick_params(labelsize=11)

# Annotate p-values
for r in results_all:
    p_str = f"p={r['perm_p']:.4f}" if r['perm_p'] >= 0.0001 else f"p<0.0001"
    ax.annotate(p_str, xy=(r['rho'], r['gap']),
                xytext=(0, -18), textcoords='offset points',
                ha='center', fontsize=7.5, color='#555555')

plt.tight_layout()
fig.savefig(str(FIGURE_FILE), bbox_inches='tight', dpi=300)
print(f"\nSaved figure: {FIGURE_FILE}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save results JSON
# ---------------------------------------------------------------------------
output = {
    "experiment": "noether_sensitivity",
    "config": {
        "P": P, "G": G, "group_size": GROUP_SIZE,
        "betas": BETAS.tolist(),
        "rho_between": RHO_BETWEEN,
        "rho_values": RHO_VALUES,
        "n_train": N_TRAIN, "noise_std": NOISE_STD,
        "n_models": N_MODELS, "seed": SEED,
        "model": "Ridge(alpha=0.01)",
        "n_permutations": N_PERM,
    },
    "results": [],
    "_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
}

for r in results_all:
    entry = {
        "rho": r["rho"],
        "mean_within": r["mean_within"],
        "mean_between": r["mean_between"],
        "gap": r["gap"],
        "gap_ci_95": [r["gap_ci_lo"], r["gap_ci_hi"]],
        "permutation_p": r["perm_p"],
        "within_rates": r["within_rates"],
        "between_rates": r["between_rates"],
    }
    if r.get("benjamini_hochberg"):
        entry["benjamini_hochberg"] = r["benjamini_hochberg"]
    output["results"].append(entry)

with open(str(RESULTS_FILE), 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved results: {RESULTS_FILE}")
