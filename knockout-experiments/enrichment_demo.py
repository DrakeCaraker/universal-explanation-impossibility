#!/usr/bin/env python3
"""
Empirical demonstration of the bilemma's enrichment prediction.

Shows that adding a "tied" option to binary feature rankings restores
stability at the cost of decisiveness — the tradeoff curve predicted
by the enrichment theorem.

Setup:
  P=8 features, 2 groups of 4, rho_within=0.90
  beta = [3,3,3,3, 1,1,1,1], N=500, noise=1.0
  100 XGBoost models with bootstrap resampling

Without enrichment (binary): within-group flip rate ~50%
With enrichment (ternary):   tied pairs reduce flip rate, but lose decisiveness
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ─── Data generation ─────────────────────────────────────────────────

def generate_correlated_data(n=500, p=8, rho_within=0.90, seed=0):
    """Generate data with two groups of correlated features."""
    rng = np.random.default_rng(seed)
    # Build correlation matrix: two blocks of size 4
    Sigma = np.eye(p)
    for i in range(4):
        for j in range(4):
            if i != j:
                Sigma[i, j] = rho_within
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                Sigma[i, j] = rho_within

    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T

    beta = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    y = X @ beta + rng.standard_normal(n) * 1.0
    return X, y, beta


# ─── Train models ────────────────────────────────────────────────────

def train_models(X, y, n_models=100):
    """Train n_models XGBoost regressors with bootstrap resampling."""
    n = X.shape[0]
    models = []
    importances = []
    for i in range(n_models):
        rng = np.random.default_rng(42 + i)
        idx = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42 + i,
            verbosity=0,
        )
        model.fit(X_boot, y_boot)
        models.append(model)
        importances.append(model.feature_importances_)

    return models, np.array(importances)


# ─── Binary flip rate (no enrichment) ────────────────────────────────

def binary_flip_rate(importances):
    """
    For each feature pair (j,k), compute fraction of model pairs
    that disagree on the ranking (j>k vs k>j).
    """
    n_models, p = importances.shape
    pairs = list(combinations(range(p), 2))
    flip_rates = {}

    for j, k in pairs:
        disagree = 0
        total = 0
        for m1 in range(n_models):
            for m2 in range(m1 + 1, n_models):
                rank1 = importances[m1, j] > importances[m1, k]
                rank2 = importances[m2, j] > importances[m2, k]
                if rank1 != rank2:
                    disagree += 1
                total += 1
        flip_rates[(j, k)] = disagree / total if total > 0 else 0.0

    return flip_rates


# ─── Enriched flip rate (with "tied" option) ─────────────────────────

def enriched_ranking(imp_j, imp_k, threshold, max_imp):
    """Classify pair as 'j>k', 'k>j', or 'tied'."""
    if max_imp == 0:
        return 'tied'
    diff = abs(imp_j - imp_k)
    if diff <= threshold * max_imp:
        return 'tied'
    elif imp_j > imp_k:
        return 'j>k'
    else:
        return 'k>j'


def enriched_flip_rate(importances, threshold):
    """
    With enrichment: "tied" is compatible with everything (the enrichment
    theorem's key property). Only j>k vs k>j is a genuine disagreement.

    Compatibility:
      tied  vs tied  → agree  (both conservative)
      tied  vs j>k   → agree  (tied is compatible with both orderings)
      j>k   vs j>k   → agree
      j>k   vs k>j   → DISAGREE (genuine flip)
    """
    n_models, p = importances.shape
    pairs = list(combinations(range(p), 2))

    # Per-model max importance for normalization
    max_imps = importances.max(axis=1)

    flip_rates = {}
    tied_counts = {}

    for j, k in pairs:
        rankings = []
        for m in range(n_models):
            rankings.append(enriched_ranking(
                importances[m, j], importances[m, k],
                threshold, max_imps[m]
            ))

        # Count tied fraction for this pair
        n_tied = sum(1 for r in rankings if r == 'tied')
        tied_counts[(j, k)] = n_tied / n_models

        # Flip rate: only count genuine flips (j>k vs k>j)
        disagree = 0
        total = 0
        for m1 in range(n_models):
            for m2 in range(m1 + 1, n_models):
                r1 = rankings[m1]
                r2 = rankings[m2]
                # Only a genuine flip when both are decisive and they disagree
                if r1 != 'tied' and r2 != 'tied' and r1 != r2:
                    disagree += 1
                total += 1
        flip_rates[(j, k)] = disagree / total if total > 0 else 0.0

    return flip_rates, tied_counts


# ─── Categorize pairs ────────────────────────────────────────────────

def categorize_pairs(p=8):
    """Classify pairs as within-group or between-group."""
    within = []
    between = []
    for j, k in combinations(range(p), 2):
        if (j < 4 and k < 4) or (j >= 4 and k >= 4):
            within.append((j, k))
        else:
            between.append((j, k))
    return within, between


# ─── Main experiment ─────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("Enrichment Demo: Bilemma Tradeoff Curve")
    print("=" * 60)

    # Generate data and train models
    print("\nGenerating correlated data (P=8, rho=0.90)...")
    X, y, beta = generate_correlated_data()
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  beta: {beta}")

    print("\nTraining 100 XGBoost models with bootstrap resampling...")
    models, importances = train_models(X, y, n_models=100)
    print(f"  Importance shape: {importances.shape}")
    print(f"  Mean importances: {importances.mean(axis=0).round(4)}")

    within, between = categorize_pairs()
    print(f"\n  Within-group pairs: {len(within)}")
    print(f"  Between-group pairs: {len(between)}")

    # Binary flip rate (no enrichment)
    print("\n--- Binary ranking (no enrichment) ---")
    binary_flips = binary_flip_rate(importances)

    within_flip = np.mean([binary_flips[p] for p in within])
    between_flip = np.mean([binary_flips[p] for p in between])
    overall_flip = np.mean(list(binary_flips.values()))

    print(f"  Within-group mean flip rate:  {within_flip:.4f}")
    print(f"  Between-group mean flip rate: {between_flip:.4f}")
    print(f"  Overall mean flip rate:       {overall_flip:.4f}")

    # Enriched flip rate across thresholds
    thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results_by_threshold = []

    print("\n--- Enriched ranking (with 'tied' option) ---")
    print(f"{'thresh':>8s} {'decisive':>10s} {'stability':>10s} "
          f"{'flip_within':>12s} {'flip_between':>13s} {'flip_all':>10s}")
    print("-" * 68)

    for thresh in thresholds:
        flips, tied_counts = enriched_flip_rate(importances, thresh)

        # Decisiveness: fraction of (pair, model) that are NOT tied
        mean_tied = np.mean(list(tied_counts.values()))
        decisiveness = 1.0 - mean_tied

        # Stability: 1 - flip rate
        within_fl = np.mean([flips[p] for p in within])
        between_fl = np.mean([flips[p] for p in between])
        overall_fl = np.mean(list(flips.values()))
        stability = 1.0 - overall_fl

        results_by_threshold.append({
            'threshold': thresh,
            'decisiveness': round(decisiveness, 4),
            'stability': round(stability, 4),
            'flip_rate_within': round(within_fl, 4),
            'flip_rate_between': round(between_fl, 4),
            'flip_rate_overall': round(overall_fl, 4),
            'tied_fraction': round(mean_tied, 4),
        })

        print(f"{thresh:8.2f} {decisiveness:10.4f} {stability:10.4f} "
              f"{within_fl:12.4f} {between_fl:13.4f} {overall_fl:10.4f}")

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDING: Enrichment tradeoff curve")
    print("=" * 60)
    r0 = results_by_threshold[0]
    print(f"  At threshold=0.00 (full decisiveness={r0['decisiveness']:.2f}):")
    print(f"    Stability = {r0['stability']:.4f}")
    print(f"    Within-group flip rate = {r0['flip_rate_within']:.4f} (should be ~0.50)")

    best_stable = max(results_by_threshold, key=lambda r: r['stability'])
    print(f"  At threshold={best_stable['threshold']:.2f} (max stability={best_stable['stability']:.4f}):")
    print(f"    Decisiveness = {best_stable['decisiveness']:.4f}")
    print(f"    => Cannot have BOTH decisiveness=1.0 AND stability=1.0")

    # Find the point closest to both being 1.0
    pareto = [(r['decisiveness'], r['stability']) for r in results_by_threshold]
    dist_to_ideal = [np.sqrt((1 - d)**2 + (1 - s)**2) for d, s in pareto]
    best_idx = np.argmin(dist_to_ideal)
    best_r = results_by_threshold[best_idx]
    print(f"\n  Closest to ideal (1,1): threshold={best_r['threshold']:.2f}")
    print(f"    Decisiveness={best_r['decisiveness']:.4f}, Stability={best_r['stability']:.4f}")
    print(f"    Distance to (1,1) = {dist_to_ideal[best_idx]:.4f}")

    # ─── Save results ────────────────────────────────────────────
    output = {
        'experiment': 'enrichment_demo',
        'description': 'Bilemma enrichment prediction: decisiveness-stability tradeoff',
        'setup': {
            'P': 8, 'N': 500, 'noise': 1.0,
            'rho_within': 0.90,
            'beta': [3, 3, 3, 3, 1, 1, 1, 1],
            'n_models': 100,
            'n_estimators': 100, 'max_depth': 4,
        },
        'binary_baseline': {
            'within_group_flip_rate': round(within_flip, 4),
            'between_group_flip_rate': round(between_flip, 4),
            'overall_flip_rate': round(overall_flip, 4),
        },
        'enrichment_tradeoff': results_by_threshold,
        'closest_to_ideal': {
            'threshold': best_r['threshold'],
            'decisiveness': best_r['decisiveness'],
            'stability': best_r['stability'],
            'distance_to_1_1': round(dist_to_ideal[best_idx], 4),
        },
    }

    out_path = OUT_DIR / 'results_enrichment_demo.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ─── Figure ──────────────────────────────────────────────────
    plot_results(results_by_threshold, output)
    return output


def plot_results(results, output):
    """Two-panel figure: tradeoff curve + threshold sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dec = [r['decisiveness'] for r in results]
    stab = [r['stability'] for r in results]
    thresh = [r['threshold'] for r in results]
    flip_w = [r['flip_rate_within'] for r in results]
    flip_b = [r['flip_rate_between'] for r in results]
    tied = [r['tied_fraction'] for r in results]

    # ── Panel A: Decisiveness vs Stability tradeoff ──
    ax = axes[0]
    ax.plot(dec, stab, 'o-', color='#2c7bb6', linewidth=2, markersize=7, zorder=3)

    # Annotate a few thresholds
    for i, t in enumerate(thresh):
        if t in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
            ax.annotate(f't={t:.2f}', (dec[i], stab[i]),
                        textcoords='offset points', xytext=(8, -8),
                        fontsize=7.5, color='#555555')

    # Mark the ideal point (1,1) as unreachable
    ax.plot(1.0, 1.0, 'x', color='red', markersize=12, markeredgewidth=2.5,
            label='Ideal (unreachable)', zorder=4)

    # Shade the "impossible" region
    ax.fill_between([0.85, 1.02], 0.85, 1.02, alpha=0.08, color='red',
                    label='Bilemma forbidden zone')

    ax.set_xlabel('Decisiveness (fraction non-tied)', fontsize=11)
    ax.set_ylabel('Stability (1 - flip rate)', fontsize=11)
    ax.set_title('A. Enrichment Tradeoff Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Panel B: Threshold sweep ──
    ax = axes[1]
    ax.plot(thresh, flip_w, 's-', color='#d7191c', linewidth=2, markersize=6,
            label='Within-group flip rate')
    ax.plot(thresh, flip_b, '^-', color='#2c7bb6', linewidth=2, markersize=6,
            label='Between-group flip rate')
    ax.plot(thresh, tied, 'D-', color='#636363', linewidth=2, markersize=5,
            label='Tied fraction (1-decisiveness)')
    ax.plot(thresh, stab, 'o-', color='#1a9641', linewidth=2, markersize=5,
            label='Stability')

    ax.set_xlabel('Tie threshold (fraction of max importance)', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('B. Threshold Sweep', fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=8.5)
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Bilemma Enrichment: Decisiveness-Stability Tradeoff',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    fig_path = FIG_DIR / 'enrichment_demo.pdf'
    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f"Figure saved to {fig_path}")

    # Also save PNG for quick viewing
    fig.savefig(FIG_DIR / 'enrichment_demo.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run_experiment()
