"""
Adversarial Max Instability — Grid search for worst-case SHAP flip rate.

Searches over correlation (rho), group size (m), tree count (T), and depth
to find the XGBoost configuration that MAXIMIZES within-group SHAP instability.

Grid:
  rho        ∈ [0.90, 0.95, 0.99]
  m (group)  ∈ [2, 3, 5, 10]
  P = L*m    for L=2 groups (so P ∈ [4, 6, 10, 20])
  T (trees)  ∈ [50, 100, 500]
  max_depth  ∈ [1, 3, 6]

That is 3×4×3×3 = 108 configs × 20 seeds = 2,160 fits.

For each config: generate Gaussian data (N=2000, Y=ΣX_j+ε), train 20
XGBoost models (subsample=0.8, different seeds), compute TreeSHAP on
200 eval samples, measure the max within-group flip rate.

Output: top 10 worst-case configs by flip rate → paper/results_adversarial.json.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import os
import sys
import time
from itertools import combinations

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Install with: pip install shap")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
RHO_VALUES = [0.90, 0.95, 0.99]
GROUP_SIZES = [2, 3, 5, 10]
N_GROUPS = 2                    # L = 2 groups
N_ESTIMATORS_VALUES = [50, 100, 500]
MAX_DEPTH_VALUES = [1, 3, 6]

N_TRAIN = 2000                  # training samples
N_EVAL = 200                    # evaluation samples for SHAP
N_SEEDS = 20                    # independent models per config
SUBSAMPLE = 0.8                 # XGBoost subsampling
DATA_SEED = 42                  # fixed data generation seed

# ── Data generation ───────────────────────────────────────────────────────────

def generate_correlated_data(rho, n_samples, p_per_group, n_groups, rng):
    """Generate Gaussian data with block-diagonal correlation structure.

    Y = sum(X_j) + epsilon, epsilon ~ N(0, 1).
    """
    p = p_per_group * n_groups

    # Block-diagonal covariance
    cov = np.eye(p)
    for g in range(n_groups):
        start = g * p_per_group
        end = start + p_per_group
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    cov[i, j] = rho

    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    Y = X.sum(axis=1) + rng.standard_normal(n_samples)
    return X, Y


def compute_within_group_flip_rate(shap_rankings, p_per_group, n_groups):
    """Compute the max within-group flip rate across all within-group pairs.

    For each pair (i, j) within the same group, the flip rate is the fraction
    of model pairs where the ranking order of i and j disagrees.

    Returns the maximum flip rate across all within-group pairs.
    """
    n_models = len(shap_rankings)
    if n_models < 2:
        return 0.0

    max_flip = 0.0

    for g in range(n_groups):
        start = g * p_per_group
        end = start + p_per_group
        group_features = list(range(start, end))

        for fi, fj in combinations(group_features, 2):
            # Count how many models rank fi > fj vs fj > fi
            fi_wins = 0
            fj_wins = 0
            for ranking in shap_rankings:
                # ranking[k] = mean |SHAP| for feature k
                if ranking[fi] > ranking[fj]:
                    fi_wins += 1
                elif ranking[fj] > ranking[fi]:
                    fj_wins += 1
                # ties don't count toward either

            total_decided = fi_wins + fj_wins
            if total_decided > 0:
                minority = min(fi_wins, fj_wins)
                flip_rate = minority / total_decided
                max_flip = max(max_flip, flip_rate)

    return max_flip


# ── Main experiment ───────────────────────────────────────────────────────────

def run_config(rho, m, n_estimators, max_depth):
    """Run a single configuration and return the max within-group flip rate."""
    p = m * N_GROUPS
    rng_data = np.random.default_rng(DATA_SEED)

    # Generate data once per config
    X_train, Y_train = generate_correlated_data(rho, N_TRAIN, m, N_GROUPS, rng_data)
    X_eval, _ = generate_correlated_data(rho, N_EVAL, m, N_GROUPS, rng_data)

    shap_rankings = []

    for seed in range(N_SEEDS):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=SUBSAMPLE,
            learning_rate=0.1,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X_train, Y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        # mean |SHAP| per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_rankings.append(mean_abs_shap)

    flip_rate = compute_within_group_flip_rate(shap_rankings, m, N_GROUPS)
    return flip_rate


def main():
    t0 = time.time()

    configs = []
    for rho in RHO_VALUES:
        for m in GROUP_SIZES:
            for n_est in N_ESTIMATORS_VALUES:
                for depth in MAX_DEPTH_VALUES:
                    configs.append((rho, m, n_est, depth))

    print(f"Adversarial max instability: {len(configs)} configs × {N_SEEDS} seeds "
          f"= {len(configs) * N_SEEDS} fits")
    print(f"Grid: rho={RHO_VALUES}, m={GROUP_SIZES}, T={N_ESTIMATORS_VALUES}, "
          f"depth={MAX_DEPTH_VALUES}")
    print()

    results = []

    for idx, (rho, m, n_est, depth) in enumerate(configs):
        p = m * N_GROUPS
        t_cfg = time.time()

        flip_rate = run_config(rho, m, n_est, depth)

        elapsed = time.time() - t_cfg
        results.append({
            "rho": rho,
            "group_size": m,
            "P": p,
            "n_estimators": n_est,
            "max_depth": depth,
            "max_flip_rate": round(flip_rate, 4),
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            total_elapsed = time.time() - t0
            remaining = (total_elapsed / (idx + 1)) * (len(configs) - idx - 1)
            print(f"  [{idx+1:3d}/{len(configs)}] rho={rho} m={m} T={n_est} "
                  f"depth={depth} → flip={flip_rate:.4f}  "
                  f"({elapsed:.1f}s, ETA {remaining/60:.1f}min)")

    # Sort by flip rate descending
    results.sort(key=lambda x: x["max_flip_rate"], reverse=True)

    # Top 10
    top10 = results[:10]

    print()
    print("=" * 72)
    print("TOP 10 WORST-CASE CONFIGURATIONS BY MAX WITHIN-GROUP FLIP RATE")
    print("=" * 72)
    print(f"{'Rank':>4}  {'rho':>5}  {'m':>3}  {'P':>3}  {'T':>4}  {'depth':>5}  {'flip_rate':>9}")
    print("-" * 72)
    for i, r in enumerate(top10):
        print(f"{i+1:4d}  {r['rho']:5.2f}  {r['group_size']:3d}  {r['P']:3d}  "
              f"{r['n_estimators']:4d}  {r['max_depth']:5d}  {r['max_flip_rate']:9.4f}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save results
    output = {
        "description": "Adversarial grid search for max SHAP instability",
        "grid": {
            "rho": RHO_VALUES,
            "group_sizes": GROUP_SIZES,
            "n_estimators": N_ESTIMATORS_VALUES,
            "max_depth": MAX_DEPTH_VALUES,
        },
        "settings": {
            "N_TRAIN": N_TRAIN,
            "N_EVAL": N_EVAL,
            "N_SEEDS": N_SEEDS,
            "SUBSAMPLE": SUBSAMPLE,
            "DATA_SEED": DATA_SEED,
            "N_GROUPS": N_GROUPS,
        },
        "total_configs": len(configs),
        "total_fits": len(configs) * N_SEEDS,
        "total_time_seconds": round(total_time, 1),
        "top10": top10,
        "all_results": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "results_adversarial.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
