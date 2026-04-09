"""
Validate the theoretical attribution ratio 1/(1-ρ²) with targeted experiments.

Measures the WITHIN-GROUP ratio (first-mover / non-first-mover in the same
collinear group), which is what the theory predicts. The previous figure
incorrectly measured the global top-1/top-2 ratio.

Experiments:
1. Split count ratio vs ρ (stumps, depth=1)
2. SHAP ratio vs ρ (stumps, depth=1)
3. Depth robustness (depth ∈ {1, 3, 6})
4. Learning rate robustness (η ∈ {0.1, 0.3, 1.0})

DGP: P=10 features, 2 groups of 5, N=2000, Y = ΣX_j + ε
"""

import numpy as np
import json
import os

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Install with: pip install shap")

# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_data(n, p_per_group, n_groups, rho, seed):
    """Generate Gaussian data with block-diagonal correlation structure."""
    rng = np.random.default_rng(seed)
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

    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    # Equal coefficients within groups
    beta = np.ones(p)
    y = X @ beta + rng.normal(0, 0.1, size=n)
    return X, y


def get_first_mover_and_ratio(model, X, p_per_group, n_groups, use_shap=False):
    """Compute within-group first-mover ratio.

    Returns dict with:
    - split_ratios: list of (first_mover_splits / non_first_mover_splits) per group
    - shap_ratios: list of SHAP ratios per group (if use_shap=True)
    - first_movers: list of first-mover feature indices per group
    """
    booster = model.get_booster()

    # Get split counts per feature
    score = booster.get_score(importance_type='weight')
    p = p_per_group * n_groups
    split_counts = np.zeros(p)
    for feat, count in score.items():
        idx = int(feat.replace('f', ''))
        split_counts[idx] = count

    results = {
        'split_ratios': [],
        'shap_ratios': [],
        'first_movers': [],
        'split_counts_by_group': [],
    }

    for g in range(n_groups):
        start = g * p_per_group
        end = start + p_per_group
        group_splits = split_counts[start:end]

        if group_splits.max() == 0:
            continue

        fm_idx = np.argmax(group_splits)
        fm_splits = group_splits[fm_idx]

        # Non-first-mover splits (average of others in group)
        non_fm_mask = np.ones(p_per_group, dtype=bool)
        non_fm_mask[fm_idx] = False
        non_fm_splits = group_splits[non_fm_mask]

        if non_fm_splits.mean() > 0:
            results['split_ratios'].append(fm_splits / non_fm_splits.mean())
        results['first_movers'].append(start + fm_idx)
        results['split_counts_by_group'].append(group_splits.tolist())

    # SHAP ratios
    if use_shap and HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:200])  # subset for speed
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        for g in range(n_groups):
            start = g * p_per_group
            end = start + p_per_group
            group_shap = mean_abs_shap[start:end]

            if group_shap.max() == 0:
                continue

            fm_idx = np.argmax(group_shap)
            fm_shap = group_shap[fm_idx]
            non_fm_mask = np.ones(p_per_group, dtype=bool)
            non_fm_mask[fm_idx] = False
            non_fm_shap = group_shap[non_fm_mask]

            if non_fm_shap.mean() > 0:
                results['shap_ratios'].append(fm_shap / non_fm_shap.mean())

    return results


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def run_experiments():
    if not HAS_XGB:
        print("Cannot run experiments without xgboost. Exiting.")
        return

    P_PER_GROUP = 5
    N_GROUPS = 2
    N_SAMPLES = 2000
    N_SEEDS = 50
    N_TREES = 100

    rho_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    all_results = {}

    # -----------------------------------------------------------------------
    # Experiment 1+2: Split count + SHAP ratio at depth=1 (stumps)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Experiment 1+2: Stumps (max_depth=1), split count + SHAP ratio")
    print("=" * 60)

    for rho in rho_values:
        theory = 1.0 / (1.0 - rho ** 2)
        split_ratios_all = []
        shap_ratios_all = []

        for seed in range(N_SEEDS):
            X, y = generate_data(N_SAMPLES, P_PER_GROUP, N_GROUPS, rho, seed)

            model = xgb.XGBRegressor(
                n_estimators=N_TREES,
                max_depth=1,  # stumps — matches theory
                learning_rate=1.0,  # full residual fitting
                reg_alpha=0,
                reg_lambda=0,
                colsample_bytree=1.0,
                subsample=1.0,
                random_state=seed + 1000,
            )
            model.fit(X, y, verbose=False)

            res = get_first_mover_and_ratio(
                model, X, P_PER_GROUP, N_GROUPS,
                use_shap=(HAS_SHAP and seed < 10)  # SHAP on first 10 seeds (slow)
            )
            split_ratios_all.extend(res['split_ratios'])
            shap_ratios_all.extend(res['shap_ratios'])

        mean_split = np.mean(split_ratios_all) if split_ratios_all else 0
        se_split = np.std(split_ratios_all) / np.sqrt(len(split_ratios_all)) if split_ratios_all else 0
        mean_shap = np.mean(shap_ratios_all) if shap_ratios_all else 0

        print(f"  ρ={rho:.2f}: theory={theory:.3f}, "
              f"split_ratio={mean_split:.3f}±{se_split:.3f}, "
              f"shap_ratio={mean_shap:.3f} "
              f"(n={len(split_ratios_all)} groups)")

        all_results[f"depth1_rho{rho}"] = {
            'rho': rho,
            'theory': theory,
            'split_ratio_mean': mean_split,
            'split_ratio_se': se_split,
            'split_ratios': split_ratios_all,
            'shap_ratio_mean': mean_shap,
            'n_obs': len(split_ratios_all),
        }

    # -----------------------------------------------------------------------
    # Experiment 3: Depth robustness
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 3: Depth robustness at ρ=0.9")
    print("=" * 60)

    rho_fixed = 0.9
    theory_fixed = 1.0 / (1.0 - rho_fixed ** 2)

    for depth in [1, 3, 6]:
        split_ratios_all = []
        for seed in range(N_SEEDS):
            X, y = generate_data(N_SAMPLES, P_PER_GROUP, N_GROUPS, rho_fixed, seed)
            model = xgb.XGBRegressor(
                n_estimators=N_TREES,
                max_depth=depth,
                learning_rate=0.3 if depth > 1 else 1.0,
                reg_alpha=0, reg_lambda=0,
                colsample_bytree=1.0, subsample=1.0,
                random_state=seed + 2000,
            )
            model.fit(X, y, verbose=False)
            res = get_first_mover_and_ratio(model, X, P_PER_GROUP, N_GROUPS)
            split_ratios_all.extend(res['split_ratios'])

        mean_r = np.mean(split_ratios_all) if split_ratios_all else 0
        se_r = np.std(split_ratios_all) / np.sqrt(len(split_ratios_all)) if split_ratios_all else 0
        print(f"  depth={depth}: theory={theory_fixed:.3f}, "
              f"empirical={mean_r:.3f}±{se_r:.3f}")

        all_results[f"depth{depth}_rho{rho_fixed}"] = {
            'depth': depth, 'rho': rho_fixed,
            'theory': theory_fixed,
            'split_ratio_mean': mean_r, 'split_ratio_se': se_r,
        }

    # -----------------------------------------------------------------------
    # Experiment 4: Learning rate robustness at depth=1, ρ=0.9
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 4: Learning rate robustness at depth=1, ρ=0.9")
    print("=" * 60)

    for eta in [0.1, 0.3, 1.0]:
        split_ratios_all = []
        for seed in range(N_SEEDS):
            X, y = generate_data(N_SAMPLES, P_PER_GROUP, N_GROUPS, rho_fixed, seed)
            model = xgb.XGBRegressor(
                n_estimators=N_TREES,
                max_depth=1,
                learning_rate=eta,
                reg_alpha=0, reg_lambda=0,
                colsample_bytree=1.0, subsample=1.0,
                random_state=seed + 3000,
            )
            model.fit(X, y, verbose=False)
            res = get_first_mover_and_ratio(model, X, P_PER_GROUP, N_GROUPS)
            split_ratios_all.extend(res['split_ratios'])

        mean_r = np.mean(split_ratios_all) if split_ratios_all else 0
        se_r = np.std(split_ratios_all) / np.sqrt(len(split_ratios_all)) if split_ratios_all else 0
        print(f"  η={eta}: theory={theory_fixed:.3f}, "
              f"empirical={mean_r:.3f}±{se_r:.3f}")

        all_results[f"eta{eta}_rho{rho_fixed}"] = {
            'eta': eta, 'rho': rho_fixed,
            'theory': theory_fixed,
            'split_ratio_mean': mean_r, 'split_ratio_se': se_r,
        }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "..", "results_validation.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: convert(vv) for kk, vv in v.items()}

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiments()
