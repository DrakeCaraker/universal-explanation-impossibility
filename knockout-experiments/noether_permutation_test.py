#!/usr/bin/env python3
"""
Noether Permutation Test (Task B)

Generates synthetic data matching the Noether experiment design:
  - P=8 features, 2 groups of 4, rho_within=0.90
  - Trains 30 XGBoost models (seeds 42-71)
  - Computes permutation importance per model (introduces stochasticity)
  - Computes within-group and between-group flip rates on importance rankings
  - Permutation test (10000 permutations) on the gap statistic

Saves results to results_noether_permutation_test.json
"""

import json
import warnings
import numpy as np
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

# --- Configuration ---
P = 8                   # number of features
G = 2                   # number of groups
GROUP_SIZE = 4          # features per group
RHO_WITHIN = 0.90       # within-group correlation
RHO_BETWEEN = 0.0       # between-group correlation
N_TRAIN = 500           # training samples
N_TEST = 200            # test samples for permutation importance
NOISE_STD = 1.0
N_MODELS = 30           # seeds 42-71
N_PERMUTATIONS = 10000
SEED_BASE = 42
BETAS = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0])


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_correlated_features(n_samples, rng):
    """Generate P features in G groups with within-group correlation."""
    cov = np.full((P, P), RHO_BETWEEN)
    for g in range(G):
        start = g * GROUP_SIZE
        end = start + GROUP_SIZE
        cov[start:end, start:end] = RHO_WITHIN
    np.fill_diagonal(cov, 1.0)
    return rng.multivariate_normal(np.zeros(P), cov, size=n_samples)


def compute_permutation_importance(model, X_test, y_test, rng, n_repeats=5):
    """
    Compute permutation importance for a model.
    Returns importance array of shape (P,).
    Uses random shuffling which introduces stochasticity across models.
    """
    from sklearn.metrics import accuracy_score
    baseline_score = accuracy_score(y_test, model.predict(X_test))

    importances = np.zeros(P)
    for f in range(P):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            perm_score = accuracy_score(y_test, model.predict(X_perm))
            scores.append(baseline_score - perm_score)
        importances[f] = np.mean(scores)

    return importances


def compute_flip_rates_from_importances(importance_matrix):
    """
    Given importance_matrix of shape (n_models, P),
    compute flip rate for each feature pair.

    A flip occurs when two models disagree on the relative importance
    of two features.
    """
    n_models = importance_matrix.shape[0]
    feature_pairs = list(combinations(range(P), 2))

    pair_flip_rates = {}
    for fi, fj in feature_pairs:
        # For each model, is feature fi more important than fj?
        fi_higher = importance_matrix[:, fi] > importance_matrix[:, fj]

        n_flips = 0
        n_model_pairs = 0
        for mi in range(n_models):
            for mj in range(mi + 1, n_models):
                if fi_higher[mi] != fi_higher[mj]:
                    n_flips += 1
                n_model_pairs += 1

        pair_flip_rates[(fi, fj)] = n_flips / n_model_pairs if n_model_pairs > 0 else 0.0

    return pair_flip_rates


def classify_pairs(pair_flip_rates, groups):
    """Classify feature pairs as within-group or between-group."""
    within_rates = []
    between_rates = []

    # Build feature-to-group mapping
    feat_to_group = {}
    for g_idx, group in enumerate(groups):
        for f in group:
            feat_to_group[f] = g_idx

    for (fi, fj), rate in pair_flip_rates.items():
        if feat_to_group.get(fi) == feat_to_group.get(fj):
            within_rates.append(rate)
        else:
            between_rates.append(rate)

    return within_rates, between_rates


def run_experiment():
    """Run the full Noether permutation test."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("ERROR: xgboost not installed. Install with: pip install xgboost")
        return None

    print("Generating synthetic data and training models...")

    # True group assignments
    true_groups = [
        list(range(0, GROUP_SIZE)),
        list(range(GROUP_SIZE, 2 * GROUP_SIZE))
    ]

    # Generate data
    rng_data = np.random.default_rng(seed=0)
    X_train = make_correlated_features(N_TRAIN, rng_data)
    y_train = (X_train @ BETAS + rng_data.normal(0, NOISE_STD, N_TRAIN) > 0).astype(int)

    X_test = make_correlated_features(N_TEST, rng_data)
    y_test = (X_test @ BETAS + rng_data.normal(0, NOISE_STD, N_TEST) > 0).astype(int)

    # Train models and compute permutation importance
    importance_matrix = np.zeros((N_MODELS, P))
    for i, seed in enumerate(range(SEED_BASE, SEED_BASE + N_MODELS)):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            tree_method='hist',
            random_state=seed,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)

        # Use per-model RNG for permutation importance (different stochasticity per model)
        rng_imp = np.random.default_rng(seed=seed)
        importance_matrix[i] = compute_permutation_importance(
            model, X_test, y_test, rng_imp, n_repeats=10
        )

    print(f"Trained {N_MODELS} XGBoost models and computed permutation importances.")

    # Show importance summary
    print(f"\n  Mean importances per feature:")
    for f in range(P):
        group_label = "G1" if f < GROUP_SIZE else "G2"
        print(f"    Feature {f} ({group_label}): {np.mean(importance_matrix[:, f]):.4f} "
              f"+/- {np.std(importance_matrix[:, f]):.4f}")

    # Compute flip rates
    pair_flip_rates = compute_flip_rates_from_importances(importance_matrix)
    within_rates, between_rates = classify_pairs(pair_flip_rates, true_groups)

    observed_within = float(np.mean(within_rates)) if within_rates else 0.0
    observed_between = float(np.mean(between_rates)) if between_rates else 0.0
    observed_gap = observed_within - observed_between

    print(f"\nObserved results:")
    print(f"  Mean within-group flip rate:  {observed_within:.6f}")
    print(f"  Mean between-group flip rate: {observed_between:.6f}")
    print(f"  Gap (within - between):       {observed_gap:.6f}")
    print(f"  Within rates:  {[round(r, 4) for r in within_rates]}")
    print(f"  Between rates: {[round(r, 4) for r in between_rates]}")

    # --- Permutation test ---
    print(f"\nRunning permutation test ({N_PERMUTATIONS} permutations)...")

    rng_perm = np.random.default_rng(seed=42)
    features = list(range(P))
    perm_gaps = []

    for i in range(N_PERMUTATIONS):
        shuffled = rng_perm.permutation(features)
        perm_groups = [
            list(shuffled[:GROUP_SIZE]),
            list(shuffled[GROUP_SIZE:])
        ]

        within_r, between_r = classify_pairs(pair_flip_rates, perm_groups)
        perm_within = float(np.mean(within_r)) if within_r else 0.0
        perm_between = float(np.mean(between_r)) if between_r else 0.0
        perm_gaps.append(perm_within - perm_between)

        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{N_PERMUTATIONS} permutations done")

    perm_gaps = np.array(perm_gaps)

    n_extreme = int(np.sum(perm_gaps >= observed_gap))
    p_value = (n_extreme + 1) / (N_PERMUTATIONS + 1)

    perm_mean = float(np.mean(perm_gaps))
    perm_std = float(np.std(perm_gaps, ddof=1))
    perm_95 = float(np.percentile(perm_gaps, 95))
    perm_99 = float(np.percentile(perm_gaps, 99))

    print(f"\nPermutation test results:")
    print(f"  Observed gap:        {observed_gap:.6f}")
    print(f"  Permutation mean:    {perm_mean:.6f}")
    print(f"  Permutation std:     {perm_std:.6f}")
    print(f"  Permutation 95th:    {perm_95:.6f}")
    print(f"  Permutation 99th:    {perm_99:.6f}")
    print(f"  N extreme (>= obs):  {n_extreme}")
    print(f"  p-value:             {p_value:.6f}")
    print(f"  Significant (p<0.05): {p_value < 0.05}")

    results = {
        "experiment": "noether_permutation_test",
        "description": (
            "Permutation test for Noether symmetry prediction: "
            "within-group flip rates should exceed between-group flip rates "
            "when features are correlated within groups (rho_within=0.90)."
        ),
        "config": {
            "P": P,
            "G": G,
            "group_size": GROUP_SIZE,
            "rho_within": RHO_WITHIN,
            "rho_between": RHO_BETWEEN,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "noise_std": NOISE_STD,
            "n_models": N_MODELS,
            "seed_range": f"{SEED_BASE}-{SEED_BASE + N_MODELS - 1}",
            "n_permutations": N_PERMUTATIONS,
            "betas": BETAS.tolist(),
            "model": "XGBClassifier(n_estimators=100, max_depth=4, tree_method=hist)",
            "importance_method": "permutation_importance(n_repeats=10)"
        },
        "observed": {
            "mean_within_flip_rate": round(observed_within, 6),
            "mean_between_flip_rate": round(observed_between, 6),
            "gap": round(observed_gap, 6),
            "within_rates": [round(float(r), 6) for r in within_rates],
            "between_rates": [round(float(r), 6) for r in between_rates],
            "true_groups": true_groups,
            "mean_importances_per_feature": [
                round(float(np.mean(importance_matrix[:, f])), 6) for f in range(P)
            ],
            "std_importances_per_feature": [
                round(float(np.std(importance_matrix[:, f])), 6) for f in range(P)
            ]
        },
        "permutation_test": {
            "n_permutations": N_PERMUTATIONS,
            "n_extreme": n_extreme,
            "p_value": round(p_value, 6),
            "significant_at_05": bool(p_value < 0.05),
            "significant_at_01": bool(p_value < 0.01),
            "permutation_distribution": {
                "mean": round(perm_mean, 6),
                "std": round(perm_std, 6),
                "percentile_95": round(perm_95, 6),
                "percentile_99": round(perm_99, 6),
                "min": round(float(np.min(perm_gaps)), 6),
                "max": round(float(np.max(perm_gaps)), 6)
            }
        },
        "interpretation": (
            "The Noether prediction states that correlated features (within-group) "
            "should show higher explanation flip rates than uncorrelated features "
            "(between-group). The permutation test assesses whether the observed "
            "within-vs-between gap is larger than expected by random group assignment."
        )
    }

    out_path = BASE_DIR / "results_noether_permutation_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")

    return results


def main():
    print("=" * 60)
    print("TASK B: Noether Permutation Test")
    print("=" * 60)
    run_experiment()


if __name__ == "__main__":
    main()
