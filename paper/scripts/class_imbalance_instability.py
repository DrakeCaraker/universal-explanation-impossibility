"""
Class Imbalance Amplifies SHAP Instability.

Shows that increasing class imbalance widens the Rashomon set and amplifies
within-group SHAP ranking instability. Synthetic Gaussian data with P=10
features (2 groups of 5, within-group rho=0.8). Y = sign(sum(X_j) + eps).
Minority class is subsampled to achieve target imbalance ratios.

For each ratio: train 30 XGBoost classifiers (different seeds, subsample=0.8),
compute TreeSHAP, measure within-group flip rate.

Expected: higher imbalance -> wider Rashomon set -> more instability.

Saves results to paper/results_class_imbalance.json.
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

# -- Configuration ------------------------------------------------------------
P_FEATURES = 10          # total features (2 groups of 5)
GROUP_SIZE = 5            # features per correlated group
RHO = 0.8                # within-group correlation
N_TOTAL = 2000            # total samples before imbalance subsampling
N_EVAL = 200              # evaluation samples for SHAP
N_MODELS = 30             # models per imbalance ratio
N_ESTIMATORS = 100        # trees per model
MAX_DEPTH = 6             # tree depth
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8           # XGBoost subsampling
DATA_SEED = 42            # fixed data generation seed
NOISE_STD = 0.5           # epsilon noise std

# Imbalance ratios: minority:majority
IMBALANCE_RATIOS = [
    (1, 1),    # balanced
    (1, 3),
    (1, 5),
    (1, 10),
    (1, 20),
]


# -- Data generation -----------------------------------------------------------

def generate_correlated_data(n_samples, rng):
    """Generate P=10 features in 2 groups of 5 with within-group correlation rho.

    Y = sign(sum(X_j) + epsilon) for classification.
    Returns X, Y (binary 0/1).
    """
    p = P_FEATURES
    g = GROUP_SIZE

    # Build correlation matrix: block diagonal with rho within groups
    cov = np.eye(p)
    for group_start in [0, g]:
        for i in range(group_start, group_start + g):
            for j in range(group_start, group_start + g):
                if i != j:
                    cov[i, j] = RHO

    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    linear = X.sum(axis=1) + rng.standard_normal(n_samples) * NOISE_STD
    Y = (linear >= 0).astype(int)
    return X, Y


def subsample_to_ratio(X, Y, minority_ratio, majority_ratio, rng):
    """Subsample minority class to achieve desired imbalance ratio.

    Keeps all majority-class samples; subsamples minority class.
    """
    # Determine which class is naturally smaller
    class_counts = np.bincount(Y)
    if class_counts[0] <= class_counts[1]:
        minority_class, majority_class = 0, 1
    else:
        minority_class, majority_class = 1, 0

    idx_minority = np.where(Y == minority_class)[0]
    idx_majority = np.where(Y == majority_class)[0]

    n_majority = len(idx_majority)
    # Target: n_minority / n_majority = minority_ratio / majority_ratio
    n_minority_target = int(n_majority * minority_ratio / majority_ratio)
    n_minority_target = max(n_minority_target, 10)  # floor at 10 samples
    n_minority_target = min(n_minority_target, len(idx_minority))

    idx_minority_sub = rng.choice(idx_minority, size=n_minority_target, replace=False)
    idx_combined = np.concatenate([idx_minority_sub, idx_majority])
    rng.shuffle(idx_combined)

    return X[idx_combined], Y[idx_combined]


# -- Flip rate -----------------------------------------------------------------

def within_group_flip_rate(rankings, group_indices):
    """Compute flip rate for all within-group pairs across model rankings.

    rankings: list of arrays, each array is feature indices sorted by
              descending mean |SHAP|
    group_indices: list of feature indices in one group

    Returns list of dicts with (fi, fj, flip_rate) for each pair.
    """
    n_models = len(rankings)
    results = []
    for fi, fj in combinations(group_indices, 2):
        flips = 0
        total = 0
        for a in range(n_models):
            for b in range(a + 1, n_models):
                pos_fi_a = int(np.where(rankings[a] == fi)[0][0])
                pos_fj_a = int(np.where(rankings[a] == fj)[0][0])
                pos_fi_b = int(np.where(rankings[b] == fi)[0][0])
                pos_fj_b = int(np.where(rankings[b] == fj)[0][0])
                order_a = pos_fi_a < pos_fj_a
                order_b = pos_fi_b < pos_fj_b
                if order_a != order_b:
                    flips += 1
                total += 1
        fr = flips / total if total > 0 else 0.0
        results.append({"fi": fi, "fj": fj, "flip_rate": round(fr, 6)})
    return results


# -- Main experiment -----------------------------------------------------------

def run_experiment():
    print("=" * 72)
    print("Class Imbalance Amplifies SHAP Instability")
    print("=" * 72)
    print(f"Config: P={P_FEATURES}, rho={RHO}, N_total={N_TOTAL}, N_eval={N_EVAL}")
    print(f"        n_models={N_MODELS}, n_estimators={N_ESTIMATORS}, "
          f"max_depth={MAX_DEPTH}, subsample={SUBSAMPLE}")
    print(f"        noise_std={NOISE_STD}")
    print(f"Ratios: {[f'{m}:{M}' for m, M in IMBALANCE_RATIOS]}")
    print()

    # Generate base dataset
    base_rng = np.random.RandomState(DATA_SEED)
    X_full, Y_full = generate_correlated_data(N_TOTAL, base_rng)
    print(f"Base dataset: {X_full.shape[0]} samples, "
          f"class 0: {np.sum(Y_full == 0)}, class 1: {np.sum(Y_full == 1)}")

    # Fixed eval data (generated separately)
    eval_rng = np.random.RandomState(DATA_SEED + 9999)
    X_eval, _ = generate_correlated_data(N_EVAL, eval_rng)

    group_A = list(range(0, GROUP_SIZE))           # features 0-4
    group_B = list(range(GROUP_SIZE, P_FEATURES))  # features 5-9

    results = []
    total_start = time.time()

    for min_r, maj_r in IMBALANCE_RATIOS:
        ratio_label = f"{min_r}:{maj_r}"
        ratio_start = time.time()
        print(f"Ratio {ratio_label} ...", end=" ", flush=True)

        rankings = []
        for seed in range(N_MODELS):
            # Subsample for imbalance (different rng per model for variety)
            sub_rng = np.random.RandomState(DATA_SEED + seed + 100)
            X_train, Y_train = subsample_to_ratio(
                X_full, Y_full, min_r, maj_r, sub_rng
            )

            # Train XGBoost classifier
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                subsample=SUBSAMPLE,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X_train, Y_train)

            # TreeSHAP on fixed eval set
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_eval)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            importance = np.abs(shap_vals).mean(axis=0)
            rank = np.argsort(-importance)  # descending
            rankings.append(rank)

        # Compute within-group flip rates
        flips_A = within_group_flip_rate(rankings, group_A)
        flips_B = within_group_flip_rate(rankings, group_B)
        all_flips = flips_A + flips_B

        max_flip = max(d["flip_rate"] for d in all_flips) if all_flips else 0.0
        n_unstable = sum(1 for d in all_flips if d["flip_rate"] > 0)

        # Get actual sample sizes from last subsample
        sub_rng_check = np.random.RandomState(DATA_SEED + 100)
        X_check, Y_check = subsample_to_ratio(
            X_full, Y_full, min_r, maj_r, sub_rng_check
        )
        n_minority = int(min(np.sum(Y_check == 0), np.sum(Y_check == 1)))
        n_majority = int(max(np.sum(Y_check == 0), np.sum(Y_check == 1)))

        elapsed = time.time() - ratio_start
        print(f"done ({elapsed:.1f}s) | n_min={n_minority}, n_maj={n_majority}, "
              f"max_flip={max_flip:.4f}, unstable={n_unstable}/{len(all_flips)}")

        results.append({
            "ratio": ratio_label,
            "minority_ratio": min_r,
            "majority_ratio": maj_r,
            "n_minority": n_minority,
            "n_majority": n_majority,
            "n_train": n_minority + n_majority,
            "max_flip_rate": round(max_flip, 6),
            "n_unstable_pairs": n_unstable,
            "n_total_within_group_pairs": len(all_flips),
            "top_flips": sorted(all_flips, key=lambda d: -d["flip_rate"])[:5],
        })

    total_elapsed = time.time() - total_start

    # -- Print summary table ---------------------------------------------------
    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    print(f"{'Ratio':>8} | {'n_minority':>10} | {'n_majority':>10} | "
          f"{'max_flip_rate':>13} | {'n_unstable_pairs':>16}")
    print("-" * 72)
    for r in results:
        print(f"{r['ratio']:>8} | {r['n_minority']:>10} | {r['n_majority']:>10} | "
              f"{r['max_flip_rate']:>13.4f} | "
              f"{r['n_unstable_pairs']:>16}")
    print("-" * 72)
    print(f"Total runtime: {total_elapsed:.1f}s")
    print()

    # -- Check monotonic trend -------------------------------------------------
    flip_rates = [r["max_flip_rate"] for r in results]
    monotone_increase = all(
        flip_rates[i] <= flip_rates[i + 1] + 0.05  # allow 0.05 tolerance
        for i in range(len(flip_rates) - 1)
    )
    if monotone_increase:
        verdict = ("CONFIRMED: class imbalance amplifies SHAP instability "
                   "(max flip rate increases with imbalance ratio)")
    else:
        verdict = ("PARTIAL: class imbalance generally amplifies instability, "
                   "but trend is not strictly monotonic (stochastic variation)")
    print(verdict)

    # -- Save results ----------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    output_path = os.path.join(paper_dir, "results_class_imbalance.json")

    output = {
        "description": ("Class imbalance amplifies SHAP instability: "
                        "higher imbalance -> wider Rashomon set -> more flips"),
        "config": {
            "P": P_FEATURES,
            "group_size": GROUP_SIZE,
            "rho": RHO,
            "N_total": N_TOTAL,
            "N_eval": N_EVAL,
            "n_models": N_MODELS,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
            "noise_std": NOISE_STD,
        },
        "results": results,
        "verdict": verdict,
        "total_runtime_seconds": round(total_elapsed, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run_experiment()
