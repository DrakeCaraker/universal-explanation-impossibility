"""
Longitudinal Retraining — Track SHAP instability over 50 retrain cycles
with simulated data drift.

Setup:
  - Breast Cancer dataset (sklearn)
  - 50 rounds: each round adds 5% Gaussian noise to features (cumulative drift)
  - At each round: train XGBoost (n_estimators=100, max_depth=6, subsample=0.8,
    random_state=round)
  - Compute TreeSHAP on a fixed test set (200 samples)
  - Record feature ranking (by mean |SHAP|) at each round

Metrics per round:
  - Spearman correlation between round t and round 0 rankings
  - Cumulative flip count (pairs that changed order from round 0)
  - Max flip rate across all correlated feature pairs at each round

Output: table → paper/results_longitudinal.json.
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ── Configuration ─────────────────────────────────────────────────────────────
N_ROUNDS = 50                   # retrain cycles
NOISE_FRACTION = 0.05           # 5% Gaussian noise per round (cumulative)
N_ESTIMATORS = 100
MAX_DEPTH = 6
SUBSAMPLE = 0.8
LEARNING_RATE = 0.1
N_EVAL = 200                    # fixed test set size for SHAP
MASTER_SEED = 42
CORR_THRESHOLD = 0.7            # pairs with |Pearson r| > this are "correlated"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ranking(mean_abs_shap):
    """Return rank array (1 = most important) from mean |SHAP| values."""
    # argsort of negative values gives descending order
    order = np.argsort(-mean_abs_shap)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(mean_abs_shap) + 1)
    return ranks


def count_flips(ranking_a, ranking_b):
    """Count the number of pairwise flips between two rankings."""
    n = len(ranking_a)
    flips = 0
    for i, j in combinations(range(n), 2):
        # A flip occurs when the relative order of i, j differs
        order_a = ranking_a[i] < ranking_a[j]  # i ranked higher in a
        order_b = ranking_b[i] < ranking_b[j]  # i ranked higher in b
        if order_a != order_b:
            flips += 1
    return flips


def find_correlated_pairs(X, threshold):
    """Find feature pairs with |Pearson correlation| > threshold."""
    corr_matrix = np.corrcoef(X, rowvar=False)
    n_features = X.shape[1]
    pairs = []
    for i, j in combinations(range(n_features), 2):
        if abs(corr_matrix[i, j]) > threshold:
            pairs.append((i, j))
    return pairs


def max_flip_rate_correlated(rankings_list, corr_pairs):
    """Compute the max flip rate across correlated pairs, comparing each
    round's ranking against round 0.

    For a single round t, the 'flip rate' for pair (i,j) is 1 if the order
    flipped from round 0 to round t, 0 otherwise. The max flip rate at round t
    is the fraction of correlated pairs that have flipped from round 0.
    """
    if not corr_pairs:
        return [0.0] * len(rankings_list)

    base_ranking = rankings_list[0]
    rates = [0.0]  # round 0 vs itself = 0

    for t in range(1, len(rankings_list)):
        current = rankings_list[t]
        flipped = 0
        for i, j in corr_pairs:
            order_base = base_ranking[i] < base_ranking[j]
            order_curr = current[i] < current[j]
            if order_base != order_curr:
                flipped += 1
        rates.append(flipped / len(corr_pairs))

    return rates


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Load data
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target
    feature_names = list(data.feature_names)

    # Fixed train/test split
    X_train_base, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=N_EVAL, random_state=MASTER_SEED
    )

    print(f"Longitudinal retraining: {N_ROUNDS} rounds, "
          f"{NOISE_FRACTION*100:.0f}% cumulative noise per round")
    print(f"Train: {X_train_base.shape}, Test: {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    # Identify correlated pairs on the original data
    corr_pairs = find_correlated_pairs(X_train_base, CORR_THRESHOLD)
    print(f"Correlated pairs (|r| > {CORR_THRESHOLD}): {len(corr_pairs)}")
    print()

    rng = np.random.default_rng(MASTER_SEED)

    rankings_list = []      # list of rank arrays, one per round
    mean_shap_list = []     # list of mean |SHAP| arrays
    round_results = []

    # Feature standard deviations for noise scaling
    feature_stds = X_train_base.std(axis=0)

    for rnd in range(N_ROUNDS):
        t_rnd = time.time()

        # Apply cumulative noise: round t adds t * 5% noise
        noise_scale = rnd * NOISE_FRACTION
        if noise_scale > 0:
            noise = rng.normal(0, 1, size=X_train_base.shape) * feature_stds * noise_scale
            X_train = X_train_base + noise
        else:
            X_train = X_train_base.copy()

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            subsample=SUBSAMPLE,
            learning_rate=LEARNING_RATE,
            n_jobs=1,
            random_state=rnd,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Compute TreeSHAP on fixed test set
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # use class 1

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        ranking = get_ranking(mean_abs_shap)

        rankings_list.append(ranking)
        mean_shap_list.append(mean_abs_shap)

        # Spearman vs round 0
        if rnd == 0:
            spearman_vs_r0 = 1.0
            cum_flips = 0
        else:
            corr, _ = spearmanr(rankings_list[0], ranking)
            spearman_vs_r0 = round(float(corr), 4)
            cum_flips = count_flips(rankings_list[0], ranking)

        elapsed = time.time() - t_rnd

        round_results.append({
            "round": rnd,
            "noise_scale": round(noise_scale, 4),
            "spearman_vs_round0": spearman_vs_r0,
            "cumulative_flips": int(cum_flips),
        })

        if rnd % 10 == 0 or rnd == N_ROUNDS - 1:
            print(f"  Round {rnd:3d}: noise={noise_scale:.2f}  "
                  f"spearman={spearman_vs_r0:.4f}  flips={cum_flips:3d}  "
                  f"({elapsed:.2f}s)")

    # Compute max flip rate across correlated pairs at each round
    flip_rates = max_flip_rate_correlated(rankings_list, corr_pairs)
    for i, r in enumerate(round_results):
        r["max_flip_rate"] = round(flip_rates[i], 4)

    total_time = time.time() - t0

    # Summary table
    print()
    print("=" * 76)
    print("LONGITUDINAL RETRAINING RESULTS")
    print("=" * 76)
    print(f"{'Round':>5}  {'Noise%':>6}  {'Spearman':>8}  {'Flips':>5}  {'FlipRate':>8}")
    print("-" * 76)
    for r in round_results:
        print(f"{r['round']:5d}  {r['noise_scale']*100:5.1f}%  "
              f"{r['spearman_vs_round0']:8.4f}  {r['cumulative_flips']:5d}  "
              f"{r['max_flip_rate']:8.4f}")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Correlated pairs tracked: {len(corr_pairs)}")

    # Compute total features for pair count context
    n_features = len(feature_names)
    total_pairs = n_features * (n_features - 1) // 2
    print(f"Total feature pairs: {total_pairs}")

    # Save results
    output = {
        "description": "Longitudinal retraining with simulated data drift",
        "settings": {
            "dataset": "breast_cancer",
            "n_rounds": N_ROUNDS,
            "noise_fraction_per_round": NOISE_FRACTION,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
            "learning_rate": LEARNING_RATE,
            "n_eval": N_EVAL,
            "master_seed": MASTER_SEED,
            "corr_threshold": CORR_THRESHOLD,
            "n_features": n_features,
            "n_correlated_pairs": len(corr_pairs),
            "total_pairs": total_pairs,
        },
        "total_time_seconds": round(total_time, 1),
        "rounds": round_results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "results_longitudinal.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
