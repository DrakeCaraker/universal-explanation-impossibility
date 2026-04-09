#!/usr/bin/env python3
"""
Time-Series Feature Instability — Attribution Impossibility for temporal features.

Temporal features (lags, rolling means) are naturally correlated within each
series, creating the collinearity that triggers the Attribution Impossibility.
This script demonstrates that feature engineering on time-series data produces
correlated feature groups where SHAP-based rankings are unstable across
near-equivalent XGBoost models.

Setup (synthetic):
  - 10 raw features: 5 pairs of AR(1) processes with cross-correlation rho=0.8
  - Each raw series expanded to 3 features: [raw, lag1, rolling_mean_5]
  - Total: P = 30 features, highly correlated within each group
  - Target: Y(t) = sum of raw features + noise
  - 30 XGBoost models (different seeds, subsample=0.8)
  - TreeSHAP on 200 evaluation samples

Output:
  - Table: feature_pair_type | n_pairs | max_flip_rate | mean_flip_rate
  - Verdict on temporal feature instability
  - Saved to paper/results_timeseries.json

Note: Real stock return data could be substituted for the synthetic AR(1)
      processes; the collinearity structure would be similar or stronger.
"""

import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np

warnings.filterwarnings("ignore")

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

from scipy.stats import spearmanr

# ── Configuration ─────────────────────────────────────────────────────────────
N_RAW_FEATURES = 10          # 5 pairs of cross-correlated AR(1) processes
N_PAIRS = 5                  # number of AR(1) pairs
AR_COEFF = 0.7               # autoregressive coefficient
CROSS_COEFF = 0.3            # cross-correlation coefficient within pair
T_TOTAL = 2500               # total time steps
T_BURNIN = 500               # burn-in period (discard first 500)
ROLLING_WINDOW = 5           # rolling mean window size
N_MODELS = 30                # number of XGBoost models
N_EVAL = 200                 # evaluation samples for SHAP
MASTER_SEED = 42
NOISE_STD = 0.5              # target noise

# ── Generate synthetic AR(1) time-series data ────────────────────────────────

def generate_ar1_data(seed=MASTER_SEED):
    """
    Generate 10 AR(1) series in 5 cross-correlated pairs.

    For paired (j, k):
      X_j(t) = 0.7 * X_j(t-1) + 0.3 * X_k(t-1) + eps_j(t)
      X_k(t) = 0.7 * X_k(t-1) + 0.3 * X_j(t-1) + eps_k(t)

    Returns raw series of shape (T_TOTAL, 10).
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((T_TOTAL, N_RAW_FEATURES))

    # Initialize
    X[0, :] = rng.randn(N_RAW_FEATURES)

    for t in range(1, T_TOTAL):
        eps = rng.randn(N_RAW_FEATURES)
        for p in range(N_PAIRS):
            j = 2 * p
            k = 2 * p + 1
            X[t, j] = AR_COEFF * X[t - 1, j] + CROSS_COEFF * X[t - 1, k] + eps[j]
            X[t, k] = AR_COEFF * X[t - 1, k] + CROSS_COEFF * X[t - 1, j] + eps[k]

    return X


def build_features(X_raw):
    """
    Expand each raw series to 3 features: [raw, lag1, rolling_mean_5].

    Returns (feature_matrix, feature_names, feature_group_map).
    feature_group_map[col_idx] = (raw_series_idx, transform_type).
    """
    T, P = X_raw.shape
    features = []
    names = []
    group_map = {}
    col = 0

    for j in range(P):
        # Raw
        features.append(X_raw[:, j])
        names.append(f"X{j}_raw")
        group_map[col] = (j, "raw")
        col += 1

        # Lag-1
        lag1 = np.zeros(T)
        lag1[1:] = X_raw[:-1, j]
        lag1[0] = 0.0
        features.append(lag1)
        names.append(f"X{j}_lag1")
        group_map[col] = (j, "lag1")
        col += 1

        # Rolling mean (window=5)
        rm = np.zeros(T)
        for t in range(T):
            start = max(0, t - ROLLING_WINDOW + 1)
            rm[t] = X_raw[start:t + 1, j].mean()
        features.append(rm)
        names.append(f"X{j}_rmean5")
        group_map[col] = (j, "rmean5")
        col += 1

    feature_matrix = np.column_stack(features)
    return feature_matrix, names, group_map


def classify_pair(group_map, i, j):
    """
    Classify a feature pair into one of:
      - within_series: same raw series, different transforms
      - cross_series_same_group: different raw series in the same AR(1) pair
      - cross_series_diff_group: different raw series in different AR(1) pairs
    """
    series_i, _ = group_map[i]
    series_j, _ = group_map[j]

    if series_i == series_j:
        return "within_series"

    pair_i = series_i // 2
    pair_j = series_j // 2

    if pair_i == pair_j:
        return "cross_series_same_group"
    else:
        return "cross_series_diff_group"


def flip_rate_from_shap(shap_arrays, i, j):
    """
    Compute flip rate for features i and j across models.

    A flip occurs when model a ranks i > j but model b ranks j > i
    (based on mean |SHAP| importance).

    Returns fraction of model pairs that disagree.
    """
    n = len(shap_arrays)
    # For each model, does feature i have higher mean |SHAP| than j?
    orderings = []
    for imp in shap_arrays:
        orderings.append(imp[i] > imp[j])

    flips = 0
    total = 0
    for a in range(n):
        for b in range(a + 1, n):
            if orderings[a] != orderings[b]:
                flips += 1
            total += 1

    return flips / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Time-Series Feature Instability")
    print("  Attribution Impossibility for temporal (lag/rolling) features")
    print(f"  {N_RAW_FEATURES} raw AR(1) series -> {N_RAW_FEATURES * 3} engineered features")
    print(f"  {N_MODELS} XGBoost models, {N_EVAL} eval samples")
    print("=" * 70)

    # Step 1: Generate data
    print("\nGenerating synthetic AR(1) time-series data...")
    X_raw = generate_ar1_data(seed=MASTER_SEED)

    # Target: sum of raw features + noise
    rng = np.random.RandomState(MASTER_SEED + 1000)
    y_full = X_raw.sum(axis=1) + NOISE_STD * rng.randn(T_TOTAL)

    # Build engineered features
    X_feat, feat_names, group_map = build_features(X_raw)

    # Discard burn-in
    X_feat = X_feat[T_BURNIN:]
    y_full = y_full[T_BURNIN:]
    T_use = X_feat.shape[0]  # 2000
    n_features = X_feat.shape[1]  # 30

    print(f"  Time steps after burn-in: {T_use}")
    print(f"  Engineered features: {n_features}")
    print(f"  Feature names: {feat_names[:6]} ... (30 total)")

    # Train/eval split (last N_EVAL as eval)
    X_train = X_feat[:T_use - N_EVAL]
    y_train = y_full[:T_use - N_EVAL]
    X_eval = X_feat[T_use - N_EVAL:]
    y_eval = y_full[T_use - N_EVAL:]

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Eval samples: {X_eval.shape[0]}")

    # Check within-group correlations
    corr_matrix = np.corrcoef(X_train.T)
    print("\n  Sample within-series correlations:")
    for j in range(min(3, N_RAW_FEATURES)):
        raw_idx = j * 3
        lag_idx = j * 3 + 1
        rm_idx = j * 3 + 2
        print(f"    X{j} raw-lag1: {corr_matrix[raw_idx, lag_idx]:.3f}, "
              f"raw-rmean5: {corr_matrix[raw_idx, rm_idx]:.3f}, "
              f"lag1-rmean5: {corr_matrix[lag_idx, rm_idx]:.3f}")

    # Step 2: Train N_MODELS XGBoost models
    print(f"\nTraining {N_MODELS} XGBoost models...")
    shap_importances = []

    for s in range(N_MODELS):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            n_jobs=1,
            random_state=MASTER_SEED + s,
        )
        model.fit(X_train, y_train, verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_eval)
        importance = np.abs(shap_vals).mean(axis=0)
        shap_importances.append(importance)

        if (s + 1) % 10 == 0:
            print(f"  Trained and explained model {s + 1}/{N_MODELS}")

    print(f"  All {N_MODELS} models trained and explained.")

    # Step 3: Identify correlated pairs and compute flip rates
    print("\nComputing flip rates by pair type...")

    # Only consider pairs with |correlation| >= 0.5
    CORR_THRESHOLD = 0.5
    pair_stats = {
        "within_series": [],
        "cross_series_same_group": [],
        "cross_series_diff_group": [],
    }

    for i, j in combinations(range(n_features), 2):
        rho = abs(corr_matrix[i, j])
        if rho < CORR_THRESHOLD:
            continue

        ptype = classify_pair(group_map, i, j)
        fr = flip_rate_from_shap(shap_importances, i, j)
        pair_stats[ptype].append({
            "feature_i": feat_names[i],
            "feature_j": feat_names[j],
            "rho": float(rho),
            "flip_rate": float(fr),
        })

    # Step 4: Compute Spearman correlations between models
    print("Computing pairwise Spearman rank correlations...")
    spearman_corrs = []
    for a in range(N_MODELS):
        for b in range(a + 1, N_MODELS):
            corr, _ = spearmanr(shap_importances[a], shap_importances[b])
            spearman_corrs.append(corr)

    mean_spearman = float(np.mean(spearman_corrs))
    min_spearman = float(np.min(spearman_corrs))

    # Step 5: Build summary table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    table_rows = []
    header = f"{'pair_type':<30} {'n_pairs':>8} {'max_flip':>10} {'mean_flip':>10}"
    print(f"\n{header}")
    print("-" * 60)

    for ptype in ["within_series", "cross_series_same_group", "cross_series_diff_group"]:
        pairs = pair_stats[ptype]
        n_pairs = len(pairs)
        if n_pairs > 0:
            max_flip = max(p["flip_rate"] for p in pairs)
            mean_flip = float(np.mean([p["flip_rate"] for p in pairs]))
        else:
            max_flip = 0.0
            mean_flip = 0.0

        print(f"{ptype:<30} {n_pairs:>8} {max_flip:>10.4f} {mean_flip:>10.4f}")
        table_rows.append({
            "pair_type": ptype,
            "n_pairs": n_pairs,
            "max_flip_rate": round(max_flip, 4),
            "mean_flip_rate": round(mean_flip, 4),
        })

    print(f"\nPairwise Spearman rank correlations across models:")
    print(f"  Mean: {mean_spearman:.4f}")
    print(f"  Min:  {min_spearman:.4f}")

    # Top unstable within-series pairs
    within_sorted = sorted(pair_stats["within_series"], key=lambda p: -p["flip_rate"])
    print(f"\nTop 10 most unstable within-series pairs:")
    for p in within_sorted[:10]:
        print(f"  {p['feature_i']} vs {p['feature_j']}: "
              f"rho={p['rho']:.3f}, flip_rate={p['flip_rate']:.4f}")

    # Top unstable cross-series same-group pairs
    cross_same_sorted = sorted(pair_stats["cross_series_same_group"],
                               key=lambda p: -p["flip_rate"])
    if cross_same_sorted:
        print(f"\nTop 5 most unstable cross-series same-group pairs:")
        for p in cross_same_sorted[:5]:
            print(f"  {p['feature_i']} vs {p['feature_j']}: "
                  f"rho={p['rho']:.3f}, flip_rate={p['flip_rate']:.4f}")

    # Verdict
    all_within_flips = [p["flip_rate"] for p in pair_stats["within_series"]]
    pct_unstable_within = (
        100.0 * sum(1 for f in all_within_flips if f > 0.10) / max(len(all_within_flips), 1)
    )

    print("\n" + "-" * 70)
    verdict = (
        f"Temporal feature engineering creates correlated groups subject to "
        f"the impossibility. {pct_unstable_within:.0f}% of within-series pairs "
        f"have flip rate > 10%."
    )
    print(verdict)
    print("-" * 70)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment": "timeseries_instability",
        "n_raw_features": N_RAW_FEATURES,
        "n_engineered_features": n_features,
        "ar_coeff": AR_COEFF,
        "cross_coeff": CROSS_COEFF,
        "t_total": T_TOTAL,
        "t_burnin": T_BURNIN,
        "t_used": T_use,
        "rolling_window": ROLLING_WINDOW,
        "n_models": N_MODELS,
        "n_eval": N_EVAL,
        "correlation_threshold": CORR_THRESHOLD,
        "table": table_rows,
        "rank_correlations": {
            "mean_spearman": round(mean_spearman, 4),
            "min_spearman": round(min_spearman, 4),
        },
        "top_unstable_within_series": [
            {
                "feature_i": p["feature_i"],
                "feature_j": p["feature_j"],
                "rho": p["rho"],
                "flip_rate": p["flip_rate"],
            }
            for p in within_sorted[:10]
        ],
        "top_unstable_cross_same_group": [
            {
                "feature_i": p["feature_i"],
                "feature_j": p["feature_j"],
                "rho": p["rho"],
                "flip_rate": p["flip_rate"],
            }
            for p in cross_same_sorted[:5]
        ],
        "pct_within_series_unstable": round(pct_unstable_within, 1),
        "verdict": verdict,
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(out_dir, "results_timeseries.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
