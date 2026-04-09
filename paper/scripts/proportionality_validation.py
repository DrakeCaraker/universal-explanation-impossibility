#!/usr/bin/env python3
"""
Validate the proportionality axiom phi_j = c * n_j on the Breast Cancer dataset.

For each of 50 XGBoost models (different seeds), we:
  - Use a different train/test split (seed varies the split)
  - Add stochastic training (subsample, colsample_bytree) for model diversity
  - Compute mean |SHAP_j| per feature via TreeExplainer on 200 test samples
  - Compute split count n_j per feature via get_booster().get_score(importance_type='weight')
  - Compute proportionality "constant" c_j = SHAP_j / n_j (features with n_j=0 excluded)
  - Compute coefficient of variation CV = std(c) / mean(c)

We repeat for max_depth in {1, 3, 6, 10} and report which features break
proportionality the most.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap


def run_single_model(X, y, feature_names, seed, max_depth, lr=0.1):
    """Train one XGBoost model and return (cv_of_c, c_per_feature_dict)."""
    # Different train/test split per seed for genuine model diversity
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # --- SHAP values (mean |SHAP_j|) on up to 200 test samples ---
    n_test = min(200, len(X_test))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:n_test])
    # For binary classification shap may return a list or a single array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive-class explanations
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # shape (n_features,)

    # --- Split counts n_j ---
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="weight")
    # score_dict keys are like "f0", "f1", ...
    n_features = len(feature_names)
    split_counts = np.zeros(n_features)
    for key, val in score_dict.items():
        idx = int(key[1:])  # "f12" -> 12
        split_counts[idx] = val

    # --- Proportionality constant c_j = SHAP_j / n_j ---
    # Exclude features with n_j == 0
    mask = split_counts > 0
    if mask.sum() < 2:
        return None, None, mask

    c = mean_abs_shap[mask] / split_counts[mask]

    cv = np.std(c) / np.mean(c) if np.mean(c) > 0 else np.nan

    # Build per-feature c dict (only for features with splits)
    c_dict = {}
    for i in range(n_features):
        if split_counts[i] > 0:
            c_dict[feature_names[i]] = mean_abs_shap[i] / split_counts[i]

    return cv, c_dict, mask


def main():
    # ---- Data ----
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    depths = [1, 3, 6, 10]
    n_models = 50

    print("=" * 72)
    print("Proportionality Axiom Validation: phi_j = c * n_j")
    print("Dataset: Breast Cancer (sklearn), 50 XGBoost models per depth")
    print("XGBoost: n_estimators=100, lr=0.1, subsample=0.8, colsample=0.8")
    print("Each model uses a different train/test split (seed 0..49)")
    print("=" * 72)

    depth_cv_means = {}
    depth_cv_stds = {}

    # Track per-feature variability at the reference depth (6)
    feature_c_all = {fn: [] for fn in feature_names}

    for depth in depths:
        cvs = []
        for seed in range(n_models):
            cv, c_dict, _ = run_single_model(
                X, y, feature_names, seed, max_depth=depth
            )
            if cv is not None and not np.isnan(cv):
                cvs.append(cv)
            # Collect per-feature c at reference depth
            if depth == 6 and c_dict is not None:
                for fn, c_val in c_dict.items():
                    feature_c_all[fn].append(c_val)

        mean_cv = np.mean(cvs)
        std_cv = np.std(cvs)
        depth_cv_means[depth] = mean_cv
        depth_cv_stds[depth] = std_cv

        print(f"\nmax_depth = {depth}:")
        print(f"  Models evaluated: {len(cvs)}")
        print(f"  Mean CV of c_j: {mean_cv:.6f}  (std across models: {std_cv:.6f})")
        print(f"  Min CV: {min(cvs):.6f}  Max CV: {max(cvs):.6f}")
        print(f"  => Proportionality holds within ~{mean_cv * 100:.1f}%")

    # ---- Which features have the most variable c (at depth=6)? ----
    print("\n" + "-" * 72)
    print("Per-feature variability of c_j = SHAP_j / n_j  (depth=6, across 50 models)")
    print("-" * 72)

    feature_cv = {}
    for fn in feature_names:
        vals = feature_c_all[fn]
        if len(vals) >= 2 and np.mean(vals) > 0:
            feature_cv[fn] = np.std(vals) / np.mean(vals)

    # Sort by CV descending
    sorted_features = sorted(feature_cv.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<30s} {'CV of c_j':>12s}  {'Mean c_j':>12s}  "
          f"{'# models':>8s}")
    print("-" * 68)
    for fn, cv_val in sorted_features:
        vals = feature_c_all[fn]
        mean_c = np.mean(vals)
        n_obs = len(vals)
        print(f"  {fn:<28s} {cv_val:>12.6f}  {mean_c:>12.6f}  {n_obs:>8d}")

    top5 = sorted_features[:5]
    print(f"\nTop 5 features where proportionality breaks most (highest cross-model CV):")
    for fn, cv_val in top5:
        print(f"  {fn}: CV = {cv_val:.6f}")

    bot5 = sorted_features[-5:]
    print(f"\nTop 5 features where proportionality is most stable (lowest cross-model CV):")
    for fn, cv_val in bot5:
        print(f"  {fn}: CV = {cv_val:.6f}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print("\n  Depth  |  Mean CV  |  Std CV   |  ~Holds within")
    print("  -------+-----------+-----------+----------------")
    for d in depths:
        print(f"    {d:>2d}   | {depth_cv_means[d]:>9.4f} | {depth_cv_stds[d]:>9.4f} | "
              f"~{depth_cv_means[d] * 100:.1f}%")

    # Find largest depth where mean CV <= threshold
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        qualifying = [d for d in depths if depth_cv_means[d] <= threshold]
        if qualifying:
            max_d = max(qualifying)
            pct = threshold * 100
            print(f"\n  Proportionality holds within {pct:.0f}% (CV <= {threshold}) "
                  f"for depth <= {max_d}")

    best_depth = min(depth_cv_means, key=depth_cv_means.get)
    worst_depth = max(depth_cv_means, key=depth_cv_means.get)
    print(f"\n  Best depth: {best_depth} "
          f"(mean CV = {depth_cv_means[best_depth]:.4f}, "
          f"~{depth_cv_means[best_depth] * 100:.1f}%)")
    print(f"  Worst depth: {worst_depth} "
          f"(mean CV = {depth_cv_means[worst_depth]:.4f}, "
          f"~{depth_cv_means[worst_depth] * 100:.1f}%)")

    # Final one-liner
    overall_mean = np.mean(list(depth_cv_means.values()))
    print(f"\n  Overall mean CV across all depths: {overall_mean:.4f} "
          f"(~{overall_mean * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
