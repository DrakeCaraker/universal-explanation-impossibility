#!/usr/bin/env python3
"""
Multiverse Layer Decomposition: Within-class vs Cross-class Analysis

The expanded multiverse experiment showed rho > 0.6 on only 2/5 datasets when
mixing model classes. This decomposition tests whether:

1. WITHIN-CLASS multiverse (same model type, different seeds/scalers) is predictable
   by the Gaussian flip formula (expected: yes, this is the Rashomon regime)
2. CROSS-CLASS multiverse (different model types) is NOT predictable
   (expected: no, different model classes use different importance definitions)

The bilemma predicts: instability WITHIN a model class follows the Rashomon structure
(Gaussian flip applies). Instability ACROSS model classes follows a different mechanism
(different importance definitions, not the same Rashomon set).

This is the multiverse analogue of the within-group/between-group distinction.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def get_importance(model, model_class, X_test):
    """Extract feature importance from a fitted model."""
    if model_class in ('xgboost', 'random_forest'):
        return model.feature_importances_
    elif model_class == 'ridge':
        return np.abs(model.coef_.ravel())
    return None


def train_pipeline(X, y, seed, model_class, scaler_name, is_classifier=True):
    """Train a single pipeline and return importance vector."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=len(X), replace=True)
    X_train, y_train = X[idx], y[idx]

    # Scale
    if scaler_name == 'standard':
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
    elif scaler_name == 'minmax':
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)

    # Train
    if model_class == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=seed,
            verbosity=0, tree_method='hist', eval_metric='logloss'
        )
    elif model_class == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=seed
        )
    elif model_class == 'ridge':
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(alpha=1.0)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.fit(X_train, y_train)
    return get_importance(model, model_class, X_train)


def gaussian_flip_rate(imp_matrix, j, k):
    """Compute Gaussian-predicted flip rate for feature pair (j, k)."""
    diff = imp_matrix[:, j] - imp_matrix[:, k]
    mu = np.mean(diff)
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0 if abs(mu) > 1e-12 else 0.5
    return float(norm.cdf(-abs(mu) / sd))


def observed_flip_rate(imp_matrix, j, k):
    """Compute observed flip rate for feature pair (j, k)."""
    n = imp_matrix.shape[0]
    disagree = 0
    total = 0
    for m1 in range(n):
        for m2 in range(m1 + 1, n):
            d1 = imp_matrix[m1, j] - imp_matrix[m1, k]
            d2 = imp_matrix[m2, j] - imp_matrix[m2, k]
            if d1 * d2 < 0:
                disagree += 1
            total += 1
    return disagree / total if total > 0 else 0.0


def analyze_layer(imp_cal, imp_val, layer_name, max_pairs=200):
    """Analyze a single layer (within-class or cross-class) using cal/val split."""
    P = imp_cal.shape[1]
    all_pairs = list(combinations(range(P), 2))

    if len(all_pairs) > max_pairs:
        rng = np.random.RandomState(42)
        pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=max_pairs, replace=False)]
    else:
        pairs = all_pairs

    predicted = []
    observed = []

    for j, k in pairs:
        pred = gaussian_flip_rate(imp_cal, j, k)
        obs = observed_flip_rate(imp_val, j, k)
        predicted.append(pred)
        observed.append(obs)

    predicted = np.array(predicted)
    observed = np.array(observed)

    # Correlation
    if len(predicted) > 2 and np.std(predicted) > 1e-12 and np.std(observed) > 1e-12:
        rho, p_val = spearmanr(predicted, observed)
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    else:
        rho, p_val, r2 = 0.0, 1.0, 0.0

    return {
        "layer": layer_name,
        "n_pairs": len(pairs),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "oos_r2": float(r2),
        "mean_predicted": float(np.mean(predicted)),
        "mean_observed": float(np.mean(observed)),
    }


def run_dataset(name, X, y):
    """Run layer decomposition for one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {name} (P={X.shape[1]}, N={X.shape[0]})")
    print(f"{'='*60}")
    t0 = time.time()

    model_classes = ['xgboost', 'random_forest', 'ridge']
    scalers = ['none', 'standard', 'minmax']
    cal_seeds = list(range(42, 52))   # 10 cal seeds
    val_seeds = list(range(142, 152)) # 10 val seeds

    P = X.shape[1]

    # Train all pipelines
    within_class_results = {}
    all_cal_imps = []
    all_val_imps = []

    for mc in model_classes:
        print(f"\n  Model class: {mc}")

        # Within-class: same model, different seeds × scalers
        cal_imps = []
        val_imps = []

        for scaler in scalers:
            for seed in cal_seeds:
                imp = train_pipeline(X, y, seed, mc, scaler)
                if imp is not None:
                    cal_imps.append(imp)
                    all_cal_imps.append((mc, imp))

            for seed in val_seeds:
                imp = train_pipeline(X, y, seed, mc, scaler)
                if imp is not None:
                    val_imps.append(imp)
                    all_val_imps.append((mc, imp))

        cal_matrix = np.array(cal_imps)  # (30, P) — 10 seeds × 3 scalers
        val_matrix = np.array(val_imps)

        print(f"    Cal models: {cal_matrix.shape[0]}, Val models: {val_matrix.shape[0]}")

        layer_result = analyze_layer(cal_matrix, val_matrix, f"within_{mc}")
        within_class_results[mc] = layer_result
        print(f"    Within-class rho: {layer_result['spearman_rho']:.3f}, "
              f"R²: {layer_result['oos_r2']:.3f}")

    # Cross-class analysis: mix all model classes
    all_cal_matrix = np.array([imp for _, imp in all_cal_imps])  # (90, P)
    all_val_matrix = np.array([imp for _, imp in all_val_imps])

    print(f"\n  Cross-class: Cal={all_cal_matrix.shape[0]}, Val={all_val_matrix.shape[0]}")
    cross_result = analyze_layer(all_cal_matrix, all_val_matrix, "cross_class")
    print(f"    Cross-class rho: {cross_result['spearman_rho']:.3f}, "
          f"R²: {cross_result['oos_r2']:.3f}")

    # Compute within-class average
    within_rhos = [within_class_results[mc]["spearman_rho"] for mc in model_classes]
    within_r2s = [within_class_results[mc]["oos_r2"] for mc in model_classes]
    mean_within_rho = float(np.mean(within_rhos))
    mean_within_r2 = float(np.mean(within_r2s))

    elapsed = time.time() - t0

    return {
        "dataset": name,
        "n_features": P,
        "n_samples": X.shape[0],
        "within_class": within_class_results,
        "cross_class": cross_result,
        "mean_within_rho": mean_within_rho,
        "mean_within_r2": mean_within_r2,
        "cross_rho": cross_result["spearman_rho"],
        "cross_r2": cross_result["oos_r2"],
        "within_better_than_cross": mean_within_rho > cross_result["spearman_rho"],
        "elapsed_seconds": round(elapsed, 1),
    }


# =====================================================================
# Datasets
# =====================================================================

def load_datasets():
    datasets = []

    # 1. Breast Cancer
    bc = load_breast_cancer()
    datasets.append(("Breast Cancer", bc.data, bc.target))

    # 2. Wine
    wine = load_wine()
    y_wine = (wine.target > 0).astype(int)  # Binary for Ridge compatibility
    datasets.append(("Wine", wine.data, y_wine))

    # 3. Heart Disease
    try:
        X_h, y_h = fetch_openml('heart-statlog', version=1, return_X_y=True, as_frame=False, parser='auto')
        y_h = LabelEncoder().fit_transform(y_h)
        datasets.append(("Heart Disease", X_h.astype(float), y_h))
    except Exception as e:
        print(f"  Heart Disease failed: {e}")

    # 4. Diabetes
    try:
        X_d, y_d = fetch_openml('diabetes', version=1, return_X_y=True, as_frame=False, parser='auto')
        y_d = LabelEncoder().fit_transform(y_d)
        datasets.append(("Diabetes", X_d.astype(float), y_d))
    except Exception as e:
        print(f"  Diabetes failed: {e}")

    # 5. German Credit
    try:
        X_g, y_g = fetch_openml('credit-g', version=1, return_X_y=True, as_frame=True, parser='auto')
        for col in X_g.columns:
            if X_g[col].dtype == 'object' or X_g[col].dtype.name == 'category':
                X_g[col] = OrdinalEncoder().fit_transform(X_g[[col]]).ravel()
        X_g = X_g.values.astype(float)
        y_g = LabelEncoder().fit_transform(y_g)
        datasets.append(("German Credit", X_g, y_g))
    except Exception as e:
        print(f"  German Credit failed: {e}")

    return datasets


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("Multiverse Layer Decomposition")
    print("=" * 60)
    print("PREDICTION: Within-class multiverse is predictable (Rashomon)")
    print("            Cross-class multiverse is NOT (different mechanisms)\n")

    datasets = load_datasets()
    all_results = {}

    for name, X, y in datasets:
        r = run_dataset(name, X, y)
        all_results[name] = r

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Layer Decomposition")
    print("=" * 60)

    within_rhos = []
    cross_rhos = []
    separation_confirmed = 0

    for name, r in all_results.items():
        w = r["mean_within_rho"]
        c = r["cross_rho"]
        within_rhos.append(w)
        cross_rhos.append(c)
        sep = w - c
        if r["within_better_than_cross"]:
            separation_confirmed += 1

        print(f"\n  {name}:")
        print(f"    Within-class mean ρ: {w:.3f}")
        print(f"    Cross-class ρ:       {c:.3f}")
        print(f"    Separation:          {sep:+.3f}")
        print(f"    Within > Cross:      {r['within_better_than_cross']}")

        # Per model class detail
        for mc, wr in r["within_class"].items():
            print(f"      {mc}: ρ={wr['spearman_rho']:.3f}, R²={wr['oos_r2']:.3f}")

    mean_within = float(np.mean(within_rhos))
    mean_cross = float(np.mean(cross_rhos))

    print(f"\n  {'='*40}")
    print(f"  Mean within-class ρ: {mean_within:.3f}")
    print(f"  Mean cross-class ρ:  {mean_cross:.3f}")
    print(f"  Separation confirmed: {separation_confirmed}/{len(all_results)} datasets")
    print(f"  {'='*40}")

    # Save
    output = {
        "experiment": "multiverse_layer_decomposition",
        "description": "Within-class vs cross-class Gaussian flip prediction",
        "prediction": "Within-class (same model type) is more predictable than cross-class",
        "n_datasets": len(all_results),
        "mean_within_rho": mean_within,
        "mean_cross_rho": mean_cross,
        "separation_confirmed_count": separation_confirmed,
        "separation_confirmed_rate": separation_confirmed / len(all_results) if all_results else 0,
        "per_dataset": all_results,
    }

    json_path = OUT_DIR / 'results_multiverse_layer_decomposition.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("Done.")
