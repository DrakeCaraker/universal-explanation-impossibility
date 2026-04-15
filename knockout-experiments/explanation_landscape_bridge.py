#!/usr/bin/env python3
"""
Explanation Landscape Bridge: Coverage Conflict → SNR → Flip Rate

Validates the formal bridge chain:
  Characterization (Lean) → Coverage Conflict (Lean) → SNR (statistics) → Flip Rate (empirical)

For each dataset, computes:
1. Per-pair coverage conflict: does a "compatible" direction exist?
   Operationalized: a pair has coverage conflict if its SNR < threshold
2. Coverage conflict degree: fraction of pairs with coverage conflict
3. Observed mean instability (flip rate)

Prediction: coverage conflict degree predicts observed instability.
The Gaussian flip formula's SNR IS the continuous measure of per-pair
fiber compatibility from the bilemma characterization.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import xgboost as xgb
from scipy.stats import norm, spearmanr, pearsonr
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEEDS_CAL = list(range(42, 72))
SEEDS_VAL = list(range(142, 172))


def train_models(X, y, seeds, n_classes=2):
    n = len(seeds)
    P = X.shape[1]
    imp = np.zeros((n, P))
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        params = dict(n_estimators=100, max_depth=4, random_state=seed,
                      verbosity=0, tree_method='hist')
        if n_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
        else:
            params['eval_metric'] = 'logloss'
        model = xgb.XGBClassifier(**params)
        model.fit(X[idx], y[idx])
        imp[i] = model.feature_importances_
    return imp


def compute_landscape(imp_cal, imp_val, snr_threshold=0.5):
    """Compute the landscape bridge metrics for one dataset."""
    P = imp_cal.shape[1]
    pairs = list(combinations(range(P), 2))
    if len(pairs) > 500:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=500, replace=False)]

    snrs = []
    predicted_flips = []
    observed_flips = []

    for j, k in pairs:
        # Calibration: compute SNR
        diff_cal = imp_cal[:, j] - imp_cal[:, k]
        mu = np.mean(diff_cal)
        sd = np.std(diff_cal, ddof=1)
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred_flip = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

        # Validation: compute observed flip rate
        n_val = imp_val.shape[0]
        disagree = 0
        total = 0
        for m1 in range(n_val):
            for m2 in range(m1 + 1, n_val):
                d1 = imp_val[m1, j] - imp_val[m1, k]
                d2 = imp_val[m2, j] - imp_val[m2, k]
                if d1 * d2 < 0:
                    disagree += 1
                total += 1
        obs_flip = disagree / total if total > 0 else 0.0

        snrs.append(snr)
        predicted_flips.append(pred_flip)
        observed_flips.append(obs_flip)

    snrs = np.array(snrs)
    predicted_flips = np.array(predicted_flips)
    observed_flips = np.array(observed_flips)

    # Coverage conflict degree: fraction of pairs with SNR < threshold
    coverage_conflict_degree = float(np.mean(snrs < snr_threshold))

    # Mean observed instability
    mean_instability = float(np.mean(observed_flips))

    # Fraction of reliable pairs (SNR > 2)
    reliable_fraction = float(np.mean(snrs > 2.0))

    # Gaussian flip R² (OOS)
    if np.std(predicted_flips) > 1e-12 and np.std(observed_flips) > 1e-12:
        ss_res = np.sum((observed_flips - predicted_flips) ** 2)
        ss_tot = np.sum((observed_flips - np.mean(observed_flips)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rho, _ = spearmanr(predicted_flips, observed_flips)
    else:
        r2 = 0.0
        rho = 0.0

    return {
        "n_pairs": len(pairs),
        "coverage_conflict_degree": coverage_conflict_degree,
        "reliable_fraction": reliable_fraction,
        "mean_instability": mean_instability,
        "mean_snr": float(np.mean(snrs)),
        "median_snr": float(np.median(snrs)),
        "gaussian_r2": float(r2),
        "gaussian_rho": float(rho),
    }


def load_datasets():
    datasets = []

    bc = load_breast_cancer()
    datasets.append(("Breast Cancer", bc.data, bc.target, 2))

    wine = load_wine()
    datasets.append(("Wine", wine.data, wine.target, 3))

    iris = load_iris()
    datasets.append(("Iris", iris.data, iris.target, 3))

    digits = load_digits()
    datasets.append(("Digits", digits.data, digits.target, 10))

    try:
        X, y = fetch_openml('heart-statlog', version=1, return_X_y=True, as_frame=False, parser='auto')
        y = LabelEncoder().fit_transform(y)
        datasets.append(("Heart Disease", X.astype(float), y, 2))
    except Exception:
        pass

    try:
        X, y = fetch_openml('diabetes', version=1, return_X_y=True, as_frame=False, parser='auto')
        y = LabelEncoder().fit_transform(y)
        datasets.append(("Diabetes", X.astype(float), y, 2))
    except Exception:
        pass

    try:
        X, y = fetch_openml('credit-g', version=1, return_X_y=True, as_frame=True, parser='auto')
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = OrdinalEncoder().fit_transform(X[[col]]).ravel()
        X = X.values.astype(float)
        y = LabelEncoder().fit_transform(y)
        datasets.append(("German Credit", X, y, 2))
    except Exception:
        pass

    return datasets


if __name__ == '__main__':
    print("Explanation Landscape Bridge")
    print("=" * 60)
    print("Chain: Characterization → Coverage Conflict → SNR → Flip Rate\n")

    datasets = load_datasets()
    all_results = {}

    for name, X, y, nc in datasets:
        print(f"\n{name} (P={X.shape[1]}, N={X.shape[0]}, classes={nc})")

        imp_cal = train_models(X, y, SEEDS_CAL, nc)
        imp_val = train_models(X, y, SEEDS_VAL, nc)

        r = compute_landscape(imp_cal, imp_val)
        all_results[name] = r

        print(f"  Coverage conflict degree: {r['coverage_conflict_degree']:.3f}")
        print(f"  Reliable fraction (SNR>2): {r['reliable_fraction']:.3f}")
        print(f"  Mean instability: {r['mean_instability']:.3f}")
        print(f"  Gaussian flip R²: {r['gaussian_r2']:.3f}")

    # Cross-dataset analysis
    print("\n" + "=" * 60)
    print("CROSS-DATASET: Coverage Conflict vs Instability")
    print("=" * 60)

    cc_degrees = [all_results[n]["coverage_conflict_degree"] for n in all_results]
    instabilities = [all_results[n]["mean_instability"] for n in all_results]
    reliable_fracs = [all_results[n]["reliable_fraction"] for n in all_results]

    if len(cc_degrees) > 2:
        rho_cc, p_cc = spearmanr(cc_degrees, instabilities)
        rho_rel, p_rel = spearmanr(reliable_fracs, instabilities)
        r_cc, _ = pearsonr(cc_degrees, instabilities)
        print(f"\n  Coverage conflict degree vs instability:")
        print(f"    Spearman ρ = {rho_cc:.3f} (p = {p_cc:.4f})")
        print(f"    Pearson r  = {r_cc:.3f}")
        print(f"\n  Reliable fraction vs instability:")
        print(f"    Spearman ρ = {rho_rel:.3f} (p = {p_rel:.4f})")

    print("\n  Per-dataset summary:")
    for name, r in all_results.items():
        print(f"    {name:20s}: CC={r['coverage_conflict_degree']:.3f}  "
              f"η≈{r['reliable_fraction']:.3f}  "
              f"instab={r['mean_instability']:.3f}  "
              f"R²={r['gaussian_r2']:.3f}")

    # Save
    output = {
        "experiment": "explanation_landscape_bridge",
        "description": "Coverage conflict degree predicts instability across datasets",
        "chain": "Characterization → Coverage Conflict → SNR → Flip Rate",
        "per_dataset": all_results,
        "cross_dataset": {
            "coverage_conflict_vs_instability_rho": float(rho_cc) if len(cc_degrees) > 2 else None,
            "reliable_fraction_vs_instability_rho": float(rho_rel) if len(cc_degrees) > 2 else None,
        }
    }

    json_path = OUT_DIR / 'results_explanation_landscape_bridge.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
