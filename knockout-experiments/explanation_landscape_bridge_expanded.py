#!/usr/bin/env python3
"""
Explanation Landscape Bridge (Expanded): Coverage Conflict → SNR → Flip Rate

Expanded version with 15+ datasets, SNR threshold sensitivity, and LOO robustness.

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
from sklearn.datasets import (load_breast_cancer, load_wine, load_iris,
                               load_digits, fetch_openml, fetch_california_housing)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEEDS_CAL = list(range(42, 72))    # 30 calibration seeds
SEEDS_VAL = list(range(142, 172))  # 30 validation seeds
SNR_THRESHOLDS = [0.3, 0.5, 1.0, 2.0]


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

    # Coverage conflict degree at multiple thresholds
    cc_by_threshold = {}
    for t in SNR_THRESHOLDS:
        cc_by_threshold[str(t)] = float(np.mean(snrs < t))

    # Default threshold
    coverage_conflict_degree = float(np.mean(snrs < snr_threshold))

    # Mean observed instability
    mean_instability = float(np.mean(observed_flips))

    # Fraction of reliable pairs (SNR > 2)
    reliable_fraction = float(np.mean(snrs > 2.0))

    # Gaussian flip R^2 (OOS)
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
        "coverage_conflict_by_threshold": cc_by_threshold,
        "reliable_fraction": reliable_fraction,
        "mean_instability": mean_instability,
        "mean_snr": float(np.mean(snrs)),
        "median_snr": float(np.median(snrs)),
        "gaussian_r2": float(r2),
        "gaussian_rho": float(rho),
    }


def prepare_openml(name, version=1, n_classes=None, max_samples=5000):
    """Fetch an OpenML dataset, encode categoricals, handle NaN, subsample."""
    try:
        X, y = fetch_openml(name, version=version, return_X_y=True,
                            as_frame=True, parser='auto')
        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = OrdinalEncoder().fit_transform(X[[col]].astype(str)).ravel()
        X = X.values.astype(float)
        # Handle NaN
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        # Encode labels
        y = LabelEncoder().fit_transform(y.astype(str))
        nc = n_classes if n_classes is not None else len(np.unique(y))
        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X, y = X[idx], y[idx]
        return X, y, nc
    except Exception as e:
        print(f"  [SKIP] Failed to load '{name}': {e}")
        return None, None, None


def load_datasets():
    datasets = []

    # --- Original 7 datasets ---
    bc = load_breast_cancer()
    datasets.append(("Breast Cancer", bc.data, bc.target, 2))

    wine = load_wine()
    datasets.append(("Wine", wine.data, wine.target, 3))

    iris = load_iris()
    datasets.append(("Iris", iris.data, iris.target, 3))

    digits = load_digits()
    datasets.append(("Digits", digits.data, digits.target, 10))

    X, y, nc = prepare_openml('heart-statlog', version=1, n_classes=2)
    if X is not None:
        datasets.append(("Heart Disease", X, y, nc))

    X, y, nc = prepare_openml('diabetes', version=1, n_classes=2)
    if X is not None:
        datasets.append(("Diabetes", X, y, nc))

    X, y, nc = prepare_openml('credit-g', version=1, n_classes=2)
    if X is not None:
        datasets.append(("German Credit", X, y, nc))

    # --- New datasets (8 additional) ---

    # California Housing (binarize at median)
    try:
        cal = fetch_california_housing()
        X_cal = cal.data
        y_cal = (cal.target >= np.median(cal.target)).astype(int)
        if len(X_cal) > 5000:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X_cal), size=5000, replace=False)
            X_cal, y_cal = X_cal[idx], y_cal[idx]
        datasets.append(("California Housing", X_cal, y_cal, 2))
    except Exception as e:
        print(f"  [SKIP] California Housing: {e}")

    # Adult/Census Income
    X, y, nc = prepare_openml('adult', version=2, n_classes=2, max_samples=5000)
    if X is not None:
        datasets.append(("Adult Income", X, y, nc))

    # Ionosphere
    X, y, nc = prepare_openml('ionosphere', version=1, n_classes=2)
    if X is not None:
        datasets.append(("Ionosphere", X, y, nc))

    # Sonar
    X, y, nc = prepare_openml('sonar', version=1, n_classes=2)
    if X is not None:
        datasets.append(("Sonar", X, y, nc))

    # Vehicle
    X, y, nc = prepare_openml('vehicle', version=1)
    if X is not None:
        datasets.append(("Vehicle", X, y, nc))

    # Segment
    X, y, nc = prepare_openml('segment', version=1)
    if X is not None:
        datasets.append(("Segment", X, y, nc))

    # Satimage
    X, y, nc = prepare_openml('satimage', version=1)
    if X is not None:
        datasets.append(("Satimage", X, y, nc))

    # Vowel
    X, y, nc = prepare_openml('vowel', version=2)
    if X is not None:
        datasets.append(("Vowel", X, y, nc))

    return datasets


if __name__ == '__main__':
    print("Explanation Landscape Bridge (Expanded)")
    print("=" * 70)
    print("Chain: Characterization -> Coverage Conflict -> SNR -> Flip Rate")
    print(f"Calibration seeds: {len(SEEDS_CAL)}, Validation seeds: {len(SEEDS_VAL)}")
    print(f"SNR thresholds for sensitivity: {SNR_THRESHOLDS}\n")

    datasets = load_datasets()
    print(f"\nLoaded {len(datasets)} datasets.\n")

    all_results = {}

    for name, X, y, nc in datasets:
        print(f"\n{name} (P={X.shape[1]}, N={X.shape[0]}, classes={nc})")

        imp_cal = train_models(X, y, SEEDS_CAL, nc)
        imp_val = train_models(X, y, SEEDS_VAL, nc)

        r = compute_landscape(imp_cal, imp_val)
        all_results[name] = r

        print(f"  Coverage conflict degree (t=0.5): {r['coverage_conflict_degree']:.3f}")
        print(f"  Reliable fraction (SNR>2):        {r['reliable_fraction']:.3f}")
        print(f"  Mean instability:                 {r['mean_instability']:.3f}")
        print(f"  Gaussian flip R^2:                {r['gaussian_r2']:.3f}")
        cc_str = ", ".join(f"t={t}: {r['coverage_conflict_by_threshold'][str(t)]:.3f}"
                           for t in SNR_THRESHOLDS)
        print(f"  CC by threshold: {cc_str}")

    # ==========================================
    # Cross-dataset analysis
    # ==========================================
    print("\n" + "=" * 70)
    print("CROSS-DATASET: Coverage Conflict vs Instability")
    print("=" * 70)

    dataset_names = list(all_results.keys())
    cc_degrees = [all_results[n]["coverage_conflict_degree"] for n in dataset_names]
    instabilities = [all_results[n]["mean_instability"] for n in dataset_names]
    reliable_fracs = [all_results[n]["reliable_fraction"] for n in dataset_names]

    cross_dataset = {}

    if len(cc_degrees) > 2:
        rho_cc, p_cc = spearmanr(cc_degrees, instabilities)
        rho_rel, p_rel = spearmanr(reliable_fracs, instabilities)
        r_cc, _ = pearsonr(cc_degrees, instabilities)
        print(f"\n  Coverage conflict degree vs instability (default t=0.5):")
        print(f"    Spearman rho = {rho_cc:.3f} (p = {p_cc:.4f})")
        print(f"    Pearson r    = {r_cc:.3f}")
        print(f"\n  Reliable fraction vs instability:")
        print(f"    Spearman rho = {rho_rel:.3f} (p = {p_rel:.4f})")

        cross_dataset["coverage_conflict_vs_instability_rho"] = float(rho_cc)
        cross_dataset["coverage_conflict_vs_instability_p"] = float(p_cc)
        cross_dataset["reliable_fraction_vs_instability_rho"] = float(rho_rel)
        cross_dataset["reliable_fraction_vs_instability_p"] = float(p_rel)

    # ==========================================
    # SNR Threshold Sensitivity
    # ==========================================
    print("\n" + "=" * 70)
    print("SNR THRESHOLD SENSITIVITY")
    print("=" * 70)

    threshold_sensitivity = {}
    for t in SNR_THRESHOLDS:
        cc_at_t = [all_results[n]["coverage_conflict_by_threshold"][str(t)]
                   for n in dataset_names]
        if len(cc_at_t) > 2 and np.std(cc_at_t) > 1e-12 and np.std(instabilities) > 1e-12:
            rho_t, p_t = spearmanr(cc_at_t, instabilities)
        else:
            rho_t, p_t = 0.0, 1.0
        threshold_sensitivity[str(t)] = {
            "spearman_rho": float(rho_t),
            "p_value": float(p_t),
        }
        print(f"  Threshold {t}: Spearman rho = {rho_t:.3f} (p = {p_t:.4f})")

    # ==========================================
    # LOO Robustness
    # ==========================================
    print("\n" + "=" * 70)
    print("LOO ROBUSTNESS (drop each dataset, recompute rho)")
    print("=" * 70)

    loo_results = {}
    loo_rhos = []

    if len(dataset_names) > 3:
        for drop_name in dataset_names:
            remaining = [n for n in dataset_names if n != drop_name]
            cc_loo = [all_results[n]["coverage_conflict_degree"] for n in remaining]
            inst_loo = [all_results[n]["mean_instability"] for n in remaining]
            if np.std(cc_loo) > 1e-12 and np.std(inst_loo) > 1e-12:
                rho_loo, p_loo = spearmanr(cc_loo, inst_loo)
            else:
                rho_loo, p_loo = 0.0, 1.0
            loo_results[drop_name] = {
                "rho_without": float(rho_loo),
                "p_without": float(p_loo),
            }
            loo_rhos.append(rho_loo)
            print(f"  Drop {drop_name:20s}: rho = {rho_loo:.3f} (p = {p_loo:.4f})")

        min_rho = float(np.min(loo_rhos))
        max_rho = float(np.max(loo_rhos))
        mean_rho = float(np.mean(loo_rhos))
        print(f"\n  LOO rho range: [{min_rho:.3f}, {max_rho:.3f}], mean = {mean_rho:.3f}")
    else:
        min_rho = max_rho = mean_rho = None
        print("  Not enough datasets for LOO analysis.")

    # Per-dataset summary table
    print("\n" + "=" * 70)
    print("PER-DATASET SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':20s}  {'CC(0.5)':>7s}  {'eta':>7s}  {'instab':>7s}  {'R^2':>7s}  {'rho':>7s}")
    print("  " + "-" * 62)
    for name, r in all_results.items():
        print(f"  {name:20s}  {r['coverage_conflict_degree']:7.3f}  "
              f"{r['reliable_fraction']:7.3f}  "
              f"{r['mean_instability']:7.3f}  "
              f"{r['gaussian_r2']:7.3f}  "
              f"{r['gaussian_rho']:7.3f}")

    # ==========================================
    # Save results
    # ==========================================
    output = {
        "experiment": "explanation_landscape_bridge_expanded",
        "description": "Coverage conflict degree predicts instability across 15+ datasets",
        "chain": "Characterization -> Coverage Conflict -> SNR -> Flip Rate",
        "n_datasets": len(dataset_names),
        "n_calibration_seeds": len(SEEDS_CAL),
        "n_validation_seeds": len(SEEDS_VAL),
        "snr_thresholds": SNR_THRESHOLDS,
        "per_dataset": all_results,
        "cross_dataset": cross_dataset,
        "snr_threshold_sensitivity": threshold_sensitivity,
        "loo_robustness": {
            "per_drop": loo_results,
            "min_rho": min_rho,
            "max_rho": max_rho,
            "mean_rho": mean_rho,
        },
    }

    json_path = OUT_DIR / 'results_explanation_landscape_bridge_expanded.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
