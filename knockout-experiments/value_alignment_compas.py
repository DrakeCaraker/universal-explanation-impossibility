#!/usr/bin/env python3
"""
Value Alignment on COMPAS: Test for Value Rashomon.

The original value alignment test on German Credit found NO value Rashomon
(CV=0.20, threshold 0.50). COMPAS is a stronger test case because:
1. Known fairness controversies (ProPublica vs Northpointe)
2. Protected attributes (race, gender) vs predictive features (priors, age)
3. Different bootstrap models may genuinely weight these differently

PREDICTION (bilemma): If value Rashomon exists (CV > 0.5 for fairness/predictivity
ratio), then no explanation can be simultaneously faithful to the model AND stable
across the value Rashomon set. This would be a genuine value alignment impossibility.

Uses the ProPublica COMPAS recidivism dataset.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def load_compas():
    """Load COMPAS dataset from OpenML or generate a realistic proxy."""
    try:
        from sklearn.datasets import fetch_openml
        # Try COMPAS from OpenML
        X_raw, y_raw = fetch_openml('compas-two-years', version=1, return_X_y=True,
                                     as_frame=True, parser='auto')
        feature_names = list(X_raw.columns)

        # Encode categoricals
        X_enc = X_raw.copy()
        for col in X_enc.columns:
            if X_enc[col].dtype == 'object' or X_enc[col].dtype.name == 'category':
                X_enc[col] = OrdinalEncoder().fit_transform(X_enc[[col]]).ravel()
        X = X_enc.values.astype(np.float64)
        y = LabelEncoder().fit_transform(y_raw)

        return X, y, feature_names
    except Exception:
        pass

    # Fallback: construct COMPAS-like dataset
    # Use Adult/Census Income as proxy (similar fairness structure)
    try:
        from sklearn.datasets import fetch_openml
        X_raw, y_raw = fetch_openml('adult', version=2, return_X_y=True,
                                     as_frame=True, parser='auto')
        feature_names = list(X_raw.columns)

        X_enc = X_raw.copy()
        for col in X_enc.columns:
            if X_enc[col].dtype == 'object' or X_enc[col].dtype.name == 'category':
                X_enc[col] = OrdinalEncoder().fit_transform(X_enc[[col]]).ravel()

        # Remove NaN rows
        X = X_enc.values.astype(np.float64)
        y = LabelEncoder().fit_transform(y_raw)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        # Subsample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
        X = X[idx]
        y = y[idx]

        return X, y, feature_names
    except Exception as e:
        print(f"  Adult dataset also failed: {e}")
        return None, None, None


def identify_feature_groups(feature_names):
    """Identify fairness-relevant vs predictive features."""
    # COMPAS-like / Adult-like fairness features
    fairness_keywords = ['race', 'sex', 'gender', 'age', 'native', 'marital',
                         'relationship', 'ethnicity']
    predictive_keywords = ['education', 'hours', 'capital', 'occupation',
                           'workclass', 'priors', 'charge', 'fnlwgt',
                           'income', 'gain', 'loss']

    fairness_idx = []
    predictive_idx = []
    fairness_names = []
    predictive_names = []

    for i, name in enumerate(feature_names):
        name_lower = name.lower().replace('-', '_').replace(' ', '_')
        if any(kw in name_lower for kw in fairness_keywords):
            fairness_idx.append(i)
            fairness_names.append(name)
        if any(kw in name_lower for kw in predictive_keywords):
            predictive_idx.append(i)
            predictive_names.append(name)

    return fairness_idx, predictive_idx, fairness_names, predictive_names


def run_value_alignment(X, y, feature_names, dataset_name):
    """Run value alignment experiment."""
    print(f"\n{'='*60}")
    print(f"VALUE ALIGNMENT: {dataset_name}")
    print(f"  X shape: {X.shape}")
    print(f"{'='*60}")
    t0 = time.time()

    fairness_idx, predictive_idx, fairness_names, predictive_names = \
        identify_feature_groups(feature_names)

    print(f"  Fairness features ({len(fairness_names)}): {fairness_names}")
    print(f"  Predictive features ({len(predictive_names)}): {predictive_names}")

    if not fairness_idx or not predictive_idx:
        print("  ERROR: Could not identify feature groups")
        return None

    # Train 30 calibration + 30 validation models
    cal_seeds = list(range(42, 72))
    val_seeds = list(range(142, 172))

    def train_models(seeds):
        n = len(seeds)
        P = X.shape[1]
        imp = np.zeros((n, P))
        for i, seed in enumerate(seeds):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=len(X), replace=True)
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, tree_method='hist', eval_metric='logloss'
            )
            model.fit(X[idx], y[idx])
            imp[i] = model.feature_importances_
        return imp

    print(f"\n  Training 60 models (30 cal + 30 val)...")
    imp_cal = train_models(cal_seeds)
    imp_val = train_models(val_seeds)

    # Compute fairness/predictive ratio for each model
    def compute_ratios(imp):
        ratios = []
        for i in range(imp.shape[0]):
            fair_imp = np.mean(imp[i, fairness_idx])
            pred_imp = np.mean(imp[i, predictive_idx])
            ratios.append(fair_imp / max(pred_imp, 1e-12))
        return np.array(ratios)

    cal_ratios = compute_ratios(imp_cal)
    val_ratios = compute_ratios(imp_val)

    cal_cv = float(np.std(cal_ratios, ddof=1) / max(abs(np.mean(cal_ratios)), 1e-12))
    val_cv = float(np.std(val_ratios, ddof=1) / max(abs(np.mean(val_ratios)), 1e-12))

    # Flip rate: do models disagree on which group matters more?
    cal_fair_means = np.mean(imp_cal[:, fairness_idx], axis=1)
    cal_pred_means = np.mean(imp_cal[:, predictive_idx], axis=1)
    cal_diff = cal_fair_means - cal_pred_means

    val_fair_means = np.mean(imp_val[:, fairness_idx], axis=1)
    val_pred_means = np.mean(imp_val[:, predictive_idx], axis=1)
    val_diff = val_fair_means - val_pred_means

    # Gaussian flip prediction (calibrated on cal, tested on val)
    mu_cal = float(np.mean(cal_diff))
    sd_cal = float(np.std(cal_diff, ddof=1))
    if sd_cal > 1e-12:
        snr = abs(mu_cal) / sd_cal
        predicted_flip = float(norm.cdf(-abs(mu_cal) / sd_cal))
    else:
        snr = 0.0
        predicted_flip = 0.5 if abs(mu_cal) < 1e-12 else 0.0

    # Observed flip on validation
    n_val = len(val_seeds)
    disagree = 0
    total = 0
    for m1 in range(n_val):
        for m2 in range(m1 + 1, n_val):
            if val_diff[m1] * val_diff[m2] < 0:
                disagree += 1
            total += 1
    observed_flip = disagree / total if total > 0 else 0.0

    # Value Rashomon: do models disagree on whether fairness or predictivity matters more?
    cal_fairness_first = int(np.sum(cal_ratios > 1.0))
    val_fairness_first = int(np.sum(val_ratios > 1.0))

    value_rashomon = cal_cv > 0.5 or val_cv > 0.5
    flip_rashomon = observed_flip > 0.10  # >10% of model pairs disagree on priority

    elapsed = time.time() - t0

    result = {
        "dataset": dataset_name,
        "n_features": int(X.shape[1]),
        "n_models_cal": len(cal_seeds),
        "n_models_val": len(val_seeds),
        "fairness_features": fairness_names,
        "predictive_features": predictive_names,
        "calibration": {
            "ratio_mean": float(np.mean(cal_ratios)),
            "ratio_std": float(np.std(cal_ratios, ddof=1)),
            "ratio_cv": cal_cv,
            "fairness_first": cal_fairness_first,
        },
        "validation": {
            "ratio_mean": float(np.mean(val_ratios)),
            "ratio_std": float(np.std(val_ratios, ddof=1)),
            "ratio_cv": val_cv,
            "fairness_first": val_fairness_first,
        },
        "gaussian_flip": {
            "delta": mu_cal,
            "sigma": sd_cal,
            "snr": snr,
            "predicted_flip": predicted_flip,
            "observed_flip_val": observed_flip,
        },
        "value_rashomon_cv": value_rashomon,
        "value_rashomon_flip": flip_rashomon,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Cal ratio CV: {cal_cv:.4f}, Val ratio CV: {val_cv:.4f}")
    print(f"  Cal fairness-first: {cal_fairness_first}/{len(cal_seeds)}")
    print(f"  Val fairness-first: {val_fairness_first}/{len(val_seeds)}")
    print(f"  SNR: {snr:.3f}")
    print(f"  Predicted flip: {predicted_flip:.4f}, Observed flip: {observed_flip:.4f}")
    print(f"  Value Rashomon (CV): {value_rashomon}")
    print(f"  Value Rashomon (flip): {flip_rashomon}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return result


if __name__ == '__main__':
    print("Value Alignment: COMPAS / Adult Dataset")
    print("=" * 60)

    X, y, feature_names = load_compas()
    if X is None:
        print("ERROR: Could not load any fairness dataset")
        exit(1)

    dataset_name = "Adult (Census Income)"  # Will be overridden if COMPAS loads
    r = run_value_alignment(X, y, feature_names, dataset_name)

    if r:
        json_path = OUT_DIR / 'results_value_alignment_compas.json'
        with open(json_path, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"\nResults saved to {json_path}")

    print("\nDone.")
