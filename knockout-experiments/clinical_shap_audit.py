#!/usr/bin/env python3
"""
Clinical SHAP Audit: Apply Gaussian flip formula to published SHAP analyses.

For each clinical/financial dataset commonly used in published SHAP studies:
1. Train 30 calibration + 30 validation XGBoost models (bootstrap)
2. Compute TreeSHAP importance for each model
3. Apply Gaussian flip formula to all feature pairs
4. Identify: which top-feature claims are reliable (SNR > 2) vs unreliable (SNR < 0.5)
5. Report: "Of the C(K,2) implied rankings among the top K features, X% are unreliable"

This directly audits the type of claims made in published clinical ML papers.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

SEEDS_CAL = list(range(42, 72))
SEEDS_VAL = list(range(142, 172))


def train_and_shap(X_train, y_train, X_test, seeds, n_classes=2):
    """Train XGBoost models and compute TreeSHAP importance."""
    n_models = len(seeds)
    P = X_train.shape[1]
    imp_matrix = np.zeros((n_models, P))

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)

        params = dict(n_estimators=100, max_depth=4, random_state=seed,
                      verbosity=0, tree_method='hist')
        if n_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
        else:
            params['eval_metric'] = 'logloss'

        model = xgb.XGBClassifier(**params)
        model.fit(X_train[idx], y_train[idx])

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1] if n_classes == 2 else sv[0]
        imp_matrix[i] = np.mean(np.abs(sv), axis=0)

    return imp_matrix


def audit_top_features(imp_cal, imp_val, feature_names, top_k=5):
    """Audit the top-K feature ranking for reliability."""
    P = imp_cal.shape[1]

    # Determine top-K by mean calibration importance
    mean_imp = np.mean(imp_cal, axis=0)
    top_indices = np.argsort(-mean_imp)[:top_k]
    top_names = [feature_names[i] for i in top_indices]

    # For all pairs among top-K features
    top_pairs = list(combinations(range(top_k), 2))
    pair_results = []

    for rank_j, rank_k in top_pairs:
        j = top_indices[rank_j]
        k = top_indices[rank_k]

        # Calibration: Gaussian prediction
        diff_cal = imp_cal[:, j] - imp_cal[:, k]
        mu = float(np.mean(diff_cal))
        sd = float(np.std(diff_cal, ddof=1))
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred_flip = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

        # Validation: observed flip
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

        reliability = "RELIABLE" if snr > 2.0 else ("UNRELIABLE" if snr < 0.5 else "MARGINAL")

        pair_results.append({
            "feature_1": top_names[rank_j],
            "feature_2": top_names[rank_k],
            "rank_1": rank_j + 1,
            "rank_2": rank_k + 1,
            "snr": round(snr, 3),
            "predicted_flip": round(pred_flip, 4),
            "observed_flip": round(obs_flip, 4),
            "reliability": reliability,
        })

    n_reliable = sum(1 for r in pair_results if r["reliability"] == "RELIABLE")
    n_unreliable = sum(1 for r in pair_results if r["reliability"] == "UNRELIABLE")
    n_marginal = sum(1 for r in pair_results if r["reliability"] == "MARGINAL")
    n_total = len(pair_results)

    return {
        "top_features": top_names,
        "top_indices": [int(i) for i in top_indices],
        "top_importances": [round(float(mean_imp[i]), 6) for i in top_indices],
        "n_pairs": n_total,
        "n_reliable": n_reliable,
        "n_unreliable": n_unreliable,
        "n_marginal": n_marginal,
        "pct_unreliable": round(n_unreliable / n_total * 100, 1) if n_total > 0 else 0,
        "pairs": pair_results,
    }


def full_audit(imp_cal, imp_val):
    """Full audit: all feature pairs."""
    P = imp_cal.shape[1]
    pairs = list(combinations(range(P), 2))

    snrs = []
    predicted = []
    observed = []

    for j, k in pairs:
        diff_cal = imp_cal[:, j] - imp_cal[:, k]
        mu = float(np.mean(diff_cal))
        sd = float(np.std(diff_cal, ddof=1))
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

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
        obs = disagree / total if total > 0 else 0.0

        snrs.append(snr)
        predicted.append(pred)
        observed.append(obs)

    snrs = np.array(snrs)
    predicted = np.array(predicted)
    observed = np.array(observed)

    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    rho, p = spearmanr(predicted, observed) if len(predicted) > 2 else (0, 1)

    return {
        "n_pairs": len(pairs),
        "oos_r2": round(float(r2), 3),
        "spearman_rho": round(float(rho), 3),
        "spearman_p": float(p),
        "pct_unreliable": round(float(np.mean(snrs < 0.5)) * 100, 1),
        "pct_reliable": round(float(np.mean(snrs > 2.0)) * 100, 1),
        "mean_flip_rate": round(float(np.mean(observed)), 3),
    }


def load_datasets():
    """Load clinical/financial datasets commonly used in published SHAP studies."""
    datasets = []

    # 1. Breast Cancer Wisconsin — most common SHAP demo dataset
    bc = load_breast_cancer()
    datasets.append(("Breast Cancer (Wisconsin)", bc.data, bc.target,
                      list(bc.feature_names), 2))

    # 2. Heart Disease (Cleveland/Statlog)
    try:
        X, y = fetch_openml('heart-statlog', version=1, return_X_y=True,
                             as_frame=False, parser='auto')
        y = LabelEncoder().fit_transform(y)
        names = ['age', 'sex', 'chest_pain', 'rest_bp', 'cholesterol',
                 'fasting_bs', 'rest_ecg', 'max_hr', 'exercise_angina',
                 'oldpeak', 'slope', 'n_vessels', 'thal']
        datasets.append(("Heart Disease", X.astype(float), y, names, 2))
    except Exception as e:
        print(f"  Heart Disease failed: {e}")

    # 3. Diabetes (Pima Indians)
    try:
        X, y = fetch_openml('diabetes', version=1, return_X_y=True,
                             as_frame=False, parser='auto')
        y = LabelEncoder().fit_transform(y)
        names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                 'insulin', 'bmi', 'diabetes_pedigree', 'age']
        datasets.append(("Diabetes (Pima)", X.astype(float), y, names, 2))
    except Exception as e:
        print(f"  Diabetes failed: {e}")

    # 4. German Credit
    try:
        X_raw, y_raw = fetch_openml('credit-g', version=1, return_X_y=True,
                                     as_frame=True, parser='auto')
        names = list(X_raw.columns)
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'category':
                X_raw[col] = OrdinalEncoder().fit_transform(X_raw[[col]]).ravel()
        X = X_raw.values.astype(float)
        y = LabelEncoder().fit_transform(y_raw)
        datasets.append(("German Credit", X, y, names, 2))
    except Exception as e:
        print(f"  German Credit failed: {e}")

    # 5. Adult Income (Census)
    try:
        X_raw, y_raw = fetch_openml('adult', version=2, return_X_y=True,
                                     as_frame=True, parser='auto')
        names = list(X_raw.columns)
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'category':
                X_raw[col] = OrdinalEncoder().fit_transform(X_raw[[col]]).ravel()
        X = X_raw.values.astype(float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y_raw = np.array(y_raw)[mask]
        y = LabelEncoder().fit_transform(y_raw)
        # Subsample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
        X = X[idx]
        y = y[idx]
        datasets.append(("Adult Income", X, y, names, 2))
    except Exception as e:
        print(f"  Adult Income failed: {e}")

    return datasets


if __name__ == '__main__':
    print("Clinical SHAP Audit: Gaussian Flip on TreeSHAP")
    print("=" * 60)
    print("Auditing published-style SHAP feature rankings for structural reliability\n")

    datasets = load_datasets()
    all_results = {}

    for name, X, y, feature_names, nc in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {name} (N={X.shape[0]}, P={X.shape[1]})")
        print(f"{'='*60}")
        t0 = time.time()

        # Split for SHAP computation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Limit test set for SHAP speed
        if len(X_test) > 100:
            rng = np.random.RandomState(42)
            test_idx = rng.choice(len(X_test), size=100, replace=False)
            X_test_shap = X_test[test_idx]
        else:
            X_test_shap = X_test

        print(f"  Training 30 calibration models + TreeSHAP...")
        imp_cal = train_and_shap(X_train, y_train, X_test_shap, SEEDS_CAL, nc)

        print(f"  Training 30 validation models + TreeSHAP...")
        imp_val = train_and_shap(X_train, y_train, X_test_shap, SEEDS_VAL, nc)

        # Top-5 audit
        print(f"  Auditing top-5 feature ranking...")
        top5 = audit_top_features(imp_cal, imp_val, feature_names, top_k=5)

        print(f"\n  TOP 5 FEATURES: {top5['top_features']}")
        print(f"  Pairwise rankings: {top5['n_reliable']} reliable, "
              f"{top5['n_unreliable']} unreliable, {top5['n_marginal']} marginal")
        print(f"  % unreliable: {top5['pct_unreliable']}%")

        for pr in top5['pairs']:
            print(f"    {pr['feature_1']:20s} vs {pr['feature_2']:20s}: "
                  f"SNR={pr['snr']:.2f} flip={pr['observed_flip']:.3f} → {pr['reliability']}")

        # Full audit
        print(f"\n  Full audit (all {X.shape[1]} features)...")
        full = full_audit(imp_cal, imp_val)
        print(f"  All pairs: {full['pct_unreliable']}% unreliable, "
              f"{full['pct_reliable']}% reliable")
        print(f"  Gaussian flip OOS R²={full['oos_r2']}, ρ={full['spearman_rho']}")

        elapsed = time.time() - t0
        all_results[name] = {
            "dataset": name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "top_5_audit": top5,
            "full_audit": full,
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"  Elapsed: {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("HEADLINE SUMMARY")
    print(f"{'='*60}")

    total_unreliable_top5 = 0
    total_pairs_top5 = 0
    for name, r in all_results.items():
        t5 = r['top_5_audit']
        total_unreliable_top5 += t5['n_unreliable']
        total_pairs_top5 += t5['n_pairs']
        print(f"\n  {name}:")
        print(f"    Top-5 ranking: {t5['pct_unreliable']}% unreliable ({t5['n_unreliable']}/{t5['n_pairs']})")
        print(f"    All pairs: {r['full_audit']['pct_unreliable']}% unreliable")
        print(f"    TreeSHAP R²={r['full_audit']['oos_r2']}, ρ={r['full_audit']['spearman_rho']}")

    pct = round(total_unreliable_top5 / total_pairs_top5 * 100, 1) if total_pairs_top5 > 0 else 0
    print(f"\n  HEADLINE: Across {len(all_results)} clinical/financial datasets,")
    print(f"  {pct}% of top-5 SHAP feature ranking comparisons are structurally unreliable.")

    # Save
    output = {
        "experiment": "clinical_shap_audit",
        "description": "TreeSHAP-based audit of published-style clinical feature rankings",
        "n_datasets": len(all_results),
        "headline_pct_unreliable_top5": pct,
        "total_unreliable_top5": total_unreliable_top5,
        "total_pairs_top5": total_pairs_top5,
        "per_dataset": all_results,
    }

    json_path = OUT_DIR / 'results_clinical_shap_audit.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
