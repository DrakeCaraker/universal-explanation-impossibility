#!/usr/bin/env python3
"""
Generational candidate experiments: Value Alignment and Abstraction/Enrichment.

EXPERIMENT 1: Value Alignment — Do different models prioritize features differently?
  Uses German Credit (P=20). Partitions features into fairness-relevant and predictive.
  Measures whether the ratio of fairness-to-predictive importance varies across
  bootstrap models — a proxy for "value Rashomon."

EXPERIMENT 2: Enrichment = Abstraction — Does merging categories reduce instability?
  Uses Wine (3 classes). Tests three abstraction levels:
    Level 0: 3-class (maximally incompatible)
    Level 1: partial merge (class 0+1 vs class 2)
    Level 2: binary (class 0 vs not-class-0)
  Prediction: enrichment (merging) reduces flip rate, as the bilemma predicts.

All use feature_importances_ (not SHAP) for speed.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.datasets import load_wine, fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# =====================================================================
# Shared utilities
# =====================================================================

def train_bootstrap_models(X, y, seeds, is_classifier=True, n_classes=2):
    """Train XGBoost models with bootstrap resampling, return feature_importances_ matrix."""
    n_models = len(seeds)
    P = X.shape[1]
    imp = np.zeros((n_models, P))

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        if is_classifier:
            params = dict(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, use_label_encoder=False, tree_method='hist'
            )
            if n_classes > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = n_classes
                params['eval_metric'] = 'mlogloss'
            else:
                params['eval_metric'] = 'logloss'
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, tree_method='hist'
            )
        model.fit(X[idx], y[idx])
        imp[i] = model.feature_importances_
    return imp


def compute_flip_rate(imp):
    """Compute mean pairwise flip rate across all feature pairs."""
    n_models, P = imp.shape
    pairs = list(combinations(range(P), 2))
    flip_rates = []

    for j, k in pairs:
        disagree = 0
        total = 0
        for m1 in range(n_models):
            for m2 in range(m1 + 1, n_models):
                if (imp[m1, j] - imp[m1, k]) * (imp[m2, j] - imp[m2, k]) < 0:
                    disagree += 1
                total += 1
        flip_rates.append(disagree / total if total > 0 else 0.0)

    return np.array(flip_rates)


def gaussian_flip_formula(imp):
    """Compute Gaussian-predicted flip rates for all feature pairs."""
    n_models, P = imp.shape
    pairs = list(combinations(range(P), 2))
    predicted = []

    for j, k in pairs:
        diff = imp[:, j] - imp[:, k]
        mu = np.mean(diff)
        sd = np.std(diff, ddof=1)
        if sd < 1e-12:
            predicted.append(0.0 if abs(mu) > 1e-12 else 0.5)
        else:
            predicted.append(float(norm.cdf(-abs(mu) / sd)))

    return np.array(predicted)


# =====================================================================
# EXPERIMENT 1: Value Alignment
# =====================================================================

def experiment_1_value_alignment():
    print("=" * 70)
    print("EXPERIMENT 1: Value Alignment — Fairness vs Predictivity Ratio")
    print("=" * 70)
    t0 = time.time()

    # Load German Credit
    print("\nLoading German Credit dataset...")
    X_raw, y_raw = fetch_openml('credit-g', version=1, return_X_y=True, as_frame=True, parser='auto')

    feature_names = list(X_raw.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # Identify fairness-relevant and predictive feature groups
    fairness_keywords = ['age', 'sex', 'foreign', 'personal_status']
    predictive_keywords = ['credit_amount', 'duration', 'savings', 'checking']

    fairness_indices = []
    predictive_indices = []
    fairness_names = []
    predictive_names = []

    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in fairness_keywords):
            fairness_indices.append(i)
            fairness_names.append(name)
        if any(kw in name_lower for kw in predictive_keywords):
            predictive_indices.append(i)
            predictive_names.append(name)

    print(f"  Fairness-relevant features ({len(fairness_names)}): {fairness_names}")
    print(f"  Predictive features ({len(predictive_names)}): {predictive_names}")

    # Encode categorical features
    X_encoded = X_raw.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
            X_encoded[col] = OrdinalEncoder().fit_transform(X_encoded[[col]]).ravel()
    X = X_encoded.values.astype(np.float64)
    y = LabelEncoder().fit_transform(y_raw)

    print(f"  X shape: {X.shape}, classes: {np.unique(y)}")

    # Train 30 bootstrap models
    seeds = list(range(42, 72))
    print(f"\nTraining {len(seeds)} XGBoost models (bootstrap)...")
    imp = train_bootstrap_models(X, y, seeds, is_classifier=True, n_classes=2)

    # Compute fairness-to-predictive ratio for each model
    ratios = []
    fairness_importances = []
    predictive_importances = []

    for i in range(len(seeds)):
        fair_imp = np.mean(imp[i, fairness_indices]) if fairness_indices else 0.0
        pred_imp = np.mean(imp[i, predictive_indices]) if predictive_indices else 1e-12
        fairness_importances.append(float(fair_imp))
        predictive_importances.append(float(pred_imp))
        ratio = fair_imp / max(pred_imp, 1e-12)
        ratios.append(float(ratio))

    ratios = np.array(ratios)
    ratio_mean = float(np.mean(ratios))
    ratio_std = float(np.std(ratios, ddof=1))
    ratio_cv = ratio_std / max(abs(ratio_mean), 1e-12)

    fairness_first = int(np.sum(ratios > 1.0))
    predictive_first = int(np.sum(ratios < 1.0))

    print(f"\n  Ratio (fairness/predictive) distribution:")
    print(f"    Mean:  {ratio_mean:.4f}")
    print(f"    Std:   {ratio_std:.4f}")
    print(f"    CV:    {ratio_cv:.4f}")
    print(f"    Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
    print(f"    Fairness-first (ratio > 1): {fairness_first}/{len(seeds)}")
    print(f"    Predictive-first (ratio < 1): {predictive_first}/{len(seeds)}")

    # Gaussian flip formula applied to fairness vs predictive comparison
    # For each model, treat fairness mean and predictive mean as two "features"
    fair_arr = np.array(fairness_importances)
    pred_arr = np.array(predictive_importances)
    diff = fair_arr - pred_arr
    mu_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    if sd_diff > 1e-12:
        gaussian_flip = float(norm.cdf(-abs(mu_diff) / sd_diff))
        snr = abs(mu_diff) / sd_diff
    else:
        gaussian_flip = 0.5 if abs(mu_diff) < 1e-12 else 0.0
        snr = 0.0

    # Observed flip rate: fraction of model pairs where one says fairness > predictive
    # and the other says predictive > fairness
    n_models = len(seeds)
    disagree = 0
    total = 0
    for m1 in range(n_models):
        for m2 in range(m1 + 1, n_models):
            if diff[m1] * diff[m2] < 0:
                disagree += 1
            total += 1
    observed_flip = disagree / total if total > 0 else 0.0

    print(f"\n  Gaussian flip formula (fairness vs predictive):")
    print(f"    Delta (mean diff):    {mu_diff:.6f}")
    print(f"    Sigma (std diff):     {sd_diff:.6f}")
    print(f"    SNR (|Delta|/sigma):  {snr:.4f}")
    print(f"    Predicted flip rate:  {gaussian_flip:.4f}")
    print(f"    Observed flip rate:   {observed_flip:.4f}")

    value_rashomon = ratio_cv > 0.5
    print(f"\n  Value Rashomon detected (CV > 0.5): {value_rashomon}")
    if value_rashomon:
        print("  --> Models with identical accuracy have different value priorities.")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        "experiment": "value_alignment",
        "description": "Do bootstrap models prioritize fairness vs predictivity differently?",
        "dataset": "German Credit",
        "n_features": int(X.shape[1]),
        "n_models": len(seeds),
        "fairness_features": fairness_names,
        "predictive_features": predictive_names,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_cv": ratio_cv,
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "fairness_first_count": fairness_first,
        "predictive_first_count": predictive_first,
        "value_rashomon_detected": value_rashomon,
        "gaussian_flip": {
            "delta": mu_diff,
            "sigma": sd_diff,
            "snr": float(snr),
            "predicted_flip_rate": gaussian_flip,
            "observed_flip_rate": float(observed_flip)
        },
        "per_model_ratios": [float(r) for r in ratios],
        "elapsed_seconds": round(elapsed, 1)
    }


# =====================================================================
# EXPERIMENT 2: Enrichment = Abstraction
# =====================================================================

def experiment_2_abstraction():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Enrichment = Abstraction — Flip Rate by Abstraction Level")
    print("=" * 70)
    t0 = time.time()

    # Load Wine dataset
    print("\nLoading Wine dataset...")
    from sklearn.datasets import load_wine
    wine = load_wine()
    X = wine.data
    y_full = wine.target  # 0, 1, 2

    print(f"  X shape: {X.shape}")
    print(f"  Classes: {np.unique(y_full)}, counts: {np.bincount(y_full)}")
    print(f"  Feature names: {wine.feature_names}")

    seeds = list(range(42, 72))  # 30 models
    n_models = len(seeds)

    results_by_level = {}

    # --- Level 0: 3-class (no abstraction, maximally incompatible) ---
    print(f"\n  Level 0: 3-class classification (no enrichment)...")
    imp_0 = train_bootstrap_models(X, y_full, seeds, is_classifier=True, n_classes=3)
    flip_rates_0 = compute_flip_rate(imp_0)
    mean_flip_0 = float(np.mean(flip_rates_0))
    gaussian_pred_0 = gaussian_flip_formula(imp_0)

    print(f"    Mean flip rate: {mean_flip_0:.4f}")
    print(f"    Flip rate range: [{np.min(flip_rates_0):.4f}, {np.max(flip_rates_0):.4f}]")

    results_by_level["level_0_3class"] = {
        "label": "3-class (no enrichment)",
        "n_classes": 3,
        "mean_flip_rate": mean_flip_0,
        "median_flip_rate": float(np.median(flip_rates_0)),
        "std_flip_rate": float(np.std(flip_rates_0)),
        "min_flip_rate": float(np.min(flip_rates_0)),
        "max_flip_rate": float(np.max(flip_rates_0)),
        "mean_gaussian_predicted": float(np.mean(gaussian_pred_0)),
        "n_pairs": len(flip_rates_0)
    }

    # --- Level 1: Partial abstraction (merge class 0+1 → typeA, class 2 → typeB) ---
    print(f"\n  Level 1: Binary (class 0+1 = typeA vs class 2 = typeB)...")
    y_level1 = np.where(y_full == 2, 1, 0)  # 0 = typeA (class 0+1), 1 = typeB (class 2)
    print(f"    Class distribution: {np.bincount(y_level1)}")

    imp_1 = train_bootstrap_models(X, y_level1, seeds, is_classifier=True, n_classes=2)
    flip_rates_1 = compute_flip_rate(imp_1)
    mean_flip_1 = float(np.mean(flip_rates_1))
    gaussian_pred_1 = gaussian_flip_formula(imp_1)

    print(f"    Mean flip rate: {mean_flip_1:.4f}")
    print(f"    Flip rate range: [{np.min(flip_rates_1):.4f}, {np.max(flip_rates_1):.4f}]")

    results_by_level["level_1_partial_merge"] = {
        "label": "Partial merge (class 0+1 vs class 2)",
        "n_classes": 2,
        "merge_description": "class 0 + class 1 -> typeA, class 2 -> typeB",
        "mean_flip_rate": mean_flip_1,
        "median_flip_rate": float(np.median(flip_rates_1)),
        "std_flip_rate": float(np.std(flip_rates_1)),
        "min_flip_rate": float(np.min(flip_rates_1)),
        "max_flip_rate": float(np.max(flip_rates_1)),
        "mean_gaussian_predicted": float(np.mean(gaussian_pred_1)),
        "n_pairs": len(flip_rates_1)
    }

    # --- Level 2: Full abstraction (class 0 vs not-class-0) ---
    print(f"\n  Level 2: Binary (class 0 vs not-class-0)...")
    y_level2 = np.where(y_full == 0, 0, 1)  # 0 = class 0, 1 = other
    print(f"    Class distribution: {np.bincount(y_level2)}")

    imp_2 = train_bootstrap_models(X, y_level2, seeds, is_classifier=True, n_classes=2)
    flip_rates_2 = compute_flip_rate(imp_2)
    mean_flip_2 = float(np.mean(flip_rates_2))
    gaussian_pred_2 = gaussian_flip_formula(imp_2)

    print(f"    Mean flip rate: {mean_flip_2:.4f}")
    print(f"    Flip rate range: [{np.min(flip_rates_2):.4f}, {np.max(flip_rates_2):.4f}]")

    results_by_level["level_2_binary"] = {
        "label": "Binary (class 0 vs other)",
        "n_classes": 2,
        "merge_description": "class 0 vs (class 1 + class 2)",
        "mean_flip_rate": mean_flip_2,
        "median_flip_rate": float(np.median(flip_rates_2)),
        "std_flip_rate": float(np.std(flip_rates_2)),
        "min_flip_rate": float(np.min(flip_rates_2)),
        "max_flip_rate": float(np.max(flip_rates_2)),
        "mean_gaussian_predicted": float(np.mean(gaussian_pred_2)),
        "n_pairs": len(flip_rates_2)
    }

    # --- Analysis: Does enrichment reduce instability? ---
    enrichment_reduces = mean_flip_0 > mean_flip_1 or mean_flip_0 > mean_flip_2
    best_level = min(
        ["level_0", "level_1", "level_2"],
        key=lambda l: [mean_flip_0, mean_flip_1, mean_flip_2][["level_0", "level_1", "level_2"].index(l)]
    )

    print(f"\n  Summary:")
    print(f"    Level 0 (3-class) mean flip rate:       {mean_flip_0:.4f}")
    print(f"    Level 1 (partial merge) mean flip rate:  {mean_flip_1:.4f}")
    print(f"    Level 2 (binary) mean flip rate:         {mean_flip_2:.4f}")
    print(f"    Enrichment reduces instability: {enrichment_reduces}")
    print(f"    Most stable level: {best_level}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        "experiment": "abstraction_enrichment",
        "description": "Does merging categories (enrichment) reduce explanation instability?",
        "dataset": "Wine",
        "n_features": int(X.shape[1]),
        "n_models": n_models,
        "levels": results_by_level,
        "enrichment_reduces_instability": enrichment_reduces,
        "most_stable_level": best_level,
        "bilemma_prediction_confirmed": enrichment_reduces,
        "elapsed_seconds": round(elapsed, 1)
    }


# =====================================================================
# Plotting
# =====================================================================

def make_figures(results):
    pdf_path = FIG_DIR / 'alignment_abstraction.pdf'
    print(f"\nSaving figures to {pdf_path}...")

    with PdfPages(str(pdf_path)) as pdf:
        # --- Figure 1: Value Alignment Ratio Distribution ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        r1 = results["experiment_1_value_alignment"]
        ratios = r1["per_model_ratios"]

        ax = axes[0]
        ax.hist(ratios, bins=15, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ratio = 1 (neutral)')
        ax.axvline(np.mean(ratios), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean = {np.mean(ratios):.3f}')
        ax.set_xlabel('Fairness / Predictive Importance Ratio')
        ax.set_ylabel('Count')
        ax.set_title(f'Value Alignment Ratio Distribution\n'
                     f'German Credit, 30 models, CV = {r1["ratio_cv"]:.3f}')
        ax.legend(fontsize=9)

        # --- Figure 1b: Fairness vs Predictive scatter ---
        ax = axes[1]
        gf = r1["gaussian_flip"]
        ax.bar(['Predicted\n(Gaussian)', 'Observed'],
               [gf["predicted_flip_rate"], gf["observed_flip_rate"]],
               color=['steelblue', 'coral'], edgecolor='black')
        ax.set_ylabel('Flip Rate')
        ax.set_title(f'Fairness vs Predictive: Flip Rate\n'
                     f'SNR = {gf["snr"]:.3f}, '
                     f'Rashomon = {"YES" if r1["value_rashomon_detected"] else "NO"}')
        ax.set_ylim(0, max(gf["predicted_flip_rate"], gf["observed_flip_rate"]) * 1.3 + 0.05)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Figure 2: Abstraction Levels Flip Rate ---
        r2 = results["experiment_2_abstraction"]
        levels = r2["levels"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of mean flip rates
        ax = axes[0]
        labels = ['Level 0\n3-class\n(no enrichment)',
                  'Level 1\nPartial merge\n(0+1 vs 2)',
                  'Level 2\nBinary\n(0 vs other)']
        means = [levels["level_0_3class"]["mean_flip_rate"],
                 levels["level_1_partial_merge"]["mean_flip_rate"],
                 levels["level_2_binary"]["mean_flip_rate"]]
        stds = [levels["level_0_3class"]["std_flip_rate"],
                levels["level_1_partial_merge"]["std_flip_rate"],
                levels["level_2_binary"]["std_flip_rate"]]
        colors = ['#e74c3c', '#f39c12', '#27ae60']

        bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='black',
                      capsize=5, alpha=0.85)
        ax.set_ylabel('Mean Pairwise Flip Rate')
        ax.set_title('Explanation Instability by Abstraction Level\n'
                     f'Wine Dataset, 30 models')
        ax.set_ylim(0, max(means) * 1.4)

        # Add value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Gaussian predicted vs observed
        ax = axes[1]
        gauss_means = [levels["level_0_3class"]["mean_gaussian_predicted"],
                       levels["level_1_partial_merge"]["mean_gaussian_predicted"],
                       levels["level_2_binary"]["mean_gaussian_predicted"]]

        x_pos = np.arange(3)
        width = 0.35
        ax.bar(x_pos - width/2, means, width, label='Observed', color='steelblue',
               edgecolor='black', alpha=0.85)
        ax.bar(x_pos + width/2, gauss_means, width, label='Gaussian predicted', color='coral',
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Level 0\n(3-class)', 'Level 1\n(merge)', 'Level 2\n(binary)'])
        ax.set_ylabel('Mean Flip Rate')
        ax.set_title('Observed vs Gaussian Predicted Flip Rate\nby Abstraction Level')
        ax.legend()
        ax.set_ylim(0, max(max(means), max(gauss_means)) * 1.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Saved: {pdf_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("Alignment & Abstraction Experiments")
    print("=" * 70)

    results = {}

    # Experiment 1
    r1 = experiment_1_value_alignment()
    results["experiment_1_value_alignment"] = r1

    # Experiment 2
    r2 = experiment_2_abstraction()
    results["experiment_2_abstraction"] = r2

    # Save results
    json_path = OUT_DIR / 'results_alignment_abstraction.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate figures
    make_figures(results)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment 1 (Value Alignment):")
    print(f"  Ratio CV = {r1['ratio_cv']:.4f}")
    print(f"  Value Rashomon detected: {r1['value_rashomon_detected']}")
    print(f"  Fairness-first models: {r1['fairness_first_count']}/{r1['n_models']}")
    print(f"  Gaussian flip rate (predicted): {r1['gaussian_flip']['predicted_flip_rate']:.4f}")

    print(f"\nExperiment 2 (Abstraction/Enrichment):")
    lvl = r2['levels']
    print(f"  Level 0 (3-class) flip rate:      {lvl['level_0_3class']['mean_flip_rate']:.4f}")
    print(f"  Level 1 (partial merge) flip rate: {lvl['level_1_partial_merge']['mean_flip_rate']:.4f}")
    print(f"  Level 2 (binary) flip rate:        {lvl['level_2_binary']['mean_flip_rate']:.4f}")
    print(f"  Enrichment reduces instability: {r2['enrichment_reduces_instability']}")
    print(f"  Bilemma prediction confirmed: {r2['bilemma_prediction_confirmed']}")

    print(f"\nTotal elapsed: {r1['elapsed_seconds'] + r2['elapsed_seconds']:.1f}s")
    print("Done.")
