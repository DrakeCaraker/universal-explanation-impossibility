#!/usr/bin/env python3
"""
Two highest-priority remaining experiments:

EXPERIMENT 1 (#29): Gene Expression SHAP Reliability
  - AP_Breast_Lung dataset (470 samples, 10935 features) from OpenML
  - Top 50 genes by variance, 30+30 bootstrap XGBoost models
  - Gain-based importance (fast), Gaussian flip formula calibration
  - Headline: fraction of gene importance comparisons unreliable

EXPERIMENT 2 (#27 extended): High-Dimensional with TreeSHAP at P=50
  - make_classification(n_features=50, n_informative=20, n_redundant=15)
  - 15+15 models, actual TreeSHAP importance
  - Compare SHAP-based R² vs gain-based R²
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, pearsonr
from sklearn.datasets import fetch_openml, make_classification
from sklearn.preprocessing import LabelEncoder
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

def train_bootstrap_models(X, y, seeds, is_classifier=True):
    """Train XGBoost models with bootstrap resampling, return models and gain importance matrix."""
    n_models = len(seeds)
    P = X.shape[1]
    imp = np.zeros((n_models, P))
    models = []

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        if is_classifier:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, use_label_encoder=False, eval_metric='logloss',
                tree_method='hist'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, tree_method='hist'
            )
        model.fit(X[idx], y[idx])
        imp[i] = model.feature_importances_
        models.append(model)
    return models, imp


def compute_shap_importance(models, X_sample):
    """Compute mean |SHAP| importance for each model on X_sample."""
    n_models = len(models)
    P = X_sample.shape[1]
    imp = np.zeros((n_models, P))

    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            # Multi-class: average across classes
            imp[i] = np.mean(np.stack([np.mean(np.abs(s), axis=0) for s in sv]), axis=0)
        elif sv.ndim == 3:
            imp[i] = np.mean(np.abs(sv), axis=(0, 2))
        else:
            imp[i] = np.mean(np.abs(sv), axis=0)
    return imp


def gaussian_flip_predictions(cal_imp, val_imp):
    """
    Compute Gaussian flip formula predictions on calibration set,
    measure observed flip rates on validation set.
    """
    P = cal_imp.shape[1]
    pairs = list(combinations(range(P), 2))
    n_pairs = len(pairs)

    predicted = np.zeros(n_pairs)
    observed = np.zeros(n_pairs)
    snr = np.zeros(n_pairs)

    for idx, (j, k) in enumerate(pairs):
        # Calibration: estimate Delta and sigma
        diff_cal = cal_imp[:, j] - cal_imp[:, k]
        mu = np.mean(diff_cal)
        sd = np.std(diff_cal, ddof=1)

        if sd < 1e-12:
            predicted[idx] = 0.0 if abs(mu) > 1e-12 else 0.5
            snr[idx] = np.inf if abs(mu) > 1e-12 else 0.0
        else:
            predicted[idx] = norm.cdf(-abs(mu) / sd)
            snr[idx] = abs(mu) / sd

        # Validation: observed flip rate
        diff_val = val_imp[:, j] - val_imp[:, k]
        n_val = len(diff_val)
        flips = 0
        total = 0
        for m1 in range(n_val):
            for m2 in range(m1 + 1, n_val):
                if (diff_val[m1] > 0) != (diff_val[m2] > 0):
                    flips += 1
                total += 1
        observed[idx] = flips / total if total > 0 else 0.0

    return predicted, observed, snr, pairs


def compute_oos_r2(predicted, observed):
    """Out-of-sample R^2."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


# =====================================================================
# EXPERIMENT 1: Gene Expression SHAP Reliability (#29)
# =====================================================================

def experiment_gene_expression():
    print("=" * 70)
    print("EXPERIMENT 1: Gene Expression SHAP Reliability (#29)")
    print("=" * 70)
    t0 = time.time()

    # Fetch AP_Breast_Lung dataset
    dataset_name = 'AP_Breast_Lung'
    print(f"\nFetching {dataset_name} from OpenML...")
    try:
        data = fetch_openml(name='AP_Breast_Lung', version=1, as_frame=False, parser='auto')
        X_raw, y_raw = data.data, data.target
        print(f"  Loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    except Exception as e:
        print(f"  AP_Breast_Lung failed ({e}), falling back to micro-mass...")
        dataset_name = 'micro-mass'
        data = fetch_openml(name='micro-mass', version=1, as_frame=False, parser='auto')
        X_raw, y_raw = data.data, data.target
        print(f"  Loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes: {len(le.classes_)} ({le.classes_})")

    # Handle NaN
    if np.any(np.isnan(X_raw)):
        print("  Replacing NaN with column medians...")
        col_medians = np.nanmedian(X_raw, axis=0)
        nan_mask = np.isnan(X_raw)
        X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Select top 50 genes by variance
    variances = np.var(X_raw, axis=0)
    top50_idx = np.argsort(variances)[-50:]
    X = X_raw[:, top50_idx]
    P = X.shape[1]
    print(f"\n  Selected top {P} genes by variance")
    print(f"  Variance range: [{variances[top50_idx].min():.2f}, {variances[top50_idx].max():.2f}]")

    # Train calibration and validation models
    cal_seeds = list(range(42, 72))      # 30 models
    val_seeds = list(range(142, 172))    # 30 models

    print(f"\nTraining {len(cal_seeds)} calibration models (seeds 42-71)...")
    _, cal_imp = train_bootstrap_models(X, y, cal_seeds, is_classifier=True)

    print(f"Training {len(val_seeds)} validation models (seeds 142-171)...")
    _, val_imp = train_bootstrap_models(X, y, val_seeds, is_classifier=True)

    # Compute Gaussian flip predictions
    n_pairs = P * (P - 1) // 2
    print(f"\nComputing Gaussian flip formula for {n_pairs} feature pairs...")
    predicted, observed, snr, pairs = gaussian_flip_predictions(cal_imp, val_imp)

    r2 = compute_oos_r2(predicted, observed)
    corr, pval = pearsonr(predicted, observed)

    # Categorize by SNR
    finite_snr = snr[np.isfinite(snr)]
    unreliable = np.sum(snr < 0.5)
    moderate = np.sum((snr >= 0.5) & (snr <= 2.0))
    reliable = np.sum(snr > 2.0)

    elapsed = time.time() - t0

    print(f"\n--- RESULTS ---")
    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {X.shape[0]}, Features (top genes): {P}")
    print(f"  Total pairs tested: {n_pairs}")
    print(f"  OOS R^2:  {r2:.4f}")
    print(f"  Pearson r: {corr:.4f} (p = {pval:.2e})")
    print(f"  SNR < 0.5 (unreliable): {unreliable} ({100*unreliable/n_pairs:.1f}%)")
    print(f"  SNR 0.5-2 (moderate):   {moderate} ({100*moderate/n_pairs:.1f}%)")
    print(f"  SNR > 2 (reliable):     {reliable} ({100*reliable/n_pairs:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    headline = (
        f"In {dataset_name} cancer gene expression (P={P} top-variance genes), "
        f"{100*unreliable/n_pairs:.0f}% of gene importance comparisons are unreliable (SNR < 0.5)."
    )
    print(f"\n  HEADLINE: {headline}")

    result = {
        'experiment': 'gene_expression_reliability',
        'description': f'Gene expression SHAP reliability ({dataset_name}, top {P} genes)',
        'dataset': dataset_name,
        'n_samples': int(X.shape[0]),
        'n_features_original': int(X_raw.shape[1]),
        'n_features_selected': P,
        'selection_method': 'top_50_by_variance',
        'importance_method': 'gain',
        'n_calibration_models': len(cal_seeds),
        'n_validation_models': len(val_seeds),
        'n_pairs_tested': n_pairs,
        'oos_r2': round(r2, 4),
        'pearson_r': round(corr, 4),
        'pearson_p': float(f'{pval:.2e}'),
        'snr_unreliable_count': int(unreliable),
        'snr_moderate_count': int(moderate),
        'snr_reliable_count': int(reliable),
        'snr_unreliable_frac': round(unreliable / n_pairs, 4),
        'snr_moderate_frac': round(moderate / n_pairs, 4),
        'snr_reliable_frac': round(reliable / n_pairs, 4),
        'headline': headline,
        'elapsed_seconds': round(elapsed, 1),
    }

    return result, predicted, observed, snr


# =====================================================================
# EXPERIMENT 2: High-Dimensional with TreeSHAP at P=50 (#27 extended)
# =====================================================================

def experiment_high_dim_shap():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: High-Dimensional with TreeSHAP at P=50 (#27 extended)")
    print("=" * 70)
    t0 = time.time()

    # Generate data
    print("\nGenerating P=50 data (20 informative, 15 redundant)...")
    X, y = make_classification(
        n_features=50, n_informative=20, n_redundant=15,
        n_clusters_per_class=1, random_state=42, n_samples=1000
    )
    P = X.shape[1]
    print(f"  X shape: {X.shape}")

    # Smaller M to keep SHAP tractable
    cal_seeds = list(range(42, 57))      # 15 models
    val_seeds = list(range(142, 157))    # 15 models

    # Train models (keep model objects for SHAP)
    print(f"\nTraining {len(cal_seeds)} calibration models (seeds 42-56)...")
    cal_models, cal_gain_imp = train_bootstrap_models(X, y, cal_seeds, is_classifier=True)

    print(f"Training {len(val_seeds)} validation models (seeds 142-156)...")
    val_models, val_gain_imp = train_bootstrap_models(X, y, val_seeds, is_classifier=True)

    # --- GAIN-BASED flip predictions (baseline) ---
    n_pairs = P * (P - 1) // 2
    print(f"\n--- Gain-based importance ({n_pairs} pairs) ---")
    pred_gain, obs_gain, snr_gain, pairs = gaussian_flip_predictions(cal_gain_imp, val_gain_imp)
    r2_gain = compute_oos_r2(pred_gain, obs_gain)
    corr_gain, pval_gain = pearsonr(pred_gain, obs_gain)
    print(f"  OOS R^2 (gain): {r2_gain:.4f}")
    print(f"  Pearson r (gain): {corr_gain:.4f}")

    # --- SHAP-BASED flip predictions ---
    X_sample = X[:100]  # Use first 100 samples for SHAP
    print(f"\nComputing TreeSHAP for {len(cal_models)} calibration models (X_sample={X_sample.shape[0]})...")
    cal_shap_imp = compute_shap_importance(cal_models, X_sample)

    print(f"Computing TreeSHAP for {len(val_models)} validation models...")
    val_shap_imp = compute_shap_importance(val_models, X_sample)

    print(f"\n--- SHAP-based importance ({n_pairs} pairs) ---")
    pred_shap, obs_shap, snr_shap, _ = gaussian_flip_predictions(cal_shap_imp, val_shap_imp)
    r2_shap = compute_oos_r2(pred_shap, obs_shap)
    corr_shap, pval_shap = pearsonr(pred_shap, obs_shap)
    print(f"  OOS R^2 (SHAP): {r2_shap:.4f}")
    print(f"  Pearson r (SHAP): {corr_shap:.4f}")

    # SNR breakdown for SHAP
    unreliable_shap = np.sum(snr_shap < 0.5)
    moderate_shap = np.sum((snr_shap >= 0.5) & (snr_shap <= 2.0))
    reliable_shap = np.sum(snr_shap > 2.0)

    unreliable_gain = np.sum(snr_gain < 0.5)
    moderate_gain = np.sum((snr_gain >= 0.5) & (snr_gain <= 2.0))
    reliable_gain = np.sum(snr_gain > 2.0)

    elapsed = time.time() - t0

    print(f"\n--- COMPARISON ---")
    print(f"  {'Metric':<30s} {'Gain':>10s} {'SHAP':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'OOS R^2':<30s} {r2_gain:>10.4f} {r2_shap:>10.4f}")
    print(f"  {'Pearson r':<30s} {corr_gain:>10.4f} {corr_shap:>10.4f}")
    print(f"  {'Unreliable (SNR<0.5)':<30s} {unreliable_gain:>10d} {unreliable_shap:>10d}")
    print(f"  {'Moderate (SNR 0.5-2)':<30s} {moderate_gain:>10d} {moderate_shap:>10d}")
    print(f"  {'Reliable (SNR>2)':<30s} {reliable_gain:>10d} {reliable_shap:>10d}")
    r2_improvement = r2_shap - r2_gain
    print(f"\n  SHAP vs gain R^2 improvement: {r2_improvement:+.4f}")
    print(f"  (P=200 gain-only R^2 was 0.53 for reference)")
    print(f"  Time: {elapsed:.1f}s")

    result = {
        'experiment': 'high_dim_shap_p50',
        'description': 'High-dimensional TreeSHAP vs gain at P=50 (#27 extended)',
        'n_features': P,
        'n_samples': 1000,
        'n_informative': 20,
        'n_redundant': 15,
        'shap_sample_size': int(X_sample.shape[0]),
        'n_calibration_models': len(cal_seeds),
        'n_validation_models': len(val_seeds),
        'n_pairs_tested': n_pairs,
        'gain': {
            'oos_r2': round(r2_gain, 4),
            'pearson_r': round(corr_gain, 4),
            'pearson_p': float(f'{pval_gain:.2e}'),
            'snr_unreliable_frac': round(unreliable_gain / n_pairs, 4),
            'snr_moderate_frac': round(moderate_gain / n_pairs, 4),
            'snr_reliable_frac': round(reliable_gain / n_pairs, 4),
        },
        'shap': {
            'oos_r2': round(r2_shap, 4),
            'pearson_r': round(corr_shap, 4),
            'pearson_p': float(f'{pval_shap:.2e}'),
            'snr_unreliable_frac': round(unreliable_shap / n_pairs, 4),
            'snr_moderate_frac': round(moderate_shap / n_pairs, 4),
            'snr_reliable_frac': round(reliable_shap / n_pairs, 4),
        },
        'r2_improvement_shap_over_gain': round(r2_improvement, 4),
        'reference_p200_gain_r2': 0.53,
        'elapsed_seconds': round(elapsed, 1),
    }

    return result, pred_gain, obs_gain, snr_gain, pred_shap, obs_shap, snr_shap


# =====================================================================
# Figure generation
# =====================================================================

def make_figure(gene_data, shap_data):
    """2-panel figure: gene expression scatter + high-dim SHAP scatter."""
    gene_result, gene_pred, gene_obs, gene_snr = gene_data
    (shap_result, pred_gain, obs_gain, snr_gain,
     pred_shap, obs_shap, snr_shap) = shap_data

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Gene expression scatter
    ax = axes[0]
    finite = np.isfinite(gene_snr)
    sc = ax.scatter(gene_pred[finite], gene_obs[finite],
                    c=gene_snr[finite], cmap='RdYlGn', s=4, alpha=0.5,
                    vmin=0, vmax=3, rasterized=True)
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('Predicted flip rate (Gaussian)')
    ax.set_ylabel('Observed flip rate (OOS)')
    ax.set_title(
        f"A. Gene Expression ({gene_result['dataset']})\n"
        f"OOS R² = {gene_result['oos_r2']:.3f}, "
        f"{gene_result['snr_unreliable_frac']*100:.0f}% unreliable",
        fontsize=10
    )
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.02, 0.52)
    ax.legend(fontsize=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('SNR', fontsize=8)

    # Panel B: SHAP vs Gain scatter (overlay)
    ax = axes[1]
    finite_g = np.isfinite(snr_gain)
    finite_s = np.isfinite(snr_shap)
    ax.scatter(pred_gain[finite_g], obs_gain[finite_g],
               c='steelblue', s=6, alpha=0.3, label=f"Gain (R²={shap_result['gain']['oos_r2']:.3f})",
               rasterized=True)
    ax.scatter(pred_shap[finite_s], obs_shap[finite_s],
               c='darkorange', s=6, alpha=0.3, label=f"SHAP (R²={shap_result['shap']['oos_r2']:.3f})",
               rasterized=True)
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Predicted flip rate (Gaussian)')
    ax.set_ylabel('Observed flip rate (OOS)')
    r2_delta = shap_result['r2_improvement_shap_over_gain']
    ax.set_title(
        f"B. P=50 Synthetic: SHAP vs Gain\n"
        f"R² improvement: {r2_delta:+.3f}",
        fontsize=10
    )
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.02, 0.52)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()

    out_path = FIG_DIR / 'gene_expression.pdf'
    with PdfPages(str(out_path)) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("Gene Expression + High-Dim SHAP Experiments")
    print("=" * 70)
    t_total = time.time()

    # Experiment 1: Gene expression
    gene_result, gene_pred, gene_obs, gene_snr = experiment_gene_expression()

    # Experiment 2: High-dim SHAP
    (shap_result, pred_gain, obs_gain, snr_gain,
     pred_shap, obs_shap, snr_shap) = experiment_high_dim_shap()

    # Save results
    results = {
        'gene_expression': gene_result,
        'high_dim_shap': shap_result,
        'total_elapsed_seconds': round(time.time() - t_total, 1),
    }

    out_path = OUT_DIR / 'results_gene_expression.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Generate figure
    gene_data = (gene_result, gene_pred, gene_obs, gene_snr)
    shap_data = (shap_result, pred_gain, obs_gain, snr_gain,
                 pred_shap, obs_shap, snr_shap)
    make_figure(gene_data, shap_data)

    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {time.time() - t_total:.1f}s")
    print(f"{'=' * 70}")
