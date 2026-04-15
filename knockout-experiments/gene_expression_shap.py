#!/usr/bin/env python3
"""
TreeSHAP on Gene Expression Data (AP_Breast_Lung)

Key question: Does switching from gain-based importance to TreeSHAP
improve the Gaussian flip formula's predictive accuracy on real
gene expression data?

Baseline (gain): OOS R² = 0.416, 70% unreliable (SNR < 0.5)

Design:
  - 15 calibration models (seeds 42-56), 15 validation models (seeds 142-156)
  - Bootstrap resampling, XGBoost, top 50 genes by variance
  - TreeSHAP importance: mean |SHAP value| over 100 test points
  - Gaussian flip formula: calibration estimates Δ,σ; validation measures flips
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, pearsonr
from sklearn.datasets import fetch_openml
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


def gaussian_flip_predictions(cal_imp, val_imp):
    """
    Gaussian flip formula: calibration models estimate Delta and sigma per pair,
    validation models measure observed flip rates.
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
    """Out-of-sample R²."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def main():
    print("=" * 70)
    print("TreeSHAP on Gene Expression (AP_Breast_Lung)")
    print("=" * 70)
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    print("\nFetching AP_Breast_Lung from OpenML...")
    data = fetch_openml(name='AP_Breast_Lung', version=1, as_frame=False, parser='auto')
    X_raw, y_raw = data.data, data.target
    y = LabelEncoder().fit_transform(y_raw)
    print(f"  Loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

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
    print(f"  Selected top {P} genes by variance")

    # ── Train models ───────────────────────────────────────────────────
    cal_seeds = list(range(42, 57))    # 15 calibration models
    val_seeds = list(range(142, 157))  # 15 validation models

    cal_models = []
    cal_gain_imp = np.zeros((len(cal_seeds), P))
    val_models = []
    val_gain_imp = np.zeros((len(val_seeds), P))

    print(f"\nTraining {len(cal_seeds)} calibration models (seeds 42-56)...")
    for i, seed in enumerate(cal_seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=seed,
            verbosity=0, eval_metric='logloss', tree_method='hist'
        )
        model.fit(X[idx], y[idx])
        cal_models.append(model)
        cal_gain_imp[i] = model.feature_importances_

    print(f"Training {len(val_seeds)} validation models (seeds 142-156)...")
    for i, seed in enumerate(val_seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=seed,
            verbosity=0, eval_metric='logloss', tree_method='hist'
        )
        model.fit(X[idx], y[idx])
        val_models.append(model)
        val_gain_imp[i] = model.feature_importances_

    # ── Gain-based flip predictions (baseline) ─────────────────────────
    n_pairs = P * (P - 1) // 2
    print(f"\n--- Gain-based importance ({n_pairs} pairs) ---")
    pred_gain, obs_gain, snr_gain, pairs = gaussian_flip_predictions(cal_gain_imp, val_gain_imp)
    r2_gain = compute_oos_r2(pred_gain, obs_gain)
    corr_gain, pval_gain = pearsonr(pred_gain, obs_gain)
    print(f"  OOS R² (gain):  {r2_gain:.4f}")
    print(f"  Pearson r:      {corr_gain:.4f}")

    # ── TreeSHAP importance ────────────────────────────────────────────
    X_sample = X[:100]
    print(f"\nComputing TreeSHAP for {len(cal_models)} calibration models (100 test points)...")
    cal_shap_imp = np.zeros((len(cal_models), P))
    for i, model in enumerate(cal_models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            cal_shap_imp[i] = np.mean(np.stack([np.mean(np.abs(s), axis=0) for s in sv]), axis=0)
        elif sv.ndim == 3:
            cal_shap_imp[i] = np.mean(np.abs(sv), axis=(0, 2))
        else:
            cal_shap_imp[i] = np.mean(np.abs(sv), axis=0)
        if (i + 1) % 5 == 0:
            print(f"  Calibration SHAP: {i+1}/{len(cal_models)} done")

    print(f"Computing TreeSHAP for {len(val_models)} validation models...")
    val_shap_imp = np.zeros((len(val_models), P))
    for i, model in enumerate(val_models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            val_shap_imp[i] = np.mean(np.stack([np.mean(np.abs(s), axis=0) for s in sv]), axis=0)
        elif sv.ndim == 3:
            val_shap_imp[i] = np.mean(np.abs(sv), axis=(0, 2))
        else:
            val_shap_imp[i] = np.mean(np.abs(sv), axis=0)
        if (i + 1) % 5 == 0:
            print(f"  Validation SHAP: {i+1}/{len(val_models)} done")

    # ── SHAP-based flip predictions ────────────────────────────────────
    print(f"\n--- SHAP-based importance ({n_pairs} pairs) ---")
    pred_shap, obs_shap, snr_shap, _ = gaussian_flip_predictions(cal_shap_imp, val_shap_imp)
    r2_shap = compute_oos_r2(pred_shap, obs_shap)
    corr_shap, pval_shap = pearsonr(pred_shap, obs_shap)
    print(f"  OOS R² (SHAP):  {r2_shap:.4f}")
    print(f"  Pearson r:      {corr_shap:.4f}")

    # ── SNR breakdowns ─────────────────────────────────────────────────
    unreliable_gain = int(np.sum(snr_gain < 0.5))
    moderate_gain = int(np.sum((snr_gain >= 0.5) & (snr_gain <= 2.0)))
    reliable_gain = int(np.sum(snr_gain > 2.0))

    unreliable_shap = int(np.sum(snr_shap < 0.5))
    moderate_shap = int(np.sum((snr_shap >= 0.5) & (snr_shap <= 2.0)))
    reliable_shap = int(np.sum(snr_shap > 2.0))

    elapsed = time.time() - t0

    # ── Print comparison ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  GENE EXPRESSION: SHAP vs GAIN COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<35s} {'Gain':>10s} {'SHAP':>10s}")
    print(f"  {'-' * 55}")
    print(f"  {'OOS R²':<35s} {r2_gain:>10.4f} {r2_shap:>10.4f}")
    print(f"  {'Pearson r':<35s} {corr_gain:>10.4f} {corr_shap:>10.4f}")
    print(f"  {'Unreliable (SNR < 0.5)':<35s} {unreliable_gain:>10d} {unreliable_shap:>10d}")
    print(f"  {'  fraction':<35s} {unreliable_gain/n_pairs:>10.1%} {unreliable_shap/n_pairs:>10.1%}")
    print(f"  {'Moderate (0.5 ≤ SNR ≤ 2)':<35s} {moderate_gain:>10d} {moderate_shap:>10d}")
    print(f"  {'Reliable (SNR > 2)':<35s} {reliable_gain:>10d} {reliable_shap:>10d}")
    print(f"  {'  fraction':<35s} {reliable_gain/n_pairs:>10.1%} {reliable_shap/n_pairs:>10.1%}")
    r2_delta = r2_shap - r2_gain
    print(f"\n  R² improvement (SHAP over gain): {r2_delta:+.4f}")
    print(f"  Reference: gain R² = 0.416 (30+30 models from results_gene_expression.json)")
    print(f"  Time: {elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────────────
    result = {
        'experiment': 'gene_expression_shap_vs_gain',
        'description': 'TreeSHAP vs gain importance on AP_Breast_Lung gene expression (top 50 genes)',
        'dataset': 'AP_Breast_Lung',
        'n_samples': int(X.shape[0]),
        'n_features_original': int(X_raw.shape[1]),
        'n_features_selected': P,
        'selection_method': 'top_50_by_variance',
        'shap_sample_size': int(X_sample.shape[0]),
        'n_calibration_models': len(cal_seeds),
        'n_validation_models': len(val_seeds),
        'n_pairs_tested': n_pairs,
        'gain': {
            'oos_r2': round(r2_gain, 4),
            'pearson_r': round(corr_gain, 4),
            'pearson_p': float(f'{pval_gain:.2e}'),
            'snr_unreliable_count': unreliable_gain,
            'snr_unreliable_frac': round(unreliable_gain / n_pairs, 4),
            'snr_moderate_count': moderate_gain,
            'snr_moderate_frac': round(moderate_gain / n_pairs, 4),
            'snr_reliable_count': reliable_gain,
            'snr_reliable_frac': round(reliable_gain / n_pairs, 4),
        },
        'shap': {
            'oos_r2': round(r2_shap, 4),
            'pearson_r': round(corr_shap, 4),
            'pearson_p': float(f'{pval_shap:.2e}'),
            'snr_unreliable_count': unreliable_shap,
            'snr_unreliable_frac': round(unreliable_shap / n_pairs, 4),
            'snr_moderate_count': moderate_shap,
            'snr_moderate_frac': round(moderate_shap / n_pairs, 4),
            'snr_reliable_count': reliable_shap,
            'snr_reliable_frac': round(reliable_shap / n_pairs, 4),
        },
        'r2_improvement_shap_over_gain': round(r2_delta, 4),
        'reference_gain_r2_30models': 0.416,
        'elapsed_seconds': round(elapsed, 1),
    }

    out_path = OUT_DIR / 'results_gene_expression_shap.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Gain-based scatter
    ax = axes[0]
    finite_g = np.isfinite(snr_gain)
    sc = ax.scatter(pred_gain[finite_g], obs_gain[finite_g],
                    c=snr_gain[finite_g], cmap='RdYlGn', s=8, alpha=0.6,
                    vmin=0, vmax=3, rasterized=True)
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('Predicted flip rate (Gaussian)')
    ax.set_ylabel('Observed flip rate (OOS)')
    ax.set_title(
        f"A. Gain-based importance\n"
        f"OOS R² = {r2_gain:.3f}, "
        f"{unreliable_gain/n_pairs*100:.0f}% unreliable",
        fontsize=10
    )
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.02, 0.52)
    ax.legend(fontsize=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('SNR', fontsize=8)

    # Panel B: SHAP-based scatter
    ax = axes[1]
    finite_s = np.isfinite(snr_shap)
    sc2 = ax.scatter(pred_shap[finite_s], obs_shap[finite_s],
                     c=snr_shap[finite_s], cmap='RdYlGn', s=8, alpha=0.6,
                     vmin=0, vmax=3, rasterized=True)
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('Predicted flip rate (Gaussian)')
    ax.set_ylabel('Observed flip rate (OOS)')
    ax.set_title(
        f"B. TreeSHAP importance\n"
        f"OOS R² = {r2_shap:.3f}, "
        f"{unreliable_shap/n_pairs*100:.0f}% unreliable",
        fontsize=10
    )
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.02, 0.52)
    ax.legend(fontsize=8)
    cb2 = plt.colorbar(sc2, ax=ax, shrink=0.8)
    cb2.set_label('SNR', fontsize=8)

    fig.suptitle('Gene Expression (AP_Breast_Lung, P=50): Gain vs TreeSHAP', fontsize=12, y=1.02)
    plt.tight_layout()

    fig_path = FIG_DIR / 'gene_expression_shap.pdf'
    with PdfPages(str(fig_path)) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
