#!/usr/bin/env python3
"""
Multiverse Analysis Simulation — Replication Crisis Proxy

The "garden of forking paths" (Gelman & Loken 2013) IS the Rashomon property:
multiple reasonable analysis pipelines produce the same summary fit but different
conclusions about which features matter.

Design:
  - Base dataset: Breast Cancer (N=569, P=30)
  - 30 analysis pipelines varying model class, scaling, feature selection, and
    bootstrap seed
  - For each pipeline: fit, compute feature importance, record top-5
  - Compute pairwise flip rates, fit Gaussian flip formula, report robustness

Headline: "X% of feature importance claims are non-replicable across reasonable
analysis choices, as predicted by the Gaussian flip formula."
"""

import warnings
warnings.filterwarnings('ignore')

import json
import itertools
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from scipy.stats import norm
from scipy.optimize import minimize_scalar

OUT_DIR = Path(__file__).parent
np.random.seed(2026)

# ──────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X_full, y = data.data, data.target
feature_names = list(data.feature_names)
N, P = X_full.shape
print(f"Dataset: Breast Cancer, N={N}, P={P}")

# ──────────────────────────────────────────────────────────────────────
# 2. Define multiverse dimensions
# ──────────────────────────────────────────────────────────────────────
model_classes = ['xgboost', 'random_forest', 'ridge']

scalers = {
    'none': None,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
}

feature_selections = ['none', 'top20_variance', 'top20_correlation']

# Base seeds for the 27 combinations + 3 extra bootstrap variations
base_seed = 42


def select_features(X, y, method, k=20):
    """Select top-k features by variance or correlation with target."""
    if method == 'none':
        return X, list(range(X.shape[1]))
    elif method == 'top20_variance':
        variances = np.var(X, axis=0)
        idx = np.argsort(variances)[-k:]
        return X[:, idx], list(idx)
    elif method == 'top20_correlation':
        corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
        idx = np.argsort(corrs)[-k:]
        return X[:, idx], list(idx)
    else:
        raise ValueError(f"Unknown feature selection: {method}")


def fit_and_importance(X_train, y_train, model_class, seed):
    """Fit model and return feature importance vector (length = X_train.shape[1])."""
    if model_class == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=seed, use_label_encoder=False, eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X_train, y_train)
        # Use gain-based importance
        imp = model.feature_importances_
    elif model_class == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=seed,
        )
        model.fit(X_train, y_train)
        imp = model.feature_importances_
    elif model_class == 'ridge':
        model = RidgeClassifier(alpha=1.0, random_state=seed)
        model.fit(X_train, y_train)
        imp = np.abs(model.coef_).ravel()
        if imp.ndim > 1:
            imp = imp.mean(axis=0)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    return imp, model


# ──────────────────────────────────────────────────────────────────────
# 3. Run the multiverse
# ──────────────────────────────────────────────────────────────────────
pipelines = []
pipeline_id = 0

# 27 systematic combinations
for model_class, (scaler_name, scaler_obj), feat_sel in itertools.product(
    model_classes, scalers.items(), feature_selections
):
    pipelines.append({
        'id': pipeline_id,
        'model_class': model_class,
        'scaler': scaler_name,
        'feature_selection': feat_sel,
        'bootstrap_seed': base_seed + pipeline_id,
    })
    pipeline_id += 1

# 3 extra bootstrap variations (same config as first 3 pipelines, different seeds)
for i in range(3):
    base = pipelines[i].copy()
    base['id'] = pipeline_id
    base['bootstrap_seed'] = 1000 + i
    pipelines.append(base)
    pipeline_id += 1

print(f"Total pipelines: {len(pipelines)}")

# Run each pipeline
results = []
importance_matrix = np.zeros((len(pipelines), P))  # Always in original feature space

for pipe in pipelines:
    seed = pipe['bootstrap_seed']
    rng = np.random.RandomState(seed)

    # Bootstrap resample
    idx = rng.choice(N, size=N, replace=True)
    X_boot, y_boot = X_full[idx], y[idx]

    # Feature scaling
    scaler_obj = scalers[pipe['scaler']]
    if scaler_obj is not None:
        # Clone the scaler to avoid state leakage
        from sklearn.base import clone
        sc = clone(scaler_obj)
        X_proc = sc.fit_transform(X_boot)
    else:
        X_proc = X_boot.copy()

    # Feature selection
    X_sel, selected_idx = select_features(X_proc, y_boot, pipe['feature_selection'])

    # Fit and get importance
    imp, model = fit_and_importance(X_sel, y_boot, pipe['model_class'], seed)

    # Map importance back to original feature space
    full_imp = np.zeros(P)
    for local_j, global_j in enumerate(selected_idx):
        full_imp[global_j] = imp[local_j]

    importance_matrix[pipe['id']] = full_imp

    # Record top-5 features
    top5_idx = np.argsort(full_imp)[-5:][::-1]
    top5_names = [feature_names[j] for j in top5_idx]

    # Cross-val accuracy on the bootstrap sample
    acc = cross_val_score(model, X_sel, y_boot, cv=3, scoring='accuracy').mean()

    results.append({
        'pipeline_id': pipe['id'],
        'model_class': pipe['model_class'],
        'scaler': pipe['scaler'],
        'feature_selection': pipe['feature_selection'],
        'bootstrap_seed': pipe['bootstrap_seed'],
        'top5_features': top5_names,
        'top5_indices': [int(j) for j in top5_idx],
        'cv_accuracy': float(acc),
    })

    print(f"  Pipeline {pipe['id']:2d}: {pipe['model_class']:14s} | "
          f"{pipe['scaler']:8s} | {pipe['feature_selection']:18s} | "
          f"acc={acc:.3f} | top={top5_names[0]}")

# ──────────────────────────────────────────────────────────────────────
# 4. Compute pairwise flip rates per feature pair
# ──────────────────────────────────────────────────────────────────────
n_pipelines = len(pipelines)

# For each feature: is it in the top-5 across pipelines?
in_top5 = np.zeros((n_pipelines, P), dtype=bool)
for r in results:
    for j in r['top5_indices']:
        in_top5[r['pipeline_id'], j] = True

# Per-feature: fraction of pipelines where it appears in top-5
feature_inclusion_rate = in_top5.mean(axis=0)

# Pairwise flip rate for each feature: across all pipeline pairs,
# how often does the feature's top-5 status change?
feature_flip_rates = []
for j in range(P):
    agreements = 0
    total = 0
    for i1 in range(n_pipelines):
        for i2 in range(i1 + 1, n_pipelines):
            if in_top5[i1, j] != in_top5[i2, j]:
                agreements += 1
            total += 1
    flip_rate = agreements / total if total > 0 else 0
    feature_flip_rates.append(flip_rate)

feature_flip_rates = np.array(feature_flip_rates)

# ──────────────────────────────────────────────────────────────────────
# 5. Gaussian flip formula: predict flip rate from Delta/sigma
# ──────────────────────────────────────────────────────────────────────
# Approach: For each pipeline, rank-normalize importances to [0,1] so that
# different model classes are comparable. Then compute per-feature:
#   Delta_j = |mean rank-normalized importance - threshold|
#   sigma_j = std of rank-normalized importance across pipelines
# Predicted flip rate = 2 * Phi(-|Delta_j| / sigma_j)

# Rank-normalize importance within each pipeline to handle scale differences
rank_importance = np.zeros_like(importance_matrix)
for i in range(n_pipelines):
    imp_i = importance_matrix[i]
    # Rank-based normalization: rank / P
    order = np.argsort(np.argsort(imp_i))  # ranks 0..P-1
    rank_importance[i] = order / (P - 1)  # normalize to [0, 1]

# Compute threshold per pipeline (rank-normalized importance of the 5th-ranked feature)
thresholds = np.sort(rank_importance, axis=1)[:, -5]  # 5th largest rank-norm
mean_threshold = thresholds.mean()

# Per-feature statistics in rank-normalized space
mean_imp = rank_importance.mean(axis=0)
std_imp = rank_importance.std(axis=0)

delta = np.abs(mean_imp - mean_threshold)
# Avoid division by zero
sigma = np.maximum(std_imp, 1e-10)

# Gaussian predicted flip rate (with calibration parameter alpha)
# Fit alpha to minimize MSE between predicted and observed flip rates
# predicted_flip = 2 * Phi(-alpha * Delta_j / sigma_j)
active_features = np.where(feature_inclusion_rate > 0)[0]

def gaussian_mse(alpha):
    pred = 2 * norm.cdf(-alpha * delta[active_features] / sigma[active_features])
    return np.mean((feature_flip_rates[active_features] - pred) ** 2)

# Optimize alpha
opt = minimize_scalar(gaussian_mse, bounds=(0.01, 10.0), method='bounded')
alpha_opt = opt.x

predicted_flip = 2 * norm.cdf(-alpha_opt * delta / sigma)
predicted_flip = np.clip(predicted_flip, 0, 1)

# R² for the calibrated Gaussian formula
mask = std_imp > 1e-8  # Only features with nonzero variation
if mask.sum() > 2:
    observed = feature_flip_rates[mask]
    predicted = predicted_flip[mask]
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    r2_gaussian = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
else:
    r2_gaussian = float('nan')

# Also compute uncalibrated R² (alpha=1) for comparison
predicted_flip_uncal = 2 * norm.cdf(-delta / sigma)
predicted_flip_uncal = np.clip(predicted_flip_uncal, 0, 1)
if mask.sum() > 2:
    obs_uc = feature_flip_rates[mask]
    pred_uc = predicted_flip_uncal[mask]
    ss_res_uc = np.sum((obs_uc - pred_uc) ** 2)
    r2_uncalibrated = 1 - ss_res_uc / ss_tot if ss_tot > 0 else float('nan')
else:
    r2_uncalibrated = float('nan')

# ──────────────────────────────────────────────────────────────────────
# 5b. Within-model-class Gaussian analysis (where formula should work best)
# ──────────────────────────────────────────────────────────────────────
from scipy.stats import spearmanr

within_class_r2 = {}
for mc in model_classes:
    mc_ids = [r['pipeline_id'] for r in results if r['model_class'] == mc]
    if len(mc_ids) < 3:
        continue
    mc_imp = importance_matrix[mc_ids]
    mc_in_top5 = in_top5[mc_ids]
    n_mc = len(mc_ids)

    # Per-feature flip rate within this model class
    mc_flip = np.zeros(P)
    for j in range(P):
        flips = 0
        pairs = 0
        for i1 in range(n_mc):
            for i2 in range(i1 + 1, n_mc):
                if mc_in_top5[i1, j] != mc_in_top5[i2, j]:
                    flips += 1
                pairs += 1
        mc_flip[j] = flips / pairs if pairs > 0 else 0

    # Rank-normalize within model class
    mc_rank = np.zeros_like(mc_imp)
    for i in range(n_mc):
        order = np.argsort(np.argsort(mc_imp[i]))
        mc_rank[i] = order / (P - 1)

    mc_thresholds = np.sort(mc_rank, axis=1)[:, -5]
    mc_mean_thresh = mc_thresholds.mean()
    mc_mean_imp = mc_rank.mean(axis=0)
    mc_std_imp = mc_rank.std(axis=0)
    mc_delta = np.abs(mc_mean_imp - mc_mean_thresh)
    mc_sigma = np.maximum(mc_std_imp, 1e-10)

    mc_pred_flip = 2 * norm.cdf(-mc_delta / mc_sigma)
    mc_pred_flip = np.clip(mc_pred_flip, 0, 1)

    mc_mask = mc_std_imp > 1e-8
    if mc_mask.sum() > 2:
        obs = mc_flip[mc_mask]
        pred = mc_pred_flip[mc_mask]
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        mc_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        # Spearman correlation (more robust)
        rho_s, _ = spearmanr(obs, pred)
    else:
        mc_r2 = float('nan')
        rho_s = float('nan')

    within_class_r2[mc] = {'r2': float(mc_r2), 'spearman_rho': float(rho_s)}

# Overall Spearman correlation (rank-based, more robust than R²)
active_obs = feature_flip_rates[active_features]
active_pred = predicted_flip[active_features]
overall_spearman, _ = spearmanr(active_obs, active_pred)

# ──────────────────────────────────────────────────────────────────────
# 6. Robustness analysis
# ──────────────────────────────────────────────────────────────────────

# A feature importance claim is "robust" if the feature appears in top-5
# in >= 80% of pipelines
robust_threshold = 0.80
robust_features = [feature_names[j] for j in range(P) if feature_inclusion_rate[j] >= robust_threshold]
non_robust_count = sum(1 for j in range(P) if 0 < feature_inclusion_rate[j] < robust_threshold)
total_claimed = sum(1 for j in range(P) if feature_inclusion_rate[j] > 0)

non_replicable_fraction = non_robust_count / total_claimed if total_claimed > 0 else 0

# Average flip rate across features that appear at least once in top-5
active_mask = feature_inclusion_rate > 0
mean_flip_rate = feature_flip_rates[active_mask].mean() if active_mask.sum() > 0 else 0

# Agreement across pipelines on the exact top-5 set
top5_sets = [frozenset(r['top5_indices']) for r in results]
jaccard_pairs = []
for i1 in range(n_pipelines):
    for i2 in range(i1 + 1, n_pipelines):
        inter = len(top5_sets[i1] & top5_sets[i2])
        union = len(top5_sets[i1] | top5_sets[i2])
        jaccard_pairs.append(inter / union if union > 0 else 1.0)
mean_jaccard = np.mean(jaccard_pairs)

# ──────────────────────────────────────────────────────────────────────
# 7. Summary and output
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MULTIVERSE ANALYSIS RESULTS")
print("=" * 70)
print(f"Dataset: Breast Cancer (N={N}, P={P})")
print(f"Pipelines: {n_pipelines}")
print(f"Mean CV accuracy across pipelines: {np.mean([r['cv_accuracy'] for r in results]):.3f}")
print(f"  (range: {np.min([r['cv_accuracy'] for r in results]):.3f} - "
      f"{np.max([r['cv_accuracy'] for r in results]):.3f})")
print()
print(f"Features ever appearing in top-5: {total_claimed}")
print(f"Robust features (>=80% inclusion): {len(robust_features)}")
print(f"  {robust_features}")
print(f"Non-robust features (0% < inclusion < 80%): {non_robust_count}")
print(f"Non-replicable fraction: {non_replicable_fraction:.1%}")
print()
print(f"Mean pairwise flip rate (active features): {mean_flip_rate:.3f}")
print(f"Mean pairwise Jaccard similarity of top-5 sets: {mean_jaccard:.3f}")
print()
print(f"Gaussian flip formula R² (full multiverse): {r2_gaussian:.3f}")
print(f"Gaussian flip formula Spearman rho (full multiverse): {overall_spearman:.3f}")
print(f"Within-model-class Gaussian R²:")
for mc, vals in within_class_r2.items():
    print(f"  {mc:14s}: R²={vals['r2']:.3f}, Spearman={vals['spearman_rho']:.3f}")
print()

# Best within-class R²
best_mc = max(within_class_r2, key=lambda k: within_class_r2[k]['r2'])
best_mc_r2 = within_class_r2[best_mc]['r2']

headline = (
    f"{non_replicable_fraction:.0%} of feature importance claims are non-replicable "
    f"across reasonable analysis choices (Jaccard similarity = {mean_jaccard:.2f}). "
    f"The Gaussian flip formula achieves Spearman rho = {overall_spearman:.2f} across "
    f"the full multiverse and R² = {best_mc_r2:.2f} within {best_mc}."
)
print(f"HEADLINE: {headline}")

# Per-feature detail
print("\nPer-feature detail (sorted by inclusion rate):")
sorted_idx = np.argsort(feature_inclusion_rate)[::-1]
for j in sorted_idx:
    if feature_inclusion_rate[j] > 0:
        print(f"  {feature_names[j]:30s}  inclusion={feature_inclusion_rate[j]:.2f}  "
              f"flip={feature_flip_rates[j]:.3f}  pred_flip={predicted_flip[j]:.3f}  "
              f"Δ/σ={delta[j]/sigma[j]:.2f}")

# ──────────────────────────────────────────────────────────────────────
# 8. Save results
# ──────────────────────────────────────────────────────────────────────
output = {
    'experiment': 'multiverse_analysis',
    'dataset': 'Breast Cancer',
    'N': int(N),
    'P': int(P),
    'n_pipelines': n_pipelines,
    'pipeline_dimensions': {
        'model_classes': model_classes,
        'scalers': list(scalers.keys()),
        'feature_selections': feature_selections,
        'n_systematic': 27,
        'n_bootstrap_extra': 3,
    },
    'accuracy': {
        'mean': float(np.mean([r['cv_accuracy'] for r in results])),
        'min': float(np.min([r['cv_accuracy'] for r in results])),
        'max': float(np.max([r['cv_accuracy'] for r in results])),
    },
    'robustness': {
        'total_features_claimed': int(total_claimed),
        'robust_features_count': len(robust_features),
        'robust_features': robust_features,
        'non_robust_count': int(non_robust_count),
        'non_replicable_fraction': float(non_replicable_fraction),
        'robust_threshold': robust_threshold,
    },
    'flip_rates': {
        'mean_pairwise_flip_rate': float(mean_flip_rate),
        'mean_jaccard_similarity': float(mean_jaccard),
    },
    'gaussian_formula': {
        'full_multiverse_r2': float(r2_gaussian),
        'full_multiverse_spearman': float(overall_spearman),
        'within_class_r2': within_class_r2,
        'calibration_alpha': float(alpha_opt),
        'per_feature': {
            feature_names[j]: {
                'inclusion_rate': float(feature_inclusion_rate[j]),
                'observed_flip_rate': float(feature_flip_rates[j]),
                'predicted_flip_rate': float(predicted_flip[j]),
                'delta_over_sigma': float(delta[j] / sigma[j]),
            }
            for j in range(P) if feature_inclusion_rate[j] > 0
        },
    },
    'headline': headline,
    'pipelines': results,
}

out_path = OUT_DIR / 'results_multiverse.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
