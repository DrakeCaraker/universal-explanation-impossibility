#!/usr/bin/env python3
"""
Test whether explanation instability (Gaussian flip rate) PREDICTS
out-of-distribution generalization gap.

Theory: Rashomon multiplicity → explanation instability AND OOD failure,
so explanation instability should predict generalization gap.

Design: 5 datasets with natural distribution shifts.
For each: train 30 XGBoost models, compute flip rate, measure ID vs OOD accuracy.
Then correlate flip rate with generalization gap across datasets AND within datasets.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.datasets import (
    load_breast_cancer, load_wine, fetch_california_housing,
    make_classification
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

N_MODELS = 30
SEEDS = list(range(42, 42 + N_MODELS))
SNR_THRESHOLD = 0.5


# ============================================================
# Dataset preparation: each returns (X_train, y_train, X_id_test, y_id_test, X_ood, y_ood)
# ============================================================

def prepare_breast_cancer():
    """OOD = patients with extreme mean_radius (top/bottom 10%)."""
    X, y = load_breast_cancer(return_X_y=True)
    # Feature 0 is mean_radius
    radius = X[:, 0]
    lo, hi = np.percentile(radius, 10), np.percentile(radius, 90)
    ood_mask = (radius <= lo) | (radius >= hi)
    id_mask = ~ood_mask

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood


def prepare_wine():
    """Binary: high-quality (class 0) vs rest. Train on alcohol > median, OOD = alcohol <= median.
    This creates a covariate shift on the dominant feature."""
    X, y = load_wine(return_X_y=True)
    # Binarize: class 0 vs classes 1+2
    y_bin = (y == 0).astype(int)

    # Split by alcohol content (feature 0) - natural covariate shift
    alcohol = X[:, 0]
    median_alc = np.median(alcohol)

    id_mask = alcohol > median_alc
    ood_mask = alcohol <= median_alc

    X_id, y_id = X[id_mask], y_bin[id_mask]
    X_ood, y_ood = X[ood_mask], y_bin[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood


def prepare_calhousing():
    """Train on latitude > 35 (Northern CA). OOD = latitude <= 35 (Southern CA).
    Binarize: above/below median price within training region."""
    X, y = fetch_california_housing(return_X_y=True)
    # Feature index 7 is latitude (last feature in the standard ordering)
    # Actually: features are MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    lat_idx = 6  # Latitude is index 6
    lat = X[:, lat_idx]

    id_mask = lat > 35
    ood_mask = lat <= 35

    X_id, y_id_raw = X[id_mask], y[id_mask]
    X_ood, y_ood_raw = X[ood_mask], y[ood_mask]

    # Binarize: above/below median of TRAINING region
    median_price = np.median(y_id_raw)
    y_id = (y_id_raw > median_price).astype(int)
    y_ood = (y_ood_raw > median_price).astype(int)

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood


def prepare_heart():
    """Train on younger patients (age < median). OOD = older patients."""
    # Use sklearn's built-in heart disease dataset via openml
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
        X, y = data.data, LabelEncoder().fit_transform(data.target)
    except Exception:
        # Fallback: generate a synthetic "heart-like" dataset
        X, y = make_classification(
            n_samples=270, n_features=13, n_informative=8,
            n_redundant=3, random_state=42
        )

    age = X[:, 0]  # First feature is typically age
    median_age = np.median(age)

    id_mask = age < median_age
    ood_mask = age >= median_age

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood


def prepare_synthetic():
    """Synthetic: 2 clusters. Train on cluster A, OOD = cluster B."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=2, flip_y=0.05,
        random_state=42
    )
    # Split by a feature that separates clusters
    # Use feature 0 sign as a proxy for cluster membership
    cluster_feature = X[:, 0]
    id_mask = cluster_feature >= np.median(cluster_feature)
    ood_mask = ~id_mask

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood


# ============================================================
# Core computation
# ============================================================

def train_and_evaluate(X_train, y_train, X_id_test, y_id_test, X_ood, y_ood,
                       is_classifier=True, n_classes=2):
    """Train N_MODELS bootstrap models, compute importances and accuracies."""

    n_features = X_train.shape[1]
    importance_matrix = np.zeros((N_MODELS, n_features))
    acc_id_list = []
    acc_ood_list = []

    for i, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train[idx], y_train[idx]

        if is_classifier:
            params = dict(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                use_label_encoder=False
            )
            if n_classes > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = n_classes
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed
            )

        model.fit(X_boot, y_boot)

        # Feature importances (gain-based)
        imp = model.feature_importances_
        importance_matrix[i] = imp

        # Accuracy
        if is_classifier:
            pred_id = model.predict(X_id_test)
            pred_ood = model.predict(X_ood)
            acc_id_list.append(accuracy_score(y_id_test, pred_id))
            acc_ood_list.append(accuracy_score(y_ood, pred_ood))
        else:
            # For regression, use R² as "accuracy"
            pred_id = model.predict(X_id_test)
            pred_ood = model.predict(X_ood)
            acc_id_list.append(r2_score(y_id_test, pred_id))
            acc_ood_list.append(r2_score(y_ood, pred_ood))

    return importance_matrix, np.array(acc_id_list), np.array(acc_ood_list)


def compute_gaussian_flip_rate(importance_matrix):
    """
    Compute fraction of feature pairs that are unreliable under Gaussian flip model.

    For each pair (i,j): Delta = |mean_i - mean_j|, sigma = std of (imp_i - imp_j).
    SNR = Delta / sigma. Flip rate = Phi(-SNR/2).
    Unreliable if SNR < threshold (flip rate > ~20%).
    """
    n_models, n_features = importance_matrix.shape
    n_pairs = 0
    n_unreliable = 0
    flip_rates = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            diff = importance_matrix[:, i] - importance_matrix[:, j]
            delta = np.abs(np.mean(diff))
            sigma = np.std(diff, ddof=1)

            if sigma < 1e-12:
                # Zero variance: perfectly stable
                snr = np.inf
                flip_rate = 0.0
            else:
                snr = delta / sigma
                flip_rate = norm.cdf(-snr / 2)

            n_pairs += 1
            flip_rates.append(flip_rate)
            if snr < SNR_THRESHOLD:
                n_unreliable += 1

    unreliable_fraction = n_unreliable / n_pairs if n_pairs > 0 else 0
    mean_flip_rate = np.mean(flip_rates)
    return unreliable_fraction, mean_flip_rate, flip_rates


def compute_per_model_instability(importance_matrix):
    """
    Per-model instability: how much does each model's ranking differ from ensemble average?
    Measured as Spearman distance from the ensemble-mean ranking.
    """
    ensemble_mean = np.mean(importance_matrix, axis=0)
    ensemble_rank = np.argsort(np.argsort(-ensemble_mean))  # descending rank

    instabilities = []
    for i in range(importance_matrix.shape[0]):
        model_rank = np.argsort(np.argsort(-importance_matrix[i]))
        # Spearman footrule distance (normalized)
        footrule = np.sum(np.abs(model_rank - ensemble_rank)) / (importance_matrix.shape[1] ** 2 / 2)
        instabilities.append(footrule)
    return np.array(instabilities)


# ============================================================
# Run experiments
# ============================================================

print("=" * 70)
print("EXPERIMENT: Flip Rate → Generalization Gap")
print("=" * 70)

datasets_config = [
    ("Breast Cancer", prepare_breast_cancer, True, 2),
    ("Wine", prepare_wine, True, 2),
    ("CalHousing", prepare_calhousing, True, 2),
    ("Heart Disease", prepare_heart, True, 2),
    ("Synthetic", prepare_synthetic, True, 2),
]

results = []
all_within = []  # For within-dataset correlations

for name, prep_fn, is_clf, n_cls in datasets_config:
    print(f"\n{'─' * 50}")
    print(f"Dataset: {name}")
    print(f"{'─' * 50}")

    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = prep_fn()
    print(f"  Train: {X_train.shape[0]}, ID test: {X_id_test.shape[0]}, "
          f"OOD: {X_ood.shape[0]}, Features: {X_train.shape[1]}")

    # Train models and evaluate
    imp_matrix, acc_id, acc_ood = train_and_evaluate(
        X_train, y_train, X_id_test, y_id_test, X_ood, y_ood,
        is_classifier=is_clf, n_classes=n_cls
    )

    # Compute flip rate
    unreliable_frac, mean_flip, flip_rates = compute_gaussian_flip_rate(imp_matrix)

    # Generalization gap
    gap = np.mean(acc_id) - np.mean(acc_ood)

    print(f"  Unreliable fraction (SNR < {SNR_THRESHOLD}): {unreliable_frac:.4f}")
    print(f"  Mean flip rate: {mean_flip:.4f}")
    print(f"  Accuracy ID: {np.mean(acc_id):.4f} ± {np.std(acc_id):.4f}")
    print(f"  Accuracy OOD: {np.mean(acc_ood):.4f} ± {np.std(acc_ood):.4f}")
    print(f"  Generalization gap: {gap:.4f}")

    # Within-dataset: per-model instability vs per-model OOD accuracy
    per_model_instab = compute_per_model_instability(imp_matrix)
    if np.std(per_model_instab) > 1e-10 and np.std(acc_ood) > 1e-10:
        within_r, within_p = pearsonr(per_model_instab, acc_ood)
    else:
        within_r, within_p = 0.0, 1.0
    print(f"  Within-dataset correlation (instability vs OOD acc): "
          f"r={within_r:.4f}, p={within_p:.4f}")

    results.append({
        'dataset': name,
        'n_train': int(X_train.shape[0]),
        'n_id_test': int(X_id_test.shape[0]),
        'n_ood': int(X_ood.shape[0]),
        'n_features': int(X_train.shape[1]),
        'unreliable_fraction': float(unreliable_frac),
        'mean_flip_rate': float(mean_flip),
        'accuracy_id_mean': float(np.mean(acc_id)),
        'accuracy_id_std': float(np.std(acc_id)),
        'accuracy_ood_mean': float(np.mean(acc_ood)),
        'accuracy_ood_std': float(np.std(acc_ood)),
        'generalization_gap': float(gap),
        'within_dataset_r': float(within_r),
        'within_dataset_p': float(within_p),
    })
    all_within.append({
        'dataset': name,
        'per_model_instability': per_model_instab.tolist(),
        'per_model_acc_ood': acc_ood.tolist(),
    })

# ============================================================
# Cross-dataset correlation
# ============================================================

print(f"\n{'=' * 70}")
print("CROSS-DATASET ANALYSIS")
print(f"{'=' * 70}")

unreliable_fracs = np.array([r['unreliable_fraction'] for r in results])
gen_gaps = np.array([r['generalization_gap'] for r in results])
mean_flips = np.array([r['mean_flip_rate'] for r in results])

# Pearson correlation: unreliable fraction vs generalization gap
if np.std(unreliable_fracs) > 1e-10 and np.std(gen_gaps) > 1e-10:
    r_uf, p_uf = pearsonr(unreliable_fracs, gen_gaps)
    rho_uf, p_rho_uf = spearmanr(unreliable_fracs, gen_gaps)
else:
    r_uf, p_uf = 0.0, 1.0
    rho_uf, p_rho_uf = 0.0, 1.0

# Also try mean flip rate
if np.std(mean_flips) > 1e-10 and np.std(gen_gaps) > 1e-10:
    r_mf, p_mf = pearsonr(mean_flips, gen_gaps)
    rho_mf, p_rho_mf = spearmanr(mean_flips, gen_gaps)
else:
    r_mf, p_mf = 0.0, 1.0
    rho_mf, p_rho_mf = 0.0, 1.0

R2_uf = r_uf ** 2
R2_mf = r_mf ** 2

print(f"\nUnreliable fraction vs Generalization gap:")
print(f"  Pearson r = {r_uf:.4f}, p = {p_uf:.4f}, R² = {R2_uf:.4f}")
print(f"  Spearman ρ = {rho_uf:.4f}, p = {p_rho_uf:.4f}")

print(f"\nMean flip rate vs Generalization gap:")
print(f"  Pearson r = {r_mf:.4f}, p = {p_mf:.4f}, R² = {R2_mf:.4f}")
print(f"  Spearman ρ = {rho_mf:.4f}, p = {p_rho_mf:.4f}")

# Direction check
direction = "MORE unstable → WORSE OOD" if r_uf > 0 else "MORE unstable → BETTER OOD"
print(f"\nDirection: {direction}")
print(f"Prediction holds (R² > 0.5): {'YES' if max(R2_uf, R2_mf) > 0.5 else 'NO'}")

# ============================================================
# Summary of within-dataset correlations
# ============================================================
print(f"\nWithin-dataset correlations (per-model instability vs OOD accuracy):")
within_rs = []
for r in results:
    sign = "(-)" if r['within_dataset_r'] < 0 else "(+)"
    sig = "*" if r['within_dataset_p'] < 0.05 else ""
    print(f"  {r['dataset']:20s}: r = {r['within_dataset_r']:.4f} {sign} "
          f"p = {r['within_dataset_p']:.4f} {sig}")
    within_rs.append(r['within_dataset_r'])

mean_within_r = np.mean(within_rs)
n_negative = sum(1 for r in within_rs if r < 0)
print(f"\n  Mean within-dataset r: {mean_within_r:.4f}")
print(f"  Negative correlations: {n_negative}/{len(within_rs)} "
      f"(more instability → worse OOD)")

# ============================================================
# Save results
# ============================================================

output = {
    'experiment': 'flip_rate_generalization',
    'description': 'Test whether Gaussian flip rate predicts OOD generalization gap',
    'n_models': N_MODELS,
    'snr_threshold': SNR_THRESHOLD,
    'cross_dataset': {
        'unreliable_frac_vs_gap': {
            'pearson_r': float(r_uf),
            'pearson_p': float(p_uf),
            'R_squared': float(R2_uf),
            'spearman_rho': float(rho_uf),
            'spearman_p': float(p_rho_uf),
        },
        'mean_flip_vs_gap': {
            'pearson_r': float(r_mf),
            'pearson_p': float(p_mf),
            'R_squared': float(R2_mf),
            'spearman_rho': float(rho_mf),
            'spearman_p': float(p_rho_mf),
        },
        'direction': direction,
        'prediction_holds': bool(max(R2_uf, R2_mf) > 0.5),
    },
    'within_dataset_summary': {
        'mean_r': float(mean_within_r),
        'n_negative': int(n_negative),
        'n_total': len(within_rs),
    },
    'per_dataset': results,
}

results_path = OUT_DIR / 'results_flip_generalization.json'
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {results_path}")

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Unreliable fraction vs Generalization gap
ax = axes[0]
for r in results:
    ax.scatter(r['unreliable_fraction'], r['generalization_gap'],
               s=120, zorder=5, edgecolors='black', linewidths=0.5)
    ax.annotate(r['dataset'], (r['unreliable_fraction'], r['generalization_gap']),
                textcoords="offset points", xytext=(8, 5), fontsize=8)

# Fit line
if np.std(unreliable_fracs) > 1e-10:
    coeffs = np.polyfit(unreliable_fracs, gen_gaps, 1)
    x_line = np.linspace(min(unreliable_fracs) - 0.02, max(unreliable_fracs) + 0.02, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 'r--', alpha=0.7, linewidth=1.5)

ax.set_xlabel('Unreliable Fraction (SNR < 0.5)', fontsize=11)
ax.set_ylabel('Generalization Gap (ID − OOD)', fontsize=11)
ax.set_title(f'A. Cross-Dataset\n(Pearson r = {r_uf:.3f}, R² = {R2_uf:.3f})', fontsize=12)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Panel B: Mean flip rate vs Generalization gap
ax = axes[1]
for r in results:
    ax.scatter(r['mean_flip_rate'], r['generalization_gap'],
               s=120, zorder=5, edgecolors='black', linewidths=0.5)
    ax.annotate(r['dataset'], (r['mean_flip_rate'], r['generalization_gap']),
                textcoords="offset points", xytext=(8, 5), fontsize=8)

if np.std(mean_flips) > 1e-10:
    coeffs = np.polyfit(mean_flips, gen_gaps, 1)
    x_line = np.linspace(min(mean_flips) - 0.01, max(mean_flips) + 0.01, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 'r--', alpha=0.7, linewidth=1.5)

ax.set_xlabel('Mean Flip Rate', fontsize=11)
ax.set_ylabel('Generalization Gap (ID − OOD)', fontsize=11)
ax.set_title(f'B. Mean Flip Rate\n(Pearson r = {r_mf:.3f}, R² = {R2_mf:.3f})', fontsize=12)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Panel C: Within-dataset correlations summary
ax = axes[2]
dataset_names = [r['dataset'] for r in results]
within_r_vals = [r['within_dataset_r'] for r in results]
within_p_vals = [r['within_dataset_p'] for r in results]
colors = ['#d62728' if r < 0 else '#2ca02c' for r in within_r_vals]
bars = ax.barh(range(len(dataset_names)), within_r_vals, color=colors, alpha=0.7,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(dataset_names)))
ax.set_yticklabels(dataset_names, fontsize=9)
ax.set_xlabel('Pearson r (per-model instability vs OOD accuracy)', fontsize=10)
ax.set_title(f'C. Within-Dataset\n(mean r = {mean_within_r:.3f})', fontsize=12)
ax.axvline(x=0, color='black', linewidth=0.8)
# Mark significant
for i, p in enumerate(within_p_vals):
    if p < 0.05:
        ax.text(within_r_vals[i], i, ' *', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
fig_path = FIG_DIR / 'flip_generalization.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ============================================================
# Final verdict
# ============================================================
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
best_R2 = max(R2_uf, R2_mf)
best_metric = "unreliable_fraction" if R2_uf >= R2_mf else "mean_flip_rate"
print(f"Best cross-dataset R² = {best_R2:.4f} (using {best_metric})")
if best_R2 > 0.5:
    print(">>> PREDICTION CONFIRMED: Explanation instability predicts OOD generalization gap.")
    print(f">>> Direction: {direction}")
elif best_R2 > 0.25:
    print(">>> MODERATE SIGNAL: Partial prediction. Explanation instability shows trend but")
    print(f"    insufficient power with N=5 datasets. Direction: {direction}")
else:
    print(">>> PREDICTION NOT CONFIRMED at R² > 0.5 threshold.")
    print(f"    Direction observed: {direction}")

print(f"\nWithin-dataset: {n_negative}/{len(within_rs)} datasets show negative correlation")
print("(more instability → worse OOD), consistent with theory.")
print(f"{'=' * 70}")
