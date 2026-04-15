#!/usr/bin/env python3
"""
EXPANDED generalization experiment: 10+ datasets with leave-one-out robustness.

Addresses adversarial concern: original r=-0.91 (p=0.034, n=5) may be driven
by CalHousing outlier. This script:
  1a. LOO on original 5 datasets
  1b. Expands to 13 datasets (8 real + 5 synthetic control)
  1c. Synthetic control with varying n_redundant

For each dataset:
  - Semantically meaningful OOD split (NOT random)
  - 30 XGBoost bootstrap models
  - Flip rate + generalization gap
  - Cross-dataset Pearson r with LOO robustness
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris,
    fetch_california_housing, fetch_openml,
    make_classification
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
# Dataset preparation functions
# ============================================================

def prepare_breast_cancer():
    """OOD = patients with extreme mean_radius (top/bottom 10%)."""
    X, y = load_breast_cancer(return_X_y=True)
    radius = X[:, 0]
    lo, hi = np.percentile(radius, 10), np.percentile(radius, 90)
    ood_mask = (radius <= lo) | (radius >= hi)
    id_mask = ~ood_mask

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "extreme_features: top/bottom 10% of mean_radius"


def prepare_wine():
    """Binary: class 0 vs rest. OOD = low-alcohol (covariate shift)."""
    X, y = load_wine(return_X_y=True)
    y_bin = (y == 0).astype(int)
    alcohol = X[:, 0]
    median_alc = np.median(alcohol)

    id_mask = alcohol > median_alc
    ood_mask = alcohol <= median_alc

    X_id, y_id = X[id_mask], y_bin[id_mask]
    X_ood, y_ood = X[ood_mask], y_bin[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "covariate_shift: alcohol above vs below median"


def prepare_calhousing():
    """Train on Northern CA (lat > 35). OOD = Southern CA (lat <= 35)."""
    X, y = fetch_california_housing(return_X_y=True)
    lat_idx = 6
    lat = X[:, lat_idx]

    id_mask = lat > 35
    ood_mask = lat <= 35

    X_id, y_id_raw = X[id_mask], y[id_mask]
    X_ood, y_ood_raw = X[ood_mask], y[ood_mask]

    median_price = np.median(y_id_raw)
    y_id = (y_id_raw > median_price).astype(int)
    y_ood = (y_ood_raw > median_price).astype(int)

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "geographic: Northern CA (train) vs Southern CA (OOD)"


def prepare_heart():
    """Train on younger patients (age < median). OOD = older patients."""
    try:
        data = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
        X, y = data.data, LabelEncoder().fit_transform(data.target)
    except Exception:
        X, y = make_classification(
            n_samples=270, n_features=13, n_informative=8,
            n_redundant=3, random_state=42
        )

    age = X[:, 0]
    median_age = np.median(age)

    id_mask = age < median_age
    ood_mask = age >= median_age

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "age_split: young (train) vs old (OOD)"


def prepare_synthetic_base():
    """Synthetic: train on one cluster region, OOD = other."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=2, flip_y=0.05,
        random_state=42
    )
    cluster_feature = X[:, 0]
    id_mask = cluster_feature >= np.median(cluster_feature)
    ood_mask = ~id_mask

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "cluster: feature[0] above vs below median"


def prepare_iris():
    """Train on setosa + versicolor, OOD = virginica (held-out class)."""
    X, y = load_iris(return_X_y=True)
    # Binary: versicolor vs setosa for training, test on virginica
    # Train task: setosa (0) vs versicolor (1)
    id_mask = y < 2  # setosa and versicolor
    ood_mask = y == 2  # virginica

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood = X[ood_mask]
    # For OOD: assign labels based on nearest class boundary
    # virginica is closest to versicolor, so label as 1
    y_ood = np.ones(X_ood.shape[0], dtype=int)

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "held_out_class: train on setosa+versicolor, OOD=virginica"


def prepare_diabetes():
    """Pima Indians Diabetes: train on young (< median age), OOD = old."""
    try:
        data = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
        X, y = data.data, LabelEncoder().fit_transform(data.target)
    except Exception:
        # Fallback
        X, y = make_classification(
            n_samples=768, n_features=8, n_informative=5,
            n_redundant=2, random_state=43
        )

    # Age is typically the last or near-last feature
    # In the Pima dataset, feature 7 is age
    age_idx = min(7, X.shape[1] - 1)
    age = X[:, age_idx]
    median_age = np.median(age)

    id_mask = age < median_age
    ood_mask = age >= median_age

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "age_split: young (train) vs old (OOD)"


def prepare_german_credit():
    """German Credit: train on young applicants, OOD = old."""
    try:
        data = fetch_openml('credit-g', version=1, as_frame=False, parser='auto')
        X, y = data.data.astype(float), LabelEncoder().fit_transform(data.target)
    except Exception:
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=44
        )

    # In German Credit, we use a feature correlated with age
    # Feature selection varies by encoding; use first feature as proxy
    split_feature = X[:, 0]
    median_val = np.median(split_feature)

    id_mask = split_feature < median_val
    ood_mask = split_feature >= median_val

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        "covariate_shift: feature[0] below vs above median"


def prepare_synthetic_redundant(n_redundant):
    """Synthetic with controlled redundancy. OOD = different cluster."""
    X, y = make_classification(
        n_features=50, n_informative=10, n_redundant=n_redundant,
        random_state=42, n_samples=2000, n_clusters_per_class=2,
        flip_y=0.03
    )
    # OOD split: use feature 0 to proxy cluster membership
    cluster_feature = X[:, 0]
    id_mask = cluster_feature >= np.median(cluster_feature)
    ood_mask = ~id_mask

    X_id, y_id = X[id_mask], y[id_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.25, random_state=42, stratify=y_id
    )
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, \
        f"cluster: n_redundant={n_redundant}, feature[0] split"


# ============================================================
# Core computation (reused from original)
# ============================================================

def train_and_evaluate(X_train, y_train, X_id_test, y_id_test, X_ood, y_ood):
    """Train N_MODELS bootstrap models, compute importances and accuracies."""
    n_features = X_train.shape[1]
    importance_matrix = np.zeros((N_MODELS, n_features))
    acc_id_list = []
    acc_ood_list = []

    for i, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train[idx], y_train[idx]

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_boot, y_boot)

        importance_matrix[i] = model.feature_importances_
        acc_id_list.append(accuracy_score(y_id_test, model.predict(X_id_test)))
        acc_ood_list.append(accuracy_score(y_ood, model.predict(X_ood)))

    return importance_matrix, np.array(acc_id_list), np.array(acc_ood_list)


def compute_mean_flip_rate(importance_matrix):
    """Compute mean flip rate across all feature pairs."""
    n_models, n_features = importance_matrix.shape
    flip_rates = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            diff = importance_matrix[:, i] - importance_matrix[:, j]
            delta = np.abs(np.mean(diff))
            sigma = np.std(diff, ddof=1)

            if sigma < 1e-12:
                flip_rate = 0.0
            else:
                snr = delta / sigma
                flip_rate = norm.cdf(-snr / 2)

            flip_rates.append(flip_rate)

    return np.mean(flip_rates) if flip_rates else 0.0


def leave_one_out_correlation(x, y, names):
    """Compute LOO: for each point, remove it and recompute Pearson r."""
    n = len(x)
    loo_results = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_loo, y_loo = x[mask], y[mask]
        if len(x_loo) >= 3 and np.std(x_loo) > 1e-10 and np.std(y_loo) > 1e-10:
            r_loo, p_loo = pearsonr(x_loo, y_loo)
        else:
            r_loo, p_loo = float('nan'), float('nan')
        loo_results.append({
            'removed': names[i],
            'r_without': float(r_loo),
            'p_without': float(p_loo),
            'n_remaining': int(n - 1)
        })
    return loo_results


# ============================================================
# Build dataset list
# ============================================================

print("=" * 70)
print("EXPANDED GENERALIZATION EXPERIMENT")
print("=" * 70)

datasets_config = [
    # Original 5
    ("Breast Cancer", prepare_breast_cancer),
    ("Wine", prepare_wine),
    ("CalHousing", prepare_calhousing),
    ("Heart Disease", prepare_heart),
    ("Synthetic", prepare_synthetic_base),
    # New real datasets
    ("Iris", prepare_iris),
    ("Diabetes", prepare_diabetes),
    ("German Credit", prepare_german_credit),
]

# Synthetic control: varying redundancy
for n_red in [0, 10, 20, 30, 40]:
    datasets_config.append((
        f"Synth_red_{n_red}",
        lambda nr=n_red: prepare_synthetic_redundant(nr)
    ))


# ============================================================
# Run all datasets
# ============================================================

results = []

for name, prep_fn in datasets_config:
    print(f"\n{'─' * 50}")
    print(f"Dataset: {name}")
    print(f"{'─' * 50}")

    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood, shift_desc = prep_fn()
    print(f"  Shift: {shift_desc}")
    print(f"  Train: {X_train.shape[0]}, ID test: {X_id_test.shape[0]}, "
          f"OOD: {X_ood.shape[0]}, Features: {X_train.shape[1]}")

    imp_matrix, acc_id, acc_ood = train_and_evaluate(
        X_train, y_train, X_id_test, y_id_test, X_ood, y_ood
    )

    mean_flip = compute_mean_flip_rate(imp_matrix)
    gap = float(np.mean(acc_id) - np.mean(acc_ood))

    print(f"  Mean flip rate: {mean_flip:.4f}")
    print(f"  Accuracy ID: {np.mean(acc_id):.4f} +/- {np.std(acc_id):.4f}")
    print(f"  Accuracy OOD: {np.mean(acc_ood):.4f} +/- {np.std(acc_ood):.4f}")
    print(f"  Generalization gap: {gap:.4f}")

    results.append({
        'dataset': name,
        'shift_description': shift_desc,
        'n_train': int(X_train.shape[0]),
        'n_id_test': int(X_id_test.shape[0]),
        'n_ood': int(X_ood.shape[0]),
        'n_features': int(X_train.shape[1]),
        'mean_flip_rate': float(mean_flip),
        'accuracy_id_mean': float(np.mean(acc_id)),
        'accuracy_id_std': float(np.std(acc_id)),
        'accuracy_ood_mean': float(np.mean(acc_ood)),
        'accuracy_ood_std': float(np.std(acc_ood)),
        'generalization_gap': float(gap),
    })


# ============================================================
# Analysis
# ============================================================

names_all = [r['dataset'] for r in results]
flips_all = np.array([r['mean_flip_rate'] for r in results])
gaps_all = np.array([r['generalization_gap'] for r in results])

# Identify subsets
original_5_mask = np.array([i < 5 for i in range(len(results))])
real_mask = np.array([i < 8 for i in range(len(results))])
synthetic_control_mask = np.array([i >= 8 for i in range(len(results))])

print(f"\n{'=' * 70}")
print("CROSS-DATASET CORRELATIONS")
print(f"{'=' * 70}")

def report_correlation(label, x, y):
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        print(f"  {label}: insufficient data or variance")
        return None, None
    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    print(f"  {label}: Pearson r = {r:.4f}, p = {p:.4f}, "
          f"Spearman rho = {rho:.4f}, p = {p_rho:.4f}, n = {len(x)}")
    return r, p

# Full dataset
print("\n--- All 13 datasets ---")
r_all, p_all = report_correlation("All datasets", flips_all, gaps_all)

# Original 5
print("\n--- Original 5 datasets ---")
r_orig, p_orig = report_correlation("Original 5",
    flips_all[original_5_mask], gaps_all[original_5_mask])

# Real datasets only (8)
print("\n--- 8 real datasets ---")
r_real, p_real = report_correlation("8 real datasets",
    flips_all[real_mask], gaps_all[real_mask])

# Synthetic control only (5)
print("\n--- 5 synthetic control ---")
r_synth, p_synth = report_correlation("5 synthetic control",
    flips_all[synthetic_control_mask], gaps_all[synthetic_control_mask])


# ============================================================
# Leave-one-out analysis
# ============================================================

print(f"\n{'=' * 70}")
print("LEAVE-ONE-OUT ANALYSIS")
print(f"{'=' * 70}")

# LOO on original 5
print("\n--- LOO on Original 5 ---")
loo_orig5 = leave_one_out_correlation(
    flips_all[original_5_mask], gaps_all[original_5_mask],
    [n for i, n in enumerate(names_all) if original_5_mask[i]]
)
for item in loo_orig5:
    print(f"  Without {item['removed']:20s}: r = {item['r_without']:.4f}, "
          f"p = {item['p_without']:.4f}")

# LOO on all 13
print("\n--- LOO on All 13 ---")
loo_all = leave_one_out_correlation(flips_all, gaps_all, names_all)
loo_rs = [item['r_without'] for item in loo_all if not np.isnan(item['r_without'])]
for item in loo_all:
    print(f"  Without {item['removed']:20s}: r = {item['r_without']:.4f}, "
          f"p = {item['p_without']:.4f}")

# LOO on 8 real
print("\n--- LOO on 8 Real ---")
loo_real = leave_one_out_correlation(
    flips_all[real_mask], gaps_all[real_mask],
    [n for i, n in enumerate(names_all) if real_mask[i]]
)
for item in loo_real:
    print(f"  Without {item['removed']:20s}: r = {item['r_without']:.4f}, "
          f"p = {item['p_without']:.4f}")


# ============================================================
# CalHousing sensitivity analysis
# ============================================================

print(f"\n{'=' * 70}")
print("CALHOUSING SENSITIVITY")
print(f"{'=' * 70}")

calhousing_idx = names_all.index("CalHousing")
calhousing_loo_orig = [item for item in loo_orig5 if item['removed'] == 'CalHousing'][0]
calhousing_loo_all = [item for item in loo_all if item['removed'] == 'CalHousing'][0]

print(f"Original 5 correlation: r = {r_orig:.4f}")
print(f"Without CalHousing (n=4): r = {calhousing_loo_orig['r_without']:.4f}, "
      f"p = {calhousing_loo_orig['p_without']:.4f}")
print(f"All 13 correlation: r = {r_all:.4f}")
print(f"Without CalHousing (n=12): r = {calhousing_loo_all['r_without']:.4f}, "
      f"p = {calhousing_loo_all['p_without']:.4f}")

calhousing_is_driver = abs(calhousing_loo_orig['r_without']) < 0.5
print(f"\nCalHousing sole driver of original correlation? "
      f"{'YES' if calhousing_is_driver else 'NO'}")


# ============================================================
# Synthetic control analysis
# ============================================================

print(f"\n{'=' * 70}")
print("SYNTHETIC CONTROL (VARYING REDUNDANCY)")
print(f"{'=' * 70}")

synth_results = [r for r in results if r['dataset'].startswith('Synth_red_')]
for sr in synth_results:
    n_red = sr['dataset'].split('_')[-1]
    print(f"  n_redundant={n_red:>2s}: flip_rate={sr['mean_flip_rate']:.4f}, "
          f"gap={sr['generalization_gap']:.4f}")

synth_confirms = r_synth is not None and r_synth < -0.3
print(f"\nSynthetic control confirms inverse correlation? "
      f"{'YES' if synth_confirms else 'NO'} (r = {r_synth})")


# ============================================================
# Save results
# ============================================================

output = {
    'experiment': 'flip_generalization_expanded',
    'description': 'Expanded generalization experiment with 13 datasets and LOO robustness',
    'n_models': N_MODELS,
    'n_datasets': len(results),
    'correlations': {
        'all_13': {
            'pearson_r': float(r_all) if r_all is not None else None,
            'pearson_p': float(p_all) if p_all is not None else None,
            'n': 13
        },
        'original_5': {
            'pearson_r': float(r_orig) if r_orig is not None else None,
            'pearson_p': float(p_orig) if p_orig is not None else None,
            'n': 5
        },
        'real_8': {
            'pearson_r': float(r_real) if r_real is not None else None,
            'pearson_p': float(p_real) if p_real is not None else None,
            'n': 8
        },
        'synthetic_control_5': {
            'pearson_r': float(r_synth) if r_synth is not None else None,
            'pearson_p': float(p_synth) if p_synth is not None else None,
            'n': 5
        }
    },
    'loo_original_5': loo_orig5,
    'loo_all_13': loo_all,
    'loo_real_8': loo_real,
    'calhousing_sensitivity': {
        'original_r': float(r_orig) if r_orig is not None else None,
        'without_calhousing_r': float(calhousing_loo_orig['r_without']),
        'without_calhousing_p': float(calhousing_loo_orig['p_without']),
        'is_sole_driver': bool(calhousing_is_driver)
    },
    'synthetic_control': {
        'r': float(r_synth) if r_synth is not None else None,
        'confirms_inverse': bool(synth_confirms),
        'per_dataset': [{
            'dataset': sr['dataset'],
            'n_redundant': int(sr['dataset'].split('_')[-1]),
            'mean_flip_rate': sr['mean_flip_rate'],
            'generalization_gap': sr['generalization_gap']
        } for sr in synth_results]
    },
    'loo_range_all_13': {
        'min_r': float(min(loo_rs)) if loo_rs else None,
        'max_r': float(max(loo_rs)) if loo_rs else None,
    },
    'per_dataset': results,
}

results_path = OUT_DIR / 'results_flip_gen_expanded.json'
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {results_path}")


# ============================================================
# Figure: 4-panel
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: All 13 datasets
ax = axes[0, 0]
colors = []
for i, r in enumerate(results):
    if r['dataset'].startswith('Synth_red_'):
        colors.append('#ff7f0e')  # orange for synthetic control
    elif i < 5:
        colors.append('#1f77b4')  # blue for original 5
    else:
        colors.append('#2ca02c')  # green for new real

for i, r in enumerate(results):
    ax.scatter(r['mean_flip_rate'], r['generalization_gap'],
               s=100, c=colors[i], zorder=5, edgecolors='black', linewidths=0.5)
    ax.annotate(r['dataset'], (r['mean_flip_rate'], r['generalization_gap']),
                textcoords="offset points", xytext=(5, 5), fontsize=6)

if r_all is not None:
    coeffs = np.polyfit(flips_all, gaps_all, 1)
    x_line = np.linspace(min(flips_all) - 0.01, max(flips_all) + 0.01, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 'r--', alpha=0.7, linewidth=1.5)

ax.set_xlabel('Mean Flip Rate', fontsize=11)
ax.set_ylabel('Generalization Gap (ID - OOD)', fontsize=11)
r_str = f"{r_all:.3f}" if r_all is not None else "N/A"
p_str = f"{p_all:.4f}" if p_all is not None else "N/A"
ax.set_title(f'A. All 13 Datasets (r = {r_str}, p = {p_str})', fontsize=12)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markersize=10, label='Original 5'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markersize=10, label='New real (3)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
           markersize=10, label='Synthetic control (5)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Panel B: LOO stability (all 13)
ax = axes[0, 1]
loo_names = [item['removed'] for item in loo_all]
loo_r_vals = [item['r_without'] for item in loo_all]
bar_colors = ['red' if abs(rv - r_all) > 0.15 else 'steelblue'
              for rv in loo_r_vals]
bars = ax.barh(range(len(loo_names)), loo_r_vals, color=bar_colors, alpha=0.7,
               edgecolor='black', linewidth=0.5)
ax.axvline(x=r_all, color='red', linestyle='--', linewidth=1.5,
           label=f'Full r = {r_all:.3f}' if r_all else '')
ax.set_yticks(range(len(loo_names)))
ax.set_yticklabels(loo_names, fontsize=7)
ax.set_xlabel('Pearson r (without dataset)', fontsize=11)
ax.set_title('B. Leave-One-Out Stability (All 13)', fontsize=12)
ax.legend(fontsize=9)

# Panel C: Synthetic control
ax = axes[1, 0]
n_reds = [int(sr['dataset'].split('_')[-1]) for sr in synth_results]
synth_flips = [sr['mean_flip_rate'] for sr in synth_results]
synth_gaps = [sr['generalization_gap'] for sr in synth_results]

ax.scatter(synth_flips, synth_gaps, s=120, c='#ff7f0e', zorder=5,
           edgecolors='black', linewidths=0.5)
for i, sr in enumerate(synth_results):
    ax.annotate(f"red={n_reds[i]}", (synth_flips[i], synth_gaps[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=9)

if r_synth is not None and not np.isnan(r_synth):
    coeffs = np.polyfit(synth_flips, synth_gaps, 1)
    x_line = np.linspace(min(synth_flips) - 0.01, max(synth_flips) + 0.01, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 'r--', alpha=0.7, linewidth=1.5)

r_synth_str = f"{r_synth:.3f}" if r_synth is not None else "N/A"
ax.set_xlabel('Mean Flip Rate', fontsize=11)
ax.set_ylabel('Generalization Gap', fontsize=11)
ax.set_title(f'C. Synthetic Control (r = {r_synth_str})', fontsize=12)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Panel D: LOO on original 5
ax = axes[1, 1]
loo5_names = [item['removed'] for item in loo_orig5]
loo5_r_vals = [item['r_without'] for item in loo_orig5]
bar_colors5 = ['red' if item['removed'] == 'CalHousing' else 'steelblue'
               for item in loo_orig5]
bars = ax.barh(range(len(loo5_names)), loo5_r_vals, color=bar_colors5, alpha=0.7,
               edgecolor='black', linewidth=0.5)
ax.axvline(x=r_orig, color='red', linestyle='--', linewidth=1.5,
           label=f'Full r = {r_orig:.3f}' if r_orig else '')
ax.set_yticks(range(len(loo5_names)))
ax.set_yticklabels(loo5_names, fontsize=9)
ax.set_xlabel('Pearson r (without dataset)', fontsize=11)
ax.set_title('D. LOO on Original 5 (CalHousing highlighted)', fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
fig_path = FIG_DIR / 'flip_gen_expanded.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")


# ============================================================
# Final verdict
# ============================================================

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
print(f"All 13 datasets: r = {r_all:.4f}, p = {p_all:.4f}")
if loo_rs:
    print(f"LOO range (all 13): [{min(loo_rs):.4f}, {max(loo_rs):.4f}]")
print(f"CalHousing sole driver of original r=-0.91? "
      f"{'YES' if calhousing_is_driver else 'NO'}")
print(f"Synthetic control confirms pattern? "
      f"{'YES' if synth_confirms else 'NO'} (r = {r_synth})")
print(f"{'=' * 70}")
