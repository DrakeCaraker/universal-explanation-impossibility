#!/usr/bin/env python3
"""
Multi-model generalization test for the Gaussian flip rate formula.

Tests that the Gaussian CDF flip-rate prediction Φ(-|Δ|/(σ√2)) generalizes
beyond XGBoost to Random Forest and Ridge regression across 3 datasets.

Design: 3 datasets × 3 model classes = 9 combinations.
Each combination: 30 calibration + 30 validation bootstrap models.
OOS R² measured per combination.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---------- datasets ----------
datasets = [
    ('Breast Cancer', *load_breast_cancer(return_X_y=True), True),
    ('Wine', *load_wine(return_X_y=True), True),
    ('CalHousing', *fetch_california_housing(return_X_y=True), False),
]

# ---------- importance extractors ----------

def shap_importance(model, X_sample):
    """Mean |SHAP| via TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        return np.mean(np.stack([np.mean(np.abs(s), axis=0) for s in sv]), axis=0)
    elif sv.ndim == 3:
        return np.mean(np.abs(sv), axis=(0, 2))
    else:
        return np.mean(np.abs(sv), axis=0)


def rf_importance(model, X_sample):
    """Gain-based feature_importances_ (fast, no extra computation)."""
    return model.feature_importances_


def ridge_importance(model, X_sample):
    """|coefficient| importance for linear models."""
    coef = model.coef_
    if coef.ndim == 1:
        return np.abs(coef)
    else:
        # Multi-class: average across classes
        return np.abs(coef).mean(axis=0)


# ---------- model builders ----------

def build_xgb(is_clf, seed):
    if is_clf:
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=seed,
            verbosity=0, use_label_encoder=False, eval_metric='logloss'
        )
    else:
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=4, random_state=seed, verbosity=0
        )


def build_rf(is_clf, seed):
    if is_clf:
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=seed)


def build_ridge(is_clf, seed):
    if is_clf:
        return RidgeClassifier(alpha=1.0)
    else:
        return Ridge(alpha=1.0)


MODEL_CLASSES = [
    ('XGBoost', build_xgb, shap_importance),
    ('RandomForest', build_rf, rf_importance),
    ('Ridge', build_ridge, ridge_importance),
]

# ---------- core functions ----------

def train_models_and_get_importance(X, y, is_clf, seeds, build_fn, importance_fn, n_explain=200):
    """Train bootstrap models and return importance matrix (n_models x P)."""
    P = X.shape[1]
    n_models = len(seeds)
    imp = np.zeros((n_models, P))
    X_explain = X[:n_explain]

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        model = build_fn(is_clf, seed)
        model.fit(X[idx], y[idx])
        imp[i] = importance_fn(model, X_explain)
    return imp


def measure_flip_rates(imp_matrix):
    """Compute pairwise observed flip rates from importance matrix."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    observed = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            n_flips = 0
            n_comparisons = 0
            for a in range(n_models):
                for b in range(a + 1, n_models):
                    if (imp_matrix[a, j] > imp_matrix[a, k]) != (imp_matrix[b, j] > imp_matrix[b, k]):
                        n_flips += 1
                    n_comparisons += 1
            observed[pair_idx] = n_flips / n_comparisons if n_comparisons > 0 else 0.0
            pair_idx += 1
    return observed


def predict_flip_rates_gaussian(imp_matrix):
    """Predict flip rates using Gaussian CDF: Φ(-|Δ|/(σ√2))."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    predicted = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            diffs = imp_matrix[:, j] - imp_matrix[:, k]
            delta = np.mean(diffs)
            sigma = np.std(diffs)
            if sigma > 1e-10:
                predicted[pair_idx] = norm.cdf(-abs(delta) / (sigma * np.sqrt(2)))
            else:
                predicted[pair_idx] = 0.0
            pair_idx += 1
    return predicted


def r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


# ---------- main experiment ----------
print("=" * 70)
print("MULTI-MODEL GENERALIZATION: Gaussian Flip Rate Formula")
print("3 datasets × 3 model classes = 9 combinations")
print("Calibration: seeds 42-71 | Validation: seeds 142-171")
print("=" * 70)

cal_seeds = list(range(42, 72))   # 30 calibration models
val_seeds = list(range(142, 172)) # 30 validation models

results = {
    "design": "3 datasets x 3 model classes, independent calibration/validation",
    "cal_seeds": "42-71",
    "val_seeds": "142-171",
    "combinations": {}
}

plot_data = {}

for ds_name, X, y, is_clf in datasets:
    P = X.shape[1]
    n_pairs = P * (P - 1) // 2
    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name} (P={P}, {n_pairs} feature pairs)")
    print(f"{'='*50}")

    for model_name, build_fn, importance_fn in MODEL_CLASSES:
        combo_key = f"{ds_name} / {model_name}"
        print(f"\n  [{model_name}] Training {len(cal_seeds)} calibration models...")
        cal_imp = train_models_and_get_importance(
            X, y, is_clf, cal_seeds, build_fn, importance_fn
        )

        print(f"  [{model_name}] Training {len(val_seeds)} validation models...")
        val_imp = train_models_and_get_importance(
            X, y, is_clf, val_seeds, build_fn, importance_fn
        )

        print(f"  [{model_name}] Computing predicted flip rates (calibration)...")
        predicted = predict_flip_rates_gaussian(cal_imp)

        print(f"  [{model_name}] Computing observed flip rates (validation)...")
        observed = measure_flip_rates(val_imp)

        oos_r2 = r_squared(observed, predicted)
        oos_rmse = np.sqrt(np.mean((observed - predicted) ** 2))

        results["combinations"][combo_key] = {
            "dataset": ds_name,
            "model_class": model_name,
            "P": int(P),
            "n_pairs": int(n_pairs),
            "oos_r2": round(float(oos_r2), 4),
            "oos_rmse": round(float(oos_rmse), 6),
            "is_classification": is_clf,
        }

        plot_data[combo_key] = {
            'predicted': predicted,
            'observed': observed,
            'oos_r2': oos_r2,
            'dataset': ds_name,
            'model_class': model_name,
        }

        print(f"  [{model_name}] OOS R² = {oos_r2:.4f}  |  RMSE = {oos_rmse:.6f}")

# ---------- Save results ----------
with open(OUT_DIR / 'results_gaussian_multimodel.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR / 'results_gaussian_multimodel.json'}")

# ---------- Figure: 3×3 grid ----------
fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)

ds_names = ['Breast Cancer', 'Wine', 'CalHousing']
model_names = ['XGBoost', 'RandomForest', 'Ridge']

for row, ds_name in enumerate(ds_names):
    for col, model_name in enumerate(model_names):
        ax = axes[row, col]
        combo_key = f"{ds_name} / {model_name}"

        if combo_key not in plot_data:
            ax.set_visible(False)
            continue

        d = plot_data[combo_key]
        pred = d['predicted']
        obs = d['observed']

        ax.scatter(pred, obs, alpha=0.3, s=8, c='steelblue', edgecolors='none')
        max_val = max(max(pred), max(obs)) * 1.05 + 0.01
        lims = [0, max_val]
        ax.plot(lims, lims, 'k--', lw=1, label='y = x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Predicted flip rate')
        ax.set_ylabel('Observed flip rate')

        r2 = d['oos_r2']
        ax.set_title(f"{ds_name} / {model_name}", fontsize=10)
        ax.text(0.05, 0.92, f"OOS R$^2$ = {r2:.3f}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Multi-Model Generalization: Gaussian Flip Rate Formula\n'
             r'Predicted = $\Phi(-|\Delta|/\sigma\sqrt{2})$ from calibration; '
             'observed from independent validation',
             fontsize=12, fontweight='bold')

fig.savefig(FIG_DIR / 'gaussian_multimodel.pdf', dpi=150, bbox_inches='tight')
print(f"Figure saved to {FIG_DIR / 'gaussian_multimodel.pdf'}")

# ---------- Summary table ----------
print("\n" + "=" * 80)
print(f"{'Combination':<35} {'P':>4} {'Pairs':>6} {'OOS R²':>8} {'RMSE':>10}")
print("-" * 80)
for combo_key, info in results["combinations"].items():
    print(f"{combo_key:<35} {info['P']:>4} {info['n_pairs']:>6} "
          f"{info['oos_r2']:>8.4f} {info['oos_rmse']:>10.6f}")
print("-" * 80)

# R² summary matrix
print("\n" + "=" * 60)
print("R² SUMMARY MATRIX")
print(f"{'':>18}", end="")
for mn in model_names:
    print(f"{mn:>14}", end="")
print()
print("-" * 60)
for ds_name in ds_names:
    print(f"{ds_name:>18}", end="")
    for model_name in model_names:
        combo_key = f"{ds_name} / {model_name}"
        r2 = results["combinations"][combo_key]["oos_r2"]
        print(f"{r2:>14.4f}", end="")
    print()
print("=" * 60)
