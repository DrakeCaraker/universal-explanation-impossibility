#!/usr/bin/env python3
"""
Out-of-sample validation of the Gaussian flip rate formula.

CRITICAL DESIGN: Independent calibration and validation model sets.
- Calibration models (seeds 42..71): estimate Delta and sigma
- Validation models (seeds 142..171): measure observed flip rates
- This is genuinely out-of-sample — no information leakage.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, pearsonr
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---------- datasets ----------
datasets = []
datasets.append(('Breast Cancer', *load_breast_cancer(return_X_y=True), True))
datasets.append(('Wine', *load_wine(return_X_y=True), True))
datasets.append(('CalHousing', *fetch_california_housing(return_X_y=True), False))

for name, oml_name in [('Diabetes', 'diabetes'), ('Heart', 'heart-statlog')]:
    try:
        X, y = fetch_openml(oml_name, version=1, return_X_y=True, as_frame=False, parser='auto')
        y = LabelEncoder().fit_transform(y)
        datasets.append((name, X, y, True))
    except Exception as e:
        print(f"Warning: could not load {name}: {e}")


def compute_shap_importance(model, X_sample, is_classifier):
    """Compute mean |SHAP| per feature."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        return np.mean(np.stack([np.mean(np.abs(s), axis=0) for s in sv]), axis=0)
    elif sv.ndim == 3:
        return np.mean(np.abs(sv), axis=(0, 2))
    else:
        return np.mean(np.abs(sv), axis=0)


def train_models_and_get_importance(X, y, is_classifier, seeds, n_explain=200):
    """Train bootstrap models and return SHAP importance matrix (n_models x P)."""
    P = X.shape[1]
    n_models = len(seeds)
    imp = np.zeros((n_models, P))
    X_explain = X[:n_explain]

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        if is_classifier:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, random_state=seed,
                verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=seed, verbosity=0
            )
        model.fit(X[idx], y[idx])
        imp[i] = compute_shap_importance(model, X_explain, is_classifier)
    return imp


def measure_flip_rates(imp_matrix):
    """Compute pairwise observed flip rates from an importance matrix (n_models x P)."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    observed = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            n_flips = 0
            n_pairs = 0
            for a in range(n_models):
                for b in range(a + 1, n_models):
                    if (imp_matrix[a, j] > imp_matrix[a, k]) != (imp_matrix[b, j] > imp_matrix[b, k]):
                        n_flips += 1
                    n_pairs += 1
            observed[pair_idx] = n_flips / n_pairs if n_pairs > 0 else 0.0
            pair_idx += 1
    return observed


def predict_flip_rates_gaussian(imp_matrix):
    """Predict flip rates from calibration importance matrix using Gaussian CDF formula."""
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


def predict_flip_raw_gap(imp_matrix):
    """Baseline: predict flip from |mean gap| only (no sigma normalization)."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    raw_gaps = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            diffs = imp_matrix[:, j] - imp_matrix[:, k]
            raw_gaps[pair_idx] = abs(np.mean(diffs))
            pair_idx += 1
    return raw_gaps


def compute_correlation_baseline(X, imp_cal, imp_val):
    """Baseline: predict flip from |Pearson r(feature_j, feature_k)| on raw data.
    Fit on calibration observed flips, evaluate on validation observed flips."""
    P = X.shape[1]
    n_pairs_total = P * (P - 1) // 2

    # Compute |r| for each pair
    abs_corrs = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            r, _ = pearsonr(X[:, j], X[:, k])
            abs_corrs[pair_idx] = abs(r) if np.isfinite(r) else 0.0
            pair_idx += 1

    # Observed flips from calibration (to fit the linear model)
    obs_cal = measure_flip_rates(imp_cal)

    # Fit linear regression: flip ~ |r| on calibration
    reg = LinearRegression()
    reg.fit(abs_corrs.reshape(-1, 1), obs_cal)

    # Predict on validation
    predicted = reg.predict(abs_corrs.reshape(-1, 1))
    return predicted


def r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))


# ---------- main experiment ----------
print("=" * 70)
print("OUT-OF-SAMPLE VALIDATION: Gaussian Flip Rate Formula")
print("Calibration: seeds 42-71 | Validation: seeds 142-171")
print("=" * 70)

results = {
    "design": "Independent calibration (seeds 42-71) and validation (seeds 142-171) model sets",
    "datasets": {}
}

all_plot_data = {}

for ds_name, X, y, is_clf in datasets:
    P = X.shape[1]
    n_pairs = P * (P - 1) // 2
    print(f"\n--- {ds_name} (P={P}, {n_pairs} pairs) ---")

    # Calibration models
    cal_seeds = list(range(42, 72))  # 30 models
    print(f"  Training {len(cal_seeds)} calibration models...")
    cal_imp = train_models_and_get_importance(X, y, is_clf, cal_seeds)

    # Validation models
    val_seeds = list(range(142, 172))  # 30 models
    print(f"  Training {len(val_seeds)} validation models...")
    val_imp = train_models_and_get_importance(X, y, is_clf, val_seeds)

    # Predict from calibration
    print("  Computing predicted flip rates (from calibration)...")
    predicted = predict_flip_rates_gaussian(cal_imp)

    # Observe from validation
    print("  Computing observed flip rates (from validation)...")
    observed = measure_flip_rates(val_imp)

    # Gaussian formula R²
    oos_r2 = r_squared(observed, predicted)
    oos_rmse_val = rmse(observed, predicted)

    # Baseline 1: correlation
    print("  Computing correlation baseline...")
    corr_predicted = compute_correlation_baseline(X, cal_imp, val_imp)
    corr_r2 = r_squared(observed, corr_predicted)

    # Baseline 2: raw gap (no sigma normalization)
    raw_gaps_cal = predict_flip_raw_gap(cal_imp)
    obs_cal = measure_flip_rates(cal_imp)
    # Fit linear: flip ~ raw_gap on calibration
    reg_gap = LinearRegression()
    reg_gap.fit(raw_gaps_cal.reshape(-1, 1), obs_cal)
    raw_gaps_val = predict_flip_raw_gap(cal_imp)  # predict using cal params
    gap_predicted = reg_gap.predict(raw_gaps_val.reshape(-1, 1))
    gap_r2 = r_squared(observed, gap_predicted)

    # Fraction of variance explained by Gaussian over correlation baseline
    if corr_r2 < oos_r2:
        gaussian_fraction = (oos_r2 - corr_r2) / (1.0 - corr_r2) if corr_r2 < 1.0 else 0.0
    else:
        gaussian_fraction = 0.0

    results["datasets"][ds_name] = {
        "P": int(P),
        "n_pairs": int(n_pairs),
        "oos_r2": round(float(oos_r2), 4),
        "oos_rmse": round(float(oos_rmse_val), 6),
        "correlation_baseline_r2": round(float(corr_r2), 4),
        "raw_gap_baseline_r2": round(float(gap_r2), 4),
        "gaussian_fraction": round(float(gaussian_fraction), 4),
        "n_cal": len(cal_seeds),
        "n_val": len(val_seeds)
    }

    all_plot_data[ds_name] = {
        'predicted': predicted,
        'observed': observed,
        'oos_r2': oos_r2,
        'corr_r2': corr_r2,
        'gap_r2': gap_r2
    }

    print(f"  OOS R² = {oos_r2:.4f}  |  Corr baseline R² = {corr_r2:.4f}  |  Raw gap R² = {gap_r2:.4f}")
    print(f"  OOS RMSE = {oos_rmse_val:.6f}")

# ---------- Breast Cancer M=100 ----------
print("\n--- Breast Cancer M=100 (50 cal + 50 val) ---")
bc_X, bc_y = load_breast_cancer(return_X_y=True)
cal_seeds_100 = list(range(42, 92))   # 50 calibration
val_seeds_100 = list(range(142, 192)) # 50 validation

print(f"  Training {len(cal_seeds_100)} calibration models...")
cal_imp_100 = train_models_and_get_importance(bc_X, bc_y, True, cal_seeds_100)
print(f"  Training {len(val_seeds_100)} validation models...")
val_imp_100 = train_models_and_get_importance(bc_X, bc_y, True, val_seeds_100)

predicted_100 = predict_flip_rates_gaussian(cal_imp_100)
observed_100 = measure_flip_rates(val_imp_100)
r2_100 = r_squared(observed_100, predicted_100)

results["breast_cancer_m100"] = {
    "oos_r2": round(float(r2_100), 4),
    "n_cal": 50,
    "n_val": 50
}
print(f"  OOS R² (M=100) = {r2_100:.4f}")

# ---------- Save results ----------
with open(OUT_DIR / 'results_gaussian_flip_validated.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR / 'results_gaussian_flip_validated.json'}")

# ---------- Figure ----------
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)
ds_names = list(all_plot_data.keys())

for i, ax in enumerate(axes):
    if i >= len(ds_names):
        ax.set_visible(False)
        continue
    name = ds_names[i]
    d = all_plot_data[name]

    ax.scatter(d['predicted'], d['observed'], alpha=0.3, s=8, c='steelblue', edgecolors='none')
    lims = [0, max(max(d['predicted']), max(d['observed'])) * 1.05 + 0.01]
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Predicted flip rate')
    ax.set_ylabel('Observed flip rate')
    ax.set_title(name)
    ax.text(0.05, 0.92, f"OOS R$^2$ = {d['oos_r2']:.3f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0.05, 0.78, f"Corr baseline = {d['corr_r2']:.3f}\nRaw gap = {d['gap_r2']:.3f}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))

fig.suptitle('Out-of-Sample Validation: Gaussian Flip Rate Formula', fontsize=13, fontweight='bold')
fig.savefig(FIG_DIR / 'gaussian_flip_validated.pdf', dpi=150, bbox_inches='tight')
print(f"Figure saved to {FIG_DIR / 'gaussian_flip_validated.pdf'}")

# ---------- Summary table ----------
print("\n" + "=" * 90)
print(f"{'Dataset':<16} {'P':>4} {'Pairs':>6} {'OOS R²':>8} {'Corr BL':>8} {'Gap BL':>8} {'RMSE':>10}")
print("-" * 90)
for ds_name, info in results["datasets"].items():
    print(f"{ds_name:<16} {info['P']:>4} {info['n_pairs']:>6} {info['oos_r2']:>8.4f} "
          f"{info['correlation_baseline_r2']:>8.4f} {info['raw_gap_baseline_r2']:>8.4f} {info['oos_rmse']:>10.6f}")
print("-" * 90)
print(f"Breast Cancer M=100: OOS R² = {results['breast_cancer_m100']['oos_r2']:.4f}")
print("=" * 90)
