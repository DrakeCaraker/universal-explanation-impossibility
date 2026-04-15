#!/usr/bin/env python3
"""
Gaussian diagnostics: 5 experiments addressing reviewer recommendations.

Experiment 1 (#2a): Linear baseline comparison — is CDF shape needed beyond linear?
Experiment 2 (#26, #2b): Gaussianity diagnostics — Shapiro-Wilk, Q-Q plots
Experiment 3 (#25): M sensitivity — minimum models for R² > 0.7
Experiment 4 (#31): Leave-one-out for η regression
Experiment 5 (#30): Per-pair to top-k connection
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, shapiro, skew, kurtosis
from scipy.special import erfc
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ============================================================
# Shared utilities (from gaussian_flip_cv.py)
# ============================================================

def compute_shap_importance(model, X_sample):
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
    """Train bootstrap XGBoost models and return SHAP importance matrix (n_models x P)."""
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
        imp[i] = compute_shap_importance(model, X_explain)
    return imp


def measure_flip_rates(imp_matrix):
    """Compute pairwise observed flip rates from an importance matrix."""
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
    """Predict flip rates using Gaussian CDF: Phi(-|Delta|/(sigma*sqrt(2)))."""
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


def get_pair_snr(imp_matrix):
    """Get per-pair |Delta|/sigma values from calibration importance matrix."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    snr = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            diffs = imp_matrix[:, j] - imp_matrix[:, k]
            delta = np.mean(diffs)
            sigma = np.std(diffs)
            if sigma > 1e-10:
                snr[pair_idx] = abs(delta) / sigma
            else:
                snr[pair_idx] = 10.0  # large SNR = no flip
            pair_idx += 1
    return snr


def r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


# ============================================================
# Load datasets
# ============================================================
print("Loading datasets...")
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

results = {}

# ============================================================
# EXPERIMENT 1: Linear baseline comparison (#2a)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Linear baseline comparison")
print("Does the Gaussian CDF shape add value over a linear fit?")
print("=" * 70)

exp1_results = {}

for ds_name, X, y, is_clf in datasets:
    P = X.shape[1]
    n_pairs = P * (P - 1) // 2
    print(f"\n--- {ds_name} (P={P}, {n_pairs} pairs) ---")

    # Train calibration and validation models
    cal_seeds = list(range(42, 72))   # 30 models
    val_seeds = list(range(142, 172)) # 30 models

    print(f"  Training {len(cal_seeds)} calibration models...")
    cal_imp = train_models_and_get_importance(X, y, is_clf, cal_seeds)
    print(f"  Training {len(val_seeds)} validation models...")
    val_imp = train_models_and_get_importance(X, y, is_clf, val_seeds)

    # Gaussian CDF prediction
    predicted_gauss = predict_flip_rates_gaussian(cal_imp)
    observed = measure_flip_rates(val_imp)
    r2_gauss = r_squared(observed, predicted_gauss)

    # Linear baseline: flip = a * |Delta|/sigma + b, fitted on calibration data
    cal_snr = get_pair_snr(cal_imp)
    cal_observed = measure_flip_rates(cal_imp)

    # Fit linear on calibration
    lin_reg = LinearRegression()
    lin_reg.fit(cal_snr.reshape(-1, 1), cal_observed)

    # Predict on validation using calibration SNR parameters
    # But we need validation's observed vs prediction from calibration SNR
    # The SNR comes from calibration (same as Gaussian), prediction goes to validation observed
    val_predicted_linear = lin_reg.predict(cal_snr.reshape(-1, 1))
    r2_linear = r_squared(observed, val_predicted_linear)

    # Marginal improvement
    improvement = r2_gauss - r2_linear

    exp1_results[ds_name] = {
        "P": int(P),
        "n_pairs": int(n_pairs),
        "gaussian_cdf_r2": round(float(r2_gauss), 4),
        "linear_baseline_r2": round(float(r2_linear), 4),
        "marginal_improvement": round(float(improvement), 4),
        "linear_slope": round(float(lin_reg.coef_[0]), 6),
        "linear_intercept": round(float(lin_reg.intercept_), 6),
    }
    print(f"  Gaussian CDF R² = {r2_gauss:.4f}")
    print(f"  Linear baseline R² = {r2_linear:.4f}")
    print(f"  Marginal improvement = {improvement:.4f}")

results["experiment_1_linear_baseline"] = exp1_results

print("\n  SUMMARY: Linear baseline comparison")
print(f"  {'Dataset':<16} {'Gauss R²':>10} {'Linear R²':>10} {'Marginal':>10}")
print(f"  {'-'*50}")
for ds, info in exp1_results.items():
    print(f"  {ds:<16} {info['gaussian_cdf_r2']:>10.4f} {info['linear_baseline_r2']:>10.4f} {info['marginal_improvement']:>10.4f}")


# ============================================================
# EXPERIMENT 2: Gaussianity diagnostics (#26, #2b)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Gaussianity diagnostics for Breast Cancer")
print("Shapiro-Wilk, skewness, kurtosis, Q-Q plots on importance differences")
print("=" * 70)

bc_X, bc_y = load_breast_cancer(return_X_y=True)
P_bc = bc_X.shape[1]

# Train 30 models (seeds 42-71)
print("Training 30 XGBoost models (seeds 42-71)...")
bc_seeds = list(range(42, 72))
bc_imp = train_models_and_get_importance(bc_X, bc_y, True, bc_seeds)

# Compute flip rates for all pairs to select representative pairs
print("Computing flip rates for all 435 pairs...")
all_flip_rates = {}
pair_diffs = {}
pair_idx = 0
for j in range(P_bc):
    for k in range(j + 1, P_bc):
        diffs = bc_imp[:, j] - bc_imp[:, k]
        delta = np.mean(diffs)
        sigma = np.std(diffs)
        if sigma > 1e-10:
            flip_pred = norm.cdf(-abs(delta) / (sigma * np.sqrt(2)))
        else:
            flip_pred = 0.0
        all_flip_rates[(j, k)] = flip_pred
        pair_diffs[(j, k)] = diffs
        pair_idx += 1

# Sort pairs by predicted flip rate and select 10 representative ones
sorted_pairs = sorted(all_flip_rates.items(), key=lambda x: x[1])
n_total = len(sorted_pairs)

# Select 10 pairs spanning high/medium/low flip rates
# Pick from quantiles: 5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95%
quantile_indices = [int(q * n_total) for q in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]]
quantile_indices = [min(i, n_total - 1) for i in quantile_indices]
selected_pairs = [sorted_pairs[i] for i in quantile_indices]

print(f"\nSelected 10 representative pairs (spanning flip rate range):")
exp2_pairs = []
shapiro_results = []
skew_values = []
kurt_values = []

for (j, k), flip_rate in selected_pairs:
    diffs = pair_diffs[(j, k)]
    stat, p_val = shapiro(diffs)
    sk = skew(diffs)
    kt = kurtosis(diffs)  # excess kurtosis (0 for normal)

    pair_info = {
        "feature_j": int(j),
        "feature_k": int(k),
        "predicted_flip_rate": round(float(flip_rate), 4),
        "shapiro_stat": round(float(stat), 4),
        "shapiro_p": round(float(p_val), 4),
        "shapiro_pass": bool(p_val > 0.05),
        "skewness": round(float(sk), 4),
        "kurtosis": round(float(kt), 4),
    }
    exp2_pairs.append(pair_info)
    shapiro_results.append(p_val > 0.05)
    skew_values.append(sk)
    kurt_values.append(kt)

    print(f"  Pair ({j},{k}): flip={flip_rate:.3f}  Shapiro p={p_val:.4f}{'*' if p_val>0.05 else ''}  "
          f"skew={sk:.3f}  kurt={kt:.3f}")

frac_passing = sum(shapiro_results) / len(shapiro_results)
mean_skew = np.mean(np.abs(skew_values))
mean_kurt = np.mean(kurt_values)

exp2_summary = {
    "n_models": 30,
    "n_features": int(P_bc),
    "n_pairs_tested": 10,
    "fraction_passing_shapiro": round(float(frac_passing), 2),
    "mean_abs_skewness": round(float(mean_skew), 4),
    "mean_excess_kurtosis": round(float(mean_kurt), 4),
    "pairs": exp2_pairs,
}

print(f"\n  Fraction passing Shapiro-Wilk (p>0.05): {frac_passing:.0%}")
print(f"  Mean |skewness|: {mean_skew:.4f}")
print(f"  Mean excess kurtosis: {mean_kurt:.4f}")

results["experiment_2_gaussianity"] = exp2_summary

# Q-Q plot: 2x5 grid
fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
axes_flat = axes.flatten()

for idx, ((j, k), flip_rate) in enumerate(selected_pairs):
    ax = axes_flat[idx]
    diffs = pair_diffs[(j, k)]
    n = len(diffs)

    # Q-Q: sort data, compute theoretical quantiles
    sorted_diffs = np.sort(diffs)
    theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    # Standardize for comparison
    mu, sig = np.mean(diffs), np.std(diffs)
    standardized = (sorted_diffs - mu) / sig if sig > 1e-10 else sorted_diffs

    ax.scatter(theoretical_quantiles, standardized, s=15, alpha=0.7, c='steelblue', edgecolors='none')
    lims = [min(theoretical_quantiles.min(), standardized.min()) - 0.2,
            max(theoretical_quantiles.max(), standardized.max()) + 0.2]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('Theoretical quantiles', fontsize=8)
    ax.set_ylabel('Sample quantiles', fontsize=8)

    p_info = exp2_pairs[idx]
    ax.set_title(f"({j},{k}) flip={flip_rate:.3f}\nSW p={p_info['shapiro_p']:.3f}", fontsize=9)
    ax.tick_params(labelsize=7)

fig.suptitle('Q-Q Plots: Importance Difference Distributions (Breast Cancer, M=30)\n'
             '10 representative feature pairs spanning low to high flip rates',
             fontsize=12, fontweight='bold')
fig.savefig(FIG_DIR / 'gaussianity_qq.pdf', dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {FIG_DIR / 'gaussianity_qq.pdf'}")


# ============================================================
# EXPERIMENT 3: M sensitivity (#25)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: M sensitivity for Breast Cancer")
print("OOS R² at M in {5, 10, 15, 20, 30, 50}")
print("=" * 70)

M_values = [5, 10, 15, 20, 30, 50]
exp3_results = {}

for M in M_values:
    n_cal = M // 2
    n_val = M - n_cal
    cal_seeds_m = list(range(42, 42 + n_cal))
    val_seeds_m = list(range(142, 142 + n_val))

    print(f"\n  M={M}: {n_cal} calibration + {n_val} validation models")
    print(f"    Training calibration models (seeds {cal_seeds_m[0]}-{cal_seeds_m[-1]})...")
    cal_imp_m = train_models_and_get_importance(bc_X, bc_y, True, cal_seeds_m)

    print(f"    Training validation models (seeds {val_seeds_m[0]}-{val_seeds_m[-1]})...")
    val_imp_m = train_models_and_get_importance(bc_X, bc_y, True, val_seeds_m)

    predicted_m = predict_flip_rates_gaussian(cal_imp_m)
    observed_m = measure_flip_rates(val_imp_m)
    r2_m = r_squared(observed_m, predicted_m)
    rmse_m = np.sqrt(np.mean((observed_m - predicted_m) ** 2))

    exp3_results[str(M)] = {
        "M": M,
        "n_cal": n_cal,
        "n_val": n_val,
        "oos_r2": round(float(r2_m), 4),
        "oos_rmse": round(float(rmse_m), 6),
    }
    print(f"    OOS R² = {r2_m:.4f}  |  RMSE = {rmse_m:.6f}")

# Find minimum M for R² > 0.7
min_m_for_07 = None
for M in M_values:
    if exp3_results[str(M)]["oos_r2"] > 0.7:
        min_m_for_07 = M
        break

exp3_summary = {
    "dataset": "Breast Cancer",
    "P": int(P_bc),
    "M_values": M_values,
    "results_by_M": exp3_results,
    "minimum_M_for_r2_07": min_m_for_07,
}

results["experiment_3_m_sensitivity"] = exp3_summary

print(f"\n  Minimum M for R² > 0.7: {min_m_for_07}")

# Figure: R² vs M
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ms = [exp3_results[str(M)]["M"] for M in M_values]
r2s = [exp3_results[str(M)]["oos_r2"] for M in M_values]

ax.plot(ms, r2s, 'o-', color='steelblue', markersize=8, linewidth=2)
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='R² = 0.7 threshold')
ax.set_xlabel('Total models M (M/2 cal + M/2 val)', fontsize=12)
ax.set_ylabel('Out-of-sample R²', fontsize=12)
ax.set_title('M Sensitivity: Breast Cancer (P=30, 435 pairs)', fontsize=13, fontweight='bold')
ax.set_xticks(M_values)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.0])

for m_val, r2_val in zip(ms, r2s):
    ax.annotate(f'{r2_val:.3f}', (m_val, r2_val), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9)

fig.savefig(FIG_DIR / 'm_sensitivity.pdf', dpi=150, bbox_inches='tight')
print(f"Figure saved to {FIG_DIR / 'm_sensitivity.pdf'}")


# ============================================================
# EXPERIMENT 4: Leave-one-out for η regression (#31)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Leave-one-out for η regression")
print("7 well-characterized points from results_gaussian_eta_rescue.json")
print("=" * 70)

# Load the eta rescue data
with open(OUT_DIR / 'results_gaussian_eta_rescue.json') as f:
    eta_data = json.load(f)

# Identify the 7 well-characterized points
well_characterized = []
for pt in eta_data["points"]:
    if pt["correction"] in ["none (well-characterized)", "gaussian_corrected"]:
        # These are the ones with reliable predictions
        pass
    # Actually: the rescue file marks "well-characterized" explicitly
    # Let's collect them properly

# The well-characterized points are those marked as such or gaussian_corrected
# From the rescue file: Attribution, Concept probe, Codon S2/S4/S6, Stat mech, Parser, Gauge, Phase retrieval
# But stat mech is analytical (exact), so it's well-characterized
# Let's pick the 7 that are well-characterized based on the rescue verdict
well_char_labels = set()
for pt in eta_data["points"]:
    corr = pt["correction"]
    if "well-characterized" in corr or "gaussian_corrected" in corr:
        well_char_labels.add(pt["domain"])

print(f"  Well-characterized points ({len(well_char_labels)}):")
well_char_points = []
for pt in eta_data["points"]:
    if pt["domain"] in well_char_labels:
        well_char_points.append(pt)
        print(f"    {pt['domain']}: pred={pt['predicted_instability']:.4f}, obs={pt['observed_instability']:.4f}")

n_wc = len(well_char_points)
preds = np.array([p["predicted_instability"] for p in well_char_points])
obs = np.array([p["observed_instability"] for p in well_char_points])

# Full regression
from sklearn.linear_model import LinearRegression as LR
reg_full = LR()
reg_full.fit(preds.reshape(-1, 1), obs)
full_pred = reg_full.predict(preds.reshape(-1, 1))
full_r2 = r_squared(obs, full_pred)
print(f"\n  Full regression R² ({n_wc} points): {full_r2:.4f}")

# LOO
loo_predictions = np.zeros(n_wc)
loo_r2_drops = []

for i in range(n_wc):
    # Hold out point i
    mask = np.ones(n_wc, dtype=bool)
    mask[i] = False

    reg_loo = LR()
    reg_loo.fit(preds[mask].reshape(-1, 1), obs[mask])
    loo_predictions[i] = reg_loo.predict(preds[i:i+1].reshape(-1, 1))[0]

    # R² on remaining n-1 points
    train_pred = reg_loo.predict(preds[mask].reshape(-1, 1))
    train_r2 = r_squared(obs[mask], train_pred)
    loo_r2_drops.append({
        "held_out": well_char_points[i]["domain"],
        "predicted": round(float(preds[i]), 4),
        "observed": round(float(obs[i]), 4),
        "loo_predicted": round(float(loo_predictions[i]), 4),
        "residual": round(float(obs[i] - loo_predictions[i]), 4),
        "train_r2_without": round(float(train_r2), 4),
    })

loo_r2 = r_squared(obs, loo_predictions)

# Find worst point
worst_idx = np.argmax(np.abs(obs - loo_predictions))
worst_point = well_char_points[worst_idx]["domain"]

exp4_results = {
    "n_points": n_wc,
    "full_r2": round(float(full_r2), 4),
    "loo_r2": round(float(loo_r2), 4),
    "worst_point": worst_point,
    "worst_residual": round(float(obs[worst_idx] - loo_predictions[worst_idx]), 4),
    "loo_details": loo_r2_drops,
}

results["experiment_4_loo_eta"] = exp4_results

print(f"  LOO R²: {loo_r2:.4f}")
print(f"  Worst point: {worst_point} (residual = {obs[worst_idx] - loo_predictions[worst_idx]:.4f})")

print(f"\n  {'Point':<45} {'Pred':>6} {'Obs':>6} {'LOO':>6} {'Resid':>7}")
print(f"  {'-'*75}")
for info in loo_r2_drops:
    print(f"  {info['held_out']:<45} {info['predicted']:>6.3f} {info['observed']:>6.3f} "
          f"{info['loo_predicted']:>6.3f} {info['residual']:>7.4f}")


# ============================================================
# EXPERIMENT 5: Per-pair to top-k connection (#30)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Per-pair to top-k connection (Breast Cancer)")
print("Top-5 agreement across 30 validation models")
print("=" * 70)

# Use the validation models from Experiment 1 (already trained seeds 142-171)
# Re-train for consistency (we need the importance matrix)
print("Training 30 validation models (seeds 142-171)...")
val_seeds_5 = list(range(142, 172))
val_imp_5 = train_models_and_get_importance(bc_X, bc_y, True, val_seeds_5)

n_val_models = val_imp_5.shape[0]
K = 5

# For each model, compute top-5 features
top_k_per_model = []
for i in range(n_val_models):
    top_k = set(np.argsort(val_imp_5[i])[-K:][::-1].tolist())
    top_k_ordered = tuple(np.argsort(val_imp_5[i])[-K:][::-1].tolist())
    top_k_per_model.append((top_k, top_k_ordered))

# Pairwise agreement: exact ordered top-5
n_pairs_exact = 0
n_pairs_set = 0
n_total_pairs = 0

for a in range(n_val_models):
    for b in range(a + 1, n_val_models):
        n_total_pairs += 1
        set_a, ord_a = top_k_per_model[a]
        set_b, ord_b = top_k_per_model[b]

        if ord_a == ord_b:
            n_pairs_exact += 1
        if set_a == set_b:
            n_pairs_set += 1

frac_exact = n_pairs_exact / n_total_pairs
frac_set = n_pairs_set / n_total_pairs

print(f"\n  Total model pairs: {n_total_pairs}")
print(f"  Exact top-5 agreement (ordered): {n_pairs_exact}/{n_total_pairs} = {frac_exact:.4f}")
print(f"  Set top-5 agreement (unordered): {n_pairs_set}/{n_total_pairs} = {frac_set:.4f}")

# Compute the Gaussian-predicted flip rates for top features
# This connects per-pair flips to top-k stability
print("\n  Connecting per-pair flips to top-k stability:")

# Use calibration importance to predict which pairs involving top features flip
cal_seeds_5 = list(range(42, 72))
cal_imp_5 = train_models_and_get_importance(bc_X, bc_y, True, cal_seeds_5)

# Identify consensus top-5 from calibration
mean_imp_cal = np.mean(cal_imp_5, axis=0)
consensus_top5 = set(np.argsort(mean_imp_cal)[-K:][::-1].tolist())
print(f"  Calibration consensus top-5: {sorted(consensus_top5)}")

# For each top-5 feature vs the 6th-ranked feature, compute predicted flip rate
sorted_features = np.argsort(mean_imp_cal)[::-1]
boundary_pairs_flip = []
sixth_feature = sorted_features[K]

for feat in sorted(consensus_top5):
    j, k = min(feat, sixth_feature), max(feat, sixth_feature)
    diffs = cal_imp_5[:, j] - cal_imp_5[:, k]
    delta = np.mean(diffs)
    sigma = np.std(diffs)
    if sigma > 1e-10:
        flip = norm.cdf(-abs(delta) / (sigma * np.sqrt(2)))
    else:
        flip = 0.0
    boundary_pairs_flip.append({
        "top_feature": int(feat),
        "boundary_feature": int(sixth_feature),
        "predicted_flip": round(float(flip), 4),
        "snr": round(float(abs(delta)/sigma if sigma > 1e-10 else 999), 4),
    })
    print(f"    Feature {feat} vs boundary feature {sixth_feature}: predicted flip = {flip:.4f}, SNR = {abs(delta)/sigma:.4f}" if sigma > 1e-10 else f"    Feature {feat} vs {sixth_feature}: no variance")

# Theoretical top-k stability from per-pair flips
# Top-k set is stable iff none of the K boundary pairs flip
# Under independence approximation: P(stable) ≈ prod(1 - flip_i)
boundary_flips = [bp["predicted_flip"] for bp in boundary_pairs_flip]
predicted_stability = np.prod([1 - f for f in boundary_flips])
print(f"\n  Predicted top-5 set stability (independence approx): {predicted_stability:.4f}")
print(f"  Observed top-5 set agreement fraction: {frac_set:.4f}")
print(f"  Note: The approximation is a lower bound; actual stability can be higher")
print(f"        because boundary flips are correlated (a flip in one pair makes")
print(f"        another less likely if the features are correlated)")

exp5_results = {
    "dataset": "Breast Cancer",
    "P": int(P_bc),
    "K": K,
    "n_validation_models": n_val_models,
    "n_model_pairs": n_total_pairs,
    "exact_top5_agreement": round(float(frac_exact), 4),
    "set_top5_agreement": round(float(frac_set), 4),
    "consensus_top5_features": sorted(list(consensus_top5)),
    "boundary_feature": int(sixth_feature),
    "boundary_pairs": boundary_pairs_flip,
    "predicted_set_stability_independence": round(float(predicted_stability), 4),
    "connection": "Per-pair Gaussian flip rates at the top-k boundary determine top-k stability. "
                  "Under independence: P(top-k stable) = prod(1 - flip_boundary_pair). "
                  "The Gaussian formula provides the mechanistic link between per-pair SNR and aggregate top-k agreement."
}

results["experiment_5_topk_connection"] = exp5_results


# ============================================================
# Save all results
# ============================================================
with open(OUT_DIR / 'results_gaussian_diagnostics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("ALL RESULTS SAVED")
print(f"  Results: {OUT_DIR / 'results_gaussian_diagnostics.json'}")
print(f"  Figure:  {FIG_DIR / 'gaussianity_qq.pdf'}")
print(f"  Figure:  {FIG_DIR / 'm_sensitivity.pdf'}")
print("=" * 70)

# Final summary
print("\n" + "=" * 70)
print("SUMMARY OF ALL 5 EXPERIMENTS")
print("=" * 70)

print("\n1. LINEAR BASELINE (#2a):")
for ds, info in exp1_results.items():
    print(f"   {ds}: Gaussian R²={info['gaussian_cdf_r2']:.4f}, Linear R²={info['linear_baseline_r2']:.4f}, "
          f"Marginal={info['marginal_improvement']:.4f}")

print(f"\n2. GAUSSIANITY (#26, #2b):")
print(f"   Shapiro-Wilk pass rate: {frac_passing:.0%} ({sum(shapiro_results)}/{len(shapiro_results)} pairs)")
print(f"   Mean |skewness|: {mean_skew:.4f}")
print(f"   Mean excess kurtosis: {mean_kurt:.4f}")

print(f"\n3. M SENSITIVITY (#25):")
for M in M_values:
    r2_val = exp3_results[str(M)]["oos_r2"]
    marker = " <-- minimum for R²>0.7" if M == min_m_for_07 else ""
    print(f"   M={M:3d}: R²={r2_val:.4f}{marker}")

print(f"\n4. LOO η REGRESSION (#31):")
print(f"   Full R²: {full_r2:.4f}")
print(f"   LOO R²: {loo_r2:.4f}")
print(f"   Worst point: {worst_point}")

print(f"\n5. TOP-K CONNECTION (#30):")
print(f"   Exact top-5 agreement: {frac_exact:.4f}")
print(f"   Set top-5 agreement: {frac_set:.4f}")
print(f"   Predicted stability (independence): {predicted_stability:.4f}")
