#!/usr/bin/env python3
"""
Bootstrap Calibration: Does Phi(-SNR) predict actual flip rates?

For the 5 validated datasets (Breast Cancer, Wine, California Housing,
Diabetes Pima, Heart Disease):

1. Extract per-pair: (predicted_flip_rate, observed_flip_rate)
2. Calibration plot: bin predicted into deciles, compute mean observed per bin
3. Per-dataset: R^2, MAE, calibration slope (ideal=1), intercept (ideal=0)
4. Aggregate across all ~647 pairs: R^2, calibration slope, Hosmer-Lemeshow test
5. By collinearity bin: low (|rho|<0.3), medium (0.3-0.7), high (>0.7)
6. Bootstrap 95% CIs on all metrics (1000 resamples)

Uses independent calibration (seeds 42-71) and validation (seeds 142-171) models,
matching the validated experimental design.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import norm, pearsonr, chi2
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from pathlib import Path

OUT_DIR = Path(__file__).parent

N_BOOTSTRAP = 1000
SEED = 42

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
    """Compute pairwise observed flip rates."""
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
    """Predict flip rates using Gaussian CDF formula from calibration data."""
    n_models, P = imp_matrix.shape
    n_pairs_total = P * (P - 1) // 2
    predicted = np.zeros(n_pairs_total)
    snr_values = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            diffs = imp_matrix[:, j] - imp_matrix[:, k]
            delta = np.mean(diffs)
            sigma = np.std(diffs)
            if sigma > 1e-10:
                snr = abs(delta) / (sigma * np.sqrt(2))
                predicted[pair_idx] = norm.cdf(-snr)
                snr_values[pair_idx] = snr
            else:
                predicted[pair_idx] = 0.0
                snr_values[pair_idx] = float('inf')
            pair_idx += 1
    return predicted, snr_values


def compute_pair_correlations(X):
    """Compute |Pearson r| for each feature pair."""
    P = X.shape[1]
    n_pairs_total = P * (P - 1) // 2
    corrs = np.zeros(n_pairs_total)
    pair_idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            r, _ = pearsonr(X[:, j], X[:, k])
            corrs[pair_idx] = abs(r) if np.isfinite(r) else 0.0
            pair_idx += 1
    return corrs


def r_squared(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))


def calibration_slope_intercept(obs, pred):
    """OLS: obs = intercept + slope * pred. Ideal: slope=1, intercept=0."""
    if len(obs) < 2 or np.std(pred) < 1e-15:
        return float('nan'), float('nan')
    reg = LinearRegression()
    reg.fit(pred.reshape(-1, 1), obs)
    return float(reg.coef_[0]), float(reg.intercept_)


def hosmer_lemeshow(obs, pred, n_bins=10):
    """Hosmer-Lemeshow-style calibration test for continuous outcomes.
    Bin predicted into deciles, compare mean observed vs mean predicted per bin.
    Returns chi-square statistic and p-value."""
    if len(obs) < n_bins:
        return float('nan'), float('nan')

    # Use quantile-based bins
    try:
        bin_edges = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10  # ensure max is included
        bin_indices = np.digitize(pred, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    except Exception:
        return float('nan'), float('nan')

    chi2_stat = 0.0
    n_valid_bins = 0
    for b in range(n_bins):
        mask = bin_indices == b
        n_b = np.sum(mask)
        if n_b == 0:
            continue
        obs_mean = np.mean(obs[mask])
        pred_mean = np.mean(pred[mask])
        if pred_mean > 1e-10 and (1 - pred_mean) > 1e-10:
            # Variance of proportion ~ p*(1-p)/n
            var_est = pred_mean * (1 - pred_mean) / n_b
            if var_est > 1e-15:
                chi2_stat += (obs_mean - pred_mean) ** 2 / var_est
                n_valid_bins += 1

    df = max(n_valid_bins - 2, 1)  # subtract 2 for estimated parameters
    p_val = 1 - chi2.cdf(chi2_stat, df)
    return float(chi2_stat), float(p_val)


def calibration_bins(obs, pred, n_bins=10):
    """Compute calibration curve: binned means."""
    if len(obs) < n_bins:
        return [], []
    try:
        bin_edges = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10
        bin_indices = np.digitize(pred, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    except Exception:
        return [], []

    bin_pred_means = []
    bin_obs_means = []
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_pred_means.append(float(np.mean(pred[mask])))
            bin_obs_means.append(float(np.mean(obs[mask])))
    return bin_pred_means, bin_obs_means


def bootstrap_metric(obs, pred, metric_fn, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(obs)
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        val = metric_fn(obs[idx], pred[idx])
        if np.isfinite(val):
            values.append(val)
    if len(values) < 10:
        return float('nan'), float('nan'), float('nan')
    values = np.array(values)
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---- Main experiment ----
print("=" * 75)
print("EXPERIMENT 4: Bootstrap Calibration of Phi(-SNR) Flip Rate Formula")
print("=" * 75)

results = {
    "experiment": "Bootstrap Calibration of Gaussian Flip Rate Formula",
    "design": "Independent calibration (seeds 42-71) and validation (seeds 142-171)",
    "n_bootstrap": N_BOOTSTRAP,
    "datasets": {},
    "aggregate": {},
    "by_collinearity": {},
}

# Collect all pairs across datasets
all_predicted = []
all_observed = []
all_corrs = []
all_snrs = []
all_dataset_labels = []

for ds_name, X, y, is_clf in datasets:
    P = X.shape[1]
    n_pairs = P * (P - 1) // 2
    print(f"\n--- {ds_name} (P={P}, {n_pairs} pairs) ---")

    # Train models
    cal_seeds = list(range(42, 72))  # 30 models
    val_seeds = list(range(142, 172))  # 30 models

    print(f"  Training {len(cal_seeds)} calibration models...")
    cal_imp = train_models_and_get_importance(X, y, is_clf, cal_seeds)

    print(f"  Training {len(val_seeds)} validation models...")
    val_imp = train_models_and_get_importance(X, y, is_clf, val_seeds)

    # Predictions from calibration
    predicted, snr_vals = predict_flip_rates_gaussian(cal_imp)

    # Observations from validation
    observed = measure_flip_rates(val_imp)

    # Feature correlations
    corrs = compute_pair_correlations(X)

    # Per-dataset metrics
    ds_r2 = r_squared(observed, predicted)
    ds_mae = mae(observed, predicted)
    slope, intercept = calibration_slope_intercept(observed, predicted)
    hl_stat, hl_p = hosmer_lemeshow(observed, predicted)
    bin_pred, bin_obs = calibration_bins(observed, predicted)

    # Bootstrap CIs
    print("  Bootstrapping R^2...")
    r2_mean, r2_lo, r2_hi = bootstrap_metric(observed, predicted, r_squared)
    print("  Bootstrapping MAE...")
    mae_mean, mae_lo, mae_hi = bootstrap_metric(observed, predicted, mae)
    print("  Bootstrapping slope...")

    def slope_fn(o, p):
        s, _ = calibration_slope_intercept(o, p)
        return s

    def intercept_fn(o, p):
        _, i = calibration_slope_intercept(o, p)
        return i

    slope_mean, slope_lo, slope_hi = bootstrap_metric(observed, predicted, slope_fn)
    intercept_mean, intercept_lo, intercept_hi = bootstrap_metric(observed, predicted, intercept_fn)

    ds_result = {
        "P": P,
        "n_pairs": n_pairs,
        "r2": round(ds_r2, 4),
        "r2_bootstrap_ci": [round(r2_lo, 4), round(r2_hi, 4)],
        "mae": round(ds_mae, 6),
        "mae_bootstrap_ci": [round(mae_lo, 6), round(mae_hi, 6)],
        "calibration_slope": round(slope, 4),
        "calibration_slope_bootstrap_ci": [round(slope_lo, 4), round(slope_hi, 4)],
        "calibration_intercept": round(intercept, 6),
        "calibration_intercept_bootstrap_ci": [round(intercept_lo, 6), round(intercept_hi, 6)],
        "hosmer_lemeshow_chi2": round(hl_stat, 4),
        "hosmer_lemeshow_p": round(hl_p, 4),
        "calibration_curve": {
            "bin_predicted_mean": [round(x, 6) for x in bin_pred],
            "bin_observed_mean": [round(x, 6) for x in bin_obs],
        },
        "mean_snr": round(float(np.mean(snr_vals[np.isfinite(snr_vals)])), 4),
        "mean_abs_correlation": round(float(np.mean(corrs)), 4),
    }

    results["datasets"][ds_name] = ds_result

    print(f"  R^2 = {ds_r2:.4f} [{r2_lo:.4f}, {r2_hi:.4f}]")
    print(f"  MAE = {ds_mae:.6f} [{mae_lo:.6f}, {mae_hi:.6f}]")
    print(f"  Slope = {slope:.4f} [{slope_lo:.4f}, {slope_hi:.4f}] (ideal=1)")
    print(f"  Intercept = {intercept:.6f} [{intercept_lo:.6f}, {intercept_hi:.6f}] (ideal=0)")
    print(f"  Hosmer-Lemeshow: chi2={hl_stat:.4f}, p={hl_p:.4f}")

    # Accumulate for aggregate
    all_predicted.extend(predicted.tolist())
    all_observed.extend(observed.tolist())
    all_corrs.extend(corrs.tolist())
    all_snrs.extend(snr_vals.tolist())
    all_dataset_labels.extend([ds_name] * n_pairs)

# ---- Aggregate analysis ----
print(f"\n{'='*75}")
print(f"AGGREGATE ANALYSIS ({len(all_predicted)} total pairs)")
print(f"{'='*75}")

all_predicted = np.array(all_predicted)
all_observed = np.array(all_observed)
all_corrs = np.array(all_corrs)
all_snrs = np.array(all_snrs)

agg_r2 = r_squared(all_observed, all_predicted)
agg_mae = mae(all_observed, all_predicted)
agg_slope, agg_intercept = calibration_slope_intercept(all_observed, all_predicted)
agg_hl_stat, agg_hl_p = hosmer_lemeshow(all_observed, all_predicted)
agg_bin_pred, agg_bin_obs = calibration_bins(all_observed, all_predicted)

# Bootstrap aggregate
print("  Bootstrapping aggregate R^2...")
agg_r2_mean, agg_r2_lo, agg_r2_hi = bootstrap_metric(all_observed, all_predicted, r_squared)
print("  Bootstrapping aggregate MAE...")
agg_mae_mean, agg_mae_lo, agg_mae_hi = bootstrap_metric(all_observed, all_predicted, mae)
print("  Bootstrapping aggregate slope...")
agg_slope_mean, agg_slope_lo, agg_slope_hi = bootstrap_metric(all_observed, all_predicted, slope_fn)
agg_int_mean, agg_int_lo, agg_int_hi = bootstrap_metric(all_observed, all_predicted, intercept_fn)

results["aggregate"] = {
    "n_total_pairs": len(all_predicted),
    "r2": round(agg_r2, 4),
    "r2_bootstrap_ci": [round(agg_r2_lo, 4), round(agg_r2_hi, 4)],
    "mae": round(agg_mae, 6),
    "mae_bootstrap_ci": [round(agg_mae_lo, 6), round(agg_mae_hi, 6)],
    "calibration_slope": round(agg_slope, 4),
    "calibration_slope_bootstrap_ci": [round(agg_slope_lo, 4), round(agg_slope_hi, 4)],
    "calibration_intercept": round(agg_intercept, 6),
    "calibration_intercept_bootstrap_ci": [round(agg_int_lo, 6), round(agg_int_hi, 6)],
    "hosmer_lemeshow_chi2": round(agg_hl_stat, 4),
    "hosmer_lemeshow_p": round(agg_hl_p, 4),
    "calibration_curve": {
        "bin_predicted_mean": [round(x, 6) for x in agg_bin_pred],
        "bin_observed_mean": [round(x, 6) for x in agg_bin_obs],
    },
}

print(f"\n  Aggregate R^2 = {agg_r2:.4f} [{agg_r2_lo:.4f}, {agg_r2_hi:.4f}]")
print(f"  Aggregate MAE = {agg_mae:.6f} [{agg_mae_lo:.6f}, {agg_mae_hi:.6f}]")
print(f"  Aggregate Slope = {agg_slope:.4f} [{agg_slope_lo:.4f}, {agg_slope_hi:.4f}]")
print(f"  Aggregate Intercept = {agg_intercept:.6f} [{agg_int_lo:.6f}, {agg_int_hi:.6f}]")
print(f"  Hosmer-Lemeshow: chi2={agg_hl_stat:.4f}, p={agg_hl_p:.4f}")

# ---- By collinearity bin ----
print(f"\n{'='*75}")
print("BY COLLINEARITY BIN")
print(f"{'='*75}")

collinearity_bins = {
    "low (|rho|<0.3)": all_corrs < 0.3,
    "medium (0.3<=|rho|<0.7)": (all_corrs >= 0.3) & (all_corrs < 0.7),
    "high (|rho|>=0.7)": all_corrs >= 0.7,
}

for bin_name, mask in collinearity_bins.items():
    n_in_bin = int(np.sum(mask))
    if n_in_bin < 5:
        print(f"\n  {bin_name}: {n_in_bin} pairs (too few for analysis)")
        results["by_collinearity"][bin_name] = {"n_pairs": n_in_bin, "note": "too few pairs"}
        continue

    obs_bin = all_observed[mask]
    pred_bin = all_predicted[mask]

    bin_r2 = r_squared(obs_bin, pred_bin)
    bin_mae_val = mae(obs_bin, pred_bin)
    bin_slope, bin_intercept = calibration_slope_intercept(obs_bin, pred_bin)

    # Bootstrap
    bin_r2_mean, bin_r2_lo, bin_r2_hi = bootstrap_metric(obs_bin, pred_bin, r_squared)
    bin_mae_mean, bin_mae_lo, bin_mae_hi = bootstrap_metric(obs_bin, pred_bin, mae)
    bin_slope_mean, bin_slope_lo, bin_slope_hi = bootstrap_metric(obs_bin, pred_bin, slope_fn)

    results["by_collinearity"][bin_name] = {
        "n_pairs": n_in_bin,
        "r2": round(bin_r2, 4),
        "r2_bootstrap_ci": [round(bin_r2_lo, 4), round(bin_r2_hi, 4)],
        "mae": round(bin_mae_val, 6),
        "mae_bootstrap_ci": [round(bin_mae_lo, 6), round(bin_mae_hi, 6)],
        "calibration_slope": round(bin_slope, 4),
        "calibration_slope_bootstrap_ci": [round(bin_slope_lo, 4), round(bin_slope_hi, 4)],
        "calibration_intercept": round(bin_intercept, 6),
        "mean_observed_flip": round(float(np.mean(obs_bin)), 6),
        "mean_predicted_flip": round(float(np.mean(pred_bin)), 6),
    }

    print(f"\n  {bin_name}: {n_in_bin} pairs")
    print(f"    R^2 = {bin_r2:.4f} [{bin_r2_lo:.4f}, {bin_r2_hi:.4f}]")
    print(f"    MAE = {bin_mae_val:.6f} [{bin_mae_lo:.6f}, {bin_mae_hi:.6f}]")
    print(f"    Slope = {bin_slope:.4f} [{bin_slope_lo:.4f}, {bin_slope_hi:.4f}]")
    print(f"    Mean obs flip: {np.mean(obs_bin):.4f}, Mean pred flip: {np.mean(pred_bin):.4f}")

# ---- Final verdict ----
print(f"\n{'='*75}")
print("VERDICT")
print(f"{'='*75}")

well_calibrated = (
    agg_r2 > 0.7
    and agg_slope_lo < 1.0 < agg_slope_hi  # 95% CI contains 1
    and abs(agg_intercept) < 0.05
)

if well_calibrated:
    verdict = (
        f"WELL CALIBRATED: Aggregate R^2={agg_r2:.4f}, "
        f"slope CI [{agg_slope_lo:.3f}, {agg_slope_hi:.3f}] contains 1.0, "
        f"|intercept|={abs(agg_intercept):.4f}<0.05"
    )
else:
    issues = []
    if agg_r2 <= 0.7:
        issues.append(f"R^2={agg_r2:.4f}<=0.7")
    if not (agg_slope_lo < 1.0 < agg_slope_hi):
        issues.append(f"slope CI [{agg_slope_lo:.3f}, {agg_slope_hi:.3f}] excludes 1.0")
    if abs(agg_intercept) >= 0.05:
        issues.append(f"|intercept|={abs(agg_intercept):.4f}>=0.05")
    verdict = f"CALIBRATION ISSUES: {'; '.join(issues)}"

results["verdict"] = verdict
print(f"\n  {verdict}")

# Save results
out_path = OUT_DIR / "results_bootstrap_calibration.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
