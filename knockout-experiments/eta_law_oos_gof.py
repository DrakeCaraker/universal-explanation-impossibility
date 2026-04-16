#!/usr/bin/env python3
"""
Eta Law: Leave-One-Out Cross-Validation + Goodness of Fit

1. LOO-CV: For each of 7 folds, fit linear regression on 6, predict 7th.
   Report: LOO R^2, LOO MAE, per-fold prediction vs actual.

2. GoF: F-test for linear regression significance.
   Report: F(1,5), p-value, adjusted R^2.

3. Residual diagnostics: Shapiro-Wilk normality, Cook's distance for
   influential points, leverage values.

4. Permutation null: 10000 random permutations of eta values.
   Report: p-value (fraction with R^2 >= observed R^2).

5. Prediction for 8th instance: Use all 7 to predict held-out instances
   from the 16-point dataset.

Data source: results_universal_eta.json (16 points total, 7 well-characterized)
"""

import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────
with open(BASE / "results_universal_eta.json") as f:
    eta_data = json.load(f)

all_points = eta_data["points"]

# The 7 well-characterized instances (from the user specification)
well_characterized_names = [
    "Attribution (SHAP, $S_2$)",
    "Attention (argmax, $S_6$)",
    "Counterfactual (direction, $\\mathbb{Z}_2$)",
    "Concept probe (TCAV, $O(64)$)",
    "Model selection ($S_{11}$ winners)",
    "Codon ($S_2$)",
    "Stat mech ($S_{252}$, $N$=10)",
]

# Extract data
core_7 = []
for name in well_characterized_names:
    for pt in all_points:
        if pt["domain"] == name:
            core_7.append(pt)
            break

assert len(core_7) == 7, f"Expected 7 core points, found {len(core_7)}"

# Also collect holdout points (the other 9)
holdout_points = [pt for pt in all_points if pt["domain"] not in well_characterized_names]

print("=" * 80)
print("ETA LAW: OUT-OF-SAMPLE + GOODNESS OF FIT")
print("=" * 80)

# Print core data
eta_vals = np.array([pt["predicted_instability"] for pt in core_7])
obs_vals = np.array([pt["observed_instability"] for pt in core_7])

print(f"\nCore 7 instances:")
print(f"{'Domain':45s} {'eta (pred)':>12s} {'obs':>12s}")
print("─" * 70)
for pt, e, o in zip(core_7, eta_vals, obs_vals):
    print(f"  {pt['domain']:43s} {e:12.4f} {o:12.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOO-CV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. LEAVE-ONE-OUT CROSS-VALIDATION (n=7)")
print("=" * 80)

loo = LeaveOneOut()
loo_predictions = np.zeros(7)
loo_details = []

for train_idx, test_idx in loo.split(eta_vals):
    lr = LinearRegression()
    lr.fit(eta_vals[train_idx].reshape(-1, 1), obs_vals[train_idx])
    pred = lr.predict(eta_vals[test_idx].reshape(-1, 1))[0]
    loo_predictions[test_idx[0]] = pred

    detail = {
        "held_out": core_7[test_idx[0]]["domain"],
        "eta": round(float(eta_vals[test_idx[0]]), 4),
        "observed": round(float(obs_vals[test_idx[0]]), 6),
        "predicted": round(float(pred), 6),
        "error": round(float(pred - obs_vals[test_idx[0]]), 6),
        "abs_error": round(abs(float(pred - obs_vals[test_idx[0]])), 6),
        "slope": round(float(lr.coef_[0]), 4),
        "intercept": round(float(lr.intercept_), 4),
    }
    loo_details.append(detail)

# LOO R^2
ss_res = np.sum((obs_vals - loo_predictions) ** 2)
ss_tot = np.sum((obs_vals - np.mean(obs_vals)) ** 2)
loo_r2 = 1.0 - ss_res / ss_tot
loo_mae = float(np.mean(np.abs(obs_vals - loo_predictions)))

print(f"\n  LOO R^2: {loo_r2:.4f}")
print(f"  LOO MAE: {loo_mae:.4f}")
print(f"\n  Per-fold details:")
print(f"  {'Held out':45s} {'eta':>8s} {'Obs':>10s} {'Pred':>10s} {'Error':>10s}")
print("  " + "─" * 85)
for d in loo_details:
    print(f"  {d['held_out']:45s} {d['eta']:8.4f} {d['observed']:10.4f} "
          f"{d['predicted']:10.4f} {d['error']:+10.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. GOF: F-test for linear regression significance
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. GOODNESS OF FIT: F-test + Adjusted R^2")
print("=" * 80)

# Full model fit
lr_full = LinearRegression()
lr_full.fit(eta_vals.reshape(-1, 1), obs_vals)
full_pred = lr_full.predict(eta_vals.reshape(-1, 1))
residuals = obs_vals - full_pred

n = len(obs_vals)
p = 1  # one predictor

ss_reg = np.sum((full_pred - np.mean(obs_vals)) ** 2)
ss_res_full = np.sum(residuals ** 2)
ss_tot_full = np.sum((obs_vals - np.mean(obs_vals)) ** 2)

r2_full = 1.0 - ss_res_full / ss_tot_full
adj_r2 = 1.0 - (1 - r2_full) * (n - 1) / (n - p - 1)

# F-statistic: F(p, n-p-1) = (SS_reg/p) / (SS_res/(n-p-1))
ms_reg = ss_reg / p
ms_res = ss_res_full / (n - p - 1)
F_stat = ms_reg / ms_res
F_pvalue = 1.0 - stats.f.cdf(F_stat, p, n - p - 1)

print(f"\n  Slope:       {lr_full.coef_[0]:.4f}")
print(f"  Intercept:   {lr_full.intercept_:.4f}")
print(f"  R^2:         {r2_full:.4f}")
print(f"  Adjusted R^2: {adj_r2:.4f}")
print(f"  F({p},{n-p-1}):      {F_stat:.4f}")
print(f"  p-value:     {F_pvalue:.6f}")
print(f"  Significant at alpha=0.05: {'YES' if F_pvalue < 0.05 else 'NO'}")
print(f"  Significant at alpha=0.01: {'YES' if F_pvalue < 0.01 else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. RESIDUAL DIAGNOSTICS")
print("=" * 80)

# Shapiro-Wilk
sw_stat, sw_pvalue = stats.shapiro(residuals)
print(f"\n  Shapiro-Wilk test for normality of residuals:")
print(f"    W = {sw_stat:.4f}, p = {sw_pvalue:.4f}")
print(f"    Residuals {'are' if sw_pvalue > 0.05 else 'are NOT'} normally distributed (alpha=0.05)")

# Cook's distance
# Cook's D_i = (e_i^2 / (p * MSE)) * (h_i / (1 - h_i)^2)
X_mat = np.column_stack([np.ones(n), eta_vals])
H = X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T
leverage = np.diag(H)
mse = ss_res_full / (n - p - 1)

cooks_d = (residuals ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
# Standard threshold: Cook's D > 4/n is influential
threshold = 4.0 / n

print(f"\n  Cook's distance (threshold = 4/n = {threshold:.4f}):")
influential_points = []
for i, (pt, cd, lev) in enumerate(zip(core_7, cooks_d, leverage)):
    flag = " ***INFLUENTIAL***" if cd > threshold else ""
    print(f"    {pt['domain']:45s}  D={cd:.4f}  leverage={lev:.4f}{flag}")
    if cd > threshold:
        influential_points.append(pt["domain"])

print(f"\n  Leverage values:")
print(f"    Mean leverage: {np.mean(leverage):.4f} (expected: {(p+1)/n:.4f})")
print(f"    Max leverage:  {np.max(leverage):.4f} (high if > {2*(p+1)/n:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# 4. PERMUTATION NULL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. PERMUTATION NULL (10000 permutations)")
print("=" * 80)

n_perm = 10000
perm_r2s = []
for _ in range(n_perm):
    perm_eta = np.random.permutation(eta_vals)
    lr_perm = LinearRegression()
    lr_perm.fit(perm_eta.reshape(-1, 1), obs_vals)
    perm_pred = lr_perm.predict(perm_eta.reshape(-1, 1))
    ss_res_perm = np.sum((obs_vals - perm_pred) ** 2)
    perm_r2s.append(1.0 - ss_res_perm / ss_tot_full)

perm_r2s = np.array(perm_r2s)
perm_pvalue = float(np.mean(perm_r2s >= r2_full))

print(f"\n  Observed R^2:      {r2_full:.4f}")
print(f"  Permutation mean:  {np.mean(perm_r2s):.4f}")
print(f"  Permutation std:   {np.std(perm_r2s):.4f}")
print(f"  Permutation 95th:  {np.percentile(perm_r2s, 95):.4f}")
print(f"  Permutation 99th:  {np.percentile(perm_r2s, 99):.4f}")
print(f"  p-value:           {perm_pvalue:.4f}")
print(f"  Significant at alpha=0.05: {'YES' if perm_pvalue < 0.05 else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PREDICTION FOR HELD-OUT INSTANCES (using all 7 to predict 9 holdout)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. OUT-OF-SAMPLE PREDICTIONS (7 -> 9 holdout instances)")
print("=" * 80)

holdout_eta = np.array([pt["predicted_instability"] for pt in holdout_points])
holdout_obs = np.array([pt["observed_instability"] for pt in holdout_points])

# Fit on all 7 core
lr_all7 = LinearRegression()
lr_all7.fit(eta_vals.reshape(-1, 1), obs_vals)
holdout_pred = lr_all7.predict(holdout_eta.reshape(-1, 1))

print(f"\n  Model trained on 7 core instances:")
print(f"    slope = {lr_all7.coef_[0]:.4f}, intercept = {lr_all7.intercept_:.4f}")
print(f"\n  {'Holdout instance':50s} {'eta':>8s} {'Obs':>10s} {'Pred':>10s} {'Error':>10s}")
print("  " + "─" * 90)

holdout_details = []
for pt, e, o, p_val in zip(holdout_points, holdout_eta, holdout_obs, holdout_pred):
    err = p_val - o
    print(f"  {pt['domain']:50s} {e:8.4f} {o:10.4f} {p_val:10.4f} {err:+10.4f}")
    holdout_details.append({
        "domain": pt["domain"],
        "eta": round(float(e), 4),
        "observed": round(float(o), 6),
        "predicted": round(float(p_val), 6),
        "error": round(float(err), 6),
    })

# Holdout R^2
if len(holdout_obs) > 1:
    ss_res_ho = np.sum((holdout_obs - holdout_pred) ** 2)
    ss_tot_ho = np.sum((holdout_obs - np.mean(holdout_obs)) ** 2)
    holdout_r2 = 1.0 - ss_res_ho / ss_tot_ho
    holdout_mae_val = float(np.mean(np.abs(holdout_obs - holdout_pred)))
    holdout_spearman = float(stats.spearmanr(holdout_obs, holdout_pred).statistic)
    print(f"\n  Holdout R^2:       {holdout_r2:.4f}")
    print(f"  Holdout MAE:       {holdout_mae_val:.4f}")
    print(f"  Holdout Spearman:  {holdout_spearman:.4f}")
else:
    holdout_r2 = float('nan')
    holdout_mae_val = float('nan')
    holdout_spearman = float('nan')

# ═══════════════════════════════════════════════════════════════════════════
# 6. COMBINED SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n  In-sample R^2:     {r2_full:.4f}")
print(f"  Adjusted R^2:      {adj_r2:.4f}")
print(f"  LOO-CV R^2:        {loo_r2:.4f}")
print(f"  LOO-CV MAE:        {loo_mae:.4f}")
print(f"  F-test p-value:    {F_pvalue:.6f}")
print(f"  Permutation p:     {perm_pvalue:.4f}")
print(f"  Shapiro-Wilk p:    {sw_pvalue:.4f}")
print(f"  Holdout R^2 (9pt): {holdout_r2:.4f}")
if influential_points:
    print(f"  Influential pts:   {influential_points}")
else:
    print(f"  Influential pts:   None")

# ── Save results ──────────────────────────────────────────────────────────
output = {
    "experiment": "eta_law_oos_gof",
    "n_core": 7,
    "n_holdout": len(holdout_points),
    "core_instances": [pt["domain"] for pt in core_7],
    "section_1_loo_cv": {
        "LOO_R2": round(loo_r2, 4),
        "LOO_MAE": round(loo_mae, 4),
        "per_fold": loo_details,
    },
    "section_2_gof": {
        "R2": round(r2_full, 4),
        "adjusted_R2": round(adj_r2, 4),
        "slope": round(float(lr_full.coef_[0]), 4),
        "intercept": round(float(lr_full.intercept_), 4),
        "F_statistic": round(F_stat, 4),
        "F_df": [p, n - p - 1],
        "F_pvalue": round(F_pvalue, 6),
        "significant_005": F_pvalue < 0.05,
        "significant_001": F_pvalue < 0.01,
    },
    "section_3_diagnostics": {
        "shapiro_wilk_W": round(sw_stat, 4),
        "shapiro_wilk_p": round(sw_pvalue, 4),
        "residuals_normal": sw_pvalue > 0.05,
        "cooks_distance": {core_7[i]["domain"]: round(float(cooks_d[i]), 4)
                           for i in range(len(core_7))},
        "leverage": {core_7[i]["domain"]: round(float(leverage[i]), 4)
                     for i in range(len(core_7))},
        "influential_points": influential_points,
        "cooks_threshold": round(threshold, 4),
    },
    "section_4_permutation": {
        "observed_R2": round(r2_full, 4),
        "permutation_mean_R2": round(float(np.mean(perm_r2s)), 4),
        "permutation_std_R2": round(float(np.std(perm_r2s)), 4),
        "permutation_p95": round(float(np.percentile(perm_r2s, 95)), 4),
        "permutation_p99": round(float(np.percentile(perm_r2s, 99)), 4),
        "p_value": round(perm_pvalue, 4),
        "n_permutations": n_perm,
        "significant_005": perm_pvalue < 0.05,
    },
    "section_5_holdout": {
        "model_slope": round(float(lr_all7.coef_[0]), 4),
        "model_intercept": round(float(lr_all7.intercept_), 4),
        "holdout_R2": round(holdout_r2, 4) if np.isfinite(holdout_r2) else None,
        "holdout_MAE": round(holdout_mae_val, 4) if np.isfinite(holdout_mae_val) else None,
        "holdout_Spearman": round(holdout_spearman, 4) if np.isfinite(holdout_spearman) else None,
        "holdout_details": holdout_details,
    },
}

out_path = BASE / "results_eta_law_oos_gof.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(out_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {out_path}")
