#!/usr/bin/env python3
"""
SAGE Baseline Comparison: Is SAGE better than naive predictors?

Baselines:
1. Max pairwise |rho| -> predict instability = max_corr
2. Mean pairwise |rho| -> predict instability = mean_corr
3. Fraction of pairs with |rho| > 0.5
4. 1 - 1/P (number of features)
5. Random predictor (permutation null, 10000 permutations)

For each: R^2, MAE, Spearman rho with bootstrap 95% CIs.
SAGE must beat ALL baselines to be meaningful.

Data sources:
- results_sage_audit.json (test4_expanded_datasets: 8 datasets with calibrated predictions)
- results_sage_discovery.json (5 datasets, raw predictions)
- results_sage_real.json (5 datasets, real-world predictions)
"""

import json
import numpy as np
from scipy import stats
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_wine,
    fetch_california_housing, load_iris, load_digits
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE = Path(__file__).parent

# ── Load SAGE results ──────────────────────────────────────────────────────
with open(BASE / "results_sage_audit.json") as f:
    audit = json.load(f)

# Primary dataset: test4_expanded_datasets (8 datasets, calibrated R^2 = 0.809)
expanded = audit["tests"]["test4_expanded_datasets"]["datasets"]

# Extract SAGE predictions and observed instability
datasets_info = {}
for name, info in expanded.items():
    datasets_info[name] = {
        "n_features": info["n_features"],
        "sage_predicted": info["instability_predicted"],
        "observed": info["instability_observed"],
    }

print("=" * 80)
print("SAGE BASELINE COMPARISON EXPERIMENT")
print("=" * 80)
print(f"\nDatasets (n={len(datasets_info)}):")
for name, info in datasets_info.items():
    print(f"  {name:30s}  P={info['n_features']:3d}  "
          f"SAGE_pred={info['sage_predicted']:.4f}  obs={info['observed']:.4f}")

# ── Load actual datasets and compute correlation matrices ──────────────────
def load_dataset(name):
    """Load sklearn dataset and return feature matrix."""
    if name == "Breast Cancer":
        X = load_breast_cancer().data
    elif name == "Diabetes":
        X = load_diabetes().data
    elif name == "Wine":
        X = load_wine().data
    elif name == "California Housing":
        X = fetch_california_housing().data
    elif name == "Iris (regression)":
        X = load_iris().data[:, :3]  # first 3 features
    elif name == "Digits (PCA-10)":
        from sklearn.decomposition import PCA
        X_raw = load_digits().data
        X = PCA(n_components=10, random_state=42).fit_transform(X_raw)
    elif name == "Synthetic Classification":
        from sklearn.datasets import make_classification
        X, _ = make_classification(n_samples=500, n_features=20,
                                   n_informative=10, n_redundant=5,
                                   random_state=42)
    elif name == "Synthetic Regression":
        from sklearn.datasets import make_regression
        X, _ = make_regression(n_samples=500, n_features=15,
                               n_informative=8, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return StandardScaler().fit_transform(X)


def compute_pairwise_correlations(X):
    """Compute absolute pairwise Pearson correlations (upper triangle)."""
    corr = np.abs(np.corrcoef(X.T))
    P = corr.shape[0]
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            pairs.append(corr[i, j])
    return np.array(pairs)


# ── Compute baseline predictions for each dataset ─────────────────────────
print("\n" + "─" * 80)
print("Computing baseline predictions...")

dataset_names = list(datasets_info.keys())
observed = np.array([datasets_info[n]["observed"] for n in dataset_names])
sage_pred = np.array([datasets_info[n]["sage_predicted"] for n in dataset_names])
n_features = np.array([datasets_info[n]["n_features"] for n in dataset_names])

baseline_preds = {
    "max_abs_corr": [],
    "mean_abs_corr": [],
    "frac_above_05": [],
    "one_minus_1_over_P": [],
}

for name in dataset_names:
    X = load_dataset(name)
    corrs = compute_pairwise_correlations(X)

    baseline_preds["max_abs_corr"].append(float(np.max(corrs)))
    baseline_preds["mean_abs_corr"].append(float(np.mean(corrs)))
    baseline_preds["frac_above_05"].append(float(np.mean(corrs > 0.5)))
    P = datasets_info[name]["n_features"]
    baseline_preds["one_minus_1_over_P"].append(1.0 - 1.0 / P)

for k in baseline_preds:
    baseline_preds[k] = np.array(baseline_preds[k])


# ── Metrics ────────────────────────────────────────────────────────────────
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_rho(y_true, y_pred):
    return float(stats.spearmanr(y_true, y_pred).statistic)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for a metric."""
    n = len(y_true)
    boot_vals = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception:
            pass
    if len(boot_vals) < 100:
        return (np.nan, np.nan)
    boot_vals = np.sort(boot_vals)
    lo = np.percentile(boot_vals, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_vals, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


# ── Evaluate all methods ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESULTS TABLE")
print("=" * 80)

all_methods = {
    "SAGE (calibrated)": sage_pred,
    "Baseline 1: max |rho|": baseline_preds["max_abs_corr"],
    "Baseline 2: mean |rho|": baseline_preds["mean_abs_corr"],
    "Baseline 3: frac |rho|>0.5": baseline_preds["frac_above_05"],
    "Baseline 4: 1-1/P": baseline_preds["one_minus_1_over_P"],
}

results_table = {}

header = f"{'Method':35s} {'R^2':>8s} {'MAE':>8s} {'Spearman':>10s} {'R^2 95% CI':>20s} {'Spearman 95% CI':>20s}"
print(header)
print("─" * len(header))

for method_name, pred in all_methods.items():
    r2 = r_squared(observed, pred)
    m = mae(observed, pred)
    rho = spearman_rho(observed, pred)
    r2_ci = bootstrap_ci(observed, pred, r_squared)
    rho_ci = bootstrap_ci(observed, pred, spearman_rho)

    results_table[method_name] = {
        "R2": round(r2, 4),
        "MAE": round(m, 4),
        "Spearman_rho": round(rho, 4),
        "R2_95CI": [round(r2_ci[0], 4), round(r2_ci[1], 4)],
        "Spearman_95CI": [round(rho_ci[0], 4), round(rho_ci[1], 4)],
        "predictions": {n: round(float(p), 4) for n, p in zip(dataset_names, pred)},
    }

    print(f"{method_name:35s} {r2:8.4f} {m:8.4f} {rho:10.4f} "
          f"[{r2_ci[0]:7.3f}, {r2_ci[1]:7.3f}] "
          f"[{rho_ci[0]:7.3f}, {rho_ci[1]:7.3f}]")


# ── Permutation null (random predictor) ───────────────────────────────────
print("\n" + "─" * 80)
print("Permutation null distribution (10000 permutations)...")

n_perm = 10000
perm_r2s = []
for _ in range(n_perm):
    perm_obs = np.random.permutation(observed)
    perm_r2s.append(r_squared(perm_obs, sage_pred))

perm_r2s = np.array(perm_r2s)
sage_r2 = r_squared(observed, sage_pred)
p_value_perm = float(np.mean(perm_r2s >= sage_r2))

print(f"  SAGE R^2:            {sage_r2:.4f}")
print(f"  Permutation mean R^2: {np.mean(perm_r2s):.4f}")
print(f"  Permutation std R^2:  {np.std(perm_r2s):.4f}")
print(f"  Permutation 95th:     {np.percentile(perm_r2s, 95):.4f}")
print(f"  Permutation 99th:     {np.percentile(perm_r2s, 99):.4f}")
print(f"  p-value (SAGE >= perm): {p_value_perm:.4f}")

results_table["Baseline 5: Random (permutation)"] = {
    "R2_mean": round(float(np.mean(perm_r2s)), 4),
    "R2_std": round(float(np.std(perm_r2s)), 4),
    "R2_p95": round(float(np.percentile(perm_r2s, 95)), 4),
    "R2_p99": round(float(np.percentile(perm_r2s, 99)), 4),
    "p_value": round(p_value_perm, 4),
    "n_permutations": n_perm,
}

# ── LOO-CV comparison ────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("LOO-CV Comparison (linear calibration for each baseline)...")

from sklearn.linear_model import LinearRegression

loo_results = {}
for method_name, pred in all_methods.items():
    n = len(observed)
    loo_preds = np.zeros(n)
    for i in range(n):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        test_idx = np.array([i])
        lr = LinearRegression()
        lr.fit(pred[train_idx].reshape(-1, 1), observed[train_idx])
        loo_preds[i] = lr.predict(pred[test_idx].reshape(-1, 1))[0]

    loo_r2 = r_squared(observed, loo_preds)
    loo_mae_val = mae(observed, loo_preds)
    loo_results[method_name] = {
        "LOO_R2": round(loo_r2, 4),
        "LOO_MAE": round(loo_mae_val, 4),
    }
    print(f"  {method_name:35s}  LOO R^2={loo_r2:8.4f}  LOO MAE={loo_mae_val:.4f}")

# ── SAGE beats all baselines? ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERDICT: Does SAGE beat ALL baselines?")
print("=" * 80)

sage_r2_val = results_table["SAGE (calibrated)"]["R2"]
sage_spearman = results_table["SAGE (calibrated)"]["Spearman_rho"]

all_beat = True
for bname in ["Baseline 1: max |rho|", "Baseline 2: mean |rho|",
              "Baseline 3: frac |rho|>0.5", "Baseline 4: 1-1/P"]:
    b_r2 = results_table[bname]["R2"]
    b_sp = results_table[bname]["Spearman_rho"]
    r2_win = sage_r2_val > b_r2
    sp_win = sage_spearman > b_sp
    status = "SAGE WINS" if (r2_win and sp_win) else "SAGE LOSES" if (not r2_win and not sp_win) else "MIXED"
    if not (r2_win and sp_win):
        all_beat = False
    print(f"  vs {bname:35s}: R^2 {'>' if r2_win else '<='} baseline ({sage_r2_val:.4f} vs {b_r2:.4f}), "
          f"Spearman {'>' if sp_win else '<='} baseline ({sage_spearman:.4f} vs {b_sp:.4f}) -> {status}")

perm_sig = p_value_perm < 0.05
print(f"  vs Random permutation: p={p_value_perm:.4f} -> {'SIGNIFICANT' if perm_sig else 'NOT SIGNIFICANT'}")
if not perm_sig:
    all_beat = False

verdict = "SAGE beats ALL baselines" if all_beat else "SAGE does NOT beat all baselines"
print(f"\n  >>> {verdict} <<<")

# ── Save results ──────────────────────────────────────────────────────────
output = {
    "experiment": "sage_baseline_comparison",
    "n_datasets": len(datasets_info),
    "dataset_names": dataset_names,
    "observed_instability": {n: round(v, 6) for n, v in zip(dataset_names, observed)},
    "sage_calibrated_r2": round(sage_r2_val, 4),
    "results": results_table,
    "loo_cv_comparison": loo_results,
    "permutation_null": results_table["Baseline 5: Random (permutation)"],
    "verdict": verdict,
    "sage_beats_all": all_beat,
}

out_path = BASE / "results_sage_baseline_comparison.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
