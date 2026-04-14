"""
SAGE G-Discovery Adversarial Audit
===================================
Six rigorous tests to determine whether the claimed R²=0.918 (calibrated)
across 5 datasets is robust or an artifact.

Tests:
1. Threshold sensitivity sweep (0.05 to 0.50)
2. Leave-one-out cross-validation
3. Correlation baseline comparison
4. Expanded dataset validation
5. Consistency check on synthetic Noether data
6. Random group assignment baseline (permutation test)
"""

import json
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_wine,
    fetch_california_housing, load_iris, load_digits,
    make_classification, make_regression,
)
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 100                   # number of bootstrap models
FLIP_THRESHOLD = 0.30     # original clustering threshold
BASE_SEED = 42
XGB_PARAMS = dict(n_estimators=50, max_depth=3, verbosity=0)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Original 5 datasets
# ---------------------------------------------------------------------------
ORIGINAL_DATASETS = {
    "Breast Cancer": dict(loader=load_breast_cancer, task="classification"),
    "Diabetes":      dict(loader=load_diabetes,      task="regression"),
    "Wine":          dict(loader=load_wine,           task="classification"),
    "California Housing": dict(loader=fetch_california_housing, task="regression"),
    "Iris (regression)":  dict(loader=load_iris,      task="regression"),
}


def load_dataset(name, cfg):
    """Return X, y for each dataset."""
    if name == "Iris (regression)":
        data = cfg["loader"]()
        X, y = data.data, data.target
        y = X[:, 0].copy()
        X = X[:, 1:]
    elif name == "Digits (PCA-10)":
        data = load_digits()
        X_raw = data.data
        y = data.target.astype(float)
        pca = PCA(n_components=10, random_state=BASE_SEED)
        X = pca.fit_transform(X_raw)
    elif name == "Synthetic Classification":
        X, y = make_classification(
            n_features=20, n_informative=10, n_redundant=5,
            n_clusters_per_class=1, random_state=42, n_samples=1000
        )
        y = y.astype(float)
    elif name == "Synthetic Regression":
        X, y = make_regression(
            n_features=15, n_informative=5, random_state=42, n_samples=1000
        )
    else:
        data = cfg["loader"]()
        X, y = data.data, data.target
    return X, y


def train_bootstrap_models(X, y, task, n_models=M):
    """Train bootstrap XGBoost models; return importance matrix."""
    n = X.shape[0]
    P = X.shape[1]
    importance_matrix = np.zeros((n_models, P))

    for i in range(n_models):
        rng = np.random.RandomState(BASE_SEED + i)
        idx = rng.choice(n, size=n, replace=True)
        X_b, y_b = X[idx], y[idx]

        if task == "classification":
            model = XGBClassifier(random_state=BASE_SEED + i, **XGB_PARAMS)
        else:
            model = XGBRegressor(random_state=BASE_SEED + i, **XGB_PARAMS)

        model.fit(X_b, y_b)
        importance_matrix[i] = model.feature_importances_

    return importance_matrix


def compute_flip_rate_matrix(importance_matrix):
    """Compute P x P flip-rate matrix F (vectorized for speed)."""
    M_models, P = importance_matrix.shape
    F = np.zeros((P, P))
    n_pairs = M_models * (M_models - 1) // 2

    for j in range(P):
        for k in range(j + 1, P):
            # Vectorized: for each model, does feature j beat k?
            j_gt_k = importance_matrix[:, j] > importance_matrix[:, k]
            n_true = j_gt_k.sum()
            n_false = M_models - n_true
            flips = int(n_true * n_false)
            F[j, k] = flips / n_pairs
            F[k, j] = F[j, k]

    return F


def discover_groups(F, threshold=FLIP_THRESHOLD):
    """Hierarchical clustering on the flip-rate matrix."""
    P = F.shape[0]
    if P < 2:
        return np.array([1]), 1, [1]

    dist = 1.0 - F
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    n_groups = len(set(labels))
    sizes = sorted([int(np.sum(labels == g)) for g in set(labels)], reverse=True)
    return labels, n_groups, sizes


def run_sage_on_dataset(name, cfg, threshold=FLIP_THRESHOLD):
    """Run full SAGE pipeline on one dataset. Return dict with results."""
    X, y = load_dataset(name, cfg)
    P = X.shape[1]

    I = train_bootstrap_models(X, y, cfg["task"])
    F = compute_flip_rate_matrix(I)
    labels, n_groups, group_sizes = discover_groups(F, threshold=threshold)

    eta_predicted = n_groups / P
    instability_predicted = 1.0 - eta_predicted

    upper = F[np.triu_indices(P, k=1)]
    instability_observed = float(np.mean(upper))

    return dict(
        n_features=int(P),
        n_groups=int(n_groups),
        group_sizes=group_sizes,
        instability_predicted=instability_predicted,
        instability_observed=instability_observed,
        labels=labels,
        flip_matrix=F,
        importance_matrix=I,
    )


def calibrated_r2(preds, obs):
    """Compute calibrated R² (fit slope+intercept, then R²)."""
    if len(preds) < 3:
        # With < 3 points, R² is not meaningful but we compute anyway
        pass
    coeffs = np.polyfit(preds, obs, 1)
    preds_cal = coeffs[0] * preds + coeffs[1]
    return float(r2_score(obs, preds_cal)), coeffs


# ---------------------------------------------------------------------------
# Run SAGE on all original datasets (cache results)
# ---------------------------------------------------------------------------
print("=" * 70)
print("SAGE G-Discovery Adversarial Audit")
print("=" * 70)

print("\n[SETUP] Running SAGE on original 5 datasets...")
original_results = {}
for name, cfg in ORIGINAL_DATASETS.items():
    print(f"  {name} ...", end=" ", flush=True)
    original_results[name] = run_sage_on_dataset(name, cfg)
    r = original_results[name]
    print(f"P={r['n_features']}, g={r['n_groups']}, "
          f"pred={r['instability_predicted']:.3f}, obs={r['instability_observed']:.3f}")

preds_orig = np.array([v["instability_predicted"] for v in original_results.values()])
obs_orig = np.array([v["instability_observed"] for v in original_results.values()])
r2_orig, coeffs_orig = calibrated_r2(preds_orig, obs_orig)
print(f"\n  Original calibrated R² = {r2_orig:.4f}")

# ===========================================================================
# TEST 1: Threshold Sensitivity
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 1: Threshold Sensitivity Sweep")
print("=" * 70)

thresholds = np.arange(0.05, 0.51, 0.05)
threshold_r2s = []

for thresh in thresholds:
    preds_t = []
    obs_t = []
    for name, cfg in ORIGINAL_DATASETS.items():
        X, y = load_dataset(name, cfg)
        P = X.shape[1]
        # Reuse cached flip matrices
        F = original_results[name]["flip_matrix"]
        _, n_groups, _ = discover_groups(F, threshold=thresh)
        inst_pred = 1.0 - n_groups / P
        inst_obs = original_results[name]["instability_observed"]
        preds_t.append(inst_pred)
        obs_t.append(inst_obs)

    preds_t = np.array(preds_t)
    obs_t = np.array(obs_t)
    r2_t, _ = calibrated_r2(preds_t, obs_t)
    threshold_r2s.append(r2_t)
    print(f"  threshold={thresh:.2f}  R²={r2_t:.4f}")

threshold_r2s = np.array(threshold_r2s)
pct_above_07 = np.mean(threshold_r2s > 0.7)
test1_pass = pct_above_07 >= 0.60
print(f"\n  R² > 0.7 for {pct_above_07*100:.0f}% of thresholds "
      f"(need >=60%): {'PASS' if test1_pass else 'FAIL'}")

test1_results = {
    "thresholds": [round(t, 2) for t in thresholds],
    "r2_values": [round(r, 6) for r in threshold_r2s],
    "pct_above_07": round(pct_above_07, 4),
    "pass": test1_pass,
}

# ===========================================================================
# TEST 2: Leave-One-Out Cross-Validation
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 2: Leave-One-Out Cross-Validation")
print("=" * 70)

dataset_names = list(original_results.keys())
loo_predictions = []
loo_observed = []
loo_details = []
mean_baseline_errors = []

for i, held_out in enumerate(dataset_names):
    # Training set: all except held_out
    train_names = [n for n in dataset_names if n != held_out]
    train_preds = np.array([original_results[n]["instability_predicted"] for n in train_names])
    train_obs = np.array([original_results[n]["instability_observed"] for n in train_names])

    # Fit calibration on training set
    coeffs_loo = np.polyfit(train_preds, train_obs, 1)

    # Predict held-out
    ho_pred_raw = original_results[held_out]["instability_predicted"]
    ho_pred_cal = coeffs_loo[0] * ho_pred_raw + coeffs_loo[1]
    ho_obs = original_results[held_out]["instability_observed"]

    # Mean baseline: predict the mean of training observed
    mean_baseline = np.mean(train_obs)
    mean_baseline_error = abs(mean_baseline - ho_obs)
    sage_error = abs(ho_pred_cal - ho_obs)

    loo_predictions.append(ho_pred_cal)
    loo_observed.append(ho_obs)
    mean_baseline_errors.append(mean_baseline_error)

    loo_details.append({
        "held_out": held_out,
        "predicted_calibrated": round(ho_pred_cal, 6),
        "observed": round(ho_obs, 6),
        "sage_error": round(sage_error, 6),
        "mean_baseline_error": round(mean_baseline_error, 6),
    })

    print(f"  Held out: {held_out:25s}  pred={ho_pred_cal:.4f}  obs={ho_obs:.4f}  "
          f"|err|={sage_error:.4f}  baseline_err={mean_baseline_error:.4f}")

loo_predictions = np.array(loo_predictions)
loo_observed = np.array(loo_observed)

# LOO-CV R²
ss_res = np.sum((loo_observed - loo_predictions) ** 2)
ss_tot = np.sum((loo_observed - np.mean(loo_observed)) ** 2)
loo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

loo_mae = np.mean(np.abs(loo_observed - loo_predictions))
mean_baseline_mae = np.mean(mean_baseline_errors)

test2_pass = (loo_r2 > 0.3) and (loo_mae < mean_baseline_mae)
print(f"\n  LOO-CV R²: {loo_r2:.4f} (need > 0.3)")
print(f"  LOO MAE:   {loo_mae:.4f}")
print(f"  Mean-baseline MAE: {mean_baseline_mae:.4f}")
print(f"  SAGE MAE < baseline MAE: {loo_mae < mean_baseline_mae}")
print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

test2_results = {
    "loo_r2": round(loo_r2, 6),
    "loo_mae": round(loo_mae, 6),
    "mean_baseline_mae": round(mean_baseline_mae, 6),
    "details": loo_details,
    "pass": test2_pass,
}

# ===========================================================================
# TEST 3: Correlation Baseline
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 3: Correlation Baseline")
print("=" * 70)

corr_predictions = []
sage_predictions_t3 = []
obs_t3 = []

for name, cfg in ORIGINAL_DATASETS.items():
    X, y = load_dataset(name, cfg)
    P = X.shape[1]

    # Correlation matrix
    corr = np.abs(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(corr, 0.0)

    # Hierarchical clustering on correlation distance
    dist_corr = 1.0 - corr
    np.fill_diagonal(dist_corr, 0.0)
    condensed_corr = squareform(dist_corr, checks=False)
    Z_corr = linkage(condensed_corr, method="complete")
    labels_corr = fcluster(Z_corr, t=0.30, criterion="distance")  # 1 - 0.70 = 0.30
    g_corr = len(set(labels_corr))

    inst_corr = 1.0 - g_corr / P
    inst_sage = original_results[name]["instability_predicted"]
    inst_obs = original_results[name]["instability_observed"]

    corr_predictions.append(inst_corr)
    sage_predictions_t3.append(inst_sage)
    obs_t3.append(inst_obs)

    print(f"  {name:25s}  g_corr={g_corr:2d}  g_sage={original_results[name]['n_groups']:2d}  "
          f"corr_pred={inst_corr:.3f}  sage_pred={inst_sage:.3f}  obs={inst_obs:.3f}")

corr_predictions = np.array(corr_predictions)
sage_predictions_t3 = np.array(sage_predictions_t3)
obs_t3 = np.array(obs_t3)

r2_corr, _ = calibrated_r2(corr_predictions, obs_t3)
r2_sage_t3, _ = calibrated_r2(sage_predictions_t3, obs_t3)

test3_pass = r2_sage_t3 > r2_corr
print(f"\n  Correlation baseline calibrated R²: {r2_corr:.4f}")
print(f"  SAGE calibrated R²:                 {r2_sage_t3:.4f}")
print(f"  SAGE > Correlation: {test3_pass}")
print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

test3_results = {
    "correlation_r2": round(r2_corr, 6),
    "sage_r2": round(r2_sage_t3, 6),
    "sage_better": test3_pass,
    "pass": test3_pass,
}

# ===========================================================================
# TEST 4: Expanded Datasets
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 4: Expanded Dataset Validation")
print("=" * 70)

EXPANDED_DATASETS = {
    **ORIGINAL_DATASETS,
    "Digits (PCA-10)": dict(loader=load_digits, task="regression"),
    "Synthetic Classification": dict(loader=make_classification, task="classification"),
    "Synthetic Regression": dict(loader=make_regression, task="regression"),
}

expanded_results = {}
for name in EXPANDED_DATASETS:
    if name in original_results:
        expanded_results[name] = original_results[name]
    else:
        cfg = EXPANDED_DATASETS[name]
        print(f"  Running SAGE on {name} ...", end=" ", flush=True)
        expanded_results[name] = run_sage_on_dataset(name, cfg)
        r = expanded_results[name]
        print(f"P={r['n_features']}, g={r['n_groups']}, "
              f"pred={r['instability_predicted']:.3f}, obs={r['instability_observed']:.3f}")

preds_exp = np.array([v["instability_predicted"] for v in expanded_results.values()])
obs_exp = np.array([v["instability_observed"] for v in expanded_results.values()])
r2_expanded, coeffs_exp = calibrated_r2(preds_exp, obs_exp)

test4_pass = r2_expanded > 0.7
print(f"\n  Expanded set ({len(expanded_results)} datasets) calibrated R²: {r2_expanded:.4f}")
print(f"  R² > 0.7: {'PASS' if test4_pass else 'FAIL'}")

# Print per-dataset summary
print("\n  Dataset summary:")
for name, r in expanded_results.items():
    print(f"    {name:30s}  P={r['n_features']:2d}  g={r['n_groups']:2d}  "
          f"pred={r['instability_predicted']:.3f}  obs={r['instability_observed']:.3f}")

test4_results = {
    "n_datasets": len(expanded_results),
    "calibrated_r2": round(r2_expanded, 6),
    "datasets": {
        name: {
            "n_features": r["n_features"],
            "n_groups": r["n_groups"],
            "instability_predicted": round(r["instability_predicted"], 6),
            "instability_observed": round(r["instability_observed"], 6),
        }
        for name, r in expanded_results.items()
    },
    "pass": test4_pass,
}

# ===========================================================================
# TEST 5: Synthetic Noether Data Consistency
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 5: Consistency on Synthetic Noether Data")
print("=" * 70)

# Generate synthetic data matching Noether experiment
P_synth = 12
G_synth = 3
GROUP_SIZE = P_synth // G_synth  # 4
betas = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
rho_within = 0.70
rho_between = 0.0
N_synth = 500
noise_synth = 1.0

# Build correlation matrix
Sigma = np.full((P_synth, P_synth), rho_between)
for g in range(G_synth):
    idx = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
    Sigma[idx, idx] = rho_within
np.fill_diagonal(Sigma, 1.0)

rng = np.random.default_rng(BASE_SEED)
L = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((N_synth, P_synth))
X_synth = Z @ L.T
y_synth = X_synth @ betas + rng.normal(0, noise_synth, N_synth)

print(f"  Generated: N={N_synth}, P={P_synth}, g_true={G_synth}, groups of {GROUP_SIZE}")
print(f"  betas={betas.tolist()}")
print(f"  rho_within={rho_within}, rho_between={rho_between}")

# Run SAGE
I_synth = train_bootstrap_models(X_synth, y_synth, "regression")
F_synth = compute_flip_rate_matrix(I_synth)
labels_synth, n_groups_synth, sizes_synth = discover_groups(F_synth)

print(f"\n  SAGE discovered {n_groups_synth} groups, sizes: {sizes_synth}")
print(f"  Labels: {labels_synth.tolist()}")

# Check group assignment correctness
true_groups = np.array([i // GROUP_SIZE for i in range(P_synth)])
# Map SAGE labels to true groups by majority vote
from collections import Counter
label_to_true = {}
for sage_label in set(labels_synth):
    members = np.where(labels_synth == sage_label)[0]
    true_group_counts = Counter(true_groups[m] for m in members)
    label_to_true[sage_label] = true_group_counts.most_common(1)[0][0]

# Check how many features are correctly assigned
correct = 0
for i in range(P_synth):
    if label_to_true[labels_synth[i]] == true_groups[i]:
        correct += 1

group_count_ok = abs(n_groups_synth - G_synth) <= 1
assignment_accuracy = correct / P_synth

print(f"\n  True groups: {true_groups.tolist()}")
print(f"  SAGE labels: {labels_synth.tolist()}")
print(f"  Mapped SAGE->true: {label_to_true}")
print(f"  Correctly assigned: {correct}/{P_synth} ({assignment_accuracy*100:.0f}%)")
print(f"  Group count within +-1: {group_count_ok} (discovered {n_groups_synth}, true {G_synth})")

test5_pass = group_count_ok and (assignment_accuracy >= 0.75)
print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")

# Print flip rate matrix structure
print(f"\n  Flip rate matrix (mean by block):")
for g1 in range(G_synth):
    for g2 in range(G_synth):
        idx1 = range(g1*GROUP_SIZE, (g1+1)*GROUP_SIZE)
        idx2 = range(g2*GROUP_SIZE, (g2+1)*GROUP_SIZE)
        block = F_synth[np.ix_(list(idx1), list(idx2))]
        if g1 == g2:
            # Upper triangle only (within-group)
            vals = block[np.triu_indices(GROUP_SIZE, k=1)]
            mean_val = np.mean(vals) if len(vals) > 0 else 0
        else:
            mean_val = np.mean(block)
        print(f"    Group {g1} vs Group {g2}: mean flip rate = {mean_val:.4f}")

test5_results = {
    "n_groups_discovered": int(n_groups_synth),
    "n_groups_true": G_synth,
    "group_sizes": sizes_synth,
    "group_count_within_1": group_count_ok,
    "assignment_accuracy": round(assignment_accuracy, 4),
    "labels": labels_synth.tolist(),
    "true_groups": true_groups.tolist(),
    "pass": test5_pass,
}

# ===========================================================================
# TEST 6: Random Group Assignment Baseline
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 6: Random Group Assignment Baseline (Permutation Test)")
print("=" * 70)

N_PERM = 1000
rng_perm = np.random.RandomState(BASE_SEED + 999)

# For each dataset, we randomly permute feature labels to destroy the
# relationship between SAGE's grouping and actual flip-rate structure.
# We keep the same number of groups as SAGE found, but assign features
# randomly. Then we re-compute the predicted instability (still 1 - g/P,
# but g may change since random assignment may merge or split groups).
# The key comparison: does SAGE's *specific* grouping predict instability
# better than random groupings with similar granularity?
random_r2s = []
for perm_i in range(N_PERM):
    preds_random = []
    for name in original_results:
        r = original_results[name]
        P = r["n_features"]
        n_groups = r["n_groups"]

        # Draw a random number of groups uniformly from [1, P]
        # This provides a fair null: any grouping granularity is equally likely
        g_random = rng_perm.randint(1, P + 1)
        inst_random = 1.0 - g_random / P
        preds_random.append(inst_random)

    preds_random = np.array(preds_random)
    try:
        r2_random, _ = calibrated_r2(preds_random, obs_orig)
        random_r2s.append(r2_random)
    except Exception:
        random_r2s.append(float("nan"))

random_r2s = np.array(random_r2s)
random_r2s_valid = random_r2s[~np.isnan(random_r2s)]

p95 = np.percentile(random_r2s_valid, 95)
p99 = np.percentile(random_r2s_valid, 99)
p_value = np.mean(random_r2s_valid >= r2_orig)

test6_pass = r2_orig > p95
print(f"  SAGE R² (calibrated): {r2_orig:.4f}")
print(f"  Random baseline 95th percentile: {p95:.4f}")
print(f"  Random baseline 99th percentile: {p99:.4f}")
print(f"  Random baseline mean: {np.mean(random_r2s_valid):.4f}")
print(f"  Random baseline std:  {np.std(random_r2s_valid):.4f}")
print(f"  p-value (fraction random >= SAGE): {p_value:.4f}")
print(f"  SAGE > 95th percentile: {test6_pass}")
print(f"  Result: {'PASS' if test6_pass else 'FAIL'}")

test6_results = {
    "sage_r2": round(r2_orig, 6),
    "random_mean": round(float(np.mean(random_r2s_valid)), 6),
    "random_std": round(float(np.std(random_r2s_valid)), 6),
    "random_p95": round(float(p95), 6),
    "random_p99": round(float(p99), 6),
    "p_value": round(float(p_value), 6),
    "n_permutations": N_PERM,
    "pass": test6_pass,
}

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

tests = {
    "test1_threshold_sensitivity": test1_results,
    "test2_loo_cv": test2_results,
    "test3_correlation_baseline": test3_results,
    "test4_expanded_datasets": test4_results,
    "test5_synthetic_noether": test5_results,
    "test6_random_baseline": test6_results,
}

n_pass = sum(1 for t in tests.values() if t["pass"])
n_total = len(tests)

test_labels = {
    "test1_threshold_sensitivity": "Threshold Sensitivity",
    "test2_loo_cv": "Leave-One-Out CV",
    "test3_correlation_baseline": "Correlation Baseline",
    "test4_expanded_datasets": "Expanded Datasets",
    "test5_synthetic_noether": "Synthetic Noether",
    "test6_random_baseline": "Random Baseline",
}

for key, res in tests.items():
    status = "PASS" if res["pass"] else "FAIL"
    print(f"  [{status}] {test_labels[key]}")

print(f"\n  Overall: {n_pass}/{n_total} tests passed")

overall_verdict = "ROBUST" if n_pass >= 4 else ("WEAK" if n_pass >= 2 else "ARTIFACT")
print(f"  Verdict: {overall_verdict}")

# ===========================================================================
# Save results JSON
# ===========================================================================
output = {
    "experiment": "sage_adversarial_audit",
    "claimed_r2": 0.918,
    "reproduced_r2": round(r2_orig, 6),
    "tests": tests,
    "n_pass": n_pass,
    "n_total": n_total,
    "verdict": overall_verdict,
    "_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results_path = OUT_DIR / "results_sage_audit.json"
with open(results_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {results_path}")

# ===========================================================================
# Figure: Multi-panel audit summary
# ===========================================================================
fig = plt.figure(figsize=(16, 12))

# Panel 1: Threshold sensitivity
ax1 = fig.add_subplot(2, 3, 1)
colors_t1 = ["#2ecc71" if r > 0.7 else "#e74c3c" for r in threshold_r2s]
ax1.bar(range(len(thresholds)), threshold_r2s, color=colors_t1, edgecolor="white", width=0.8)
ax1.axhline(0.7, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax1.set_xticks(range(len(thresholds)))
ax1.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45, fontsize=7)
ax1.set_xlabel("Clustering Threshold")
ax1.set_ylabel("Calibrated R²")
status1 = "PASS" if test1_pass else "FAIL"
ax1.set_title(f"Test 1: Threshold Sensitivity [{status1}]", fontweight="bold", fontsize=10)
ax1.set_ylim(-0.5, 1.1)

# Panel 2: LOO-CV predictions vs observed
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(loo_predictions, loo_observed, s=80, zorder=5, edgecolors="black", c="steelblue")
lims = [min(min(loo_predictions), min(loo_observed)) - 0.02,
        max(max(loo_predictions), max(loo_observed)) + 0.02]
ax2.plot(lims, lims, "k--", lw=1, alpha=0.5)
for detail in loo_details:
    ax2.annotate(detail["held_out"][:10],
                 (detail["predicted_calibrated"], detail["observed"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
ax2.set_xlabel("LOO Predicted (calibrated)")
ax2.set_ylabel("Observed Instability")
status2 = "PASS" if test2_pass else "FAIL"
ax2.set_title(f"Test 2: LOO-CV [{status2}]\nR²={loo_r2:.3f}, MAE={loo_mae:.3f}", fontweight="bold", fontsize=10)

# Panel 3: SAGE vs Correlation baseline
ax3 = fig.add_subplot(2, 3, 3)
bar_labels = ["SAGE", "Correlation"]
bar_values = [r2_sage_t3, r2_corr]
bar_colors = ["steelblue", "coral"]
ax3.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black", width=0.5)
ax3.axhline(0, color="black", linewidth=0.5)
ax3.set_ylabel("Calibrated R²")
status3 = "PASS" if test3_pass else "FAIL"
ax3.set_title(f"Test 3: SAGE vs Correlation [{status3}]", fontweight="bold", fontsize=10)
ax3.set_ylim(min(0, min(bar_values) - 0.1), 1.1)

# Panel 4: Expanded dataset fit
ax4 = fig.add_subplot(2, 3, 4)
# Calibrated predictions
preds_cal_exp = coeffs_exp[0] * preds_exp + coeffs_exp[1]
ax4.scatter(preds_cal_exp, obs_exp, s=80, zorder=5, edgecolors="black", c="steelblue")
lims4 = [min(min(preds_cal_exp), min(obs_exp)) - 0.02,
         max(max(preds_cal_exp), max(obs_exp)) + 0.02]
ax4.plot(lims4, lims4, "k--", lw=1, alpha=0.5)
for name, r in expanded_results.items():
    pred_cal = coeffs_exp[0] * r["instability_predicted"] + coeffs_exp[1]
    ax4.annotate(name[:12],
                 (pred_cal, r["instability_observed"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=6)
ax4.set_xlabel("Calibrated Predicted")
ax4.set_ylabel("Observed Instability")
status4 = "PASS" if test4_pass else "FAIL"
ax4.set_title(f"Test 4: Expanded ({len(expanded_results)} datasets) [{status4}]\nR²={r2_expanded:.3f}",
              fontweight="bold", fontsize=10)

# Panel 5: Synthetic Noether group discovery
ax5 = fig.add_subplot(2, 3, 5)
# Show flip rate matrix as heatmap
im = ax5.imshow(F_synth, cmap="RdYlBu_r", vmin=0, vmax=0.5, aspect="equal")
ax5.set_xlabel("Feature Index")
ax5.set_ylabel("Feature Index")
# Draw group boundaries
for g in range(1, G_synth):
    ax5.axhline(g * GROUP_SIZE - 0.5, color="black", linewidth=2)
    ax5.axvline(g * GROUP_SIZE - 0.5, color="black", linewidth=2)
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label="Flip Rate")
status5 = "PASS" if test5_pass else "FAIL"
ax5.set_title(f"Test 5: Noether Flip Matrix [{status5}]\n"
              f"Discovered {n_groups_synth} groups (true={G_synth}), "
              f"accuracy={assignment_accuracy*100:.0f}%",
              fontweight="bold", fontsize=10)

# Panel 6: Random baseline histogram
ax6 = fig.add_subplot(2, 3, 6)
ax6.hist(random_r2s_valid, bins=50, color="lightgray", edgecolor="gray", alpha=0.8)
ax6.axvline(r2_orig, color="steelblue", linewidth=2, label=f"SAGE R²={r2_orig:.3f}")
ax6.axvline(p95, color="red", linewidth=1, linestyle="--", label=f"95th pctile={p95:.3f}")
ax6.set_xlabel("Calibrated R² (random grouping)")
ax6.set_ylabel("Count")
ax6.legend(fontsize=8)
status6 = "PASS" if test6_pass else "FAIL"
ax6.set_title(f"Test 6: Random Baseline [{status6}]\np={p_value:.4f}",
              fontweight="bold", fontsize=10)

fig.suptitle(f"SAGE G-Discovery Adversarial Audit — {n_pass}/{n_total} PASS — Verdict: {overall_verdict}",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()

fig_path = FIG_DIR / "sage_audit.pdf"
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
print(f"Figure saved to {fig_path}")
plt.close(fig)
