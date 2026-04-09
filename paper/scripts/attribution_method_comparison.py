"""
Attribution Method Stability Comparison.

Compares ranking stability across three attribution methods:
  1. TreeSHAP (mean |SHAP| per feature)
  2. Permutation importance (sklearn, n_repeats=10)
  3. XGBoost gain-based feature importance (model.feature_importances_)

All three methods produce feature rankings. The impossibility applies to
any method satisfying proportionality-like conditions.

Dataset: Breast Cancer (sklearn)
Models:  50 XGBoost classifiers (different seeds, subsample=0.8)

Key question: do alternative importance methods escape the impossibility,
or do the same pairs flip regardless of attribution method?

Saves results to paper/results_pdp_comparison.json.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import json
import numpy as np
from scipy.stats import pearsonr

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Install with: pip install shap")
    sys.exit(1)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ── Configuration ─────────────────────────────────────────────────────────────
N_SEEDS = 50
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
PERM_N_REPEATS = 10
FLIP_THRESHOLD = 0.10          # >10% flip rate => unstable pair
TEST_SIZE = 0.3
SHAP_MAX_SAMPLES = 200         # cap test samples for SHAP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results_pdp_comparison.json")

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
P = X.shape[1]                  # 30 features -> 435 pairs
N_PAIRS = P * (P - 1) // 2

print("=" * 72)
print("PDP / Gain-Based Importance Stability Comparison")
print("=" * 72)
print(f"Dataset : Breast Cancer ({X.shape[0]} samples, {P} features)")
print(f"Models  : {N_SEEDS} XGBoost classifiers (different seeds)")
print(f"XGBoost : n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
      f"lr={LEARNING_RATE}, sub={SUBSAMPLE}, col={COLSAMPLE_BYTREE}")
print(f"Methods : TreeSHAP, Permutation Importance, Gain-based Importance")
print(f"Pairs   : {N_PAIRS} (all feature pairs)")
print()

# ── Train models and collect attributions ─────────────────────────────────────
all_shap = np.zeros((N_SEEDS, P))
all_perm = np.zeros((N_SEEDS, P))
all_gain = np.zeros((N_SEEDS, P))

print(f"Training {N_SEEDS} models and computing attributions...")

for seed in range(N_SEEDS):
    if seed % 10 == 0:
        print(f"  seed {seed}/{N_SEEDS} ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # --- TreeSHAP attribution (mean |SHAP_j|) ---
    n_test = min(SHAP_MAX_SAMPLES, len(X_test))
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test[:n_test])
    if isinstance(sv, list):
        sv = sv[1]              # positive class for binary classification
    all_shap[seed] = np.mean(np.abs(sv), axis=0)

    # --- Permutation importance ---
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=PERM_N_REPEATS,
        random_state=seed,
        scoring="accuracy",
        n_jobs=1,
    )
    all_perm[seed] = perm_result.importances_mean

    # --- Gain-based feature importance ---
    all_gain[seed] = model.feature_importances_

print("  Done.\n")


# ── Flip rate computation ─────────────────────────────────────────────────────
def compute_pair_stats(attr_matrix):
    """
    Given attr_matrix of shape (M, P), compute for all C(P,2) pairs:
      - flip_rate: fraction of models where the minority ranking wins
      - z_f1: |mean(phi_j - phi_k)| / SE(phi_j - phi_k)

    Returns arrays of length C(P,2), plus list of (j, k) pair indices.
    """
    M = attr_matrix.shape[0]
    flip_rates = []
    z_stats = []
    pairs = []

    for j in range(P):
        for k in range(j + 1, P):
            diffs = attr_matrix[:, j] - attr_matrix[:, k]
            j_wins = np.sum(diffs > 0)
            k_wins = np.sum(diffs < 0)
            total = j_wins + k_wins
            flip = (min(j_wins, k_wins) / total) if total > 0 else 0.0
            flip_rates.append(flip)

            mu_d = np.mean(diffs)
            se_d = np.std(diffs, ddof=1) / np.sqrt(M)
            z = abs(mu_d) / se_d if se_d > 1e-12 else 999.0
            z_stats.append(z)
            pairs.append((j, k))

    return np.array(flip_rates), np.array(z_stats), pairs


print("Computing pair statistics for TreeSHAP ...")
shap_flips, shap_zs, pair_indices = compute_pair_stats(all_shap)

print("Computing pair statistics for Permutation Importance ...")
perm_flips, perm_zs, _ = compute_pair_stats(all_perm)

print("Computing pair statistics for Gain-based Importance ...")
gain_flips, gain_zs, _ = compute_pair_stats(all_gain)
print()


# ── Summary metrics ───────────────────────────────────────────────────────────
def summarize(name, flips, zs):
    unstable = int(np.sum(flips > FLIP_THRESHOLD))
    max_flip = float(np.max(flips))
    mean_flip = float(np.mean(flips))
    n_pairs = len(flips)

    print(f"  Method: {name}")
    print(f"    Total pairs          : {n_pairs}")
    print(f"    Unstable (flip>10%)  : {unstable}  ({100*unstable/n_pairs:.1f}%)")
    print(f"    Max flip rate        : {max_flip:.4f}")
    print(f"    Mean flip rate       : {mean_flip:.4f}")
    print()

    return {
        "method": name,
        "n_pairs": n_pairs,
        "n_unstable_pairs": unstable,
        "pct_unstable": round(100 * unstable / n_pairs, 2),
        "max_flip_rate": round(max_flip, 4),
        "mean_flip_rate": round(mean_flip, 4),
    }


print("-" * 72)
print("Comparison Table")
print("-" * 72)
shap_summary = summarize("TreeSHAP", shap_flips, shap_zs)
perm_summary = summarize("Permutation Importance", perm_flips, perm_zs)
gain_summary = summarize("Gain-based Importance", gain_flips, gain_zs)

# ── Cross-method correlation ──────────────────────────────────────────────────
print("-" * 72)
print("Cross-method Correlation (flip rates)")
print("-" * 72)

correlations = {}

for name_a, flips_a, name_b, flips_b in [
    ("TreeSHAP", shap_flips, "Permutation", perm_flips),
    ("TreeSHAP", shap_flips, "Gain-based", gain_flips),
    ("Permutation", perm_flips, "Gain-based", gain_flips),
]:
    r, p_val = pearsonr(flips_a, flips_b)
    key = f"{name_a}_vs_{name_b}"
    correlations[key] = {"r": round(float(r), 4), "p_value": float(p_val)}
    print(f"  {name_a} vs {name_b}: r = {r:.4f}  (p = {p_val:.2e})")

    # Pairs that flip in BOTH methods (above threshold)
    both_unstable = int(np.sum((flips_a > FLIP_THRESHOLD) & (flips_b > FLIP_THRESHOLD)))
    only_a = int(np.sum((flips_a > FLIP_THRESHOLD) & (flips_b <= FLIP_THRESHOLD)))
    only_b = int(np.sum((flips_a <= FLIP_THRESHOLD) & (flips_b > FLIP_THRESHOLD)))
    correlations[key]["both_unstable"] = both_unstable
    correlations[key][f"only_{name_a}"] = only_a
    correlations[key][f"only_{name_b}"] = only_b
    print(f"    Both unstable: {both_unstable}, "
          f"only {name_a}: {only_a}, only {name_b}: {only_b}")

print()

# ── Key finding ───────────────────────────────────────────────────────────────
print("-" * 72)
print("Key Finding")
print("-" * 72)

all_methods_unstable = (
    shap_summary["n_unstable_pairs"] > 0
    and perm_summary["n_unstable_pairs"] > 0
    and gain_summary["n_unstable_pairs"] > 0
)

if all_methods_unstable:
    finding = (
        "All three methods exhibit ranking instability. The impossibility "
        "is structural (Rashomon property), not an artifact of any single "
        "attribution method."
    )
else:
    finding = (
        "Not all methods show instability above threshold. Check if the "
        "dataset has sufficient collinearity for the impossibility to bite."
    )

print(f"  {finding}")
print()

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "experiment": "pdp_stability_comparison",
    "dataset": "breast_cancer",
    "n_models": N_SEEDS,
    "n_features": P,
    "n_pairs": N_PAIRS,
    "flip_threshold": FLIP_THRESHOLD,
    "xgb_params": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
    },
    "comparison_table": [shap_summary, perm_summary, gain_summary],
    "cross_method_correlations": correlations,
    "finding": finding,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_PATH}")
print("Done.")
