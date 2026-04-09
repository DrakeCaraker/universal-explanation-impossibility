"""
KernelSHAP noise control experiment.

Measures how much of the observed NN SHAP instability is due to KernelSHAP
approximation noise vs genuine model instability from random initialization.

Design:
  - Train ONE MLP model (random_state=0) on Breast Cancer Wisconsin
  - Compute KernelSHAP 20 times with DIFFERENT random background samples
  - For each pair of SHAP runs (190 pairs): compute flip rate
  - This isolates SHAP-noise-only instability (no model variation)
  - Repeat with 200 background samples to test if more background helps
  - Compare noise-only flip rates to the 87% model-variation flip rate

Reference: Supplement Section S4.3 (Neural network instantiation).
"""

import sys
import time
import os
import numpy as np

try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import shap
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install scikit-learn shap numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SHAP_RUNS = 20
N_EXPLAIN = 100
RANDOM_STATE_SPLIT = 42
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 500
MLP_ACTIVATION = "relu"
MLP_SEED = 0

BACKGROUND_SIZES = [50, 200]

# Known high-correlation pair on Breast Cancer Wisconsin
CORR_IDX_J = 22  # worst perimeter
CORR_IDX_K = 23  # worst area

# Threshold for declaring a pair "unstable"
FLIP_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
X, y = data.data.astype(np.float64), data.target.astype(np.float64)
feature_names = list(data.feature_names)
P = X.shape[1]  # 30 features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE_SPLIT
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_explain = X_test_sc[:N_EXPLAIN]

# ---------------------------------------------------------------------------
# Train ONE model
# ---------------------------------------------------------------------------
print(f"Training single MLP (random_state={MLP_SEED})...")
mlp = MLPRegressor(
    hidden_layer_sizes=MLP_HIDDEN,
    activation=MLP_ACTIVATION,
    max_iter=MLP_MAX_ITER,
    random_state=MLP_SEED,
)
mlp.fit(X_train_sc, y_train)
print("  Model trained.")
print()

# ---------------------------------------------------------------------------
# Helper: compute pairwise flip rates from SHAP importance arrays
# ---------------------------------------------------------------------------
def compute_flip_rates(all_mean_abs_shap):
    """Compute flip rates for all 435 feature pairs across SHAP runs."""
    n_runs = all_mean_abs_shap.shape[0]
    pair_flips = {}
    for j in range(P):
        for k in range(j + 1, P):
            phi_j = all_mean_abs_shap[:, j]
            phi_k = all_mean_abs_shap[:, k]

            n_j_wins = int(np.sum(phi_j > phi_k))
            n_k_wins = int(np.sum(phi_k > phi_j))
            total = n_j_wins + n_k_wins
            flip_rate = min(n_j_wins, n_k_wins) / total if total > 0 else 0.0

            pair_flips[(j, k)] = flip_rate

    return pair_flips


# ---------------------------------------------------------------------------
# Run KernelSHAP with different background samples
# ---------------------------------------------------------------------------
results_by_bg = {}

for n_bg in BACKGROUND_SIZES:
    print(f"=== Background size = {n_bg} ===")
    all_mean_abs_shap = []
    t0 = time.time()

    for run_seed in range(N_SHAP_RUNS):
        print(f"  SHAP run {run_seed + 1}/{N_SHAP_RUNS} (bg seed={run_seed})...",
              end=" ", flush=True)

        # Draw different background samples each run
        rng = np.random.default_rng(run_seed)
        bg_idx = rng.choice(len(X_train_sc), size=n_bg, replace=False)
        X_background = X_train_sc[bg_idx]

        explainer = shap.KernelExplainer(mlp.predict, X_background)
        shap_values = explainer.shap_values(X_explain, silent=True)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        all_mean_abs_shap.append(mean_abs)

        print("done.", flush=True)

    all_mean_abs_shap = np.array(all_mean_abs_shap)  # (N_SHAP_RUNS, P)
    elapsed = time.time() - t0

    pair_flips = compute_flip_rates(all_mean_abs_shap)
    flip_rates = np.array(list(pair_flips.values()))

    n_unstable = int(np.sum(flip_rates > FLIP_THRESHOLD))
    mean_flip = float(np.mean(flip_rates))
    key_flip = pair_flips[(CORR_IDX_J, CORR_IDX_K)]

    results_by_bg[n_bg] = {
        "n_unstable": n_unstable,
        "mean_flip": mean_flip,
        "key_flip": key_flip,
        "elapsed": elapsed,
    }

    print(f"  Elapsed: {elapsed:.1f}s")
    print()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
N_PAIRS = P * (P - 1) // 2  # 435

print()
print("=" * 45)
print("  KernelSHAP Noise Control")
print("=" * 45)
print(f"Single model (random_state={MLP_SEED}), Breast Cancer")
print()

for n_bg in BACKGROUND_SIZES:
    r = results_by_bg[n_bg]
    pct = 100.0 * r["n_unstable"] / N_PAIRS
    print(f"Background={n_bg}:")
    print(f"  Noise-only unstable pairs (flip > 10%): {r['n_unstable']}/{N_PAIRS}")
    print(f"  Mean noise-only flip rate: {r['mean_flip']:.2f}")
    print(f"  Key pair (worst perim vs area) noise flip: {r['key_flip']:.2f}")
    print()

# Model-variation reference from nn_shap_validation.py
MODEL_VAR_UNSTABLE = 380
MODEL_VAR_PCT = 100.0 * MODEL_VAR_UNSTABLE / N_PAIRS

print("Comparison:")
print(f"  Model-variation unstable (from nn_shap): {MODEL_VAR_UNSTABLE}/{N_PAIRS} ({MODEL_VAR_PCT:.0f}%)")
for n_bg in BACKGROUND_SIZES:
    r = results_by_bg[n_bg]
    pct = 100.0 * r["n_unstable"] / N_PAIRS
    print(f"  Noise-only unstable (bg={n_bg}): {r['n_unstable']}/{N_PAIRS} ({pct:.0f}%)")

print()

# Verdict
noise_50 = results_by_bg[50]["n_unstable"]
noise_200 = results_by_bg[200]["n_unstable"]
max_noise_pct = 100.0 * max(noise_50, noise_200) / N_PAIRS

if max_noise_pct < MODEL_VAR_PCT * 0.25:
    verdict = "Model instability dominates"
else:
    verdict = "KernelSHAP noise is a confounder"

print(f"VERDICT: {verdict}")
print(f"  (noise accounts for at most {max_noise_pct:.0f}% unstable pairs vs {MODEL_VAR_PCT:.0f}% from model variation)")
