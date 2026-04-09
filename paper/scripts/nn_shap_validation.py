"""
Neural network SHAP validation of the Attribution Impossibility theorem.

Validates that the impossibility result (Trilemma.lean) is not specific to
tree ensembles: any model class satisfying the Rashomon property — including
MLPs under random initialization — exhibits attribution instability for
correlated features.

Experimental design:
  - Dataset: Breast Cancer Wisconsin (30 features, 569 samples)
  - Models: 20 MLPRegressor, each with a different random seed
  - Attribution: KernelSHAP (approximation; slower than TreeSHAP but
    model-agnostic — see note below)
  - Instability measure: ranking flip rate across the 20 models

NOTE: KernelSHAP is a model-agnostic approximation of Shapley values based on
weighted linear regression over feature coalitions. It is substantially slower
than the exact TreeSHAP algorithm used for gradient-boosted trees elsewhere in
this supplement, but is applicable to any differentiable or black-box model.
Expected runtime: 30-60 minutes for 20 models x 100 test samples.

Reference: Supplement Section S4.3 (Neural network instantiation).
"""

import sys
import time
import os
import json
import numpy as np
from scipy.stats import pearsonr

try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import shap
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install scikit-learn shap numpy scipy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_MODELS = 20
N_BACKGROUND = 50    # KernelSHAP background samples (more = slower but more accurate)
N_EXPLAIN = 100      # test samples to explain
RANDOM_STATE_SPLIT = 42
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 500
MLP_ACTIVATION = "relu"

# Known high-correlation pair on Breast Cancer Wisconsin
CORR_IDX_J = 22  # worst perimeter
CORR_IDX_K = 23  # worst area

# Threshold for declaring a pair "unstable"
FLIP_THRESHOLD = 0.10

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
X, y = data.data.astype(np.float64), data.target.astype(np.float64)
feature_names = list(data.feature_names)
P = X.shape[1]  # 30 features

corr_matrix = np.corrcoef(X.T)

# Fixed train/test split — same for all models (only random_state of MLP varies)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE_SPLIT
)

# Standardize (fit only on training data)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Background and explanation sets
rng = np.random.default_rng(0)
bg_idx = rng.choice(len(X_train_sc), size=N_BACKGROUND, replace=False)
X_background = X_train_sc[bg_idx]
X_explain = X_test_sc[:N_EXPLAIN]

print(f"Train: {X_train_sc.shape[0]} samples | Test: {X_test_sc.shape[0]} samples")
print(f"KernelSHAP background: {N_BACKGROUND} | Explained: {N_EXPLAIN}")
print()

# ---------------------------------------------------------------------------
# Train 20 MLPs and compute KernelSHAP
# ---------------------------------------------------------------------------
all_mean_abs_shap = []  # (N_MODELS, P)
t_start = time.time()

for seed in range(N_MODELS):
    t0 = time.time()
    print(f"Model {seed + 1}/{N_MODELS}: training MLP (random_state={seed})...", end=" ", flush=True)

    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        activation=MLP_ACTIVATION,
        max_iter=MLP_MAX_ITER,
        random_state=seed,
    )
    mlp.fit(X_train_sc, y_train)

    print(f"done. Computing KernelSHAP...", end=" ", flush=True)

    # KernelSHAP: model-agnostic, works with any predict function.
    # NOTE: This is an approximation — see module docstring.
    explainer = shap.KernelExplainer(mlp.predict, X_background)

    # silent=True suppresses the per-sample tqdm bar; we print our own progress.
    shap_values = explainer.shap_values(X_explain, silent=True)
    # shap_values: (N_EXPLAIN, P)

    mean_abs = np.mean(np.abs(shap_values), axis=0)  # (P,)
    all_mean_abs_shap.append(mean_abs)

    elapsed = time.time() - t0
    print(f"done. ({elapsed:.1f}s)")

all_mean_abs_shap = np.array(all_mean_abs_shap)  # (N_MODELS, P)
t_total = time.time() - t_start

# ---------------------------------------------------------------------------
# Compute flip rates for all 435 feature pairs
# ---------------------------------------------------------------------------
print("\nComputing flip rates for all feature pairs...")

pairs = []
for j in range(P):
    for k in range(j + 1, P):
        phi_j = all_mean_abs_shap[:, j]
        phi_k = all_mean_abs_shap[:, k]

        n_j_wins = int(np.sum(phi_j > phi_k))
        n_k_wins = int(np.sum(phi_k > phi_j))
        total = n_j_wins + n_k_wins
        flip_rate = min(n_j_wins, n_k_wins) / total if total > 0 else 0.0

        rho_jk = float(abs(corr_matrix[j, k]))

        pairs.append({
            "j": j,
            "k": k,
            "j_name": feature_names[j],
            "k_name": feature_names[k],
            "rho": rho_jk,
            "flip": flip_rate,
            "n_j_wins": n_j_wins,
            "n_k_wins": n_k_wins,
        })

# ---------------------------------------------------------------------------
# F1 diagnostic: Z_{jk} = |mean(phi_j - phi_k)| / SE, correlation with flip
# ---------------------------------------------------------------------------
z_scores = []
flip_rates = []

for p in pairs:
    j, k = p["j"], p["k"]
    diff = all_mean_abs_shap[:, j] - all_mean_abs_shap[:, k]
    mu = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(N_MODELS)
    z = abs(mu) / se if se > 1e-10 else 999.0
    z_scores.append(z)
    flip_rates.append(p["flip"])

z_scores = np.array(z_scores)
flip_rates = np.array(flip_rates)
z_clip = np.clip(z_scores, 0, 20)

r_z_flip, p_val_z_flip = pearsonr(z_clip, flip_rates)

# ---------------------------------------------------------------------------
# Key pair: worst perimeter vs worst area
# ---------------------------------------------------------------------------
key_pair = next(
    p for p in pairs if p["j"] == CORR_IDX_J and p["k"] == CORR_IDX_K
)
nn_flip_key = key_pair["flip"]
rho_key = key_pair["rho"]

# ---------------------------------------------------------------------------
# Unstable pairs
# ---------------------------------------------------------------------------
n_unstable = int(np.sum(flip_rates > FLIP_THRESHOLD))

# ---------------------------------------------------------------------------
# Cross-method comparison: XGBoost flip rates (paper baseline)
# We do not rerun XGBoost here; we compare the distributions qualitatively
# using the known paper figures.  The cross-method correlation is assessed
# by comparing our NN flip rates against the paper-reported XGBoost values
# stored in results_f1_f5.json if available.
# ---------------------------------------------------------------------------
xgb_results_path = os.path.join(os.path.dirname(__file__), "..", "results_f1_f5.json")
cross_method_r = None
xgb_pairs_lookup = {}

if os.path.exists(xgb_results_path):
    with open(xgb_results_path) as f:
        xgb_data = json.load(f)
    for xp in xgb_data.get("pairs", []):
        xgb_pairs_lookup[(xp["j"], xp["k"])] = xp["flip"]

    if xgb_pairs_lookup:
        nn_flips_matched = []
        xgb_flips_matched = []
        for p in pairs:
            key = (p["j"], p["k"])
            if key in xgb_pairs_lookup:
                nn_flips_matched.append(p["flip"])
                xgb_flips_matched.append(xgb_pairs_lookup[key])
        if len(nn_flips_matched) > 10:
            cross_method_r, _ = pearsonr(nn_flips_matched, xgb_flips_matched)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("  Neural Network SHAP Validation")
print("=" * 55)
print(f"Models trained: {N_MODELS} MLPRegressor")
print(f"  hidden_layer_sizes={MLP_HIDDEN}, activation={MLP_ACTIVATION}")
print(f"  max_iter={MLP_MAX_ITER}, random_state 0–{N_MODELS - 1}")
print(f"Attribution: KernelSHAP ({N_BACKGROUND} background, {N_EXPLAIN} test samples)")
print(f"Elapsed: {t_total / 60:.1f} min")
print()
print(f"Key pair: {feature_names[CORR_IDX_J]} vs {feature_names[CORR_IDX_K]}")
print(f"  Correlation: |ρ| = {rho_key:.3f}")
print(f"  NN flip rate:              {nn_flip_key:.3f}")
print(f"  XGBoost flip rate (paper): 0.48")
print()
print("Overall:")
print(f"  Unstable pairs (flip > {FLIP_THRESHOLD:.0%}): {n_unstable}/435")
print(f"  F1 correlation r(Z, flip): {r_z_flip:.3f}  (p={p_val_z_flip:.2e})")
if cross_method_r is not None:
    print()
    print("Comparison to XGBoost:")
    print(f"  Cross-method flip rate correlation: {cross_method_r:.3f}")
else:
    print()
    print("Comparison to XGBoost:")
    print("  (results_f1_f5.json not found — run f1_f5_validation.py first)")
print()

# Verdict
passes = nn_flip_key > 0.20
status = "DO" if passes else "DO NOT"
print(f"RESULT: Neural networks {status} exhibit attribution instability")
print("        (criterion: key correlated pair flip rate > 20%)")
print()

# Most unstable pairs
print("Top 10 most unstable pairs (NN):")
for p in sorted(pairs, key=lambda x: -x["flip"])[:10]:
    print(f"  flip={p['flip']:.3f}  |ρ|={p['rho']:.3f}  "
          f"{p['j_name'][:22]:22s} <-> {p['k_name']}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "dataset": "Breast Cancer Wisconsin",
    "n_models": N_MODELS,
    "n_background": N_BACKGROUND,
    "n_explain": N_EXPLAIN,
    "mlp_hidden": list(MLP_HIDDEN),
    "mlp_activation": MLP_ACTIVATION,
    "mlp_max_iter": MLP_MAX_ITER,
    "elapsed_seconds": t_total,
    "key_pair": {
        "j": CORR_IDX_J,
        "k": CORR_IDX_K,
        "j_name": feature_names[CORR_IDX_J],
        "k_name": feature_names[CORR_IDX_K],
        "rho": rho_key,
        "nn_flip_rate": nn_flip_key,
        "xgb_flip_rate_paper": 0.48,
    },
    "n_unstable_pairs": n_unstable,
    "flip_threshold": FLIP_THRESHOLD,
    "r_z_flip": float(r_z_flip),
    "p_z_flip": float(p_val_z_flip),
    "cross_method_r": float(cross_method_r) if cross_method_r is not None else None,
    "pairs": pairs,
}

json_path = os.path.join(os.path.dirname(__file__), "..", "results_nn_shap.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved {json_path}")

sys.exit(0 if passes else 1)
