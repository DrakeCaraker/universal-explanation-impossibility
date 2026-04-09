"""
lending_club_validation.py

Validates the F5→F1→DASH diagnostic pipeline on real Lending Club data.
Dataset: 32,581 rows, 12 features from OpenML (downloaded to /tmp/lending_club.csv).

Pipeline:
  Step 0: Identify correlated groups (|ρ| > 0.5)
  Step 1: Train 1 XGBoost model, compute F5 Z-statistics for correlated pairs
  Step 2: Flag pairs with Z < 1.96
  Step 3: Train 20 XGBoost models (seeds 0–19), compute F1 Z and flip rates
  Step 4: For flagged pairs, compute DASH consensus with M=20
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "/tmp/lending_club.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df.shape}")

# Target column
TARGET = "loan_status"
y = df[TARGET].values

# Feature columns
feature_cols = [c for c in df.columns if c != TARGET]

# Separate numeric vs. categorical
cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"  Numeric features: {num_cols}")
print(f"  Categorical features: {cat_cols}")

# One-hot encode categoricals; cast to int so select_dtypes won't drop bool cols
if cat_cols:
    dummies = pd.get_dummies(df[cat_cols], drop_first=True).astype(int)
    X = pd.concat([df[num_cols], dummies], axis=1)
else:
    X = df[num_cols].copy()

# Drop any remaining non-numeric columns (safety net — booleans already cast to int)
X = X.select_dtypes(include=[np.number])

# Drop rows with NaN
mask = ~(X.isnull().any(axis=1) | pd.isnull(y))
X = X[mask].reset_index(drop=True)
y = y[mask]

print(f"  After encoding & dropping NaNs: {X.shape[0]} rows, {X.shape[1]} features")
feature_names = list(X.columns)
N_FEATURES = len(feature_names)

# ---------------------------------------------------------------------------
# 2. Train/test split (80/20, seed 42)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.20, random_state=42
)
print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# Step 0: Identify correlated groups (|ρ| > 0.5)
# ---------------------------------------------------------------------------
print("\n--- Step 0: Correlated feature groups ---")
CORR_THRESH = 0.5
corr_matrix = np.corrcoef(X_train.T)

correlated_pairs = []
for i, j in combinations(range(N_FEATURES), 2):
    r = corr_matrix[i, j]
    if abs(r) > CORR_THRESH:
        correlated_pairs.append((i, j, r))

# Build correlated groups via union-find
parent = list(range(N_FEATURES))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

for i, j, _ in correlated_pairs:
    union(i, j)

groups_map = {}
for idx in range(N_FEATURES):
    root = find(idx)
    groups_map.setdefault(root, []).append(idx)

corr_groups = [g for g in groups_map.values() if len(g) > 1]
print(f"  Correlated groups (|ρ| > {CORR_THRESH}): {len(corr_groups)}")
for g in corr_groups:
    print(f"    {[feature_names[i] for i in g]}")
print(f"  Total pairs: {N_FEATURES*(N_FEATURES-1)//2}, Correlated pairs: {len(correlated_pairs)}")
for i, j, r in correlated_pairs:
    print(f"    {feature_names[i]} ↔ {feature_names[j]}: ρ={r:.3f}")

# ---------------------------------------------------------------------------
# Helper: compute SHAP-based attributions for a fitted model
# ---------------------------------------------------------------------------
def get_shap_values(model, X_data, feature_names):
    """Return mean |SHAP| per feature (shape: n_features)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_data)
    # For binary XGBoost, shap_values returns a 2D array (n_samples x n_features)
    if isinstance(sv, list):
        sv = sv[1]
    return np.abs(sv).mean(axis=0)   # shape: (n_features,)

# ---------------------------------------------------------------------------
# Step 1: Single XGBoost model → F5 Z-statistics for correlated pairs
# ---------------------------------------------------------------------------
print("\n--- Step 1: Single XGBoost model (F5 Z-statistics) ---")

params_single = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=0,
    verbosity=0,
)
model_single = xgb.XGBClassifier(**params_single)
model_single.fit(X_train, y_train)
print("  Model trained.")

shap_single = get_shap_values(model_single, X_test, feature_names)

def f5_z_statistic(phi_i, phi_j):
    """
    F5 Z-statistic: tests H0: φ_i = φ_j using attributions across test samples.
    Here phi_i, phi_j are per-sample attributions (1-D arrays).
    Z = (mean_i - mean_j) / (std(phi_i - phi_j) / sqrt(n))
    """
    diff = phi_i - phi_j
    n = len(diff)
    mu = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return np.inf if mu != 0 else 0.0
    return mu / se

# Get per-sample SHAP values for correlated pairs
explainer_single = shap.TreeExplainer(model_single)
sv_single_raw = explainer_single.shap_values(X_test)
if isinstance(sv_single_raw, list):
    sv_single_raw = sv_single_raw[1]

f5_z = {}
for i, j, r in correlated_pairs:
    z = f5_z_statistic(sv_single_raw[:, i], sv_single_raw[:, j])
    f5_z[(i, j)] = z
    print(f"  F5 Z({feature_names[i]} vs {feature_names[j]}): Z={z:.3f}  (|ρ|={abs(r):.3f})")

# ---------------------------------------------------------------------------
# Step 2: Flag unstable pairs (|Z| < 1.96 from F5 test)
# Use adaptive threshold: if no pairs flagged at 1.96, report that and use
# the weakest pair for DASH demonstration.
# ---------------------------------------------------------------------------
print("\n--- Step 2: Flag unstable pairs (F5 |Z| < 1.96) ---")
flagged_pairs = [(i, j, r) for (i, j, r) in correlated_pairs if abs(f5_z[(i, j)]) < 1.96]
if not flagged_pairs and correlated_pairs:
    # No pairs unstable at 1.96 — all rankings are stable on this dataset.
    # For DASH demonstration, use the pair with smallest |Z| (closest to instability).
    weakest = min(correlated_pairs, key=lambda x: abs(f5_z[(x[0], x[1])]))
    flagged_pairs = [weakest]
    print(f"  No pairs flagged at |Z| < 1.96 (dataset has clear rankings).")
    print(f"  Using weakest pair for DASH demonstration: "
          f"{feature_names[weakest[0]]} vs {feature_names[weakest[1]]} "
          f"(Z={f5_z[(weakest[0], weakest[1])]:.3f})")
else:
    print(f"  Flagged pairs: {len(flagged_pairs)}")
for i, j, r in flagged_pairs:
    print(f"    {feature_names[i]} vs {feature_names[j]}: Z={f5_z[(i,j)]:.3f}, |ρ|={abs(r):.3f}")

# ---------------------------------------------------------------------------
# Step 3: Train 20 XGBoost models (seeds 0–19), compute F1 Z and flip rates
# ---------------------------------------------------------------------------
print("\n--- Step 3: 20 XGBoost models → F1 Z + flip rates ---")
N_MODELS = 20
all_shap_per_sample = []   # shape: (N_MODELS, n_test, n_features)

for seed in range(N_MODELS):
    params_m = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    m = xgb.XGBClassifier(**params_m)
    m.fit(X_train, y_train)
    exp_m = shap.TreeExplainer(m)
    sv_m = exp_m.shap_values(X_test)
    if isinstance(sv_m, list):
        sv_m = sv_m[1]
    all_shap_per_sample.append(sv_m)
    print(f"  Model {seed:2d}/19 trained.")

all_shap_per_sample = np.array(all_shap_per_sample)  # (20, n_test, n_features)
mean_shap_per_model = np.abs(all_shap_per_sample).mean(axis=1)   # (20, n_features)

def f1_z_statistic(phi_i_across_models, phi_j_across_models):
    """
    F1 Z-statistic across M models: tests whether mean attribution of i
    consistently exceeds j.
    Z = (mean_m[φ_i - φ_j]) / (std_m[φ_i - φ_j] / sqrt(M))
    """
    diff = phi_i_across_models - phi_j_across_models
    M = len(diff)
    mu = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(M)
    if se == 0:
        return np.inf if mu != 0 else 0.0
    return mu / se

def flip_rate(phi_i_across_models, phi_j_across_models):
    """Fraction of models where ranking of i vs j flips from the majority."""
    signs = np.sign(phi_i_across_models - phi_j_across_models)
    majority = np.sign(signs.mean())
    if majority == 0:
        return 0.5
    flips = (signs != majority).sum()
    return flips / len(signs)

f1_z_dict = {}
flip_dict = {}
for i, j, r in correlated_pairs:
    phi_i = mean_shap_per_model[:, i]
    phi_j = mean_shap_per_model[:, j]
    fz = f1_z_statistic(phi_i, phi_j)
    fr = flip_rate(phi_i, phi_j)
    f1_z_dict[(i, j)] = fz
    flip_dict[(i, j)] = fr
    print(f"  F1 Z({feature_names[i]} vs {feature_names[j]}): Z={fz:.3f}, flip={fr:.3f}")

# Unstable pairs: flip > 10%
unstable_pairs = [(i, j, r) for (i, j, r) in correlated_pairs if flip_dict[(i, j)] > 0.10]
print(f"\n  Unstable pairs (flip > 10%): {len(unstable_pairs)}")

# Correlation between F1 |Z| and flip rate
r_pearson = float("nan")
if len(correlated_pairs) >= 2:
    z_vals = np.array([abs(f1_z_dict[(i, j)]) for i, j, _ in correlated_pairs])
    flip_vals = np.array([flip_dict[(i, j)] for i, j, _ in correlated_pairs])
    # Clip infinite Z values
    z_vals_clipped = np.where(np.isinf(z_vals), 1e6, z_vals)
    if flip_vals.std() == 0:
        print(f"  F1 correlation r(|Z|, flip): N/A (all flip rates = 0; "
              f"rankings are perfectly stable on this dataset)")
    else:
        r_pearson, p_val = stats.pearsonr(z_vals_clipped, flip_vals)
        print(f"  F1 correlation r(|Z|, flip): {r_pearson:.3f}  (p={p_val:.3g})")
else:
    print("  Too few pairs for correlation.")

# ---------------------------------------------------------------------------
# Step 4: DASH consensus (M=20) for flagged pairs
# ---------------------------------------------------------------------------
print("\n--- Step 4: DASH Consensus (M=20) ---")

# DASH: average attributions across all 20 models → consensus ranking
dash_mean = mean_shap_per_model.mean(axis=0)   # (n_features,)

# Determine tied groups in DASH: features where |φ_i - φ_j| < std_m * 1.96/sqrt(M)
std_shap = mean_shap_per_model.std(axis=0)
se_shap = std_shap / np.sqrt(N_MODELS)
DASH_TIE_THRESH = 1.96

tied_groups_dash = []
visited = set()
for i, j, r in flagged_pairs:
    gap = abs(dash_mean[i] - dash_mean[j])
    se_gap = np.sqrt(se_shap[i]**2 + se_shap[j]**2)
    z_gap = gap / se_gap if se_gap > 0 else np.inf
    if z_gap < DASH_TIE_THRESH:
        pair_str = f"({feature_names[i]}, {feature_names[j]})"
        tied_groups_dash.append(pair_str)
        print(f"  DASH tie: {feature_names[i]} vs {feature_names[j]}: gap={gap:.4f}, Z_gap={z_gap:.3f} → TIED")
    else:
        print(f"  DASH resolved: {feature_names[i]} > {feature_names[j]}: gap={gap:.4f}, Z_gap={z_gap:.3f}")

# Between-group ranking stability: check that for all non-flagged correlated pairs
# DASH ranking is consistent (|Z_gap| > 1.96)
between_group_stable = True
for i, j, r in correlated_pairs:
    if (i, j) not in [(ii, jj) for ii, jj, _ in flagged_pairs]:
        gap = abs(dash_mean[i] - dash_mean[j])
        se_gap = np.sqrt(se_shap[i]**2 + se_shap[j]**2)
        z_gap = gap / se_gap if se_gap > 0 else np.inf
        if z_gap < DASH_TIE_THRESH:
            between_group_stable = False

# Also check all correlated pairs
all_between_stable = all(
    (abs(dash_mean[i] - dash_mean[j]) / (np.sqrt(se_shap[i]**2 + se_shap[j]**2) + 1e-12)) >= DASH_TIE_THRESH
    or (i, j) in {(ii, jj) for ii, jj, _ in flagged_pairs}
    for i, j, _ in correlated_pairs
)

# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------
print("\n" + "="*50)
print("=== Lending Club Validation ===")
print(f"Dataset: {X.shape[0]} loans, {N_FEATURES} features (after encoding)")
print(f"Correlated groups: {len(corr_groups)} groups")
print(f"Total pairs: {N_FEATURES*(N_FEATURES-1)//2}, Correlated pairs: {len(correlated_pairs)}")
print(f"Unstable pairs (F1 flip > 10%): {len(unstable_pairs)}")

if np.isnan(r_pearson):
    print("F1 correlation r(Z, flip): N/A (all flip rates = 0; rankings perfectly stable)")
else:
    print(f"F1 correlation r(Z, flip): {r_pearson:.3f}")

print("\nTop unstable pairs:")
if unstable_pairs:
    sorted_unstable = sorted(unstable_pairs, key=lambda x: flip_dict[(x[0], x[1])], reverse=True)
    for i, j, r in sorted_unstable[:5]:
        print(f"  [{feature_names[i]}] vs [{feature_names[j]}]: "
              f"flip={flip_dict[(i,j)]:.2f}, |ρ|={abs(r):.2f}, Z={f1_z_dict[(i,j)]:.2f}")
else:
    # Show all correlated pairs sorted by flip rate
    sorted_corr = sorted(correlated_pairs, key=lambda x: flip_dict[(x[0], x[1])], reverse=True)
    for i, j, r in sorted_corr[:5]:
        print(f"  [{feature_names[i]}] vs [{feature_names[j]}]: "
              f"flip={flip_dict[(i,j)]:.2f}, |ρ|={abs(r):.2f}, Z={f1_z_dict[(i,j)]:.2f}")

print("\nDASH Consensus (M=20):")
if tied_groups_dash:
    print(f"  Tied groups: {tied_groups_dash}")
else:
    print("  Tied groups: (none — all flagged pairs resolved by DASH)")
print(f"  Between-group ranking stable: {'YES' if all_between_stable else 'NO'}")
print("="*50)
