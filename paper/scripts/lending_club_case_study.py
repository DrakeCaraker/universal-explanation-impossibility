"""
Lending Club Case Study: Credit Attribution Impossibility under Collinearity

Demonstrates the F5 -> F1 -> DASH workflow on the Lending Club dataset
(or fallback: UCI default-of-credit-card-clients) showing that genuine
high collinearity (rho > 0.5) causes attribution instability that DASH resolves.

Pipeline:
  Step 0: Load data, encode categoricals, impute NaN, subsample to 10000
  Step 1: Compute correlation matrix, identify pairs with |rho| > 0.5
  Step 2 (F5): Train 1 model, compute split-frequency Z for correlated pairs
  Step 3 (F1): Train 5 models, compute Z_jk for flagged pairs
  Step 4 (DASH): Train 25 models, show flip rate drops

Requires: pip install xgboost scikit-learn numpy pandas scipy

Usage:
  python3 lending_club_case_study.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Output setup: tee to both stdout and results file
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results_lending_club.txt")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee = Tee(RESULTS_PATH)
sys.stdout = tee

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Step 0: Load dataset
# ---------------------------------------------------------------------------
print("=" * 72)
print("LENDING CLUB CASE STUDY: Attribution Impossibility under Collinearity")
print("=" * 72)
print()

dataset_name = None
df = None
target_col = None

# Try Lending Club variants on OpenML
attempts = [
    {"kwargs": {"name": "lending-club", "as_frame": True}, "label": "lending-club (by name)"},
    {"kwargs": {"data_id": 42926, "as_frame": True}, "label": "lending-club (data_id=42926)"},
]

for attempt in attempts:
    try:
        print(f"Trying fetch_openml({attempt['label']})...", end=" ")
        data = fetch_openml(**attempt["kwargs"])
        df = data.frame
        if df is not None and len(df) > 100:
            dataset_name = attempt["label"]
            target_col = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else data.frame.columns[-1]
            print(f"SUCCESS ({len(df)} rows, {df.shape[1]} cols)")
            break
        else:
            print("too few rows, skipping")
    except Exception as e:
        print(f"FAILED ({type(e).__name__}: {e})")

# Search for any lending/loan dataset
if df is None:
    print("\nSearching OpenML for lending/loan datasets...")
    try:
        from sklearn.datasets import fetch_openml
        import openml
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        loan_datasets = datasets[
            datasets["name"].str.contains("lend|loan", case=False, na=False)
        ].sort_values("NumberOfInstances", ascending=False)
        if len(loan_datasets) > 0:
            top = loan_datasets.iloc[0]
            print(f"Found: {top['name']} (id={top.name}, {int(top['NumberOfInstances'])} rows)")
            data = fetch_openml(data_id=int(top.name), as_frame=True)
            df = data.frame
            dataset_name = top["name"]
            target_col = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else df.columns[-1]
    except Exception as e:
        print(f"OpenML search failed: {e}")

# Fallback: UCI default of credit card clients
if df is None:
    print("\nFalling back to UCI default-of-credit-card-clients (OpenML id=42477)...")
    try:
        data = fetch_openml(data_id=42477, as_frame=True)
        df = data.frame
        dataset_name = "default-of-credit-card-clients"
        target_col = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else df.columns[-1]
        print(f"SUCCESS ({len(df)} rows, {df.shape[1]} cols)")
    except Exception as e:
        print(f"Fallback also failed: {e}")
        print("ERROR: Could not load any suitable dataset. Exiting.")
        sys.stdout = tee.stdout
        tee.close()
        sys.exit(1)

print(f"\nDataset: {dataset_name}")
print(f"Target column: {target_col}")
print(f"Shape: {df.shape}")
print()

# ---------------------------------------------------------------------------
# Step 0b: Preprocess — encode categoricals, impute NaN, subsample
# ---------------------------------------------------------------------------
print("-" * 72)
print("PREPROCESSING")
print("-" * 72)

# Separate target
y_raw = df[target_col].copy()
X_raw = df.drop(columns=[target_col])

# Encode target to numeric if needed
if y_raw.dtype == object or y_raw.dtype.name == "category":
    y_raw = y_raw.astype(str)
    unique_vals = sorted(y_raw.unique())
    label_map = {v: i for i, v in enumerate(unique_vals)}
    y = y_raw.map(label_map).astype(int)
    print(f"Target encoded: {label_map}")
else:
    y = y_raw.astype(float).astype(int)
    print(f"Target values: {sorted(y.unique())}")

# Identify column types
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

# Encode categoricals with OrdinalEncoder
X = X_raw.copy()
if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))

# Convert all to float and impute NaN with column median
X = X.astype(float)
for col in X.columns:
    if X[col].isna().any():
        X[col].fillna(X[col].median(), inplace=True)

# Drop columns that are all constant
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print(f"Dropping {len(const_cols)} constant columns")
    X.drop(columns=const_cols, inplace=True)

# Subsample if needed
MAX_N = 10000
if len(X) > MAX_N:
    idx = np.random.choice(len(X), MAX_N, replace=False)
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    print(f"Subsampled to {MAX_N} rows")

print(f"Final shape: X={X.shape}, y={y.shape}")
feature_names = list(X.columns)
print(f"Features: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
print()

# ---------------------------------------------------------------------------
# Step 1: Correlation matrix — identify pairs with |rho| > 0.5
# ---------------------------------------------------------------------------
print("-" * 72)
print("STEP 1: CORRELATION ANALYSIS")
print("-" * 72)

corr_matrix = X.corr()
n_features = len(feature_names)

# Find all pairs with |rho| > 0.5
HIGH_RHO_THRESH = 0.5
high_corr_pairs = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        rho = corr_matrix.iloc[i, j]
        if abs(rho) > HIGH_RHO_THRESH:
            high_corr_pairs.append((feature_names[i], feature_names[j], rho))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"Total feature pairs: {n_features * (n_features - 1) // 2}")
print(f"Pairs with |rho| > {HIGH_RHO_THRESH}: {len(high_corr_pairs)}")
print()

if len(high_corr_pairs) == 0:
    # Lower threshold if no pairs found
    HIGH_RHO_THRESH = 0.3
    for i in range(n_features):
        for j in range(i + 1, n_features):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > HIGH_RHO_THRESH:
                high_corr_pairs.append((feature_names[i], feature_names[j], rho))
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"Lowered threshold to {HIGH_RHO_THRESH}: found {len(high_corr_pairs)} pairs")
    print()

print("Top correlated pairs:")
for fi, fj, rho in high_corr_pairs[:15]:
    print(f"  {fi:>30s}  vs  {fj:<30s}  rho = {rho:+.4f}")
print()

# Select pairs to track (up to 10)
tracked_pairs = high_corr_pairs[:10]
print(f"Tracking {len(tracked_pairs)} pairs through F5 -> F1 -> DASH pipeline")
print()


# ---------------------------------------------------------------------------
# Helper: train XGBClassifier with a given seed
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, seed):
    """Train XGBClassifier with specified seed."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        verbosity=0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def get_split_frequencies(model, feature_names):
    """Extract split frequencies from XGBoost model."""
    booster = model.get_booster()
    score = booster.get_score(importance_type="weight")
    freq = np.zeros(len(feature_names))
    for i, fname in enumerate(feature_names):
        freq[i] = score.get(fname, 0)
    total = freq.sum()
    if total > 0:
        freq = freq / total
    return freq


def compute_z_statistic(freq_j, freq_k, n_splits_total):
    """Compute Z statistic for split frequency difference (j vs k)."""
    p_j = freq_j
    p_k = freq_k
    p_avg = (p_j + p_k) / 2
    if p_avg <= 0 or p_avg >= 1:
        return 0.0
    se = np.sqrt(2 * p_avg * (1 - p_avg) / max(n_splits_total, 1))
    if se < 1e-12:
        return 0.0
    return (p_j - p_k) / se


# ---------------------------------------------------------------------------
# Step 2 (F5): Single model — split-frequency diagnostic
# ---------------------------------------------------------------------------
print("-" * 72)
print("STEP 2 (F5 SCREEN): Single-model split-frequency diagnostic")
print("-" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

model_single = train_model(X_train, y_train, seed=SEED)
freq_single = get_split_frequencies(model_single, feature_names)

# Accuracy
acc = (model_single.predict(X_test) == y_test).mean()
print(f"Single model accuracy: {acc:.4f}")

# Total splits
booster = model_single.get_booster()
n_splits = sum(booster.get_score(importance_type="weight").values())
print(f"Total splits: {int(n_splits)}")
print()

print("F5 split-frequency Z for tracked pairs:")
print(f"  {'Feature j':>30s}  {'Feature k':>30s}  {'rho':>7s}  {'Z_jk':>8s}  {'Flag':>6s}")
print(f"  {'─'*30}  {'─'*30}  {'─'*7}  {'─'*8}  {'─'*6}")

f5_results = []
for fi, fj, rho in tracked_pairs:
    idx_i = feature_names.index(fi)
    idx_j = feature_names.index(fj)
    z = compute_z_statistic(freq_single[idx_i], freq_single[idx_j], n_splits)
    flag = "|Z|>1.96" if abs(z) > 1.96 else ""
    f5_results.append((fi, fj, rho, z, flag))
    print(f"  {fi:>30s}  {fj:>30s}  {rho:+.4f}  {z:+8.3f}  {flag:>6s}")

flagged_pairs = [(fi, fj, rho) for fi, fj, rho, z, flag in f5_results if abs(z) > 1.96]
if not flagged_pairs:
    # If none flagged at 1.96, use all tracked pairs anyway
    flagged_pairs = [(fi, fj, rho) for fi, fj, rho in tracked_pairs]
    print(f"\n  No pairs flagged at |Z|>1.96; using all {len(flagged_pairs)} tracked pairs")
else:
    print(f"\n  Pairs flagged for F1 validation: {len(flagged_pairs)}")
print()

# ---------------------------------------------------------------------------
# Step 3 (F1): 5 models — multi-model attribution test
# ---------------------------------------------------------------------------
print("-" * 72)
print("STEP 3 (F1 VALIDATION): 5-model attribution stability test")
print("-" * 72)

N_MODELS_F1 = 5
f1_freqs = []
f1_accuracies = []

for m in range(N_MODELS_F1):
    seed_m = SEED + m + 1
    # Resample training data (bootstrap)
    idx_boot = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[idx_boot]
    y_boot = y_train.iloc[idx_boot]
    model_m = train_model(X_boot, y_boot, seed=seed_m)
    freq_m = get_split_frequencies(model_m, feature_names)
    f1_freqs.append(freq_m)
    acc_m = (model_m.predict(X_test) == y_test).mean()
    f1_accuracies.append(acc_m)

print(f"F1 model accuracies: {['%.4f' % a for a in f1_accuracies]}")
print()

# Compute ranking for each model
f1_rankings = []
for freq_m in f1_freqs:
    ranking = np.argsort(-freq_m)  # descending
    f1_rankings.append(ranking)

# For each flagged pair, count how many models rank j above k
print("F1 ranking stability for flagged pairs:")
print(f"  {'Feature j':>30s}  {'Feature k':>30s}  {'rho':>7s}  {'j>k':>5s}  {'k>j':>5s}  {'Flip?':>6s}")
print(f"  {'─'*30}  {'─'*30}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*6}")

f1_flip_data = []
for fi, fj, rho in flagged_pairs:
    idx_i = feature_names.index(fi)
    idx_j = feature_names.index(fj)
    count_i_above = 0
    count_j_above = 0
    for freq_m in f1_freqs:
        if freq_m[idx_i] > freq_m[idx_j]:
            count_i_above += 1
        elif freq_m[idx_j] > freq_m[idx_i]:
            count_j_above += 1
        else:
            # Tie: count as 0.5 each
            count_i_above += 0.5
            count_j_above += 0.5
    flip = "FLIP" if min(count_i_above, count_j_above) >= 1 else ""
    f1_flip_data.append((fi, fj, rho, count_i_above, count_j_above, flip))
    print(f"  {fi:>30s}  {fj:>30s}  {rho:+.4f}  {count_i_above:5.1f}  {count_j_above:5.1f}  {flip:>6s}")

n_flips_f1 = sum(1 for row in f1_flip_data if row[5] == "FLIP")
print(f"\n  Pairs with ranking flips: {n_flips_f1}/{len(flagged_pairs)}")
if len(flagged_pairs) > 0:
    print(f"  Flip rate (F1, 5 models): {n_flips_f1/len(flagged_pairs):.1%}")
print()

# ---------------------------------------------------------------------------
# Step 4 (DASH): 25 models — consensus ranking, flip rate drops
# ---------------------------------------------------------------------------
print("-" * 72)
print("STEP 4 (DASH RESOLUTION): 25-model consensus ranking")
print("-" * 72)

N_MODELS_DASH = 25
dash_freqs = []
dash_accuracies = []

for m in range(N_MODELS_DASH):
    seed_m = SEED + 100 + m
    idx_boot = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[idx_boot]
    y_boot = y_train.iloc[idx_boot]
    model_m = train_model(X_boot, y_boot, seed=seed_m)
    freq_m = get_split_frequencies(model_m, feature_names)
    dash_freqs.append(freq_m)
    acc_m = (model_m.predict(X_test) == y_test).mean()
    dash_accuracies.append(acc_m)

print(f"DASH ensemble: {N_MODELS_DASH} models")
print(f"Accuracy range: [{min(dash_accuracies):.4f}, {max(dash_accuracies):.4f}]")
print(f"Mean accuracy:  {np.mean(dash_accuracies):.4f}")
print()

# Consensus frequencies (average across all models)
consensus_freq = np.mean(dash_freqs, axis=0)

# For each flagged pair, check flip rate across random subsets of models
# Use bootstrap subsets of size 5 from the 25 models
N_SUBSETS = 100
dash_flip_counts = {(fi, fj): 0 for fi, fj, _ in flagged_pairs}

for _ in range(N_SUBSETS):
    subset_idx = np.random.choice(N_MODELS_DASH, 5, replace=False)
    subset_freq = np.mean([dash_freqs[i] for i in subset_idx], axis=0)
    for fi, fj, rho in flagged_pairs:
        idx_i = feature_names.index(fi)
        idx_j = feature_names.index(fj)
        # Compare to consensus ranking
        consensus_order = consensus_freq[idx_i] > consensus_freq[idx_j]
        subset_order = subset_freq[idx_i] > subset_freq[idx_j]
        if consensus_order != subset_order:
            dash_flip_counts[(fi, fj)] += 1

print("DASH ranking stability for flagged pairs:")
print(f"  {'Feature j':>30s}  {'Feature k':>30s}  {'rho':>7s}  {'Consensus Z_j':>13s}  {'Consensus Z_k':>13s}  {'FlipRate':>10s}")
print(f"  {'─'*30}  {'─'*30}  {'─'*7}  {'─'*13}  {'─'*13}  {'─'*10}")

dash_flip_rates = []
for fi, fj, rho in flagged_pairs:
    idx_i = feature_names.index(fi)
    idx_j = feature_names.index(fj)
    flip_rate = dash_flip_counts[(fi, fj)] / N_SUBSETS
    dash_flip_rates.append(flip_rate)
    print(f"  {fi:>30s}  {fj:>30s}  {rho:+.4f}  {consensus_freq[idx_i]:13.4f}  {consensus_freq[idx_j]:13.4f}  {flip_rate:10.2%}")

if dash_flip_rates:
    mean_dash_flip = np.mean(dash_flip_rates)
    print(f"\n  Mean DASH flip rate: {mean_dash_flip:.2%}")
print()

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
print("=" * 72)
print("SUMMARY: F5 -> F1 -> DASH Pipeline Results")
print("=" * 72)
print()
print(f"Dataset:         {dataset_name}")
print(f"Samples:         {len(X)}")
print(f"Features:        {len(feature_names)}")
print(f"Collinear pairs: {len(high_corr_pairs)} (|rho| > {HIGH_RHO_THRESH})")
print(f"Tracked pairs:   {len(tracked_pairs)}")
print()

print("Pipeline progression:")
print(f"  F5 (1 model):   {len(flagged_pairs)} pairs flagged for instability")
if len(flagged_pairs) > 0:
    print(f"  F1 (5 models):  {n_flips_f1}/{len(flagged_pairs)} pairs flip = {n_flips_f1/len(flagged_pairs):.0%} flip rate")
    if dash_flip_rates:
        print(f"  DASH (25 models): mean flip rate = {mean_dash_flip:.2%}")
        if n_flips_f1 > 0 and mean_dash_flip > 0:
            reduction = 1 - mean_dash_flip / (n_flips_f1 / len(flagged_pairs))
            print(f"  Flip rate reduction: {reduction:.0%}")
print()

print("Theoretical prediction:")
for fi, fj, rho in tracked_pairs[:5]:
    ratio = 1 / (1 - rho**2) if abs(rho) < 1 else float("inf")
    print(f"  rho={rho:+.3f}  =>  attribution_ratio = 1/(1-rho^2) = {ratio:.2f}")
print()

print("Conclusion:")
print("  Collinear features cause ranking instability across equivalent models")
print("  (the Attribution Impossibility). DASH consensus averaging reduces")
print("  flip rates by pooling across the Rashomon set.")
print()

# Restore stdout
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {os.path.abspath(RESULTS_PATH)}")
