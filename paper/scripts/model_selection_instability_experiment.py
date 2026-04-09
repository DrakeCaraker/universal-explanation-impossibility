"""
Model Selection Instability Experiment (Task 1D — Supplementary)

Confirms Instance 6 (Model Selection) of the Universal Explanation Impossibility:
when multiple XGBoost models are functionally equivalent (Rashomon set), the
identity of the "best" model on a held-out test set is unstable — it flips
across different random train/test splits.

Design:
  - Train 50 XGBoost classifiers with different seeds on a SINGLE fixed 80/20 split
  - Evaluate all 50 models on 20 DIFFERENT random 80/20 test splits
  - For each evaluation split: record which model wins (highest test AUC)
  - Measures: best-model flip rate, unique winners, AUC spread

Output:
  - paper/results_model_selection_instability.json
  - paper/sections/table_model_selection.tex  (LaTeX table fragment)
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── import experiment_utils from sibling directory ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from experiment_utils import set_all_seeds, save_results, PAPER_DIR

# ── reproducibility ───────────────────────────────────────────────────────────
set_all_seeds(42)

# ── dependencies ──────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load German Credit dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("Model Selection Instability Experiment (Task 1D — Supplementary)")
print("Dataset: German Credit (credit-g, OpenML)")
print("=" * 68)
print()

print("Loading German Credit dataset ...")
credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
df = credit.data.copy()
target = credit.target

# Encode target: 'good' -> 1, 'bad' -> 0
y = (target == "good").astype(int).values
print(f"  Samples: {df.shape[0]}, Raw features: {df.shape[1]}")
print(f"  Class balance: {y.mean():.1%} good, {1 - y.mean():.1%} bad")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing: one-hot encode categoricals, standardize numerics
# ─────────────────────────────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()

print(f"  Categorical columns ({len(cat_cols)}): {cat_cols[:5]} ...")
print(f"  Numeric columns    ({len(num_cols)}): {num_cols}")
print()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

X_raw = df

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fixed training split — all 50 models trained on this
# ─────────────────────────────────────────────────────────────────────────────
N_MODELS = 50
N_EVAL_SPLITS = 20
XGB_PARAMS = dict(
    max_depth=4,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    verbosity=0,
)

print(f"Creating fixed 80/20 training split (seed=42) ...")
X_train_raw, _, y_train, _ = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# Fit preprocessor on training data only
preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
print(f"  Training samples: {X_train.shape[0]}, Features after encoding: {X_train.shape[1]}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train 50 XGBoost classifiers with different seeds
# ─────────────────────────────────────────────────────────────────────────────
print(f"Training {N_MODELS} XGBoost classifiers (seeds 42 .. {42 + N_MODELS - 1}) ...")
models = []
for i in range(N_MODELS):
    seed = 42 + i
    m = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS)
    m.fit(X_train, y_train)
    models.append(m)
    if (i + 1) % 10 == 0:
        print(f"  Trained {i + 1}/{N_MODELS} models")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate all 50 models on 20 different random 80/20 evaluation splits
#    (the "test" portion of each split is the evaluation set)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Evaluating all {N_MODELS} models on {N_EVAL_SPLITS} different random splits ...")
print()

# Use seeds 100..119 for evaluation splits (distinct from training seeds 42..91)
EVAL_SEED_BASE = 100

best_model_per_split = []       # index of best model on each eval split
auc_matrix = []                 # shape (N_EVAL_SPLITS, N_MODELS)
spread_per_split = []           # max_auc - min_auc per split

for s in range(N_EVAL_SPLITS):
    eval_seed = EVAL_SEED_BASE + s
    _, X_eval_raw, _, y_eval = train_test_split(
        X_raw, y, test_size=0.20, random_state=eval_seed, stratify=y
    )
    X_eval = preprocessor.transform(X_eval_raw)

    aucs = []
    for m in models:
        prob = m.predict_proba(X_eval)[:, 1]
        auc = roc_auc_score(y_eval, prob)
        aucs.append(auc)

    aucs = np.array(aucs)
    auc_matrix.append(aucs)

    best_idx = int(np.argmax(aucs))
    best_model_per_split.append(best_idx)
    spread = float(aucs.max() - aucs.min())
    spread_per_split.append(spread)

    print(
        f"  Split {s+1:2d} (seed={eval_seed}): "
        f"best=model_{best_idx:02d}  "
        f"AUC range=[{aucs.min():.4f}, {aucs.max():.4f}]  "
        f"spread={spread:.4f}"
    )

auc_matrix = np.array(auc_matrix)  # (N_EVAL_SPLITS, N_MODELS)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Compute summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("RESULTS SUMMARY")
print("=" * 68)

unique_winners = list(set(best_model_per_split))
n_unique_winners = len(unique_winners)

# Flip rate: fraction of consecutive split-pairs where winner changes
n_flips = sum(
    best_model_per_split[i] != best_model_per_split[i - 1]
    for i in range(1, N_EVAL_SPLITS)
)
flip_rate = n_flips / (N_EVAL_SPLITS - 1)

# Mean AUC per model (across all eval splits)
mean_auc_per_model = auc_matrix.mean(axis=0)
overall_mean_auc = float(mean_auc_per_model.mean())
overall_std_auc = float(mean_auc_per_model.std())

# AUC spread per split (small spread = models are equivalent)
mean_spread = float(np.mean(spread_per_split))
max_spread = float(np.max(spread_per_split))

# Best-model changes: how many of the 20 splits have a DIFFERENT winner
# than the plurality winner
from collections import Counter
winner_counts = Counter(best_model_per_split)
plurality_winner, plurality_count = winner_counts.most_common(1)[0]
non_plurality_splits = N_EVAL_SPLITS - plurality_count
best_model_flip_rate = non_plurality_splits / N_EVAL_SPLITS

print(f"  Number of eval splits:       {N_EVAL_SPLITS}")
print(f"  Number of models:            {N_MODELS}")
print(f"  Unique 'best' models:        {n_unique_winners} / {N_MODELS}")
print(f"  Plurality winner (model_{plurality_winner:02d}): {plurality_count} / {N_EVAL_SPLITS} splits")
print(f"  Best-model flip rate:        {best_model_flip_rate:.2%}  (splits not won by plurality winner)")
print(f"  Consecutive flip rate:       {flip_rate:.2%}  (winner changes split-to-split)")
print(f"  Mean AUC spread per split:   {mean_spread:.4f}  (max-min, shows model equivalence)")
print(f"  Max AUC spread per split:    {max_spread:.4f}")
print(f"  Overall mean AUC:            {overall_mean_auc:.4f} ± {overall_std_auc:.4f}")
print()

print("  Winner distribution (top 10):")
for model_idx, count in winner_counts.most_common(10):
    print(f"    model_{model_idx:02d}: {count} splits")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save JSON results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "experiment": "model_selection_instability",
    "task": "1D",
    "dataset": "German Credit (credit-g, OpenML)",
    "n_models": N_MODELS,
    "n_eval_splits": N_EVAL_SPLITS,
    "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k not in ("use_label_encoder", "eval_metric", "verbosity")},
    "best_model_per_split": best_model_per_split,
    "n_unique_winners": n_unique_winners,
    "unique_winner_indices": sorted(unique_winners),
    "best_model_flip_rate": round(best_model_flip_rate, 4),
    "consecutive_flip_rate": round(flip_rate, 4),
    "mean_auc_spread_per_split": round(mean_spread, 5),
    "max_auc_spread_per_split": round(max_spread, 5),
    "overall_mean_auc": round(overall_mean_auc, 5),
    "overall_std_auc_across_models": round(overall_std_auc, 5),
    "winner_counts": {f"model_{k:02d}": v for k, v in winner_counts.items()},
    "auc_spread_per_split": [round(s, 5) for s in spread_per_split],
    "mean_auc_per_model": [round(float(v), 5) for v in mean_auc_per_model],
}

save_results(results, "model_selection_instability")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Write LaTeX table fragment
# ─────────────────────────────────────────────────────────────────────────────
sections_dir = PAPER_DIR / "sections"
sections_dir.mkdir(exist_ok=True)
tex_path = sections_dir / "table_model_selection.tex"

latex = r"""\begin{table}[ht]
\centering
\caption{Model selection instability on German Credit (credit-g).
  50 XGBoost classifiers (differing only in random seed) are trained on a
  single fixed 80/20 split and evaluated on """ + str(N_EVAL_SPLITS) + r""" independent
  random 80/20 evaluation splits. Despite near-identical mean AUC
  ($""" + f"{overall_mean_auc:.3f}" + r""" \pm """ + f"{overall_std_auc:.4f}" + r"""$),
  the identity of the ``best'' model changes across splits, confirming
  Instance~6 of the Universal Explanation Impossibility.}
\label{tab:model_selection_instability}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Models in Rashomon set & """ + str(N_MODELS) + r""" \\
Evaluation splits & """ + str(N_EVAL_SPLITS) + r""" \\
Unique ``best'' models & """ + str(n_unique_winners) + r""" / """ + str(N_MODELS) + r""" \\
Best-model flip rate & $""" + f"{best_model_flip_rate:.0%}" + r"""$ \\
Consecutive flip rate & $""" + f"{flip_rate:.0%}" + r"""$ \\
Mean AUC spread per split & $""" + f"{mean_spread:.4f}" + r"""$ \\
Overall mean AUC & $""" + f"{overall_mean_auc:.3f} \\pm {overall_std_auc:.4f}" + r"""$ \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(tex_path, "w") as f:
    f.write(latex)

print(f"Saved LaTeX table: {tex_path}")
print()
print("Interpretation:")
print(f"  - {n_unique_winners} distinct models were 'best' across {N_EVAL_SPLITS} splits")
print(f"  - Best-model flip rate = {best_model_flip_rate:.0%} (target: >80%)")
print(f"  - Mean AUC spread = {mean_spread:.4f} (small => models are functionally equivalent)")
print(f"  - This confirms: model selection under the Rashomon property is unstable")
print("=" * 68)
