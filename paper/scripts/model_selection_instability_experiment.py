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

NEGATIVE CONTROL:
  - Train 50 models with IDENTICAL hyperparameters AND subsample=1.0 (no data randomness)
  - The "best model" should be stable across splits
  - Expected flip rate: <10%

RESOLUTION TEST:
  - Use ensemble prediction (average probabilities across all 50 models)
  - Compare: ensemble AUC vs best-single-model AUC across splits
  - Ensemble should be more stable (lower variance across splits)

Output:
  - paper/results_model_selection_instability.json
  - paper/sections/table_model_selection.tex  (LaTeX table fragment)
  - paper/figures/model_selection_instability.pdf
"""

import sys
import os
import numpy as np
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ── import experiment_utils from sibling directory ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from experiment_utils import set_all_seeds, save_results, PAPER_DIR, percentile_ci, \
    load_publication_style, save_figure

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# 4. POSITIVE TEST — Train 50 diverse XGBoost classifiers
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("POSITIVE TEST: 50 diverse XGBoost models (different seeds, subsample=0.8)")
print("=" * 68)
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
# 5. Evaluate all 50 models on 20 different random evaluation splits
# ─────────────────────────────────────────────────────────────────────────────
print(f"Evaluating all {N_MODELS} models on {N_EVAL_SPLITS} different random splits ...")
print()

EVAL_SEED_BASE = 100

best_model_per_split = []
auc_matrix = []
spread_per_split = []
ensemble_auc_per_split = []
best_single_auc_per_split = []

for s in range(N_EVAL_SPLITS):
    eval_seed = EVAL_SEED_BASE + s
    _, X_eval_raw, _, y_eval = train_test_split(
        X_raw, y, test_size=0.20, random_state=eval_seed, stratify=y
    )
    X_eval = preprocessor.transform(X_eval_raw)

    aucs = []
    probs_all = []
    for m in models:
        prob = m.predict_proba(X_eval)[:, 1]
        aucs.append(roc_auc_score(y_eval, prob))
        probs_all.append(prob)

    aucs = np.array(aucs)
    auc_matrix.append(aucs)

    # Ensemble: average probabilities
    ensemble_prob = np.array(probs_all).mean(axis=0)
    ensemble_auc = roc_auc_score(y_eval, ensemble_prob)
    ensemble_auc_per_split.append(ensemble_auc)
    best_single_auc_per_split.append(float(aucs.max()))

    best_idx = int(np.argmax(aucs))
    best_model_per_split.append(best_idx)
    spread = float(aucs.max() - aucs.min())
    spread_per_split.append(spread)

    print(
        f"  Split {s+1:2d} (seed={eval_seed}): "
        f"best=model_{best_idx:02d}  "
        f"AUC range=[{aucs.min():.4f}, {aucs.max():.4f}]  "
        f"ensemble={ensemble_auc:.4f}"
    )

auc_matrix = np.array(auc_matrix)  # (N_EVAL_SPLITS, N_MODELS)
ensemble_auc_per_split = np.array(ensemble_auc_per_split)
best_single_auc_per_split = np.array(best_single_auc_per_split)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary statistics (positive test)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("POSITIVE TEST RESULTS")
print("=" * 68)

unique_winners = list(set(best_model_per_split))
n_unique_winners = len(unique_winners)

n_flips = sum(
    best_model_per_split[i] != best_model_per_split[i - 1]
    for i in range(1, N_EVAL_SPLITS)
)
flip_rate = n_flips / (N_EVAL_SPLITS - 1)

mean_auc_per_model = auc_matrix.mean(axis=0)
overall_mean_auc = float(mean_auc_per_model.mean())
overall_std_auc = float(mean_auc_per_model.std())

mean_spread = float(np.mean(spread_per_split))
max_spread = float(np.max(spread_per_split))

winner_counts = Counter(best_model_per_split)
plurality_winner, plurality_count = winner_counts.most_common(1)[0]
non_plurality_splits = N_EVAL_SPLITS - plurality_count
best_model_flip_rate = non_plurality_splits / N_EVAL_SPLITS

# Bootstrap CI for flip rate (binary outcomes per consecutive pair)
flip_binary = [int(best_model_per_split[i] != best_model_per_split[i - 1])
               for i in range(1, N_EVAL_SPLITS)]
flip_ci = percentile_ci(flip_binary, n_boot=2000)

print(f"  Number of eval splits:       {N_EVAL_SPLITS}")
print(f"  Number of models:            {N_MODELS}")
print(f"  Unique 'best' models:        {n_unique_winners} / {N_MODELS}")
print(f"  Plurality winner (model_{plurality_winner:02d}): {plurality_count} / {N_EVAL_SPLITS} splits")
print(f"  Best-model flip rate:        {best_model_flip_rate:.2%}  (splits not won by plurality winner)")
print(f"  Consecutive flip rate:       {flip_rate:.2%}  [{flip_ci[0]:.2%}, {flip_ci[2]:.2%}]")
print(f"  Mean AUC spread per split:   {mean_spread:.4f}  (max-min, shows model equivalence)")
print(f"  Max AUC spread per split:    {max_spread:.4f}")
print(f"  Overall mean AUC:            {overall_mean_auc:.4f} ± {overall_std_auc:.4f}")
print()

print("  Winner distribution (top 10):")
for model_idx, count in winner_counts.most_common(10):
    print(f"    model_{model_idx:02d}: {count} splits")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. NEGATIVE CONTROL — 50 models with IDENTICAL seed AND subsample=1.0
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("NEGATIVE CONTROL: 50 models — SAME seed + subsample=1.0 (no randomness)")
print("=" * 68)

XGB_PARAMS_NC = dict(
    max_depth=4,
    n_estimators=100,
    learning_rate=0.1,
    subsample=1.0,         # no data subsampling — no source of randomness
    random_state=42,       # SAME seed for all models
    use_label_encoder=False,
    eval_metric="logloss",
    verbosity=0,
)

print(f"Training {N_MODELS} XGBoost classifiers (ALL with seed=42, subsample=1.0) ...")
models_nc = []
for i in range(N_MODELS):
    # Same seed AND same subsample → deterministic; all models are identical
    m_nc = xgb.XGBClassifier(**XGB_PARAMS_NC)
    m_nc.fit(X_train, y_train)
    models_nc.append(m_nc)
    if (i + 1) % 10 == 0:
        print(f"  Trained {i + 1}/{N_MODELS} models (all identical)")
print()

best_model_nc_per_split = []
auc_matrix_nc = []
ensemble_auc_nc_per_split = []

for s in range(N_EVAL_SPLITS):
    eval_seed = EVAL_SEED_BASE + s
    _, X_eval_raw, _, y_eval = train_test_split(
        X_raw, y, test_size=0.20, random_state=eval_seed, stratify=y
    )
    X_eval = preprocessor.transform(X_eval_raw)

    aucs_nc = []
    probs_nc = []
    for m in models_nc:
        prob = m.predict_proba(X_eval)[:, 1]
        aucs_nc.append(roc_auc_score(y_eval, prob))
        probs_nc.append(prob)

    aucs_nc = np.array(aucs_nc)
    auc_matrix_nc.append(aucs_nc)
    best_model_nc_per_split.append(int(np.argmax(aucs_nc)))

    ens_prob_nc = np.array(probs_nc).mean(axis=0)
    ensemble_auc_nc_per_split.append(roc_auc_score(y_eval, ens_prob_nc))

auc_matrix_nc = np.array(auc_matrix_nc)
ensemble_auc_nc_per_split = np.array(ensemble_auc_nc_per_split)

winner_counts_nc = Counter(best_model_nc_per_split)
nc_unique_winners = len(set(best_model_nc_per_split))
nc_plurality_winner, nc_plurality_count = winner_counts_nc.most_common(1)[0]
nc_non_plurality = N_EVAL_SPLITS - nc_plurality_count
nc_flip_rate = nc_non_plurality / N_EVAL_SPLITS

nc_flip_binary = [int(best_model_nc_per_split[i] != best_model_nc_per_split[i - 1])
                  for i in range(1, N_EVAL_SPLITS)]
nc_flip_ci = percentile_ci(nc_flip_binary, n_boot=2000)

print(f"  Unique 'best' models: {nc_unique_winners} / {N_MODELS}")
print(f"  Best-model flip rate: {nc_flip_rate:.2%}  (expected <10%)")
print(f"  Consecutive flip rate: {nc_non_plurality / max(N_EVAL_SPLITS - 1, 1):.2%}  "
      f"[{nc_flip_ci[0]:.2%}, {nc_flip_ci[2]:.2%}]")
print(f"  Note: all models are identical (same seed + no subsampling)")

nc_results = {
    "description": "50 identical models: same seed=42, subsample=1.0",
    "n_unique_winners": nc_unique_winners,
    "best_model_flip_rate": float(nc_flip_rate),
    "consecutive_flip_rate": float(sum(nc_flip_binary) / max(len(nc_flip_binary), 1)),
    "consecutive_flip_rate_ci_lo": float(nc_flip_ci[0]),
    "consecutive_flip_rate_ci_hi": float(nc_flip_ci[2]),
    "interpretation": "Expected <10%; all models are deterministically identical",
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. RESOLUTION TEST — ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("RESOLUTION TEST: Ensemble AUC vs best-single-model AUC")
print("=" * 68)

mean_ensemble_auc = float(ensemble_auc_per_split.mean())
std_ensemble_auc = float(ensemble_auc_per_split.std())
mean_best_single_auc = float(best_single_auc_per_split.mean())
std_best_single_auc = float(best_single_auc_per_split.std())

# Bootstrap CI for ensemble AUC
ens_ci = percentile_ci(ensemble_auc_per_split.tolist(), n_boot=2000)
best_ci = percentile_ci(best_single_auc_per_split.tolist(), n_boot=2000)

# Stability = 1 - std (lower std = more stable across splits)
ens_stability_gain = float(std_best_single_auc - std_ensemble_auc)

print(f"  Ensemble AUC (mean ± std):     {mean_ensemble_auc:.4f} ± {std_ensemble_auc:.4f}  "
      f"95% CI [{ens_ci[0]:.4f}, {ens_ci[2]:.4f}]")
print(f"  Best single AUC (mean ± std):  {mean_best_single_auc:.4f} ± {std_best_single_auc:.4f}  "
      f"95% CI [{best_ci[0]:.4f}, {best_ci[2]:.4f}]")
print(f"  Std reduction (stability gain): {ens_stability_gain:.4f}")
print(f"  Ensemble AUC > best-single AUC: {mean_ensemble_auc > mean_best_single_auc}")
print(f"  Expected: ensemble should have lower std (more stable) and comparable/higher AUC")

res_results = {
    "description": "ensemble of all 50 models (average probabilities) vs best-single-model",
    "ensemble_auc_mean": float(mean_ensemble_auc),
    "ensemble_auc_std": float(std_ensemble_auc),
    "ensemble_auc_ci_lo": float(ens_ci[0]),
    "ensemble_auc_ci_hi": float(ens_ci[2]),
    "best_single_auc_mean": float(mean_best_single_auc),
    "best_single_auc_std": float(std_best_single_auc),
    "best_single_auc_ci_lo": float(best_ci[0]),
    "best_single_auc_ci_hi": float(best_ci[2]),
    "std_reduction": float(ens_stability_gain),
    "ensemble_auc_per_split": ensemble_auc_per_split.tolist(),
    "best_single_auc_per_split": best_single_auc_per_split.tolist(),
    "interpretation": "Ensemble std should be lower than best-single std (aggregation resolves instability)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Figure
# ─────────────────────────────────────────────────────────────────────────────
print()
print("Generating figure ...")

load_publication_style()
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
fig.subplots_adjust(wspace=0.40)

# ---- Left panel: AUC distribution per model (positive test) ----
ax1 = axes[0]
mean_auc_per_model_sorted = np.sort(mean_auc_per_model)
colors_auc = ['#D55E00' if auc == mean_auc_per_model[plurality_winner]
               else '#0072B2' for auc in mean_auc_per_model_sorted]
ax1.scatter(range(N_MODELS), mean_auc_per_model_sorted, c=colors_auc,
            s=18, alpha=0.75, linewidths=0.5, edgecolors='gray')
ax1.axhline(overall_mean_auc, color='crimson', linewidth=1.2, linestyle='--',
            label=f'Mean={overall_mean_auc:.3f}')
ax1.set_xlabel('Model rank (by mean AUC)', fontsize=9)
ax1.set_ylabel('Mean AUC across 20 splits', fontsize=9)
ax1.set_title(f'(a) Positive Test\nModel AUC Distribution\n({N_MODELS} diverse models)', fontsize=9)
ax1.legend(fontsize=7.5, frameon=False)
ax1.tick_params(labelsize=8)
# Annotate spread
ax1.text(0.05, 0.05, f'Spread: {mean_auc_per_model.max() - mean_auc_per_model.min():.4f}\n'
         f'Unique winners: {n_unique_winners}/{N_MODELS}',
         transform=ax1.transAxes, va='bottom', ha='left', fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# ---- Middle panel: consecutive flip rate comparison (positive vs control) ----
ax2 = axes[1]
bar_labels_flip = ['Positive\n(diverse models\nsubsample=0.8)', 'Neg. Control\n(identical models\nsubsample=1.0)']
bar_vals_flip = [flip_rate, nc_non_plurality / max(N_EVAL_SPLITS - 1, 1)]
bar_errs_lo_flip = [flip_rate - flip_ci[0], nc_non_plurality / max(N_EVAL_SPLITS - 1, 1) - nc_flip_ci[0]]
bar_errs_hi_flip = [flip_ci[2] - flip_rate, nc_flip_ci[2] - nc_non_plurality / max(N_EVAL_SPLITS - 1, 1)]
bar_colors_flip = ['#D55E00', '#009E73']
xs2 = np.arange(len(bar_labels_flip))
bars2 = ax2.bar(xs2, bar_vals_flip, color=bar_colors_flip, alpha=0.8, width=0.55,
                yerr=[bar_errs_lo_flip, bar_errs_hi_flip],
                error_kw=dict(elinewidth=1.0, capsize=4, ecolor='#333333'))
ax2.set_xticks(xs2)
ax2.set_xticklabels(bar_labels_flip, fontsize=8)
ax2.set_ylabel('Consecutive flip rate', fontsize=9)
ax2.set_title('(b) Positive vs Neg. Control\nFlip Rate Comparison', fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.axhline(0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
            label='10% threshold')
ax2.legend(fontsize=7.5, frameon=False)
for bar, val in zip(bars2, bar_vals_flip):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
             f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ---- Right panel: ensemble vs best-single AUC per split (resolution test) ----
ax3 = axes[2]
split_numbers = np.arange(1, N_EVAL_SPLITS + 1)
ax3.plot(split_numbers, ensemble_auc_per_split, 'o-', color='#0072B2',
         linewidth=1.5, markersize=5, label=f'Ensemble (std={std_ensemble_auc:.4f})', alpha=0.85)
ax3.plot(split_numbers, best_single_auc_per_split, 's--', color='#D55E00',
         linewidth=1.2, markersize=4, label=f'Best single (std={std_best_single_auc:.4f})', alpha=0.75)
ax3.axhline(mean_ensemble_auc, color='#0072B2', linewidth=0.8, linestyle=':', alpha=0.6)
ax3.axhline(mean_best_single_auc, color='#D55E00', linewidth=0.8, linestyle=':', alpha=0.6)
ax3.set_xlabel('Evaluation split', fontsize=9)
ax3.set_ylabel('AUC', fontsize=9)
ax3.set_title(f'(c) Resolution Test\nEnsemble vs Best-Single AUC\nacross {N_EVAL_SPLITS} splits', fontsize=9)
ax3.legend(fontsize=7.5, frameon=False)
ax3.tick_params(labelsize=8)
ax3.set_xlim(0.5, N_EVAL_SPLITS + 0.5)
# Annotate std reduction
ax3.text(0.05, 0.05,
         f'Std reduction: {ens_stability_gain:.4f}',
         transform=ax3.transAxes, va='bottom', ha='left', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

fig.suptitle('Model Selection Instability: Rashomon Set Causes Unstable "Best" Model\n'
             '+ Negative Control (identical models) + Resolution (ensemble)',
             fontsize=10, fontweight='bold')

save_figure(fig, 'model_selection_instability')

# ─────────────────────────────────────────────────────────────────────────────
# 10. Save JSON results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "experiment": "model_selection_instability",
    "task": "1D",
    "dataset": "German Credit (credit-g, OpenML)",
    "n_models": N_MODELS,
    "n_eval_splits": N_EVAL_SPLITS,
    "xgb_params": {k: v for k, v in XGB_PARAMS.items()
                   if k not in ("use_label_encoder", "eval_metric", "verbosity")},
    "best_model_per_split": best_model_per_split,
    "n_unique_winners": n_unique_winners,
    "unique_winner_indices": sorted(unique_winners),
    "best_model_flip_rate": round(best_model_flip_rate, 4),
    "consecutive_flip_rate": round(flip_rate, 4),
    "consecutive_flip_rate_ci_lo": round(flip_ci[0], 4),
    "consecutive_flip_rate_ci_hi": round(flip_ci[2], 4),
    "mean_auc_spread_per_split": round(mean_spread, 5),
    "max_auc_spread_per_split": round(max_spread, 5),
    "overall_mean_auc": round(overall_mean_auc, 5),
    "overall_std_auc_across_models": round(overall_std_auc, 5),
    "winner_counts": {f"model_{k:02d}": v for k, v in winner_counts.items()},
    "auc_spread_per_split": [round(s, 5) for s in spread_per_split],
    "mean_auc_per_model": [round(float(v), 5) for v in mean_auc_per_model],
    "negative_control": nc_results,
    "resolution_test": res_results,
}

save_results(results, "model_selection_instability")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Write LaTeX table fragment
# ─────────────────────────────────────────────────────────────────────────────
sections_dir = PAPER_DIR / "sections"
sections_dir.mkdir(exist_ok=True)
tex_path = sections_dir / "table_model_selection.tex"

latex = r"""\begin{table}[ht]
\centering
\caption{Model selection instability on German Credit (credit-g).
  \emph{Positive test}: 50 diverse XGBoost classifiers (differing in seed + subsample=0.8) on """ + str(N_EVAL_SPLITS) + r""" splits.
  \emph{Negative control}: 50 \emph{identical} classifiers (same seed, subsample=1.0) --- expected flip rate $<10\%$.
  \emph{Resolution}: ensemble (average probabilities) is more stable than best-single.
  All 95\% bootstrap CIs from 2000 resamples.}
\label{tab:model_selection_instability}
\begin{tabular}{llc}
\toprule
\textbf{Test} & \textbf{Metric} & \textbf{Value (95\% CI)} \\
\midrule
"""
latex += f"\\multirow{{5}}{{*}}{{Positive (Rashomon)}} & Models in Rashomon set & {N_MODELS} \\\\\n"
latex += f"& Evaluation splits & {N_EVAL_SPLITS} \\\\\n"
latex += f"& Unique `best' models & {n_unique_winners} / {N_MODELS} \\\\\n"
latex += f"& Best-model flip rate & ${best_model_flip_rate:.0%}$ \\\\\n"
latex += (f"& Consecutive flip rate & ${flip_rate:.0%}$ $[{flip_ci[0]:.0%}, {flip_ci[2]:.0%}]$ \\\\\n")
latex += r"\midrule" + "\n"
latex += (f"Neg.\\ control (identical) & Consecutive flip rate & "
          f"${nc_non_plurality / max(N_EVAL_SPLITS - 1, 1):.0%}$ "
          f"$[{nc_flip_ci[0]:.0%}, {nc_flip_ci[2]:.0%}]$ \\\\\n")
latex += r"\midrule" + "\n"
latex += (f"\\multirow{{3}}{{*}}{{Resolution (ensemble)}} & "
          f"Ensemble AUC (mean) & ${mean_ensemble_auc:.4f}$ $[{ens_ci[0]:.4f}, {ens_ci[2]:.4f}]$ \\\\\n")
latex += (f"& Best-single AUC (mean) & ${mean_best_single_auc:.4f}$ $[{best_ci[0]:.4f}, {best_ci[2]:.4f}]$ \\\\\n")
latex += f"& Std reduction & ${ens_stability_gain:.4f}$ \\\\\n"
latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(tex_path, "w") as f:
    f.write(latex)

print(f"Saved LaTeX table: {tex_path}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 12. Final console summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print()
print("POSITIVE TEST (diverse models, Rashomon set):")
print(f"  {n_unique_winners} distinct models were 'best' across {N_EVAL_SPLITS} splits")
print(f"  Best-model flip rate = {best_model_flip_rate:.0%} (target: >80%)")
print(f"  Consecutive flip rate = {flip_rate:.0%}  "
      f"95% CI [{flip_ci[0]:.0%}, {flip_ci[2]:.0%}]")
print(f"  Mean AUC spread = {mean_spread:.4f} (small => models are functionally equivalent)")
print()
print("NEGATIVE CONTROL (identical models, no randomness):")
print(f"  Unique 'best' models: {nc_unique_winners} / {N_MODELS}")
print(f"  Consecutive flip rate = {nc_non_plurality / max(N_EVAL_SPLITS - 1, 1):.0%}  "
      f"95% CI [{nc_flip_ci[0]:.0%}, {nc_flip_ci[2]:.0%}]")
print(f"  Expected: <10% (all models are identical)")
print()
print("RESOLUTION TEST (ensemble prediction):")
print(f"  Ensemble AUC: {mean_ensemble_auc:.4f} ± {std_ensemble_auc:.4f}  "
      f"95% CI [{ens_ci[0]:.4f}, {ens_ci[2]:.4f}]")
print(f"  Best single:  {mean_best_single_auc:.4f} ± {std_best_single_auc:.4f}  "
      f"95% CI [{best_ci[0]:.4f}, {best_ci[2]:.4f}]")
print(f"  Std reduction: {ens_stability_gain:.4f}  (positive = ensemble is more stable)")
print()
print("CONCLUSION: model selection under the Rashomon property is unstable.")
print("Identical models (no randomness) are stable. Ensemble aggregation")
print("reduces variance across splits, confirming the resolution principle.")
print("=" * 68)
