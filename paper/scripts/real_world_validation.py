"""
Real-world validation: SHAP instability on California Housing.

Demonstrates on public data that:
1. Within-pair SHAP instability increases with feature correlation
2. DASH ensemble averaging resolves the instability
3. Low-correlation pairs remain stable without intervention

Output: paper/figures/real_world_instability.pdf
"""

import numpy as np
import os
import json

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install scikit-learn xgboost shap matplotlib")
    exit(1)

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.figsize': (3.4, 2.6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.formatter.use_mathtext': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS = 50
N_TREES = 100
M_DASH = 25  # ensemble size for DASH
N_SHAP_BACKGROUND = 200  # background samples for TreeSHAP

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading California Housing...")
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
P = X.shape[1]  # 8 features

# Compute pairwise correlations
corr_matrix = np.corrcoef(X.T)
print(f"Features: {feature_names}")
print(f"Correlation range: [{corr_matrix[np.triu_indices(P, k=1)].min():.3f}, "
      f"{corr_matrix[np.triu_indices(P, k=1)].max():.3f}]")

# ---------------------------------------------------------------------------
# Train models and compute SHAP
# ---------------------------------------------------------------------------
print(f"\nTraining {N_SEEDS} XGBoost models...")

all_shap_rankings = []  # (N_SEEDS, P) — rank of each feature per seed

for seed in range(N_SEEDS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = xgb.XGBRegressor(
        n_estimators=N_TREES,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed + 1000,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Compute mean |SHAP| per feature
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:N_SHAP_BACKGROUND])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Store rankings (higher SHAP = higher rank)
    ranking = np.argsort(np.argsort(mean_abs_shap))  # rank transform
    all_shap_rankings.append(ranking)

    if (seed + 1) % 10 == 0:
        print(f"  Seed {seed + 1}/{N_SEEDS} done")

all_shap_rankings = np.array(all_shap_rankings)  # (N_SEEDS, P)

# Also compute SHAP values (not just rankings) for DASH
print(f"\nComputing SHAP values for DASH (M={M_DASH})...")
all_shap_values = []  # (N_SEEDS, P) — mean |SHAP| per feature per seed

for seed in range(N_SEEDS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.1,
        random_state=seed + 1000, verbosity=0,
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test[:N_SHAP_BACKGROUND])
    all_shap_values.append(np.mean(np.abs(sv), axis=0))

all_shap_values = np.array(all_shap_values)  # (N_SEEDS, P)

# ---------------------------------------------------------------------------
# Compute flip rates per feature pair
# ---------------------------------------------------------------------------
print("\nComputing flip rates...")

pairs = []
for j in range(P):
    for k in range(j + 1, P):
        rho_jk = abs(corr_matrix[j, k])

        # Single-model flip rate: fraction of seed pairs where j vs k ranking flips
        n_jk_wins = 0
        n_kj_wins = 0
        for s in range(N_SEEDS):
            if all_shap_values[s, j] > all_shap_values[s, k]:
                n_jk_wins += 1
            elif all_shap_values[s, k] > all_shap_values[s, j]:
                n_kj_wins += 1

        # Flip rate = min(wins_j, wins_k) / total — measures how often the
        # minority ordering occurs. 0 = perfectly stable, 0.5 = maximally unstable
        total = n_jk_wins + n_kj_wins
        if total > 0:
            flip_rate_single = min(n_jk_wins, n_kj_wins) / total
        else:
            flip_rate_single = 0.0

        # DASH flip rate: for non-overlapping ensembles of size M_DASH
        n_ensembles = N_SEEDS // M_DASH
        dash_jk_wins = 0
        dash_kj_wins = 0
        for e in range(n_ensembles):
            start = e * M_DASH
            end = start + M_DASH
            dash_j = np.mean(all_shap_values[start:end, j])
            dash_k = np.mean(all_shap_values[start:end, k])
            if dash_j > dash_k:
                dash_jk_wins += 1
            elif dash_k > dash_j:
                dash_kj_wins += 1

        dash_total = dash_jk_wins + dash_kj_wins
        if dash_total > 0:
            flip_rate_dash = min(dash_jk_wins, dash_kj_wins) / dash_total
        else:
            flip_rate_dash = 0.0

        pairs.append({
            'j': j, 'k': k,
            'j_name': feature_names[j],
            'k_name': feature_names[k],
            'rho': rho_jk,
            'flip_single': flip_rate_single,
            'flip_dash': flip_rate_dash,
        })

        if rho_jk > 0.3:
            print(f"  {feature_names[j]:12s} ↔ {feature_names[k]:12s}: "
                  f"|ρ|={rho_jk:.3f}, flip_single={flip_rate_single:.3f}, "
                  f"flip_dash={flip_rate_dash:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nGenerating figure...")

rhos = [p['rho'] for p in pairs]
flips_single = [p['flip_single'] for p in pairs]
flips_dash = [p['flip_dash'] for p in pairs]

fig, ax = plt.subplots()

ax.scatter(rhos, flips_single, s=30, c='#d62728', alpha=0.7, zorder=3,
           label='Single model', marker='o', edgecolors='white', linewidths=0.3)
ax.scatter(rhos, flips_dash, s=30, c='#1f77b4', alpha=0.7, zorder=4,
           label=f'DASH (M={M_DASH})', marker='s', edgecolors='white', linewidths=0.3)

# Add theory line: flip rate = 0.5 for perfectly symmetric features
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5,
           label='Theory max (symmetric)')

ax.set_xlabel(r'Feature pair correlation $|\rho|$')
ax.set_ylabel('Ranking flip rate')
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(-0.02, 0.55)
ax.grid(True, linewidth=0.3, alpha=0.2, color='#cccccc')
ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "real_world_instability.pdf")
fig.savefig(out_path)
print(f"Saved {out_path}")

# Save data
results = {
    'dataset': 'California Housing',
    'n_seeds': N_SEEDS,
    'n_trees': N_TREES,
    'M_dash': M_DASH,
    'pairs': pairs,
    'feature_names': list(feature_names),
}
json_path = os.path.join(os.path.dirname(__file__), "..", "results_real_world.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved {json_path}")
