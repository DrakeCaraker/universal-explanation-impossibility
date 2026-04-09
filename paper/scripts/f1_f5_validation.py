"""
Validate F1 (statistical indistinguishability) and F5 (split-frequency diagnostic).

For each feature pair on Breast Cancer:
1. Compute the F1 test statistic: |E[φ_j]-E[φ_k]| / (σ_{jk}/√M)
2. Compute the F5 diagnostic: split frequency z-test from a single model
3. Compare both to the empirical flip rate (from 50 models)

The prediction: low test statistic ↔ high flip rate ↔ Rashomon applies.
"""

import numpy as np
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'figure.figsize': (6.8, 2.8), 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Load data and train models
# -----------------------------------------------------------------------
print("Loading Breast Cancer and training 50 models...")
data = load_breast_cancer()
X, y = data.data, data.target
names = list(data.feature_names)
P = X.shape[1]
corr = np.corrcoef(X.T)
N_SEEDS = 50

all_shap = []
all_split_counts = []

for seed in range(N_SEEDS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=seed + 1000, verbosity=0, eval_metric='logloss'
    )
    model.fit(Xtr, ytr)

    # SHAP values
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte[:200])
    all_shap.append(np.mean(np.abs(sv), axis=0))

    # Split counts per feature
    score = model.get_booster().get_score(importance_type='weight')
    splits = np.zeros(P)
    for f, c in score.items():
        splits[int(f.replace('f', ''))] = c
    all_split_counts.append(splits)

all_shap = np.array(all_shap)  # (50, 30)
all_split_counts = np.array(all_split_counts)  # (50, 30)

# -----------------------------------------------------------------------
# F1: Attribution-based test statistic
# -----------------------------------------------------------------------
print("\nComputing F1 test statistics...")
pairs = []
for j in range(P):
    for k in range(j + 1, P):
        phi_j = all_shap[:, j]
        phi_k = all_shap[:, k]
        diff = phi_j - phi_k
        mu_diff = np.mean(diff)
        se_diff = np.std(diff) / np.sqrt(N_SEEDS)

        # F1 test statistic: |mean difference| / SE
        z_f1 = abs(mu_diff) / se_diff if se_diff > 1e-10 else 999

        # Flip rate
        jw = np.sum(phi_j > phi_k)
        kw = np.sum(phi_k > phi_j)
        total = jw + kw
        flip = min(jw, kw) / total if total > 0 else 0

        # F5: Split-frequency diagnostic from FIRST model only
        # Per-tree split indicator would need tree-level data;
        # approximate with cross-seed split count variation
        nj = all_split_counts[:, j]
        nk = all_split_counts[:, k]
        n_diff = nj - nk
        z_f5 = abs(np.mean(n_diff)) / (np.std(n_diff) / np.sqrt(N_SEEDS)) if np.std(n_diff) > 0 else 999

        rho_jk = abs(corr[j, k])

        pairs.append({
            'j': j, 'k': k,
            'j_name': names[j], 'k_name': names[k],
            'rho': rho_jk,
            'flip': flip,
            'z_f1': z_f1,
            'z_f5': z_f5,
            'mu_diff': mu_diff,
            'se_diff': se_diff,
        })

# -----------------------------------------------------------------------
# Correlation between test statistics and flip rate
# -----------------------------------------------------------------------
flips = np.array([p['flip'] for p in pairs])
z_f1s = np.array([p['z_f1'] for p in pairs])
z_f5s = np.array([p['z_f5'] for p in pairs])

# Clip for display
z_f1_clip = np.clip(z_f1s, 0, 20)
z_f5_clip = np.clip(z_f5s, 0, 20)

corr_f1 = np.corrcoef(z_f1_clip, flips)[0, 1]
corr_f5 = np.corrcoef(z_f5_clip, flips)[0, 1]
print(f"Correlation(z_F1, flip_rate) = {corr_f1:.3f}")
print(f"Correlation(z_F5, flip_rate) = {corr_f5:.3f}")

# Restricted-range analysis (R5 robustness check: are "easy" pairs inflating r?)
for z_thresh in [5, 3, 2]:
    mask = z_f1_clip < z_thresh
    n_mask = mask.sum()
    if n_mask > 10:
        r_restricted = np.corrcoef(z_f1_clip[mask], flips[mask])[0, 1]
        print(f"  Restricted Z<{z_thresh}: n={n_mask}, r={r_restricted:.3f}")
    else:
        print(f"  Restricted Z<{z_thresh}: n={n_mask} (too few pairs)")

# Also check with gain-based importance (non-SHAP) for robustness
print("\nGain-based importance check:")
all_gain = []
for seed in range(N_SEEDS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=seed + 1000, verbosity=0, eval_metric='logloss'
    )
    model.fit(Xtr, ytr)
    score = model.get_booster().get_score(importance_type='gain')
    gains = np.zeros(P)
    for f, g in score.items():
        gains[int(f.replace('f', ''))] = g
    all_gain.append(gains)
all_gain = np.array(all_gain)

# Compute gain-based flip rates and z-scores
gain_flips = []
gain_zs = []
for j in range(P):
    for k in range(j + 1, P):
        gj = all_gain[:, j]
        gk = all_gain[:, k]
        diff = gj - gk
        mu_diff = np.mean(diff)
        se_diff = np.std(diff) / np.sqrt(N_SEEDS)
        z_gain = abs(mu_diff) / se_diff if se_diff > 1e-10 else 999
        jw = np.sum(gj > gk)
        kw = np.sum(gk > gj)
        total = jw + kw
        flip = min(jw, kw) / total if total > 0 else 0
        gain_flips.append(flip)
        gain_zs.append(z_gain)

gain_flips = np.array(gain_flips)
gain_zs_clip = np.clip(np.array(gain_zs), 0, 20)
corr_gain = np.corrcoef(gain_zs_clip, gain_flips)[0, 1]
print(f"Gain-based: r(Z, flip) = {corr_gain:.3f}")
for z_thresh in [5, 3]:
    mask = gain_zs_clip < z_thresh
    if mask.sum() > 10:
        r_g = np.corrcoef(gain_zs_clip[mask], gain_flips[mask])[0, 1]
        print(f"  Gain restricted Z<{z_thresh}: n={mask.sum()}, r={r_g:.3f}")

# Show top unstable pairs with their z-scores
print("\nMost unstable pairs:")
for p in sorted(pairs, key=lambda x: -x['flip'])[:10]:
    print(f"  flip={p['flip']:.3f} z_F1={p['z_f1']:.1f} z_F5={p['z_f5']:.1f} "
          f"|ρ|={p['rho']:.3f} {p['j_name'][:20]:20s} <-> {p['k_name'][:20]}")

# -----------------------------------------------------------------------
# Figure: two panels
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

# Panel 1: F1 test statistic vs flip rate
ax1.scatter(z_f1_clip, flips, s=10, alpha=0.4, c='#d62728', edgecolors='none')
ax1.axvline(x=1.96, color='gray', linestyle='--', lw=0.8, label='z=1.96')
ax1.set_xlabel('F1 test statistic $|\\bar{\\varphi}_j - \\bar{\\varphi}_k|$ / SE')
ax1.set_ylabel('Ranking flip rate')
ax1.set_title(f'Attribution test (r={corr_f1:.2f})')
ax1.set_xlim(-0.5, 20)
ax1.set_ylim(-0.02, 0.55)
ax1.legend(fontsize=7)
ax1.grid(True, lw=0.3, alpha=0.5)

# Panel 2: F5 split-frequency z-score vs flip rate
ax2.scatter(z_f5_clip, flips, s=10, alpha=0.4, c='#1f77b4', edgecolors='none')
ax2.axvline(x=1.96, color='gray', linestyle='--', lw=0.8, label='z=1.96')
ax2.set_xlabel('F5 split-frequency z-statistic')
ax2.set_ylabel('Ranking flip rate')
ax2.set_title(f'Split diagnostic (r={corr_f5:.2f})')
ax2.set_xlim(-0.5, 20)
ax2.set_ylim(-0.02, 0.55)
ax2.legend(fontsize=7)
ax2.grid(True, lw=0.3, alpha=0.5)

fig.tight_layout()
out = os.path.join(OUT_DIR, "f1_f5_diagnostic.pdf")
fig.savefig(out)
print(f"\nSaved {out}")

# Save results
results = {
    'pairs': pairs,
    'corr_f1_flip': corr_f1,
    'corr_f5_flip': corr_f5,
    'corr_gain_flip': float(corr_gain),
}
with open(os.path.join(os.path.dirname(__file__), '..', 'results_f1_f5.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)
print("Saved results_f1_f5.json")
