#!/usr/bin/env python3
"""
Phase 3 Task 5: Continuous Symmetry Empirical — dim(V^G) vs task complexity
============================================================================
Investigates how the number of stable CCA dimensions relates to:
  (A) number of classes k in MNIST
  (B) hidden layer size

Key question: Is dim(V^G) ≈ k (linear relationship)?
"""

import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIR / "results_continuous_symmetry.json"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_FILE = FIGURES_DIR / "continuous_symmetry.pdf"

# ---------------------------------------------------------------------------
# CCA computation (SVD-based, numerically stable)
# ---------------------------------------------------------------------------
def compute_cca_correlations(H_i, H_j, hidden):
    """SVD-based CCA between two activation matrices."""
    H_i = H_i - H_i.mean(axis=0)
    H_j = H_j - H_j.mean(axis=0)
    n = H_i.shape[0]
    reg = 1e-6 * np.eye(hidden)
    C_ii = H_i.T @ H_i / n + reg
    C_jj = H_j.T @ H_j / n + reg
    C_ij = H_i.T @ H_j / n
    U_i, S_i, _ = np.linalg.svd(C_ii)
    W_i = U_i @ np.diag(1.0 / np.sqrt(S_i))
    U_j, S_j, _ = np.linalg.svd(C_jj)
    W_j = U_j @ np.diag(1.0 / np.sqrt(S_j))
    T = W_i.T @ C_ij @ W_j
    _, corrs, _ = np.linalg.svd(T)
    return np.clip(corrs, 0, 1)


def count_stable_dims(corrs, threshold):
    """Count dimensions with CCA correlation above threshold."""
    return int(np.sum(corrs > threshold))


# ---------------------------------------------------------------------------
# Load MNIST once
# ---------------------------------------------------------------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist['data'].astype(np.float32) / 255.0
y_all = mnist['target'].astype(int)

# Use a fixed train/test split
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_all))
X_all = X_all[indices]
y_all = y_all[indices]

N_TRAIN = 5000
N_TEST = 1000
N_MODELS = 5
THRESHOLDS = [0.99, 0.95, 0.90]

# ---------------------------------------------------------------------------
# Part A: Vary number of classes k, fixed hidden_size=128
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART A: dim(V^G) vs number of classes k")
print("=" * 60)

HIDDEN_A = 128
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

results_part_a = {}

for k in K_VALUES:
    print(f"\n--- k = {k} classes (hidden={HIDDEN_A}) ---")

    # Filter to first k digits
    mask_all = y_all < k
    X_filt = X_all[mask_all]
    y_filt = y_all[mask_all]

    X_train = X_filt[:N_TRAIN]
    y_train = y_filt[:N_TRAIN]
    X_test = X_filt[N_TRAIN:N_TRAIN + N_TEST]
    y_test = y_filt[N_TRAIN:N_TRAIN + N_TEST]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train N_MODELS MLPs
    activations = []
    accuracies = []
    for i in range(N_MODELS):
        clf = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_A,),
            max_iter=200,
            random_state=42 + i,
            early_stopping=False,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)
        print(f"  Model {i}: accuracy = {acc:.4f}")

        # Extract hidden activations: h = relu(X_test @ W1 + b1)
        W1 = clf.coefs_[0]     # (784, hidden)
        b1 = clf.intercepts_[0]  # (hidden,)
        H = np.maximum(0, X_test @ W1 + b1)  # ReLU
        activations.append(H)

    # Compute CCA between all model pairs
    pair_corrs = []
    for i_idx, j_idx in combinations(range(N_MODELS), 2):
        corrs = compute_cca_correlations(activations[i_idx], activations[j_idx], HIDDEN_A)
        pair_corrs.append(corrs)

    # Average CCA correlations across pairs
    mean_corrs = np.mean(pair_corrs, axis=0)

    # Count stable dims at each threshold
    stable_counts = {}
    for thresh in THRESHOLDS:
        # Count per-pair, then take mean
        per_pair = [count_stable_dims(c, thresh) for c in pair_corrs]
        stable_counts[f"{thresh}"] = {
            "mean": float(np.mean(per_pair)),
            "std": float(np.std(per_pair)),
            "min": int(np.min(per_pair)),
            "max": int(np.max(per_pair)),
            "per_pair": per_pair,
        }
        print(f"  Stable dims (>{thresh}): mean={np.mean(per_pair):.1f}, "
              f"std={np.std(per_pair):.1f}, range=[{np.min(per_pair)}, {np.max(per_pair)}]")

    results_part_a[str(k)] = {
        "k": k,
        "hidden_size": HIDDEN_A,
        "n_models": N_MODELS,
        "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "stable_dims": stable_counts,
        "mean_cca_spectrum": mean_corrs.tolist(),
    }

# ---------------------------------------------------------------------------
# Part B: Vary hidden size, fixed k=10
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: dim(V^G) vs hidden_size (k=10)")
print("=" * 60)

K_B = 10
HIDDEN_SIZES = [32, 64, 128, 256]

# Full MNIST (all 10 digits)
mask_all = y_all < K_B
X_filt = X_all[mask_all]
y_filt = y_all[mask_all]
X_train_b = X_filt[:N_TRAIN]
y_train_b = y_filt[:N_TRAIN]
X_test_b = X_filt[N_TRAIN:N_TRAIN + N_TEST]
y_test_b = y_filt[N_TRAIN:N_TRAIN + N_TEST]

results_part_b = {}

for hidden in HIDDEN_SIZES:
    print(f"\n--- hidden_size = {hidden} (k={K_B}) ---")

    activations = []
    accuracies = []
    for i in range(N_MODELS):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            max_iter=200,
            random_state=42 + i,
            early_stopping=False,
        )
        clf.fit(X_train_b, y_train_b)
        acc = clf.score(X_test_b, y_test_b)
        accuracies.append(acc)
        print(f"  Model {i}: accuracy = {acc:.4f}")

        W1 = clf.coefs_[0]
        b1 = clf.intercepts_[0]
        H = np.maximum(0, X_test_b @ W1 + b1)
        activations.append(H)

    pair_corrs = []
    for i_idx, j_idx in combinations(range(N_MODELS), 2):
        corrs = compute_cca_correlations(activations[i_idx], activations[j_idx], hidden)
        pair_corrs.append(corrs)

    mean_corrs = np.mean(pair_corrs, axis=0)

    stable_counts = {}
    for thresh in THRESHOLDS:
        per_pair = [count_stable_dims(c, thresh) for c in pair_corrs]
        stable_counts[f"{thresh}"] = {
            "mean": float(np.mean(per_pair)),
            "std": float(np.std(per_pair)),
            "min": int(np.min(per_pair)),
            "max": int(np.max(per_pair)),
            "per_pair": per_pair,
        }
        print(f"  Stable dims (>{thresh}): mean={np.mean(per_pair):.1f}, "
              f"std={np.std(per_pair):.1f}, range=[{np.min(per_pair)}, {np.max(per_pair)}]")

    results_part_b[str(hidden)] = {
        "hidden_size": hidden,
        "k": K_B,
        "n_models": N_MODELS,
        "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "stable_dims": stable_counts,
        "mean_cca_spectrum": mean_corrs.tolist(),
    }

# ---------------------------------------------------------------------------
# Analysis: Regression of dim(V^G) vs k
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

k_vals = np.array(K_VALUES, dtype=float)
dim_vals_99 = np.array([results_part_a[str(k)]["stable_dims"]["0.99"]["mean"] for k in K_VALUES])
dim_vals_95 = np.array([results_part_a[str(k)]["stable_dims"]["0.95"]["mean"] for k in K_VALUES])
dim_vals_90 = np.array([results_part_a[str(k)]["stable_dims"]["0.9"]["mean"] for k in K_VALUES])

# Linear regression: dim(V^G) = slope * k + intercept
slope_99, intercept_99, r_99, p_99, se_99 = stats.linregress(k_vals, dim_vals_99)
slope_95, intercept_95, r_95, p_95, se_95 = stats.linregress(k_vals, dim_vals_95)
slope_90, intercept_90, r_90, p_90, se_90 = stats.linregress(k_vals, dim_vals_90)

print(f"\nLinear regression: dim(V^G) = slope * k + intercept")
print(f"  Threshold 0.99: slope={slope_99:.3f}, intercept={intercept_99:.3f}, R²={r_99**2:.4f}, p={p_99:.2e}")
print(f"  Threshold 0.95: slope={slope_95:.3f}, intercept={intercept_95:.3f}, R²={r_95**2:.4f}, p={p_95:.2e}")
print(f"  Threshold 0.90: slope={slope_90:.3f}, intercept={intercept_90:.3f}, R²={r_90**2:.4f}, p={p_90:.2e}")

# Full table
print(f"\n{'k':>3} {'dim@0.99':>10} {'dim@0.95':>10} {'dim@0.90':>10} {'accuracy':>10}")
print("-" * 48)
for k in K_VALUES:
    d99 = results_part_a[str(k)]["stable_dims"]["0.99"]["mean"]
    d95 = results_part_a[str(k)]["stable_dims"]["0.95"]["mean"]
    d90 = results_part_a[str(k)]["stable_dims"]["0.9"]["mean"]
    acc = results_part_a[str(k)]["mean_accuracy"]
    print(f"{k:>3} {d99:>10.1f} {d95:>10.1f} {d90:>10.1f} {acc:>10.4f}")

print(f"\n{'hidden':>6} {'dim@0.99':>10} {'dim@0.95':>10} {'dim@0.90':>10} {'accuracy':>10}")
print("-" * 52)
for h in HIDDEN_SIZES:
    d99 = results_part_b[str(h)]["stable_dims"]["0.99"]["mean"]
    d95 = results_part_b[str(h)]["stable_dims"]["0.95"]["mean"]
    d90 = results_part_b[str(h)]["stable_dims"]["0.9"]["mean"]
    acc = results_part_b[str(h)]["mean_accuracy"]
    print(f"{h:>6} {d99:>10.1f} {d95:>10.1f} {d90:>10.1f} {acc:>10.4f}")

# Honest assessment
print("\n=== HONEST ASSESSMENT ===")
if slope_99 > 0.5 and r_99**2 > 0.8:
    print(f"dim(V^G) at 0.99 threshold has a positive relationship with k "
          f"(slope={slope_99:.2f}, R²={r_99**2:.3f})")
    if abs(slope_99 - 1.0) < 0.3:
        print("The slope is close to 1, consistent with dim(V^G) ≈ k.")
    else:
        print(f"However, the slope ({slope_99:.2f}) deviates from 1.0.")
        print(f"The relationship is better described as dim(V^G) ≈ {slope_99:.2f}*k + {intercept_99:.1f}")
else:
    print(f"The linear relationship is weak (slope={slope_99:.2f}, R²={r_99**2:.3f})")
    print("dim(V^G) ≈ k does NOT hold cleanly.")

# Check if the k=10 match was coincidence
if abs(dim_vals_99[0] - K_VALUES[0]) > 2:
    print(f"\nNote: at k=2, dim(V^G)={dim_vals_99[0]:.1f} (predicted 2) — large offset")
    print("This suggests a floor effect: even binary tasks have multiple stable dims.")

# ---------------------------------------------------------------------------
# Figure: 3-panel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: dim(V^G) vs k with regression line
ax = axes[0]
ax.plot(k_vals, dim_vals_99, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Observed (>0.99)')
ax.plot(k_vals, slope_99 * k_vals + intercept_99, '--', color='#1f77b4', alpha=0.5,
        label=f'Fit: {slope_99:.2f}k + {intercept_99:.1f} (R²={r_99**2:.3f})')
ax.plot(k_vals, k_vals, ':', color='gray', alpha=0.5, label='dim = k (prediction)')
ax.set_xlabel('Number of classes (k)', fontsize=12)
ax.set_ylabel('dim(V^G) at threshold 0.99', fontsize=12)
ax.set_title('A: Stable CCA dims vs task complexity', fontsize=13)
ax.legend(fontsize=9)
ax.set_xticks(K_VALUES)

# Panel B: dim(V^G) vs hidden_size
ax = axes[1]
h_vals = np.array(HIDDEN_SIZES, dtype=float)
dim_h_99 = np.array([results_part_b[str(h)]["stable_dims"]["0.99"]["mean"] for h in HIDDEN_SIZES])
dim_h_95 = np.array([results_part_b[str(h)]["stable_dims"]["0.95"]["mean"] for h in HIDDEN_SIZES])
dim_h_90 = np.array([results_part_b[str(h)]["stable_dims"]["0.9"]["mean"] for h in HIDDEN_SIZES])

ax.plot(h_vals, dim_h_99, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='>0.99')
ax.plot(h_vals, dim_h_95, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='>0.95')
ax.plot(h_vals, dim_h_90, '^-', color='#2ca02c', linewidth=2, markersize=8, label='>0.90')
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='k=10')
ax.set_xlabel('Hidden layer size', fontsize=12)
ax.set_ylabel('dim(V^G)', fontsize=12)
ax.set_title('B: Stable dims vs hidden size (k=10)', fontsize=13)
ax.legend(fontsize=9)
ax.set_xticks(HIDDEN_SIZES)

# Panel C: CCA spectra overlay for k=2, 5, 10
ax = axes[2]
for k_show, color, ls in [(2, '#d62728', '-'), (5, '#9467bd', '--'), (10, '#1f77b4', '-.')]:
    spectrum = np.array(results_part_a[str(k_show)]["mean_cca_spectrum"])
    ax.plot(range(1, len(spectrum) + 1), spectrum, ls, color=color,
            linewidth=2, label=f'k={k_show}')

ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.3, label='0.99 threshold')
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.3, label='0.95 threshold')
ax.set_xlabel('CCA dimension', fontsize=12)
ax.set_ylabel('Mean CCA correlation', fontsize=12)
ax.set_title('C: CCA spectra by task complexity', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0, 50)  # zoom to first 50 dims

plt.tight_layout()
fig.savefig(str(FIGURE_FILE), bbox_inches='tight', dpi=300)
print(f"\nSaved figure: {FIGURE_FILE}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "experiment": "continuous_symmetry_empirical",
    "description": "dim(V^G) vs task complexity (k) and hidden size for MNIST MLPs",
    "config": {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_models": N_MODELS,
        "thresholds": THRESHOLDS,
        "max_iter": 200,
    },
    "part_a_vary_k": {
        "hidden_size": HIDDEN_A,
        "results": results_part_a,
        "regression_099": {
            "slope": float(slope_99),
            "intercept": float(intercept_99),
            "r_squared": float(r_99**2),
            "p_value": float(p_99),
            "std_err": float(se_99),
        },
        "regression_095": {
            "slope": float(slope_95),
            "intercept": float(intercept_95),
            "r_squared": float(r_95**2),
            "p_value": float(p_95),
        },
        "regression_090": {
            "slope": float(slope_90),
            "intercept": float(intercept_90),
            "r_squared": float(r_90**2),
            "p_value": float(p_90),
        },
    },
    "part_b_vary_hidden": {
        "k": K_B,
        "results": results_part_b,
    },
    "_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
}

with open(str(RESULTS_FILE), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {RESULTS_FILE}")
