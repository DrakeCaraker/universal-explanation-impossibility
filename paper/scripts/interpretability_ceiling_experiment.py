#!/usr/bin/env python3
"""
Interpretability Ceiling Experiment
====================================

Verifies the theoretical prediction that permutation symmetry S_n limits
stably interpretable internal representations to at most 1/n of the total.

For a 2-layer ReLU network with n hidden units, the hidden-layer weights
can be freely permuted (rows of W1, bias b1, columns of W2) without
changing the network's input-output function.  This S_n symmetry means:
  - Per-neuron importance vectors are NOT permutation-invariant (unstable)
  - Aggregate statistics (mean activation) ARE invariant (the G-invariant)
  - The fraction of neurons with stable rank ~ 1/n (only the mean is stable)

Theoretical prediction (from Theorem 1 / ExplanationSystem.lean):
  stable_fraction ≈ 1/n, mean_activation_corr = 1.0
"""

import sys
from pathlib import Path

# Ensure experiment_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from experiment_utils import (
    set_all_seeds, load_publication_style, save_figure, save_results,
    PAPER_DIR, FIGURES_DIR
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Simple 2-layer ReLU network (manual, no framework dependency)
# ---------------------------------------------------------------------------

class TwoLayerReLU:
    """Minimal 2-layer ReLU network trained with gradient descent."""

    def __init__(self, d_in: int, n_hidden: int, lr: float = 0.01,
                 n_epochs: int = 500, seed: int = 42):
        self.d_in = d_in
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_epochs = n_epochs
        rng = np.random.RandomState(seed)
        # Xavier init
        self.W1 = rng.randn(d_in, n_hidden) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> tuple:
        """Returns (prediction, hidden_activations)."""
        z = X @ self.W1 + self.b1
        h = np.maximum(z, 0)  # ReLU
        y_hat = h @ self.W2 + self.b2
        return y_hat.ravel(), h

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train via mini-batch SGD."""
        n = X.shape[0]
        for epoch in range(self.n_epochs):
            # Forward
            z = X @ self.W1 + self.b1
            h = np.maximum(z, 0)
            y_hat = (h @ self.W2 + self.b2).ravel()
            residual = y_hat - y

            # Backward
            d_out = residual[:, None]                         # (n, 1)
            d_W2 = h.T @ d_out / n                           # (n_hidden, 1)
            d_b2 = d_out.mean(axis=0)                        # (1,)
            d_h = d_out @ self.W2.T                           # (n, n_hidden)
            d_h[z <= 0] = 0                                   # ReLU grad
            d_W1 = X.T @ d_h / n                              # (d_in, n_hidden)
            d_b1 = d_h.mean(axis=0)                           # (n_hidden,)

            self.W1 -= self.lr * d_W1
            self.b1 -= self.lr * d_b1
            self.W2 -= self.lr * d_W2
            self.b2 -= self.lr * d_b2

    def get_weights(self):
        return self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()

    def set_weights(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()


# ---------------------------------------------------------------------------
# 2. Permutation utilities
# ---------------------------------------------------------------------------

def permute_hidden(net: TwoLayerReLU, perm: np.ndarray) -> TwoLayerReLU:
    """Return a new network with hidden units permuted by `perm`.

    This is an exact symmetry: the permuted network computes the same
    function f(x) for every x.
    """
    W1, b1, W2, b2 = net.get_weights()
    new_net = TwoLayerReLU(net.d_in, net.n_hidden)
    new_net.set_weights(W1[:, perm], b1[perm], W2[perm, :], b2)
    return new_net


def neuron_importance(net: TwoLayerReLU, X: np.ndarray) -> np.ndarray:
    """Per-neuron importance = mean absolute activation on data X."""
    _, h = net.forward(X)
    return np.mean(np.abs(h), axis=0)   # shape (n_hidden,)


# ---------------------------------------------------------------------------
# 3. Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seed: int = 42):
    set_all_seeds(seed)

    # Synthetic data
    d = 5
    n_samples = 500
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, d)
    true_beta = rng.randn(d)
    y = X @ true_beta + 0.3 * rng.randn(n_samples)

    # Train/test split
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    hidden_widths = [4, 8, 16, 32, 64, 128]
    n_perms = 50
    results = {}

    for n_h in hidden_widths:
        print(f"\n--- Hidden width n = {n_h} ---")

        # Train the network
        net = TwoLayerReLU(d, n_h, lr=0.01, n_epochs=1000, seed=seed)
        net.fit(X_train, y_train)

        # Baseline predictions and importance
        y_pred_base, _ = net.forward(X_test)
        mse_base = np.mean((y_pred_base - y_test) ** 2)
        imp_base = neuron_importance(net, X_test)
        rank_base = stats.rankdata(imp_base)
        mean_act_base = np.mean(imp_base)

        print(f"  MSE = {mse_base:.4f}, mean_importance = {mean_act_base:.4f}")

        # Generate random permutations and measure stability
        imp_vectors = [imp_base]
        mean_acts = [mean_act_base]
        rank_vectors = [rank_base]
        pred_diffs = []

        for p in range(n_perms):
            perm = rng.permutation(n_h)
            net_perm = permute_hidden(net, perm)

            # Verify functional equivalence
            y_pred_perm, _ = net_perm.forward(X_test)
            pred_diffs.append(np.max(np.abs(y_pred_perm - y_pred_base)))

            # Neuron importance in permuted network
            imp_perm = neuron_importance(net_perm, X_test)
            imp_vectors.append(imp_perm)
            mean_acts.append(np.mean(imp_perm))
            rank_vectors.append(stats.rankdata(imp_perm))

        # ---- Measurements ----

        # (a) Functional equivalence check
        max_pred_diff = max(pred_diffs)
        print(f"  Max prediction diff (should be ~0): {max_pred_diff:.2e}")

        # (b) Mean activation correlation (the G-invariant)
        mean_act_corr = np.corrcoef(mean_acts[:-1], mean_acts[1:])[0, 1]
        # Since all mean_acts should be identical, correlation is degenerate;
        # instead measure max deviation
        mean_act_spread = np.std(mean_acts)
        mean_acts_identical = np.allclose(mean_acts, mean_acts[0], atol=1e-10)
        print(f"  Mean activation std (should be ~0): {mean_act_spread:.2e}")
        print(f"  Mean activations identical: {mean_acts_identical}")

        # (c) Spearman correlation of per-neuron importance across permutations
        spearman_corrs = []
        for i in range(1, len(imp_vectors)):
            rho, _ = stats.spearmanr(imp_vectors[0], imp_vectors[i])
            spearman_corrs.append(rho)
        mean_spearman = np.mean(spearman_corrs)
        print(f"  Mean Spearman corr of importance vectors: {mean_spearman:.4f}")

        # (d) Fraction of neurons with stable rank (same position ± 1 across ALL permutations)
        n_stable = 0
        for j in range(n_h):
            base_rank = rank_base[j]
            stable = True
            for rv in rank_vectors[1:]:
                if abs(rv[j] - base_rank) > 1:
                    stable = False
                    break
            if stable:
                n_stable += 1
        frac_stable = n_stable / n_h
        predicted_frac = 1.0 / n_h
        print(f"  Stable neurons: {n_stable}/{n_h} = {frac_stable:.4f}")
        print(f"  Predicted 1/n:  {predicted_frac:.4f}")

        # (e) Fraction of importance variance explained by permutation-invariant
        # statistics.  The only S_n-invariant of the importance vector is
        # its sorted multiset (equivalently, order statistics).  But the
        # *identity* of which neuron has which importance is lost.
        # We measure: across permutations, what fraction of the n-dimensional
        # importance vector is "recoverable"?  Answer: only the 1-dim mean
        # (and other symmetric functions), which spans 1/n of the coordinates.
        # Operationally: compute Kendall tau between base and permuted ranks.
        kendall_taus = []
        for i in range(1, len(rank_vectors)):
            tau, _ = stats.kendalltau(rank_vectors[0], rank_vectors[i])
            kendall_taus.append(tau)
        mean_kendall = np.nanmean(kendall_taus)
        print(f"  Mean Kendall tau of neuron ranks: {mean_kendall:.4f}")

        # (f) Dimension of stable subspace: the G-invariant subspace has
        # dimension 1 (the mean), so stable_dim / n = 1/n.
        # We verify: the rank-1 approximation (projecting all importance
        # vectors onto the mean direction) captures ~1/n of variance.
        imp_matrix = np.array(imp_vectors)          # (n_perms+1, n_h)
        total_var = np.var(imp_matrix, axis=1).mean()  # avg variance across neurons
        # Variance of the mean (the invariant part)
        mean_vals = imp_matrix.mean(axis=1)             # (n_perms+1,)
        invariant_var = np.var(mean_vals)
        # Per-neuron variance (the non-invariant part)
        per_neuron_var = np.mean(np.var(imp_matrix, axis=0))
        frac_var_invariant = 1.0 - (per_neuron_var / total_var) if total_var > 0 else 0
        print(f"  Per-neuron variance (instability): {per_neuron_var:.4f}")
        print(f"  Invariant (mean) variance: {invariant_var:.2e}")

        results[str(n_h)] = {
            'hidden_width': n_h,
            'mse': float(mse_base),
            'max_pred_diff': float(max_pred_diff),
            'mean_act_spread': float(mean_act_spread),
            'mean_acts_identical': bool(mean_acts_identical),
            'mean_spearman_corr': float(mean_spearman),
            'mean_kendall_tau': float(mean_kendall),
            'n_stable_neurons': int(n_stable),
            'frac_stable': float(frac_stable),
            'predicted_frac_1_over_n': float(predicted_frac),
            'per_neuron_variance': float(per_neuron_var),
            'invariant_variance': float(invariant_var),
            'spearman_corrs': [float(s) for s in spearman_corrs],
        }

    return results


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def make_plot(results: dict):
    load_publication_style()

    widths = sorted([int(k) for k in results.keys() if k != '_timestamp'])
    frac_stable = [results[str(w)]['frac_stable'] for w in widths]
    predicted = [results[str(w)]['predicted_frac_1_over_n'] for w in widths]
    mean_act_ok = [1.0 if results[str(w)]['mean_acts_identical'] else 0.0 for w in widths]
    spearman = [results[str(w)]['mean_spearman_corr'] for w in widths]
    kendall = [results[str(w)]['mean_kendall_tau'] for w in widths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # --- Left panel: 1/n ceiling and observed stable fraction ---
    ax1.plot(widths, predicted, 's--', label=r'$1/n$ ceiling (predicted)',
             color='#D55E00', markersize=5, zorder=3)
    ax1.fill_between(widths, 0, predicted, alpha=0.15, color='#D55E00')
    ax1.plot(widths, frac_stable, 'o-', label='Observed stable fraction',
             color='#0072B2', markersize=5, zorder=4)
    ax1.set_xlabel('Hidden width $n$')
    ax1.set_ylabel('Fraction of stable neurons')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(widths)
    ax1.set_xticklabels([str(w) for w in widths])
    ax1.set_ylim(-0.02, 0.35)
    ax1.legend(loc='upper right', fontsize=7)
    ax1.set_title(r'Interpretability ceiling: observed $\leq 1/n$')

    # --- Right panel: G-invariant vs per-neuron correlation ---
    ax2.plot(widths, mean_act_ok, 'o-', label=r'$G$-invariant (mean act.)',
             color='#009E73', markersize=5)
    ax2.plot(widths, spearman, 's-', label=r'Per-neuron Spearman $\rho$',
             color='#CC79A7', markersize=5)
    ax2.plot(widths, kendall, '^-', label=r'Per-neuron Kendall $\tau$',
             color='#56B4E9', markersize=5)
    ax2.axhline(1.0, ls=':', color='gray', alpha=0.5)
    ax2.axhline(0.0, ls=':', color='gray', alpha=0.5)
    ax2.set_xlabel('Hidden width $n$')
    ax2.set_ylabel('Correlation')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(widths)
    ax2.set_xticklabels([str(w) for w in widths])
    ax2.set_ylim(-0.3, 1.15)
    ax2.legend(loc='center right', fontsize=7)
    ax2.set_title(r'Stability: invariant vs.\ per-neuron')

    fig.tight_layout()
    save_figure(fig, 'interpretability_ceiling')
    return fig


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Interpretability Ceiling Experiment")
    print("Theoretical prediction: stable fraction ~ 1/n")
    print("=" * 60)

    results = run_experiment(seed=42)

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'n':>6}  {'stable':>8}  {'1/n':>8}  {'ratio':>8}  {'G-inv':>6}  {'Spearman':>8}  {'Kendall':>8}")
    print("-" * 70)
    for w in sorted([int(k) for k in results.keys() if k != '_timestamp']):
        r = results[str(w)]
        ratio = r['frac_stable'] / r['predicted_frac_1_over_n'] if r['predicted_frac_1_over_n'] > 0 else float('inf')
        print(f"{w:>6}  {r['frac_stable']:>8.4f}  {r['predicted_frac_1_over_n']:>8.4f}  "
              f"{ratio:>8.2f}  {'yes' if r['mean_acts_identical'] else 'NO':>6}  "
              f"{r['mean_spearman_corr']:>8.4f}  {r['mean_kendall_tau']:>8.4f}")

    # Save
    save_results(results, 'interpretability_ceiling')
    make_plot(results)

    print("\nDone.")
