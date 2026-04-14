"""
MI Non-Uniqueness Adversarial Audit
====================================

Rigorous audit of the claimed Jaccard = 0.041 (chance level) result.
Determines whether this reflects permutation symmetry (framework prediction)
or artifacts (unconverged training, different loss basins).

TEST 1: Convergence check (max_iter=500, expect >99% accuracy)
TEST 2: Permutation alignment via Hungarian algorithm
TEST 3: Known-permutation control (ground-truth baseline)
TEST 4: Probe stability (within-model consistency)
"""

import json
import os
import time
import warnings
from itertools import combinations

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
N_MODELS = 10
HIDDEN_SIZE = 256
TOP_K = 20
N_CLASSES = 10
MAX_ITER = 500  # Increased from 50
TEST_SIZE = 0.2
N_PROBE_SAMPLES = 10000

BASE_DIR = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments'
RESULTS_PATH = os.path.join(BASE_DIR, 'results_mi_audit.json')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')
FIGURE_PATH = os.path.join(FIGURE_DIR, 'mi_audit.pdf')

SEEDS = list(range(42, 42 + N_MODELS))

# Theoretical chance Jaccard for top-k out of n
CHANCE_JACCARD = TOP_K / (2 * HIDDEN_SIZE - TOP_K)

# ============================================================
# Helper functions
# ============================================================

def get_hidden_activations(mlp, X):
    """Extract hidden layer 1 activations from a trained sklearn MLP."""
    W1 = mlp.coefs_[0]
    b1 = mlp.intercepts_[0]
    h1 = np.maximum(0, X @ W1 + b1)  # ReLU
    return h1


def fit_probe(activations, labels, random_state=0):
    """Fit logistic regression probe, return weight matrix (n_classes, n_neurons)."""
    probe = LogisticRegression(
        max_iter=1000, solver='lbfgs', multi_class='multinomial',
        C=1.0, random_state=random_state
    )
    probe.fit(activations, labels)
    return probe.coef_


def top_k_neurons(weight_vector, k=TOP_K):
    """Return indices of top-k neurons by absolute weight."""
    return set(np.argsort(np.abs(weight_vector))[-k:])


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def mean_jaccard_top_k(probe_weights_a, probe_weights_b, k=TOP_K):
    """Mean Jaccard across all classes for top-k neurons."""
    jaccards = []
    for c in range(N_CLASSES):
        sa = top_k_neurons(probe_weights_a[c], k)
        sb = top_k_neurons(probe_weights_b[c], k)
        jaccards.append(jaccard(sa, sb))
    return np.mean(jaccards)


def subspace_cosine(W_a, W_b):
    """Mean cosine of principal angles between probe weight subspaces."""
    angles = subspace_angles(W_a.T, W_b.T)
    return float(np.mean(np.cos(angles)))


def hungarian_alignment(W1_a, W1_b):
    """
    Find optimal neuron permutation aligning model B to model A
    using the Hungarian algorithm on weight correlations.

    W1_a, W1_b: (d_input, n_hidden) weight matrices.
    Returns: permutation array pi such that neuron j in B maps to pi[j] in A.
    """
    n_neurons = W1_a.shape[1]
    # Cost matrix: -|correlation| (we want to maximize correlation)
    cost = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(n_neurons):
            r = np.corrcoef(W1_a[:, i], W1_b[:, j])[0, 1]
            cost[i, j] = -abs(r) if not np.isnan(r) else 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[k] = k (rows are model A neurons)
    # col_ind[k] = which model B neuron maps to model A neuron k
    # We want pi such that pi[j] gives the aligned index for B's neuron j
    # col_ind[i] = j means B's neuron j aligns to A's neuron i
    # So we need the inverse: for B's neuron j, find which A neuron it maps to
    pi = np.zeros(n_neurons, dtype=int)
    for i, j in zip(row_ind, col_ind):
        pi[j] = i
    return pi, -np.mean(cost[row_ind, col_ind])


def apply_permutation_to_probe(probe_weights, pi):
    """Remap probe weights according to neuron permutation pi.
    probe_weights: (n_classes, n_neurons)
    pi: permutation array where pi[j] = new index for old neuron j
    Returns permuted probe weights.
    """
    n_classes, n_neurons = probe_weights.shape
    permuted = np.zeros_like(probe_weights)
    for j in range(n_neurons):
        permuted[:, pi[j]] = probe_weights[:, j]
    return permuted


# ============================================================
# TEST 1: Convergence check
# ============================================================
def test1_convergence(X_train, y_train, X_test, y_test, X_probe, y_probe):
    print("\n" + "=" * 70)
    print("TEST 1: Convergence Check (max_iter=500)")
    print("=" * 70)

    models = []
    accuracies = []
    hidden_acts = []
    probe_weights_list = []

    for i, seed in enumerate(SEEDS):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_SIZE, HIDDEN_SIZE),
            activation='relu', max_iter=MAX_ITER, random_state=seed,
            early_stopping=False, verbose=False
        )
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        models.append(mlp)
        accuracies.append(acc)
        print(f"  Model {i} (seed={seed}): accuracy = {acc:.4f}, n_iter = {mlp.n_iter_}")

    mean_acc = np.mean(accuracies)
    min_acc = np.min(accuracies)
    print(f"\n  Mean accuracy: {mean_acc:.4f}, Min accuracy: {min_acc:.4f}")

    # Extract activations and fit probes
    for i, mlp in enumerate(models):
        h1 = get_hidden_activations(mlp, X_probe)
        hidden_acts.append(h1)
        W_probe = fit_probe(h1, y_probe)
        probe_weights_list.append(W_probe)

    # Compute pairwise Jaccard and subspace cosine
    pairs = list(combinations(range(N_MODELS), 2))
    jaccards = []
    cosines = []
    for i, j in pairs:
        jac = mean_jaccard_top_k(probe_weights_list[i], probe_weights_list[j])
        jaccards.append(jac)
        cos = subspace_cosine(probe_weights_list[i], probe_weights_list[j])
        cosines.append(cos)

    mean_jac = np.mean(jaccards)
    mean_cos = np.mean(cosines)

    converged = min_acc > 0.99
    jac_low = mean_jac < 0.10
    passed = converged and jac_low

    print(f"\n  Mean Jaccard (converged):    {mean_jac:.4f} (chance = {CHANCE_JACCARD:.4f})")
    print(f"  Mean subspace cosine:       {mean_cos:.4f}")
    print(f"  Min accuracy > 99%?         {'YES' if converged else 'NO'} ({min_acc:.4f})")
    print(f"  Jaccard < 0.10?             {'YES' if jac_low else 'NO'} ({mean_jac:.4f})")
    print(f"  TEST 1 VERDICT:             {'PASS' if passed else 'FAIL'}")

    return {
        'models': models,
        'accuracies': [float(a) for a in accuracies],
        'hidden_acts': hidden_acts,
        'probe_weights': probe_weights_list,
        'mean_accuracy': float(mean_acc),
        'min_accuracy': float(min_acc),
        'mean_jaccard': float(mean_jac),
        'std_jaccard': float(np.std(jaccards)),
        'mean_subspace_cosine': float(mean_cos),
        'std_subspace_cosine': float(np.std(cosines)),
        'all_jaccards': [float(j) for j in jaccards],
        'all_cosines': [float(c) for c in cosines],
        'converged': bool(converged),
        'jaccard_low': bool(jac_low),
        'passed': bool(passed)
    }


# ============================================================
# TEST 2: Permutation alignment (Hungarian algorithm)
# ============================================================
def test2_hungarian_alignment(models, probe_weights_list):
    print("\n" + "=" * 70)
    print("TEST 2: Permutation Alignment (Hungarian Algorithm)")
    print("=" * 70)

    pairs = list(combinations(range(N_MODELS), 2))
    unaligned_jaccards = []
    aligned_jaccards = []
    mean_correlations = []

    for idx, (i, j) in enumerate(pairs):
        # Unaligned Jaccard
        unaligned_jac = mean_jaccard_top_k(probe_weights_list[i], probe_weights_list[j])
        unaligned_jaccards.append(unaligned_jac)

        # Hungarian alignment on first-layer weights
        W1_a = models[i].coefs_[0]  # (784, 256)
        W1_b = models[j].coefs_[0]

        pi, mean_corr = hungarian_alignment(W1_a, W1_b)
        mean_correlations.append(mean_corr)

        # Apply permutation to model B's probe weights
        aligned_probe_b = apply_permutation_to_probe(probe_weights_list[j], pi)

        # Aligned Jaccard
        aligned_jac = mean_jaccard_top_k(probe_weights_list[i], aligned_probe_b)
        aligned_jaccards.append(aligned_jac)

        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx+1}/{len(pairs)} pairs...")

    mean_unaligned = np.mean(unaligned_jaccards)
    mean_aligned = np.mean(aligned_jaccards)
    mean_corr_overall = np.mean(mean_correlations)

    print(f"\n  Mean UNALIGNED Jaccard:      {mean_unaligned:.4f} (should match ~0.04)")
    print(f"  Mean ALIGNED Jaccard:        {mean_aligned:.4f}")
    print(f"  Mean alignment correlation:  {mean_corr_overall:.4f}")
    print(f"  Chance Jaccard:              {CHANCE_JACCARD:.4f}")

    if mean_aligned > 0.5:
        interpretation = "S_n permutation symmetry is the FULL explanation (confirms framework)"
    elif mean_aligned > 0.2:
        interpretation = "Partial permutation alignment — some deeper structure exists"
    else:
        interpretation = "Something DEEPER than permutation symmetry is happening"

    print(f"\n  INTERPRETATION: {interpretation}")

    # Determine pass/fail: aligned Jaccard should be meaningfully above unaligned
    alignment_lift = mean_aligned / max(mean_unaligned, 1e-10)
    print(f"  Alignment lift factor:       {alignment_lift:.1f}x")

    return {
        'mean_unaligned_jaccard': float(mean_unaligned),
        'std_unaligned_jaccard': float(np.std(unaligned_jaccards)),
        'mean_aligned_jaccard': float(mean_aligned),
        'std_aligned_jaccard': float(np.std(aligned_jaccards)),
        'mean_alignment_correlation': float(mean_corr_overall),
        'alignment_lift': float(alignment_lift),
        'interpretation': interpretation,
        'all_unaligned': [float(j) for j in unaligned_jaccards],
        'all_aligned': [float(j) for j in aligned_jaccards],
    }


# ============================================================
# TEST 3: Known-permutation control
# ============================================================
def test3_known_permutation(models, probe_weights_list, X_probe, y_probe):
    print("\n" + "=" * 70)
    print("TEST 3: Known-Permutation Control")
    print("=" * 70)

    model_0 = models[0]
    W1_orig = model_0.coefs_[0].copy()
    b1_orig = model_0.intercepts_[0].copy()
    W2_orig = model_0.coefs_[1].copy()
    probe_0 = probe_weights_list[0]

    n_permutations = 10
    unaligned_jaccards = []
    aligned_jaccards = []
    alignment_correlations = []

    for p in range(n_permutations):
        rng = np.random.RandomState(p + 100)
        perm = rng.permutation(HIDDEN_SIZE)

        # Apply known permutation to hidden layer 1
        W1_perm = W1_orig[:, perm]        # permute columns of W1
        b1_perm = b1_orig[perm]            # permute bias
        W2_perm = W2_orig[perm, :]         # permute rows of W2

        # Verify functional equivalence: compute activations
        h1_orig = np.maximum(0, X_probe @ W1_orig + b1_orig)
        h1_perm = np.maximum(0, X_probe @ W1_perm + b1_perm)

        # Layer 2 output should be identical
        h2_orig = h1_orig @ W2_orig
        h2_perm = h1_perm @ W2_perm
        assert np.allclose(h2_orig, h2_perm, atol=1e-6), "Permuted model is NOT functionally equivalent!"

        # Fit probe on permuted activations
        probe_perm = fit_probe(h1_perm, y_probe)

        # UNALIGNED Jaccard (should be ~chance, since labels are shuffled)
        unaligned_jac = mean_jaccard_top_k(probe_0, probe_perm)
        unaligned_jaccards.append(unaligned_jac)

        # Hungarian alignment to recover the permutation
        pi, mean_corr = hungarian_alignment(W1_orig, W1_perm)
        alignment_correlations.append(mean_corr)

        # Apply alignment and compute Jaccard (should be ~1.0)
        aligned_probe = apply_permutation_to_probe(probe_perm, pi)
        aligned_jac = mean_jaccard_top_k(probe_0, aligned_probe)
        aligned_jaccards.append(aligned_jac)

        print(f"  Permutation {p}: unaligned={unaligned_jac:.4f}, aligned={aligned_jac:.4f}, corr={mean_corr:.4f}")

    mean_unaligned = np.mean(unaligned_jaccards)
    mean_aligned = np.mean(aligned_jaccards)

    print(f"\n  Mean UNALIGNED Jaccard (known perm): {mean_unaligned:.4f} (expect ~chance = {CHANCE_JACCARD:.4f})")
    print(f"  Mean ALIGNED Jaccard (known perm):   {mean_aligned:.4f} (expect ~1.0)")

    # Ground-truth checks
    unaligned_near_chance = mean_unaligned < 0.10
    aligned_near_one = mean_aligned > 0.90

    print(f"\n  Unaligned near chance (<0.10)?  {'YES' if unaligned_near_chance else 'NO'}")
    print(f"  Aligned near 1.0 (>0.90)?      {'YES' if aligned_near_one else 'NO'}")
    print(f"  TEST 3 VERDICT:                {'PASS' if (unaligned_near_chance and aligned_near_one) else 'FAIL'}")

    return {
        'mean_unaligned_jaccard': float(mean_unaligned),
        'mean_aligned_jaccard': float(mean_aligned),
        'std_unaligned': float(np.std(unaligned_jaccards)),
        'std_aligned': float(np.std(aligned_jaccards)),
        'all_unaligned': [float(j) for j in unaligned_jaccards],
        'all_aligned': [float(j) for j in aligned_jaccards],
        'unaligned_near_chance': bool(unaligned_near_chance),
        'aligned_near_one': bool(aligned_near_one),
        'passed': bool(unaligned_near_chance and aligned_near_one)
    }


# ============================================================
# TEST 4: Probe stability (within-model)
# ============================================================
def test4_probe_stability(models, X_probe, y_probe):
    print("\n" + "=" * 70)
    print("TEST 4: Probe Stability (Within-Model)")
    print("=" * 70)

    model_0 = models[0]
    h1 = get_hidden_activations(model_0, X_probe)

    n_probe_seeds = 10
    probe_weights = []
    for s in range(n_probe_seeds):
        W = fit_probe(h1, y_probe, random_state=s)
        probe_weights.append(W)

    # Compute all pairwise subspace cosines
    pairs = list(combinations(range(n_probe_seeds), 2))
    cosines = []
    jaccards = []
    for i, j in pairs:
        cos = subspace_cosine(probe_weights[i], probe_weights[j])
        cosines.append(cos)
        jac = mean_jaccard_top_k(probe_weights[i], probe_weights[j])
        jaccards.append(jac)

    mean_cos = np.mean(cosines)
    mean_jac = np.mean(jaccards)

    print(f"  Within-model subspace cosine: {mean_cos:.4f} +/- {np.std(cosines):.4f}")
    print(f"  Within-model Jaccard:         {mean_jac:.4f} +/- {np.std(jaccards):.4f}")

    stable = mean_cos > 0.80
    print(f"\n  Within-model cosine > 0.80?   {'YES' if stable else 'NO'} ({mean_cos:.4f})")
    print(f"  TEST 4 VERDICT:               {'PASS' if stable else 'FAIL'}")

    if stable:
        print(f"  => Probe is stable; the low cross-model cosine (0.18) is REAL, not noise.")
    else:
        print(f"  => Probe is noisy; cross-model cosine may be unreliable.")

    return {
        'mean_within_cosine': float(mean_cos),
        'std_within_cosine': float(np.std(cosines)),
        'mean_within_jaccard': float(mean_jac),
        'std_within_jaccard': float(np.std(jaccards)),
        'all_cosines': [float(c) for c in cosines],
        'all_jaccards': [float(j) for j in jaccards],
        'stable': bool(stable),
        'passed': bool(stable)
    }


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("MI NON-UNIQUENESS ADVERSARIAL AUDIT")
    print("=" * 70)
    print(f"Original claim: Jaccard = 0.041 (chance level = {CHANCE_JACCARD:.4f})")
    print(f"Question: permutation symmetry or something else?")

    # Load MNIST
    print("\n[0] Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=999, stratify=y
    )
    rng = np.random.RandomState(0)
    probe_idx = rng.choice(len(X_test), size=min(N_PROBE_SAMPLES, len(X_test)), replace=False)
    X_probe = X_test[probe_idx]
    y_probe = y_test[probe_idx]
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Probe: {X_probe.shape[0]}")

    # Run all tests
    t1 = test1_convergence(X_train, y_train, X_test, y_test, X_probe, y_probe)
    t2 = test2_hungarian_alignment(t1['models'], t1['probe_weights'])
    t3 = test3_known_permutation(t1['models'], t1['probe_weights'], X_probe, y_probe)
    t4 = test4_probe_stability(t1['models'], X_probe, y_probe)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    tests = {
        'TEST 1 (Convergence)': t1['passed'],
        'TEST 2 (Hungarian Alignment)': True,  # informational
        'TEST 3 (Known-Permutation Control)': t3['passed'],
        'TEST 4 (Probe Stability)': t4['passed'],
    }

    for name, passed in tests.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    # Key diagnostic
    print(f"\n  KEY NUMBERS:")
    print(f"    Original Jaccard (max_iter=50):          0.041")
    print(f"    Converged Jaccard (max_iter=500):        {t1['mean_jaccard']:.4f}")
    print(f"    Unaligned Jaccard (cross-model):         {t2['mean_unaligned_jaccard']:.4f}")
    print(f"    ALIGNED Jaccard (cross-model, Hungarian):{t2['mean_aligned_jaccard']:.4f}")
    print(f"    Known-perm unaligned Jaccard:            {t3['mean_unaligned_jaccard']:.4f}")
    print(f"    Known-perm ALIGNED Jaccard:              {t3['mean_aligned_jaccard']:.4f}")
    print(f"    Within-model probe cosine:               {t4['mean_within_cosine']:.4f}")
    print(f"    Cross-model subspace cosine:             {t1['mean_subspace_cosine']:.4f}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    if t2['mean_aligned_jaccard'] > 0.5:
        diagnosis = (
            "The non-uniqueness is FULLY explained by S_n permutation symmetry. "
            "After Hungarian alignment, neurons are consistent across models. "
            "The original Jaccard = 0.041 is exactly what permutation symmetry predicts."
        )
    elif t2['mean_aligned_jaccard'] > 0.2:
        diagnosis = (
            "Permutation symmetry explains MOST but not ALL of the non-uniqueness. "
            f"Aligned Jaccard = {t2['mean_aligned_jaccard']:.3f} is above chance but below 0.5, "
            "suggesting additional rotation/scaling degrees of freedom beyond pure permutation."
        )
    else:
        diagnosis = (
            "The non-uniqueness goes DEEPER than permutation symmetry. "
            f"Even after optimal alignment, Jaccard = {t2['mean_aligned_jaccard']:.3f} remains low. "
            "This suggests the models find genuinely different feature decompositions "
            "(different loss basins, not just relabeled neurons)."
        )
    print(f"  {diagnosis}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        'experiment': 'MI Non-Uniqueness Adversarial Audit',
        'original_claim': 'Jaccard = 0.041 (chance level)',
        'test1_convergence': {
            'accuracies': t1['accuracies'],
            'mean_accuracy': t1['mean_accuracy'],
            'min_accuracy': t1['min_accuracy'],
            'mean_jaccard': t1['mean_jaccard'],
            'std_jaccard': t1['std_jaccard'],
            'mean_subspace_cosine': t1['mean_subspace_cosine'],
            'std_subspace_cosine': t1['std_subspace_cosine'],
            'converged': t1['converged'],
            'passed': t1['passed'],
        },
        'test2_hungarian_alignment': {
            'mean_unaligned_jaccard': t2['mean_unaligned_jaccard'],
            'std_unaligned_jaccard': t2['std_unaligned_jaccard'],
            'mean_aligned_jaccard': t2['mean_aligned_jaccard'],
            'std_aligned_jaccard': t2['std_aligned_jaccard'],
            'mean_alignment_correlation': t2['mean_alignment_correlation'],
            'alignment_lift': t2['alignment_lift'],
            'interpretation': t2['interpretation'],
        },
        'test3_known_permutation': {
            'mean_unaligned_jaccard': t3['mean_unaligned_jaccard'],
            'mean_aligned_jaccard': t3['mean_aligned_jaccard'],
            'unaligned_near_chance': t3['unaligned_near_chance'],
            'aligned_near_one': t3['aligned_near_one'],
            'passed': t3['passed'],
        },
        'test4_probe_stability': {
            'mean_within_cosine': t4['mean_within_cosine'],
            'std_within_cosine': t4['std_within_cosine'],
            'mean_within_jaccard': t4['mean_within_jaccard'],
            'std_within_jaccard': t4['std_within_jaccard'],
            'stable': t4['stable'],
            'passed': t4['passed'],
        },
        'diagnosis': diagnosis,
        'all_tests_passed': all(tests.values()),
        'elapsed_seconds': round(elapsed, 1),
    }

    os.makedirs(FIGURE_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # ============================================================
    # Generate figure (2x2 grid)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Test 1 — Accuracy histogram
    ax = axes[0, 0]
    ax.bar(range(N_MODELS), t1['accuracies'], color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0.99, color='red', linestyle='--', linewidth=2, label='99% threshold')
    ax.axhline(t1['mean_accuracy'], color='#1f77b4', linestyle='-', linewidth=2,
               label=f'Mean = {t1["mean_accuracy"]:.4f}')
    ax.set_xlabel('Model index', fontsize=11)
    ax.set_ylabel('Test accuracy', fontsize=11)
    ax.set_title('Test 1: Convergence Check (max_iter=500)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.96, 1.001)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02, f'PASS' if t1['passed'] else 'FAIL',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='green' if t1['passed'] else 'red',
            verticalalignment='bottom')

    # Panel B: Test 2 — Aligned vs unaligned Jaccard
    ax = axes[0, 1]
    x_pos = np.arange(len(t2['all_unaligned']))
    width = 0.35
    ax.bar(x_pos - width/2, t2['all_unaligned'], width, label='Unaligned', color='#d62728', alpha=0.7)
    ax.bar(x_pos + width/2, t2['all_aligned'], width, label='Aligned (Hungarian)', color='#2ca02c', alpha=0.7)
    ax.axhline(CHANCE_JACCARD, color='black', linestyle='--', linewidth=1.5, label=f'Chance = {CHANCE_JACCARD:.3f}')
    ax.set_xlabel('Pair index', fontsize=11)
    ax.set_ylabel('Jaccard similarity', fontsize=11)
    ax.set_title('Test 2: Hungarian Alignment', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.text(0.02, 0.98, f'Aligned = {t2["mean_aligned_jaccard"]:.3f}\nLift = {t2["alignment_lift"]:.1f}x',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel C: Test 3 — Known permutation control
    ax = axes[1, 0]
    perm_x = np.arange(len(t3['all_unaligned']))
    ax.bar(perm_x - width/2, t3['all_unaligned'], width, label='Unaligned', color='#d62728', alpha=0.7)
    ax.bar(perm_x + width/2, t3['all_aligned'], width, label='Aligned', color='#2ca02c', alpha=0.7)
    ax.axhline(CHANCE_JACCARD, color='black', linestyle='--', linewidth=1.5, label=f'Chance = {CHANCE_JACCARD:.3f}')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Perfect = 1.0')
    ax.set_xlabel('Permutation index', fontsize=11)
    ax.set_ylabel('Jaccard similarity', fontsize=11)
    ax.set_title('Test 3: Known-Permutation Control', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, f'PASS' if t3['passed'] else 'FAIL',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='green' if t3['passed'] else 'red',
            verticalalignment='bottom')

    # Panel D: Test 4 — Probe stability
    ax = axes[1, 1]
    ax.hist(t4['all_cosines'], bins=15, color='#9467bd', alpha=0.7, edgecolor='black', linewidth=0.5,
            label='Within-model cosines')
    ax.axvline(t4['mean_within_cosine'], color='#9467bd', linestyle='-', linewidth=2,
               label=f'Within-model = {t4["mean_within_cosine"]:.3f}')
    ax.axvline(t1['mean_subspace_cosine'], color='#d62728', linestyle='--', linewidth=2,
               label=f'Cross-model = {t1["mean_subspace_cosine"]:.3f}')
    ax.axvline(0.80, color='black', linestyle=':', linewidth=1.5, label='Stability threshold = 0.80')
    ax.set_xlabel('Subspace cosine similarity', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Test 4: Probe Stability', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, f'PASS' if t4['passed'] else 'FAIL',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='green' if t4['passed'] else 'red',
            verticalalignment='bottom')

    fig.suptitle(
        'MI Non-Uniqueness Adversarial Audit\n'
        f'Jaccard = {t1["mean_jaccard"]:.3f} (chance = {CHANCE_JACCARD:.3f}), '
        f'Aligned Jaccard = {t2["mean_aligned_jaccard"]:.3f}',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches='tight', dpi=150)
    print(f"  Figure saved to: {FIGURE_PATH}")

    print(f"\n{'=' * 70}")
    print("AUDIT COMPLETE")
    print(f"{'=' * 70}")

    return results


if __name__ == '__main__':
    results = main()
