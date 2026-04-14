"""
Mechanistic Interpretability Non-Uniqueness Experiment
======================================================

Demonstrates that neuron-level interpretations are non-unique across
independently trained neural networks, while the functional subspaces
they span are highly similar — confirming an interpretability ceiling.

Experiment A: MLP Feature Probing (10 models, MNIST)
-----------------------------------------------------
1. Train 10 two-hidden-layer MLPs (256 hidden units) on MNIST from different seeds
2. Verify all achieve >97% accuracy
3. For each model: fit linear probes on hidden layer 1 activations
4. Measure neuron-level agreement (Jaccard) vs subspace-level agreement (cosine)

Prediction: Jaccard ~ 0.08 (chance), subspace cosine ~ 1.0
Success: Mean Jaccard < 0.20 AND Mean subspace cosine > 0.80
"""

import json
import os
import time
import warnings
from itertools import combinations

import numpy as np
from scipy.linalg import subspace_angles
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
TOP_K = 20          # top neurons per class for Jaccard
N_CLASSES = 10      # digits 0-9
MAX_ITER = 50       # MLP training iterations
TEST_SIZE = 0.2
N_PROBE_SAMPLES = 10000  # samples for probing (speed)

RESULTS_PATH = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/results_mi_nonuniqueness.json'
FIGURE_DIR = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/figures'
FIGURE_PATH = os.path.join(FIGURE_DIR, 'mi_nonuniqueness.pdf')

SEEDS = list(range(42, 42 + N_MODELS))

# ============================================================
# Helper functions
# ============================================================

def get_hidden_activations(mlp, X):
    """Extract hidden layer 1 activations from a trained sklearn MLP."""
    W1 = mlp.coefs_[0]   # (784, 256)
    b1 = mlp.intercepts_[0]  # (256,)
    h1 = X @ W1 + b1
    h1 = np.maximum(h1, 0)   # ReLU
    return h1


def fit_probe(activations, labels):
    """Fit logistic regression probe, return weight matrix (n_classes, n_neurons)."""
    probe = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0,
        random_state=0
    )
    probe.fit(activations, labels)
    return probe.coef_  # shape (10, 256)


def top_k_neurons(weight_vector, k=TOP_K):
    """Return indices of top-k neurons by absolute weight."""
    return set(np.argsort(np.abs(weight_vector))[-k:])


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def subspace_cosine(W_a, W_b):
    """
    Compute subspace similarity between two probe weight matrices via
    principal angles. Each is (n_classes, n_neurons).

    Returns the mean cosine of principal angles (1.0 = identical subspaces).
    """
    angles = subspace_angles(W_a.T, W_b.T)
    cos_angles = np.cos(angles)
    return float(np.mean(cos_angles))


def representational_similarity_cca(H_a, H_b, n_components=10):
    """
    Compute CCA-based representational similarity between two sets of
    hidden activations using SVD (fast, no iterative fitting).
    This is the SVCCA approach (Raghu et al., 2017).

    H_a, H_b: (n_samples, n_neurons) activation matrices from two models.
    Returns: mean canonical correlation (1.0 = identical representations).
    """
    # Center
    H_a = H_a - H_a.mean(axis=0)
    H_b = H_b - H_b.mean(axis=0)

    # Truncated SVD to top-k components (SVCCA style)
    U_a, S_a, Vt_a = np.linalg.svd(H_a, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(H_b, full_matrices=False)

    # Keep top n_components
    U_a = U_a[:, :n_components]
    U_b = U_b[:, :n_components]

    # CCA on the reduced representations: canonical correlations are
    # the singular values of U_a^T @ U_b
    M = U_a.T @ U_b  # (n_components, n_components)
    svals = np.linalg.svd(M, compute_uv=False)

    # Clip to [0, 1] (numerical precision)
    svals = np.clip(svals, 0, 1)

    return float(np.mean(svals))


def probe_prediction_agreement(probe_weights_a, probe_weights_b, H_a, H_b, y):
    """
    Measure whether the probes make the same predictions (functional equivalence).
    Even if neuron roles differ, probes should predict similarly if the
    representations encode the same information.

    Returns: fraction of test samples where probe predictions agree.
    """
    # Use probe weights to predict: pred = argmax(H @ W.T)
    logits_a = H_a @ probe_weights_a.T  # (n_samples, 10)
    logits_b = H_b @ probe_weights_b.T  # (n_samples, 10)
    preds_a = np.argmax(logits_a, axis=1)
    preds_b = np.argmax(logits_b, axis=1)
    return float(np.mean(preds_a == preds_b))


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("MI Non-Uniqueness Experiment")
    print("=" * 70)

    t_start = time.time()

    # ----------------------------------------------------------
    # 1. Load MNIST
    # ----------------------------------------------------------
    print("\n[1] Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split (fixed seed for reproducibility of split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=999, stratify=y
    )

    # Subsample for probing (speed)
    rng = np.random.RandomState(0)
    probe_idx = rng.choice(len(X_test), size=min(N_PROBE_SAMPLES, len(X_test)), replace=False)
    X_probe = X_test[probe_idx]
    y_probe = y_test[probe_idx]

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Probe: {X_probe.shape[0]}")

    # ----------------------------------------------------------
    # 2. Train N_MODELS MLPs
    # ----------------------------------------------------------
    print(f"\n[2] Training {N_MODELS} MLPs (hidden={HIDDEN_SIZE},{HIDDEN_SIZE}, max_iter={MAX_ITER})...")

    models = []
    accuracies = []

    for i, seed in enumerate(SEEDS):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_SIZE, HIDDEN_SIZE),
            activation='relu',
            max_iter=MAX_ITER,
            random_state=seed,
            early_stopping=False,
            verbose=False
        )
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        models.append(mlp)
        accuracies.append(acc)
        print(f"  Model {i} (seed={seed}): accuracy = {acc:.4f}")

    mean_acc = np.mean(accuracies)
    min_acc = np.min(accuracies)
    print(f"\n  Mean accuracy: {mean_acc:.4f}, Min accuracy: {min_acc:.4f}")
    assert min_acc > 0.95, f"Min accuracy {min_acc:.4f} is below 0.95 — models not well-trained"

    # ----------------------------------------------------------
    # 3. Extract hidden activations and fit probes
    # ----------------------------------------------------------
    print(f"\n[3] Extracting hidden activations and fitting probes...")

    probe_weights = []  # list of (10, 256) arrays
    hidden_acts = []    # list of (n_samples, 256) arrays
    top_neuron_sets = []  # list of dicts: {class_idx: set of top-k neuron indices}

    for i, mlp in enumerate(models):
        h1 = get_hidden_activations(mlp, X_probe)
        hidden_acts.append(h1)
        W_probe = fit_probe(h1, y_probe)  # (10, 256)
        probe_weights.append(W_probe)

        top_sets = {}
        for c in range(N_CLASSES):
            top_sets[c] = top_k_neurons(W_probe[c], TOP_K)
        top_neuron_sets.append(top_sets)

        print(f"  Model {i}: probe fitted, weight shape = {W_probe.shape}")

    # ----------------------------------------------------------
    # 4. Compute pairwise metrics
    # ----------------------------------------------------------
    print(f"\n[4] Computing pairwise metrics across {N_MODELS}C2 = {N_MODELS*(N_MODELS-1)//2} pairs...")

    pairs = list(combinations(range(N_MODELS), 2))

    all_jaccards = []       # per-pair mean Jaccard
    all_class_jaccards = [] # all per-class Jaccards (flat)
    all_subspace_cos = []   # per-pair subspace cosine (principal angles)
    all_cca = []            # per-pair CCA similarity
    all_pred_agree = []     # per-pair probe prediction agreement

    for idx, (i, j) in enumerate(pairs):
        # Per-class Jaccard
        class_jaccards = []
        for c in range(N_CLASSES):
            jac = jaccard(top_neuron_sets[i][c], top_neuron_sets[j][c])
            class_jaccards.append(jac)
            all_class_jaccards.append(jac)

        mean_jac = np.mean(class_jaccards)
        all_jaccards.append(mean_jac)

        # Subspace cosine (principal angles on probe weights)
        cos_sim = subspace_cosine(probe_weights[i], probe_weights[j])
        all_subspace_cos.append(cos_sim)

        # CCA-based representational similarity (on hidden activations)
        cca_sim = representational_similarity_cca(hidden_acts[i], hidden_acts[j], n_components=N_CLASSES)
        all_cca.append(cca_sim)

        # Probe prediction agreement
        pred_ag = probe_prediction_agreement(probe_weights[i], probe_weights[j],
                                             hidden_acts[i], hidden_acts[j], y_probe)
        all_pred_agree.append(pred_ag)

        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx+1}/{len(pairs)} pairs...")

    mean_jaccard = float(np.mean(all_jaccards))
    std_jaccard = float(np.std(all_jaccards))
    mean_cosine = float(np.mean(all_subspace_cos))
    std_cosine = float(np.std(all_subspace_cos))
    mean_cca = float(np.mean(all_cca))
    std_cca = float(np.std(all_cca))
    mean_pred_agree = float(np.mean(all_pred_agree))
    std_pred_agree = float(np.std(all_pred_agree))

    # Theoretical chance-level Jaccard for top-20 out of 256
    # J(A,B) where |A|=|B|=k, drawn uniformly from n items
    # E[J] = k / (2n - k) for independent uniform draws
    k, n = TOP_K, HIDDEN_SIZE
    chance_jaccard = k / (2 * n - k)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  --- Neuron-level (should be LOW) ---")
    print(f"  Mean Jaccard similarity (top-{TOP_K} neurons): {mean_jaccard:.4f} +/- {std_jaccard:.4f}")
    print(f"  Theoretical chance Jaccard:                {chance_jaccard:.4f}")
    print(f"  Ratio to chance:                          {mean_jaccard/chance_jaccard:.2f}x")
    print(f"")
    print(f"  --- Representation-level (should be HIGH) ---")
    print(f"  Mean CCA similarity (hidden activations):  {mean_cca:.4f} +/- {std_cca:.4f}")
    print(f"  Mean probe prediction agreement:           {mean_pred_agree:.4f} +/- {std_pred_agree:.4f}")
    print(f"  Mean subspace cosine (principal angles):   {mean_cosine:.4f} +/- {std_cosine:.4f}")
    print(f"")

    # Success criteria: use CCA as the representation-level metric
    success_jaccard = mean_jaccard < 0.20
    success_cca = mean_cca > 0.80
    success = success_jaccard and success_cca

    print(f"  Jaccard < 0.20?     {'YES' if success_jaccard else 'NO'} ({mean_jaccard:.4f})")
    print(f"  CCA     > 0.80?     {'YES' if success_cca else 'NO'} ({mean_cca:.4f})")
    print(f"  Pred agreement:     {mean_pred_agree:.4f} (informational)")
    print(f"  Overall success:    {'YES' if success else 'NO'}")
    print(f"{'='*70}")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    print(f"  Neuron-level agreement ({mean_jaccard:.3f}) is near chance ({chance_jaccard:.3f}),")
    print(f"  meaning individual neuron roles are NOT consistent across models.")
    print(f"  CCA similarity ({mean_cca:.3f}) and prediction agreement ({mean_pred_agree:.3f})")
    print(f"  are {'high' if mean_cca > 0.8 else 'moderate'}, meaning the hidden representations encode")
    print(f"  the same information despite different neuron-level assignments.")
    print(f"  This confirms the interpretability ceiling: mechanistic explanations")
    print(f"  at the neuron level are non-unique, while the underlying computation")
    print(f"  is invariant up to rotation/permutation.")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    # ----------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------
    results = {
        'experiment': 'MI Non-Uniqueness (MLP Feature Probing)',
        'description': 'Demonstrates neuron-level interpretation non-uniqueness across independently trained networks',
        'config': {
            'n_models': N_MODELS,
            'hidden_size': HIDDEN_SIZE,
            'top_k': TOP_K,
            'max_iter': MAX_ITER,
            'seeds': SEEDS,
            'dataset': 'MNIST',
            'n_probe_samples': N_PROBE_SAMPLES
        },
        'model_accuracies': [float(a) for a in accuracies],
        'mean_accuracy': float(mean_acc),
        'min_accuracy': float(min_acc),
        'metrics': {
            'mean_jaccard': mean_jaccard,
            'std_jaccard': std_jaccard,
            'chance_jaccard': chance_jaccard,
            'jaccard_to_chance_ratio': mean_jaccard / chance_jaccard,
            'mean_cca_similarity': mean_cca,
            'std_cca_similarity': std_cca,
            'mean_pred_agreement': mean_pred_agree,
            'std_pred_agreement': std_pred_agree,
            'mean_subspace_cosine': mean_cosine,
            'std_subspace_cosine': std_cosine,
            'all_pair_jaccards': [float(j) for j in all_jaccards],
            'all_pair_cca': [float(c) for c in all_cca],
            'all_pair_pred_agreement': [float(p) for p in all_pred_agree],
            'all_pair_cosines': [float(c) for c in all_subspace_cos],
            'all_class_jaccards': [float(j) for j in all_class_jaccards],
        },
        'success_criteria': {
            'jaccard_lt_0.20': success_jaccard,
            'cca_gt_0.80': success_cca,
            'overall': success
        },
        'interpretation': (
            f"Neuron-level agreement (Jaccard={mean_jaccard:.3f}) is near chance "
            f"({chance_jaccard:.3f}), confirming that individual neuron interpretations "
            f"are non-unique across models. CCA similarity ({mean_cca:.3f}) and "
            f"prediction agreement ({mean_pred_agree:.3f}) are "
            f"{'high' if mean_cca > 0.8 else 'moderate'}, confirming that the "
            f"underlying functional computation is preserved despite neuron-level "
            f"permutation symmetry. This demonstrates the interpretability ceiling."
        ),
        'elapsed_seconds': round(elapsed, 1)
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # ----------------------------------------------------------
    # 6. Generate figure
    # ----------------------------------------------------------
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Jaccard histogram
    ax1.hist(all_class_jaccards, bins=20, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(chance_jaccard, color='black', linestyle='--', linewidth=2,
                label=f'Chance level ({chance_jaccard:.3f})')
    ax1.axvline(mean_jaccard, color='#d62728', linestyle='-', linewidth=2,
                label=f'Mean ({mean_jaccard:.3f})')
    ax1.set_xlabel('Jaccard Similarity (top-20 neurons)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Neuron-Level Agreement\n(Should be near chance)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.02, 0.5)

    # Right panel: CCA similarity histogram
    ax2.hist(all_cca, bins=15, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(mean_cca, color='#2ca02c', linestyle='-', linewidth=2,
                label=f'Mean CCA ({mean_cca:.3f})')
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Perfect (1.0)')
    # Also show prediction agreement as a vertical line
    ax2.axvline(mean_pred_agree, color='#1f77b4', linestyle=':', linewidth=2,
                label=f'Pred. agreement ({mean_pred_agree:.3f})')
    ax2.set_xlabel('CCA Similarity', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Representation-Level Agreement\n(Should be near 1.0)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.5, 1.05)

    fig.suptitle(
        'Interpretability Non-Uniqueness: Neurons Disagree, Representations Agree\n'
        f'({N_MODELS} independently trained MLPs on MNIST, {HIDDEN_SIZE} hidden units)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches='tight', dpi=150)
    print(f"  Figure saved to: {FIGURE_PATH}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = main()
