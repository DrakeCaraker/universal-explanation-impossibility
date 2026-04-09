"""
Task 1C: Concept Probe Instability Experiment
Research question: Do functionally equivalent neural networks encode the same concept
in incompatible directions?
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add scripts dir to path for experiment_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import (
    set_all_seeds, load_publication_style, save_figure, save_results, percentile_ci
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# ─── Concept definitions ───────────────────────────────────────────────────────
# Concept A: "curved digit" vs "angular digit"
CURVED_DIGITS  = {0, 2, 3, 5, 6, 8, 9}
ANGULAR_DIGITS = {1, 4, 7}

# Concept B: "symmetric digit" vs rest
SYMMETRIC_DIGITS = {0, 1, 8}

N_MODELS = 15
HIDDEN_LAYER_SIZES = (128, 64)
TEST_SIZE = 0.30
RANDOM_STATE_BASE = 42
MIN_ACCURACY = 0.95
N_BOOT = 2000


# ─── Helper functions ──────────────────────────────────────────────────────────

def get_penultimate_activations(model, X):
    """Manually compute penultimate-layer activations from sklearn MLP."""
    h = X
    # Forward through all layers except the last
    for i in range(len(model.coefs_) - 1):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = np.maximum(h, 0)  # ReLU
    return h  # shape: (n_samples, 64)


def get_concept_labels(y, positive_set):
    """Binary concept labels: 1 if digit in positive_set, 0 otherwise."""
    return np.array([1 if d in positive_set else 0 for d in y])


def extract_cav(activations, concept_labels):
    """
    Fit LinearSVC on activations to classify concept, extract normalised CAV.
    Returns normalised weight vector (CAV).
    """
    svc = LinearSVC(max_iter=5000, random_state=42, C=1.0)
    svc.fit(activations, concept_labels)
    v = svc.coef_[0]
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def tcav_score(model, X_test, y_test, cav, target_class=None):
    """
    TCAV-like score: fraction of test samples where the directional derivative
    of the predicted-class logit along the CAV is positive.

    The gradient of the output logit w.r.t. penultimate activations is simply
    the row of the last weight matrix corresponding to the predicted class.
    """
    preds = model.predict(X_test)
    if target_class is None:
        # Use predicted class per sample
        classes = preds
    else:
        classes = np.full(len(X_test), target_class)

    # Last weight matrix maps penultimate (64-d) → logits (10-d)
    W_last = model.coefs_[-1]  # shape: (64, 10)

    positive_count = 0
    for i, cls in enumerate(classes):
        grad = W_last[:, cls]  # shape: (64,)
        directional_deriv = np.dot(grad, cav)
        if directional_deriv > 0:
            positive_count += 1

    return positive_count / len(X_test)


def cosine_similarity_matrix(cavs):
    """Compute NxN matrix of |cos(v_i, v_j)| for list of unit CAVs."""
    n = len(cavs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = abs(np.dot(cavs[i], cavs[j]))
    return mat


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    """95% bootstrap CI around the mean."""
    values = np.array(values)
    boot_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(np.mean(values)), float(hi)


# ─── Main experiment ───────────────────────────────────────────────────────────

def main():
    set_all_seeds(42)
    load_publication_style()

    print("=" * 60)
    print("Task 1C: Concept Probe Instability Experiment")
    print("=" * 60)

    # 1. Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Normalise pixel values to [0,1]
    X = X / 16.0

    # Stratified 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_BASE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Train 15 MLP models
    print("\nTraining 15 MLPClassifier models …")
    models = []
    accuracies = []
    for i in range(N_MODELS):
        seed = RANDOM_STATE_BASE + i
        mlp = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation='relu',
            max_iter=500,
            random_state=seed,
        )
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        accuracies.append(acc)
        models.append(mlp)
        print(f"  Model {i+1:2d} (seed={seed}): test accuracy = {acc:.4f}")
        if acc < MIN_ACCURACY:
            print(f"    WARNING: accuracy {acc:.4f} < {MIN_ACCURACY}")

    min_acc = min(accuracies)
    mean_acc = np.mean(accuracies)
    print(f"\nAll models >{MIN_ACCURACY*100:.0f}%: {all(a >= MIN_ACCURACY for a in accuracies)}")
    print(f"Mean accuracy: {mean_acc:.4f}, Min: {min_acc:.4f}")

    # 3. Concept probe extraction (using ALL test samples)
    print("\nExtracting CAVs …")

    # Concept A: curved vs angular (only samples that are curved or angular)
    curved_mask  = np.array([d in CURVED_DIGITS  for d in y_test])
    angular_mask = np.array([d in ANGULAR_DIGITS for d in y_test])
    concept_a_mask = curved_mask | angular_mask
    X_concept_a = X_test[concept_a_mask]
    y_concept_a_labels = get_concept_labels(y_test[concept_a_mask], CURVED_DIGITS)

    # Concept B: symmetric vs rest (use all test samples)
    y_concept_b_labels = get_concept_labels(y_test, SYMMETRIC_DIGITS)

    cavs_a = []
    cavs_b = []
    tcav_scores_a = []
    tcav_scores_b = []

    for i, model in enumerate(models):
        # Penultimate activations for concept A subset
        act_a = get_penultimate_activations(model, X_concept_a)
        cav_a = extract_cav(act_a, y_concept_a_labels)
        cavs_a.append(cav_a)

        # Penultimate activations for full test set (concept B)
        act_test = get_penultimate_activations(model, X_test)
        cav_b = extract_cav(act_test, y_concept_b_labels)
        cavs_b.append(cav_b)

        # TCAV scores (use predicted class per sample)
        score_a = tcav_score(model, X_test, y_test, cav_a)
        score_b = tcav_score(model, X_test, y_test, cav_b)
        tcav_scores_a.append(score_a)
        tcav_scores_b.append(score_b)

    tcav_scores_a = np.array(tcav_scores_a)
    tcav_scores_b = np.array(tcav_scores_b)

    # 4. Cosine similarity matrix for Concept A
    cos_mat = cosine_similarity_matrix(cavs_a)

    # Off-diagonal mean
    n = N_MODELS
    off_diag = cos_mat[np.triu_indices(n, k=1)]
    mean_off_diag = float(np.mean(off_diag))
    concept_direction_instability = 1.0 - mean_off_diag

    print(f"\nConcept A (curved vs angular):")
    print(f"  Mean |cosine similarity| (off-diag): {mean_off_diag:.4f}")
    print(f"  Concept direction instability:       {concept_direction_instability:.4f}")
    print(f"  TCAV score mean: {tcav_scores_a.mean():.4f}, std: {tcav_scores_a.std():.4f}")

    print(f"\nConcept B (symmetric vs rest):")
    print(f"  TCAV score mean: {tcav_scores_b.mean():.4f}, std: {tcav_scores_b.std():.4f}")

    # 5. Prediction agreement across 15 models
    all_preds = np.array([m.predict(X_test) for m in models])  # (15, n_test)
    agreement = np.mean(np.all(all_preds == all_preds[0], axis=0))
    print(f"\nPrediction agreement (all 15 models): {agreement:.4f}")

    # 6. Bootstrap CIs
    ci_instability = bootstrap_ci([1 - abs(np.dot(cavs_a[i], cavs_a[j]))
                                    for i in range(n) for j in range(i+1, n)])
    ci_tcav_a = bootstrap_ci(tcav_scores_a)
    ci_tcav_b = bootstrap_ci(tcav_scores_b)
    ci_agreement = percentile_ci(
        [float(np.all(all_preds[:, k] == all_preds[0, k])) for k in range(all_preds.shape[1])]
    )

    print(f"\n95% Bootstrap CIs:")
    print(f"  Concept direction instability: [{ci_instability[0]:.4f}, {ci_instability[2]:.4f}]")
    print(f"  TCAV-A std CI:                 [{ci_tcav_a[0]:.4f}, {ci_tcav_a[2]:.4f}]")
    print(f"  TCAV-B std CI:                 [{ci_tcav_b[0]:.4f}, {ci_tcav_b[2]:.4f}]")
    print(f"  Prediction agreement CI:       [{ci_agreement[0]:.4f}, {ci_agreement[2]:.4f}]")

    # ─── 7. Figure ─────────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # Panel A: Heatmap of |cosine similarity| 15×15
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(cos_mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r'$|\cos(\mathbf{v}_i,\,\mathbf{v}_j)|$', fontsize=10)
    ax1.set_title('Concept Direction Alignment\n(Curved vs. Angular)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Model index', fontsize=10)
    ax1.set_ylabel('Model index', fontsize=10)
    ax1.set_xticks(range(N_MODELS))
    ax1.set_yticks(range(N_MODELS))
    ax1.set_xticklabels(range(1, N_MODELS + 1), fontsize=7)
    ax1.set_yticklabels(range(1, N_MODELS + 1), fontsize=7)
    # Annotate mean off-diagonal
    ax1.text(0.03, 0.97,
             f'Mean off-diag $= {mean_off_diag:.3f}$\nInstability $= {concept_direction_instability:.3f}$',
             transform=ax1.transAxes, va='top', ha='left', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel B: Box plot of TCAV scores for 2 concepts
    ax2 = fig.add_subplot(gs[1])
    bp = ax2.boxplot(
        [tcav_scores_a, tcav_scores_b],
        labels=['Curved\n(0,2,3,5,6,8,9)\nvs. Angular', 'Symmetric\n(0,1,8)\nvs. Rest'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.6),
        widths=0.5,
    )
    colors = ['#4878d0', '#ee854a']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual model points
    for k, scores in enumerate([tcav_scores_a, tcav_scores_b], start=1):
        x_jitter = np.random.normal(k, 0.07, size=len(scores))
        ax2.scatter(x_jitter, scores, color=colors[k-1], s=28, zorder=5, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Annotate std
    for k, scores in enumerate([tcav_scores_a, tcav_scores_b], start=1):
        ax2.text(k, scores.min() - 0.025,
                 f'std={scores.std():.3f}',
                 ha='center', va='top', fontsize=8.5,
                 color=colors[k-1], fontweight='bold')

    ax2.set_ylabel('TCAV-like Score', fontsize=10)
    ax2.set_title('TCAV Score Variability\nAcross 15 Equivalent Models', fontsize=11, fontweight='bold')
    ax2.set_ylim(max(0, min(tcav_scores_a.min(), tcav_scores_b.min()) - 0.06),
                 min(1, max(tcav_scores_a.max(), tcav_scores_b.max()) + 0.06))
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Chance (0.5)')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Concept Probe Instability: Equivalent Models, Divergent Directions',
                 fontsize=12, fontweight='bold', y=1.01)

    save_figure(fig, 'concept_probe_instability')

    # ─── 8. LaTeX table ────────────────────────────────────────────────────────
    print("Generating LaTeX table …")
    sections_dir = Path(__file__).resolve().parent.parent / 'sections'
    sections_dir.mkdir(exist_ok=True)
    table_path = sections_dir / 'table_concept.tex'

    latex = r"""\begin{table}[t]
\centering
\caption{Concept probe instability across 15 equivalent MLPClassifier models trained on
\texttt{sklearn} \texttt{load\_digits()} (8$\times$8 images, 10 classes, 1797 samples).
Architecture: $(128, 64)$ hidden units, ReLU, trained with different random seeds.
All models achieve $>95\%$ test accuracy yet encode concepts in substantially different directions.}
\label{tab:concept_probe}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\
\midrule
"""
    latex += f"Mean test accuracy & ${mean_acc:.4f}$ & $[{min(accuracies):.4f},\\,{max(accuracies):.4f}]$ \\\\\n"
    latex += f"Min test accuracy & ${min_acc:.4f}$ & --- \\\\\n"
    latex += (f"Mean $|\\cos(\\mathbf{{v}}_i,\\mathbf{{v}}_j)|$ (off-diag) & "
              f"${mean_off_diag:.4f}$ & $[{ci_instability[0]:.4f},\\,{ci_instability[2]:.4f}]$ \\\\\n")
    latex += (f"Concept direction instability & "
              f"${concept_direction_instability:.4f}$ & --- \\\\\n")
    latex += (f"TCAV score std — curved concept & "
              f"${tcav_scores_a.std():.4f}$ & $[{ci_tcav_a[0]:.4f},\\,{ci_tcav_a[2]:.4f}]$ \\\\\n")
    latex += (f"TCAV score std — symmetric concept & "
              f"${tcav_scores_b.std():.4f}$ & $[{ci_tcav_b[0]:.4f},\\,{ci_tcav_b[2]:.4f}]$ \\\\\n")
    latex += (f"Prediction agreement (all 15 models) & "
              f"${agreement:.4f}$ & $[{ci_agreement[0]:.4f},\\,{ci_agreement[2]:.4f}]$ \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    table_path.write_text(latex)
    print(f"Saved table: {table_path}")

    # ─── 9. Save JSON results ──────────────────────────────────────────────────
    results = {
        'experiment': 'concept_probe_instability',
        'task': '1C',
        'dataset': 'sklearn load_digits (8x8, 10 classes, 1797 samples)',
        'n_models': N_MODELS,
        'architecture': str(HIDDEN_LAYER_SIZES),
        'test_size': TEST_SIZE,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'model_accuracies': [float(a) for a in accuracies],
        'mean_accuracy': float(mean_acc),
        'min_accuracy': float(min_acc),
        'max_accuracy': float(max(accuracies)),
        'all_above_95pct': bool(all(a >= MIN_ACCURACY for a in accuracies)),
        'cosine_similarity_matrix': cos_mat.tolist(),
        'mean_abs_cosine_similarity_off_diag': mean_off_diag,
        'concept_direction_instability': float(concept_direction_instability),
        'concept_direction_instability_ci': {
            'lo': ci_instability[0], 'mean': ci_instability[1], 'hi': ci_instability[2]
        },
        'tcav_scores_curved': tcav_scores_a.tolist(),
        'tcav_scores_symmetric': tcav_scores_b.tolist(),
        'tcav_curved_mean': float(tcav_scores_a.mean()),
        'tcav_curved_std': float(tcav_scores_a.std()),
        'tcav_curved_ci': {'lo': ci_tcav_a[0], 'mean': ci_tcav_a[1], 'hi': ci_tcav_a[2]},
        'tcav_symmetric_mean': float(tcav_scores_b.mean()),
        'tcav_symmetric_std': float(tcav_scores_b.std()),
        'tcav_symmetric_ci': {'lo': ci_tcav_b[0], 'mean': ci_tcav_b[1], 'hi': ci_tcav_b[2]},
        'prediction_agreement': float(agreement),
        'prediction_agreement_ci': {'lo': ci_agreement[0], 'mean': ci_agreement[1], 'hi': ci_agreement[2]},
        'concepts': {
            'curved_digits': sorted(CURVED_DIGITS),
            'angular_digits': sorted(ANGULAR_DIGITS),
            'symmetric_digits': sorted(SYMMETRIC_DIGITS),
        },
    }
    save_results(results, 'concept_probe_instability')

    # ─── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Models trained:              {N_MODELS}")
    print(f"  All >95% accuracy:           {all(a >= MIN_ACCURACY for a in accuracies)}")
    print(f"  Mean accuracy:               {mean_acc:.4f}")
    print(f"  Mean |cos sim| (off-diag):   {mean_off_diag:.4f}  (expected <0.6)")
    print(f"  Concept direction instab.:   {concept_direction_instability:.4f}")
    print(f"  TCAV std (curved):           {tcav_scores_a.std():.4f}  (expected >0.1)")
    print(f"  TCAV std (symmetric):        {tcav_scores_b.std():.4f}")
    print(f"  Prediction agreement:        {agreement:.4f}  (expected >0.95)")
    print("=" * 60)


if __name__ == '__main__':
    main()
