"""
Task 1C: Concept Probe Instability Experiment
Research question: Do functionally equivalent neural networks encode the same concept
in incompatible directions?

NEGATIVE CONTROL:
- Probe for the OUTPUT CLASS ITSELF (e.g., "is this digit 0?")
- All 15 models achieve >95% accuracy on the same class
- Probe direction for the output class should be stable
- Expected |cosine similarity|: >0.8

RESOLUTION TEST:
- Compute averaged CAV across all 15 models
- Measure cosine similarity between averaged CAV and each individual CAV
- Averaged CAV should be a better "consensus direction"
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


# ─── NEGATIVE CONTROL ──────────────────────────────────────────────────────────

def run_negative_control_concept(models, X_test, y_test, target_class=0):
    """
    Negative control: probe for the OUTPUT CLASS ITSELF.

    All 15 models achieve >95% accuracy on the same class.
    The probe direction for the output class should be stable across models.
    Expected |cosine similarity|: >0.8.

    We probe for "is this digit TARGET_CLASS?" using the penultimate activations.
    """
    print("\n" + "=" * 60)
    print(f"NEGATIVE CONTROL: Probing for output class {target_class} itself")
    print("=" * 60)

    # Binary labels: 1 if digit == target_class, 0 otherwise
    class_labels = (y_test == target_class).astype(int)
    print(f"  Positive class: digit {target_class} ({class_labels.sum()} samples)")
    print(f"  Negative class: all other digits ({(class_labels == 0).sum()} samples)")

    cavs_class = []
    for i, model in enumerate(models):
        act = get_penultimate_activations(model, X_test)
        cav = extract_cav(act, class_labels)
        cavs_class.append(cav)

    # Cosine similarity matrix
    cos_mat_class = cosine_similarity_matrix(cavs_class)
    n = len(cavs_class)
    off_diag = cos_mat_class[np.triu_indices(n, k=1)]
    mean_cos_class = float(np.mean(off_diag))

    ci_cos_class = bootstrap_ci(off_diag.tolist())
    print(f"  Mean |cos sim| (off-diag): {mean_cos_class:.4f}  "
          f"95% CI [{ci_cos_class[0]:.4f}, {ci_cos_class[2]:.4f}]")
    print(f"  Expected: >0.80 (class probe direction is stable)")
    print(f"  Compared to concept-A instability — larger value = more stable")

    return {
        "description": f"probe for output class {target_class} itself",
        "target_class": int(target_class),
        "n_positive_samples": int(class_labels.sum()),
        "mean_abs_cosine_similarity": float(mean_cos_class),
        "cosine_similarity_ci_lo": float(ci_cos_class[0]),
        "cosine_similarity_ci_hi": float(ci_cos_class[2]),
        "cavs_class": [v.tolist() for v in cavs_class],
        "interpretation": "Expected >0.80; class probe is stable because all models solve same class",
    }


# ─── RESOLUTION TEST ───────────────────────────────────────────────────────────

def run_resolution_test_concept(cavs_a, X_test, y_test, models):
    """
    Resolution test: compute the averaged CAV across all 15 models.

    Measure cosine similarity between averaged CAV and each individual CAV.
    The averaged CAV should be a better "consensus direction" — each individual
    CAV should be closer to the average than to a random direction.
    """
    print("\n" + "=" * 60)
    print("RESOLUTION TEST: Averaged CAV as consensus direction")
    print("=" * 60)

    # Average CAV (sum of unit vectors, then re-normalize)
    cavs_arr = np.array(cavs_a)  # (15, 64)
    avg_cav = cavs_arr.mean(axis=0)
    avg_norm = np.linalg.norm(avg_cav)
    if avg_norm > 0:
        avg_cav = avg_cav / avg_norm

    # Cosine similarity of each individual CAV with the averaged CAV
    cos_sims_with_avg = []
    for cav in cavs_a:
        cos_sims_with_avg.append(abs(np.dot(cav, avg_cav)))

    cos_sims_with_avg = np.array(cos_sims_with_avg)
    mean_cos_with_avg = float(np.mean(cos_sims_with_avg))
    ci_avg = bootstrap_ci(cos_sims_with_avg.tolist())

    print(f"  Mean |cos(individual, average)|: {mean_cos_with_avg:.4f}  "
          f"95% CI [{ci_avg[0]:.4f}, {ci_avg[2]:.4f}]")

    # For comparison: pairwise individual-to-individual cosine similarity
    n = len(cavs_a)
    pairwise_cos = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_cos.append(abs(np.dot(cavs_a[i], cavs_a[j])))
    mean_pairwise = float(np.mean(pairwise_cos))
    ci_pairwise = bootstrap_ci(pairwise_cos)

    print(f"  Mean pairwise |cos(i, j)|:       {mean_pairwise:.4f}  "
          f"95% CI [{ci_pairwise[0]:.4f}, {ci_pairwise[2]:.4f}]")
    print(f"  Improvement (avg vs pairwise): {(mean_cos_with_avg - mean_pairwise)*100:.1f} pp")
    print(f"  Expected: mean_cos_with_avg > mean_pairwise (averaged CAV is more central)")

    return {
        "description": "averaged CAV as consensus direction for concept A (curved vs angular)",
        "mean_cos_individual_to_avg": float(mean_cos_with_avg),
        "cos_individual_to_avg_ci_lo": float(ci_avg[0]),
        "cos_individual_to_avg_ci_hi": float(ci_avg[2]),
        "mean_pairwise_cos": float(mean_pairwise),
        "pairwise_cos_ci_lo": float(ci_pairwise[0]),
        "pairwise_cos_ci_hi": float(ci_pairwise[2]),
        "improvement_pp": float((mean_cos_with_avg - mean_pairwise) * 100),
        "averaged_cav": avg_cav.tolist(),
        "interpretation": "avg CAV should have higher similarity to each individual than pairwise average",
    }


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

    # ─── POSITIVE TEST ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POSITIVE TEST: Rashomon concept probes (curved vs angular)")
    print("=" * 60)

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
        act_a = get_penultimate_activations(model, X_concept_a)
        cav_a = extract_cav(act_a, y_concept_a_labels)
        cavs_a.append(cav_a)

        act_test = get_penultimate_activations(model, X_test)
        cav_b = extract_cav(act_test, y_concept_b_labels)
        cavs_b.append(cav_b)

        score_a = tcav_score(model, X_test, y_test, cav_a)
        score_b = tcav_score(model, X_test, y_test, cav_b)
        tcav_scores_a.append(score_a)
        tcav_scores_b.append(score_b)

    tcav_scores_a = np.array(tcav_scores_a)
    tcav_scores_b = np.array(tcav_scores_b)

    # Cosine similarity matrix for Concept A
    cos_mat = cosine_similarity_matrix(cavs_a)

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

    # Prediction agreement across 15 models
    all_preds = np.array([m.predict(X_test) for m in models])
    agreement = np.mean(np.all(all_preds == all_preds[0], axis=0))
    print(f"\nPrediction agreement (all 15 models): {agreement:.4f}")

    # Bootstrap CIs (positive test)
    ci_instability = bootstrap_ci([1 - abs(np.dot(cavs_a[i], cavs_a[j]))
                                    for i in range(n) for j in range(i+1, n)])
    ci_tcav_a = bootstrap_ci(tcav_scores_a)
    ci_tcav_b = bootstrap_ci(tcav_scores_b)
    ci_agreement = percentile_ci(
        [float(np.all(all_preds[:, k] == all_preds[0, k])) for k in range(all_preds.shape[1])]
    )

    print(f"\n95% Bootstrap CIs (positive test):")
    print(f"  Concept direction instability: [{ci_instability[0]:.4f}, {ci_instability[2]:.4f}]")
    print(f"  TCAV-A std CI:                 [{ci_tcav_a[0]:.4f}, {ci_tcav_a[2]:.4f}]")
    print(f"  TCAV-B std CI:                 [{ci_tcav_b[0]:.4f}, {ci_tcav_b[2]:.4f}]")
    print(f"  Prediction agreement CI:       [{ci_agreement[0]:.4f}, {ci_agreement[2]:.4f}]")

    # ─── NEGATIVE CONTROL ─────────────────────────────────────────────────────
    nc_results = run_negative_control_concept(models, X_test, y_test, target_class=0)

    # ─── RESOLUTION TEST ──────────────────────────────────────────────────────
    res_results = run_resolution_test_concept(cavs_a, X_test, y_test, models)

    # ─── Figure ────────────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    # Panel A: Heatmap of |cosine similarity| 15×15 (positive test - concept A)
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(cos_mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r'$|\cos(\mathbf{v}_i,\,\mathbf{v}_j)|$', fontsize=10)
    ax1.set_title('(a) Positive Test\nConcept Direction Alignment\n(Curved vs. Angular)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Model index', fontsize=10)
    ax1.set_ylabel('Model index', fontsize=10)
    ax1.set_xticks(range(N_MODELS))
    ax1.set_yticks(range(N_MODELS))
    ax1.set_xticklabels(range(1, N_MODELS + 1), fontsize=7)
    ax1.set_yticklabels(range(1, N_MODELS + 1), fontsize=7)
    ax1.text(0.03, 0.97,
             f'Mean off-diag $= {mean_off_diag:.3f}$\nInstability $= {concept_direction_instability:.3f}$',
             transform=ax1.transAxes, va='top', ha='left', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel B: Negative control cosine matrix (output class probe)
    nc_cavs = nc_results['cavs_class']
    nc_cavs_arr = [np.array(v) for v in nc_cavs]
    cos_mat_nc = cosine_similarity_matrix(nc_cavs_arr)
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(cos_mat_nc, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r'$|\cos(\mathbf{v}_i,\,\mathbf{v}_j)|$', fontsize=10)
    nc_off_diag = cos_mat_nc[np.triu_indices(N_MODELS, k=1)]
    ax2.set_title(f'(b) Negative Control\nClass-{nc_results["target_class"]} Probe Alignment\n(Expected: stable)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Model index', fontsize=10)
    ax2.set_ylabel('Model index', fontsize=10)
    ax2.set_xticks(range(N_MODELS))
    ax2.set_yticks(range(N_MODELS))
    ax2.set_xticklabels(range(1, N_MODELS + 1), fontsize=7)
    ax2.set_yticklabels(range(1, N_MODELS + 1), fontsize=7)
    ax2.text(0.03, 0.97,
             f'Mean off-diag $= {nc_results["mean_abs_cosine_similarity"]:.3f}$\n(vs. concept A: {mean_off_diag:.3f})',
             transform=ax2.transAxes, va='top', ha='left', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Panel C: Resolution — bar chart comparing avg CAV vs pairwise
    ax3 = fig.add_subplot(gs[2])
    bar_labels_res = ['Pairwise\n|cos(i,j)|\n(instability)', 'Indiv. vs\nAvg CAV\n(resolution)',
                      'Neg. Control\n|cos(i,j)|\n(class probe)']
    bar_vals_res = [res_results['mean_pairwise_cos'],
                    res_results['mean_cos_individual_to_avg'],
                    nc_results['mean_abs_cosine_similarity']]
    bar_errs_lo_res = [res_results['mean_pairwise_cos'] - res_results['pairwise_cos_ci_lo'],
                       res_results['mean_cos_individual_to_avg'] - res_results['cos_individual_to_avg_ci_lo'],
                       nc_results['mean_abs_cosine_similarity'] - nc_results['cosine_similarity_ci_lo']]
    bar_errs_hi_res = [res_results['pairwise_cos_ci_hi'] - res_results['mean_pairwise_cos'],
                       res_results['cos_individual_to_avg_ci_hi'] - res_results['mean_cos_individual_to_avg'],
                       nc_results['cosine_similarity_ci_hi'] - nc_results['mean_abs_cosine_similarity']]

    bar_colors_res = ['#D55E00', '#0072B2', '#009E73']
    xs_res = np.arange(len(bar_labels_res))
    bars_res = ax3.bar(xs_res, bar_vals_res, color=bar_colors_res, alpha=0.8, width=0.6,
                       yerr=[bar_errs_lo_res, bar_errs_hi_res],
                       error_kw=dict(elinewidth=1.0, capsize=3, ecolor='#333333'))
    ax3.set_xticks(xs_res)
    ax3.set_xticklabels(bar_labels_res, fontsize=8)
    ax3.set_ylabel('Mean |cosine similarity|', fontsize=9)
    ax3.set_title('(c) Resolution Test\nAveraged CAV vs Pairwise', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
                label='Expected control (0.8)')
    ax3.legend(fontsize=7.5, loc='lower right')

    for bar, val in zip(bars_res, bar_vals_res):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    fig.suptitle('Concept Probe Instability: Equivalent Models, Divergent Directions\n'
                 '+ Negative Control (class probe) + Resolution (averaged CAV)',
                 fontsize=11, fontweight='bold', y=1.01)

    save_figure(fig, 'concept_probe_instability')

    # ─── LaTeX table ────────────────────────────────────────────────────────────
    print("Generating LaTeX table …")
    sections_dir = Path(__file__).resolve().parent.parent / 'sections'
    sections_dir.mkdir(exist_ok=True)
    table_path = sections_dir / 'table_concept.tex'

    latex = r"""\begin{table}[t]
\centering
\caption{Concept probe instability across 15 equivalent MLPClassifier models on
\texttt{sklearn} \texttt{load\_digits()} (8$\times$8 images, 10 classes).
Architecture: $(128, 64)$ hidden units, ReLU.
\emph{Negative control}: probe for output class itself --- directions should be stable ($>0.80$).
\emph{Resolution}: averaged CAV across 15 models --- individual CAVs should be closer to the average.
All 95\% CIs from 2000 bootstrap resamples.}
\label{tab:concept_probe}
\begin{tabular}{llcc}
\toprule
\textbf{Test} & \textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\
\midrule
"""
    latex += f"\\multirow{{5}}{{*}}{{Positive (Rashomon)}} & Mean test accuracy & ${mean_acc:.4f}$ & $[{min(accuracies):.4f},\\,{max(accuracies):.4f}]$ \\\\\n"
    latex += f"& Min test accuracy & ${min_acc:.4f}$ & --- \\\\\n"
    latex += (f"& Mean $|\\cos(\\mathbf{{v}}_i,\\mathbf{{v}}_j)|$ (off-diag) & "
              f"${mean_off_diag:.4f}$ & $[{ci_instability[0]:.4f},\\,{ci_instability[2]:.4f}]$ \\\\\n")
    latex += (f"& Concept direction instability & "
              f"${concept_direction_instability:.4f}$ & --- \\\\\n")
    latex += (f"& Prediction agreement & "
              f"${agreement:.4f}$ & $[{ci_agreement[0]:.4f},\\,{ci_agreement[2]:.4f}]$ \\\\\n")
    latex += r"\midrule" + "\n"
    latex += (f"Neg.\\ control (class-{nc_results['target_class']} probe) & "
              f"Mean $|\\cos(\\mathbf{{v}}_i,\\mathbf{{v}}_j)|$ & "
              f"${nc_results['mean_abs_cosine_similarity']:.4f}$ & "
              f"$[{nc_results['cosine_similarity_ci_lo']:.4f},\\,{nc_results['cosine_similarity_ci_hi']:.4f}]$ \\\\\n")
    latex += r"\midrule" + "\n"
    latex += (f"\\multirow{{2}}{{*}}{{Resolution (avg.\\ CAV)}} & "
              f"Mean $|\\cos(\\text{{indiv}},\\text{{avg}})|$ & "
              f"${res_results['mean_cos_individual_to_avg']:.4f}$ & "
              f"$[{res_results['cos_individual_to_avg_ci_lo']:.4f},\\,{res_results['cos_individual_to_avg_ci_hi']:.4f}]$ \\\\\n")
    latex += (f"& Mean pairwise $|\\cos(i,j)|$ & "
              f"${res_results['mean_pairwise_cos']:.4f}$ & "
              f"$[{res_results['pairwise_cos_ci_lo']:.4f},\\,{res_results['pairwise_cos_ci_hi']:.4f}]$ \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    table_path.write_text(latex)
    print(f"Saved table: {table_path}")

    # ─── Save JSON results ──────────────────────────────────────────────────────
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
        'negative_control': {k: v for k, v in nc_results.items() if k != 'cavs_class'},
        'resolution_test': {k: v for k, v in res_results.items() if k != 'averaged_cav'},
    }
    save_results(results, 'concept_probe_instability')

    # ─── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("POSITIVE TEST (Rashomon models):")
    print(f"  Models trained:              {N_MODELS}")
    print(f"  All >95% accuracy:           {all(a >= MIN_ACCURACY for a in accuracies)}")
    print(f"  Mean accuracy:               {mean_acc:.4f}")
    print(f"  Mean |cos sim| (off-diag):   {mean_off_diag:.4f}  (expected <0.6)")
    print(f"  Concept direction instab.:   {concept_direction_instability:.4f}")
    print(f"  TCAV std (curved):           {tcav_scores_a.std():.4f}  (expected >0.1)")
    print(f"  Prediction agreement:        {agreement:.4f}  (expected >0.95)")
    print()
    print("NEGATIVE CONTROL (class probe):")
    print(f"  Mean |cos sim| (class {nc_results['target_class']} probe): "
          f"{nc_results['mean_abs_cosine_similarity']:.4f}  "
          f"[{nc_results['cosine_similarity_ci_lo']:.4f}, {nc_results['cosine_similarity_ci_hi']:.4f}]")
    print(f"  Expected: >0.80")
    print()
    print("RESOLUTION TEST (averaged CAV):")
    print(f"  Mean |cos(indiv, avg)|:  {res_results['mean_cos_individual_to_avg']:.4f}  "
          f"[{res_results['cos_individual_to_avg_ci_lo']:.4f}, {res_results['cos_individual_to_avg_ci_hi']:.4f}]")
    print(f"  Mean pairwise |cos|:     {res_results['mean_pairwise_cos']:.4f}  "
          f"[{res_results['pairwise_cos_ci_lo']:.4f}, {res_results['pairwise_cos_ci_hi']:.4f}]")
    print(f"  Improvement:             {res_results['improvement_pp']:.1f} pp")
    print("=" * 60)


if __name__ == '__main__':
    main()
