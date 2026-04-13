"""
Ensemble Causal Discovery Experiment — Cross-Domain Transfer Test
=================================================================

Research question: Does DASH-style ensemble averaging (orbit averaging)
transfer from ML attributions to causal discovery?

The universal impossibility framework predicts that "orbit averaging"
(ensemble methods) is the Pareto-optimal resolution for ANY domain with
the Rashomon property. DASH (ensemble averaging of SHAP values) was
developed for ML attributions. This experiment tests the cross-domain
transfer prediction: applying DASH-style ensemble averaging to CAUSAL
DISCOVERY should improve orientation stability.

DAG: Asia network (8 nodes, linear Gaussian) — same as
     causal_discovery_experiment.py

Three methods:
  1. SINGLE-RUN: PC(alpha=0.05) once → single CPDAG
  2. BOOTSTRAP-VOTE: PC on 50 bootstrap resamples, majority orientation
  3. DASH-STYLE ENSEMBLE: PC on 50 bootstrap resamples, mean orientation
     confidence → directed only if confidence > 0.7, else undirected

Metrics (trilemma-aware):
  - Orientation accuracy: among directed edges in true DAG that the method
    also directs, fraction with correct orientation (faithfulness)
  - Decisiveness: fraction of true DAG edges that the method commits to
    directing (vs. leaving undirected)
  - Cross-seed stability: for each pair of seeds, compute orientation
    agreement on the edges they both direct (direct analog of DASH
    stability in attribution space)

Framework prediction: Ensemble methods reduce instability (the variance
across datasets of what gets directed and how). DASH should be Pareto-
optimal: among methods with comparable accuracy, it has lowest instability.

Output:
  paper/results_ensemble_causal_transfer.json
  paper/figures/ensemble_causal_transfer.pdf
"""

import sys
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, '/Users/drake.caraker/Library/Python/3.9/lib/python/site-packages')

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    PAPER_DIR,
)

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Asia DAG (identical to causal_discovery_experiment.py) ────────────────────
NODE_NAMES = ['Asia', 'Tub', 'Smoke', 'Lung', 'Bronch', 'Either', 'Xray', 'Dysp']
N_NODES = 8

# True directed edges as (parent, child) pairs
TRUE_EDGES = [
    (0, 1),  # Asia -> Tub
    (2, 3),  # Smoke -> Lung
    (2, 4),  # Smoke -> Bronch
    (1, 5),  # Tub -> Either
    (3, 5),  # Lung -> Either
    (5, 6),  # Either -> Xray
    (5, 7),  # Either -> Dysp
    (4, 7),  # Bronch -> Dysp
]

# All possible canonical edge keys (i < j) for 8 nodes
ALL_PAIR_KEYS = [(i, j) for i in range(N_NODES) for j in range(i+1, N_NODES)]


def sample_asia_linear_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N observations from a linear Gaussian version of the Asia DAG.
    Identical to causal_discovery_experiment.py.
    """
    noise_std = 0.5
    coeff = 0.8

    asia   = rng.normal(0, 1, n)
    smoke  = rng.normal(0, 1, n)
    tub    = coeff * asia  + rng.normal(0, noise_std, n)
    lung   = coeff * smoke + rng.normal(0, noise_std, n)
    bronc  = coeff * smoke + rng.normal(0, noise_std, n)
    either = coeff * tub   + coeff * lung + rng.normal(0, noise_std, n)
    xray   = coeff * either + rng.normal(0, noise_std, n)
    dysp   = coeff * either + coeff * bronc + rng.normal(0, noise_std, n)

    data = np.column_stack([asia, tub, smoke, lung, bronc, either, xray, dysp])
    return data


def extract_adj(graph_obj) -> np.ndarray:
    """Extract adjacency matrix from causallearn GeneralGraph."""
    return graph_obj.graph.copy()


def adj_to_edge_dict(adj: np.ndarray) -> dict:
    """
    Convert adjacency matrix to dict mapping (i,j) -> orientation string.
    Only canonical pairs (i < j).
    """
    n = adj.shape[0]
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            aij = adj[i, j]
            aji = adj[j, i]
            if aij == 0 and aji == 0:
                continue
            key = (i, j)
            if aij == -1 and aji == -1:
                edges[key] = 'undirected'
            elif aij == -1 and aji == 1:
                edges[key] = 'i->j'
            elif aij == 1 and aji == -1:
                edges[key] = 'j->i'
            else:
                edges[key] = f'other({aij},{aji})'
    return edges


def run_pc_single(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Run PC algorithm once, return edge dict."""
    from causallearn.search.ConstraintBased.PC import pc
    result = pc(data, alpha=alpha, indep_test='fisherz',
                show_progress=False, verbose=False)
    adj = extract_adj(result.G)
    return adj_to_edge_dict(adj)


def true_edge_orientation(i: int, j: int) -> str:
    """Return true orientation for canonical pair (i < j)."""
    for (p, c) in TRUE_EDGES:
        if (p, c) == (i, j):
            return 'i->j'
        if (p, c) == (j, i):
            return 'j->i'
    return 'none'


def edges_to_orientation_vector(edges: dict) -> np.ndarray:
    """
    Convert edge dict to a vector over ALL possible pairs.
    Encoding: +1 = i->j, -1 = j->i, 0 = undirected/absent.
    This gives a fixed-length representation for cross-seed comparison.
    """
    vec = np.zeros(len(ALL_PAIR_KEYS))
    for idx, key in enumerate(ALL_PAIR_KEYS):
        if key in edges:
            orient = edges[key]
            if orient == 'i->j':
                vec[idx] = 1.0
            elif orient == 'j->i':
                vec[idx] = -1.0
            # undirected or other → 0
    return vec


def compute_accuracy_decisiveness(edges: dict) -> dict:
    """
    Compute trilemma metrics against the true DAG.
    accuracy: among true-DAG edges the method directs, fraction correct.
    decisiveness: fraction of true-DAG edges that the method directs.
    """
    n_correct = 0
    n_directed_true = 0
    n_true = len(TRUE_EDGES)

    for (p, c) in TRUE_EDGES:
        key = (min(p, c), max(p, c))
        if key not in edges:
            continue
        orient = edges[key]
        true_orient = true_edge_orientation(key[0], key[1])
        if orient in ('i->j', 'j->i'):
            n_directed_true += 1
            if orient == true_orient:
                n_correct += 1

    accuracy = n_correct / n_directed_true if n_directed_true > 0 else float('nan')
    decisiveness = n_directed_true / n_true

    return {
        'accuracy': accuracy,
        'decisiveness': decisiveness,
        'n_correct': n_correct,
        'n_directed_true': n_directed_true,
    }


def bootstrap_resample(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap resample rows of data."""
    n = data.shape[0]
    indices = rng.integers(0, n, size=n)
    return data[indices]


def run_bootstrap_pc(data: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Run PC on n_boot bootstrap resamples. Return list of edge dicts."""
    all_edges = []
    for b in range(n_boot):
        boot_data = bootstrap_resample(data, rng)
        edges = run_pc_single(boot_data, alpha=0.05)
        all_edges.append(edges)
    return all_edges


def aggregate_edges(all_edges: list, n_boot: int, mode: str,
                    threshold: float = 0.7) -> dict:
    """
    Aggregate bootstrap edge results.
    mode='vote': majority orientation wins
    mode='dash': directed only if confidence > threshold, else undirected
    """
    all_keys = set()
    for e in all_edges:
        all_keys.update(e.keys())

    consensus = {}
    for key in all_keys:
        counts = {'i->j': 0, 'j->i': 0, 'undirected': 0, 'absent': 0}
        for e in all_edges:
            if key in e:
                orient = e[key]
                if orient in counts:
                    counts[orient] += 1
                else:
                    counts['undirected'] += 1
            else:
                counts['absent'] += 1

        present_total = counts['i->j'] + counts['j->i'] + counts['undirected']
        if present_total < n_boot * 0.5:
            continue

        if mode == 'vote':
            if counts['i->j'] > counts['j->i'] and counts['i->j'] > counts['undirected']:
                consensus[key] = 'i->j'
            elif counts['j->i'] > counts['i->j'] and counts['j->i'] > counts['undirected']:
                consensus[key] = 'j->i'
            else:
                consensus[key] = 'undirected'

        elif mode == 'dash':
            conf_ij = counts['i->j'] / present_total
            conf_ji = counts['j->i'] / present_total
            if conf_ij > threshold:
                consensus[key] = 'i->j'
            elif conf_ji > threshold:
                consensus[key] = 'j->i'
            else:
                consensus[key] = 'undirected'

    return consensus


def run_one_seed(seed: int, n_samples: int = 1000, n_boot: int = 50):
    """Run all three methods for one seed. Return edges and metrics."""
    rng = np.random.default_rng(seed)
    data = sample_asia_linear_gaussian(n_samples, rng)

    # Method 1: Single run
    edges_single = run_pc_single(data, alpha=0.05)

    # Methods 2 and 3 share the same bootstrap resamples
    rng_boot = np.random.default_rng(seed + 10000)
    all_boot_edges = run_bootstrap_pc(data, n_boot, rng_boot)

    edges_vote = aggregate_edges(all_boot_edges, n_boot, mode='vote')
    edges_dash = aggregate_edges(all_boot_edges, n_boot, mode='dash', threshold=0.7)

    return {
        'seed': seed,
        'edges_single': edges_single,
        'edges_vote': edges_vote,
        'edges_dash': edges_dash,
        'metrics_single': compute_accuracy_decisiveness(edges_single),
        'metrics_vote': compute_accuracy_decisiveness(edges_vote),
        'metrics_dash': compute_accuracy_decisiveness(edges_dash),
        'vec_single': edges_to_orientation_vector(edges_single),
        'vec_vote': edges_to_orientation_vector(edges_vote),
        'vec_dash': edges_to_orientation_vector(edges_dash),
    }


def compute_cross_seed_stability(all_results: list, vec_key: str) -> dict:
    """
    Compute cross-seed orientation stability.
    For each pair of seeds, compute agreement on the orientation vector.
    Agreement = fraction of pairs where orientation matches (same sign,
    including both being 0).
    This is the direct causal-discovery analog of DASH's cross-model
    attribution stability.
    """
    n = len(all_results)
    vecs = [r[vec_key] for r in all_results]
    n_pairs = n * (n - 1) // 2

    agreements = []
    flip_rates = []
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = vecs[i], vecs[j]
            # Agreement: same orientation (including both absent/undirected)
            agree = np.mean(vi == vj)
            agreements.append(agree)
            # Flip: one directs i->j, the other directs j->i (sign flip)
            flips = np.mean((vi != 0) & (vj != 0) & (vi != vj))
            flip_rates.append(flips)

    return {
        'mean_agreement': float(np.mean(agreements)),
        'std_agreement': float(np.std(agreements)),
        'mean_flip_rate': float(np.mean(flip_rates)),
        'std_flip_rate': float(np.std(flip_rates)),
        'agreements': agreements,
        'flip_rates': flip_rates,
    }


def make_figure(all_results: list, stab_single: dict, stab_vote: dict,
                stab_dash: dict, out_name: str):
    """Three-panel figure: accuracy, decisiveness, cross-seed stability."""
    load_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods_keys = ['metrics_single', 'metrics_vote', 'metrics_dash']
    labels = ['Single-run\nPC(0.05)', 'Bootstrap\nmajority vote', 'DASH-style\nensemble']
    colors = ['#4ECDC4', '#FFB347', '#FF6B6B']

    # ── Panel A: Accuracy ──
    ax = axes[0]
    acc_data = []
    for mk in methods_keys:
        vals = [r[mk]['accuracy'] for r in all_results
                if not np.isnan(r[mk]['accuracy'])]
        acc_data.append(vals)

    bp = ax.boxplot(acc_data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(acc_data):
        if vals:
            ax.text(i + 1, 1.08,
                    f'$\\mu$={np.mean(vals):.3f}\n$\\sigma$={np.std(vals):.3f}',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_ylabel('Orientation accuracy', fontsize=10)
    ax.set_title('(A) Accuracy of directed edges', fontsize=11)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 1.25)

    # ── Panel B: Decisiveness ──
    ax = axes[1]
    dec_data = []
    for mk in methods_keys:
        vals = [r[mk]['decisiveness'] for r in all_results]
        dec_data.append(vals)

    bp = ax.boxplot(dec_data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(dec_data):
        ax.text(i + 1, max(vals) + 0.05,
                f'$\\mu$={np.mean(vals):.3f}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_ylabel('Decisiveness\n(frac. true edges directed)', fontsize=10)
    ax.set_title('(B) Decisiveness', fontsize=11)
    ax.set_ylim(0, 0.8)

    # ── Panel C: Cross-seed orientation stability ──
    ax = axes[2]
    stab_data = [stab_single['agreements'], stab_vote['agreements'], stab_dash['agreements']]
    bp = ax.boxplot(stab_data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(stab_data):
        ax.text(i + 1, max(vals) + 0.005,
                f'$\\mu$={np.mean(vals):.4f}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_ylabel('Cross-seed agreement', fontsize=10)
    ax.set_title('(C) Orientation stability across datasets', fontsize=11)

    fig.suptitle(
        'Cross-Domain Transfer: DASH-Style Ensemble for Causal Discovery\n'
        '(Asia DAG, N=1000, 100 seeds, 50 bootstrap resamples)',
        fontsize=12, y=1.03
    )
    fig.tight_layout()
    save_figure(fig, out_name)


def main():
    set_all_seeds(42)

    N_SEEDS = 100
    N_SAMPLES = 1000
    N_BOOT = 50
    BASE_SEEDS = list(range(10, 10 + N_SEEDS))

    print("=" * 70)
    print("Ensemble Causal Discovery — Cross-Domain Transfer Experiment")
    print("Asia DAG, 8 nodes, linear Gaussian")
    print(f"N={N_SAMPLES}, {N_SEEDS} seeds, {N_BOOT} bootstrap resamples")
    print("Methods: Single-run PC, Bootstrap majority vote, DASH ensemble")
    print("=" * 70)

    all_results = []
    for idx, seed in enumerate(BASE_SEEDS):
        res = run_one_seed(seed, n_samples=N_SAMPLES, n_boot=N_BOOT)
        all_results.append(res)
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{N_SEEDS} seeds complete")

    # ── Cross-seed stability (the key metric for orbit averaging) ──
    print("\n  Computing cross-seed orientation stability...")
    stab_single = compute_cross_seed_stability(all_results, 'vec_single')
    stab_vote = compute_cross_seed_stability(all_results, 'vec_vote')
    stab_dash = compute_cross_seed_stability(all_results, 'vec_dash')

    # ── Extract per-seed accuracy/decisiveness ──
    methods = ['metrics_single', 'metrics_vote', 'metrics_dash']
    method_names = ['Single-run', 'Bootstrap-vote', 'DASH-ensemble']

    acc = {}
    dec = {}
    for m in methods:
        acc[m] = np.array([r[m]['accuracy'] for r in all_results])
        dec[m] = np.array([r[m]['decisiveness'] for r in all_results])

    acc_clean = {}
    for m in methods:
        mask = ~np.isnan(acc[m])
        acc_clean[m] = acc[m][mask]

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n  Orientation accuracy (correct / committed-directed, on true-DAG edges):")
    for m, label in zip(methods, method_names):
        v = acc_clean[m]
        print(f"    {label:20s}: mean={np.mean(v):.4f}  std={np.std(v):.4f}  "
              f"({len(v)}/{N_SEEDS} seeds with directed edges)")

    print("\n  Decisiveness (fraction of true edges directed):")
    for m, label in zip(methods, method_names):
        v = dec[m]
        print(f"    {label:20s}: mean={np.mean(v):.4f}  std={np.std(v):.4f}")

    print("\n  Cross-seed orientation stability (THE KEY METRIC):")
    for stab, label in zip([stab_single, stab_vote, stab_dash], method_names):
        print(f"    {label:20s}: agreement={stab['mean_agreement']:.4f} "
              f"(std={stab['std_agreement']:.4f})  "
              f"flip_rate={stab['mean_flip_rate']:.4f} "
              f"(std={stab['std_flip_rate']:.4f})")

    # ── Pairwise Wilcoxon on cross-seed agreement ──
    print("\n  Pairwise Wilcoxon signed-rank tests (cross-seed agreement):")

    agree_s = np.array(stab_single['agreements'])
    agree_v = np.array(stab_vote['agreements'])
    agree_d = np.array(stab_dash['agreements'])

    def safe_wilcoxon(a, b, label):
        diff = a - b
        nonzero = diff != 0
        n_nz = np.sum(nonzero)
        if n_nz < 10:
            print(f"    {label}: too few non-tied pairs ({n_nz})")
            return float('nan'), float('nan')
        try:
            stat, p = wilcoxon(diff[nonzero], alternative='greater')
            print(f"    {label}: stat={stat:.1f}, p={p:.4e} (N={n_nz} non-tied)")
            return float(stat), float(p)
        except ValueError as e:
            print(f"    {label}: {e}")
            return float('nan'), float('nan')

    stat_vs, p_vs = safe_wilcoxon(agree_v, agree_s, 'Bootstrap-vote > Single-run')
    stat_ds, p_ds = safe_wilcoxon(agree_d, agree_s, 'DASH-ensemble > Single-run')
    stat_dv, p_dv = safe_wilcoxon(agree_d, agree_v, 'DASH-ensemble > Bootstrap-vote')

    # ── Also test flip rates (lower = more stable) ──
    print("\n  Pairwise Wilcoxon on flip rate (lower = better):")
    flip_s = np.array(stab_single['flip_rates'])
    flip_v = np.array(stab_vote['flip_rates'])
    flip_d = np.array(stab_dash['flip_rates'])

    stat_fvs, p_fvs = safe_wilcoxon(flip_s, flip_v, 'Single > Vote (Vote lower)')
    stat_fds, p_fds = safe_wilcoxon(flip_s, flip_d, 'Single > DASH (DASH lower)')
    stat_fdv, p_fdv = safe_wilcoxon(flip_v, flip_d, 'Vote > DASH (DASH lower)')

    # ── Framework prediction ──
    print("\n" + "=" * 70)
    print("FRAMEWORK PREDICTION ANALYSIS")
    print("=" * 70)

    mean_agree_s = stab_single['mean_agreement']
    mean_agree_v = stab_vote['mean_agreement']
    mean_agree_d = stab_dash['mean_agreement']
    mean_flip_s = stab_single['mean_flip_rate']
    mean_flip_v = stab_vote['mean_flip_rate']
    mean_flip_d = stab_dash['mean_flip_rate']

    mean_acc_s = float(np.mean(acc_clean['metrics_single']))
    mean_acc_v = float(np.mean(acc_clean['metrics_vote']))
    mean_acc_d = float(np.mean(acc_clean['metrics_dash']))
    mean_dec_s = float(np.mean(dec['metrics_single']))
    mean_dec_v = float(np.mean(dec['metrics_vote']))
    mean_dec_d = float(np.mean(dec['metrics_dash']))

    print(f"\n  Method            Accuracy  Decisive  CrossSeedAgree  FlipRate")
    print(f"  {'Single-run':20s} {mean_acc_s:.4f}    {mean_dec_s:.4f}    "
          f"{mean_agree_s:.4f}          {mean_flip_s:.4f}")
    print(f"  {'Bootstrap-vote':20s} {mean_acc_v:.4f}    {mean_dec_v:.4f}    "
          f"{mean_agree_v:.4f}          {mean_flip_v:.4f}")
    print(f"  {'DASH-ensemble':20s} {mean_acc_d:.4f}    {mean_dec_d:.4f}    "
          f"{mean_agree_d:.4f}          {mean_flip_d:.4f}")

    # Key predictions:
    # 1. Ensemble > single on cross-seed stability (agreement)
    pred1_vote = mean_agree_v > mean_agree_s
    pred1_dash = mean_agree_d > mean_agree_s
    # 2. Ensemble has lower flip rate
    pred2_vote = mean_flip_v < mean_flip_s
    pred2_dash = mean_flip_d < mean_flip_s
    # 3. DASH trades decisiveness for stability (Pareto tradeoff)
    pred3 = mean_dec_d < mean_dec_v and mean_agree_d > mean_agree_v

    print(f"\n  Prediction 1: Ensemble > Single on cross-seed stability")
    print(f"    Bootstrap-vote higher agreement than Single: {pred1_vote} "
          f"({mean_agree_v:.4f} vs {mean_agree_s:.4f})")
    print(f"    DASH higher agreement than Single: {pred1_dash} "
          f"({mean_agree_d:.4f} vs {mean_agree_s:.4f})")

    print(f"\n  Prediction 2: Ensemble has lower orientation flip rate")
    print(f"    Bootstrap-vote lower flip rate: {pred2_vote} "
          f"({mean_flip_v:.4f} vs {mean_flip_s:.4f})")
    print(f"    DASH lower flip rate: {pred2_dash} "
          f"({mean_flip_d:.4f} vs {mean_flip_s:.4f})")

    print(f"\n  Prediction 3: DASH is Pareto-optimal (less decisive, more stable)")
    print(f"    DASH less decisive than Vote: {mean_dec_d < mean_dec_v} "
          f"({mean_dec_d:.4f} vs {mean_dec_v:.4f})")
    print(f"    DASH more stable than Vote: {mean_agree_d > mean_agree_v} "
          f"({mean_agree_d:.4f} vs {mean_agree_v:.4f})")

    ensemble_improves_stability = pred1_vote or pred1_dash
    ensemble_reduces_flips = pred2_vote or pred2_dash

    print(f"\n  CROSS-DOMAIN TRANSFER: Ensemble improves stability: "
          f"{'CONFIRMED' if ensemble_improves_stability else 'NOT CONFIRMED'}")
    print(f"  CROSS-DOMAIN TRANSFER: Ensemble reduces flip rate: "
          f"{'CONFIRMED' if ensemble_reduces_flips else 'NOT CONFIRMED'}")
    if pred3:
        print(f"  PARETO OPTIMALITY: DASH trades decisiveness for stability: CONFIRMED")

    # ── Figure ──
    make_figure(all_results, stab_single, stab_vote, stab_dash,
                'ensemble_causal_transfer')

    # ── Save results ──
    output = {
        'experiment': 'ensemble_causal_discovery_transfer',
        'dag': 'Asia (8 nodes, linear Gaussian)',
        'n_samples': N_SAMPLES,
        'n_seeds': N_SEEDS,
        'n_bootstrap': N_BOOT,
        'dash_threshold': 0.7,
        'methods': ['Single-run PC(0.05)', 'Bootstrap majority vote',
                    'DASH-style ensemble'],
        'true_edges': TRUE_EDGES,
        'summary': {
            'single_run': {
                'accuracy_mean': mean_acc_s,
                'accuracy_std': float(np.std(acc_clean['metrics_single'])),
                'decisiveness_mean': mean_dec_s,
                'decisiveness_std': float(np.std(dec['metrics_single'])),
                'cross_seed_agreement': mean_agree_s,
                'flip_rate': mean_flip_s,
            },
            'bootstrap_vote': {
                'accuracy_mean': mean_acc_v,
                'accuracy_std': float(np.std(acc_clean['metrics_vote'])),
                'decisiveness_mean': mean_dec_v,
                'decisiveness_std': float(np.std(dec['metrics_vote'])),
                'cross_seed_agreement': mean_agree_v,
                'flip_rate': mean_flip_v,
            },
            'dash_ensemble': {
                'accuracy_mean': mean_acc_d,
                'accuracy_std': float(np.std(acc_clean['metrics_dash'])),
                'decisiveness_mean': mean_dec_d,
                'decisiveness_std': float(np.std(dec['metrics_dash'])),
                'cross_seed_agreement': mean_agree_d,
                'flip_rate': mean_flip_d,
            },
        },
        'pairwise_tests_stability': {
            'bootstrap_vote_gt_single': {
                'test': 'Wilcoxon signed-rank (one-sided, cross-seed agreement)',
                'statistic': stat_vs if not np.isnan(stat_vs) else None,
                'p_value': p_vs if not np.isnan(p_vs) else None,
            },
            'dash_gt_single': {
                'test': 'Wilcoxon signed-rank (one-sided, cross-seed agreement)',
                'statistic': stat_ds if not np.isnan(stat_ds) else None,
                'p_value': p_ds if not np.isnan(p_ds) else None,
            },
            'dash_gt_vote': {
                'test': 'Wilcoxon signed-rank (one-sided, cross-seed agreement)',
                'statistic': stat_dv if not np.isnan(stat_dv) else None,
                'p_value': p_dv if not np.isnan(p_dv) else None,
            },
        },
        'pairwise_tests_flip_rate': {
            'vote_lower_than_single': {
                'statistic': stat_fvs if not np.isnan(stat_fvs) else None,
                'p_value': p_fvs if not np.isnan(p_fvs) else None,
            },
            'dash_lower_than_single': {
                'statistic': stat_fds if not np.isnan(stat_fds) else None,
                'p_value': p_fds if not np.isnan(p_fds) else None,
            },
            'dash_lower_than_vote': {
                'statistic': stat_fdv if not np.isnan(stat_fdv) else None,
                'p_value': p_fdv if not np.isnan(p_fdv) else None,
            },
        },
        'prediction': {
            'ensemble_improves_stability': bool(ensemble_improves_stability),
            'ensemble_reduces_flips': bool(ensemble_reduces_flips),
            'dash_pareto_optimal': bool(pred3),
        },
        'per_seed_results': [
            {
                'seed': r['seed'],
                'single_acc': r['metrics_single']['accuracy'],
                'single_dec': r['metrics_single']['decisiveness'],
                'vote_acc': r['metrics_vote']['accuracy'],
                'vote_dec': r['metrics_vote']['decisiveness'],
                'dash_acc': r['metrics_dash']['accuracy'],
                'dash_dec': r['metrics_dash']['decisiveness'],
            }
            for r in all_results
        ],
        'interpretation': (
            f"Cross-domain transfer test: DASH-style ensemble averaging applied to causal "
            f"discovery (Asia DAG, N={N_SAMPLES}, {N_SEEDS} seeds). "
            f"Cross-seed orientation stability — the key metric for orbit averaging — shows: "
            f"Single-run agreement={mean_agree_s:.4f}, "
            f"Bootstrap-vote={mean_agree_v:.4f}, DASH={mean_agree_d:.4f}. "
            f"Flip rates: Single={mean_flip_s:.4f}, Vote={mean_flip_v:.4f}, "
            f"DASH={mean_flip_d:.4f}. "
            f"Accuracy (correct/directed): Single={mean_acc_s:.3f}, "
            f"Vote={mean_acc_v:.3f}, DASH={mean_acc_d:.3f}. "
            f"Decisiveness: Single={mean_dec_s:.3f}, Vote={mean_dec_v:.3f}, "
            f"DASH={mean_dec_d:.3f}. "
            f"Ensemble improves stability: "
            f"{'CONFIRMED' if ensemble_improves_stability else 'NOT CONFIRMED'}. "
            f"Ensemble reduces flips: "
            f"{'CONFIRMED' if ensemble_reduces_flips else 'NOT CONFIRMED'}. "
            f"DASH Pareto-optimal (less decisive, more stable): "
            f"{'CONFIRMED' if pred3 else 'NOT CONFIRMED'}."
        ),
    }
    save_results(output, 'ensemble_causal_transfer')

    print("\nDone.")


if __name__ == '__main__':
    main()
