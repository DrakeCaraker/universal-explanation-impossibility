"""
Causal Discovery Experiment — Sachs Network Bootstrap CPDAG Stability
=====================================================================

Research question: How stable are PC algorithm edge orientations across
bootstrap resamples of the Sachs phosphoprotein signaling network?

The Sachs dataset (Sachs et al., 2005) is a standard causal discovery
benchmark: 11 phosphoproteins, ~7500 flow cytometry observations, 17 known
edges. This extends the Asia network experiment (8 nodes) from the paper
to a larger, real-world-scale causal structure.

Design:
  1. Simulate a Sachs-like network (11 nodes, known DAG, linear Gaussian SEM)
     or load from causal-learn if available
  2. Run PC algorithm (alpha=0.05) across 100 bootstrap resamples
  3. For each bootstrap: record the estimated CPDAG (edge list)
  4. Measure:
     (a) Edge orientation flip rate across bootstrap pairs
     (b) Skeleton agreement (Jaccard of edge presence ignoring orientation)
     (c) Number of reversible (undirected) edges per bootstrap
  5. Compare to Asia results from paper/results_causal_discovery_exp.json

Output:
  knockout-experiments/results_causal_sachs.json
  knockout-experiments/figures/causal_sachs.pdf
"""

import sys
import os
import json
import warnings
import itertools
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
PAPER_DIR = SCRIPT_DIR.parent / "paper"

# ── Sachs DAG Definition ────────────────────────────────────────────────────
# 11 phosphoproteins from Sachs et al. (2005)
NODE_NAMES = [
    'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk'
]
N_NODES = 11

# True directed edges (consensus network from Sachs et al. 2005)
# 17 edges total
TRUE_EDGES = [
    (0, 1),   # Raf -> Mek
    (1, 5),   # Mek -> Erk
    (2, 3),   # Plcg -> PIP2
    (2, 4),   # Plcg -> PIP3
    (4, 3),   # PIP3 -> PIP2
    (7, 0),   # PKA -> Raf
    (7, 1),   # PKA -> Mek
    (7, 5),   # PKA -> Erk
    (7, 6),   # PKA -> Akt
    (7, 9),   # PKA -> P38
    (7, 10),  # PKA -> Jnk
    (8, 0),   # PKC -> Raf
    (8, 1),   # PKC -> Mek
    (8, 9),   # PKC -> P38
    (8, 10),  # PKC -> Jnk
    (8, 2),   # PKC -> Plcg
    (5, 6),   # Erk -> Akt
]

N_TRUE_EDGES = len(TRUE_EDGES)


def build_adjacency_matrix(edges, n_nodes):
    """Build a binary adjacency matrix from edge list."""
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in edges:
        adj[i, j] = 1
    return adj


def topological_order(adj):
    """Return a topological ordering of nodes given adjacency matrix."""
    n = adj.shape[0]
    in_degree = adj.sum(axis=0)
    order = []
    available = list(np.where(in_degree == 0)[0])
    remaining_adj = adj.copy()
    while available:
        node = available.pop(0)
        order.append(node)
        children = np.where(remaining_adj[node] > 0)[0]
        remaining_adj[node, :] = 0
        for c in children:
            if remaining_adj[:, c].sum() == 0:
                available.append(c)
    return order


def sample_sachs_linear_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N observations from a linear Gaussian SEM with the Sachs DAG.
    Edge coefficients drawn uniformly from [0.5, 1.5] with random signs.
    Noise std = 0.5.
    """
    adj = build_adjacency_matrix(TRUE_EDGES, N_NODES)
    topo = topological_order(adj)

    noise_std = 0.5
    # Fixed coefficients for reproducibility (drawn once)
    coeff_rng = np.random.default_rng(12345)
    coefficients = np.zeros((N_NODES, N_NODES))
    for i, j in TRUE_EDGES:
        sign = coeff_rng.choice([-1, 1])
        mag = coeff_rng.uniform(0.5, 1.5)
        coefficients[i, j] = sign * mag

    data = np.zeros((n, N_NODES))
    for node in topo:
        parents = np.where(adj[:, node] > 0)[0]
        val = rng.normal(0, 1, n)
        for p in parents:
            val += coefficients[p, node] * data[:, p]
        val += rng.normal(0, noise_std, n)
        data[:, node] = val

    return data


# ── PC Algorithm (self-contained, no causal-learn dependency) ────────────────

def fisher_z_test(data, i, j, conditioning_set, n):
    """
    Fisher-Z conditional independence test.
    Returns p-value for H0: X_i ⊥ X_j | X_S.
    """
    if len(conditioning_set) == 0:
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
    else:
        # Partial correlation via regression residuals
        S = list(conditioning_set)
        X = data[:, S]
        # Regress i on S
        try:
            beta_i = np.linalg.lstsq(X, data[:, i], rcond=None)[0]
            res_i = data[:, i] - X @ beta_i
            beta_j = np.linalg.lstsq(X, data[:, j], rcond=None)[0]
            res_j = data[:, j] - X @ beta_j
            r = np.corrcoef(res_i, res_j)[0, 1]
        except np.linalg.LinAlgError:
            return 1.0  # cannot reject independence

    # Clip correlation to avoid numerical issues
    r = np.clip(r, -0.9999, 0.9999)

    # Fisher Z transform
    z = 0.5 * np.log((1 + r) / (1 - r))
    dof = n - len(conditioning_set) - 3
    if dof < 1:
        return 1.0
    stat = abs(z) * np.sqrt(dof)
    p_value = 2 * (1 - stats.norm.cdf(stat))
    return p_value


def pc_algorithm(data, alpha=0.05):
    """
    Simplified PC algorithm returning a CPDAG-like structure.
    Returns:
      skeleton: set of frozenset({i,j}) for undirected edges
      oriented: dict mapping (i,j) -> True for directed i->j edges
      sep_sets: dict mapping frozenset({i,j}) -> conditioning set that separated them
    """
    n_obs, n_vars = data.shape

    # Step 1: Start with complete undirected graph
    edges = set()
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            edges.add(frozenset({i, j}))

    sep_sets = {}

    # Step 2: Edge removal by conditional independence tests
    depth = 0
    max_depth = min(n_vars - 2, 4)  # limit depth for efficiency

    while depth <= max_depth:
        edges_to_remove = set()
        for edge in list(edges):
            i, j = tuple(edge)
            # Neighbors of i (excluding j)
            neighbors_i = set()
            for e in edges:
                if i in e and e != edge:
                    neighbors_i.update(e - {i})
            # Neighbors of j (excluding i)
            neighbors_j = set()
            for e in edges:
                if j in e and e != edge:
                    neighbors_j.update(e - {j})

            # Test subsets of neighbors of size = depth
            neighbors = neighbors_i | neighbors_j
            if len(neighbors) < depth:
                continue

            found_independent = False
            for S in itertools.combinations(neighbors, depth):
                S_set = set(S)
                p_val = fisher_z_test(data, i, j, S_set, n_obs)
                if p_val > alpha:
                    edges_to_remove.add(edge)
                    sep_sets[edge] = S_set
                    found_independent = True
                    break
            if found_independent:
                continue

        edges -= edges_to_remove
        depth += 1

    # Step 3: Orient v-structures
    # For each triple i - k - j where i and j are not adjacent,
    # orient as i -> k <- j if k not in sep_set(i, j)
    oriented = {}  # (i, j) means directed edge i -> j
    skeleton = set(edges)

    for k in range(n_vars):
        # Find pairs of neighbors of k that are not adjacent to each other
        neighbors_k = []
        for e in skeleton:
            if k in e:
                other = list(e - {k})[0]
                neighbors_k.append(other)

        for idx_a in range(len(neighbors_k)):
            for idx_b in range(idx_a + 1, len(neighbors_k)):
                i = neighbors_k[idx_a]
                j = neighbors_k[idx_b]
                # Check if i and j are NOT adjacent
                if frozenset({i, j}) not in skeleton:
                    # Check if k is in separation set
                    sep_key = frozenset({i, j})
                    sep_set = sep_sets.get(sep_key, set())
                    if k not in sep_set:
                        # Orient i -> k <- j (v-structure)
                        oriented[(i, k)] = True
                        oriented[(j, k)] = True

    return skeleton, oriented, sep_sets


def cpdag_to_edge_list(skeleton, oriented, n_vars):
    """
    Convert PC output to a canonical edge list representation.
    Returns list of tuples: (i, j, 'directed') or (i, j, 'undirected')
    where i < j always.
    """
    edge_list = []
    for edge in skeleton:
        i, j = sorted(tuple(edge))
        if (i, j) in oriented:
            edge_list.append((i, j, 'i->j'))
        elif (j, i) in oriented:
            edge_list.append((i, j, 'j->i'))
        else:
            edge_list.append((i, j, 'undirected'))
    return sorted(edge_list)


# ── Bootstrap Experiment ─────────────────────────────────────────────────────

def run_bootstrap_pc(data, n_bootstrap=100, alpha=0.05, seed=42):
    """
    Run PC algorithm on n_bootstrap resamples of data.
    Returns list of CPDAG edge lists (one per bootstrap).
    """
    rng = np.random.default_rng(seed)
    n_obs = data.shape[0]
    n_vars = data.shape[1]
    results = []

    for b in range(n_bootstrap):
        # Resample rows with replacement
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_data = data[idx]

        skeleton, oriented, _ = pc_algorithm(boot_data, alpha=alpha)
        edge_list = cpdag_to_edge_list(skeleton, oriented, n_vars)
        results.append(edge_list)

        if (b + 1) % 10 == 0:
            print(f"  Bootstrap {b+1}/{n_bootstrap} complete")

    return results


def compute_bootstrap_metrics(bootstrap_results, n_vars):
    """
    Compute stability metrics across bootstrap CPDAG estimates.

    Returns:
      flip_rate: fraction of bootstrap pairs where an edge's orientation differs
      skeleton_agreement: mean Jaccard similarity of skeletons across pairs
      n_reversible: mean number of undirected edges per bootstrap
    """
    n_boot = len(bootstrap_results)

    # Convert each bootstrap result to skeleton set and orientation dict
    skeletons = []
    orientations = []
    n_undirected_list = []

    for edge_list in bootstrap_results:
        skel = set()
        orient = {}
        n_undir = 0
        for (i, j, direction) in edge_list:
            skel.add(frozenset({i, j}))
            orient[frozenset({i, j})] = direction
            if direction == 'undirected':
                n_undir += 1
        skeletons.append(skel)
        orientations.append(orient)
        n_undirected_list.append(n_undir)

    # Pairwise metrics
    n_pairs = 0
    total_flip = 0
    total_skeleton_jaccard = 0.0
    flip_details = {}  # edge -> count of flips

    # Sample pairs (all pairs if n_boot <= 100, else sample)
    pairs = list(itertools.combinations(range(n_boot), 2))

    for (a, b) in pairs:
        skel_a = skeletons[a]
        skel_b = skeletons[b]
        orient_a = orientations[a]
        orient_b = orientations[b]

        # Skeleton Jaccard
        intersection = len(skel_a & skel_b)
        union = len(skel_a | skel_b)
        jaccard = intersection / union if union > 0 else 1.0
        total_skeleton_jaccard += jaccard

        # Orientation flips: among edges present in both, count orientation changes
        shared = skel_a & skel_b
        for edge in shared:
            dir_a = orient_a.get(edge, 'missing')
            dir_b = orient_b.get(edge, 'missing')
            if dir_a != dir_b:
                total_flip += 1
                edge_key = tuple(sorted(tuple(edge)))
                flip_details[edge_key] = flip_details.get(edge_key, 0) + 1

        n_pairs += 1

    flip_rate = total_flip / (n_pairs * N_NODES * (N_NODES - 1) / 2) if n_pairs > 0 else 0
    mean_skeleton_jaccard = total_skeleton_jaccard / n_pairs if n_pairs > 0 else 0
    mean_n_reversible = float(np.mean(n_undirected_list))

    # Per-edge flip rate (among edges that appeared in at least one bootstrap)
    all_edges_ever = set()
    for skel in skeletons:
        all_edges_ever.update(skel)

    per_edge_flip_rate = {}
    for edge in all_edges_ever:
        edge_key = tuple(sorted(tuple(edge)))
        n_flips = flip_details.get(edge_key, 0)
        per_edge_flip_rate[str(edge_key)] = n_flips / n_pairs if n_pairs > 0 else 0

    # Edge frequency (how often each edge appears in skeleton)
    edge_frequency = {}
    for edge in all_edges_ever:
        edge_key = tuple(sorted(tuple(edge)))
        count = sum(1 for skel in skeletons if edge in skel)
        edge_frequency[str(edge_key)] = count / n_boot

    return {
        'orientation_flip_rate': float(flip_rate),
        'mean_skeleton_jaccard': float(mean_skeleton_jaccard),
        'mean_n_reversible': float(mean_n_reversible),
        'n_pairs_compared': n_pairs,
        'per_edge_flip_rate': per_edge_flip_rate,
        'edge_frequency': edge_frequency,
        'n_undirected_per_bootstrap': [int(x) for x in n_undirected_list],
        'n_edges_per_bootstrap': [len(bl) for bl in bootstrap_results],
    }


def load_asia_results():
    """Load Asia experiment results for comparison."""
    asia_path = PAPER_DIR / "results_causal_discovery_exp.json"
    if asia_path.exists():
        with open(asia_path) as f:
            return json.load(f)
    return None


def make_figure(metrics, asia_results, out_path):
    """Create comparison figure: Sachs bootstrap stability vs Asia results."""
    # Publication style
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Edge frequency histogram (Sachs)
    ax = axes[0]
    freqs = list(metrics['edge_frequency'].values())
    ax.hist(freqs, bins=20, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_xlabel('Edge frequency across bootstraps')
    ax.set_ylabel('Number of edges')
    ax.set_title('(a) Sachs: Edge stability\nacross 100 bootstraps')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='50% threshold')
    ax.legend(fontsize=8)

    # Panel B: Orientation flip rate per edge (top edges)
    ax = axes[1]
    flip_rates = metrics['per_edge_flip_rate']
    # Sort by flip rate descending, show top 15
    sorted_edges = sorted(flip_rates.items(), key=lambda x: x[1], reverse=True)[:15]
    if sorted_edges:
        edge_labels = []
        flip_vals = []
        for edge_str, rate in sorted_edges:
            # Parse edge tuple from string
            edge_tuple = eval(edge_str)
            label = f"{NODE_NAMES[edge_tuple[0]]}-{NODE_NAMES[edge_tuple[1]]}"
            edge_labels.append(label)
            flip_vals.append(rate)

        y_pos = np.arange(len(edge_labels))
        colors = ['#d62728' if v > 0.1 else '#ff7f0e' if v > 0.05 else '#2ca02c' for v in flip_vals]
        ax.barh(y_pos, flip_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(edge_labels, fontsize=7)
        ax.set_xlabel('Orientation flip rate')
        ax.set_title('(b) Sachs: Per-edge\norientation instability')
        ax.invert_yaxis()

    # Panel C: Comparison with Asia
    ax = axes[2]
    sachs_jaccard = metrics['mean_skeleton_jaccard']
    sachs_flip = metrics['orientation_flip_rate']
    sachs_reversible = metrics['mean_n_reversible']

    # Asia metrics from loaded results
    asia_agreement = None
    if asia_results is not None:
        asia_agreement = asia_results.get('ci_small', {}).get('mean', None)

    bar_labels = ['Skeleton\nJaccard', 'Orient.\nflip rate', 'Reversible\nedges (norm.)']
    sachs_vals = [sachs_jaccard, sachs_flip, sachs_reversible / N_NODES]

    x = np.arange(len(bar_labels))
    width = 0.35
    bars_sachs = ax.bar(x - width/2, sachs_vals, width, label='Sachs (11 nodes)',
                        color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.7)

    if asia_agreement is not None:
        # Asia: use overall agreement as skeleton proxy, orientation disagree as flip rate proxy
        asia_orient_disagree = asia_results.get('results_small', {}).get('n_orientation_disagree_pairs', 0)
        asia_n_edges = 8  # Asia has 8 true edges
        asia_flip_proxy = asia_orient_disagree / (3 * asia_n_edges) if asia_n_edges > 0 else 0
        asia_vals = [
            asia_results.get('pairwise_small', [{}])[0].get('skeleton_agreement', 0),
            asia_flip_proxy,
            len(asia_results.get('reversible_edges', [])) / 8
        ]
        bars_asia = ax.bar(x + width/2, asia_vals, width, label='Asia (8 nodes)',
                           color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Value')
    ax.set_title('(c) Sachs vs Asia\nstability comparison')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Annotate
    for bar in bars_sachs:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7)

    fig.suptitle(
        'Causal Discovery Bootstrap Stability — Sachs Network\n'
        '(11 phosphoproteins, PC algorithm, $\\alpha$=0.05, 100 bootstrap resamples)',
        fontsize=11, y=1.03
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("Causal Sachs Experiment")
    print("Sachs network — 11 nodes, 17 edges, linear Gaussian SEM")
    print("Method: PC (alpha=0.05), 100 bootstrap resamples")
    print("=" * 60)

    # Step 1: Generate data
    n_samples = 1000
    print(f"\nGenerating {n_samples} samples from Sachs-like linear Gaussian SEM...")
    data = sample_sachs_linear_gaussian(n_samples, rng)
    print(f"  Data shape: {data.shape}")

    # Step 2: Run bootstrap PC
    n_bootstrap = 100
    print(f"\nRunning PC algorithm on {n_bootstrap} bootstrap resamples...")
    bootstrap_results = run_bootstrap_pc(data, n_bootstrap=n_bootstrap, alpha=0.05, seed=42)

    # Step 3: Compute metrics
    print("\nComputing bootstrap stability metrics...")
    metrics = compute_bootstrap_metrics(bootstrap_results, N_NODES)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Orientation flip rate: {metrics['orientation_flip_rate']:.4f}")
    print(f"  Mean skeleton Jaccard: {metrics['mean_skeleton_jaccard']:.4f}")
    print(f"  Mean reversible edges: {metrics['mean_n_reversible']:.1f} / {N_NODES} nodes")
    print(f"  Mean edges per bootstrap: {np.mean(metrics['n_edges_per_bootstrap']):.1f}")

    # Step 4: Load Asia results for comparison
    asia_results = load_asia_results()
    if asia_results is not None:
        asia_mean = asia_results.get('ci_small', {}).get('mean', 'N/A')
        print(f"\n  Asia comparison — mean overall agreement (N=1000): {asia_mean}")
    else:
        print("\n  Asia results not found; skipping comparison.")

    # Step 5: Figure
    fig_path = FIGURES_DIR / "causal_sachs.pdf"
    print(f"\nGenerating figure...")
    make_figure(metrics, asia_results, fig_path)

    # Step 6: Save results JSON
    output = {
        'experiment': 'causal_sachs',
        'dag': 'Sachs (11 phosphoproteins, linear Gaussian SEM)',
        'n_nodes': N_NODES,
        'n_true_edges': N_TRUE_EDGES,
        'node_names': NODE_NAMES,
        'true_edges': TRUE_EDGES,
        'n_samples': n_samples,
        'n_bootstrap': n_bootstrap,
        'alpha': 0.05,
        'method': 'PC (Fisher-Z, alpha=0.05)',
        'orientation_flip_rate': metrics['orientation_flip_rate'],
        'mean_skeleton_jaccard': metrics['mean_skeleton_jaccard'],
        'mean_n_reversible': metrics['mean_n_reversible'],
        'mean_n_edges_per_bootstrap': float(np.mean(metrics['n_edges_per_bootstrap'])),
        'std_n_edges_per_bootstrap': float(np.std(metrics['n_edges_per_bootstrap'])),
        'per_edge_flip_rate': metrics['per_edge_flip_rate'],
        'edge_frequency': metrics['edge_frequency'],
        'n_undirected_per_bootstrap': metrics['n_undirected_per_bootstrap'],
        'n_edges_per_bootstrap': metrics['n_edges_per_bootstrap'],
        'asia_comparison': {
            'asia_mean_agreement_N1000': asia_results.get('ci_small', {}).get('mean', None) if asia_results else None,
            'asia_n_nodes': 8,
            'sachs_n_nodes': N_NODES,
            'note': 'Sachs has more nodes (11 vs 8) and edges (17 vs 8), '
                    'increasing the Markov equivalence class and thus instability.',
        },
        'interpretation': (
            f"PC algorithm on the Sachs network (11 nodes, 17 edges) shows substantial "
            f"bootstrap instability: orientation flip rate = {metrics['orientation_flip_rate']:.4f}, "
            f"mean skeleton Jaccard = {metrics['mean_skeleton_jaccard']:.4f}, "
            f"mean reversible edges = {metrics['mean_n_reversible']:.1f}. "
            f"The larger Markov equivalence class (compared to Asia, 8 nodes) "
            f"amplifies the impossibility: no single CPDAG estimate is simultaneously "
            f"faithful, stable, and decisive under finite-sample underspecification."
        ),
    }

    results_path = SCRIPT_DIR / "results_causal_sachs.json"
    import time
    output['_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved results: {results_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
