"""
Causal Discovery Experiment — Algorithm Orientation Agreement
=============================================================

Research question: Do different causal discovery algorithms agree on
edge orientations when learning from finite data? Under underspecification
(small N), algorithms disagree on orientations within the Markov equivalence
class. At large N, they converge to the same CPDAG.

This demonstrates the impossibility theorem's causal discovery instance:
no single finite-sample algorithm is simultaneously faithful, stable,
and decisive when the Markov equivalence class is non-trivial.

DAG: Asia network (8 nodes):
  Asia(A) -> Tub(T), Smoke(S) -> Lung(L), Smoke(S) -> Bronch(B),
  Tub(T) -> Either(E), Lung(L) -> Either(E),
  Either(E) -> Xray(X), Either(E) -> Dysp(D), Bronch(B) -> Dysp(D)

Methods:
  1. PC algorithm (alpha=0.05)
  2. GES (BIC score)
  3. PC algorithm (alpha=0.01) — same constraint-based family, stricter test

Metrics:
  - Pairwise edge orientation agreement rate
  - Number of disagreements on orientation
  - 95% bootstrap CIs
  - Negative control: N=100,000 should show high agreement

Output:
  paper/figures/causal_discovery_exp.pdf
  paper/sections/table_causal_discovery_exp.tex
  paper/results_causal_discovery_exp.json
"""

import sys
import os
import json
import warnings
import itertools
from pathlib import Path
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, '/Users/drake.caraker/Library/Python/3.9/lib/python/site-packages')

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
    PAPER_DIR,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SECTIONS_DIR = PAPER_DIR / "sections"
SECTIONS_DIR.mkdir(exist_ok=True)

# ── Asia DAG Definition ───────────────────────────────────────────────────────
# Nodes: 0=Asia, 1=Tub, 2=Smoke, 3=Lung, 4=Bronch, 5=Either, 6=Xray, 7=Dysp
# Edges (parent -> child):
#   Asia(0)->Tub(1), Smoke(2)->Lung(3), Smoke(2)->Bronch(4),
#   Tub(1)->Either(5), Lung(3)->Either(5),
#   Either(5)->Xray(6), Either(5)->Dysp(7), Bronch(4)->Dysp(7)

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

# V-structures (compelled edges): edges forced by v-structures
# Tub(1)->Either(5)<-Lung(3) is a v-structure (Tub⊥Lung | {}  but not | Either)
# Asia(0)->Tub(1) is compelled because Tub is in v-structure
# Smoke(2)->Lung(3) and Smoke(2)->Bronch(4): Smoke is common cause — reversible
# Either(5)->Xray(6): Xray has only one parent — reversible
# Either(5)->Dysp(7) and Bronch(4)->Dysp(7): v-structure if Bronch⊥Either | {}
V_STRUCTURE_EDGES = {(1, 5), (3, 5), (4, 7), (5, 7)}  # edges near v-structures
REVERSIBLE_EDGES = {(0, 1), (2, 3), (2, 4), (5, 6)}    # within MEC


def sample_asia_linear_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N observations from a linear Gaussian version of the Asia DAG.
    Using continuous (Gaussian) variables so PC/GES can use Fisher-Z / BIC.
    Topological order: Asia, Smoke, Tub, Lung, Bronch, Either, Xray, Dysp
    """
    # Topo order: 0(Asia), 2(Smoke), 1(Tub), 3(Lung), 4(Bronch), 5(Either), 6(Xray), 7(Dysp)
    noise_std = 0.5
    coeff = 0.8  # edge coefficient strength

    asia  = rng.normal(0, 1, n)                                      # 0: Asia (exog)
    smoke = rng.normal(0, 1, n)                                      # 2: Smoke (exog)
    tub   = coeff * asia  + rng.normal(0, noise_std, n)              # 1: Tub <- Asia
    lung  = coeff * smoke + rng.normal(0, noise_std, n)              # 3: Lung <- Smoke
    bronc = coeff * smoke + rng.normal(0, noise_std, n)              # 4: Bronch <- Smoke
    either = coeff * tub  + coeff * lung + rng.normal(0, noise_std, n)  # 5: Either <- Tub, Lung
    xray  = coeff * either + rng.normal(0, noise_std, n)             # 6: Xray <- Either
    dysp  = coeff * either + coeff * bronc + rng.normal(0, noise_std, n)  # 7: Dysp <- Either, Bronch

    # Column order matches NODE_NAMES: Asia, Tub, Smoke, Lung, Bronch, Either, Xray, Dysp
    data = np.column_stack([asia, tub, smoke, lung, bronc, either, xray, dysp])
    return data


def extract_adj(graph_obj) -> np.ndarray:
    """
    Extract the adjacency matrix from a causallearn GeneralGraph object.
    Returns an (n, n) matrix using causallearn's encoding:
      adj[i, j] = -1: tail at j (edge exists from i side)
      adj[i, j] =  1: arrowhead at j
      adj[i, j] =  0: no edge endpoint
    An undirected edge i--j: adj[i,j]=-1, adj[j,i]=-1
    A directed edge i->j:    adj[i,j]=-1, adj[j,i]=1
    """
    return graph_obj.graph.copy()


def adj_to_edge_dict(adj: np.ndarray) -> dict:
    """
    Convert adjacency matrix to dict mapping frozenset({i,j}) -> orientation.
    Orientation values:
      'i->j': directed i to j
      'j->i': directed j to i
      'i--j': undirected
      'i<->j': bidirected
    Only returns pairs where an edge is present.
    """
    n = adj.shape[0]
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            aij = adj[i, j]
            aji = adj[j, i]
            if aij == 0 and aji == 0:
                continue  # no edge
            key = (i, j)  # canonical order
            if aij == -1 and aji == -1:
                edges[key] = 'undirected'
            elif aij == -1 and aji == 1:
                edges[key] = 'i->j'  # i -> j
            elif aij == 1 and aji == -1:
                edges[key] = 'j->i'  # j -> i (i.e., j->i in original)
            else:
                edges[key] = f'other({aij},{aji})'
    return edges


def run_pc(data: np.ndarray, alpha: float, label: str) -> dict:
    """Run PC algorithm and return edge dict."""
    from causallearn.search.ConstraintBased.PC import pc
    result = pc(data, alpha=alpha, indep_test='fisherz', show_progress=False, verbose=False)
    adj = extract_adj(result.G)
    edges = adj_to_edge_dict(adj)
    print(f"  {label}: {len(edges)} edges found")
    return {'label': label, 'edges': edges, 'adj': adj}


def run_ges(data: np.ndarray, label: str) -> dict:
    """Run GES algorithm and return edge dict."""
    from causallearn.search.ScoreBased.GES import ges
    result = ges(data)
    adj = extract_adj(result['G'])
    edges = adj_to_edge_dict(adj)
    print(f"  {label}: {len(edges)} edges found")
    return {'label': label, 'edges': edges, 'adj': adj}


def compute_pairwise_agreement(result_a: dict, result_b: dict) -> dict:
    """
    Compare two estimated graphs. For each edge present in either graph:
      - Classify as: both_same_dir, both_opposite_dir, both_undirected,
        one_directed_one_undirected, only_in_a, only_in_b
    Returns summary statistics.
    """
    edges_a = result_a['edges']
    edges_b = result_b['edges']

    all_keys = set(edges_a.keys()) | set(edges_b.keys())

    both_same = 0
    both_differ = 0
    both_undirected = 0
    mixed_dir = 0
    only_a = 0
    only_b = 0

    edge_details = []

    for key in all_keys:
        in_a = key in edges_a
        in_b = key in edges_b
        if in_a and in_b:
            oa = edges_a[key]
            ob = edges_b[key]
            if oa == ob:
                if oa == 'undirected':
                    both_undirected += 1
                    edge_details.append({'key': key, 'status': 'both_undirected'})
                else:
                    both_same += 1
                    edge_details.append({'key': key, 'status': 'both_same_dir', 'orient': oa})
            else:
                # Both directed but different, or one directed one undirected
                is_a_dir = oa != 'undirected'
                is_b_dir = ob != 'undirected'
                if is_a_dir and is_b_dir:
                    both_differ += 1
                    edge_details.append({'key': key, 'status': 'both_opposite_dir',
                                         'orient_a': oa, 'orient_b': ob})
                else:
                    mixed_dir += 1
                    edge_details.append({'key': key, 'status': 'mixed',
                                         'orient_a': oa, 'orient_b': ob})
        elif in_a:
            only_a += 1
            edge_details.append({'key': key, 'status': 'only_a', 'orient': edges_a[key]})
        else:
            only_b += 1
            edge_details.append({'key': key, 'status': 'only_b', 'orient': edges_b[key]})

    # For directed edges present in both, compute agreement
    # Agreement = edges with same orientation / edges present in both with at least one directed
    n_shared = both_same + both_differ + both_undirected + mixed_dir
    n_shared_directed = both_same + both_differ + mixed_dir  # at least one is directed
    n_only_one = only_a + only_b

    # Skeleton agreement (ignoring orientation)
    n_skeleton_agree = both_same + both_differ + both_undirected + mixed_dir
    n_total_skeleton = n_skeleton_agree + n_only_one
    skeleton_agreement = n_skeleton_agree / n_total_skeleton if n_total_skeleton > 0 else 0.0

    # Orientation agreement (among shared directed edges)
    orientation_agreement = both_same / n_shared_directed if n_shared_directed > 0 else None

    # Overall agreement: shared same direction / all edges in either graph
    total_edges = n_shared + n_only_one
    overall_agreement = both_same / total_edges if total_edges > 0 else 0.0

    return {
        'pair': (result_a['label'], result_b['label']),
        'both_same_dir': both_same,
        'both_opposite_dir': both_differ,
        'both_undirected': both_undirected,
        'mixed_dir_undirected': mixed_dir,
        'only_in_a': only_a,
        'only_in_b': only_b,
        'n_shared': n_shared,
        'skeleton_agreement': skeleton_agreement,
        'orientation_agreement': orientation_agreement,
        'overall_agreement': overall_agreement,
        'edge_details': edge_details,
    }


def run_experiment(n_samples: int, rng: np.random.Generator, label: str) -> dict:
    """Run all three methods on data of size n_samples. Return results."""
    print(f"\n--- {label} (N={n_samples:,}) ---")
    data = sample_asia_linear_gaussian(n_samples, rng)

    results = [
        run_pc(data, alpha=0.05, label=r'PC($\alpha$=0.05)'),
        run_ges(data, label='GES(BIC)'),
        run_pc(data, alpha=0.01, label=r'PC($\alpha$=0.01)'),
    ]

    method_labels = [r['label'] for r in results]
    pairs = list(itertools.combinations(range(len(results)), 2))

    pairwise = []
    overall_agreements = []
    orientation_agreements = []

    for i, j in pairs:
        comp = compute_pairwise_agreement(results[i], results[j])
        pairwise.append(comp)
        overall_agreements.append(comp['overall_agreement'])
        if comp['orientation_agreement'] is not None:
            orientation_agreements.append(comp['orientation_agreement'])
        print(f"  {comp['pair'][0]} vs {comp['pair'][1]}: "
              f"overall_agree={comp['overall_agreement']:.3f}, "
              f"orient_agree={comp['orientation_agreement']}")

    mean_overall = float(np.mean(overall_agreements))
    mean_orient = float(np.mean(orientation_agreements)) if orientation_agreements else None

    # Count disagreements on orientation
    n_orient_disagree = sum(p['both_opposite_dir'] for p in pairwise)
    n_only_one = sum(p['only_in_a'] + p['only_in_b'] for p in pairwise)

    print(f"  Mean overall agreement: {mean_overall:.3f}")
    print(f"  Mean orientation agreement: {mean_orient}")
    print(f"  Orientation disagreements (summed over pairs): {n_orient_disagree}")

    # Edge counts per method
    edge_counts = {r['label']: len(r['edges']) for r in results}

    return {
        'n_samples': n_samples,
        'label': label,
        'method_labels': method_labels,
        'edge_counts': edge_counts,
        'pairwise_comparisons': [
            {k: v for k, v in p.items() if k != 'edge_details'}  # exclude verbose details
            for p in pairwise
        ],
        'overall_agreements': overall_agreements,
        'orientation_agreements': orientation_agreements,
        'mean_overall_agreement': mean_overall,
        'mean_orientation_agreement': mean_orient,
        'n_orientation_disagree_pairs': n_orient_disagree,
        'n_edge_only_one_method': n_only_one,
    }


def bootstrap_ci_agreement(agreements: list, n_boot: int = 2000, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for mean agreement."""
    rng = np.random.default_rng(seed)
    arr = np.array(agreements)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (lo, float(np.mean(arr)), hi)


def make_figure(results_small: dict, results_large: dict, out_name: str):
    """Bar chart: agreement rate at N=1000 vs N=100,000."""
    load_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, results, n_label in zip(
        axes,
        [results_small, results_large],
        [f"N = {results_small['n_samples']:,}\n(observational)", f"N = {results_large['n_samples']:,}\n(negative control)"]
    ):
        pairs = results['pairwise_comparisons']
        pair_labels = [f"{p['pair'][0]}\nvs\n{p['pair'][1]}" for p in pairs]
        overall_vals = [p['overall_agreement'] for p in pairs]
        orient_vals = [p['orientation_agreement'] if p['orientation_agreement'] is not None else 0.0
                       for p in pairs]

        x = np.arange(len(pairs))
        width = 0.35
        colors_overall = '#1f77b4'
        colors_orient = '#ff7f0e'

        bars1 = ax.bar(x - width/2, overall_vals, width, label='Overall edge agreement',
                       color=colors_overall, alpha=0.85, edgecolor='black', linewidth=0.7)
        bars2 = ax.bar(x + width/2, orient_vals, width, label='Orientation agreement\n(directed edges)',
                       color=colors_orient, alpha=0.85, edgecolor='black', linewidth=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Agreement rate')
        ax.set_title(n_label, fontsize=10)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=7)

        # Annotate bars with values
        for bar, val in zip(bars1, overall_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars2, orient_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Summary panel inset: overall mean comparison
    mean_small = results_small['mean_overall_agreement']
    mean_large = results_large['mean_overall_agreement']

    fig.suptitle(
        r'Causal Discovery Algorithm Agreement'
        '\n'
        r'(Asia DAG, 8 nodes; PC $\alpha$=0.05, GES, PC $\alpha$=0.01)',
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, out_name)


def write_latex_table(results_small: dict, results_large: dict,
                      ci_small: tuple, ci_large: tuple, out_path: Path):
    """Write LaTeX table of pairwise agreement rates."""

    pairs_s = results_small['pairwise_comparisons']
    pairs_l = results_large['pairwise_comparisons']

    rows = []
    for ps, pl in zip(pairs_s, pairs_l):
        a_label, b_label = ps['pair']
        row = (
            f"{a_label} vs {b_label}",
            f"{ps['overall_agreement']:.3f}",
            f"{ps['orientation_agreement']:.3f}" if ps['orientation_agreement'] is not None else "---",
            f"{pl['overall_agreement']:.3f}",
            f"{pl['orientation_agreement']:.3f}" if pl['orientation_agreement'] is not None else "---",
        )
        rows.append(row)

    header = (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\caption{Causal discovery algorithm agreement on the Asia DAG (8 nodes, "
        r"linear Gaussian). \emph{Overall} counts edges present in either graph; "
        r"\emph{Orientation} counts directed edges with consistent orientation. "
        r"At $N=1{,}000$ algorithms disagree due to finite-sample underspecification; "
        r"at $N=100{,}000$ they converge.}" + "\n"
        r"\label{tab:causal_discovery_agreement}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\toprule" + "\n"
        r" & \multicolumn{2}{c}{$N=1{,}000$} & \multicolumn{2}{c}{$N=100{,}000$} \\" + "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
        r"Algorithm pair & Overall & Orient. & Overall & Orient. \\" + "\n"
        r"\midrule" + "\n"
    )

    body = ""
    for row in rows:
        body += " & ".join(row) + r" \\" + "\n"

    footer = (
        r"\midrule" + "\n"
        f"Mean (95\\% CI) & "
        f"\\multicolumn{{2}}{{c}}{{{ci_small[1]:.3f} [{ci_small[0]:.3f}, {ci_small[2]:.3f}]}} & "
        f"\\multicolumn{{2}}{{c}}{{{ci_large[1]:.3f} [{ci_large[0]:.3f}, {ci_large[2]:.3f}]}} \\\\\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )

    tex = header + body + footer
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Saved LaTeX table: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

N_SEEDS = 10  # number of random seeds per condition for statistical test
BASE_SEEDS = list(range(10, 10 + N_SEEDS))  # seeds 10..19


def run_multi_seed(n_samples: int, seeds: list, condition_label: str) -> dict:
    """
    Run the experiment with multiple random seeds. For each seed, collect the
    mean overall pairwise agreement across the 3 algorithm pairs.
    Returns list of per-seed mean overall agreements.
    """
    per_seed_mean_agreements = []
    all_seed_results = []

    print(f"\n{'='*60}")
    print(f"Multi-seed run: {condition_label} (N={n_samples:,}, {len(seeds)} seeds)")
    print(f"{'='*60}")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        res = run_experiment(n_samples=n_samples, rng=rng,
                             label=f'{condition_label}_seed{seed}')
        per_seed_mean_agreements.append(res['mean_overall_agreement'])
        all_seed_results.append(res)

    print(f"\n  {condition_label} per-seed mean agreements: "
          f"{[f'{v:.3f}' for v in per_seed_mean_agreements]}")
    print(f"  {condition_label} overall mean: {np.mean(per_seed_mean_agreements):.3f}  "
          f"std: {np.std(per_seed_mean_agreements):.3f}")

    return {
        'n_samples': n_samples,
        'seeds': seeds,
        'per_seed_mean_agreements': per_seed_mean_agreements,
        'all_seed_results': all_seed_results,
    }


def main():
    set_all_seeds(42)
    rng_small = np.random.default_rng(42)
    rng_large = np.random.default_rng(42)

    print("=" * 60)
    print("Causal Discovery Experiment")
    print("Asia DAG — 8 nodes, linear Gaussian")
    print("Methods: PC(α=0.05), GES(BIC), PC(α=0.01)")
    print("=" * 60)

    # Single-run experiments for figure / table (original behaviour)
    results_small = run_experiment(n_samples=1000, rng=rng_small, label='small')
    results_large = run_experiment(n_samples=100_000, rng=rng_large, label='large')

    # Bootstrap CIs on overall agreement (single-run)
    ci_small = bootstrap_ci_agreement(results_small['overall_agreements'])
    ci_large = bootstrap_ci_agreement(results_large['overall_agreements'])

    # ── Multi-seed statistical test ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"MULTI-SEED STATISTICAL TEST ({N_SEEDS} seeds each condition)")
    print("=" * 60)

    multi_small = run_multi_seed(1000, BASE_SEEDS, 'N=1000')
    multi_large = run_multi_seed(100_000, BASE_SEEDS, 'N=100000')

    # Mann-Whitney U test: N=100,000 agreements > N=1,000 agreements
    mw_stat, mw_p = mannwhitneyu(
        multi_large['per_seed_mean_agreements'],
        multi_small['per_seed_mean_agreements'],
        alternative='greater',
    )

    print(f"\n  Mann-Whitney U (N=100k > N=1k): stat={mw_stat:.1f}, p={mw_p:.4e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N=1,000  mean overall agreement: {ci_small[1]:.3f}  "
          f"95% CI [{ci_small[0]:.3f}, {ci_small[2]:.3f}]")
    print(f"N=100,000 mean overall agreement: {ci_large[1]:.3f}  "
          f"95% CI [{ci_large[0]:.3f}, {ci_large[2]:.3f}]")
    print(f"Orientation disagree pairs (N=1k): {results_small['n_orientation_disagree_pairs']}")
    print(f"Orientation disagree pairs (N=100k): {results_large['n_orientation_disagree_pairs']}")
    print(f"Multi-seed MW test p-value: {mw_p:.4e}")

    # Figure (uses single-run results for clean presentation)
    make_figure(results_small, results_large, 'causal_discovery_exp')

    # LaTeX table (single-run) + append p-value footnote
    tex_path = SECTIONS_DIR / 'table_causal_discovery_exp.tex'
    write_latex_table(results_small, results_large, ci_small, ci_large, tex_path)
    # Append statistical test result to table
    with open(tex_path, 'r') as f:
        tex_content = f.read()
    # Insert before \end{table}
    footnote = (
        f"\\vspace{{2pt}}\n"
        f"\\noindent\\small Mann--Whitney $U$ test ({N_SEEDS} seeds each condition, "
        f"$N$=100{{{','}}}000 vs $N$=1{{{','}}}000): $p = {mw_p:.3e}$.\n"
    )
    tex_content = tex_content.replace(r'\end{table}', footnote + r'\end{table}')
    with open(tex_path, 'w') as f:
        f.write(tex_content)
    print(f"Updated LaTeX table with p-value: {tex_path}")

    # Results JSON
    output = {
        'experiment': 'causal_discovery',
        'dag': 'Asia (8 nodes, linear Gaussian)',
        'methods': ['PC(alpha=0.05)', 'GES(BIC)', 'PC(alpha=0.01)'],  # ASCII for JSON
        'true_edges': TRUE_EDGES,
        'v_structure_edges': list(V_STRUCTURE_EDGES),
        'reversible_edges': list(REVERSIBLE_EDGES),
        'n_small': results_small['n_samples'],
        'n_large': results_large['n_samples'],
        'results_small': {k: v for k, v in results_small.items()
                          if k not in ('pairwise_comparisons',)},
        'results_large': {k: v for k, v in results_large.items()
                          if k not in ('pairwise_comparisons',)},
        'pairwise_small': results_small['pairwise_comparisons'],
        'pairwise_large': results_large['pairwise_comparisons'],
        'ci_small': {'lo': ci_small[0], 'mean': ci_small[1], 'hi': ci_small[2]},
        'ci_large': {'lo': ci_large[0], 'mean': ci_large[1], 'hi': ci_large[2]},
        'statistical_test': {
            'test': f'Mann-Whitney U ({N_SEEDS} seeds each condition, N=100k > N=1k)',
            'statistic': float(mw_stat),
            'p_value': float(mw_p),
            'n_small_per_seed_agreements': multi_small['per_seed_mean_agreements'],
            'n_large_per_seed_agreements': multi_large['per_seed_mean_agreements'],
        },
        'interpretation': (
            f"At N=1,000, mean pairwise edge agreement is {ci_small[1]:.3f} "
            f"[{ci_small[0]:.3f}, {ci_small[2]:.3f}], with "
            f"{results_small['n_orientation_disagree_pairs']} orientation disagreement pairs. "
            f"At N=100,000 (negative control), agreement rises to {ci_large[1]:.3f} "
            f"[{ci_large[0]:.3f}, {ci_large[2]:.3f}] with "
            f"{results_large['n_orientation_disagree_pairs']} disagreement pairs. "
            f"Multi-seed Mann-Whitney U test ({N_SEEDS} seeds each): p={mw_p:.3e}. "
            "This demonstrates that finite-sample underspecification prevents any single "
            "algorithm from being simultaneously faithful, stable, and decisive."
        )
    }
    save_results(output, 'causal_discovery_exp')

    print("\nDone.")


if __name__ == '__main__':
    main()
