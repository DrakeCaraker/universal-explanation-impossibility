"""
Task 1.1: Mathematics — Solver Disagreement
===========================================
Generate underdetermined linear systems Ax=b (m<n).
Vary null space dimension d = n - m from 1 to 50.
Solve with 4 methods: pseudoinverse, LSQR, Tikhonov, random null-space.
Compute pairwise RMSD. Control: square systems (d=0).

Output:
  paper/results_linear_solver.json
  paper/figures/linear_solver.pdf
  paper/sections/table_linear_solver.tex
"""

import sys
import os
import json
import numpy as np
import itertools
from pathlib import Path

# Allow importing experiment_utils from same directory
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
    PAPER_DIR,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import lsqr as scipy_lsqr

# ── Constants ──────────────────────────────────────────────────────────────
M = 10          # number of equations (fixed)
D_VALUES = list(range(1, 51))   # null space dim 1..50 → n = 11..60
REPS_PER_D = 2  # random systems per d value
TIKHONOV_LAMBDA = 0.01
N_CONTROL = 20  # fully-determined control systems (m=n=10)
N_BOOT = 2000   # bootstrap replicates


# ── Solver implementations ─────────────────────────────────────────────────

def solve_pseudoinverse(A, b):
    """Minimum-norm solution via np.linalg.lstsq."""
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def solve_lsqr(A, b):
    """LSQR iterative solver (minimum-norm)."""
    result = scipy_lsqr(A, b)
    return result[0]


def solve_tikhonov(A, b, lam=TIKHONOV_LAMBDA):
    """Tikhonov regularisation: x = (A^T A + λI)^{-1} A^T b."""
    n = A.shape[1]
    x = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)
    return x


def solve_random_null(A, b):
    """Particular solution (pseudoinverse) + random null-space component."""
    x_particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Null space basis via SVD
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > 1e-10)
    null_basis = Vt[rank:].T          # shape (n, d)
    if null_basis.shape[1] == 0:
        return x_particular            # no null space (control case)
    # Random null-space direction with unit length
    coeff = np.random.randn(null_basis.shape[1])
    coeff /= (np.linalg.norm(coeff) + 1e-14)
    return x_particular + null_basis @ coeff


SOLVERS = [
    ("pseudoinverse", solve_pseudoinverse),
    ("lsqr",         solve_lsqr),
    ("tikhonov",     solve_tikhonov),
    ("random_null",  solve_random_null),
]
SOLVER_NAMES = [name for name, _ in SOLVERS]


# ── RMSD helper ────────────────────────────────────────────────────────────

def pairwise_rmsd(solutions: dict) -> float:
    """Mean pairwise RMSD over all pairs of solver solutions."""
    names = list(solutions.keys())
    rmsds = []
    for i, j in itertools.combinations(range(len(names)), 2):
        diff = solutions[names[i]] - solutions[names[j]]
        rmsds.append(np.sqrt(np.mean(diff**2)))
    return float(np.mean(rmsds)) if rmsds else 0.0


def individual_pairwise_rmsds(solutions: dict) -> list:
    """All pairwise RMSDs as a list."""
    names = list(solutions.keys())
    rmsds = []
    for i, j in itertools.combinations(range(len(names)), 2):
        diff = solutions[names[i]] - solutions[names[j]]
        rmsds.append(float(np.sqrt(np.mean(diff**2))))
    return rmsds


# ── Experiment ─────────────────────────────────────────────────────────────

def run_one_system(m, n):
    """Generate one random system and solve with all methods. Return mean pairwise RMSD."""
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true

    solutions = {}
    for name, solver in SOLVERS:
        solutions[name] = solver(A, b)

    return pairwise_rmsd(solutions), individual_pairwise_rmsds(solutions)


def run_experiment():
    set_all_seeds(42)
    load_publication_style()

    # ── Underdetermined systems ─────────────────────────────────────────────
    records = []  # (d, mean_rmsd)
    all_rmsds_by_d = {}

    for d in D_VALUES:
        n = M + d
        rmsds_this_d = []
        for rep in range(REPS_PER_D):
            mean_rmsd, pairwise = run_one_system(M, n)
            rmsds_this_d.extend(pairwise)   # collect all pairs
            records.append({"d": d, "n": n, "rep": rep, "mean_pairwise_rmsd": mean_rmsd})
        all_rmsds_by_d[d] = rmsds_this_d

    # ── Control: fully-determined systems (m=n=10) ─────────────────────────
    control_rmsds = []
    for _ in range(N_CONTROL):
        _, pairwise = run_one_system(M, M)
        control_rmsds.extend(pairwise)

    # ── Aggregate ──────────────────────────────────────────────────────────
    d_vals = sorted(all_rmsds_by_d.keys())
    d_mean_rmsd = {d: float(np.mean(v)) for d, v in all_rmsds_by_d.items()}

    # Bootstrap CI for underdetermined (d>0) overall
    all_underdet_rmsds = [v for d in d_vals for v in all_rmsds_by_d[d]]
    ci_lo, ci_mean, ci_hi = percentile_ci(all_underdet_rmsds, n_boot=N_BOOT)

    # Bootstrap CI for control
    ctl_lo, ctl_mean, ctl_hi = percentile_ci(control_rmsds, n_boot=N_BOOT)

    # Per-d bootstrap CIs
    d_ci = {}
    for d in d_vals:
        lo, mn, hi = percentile_ci(all_rmsds_by_d[d], n_boot=N_BOOT)
        d_ci[d] = {"lo": lo, "mean": mn, "hi": hi}

    # ── Trend line (linear regression of mean RMSD vs d) ──────────────────
    xs = np.array(d_vals, dtype=float)
    ys = np.array([d_mean_rmsd[d] for d in d_vals])
    slope, intercept = np.polyfit(xs, ys, 1)

    print("\n=== Linear Solver Experiment Results ===")
    print(f"  Underdetermined RMSD: {ci_mean:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Control (d=0) RMSD:   {ctl_mean:.4f}  95% CI [{ctl_lo:.4f}, {ctl_hi:.4f}]")
    print(f"  Trend: slope = {slope:.5f}, intercept = {intercept:.5f}")
    print(f"  d=1 mean RMSD: {d_mean_rmsd[1]:.4f}")
    print(f"  d=50 mean RMSD: {d_mean_rmsd[50]:.4f}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Left panel: RMSD vs null-space dimension (scatter + trend) ---
    # Scatter: individual rep mean RMSDs
    scatter_x = [r["d"] for r in records]
    scatter_y = [r["mean_pairwise_rmsd"] for r in records]
    ax_left.scatter(scatter_x, scatter_y, alpha=0.55, s=22, color='steelblue',
                    zorder=3, label='Individual systems')

    # Trend line
    x_trend = np.array([min(d_vals), max(d_vals)])
    ax_left.plot(x_trend, slope * x_trend + intercept, color='#c0392b',
                 linewidth=2.0, linestyle='--', label=f'Trend (slope={slope:.4f})', zorder=4)

    # Per-d means
    ax_left.plot(d_vals, [d_mean_rmsd[d] for d in d_vals],
                 color='navy', linewidth=1.2, alpha=0.6, label='Per-d mean', zorder=2)

    ax_left.set_xlabel('Null space dimension $d = n - m$')
    ax_left.set_ylabel('Mean pairwise RMSD between solvers')
    ax_left.set_title('Solver disagreement vs. null space dimension')
    ax_left.legend(fontsize=8)
    ax_left.set_xlim(0, 51)

    # --- Right panel: bar chart (d>0 overall vs control) ---
    bar_labels = ['Underdetermined\n($d > 0$)', 'Determined\n($d = 0$, control)']
    bar_heights = [ci_mean, ctl_mean]
    bar_errs_lo = [ci_mean - ci_lo, ctl_mean - ctl_lo]
    bar_errs_hi = [ci_hi - ci_mean, ctl_hi - ctl_mean]
    colors = ['steelblue', '#7f8c8d']

    bars = ax_right.bar(
        bar_labels, bar_heights,
        yerr=[bar_errs_lo, bar_errs_hi],
        color=colors, width=0.45, capsize=6, error_kw={'linewidth': 1.5},
        zorder=3
    )

    ax_right.set_ylabel('Mean pairwise RMSD between solvers')
    ax_right.set_title('Underspecification vs. control')

    # Annotate bars
    for bar, h, lo, hi in zip(bars, bar_heights, bar_errs_lo, bar_errs_hi):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            h + hi + ax_right.get_ylim()[1] * 0.01,
            f'{h:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    fig.tight_layout(pad=2.0)
    save_figure(fig, "linear_solver")

    # ── LaTeX table ────────────────────────────────────────────────────────
    sections_dir = PAPER_DIR / "sections"
    sections_dir.mkdir(exist_ok=True)
    table_path = sections_dir / "table_linear_solver.tex"

    # Select a few representative d values for the table
    rep_ds = [1, 5, 10, 20, 30, 40, 50]

    with open(table_path, 'w') as f:
        f.write(r"""\begin{table}[h]
\centering
\caption{Pairwise RMSD between four linear solvers for underdetermined systems $Ax=b$
($m=10$ equations, $n=m+d$ unknowns, 2 random systems per $d$ value).
Control: $d=0$ (square, fully determined). Bootstrap 95\% CIs over all pairwise solver comparisons.}
\label{tab:linear_solver}
\begin{tabular}{rcccc}
\toprule
$d = n - m$ & $n$ & Mean RMSD & 95\% CI low & 95\% CI high \\
\midrule
""")
        for d in rep_ds:
            ci = d_ci[d]
            f.write(f"  {d:3d} & {M + d:3d} & {ci['mean']:.4f} & {ci['lo']:.4f} & {ci['hi']:.4f} \\\\\n")
        # Control row
        f.write(r"\midrule" + "\n")
        f.write(f"  0 (control) & {M:3d} & {ctl_mean:.4f} & {ctl_lo:.4f} & {ctl_hi:.4f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Saved table: {table_path}")

    # ── Results JSON ───────────────────────────────────────────────────────
    results = {
        "experiment": "linear_solver",
        "description": "Pairwise RMSD between 4 linear solvers for underdetermined Ax=b",
        "config": {
            "m": M,
            "d_range": [min(D_VALUES), max(D_VALUES)],
            "reps_per_d": REPS_PER_D,
            "n_control_systems": N_CONTROL,
            "tikhonov_lambda": TIKHONOV_LAMBDA,
            "solver_names": SOLVER_NAMES,
        },
        "underdetermined": {
            "mean_rmsd": ci_mean,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "n_pairwise_observations": len(all_underdet_rmsds),
        },
        "control_d0": {
            "mean_rmsd": ctl_mean,
            "ci_95_lo": ctl_lo,
            "ci_95_hi": ctl_hi,
            "n_pairwise_observations": len(control_rmsds),
        },
        "trend": {
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "per_d": {
            str(d): {
                "n": M + d,
                "mean_rmsd": d_ci[d]["mean"],
                "ci_95_lo": d_ci[d]["lo"],
                "ci_95_hi": d_ci[d]["hi"],
            }
            for d in d_vals
        },
        "individual_records": records,
    }

    save_results(results, "linear_solver")

    print("\n=== Done ===")
    return results


if __name__ == "__main__":
    run_experiment()
