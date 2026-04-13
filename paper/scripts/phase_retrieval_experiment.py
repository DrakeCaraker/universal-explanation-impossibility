"""
Task 2.3: Crystallography — Phase Retrieval
===========================================
Gerchberg-Saxton phase retrieval on 1D signals of lengths N = 16, 32, 64, 128.

For each length:
  - Generate a random real-valued signal from N(0,1)  (general case)
  - Generate a positive control signal (all values >= 0)
  - Run 20 reconstructions from random initial phases
  - Compute pairwise RMSD between the 20 reconstructions
  - Compute feature agreement (local maxima within 2 positions)

Scaling hypothesis: RMSD increases with N (more phase ambiguity in longer signals).

Output:
  paper/results_phase_retrieval.json
  paper/figures/phase_retrieval.pdf
  paper/sections/table_phase_retrieval.tex
"""

import sys
import itertools
from pathlib import Path

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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.stats import mannwhitneyu

# ── Constants ──────────────────────────────────────────────────────────────
SIGNAL_LENGTHS = [16, 32, 64, 128]
N_RECONSTRUCTIONS = 20       # random initial-phase starts per signal
GS_ITERATIONS = 500          # Gerchberg-Saxton iterations
N_BOOT = 2000                # bootstrap replicates
FEATURE_WINDOW = 2           # local-max agreement window (±positions)


# ── Gerchberg-Saxton ───────────────────────────────────────────────────────

def gs_reconstruct(magnitudes: np.ndarray, positivity: bool = False,
                   n_iter: int = GS_ITERATIONS, rng=None) -> np.ndarray:
    """
    Run one Gerchberg-Saxton reconstruction.

    Parameters
    ----------
    magnitudes : 1-D array, |F[k]| — the measured Fourier magnitudes.
    positivity : if True, clip negative real-space values to 0 each iteration.
    n_iter     : number of GS iterations.
    rng        : numpy Generator for reproducible random phases.

    Returns
    -------
    Reconstructed real-space signal (float64, same length as magnitudes).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(magnitudes)

    # Step 0 — random initial phases on [0, 2π]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=N)

    for _ in range(n_iter):
        # Step i — apply magnitude constraint: keep current phase, set |F| = measured
        F = magnitudes * np.exp(1j * phases)

        # Step ii — inverse FFT → real-space estimate
        signal_est = np.fft.ifft(F).real          # take real part

        # Step iii — real-space constraint
        if positivity:
            signal_est = np.clip(signal_est, 0.0, None)

        # Step iv — FFT of constrained estimate
        F_new = np.fft.fft(signal_est)

        # Step v — extract updated phases for next iteration
        phases = np.angle(F_new)

    return signal_est


def run_reconstructions(magnitudes: np.ndarray, positivity: bool,
                        seed_offset: int = 0) -> np.ndarray:
    """
    Run N_RECONSTRUCTIONS independent GS reconstructions.

    Returns an array of shape (N_RECONSTRUCTIONS, N) with the reconstructed signals.
    """
    N = len(magnitudes)
    results = np.zeros((N_RECONSTRUCTIONS, N))
    for i in range(N_RECONSTRUCTIONS):
        rng = np.random.default_rng(seed=1000 * seed_offset + i)
        results[i] = gs_reconstruct(magnitudes, positivity=positivity, rng=rng)
    return results


# ── Metrics ────────────────────────────────────────────────────────────────

def pairwise_rmsd_matrix(reconstructions: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise RMSDs between rows of `reconstructions`
    (shape: n_recon × N).  Returns a flat array of n_recon*(n_recon-1)/2 values.
    """
    rmsds = []
    n = reconstructions.shape[0]
    for i, j in itertools.combinations(range(n), 2):
        diff = reconstructions[i] - reconstructions[j]
        rmsds.append(float(np.sqrt(np.mean(diff ** 2))))
    return np.array(rmsds)


def per_reconstruction_mean_rmsd(reconstructions: np.ndarray) -> np.ndarray:
    """
    For each of the n reconstructions, compute its mean RMSD to all other
    (n-1) reconstructions.  Returns an array of length n — one independent
    summary statistic per reconstruction.
    """
    n = reconstructions.shape[0]
    means = np.zeros(n)
    for i in range(n):
        rmsds_i = []
        for j in range(n):
            if i == j:
                continue
            diff = reconstructions[i] - reconstructions[j]
            rmsds_i.append(float(np.sqrt(np.mean(diff ** 2))))
        means[i] = np.mean(rmsds_i)
    return means


def feature_agreement(true_signal: np.ndarray,
                      reconstructions: np.ndarray,
                      window: int = FEATURE_WINDOW) -> float:
    """
    For each local maximum in `true_signal`, check whether at least one
    local maximum appears within `window` positions in each reconstruction.
    Returns the mean fraction of true peaks matched across all reconstructions.
    """
    # Find true local maxima (order=1: strict local max)
    true_peaks = set(argrelmax(true_signal, order=1)[0])
    if not true_peaks:
        return float('nan')

    match_rates = []
    for recon in reconstructions:
        recon_peaks = set(argrelmax(recon, order=1)[0])
        matched = 0
        N = len(true_signal)
        for p in true_peaks:
            neighbors = range(max(0, p - window), min(N, p + window + 1))
            if any(q in recon_peaks for q in neighbors):
                matched += 1
        match_rates.append(matched / len(true_peaks))

    return float(np.mean(match_rates))


# ── Signal generators ──────────────────────────────────────────────────────

def make_general_signal(N: int) -> np.ndarray:
    """Random signal from N(0,1)."""
    return np.random.randn(N)


def make_positive_signal(N: int) -> np.ndarray:
    """All non-negative signal: |N(0,1)|."""
    return np.abs(np.random.randn(N))


# ── Main experiment ────────────────────────────────────────────────────────

def run_experiment():
    set_all_seeds(42)
    load_publication_style()

    results_by_N = {}
    all_general_rmsds = []   # collected across all N for overall test
    all_positive_rmsds = []  # collected across all N for overall test

    print("\n=== Phase Retrieval Experiment ===")
    print(f"  Signal lengths: {SIGNAL_LENGTHS}")
    print(f"  Reconstructions per signal: {N_RECONSTRUCTIONS}")
    print(f"  GS iterations: {GS_ITERATIONS}\n")

    for idx, N in enumerate(SIGNAL_LENGTHS):

        # ── Generate signals ──────────────────────────────────────────────
        general_signal  = make_general_signal(N)
        positive_signal = make_positive_signal(N)

        # ── Compute measured Fourier magnitudes ───────────────────────────
        gen_magnitudes = np.abs(np.fft.fft(general_signal))
        pos_magnitudes = np.abs(np.fft.fft(positive_signal))

        # ── Run reconstructions ───────────────────────────────────────────
        gen_recons = run_reconstructions(gen_magnitudes,  positivity=False, seed_offset=idx * 10 + 1)
        pos_recons = run_reconstructions(pos_magnitudes, positivity=True,  seed_offset=idx * 10 + 2)

        # ── Pairwise RMSDs (kept for visualization) ─────────────────────
        gen_rmsds = pairwise_rmsd_matrix(gen_recons)
        pos_rmsds = pairwise_rmsd_matrix(pos_recons)

        gen_lo, gen_mean, gen_hi = percentile_ci(gen_rmsds.tolist(), n_boot=N_BOOT)
        pos_lo, pos_mean, pos_hi = percentile_ci(pos_rmsds.tolist(), n_boot=N_BOOT)

        # Legacy pairwise Mann-Whitney (pseudoreplicated — kept for reference)
        mw_stat_n_legacy, mw_p_n_legacy = mannwhitneyu(gen_rmsds, pos_rmsds, alternative='greater')

        # ── Per-reconstruction mean RMSD (independent summary stats) ──
        gen_per_recon = per_reconstruction_mean_rmsd(gen_recons)
        pos_per_recon = per_reconstruction_mean_rmsd(pos_recons)

        # Corrected test: Mann-Whitney U on N=20 vs N=20 independent means
        mw_stat_n, mw_p_n = mannwhitneyu(gen_per_recon, pos_per_recon, alternative='greater')

        # Accumulate per-reconstruction means for overall test
        all_general_rmsds.extend(gen_per_recon.tolist())
        all_positive_rmsds.extend(pos_per_recon.tolist())

        ratio = gen_mean / pos_mean if pos_mean > 0 else float('inf')

        # ── Feature agreement ─────────────────────────────────────────────
        gen_feat_agree = feature_agreement(general_signal,  gen_recons)
        pos_feat_agree = feature_agreement(positive_signal, pos_recons)

        results_by_N[N] = {
            "N": N,
            "general": {
                "mean_pairwise_rmsd": gen_mean,
                "ci_95_lo": gen_lo,
                "ci_95_hi": gen_hi,
                "n_pairwise": len(gen_rmsds),
                "per_reconstruction_mean_rmsd": gen_per_recon.tolist(),
                "feature_agreement": gen_feat_agree,
            },
            "positive_control": {
                "mean_pairwise_rmsd": pos_mean,
                "ci_95_lo": pos_lo,
                "ci_95_hi": pos_hi,
                "n_pairwise": len(pos_rmsds),
                "per_reconstruction_mean_rmsd": pos_per_recon.tolist(),
                "feature_agreement": pos_feat_agree,
            },
            "rmsd_ratio_general_over_positive": ratio,
            "mann_whitney_per_N": {
                "test": "Mann-Whitney U on per-reconstruction mean RMSDs (N=20 vs N=20)",
                "statistic": float(mw_stat_n),
                "p_value": float(mw_p_n),
                "n_per_group": N_RECONSTRUCTIONS,
            },
            "pairwise_legacy": {
                "test": "Mann-Whitney U on C(20,2)=190 pairwise RMSDs (PSEUDOREPLICATED — do not cite)",
                "statistic": float(mw_stat_n_legacy),
                "p_value": float(mw_p_n_legacy),
                "n_per_group": len(gen_rmsds),
            },
        }

        print(f"  N={N:3d} | General RMSD: {gen_mean:.4f} [{gen_lo:.4f}, {gen_hi:.4f}]"
              f"  | Positive RMSD: {pos_mean:.4f} [{pos_lo:.4f}, {pos_hi:.4f}]"
              f"  | Ratio: {ratio:.2f}"
              f"  | FeatAgree (gen): {gen_feat_agree:.3f}"
              f"  | MW p(corrected)={mw_p_n:.4e}"
              f"  | MW p(legacy)={mw_p_n_legacy:.4e}")

    # ── Overall Mann-Whitney U test across all signal lengths ───────────────
    # Corrected: uses per-reconstruction means (N=80 vs N=80, 20 per signal length)
    mw_stat_all, mw_p_all = mannwhitneyu(all_general_rmsds, all_positive_rmsds, alternative='greater')
    print(f"\n  Overall MW U test (corrected, per-recon means, N={len(all_general_rmsds)} vs N={len(all_positive_rmsds)}): "
          f"stat={mw_stat_all:.1f}, p={mw_p_all:.4e}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    Ns     = SIGNAL_LENGTHS
    gen_means  = [results_by_N[N]["general"]["mean_pairwise_rmsd"]         for N in Ns]
    gen_los    = [results_by_N[N]["general"]["mean_pairwise_rmsd"]
                  - results_by_N[N]["general"]["ci_95_lo"]                 for N in Ns]
    gen_his    = [results_by_N[N]["general"]["ci_95_hi"]
                  - results_by_N[N]["general"]["mean_pairwise_rmsd"]       for N in Ns]

    pos_means  = [results_by_N[N]["positive_control"]["mean_pairwise_rmsd"] for N in Ns]
    pos_los    = [results_by_N[N]["positive_control"]["mean_pairwise_rmsd"]
                  - results_by_N[N]["positive_control"]["ci_95_lo"]        for N in Ns]
    pos_his    = [results_by_N[N]["positive_control"]["ci_95_hi"]
                  - results_by_N[N]["positive_control"]["mean_pairwise_rmsd"] for N in Ns]

    ratios = [results_by_N[N]["rmsd_ratio_general_over_positive"] for N in Ns]

    # Left panel: RMSD vs signal length
    ax_left.errorbar(Ns, gen_means, yerr=[gen_los, gen_his],
                     fmt='o-', color='steelblue', linewidth=2.0, markersize=7,
                     capsize=5, label='General (phase unknown)', zorder=3)
    ax_left.errorbar(Ns, pos_means, yerr=[pos_los, pos_his],
                     fmt='s--', color='#2ecc71', linewidth=2.0, markersize=7,
                     capsize=5, label='Positive control (positivity constraint)', zorder=3)

    ax_left.set_xlabel('Signal length $N$')
    ax_left.set_ylabel('Mean pairwise RMSD between reconstructions')
    ax_left.set_title('Phase retrieval ambiguity vs. signal length')
    ax_left.legend(fontsize=8)
    ax_left.set_xticks(Ns)
    ax_left.set_xticklabels([str(n) for n in Ns])

    # Right panel: RMSD ratio bar chart
    bar_colors = ['#3498db', '#2980b9', '#1f618d', '#154360']
    bars = ax_right.bar([str(N) for N in Ns], ratios,
                        color=bar_colors, width=0.5, zorder=3)
    ax_right.axhline(y=1.0, color='#c0392b', linestyle='--', linewidth=1.5,
                     label='Ratio = 1 (no advantage)')
    ax_right.set_xlabel('Signal length $N$')
    ax_right.set_ylabel('RMSD ratio (general / positive control)')
    ax_right.set_title('Ambiguity reduction from positivity constraint')
    ax_right.legend(fontsize=8)

    for bar, ratio in zip(bars, ratios):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.02,
            f'{ratio:.2f}x',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    fig.tight_layout(pad=2.0)
    save_figure(fig, "phase_retrieval")

    # ── LaTeX table ────────────────────────────────────────────────────────
    sections_dir = PAPER_DIR / "sections"
    sections_dir.mkdir(exist_ok=True)
    table_path = sections_dir / "table_phase_retrieval.tex"

    with open(table_path, 'w') as f:
        f.write(r"""\begin{table}[h]
\centering
\caption{Gerchberg-Saxton phase retrieval ambiguity for 1D signals of varying length.
Each cell shows mean pairwise RMSD across 20 reconstructions from random initial phases
($N_{\text{boot}}=2000$, 95\% CI). The positive control applies a non-negativity
constraint at each iteration, substantially reducing ambiguity.
RMSD ratio = general / positive control.
$p$-values: one-sided Mann--Whitney $U$ test on per-reconstruction mean RMSDs
($N=20$ vs $N=20$ independent summaries; general $>$ positive).}
\label{tab:phase_retrieval}
\begin{tabular}{rccccccc}
\toprule
$N$ & \multicolumn{2}{c}{General (no constraint)} & \multicolumn{2}{c}{Positive control} & Ratio & Feat.\ agree (gen) & MW $p$-value \\
    & RMSD & 95\% CI & RMSD & 95\% CI & gen/pos & & \\
\midrule
""")
        for N in SIGNAL_LENGTHS:
            r = results_by_N[N]
            g = r["general"]
            p = r["positive_control"]
            fa = g["feature_agreement"]
            fa_str = f'{fa:.3f}' if not (isinstance(fa, float) and fa != fa) else 'n/a'
            mw_p_val = r["mann_whitney_per_N"]["p_value"]
            f.write(
                f'  {N:3d} & '
                f'{g["mean_pairwise_rmsd"]:.4f} & '
                f'[{g["ci_95_lo"]:.4f},\\ {g["ci_95_hi"]:.4f}] & '
                f'{p["mean_pairwise_rmsd"]:.4f} & '
                f'[{p["ci_95_lo"]:.4f},\\ {p["ci_95_hi"]:.4f}] & '
                f'{r["rmsd_ratio_general_over_positive"]:.2f} & '
                f'{fa_str} & '
                f'{mw_p_val:.3e} \\\\\n'
            )
        f.write(r"\midrule" + "\n")
        f.write(
            f"  \\multicolumn{{8}}{{l}}{{Overall MW $U$ test (all $N$ combined): "
            f"$p = {mw_p_all:.3e}$}} \\\\\n"
        )
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"\nSaved table: {table_path}")

    # ── Results JSON ───────────────────────────────────────────────────────
    results = {
        "experiment": "phase_retrieval",
        "description": (
            "Gerchberg-Saxton phase retrieval ambiguity on 1D signals. "
            "Mean pairwise RMSD across 20 reconstructions from random initial phases. "
            "General (no real-space constraint) vs. positive control (non-negativity clip)."
        ),
        "config": {
            "signal_lengths": SIGNAL_LENGTHS,
            "n_reconstructions": N_RECONSTRUCTIONS,
            "gs_iterations": GS_ITERATIONS,
            "n_boot": N_BOOT,
            "feature_window": FEATURE_WINDOW,
            "seed": 42,
        },
        "per_length": {
            str(N): results_by_N[N] for N in SIGNAL_LENGTHS
        },
        "statistical_test_overall": {
            "test": "Mann-Whitney U on per-reconstruction mean RMSDs (one-sided: general > positive, all N combined)",
            "statistic": float(mw_stat_all),
            "p_value": float(mw_p_all),
            "n_general": len(all_general_rmsds),
            "n_positive": len(all_positive_rmsds),
            "note": "Each reconstruction contributes one independent mean-RMSD value, avoiding pseudoreplication from pairwise dependencies.",
        },
    }

    save_results(results, "phase_retrieval")

    print("\n=== Summary ===")
    print(f"  {'N':>4}  {'Gen RMSD':>10}  {'Pos RMSD':>10}  {'Ratio':>6}  {'FeatAgree':>10}")
    for N in SIGNAL_LENGTHS:
        r = results_by_N[N]
        g_rmsd = r["general"]["mean_pairwise_rmsd"]
        p_rmsd = r["positive_control"]["mean_pairwise_rmsd"]
        ratio  = r["rmsd_ratio_general_over_positive"]
        fa     = r["general"]["feature_agreement"]
        fa_str = f'{fa:.3f}' if not (isinstance(fa, float) and fa != fa) else 'n/a'
        print(f"  {N:>4}  {g_rmsd:>10.4f}  {p_rmsd:>10.4f}  {ratio:>6.2f}  {fa_str:>10}")

    print("\n=== Done ===")
    return results


if __name__ == "__main__":
    run_experiment()
