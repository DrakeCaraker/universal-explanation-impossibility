"""
Create the universal 8-panel dose-response figure for Nature submission.

Reads existing experiment result JSONs and produces a single 2×4 panel figure
showing the dose-response relationship across all 8 domains.

Usage:
    python paper/scripts/create_universal_figure.py
"""

import json
import sys
from math import comb, log
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Resolve paths via experiment_utils ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import load_publication_style, save_figure, PAPER_DIR

RESULTS_DIR = PAPER_DIR  # JSONs live directly in paper/

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(filename: str):
    """Load a JSON file defensively; return None on any error."""
    path = RESULTS_DIR / filename
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not load {filename}: {exc}")
        return None


def no_data(ax, label: str = "No data"):
    """Render a placeholder panel when data is unavailable."""
    ax.text(
        0.5, 0.5, label,
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=10, color='grey',
    )
    ax.set_xticks([])
    ax.set_yticks([])


def panel_label(ax, letter: str):
    """Add bold panel letter in the top-left corner."""
    ax.text(
        -0.18, 1.05, f"\\textbf{{{letter}}}",
        transform=ax.transAxes,
        fontsize=12, fontweight='bold',
        va='top', ha='left',
    )


# ── Individual panel drawers ─────────────────────────────────────────────────

def panel_A_mathematics(ax):
    """Mathematics: null-space dimension vs solver RMSD."""
    data = load_json("results_linear_solver.json")
    if data is None:
        no_data(ax)
        ax.set_title("Mathematics")
        return

    try:
        per_d = data["per_d"]           # keys are str "1".."50"
        ctrl  = data["control_d0"]

        ds        = sorted(int(k) for k in per_d)
        means     = [per_d[str(d)]["mean_rmsd"]  for d in ds]
        lo        = [per_d[str(d)]["ci_95_lo"]   for d in ds]
        hi        = [per_d[str(d)]["ci_95_hi"]   for d in ds]
        lo_err    = [m - l for m, l in zip(means, lo)]
        hi_err    = [h - m for m, h in zip(means, hi)]

        ctrl_mean = ctrl["mean_rmsd"]
        ctrl_lo   = ctrl["ci_95_lo"]
        ctrl_hi   = ctrl["ci_95_hi"]

        # Shade CI band
        ax.fill_between(ds,
                         [m - e for m, e in zip(means, lo_err)],
                         [m + e for m, e in zip(means, hi_err)],
                         alpha=0.2, color='#0072B2', linewidth=0)
        ax.plot(ds, means, color='#0072B2', linewidth=1.2, label='Underdetermined')

        # Control: d=0, shown as star at x=0
        ax.errorbar(
            0, ctrl_mean,
            yerr=[[ctrl_mean - ctrl_lo], [ctrl_hi - ctrl_mean]],
            fmt='r*', markersize=8, zorder=5, label='Control ($d=0$)',
            capsize=3,
        )

        ax.set_xlabel("Null-space dimension $d$")
        ax.set_ylabel("Solver RMSD")
        ax.legend(fontsize=7)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel A: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Mathematics")


def panel_B_biology(ax):
    """Biology: codon degeneracy vs entropy (bits)."""
    data = load_json("results_codon_entropy.json")
    if data is None:
        no_data(ax)
        ax.set_title("Biology")
        return

    try:
        rows = data["summary_by_degeneracy"]   # list of dicts
        degs  = [r["degeneracy"]          for r in rows]
        means = [r["obs_entropy_mean"]    for r in rows]
        stds  = [r["obs_entropy_std"]     for r in rows]
        maxes = [r["max_entropy"]         for r in rows]   # theoretical max

        colors = ['#D55E00' if d == 1 else '#0072B2' for d in degs]

        for i, (d, m, s, c) in enumerate(zip(degs, means, stds, colors)):
            marker = '*' if d == 1 else 'o'
            ms     = 9  if d == 1 else 5
            ax.errorbar(d, m, yerr=s, fmt=marker, markersize=ms,
                        color=c, capsize=3, zorder=5)

        ax.plot(degs, means, color='#0072B2', linewidth=1.0, zorder=3)
        ax.plot(degs, maxes, color='grey', linewidth=0.8, linestyle='--',
                label='Max entropy')

        # Annotate control
        ax.annotate("Control\n(H=0)", xy=(1, 0), xytext=(1.8, 0.35),
                    fontsize=7, color='#D55E00',
                    arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.8))

        ax.set_xlabel("Codon degeneracy")
        ax.set_ylabel("Shannon entropy (bits)")
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.legend(fontsize=7)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel B: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Biology")


def panel_C_gauge_theory(ax):
    """Gauge Theory: link variance vs plaquette variance across lattice sizes."""
    data = load_json("results_gauge_lattice.json")
    if data is None:
        no_data(ax)
        ax.set_title("Gauge Theory")
        return

    try:
        sizes      = data["lattice_sizes"]                       # [4,6,8,10]
        link_var   = data["mean_within_orbit_link_variance"]     # ~0.25 each
        plaq_var   = data["mean_within_orbit_plaquette_variance"]# 0.0 each

        x = np.arange(len(sizes))
        width = 0.35

        bars1 = ax.bar(x - width/2, link_var,  width, label='Gauge-variant (links)',
                       color='#0072B2', alpha=0.85)
        bars2 = ax.bar(x + width/2, plaq_var, width, label='Gauge-invariant (plaquettes)',
                       color='#D55E00', alpha=0.85)

        # Star on the control bars (plaquette = 0)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.005,
                    '$\\star$', ha='center', va='bottom',
                    color='#D55E00', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"${s}\\times{s}$" for s in sizes])
        ax.set_xlabel("Lattice size")
        ax.set_ylabel("Within-orbit variance")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel C: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Gauge Theory")


def panel_D_stat_mech(ax):
    """Statistical Mechanics: Rashomon entropy = log C(N,k) vs macrostate k."""
    # Compute directly for N=20 (confirmed in JSON key_values)
    N = 20
    ks = list(range(N + 1))
    try:
        rashomon_entropy = [log(comb(N, k)) if comb(N, k) > 0 else 0.0
                            for k in ks]
    except (ValueError, OverflowError) as exc:
        print(f"  WARNING panel D compute: {exc}")
        no_data(ax, "Compute error")
        ax.set_title("Statistical Mechanics")
        return

    # Optionally verify against JSON
    data = load_json("results_stat_mech_entropy.json")
    if data is not None:
        try:
            stored_max = data["key_values"]["20"]["S_R_at_max"]
            computed_max = max(rashomon_entropy)
            if abs(stored_max - computed_max) > 0.01:
                print(f"  WARNING panel D: computed max {computed_max:.4f} != "
                      f"stored {stored_max:.4f}")
        except (KeyError, TypeError):
            pass

    ax.plot(ks, rashomon_entropy, color='#0072B2', linewidth=1.2)
    ax.fill_between(ks, rashomon_entropy, alpha=0.15, color='#0072B2')

    # Control: k=0 (only 1 microstate, S_R=0)
    ax.plot(0, 0, 'r*', markersize=9, zorder=5, label='Control ($k=0$, $S_R=0$)')
    ax.plot(N, 0, 'r*', markersize=9, zorder=5)  # symmetric endpoint

    # Annotate peak
    k_peak = N // 2
    ax.annotate(
        f"$S_R^{{\\max}}={max(rashomon_entropy):.1f}$",
        xy=(k_peak, max(rashomon_entropy)),
        xytext=(k_peak + 3, max(rashomon_entropy) - 1.5),
        fontsize=7,
        arrowprops=dict(arrowstyle='->', lw=0.8),
    )

    ax.set_xlabel("Macrostate $k$ (spins up)")
    ax.set_ylabel(r"$S_R(k) = \ln\binom{N}{k}$")
    ax.set_xlim(0, N)
    ax.legend(fontsize=7)
    ax.set_title("Statistical Mechanics")


def panel_E_linguistics(ax):
    """Linguistics: parser UAS for ambiguous vs unambiguous sentences."""
    data = load_json("results_parser_disagreement.json")
    if data is None:
        no_data(ax)
        ax.set_title("Linguistics")
        return

    try:
        amb_vals   = data["ambiguous_agreements"]    # list[50]
        unamb_vals = data["unambiguous_agreements"]  # list[50]

        positions = [1, 2]
        parts = ax.violinplot(
            [amb_vals, unamb_vals],
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.6,
        )

        colors = ['#D55E00', '#0072B2']
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')

        # Star on ambiguous (negative control = lower agreement)
        ax.plot(1, np.mean(amb_vals), 'r*', markersize=9, zorder=5,
                label='Lower agreement (ambiguous)')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Ambiguous', 'Unambiguous'], fontsize=8)
        ax.set_ylabel("Parser agreement (UAS)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel E: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Linguistics")


def panel_F_crystallography(ax):
    """Crystallography: signal length vs phase-retrieval RMSD."""
    data = load_json("results_phase_retrieval.json")
    if data is None:
        no_data(ax)
        ax.set_title("Crystallography")
        return

    try:
        per_length = data["per_length"]
        lengths    = sorted(int(k) for k in per_length)

        gen_means  = [per_length[str(n)]["general"]["mean_pairwise_rmsd"]  for n in lengths]
        gen_lo     = [per_length[str(n)]["general"]["ci_95_lo"]            for n in lengths]
        gen_hi     = [per_length[str(n)]["general"]["ci_95_hi"]            for n in lengths]

        pos_means  = [per_length[str(n)]["positive_control"]["mean_pairwise_rmsd"] for n in lengths]
        pos_lo     = [per_length[str(n)]["positive_control"]["ci_95_lo"]           for n in lengths]
        pos_hi     = [per_length[str(n)]["positive_control"]["ci_95_hi"]           for n in lengths]

        ax.fill_between(lengths, gen_lo,  gen_hi,  alpha=0.2, color='#0072B2')
        ax.fill_between(lengths, pos_lo,  pos_hi,  alpha=0.2, color='#009E73')
        ax.plot(lengths, gen_means, 'o-', color='#0072B2', linewidth=1.2,
                markersize=4, label='Unconstrained')
        ax.plot(lengths, pos_means, 's-', color='#009E73', linewidth=1.2,
                markersize=4, label='Positive control')

        # Star at shortest length for unconstrained (largest ratio)
        ax.plot(lengths[0], gen_means[0], 'r*', markersize=9, zorder=5)

        ax.set_xscale('log', base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(n) for n in lengths])
        ax.set_xlabel("Signal length $N$")
        ax.set_ylabel("Reconstruction RMSD")
        ax.legend(fontsize=7)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel F: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Crystallography")


def panel_G_computer_science(ax):
    """Computer Science: number of counties vs KL divergence."""
    data = load_json("results_census_disagg.json")
    if data is None:
        no_data(ax)
        ax.set_title("Computer Science")
        return

    try:
        per_state = data["per_state"]  # list of dicts

        n_counties = [s["n_counties"] for s in per_state]
        mean_kl    = [s["mean_kl"]    for s in per_state]

        # Identify DC (n_counties=1, KL=0) as the control
        ctrl_idx = [i for i, s in enumerate(per_state) if s["n_counties"] == 1]

        ax.scatter(n_counties, mean_kl, s=15, alpha=0.6,
                   color='#0072B2', zorder=3, label='State')

        if ctrl_idx:
            ci = ctrl_idx[0]
            ax.scatter(n_counties[ci], mean_kl[ci], s=100, marker='*',
                       color='red', zorder=5, label='Control (DC, $n=1$)')

        # Log-linear fit from stored parameters
        try:
            slope     = data["log_linear_fit"]["slope"]
            intercept = data["log_linear_fit"]["intercept"]
            xs = np.linspace(min(n_counties), max(n_counties), 200)
            ys = slope * np.log(xs) + intercept
            ax.plot(xs, ys, '--', color='#D55E00', linewidth=0.9,
                    label=f'Log fit ($r={data["pearson_r"]:.2f}$)')
        except (KeyError, TypeError, ValueError):
            pass

        ax.set_xscale('log')
        ax.set_xlabel("Number of counties (log scale)")
        ax.set_ylabel("Mean KL divergence")
        ax.legend(fontsize=7)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel G: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Computer Science")


def panel_H_statistics(ax):
    """Statistics: causal-discovery algorithm agreement at N=1k vs N=100k."""
    data = load_json("results_causal_discovery_exp.json")
    if data is None:
        no_data(ax)
        ax.set_title("Statistics")
        return

    try:
        small_agreements = data["statistical_test"]["n_small_per_seed_agreements"]
        large_agreements = data["statistical_test"]["n_large_per_seed_agreements"]

        ci_small = data["ci_small"]
        ci_large = data["ci_large"]

        positions = [1, 2]
        parts = ax.violinplot(
            [small_agreements, large_agreements],
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.6,
        )

        colors = ['#D55E00', '#009E73']
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')

        # Mean ± CI markers
        ax.errorbar(1, ci_small["mean"],
                    yerr=[[ci_small["mean"] - ci_small["lo"]],
                          [ci_small["hi"]  - ci_small["mean"]]],
                    fmt='r*', markersize=9, zorder=5,
                    capsize=3, label='Low $N$ (underspecified)')
        ax.errorbar(2, ci_large["mean"],
                    yerr=[[ci_large["mean"] - ci_large["lo"]],
                          [ci_large["hi"]  - ci_large["mean"]]],
                    fmt='s', color='#009E73', markersize=5, zorder=5,
                    capsize=3, label='High $N$ (control)')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['$N=1{,}000$', '$N=100{,}000$'], fontsize=8)
        ax.set_ylabel("Pairwise edge agreement")
        ax.set_ylim(0, 0.75)
        ax.legend(fontsize=7)

        # p-value annotation
        p = data["statistical_test"]["p_value"]
        ax.text(0.98, 0.05, f"$p={p:.1e}$",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

    except (KeyError, TypeError, ValueError) as exc:
        print(f"  WARNING panel H: {exc}")
        no_data(ax, f"Parse error:\n{exc}")

    ax.set_title("Statistics")


# ── Main figure assembly ──────────────────────────────────────────────────────

def main():
    print("Loading publication style...")
    load_publication_style()

    print("Creating 2×4 panel figure...")
    fig, axes = plt.subplots(
        2, 4,
        figsize=(14, 7),
        constrained_layout=True,
    )

    panels = [
        ("A", panel_A_mathematics),
        ("B", panel_B_biology),
        ("C", panel_C_gauge_theory),
        ("D", panel_D_stat_mech),
        ("E", panel_E_linguistics),
        ("F", panel_F_crystallography),
        ("G", panel_G_computer_science),
        ("H", panel_H_statistics),
    ]

    for (letter, draw_fn), ax in zip(panels, axes.flat):
        print(f"  Panel {letter}...")
        try:
            draw_fn(ax)
        except Exception as exc:
            print(f"  ERROR panel {letter}: {exc}")
            no_data(ax, f"Error:\n{exc}")
            ax.set_title(letter)
        panel_label(ax, letter)

    print("Saving figure...")
    save_figure(fig, "universal_dose_response")
    print("Done.")


if __name__ == "__main__":
    main()
