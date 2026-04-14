#!/usr/bin/env python3
"""
Molecular Evolution Experiment: Testing Representation-Theoretic Predictions
in the Standard Genetic Code

Tests an EXTRAPOLATION of the Universal Explanation Impossibility framework:
the character theory of S_k predicts that the "gauge-variant" fraction of
codon space = (k-1)/k, which should correlate with synonymous substitution
properties.

Parts:
  A. Synonymous substitution rate by degeneracy (computational)
  B. Codon neighbor structure analysis
  C. Test the representation-theoretic prediction (model comparison)
  D. Extend existing codon entropy results (eta validation)
"""

import json
import os
import sys
import math
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ──────────────────────────────────────────────────────────────────────
# STANDARD GENETIC CODE (DNA alphabet: T, C, A, G)
# ──────────────────────────────────────────────────────────────────────

CODON_TABLE = {
    "TTT": "Phe", "TTC": "Phe",
    "TTA": "Leu", "TTG": "Leu",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile",
    "ATG": "Met",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr",
    "TAA": "Stop", "TAG": "Stop",
    "CAT": "His", "CAC": "His",
    "CAA": "Gln", "CAG": "Gln",
    "AAT": "Asn", "AAC": "Asn",
    "AAA": "Lys", "AAG": "Lys",
    "GAT": "Asp", "GAC": "Asp",
    "GAA": "Glu", "GAG": "Glu",
    "TGT": "Cys", "TGC": "Cys",
    "TGA": "Stop",
    "TGG": "Trp",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGT": "Ser", "AGC": "Ser",
    "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

BASES = ["T", "C", "A", "G"]


def get_sense_codons():
    """Return list of 61 sense codons."""
    return [c for c, aa in CODON_TABLE.items() if aa != "Stop"]


def get_amino_acid_codons():
    """Return dict: amino_acid -> list of codons."""
    aa_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        if aa != "Stop":
            aa_codons[aa].append(codon)
    return dict(aa_codons)


def get_degeneracy():
    """Return dict: amino_acid -> degeneracy k."""
    aa_codons = get_amino_acid_codons()
    return {aa: len(codons) for aa, codons in aa_codons.items()}


def single_nucleotide_neighbors(codon):
    """Return all 9 single-nucleotide mutations of a codon."""
    neighbors = []
    for pos in range(3):
        for base in BASES:
            if base != codon[pos]:
                mutant = codon[:pos] + base + codon[pos+1:]
                neighbors.append(mutant)
    return neighbors


# ──────────────────────────────────────────────────────────────────────
# PART B: Codon Neighbor Structure Analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_codon_neighbors():
    """
    For each sense codon, enumerate all 9 single-nucleotide mutations
    and classify as synonymous, nonsynonymous, or nonsense.
    """
    sense_codons = get_sense_codons()
    aa_codons = get_amino_acid_codons()
    degeneracy = get_degeneracy()

    codon_data = []

    for codon in sense_codons:
        aa = CODON_TABLE[codon]
        k = degeneracy[aa]
        neighbors = single_nucleotide_neighbors(codon)

        n_syn = 0
        n_nonsyn = 0
        n_nonsense = 0

        for nb in neighbors:
            nb_aa = CODON_TABLE[nb]
            if nb_aa == "Stop":
                n_nonsense += 1
            elif nb_aa == aa:
                n_syn += 1
            else:
                n_nonsyn += 1

        assert n_syn + n_nonsyn + n_nonsense == 9, f"Expected 9 neighbors for {codon}"

        codon_data.append({
            "codon": codon,
            "amino_acid": aa,
            "degeneracy": k,
            "n_synonymous": n_syn,
            "n_nonsynonymous": n_nonsyn,
            "n_nonsense": n_nonsense,
            "frac_synonymous": n_syn / 9.0,
            "frac_nonsynonymous": n_nonsyn / 9.0,
            "frac_nonsense": n_nonsense / 9.0,
        })

    return codon_data


def aggregate_by_degeneracy(codon_data):
    """Aggregate codon neighbor statistics by degeneracy class."""
    by_k = defaultdict(list)
    for cd in codon_data:
        by_k[cd["degeneracy"]].append(cd)

    summary = {}
    for k in sorted(by_k.keys()):
        codons = by_k[k]
        frac_syn = [c["frac_synonymous"] for c in codons]
        n_syn = [c["n_synonymous"] for c in codons]
        summary[k] = {
            "k": k,
            "n_codons": len(codons),
            "amino_acids": sorted(set(c["amino_acid"] for c in codons)),
            "mean_frac_synonymous": np.mean(frac_syn),
            "std_frac_synonymous": np.std(frac_syn, ddof=1) if len(frac_syn) > 1 else 0.0,
            "mean_n_synonymous": np.mean(n_syn),
            "std_n_synonymous": np.std(n_syn, ddof=1) if len(n_syn) > 1 else 0.0,
            "min_n_synonymous": min(n_syn),
            "max_n_synonymous": max(n_syn),
            "character_prediction": (k - 1) / k,  # group-theoretic prediction
            "individual_frac_syn": frac_syn,
            "individual_n_syn": n_syn,
        }

    return summary


# ──────────────────────────────────────────────────────────────────────
# PART C: Model Comparison
# ──────────────────────────────────────────────────────────────────────

def fit_models(summary):
    """
    Fit three models to the data:
    1. Representation theory: frac_syn = a * (k-1)/k + b
    2. Information-theoretic: frac_syn = a * log(k) + b
    3. Neighbor-count: frac_syn = a * mean_neighbors + b
    """
    ks = sorted(summary.keys())

    # Use per-codon data for proper regression (not just means)
    # But first do mean-level analysis for visualization
    y_mean = np.array([summary[k]["mean_frac_synonymous"] for k in ks])
    x_char = np.array([(k - 1) / k for k in ks])
    x_log = np.array([np.log(k) for k in ks])
    x_neigh = np.array([summary[k]["mean_n_synonymous"] for k in ks])

    # Weighted regression using number of codons as weights
    weights = np.array([summary[k]["n_codons"] for k in ks])

    results = {}

    # --- Per-codon level analysis (proper) ---
    all_frac_syn = []
    all_k = []
    all_n_syn = []
    for k in ks:
        for fs in summary[k]["individual_frac_syn"]:
            all_frac_syn.append(fs)
            all_k.append(k)
        for ns in summary[k]["individual_n_syn"]:
            all_n_syn.append(ns)

    all_frac_syn = np.array(all_frac_syn)
    all_k = np.array(all_k)
    all_n_syn = np.array(all_n_syn)
    all_char = (all_k - 1) / all_k
    all_log_k = np.log(all_k)

    # Model 1: Representation theory (k-1)/k
    slope1, intercept1, r1, p1, se1 = stats.linregress(all_char, all_frac_syn)
    results["representation_theory"] = {
        "model": "frac_syn = a * (k-1)/k + b",
        "a": float(slope1),
        "b": float(intercept1),
        "R2": float(r1**2),
        "r": float(r1),
        "p_value": float(p1),
        "se_slope": float(se1),
    }

    # Model 2: Information-theoretic log(k)
    slope2, intercept2, r2, p2, se2 = stats.linregress(all_log_k, all_frac_syn)
    results["information_theoretic"] = {
        "model": "frac_syn = a * log(k) + b",
        "a": float(slope2),
        "b": float(intercept2),
        "R2": float(r2**2),
        "r": float(r2),
        "p_value": float(p2),
        "se_slope": float(se2),
    }

    # Model 3: Neighbor count
    slope3, intercept3, r3, p3, se3 = stats.linregress(all_n_syn, all_frac_syn)
    results["neighbor_count"] = {
        "model": "frac_syn = a * mean_neighbors + b",
        "a": float(slope3),
        "b": float(intercept3),
        "R2": float(r3**2),
        "r": float(r3),
        "p_value": float(p3),
        "se_slope": float(se3),
    }

    # --- Residual analysis: does (k-1)/k explain variance AFTER neighbor count? ---
    # Partial correlation: correlation of frac_syn with (k-1)/k after removing
    # effect of neighbor count
    # Residualize both frac_syn and (k-1)/k on neighbor count
    s_yn, i_yn, _, _, _ = stats.linregress(all_n_syn, all_frac_syn)
    resid_y = all_frac_syn - (s_yn * all_n_syn + i_yn)

    s_xn, i_xn, _, _, _ = stats.linregress(all_n_syn, all_char)
    resid_x = all_char - (s_xn * all_n_syn + i_xn)

    if np.std(resid_x) > 1e-12 and np.std(resid_y) > 1e-12:
        partial_r, partial_p = stats.pearsonr(resid_x, resid_y)
    else:
        partial_r, partial_p = 0.0, 1.0

    results["partial_correlation"] = {
        "description": "Correlation of frac_syn with (k-1)/k AFTER controlling for neighbor count",
        "partial_r": float(partial_r),
        "partial_p": float(partial_p),
        "partial_R2": float(partial_r**2),
        "interpretation": (
            "Group structure explains additional variance beyond neighbor count"
            if partial_p < 0.05
            else "Group structure does NOT explain additional variance beyond neighbor count"
        ),
    }

    # --- Mean-level fits for plotting ---
    results["mean_level"] = {
        "ks": [int(k) for k in ks],
        "y_mean": y_mean.tolist(),
        "x_char": x_char.tolist(),
        "x_log": x_log.tolist(),
        "x_neigh": x_neigh.tolist(),
        "weights": weights.tolist(),
    }

    # Weighted R^2 at the mean level
    for name, x_arr in [("repr_theory_mean", x_char), ("info_theory_mean", x_log), ("neighbor_mean", x_neigh)]:
        sw, iw, rw, pw, sew = stats.linregress(x_arr, y_mean)
        results[name] = {"R2_mean": float(rw**2), "r_mean": float(rw), "p_mean": float(pw)}

    return results


# ──────────────────────────────────────────────────────────────────────
# PART D: Extend Existing Codon Entropy Results (eta validation)
# ──────────────────────────────────────────────────────────────────────

def compute_eta_values(existing_results):
    """
    For each degeneracy class k, compute:
    - Predicted information RETAINED = 1/k (character theory)
    - Observed information retained = 1 - H_obs/H_max
    """
    eta_data = []
    summary = existing_results.get("summary_by_degeneracy", [])

    for entry in summary:
        k = entry["degeneracy"]
        if k == 1:
            # k=1 is degenerate: H_max = 0, no information to lose
            continue

        H_obs = entry["obs_entropy_mean"]
        H_max = entry["max_entropy"]

        if H_max > 0:
            info_retained_obs = 1.0 - H_obs / H_max
            info_retained_pred = 1.0 / k
            eta_obs = H_obs / H_max  # fraction of max entropy realized

            eta_data.append({
                "k": k,
                "H_obs": H_obs,
                "H_max": H_max,
                "info_retained_obs": info_retained_obs,
                "info_retained_pred": info_retained_pred,
                "eta_obs": eta_obs,
                "eta_pred": (k - 1) / k,  # group-theoretic: fraction gauge-variant
            })

    # Also use real data if available
    real_summary = existing_results.get("real_data_summary_by_degeneracy", {})
    eta_real = []
    for k_str, data in real_summary.items():
        k = int(k_str)
        if k == 1:
            continue
        H_obs = data["mean_entropy"]
        H_max = math.log2(k)
        if H_max > 0:
            eta_real.append({
                "k": k,
                "H_obs_real": H_obs,
                "H_max": H_max,
                "info_retained_real": 1.0 - H_obs / H_max,
                "info_retained_pred": 1.0 / k,
                "eta_obs_real": H_obs / H_max,
                "eta_pred": (k - 1) / k,
            })

    return eta_data, eta_real


# ──────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def create_figure(summary, model_results, eta_data, eta_real, codon_data):
    """Create multi-panel figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Molecular Evolution: Representation-Theoretic Predictions\nin the Standard Genetic Code",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ks = sorted(summary.keys())
    y_mean = np.array([summary[k]["mean_frac_synonymous"] for k in ks])
    y_std = np.array([summary[k]["std_frac_synonymous"] for k in ks])

    # ── Panel A: Degeneracy vs. mean fraction synonymous ──
    ax = axes[0, 0]
    ax.errorbar(ks, y_mean, yerr=y_std, fmt="o-", color="steelblue",
                capsize=4, markersize=8, linewidth=2)
    ax.set_xlabel("Degeneracy k", fontsize=11)
    ax.set_ylabel("Mean fraction synonymous neighbors", fontsize=11)
    ax.set_title("A. Synonymous neighbor fraction by degeneracy", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    # ── Panel B: (k-1)/k prediction vs observed ──
    ax = axes[0, 1]
    x_char = np.array([(k - 1) / k for k in ks])
    ax.scatter(x_char, y_mean, s=100, c="steelblue", zorder=5, edgecolors="navy")
    # Fit line
    sl, ic, _, _, _ = stats.linregress(x_char, y_mean)
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, sl * x_fit + ic, "r--", linewidth=1.5, alpha=0.8,
            label=f"$R^2$ = {model_results['repr_theory_mean']['R2_mean']:.4f}")
    for i, k in enumerate(ks):
        ax.annotate(f"k={k}", (x_char[i], y_mean[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("$(k-1)/k$ (character theory prediction)", fontsize=11)
    ax.set_ylabel("Mean fraction synonymous", fontsize=11)
    ax.set_title("B. Representation theory model", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel C: Model comparison (R^2 bar chart) ──
    ax = axes[0, 2]
    model_names = ["$(k{-}1)/k$\n(repr. theory)", "$\\log(k)$\n(info. theory)", "neighbor\ncount"]
    r2_codon = [
        model_results["representation_theory"]["R2"],
        model_results["information_theoretic"]["R2"],
        model_results["neighbor_count"]["R2"],
    ]
    r2_mean = [
        model_results["repr_theory_mean"]["R2_mean"],
        model_results["info_theory_mean"]["R2_mean"],
        model_results["neighbor_mean"]["R2_mean"],
    ]
    x_pos = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, r2_codon, width, label="Per-codon", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, r2_mean, width, label="Per-degeneracy-class", color="coral", alpha=0.8)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("C. Model comparison", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    # ── Panel D: Per-codon scatter colored by degeneracy ──
    ax = axes[1, 0]
    colors_map = {1: "gray", 2: "green", 3: "orange", 4: "blue", 6: "red"}
    for cd in codon_data:
        k = cd["degeneracy"]
        ax.scatter(cd["n_synonymous"], cd["frac_synonymous"],
                   c=colors_map.get(k, "black"), s=30, alpha=0.7,
                   edgecolors="none")
    # Legend
    for k_val, col in sorted(colors_map.items()):
        ax.scatter([], [], c=col, s=50, label=f"k={k_val}")
    ax.legend(fontsize=9, title="Degeneracy", title_fontsize=9)
    ax.set_xlabel("Number of synonymous neighbors", fontsize=11)
    ax.set_ylabel("Fraction synonymous (n/9)", fontsize=11)
    ax.set_title("D. Per-codon neighbor structure", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel E: eta validation (simulated data) ──
    ax = axes[1, 1]
    if eta_data:
        k_vals = [d["k"] for d in eta_data]
        eta_obs = [d["eta_obs"] for d in eta_data]
        eta_pred = [d["eta_pred"] for d in eta_data]
        ax.scatter(eta_pred, eta_obs, s=100, c="steelblue", zorder=5,
                   edgecolors="navy", label="Simulated")
        if eta_real:
            k_real = [d["k"] for d in eta_real]
            eta_obs_r = [d["eta_obs_real"] for d in eta_real]
            eta_pred_r = [d["eta_pred"] for d in eta_real]
            ax.scatter(eta_pred_r, eta_obs_r, s=100, c="coral", zorder=5,
                       edgecolors="darkred", marker="D", label="Real (NCBI)")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="$\\eta_{obs} = \\eta_{pred}$")
        # Fit
        all_pred = eta_pred + (eta_pred_r if eta_real else [])
        all_obs = eta_obs + (eta_obs_r if eta_real else [])
        if len(all_pred) > 2:
            sl_e, ic_e, r_e, _, _ = stats.linregress(all_pred, all_obs)
            ax.plot([min(all_pred), max(all_pred)],
                    [sl_e * min(all_pred) + ic_e, sl_e * max(all_pred) + ic_e],
                    "r-", alpha=0.5, linewidth=1.5,
                    label=f"Fit: $R^2$ = {r_e**2:.4f}")
        for d in eta_data:
            ax.annotate(f"k={d['k']}", (d["eta_pred"], d["eta_obs"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9)
        ax.set_xlabel("$\\eta_{pred} = (k-1)/k$ (character theory)", fontsize=11)
        ax.set_ylabel("$\\eta_{obs} = H_{obs}/H_{max}$", fontsize=11)
        ax.set_title("E. Entropy utilization: prediction vs observation", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No existing entropy data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("E. Entropy utilization", fontsize=11, fontweight="bold")

    # ── Panel F: Partial correlation (residual plot) ──
    ax = axes[1, 2]
    # Compute residuals for visualization
    all_frac_syn = []
    all_k_arr = []
    all_n_syn = []
    for k in ks:
        for fs in summary[k]["individual_frac_syn"]:
            all_frac_syn.append(fs)
        for ns in summary[k]["individual_n_syn"]:
            all_n_syn.append(ns)
        for _ in summary[k]["individual_frac_syn"]:
            all_k_arr.append(k)
    all_frac_syn = np.array(all_frac_syn)
    all_k_arr = np.array(all_k_arr)
    all_n_syn = np.array(all_n_syn)
    all_char = (all_k_arr - 1) / all_k_arr

    # Residualize
    s_yn, i_yn, _, _, _ = stats.linregress(all_n_syn, all_frac_syn)
    resid_y = all_frac_syn - (s_yn * all_n_syn + i_yn)
    s_xn, i_xn, _, _, _ = stats.linregress(all_n_syn, all_char)
    resid_x = all_char - (s_xn * all_n_syn + i_xn)

    for i, k in enumerate(all_k_arr):
        ax.scatter(resid_x[i], resid_y[i], c=colors_map.get(k, "black"),
                   s=30, alpha=0.7, edgecolors="none")
    partial_r = model_results["partial_correlation"]["partial_r"]
    partial_p = model_results["partial_correlation"]["partial_p"]
    ax.set_xlabel("Residual $(k{-}1)/k$ | neighbor count", fontsize=11)
    ax.set_ylabel("Residual frac. syn. | neighbor count", fontsize=11)
    ax.set_title(
        f"F. Partial correlation\n$r_{{partial}}$ = {partial_r:.4f}, p = {partial_p:.2e}",
        fontsize=11, fontweight="bold",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    # Legend
    for k_val, col in sorted(colors_map.items()):
        ax.scatter([], [], c=col, s=50, label=f"k={k_val}")
    ax.legend(fontsize=8, title="Degeneracy", title_fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_path = os.path.join(script_dir, "results_molecular_evolution.json")
    figure_path = os.path.join(script_dir, "figures", "molecular_evolution.pdf")
    existing_path = os.path.join(project_dir, "paper", "results_codon_entropy.json")

    print("=" * 70)
    print("MOLECULAR EVOLUTION EXPERIMENT")
    print("Testing representation-theoretic predictions in the genetic code")
    print("=" * 70)

    # ── Part A & B: Codon neighbor structure ──
    print("\n[Part B] Analyzing codon neighbor structure...")
    codon_data = analyze_codon_neighbors()
    summary = aggregate_by_degeneracy(codon_data)

    print(f"\n{'k':>4}  {'n_codons':>8}  {'mean_syn':>10}  {'std_syn':>10}  {'(k-1)/k':>8}  {'amino_acids'}")
    print("-" * 80)
    for k in sorted(summary.keys()):
        s = summary[k]
        aa_str = ", ".join(s["amino_acids"])
        print(f"{k:>4}  {s['n_codons']:>8}  {s['mean_frac_synonymous']:>10.4f}  "
              f"{s['std_frac_synonymous']:>10.4f}  {s['character_prediction']:>8.4f}  {aa_str}")

    # ── Part C: Model comparison ──
    print("\n[Part C] Fitting and comparing models...")
    model_results = fit_models(summary)

    print("\n--- Per-codon regression results ---")
    for name in ["representation_theory", "information_theoretic", "neighbor_count"]:
        m = model_results[name]
        print(f"  {name:30s}: R^2 = {m['R2']:.6f}, r = {m['r']:.4f}, p = {m['p_value']:.2e}")

    print("\n--- Per-degeneracy-class regression results ---")
    for name in ["repr_theory_mean", "info_theory_mean", "neighbor_mean"]:
        m = model_results[name]
        print(f"  {name:30s}: R^2 = {m['R2_mean']:.6f}, r = {m['r_mean']:.4f}, p = {m['p_mean']:.2e}")

    print("\n--- Partial correlation (group structure | neighbor count) ---")
    pc = model_results["partial_correlation"]
    print(f"  partial r  = {pc['partial_r']:.6f}")
    print(f"  partial p  = {pc['partial_p']:.2e}")
    print(f"  partial R^2 = {pc['partial_R2']:.6f}")
    print(f"  Interpretation: {pc['interpretation']}")

    # ── Part D: Eta validation ──
    print("\n[Part D] Computing eta values from existing codon entropy data...")
    eta_data = []
    eta_real = []
    if os.path.exists(existing_path):
        with open(existing_path, "r") as f:
            existing = json.load(f)
        eta_data, eta_real = compute_eta_values(existing)

        print(f"\n--- Simulated data eta values ---")
        print(f"{'k':>4}  {'eta_pred':>10}  {'eta_obs':>10}  {'info_ret_pred':>14}  {'info_ret_obs':>14}")
        print("-" * 60)
        for d in eta_data:
            print(f"{d['k']:>4}  {d['eta_pred']:>10.4f}  {d['eta_obs']:>10.4f}  "
                  f"{d['info_retained_pred']:>14.4f}  {d['info_retained_obs']:>14.4f}")

        if eta_real:
            print(f"\n--- Real data (NCBI) eta values ---")
            print(f"{'k':>4}  {'eta_pred':>10}  {'eta_obs':>10}  {'info_ret_pred':>14}  {'info_ret_obs':>14}")
            print("-" * 60)
            for d in eta_real:
                print(f"{d['k']:>4}  {d['eta_pred']:>10.4f}  {d['eta_obs_real']:>10.4f}  "
                      f"{d['info_retained_pred']:>14.4f}  {d['info_retained_real']:>14.4f}")

        # Spearman correlation for eta
        if len(eta_data) >= 3:
            pred = [d["eta_pred"] for d in eta_data]
            obs = [d["eta_obs"] for d in eta_data]
            rho_s, p_s = stats.spearmanr(pred, obs)
            print(f"\n  Spearman (simulated): rho = {rho_s:.4f}, p = {p_s:.4e}")

        if eta_real and len(eta_real) >= 3:
            pred_r = [d["eta_pred"] for d in eta_real]
            obs_r = [d["eta_obs_real"] for d in eta_real]
            rho_r, p_r = stats.spearmanr(pred_r, obs_r)
            print(f"  Spearman (real NCBI): rho = {rho_r:.4f}, p = {p_r:.4e}")
    else:
        print(f"  WARNING: {existing_path} not found. Skipping Part D.")

    # ── Generate figure ──
    print("\n[Figures] Generating multi-panel figure...")
    fig = create_figure(summary, model_results, eta_data, eta_real, codon_data)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    with PdfPages(figure_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"  Saved: {figure_path}")

    # ── Assemble results JSON ──
    # Clean summary for JSON serialization
    summary_json = {}
    for k, s in summary.items():
        sj = dict(s)
        sj["individual_frac_syn"] = [float(x) for x in sj["individual_frac_syn"]]
        sj["individual_n_syn"] = [int(x) for x in sj["individual_n_syn"]]
        summary_json[str(k)] = sj

    output = {
        "experiment": "molecular_evolution",
        "description": (
            "Tests representation-theoretic predictions of the Universal Explanation "
            "Impossibility framework extrapolated to molecular evolution. The character "
            "theory of S_k predicts gauge-variant fraction = (k-1)/k."
        ),
        "part_b_neighbor_structure": summary_json,
        "part_c_model_comparison": {
            "per_codon_regression": {
                name: model_results[name]
                for name in ["representation_theory", "information_theoretic", "neighbor_count"]
            },
            "per_degeneracy_class": {
                name: model_results[name]
                for name in ["repr_theory_mean", "info_theory_mean", "neighbor_mean"]
            },
            "partial_correlation": model_results["partial_correlation"],
        },
        "part_d_eta_validation": {
            "simulated": eta_data,
            "real_ncbi": eta_real,
        },
        "conclusions": {
            "best_model_per_codon": max(
                ["representation_theory", "information_theoretic", "neighbor_count"],
                key=lambda n: model_results[n]["R2"],
            ),
            "best_R2_per_codon": max(
                model_results[n]["R2"]
                for n in ["representation_theory", "information_theoretic", "neighbor_count"]
            ),
            "group_structure_residual": model_results["partial_correlation"]["partial_R2"],
            "group_structure_significant": model_results["partial_correlation"]["partial_p"] < 0.05,
        },
        "_timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPer-codon R^2:")
    for name in ["representation_theory", "information_theoretic", "neighbor_count"]:
        print(f"  {name:30s}: {model_results[name]['R2']:.6f}")
    print(f"\nPartial correlation (group structure | neighbor count):")
    print(f"  r_partial = {pc['partial_r']:.6f}, p = {pc['partial_p']:.2e}")
    print(f"  {pc['interpretation']}")
    if eta_data:
        print(f"\nEta values (simulated):")
        for d in eta_data:
            print(f"  k={d['k']}: eta_pred = {d['eta_pred']:.4f}, eta_obs = {d['eta_obs']:.4f}")
    if eta_real:
        print(f"\nEta values (real NCBI):")
        for d in eta_real:
            print(f"  k={d['k']}: eta_pred = {d['eta_pred']:.4f}, eta_obs = {d['eta_obs_real']:.4f}")
    print()


if __name__ == "__main__":
    main()
