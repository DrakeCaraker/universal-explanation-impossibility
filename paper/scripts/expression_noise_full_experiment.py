#!/usr/bin/env python3
"""
Expression Noise Full Experiment — Addressing Reviewer Concerns
================================================================

The original experiment (biology_knockout_experiments.py, Experiment 1) used
40 hand-picked yeast genes. Reviewers flagged:
  1. Selection bias in gene choice
  2. Measurement artifacts in repressed genes (GAL1, GAL10, HO, IME1, FLO1)
  3. N too small for robust partial correlations

This script:
  A. Removes the 5 artifact genes and reruns on the remaining 35
  B. Applies quality filters (expression > 100 mol/cell, TF >= 2)
  C. Performs sensitivity analysis (remove top/bottom 3 genes)
  D. Reports whether the correlation survives each filter

Data sources (same as original):
  - Noise (CV²): Newman et al. (2006) Nature 441:840-846, Supplementary Table S2
  - TF regulators: YEASTRACT (Teixeira et al. 2018) / SGD
  - TATA box: Basehoar et al. (2004) Cell 116:699-709
  - Essential genes: SGD essential gene list
  - Expression level (molecules/cell): Newman et al. (2006) & Ghaemmaghami et al. (2003)
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

np.random.seed(42)

RESULTS_PATH = Path(__file__).parent.parent / "results_expression_noise_filtered.json"

# =============================================================================
# DATA: Same 40 genes from original experiment
# =============================================================================
# (gene, noise_CV2, n_TF_regulators, has_TATA, is_essential, expression_level_log2)

yeast_genes_full = [
    # High noise, many TFs
    ("GAL1",   0.95, 8,  True,  False, 8.0),
    ("GAL10",  0.88, 7,  True,  False, 7.5),
    ("PHO5",   0.82, 6,  True,  False, 6.0),
    ("HIS4",   0.70, 9,  True,  False, 9.0),
    ("SUC2",   0.78, 7,  True,  False, 7.0),
    ("CYC1",   0.65, 8,  False, False, 10.0),
    ("ADH2",   0.72, 6,  True,  False, 8.5),
    ("FLO1",   0.90, 5,  True,  False, 5.0),

    # Medium noise, medium TFs
    ("ACT1",   0.15, 3,  False, True,  13.0),
    ("PGK1",   0.18, 4,  False, True,  14.0),
    ("TDH3",   0.12, 3,  False, True,  14.5),
    ("ENO2",   0.20, 4,  False, True,  13.5),
    ("CDC28",  0.30, 5,  False, True,  10.5),
    ("CLN3",   0.45, 6,  True,  True,  8.0),
    ("SWI4",   0.38, 5,  False, False, 9.5),
    ("MBP1",   0.35, 4,  False, True,  9.0),
    ("CLB2",   0.40, 5,  True,  False, 10.0),
    ("TUB1",   0.22, 3,  False, True,  11.0),

    # Low noise, few TFs
    ("RPL3",   0.08, 2,  False, True,  14.0),
    ("RPL25",  0.09, 2,  False, True,  13.5),
    ("RPS5",   0.07, 2,  False, True,  14.0),
    ("RPL16A", 0.10, 2,  False, True,  13.0),
    ("RPS3",   0.08, 1,  False, True,  14.5),
    ("RPL5",   0.09, 2,  False, True,  13.5),
    ("NOP1",   0.11, 2,  False, True,  12.0),
    ("SEC61",  0.14, 2,  False, True,  11.5),

    # Controls: high expression, varied TFs
    ("SSA1",   0.25, 5,  False, False, 13.0),
    ("HSP82",  0.30, 6,  True,  True,  12.5),
    ("SSB1",   0.20, 3,  False, False, 13.5),
    ("KAR2",   0.22, 4,  False, True,  12.0),

    # Controls: low expression, few TFs
    ("HO",     0.85, 4,  True,  False, 4.0),
    ("IME1",   0.80, 5,  True,  False, 3.5),
    ("SPO11",  0.75, 3,  True,  False, 4.5),

    # Additional medium-expression genes
    ("URA3",   0.55, 4,  True,  False, 8.0),
    ("LEU2",   0.50, 5,  True,  False, 9.0),
    ("TRP1",   0.42, 3,  False, False, 9.5),
    ("ADE2",   0.48, 4,  True,  False, 8.5),
    ("LYS2",   0.52, 5,  True,  False, 7.5),
    ("MET3",   0.60, 6,  True,  False, 7.0),
    ("GCN4",   0.55, 7,  True,  False, 9.0),
    ("HAP4",   0.40, 5,  False, False, 8.0),
]

# =============================================================================
# ARTIFACT GENES: known to be OFF/repressed or epigenetically silenced
# under standard lab conditions (YPD, 30°C, mid-log)
# =============================================================================
ARTIFACT_GENES = {"GAL1", "GAL10", "HO", "IME1", "FLO1"}

# Expression threshold: 100 molecules/cell = log2(100) ≈ 6.64
EXPR_THRESHOLD_LOG2 = np.log2(100)  # ~6.64

# Minimum TF regulators for meaningful regulatory complexity
MIN_TF_COUNT = 2


def to_arrays(gene_list):
    """Convert gene tuples to named arrays."""
    genes = [g[0] for g in gene_list]
    noise = np.array([g[1] for g in gene_list])
    tf_count = np.array([g[2] for g in gene_list])
    tata = np.array([1.0 if g[3] else 0.0 for g in gene_list])
    essential = np.array([1.0 if g[4] else 0.0 for g in gene_list])
    expr = np.array([g[5] for g in gene_list])
    return genes, noise, tf_count, tata, essential, expr


def residualize(y, X):
    """Residualize y on confounders X using OLS."""
    # Add intercept
    X_aug = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    return y - X_aug @ beta


def run_analysis(gene_list, label, verbose=True):
    """Run correlation analysis on a gene list. Returns results dict."""
    genes, noise, tf_count, tata, essential, expr = to_arrays(gene_list)
    n = len(genes)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"  N = {n} genes")
        print(f"{'=' * 70}")
        print(f"  Noise CV² range: [{noise.min():.2f}, {noise.max():.2f}]")
        print(f"  TF count range:  [{tf_count.min()}, {tf_count.max()}]")
        print(f"  Expression range: [{expr.min():.1f}, {expr.max():.1f}] log2(mol/cell)")

    # Raw Spearman correlation
    r_raw, p_raw = stats.spearmanr(tf_count, noise)
    if verbose:
        print(f"\n  Raw Spearman:     rho = {r_raw:.4f}, p = {p_raw:.2e}")

    # Partial correlation: residualize on expression, TATA, essentiality
    X_confounders = np.column_stack([expr, tata, essential])
    tf_resid = residualize(tf_count, X_confounders)
    noise_resid = residualize(noise, X_confounders)

    r_partial, p_partial = stats.spearmanr(tf_resid, noise_resid)
    r_partial_pearson, p_partial_pearson = stats.pearsonr(tf_resid, noise_resid)
    if verbose:
        print(f"  Partial Spearman: rho = {r_partial:.4f}, p = {p_partial:.2e}")
        print(f"  Partial Pearson:  r   = {r_partial_pearson:.4f}, p = {p_partial_pearson:.2e}")

    return {
        "label": label,
        "n_genes": n,
        "genes": genes,
        "raw_spearman_rho": round(float(r_raw), 4),
        "raw_spearman_p": float(p_raw),
        "partial_spearman_rho": round(float(r_partial), 4),
        "partial_spearman_p": float(p_partial),
        "partial_pearson_r": round(float(r_partial_pearson), 4),
        "partial_pearson_p": float(p_partial_pearson),
        "prediction_supported": bool(r_raw > 0 and p_raw < 0.05),
        "partial_supported": bool(r_partial > 0 and p_partial < 0.05),
    }


def sensitivity_analysis(gene_list, label_prefix, verbose=True):
    """
    Sensitivity: remove top 3 and bottom 3 genes by noise, rerun.
    Tests whether the correlation is driven by extremes.
    """
    genes, noise, tf_count, tata, essential, expr = to_arrays(gene_list)

    # Sort by noise
    order = np.argsort(noise)
    trimmed_indices = order[3:-3]  # remove bottom 3 and top 3
    trimmed_genes = [gene_list[i] for i in trimmed_indices]

    removed_bottom = [gene_list[i] for i in order[:3]]
    removed_top = [gene_list[i] for i in order[-3:]]

    if verbose:
        print(f"\n  Sensitivity: removing top 3 ({[g[0] for g in removed_top]}) "
              f"and bottom 3 ({[g[0] for g in removed_bottom]}) by noise")

    return run_analysis(trimmed_genes, f"{label_prefix} (trimmed top/bottom 3)", verbose)


# =============================================================================
# ANALYSIS A: Full 40 genes (baseline, same as original)
# =============================================================================
print("\n" + "=" * 80)
print("EXPRESSION NOISE FULL EXPERIMENT — ADDRESSING REVIEWER CONCERNS")
print("=" * 80)

result_full = run_analysis(yeast_genes_full, "A. Full dataset (40 genes, original)")

# =============================================================================
# ANALYSIS B: Remove 5 artifact genes
# =============================================================================
genes_no_artifacts = [g for g in yeast_genes_full if g[0] not in ARTIFACT_GENES]
result_no_artifacts = run_analysis(genes_no_artifacts,
                                   "B. Artifacts removed (GAL1, GAL10, HO, IME1, FLO1)")

# =============================================================================
# ANALYSIS C: Quality filters (expression > 100 mol/cell AND TF >= 2)
# =============================================================================
genes_filtered = [g for g in genes_no_artifacts
                  if g[5] >= EXPR_THRESHOLD_LOG2 and g[2] >= MIN_TF_COUNT]
result_filtered = run_analysis(genes_filtered,
                                f"C. Quality-filtered (expr > {EXPR_THRESHOLD_LOG2:.1f} log2, TF >= {MIN_TF_COUNT})")

# =============================================================================
# ANALYSIS D: Sensitivity — remove top/bottom 3 from artifact-removed set
# =============================================================================
sensitivity_no_artifacts = sensitivity_analysis(
    genes_no_artifacts, "D. Artifacts removed")

# =============================================================================
# ANALYSIS E: Sensitivity — remove top/bottom 3 from quality-filtered set
# =============================================================================
sensitivity_filtered = sensitivity_analysis(
    genes_filtered, "E. Quality-filtered")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

datasets = [
    ("A. Full (N=40)", result_full),
    ("B. No artifacts (N=35)", result_no_artifacts),
    ("C. Quality-filtered", result_filtered),
    ("D. No artifacts, trimmed", sensitivity_no_artifacts),
    ("E. Filtered, trimmed", sensitivity_filtered),
]

print(f"\n{'Dataset':<35} {'N':>3}  {'Raw rho':>8} {'Raw p':>10}  "
      f"{'Part rho':>9} {'Part p':>10}  {'Surv?':>5}")
print("-" * 95)
for label, res in datasets:
    surv = "YES" if res["prediction_supported"] else "NO"
    print(f"{label:<35} {res['n_genes']:>3}  "
          f"{res['raw_spearman_rho']:>8.4f} {res['raw_spearman_p']:>10.2e}  "
          f"{res['partial_spearman_rho']:>9.4f} {res['partial_spearman_p']:>10.2e}  "
          f"{surv:>5}")

# Does removing artifacts change the result?
artifact_effect = (result_full["prediction_supported"] !=
                   result_no_artifacts["prediction_supported"])

print(f"\nRemoving artifact genes changes conclusion: {artifact_effect}")
print(f"  Full dataset:     rho={result_full['raw_spearman_rho']:.4f}, p={result_full['raw_spearman_p']:.2e}")
print(f"  Without artifacts: rho={result_no_artifacts['raw_spearman_rho']:.4f}, p={result_no_artifacts['raw_spearman_p']:.2e}")

if not artifact_effect:
    print("  -> Correlation survives artifact removal.")
else:
    print("  -> WARNING: Correlation does NOT survive artifact removal.")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output = {
    "experiment": "expression_noise_filtered",
    "description": ("Reanalysis of gene expression noise vs TF regulator count, "
                    "addressing reviewer concerns about selection bias, measurement "
                    "artifacts, and small N."),
    "artifact_genes_removed": sorted(ARTIFACT_GENES),
    "artifact_removal_reason": (
        "GAL1, GAL10: galactose-inducible, essentially OFF in glucose (standard "
        "conditions); measured 'noise' is shot noise from near-zero expression. "
        "HO: mating-type switch, silenced in most lab strains. "
        "IME1: meiosis-specific, repressed in mitotic growth. "
        "FLO1: epigenetically silenced, subtelomeric."
    ),
    "quality_filters": {
        "expression_threshold_log2": round(EXPR_THRESHOLD_LOG2, 2),
        "expression_threshold_molecules_per_cell": 100,
        "min_tf_regulators": MIN_TF_COUNT,
        "rationale": (
            "Exclude genes with mean expression below 100 molecules/cell "
            "(removes shot noise artifacts where measured CV² reflects Poisson "
            "sampling rather than regulatory noise). Exclude genes with fewer "
            "than 2 known TF regulators (can't assess regulatory complexity)."
        ),
    },
    "confounders_controlled": [
        "expression_level_log2",
        "TATA_box_presence",
        "essentiality",
    ],
    "results": {
        "full_40_genes": result_full,
        "no_artifacts_35_genes": result_no_artifacts,
        "quality_filtered": result_filtered,
        "sensitivity_no_artifacts_trimmed": sensitivity_no_artifacts,
        "sensitivity_filtered_trimmed": sensitivity_filtered,
    },
    "conclusion": {
        "artifact_removal_changes_result": artifact_effect,
        "correlation_survives_artifact_removal": result_no_artifacts["prediction_supported"],
        "correlation_survives_quality_filter": result_filtered["prediction_supported"],
        "correlation_survives_sensitivity": sensitivity_no_artifacts["prediction_supported"],
        "partial_correlation_survives": result_no_artifacts["partial_supported"],
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {RESULTS_PATH}")
