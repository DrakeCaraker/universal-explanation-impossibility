"""
Refined Functional Site Prediction: Does 1/k Deviation Add Value BEYOND Conservation?
=====================================================================================

The original experiment compared AUC of deviation vs conservation separately.
Conservation won. But the CORRECT question is: does the 1/k deviation add
predictive value BEYOND conservation?

Key refinements:
  1. EXCLUDE d=1 positions (fully conserved, 1/k framework uninformative there)
  2. Logistic regression: conservation-only vs conservation+deviation (LRT)
  3. Stratified analysis: within conservation quartiles, does deviation predict?
  4. Key insight test: high-d/low-H positions (excess conserved) vs high-d/high-H
  5. Residual predictor: isolate selection signal not explained by structural degeneracy

Reuses download/alignment/annotation infrastructure from the original experiment.

Outputs:
  paper/results_functional_site_refined.json
  paper/figures/functional_site_refined.pdf
"""

import sys
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    PAPER_DIR,
)

# Import infrastructure from original experiment
from functional_site_prediction_experiment import (
    PROTEINS,
    SEED,
    MIN_COLUMN_OCCUPANCY,
    download_sequences,
    align_sequences,
    find_human_sequence_index,
    build_alignment_to_uniprot_map,
    download_uniprot_annotations,
    MIN_SPECIES,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ── Per-position computation (d>=2 focus) ─────────────────────────────────────

def compute_position_data(aligned_seqs, aln_to_uniprot, functional_positions, protein_name):
    """
    Compute per-position metrics for ALL positions, then filter to d>=2.
    Returns list of dicts with all metrics needed for the refined analysis.
    """
    if not aligned_seqs:
        return []

    n_seqs = len(aligned_seqs)
    aln_len = len(aligned_seqs[0])
    all_positions = []

    for col in range(aln_len):
        residues = [aligned_seqs[i][col] for i in range(n_seqs)]
        aa_counts = Counter(r for r in residues if r not in ('-', '.', 'X', '*'))
        occupancy = sum(aa_counts.values()) / n_seqs

        if occupancy < MIN_COLUMN_OCCUPANCY:
            continue

        d = len(aa_counts)  # number of distinct amino acids
        if d < 1:
            continue

        H = _shannon_entropy(aa_counts)
        H_max_20 = np.log2(20)

        # UniProt mapping
        uniprot_pos = aln_to_uniprot.get(col)
        is_functional = 1 if (uniprot_pos and uniprot_pos in functional_positions) else 0

        if d == 1:
            # Record but mark as d=1 (will be excluded from main analysis)
            all_positions.append({
                "position": col,
                "d": d,
                "H": 0.0,
                "H_max": 0.0,
                "ratio": 0.0,
                "ceiling": 0.0,
                "deviation": 0.0,
                "conservation": 1.0 - H / H_max_20,
                "is_functional": is_functional,
                "protein_name": protein_name,
                "uniprot_position": uniprot_pos,
            })
        else:
            H_max = np.log2(d)
            ratio = H / H_max if H_max > 0 else 0.0
            ceiling = (d - 1) / d
            deviation = ceiling - ratio  # positive = more conserved than neutral expectation

            all_positions.append({
                "position": col,
                "d": d,
                "H": float(H),
                "H_max": float(H_max),
                "ratio": float(ratio),
                "ceiling": float(ceiling),
                "deviation": float(deviation),
                "conservation": float(1.0 - H / H_max_20),
                "is_functional": is_functional,
                "protein_name": protein_name,
                "uniprot_position": uniprot_pos,
            })

    return all_positions


def _shannon_entropy(counts):
    """Shannon entropy in bits from a Counter."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return float(-np.sum(probs * np.log2(probs)))


# ── Analysis functions ────────────────────────────────────────────────────────

def logistic_regression_comparison(labels, conservation, deviation):
    """
    Model 1: logistic(functional ~ conservation)
    Model 2: logistic(functional ~ conservation + deviation)
    Likelihood ratio test (chi-squared, 1 df).
    Returns dict with results.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss

    results = {}

    # Standardize predictors for numerical stability
    cons_mean, cons_std = np.mean(conservation), np.std(conservation)
    dev_mean, dev_std = np.mean(deviation), np.std(deviation)
    cons_std = max(cons_std, 1e-10)
    dev_std = max(dev_std, 1e-10)
    cons_z = (conservation - cons_mean) / cons_std
    dev_z = (deviation - dev_mean) / dev_std

    # Model 1: conservation only
    X1 = cons_z.reshape(-1, 1)
    model1 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    model1.fit(X1, labels)
    probs1 = model1.predict_proba(X1)[:, 1]
    ll1 = -log_loss(labels, probs1, normalize=False)  # log-likelihood (negative of total loss)
    auc1 = roc_auc_score(labels, probs1)

    results["model1_auc"] = float(auc1)
    results["model1_loglik"] = float(ll1)
    results["model1_coef_conservation"] = float(model1.coef_[0][0])
    results["model1_intercept"] = float(model1.intercept_[0])

    # Model 2: conservation + deviation
    X2 = np.column_stack([cons_z, dev_z])
    model2 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    model2.fit(X2, labels)
    probs2 = model2.predict_proba(X2)[:, 1]
    ll2 = -log_loss(labels, probs2, normalize=False)
    auc2 = roc_auc_score(labels, probs2)

    results["model2_auc"] = float(auc2)
    results["model2_loglik"] = float(ll2)
    results["model2_coef_conservation"] = float(model2.coef_[0][0])
    results["model2_coef_deviation"] = float(model2.coef_[0][1])
    results["model2_intercept"] = float(model2.intercept_[0])

    # Likelihood ratio test: 2*(ll2 - ll1) ~ chi2(1)
    lr_stat = 2 * (ll2 - ll1)
    lr_stat = max(lr_stat, 0.0)  # numerical guard
    lr_pvalue = float(stats.chi2.sf(lr_stat, df=1))

    results["lr_statistic"] = float(lr_stat)
    results["lr_pvalue"] = float(lr_pvalue)
    results["lr_significant_005"] = bool(lr_pvalue < 0.05)
    results["auc_improvement"] = float(auc2 - auc1)

    return results, probs1, probs2


def stratified_analysis(labels, conservation, deviation):
    """
    Bin positions by conservation quartile. Within each quartile,
    compute AUC of deviation for predicting functional status.
    """
    from sklearn.metrics import roc_auc_score

    quartile_edges = np.percentile(conservation, [0, 25, 50, 75, 100])
    quartile_labels = np.digitize(conservation, quartile_edges[1:-1])  # 0,1,2,3

    results = []
    for q in range(4):
        mask = quartile_labels == q
        q_labels = labels[mask]
        q_deviation = deviation[mask]
        q_conservation = conservation[mask]

        n_func = int(q_labels.sum())
        n_nonfunc = int(len(q_labels) - n_func)
        n_total = int(len(q_labels))

        q_result = {
            "quartile": q,
            "conservation_range": [float(conservation[mask].min()), float(conservation[mask].max())],
            "n_total": n_total,
            "n_functional": n_func,
            "n_nonfunctional": n_nonfunc,
            "functional_fraction": float(n_func / n_total) if n_total > 0 else 0.0,
        }

        if n_func >= 2 and n_nonfunc >= 2:
            try:
                auc = roc_auc_score(q_labels, q_deviation)
                q_result["auc_deviation"] = float(auc)

                # Mann-Whitney for significance
                func_dev = q_deviation[q_labels == 1]
                nonfunc_dev = q_deviation[q_labels == 0]
                u, p = stats.mannwhitneyu(func_dev, nonfunc_dev, alternative='two-sided')
                q_result["mannwhitney_p"] = float(p)
                q_result["significant_005"] = bool(p < 0.05)
                q_result["mean_deviation_functional"] = float(np.mean(func_dev))
                q_result["mean_deviation_nonfunctional"] = float(np.mean(nonfunc_dev))
            except ValueError:
                q_result["auc_deviation"] = None
                q_result["mannwhitney_p"] = None
                q_result["significant_005"] = False
        else:
            q_result["auc_deviation"] = None
            q_result["mannwhitney_p"] = None
            q_result["significant_005"] = False
            q_result["note"] = "insufficient samples in one class"

        results.append(q_result)

    return results


def key_insight_test(labels, d_values, H_values):
    """
    Among positions with high d (d >= 4) but low entropy (H < median),
    are these 'excess conserved' positions more likely to be functional?
    Compare functional fraction at high-d/low-H vs high-d/high-H.
    """
    median_H = np.median(H_values)

    high_d = d_values >= 4
    low_H = H_values < median_H
    high_H = H_values >= median_H

    group_A = high_d & low_H   # high structural capacity, low entropy (excess conserved)
    group_B = high_d & high_H  # high structural capacity, high entropy (neutral)

    n_A = int(group_A.sum())
    n_B = int(group_B.sum())

    results = {
        "median_H": float(median_H),
        "n_high_d_low_H": n_A,
        "n_high_d_high_H": n_B,
    }

    if n_A < 3 or n_B < 3:
        results["status"] = "insufficient_data"
        results["note"] = f"Need >=3 in each group, got {n_A} and {n_B}"
        return results

    frac_A = float(labels[group_A].mean())
    frac_B = float(labels[group_B].mean())

    results["functional_fraction_high_d_low_H"] = frac_A
    results["functional_fraction_high_d_high_H"] = frac_B
    results["enrichment_ratio"] = float(frac_A / frac_B) if frac_B > 0 else float('inf')

    # Fisher's exact test
    a = int(labels[group_A].sum())   # func in group A
    b = int(n_A - a)                 # non-func in group A
    c = int(labels[group_B].sum())   # func in group B
    d_val = int(n_B - c)             # non-func in group B

    results["contingency_table"] = [[a, b], [c, d_val]]

    try:
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d_val]], alternative='greater')
        results["fisher_odds_ratio"] = float(odds_ratio)
        results["fisher_p"] = float(fisher_p)
        results["significant_005"] = bool(fisher_p < 0.05)
    except ValueError:
        results["fisher_odds_ratio"] = None
        results["fisher_p"] = None
        results["significant_005"] = False

    results["status"] = "computed"
    return results


def residual_predictor_analysis(labels, conservation, deviation, d_values, ratio_values, ceiling_values):
    """
    Use RESIDUAL from the regression of H/H_max on (d-1)/d as the predictor.
    The residual isolates the component of entropy NOT explained by structural degeneracy.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression, LinearRegression

    results = {}

    # Regress ratio on ceiling to get residuals
    lr = LinearRegression()
    lr.fit(ceiling_values.reshape(-1, 1), ratio_values)
    predicted_ratio = lr.predict(ceiling_values.reshape(-1, 1))
    residuals = ratio_values - predicted_ratio  # negative residual = more conserved than expected

    results["ols_intercept"] = float(lr.intercept_)
    results["ols_slope"] = float(lr.coef_[0])
    results["ols_r_squared"] = float(lr.score(ceiling_values.reshape(-1, 1), ratio_values))

    # The selection signal: negative residual means MORE conserved than structural degeneracy predicts
    selection_signal = -residuals  # flip sign so positive = more selection pressure

    results["residual_mean"] = float(np.mean(residuals))
    results["residual_std"] = float(np.std(residuals))

    # AUC of residual predictor
    try:
        auc_residual = roc_auc_score(labels, selection_signal)
        results["auc_residual"] = float(auc_residual)
    except ValueError:
        results["auc_residual"] = 0.5

    # Logistic: conservation + residual
    cons_z = (conservation - np.mean(conservation)) / max(np.std(conservation), 1e-10)
    res_z = (selection_signal - np.mean(selection_signal)) / max(np.std(selection_signal), 1e-10)

    X_res = np.column_stack([cons_z, res_z])
    model_res = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    model_res.fit(X_res, labels)
    probs_res = model_res.predict_proba(X_res)[:, 1]
    auc_res_model = roc_auc_score(labels, probs_res)

    results["auc_conservation_plus_residual"] = float(auc_res_model)
    results["coef_conservation"] = float(model_res.coef_[0][0])
    results["coef_residual"] = float(model_res.coef_[0][1])

    return results, selection_signal


# ── Plotting ──────────────────────────────────────────────────────────────────

def create_figure(positions_d2, lr_results, stratified_results, insight_results,
                  labels, conservation, deviation, probs1, probs2):
    """Create 4-panel figure."""
    load_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel A: Scatter of conservation vs deviation, colored by functional ──
    ax = axes[0, 0]
    func_mask = labels == 1
    nonfunc_mask = labels == 0

    ax.scatter(conservation[nonfunc_mask], deviation[nonfunc_mask],
               alpha=0.3, s=15, c='#999999', label=f'Non-functional (n={nonfunc_mask.sum()})',
               edgecolors='none')
    ax.scatter(conservation[func_mask], deviation[func_mask],
               alpha=0.7, s=25, c='#d62728', label=f'Functional (n={func_mask.sum()})',
               edgecolors='black', linewidths=0.3)

    ax.set_xlabel('Conservation (1 - H/log2(20))')
    ax.set_ylabel('Deviation: (d-1)/d - H/H_max')
    ax.set_title('A. Conservation vs. deviation (d >= 2 only)')
    ax.legend(fontsize=8, loc='upper left')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # ── Panel B: Model comparison AUCs ────────────────────────────────────────
    ax = axes[0, 1]
    auc1 = lr_results["model1_auc"]
    auc2 = lr_results["model2_auc"]
    lr_p = lr_results["lr_pvalue"]

    bars = ax.bar(['Conservation\nonly', 'Conservation\n+ deviation'],
                  [auc1, auc2],
                  color=['#b2182b', '#2166ac'], alpha=0.85, width=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, [auc1, auc2]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('AUC-ROC')
    ax.set_title('B. Logistic regression model comparison')
    ax.set_ylim(0.4, min(1.0, max(auc1, auc2) + 0.08))

    # Significance annotation
    sig_text = f'LRT p = {lr_p:.2e}' if lr_p < 0.01 else f'LRT p = {lr_p:.4f}'
    sig_color = '#d62728' if lr_p < 0.05 else '#333333'
    ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold', color=sig_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=sig_color))

    # ── Panel C: Stratified analysis ──────────────────────────────────────────
    ax = axes[1, 0]
    quartile_aucs = []
    quartile_labels_str = []
    quartile_colors = []
    for qr in stratified_results:
        q = qr["quartile"]
        auc = qr.get("auc_deviation")
        if auc is not None:
            quartile_aucs.append(auc)
            sig_marker = " *" if qr.get("significant_005") else ""
            low, high = qr["conservation_range"]
            quartile_labels_str.append(f'Q{q+1}\n[{low:.2f},{high:.2f}]{sig_marker}')
            quartile_colors.append('#2166ac' if qr.get("significant_005") else '#999999')
        else:
            quartile_aucs.append(0.5)
            low, high = qr["conservation_range"]
            quartile_labels_str.append(f'Q{q+1}\n[{low:.2f},{high:.2f}]\n(insuff.)')
            quartile_colors.append('#dddddd')

    bars_c = ax.bar(range(len(quartile_aucs)), quartile_aucs, color=quartile_colors, alpha=0.85)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance')
    ax.set_xticks(range(len(quartile_labels_str)))
    ax.set_xticklabels(quartile_labels_str, fontsize=8)
    ax.set_ylabel('AUC of deviation')
    ax.set_title('C. Deviation AUC within conservation quartiles')
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel('Conservation quartile (* = p < 0.05)')

    # ── Panel D: Key insight — high-d/low-H enrichment ───────────────────────
    ax = axes[1, 1]
    if insight_results.get("status") == "computed":
        frac_A = insight_results["functional_fraction_high_d_low_H"]
        frac_B = insight_results["functional_fraction_high_d_high_H"]
        n_A = insight_results["n_high_d_low_H"]
        n_B = insight_results["n_high_d_high_H"]
        fisher_p = insight_results.get("fisher_p", 1.0)
        enrichment = insight_results.get("enrichment_ratio", 0)

        bars_d = ax.bar(
            ['High-d / Low-H\n(excess conserved)', 'High-d / High-H\n(near neutral)'],
            [frac_A, frac_B],
            color=['#2166ac', '#b2182b'], alpha=0.85, width=0.5
        )
        for bar, val, n in zip(bars_d, [frac_A, frac_B], [n_A, n_B]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}\n(n={n})', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Fraction functional')
        ax.set_title('D. Functional enrichment at high-d positions')

        sig_text = f'Fisher p = {fisher_p:.2e}' if fisher_p < 0.01 else f'Fisher p = {fisher_p:.4f}'
        enrich_text = f'Enrichment = {enrichment:.2f}x'
        sig_color = '#d62728' if fisher_p < 0.05 else '#333333'
        ax.text(0.5, 0.95, f'{enrich_text}\n{sig_text}', transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold', color=sig_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=sig_color))
        ax.set_ylim(0, max(frac_A, frac_B) * 1.4 + 0.01)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor key insight test',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('D. Functional enrichment (insufficient data)')

    plt.tight_layout()
    save_figure(fig, "functional_site_refined")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)
    print("=" * 70)
    print("REFINED FUNCTIONAL SITE PREDICTION")
    print("Does 1/k deviation add value BEYOND conservation?")
    print("=" * 70)

    all_positions = []

    for name, gene, alts, accession in PROTEINS:
        print(f"\n{'='*70}")
        print(f"PROTEIN: {name} ({gene}, UniProt: {accession})")
        print(f"{'='*70}")

        # Step 1: Download sequences
        print("\n[Step 1] Downloading sequences...")
        sequences = download_sequences(gene, alts)
        print(f"  {len(sequences)} sequences downloaded")

        if len(sequences) < MIN_SPECIES:
            print(f"  WARNING: Only {len(sequences)} species (need {MIN_SPECIES}), skipping")
            continue

        # Step 2: Align
        print("\n[Step 2] Aligning...")
        human_idx = find_human_sequence_index(sequences)
        aligned = align_sequences(sequences, name)
        if not aligned:
            print("  Alignment failed, skipping")
            continue

        # Step 3: Build mapping
        if human_idx is not None:
            aln_to_uniprot = build_alignment_to_uniprot_map(aligned, human_idx)
        else:
            # Approximate
            aln_to_uniprot = {col: col + 1 for col in range(len(aligned[0]))}

        # Step 4: Annotations
        print("\n[Step 3] Downloading annotations...")
        functional_positions = download_uniprot_annotations(accession, name)

        # Step 5: Compute position data
        print("\n[Step 4] Computing per-position metrics...")
        protein_positions = compute_position_data(aligned, aln_to_uniprot, functional_positions, name)

        n_total = len(protein_positions)
        n_d1 = sum(1 for p in protein_positions if p["d"] == 1)
        n_d2plus = sum(1 for p in protein_positions if p["d"] >= 2)
        n_func_all = sum(1 for p in protein_positions if p["is_functional"])
        n_func_d2 = sum(1 for p in protein_positions if p["d"] >= 2 and p["is_functional"])

        print(f"  Total positions: {n_total}")
        print(f"  d=1 (invariant): {n_d1}")
        print(f"  d>=2 (variable): {n_d2plus}")
        print(f"  Functional (all): {n_func_all}")
        print(f"  Functional (d>=2): {n_func_d2}")

        all_positions.extend(protein_positions)

    # ── Filter to d>=2 ────────────────────────────────────────────────────────
    positions_d2 = [p for p in all_positions if p["d"] >= 2]

    print(f"\n{'='*70}")
    print("POOLED ANALYSIS (d >= 2 only)")
    print(f"{'='*70}")
    print(f"Total positions (d>=2): {len(positions_d2)}")

    if len(positions_d2) < 20:
        print("ERROR: Too few d>=2 positions for meaningful analysis")
        save_results({
            "experiment": "functional_site_refined",
            "status": "failed",
            "reason": f"Only {len(positions_d2)} d>=2 positions",
        }, "functional_site_refined")
        return

    # Extract arrays
    labels = np.array([p["is_functional"] for p in positions_d2])
    conservation = np.array([p["conservation"] for p in positions_d2])
    deviation = np.array([p["deviation"] for p in positions_d2])
    d_values = np.array([p["d"] for p in positions_d2])
    H_values = np.array([p["H"] for p in positions_d2])
    ratio_values = np.array([p["ratio"] for p in positions_d2])
    ceiling_values = np.array([p["ceiling"] for p in positions_d2])

    n_functional = int(labels.sum())
    n_nonfunctional = int(len(labels) - labels.sum())
    print(f"Functional: {n_functional}")
    print(f"Non-functional: {n_nonfunctional}")

    if n_functional < 5 or n_nonfunctional < 5:
        print("ERROR: Too few in one class for meaningful analysis")
        save_results({
            "experiment": "functional_site_refined",
            "status": "failed",
            "reason": f"Only {n_functional} functional / {n_nonfunctional} non-functional",
        }, "functional_site_refined")
        return

    # ── Analysis 3a: Logistic regression comparison ───────────────────────────
    print(f"\n{'─'*50}")
    print("ANALYSIS 3a: Logistic Regression Comparison")
    print(f"{'─'*50}")

    lr_results, probs1, probs2 = logistic_regression_comparison(labels, conservation, deviation)

    print(f"  Model 1 (conservation only):       AUC = {lr_results['model1_auc']:.4f}")
    print(f"  Model 2 (conservation + deviation): AUC = {lr_results['model2_auc']:.4f}")
    print(f"  AUC improvement:                   {lr_results['auc_improvement']:+.4f}")
    print(f"  Likelihood ratio statistic:         {lr_results['lr_statistic']:.4f}")
    print(f"  LRT p-value:                        {lr_results['lr_pvalue']:.6f}")
    print(f"  Significant at 0.05?                {lr_results['lr_significant_005']}")
    print(f"  Model 2 deviation coefficient:      {lr_results['model2_coef_deviation']:.4f}")

    # ── Analysis 3b: Stratified analysis ──────────────────────────────────────
    print(f"\n{'─'*50}")
    print("ANALYSIS 3b: Stratified by Conservation Quartile")
    print(f"{'─'*50}")

    stratified_results = stratified_analysis(labels, conservation, deviation)

    for qr in stratified_results:
        q = qr["quartile"]
        auc = qr.get("auc_deviation")
        p = qr.get("mannwhitney_p")
        sig = qr.get("significant_005", False)
        n_f = qr["n_functional"]
        n_nf = qr["n_nonfunctional"]
        cons_range = qr["conservation_range"]
        if auc is not None:
            print(f"  Q{q+1} [cons {cons_range[0]:.2f}-{cons_range[1]:.2f}]: "
                  f"AUC={auc:.3f}, p={p:.4f}, n_func={n_f}, n_nonfunc={n_nf}"
                  f"{' ***' if sig else ''}")
        else:
            print(f"  Q{q+1} [cons {cons_range[0]:.2f}-{cons_range[1]:.2f}]: "
                  f"insufficient data (n_func={n_f}, n_nonfunc={n_nf})")

    any_quartile_sig = any(qr.get("significant_005", False) for qr in stratified_results)

    # ── Analysis 3c: Key insight test ─────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("ANALYSIS 3c: Key Insight Test (high-d / low-H enrichment)")
    print(f"{'─'*50}")

    insight_results = key_insight_test(labels, d_values, H_values)

    if insight_results["status"] == "computed":
        frac_A = insight_results["functional_fraction_high_d_low_H"]
        frac_B = insight_results["functional_fraction_high_d_high_H"]
        enrichment = insight_results["enrichment_ratio"]
        fisher_p = insight_results["fisher_p"]
        print(f"  High-d/Low-H:  functional fraction = {frac_A:.4f} (n={insight_results['n_high_d_low_H']})")
        print(f"  High-d/High-H: functional fraction = {frac_B:.4f} (n={insight_results['n_high_d_high_H']})")
        print(f"  Enrichment ratio:                    {enrichment:.2f}x")
        print(f"  Fisher exact p:                      {fisher_p:.6f}")
        print(f"  Significant at 0.05?                 {insight_results['significant_005']}")
    else:
        print(f"  {insight_results.get('note', 'Insufficient data')}")

    # ── Analysis 4: Residual predictor ────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("ANALYSIS 4: Residual Predictor (selection signal)")
    print(f"{'─'*50}")

    residual_results, selection_signal = residual_predictor_analysis(
        labels, conservation, deviation, d_values, ratio_values, ceiling_values
    )

    print(f"  OLS: ratio = {residual_results['ols_intercept']:.4f} + "
          f"{residual_results['ols_slope']:.4f} * ceiling")
    print(f"  R-squared: {residual_results['ols_r_squared']:.4f}")
    print(f"  AUC (residual alone):              {residual_results['auc_residual']:.4f}")
    print(f"  AUC (conservation + residual):     {residual_results['auc_conservation_plus_residual']:.4f}")
    print(f"  Residual coefficient in logistic:  {residual_results['coef_residual']:.4f}")

    # ── Correlation diagnostics ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("DIAGNOSTICS")
    print(f"{'─'*50}")

    corr_cons_dev = float(np.corrcoef(conservation, deviation)[0, 1])
    corr_cons_sig = float(np.corrcoef(conservation, selection_signal)[0, 1])
    corr_dev_sig = float(np.corrcoef(deviation, selection_signal)[0, 1])
    print(f"  Correlation(conservation, deviation): {corr_cons_dev:.4f}")
    print(f"  Correlation(conservation, residual):  {corr_cons_sig:.4f}")
    print(f"  Correlation(deviation, residual):     {corr_dev_sig:.4f}")
    print(f"  d distribution: min={d_values.min()}, median={np.median(d_values):.0f}, "
          f"max={d_values.max()}, mean={d_values.mean():.1f}")

    # ── Create figure ─────────────────────────────────────────────────────────
    print("\n[Plotting] Creating figure...")
    create_figure(positions_d2, lr_results, stratified_results, insight_results,
                  labels, conservation, deviation, probs1, probs2)

    # ── Save results ──────────────────────────────────────────────────────────
    save_data = {
        "experiment": "functional_site_refined",
        "description": "Does 1/k deviation add predictive value BEYOND conservation?",
        "data_summary": {
            "n_total_positions_all": len(all_positions),
            "n_d1_excluded": sum(1 for p in all_positions if p["d"] == 1),
            "n_d2_plus_analyzed": len(positions_d2),
            "n_functional": n_functional,
            "n_nonfunctional": n_nonfunctional,
            "proteins_included": list(set(p["protein_name"] for p in positions_d2)),
        },
        "logistic_regression": lr_results,
        "stratified_analysis": stratified_results,
        "key_insight_test": insight_results,
        "residual_predictor": residual_results,
        "diagnostics": {
            "corr_conservation_deviation": corr_cons_dev,
            "corr_conservation_residual": corr_cons_sig,
            "corr_deviation_residual": corr_dev_sig,
            "d_distribution": {
                "min": int(d_values.min()),
                "max": int(d_values.max()),
                "mean": float(d_values.mean()),
                "median": float(np.median(d_values)),
            },
        },
        "per_position_data": [
            {k: v for k, v in p.items() if k != "uniprot_position"}
            for p in positions_d2
        ],
    }

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    verdict_lines = []

    # LRT
    if lr_results["lr_significant_005"]:
        verdict_lines.append(
            f"YES: Deviation adds SIGNIFICANT value beyond conservation "
            f"(LRT p={lr_results['lr_pvalue']:.2e}, "
            f"AUC improvement={lr_results['auc_improvement']:+.4f})"
        )
    else:
        verdict_lines.append(
            f"NO: Deviation does NOT add significant value beyond conservation "
            f"(LRT p={lr_results['lr_pvalue']:.4f}, "
            f"AUC improvement={lr_results['auc_improvement']:+.4f})"
        )

    # Stratified
    if any_quartile_sig:
        sig_quartiles = [qr["quartile"] + 1 for qr in stratified_results if qr.get("significant_005")]
        verdict_lines.append(
            f"STRATIFIED: Deviation is significant within conservation quartile(s): {sig_quartiles}"
        )
    else:
        verdict_lines.append(
            "STRATIFIED: Deviation is NOT significant in any conservation quartile"
        )

    # Key insight
    if insight_results.get("significant_005"):
        verdict_lines.append(
            f"KEY INSIGHT: High-d/low-H positions ARE enriched for function "
            f"(enrichment={insight_results['enrichment_ratio']:.2f}x, "
            f"Fisher p={insight_results['fisher_p']:.4f})"
        )
    elif insight_results.get("status") == "computed":
        verdict_lines.append(
            f"KEY INSIGHT: High-d/low-H positions are NOT significantly enriched "
            f"(enrichment={insight_results.get('enrichment_ratio', 0):.2f}x, "
            f"Fisher p={insight_results.get('fisher_p', 1.0):.4f})"
        )
    else:
        verdict_lines.append("KEY INSIGHT: Insufficient data for test")

    # Residual
    verdict_lines.append(
        f"RESIDUAL: AUC of selection signal (residual) = {residual_results['auc_residual']:.4f}, "
        f"coefficient in logistic = {residual_results['coef_residual']:.4f}"
    )

    # Overall
    positive_signals = sum([
        lr_results["lr_significant_005"],
        any_quartile_sig,
        insight_results.get("significant_005", False),
    ])

    if positive_signals >= 2:
        overall = "The 1/k deviation DOES add predictive value beyond conservation (multiple convergent signals)"
    elif positive_signals == 1:
        overall = "MIXED: One signal supports deviation adding value, but not consistently"
    else:
        overall = "The 1/k deviation does NOT add clear predictive value beyond conservation in this test"

    verdict_lines.append(f"\nOVERALL: {overall}")

    for line in verdict_lines:
        print(f"  {line}")

    save_data["verdict"] = {
        "lr_significant": lr_results["lr_significant_005"],
        "lr_pvalue": lr_results["lr_pvalue"],
        "any_quartile_significant": any_quartile_sig,
        "insight_significant": insight_results.get("significant_005", False),
        "positive_signals": positive_signals,
        "overall": overall,
    }

    save_results(save_data, "functional_site_refined")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")

    return save_data


if __name__ == "__main__":
    main()
