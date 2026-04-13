"""
Cross-Level Prediction: Protein Designability and the 1/k Law
==============================================================
The universal explanation impossibility framework's character theory predicts
that for S_k acting on R^k, the G-invariant resolution preserves exactly 1/k
of the information. This was derived from codon degeneracy (biology).

CROSS-LEVEL PREDICTION: the same 1/k law should govern protein sequence
diversity at the structural level. At each position in a protein alignment,
the number of distinct amino acids observed (local "designability" d) should
predict the sequence entropy:

    H(position) / H_max  ≈  (d-1)/d  =  1 - 1/d

where H_max = log₂(d).

This experiment:
1. Downloads real cytochrome c protein sequences from NCBI (reusing the
   accession list from the codon entropy experiment)
2. Aligns them with Clustal Omega
3. At each aligned position, computes d (distinct amino acids) and H (Shannon entropy)
4. Tests whether H/H_max ≈ (d-1)/d across positions grouped by d

Outputs:
    paper/results_protein_designability.json
    paper/figures/protein_designability.pdf
"""

import sys
import json
import time
import subprocess
import tempfile
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
    percentile_ci,
    PAPER_DIR,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEED = 42
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


# ── Utility ──────────────────────────────────────────────────────────────────

def shannon_entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy H = -Σ p_i log₂(p_i) in bits."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def parse_fasta(filepath: str) -> dict:
    """Parse a FASTA file into {id: sequence} dict."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(current_seq)
    return seqs


# ── Download protein sequences ───────────────────────────────────────────────

def download_proteins() -> list:
    """
    Download cytochrome c protein sequences from NCBI using the saved
    accession list, or fall back to a fresh Entrez search.
    Returns list of (organism, protein_sequence) tuples.
    """
    acc_file = PAPER_DIR / "data" / "cytochrome_c_accessions.txt"

    # Read accession IDs
    accessions = []
    organisms_from_file = []
    if acc_file.exists():
        with open(acc_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    accessions.append(parts[0])
                    organisms_from_file.append(parts[1])
        print(f"[Download] Found {len(accessions)} accessions in {acc_file}")
    else:
        print(f"[Download] No accession file found at {acc_file}")

    try:
        from Bio import Entrez, SeqIO
        from Bio.Seq import Seq
    except ImportError:
        print("[Download] BioPython not installed; cannot download sequences.")
        return []

    Entrez.email = "drakecaraker@gmail.com"

    if accessions:
        # Download from saved accessions
        print(f"[Download] Fetching {len(accessions)} sequences from NCBI...")
        proteins = []
        batch_size = 50
        for start in range(0, len(accessions), batch_size):
            batch = accessions[start:start + batch_size]
            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=",".join(batch),
                    rettype="gb",
                    retmode="text",
                )
                for rec in SeqIO.parse(handle, "genbank"):
                    org = rec.annotations.get("organism", "unknown")
                    for feat in rec.features:
                        if feat.type == "CDS":
                            cds_seq = str(feat.extract(rec.seq)).upper()
                            if 270 <= len(cds_seq) <= 500 and len(cds_seq) % 3 == 0:
                                try:
                                    prot = str(Seq(cds_seq).translate(to_stop=True))
                                    if len(prot) >= 80:
                                        proteins.append((org, prot))
                                except Exception:
                                    pass
                                break
                handle.close()
                time.sleep(0.4)
            except Exception as e:
                print(f"[Download] Batch error: {e}")
                time.sleep(1.0)

        print(f"[Download] Got {len(proteins)} protein sequences.")
        return proteins
    else:
        # Fresh Entrez search
        print("[Download] Performing fresh NCBI search for CYCS...")
        try:
            handle = Entrez.esearch(
                db="nucleotide",
                term='CYCS[Gene] AND mRNA[Filter] AND refseq[Filter]',
                retmax=200,
            )
            search_record = Entrez.read(handle)
            handle.close()
            time.sleep(0.4)

            id_list = search_record.get("IdList", [])
            print(f"[Download] Found {len(id_list)} IDs.")

            proteins = []
            seen_orgs = set()
            batch_size = 50
            for start in range(0, min(len(id_list), 200), batch_size):
                batch = id_list[start:start + batch_size]
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    id=",".join(batch),
                    rettype="gb",
                    retmode="text",
                )
                for rec in SeqIO.parse(fetch_handle, "genbank"):
                    org = rec.annotations.get("organism", "unknown")
                    if org in seen_orgs:
                        continue
                    for feat in rec.features:
                        if feat.type == "CDS":
                            cds_seq = str(feat.extract(rec.seq)).upper()
                            if 270 <= len(cds_seq) <= 500 and len(cds_seq) % 3 == 0:
                                try:
                                    from Bio.Seq import Seq
                                    prot = str(Seq(cds_seq).translate(to_stop=True))
                                    if len(prot) >= 80:
                                        proteins.append((org, prot))
                                        seen_orgs.add(org)
                                except Exception:
                                    pass
                                break
                fetch_handle.close()
                time.sleep(0.4)

            print(f"[Download] Got {len(proteins)} protein sequences from fresh search.")
            return proteins
        except Exception as e:
            print(f"[Download] Entrez search failed: {e}")
            return []


# ── Align with Clustal Omega ─────────────────────────────────────────────────

def align_proteins(proteins: list) -> dict:
    """
    Align protein sequences with Clustal Omega.
    Input: list of (organism, sequence) tuples.
    Returns: dict of {id: aligned_sequence}.
    """
    tmp_in = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, prefix="designability_"
    )
    for i, (org, seq) in enumerate(proteins):
        org_clean = org.replace(" ", "_").replace("(", "").replace(")", "")
        tmp_in.write(f">sp{i}_{org_clean}\n{seq}\n")
    tmp_in.close()

    tmp_out_path = tmp_in.name.replace(".fasta", "_aligned.fasta")

    clustalo_bin = "/opt/homebrew/bin/clustalo"
    cmd = [
        clustalo_bin,
        "-i", tmp_in.name,
        "-o", tmp_out_path,
        "--threads=4",
        "--force",
    ]
    print(f"[Align] Running Clustal Omega: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if proc.returncode != 0 and "thread" in proc.stderr.lower():
        cmd = [c for c in cmd if not c.startswith("--threads")]
        print(f"[Align] Retrying without threads: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if proc.returncode != 0:
        print(f"[Align] Clustal Omega failed: {proc.stderr[:500]}")
        raise RuntimeError("Clustal Omega alignment failed")

    print("[Align] Alignment complete.")
    aligned = parse_fasta(tmp_out_path)

    # Cleanup
    import os
    try:
        os.unlink(tmp_in.name)
        os.unlink(tmp_out_path)
    except OSError:
        pass

    return aligned


# ── Per-position analysis ────────────────────────────────────────────────────

def analyze_positions(aligned_seqs: dict) -> dict:
    """
    For each aligned position, compute:
    - d: number of distinct standard amino acids (excluding gaps and X)
    - H: Shannon entropy of the amino acid distribution
    - H_max: log₂(d)
    - ratio: H / H_max
    - predicted: (d-1)/d

    Returns a dict with position-level data and grouped statistics.
    """
    seq_ids = list(aligned_seqs.keys())
    sequences = [aligned_seqs[sid] for sid in seq_ids]
    n_seqs = len(sequences)
    aln_len = len(sequences[0])

    print(f"\n[Analysis] Alignment: {n_seqs} sequences, {aln_len} columns")

    # Per-position computation
    positions = []  # list of dicts with d, H, H_max, ratio, predicted

    for col in range(aln_len):
        # Extract amino acids at this column (exclude gaps)
        residues = []
        for seq in sequences:
            if col < len(seq):
                aa = seq[col].upper()
                if aa in STANDARD_AMINO_ACIDS:
                    residues.append(aa)

        if len(residues) < n_seqs * 0.5:
            # Skip columns with >50% gaps
            continue

        # Count distinct amino acids
        aa_counts = Counter(residues)
        d = len(aa_counts)  # number of distinct AAs

        if d < 1:
            continue

        # Shannon entropy
        count_arr = np.array(list(aa_counts.values()), dtype=float)
        H = shannon_entropy_bits(count_arr)

        # Maximum entropy for d categories
        H_max = np.log2(d) if d > 1 else 0.0

        # Ratio and prediction
        if d == 1:
            ratio = 0.0  # perfectly conserved
            predicted = 0.0  # (1-1)/1 = 0
        else:
            ratio = H / H_max if H_max > 0 else 0.0
            predicted = (d - 1) / d

        positions.append({
            "col": col,
            "d": d,
            "H": H,
            "H_max": H_max,
            "ratio": ratio,
            "predicted": predicted,
            "n_residues": len(residues),
        })

    print(f"[Analysis] Analyzed {len(positions)} non-gap columns.")

    # Group by d
    d_values = sorted(set(p["d"] for p in positions))
    grouped = {}
    for d in d_values:
        pos_at_d = [p for p in positions if p["d"] == d]
        ratios = [p["ratio"] for p in pos_at_d]
        if d == 1:
            # All ratios are 0/0 → 0 by convention; skip for statistical analysis
            grouped[d] = {
                "n_positions": len(pos_at_d),
                "mean_ratio": 0.0,
                "std_ratio": 0.0,
                "ci_lo": 0.0,
                "ci_hi": 0.0,
                "predicted": 0.0,
                "residual": 0.0,
            }
        else:
            ci_lo, mean_r, ci_hi = percentile_ci(np.array(ratios))
            grouped[d] = {
                "n_positions": len(pos_at_d),
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "predicted": (d - 1) / d,
                "residual": float(np.mean(ratios)) - (d - 1) / d,
            }

    # Print summary
    print("\n=== Position-Level Designability Analysis ===")
    print(f"{'d':>3}  {'N_pos':>6}  {'Mean H/Hmax':>11}  {'Predicted':>9}  {'Residual':>8}")
    print("-" * 50)
    for d in d_values:
        g = grouped[d]
        print(f"{d:>3}  {g['n_positions']:>6}  {g['mean_ratio']:>11.4f}  "
              f"{g['predicted']:>9.4f}  {g['residual']:>8.4f}")

    return {
        "positions": positions,
        "grouped": grouped,
        "d_values": d_values,
        "n_seqs": n_seqs,
        "aln_len": aln_len,
        "n_analyzed": len(positions),
    }


# ── Statistical tests ────────────────────────────────────────────────────────

def compute_statistics(analysis: dict) -> dict:
    """
    Statistical tests comparing observed H/H_max to the 1-1/d prediction.
    """
    grouped = analysis["grouped"]
    d_values = [d for d in analysis["d_values"] if d >= 2]  # exclude d=1

    if len(d_values) < 3:
        return {"error": "Too few d values for statistical analysis"}

    observed = np.array([grouped[d]["mean_ratio"] for d in d_values])
    predicted_char = np.array([(d - 1) / d for d in d_values])
    predicted_uniform = np.ones(len(d_values))  # H/H_max = 1 (uniform dist)

    # Correlation: observed vs character-theory prediction
    r_char, p_char = stats.pearsonr(predicted_char, observed)
    rho_char, rho_p_char = stats.spearmanr(predicted_char, observed)

    # RMSE: character theory vs uniform
    rmse_char = float(np.sqrt(np.mean((observed - predicted_char) ** 2)))
    rmse_uniform = float(np.sqrt(np.mean((observed - predicted_uniform) ** 2)))

    # MAE
    mae_char = float(np.mean(np.abs(observed - predicted_char)))
    mae_uniform = float(np.mean(np.abs(observed - predicted_uniform)))

    # Mean signed residual (bias)
    mean_residual = float(np.mean(observed - predicted_char))

    # Per-position test: all positions with d >= 2
    all_ratios = []
    all_predicted = []
    for p in analysis["positions"]:
        if p["d"] >= 2:
            all_ratios.append(p["ratio"])
            all_predicted.append(p["predicted"])
    all_ratios = np.array(all_ratios)
    all_predicted = np.array(all_predicted)

    r_pos, p_pos = stats.pearsonr(all_predicted, all_ratios)

    # One-sample t-test: is mean residual significantly different from 0?
    residuals = all_ratios - all_predicted
    t_stat, t_pval = stats.ttest_1samp(residuals, 0.0)

    # Distance from uniform vs character theory for each position
    dist_to_char = np.abs(all_ratios - all_predicted)
    dist_to_uniform = np.abs(all_ratios - 1.0)
    closer_to_char = int(np.sum(dist_to_char < dist_to_uniform))
    closer_to_uniform = int(np.sum(dist_to_char > dist_to_uniform))

    stats_result = {
        "n_d_values": len(d_values),
        "d_values": [int(d) for d in d_values],
        "pearson_r_grouped": float(r_char),
        "pearson_p_grouped": float(p_char),
        "spearman_rho_grouped": float(rho_char),
        "spearman_p_grouped": float(rho_p_char),
        "rmse_character_theory": rmse_char,
        "rmse_uniform": rmse_uniform,
        "mae_character_theory": mae_char,
        "mae_uniform": mae_uniform,
        "rmse_ratio_uniform_over_char": rmse_uniform / rmse_char if rmse_char > 0 else float("inf"),
        "mean_signed_residual": mean_residual,
        "ttest_residual_t": float(t_stat),
        "ttest_residual_p": float(t_pval),
        "n_positions_d_ge_2": len(all_ratios),
        "pearson_r_positions": float(r_pos),
        "pearson_p_positions": float(p_pos),
        "positions_closer_to_char": closer_to_char,
        "positions_closer_to_uniform": closer_to_uniform,
        "fraction_closer_to_char": closer_to_char / (closer_to_char + closer_to_uniform)
            if (closer_to_char + closer_to_uniform) > 0 else 0.0,
    }

    print("\n=== Statistical Summary ===")
    print(f"  Pearson r (grouped, obs vs 1-1/d): {r_char:.4f}  (p={p_char:.2e})")
    print(f"  Spearman rho (grouped):            {rho_char:.4f}  (p={rho_p_char:.2e})")
    print(f"  RMSE (char theory): {rmse_char:.4f}")
    print(f"  RMSE (uniform):     {rmse_uniform:.4f}")
    print(f"  RMSE ratio (uniform/char): {stats_result['rmse_ratio_uniform_over_char']:.2f}x")
    print(f"  MAE  (char theory): {mae_char:.4f}")
    print(f"  MAE  (uniform):     {mae_uniform:.4f}")
    print(f"  Mean signed residual (obs - pred): {mean_residual:.4f}")
    print(f"  t-test on residuals: t={t_stat:.2f}, p={t_pval:.2e}")
    print(f"  Per-position Pearson r: {r_pos:.4f}  (p={p_pos:.2e})")
    print(f"  Positions closer to char theory: {closer_to_char}/{closer_to_char + closer_to_uniform} "
          f"({100*stats_result['fraction_closer_to_char']:.1f}%)")

    return stats_result


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(analysis: dict, stats_result: dict):
    """
    Create the protein designability figure with two panels:
    (a) Mean H/H_max vs d with theory overlay
    (b) Per-position scatter
    """
    set_all_seeds(SEED)
    load_publication_style()

    grouped = analysis["grouped"]
    d_values = sorted(analysis["d_values"])
    d_nontrivial = [d for d in d_values if d >= 2]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # ── Panel (a): Grouped mean H/H_max vs d ──
    ax = axes[0]

    # Theory curve
    d_theory = np.linspace(1, max(d_nontrivial) + 1, 200)
    ax.plot(d_theory, 1.0 - 1.0 / d_theory,
            color="#D55E00", ls="--", lw=1.2, label=r"$1 - 1/d$ (character theory)")
    ax.axhline(1.0, color="#009E73", ls=":", lw=0.8, alpha=0.7, label=r"Uniform ($H/H_{\max}=1$)")

    # Data points with error bars
    x_data = []
    y_data = []
    y_lo = []
    y_hi = []
    for d in d_nontrivial:
        g = grouped[d]
        if g["n_positions"] >= 2:
            x_data.append(d)
            y_data.append(g["mean_ratio"])
            y_lo.append(g["mean_ratio"] - g["ci_lo"])
            y_hi.append(g["ci_hi"] - g["mean_ratio"])

    ax.errorbar(x_data, y_data, yerr=[y_lo, y_hi],
                fmt="o", color="#0072B2", ms=4, capsize=2, lw=0.8,
                label="Observed (mean)")

    ax.set_xlabel(r"$d$ (distinct amino acids)")
    ax.set_ylabel(r"$H / H_{\max}$")
    ax.set_title(r"\textbf{(a)} Grouped by designability $d$")
    ax.set_xlim(1, max(d_nontrivial) + 1)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=6.5, loc="lower right")

    # Annotate correlation
    r_val = stats_result.get("pearson_r_grouped", 0)
    p_val = stats_result.get("pearson_p_grouped", 1)
    rmse_c = stats_result.get("rmse_character_theory", 0)
    rmse_u = stats_result.get("rmse_uniform", 0)
    ax.text(0.05, 0.92, f"$r = {r_val:.3f}$\nRMSE$_{{1/k}} = {rmse_c:.3f}$\nRMSE$_{{\\mathrm{{unif}}}} = {rmse_u:.3f}$",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.7"))

    # ── Panel (b): Per-position scatter ──
    ax2 = axes[1]

    pos_d = np.array([p["d"] for p in analysis["positions"] if p["d"] >= 2])
    pos_ratio = np.array([p["ratio"] for p in analysis["positions"] if p["d"] >= 2])
    pos_predicted = np.array([p["predicted"] for p in analysis["positions"] if p["d"] >= 2])

    # Jitter d for visibility
    rng = np.random.RandomState(SEED)
    jitter = rng.uniform(-0.25, 0.25, size=len(pos_d))

    ax2.scatter(pos_d + jitter, pos_ratio, s=6, alpha=0.35, color="#0072B2",
                edgecolors="none", label="Individual positions")

    # Theory curve
    ax2.plot(d_theory, 1.0 - 1.0 / d_theory,
             color="#D55E00", ls="--", lw=1.2, label=r"$1 - 1/d$")
    ax2.axhline(1.0, color="#009E73", ls=":", lw=0.8, alpha=0.7)

    ax2.set_xlabel(r"$d$ (distinct amino acids)")
    ax2.set_ylabel(r"$H / H_{\max}$")
    ax2.set_title(r"\textbf{(b)} Per-position")
    ax2.set_xlim(1, max(d_nontrivial) + 1)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=6.5, loc="lower right")

    # Annotate
    r_pos = stats_result.get("pearson_r_positions", 0)
    frac_closer = stats_result.get("fraction_closer_to_char", 0)
    ax2.text(0.05, 0.92,
             f"$r = {r_pos:.3f}$ (per-position)\n"
             f"{100*frac_closer:.0f}\\% closer to $1\\!-\\!1/d$",
             transform=ax2.transAxes, fontsize=6, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.7"))

    fig.tight_layout()
    save_figure(fig, "protein_designability")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)

    print("=" * 70)
    print("PROTEIN DESIGNABILITY: Testing the 1/k Law at the Sequence Level")
    print("=" * 70)

    # Step 1: Download protein sequences
    print("\n--- Step 1: Download protein sequences ---")
    proteins = download_proteins()

    if len(proteins) < 15:
        print(f"\n[FATAL] Only {len(proteins)} proteins downloaded; need >=15.")
        print("Cannot proceed with experiment.")
        sys.exit(1)

    # Deduplicate by organism
    seen = set()
    unique_proteins = []
    for org, seq in proteins:
        if org not in seen:
            unique_proteins.append((org, seq))
            seen.add(org)
    proteins = unique_proteins
    print(f"[Download] {len(proteins)} unique species after deduplication.")

    # Step 2: Align
    print("\n--- Step 2: Align with Clustal Omega ---")
    aligned = align_proteins(proteins)
    print(f"[Align] {len(aligned)} aligned sequences.")

    # Step 3: Analyze positions
    print("\n--- Step 3: Per-position designability analysis ---")
    analysis = analyze_positions(aligned)

    # Step 4: Statistical tests
    print("\n--- Step 4: Statistical tests ---")
    stats_result = compute_statistics(analysis)

    # Step 5: Plot
    print("\n--- Step 5: Generate figure ---")
    make_figure(analysis, stats_result)

    # Step 6: Save results
    print("\n--- Step 6: Save results ---")

    # Prepare serializable grouped data
    grouped_serializable = {}
    for d, g in analysis["grouped"].items():
        grouped_serializable[str(d)] = g

    results = {
        "experiment": "protein_designability",
        "description": (
            "Tests the cross-level prediction from the universal explanation "
            "impossibility framework's character theory: H/H_max ≈ (d-1)/d = 1-1/d "
            "for protein sequence diversity at each aligned position."
        ),
        "data_source": "Cytochrome c protein sequences from NCBI RefSeq",
        "n_species": len(proteins),
        "n_aligned_columns": analysis["aln_len"],
        "n_analyzed_positions": analysis["n_analyzed"],
        "grouped_by_d": grouped_serializable,
        "statistics": stats_result,
        "prediction": "H/H_max = (d-1)/d = 1 - 1/d (character theory)",
        "null_prediction": "H/H_max = 1.0 (uniform distribution)",
        "conclusion": "",
    }

    # Determine conclusion
    rmse_char = stats_result.get("rmse_character_theory", 999)
    rmse_unif = stats_result.get("rmse_uniform", 999)
    r_grouped = stats_result.get("pearson_r_grouped", 0)
    frac_closer = stats_result.get("fraction_closer_to_char", 0)

    if rmse_char < rmse_unif:
        results["conclusion"] = (
            f"The 1/k law HOLDS: observed H/H_max is closer to (d-1)/d than to 1.0 "
            f"(RMSE {rmse_char:.4f} vs {rmse_unif:.4f}; "
            f"Pearson r = {r_grouped:.3f}; "
            f"{100*frac_closer:.0f}% of positions closer to character theory)."
        )
    else:
        results["conclusion"] = (
            f"The 1/k law does NOT hold: observed H/H_max is closer to 1.0 (uniform) "
            f"than to (d-1)/d "
            f"(RMSE char={rmse_char:.4f} vs uniform={rmse_unif:.4f}; "
            f"Pearson r = {r_grouped:.3f})."
        )

    save_results(results, "protein_designability")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print(results["conclusion"])
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
