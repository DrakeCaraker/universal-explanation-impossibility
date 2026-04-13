"""
DMS Prediction Experiment: Attenuation Factor alpha vs Evolutionary Rate
=========================================================================
Tests whether the 1/k framework's attenuation parameter alpha correlates
with known evolutionary rates across diverse proteins.

For each protein:
  1. Download ortholog protein sequences from NCBI Entrez
  2. Align with Clustal Omega
  3. At each variable position (d >= 2): compute H/H_max and (d-1)/d ceiling
  4. Fit alpha = slope of H/H_max vs (d-1)/d via OLS through origin
  5. alpha measures how close to the neutral (1/k) ceiling the protein evolves

Prediction: alpha should correlate with known evolutionary rates.
Fast-evolving proteins should have alpha closer to 1 (neutral ceiling).
Slow-evolving proteins should have alpha closer to 0 (strong selection).

Outputs:
  paper/results_attenuation_factor.json
  paper/figures/attenuation_factor.pdf
"""

import sys
import json
import time
import tempfile
import subprocess
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────

SEED = 42
CACHE_DIR = PAPER_DIR / "cache_attenuation"
CACHE_DIR.mkdir(exist_ok=True)

CLUSTALO = "/opt/homebrew/bin/clustalo"

MIN_SPECIES = 20
MAX_SEQUENCES = 200
MIN_VARIABLE_POSITIONS = 15
MIN_COLUMN_OCCUPANCY = 0.80

# Proteins spanning a wide range of evolutionary rates.
# Format: (display_name, gene_symbol, alt_symbols, evolutionary_rate_PAM)
# Evolutionary rates: amino acid substitutions per site per 10^8 years (PAMs/100My)
# Sources: Dayhoff (1978), Dickerson (1971), Wilson et al. (1977), Graur & Li (2000)
PROTEINS = [
    # Very slow evolving
    ("Histone H4",          "HIST1H4A", ["H4C1", "H4-16"],   0.01),
    ("Ubiquitin",           "UBB",      ["UBC"],              0.02),
    ("Cytochrome c",        "CYCS",     ["CYC"],              0.3),
    ("Actin beta",          "ACTB",     [],                   0.3),
    # Slow-moderate
    ("Hemoglobin alpha",    "HBA1",     ["HBA"],              0.8),
    ("Hemoglobin beta",     "HBB",      [],                   0.8),
    ("Myoglobin",           "MB",       [],                   0.9),
    ("SOD1",                "SOD1",     [],                   1.0),
    # Moderate
    ("Insulin",             "INS",      [],                   1.2),
    ("Lysozyme",            "LYZ",      [],                   2.0),
    ("Growth hormone",      "GH1",      ["GH"],               2.6),
    # Fast
    ("Alpha-lactalbumin",   "LALBA",    [],                   2.7),
    ("Interleukin 2",       "IL2",      [],                   3.5),
    ("Kappa casein",        "CSN3",     [],                   3.7),
    ("Relaxin",             "RLN2",     ["RLN1"],             4.0),
    ("Fibrinopeptide A",    "FGA",      [],                   8.3),
]


# ── Entrez download ─────────────────────────────────────────────────────────

def download_sequences(gene_symbol, alt_symbols, max_seqs=MAX_SEQUENCES):
    """Download protein sequences from NCBI Entrez. Returns list of (organism, sequence)."""
    cache_file = CACHE_DIR / f"entrez_{gene_symbol}.json"
    if cache_file.exists():
        print(f"  [Cache] Loading cached sequences for {gene_symbol}")
        with open(cache_file) as f:
            return json.load(f)

    try:
        from Bio import Entrez, SeqIO
    except ImportError:
        print("  [ERROR] BioPython not installed")
        return []

    Entrez.email = "drakecaraker@gmail.com"

    symbols_to_try = [gene_symbol] + alt_symbols
    all_sequences = []

    for sym in symbols_to_try:
        if len(all_sequences) >= MIN_SPECIES:
            break

        query = f'{sym}[Gene] AND mRNA[Filter] AND refseq[Filter]'
        print(f"  [Entrez] Searching: {query}")

        try:
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_seqs)
            search_record = Entrez.read(handle)
            handle.close()
            time.sleep(0.4)

            id_list = search_record.get("IdList", [])
            n_found = int(search_record.get("Count", 0))
            print(f"  [Entrez] Found {n_found} records, retrieving {len(id_list)}")

            if not id_list:
                continue

            for batch_start in range(0, len(id_list), 50):
                batch_ids = id_list[batch_start:batch_start + 50]
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=",".join(batch_ids),
                    rettype="gb",
                    retmode="text",
                )
                records = list(SeqIO.parse(handle, "genbank"))
                handle.close()
                time.sleep(0.4)

                for record in records:
                    organism = record.annotations.get("organism", "Unknown")
                    for feature in record.features:
                        if feature.type == "CDS":
                            translation = feature.qualifiers.get("translation", [None])[0]
                            if translation and len(translation) > 10:
                                all_sequences.append((organism, translation))
                                break

        except Exception as e:
            print(f"  [Entrez] Error with {sym}: {e}")
            time.sleep(1)

    # Deduplicate by organism
    seen_orgs = set()
    deduped = []
    for org, seq in all_sequences:
        org_key = org.lower().strip()
        if org_key not in seen_orgs:
            seen_orgs.add(org_key)
            deduped.append((org, seq))

    print(f"  [Entrez] {len(deduped)} unique species for {gene_symbol}")

    with open(cache_file, 'w') as f:
        json.dump(deduped, f)

    return deduped


# ── Alignment ────────────────────────────────────────────────────────────────

def align_sequences(sequences, protein_name):
    """Align protein sequences with Clustal Omega. Returns list of aligned sequences."""
    safe_name = protein_name.replace(' ', '_').replace('-', '_')
    cache_file = CACHE_DIR / f"aligned_{safe_name}.json"
    if cache_file.exists():
        print(f"  [Cache] Loading cached alignment for {protein_name}")
        with open(cache_file) as f:
            return json.load(f)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        input_path = f.name
        for i, (org, seq) in enumerate(sequences):
            safe_org = org.replace(' ', '_').replace('/', '_')[:40]
            f.write(f">{safe_org}_{i}\n{seq}\n")

    output_path = input_path + ".aligned"

    try:
        cmd = [CLUSTALO, "-i", input_path, "-o", output_path, "--force", "--threads=4"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            cmd = [CLUSTALO, "-i", input_path, "-o", output_path, "--force"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  [ERROR] Clustal Omega failed: {result.stderr[:200]}")
                return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [ERROR] Clustal Omega error: {e}")
        return []

    aligned = []
    from Bio import SeqIO as SIO
    for record in SIO.parse(output_path, "fasta"):
        aligned.append(str(record.seq))

    print(f"  [Align] {len(aligned)} sequences, alignment length {len(aligned[0]) if aligned else 0}")

    with open(cache_file, 'w') as f:
        json.dump(aligned, f)

    Path(input_path).unlink(missing_ok=True)
    Path(output_path).unlink(missing_ok=True)

    return aligned


# ── Per-position analysis ────────────────────────────────────────────────────

def shannon_entropy(counts_dict):
    """Shannon entropy in bits from a counts dict."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts_dict.values() if c > 0])
    return float(-np.sum(probs * np.log2(probs)))


def analyze_alignment(aligned_seqs):
    """
    Analyze each column of the alignment.
    Returns list of dicts for variable positions (d >= 2) with:
      - d: number of distinct amino acids (excluding gaps)
      - H: Shannon entropy (bits)
      - H_max: log2(d)
      - H_ratio: H / H_max
      - ceiling: (d-1)/d
      - deviation: ceiling - H_ratio
      - conservation: 1 - H/log2(20)
      - occupancy: fraction non-gap
    """
    if not aligned_seqs:
        return []

    aln_len = len(aligned_seqs[0])
    n_seqs = len(aligned_seqs)
    positions = []

    for col in range(aln_len):
        residues = [aligned_seqs[s][col] for s in range(n_seqs)]
        non_gap = [r for r in residues if r not in ('-', '.', 'X', '*')]
        occupancy = len(non_gap) / n_seqs

        if occupancy < MIN_COLUMN_OCCUPANCY:
            continue

        counts = Counter(non_gap)
        d = len(counts)

        if d < 2:
            continue

        H = shannon_entropy(counts)
        H_max = np.log2(d)
        H_ratio = H / H_max if H_max > 0 else 0.0
        ceiling = (d - 1) / d
        deviation = ceiling - H_ratio
        conservation = 1.0 - H / np.log2(20)

        positions.append({
            'col': col,
            'd': d,
            'H': H,
            'H_max': H_max,
            'H_ratio': H_ratio,
            'ceiling': ceiling,
            'deviation': deviation,
            'conservation': conservation,
            'occupancy': occupancy,
        })

    return positions


def fit_attenuation_factor(positions):
    """
    Fit alpha: H/H_max = alpha * (d-1)/d via OLS through origin.
    Returns alpha, r_squared, n_positions.
    """
    if len(positions) < MIN_VARIABLE_POSITIONS:
        return None, None, len(positions)

    x = np.array([p['ceiling'] for p in positions])  # (d-1)/d
    y = np.array([p['H_ratio'] for p in positions])   # H/H_max

    # OLS through origin: alpha = sum(x*y) / sum(x^2)
    alpha = float(np.sum(x * y) / np.sum(x ** 2))

    # R^2 for regression through origin
    y_pred = alpha * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(y ** 2)  # for regression through origin, SS_tot = sum(y^2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return alpha, r_squared, len(positions)


# ── Main experiment ──────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)

    print("=" * 70)
    print("DMS Prediction Experiment: Attenuation Factor vs Evolutionary Rate")
    print("=" * 70)

    results_by_protein = {}
    successful_proteins = []

    for name, gene, alts, evo_rate in PROTEINS:
        print(f"\n{'─' * 60}")
        print(f"Processing: {name} ({gene}), known evolutionary rate = {evo_rate}")
        print(f"{'─' * 60}")

        # Step 1: Download
        sequences = download_sequences(gene, alts)
        if len(sequences) < MIN_SPECIES:
            print(f"  [SKIP] Only {len(sequences)} species (need {MIN_SPECIES})")
            results_by_protein[name] = {
                'gene': gene,
                'n_species': len(sequences),
                'status': 'insufficient_sequences',
                'evolutionary_rate': evo_rate,
            }
            continue

        # Step 2: Align
        aligned = align_sequences(sequences, name)
        if not aligned:
            print(f"  [SKIP] Alignment failed for {name}")
            results_by_protein[name] = {
                'gene': gene,
                'n_species': len(sequences),
                'status': 'alignment_failed',
                'evolutionary_rate': evo_rate,
            }
            continue

        # Step 3: Analyze positions
        positions = analyze_alignment(aligned)
        print(f"  [Analysis] {len(positions)} variable positions (d >= 2)")

        # Step 4: Fit alpha
        alpha, r_sq, n_pos = fit_attenuation_factor(positions)

        if alpha is None:
            print(f"  [SKIP] Only {n_pos} variable positions (need {MIN_VARIABLE_POSITIONS})")
            results_by_protein[name] = {
                'gene': gene,
                'n_species': len(sequences),
                'n_variable_positions': n_pos,
                'status': 'insufficient_variable_positions',
                'evolutionary_rate': evo_rate,
            }
            continue

        print(f"  [Result] alpha = {alpha:.4f}, R^2 = {r_sq:.4f}, n_positions = {n_pos}")

        # Distribution of d values
        d_counts = Counter(p['d'] for p in positions)

        results_by_protein[name] = {
            'gene': gene,
            'n_species': len(sequences),
            'alignment_length': len(aligned[0]),
            'n_variable_positions': n_pos,
            'alpha': alpha,
            'r_squared': r_sq,
            'evolutionary_rate': evo_rate,
            'mean_H_ratio': float(np.mean([p['H_ratio'] for p in positions])),
            'mean_ceiling': float(np.mean([p['ceiling'] for p in positions])),
            'mean_deviation': float(np.mean([p['deviation'] for p in positions])),
            'd_distribution': {str(k): v for k, v in sorted(d_counts.items())},
            'status': 'success',
        }

        successful_proteins.append({
            'name': name,
            'alpha': alpha,
            'r_squared': r_sq,
            'evolutionary_rate': evo_rate,
            'n_positions': n_pos,
            'n_species': len(sequences),
            'positions': positions,  # keep for plotting
        })

    # ── Cross-protein correlation ────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print(f"Cross-protein correlation: {len(successful_proteins)} proteins")
    print(f"{'=' * 70}")

    if len(successful_proteins) < 4:
        print("[FAIL] Too few successful proteins for meaningful correlation.")
        save_results({
            'experiment': 'attenuation_factor',
            'n_proteins_attempted': len(PROTEINS),
            'n_proteins_successful': len(successful_proteins),
            'results_by_protein': results_by_protein,
            'correlation': None,
            'status': 'insufficient_proteins',
        }, 'attenuation_factor')
        return

    alphas = np.array([p['alpha'] for p in successful_proteins])
    evo_rates = np.array([p['evolutionary_rate'] for p in successful_proteins])
    names = [p['name'] for p in successful_proteins]

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(evo_rates, alphas)
    # Spearman (more robust to outliers)
    r_spearman, p_spearman = stats.spearmanr(evo_rates, alphas)
    # Log-space correlation (evolutionary rates span orders of magnitude)
    log_rates = np.log10(evo_rates)
    r_log, p_log = stats.pearsonr(log_rates, alphas)

    print(f"\nPearson:  r = {r_pearson:.4f}, p = {p_pearson:.4e}")
    print(f"Spearman: r = {r_spearman:.4f}, p = {p_spearman:.4e}")
    print(f"Log-rate Pearson: r = {r_log:.4f}, p = {p_log:.4e}")

    for p in successful_proteins:
        print(f"  {p['name']:25s}  alpha={p['alpha']:.4f}  rate={p['evolutionary_rate']:.2f}  "
              f"R2={p['r_squared']:.4f}  n_pos={p['n_positions']}")

    # ── Beyond conservation test ─────────────────────────────────────────────
    # For each protein, compare alpha (1/k-based) with raw mean conservation
    # as predictors of evolutionary rate.
    mean_conservations = []
    for prot_data in successful_proteins:
        positions = prot_data['positions']
        mean_cons = np.mean([p['conservation'] for p in positions])
        mean_conservations.append(mean_cons)

    mean_conservations = np.array(mean_conservations)
    r_cons, p_cons = stats.pearsonr(evo_rates, mean_conservations)
    r_cons_sp, p_cons_sp = stats.spearmanr(evo_rates, mean_conservations)

    print(f"\nConservation baseline:")
    print(f"  Pearson (rate vs mean_conservation):  r = {r_cons:.4f}, p = {p_cons:.4e}")
    print(f"  Spearman (rate vs mean_conservation): r = {r_cons_sp:.4f}, p = {p_cons_sp:.4e}")

    # ── Partial correlation: alpha | conservation ────────────────────────────
    # Does alpha add info beyond conservation?
    from numpy.linalg import lstsq
    # Residualize alpha on conservation
    X_cons = np.column_stack([mean_conservations, np.ones(len(mean_conservations))])
    beta_alpha, _, _, _ = lstsq(X_cons, alphas, rcond=None)
    alpha_resid = alphas - X_cons @ beta_alpha
    beta_rate, _, _, _ = lstsq(X_cons, evo_rates, rcond=None)
    rate_resid = evo_rates - X_cons @ beta_rate

    if np.std(alpha_resid) > 1e-10 and np.std(rate_resid) > 1e-10:
        r_partial, p_partial = stats.pearsonr(alpha_resid, rate_resid)
        print(f"\nPartial correlation (alpha vs rate | conservation):")
        print(f"  r = {r_partial:.4f}, p = {p_partial:.4e}")
    else:
        r_partial, p_partial = 0.0, 1.0
        print(f"\nPartial correlation: degenerate (zero variance in residuals)")

    # ── Figure ───────────────────────────────────────────────────────────────

    load_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: alpha vs evolutionary rate
    ax = axes[0]
    ax.scatter(evo_rates, alphas, s=60, c='#2c7bb6', edgecolors='black',
               linewidths=0.5, zorder=5)
    for i, nm in enumerate(names):
        short = nm.split()[0] if len(nm) > 12 else nm
        ax.annotate(short, (evo_rates[i], alphas[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Fit line
    slope, intercept = np.polyfit(evo_rates, alphas, 1)
    x_fit = np.linspace(0, max(evo_rates) * 1.1, 100)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color='red', alpha=0.6,
            label=f'r = {r_pearson:.3f} (p = {p_pearson:.3e})')
    ax.set_xlabel('Evolutionary rate (PAM/100 My)')
    ax.set_ylabel(r'Attenuation factor $\alpha$')
    ax.set_title(r'$\alpha$ vs evolutionary rate')
    ax.legend(fontsize=8, loc='lower right')

    # Panel B: H/H_max vs (d-1)/d for fastest and slowest proteins
    ax = axes[1]
    sorted_by_rate = sorted(successful_proteins, key=lambda p: p['evolutionary_rate'])
    slowest = sorted_by_rate[0]
    fastest = sorted_by_rate[-1]

    for prot, color, marker in [(slowest, '#2c7bb6', 'o'), (fastest, '#d7191c', 's')]:
        pos = prot['positions']
        x = np.array([p['ceiling'] for p in pos])
        y = np.array([p['H_ratio'] for p in pos])
        ax.scatter(x, y, s=15, c=color, marker=marker, alpha=0.4,
                   label=r"{} ($\alpha$={:.3f})".format(prot['name'], prot['alpha']))
        # Fit line through origin
        a = np.sum(x * y) / np.sum(x ** 2)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, a * x_line, '--', color=color, alpha=0.7)

    # Neutral ceiling
    ax.plot(x_line, x_line, 'k-', alpha=0.3, label=r'Neutral ceiling ($\alpha$=1)')
    ax.set_xlabel(r'$(d-1)/d$ (1/k ceiling)')
    ax.set_ylabel(r'$H/H_{\max}$')
    ax.set_title('Slow vs fast protein')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Panel C: alpha vs conservation (showing they're related but not identical)
    ax = axes[2]
    ax.scatter(mean_conservations, alphas, s=60, c='#2c7bb6', edgecolors='black',
               linewidths=0.5, zorder=5)
    for i, nm in enumerate(names):
        short = nm.split()[0] if len(nm) > 12 else nm
        ax.annotate(short, (mean_conservations[i], alphas[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel('Mean conservation (1 - H/log2(20))')
    ax.set_ylabel(r'Attenuation factor $\alpha$')
    ax.set_title(r'$\alpha$ vs conservation')

    # Add partial correlation annotation
    ax.annotate('Partial r($\\alpha$, rate | cons) = {:.3f}\np = {:.3e}'.format(r_partial, p_partial),
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_figure(fig, 'attenuation_factor')

    # ── Save results ─────────────────────────────────────────────────────────

    # Remove position-level data from results (too large for JSON)
    for prot in successful_proteins:
        del prot['positions']

    correlation_results = {
        'experiment': 'attenuation_factor',
        'description': (
            'Tests whether the 1/k framework attenuation factor alpha '
            'correlates with known evolutionary rates across diverse proteins. '
            'alpha = slope of H/H_max vs (d-1)/d at variable positions.'
        ),
        'n_proteins_attempted': len(PROTEINS),
        'n_proteins_successful': len(successful_proteins),
        'results_by_protein': results_by_protein,
        'successful_proteins': [
            {k: v for k, v in p.items()}
            for p in successful_proteins
        ],
        'correlation': {
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman),
            'log_rate_pearson_r': float(r_log),
            'log_rate_pearson_p': float(p_log),
        },
        'conservation_baseline': {
            'pearson_r': float(r_cons),
            'pearson_p': float(p_cons),
            'spearman_r': float(r_cons_sp),
            'spearman_p': float(p_cons_sp),
        },
        'partial_correlation': {
            'alpha_vs_rate_given_conservation_r': float(r_partial),
            'alpha_vs_rate_given_conservation_p': float(p_partial),
        },
        'interpretation': (
            'alpha measures the fraction of the neutral 1/k ceiling realized. '
            'If alpha correlates with evolutionary rate, the 1/k framework '
            'provides a principled single-parameter model of protein evolution '
            'calibrated by character theory.'
        ),
    }

    # Remove position-level data from results_by_protein too
    for k, v in correlation_results['results_by_protein'].items():
        if 'positions' in v:
            del v['positions']

    save_results(correlation_results, 'attenuation_factor')

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Proteins analyzed: {len(successful_proteins)}/{len(PROTEINS)}")
    print(f"Pearson r(alpha, evo_rate) = {r_pearson:.4f}, p = {p_pearson:.4e}")
    print(f"Spearman r(alpha, evo_rate) = {r_spearman:.4f}, p = {p_spearman:.4e}")
    print(f"Partial r(alpha, rate | conservation) = {r_partial:.4f}, p = {p_partial:.4e}")
    print(f"Conservation baseline Spearman = {r_cons_sp:.4f}")
    print(f"Does alpha correlate with evolutionary rate? "
          f"{'YES' if p_spearman < 0.05 else 'NO'} (p = {p_spearman:.4e})")


if __name__ == "__main__":
    main()
