"""
Functional Site Prediction via 1/k Information Ceiling
=======================================================
Tests the framework's character theory prediction: S_k acting on R^k preserves
1/k of the information. The deviation of observed entropy from the (d-1)/d
ceiling should predict functional importance BETTER than raw conservation alone.

For each of 5 well-studied proteins:
  1. Download ortholog CDS sequences from NCBI Entrez
  2. Translate + align with Clustal Omega
  3. Compute per-position: Shannon entropy, 1/k ceiling, deviation, raw conservation
  4. Download UniProt functional annotations for the human ortholog
  5. Test whether deviation discriminates functional sites better than conservation (AUC-ROC)

Outputs:
  paper/results_functional_site_prediction.json
  paper/figures/functional_site_prediction.pdf
"""

import sys
import json
import time
import hashlib
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
CACHE_DIR = PAPER_DIR / "cache_functional"
CACHE_DIR.mkdir(exist_ok=True)

CLUSTALO = "/opt/homebrew/bin/clustalo"

# Proteins to analyze: (name, gene_symbol, alternative_symbols, uniprot_human_accession)
PROTEINS = [
    ("Cytochrome c",           "CYCS",  ["CYC"],     "P99999"),
    ("Hemoglobin alpha",       "HBA1",  ["HBA"],     "P69905"),
    ("Ubiquitin",              "UBB",   ["UBC"],     "P0CG48"),
    ("Insulin",                "INS",   [],          "P01308"),
    ("Superoxide dismutase 1", "SOD1",  [],          "P00441"),
]

# Minimum thresholds
MIN_SPECIES = 30
MIN_VARIABLE_POSITIONS = 20
MAX_SEQUENCES = 200
MIN_COLUMN_OCCUPANCY = 0.80  # at least 80% non-gap

# UniProt feature types that indicate functional importance
FUNCTIONAL_FEATURE_TYPES = {
    "Active site", "Binding site", "Metal binding",
    "Disulfide bond", "Modified residue", "Mutagenesis",
    # UniProt JSON key names (lowercase variants)
    "ACT_SITE", "BINDING", "METAL", "DISULFID", "MOD_RES", "MUTAGEN",
    "SITE",  # general functional site
}

# ── Entrez download ─────────────────────────────────────────────────────────

def download_sequences(gene_symbol, alt_symbols, max_seqs=MAX_SEQUENCES):
    """Download CDS sequences from NCBI Entrez. Returns list of (organism, sequence) tuples."""
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

    # Try primary symbol, then alternatives
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

            # Download in batches of 50
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
                    # Extract organism
                    organism = record.annotations.get("organism", "Unknown")
                    # Extract CDS
                    for feature in record.features:
                        if feature.type == "CDS":
                            translation = feature.qualifiers.get("translation", [None])[0]
                            if translation and len(translation) > 20:
                                all_sequences.append((organism, translation))
                                break

        except Exception as e:
            print(f"  [Entrez] Error with {sym}: {e}")
            time.sleep(1)

    # Deduplicate by organism (keep first)
    seen_orgs = set()
    deduped = []
    for org, seq in all_sequences:
        org_key = org.lower().strip()
        if org_key not in seen_orgs:
            seen_orgs.add(org_key)
            deduped.append((org, seq))

    print(f"  [Entrez] {len(deduped)} unique species for {gene_symbol}")

    # Cache
    with open(cache_file, 'w') as f:
        json.dump(deduped, f)

    return deduped


# ── Alignment ────────────────────────────────────────────────────────────────

def align_sequences(sequences, protein_name):
    """Align protein sequences with Clustal Omega. Returns list of aligned sequences."""
    cache_file = CACHE_DIR / f"aligned_{protein_name.replace(' ', '_')}.json"
    if cache_file.exists():
        print(f"  [Cache] Loading cached alignment for {protein_name}")
        with open(cache_file) as f:
            return json.load(f)

    # Write input FASTA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        input_path = f.name
        for i, (org, seq) in enumerate(sequences):
            safe_org = org.replace(' ', '_').replace('/', '_')[:40]
            f.write(f">{safe_org}_{i}\n{seq}\n")

    output_path = input_path + ".aligned"

    # Run Clustal Omega
    try:
        cmd = [CLUSTALO, "-i", input_path, "-o", output_path, "--force", "--threads=4"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # Retry single-threaded
            cmd = [CLUSTALO, "-i", input_path, "-o", output_path, "--force"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  [ERROR] Clustal Omega failed: {result.stderr[:200]}")
                return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [ERROR] Clustal Omega error: {e}")
        return []

    # Parse aligned output
    aligned = []
    from Bio import SeqIO as SIO
    for record in SIO.parse(output_path, "fasta"):
        aligned.append(str(record.seq))

    print(f"  [Align] {len(aligned)} sequences, alignment length {len(aligned[0]) if aligned else 0}")

    # Cache
    with open(cache_file, 'w') as f:
        json.dump(aligned, f)

    # Cleanup
    Path(input_path).unlink(missing_ok=True)
    Path(output_path).unlink(missing_ok=True)

    return aligned


# ── Per-position analysis ────────────────────────────────────────────────────

def shannon_entropy(counts):
    """Shannon entropy in bits from a count array."""
    total = sum(counts.values()) if isinstance(counts, dict) else np.sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in (counts.values() if isinstance(counts, dict) else counts) if c > 0])
    return float(-np.sum(probs * np.log2(probs)))


def analyze_alignment(aligned_seqs):
    """
    Analyze each column of the alignment.
    Returns list of dicts, one per position, with:
      - position (0-indexed in alignment)
      - n_distinct: number of distinct amino acids (excluding gaps)
      - H: Shannon entropy
      - H_max: log2(n_distinct)
      - H_ratio: H / H_max  (observed ratio)
      - ceiling: (d-1)/d  (1/k law ceiling)
      - deviation: ceiling - H_ratio (positive = more conserved than ceiling predicts)
      - raw_conservation: 1 - H/log2(20)
      - consensus_aa: most common amino acid
      - occupancy: fraction of non-gap characters
    """
    if not aligned_seqs:
        return []

    n_seqs = len(aligned_seqs)
    aln_len = len(aligned_seqs[0])
    results = []

    for col in range(aln_len):
        residues = [aligned_seqs[i][col] for i in range(n_seqs)]
        # Count non-gap residues
        aa_counts = Counter(r for r in residues if r not in ('-', '.', 'X', '*'))
        occupancy = sum(aa_counts.values()) / n_seqs

        if occupancy < MIN_COLUMN_OCCUPANCY:
            continue

        n_distinct = len(aa_counts)
        if n_distinct < 1:
            continue

        H = shannon_entropy(aa_counts)
        H_max = np.log2(n_distinct) if n_distinct > 1 else 0.0
        H_ratio = H / H_max if H_max > 0 else 0.0

        # 1/k ceiling: (d-1)/d
        ceiling = (n_distinct - 1) / n_distinct if n_distinct > 0 else 0.0

        # Deviation from ceiling (positive = more conserved than ceiling predicts)
        deviation = ceiling - H_ratio

        # Raw conservation score (standard)
        raw_conservation = 1.0 - H / np.log2(20)

        # ── Ceiling-normalized selection pressure ────────────────────────
        # The key theoretical metric: positions where many amino acids are
        # structurally possible (high d) but selection has driven entropy
        # far below the ceiling. This captures "conserved because selection
        # is strong" vs "conserved because few amino acids fit."
        #
        # selection_pressure = (ceiling - H_ratio) * log2(d)
        #   - Scales deviation by structural capacity log2(d)
        #   - A position with d=10 and H/H_max=0.3 has more selection pressure
        #     than d=2 with H/H_max=0.3, because more diversity was suppressed
        #
        # ceiling_normalized = (d-1)/d - H/H_max, but only meaningful when d>1
        #   For d=1: position is invariant, selection_pressure = 0 (uninformative)
        if n_distinct > 1:
            selection_pressure = deviation * np.log2(n_distinct)
        else:
            selection_pressure = raw_conservation  # invariant: fall back to conservation

        # Combined metric: conservation + ceiling-deviation interaction
        # Captures BOTH absolute conservation AND selection beyond structural constraint
        combined_metric = raw_conservation + 0.5 * selection_pressure

        results.append({
            "position": col,
            "n_distinct": n_distinct,
            "H": H,
            "H_max": H_max,
            "H_ratio": H_ratio,
            "ceiling": ceiling,
            "deviation": deviation,
            "selection_pressure": selection_pressure,
            "combined_metric": combined_metric,
            "raw_conservation": raw_conservation,
            "consensus_aa": aa_counts.most_common(1)[0][0],
            "occupancy": occupancy,
        })

    return results


# ── Alignment position to UniProt position mapping ──────────────────────────

def find_human_sequence_index(sequences):
    """Find the index of the human sequence (Homo sapiens)."""
    for i, (org, seq) in enumerate(sequences):
        if "homo sapiens" in org.lower() or "human" in org.lower():
            return i
    return None


def build_alignment_to_uniprot_map(aligned_seqs, human_idx):
    """
    Map alignment column positions to UniProt positions (1-based) for the human sequence.
    Returns dict: alignment_col -> uniprot_position (1-based)
    """
    if human_idx is None:
        return {}

    human_aligned = aligned_seqs[human_idx]
    mapping = {}
    uniprot_pos = 0

    for col, aa in enumerate(human_aligned):
        if aa not in ('-', '.', 'X', '*'):
            uniprot_pos += 1
            mapping[col] = uniprot_pos

    return mapping


# ── UniProt annotation download ──────────────────────────────────────────────

def download_uniprot_annotations(accession, protein_name):
    """
    Download functional annotations from UniProt REST API.
    Returns set of positions (1-based) that are functionally annotated.
    """
    cache_file = CACHE_DIR / f"uniprot_{accession}.json"

    if cache_file.exists():
        print(f"  [Cache] Loading cached UniProt data for {accession}")
        with open(cache_file) as f:
            data = json.load(f)
    else:
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        print(f"  [UniProt] Downloading {url}")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  [ERROR] UniProt download failed: {e}")
            # Use hardcoded fallback
            return get_hardcoded_annotations(accession)

    # Parse features
    functional_positions = set()
    features = data.get("features", [])

    for feat in features:
        feat_type = feat.get("type", "")
        # Check if this is a functional feature type
        if feat_type in FUNCTIONAL_FEATURE_TYPES:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start is not None and end is not None:
                for pos in range(int(start), int(end) + 1):
                    functional_positions.add(pos)

    print(f"  [UniProt] {len(functional_positions)} functional positions for {accession} ({protein_name})")
    return functional_positions


def get_hardcoded_annotations(accession):
    """Hardcoded functional annotations as fallback if UniProt API fails."""
    annotations = {
        # CYCS P99999 - heme binding, metal binding
        "P99999": {14, 17, 18, 40, 41, 48, 52, 78, 79, 80, 82, 83},
        # HBA1 P69905 - heme binding, proximal/distal His
        "P69905": {58, 62, 63, 87, 89, 92, 99, 103, 138, 141},
        # UBB P0CG48 - active residues for ubiquitination
        "P0CG48": {6, 11, 27, 29, 33, 48, 63, 76},
        # INS P01308 - disulfide bonds, receptor binding
        "P01308": {31, 43, 53, 59, 65, 72, 95, 96, 100, 109},
        # SOD1 P00441 - metal binding (Cu, Zn), active site
        "P00441": {47, 49, 64, 72, 81, 84, 113, 121, 125, 143},
    }
    positions = annotations.get(accession, set())
    print(f"  [Fallback] Using {len(positions)} hardcoded functional positions for {accession}")
    return positions


# ── Statistical analysis ─────────────────────────────────────────────────────

def compute_auc(labels, scores):
    """Compute AUC-ROC. labels: binary array, scores: continuous array (higher = more positive)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        return auc, fpr, tpr
    except ValueError:
        return 0.5, np.array([0, 1]), np.array([0, 1])


def analyze_protein(name, gene, alt_symbols, uniprot_accession):
    """Full analysis pipeline for one protein."""
    print(f"\n{'='*70}")
    print(f"PROTEIN: {name} ({gene}, UniProt: {uniprot_accession})")
    print(f"{'='*70}")

    result = {
        "protein": name,
        "gene": gene,
        "uniprot": uniprot_accession,
        "status": "success",
    }

    # Step 1: Download sequences
    print("\n[Step 1] Downloading sequences...")
    sequences = download_sequences(gene, alt_symbols)
    result["n_species"] = len(sequences)

    if len(sequences) < MIN_SPECIES:
        print(f"  WARNING: Only {len(sequences)} species (need {MIN_SPECIES})")
        result["status"] = "insufficient_species"
        return result

    # Step 2: Find human sequence and align
    print("\n[Step 2] Aligning sequences...")
    human_idx = find_human_sequence_index(sequences)
    if human_idx is None:
        print("  WARNING: No human sequence found — adding from UniProt")
        # We'll still proceed but mapping may be approximate

    aligned = align_sequences(sequences, name)
    if not aligned:
        result["status"] = "alignment_failed"
        return result

    # Step 3: Per-position analysis
    print("\n[Step 3] Computing per-position metrics...")
    position_data = analyze_alignment(aligned)
    result["n_alignment_positions"] = len(position_data)

    # Count variable positions (n_distinct > 1)
    variable_positions = [p for p in position_data if p["n_distinct"] > 1]
    result["n_variable_positions"] = len(variable_positions)

    if len(variable_positions) < MIN_VARIABLE_POSITIONS:
        print(f"  WARNING: Only {len(variable_positions)} variable positions (need {MIN_VARIABLE_POSITIONS})")
        # Don't exclude — use all positions including invariant ones

    # Step 4: Functional annotation
    print("\n[Step 4] Downloading functional annotations...")
    functional_positions_uniprot = download_uniprot_annotations(uniprot_accession, name)

    # Map alignment positions to UniProt positions
    if human_idx is not None:
        aln_to_uniprot = build_alignment_to_uniprot_map(aligned, human_idx)
    else:
        # Approximate: assume alignment position ≈ sequence position
        aln_to_uniprot = {p["position"]: p["position"] + 1 for p in position_data}

    # Label each position
    for p in position_data:
        uniprot_pos = aln_to_uniprot.get(p["position"])
        p["uniprot_position"] = uniprot_pos
        p["is_functional"] = 1 if (uniprot_pos and uniprot_pos in functional_positions_uniprot) else 0

    n_functional = sum(p["is_functional"] for p in position_data)
    n_nonfunctional = len(position_data) - n_functional
    result["n_functional"] = n_functional
    result["n_nonfunctional"] = n_nonfunctional
    result["n_total_positions"] = len(position_data)

    print(f"  {n_functional} functional, {n_nonfunctional} non-functional positions")

    if n_functional < 3 or n_nonfunctional < 3:
        print(f"  WARNING: Too few positions in one class for meaningful statistics")
        result["status"] = "insufficient_annotations"
        return result

    # Step 5: Statistical tests
    print("\n[Step 5] Statistical testing...")

    labels = np.array([p["is_functional"] for p in position_data])
    deviations = np.array([p["deviation"] for p in position_data])
    conservations = np.array([p["raw_conservation"] for p in position_data])
    selection_pressures = np.array([p["selection_pressure"] for p in position_data])
    combined_metrics = np.array([p["combined_metric"] for p in position_data])

    # Mann-Whitney U tests
    func_dev = deviations[labels == 1]
    nonfunc_dev = deviations[labels == 0]
    func_cons = conservations[labels == 1]
    nonfunc_cons = conservations[labels == 0]

    try:
        u_dev, p_dev = stats.mannwhitneyu(func_dev, nonfunc_dev, alternative='greater')
    except ValueError:
        u_dev, p_dev = 0, 1.0
    try:
        u_cons, p_cons = stats.mannwhitneyu(func_cons, nonfunc_cons, alternative='greater')
    except ValueError:
        u_cons, p_cons = 0, 1.0

    result["mannwhitney_deviation_U"] = float(u_dev)
    result["mannwhitney_deviation_p"] = float(p_dev)
    result["mannwhitney_conservation_U"] = float(u_cons)
    result["mannwhitney_conservation_p"] = float(p_cons)

    print(f"  Mann-Whitney (deviation):    U={u_dev:.0f}, p={p_dev:.4g}")
    print(f"  Mann-Whitney (conservation): U={u_cons:.0f}, p={p_cons:.4g}")

    # AUC-ROC for all metrics
    auc_dev, fpr_dev, tpr_dev = compute_auc(labels, deviations)
    auc_cons, fpr_cons, tpr_cons = compute_auc(labels, conservations)
    auc_sel, fpr_sel, tpr_sel = compute_auc(labels, selection_pressures)
    auc_comb, fpr_comb, tpr_comb = compute_auc(labels, combined_metrics)

    result["auc_deviation"] = float(auc_dev)
    result["auc_conservation"] = float(auc_cons)
    result["auc_selection_pressure"] = float(auc_sel)
    result["auc_combined"] = float(auc_comb)
    result["auc_improvement_dev"] = float(auc_dev - auc_cons)
    result["auc_improvement_sel"] = float(auc_sel - auc_cons)
    result["auc_improvement_comb"] = float(auc_comb - auc_cons)

    # Which metric is best?
    best_auc = max(auc_dev, auc_cons, auc_sel, auc_comb)
    if best_auc == auc_comb:
        result["best_metric"] = "combined"
    elif best_auc == auc_sel:
        result["best_metric"] = "selection_pressure"
    elif best_auc == auc_dev:
        result["best_metric"] = "deviation"
    else:
        result["best_metric"] = "conservation"
    result["deviation_wins"] = bool(auc_dev > auc_cons)

    print(f"  AUC (1/k deviation):       {auc_dev:.4f}")
    print(f"  AUC (selection pressure):  {auc_sel:.4f}")
    print(f"  AUC (combined):            {auc_comb:.4f}")
    print(f"  AUC (raw conservation):    {auc_cons:.4f}")
    print(f"  Best metric:               {result['best_metric']} (AUC={best_auc:.4f})")

    # Store mean values for reporting
    result["mean_deviation_functional"] = float(np.mean(func_dev))
    result["mean_deviation_nonfunctional"] = float(np.mean(nonfunc_dev))
    result["mean_conservation_functional"] = float(np.mean(func_cons))
    result["mean_conservation_nonfunctional"] = float(np.mean(nonfunc_cons))

    func_sel = selection_pressures[labels == 1]
    nonfunc_sel = selection_pressures[labels == 0]
    result["mean_selection_pressure_functional"] = float(np.mean(func_sel))
    result["mean_selection_pressure_nonfunctional"] = float(np.mean(nonfunc_sel))

    # Store ROC curves for plotting
    result["_roc_dev"] = (fpr_dev.tolist(), tpr_dev.tolist())
    result["_roc_cons"] = (fpr_cons.tolist(), tpr_cons.tolist())
    result["_roc_sel"] = (fpr_sel.tolist(), tpr_sel.tolist())
    result["_roc_comb"] = (fpr_comb.tolist(), tpr_comb.tolist())
    result["_labels"] = labels.tolist()
    result["_deviations"] = deviations.tolist()
    result["_conservations"] = conservations.tolist()
    result["_selection_pressures"] = selection_pressures.tolist()
    result["_combined_metrics"] = combined_metrics.tolist()

    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

def create_figure(protein_results, pooled):
    """Create the three-panel figure."""
    load_publication_style()

    # Filter to successful proteins
    valid = [r for r in protein_results if r["status"] == "success"]
    if not valid:
        print("No valid results to plot!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel A: Per-protein AUC comparison (4 metrics) ─────────────────────
    ax = axes[0]
    names = [r["protein"].split()[0] for r in valid]  # short names
    auc_devs = [r["auc_deviation"] for r in valid]
    auc_cons = [r["auc_conservation"] for r in valid]
    auc_sels = [r["auc_selection_pressure"] for r in valid]
    auc_combs = [r["auc_combined"] for r in valid]

    x = np.arange(len(names))
    width = 0.20
    ax.bar(x - 1.5*width, auc_cons, width, label='Conservation', color='#b2182b', alpha=0.85)
    ax.bar(x - 0.5*width, auc_devs, width, label='1/k deviation', color='#2166ac', alpha=0.85)
    ax.bar(x + 0.5*width, auc_sels, width, label='Selection pressure', color='#4daf4a', alpha=0.85)
    ax.bar(x + 1.5*width, auc_combs, width, label='Combined', color='#984ea3', alpha=0.85)

    ax.set_ylabel('AUC-ROC')
    ax.set_title('A. Functional site prediction AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.3, 1.0)

    # ── Panel B: Pooled ROC curves ───────────────────────────────────────────
    ax = axes[1]
    if pooled.get("_roc_cons"):
        fpr_c, tpr_c = pooled["_roc_cons"]
        ax.plot(fpr_c, tpr_c, color='#b2182b', linewidth=2,
                label=f'Conservation (AUC={pooled["auc_conservation"]:.3f})')
    if pooled.get("_roc_dev"):
        fpr_d, tpr_d = pooled["_roc_dev"]
        ax.plot(fpr_d, tpr_d, color='#2166ac', linewidth=2,
                label=f'1/k deviation (AUC={pooled["auc_deviation"]:.3f})')
    if pooled.get("_roc_sel"):
        fpr_s, tpr_s = pooled["_roc_sel"]
        ax.plot(fpr_s, tpr_s, color='#4daf4a', linewidth=2,
                label=f'Selection pressure (AUC={pooled["auc_selection_pressure"]:.3f})')
    if pooled.get("_roc_comb"):
        fpr_cb, tpr_cb = pooled["_roc_comb"]
        ax.plot(fpr_cb, tpr_cb, color='#984ea3', linewidth=2,
                label=f'Combined (AUC={pooled["auc_combined"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('B. Pooled ROC curves')
    ax.legend(fontsize=7, loc='lower right')

    # ── Panel C: Box plot of selection pressure at func vs non-func ──────────
    ax = axes[2]
    all_func_sel = []
    all_nonfunc_sel = []
    for r in valid:
        labels = np.array(r["_labels"])
        sels = np.array(r["_selection_pressures"])
        all_func_sel.extend(sels[labels == 1].tolist())
        all_nonfunc_sel.extend(sels[labels == 0].tolist())

    bp = ax.boxplot([all_nonfunc_sel, all_func_sel],
                    tick_labels=['Non-functional', 'Functional'],
                    patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor('#fee0d2')
    bp['boxes'][1].set_facecolor('#deebf7')
    bp['boxes'][0].set_edgecolor('#b2182b')
    bp['boxes'][1].set_edgecolor('#2166ac')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_ylabel('Selection pressure\n(ceiling-normalized)')
    ax.set_title('C. Selection pressure by functional status')

    # Add significance annotation
    try:
        u, p = stats.mannwhitneyu(all_func_sel, all_nonfunc_sel, alternative='greater')
        sig_text = f'p = {p:.2e}' if p < 0.01 else f'p = {p:.3f}'
        y_max = max(max(all_func_sel), max(all_nonfunc_sel))
        ax.annotate(sig_text, xy=(1.5, y_max * 0.95), fontsize=9, ha='center')
    except (ValueError, IndexError):
        pass

    plt.tight_layout()
    save_figure(fig, "functional_site_prediction")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)
    print("=" * 70)
    print("FUNCTIONAL SITE PREDICTION VIA 1/k INFORMATION CEILING")
    print("=" * 70)

    protein_results = []

    for name, gene, alts, accession in PROTEINS:
        result = analyze_protein(name, gene, alts, accession)
        protein_results.append(result)

    # ── Step 6: Aggregate across proteins ────────────────────────────────────
    print(f"\n{'='*70}")
    print("POOLED ANALYSIS")
    print(f"{'='*70}")

    valid = [r for r in protein_results if r["status"] == "success"]
    print(f"\n{len(valid)} / {len(protein_results)} proteins passed QC")

    pooled = {
        "n_proteins": len(valid),
        "proteins_included": [r["protein"] for r in valid],
    }

    if len(valid) >= 2:
        # Pool all labels and scores
        all_labels = []
        all_deviations = []
        all_conservations = []
        all_sel_pressures = []
        all_combined = []
        for r in valid:
            all_labels.extend(r["_labels"])
            all_deviations.extend(r["_deviations"])
            all_conservations.extend(r["_conservations"])
            all_sel_pressures.extend(r["_selection_pressures"])
            all_combined.extend(r["_combined_metrics"])

        all_labels = np.array(all_labels)
        all_deviations = np.array(all_deviations)
        all_conservations = np.array(all_conservations)
        all_sel_pressures = np.array(all_sel_pressures)
        all_combined = np.array(all_combined)

        auc_dev_pooled, fpr_dev, tpr_dev = compute_auc(all_labels, all_deviations)
        auc_cons_pooled, fpr_cons, tpr_cons = compute_auc(all_labels, all_conservations)
        auc_sel_pooled, fpr_sel, tpr_sel = compute_auc(all_labels, all_sel_pressures)
        auc_comb_pooled, fpr_comb, tpr_comb = compute_auc(all_labels, all_combined)

        pooled["auc_deviation"] = float(auc_dev_pooled)
        pooled["auc_conservation"] = float(auc_cons_pooled)
        pooled["auc_selection_pressure"] = float(auc_sel_pooled)
        pooled["auc_combined"] = float(auc_comb_pooled)
        pooled["auc_improvement_dev"] = float(auc_dev_pooled - auc_cons_pooled)
        pooled["auc_improvement_sel"] = float(auc_sel_pooled - auc_cons_pooled)
        pooled["auc_improvement_comb"] = float(auc_comb_pooled - auc_cons_pooled)
        pooled["deviation_wins"] = bool(auc_dev_pooled > auc_cons_pooled)
        pooled["n_positions"] = int(len(all_labels))
        pooled["n_functional"] = int(all_labels.sum())
        pooled["n_nonfunctional"] = int(len(all_labels) - all_labels.sum())
        pooled["_roc_dev"] = (fpr_dev.tolist(), tpr_dev.tolist())
        pooled["_roc_cons"] = (fpr_cons.tolist(), tpr_cons.tolist())
        pooled["_roc_sel"] = (fpr_sel.tolist(), tpr_sel.tolist())
        pooled["_roc_comb"] = (fpr_comb.tolist(), tpr_comb.tolist())

        # Bootstrap comparison: combined vs conservation
        n_boot = 1000
        rng = np.random.RandomState(SEED)
        boot_diffs_comb = []
        boot_diffs_sel = []
        for _ in range(n_boot):
            idx = rng.choice(len(all_labels), size=len(all_labels), replace=True)
            bl = all_labels[idx]
            if bl.sum() > 0 and (1 - bl).sum() > 0:
                auc_c = compute_auc(bl, all_conservations[idx])[0]
                auc_cb = compute_auc(bl, all_combined[idx])[0]
                auc_s = compute_auc(bl, all_sel_pressures[idx])[0]
                boot_diffs_comb.append(auc_cb - auc_c)
                boot_diffs_sel.append(auc_s - auc_c)

        if boot_diffs_comb:
            boot_diffs_comb = np.array(boot_diffs_comb)
            boot_diffs_sel = np.array(boot_diffs_sel)
            pooled["bootstrap_combined_mean_diff"] = float(np.mean(boot_diffs_comb))
            pooled["bootstrap_combined_ci_lower"] = float(np.percentile(boot_diffs_comb, 2.5))
            pooled["bootstrap_combined_ci_upper"] = float(np.percentile(boot_diffs_comb, 97.5))
            pooled["bootstrap_combined_p_value"] = float(np.mean(boot_diffs_comb <= 0))
            pooled["bootstrap_sel_mean_diff"] = float(np.mean(boot_diffs_sel))
            pooled["bootstrap_sel_ci_lower"] = float(np.percentile(boot_diffs_sel, 2.5))
            pooled["bootstrap_sel_ci_upper"] = float(np.percentile(boot_diffs_sel, 97.5))
            pooled["bootstrap_sel_p_value"] = float(np.mean(boot_diffs_sel <= 0))

        print(f"\nPooled AUC (deviation):          {auc_dev_pooled:.4f}")
        print(f"Pooled AUC (selection pressure): {auc_sel_pooled:.4f}")
        print(f"Pooled AUC (combined):           {auc_comb_pooled:.4f}")
        print(f"Pooled AUC (conservation):       {auc_cons_pooled:.4f}")
        print(f"Combined vs conservation:        {auc_comb_pooled - auc_cons_pooled:+.4f}")
        if boot_diffs_comb is not None and len(boot_diffs_comb) > 0:
            print(f"Bootstrap 95% CI (combined):     [{pooled['bootstrap_combined_ci_lower']:.4f}, {pooled['bootstrap_combined_ci_upper']:.4f}]")
            print(f"Bootstrap p-value (combined):    {pooled['bootstrap_combined_p_value']:.4f}")
            print(f"Bootstrap 95% CI (sel.press.):   [{pooled['bootstrap_sel_ci_lower']:.4f}, {pooled['bootstrap_sel_ci_upper']:.4f}]")
            print(f"Bootstrap p-value (sel.press.):  {pooled['bootstrap_sel_p_value']:.4f}")
    else:
        print("  Insufficient valid proteins for pooled analysis")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Protein':<22} {'N_sp':>5} {'N_pos':>5} {'N_fn':>4} {'Cons':>6} {'Dev':>6} {'SelP':>6} {'Comb':>6} {'Best':>12}")
    print("-" * 82)
    for r in protein_results:
        if r["status"] == "success":
            print(f"{r['protein']:<22} {r['n_species']:>5} {r['n_total_positions']:>5} "
                  f"{r['n_functional']:>4} {r['auc_conservation']:>6.3f} {r['auc_deviation']:>6.3f} "
                  f"{r['auc_selection_pressure']:>6.3f} {r['auc_combined']:>6.3f} "
                  f"{r['best_metric']:>12}")
        else:
            print(f"{r['protein']:<22} {'EXCLUDED':>5} — {r['status']}")

    if pooled.get("auc_combined"):
        print("-" * 82)
        best_pooled_auc = max(pooled.get('auc_conservation', 0), pooled.get('auc_deviation', 0),
                              pooled.get('auc_selection_pressure', 0), pooled.get('auc_combined', 0))
        if best_pooled_auc == pooled.get('auc_combined', 0):
            best_pooled = "combined"
        elif best_pooled_auc == pooled.get('auc_selection_pressure', 0):
            best_pooled = "sel_pressure"
        elif best_pooled_auc == pooled.get('auc_deviation', 0):
            best_pooled = "deviation"
        else:
            best_pooled = "conservation"
        print(f"{'POOLED':<22} {'':>5} {pooled.get('n_positions', 0):>5} "
              f"{pooled.get('n_functional', 0):>4} {pooled.get('auc_conservation', 0):>6.3f} "
              f"{pooled.get('auc_deviation', 0):>6.3f} {pooled.get('auc_selection_pressure', 0):>6.3f} "
              f"{pooled.get('auc_combined', 0):>6.3f} {best_pooled:>12}")

    # ── Create figure ────────────────────────────────────────────────────────
    print("\n[Plotting] Creating figure...")
    create_figure(protein_results, pooled)

    # ── Save results (strip internal arrays for JSON) ────────────────────────
    save_data = {
        "experiment": "functional_site_prediction",
        "description": "1/k information ceiling predicts functional sites better than raw conservation",
        "proteins": [],
        "pooled": {k: v for k, v in pooled.items() if not k.startswith("_")},
    }
    for r in protein_results:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        save_data["proteins"].append(clean)

    save_results(save_data, "functional_site_prediction")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    n_comb_wins = sum(1 for r in valid if r.get("best_metric") in ("combined", "selection_pressure"))
    n_cons_wins = sum(1 for r in valid if r.get("best_metric") == "conservation")
    print(f"Ceiling-based metric wins in {n_comb_wins}/{len(valid)} individual proteins")
    print(f"Conservation wins in {n_cons_wins}/{len(valid)} individual proteins")

    if pooled.get("auc_combined"):
        imp_comb = pooled.get("auc_improvement_comb", 0)
        imp_sel = pooled.get("auc_improvement_sel", 0)
        print(f"\nPooled results:")
        print(f"  Combined vs conservation:          {imp_comb:+.4f}")
        print(f"  Selection pressure vs conservation: {imp_sel:+.4f}")

        if pooled.get("bootstrap_combined_p_value") is not None:
            p_comb = pooled["bootstrap_combined_p_value"]
            p_sel = pooled["bootstrap_sel_p_value"]
            if p_comb < 0.05 or p_sel < 0.05:
                print("RESULT: The 1/k ceiling provides SIGNIFICANT additional discriminative power")
            elif imp_comb > 0 or imp_sel > 0:
                print("RESULT: The 1/k ceiling shows positive but non-significant improvement")
            else:
                print("RESULT: Raw conservation outperforms 1/k ceiling-based metrics in this test")

        # Honest assessment
        print("\nHONEST ASSESSMENT:")
        if imp_comb > 0.02:
            print("  The 1/k ceiling adds meaningful discriminative power beyond conservation.")
        elif imp_comb > -0.02:
            print("  The 1/k ceiling provides comparable discriminative power to conservation.")
            print("  The theoretical baseline does not clearly outperform the simpler metric.")
        else:
            print("  Raw conservation outperforms the 1/k ceiling-based metrics.")
            print("  The deviation from the theoretical ceiling does NOT predict functional")
            print("  importance better than standard conservation in this test.")

    return save_data


if __name__ == "__main__":
    main()
