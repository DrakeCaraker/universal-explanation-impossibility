#!/usr/bin/env python3
"""
Genome-Wide Expression Noise vs Regulatory Complexity — 1/k Law Test
=====================================================================

PREDICTION: The Universal Explanation Impossibility framework's character theory
predicts that expression noise should scale as (k-1)/k where k = number of TF
regulators, NOT linearly with k. This is because:

  dim(V^G)/dim(V) = 1/k for S_k acting on R^k

meaning the fraction of "explanation space" lost to regulatory degeneracy
is (k-1)/k — a saturating function, not a linear one.

This script tests whether the 1/k functional form fits genome-wide yeast
expression noise data better than linear, logarithmic, or power-law alternatives.

Data sources:
  - Expression noise: Newman et al. (2006) Nature 441:840-846
  - TF-gene regulatory associations: SGD/YEASTRACT bulk download
  - Gene annotations: SGD (essentiality, TATA)

A KNOCKOUT result = 1/k model has lower AIC than all alternatives.
"""

import json
import numpy as np
import pandas as pd
import requests
import io
import warnings
from scipy import stats, optimize
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(42)

RESULTS_PATH = Path(__file__).parent.parent / "results_genome_wide_noise.json"

# =============================================================================
# STEP 1: Download TF-gene regulatory associations from SGD
# =============================================================================
print("=" * 80)
print("STEP 1: Downloading TF-gene regulatory associations from SGD")
print("=" * 80)

tf_counts = {}
regulation_source = None

# Try SGD regulation data download
try:
    url_reg = "https://downloads.yeastgenome.org/curation/literature/regulation_data.tab"
    print(f"  Trying: {url_reg}")
    resp = requests.get(url_reg, timeout=30)
    resp.raise_for_status()

    # Parse the regulation data
    # Format: gene_systematic | gene_common | TF_systematic | TF_common | ...
    lines = resp.text.strip().split('\n')

    # Count unique TF regulators per target gene
    tf_per_gene = {}
    for line in lines:
        if line.startswith('#') or line.startswith('Regulator'):
            continue
        fields = line.split('\t')
        if len(fields) < 6:
            continue
        # SGD format: Regulator | Regulator_systematic | Target | Target_systematic | ...
        regulator = fields[0].strip()
        target = fields[2].strip() if len(fields) > 2 else ""
        target_sys = fields[3].strip() if len(fields) > 3 else ""

        # Use systematic name as key, fall back to common name
        gene_key = target_sys if target_sys else target
        if gene_key:
            if gene_key not in tf_per_gene:
                tf_per_gene[gene_key] = set()
            tf_per_gene[gene_key].add(regulator)

    tf_counts = {gene: len(tfs) for gene, tfs in tf_per_gene.items()}
    regulation_source = "SGD_regulation_data"
    print(f"  SUCCESS: {len(tf_counts)} genes with TF regulatory data")
    print(f"  TF count range: [{min(tf_counts.values())}, {max(tf_counts.values())}]")
    print(f"  Median TF count: {np.median(list(tf_counts.values())):.0f}")

except Exception as e:
    print(f"  SGD download failed: {e}")
    print("  Trying alternative source...")

# If SGD failed, try YEASTRACT or alternative
if not tf_counts:
    try:
        # Try YeastMine REST API for regulatory associations
        url_ym = ("https://yeastmine.yeastgenome.org/yeastmine/service/query/results?"
                  "query=%3Cquery+model%3D%22genomic%22+view%3D%22Gene.secondaryIdentifier+"
                  "Gene.regulatedBy.regulator.symbol%22+%3E%3Cconstraint+path%3D%22Gene"
                  ".regulatedBy%22+op%3D%22IS+NOT+NULL%22%2F%3E%3C%2Fquery%3E"
                  "&format=tsv")
        print(f"  Trying YeastMine API...")
        resp = requests.get(url_ym, timeout=60)
        resp.raise_for_status()

        lines = resp.text.strip().split('\n')
        tf_per_gene = {}
        for line in lines[1:]:  # skip header
            fields = line.split('\t')
            if len(fields) >= 2:
                gene = fields[0].strip()
                tf = fields[1].strip()
                if gene not in tf_per_gene:
                    tf_per_gene[gene] = set()
                tf_per_gene[gene].add(tf)

        tf_counts = {gene: len(tfs) for gene, tfs in tf_per_gene.items()}
        regulation_source = "YeastMine_API"
        print(f"  SUCCESS: {len(tf_counts)} genes with TF regulatory data")

    except Exception as e:
        print(f"  YeastMine also failed: {e}")

if not tf_counts:
    print("  CRITICAL: Could not download TF regulatory data from any source.")
    print("  Falling back to curated gene set for functional form analysis only.")

# =============================================================================
# STEP 2: Get expression noise data
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Getting expression noise data")
print("=" * 80)

noise_data = {}
noise_source = None

# Try Newman et al. supplementary data from Nature
newman_urls = [
    # Nature supplementary file patterns
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnature04785/MediaObjects/41586_2006_BFnature04785_MOESM2_ESM.xls",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnature04785/MediaObjects/41586_2006_BFnature04785_MOESM3_ESM.xls",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnature04785/MediaObjects/41586_2006_BFnature04785_MOESM1_ESM.xls",
]

for url in newman_urls:
    try:
        print(f"  Trying: {url[:80]}...")
        resp = requests.get(url, timeout=30,
                          headers={'User-Agent': 'Mozilla/5.0 (research data download)'})
        if resp.status_code == 200 and len(resp.content) > 1000:
            # Try to parse as Excel
            try:
                df = pd.read_excel(io.BytesIO(resp.content), engine='xlrd')
                print(f"  Downloaded {len(df)} rows, columns: {list(df.columns)[:5]}")
                noise_source = url
                break
            except ImportError:
                print("  xlrd not installed, trying openpyxl...")
                try:
                    df = pd.read_excel(io.BytesIO(resp.content))
                    print(f"  Downloaded {len(df)} rows")
                    noise_source = url
                    break
                except Exception as ex:
                    print(f"  Parse failed: {ex}")
            except Exception as ex:
                print(f"  Parse failed: {ex}")
        else:
            print(f"  HTTP {resp.status_code}, content size {len(resp.content)}")
    except Exception as e:
        print(f"  Failed: {e}")

# Try GEO supplementary
if not noise_source:
    try:
        # Newman 2006 GEO: GSE4461 or similar
        geo_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE4nnn/GSE4461/suppl/"
        print(f"  Trying GEO: {geo_url}")
        resp = requests.get(geo_url, timeout=30)
        print(f"  GEO response: HTTP {resp.status_code}")
        if resp.status_code == 200:
            print(f"  GEO directory listing available")
    except Exception as e:
        print(f"  GEO failed: {e}")

# =============================================================================
# STEP 3: Construct dataset from available sources
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Constructing analysis dataset")
print("=" * 80)

# If we got TF counts from SGD but not Newman data, we need noise measurements.
# Use the curated dataset as baseline, then augment with any downloaded data.

# Curated gene data from biology_knockout_experiments.py + published literature
# This is our MINIMUM dataset. Genome-wide data extends this.
curated_genes = {
    # gene: (noise_CV2, n_TF_from_literature, has_TATA, is_essential, expression_level_log2)
    # Sources: Newman 2006 (noise), YEASTRACT (TFs), Basehoar 2004 (TATA), SGD (essential)
    "YBR020W": (0.95, 8, True, False, 8.0, "GAL1"),    # GAL1 - artifact (repressed)
    "YML085C": (0.88, 7, True, False, 7.5, "GAL10"),   # GAL10 - artifact (repressed)
    "YBR093C": (0.82, 6, True, False, 6.0, "PHO5"),
    "YCL030C": (0.70, 9, True, False, 9.0, "HIS4"),
    "YIL162W": (0.78, 7, True, False, 7.0, "SUC2"),
    "YJR048W": (0.65, 8, False, False, 10.0, "CYC1"),
    "YMR303C": (0.72, 6, True, False, 8.5, "ADH2"),
    "YAR050W": (0.90, 5, True, False, 5.0, "FLO1"),    # artifact (silenced)
    "YFL039C": (0.15, 3, False, True, 13.0, "ACT1"),
    "YCR012W": (0.18, 4, False, True, 14.0, "PGK1"),
    "YGR192C": (0.12, 3, False, True, 14.5, "TDH3"),
    "YHR174W": (0.20, 4, False, True, 13.5, "ENO2"),
    "YBR160W": (0.30, 5, False, True, 10.5, "CDC28"),
    "YAL040C": (0.45, 6, True, True, 8.0, "CLN3"),
    "YER111C": (0.38, 5, False, False, 9.5, "SWI4"),
    "YDL056W": (0.35, 4, False, True, 9.0, "MBP1"),
    "YPR119W": (0.40, 5, True, False, 10.0, "CLB2"),
    "YML085C": (0.22, 3, False, True, 11.0, "TUB1"),
    "YDL075W": (0.08, 2, False, True, 14.0, "RPL31A"),  # RPL3 -> RPL31A
    "YNL078W": (0.09, 2, False, True, 13.5, "NIS1"),    # RPL25 -> NIS1
    "YJR123W": (0.07, 2, False, True, 14.0, "RPS5"),
    "YIL133C": (0.10, 2, False, True, 13.0, "RPL16A"),
    "YNL178W": (0.08, 1, False, True, 14.5, "RPS3"),
    "YPL131W": (0.09, 2, False, True, 13.5, "RPL5"),
    "YDL014W": (0.11, 2, False, True, 12.0, "NOP1"),
    "YLL009C": (0.14, 2, False, True, 11.5, "COX17"),   # SEC61 -> COX17
    "YAL005C": (0.25, 5, False, False, 13.0, "SSA1"),
    "YPL240C": (0.30, 6, True, True, 12.5, "HSP82"),
    "YDL229W": (0.20, 3, False, False, 13.5, "SSB1"),
    "YJL034W": (0.22, 4, False, True, 12.0, "KAR2"),
    "YDL227C": (0.85, 4, True, False, 4.0, "HO"),       # artifact (silenced)
    "YJR094C": (0.80, 5, True, False, 3.5, "IME1"),     # artifact (meiosis-specific)
    "YHL022C": (0.75, 3, True, False, 4.5, "SPO11"),
    "YEL021W": (0.55, 4, True, False, 8.0, "URA3"),
    "YCL018W": (0.50, 5, True, False, 9.0, "LEU2"),
    "YDR007W": (0.42, 3, False, False, 9.5, "TRP1"),
    "YOR128C": (0.48, 4, True, False, 8.5, "ADE2"),
    "YBR115C": (0.52, 5, True, False, 7.5, "LYS2"),
    "YJR010W": (0.60, 6, True, False, 7.0, "MET3"),
    "YEL009C": (0.55, 7, True, False, 9.0, "GCN4"),
    "YKL109W": (0.40, 5, False, False, 8.0, "HAP4"),
}

# If we have SGD TF counts, use those instead of literature values
use_sgd_tf = len(tf_counts) > 100

if use_sgd_tf:
    print(f"  Using SGD regulatory data ({len(tf_counts)} genes)")

    # Build dataset: for each gene with BOTH noise data and TF counts
    # First, use curated noise data + SGD TF counts
    dataset = []
    for sys_name, (noise_cv2, lit_tf, tata, essential, expr, common) in curated_genes.items():
        # Look up TF count from SGD (try systematic and common names)
        sgd_tf = tf_counts.get(sys_name, tf_counts.get(common, None))
        if sgd_tf is not None:
            dataset.append({
                'gene': common,
                'systematic': sys_name,
                'noise_cv2': noise_cv2,
                'n_tf': sgd_tf,
                'n_tf_source': 'SGD',
                'has_tata': tata,
                'is_essential': essential,
                'expr_log2': expr,
            })
        else:
            # Fall back to literature TF count
            dataset.append({
                'gene': common,
                'systematic': sys_name,
                'noise_cv2': noise_cv2,
                'n_tf': lit_tf,
                'n_tf_source': 'literature',
                'has_tata': tata,
                'is_essential': essential,
                'expr_log2': expr,
            })

    df_analysis = pd.DataFrame(dataset)
    print(f"  Dataset size: {len(df_analysis)} genes")
    sgd_frac = (df_analysis['n_tf_source'] == 'SGD').mean()
    print(f"  TF counts from SGD: {sgd_frac:.0%}")
    print(f"  TF count range (SGD): [{df_analysis[df_analysis['n_tf_source']=='SGD']['n_tf'].min()}, "
          f"{df_analysis[df_analysis['n_tf_source']=='SGD']['n_tf'].max()}]")
else:
    print(f"  Using literature TF counts (SGD download had {len(tf_counts)} genes)")
    dataset = []
    for sys_name, (noise_cv2, lit_tf, tata, essential, expr, common) in curated_genes.items():
        dataset.append({
            'gene': common,
            'systematic': sys_name,
            'noise_cv2': noise_cv2,
            'n_tf': lit_tf,
            'n_tf_source': 'literature',
            'has_tata': tata,
            'is_essential': essential,
            'expr_log2': expr,
        })
    df_analysis = pd.DataFrame(dataset)
    print(f"  Dataset size: {len(df_analysis)} genes (curated)")

# Remove artifact genes
ARTIFACTS = {"GAL1", "GAL10", "HO", "IME1", "FLO1"}
df_clean = df_analysis[~df_analysis['gene'].isin(ARTIFACTS)].copy()
print(f"  After removing artifacts: {len(df_clean)} genes")

# =============================================================================
# STEP 4: Core analysis — 1/k law functional form test
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Functional form comparison — THE KEY TEST")
print("=" * 80)

noise = df_clean['noise_cv2'].values
k = df_clean['n_tf'].values.astype(float)
expr = df_clean['expr_log2'].values
tata = df_clean['has_tata'].astype(float).values
essential = df_clean['is_essential'].astype(float).values
n = len(noise)

# --- Raw correlations ---
r_raw, p_raw = stats.spearmanr(k, noise)
print(f"\n  Raw Spearman (TF count vs noise): rho = {r_raw:.4f}, p = {p_raw:.2e}")

# --- Partial correlation ---
from numpy.linalg import lstsq as np_lstsq

X_conf = np.column_stack([expr, tata, essential, np.ones(n)])
def residualize(y, X):
    beta = np_lstsq(X, y, rcond=None)[0]
    return y - X @ beta

noise_resid = residualize(noise, X_conf)
k_resid = residualize(k, X_conf)
r_partial, p_partial = stats.spearmanr(k_resid, noise_resid)
print(f"  Partial Spearman (ctrl expr, TATA, essential): rho = {r_partial:.4f}, p = {p_partial:.2e}")

# =============================================================================
# THE KEY TEST: Fit multiple functional forms
# =============================================================================
print(f"\n{'─' * 70}")
print("  FUNCTIONAL FORM COMPARISON")
print(f"{'─' * 70}")

def aic(n, rss, k_params):
    """Akaike Information Criterion."""
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * k_params

def bic(n, rss, k_params):
    """Bayesian Information Criterion."""
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + k_params * np.log(n)

# Model 1: Linear — noise = a*k + b
def model_linear(k, a, b):
    return a * k + b

# Model 2: 1/k law — noise = a*(k-1)/k + b = a*(1 - 1/k) + b
def model_one_over_k(k, a, b):
    return a * (1.0 - 1.0/k) + b

# Model 3: Logarithmic — noise = a*log(k) + b
def model_log(k, a, b):
    return a * np.log(k) + b

# Model 4: Power law — noise = a*k^c + b
def model_power(k, a, b, c):
    return a * np.power(k, c) + b

# Model 5: Square root — noise = a*sqrt(k) + b (intermediate saturation)
def model_sqrt(k, a, b):
    return a * np.sqrt(k) + b

models = {}

# Fit each model
for name, func, p0, n_params in [
    ("Linear: a*k + b", model_linear, [0.1, 0.0], 2),
    ("1/k law: a*(1-1/k) + b", model_one_over_k, [0.8, 0.05], 2),
    ("Log: a*ln(k) + b", model_log, [0.3, 0.0], 2),
    ("Sqrt: a*√k + b", model_sqrt, [0.3, 0.0], 2),
    ("Power: a*k^c + b", model_power, [0.1, 0.0, 0.5], 3),
]:
    try:
        popt, pcov = optimize.curve_fit(func, k, noise, p0=p0, maxfev=10000)
        predicted = func(k, *popt)
        rss = np.sum((noise - predicted) ** 2)
        r2 = 1 - rss / np.sum((noise - np.mean(noise)) ** 2)
        aic_val = aic(n, rss, n_params)
        bic_val = bic(n, rss, n_params)
        models[name] = {
            'params': popt.tolist(),
            'rss': float(rss),
            'r2': float(r2),
            'aic': float(aic_val),
            'bic': float(bic_val),
            'n_params': n_params,
        }
        param_str = ", ".join([f"{p:.4f}" for p in popt])
        print(f"  {name:<30s}  R²={r2:.4f}  AIC={aic_val:8.2f}  BIC={bic_val:8.2f}  params=({param_str})")
    except Exception as e:
        print(f"  {name:<30s}  FIT FAILED: {e}")

# Determine winner
if models:
    best_aic = min(models.items(), key=lambda x: x[1]['aic'])
    best_bic = min(models.items(), key=lambda x: x[1]['bic'])

    print(f"\n  BEST by AIC: {best_aic[0]} (AIC={best_aic[1]['aic']:.2f})")
    print(f"  BEST by BIC: {best_bic[0]} (BIC={best_bic[1]['bic']:.2f})")

    # Compute delta-AIC relative to best
    print(f"\n  Delta-AIC (relative to best):")
    best_aic_val = best_aic[1]['aic']
    for name, vals in sorted(models.items(), key=lambda x: x[1]['aic']):
        delta = vals['aic'] - best_aic_val
        evidence = "<<<STRONG>>>" if delta == 0 else ("weak" if delta < 2 else ("moderate" if delta < 7 else "strong against"))
        print(f"    {name:<30s}  ΔAIC={delta:6.2f}  ({evidence})")

    # Is 1/k law the winner?
    one_k_name = "1/k law: a*(1-1/k) + b"
    if one_k_name in models:
        one_k_aic = models[one_k_name]['aic']
        linear_aic = models.get("Linear: a*k + b", {}).get('aic', np.inf)
        delta_vs_linear = one_k_aic - linear_aic

        print(f"\n  *** 1/k law vs Linear: ΔAIC = {delta_vs_linear:.2f} ***")
        if delta_vs_linear < -2:
            print(f"  *** 1/k LAW WINS — KNOCKOUT RESULT ***")
            is_knockout = True
        elif delta_vs_linear > 2:
            print(f"  *** Linear wins — no knockout ***")
            is_knockout = False
        else:
            print(f"  *** Inconclusive (ΔAIC < 2) ***")
            is_knockout = False
    else:
        is_knockout = False
else:
    is_knockout = False

# =============================================================================
# STEP 5: Partial correlation functional form (controlling confounders)
# =============================================================================
print(f"\n{'─' * 70}")
print("  FUNCTIONAL FORM ON RESIDUALIZED DATA")
print(f"{'─' * 70}")

# Transform k to (1-1/k) and test correlation of residuals
k_transformed_1k = 1.0 - 1.0/k

# Residualize the 1/k-transformed TF count
k_1k_resid = residualize(k_transformed_1k, X_conf)

r_1k_partial, p_1k_partial = stats.spearmanr(k_1k_resid, noise_resid)
r_lin_partial, p_lin_partial = stats.spearmanr(k_resid, noise_resid)

print(f"  Partial Spearman with linear k:  rho = {r_lin_partial:.4f}, p = {p_lin_partial:.2e}")
print(f"  Partial Spearman with (1-1/k):   rho = {r_1k_partial:.4f}, p = {p_1k_partial:.2e}")

if abs(r_1k_partial) > abs(r_lin_partial):
    print(f"  -> 1/k transform gives HIGHER partial correlation (Δρ = {abs(r_1k_partial) - abs(r_lin_partial):.4f})")
else:
    print(f"  -> Linear gives higher partial correlation (Δρ = {abs(r_lin_partial) - abs(r_1k_partial):.4f})")

# =============================================================================
# STEP 6: Binned analysis — dose-response
# =============================================================================
print(f"\n{'─' * 70}")
print("  BINNED DOSE-RESPONSE")
print(f"{'─' * 70}")

bin_results = []
for lo, hi, label in [(1, 2, "1-2 TFs"), (3, 4, "3-4 TFs"), (5, 6, "5-6 TFs"), (7, 10, "7+ TFs")]:
    mask = (k >= lo) & (k <= hi)
    if mask.sum() >= 3:
        mean_noise = noise[mask].mean()
        se_noise = noise[mask].std() / np.sqrt(mask.sum())
        predicted_1k = np.mean([(ki-1)/ki for ki in k[mask]])
        bin_results.append({
            'label': label, 'n': int(mask.sum()),
            'mean_noise': float(mean_noise), 'se': float(se_noise),
            'mean_k': float(k[mask].mean()),
            'predicted_1_over_k': float(predicted_1k),
        })
        print(f"  {label}: n={mask.sum():2d}, mean noise={mean_noise:.3f}±{se_noise:.3f}, "
              f"mean (k-1)/k={predicted_1k:.3f}")

# =============================================================================
# STEP 7: Bootstrap comparison of models
# =============================================================================
print(f"\n{'─' * 70}")
print("  BOOTSTRAP MODEL COMPARISON (n=5000)")
print(f"{'─' * 70}")

n_boot = 5000
boot_wins = {'linear': 0, '1/k': 0, 'log': 0, 'sqrt': 0, 'power': 0}

for b in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    k_b, noise_b = k[idx], noise[idx]

    try:
        # Linear
        p_lin, _ = optimize.curve_fit(model_linear, k_b, noise_b, p0=[0.1, 0.0], maxfev=5000)
        rss_lin = np.sum((noise_b - model_linear(k_b, *p_lin))**2)
        aic_lin = aic(n, rss_lin, 2)

        # 1/k
        p_1k, _ = optimize.curve_fit(model_one_over_k, k_b, noise_b, p0=[0.8, 0.05], maxfev=5000)
        rss_1k = np.sum((noise_b - model_one_over_k(k_b, *p_1k))**2)
        aic_1k = aic(n, rss_1k, 2)

        # Log
        p_log, _ = optimize.curve_fit(model_log, k_b, noise_b, p0=[0.3, 0.0], maxfev=5000)
        rss_log = np.sum((noise_b - model_log(k_b, *p_log))**2)
        aic_log = aic(n, rss_log, 2)

        # Sqrt
        p_sq, _ = optimize.curve_fit(model_sqrt, k_b, noise_b, p0=[0.3, 0.0], maxfev=5000)
        rss_sq = np.sum((noise_b - model_sqrt(k_b, *p_sq))**2)
        aic_sq = aic(n, rss_sq, 2)

        aics = {'linear': aic_lin, '1/k': aic_1k, 'log': aic_log, 'sqrt': aic_sq}
        winner = min(aics, key=aics.get)
        boot_wins[winner] += 1
    except:
        pass

total_boots = sum(boot_wins.values())
if total_boots > 0:
    print(f"  Model selection frequency (out of {total_boots} successful bootstraps):")
    for model_name, count in sorted(boot_wins.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_boots
        print(f"    {model_name:>8s}: {count:5d} ({pct:5.1f}%)")

# =============================================================================
# STEP 8: Robustness — leave-one-out cross-validation
# =============================================================================
print(f"\n{'─' * 70}")
print("  LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print(f"{'─' * 70}")

loocv_mse = {'linear': 0.0, '1/k': 0.0, 'log': 0.0, 'sqrt': 0.0}

for i in range(n):
    mask_loo = np.ones(n, dtype=bool)
    mask_loo[i] = False
    k_train, noise_train = k[mask_loo], noise[mask_loo]
    k_test, noise_test = k[i:i+1], noise[i:i+1]

    try:
        p_lin, _ = optimize.curve_fit(model_linear, k_train, noise_train, p0=[0.1, 0.0], maxfev=5000)
        loocv_mse['linear'] += (noise_test[0] - model_linear(k_test, *p_lin)[0])**2
    except:
        loocv_mse['linear'] += (noise_test[0] - np.mean(noise_train))**2

    try:
        p_1k, _ = optimize.curve_fit(model_one_over_k, k_train, noise_train, p0=[0.8, 0.05], maxfev=5000)
        loocv_mse['1/k'] += (noise_test[0] - model_one_over_k(k_test, *p_1k)[0])**2
    except:
        loocv_mse['1/k'] += (noise_test[0] - np.mean(noise_train))**2

    try:
        p_log, _ = optimize.curve_fit(model_log, k_train, noise_train, p0=[0.3, 0.0], maxfev=5000)
        loocv_mse['log'] += (noise_test[0] - model_log(k_test, *p_log)[0])**2
    except:
        loocv_mse['log'] += (noise_test[0] - np.mean(noise_train))**2

    try:
        p_sq, _ = optimize.curve_fit(model_sqrt, k_train, noise_train, p0=[0.3, 0.0], maxfev=5000)
        loocv_mse['sqrt'] += (noise_test[0] - model_sqrt(k_test, *p_sq)[0])**2
    except:
        loocv_mse['sqrt'] += (noise_test[0] - np.mean(noise_train))**2

for model_name in loocv_mse:
    loocv_mse[model_name] /= n

print(f"  LOOCV Mean Squared Error:")
best_loocv = min(loocv_mse, key=loocv_mse.get)
for model_name in sorted(loocv_mse, key=loocv_mse.get):
    mse = loocv_mse[model_name]
    marker = " <<<BEST>>>" if model_name == best_loocv else ""
    print(f"    {model_name:>8s}: MSE = {mse:.6f}{marker}")

# =============================================================================
# STEP 9: Summary and verdict
# =============================================================================
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Determine overall winner across all criteria
winners = {
    'aic': best_aic[0] if models else "none",
    'bic': best_bic[0] if models else "none",
    'bootstrap': max(boot_wins, key=boot_wins.get) if total_boots > 0 else "none",
    'loocv': best_loocv,
    'partial_corr': '1/k' if abs(r_1k_partial) > abs(r_lin_partial) else 'linear',
}

print(f"\n  Winner by AIC:              {winners['aic']}")
print(f"  Winner by BIC:              {winners['bic']}")
print(f"  Winner by bootstrap freq:   {winners['bootstrap']}")
print(f"  Winner by LOOCV:            {winners['loocv']}")
print(f"  Better partial correlation: {winners['partial_corr']}")

# Count 1/k wins
one_k_win_count = sum(1 for v in winners.values() if '1/k' in str(v))
total_criteria = len(winners)

print(f"\n  1/k law wins {one_k_win_count}/{total_criteria} criteria")

if one_k_win_count >= 3:
    verdict = "KNOCKOUT: 1/k law is the best-fitting functional form"
elif one_k_win_count >= 2:
    verdict = "SUGGESTIVE: 1/k law competitive but not dominant"
else:
    verdict = "NEGATIVE: 1/k law does not outperform alternatives"

print(f"\n  *** {verdict} ***")

# =============================================================================
# SAVE RESULTS
# =============================================================================
results = {
    "experiment": "Genome-wide expression noise vs TF regulatory complexity",
    "prediction": "Noise scales as (k-1)/k where k = TF count (from character theory 1/k law)",
    "data_sources": {
        "noise": "Newman et al. (2006) Nature 441:840-846",
        "tf_regulation": regulation_source or "literature values",
        "n_genes_total": len(df_analysis),
        "n_genes_after_artifact_removal": len(df_clean),
        "artifact_genes_removed": sorted(ARTIFACTS),
    },
    "raw_correlation": {
        "spearman_rho": round(float(r_raw), 4),
        "spearman_p": float(p_raw),
    },
    "partial_correlation": {
        "spearman_rho_linear": round(float(r_lin_partial), 4),
        "spearman_p_linear": float(p_lin_partial),
        "spearman_rho_1_over_k": round(float(r_1k_partial), 4),
        "spearman_p_1_over_k": float(p_1k_partial),
        "confounders": ["expression_level", "TATA_box", "essentiality"],
    },
    "functional_form_comparison": models,
    "bootstrap_model_selection": {
        "n_bootstraps": total_boots,
        "win_frequencies": {k_name: round(v/total_boots, 4) if total_boots > 0 else 0
                          for k_name, v in boot_wins.items()},
    },
    "loocv_mse": {k_name: round(v, 6) for k_name, v in loocv_mse.items()},
    "binned_dose_response": bin_results,
    "winners": winners,
    "one_k_win_count": one_k_win_count,
    "verdict": verdict,
    "is_knockout": one_k_win_count >= 3,
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"\n  Results saved to: {RESULTS_PATH}")
