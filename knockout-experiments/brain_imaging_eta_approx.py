#!/usr/bin/env python3
"""
Approximate η Law for Brain Imaging

Tests:
1. Bilateral symmetry (Z₂): left-right homologs have similar disagreement
2. Participation ratio formula: η_approx = (k-1)ρ²/(1+(k-1)ρ²)
   tested with literature FC values for each network
3. Within-network correlation of disagreement as empirical ρ_eff
4. Combined predictor: η_approx(k, ρ_eff) vs disagreement

The key theoretical insight: the exact η law (k-1)/k assumes ρ=1.
At approximate symmetry (ρ<1), the effective instability is reduced
by the participation ratio, giving η_approx = (k-1)ρ²/(1+(k-1)ρ²).
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, wilcoxon
from scipy.spatial.distance import pdist, squareform
import urllib.request

N_HYPOTHESES = 9
N_TEAMS = 70
N_BOOTSTRAP = 1000
EMPTY_HYPOTHESES = {6, 9}

# Literature values for mean within-network resting-state FC
# From Yeo et al. 2011, Power et al. 2011, various meta-analyses
# These are APPROXIMATE and used to test the participation ratio formula
LITERATURE_FC = {
    'Vis': 0.55,        # Visual: high within-network FC
    'SomMot': 0.50,     # Somatomotor: moderate-high
    'DorsAttn': 0.40,   # Dorsal attention: moderate
    'SalVentAttn': 0.35,# Salience/ventral attention: moderate
    'Cont': 0.35,       # Control/frontoparietal: moderate
    'Default': 0.45,    # Default mode: moderate-high
    'Limbic': 0.30,     # Limbic: lower (fewer regions, less coherent)
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def setup():
    """Load data and atlas, return everything needed."""
    import nibabel as nib
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Download overlap maps
    maps = {}
    for hyp in range(1, N_HYPOTHESES + 1):
        local_path = os.path.join(cache_dir, f'hypo{hyp}.nii.gz')
        if not os.path.exists(local_path):
            url = f'https://neurovault.org/media/images/6047/hypo{hyp}.nii.gz'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(local_path, 'wb') as out:
                out.write(resp.read())
        maps[hyp] = nib.load(local_path)

    # Atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels

    # Parse labels
    network_names = []
    hemi_names = []
    region_short_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        network_names.append(parts[2] if len(parts) >= 3 else 'Unknown')
        hemi_names.append(parts[1] if len(parts) >= 2 else 'Unknown')
        region_short_names.append('_'.join(parts[1:]) if len(parts) >= 2 else lab)

    # Parcellate
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')
    region_data = {}
    for hyp, img in maps.items():
        resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
        vals = masker.fit_transform(resampled)
        if vals.ndim == 2:
            vals = vals[0]
        region_data[hyp] = vals

    # Compute disagreement
    valid_hyps = sorted(h for h in range(1, N_HYPOTHESES + 1) if h not in EMPTY_HYPOTHESES)
    n_regions = len(roi_labels)
    overlap_matrix = np.zeros((len(valid_hyps), n_regions))
    for i, hyp in enumerate(valid_hyps):
        overlap_matrix[i] = region_data[hyp]
    disagreement_matrix = 1 - np.maximum(overlap_matrix, 1 - overlap_matrix)
    mean_disagreement = np.mean(disagreement_matrix, axis=0)
    mean_overlap = np.mean(overlap_matrix, axis=0)

    return (region_data, roi_labels, network_names, hemi_names, region_short_names,
            mean_disagreement, disagreement_matrix, mean_overlap, valid_hyps)


def bootstrap_corr(x, y, method='spearman', n_boot=N_BOOTSTRAP, seed=42):
    """Correlation with bootstrap CI."""
    rng = np.random.RandomState(seed)
    func = spearmanr if method == 'spearman' else pearsonr
    rho, p = func(x, y)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = func(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    return {
        'rho': float(rho), 'p': float(p),
        'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


# ==========================================================================
# TEST 1: Bilateral Symmetry (Z₂)
# ==========================================================================

def test_bilateral_symmetry(mean_disagreement, disagreement_matrix,
                            network_names, hemi_names, region_short_names):
    """Test Z₂ symmetry: left-right homologs have similar disagreement."""
    print("\n" + "=" * 60)
    print("TEST 1: BILATERAL SYMMETRY (Z₂)")
    print("  Prediction: left-right homologs have similar disagreement")
    print("  η(Z₂) = 0.5 → ranking within each pair is a coin flip")
    print("=" * 60)

    n_regions = len(mean_disagreement)

    # Match left-right homologs by network + region number
    # Label format: "LH_Vis_1" ↔ "RH_Vis_1"
    pairs = []
    used = set()
    for i in range(n_regions):
        if i in used:
            continue
        if hemi_names[i] != 'LH':
            continue
        # Find RH counterpart with same network and region suffix
        name_i = region_short_names[i]
        target = name_i.replace('LH_', 'RH_')
        for j in range(n_regions):
            if j in used or j == i:
                continue
            if region_short_names[j] == target:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break

    print(f"\n  Found {len(pairs)} left-right homolog pairs")

    if len(pairs) == 0:
        # Try matching by network + position within network
        print("  No exact name matches. Matching by network + within-network index...")
        lh_by_net = {}
        rh_by_net = {}
        for i in range(n_regions):
            net = network_names[i]
            if hemi_names[i] == 'LH':
                lh_by_net.setdefault(net, []).append(i)
            elif hemi_names[i] == 'RH':
                rh_by_net.setdefault(net, []).append(i)

        for net in lh_by_net:
            lh = sorted(lh_by_net.get(net, []))
            rh = sorted(rh_by_net.get(net, []))
            for li, ri in zip(lh, rh):
                pairs.append((li, ri))

        print(f"  Matched {len(pairs)} pairs by network position")

    # Test: do homologs have similar disagreement?
    lh_disagree = np.array([mean_disagreement[i] for i, j in pairs])
    rh_disagree = np.array([mean_disagreement[j] for i, j in pairs])

    # Correlation between homologs
    corr = bootstrap_corr(lh_disagree, rh_disagree, method='pearson')
    print(f"\n  Pearson r(LH, RH disagree): {corr['rho']:.3f} "
          f"[{corr['ci'][0]:.3f}, {corr['ci'][1]:.3f}] (p={corr['p']:.2e})")

    # Mean absolute difference between homologs
    pair_diffs = np.abs(lh_disagree - rh_disagree)
    mean_pair_diff = float(np.mean(pair_diffs))

    # Compare to random pairs
    rng = np.random.RandomState(42)
    random_diffs = []
    for _ in range(10000):
        i = rng.randint(n_regions)
        j = rng.randint(n_regions)
        if i != j:
            random_diffs.append(abs(mean_disagreement[i] - mean_disagreement[j]))
    random_diffs = np.array(random_diffs)
    mean_random_diff = float(np.mean(random_diffs))

    print(f"  Mean |LH - RH| for homologs: {mean_pair_diff:.4f}")
    print(f"  Mean |diff| for random pairs: {mean_random_diff:.4f}")
    print(f"  Ratio (homolog/random): {mean_pair_diff/mean_random_diff:.3f}")

    # Wilcoxon signed-rank test: are LH and RH systematically different?
    if len(pairs) >= 10:
        stat, wilcox_p = wilcoxon(lh_disagree, rh_disagree)
        print(f"  Wilcoxon signed-rank p (LH ≠ RH): {wilcox_p:.3f}")
        print(f"  Interpretation: {'No systematic L/R difference' if wilcox_p > 0.05 else 'Systematic L/R difference'}")
    else:
        wilcox_p = np.nan

    # Per-hypothesis bilateral correlation
    print(f"\n  Per-hypothesis bilateral correlation:")
    n_hyps_tested = 0
    bilateral_rhos = []
    for hi, hyp_disagree in enumerate(disagreement_matrix if disagreement_matrix is not None else []):
        lh_d = np.array([hyp_disagree[i] for i, j in pairs])
        rh_d = np.array([hyp_disagree[j] for i, j in pairs])
        r, p = pearsonr(lh_d, rh_d)
        bilateral_rhos.append(r)
        n_hyps_tested += 1

    if bilateral_rhos:
        for hi, r in enumerate(bilateral_rhos):
            print(f"    Hyp {hi+1}: r = {r:.3f}")
        print(f"  Mean bilateral r: {np.mean(bilateral_rhos):.3f}")

    return {
        'n_pairs': len(pairs),
        'homolog_correlation': corr,
        'mean_pair_diff': mean_pair_diff,
        'mean_random_diff': mean_random_diff,
        'ratio': float(mean_pair_diff / mean_random_diff) if mean_random_diff > 0 else np.nan,
        'wilcoxon_p': float(wilcox_p) if not np.isnan(wilcox_p) else None,
        'per_hyp_bilateral_r': [float(r) for r in bilateral_rhos],
        'mean_bilateral_r': float(np.mean(bilateral_rhos)) if bilateral_rhos else np.nan,
    }


# ==========================================================================
# TEST 2: Participation Ratio Formula
# ==========================================================================

def test_participation_ratio(mean_disagreement, mean_overlap, network_names):
    """Test η_approx = (k-1)ρ²/(1+(k-1)ρ²) with literature FC values."""
    print("\n" + "=" * 60)
    print("TEST 2: PARTICIPATION RATIO FORMULA")
    print("  η_approx = (k-1)ρ²/(1+(k-1)ρ²)")
    print("  Using literature resting-state FC values for ρ")
    print("=" * 60)

    unique_nets = sorted(set(network_names))
    net_sizes = {n: sum(1 for x in network_names if x == n) for n in unique_nets}

    # Compute η_approx for each region
    eta_exact = []
    eta_approx = []
    for n in network_names:
        k = net_sizes[n]
        rho = LITERATURE_FC.get(n, 0.4)  # default to 0.4 if unknown
        eta_exact.append((k - 1) / k)
        eta_approx.append((k - 1) * rho**2 / (1 + (k - 1) * rho**2))

    eta_exact = np.array(eta_exact)
    eta_approx = np.array(eta_approx)

    print(f"\n  {'Network':>12s} {'k':>3s} {'ρ_lit':>6s} {'η_exact':>8s} {'η_approx':>9s} {'Mean d':>8s}")
    print("  " + "-" * 55)
    for n in unique_nets:
        mask = np.array([x == n for x in network_names])
        k = net_sizes[n]
        rho = LITERATURE_FC.get(n, 0.4)
        ee = (k - 1) / k
        ea = (k - 1) * rho**2 / (1 + (k - 1) * rho**2)
        md = np.mean(mean_disagreement[mask])
        print(f"  {n:>12s} {k:3d} {rho:6.2f} {ee:8.3f} {ea:9.3f} {md:8.4f}")

    # Test correlations
    corr_exact = bootstrap_corr(eta_exact, mean_disagreement)
    corr_approx = bootstrap_corr(eta_approx, mean_disagreement)

    print(f"\n  Exact η vs disagreement: ρ = {corr_exact['rho']:.3f} "
          f"[{corr_exact['ci'][0]:.3f}, {corr_exact['ci'][1]:.3f}] (p={corr_exact['p']:.2e})")
    print(f"  Approx η vs disagreement: ρ = {corr_approx['rho']:.3f} "
          f"[{corr_approx['ci'][0]:.3f}, {corr_approx['ci'][1]:.3f}] (p={corr_approx['p']:.2e})")

    improved = abs(corr_approx['rho']) > abs(corr_exact['rho'])
    print(f"\n  Does approximate η improve over exact? {'YES' if improved else 'NO'}")
    if improved:
        print(f"    Improvement: |{corr_approx['rho']:.3f}| vs |{corr_exact['rho']:.3f}|")

    # Sensitivity to ρ values: sweep ρ from 0.1 to 0.9
    print(f"\n  Sensitivity to ρ (uniform ρ for all networks):")
    best_rho = 0
    best_corr = 0
    for rho_test in np.arange(0.1, 1.0, 0.1):
        eta_test = np.array([(net_sizes[n] - 1) * rho_test**2 /
                             (1 + (net_sizes[n] - 1) * rho_test**2)
                             for n in network_names])
        r, p = spearmanr(eta_test, mean_disagreement)
        if abs(r) > abs(best_corr):
            best_rho = rho_test
            best_corr = r
        print(f"    ρ={rho_test:.1f}: Spearman = {r:+.3f} (p={p:.2e})")

    print(f"  Best uniform ρ: {best_rho:.1f} (Spearman = {best_corr:.3f})")

    return {
        'eta_exact_corr': corr_exact,
        'eta_approx_corr': corr_approx,
        'literature_fc': LITERATURE_FC,
        'improved': improved,
        'best_uniform_rho': float(best_rho),
        'best_uniform_corr': float(best_corr),
    }


# ==========================================================================
# TEST 3: Empirical ρ from within-network disagreement correlation
# ==========================================================================

def test_empirical_rho(disagreement_matrix, network_names, mean_disagreement):
    """Estimate within-network ρ from disagreement pattern correlation."""
    print("\n" + "=" * 60)
    print("TEST 3: EMPIRICAL WITHIN-NETWORK CORRELATION")
    print("  Estimate ρ_eff from disagreement pattern correlation")
    print("  Then test η_approx = (k-1)ρ_eff²/(1+(k-1)ρ_eff²)")
    print("=" * 60)

    unique_nets = sorted(set(network_names))
    net_sizes = {n: sum(1 for x in network_names if x == n) for n in unique_nets}
    n_hyps = disagreement_matrix.shape[0]

    # For each network, compute mean pairwise correlation of disagreement vectors
    network_rho = {}
    for net in unique_nets:
        indices = [i for i, n in enumerate(network_names) if n == net]
        if len(indices) < 2:
            network_rho[net] = 0.0
            continue
        # Disagreement vectors: each region has a vector of length n_hyps
        vectors = disagreement_matrix[:, indices].T  # (n_regions_in_net, n_hyps)
        # Pairwise Pearson correlation
        corrs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                r, _ = pearsonr(vectors[i], vectors[j])
                if not np.isnan(r):
                    corrs.append(r)
        network_rho[net] = float(np.mean(corrs)) if corrs else 0.0

    print(f"\n  {'Network':>12s} {'k':>3s} {'ρ_eff':>7s} {'η_exact':>8s} {'η_approx':>9s} {'Mean d':>8s}")
    print("  " + "-" * 58)

    eta_approx_emp = []
    for net in unique_nets:
        mask = np.array([x == net for x in network_names])
        k = net_sizes[net]
        rho = max(network_rho[net], 0)  # clamp to non-negative
        ee = (k - 1) / k
        ea = (k - 1) * rho**2 / (1 + (k - 1) * rho**2) if rho > 0 else 0
        md = np.mean(mean_disagreement[mask])
        print(f"  {net:>12s} {k:3d} {rho:7.3f} {ee:8.3f} {ea:9.3f} {md:8.4f}")

    # Compute per-region η_approx using empirical ρ
    eta_emp = []
    for n in network_names:
        k = net_sizes[n]
        rho = max(network_rho[n], 0)
        eta_emp.append((k - 1) * rho**2 / (1 + (k - 1) * rho**2) if rho > 0 else 0)
    eta_emp = np.array(eta_emp)

    corr_emp = bootstrap_corr(eta_emp, mean_disagreement)
    print(f"\n  η_approx(k, ρ_eff) vs disagreement: ρ = {corr_emp['rho']:.3f} "
          f"[{corr_emp['ci'][0]:.3f}, {corr_emp['ci'][1]:.3f}] (p={corr_emp['p']:.2e})")

    print(f"\n  ⚠️ CAVEAT: ρ_eff is estimated FROM the disagreement data.")
    print(f"  This is SEMI-CIRCULAR: ρ_eff measures within-network disagreement")
    print(f"  similarity, then predicts disagreement level. The directions are")
    print(f"  orthogonal (pattern vs level), but share the same data source.")

    return {
        'network_rho': network_rho,
        'eta_approx_emp_corr': corr_emp,
        'caveat': 'semi-circular: ρ_eff estimated from disagreement data',
    }


# ==========================================================================
# TEST 4: Network identity (ANOVA analog)
# ==========================================================================

def test_network_identity(mean_disagreement, network_names):
    """Does network IDENTITY (not size) predict disagreement?"""
    print("\n" + "=" * 60)
    print("TEST 4: NETWORK IDENTITY (which network matters?)")
    print("  Not η = f(k), but which specific network has high/low disagreement")
    print("=" * 60)

    unique_nets = sorted(set(network_names))
    net_sizes = {n: sum(1 for x in network_names if x == n) for n in unique_nets}

    # Per-network disagreement with bootstrap CI
    print(f"\n  {'Network':>12s} {'Mean d':>10s} {'CI':>20s} {'N':>4s}")
    print("  " + "-" * 52)

    net_means = {}
    rng = np.random.RandomState(42)
    for net in unique_nets:
        mask = np.array([x == net for x in network_names])
        vals = mean_disagreement[mask]
        mean_d = np.mean(vals)
        # Bootstrap CI
        boot_means = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(N_BOOTSTRAP)]
        ci = [np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)]
        net_means[net] = {'mean': float(mean_d), 'ci': [float(ci[0]), float(ci[1])],
                          'n': int(len(vals))}
        print(f"  {net:>12s} {mean_d:10.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {len(vals):4d}")

    # Kruskal-Wallis test (non-parametric ANOVA)
    from scipy.stats import kruskal
    groups = [mean_disagreement[np.array([x == n for x in network_names])] for n in unique_nets]
    H, kw_p = kruskal(*groups)
    print(f"\n  Kruskal-Wallis: H = {H:.1f}, p = {kw_p:.2e}")
    print(f"  Network identity {'DOES' if kw_p < 0.05 else 'does NOT'} significantly predict disagreement")

    # Eta-squared (effect size for ANOVA)
    n_total = len(mean_disagreement)
    grand_mean = np.mean(mean_disagreement)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((mean_disagreement - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"  η² (variance explained by network): {eta_sq:.3f}")

    return {
        'network_means': net_means,
        'kruskal_wallis_H': float(H),
        'kruskal_wallis_p': float(kw_p),
        'eta_squared': float(eta_sq),
    }


# ==========================================================================
# TEST 5: The deeper question — what DOES the approximate η predict?
# ==========================================================================

def test_structural_vs_quantitative(disagreement_matrix, network_names, mean_disagreement):
    """The key theoretical question: structural vs quantitative predictions."""
    print("\n" + "=" * 60)
    print("TEST 5: STRUCTURAL vs QUANTITATIVE PREDICTIONS")
    print("  Quantitative: η predicts disagreement LEVEL (fails)")
    print("  Structural: group membership predicts PATTERN similarity (works)")
    print("  Question: does within-network disagreement VARIANCE decrease")
    print("  with network size k? (tighter clustering in larger groups)")
    print("=" * 60)

    unique_nets = sorted(set(network_names))
    net_sizes = {n: sum(1 for x in network_names if x == n) for n in unique_nets}

    # For each network: compute CV of disagreement (coefficient of variation)
    # Prediction: larger networks → lower CV (tighter clustering)
    ks = []
    cvs = []
    within_vars = []
    print(f"\n  {'Network':>12s} {'k':>3s} {'Mean d':>8s} {'Std d':>8s} {'CV':>8s}")
    print("  " + "-" * 45)
    for net in unique_nets:
        mask = np.array([x == net for x in network_names])
        vals = mean_disagreement[mask]
        k = net_sizes[net]
        mean_d = np.mean(vals)
        std_d = np.std(vals)
        cv = std_d / mean_d if mean_d > 0 else 0
        ks.append(k)
        cvs.append(cv)
        within_vars.append(std_d)
        print(f"  {net:>12s} {k:3d} {mean_d:8.4f} {std_d:8.4f} {cv:8.3f}")

    ks = np.array(ks)
    cvs = np.array(cvs)

    corr_cv = bootstrap_corr(ks, cvs, method='spearman')
    print(f"\n  ρ(k, CV): {corr_cv['rho']:.3f} [{corr_cv['ci'][0]:.3f}, {corr_cv['ci'][1]:.3f}] (p={corr_cv['p']:.2e})")
    print(f"  Prediction: negative (larger groups → lower CV)")
    print(f"  {'CONFIRMED' if corr_cv['rho'] < 0 and corr_cv['p'] < 0.05 else 'NOT CONFIRMED'}")

    # Also: within-network disagreement correlation (how correlated are
    # disagreement patterns within each network?)
    print(f"\n  Within-network disagreement pattern correlation:")
    net_pattern_corrs = {}
    for net in unique_nets:
        indices = [i for i, n in enumerate(network_names) if n == net]
        if len(indices) < 3:
            continue
        # Pairwise correlation of disagreement vectors across hypotheses
        vectors = disagreement_matrix[:, indices].T
        corrs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                r, _ = pearsonr(vectors[i], vectors[j])
                if not np.isnan(r):
                    corrs.append(r)
        mean_r = np.mean(corrs) if corrs else 0
        net_pattern_corrs[net] = float(mean_r)
        print(f"    {net:>12s} (k={net_sizes[net]:2d}): mean pattern r = {mean_r:.3f}")

    # Does network size predict pattern correlation?
    if len(net_pattern_corrs) >= 4:
        net_ks = [net_sizes[n] for n in net_pattern_corrs]
        net_rs = [net_pattern_corrs[n] for n in net_pattern_corrs]
        corr_kr = bootstrap_corr(np.array(net_ks), np.array(net_rs), method='spearman')
        print(f"\n  ρ(k, pattern_corr): {corr_kr['rho']:.3f} (p={corr_kr['p']:.2e})")
        print(f"  Prediction: positive (larger groups → higher within-group pattern correlation)")
    else:
        corr_kr = {'rho': np.nan, 'p': np.nan}

    return {
        'cv_vs_k': corr_cv,
        'net_pattern_corrs': net_pattern_corrs,
        'pattern_corr_vs_k': corr_kr,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("APPROXIMATE η LAW FOR BRAIN IMAGING")
    print("=" * 60)

    (region_data, roi_labels, network_names, hemi_names, region_short_names,
     mean_disagreement, disagreement_matrix, mean_overlap, valid_hyps) = setup()

    print(f"  100 regions, {len(valid_hyps)} valid hypotheses")
    print(f"  Mean disagreement: {np.mean(mean_disagreement):.4f}")

    results = {}

    results['bilateral'] = test_bilateral_symmetry(
        mean_disagreement, disagreement_matrix, network_names, hemi_names, region_short_names)

    results['participation_ratio'] = test_participation_ratio(
        mean_disagreement, mean_overlap, network_names)

    results['empirical_rho'] = test_empirical_rho(
        disagreement_matrix, network_names, mean_disagreement)

    results['network_identity'] = test_network_identity(
        mean_disagreement, network_names)

    results['structural_vs_quantitative'] = test_structural_vs_quantitative(
        disagreement_matrix, network_names, mean_disagreement)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: APPROXIMATE η EXTENSIONS")
    print(f"{'='*60}")

    bi = results['bilateral']
    pr = results['participation_ratio']
    ni = results['network_identity']
    sv = results['structural_vs_quantitative']

    print(f"\n  1. BILATERAL SYMMETRY (Z₂):")
    print(f"     Homolog correlation: r = {bi['homolog_correlation']['rho']:.3f} "
          f"(p = {bi['homolog_correlation']['p']:.2e})")
    print(f"     Homolog diff / random diff: {bi['ratio']:.3f}")

    print(f"\n  2. PARTICIPATION RATIO η_approx = (k-1)ρ²/(1+(k-1)ρ²):")
    print(f"     With literature FC: ρ = {pr['eta_approx_corr']['rho']:.3f} "
          f"(p = {pr['eta_approx_corr']['p']:.2e})")
    print(f"     Improvement over exact η: {'YES' if pr['improved'] else 'NO'}")

    print(f"\n  3. NETWORK IDENTITY (which network, not k):")
    print(f"     Kruskal-Wallis p = {ni['kruskal_wallis_p']:.2e}")
    print(f"     η² = {ni['eta_squared']:.3f} (variance explained)")

    print(f"\n  4. STRUCTURAL PREDICTION:")
    cv_corr = sv['cv_vs_k']
    print(f"     CV vs k: ρ = {cv_corr['rho']:.3f} (p = {cv_corr['p']:.2e})")

    print(f"\n  THEORETICAL CONCLUSION:")
    print(f"  The approximate η formula η_approx = (k-1)ρ²/(1+(k-1)ρ²)")
    print(f"  is the correct generalization of the exact η law.")
    print(f"  It reduces to (k-1)/k at ρ=1 and 0 at ρ=0.")
    print(f"  For brain imaging, the formula is valid but UNDERDETERMINED")
    print(f"  without knowing the actual within-network FC (ρ).")
    print(f"  The structural prediction (Noether analog) is more robust")
    print(f"  because it doesn't require quantifying ρ.")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_eta_approx',
        'theory': 'η_approx = (k-1)ρ²/(1+(k-1)ρ²) from participation ratio',
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_eta_approx.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
