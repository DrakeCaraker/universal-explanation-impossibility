#!/usr/bin/env python3
"""
Brain Imaging Rashomon v3: Comprehensive Expansion

Pivotal questions from v2:
Q1. Was η test underpowered? (η range only 0.80–0.96)
    → Test with hemisphere × network groups (η range 0.50–0.92)
Q2. Is the Noether result (d=0.43) trivial?
    → Random grouping null with matched group sizes
Q3. Is the Noether result just spatial autocorrelation?
    → k-means on centroids as spatial grouping control
Q4. Is the result consistent across hypotheses?
    → Per-hypothesis decomposition
Q5. Is the result sensitive to parcellation scale?
    → Schaefer-400 replication

All tests use bootstrap CIs. Reports effect sizes regardless of significance.
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import urllib.request

N_HYPOTHESES = 9
N_TEAMS = 70
N_BOOTSTRAP = 1000
EMPTY_HYPOTHESES = {6, 9}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def download_overlap_maps():
    """Download the 9 hypothesis overlap maps from NeuroVault."""
    import nibabel as nib
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_cache')
    os.makedirs(cache_dir, exist_ok=True)
    maps = {}
    for hyp in range(1, N_HYPOTHESES + 1):
        local_path = os.path.join(cache_dir, f'hypo{hyp}.nii.gz')
        if not os.path.exists(local_path):
            url = f'https://neurovault.org/media/images/6047/hypo{hyp}.nii.gz'
            print(f'  Downloading hypothesis {hyp}...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(local_path, 'wb') as out:
                out.write(resp.read())
        maps[hyp] = nib.load(local_path)
    return maps


def setup_atlas(n_rois=100):
    """Load Schaefer atlas, extract labels, networks, centroids."""
    from nilearn import datasets
    import nibabel as nib

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels

    # Skip background label
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels

    # Parse network and hemisphere from labels
    network_names = []
    hemi_network_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        if len(parts) >= 3:
            network_names.append(parts[2])
            hemi_network_names.append(f"{parts[1]}_{parts[2]}")
        else:
            network_names.append('Unknown')
            hemi_network_names.append('Unknown')

    # Compute centroids
    atlas_nii = nib.load(atlas_img) if isinstance(atlas_img, str) else atlas_img
    atlas_data = atlas_nii.get_fdata()
    affine = atlas_nii.affine

    centroids = []
    for roi_id in range(1, len(roi_labels) + 1):
        voxels = np.argwhere(atlas_data == roi_id)
        if len(voxels) > 0:
            centroid_mni = affine[:3, :3] @ voxels.mean(axis=0) + affine[:3, 3]
            centroids.append(centroid_mni)
        else:
            centroids.append(np.array([0, 0, 0]))
    centroids = np.array(centroids)

    return atlas_img, roi_labels, network_names, hemi_network_names, centroids


def parcellate(overlap_maps, atlas_img):
    """Extract per-region overlap fractions."""
    from nilearn import image
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')
    region_data = {}
    for hyp, img in overlap_maps.items():
        resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
        vals = masker.fit_transform(resampled)
        if vals.ndim == 2:
            vals = vals[0]
        region_data[hyp] = vals
    return region_data


def compute_disagreement(region_data):
    """Compute disagreement excluding empty hypotheses."""
    valid_hyps = sorted(h for h in region_data if h not in EMPTY_HYPOTHESES)
    n_regions = len(region_data[valid_hyps[0]])

    overlap_matrix = np.zeros((len(valid_hyps), n_regions))
    for i, hyp in enumerate(valid_hyps):
        overlap_matrix[i] = region_data[hyp]

    disagreement_matrix = 1 - np.maximum(overlap_matrix, 1 - overlap_matrix)
    mean_disagreement = np.mean(disagreement_matrix, axis=0)
    mean_overlap = np.mean(overlap_matrix, axis=0)

    return mean_disagreement, disagreement_matrix, mean_overlap, valid_hyps


def make_grouping(names):
    """Convert list of group names to integer assignments and size dict."""
    unique = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(unique)}
    assignments = np.array([name_to_id[n] for n in names])
    sizes = {n: sum(1 for x in names if x == n) for n in unique}
    return assignments, sizes, unique


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, seed=42):
    """Spearman with bootstrap CI."""
    rng = np.random.RandomState(seed)
    rho, p = spearmanr(x, y)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    return {
        'rho': float(rho), 'p': float(p),
        'ci_lo': float(np.percentile(boots, 2.5)) if len(boots) > 0 else np.nan,
        'ci_hi': float(np.percentile(boots, 97.5)) if len(boots) > 0 else np.nan,
    }


def partial_spearman(x, y, z, n_boot=N_BOOTSTRAP, seed=42):
    """Partial Spearman controlling for z, with bootstrap CI."""
    from scipy.stats import rankdata
    def residualize(r, rz):
        A = np.column_stack([rz, np.ones(len(rz))])
        return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    rx_res = residualize(rankdata(x), rankdata(z))
    ry_res = residualize(rankdata(y), rankdata(z))
    rho, p = spearmanr(rx_res, ry_res)

    rng = np.random.RandomState(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rx_b = residualize(rankdata(x[idx]), rankdata(z[idx]))
        ry_b = residualize(rankdata(y[idx]), rankdata(z[idx]))
        r, _ = spearmanr(rx_b, ry_b)
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    return {
        'rho': float(rho), 'p': float(p),
        'ci_lo': float(np.percentile(boots, 2.5)) if len(boots) > 0 else np.nan,
        'ci_hi': float(np.percentile(boots, 97.5)) if len(boots) > 0 else np.nan,
    }


def within_between_test(disagreement_matrix, assignments):
    """Within-group vs between-group disagreement similarity."""
    n_regions = disagreement_matrix.shape[1]
    within = []
    between = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            diff = np.mean(np.abs(disagreement_matrix[:, i] - disagreement_matrix[:, j]))
            if assignments[i] == assignments[j]:
                within.append(diff)
            else:
                between.append(diff)

    within = np.array(within)
    between = np.array(between)

    if len(within) < 2 or len(between) < 2:
        return {'mean_within': np.nan, 'mean_between': np.nan, 'cohens_d': np.nan,
                'p': 1.0, 'n_within': len(within), 'n_between': len(between)}

    mean_w = float(np.mean(within))
    mean_b = float(np.mean(between))
    pooled = np.sqrt((np.var(within) * len(within) + np.var(between) * len(between)) /
                     (len(within) + len(between)))
    d = (mean_b - mean_w) / pooled if pooled > 0 else 0
    U, p = mannwhitneyu(within, between, alternative='less')

    # Bootstrap CI on difference
    rng = np.random.RandomState(42)
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        w_b = rng.choice(within, len(within), replace=True)
        b_b = rng.choice(between, len(between), replace=True)
        boot_diffs.append(np.mean(b_b) - np.mean(w_b))
    boot_diffs = np.array(boot_diffs)

    return {
        'mean_within': mean_w, 'mean_between': mean_b,
        'difference': float(mean_b - mean_w),
        'diff_ci': [float(np.percentile(boot_diffs, 2.5)),
                    float(np.percentile(boot_diffs, 97.5))],
        'cohens_d': float(d), 'p': float(p),
        'n_within': len(within), 'n_between': len(between),
    }


# ==========================================================================
# PHASE 1: Expand η range with hemisphere × network
# ==========================================================================

def phase1_hemisphere_eta(mean_disagreement, mean_overlap, hemi_network_names):
    """Test η law with hemisphere × network grouping (wider η range)."""
    print("\n" + "=" * 60)
    print("PHASE 1: HEMISPHERE × NETWORK GROUPING (wider η range)")
    print("=" * 60)

    assignments, sizes, unique = make_grouping(hemi_network_names)
    eta = np.array([(sizes[n] - 1) / sizes[n] for n in hemi_network_names])

    print(f"\n  {len(unique)} groups, η range: [{np.min(eta):.3f}, {np.max(eta):.3f}]")
    print(f"  (vs 7-network: [0.800, 0.958])")

    print(f"\n  {'Group':>16s} {'k':>3s} {'η':>6s} {'Mean disagree':>14s}")
    print("  " + "-" * 45)
    for g in unique:
        mask = np.array([n == g for n in hemi_network_names])
        k = sizes[g]
        print(f"  {g:>16s} {k:3d} {(k-1)/k:6.3f} {np.mean(mean_disagreement[mask]):14.4f}")

    raw = bootstrap_spearman(eta, mean_disagreement)
    partial = partial_spearman(eta, mean_disagreement, mean_overlap)

    print(f"\n  Raw: ρ(η, disagree) = {raw['rho']:.3f} [{raw['ci_lo']:.3f}, {raw['ci_hi']:.3f}] (p={raw['p']:.2e})")
    print(f"  Partial: ρ = {partial['rho']:.3f} [{partial['ci_lo']:.3f}, {partial['ci_hi']:.3f}] (p={partial['p']:.2e})")

    return {
        'n_groups': len(unique),
        'eta_range': [float(np.min(eta)), float(np.max(eta))],
        'group_sizes': {g: sizes[g] for g in unique},
        'raw': raw, 'partial': partial,
    }


# ==========================================================================
# PHASE 2: Random grouping null
# ==========================================================================

def phase2_random_null(disagreement_matrix, network_names, n_perm=1000):
    """Is the within-network similarity trivial? Test with random groupings."""
    print("\n" + "=" * 60)
    print("PHASE 2: RANDOM GROUPING NULL")
    print("  If random groups show similar d, the Noether result is trivial.")
    print("=" * 60)

    # Observed result with real networks
    assignments_real, _, _ = make_grouping(network_names)
    observed = within_between_test(disagreement_matrix, assignments_real)
    observed_d = observed['cohens_d']

    print(f"\n  Observed (Yeo networks): d = {observed_d:.3f}")

    # Generate random groupings matching the size distribution
    _, sizes, unique_nets = make_grouping(network_names)
    group_size_list = [sizes[n] for n in unique_nets]
    n_regions = len(network_names)

    rng = np.random.RandomState(42)
    null_ds = []
    for _ in range(n_perm):
        # Random permutation of region indices, then assign to groups of same sizes
        perm = rng.permutation(n_regions)
        rand_assignments = np.zeros(n_regions, dtype=int)
        start = 0
        for gi, gs in enumerate(group_size_list):
            rand_assignments[perm[start:start + gs]] = gi
            start += gs
        result = within_between_test(disagreement_matrix, rand_assignments)
        null_ds.append(result['cohens_d'])

    null_ds = np.array(null_ds)
    perm_p = float(np.mean(null_ds >= observed_d))

    print(f"  Random null: d = {np.mean(null_ds):.3f} ± {np.std(null_ds):.3f}")
    print(f"  Permutation p (observed ≥ null): {perm_p:.3f}")
    print(f"  95th percentile of null: {np.percentile(null_ds, 95):.3f}")

    if perm_p < 0.05:
        print(f"  → Yeo networks show MORE within-group similarity than random groupings.")
        print(f"    The Noether result is NOT trivial.")
    else:
        print(f"  → Random groupings show similar within-group similarity.")
        print(f"    The Noether result IS trivial (any grouping would show it).")

    return {
        'observed_d': float(observed_d),
        'null_mean': float(np.mean(null_ds)),
        'null_std': float(np.std(null_ds)),
        'null_95th': float(np.percentile(null_ds, 95)),
        'permutation_p': perm_p,
        'trivial': perm_p >= 0.05,
    }


# ==========================================================================
# PHASE 3: Spatial autocorrelation control
# ==========================================================================

def phase3_spatial_control(disagreement_matrix, centroids, network_names):
    """Is within-network similarity just spatial autocorrelation?"""
    print("\n" + "=" * 60)
    print("PHASE 3: SPATIAL AUTOCORRELATION CONTROL")
    print("  Does distance-based grouping match network-based?")
    print("=" * 60)

    n_regions = len(centroids)
    assignments_net, _, _ = make_grouping(network_names)
    observed_net = within_between_test(disagreement_matrix, assignments_net)

    results = {'network_d': float(observed_net['cohens_d'])}
    spatial_ds = {}

    for k in [5, 7, 10, 14]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        spatial_assignments = km.fit_predict(centroids)
        spatial_result = within_between_test(disagreement_matrix, spatial_assignments)
        spatial_ds[k] = float(spatial_result['cohens_d'])
        print(f"\n  k-means (k={k}): d = {spatial_result['cohens_d']:.3f} "
              f"(p = {spatial_result['p']:.2e})")

    print(f"\n  Network (Yeo, k=7): d = {observed_net['cohens_d']:.3f}")
    print(f"\n  Comparison:")
    for k, d in spatial_ds.items():
        better = "NETWORK BETTER" if observed_net['cohens_d'] > d else "SPATIAL BETTER"
        print(f"    k-means k={k}: d={d:.3f} vs network d={observed_net['cohens_d']:.3f} → {better}")

    results['spatial_ds'] = spatial_ds
    net_beats_spatial = all(observed_net['cohens_d'] > d for d in spatial_ds.values())
    results['network_beats_all_spatial'] = net_beats_spatial

    if net_beats_spatial:
        print(f"\n  → Functional networks outperform ALL spatial groupings.")
        print(f"    The Noether result is NOT just spatial autocorrelation.")
    else:
        best_spatial_k = max(spatial_ds, key=spatial_ds.get)
        print(f"\n  → Spatial grouping (k={best_spatial_k}) matches or beats networks.")
        print(f"    Spatial autocorrelation may explain the within-group similarity.")

    return results


# ==========================================================================
# PHASE 4: Per-hypothesis consistency
# ==========================================================================

def phase4_per_hypothesis(region_data, network_names, valid_hyps):
    """Is the Noether result consistent across hypotheses?"""
    print("\n" + "=" * 60)
    print("PHASE 4: PER-HYPOTHESIS CONSISTENCY")
    print("=" * 60)

    assignments, _, _ = make_grouping(network_names)
    n_regions = len(network_names)

    per_hyp = {}
    print(f"\n  {'Hyp':>4s} {'d':>8s} {'p':>12s} {'within':>8s} {'between':>8s}")
    print("  " + "-" * 50)

    for hyp in valid_hyps:
        overlap = region_data[hyp]
        disagree = 1 - np.maximum(overlap, 1 - overlap)
        # Use single-hypothesis disagreement as a 1-row matrix
        disagree_mat = disagree.reshape(1, -1)

        within = []
        between = []
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                diff = abs(disagree[i] - disagree[j])
                if assignments[i] == assignments[j]:
                    within.append(diff)
                else:
                    between.append(diff)

        within = np.array(within)
        between = np.array(between)
        mean_w = np.mean(within)
        mean_b = np.mean(between)
        pooled = np.sqrt((np.var(within) * len(within) + np.var(between) * len(between)) /
                         (len(within) + len(between)))
        d = (mean_b - mean_w) / pooled if pooled > 0 else 0
        U, p = mannwhitneyu(within, between, alternative='less')

        per_hyp[hyp] = {'d': float(d), 'p': float(p),
                        'mean_within': float(mean_w), 'mean_between': float(mean_b)}
        print(f"  {hyp:4d} {d:8.3f} {p:12.2e} {mean_w:8.4f} {mean_b:8.4f}")

    n_consistent = sum(1 for h in valid_hyps if per_hyp[h]['d'] > 0 and per_hyp[h]['p'] < 0.05)
    n_positive = sum(1 for h in valid_hyps if per_hyp[h]['d'] > 0)
    print(f"\n  {n_consistent}/{len(valid_hyps)} significant (p < 0.05)")
    print(f"  {n_positive}/{len(valid_hyps)} positive direction (within < between)")

    return per_hyp


# ==========================================================================
# PHASE 5: Finer parcellation (Schaefer-400)
# ==========================================================================

def phase5_schaefer400(overlap_maps):
    """Replicate all key tests with Schaefer-400 parcellation."""
    print("\n" + "=" * 60)
    print("PHASE 5: SCHAEFER-400 REPLICATION")
    print("=" * 60)

    atlas_img, roi_labels, net_names, hemi_net_names, centroids = setup_atlas(n_rois=400)
    print(f"  {len(roi_labels)} regions")

    region_data = parcellate(overlap_maps, atlas_img)
    mean_disagree, disagree_matrix, mean_overlap, valid_hyps = compute_disagreement(region_data)

    # Group info
    assignments_net, sizes_net, unique_net = make_grouping(net_names)
    assignments_hemi, sizes_hemi, unique_hemi = make_grouping(hemi_net_names)

    # η law — 7 networks
    eta_7 = np.array([(sizes_net[n] - 1) / sizes_net[n] for n in net_names])
    raw_7 = bootstrap_spearman(eta_7, mean_disagree)
    partial_7 = partial_spearman(eta_7, mean_disagree, mean_overlap)
    print(f"\n  7-network η law:")
    print(f"    Raw: ρ = {raw_7['rho']:.3f} [{raw_7['ci_lo']:.3f}, {raw_7['ci_hi']:.3f}] (p={raw_7['p']:.2e})")
    print(f"    Partial: ρ = {partial_7['rho']:.3f} [{partial_7['ci_lo']:.3f}, {partial_7['ci_hi']:.3f}]")

    # η law — hemisphere × network
    eta_hemi = np.array([(sizes_hemi[n] - 1) / sizes_hemi[n] for n in hemi_net_names])
    raw_hemi = bootstrap_spearman(eta_hemi, mean_disagree)
    partial_hemi = partial_spearman(eta_hemi, mean_disagree, mean_overlap)
    print(f"\n  Hemisphere×network η law:")
    print(f"    η range: [{np.min(eta_hemi):.3f}, {np.max(eta_hemi):.3f}]")
    print(f"    Raw: ρ = {raw_hemi['rho']:.3f} [{raw_hemi['ci_lo']:.3f}, {raw_hemi['ci_hi']:.3f}] (p={raw_hemi['p']:.2e})")
    print(f"    Partial: ρ = {partial_hemi['rho']:.3f} [{partial_hemi['ci_lo']:.3f}, {partial_hemi['ci_hi']:.3f}]")

    # Within-vs-between (Noether)
    noether_7 = within_between_test(disagree_matrix, assignments_net)
    noether_hemi = within_between_test(disagree_matrix, assignments_hemi)
    print(f"\n  Noether (7-network): d = {noether_7['cohens_d']:.3f} (p = {noether_7['p']:.2e})")
    print(f"  Noether (hemi×net): d = {noether_hemi['cohens_d']:.3f} (p = {noether_hemi['p']:.2e})")

    return {
        'n_regions': len(roi_labels),
        'eta_7_raw': raw_7, 'eta_7_partial': partial_7,
        'eta_hemi_raw': raw_hemi, 'eta_hemi_partial': partial_hemi,
        'eta_hemi_range': [float(np.min(eta_hemi)), float(np.max(eta_hemi))],
        'noether_7': noether_7,
        'noether_hemi': noether_hemi,
    }


# ==========================================================================
# PHASE 6: Head-to-head summary
# ==========================================================================

def phase6_summary(all_results):
    """Print definitive head-to-head comparison."""
    print("\n" + "=" * 60)
    print("PHASE 6: HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    print("\n  η LAW (does network size predict disagreement level?)")
    print("  " + "-" * 55)
    tests = [
        ("Schaefer-100, 7 networks", "eta_100_7net"),
        ("Schaefer-100, 14 hemi×net", "phase1"),
        ("Schaefer-400, 7 networks", "s400_7"),
        ("Schaefer-400, 14 hemi×net", "s400_hemi"),
    ]
    for label, key in tests:
        if key in all_results:
            r = all_results[key]
            if 'raw' in r:
                raw = r['raw']
            elif 'eta_7_raw' in r:
                raw = r['eta_7_raw'] if '7' in label else r['eta_hemi_raw']
            else:
                continue
            sig = "✓" if raw['p'] < 0.05 else "✗"
            print(f"    {sig} {label:30s}: ρ = {raw['rho']:+.3f} [{raw['ci_lo']:.3f}, {raw['ci_hi']:.3f}]")

    print(f"\n  NOETHER ANALOG (within-group similarity)")
    print("  " + "-" * 55)
    noether_tests = [
        ("Yeo networks (Schaefer-100)", all_results.get('noether_100', {})),
        ("Yeo networks (Schaefer-400)", all_results.get('phase5', {}).get('noether_7', {})),
    ]
    for label, r in noether_tests:
        if r and 'cohens_d' in r:
            sig = "✓" if r['p'] < 0.05 else "✗"
            print(f"    {sig} {label:30s}: d = {r['cohens_d']:.3f} (p = {r['p']:.2e})")

    print(f"\n  CONTROLS")
    print("  " + "-" * 55)
    rn = all_results.get('phase2', {})
    if rn:
        print(f"    Random null: d = {rn.get('null_mean', 0):.3f} ± {rn.get('null_std', 0):.3f} "
              f"(perm p = {rn.get('permutation_p', 1):.3f})")
        verdict = "TRIVIAL" if rn.get('trivial', True) else "GENUINE"
        print(f"    → Noether result is {verdict}")

    sp = all_results.get('phase3', {})
    if sp:
        print(f"    Network d = {sp.get('network_d', 0):.3f} vs spatial: {sp.get('spatial_ds', {})}")
        verdict = "GENUINE" if sp.get('network_beats_all_spatial', False) else "SPATIAL"
        print(f"    → Network grouping is {verdict}")


def main():
    start = time.time()
    print("=" * 60)
    print("BRAIN IMAGING RASHOMON v3: COMPREHENSIVE EXPANSION")
    print("=" * 60)

    # Setup
    overlap_maps = download_overlap_maps()
    atlas_img, roi_labels, net_names, hemi_net_names, centroids = setup_atlas(100)
    region_data = parcellate(overlap_maps, atlas_img)
    mean_disagree, disagree_matrix, mean_overlap, valid_hyps = compute_disagreement(region_data)

    print(f"\n  Schaefer-100: {len(roi_labels)} regions, {len(valid_hyps)} valid hypotheses")
    print(f"  Mean disagreement: {np.mean(mean_disagree):.4f} ± {np.std(mean_disagree):.4f}")

    all_results = {}

    # Baseline: v2 η law with 7 networks (for comparison)
    assignments_7, sizes_7, _ = make_grouping(net_names)
    eta_7 = np.array([(sizes_7[n] - 1) / sizes_7[n] for n in net_names])
    all_results['eta_100_7net'] = {
        'raw': bootstrap_spearman(eta_7, mean_disagree),
        'partial': partial_spearman(eta_7, mean_disagree, mean_overlap),
    }
    print(f"\n  Baseline (7-net η): ρ = {all_results['eta_100_7net']['raw']['rho']:.3f}")

    # Noether baseline
    all_results['noether_100'] = within_between_test(disagree_matrix, assignments_7)
    print(f"  Baseline (Noether): d = {all_results['noether_100']['cohens_d']:.3f}")

    # Phase 1: Hemisphere × network
    all_results['phase1'] = phase1_hemisphere_eta(mean_disagree, mean_overlap, hemi_net_names)

    # Phase 2: Random null
    all_results['phase2'] = phase2_random_null(disagree_matrix, net_names)

    # Phase 3: Spatial control
    all_results['phase3'] = phase3_spatial_control(disagree_matrix, centroids, net_names)

    # Phase 4: Per-hypothesis
    all_results['phase4'] = phase4_per_hypothesis(region_data, net_names, valid_hyps)

    # Phase 5: Schaefer-400
    all_results['phase5'] = phase5_schaefer400(overlap_maps)

    # Phase 6: Head-to-head
    # Add Schaefer-400 results for comparison
    all_results['s400_7'] = {
        'raw': all_results['phase5']['eta_7_raw'],
        'partial': all_results['phase5']['eta_7_partial'],
    }
    all_results['s400_hemi'] = {
        'raw': all_results['phase5']['eta_hemi_raw'],
        'partial': all_results['phase5']['eta_hemi_partial'],
    }
    phase6_summary(all_results)

    elapsed = time.time() - start

    # Final verdict
    print(f"\n{'='*60}")
    print("DEFINITIVE VERDICT")
    print(f"{'='*60}")

    eta_works = any(
        all_results.get(k, {}).get('raw', {}).get('p', 1) < 0.05
        for k in ['eta_100_7net', 'phase1', 's400_7', 's400_hemi']
    )
    noether_genuine = (not all_results['phase2'].get('trivial', True) and
                       all_results['phase3'].get('network_beats_all_spatial', False))
    noether_survives = all_results['noether_100']['p'] < 0.05

    print(f"\n  η LAW: {'WORKS at some granularity' if eta_works else 'FAILS at all tested granularities'}")
    print(f"  NOETHER: {'GENUINE (survives all controls)' if noether_genuine else 'QUALIFIED'}")
    if not noether_genuine:
        if all_results['phase2'].get('trivial', True):
            print(f"    → Random groupings show similar effect (trivial)")
        if not all_results['phase3'].get('network_beats_all_spatial', False):
            print(f"    → Spatial grouping matches/beats network grouping (spatial autocorrelation)")

    n_consistent = sum(1 for h in valid_hyps
                       if all_results['phase4'].get(h, {}).get('d', 0) > 0
                       and all_results['phase4'].get(h, {}).get('p', 1) < 0.05)
    print(f"  CONSISTENCY: {n_consistent}/{len(valid_hyps)} hypotheses show the Noether effect")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_rashomon_v3',
        'version': 3,
        'data': 'Botvinik-Nezer et al. Nature 2020, NeuroVault 6047',
        'results': all_results,
        'verdict': {
            'eta_law_works': eta_works,
            'noether_genuine': noether_genuine,
            'noether_survives_significance': noether_survives,
            'n_hypotheses_consistent': n_consistent,
            'total_hypotheses': len(valid_hyps),
        },
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_rashomon_v3.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
