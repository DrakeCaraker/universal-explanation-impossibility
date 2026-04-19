#!/usr/bin/env python3
"""
Brain Imaging Rashomon v2: Methodologically Rigorous Reanalysis

Fixes from v1:
1. CIRCULARITY BROKEN: Uses independent Schaefer network labels (from Yeo
   resting-state parcellation) as predictor — NOT correlation derived from
   the outcome data.
2. PROPER η LAW TEST: Network size k → predicted instability η = (k-1)/k
3. THRESHOLDING CONFOUND CONTROLLED: Partial Spearman controlling for
   mean overlap level
4. BOOTSTRAP CIs: On all effect sizes (1000 resamples)
5. MULTIPLE COMPARISON CORRECTION: FDR for per-hypothesis tests
6. EMPTY HYPOTHESES EXCLUDED: Hyp 6 and 9 (all zeros) excluded from aggregates
7. DISTANCE BASELINE: Anatomical distance between region centroids
8. WITHIN-VS-BETWEEN NETWORK TEST: Direct Noether counting analog
9. CROSS-VALIDATION: Leave-one-hypothesis-out

Design (revised per vet):
- This is an ANALOGICAL extension, not a formal theorem application
- Brain regions in same network are NOT exactly exchangeable (honest disanalogy)
- The 70 teams used different methods (researcher DOF ≠ Rashomon)
- Reports effect sizes with CIs regardless of significance
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from scipy.spatial.distance import pdist, squareform
import urllib.request

N_HYPOTHESES = 9
N_TEAMS = 70
N_BOOTSTRAP = 1000
# Hypotheses with all-zero overlap maps (no team found anything significant)
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
        img = nib.load(local_path)
        data = img.get_fdata()
        maps[hyp] = img
        is_empty = np.nanmax(data) == 0
        print(f'  Hyp {hyp}: shape={img.shape}, '
              f'range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]'
              f'{" (EMPTY — will exclude)" if is_empty else ""}')

    return maps


def parcellate_and_get_networks(overlap_maps):
    """Parcellate overlap maps and extract INDEPENDENT network labels."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels

    # Extract network names from Schaefer labels
    # Format: "7Networks_LH_Vis_1" → network = "Vis"
    # Skip first label ("Background") — masker returns 100 ROI values
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels
    network_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        # Find network name (after hemisphere indicator)
        if len(parts) >= 3:
            network_names.append(parts[2])
        else:
            network_names.append('Unknown')

    # Compute network assignments and sizes
    unique_networks = sorted(set(network_names))
    network_id = {n: i for i, n in enumerate(unique_networks)}
    region_network = np.array([network_id[n] for n in network_names])
    network_sizes = {}
    for n in unique_networks:
        network_sizes[n] = sum(1 for nn in network_names if nn == n)

    print(f'\n  Networks ({len(unique_networks)}):')
    for n in unique_networks:
        print(f'    {n}: {network_sizes[n]} regions')

    # Compute region centroids from atlas
    atlas_data = nib.load(atlas_img).get_fdata() if isinstance(atlas_img, str) else atlas_img.get_fdata() if hasattr(atlas_img, 'get_fdata') else nib.load(atlas_img).get_fdata()
    # Get affine
    atlas_nii = nib.load(atlas_img) if isinstance(atlas_img, str) else atlas_img
    affine = atlas_nii.affine if hasattr(atlas_nii, 'affine') else np.eye(4)
    atlas_data_raw = atlas_nii.get_fdata() if hasattr(atlas_nii, 'get_fdata') else atlas_data

    centroids = []
    n_rois = len(roi_labels)
    for roi_id in range(1, n_rois + 1):
        voxels = np.argwhere(atlas_data_raw == roi_id)
        if len(voxels) > 0:
            centroid_vox = voxels.mean(axis=0)
            # Convert to MNI coordinates
            centroid_mni = affine[:3, :3] @ centroid_vox + affine[:3, 3]
            centroids.append(centroid_mni)
        else:
            centroids.append(np.array([0, 0, 0]))
    centroids = np.array(centroids)

    # Compute Euclidean distance matrix
    dist_matrix = squareform(pdist(centroids))

    # Parcellate overlap maps
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        strategy='mean'
    )

    region_data = {}
    for hyp, img in overlap_maps.items():
        resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
        region_vals = masker.fit_transform(resampled)
        if region_vals.ndim == 2:
            region_vals = region_vals[0]
        # Values are in [0, 1] representing fraction of teams
        # Keep as fractions for cleaner analysis
        region_data[hyp] = region_vals
        if hyp not in EMPTY_HYPOTHESES:
            print(f'  Hyp {hyp}: {len(region_vals)} regions, '
                  f'mean overlap={np.mean(region_vals):.3f}, '
                  f'max overlap={np.max(region_vals):.3f}')

    return (region_data, labels, network_names, region_network,
            unique_networks, network_sizes, centroids, dist_matrix)


def compute_disagreement(region_data):
    """Compute per-region disagreement, excluding empty hypotheses."""
    n_regions = len(region_data[1])
    valid_hyps = [h for h in range(1, N_HYPOTHESES + 1) if h not in EMPTY_HYPOTHESES]

    overlap_matrix = np.zeros((len(valid_hyps), n_regions))
    for i, hyp in enumerate(valid_hyps):
        overlap_matrix[i] = region_data[hyp]

    # Disagreement = 1 - max(overlap, 1 - overlap)
    # Max disagreement at overlap = 0.5
    agreement_matrix = np.maximum(overlap_matrix, 1 - overlap_matrix)
    disagreement_matrix = 1 - agreement_matrix

    mean_disagreement = np.mean(disagreement_matrix, axis=0)
    mean_overlap = np.mean(overlap_matrix, axis=0)

    return mean_disagreement, disagreement_matrix, mean_overlap, overlap_matrix, valid_hyps


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, seed=42):
    """Compute Spearman correlation with bootstrap CI."""
    rng = np.random.RandomState(seed)
    rho, p = spearmanr(x, y)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    boot_rhos = np.array(boot_rhos)
    ci_lo = np.percentile(boot_rhos, 2.5) if len(boot_rhos) > 0 else np.nan
    ci_hi = np.percentile(boot_rhos, 97.5) if len(boot_rhos) > 0 else np.nan
    return float(rho), float(p), float(ci_lo), float(ci_hi)


def partial_spearman(x, y, z, n_boot=N_BOOTSTRAP, seed=42):
    """Partial Spearman correlation of x and y, controlling for z."""
    from scipy.stats import rankdata
    # Rank all variables
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    # Regress out z from x and y (in rank space)
    def residualize(r, rz):
        A = np.column_stack([rz, np.ones(len(rz))])
        beta = np.linalg.lstsq(A, r, rcond=None)[0]
        return r - A @ beta
    rx_res = residualize(rx, rz)
    ry_res = residualize(ry, rz)
    rho, p = spearmanr(rx_res, ry_res)

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rx_b = residualize(rankdata(x[idx]), rankdata(z[idx]))
        ry_b = residualize(rankdata(y[idx]), rankdata(z[idx]))
        r, _ = spearmanr(rx_b, ry_b)
        if not np.isnan(r):
            boot_rhos.append(r)
    boot_rhos = np.array(boot_rhos)
    ci_lo = np.percentile(boot_rhos, 2.5) if len(boot_rhos) > 0 else np.nan
    ci_hi = np.percentile(boot_rhos, 97.5) if len(boot_rhos) > 0 else np.nan
    return float(rho), float(p), float(ci_lo), float(ci_hi)


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    significant = np.zeros(n, dtype=bool)
    # Find largest k such that p_(k) <= k*alpha/n
    below = sorted_p <= thresholds
    if np.any(below):
        max_k = np.max(np.where(below)[0])
        significant[sorted_idx[:max_k + 1]] = True
    return significant


def test_eta_law(mean_disagreement, network_names, network_sizes, mean_overlap):
    """Test η = (k-1)/k prediction using INDEPENDENT network labels."""
    print("\n" + "=" * 60)
    print("TEST 1: η LAW — Network size predicts disagreement")
    print("  Predictor: η = (k-1)/k from Schaefer/Yeo network labels")
    print("  Outcome: mean disagreement across non-empty hypotheses")
    print("  Key: network labels are INDEPENDENT of NARPS data")
    print("=" * 60)

    # Compute η for each region based on its network size
    eta = np.array([(network_sizes[n] - 1) / network_sizes[n] for n in network_names])

    # Raw correlation
    rho, p, ci_lo, ci_hi = bootstrap_spearman(eta, mean_disagreement)
    print(f"\n  Raw: ρ(η, disagreement) = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p:.2e})")

    # Partial correlation controlling for mean overlap level (thresholding confound)
    rho_partial, p_partial, ci_lo_p, ci_hi_p = partial_spearman(
        eta, mean_disagreement, mean_overlap)
    print(f"  Partial (ctrl overlap): ρ = {rho_partial:.3f} [{ci_lo_p:.3f}, {ci_hi_p:.3f}] (p={p_partial:.2e})")

    # Per-network summary
    print(f"\n  {'Network':>12s} {'k':>3s} {'η':>6s} {'Mean disagree':>14s} {'Std':>8s} {'N':>3s}")
    print("  " + "-" * 55)
    network_stats = {}
    for net in sorted(set(network_names)):
        mask = np.array([n == net for n in network_names])
        k = network_sizes[net]
        eta_val = (k - 1) / k
        md = mean_disagreement[mask]
        network_stats[net] = {
            'k': k, 'eta': float(eta_val),
            'mean_disagree': float(np.mean(md)),
            'std_disagree': float(np.std(md)),
        }
        print(f"  {net:>12s} {k:3d} {eta_val:6.3f} {np.mean(md):14.4f} {np.std(md):8.4f} {k:3d}")

    return {
        'raw_rho': rho, 'raw_p': p, 'raw_ci': [ci_lo, ci_hi],
        'partial_rho': rho_partial, 'partial_p': p_partial,
        'partial_ci': [ci_lo_p, ci_hi_p],
        'network_stats': network_stats,
    }


def test_within_vs_between(disagreement_matrix, region_network, valid_hyps):
    """Test: within-network regions disagree more similarly than between-network."""
    print("\n" + "=" * 60)
    print("TEST 2: NOETHER ANALOG — Within-network vs between-network")
    print("  Do regions in the same functional network have more similar")
    print("  disagreement patterns across hypotheses?")
    print("=" * 60)

    n_regions = disagreement_matrix.shape[1]

    # Compute pairwise disagreement-pattern similarity (correlation across hypotheses)
    # For each pair of regions, correlate their disagreement vectors across hypotheses
    within_sims = []
    between_sims = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            # Similarity = 1 - |disagree_i - disagree_j| averaged across hypotheses
            diff = np.mean(np.abs(disagreement_matrix[:, i] - disagreement_matrix[:, j]))
            if region_network[i] == region_network[j]:
                within_sims.append(diff)
            else:
                between_sims.append(diff)

    within_sims = np.array(within_sims)
    between_sims = np.array(between_sims)

    # Within-network pairs should have SMALLER differences (more similar disagreement)
    mean_within = float(np.mean(within_sims))
    mean_between = float(np.mean(between_sims))

    # Mann-Whitney test
    U, mw_p = mannwhitneyu(within_sims, between_sims, alternative='less')

    # Cohen's d
    pooled_std = np.sqrt((np.var(within_sims) * len(within_sims) +
                          np.var(between_sims) * len(between_sims)) /
                         (len(within_sims) + len(between_sims)))
    cohens_d = (mean_between - mean_within) / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI on the difference
    rng = np.random.RandomState(42)
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        w_boot = rng.choice(within_sims, len(within_sims), replace=True)
        b_boot = rng.choice(between_sims, len(between_sims), replace=True)
        boot_diffs.append(np.mean(b_boot) - np.mean(w_boot))
    boot_diffs = np.array(boot_diffs)
    diff_ci = [float(np.percentile(boot_diffs, 2.5)),
               float(np.percentile(boot_diffs, 97.5))]

    print(f"\n  Within-network mean |Δdisagree|: {mean_within:.4f} (n={len(within_sims)})")
    print(f"  Between-network mean |Δdisagree|: {mean_between:.4f} (n={len(between_sims)})")
    print(f"  Difference (between - within): {mean_between - mean_within:.4f} "
          f"CI [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mann-Whitney p (within < between): {mw_p:.2e}")
    print(f"  Prediction: within < between (same-network regions disagree similarly)")
    confirmed = mean_within < mean_between and mw_p < 0.05
    print(f"  CONFIRMED: {'YES' if confirmed else 'NO'}")

    return {
        'mean_within': mean_within, 'mean_between': mean_between,
        'difference': float(mean_between - mean_within),
        'difference_ci': diff_ci,
        'cohens_d': float(cohens_d),
        'mann_whitney_p': float(mw_p),
        'n_within': len(within_sims), 'n_between': len(between_sims),
        'confirmed': confirmed,
    }


def test_distance_baseline(mean_disagreement, dist_matrix, mean_overlap):
    """Baseline: does anatomical distance predict disagreement?"""
    print("\n" + "=" * 60)
    print("TEST 3: DISTANCE BASELINE")
    print("  Does anatomical distance predict disagreement?")
    print("=" * 60)

    n_regions = len(mean_disagreement)
    mean_dist = np.mean(dist_matrix, axis=1)  # mean distance to all other regions

    rho, p, ci_lo, ci_hi = bootstrap_spearman(mean_dist, mean_disagreement)
    print(f"\n  Raw: ρ(mean_distance, disagreement) = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p:.2e})")

    rho_partial, p_partial, ci_lo_p, ci_hi_p = partial_spearman(
        mean_dist, mean_disagreement, mean_overlap)
    print(f"  Partial (ctrl overlap): ρ = {rho_partial:.3f} [{ci_lo_p:.3f}, {ci_hi_p:.3f}] (p={p_partial:.2e})")

    return {
        'raw_rho': rho, 'raw_p': p, 'raw_ci': [ci_lo, ci_hi],
        'partial_rho': rho_partial, 'partial_p': p_partial,
        'partial_ci': [ci_lo_p, ci_hi_p],
    }


def test_overlap_baseline(mean_disagreement, mean_overlap):
    """Baseline: is disagreement just driven by how close overlap is to 0.5?"""
    print("\n" + "=" * 60)
    print("TEST 4: THRESHOLDING BASELINE")
    print("  Is disagreement driven by overlap proximity to 0.5?")
    print("=" * 60)

    # Distance from 0.5 — regions where overlap ≈ 0.5 should disagree most
    dist_from_half = np.abs(mean_overlap - 0.5)

    rho, p, ci_lo, ci_hi = bootstrap_spearman(dist_from_half, mean_disagreement)
    print(f"\n  ρ(|overlap - 0.5|, disagreement) = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p:.2e})")
    print(f"  Expected: strong NEGATIVE (regions near 0.5 disagree most)")

    # Also: raw overlap predicts disagreement
    rho2, p2, ci_lo2, ci_hi2 = bootstrap_spearman(mean_overlap, mean_disagreement)
    print(f"  ρ(overlap, disagreement) = {rho2:.3f} [{ci_lo2:.3f}, {ci_hi2:.3f}] (p={p2:.2e})")

    return {
        'dist_from_half_rho': rho, 'dist_from_half_p': p, 'dist_from_half_ci': [ci_lo, ci_hi],
        'overlap_rho': rho2, 'overlap_p': p2, 'overlap_ci': [ci_lo2, ci_hi2],
    }


def test_per_hypothesis(region_data, network_names, network_sizes, valid_hyps):
    """Per-hypothesis η law test with FDR correction."""
    print("\n" + "=" * 60)
    print("TEST 5: PER-HYPOTHESIS η LAW (with FDR correction)")
    print("=" * 60)

    eta = np.array([(network_sizes[n] - 1) / network_sizes[n] for n in network_names])

    per_hyp = {}
    p_values = []
    for hyp in valid_hyps:
        overlap = region_data[hyp]
        disagreement = 1 - np.maximum(overlap, 1 - overlap)
        rho, p, ci_lo, ci_hi = bootstrap_spearman(eta, disagreement)
        # Partial controlling for overlap
        rho_p, p_p, ci_lo_p, ci_hi_p = partial_spearman(eta, disagreement, overlap)

        per_hyp[hyp] = {
            'raw_rho': rho, 'raw_p': p, 'raw_ci': [ci_lo, ci_hi],
            'partial_rho': rho_p, 'partial_p': p_p, 'partial_ci': [ci_lo_p, ci_hi_p],
            'mean_disagreement': float(np.mean(disagreement)),
        }
        p_values.append(p)

    # FDR correction
    sig = fdr_correction(p_values)
    for i, hyp in enumerate(valid_hyps):
        per_hyp[hyp]['fdr_significant'] = bool(sig[i])

    print(f"\n  {'Hyp':>4s} {'ρ(raw)':>8s} {'CI':>16s} {'p':>10s} {'ρ(partial)':>10s} {'p(partial)':>10s} {'FDR':>4s}")
    print("  " + "-" * 75)
    for hyp in valid_hyps:
        r = per_hyp[hyp]
        print(f"  {hyp:4d} {r['raw_rho']:8.3f} [{r['raw_ci'][0]:6.3f}, {r['raw_ci'][1]:6.3f}] "
              f"{r['raw_p']:10.2e} {r['partial_rho']:10.3f} {r['partial_p']:10.2e} "
              f"{'*' if r['fdr_significant'] else ' ':>4s}")

    n_sig_raw = sum(1 for h in valid_hyps if per_hyp[h]['raw_p'] < 0.05)
    n_sig_fdr = sum(1 for h in valid_hyps if per_hyp[h]['fdr_significant'])
    n_positive = sum(1 for h in valid_hyps if per_hyp[h]['raw_rho'] > 0)
    print(f"\n  {n_sig_raw}/{len(valid_hyps)} significant (raw p < 0.05)")
    print(f"  {n_sig_fdr}/{len(valid_hyps)} significant (FDR-corrected)")
    print(f"  {n_positive}/{len(valid_hyps)} positive direction (η → more disagreement)")

    return per_hyp


def test_permutation_null(mean_disagreement, network_names, network_sizes, n_perm=1000):
    """Permutation control: shuffle network labels."""
    print("\n" + "=" * 60)
    print("TEST 6: PERMUTATION NULL — Shuffle network labels")
    print("=" * 60)

    eta = np.array([(network_sizes[n] - 1) / network_sizes[n] for n in network_names])
    observed_rho, _ = spearmanr(eta, mean_disagreement)

    rng = np.random.RandomState(42)
    perm_rhos = []
    for _ in range(n_perm):
        perm_eta = rng.permutation(eta)
        r, _ = spearmanr(perm_eta, mean_disagreement)
        perm_rhos.append(r)
    perm_rhos = np.array(perm_rhos)
    perm_p = float(np.mean(np.abs(perm_rhos) >= np.abs(observed_rho)))

    print(f"\n  Observed ρ: {observed_rho:.3f}")
    print(f"  Permutation null: mean={np.mean(perm_rhos):.3f} ± {np.std(perm_rhos):.3f}")
    print(f"  Permutation p: {perm_p:.3f}")
    print(f"  Significant: {'YES' if perm_p < 0.05 else 'NO'}")

    return {
        'observed_rho': float(observed_rho),
        'permutation_p': perm_p,
        'null_mean': float(np.mean(perm_rhos)),
        'null_std': float(np.std(perm_rhos)),
    }


def test_cross_validation(region_data, network_names, network_sizes, valid_hyps):
    """Leave-one-hypothesis-out cross-validation of η prediction."""
    print("\n" + "=" * 60)
    print("TEST 7: LEAVE-ONE-HYPOTHESIS-OUT CROSS-VALIDATION")
    print("=" * 60)

    eta = np.array([(network_sizes[n] - 1) / network_sizes[n] for n in network_names])
    n_regions = len(eta)

    cv_rhos = []
    for held_out in valid_hyps:
        # Train: compute disagreement from all hypotheses except held_out
        train_hyps = [h for h in valid_hyps if h != held_out]
        train_disagree = np.mean([
            1 - np.maximum(region_data[h], 1 - region_data[h])
            for h in train_hyps
        ], axis=0)

        # Test: predict held-out hypothesis disagreement
        test_disagree = 1 - np.maximum(region_data[held_out], 1 - region_data[held_out])

        # η should predict test disagreement
        rho, _ = spearmanr(eta, test_disagree)
        cv_rhos.append(rho)
        print(f"  Held out Hyp {held_out}: ρ(η, test_disagree) = {rho:.3f}")

    mean_cv_rho = float(np.mean(cv_rhos))
    std_cv_rho = float(np.std(cv_rhos))
    print(f"\n  Mean CV ρ: {mean_cv_rho:.3f} ± {std_cv_rho:.3f}")

    return {
        'cv_rhos': {str(h): float(r) for h, r in zip(valid_hyps, cv_rhos)},
        'mean_cv_rho': mean_cv_rho,
        'std_cv_rho': std_cv_rho,
    }


def compare_v1_to_v2(region_data, mean_disagreement, valid_hyps):
    """Run v1's circular analysis for direct comparison."""
    print("\n" + "=" * 60)
    print("COMPARISON: v1 (circular) vs v2 (independent)")
    print("=" * 60)

    n_regions = len(mean_disagreement)

    # v1 approach: correlation matrix from overlap data
    overlap_matrix = np.zeros((len(valid_hyps), n_regions))
    for i, hyp in enumerate(valid_hyps):
        overlap_matrix[i] = region_data[hyp]
    agreement_matrix = np.maximum(overlap_matrix, 1 - overlap_matrix)
    region_corr = np.corrcoef(agreement_matrix.T)

    # v1 predictor: mean |correlation| with all other regions
    mean_corr_mag = np.mean(np.abs(region_corr), axis=1)
    rho_v1, p_v1 = spearmanr(mean_corr_mag, mean_disagreement)

    print(f"\n  v1 (CIRCULAR — corr from outcome data):")
    print(f"    ρ(mean|corr|, disagreement) = {rho_v1:.3f} (p={p_v1:.2e})")
    print(f"    ⚠️ This predictor is derived from the outcome variable!")

    return {
        'v1_circular_rho': float(rho_v1),
        'v1_circular_p': float(p_v1),
    }


def main():
    start = time.time()
    print("=" * 60)
    print("BRAIN IMAGING RASHOMON v2: METHODOLOGICALLY RIGOROUS")
    print("Does independent spatial structure predict team disagreement?")
    print("(Botvinik-Nezer et al., Nature 2020)")
    print("=" * 60)
    print("\nKey fix: Predictors are INDEPENDENT of outcome data.")
    print("Network labels from Yeo resting-state parcellation (separate dataset).")
    print(f"Excluding empty hypotheses: {sorted(EMPTY_HYPOTHESES)}")

    # Phase 1: Download
    print("\n" + "=" * 60)
    print("PHASE 1: Download overlap maps")
    print("=" * 60)
    overlap_maps = download_overlap_maps()

    # Phase 2: Parcellate with independent predictors
    print("\n" + "=" * 60)
    print("PHASE 2: Parcellate and extract independent predictors")
    print("=" * 60)
    (region_data, labels, network_names, region_network,
     unique_networks, network_sizes, centroids, dist_matrix) = \
        parcellate_and_get_networks(overlap_maps)

    # Phase 3: Compute disagreement
    print("\n" + "=" * 60)
    print("PHASE 3: Compute disagreement (excluding empty hypotheses)")
    print("=" * 60)
    mean_disagreement, disagreement_matrix, mean_overlap, overlap_matrix, valid_hyps = \
        compute_disagreement(region_data)

    print(f"\n  Valid hypotheses: {valid_hyps}")
    print(f"  Mean disagreement: {np.mean(mean_disagreement):.4f}")
    print(f"  Std disagreement: {np.std(mean_disagreement):.4f}")
    print(f"  Range: [{np.min(mean_disagreement):.4f}, {np.max(mean_disagreement):.4f}]")
    print(f"  Mean overlap: {np.mean(mean_overlap):.4f}")

    # Phase 4: All tests
    results = {}

    # Test 1: η law with independent network labels
    results['eta_law'] = test_eta_law(
        mean_disagreement, network_names, network_sizes, mean_overlap)

    # Test 2: Within-network vs between-network
    results['within_vs_between'] = test_within_vs_between(
        disagreement_matrix, region_network, valid_hyps)

    # Test 3: Distance baseline
    results['distance_baseline'] = test_distance_baseline(
        mean_disagreement, dist_matrix, mean_overlap)

    # Test 4: Thresholding baseline
    results['thresholding_baseline'] = test_overlap_baseline(
        mean_disagreement, mean_overlap)

    # Test 5: Per-hypothesis
    results['per_hypothesis'] = test_per_hypothesis(
        region_data, network_names, network_sizes, valid_hyps)

    # Test 6: Permutation null
    results['permutation'] = test_permutation_null(
        mean_disagreement, network_names, network_sizes)

    # Test 7: Cross-validation
    results['cross_validation'] = test_cross_validation(
        region_data, network_names, network_sizes, valid_hyps)

    # Comparison with v1
    results['v1_comparison'] = compare_v1_to_v2(
        region_data, mean_disagreement, valid_hyps)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print("DEFINITIVE SUMMARY")
    print(f"{'='*60}")

    eta_raw = results['eta_law']['raw_rho']
    eta_partial = results['eta_law']['partial_rho']
    eta_raw_ci = results['eta_law']['raw_ci']
    eta_partial_ci = results['eta_law']['partial_ci']
    wb = results['within_vs_between']
    dist = results['distance_baseline']
    thresh = results['thresholding_baseline']
    perm = results['permutation']
    cv = results['cross_validation']
    v1 = results['v1_comparison']

    print(f"\n  1. η LAW (independent network labels):")
    print(f"     Raw: ρ = {eta_raw:.3f} [{eta_raw_ci[0]:.3f}, {eta_raw_ci[1]:.3f}]")
    print(f"     Partial (ctrl overlap): ρ = {eta_partial:.3f} [{eta_partial_ci[0]:.3f}, {eta_partial_ci[1]:.3f}]")

    print(f"\n  2. WITHIN vs BETWEEN networks:")
    print(f"     Within |Δdisagree|: {wb['mean_within']:.4f}")
    print(f"     Between |Δdisagree|: {wb['mean_between']:.4f}")
    print(f"     Cohen's d = {wb['cohens_d']:.3f}, p = {wb['mann_whitney_p']:.2e}")

    print(f"\n  3. BASELINES:")
    print(f"     Distance: ρ = {dist['raw_rho']:.3f} (partial: {dist['partial_rho']:.3f})")
    print(f"     Thresholding: ρ = {thresh['dist_from_half_rho']:.3f}")
    print(f"     v1 (CIRCULAR): ρ = {v1['v1_circular_rho']:.3f}")

    print(f"\n  4. ROBUSTNESS:")
    print(f"     Permutation p: {perm['permutation_p']:.3f}")
    print(f"     Mean CV ρ: {cv['mean_cv_rho']:.3f} ± {cv['std_cv_rho']:.3f}")

    # Verdict
    print(f"\n  VERDICT:")
    if (results['eta_law']['raw_p'] < 0.05 and perm['permutation_p'] < 0.05
            and eta_raw_ci[0] > 0):
        print(f"  ✓ Network size (independent predictor) significantly predicts")
        print(f"    team disagreement. Direction: larger networks → MORE disagreement.")
        if results['eta_law']['partial_p'] < 0.05:
            print(f"  ✓ Survives thresholding confound control (partial ρ significant).")
        else:
            print(f"  ⚠ Does NOT survive thresholding confound control.")
            print(f"    Effect may be driven by overlap proximity to 0.5.")
    else:
        print(f"  ✗ Network size does NOT significantly predict disagreement")
        print(f"    when using independent predictors.")

    if wb['confirmed']:
        print(f"  ✓ Within-network regions have more similar disagreement (Noether analog).")
    else:
        print(f"  ✗ Within-network vs between-network difference not significant.")

    print(f"\n  CAVEATS (always report):")
    print(f"  1. ANALOGICAL: Brain regions are NOT exchangeable under Sₖ.")
    print(f"     Network membership ≠ exact symmetry group.")
    print(f"  2. RESEARCHER DOF ≠ RASHOMON: 70 teams used different methods,")
    print(f"     not the same method with different random seeds.")
    print(f"  3. PARCELLATION SENSITIVITY: Results at Schaefer-100 only.")
    print(f"  4. SOFTWARE CONFOUND: Cannot control for SPM/FSL/AFNI effects")
    print(f"     (per-team metadata not in overlap maps).")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_rashomon_v2',
        'version': 2,
        'data': 'Botvinik-Nezer et al. Nature 2020, NeuroVault collection 6047',
        'independent_predictor': 'Schaefer/Yeo 7-network labels (resting-state fMRI)',
        'circularity_fix': 'Predictor (network labels) independent of outcome (disagreement)',
        'n_teams': N_TEAMS,
        'n_hypotheses_total': N_HYPOTHESES,
        'n_hypotheses_valid': len(valid_hyps),
        'excluded_hypotheses': sorted(EMPTY_HYPOTHESES),
        'n_regions': len(mean_disagreement),
        'n_networks': len(unique_networks),
        'networks': {n: network_sizes[n] for n in unique_networks},
        'results': results,
        'mean_disagreement_stats': {
            'mean': float(np.mean(mean_disagreement)),
            'std': float(np.std(mean_disagreement)),
            'min': float(np.min(mean_disagreement)),
            'max': float(np.max(mean_disagreement)),
        },
        'caveats': [
            'Analogical extension — brain regions not exchangeable under Sₖ',
            'Researcher DOF ≠ Rashomon (different methods, not same method with different seeds)',
            'Parcellation at Schaefer-100 only',
            'Cannot control for analysis software (SPM/FSL/AFNI) effects',
        ],
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_rashomon_v2.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
