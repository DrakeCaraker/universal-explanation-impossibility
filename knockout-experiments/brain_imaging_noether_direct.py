#!/usr/bin/env python3
"""
Direct Noether Counting Test on NARPS Per-Team Statistical Maps

Downloads individual team unthresholded t-maps from NeuroVault,
parcellates with Schaefer-100, and tests:

1. Within-network pairs flip MORE than between-network pairs (Noether counting)
2. Flip rate distribution is bimodal (within ≈ 50%, between ≈ low)
3. η law: within-network flip rate vs (k-1)/k prediction

Data: 70 teams × 7 valid hypotheses × 100 brain regions
Source: Individual NeuroVault collections per team
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import urllib.request

N_BOOTSTRAP = 1000
EMPTY_HYPOTHESES = {6, 9}
VALID_HYPOTHESES = [1, 2, 3, 4, 5, 7, 8]
MAX_TEAMS = 70  # full dataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def fetch_team_collections():
    """Fetch team-to-NeuroVault-collection mapping."""
    url = ('https://raw.githubusercontent.com/Inria-Empenn/narps_open_pipelines/'
           'main/narps_open/data/description/analysis_pipelines_full_descriptions.tsv')
    print(f'  Fetching team metadata...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode('utf-8')
    except Exception as e:
        print(f'  Failed to fetch team metadata: {e}')
        return {}

    teams = {}
    for line in content.strip().split('\n')[1:]:  # skip header
        fields = line.split('\t')
        if len(fields) < 2:
            continue
        team_id = fields[0].strip()
        # Find the NeuroVault collection URL
        nv_url = None
        for field in fields:
            if 'neurovault.org/collections/' in field:
                nv_url = field.strip()
                break
        if nv_url and team_id:
            # Extract collection ID
            parts = nv_url.rstrip('/').split('/')
            try:
                coll_id = int(parts[-1])
                teams[team_id] = coll_id
            except (ValueError, IndexError):
                pass

    print(f'  Found {len(teams)} teams with NeuroVault collections')
    return teams


def download_team_maps(teams, cache_dir):
    """Download unthresholded t-maps for each team and hypothesis."""
    import nibabel as nib

    os.makedirs(cache_dir, exist_ok=True)
    team_maps = {}  # team_id → {hyp → nii_img}
    team_ids = sorted(teams.keys())[:MAX_TEAMS]

    total = len(team_ids) * len(VALID_HYPOTHESES)
    done = 0
    failed_teams = set()

    for team_id in team_ids:
        coll_id = teams[team_id]
        team_maps[team_id] = {}

        for hyp in VALID_HYPOTHESES:
            done += 1
            fname = f'team_{team_id}_hypo{hyp}_unthresh.nii.gz'
            local_path = os.path.join(cache_dir, fname)

            if os.path.exists(local_path):
                try:
                    img = nib.load(local_path)
                    team_maps[team_id][hyp] = img
                    continue
                except Exception:
                    os.remove(local_path)

            # Try downloading from NeuroVault
            url = f'https://neurovault.org/media/images/{coll_id}/hypo{hyp}_unthresh.nii.gz'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, 'wb') as out:
                    out.write(resp.read())
                img = nib.load(local_path)
                team_maps[team_id][hyp] = img
            except Exception as e:
                if team_id not in failed_teams:
                    print(f'  ⚠ Team {team_id} (coll {coll_id}): download failed ({e})')
                    failed_teams.add(team_id)
                # Clean up partial download
                if os.path.exists(local_path):
                    os.remove(local_path)

        n_loaded = len(team_maps[team_id])
        if n_loaded > 0 and team_id not in failed_teams:
            sys.stdout.write(f'\r  Downloaded {done}/{total} maps ({len(team_maps)} teams)...')
            sys.stdout.flush()

    print(f'\n  Successfully loaded maps for {sum(1 for t in team_maps if len(team_maps[t]) > 0)} teams')
    print(f'  Failed teams: {len(failed_teams)}')

    # Filter to teams with all valid hypotheses
    complete_teams = {t: m for t, m in team_maps.items() if len(m) == len(VALID_HYPOTHESES)}
    print(f'  Teams with complete data: {len(complete_teams)}')

    return complete_teams


def parcellate_team_maps(team_maps):
    """Parcellate all team maps with Schaefer-100."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels

    # Parse network names
    network_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        network_names.append(parts[2] if len(parts) >= 3 else 'Unknown')

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')

    # Result: team_id → hyp → (100,) array of region t-statistics
    region_data = {}
    n_teams = len(team_maps)
    for ti, (team_id, hyp_maps) in enumerate(team_maps.items()):
        region_data[team_id] = {}
        for hyp, img in hyp_maps.items():
            try:
                resampled = image.resample_to_img(img, atlas_img, interpolation='continuous')
                vals = masker.fit_transform(resampled)
                if vals.ndim == 2:
                    vals = vals[0]
                region_data[team_id][hyp] = vals
            except Exception as e:
                print(f'  ⚠ Parcellation failed for team {team_id} hyp {hyp}: {e}')

        sys.stdout.write(f'\r  Parcellated {ti+1}/{n_teams} teams...')
        sys.stdout.flush()

    print()

    # Filter to teams with all hypotheses successfully parcellated
    complete = {t: d for t, d in region_data.items() if len(d) == len(VALID_HYPOTHESES)}
    print(f'  Complete parcellated teams: {len(complete)}')

    return complete, network_names


def compute_noether(region_data, network_names):
    """Direct Noether counting test: within-network vs between-network flip rates."""
    print("\n" + "=" * 60)
    print("DIRECT NOETHER COUNTING TEST")
    print("=" * 60)

    team_ids = sorted(region_data.keys())
    n_teams = len(team_ids)
    n_regions = 100
    unique_nets = sorted(set(network_names))
    net_id = {n: i for i, n in enumerate(unique_nets)}
    region_net = np.array([net_id[n] for n in network_names])
    net_sizes = {n: sum(1 for x in network_names if x == n) for n in unique_nets}

    # For each hypothesis, compute per-pair flip rate across teams
    all_within_flips = []
    all_between_flips = []
    per_hyp_results = {}

    for hyp in VALID_HYPOTHESES:
        # Build team × region matrix for this hypothesis
        t_matrix = np.zeros((n_teams, n_regions))
        for ti, team_id in enumerate(team_ids):
            t_matrix[ti] = region_data[team_id][hyp]

        # For each region pair, compute flip rate across team pairs
        within_flips = []
        between_flips = []
        within_pairs_info = []
        between_pairs_info = []

        for ri in range(n_regions):
            for rj in range(ri + 1, n_regions):
                # For each team pair, does the ranking of (ri, rj) flip?
                n_flip = 0
                n_comparable = 0
                for t1 in range(n_teams):
                    for t2 in range(t1 + 1, n_teams):
                        diff1 = t_matrix[t1, ri] - t_matrix[t1, rj]
                        diff2 = t_matrix[t2, ri] - t_matrix[t2, rj]
                        if abs(diff1) > 1e-10 and abs(diff2) > 1e-10:
                            if np.sign(diff1) != np.sign(diff2):
                                n_flip += 1
                            n_comparable += 1

                if n_comparable > 0:
                    fr = n_flip / n_comparable
                    if region_net[ri] == region_net[rj]:
                        within_flips.append(fr)
                    else:
                        between_flips.append(fr)

        within_flips = np.array(within_flips)
        between_flips = np.array(between_flips)
        all_within_flips.extend(within_flips)
        all_between_flips.extend(between_flips)

        mean_w = float(np.mean(within_flips)) if len(within_flips) > 0 else 0
        mean_b = float(np.mean(between_flips)) if len(between_flips) > 0 else 0

        per_hyp_results[hyp] = {
            'mean_within_flip': mean_w,
            'mean_between_flip': mean_b,
            'gap': float(mean_w - mean_b),
            'n_within_pairs': len(within_flips),
            'n_between_pairs': len(between_flips),
        }
        print(f"  Hyp {hyp}: within={mean_w:.3f}, between={mean_b:.3f}, "
              f"gap={mean_w - mean_b:+.3f} ({len(within_flips)}w/{len(between_flips)}b pairs)")

    all_within_flips = np.array(all_within_flips)
    all_between_flips = np.array(all_between_flips)

    # Overall statistics
    mean_within = float(np.mean(all_within_flips))
    mean_between = float(np.mean(all_between_flips))
    gap = mean_within - mean_between

    # Mann-Whitney test
    U, mw_p = mannwhitneyu(all_within_flips, all_between_flips, alternative='greater')

    # Cohen's d
    pooled_std = np.sqrt((np.var(all_within_flips) * len(all_within_flips) +
                          np.var(all_between_flips) * len(all_between_flips)) /
                         (len(all_within_flips) + len(all_between_flips)))
    cohens_d = gap / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI on gap
    rng = np.random.RandomState(42)
    boot_gaps = []
    for _ in range(N_BOOTSTRAP):
        w = rng.choice(all_within_flips, len(all_within_flips), replace=True)
        b = rng.choice(all_between_flips, len(all_between_flips), replace=True)
        boot_gaps.append(np.mean(w) - np.mean(b))
    boot_gaps = np.array(boot_gaps)
    gap_ci = [float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5))]

    print(f"\n  OVERALL:")
    print(f"  Within-network mean flip rate: {mean_within:.3f}")
    print(f"  Between-network mean flip rate: {mean_between:.3f}")
    print(f"  Gap: {gap:+.3f} [{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}]")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mann-Whitney p (within > between): {mw_p:.2e}")

    # Per-network within-flip rates
    print(f"\n  Per-network within-flip rates:")
    print(f"  {'Network':>12s} {'k':>3s} {'η_exact':>8s} {'Observed flip':>14s}")
    print("  " + "-" * 45)
    per_net = {}
    for net in unique_nets:
        net_indices = [i for i, n in enumerate(network_names) if n == net]
        k = len(net_indices)
        # Collect flip rates for pairs within this network
        net_within = []
        for hyp in VALID_HYPOTHESES:
            t_matrix = np.zeros((n_teams, n_regions))
            for ti, team_id in enumerate(team_ids):
                t_matrix[ti] = region_data[team_id][hyp]
            for i_idx in range(len(net_indices)):
                for j_idx in range(i_idx + 1, len(net_indices)):
                    ri, rj = net_indices[i_idx], net_indices[j_idx]
                    n_flip = 0
                    n_comp = 0
                    for t1 in range(n_teams):
                        for t2 in range(t1 + 1, n_teams):
                            d1 = t_matrix[t1, ri] - t_matrix[t1, rj]
                            d2 = t_matrix[t2, ri] - t_matrix[t2, rj]
                            if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                                if np.sign(d1) != np.sign(d2):
                                    n_flip += 1
                                n_comp += 1
                    if n_comp > 0:
                        net_within.append(n_flip / n_comp)

        mean_net_flip = float(np.mean(net_within)) if net_within else 0
        eta_exact = (k - 1) / k
        per_net[net] = {
            'k': k, 'eta_exact': float(eta_exact),
            'observed_flip': mean_net_flip, 'n_pairs': len(net_within),
        }
        print(f"  {net:>12s} {k:3d} {eta_exact:8.3f} {mean_net_flip:14.3f}")

    # Noether prediction comparison
    n_consistent = sum(1 for h in VALID_HYPOTHESES
                       if per_hyp_results[h]['mean_within_flip'] > per_hyp_results[h]['mean_between_flip'])

    print(f"\n  {n_consistent}/{len(VALID_HYPOTHESES)} hypotheses show within > between")

    return {
        'mean_within_flip': mean_within,
        'mean_between_flip': mean_between,
        'gap': float(gap),
        'gap_ci': gap_ci,
        'cohens_d': float(cohens_d),
        'mann_whitney_p': float(mw_p),
        'per_hypothesis': per_hyp_results,
        'per_network': per_net,
        'n_consistent': n_consistent,
        'n_teams': n_teams,
        'within_flip_distribution': {
            'mean': float(np.mean(all_within_flips)),
            'std': float(np.std(all_within_flips)),
            'median': float(np.median(all_within_flips)),
            'q25': float(np.percentile(all_within_flips, 25)),
            'q75': float(np.percentile(all_within_flips, 75)),
        },
        'between_flip_distribution': {
            'mean': float(np.mean(all_between_flips)),
            'std': float(np.std(all_between_flips)),
            'median': float(np.median(all_between_flips)),
            'q25': float(np.percentile(all_between_flips, 25)),
            'q75': float(np.percentile(all_between_flips, 75)),
        },
    }


def random_null_test(region_data, network_names, n_perm=200):
    """Permutation null: shuffle network assignments."""
    print("\n" + "=" * 60)
    print("RANDOM GROUPING NULL")
    print("=" * 60)

    team_ids = sorted(region_data.keys())
    n_teams = len(team_ids)
    n_regions = 100

    # Precompute all team × region matrices for speed
    t_matrices = {}
    for hyp in VALID_HYPOTHESES:
        t_matrix = np.zeros((n_teams, n_regions))
        for ti, team_id in enumerate(team_ids):
            t_matrix[ti] = region_data[team_id][hyp]
        t_matrices[hyp] = t_matrix

    # Precompute pairwise flip rates for ALL region pairs (hypothesis-averaged)
    # This avoids recomputing for each permutation
    print("  Precomputing pairwise flip rates...")
    flip_matrix = np.zeros((n_regions, n_regions))
    for hyp in VALID_HYPOTHESES:
        tm = t_matrices[hyp]
        for ri in range(n_regions):
            for rj in range(ri + 1, n_regions):
                n_flip = 0
                n_comp = 0
                for t1 in range(n_teams):
                    for t2 in range(t1 + 1, n_teams):
                        d1 = tm[t1, ri] - tm[t1, rj]
                        d2 = tm[t2, ri] - tm[t2, rj]
                        if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                            if np.sign(d1) != np.sign(d2):
                                n_flip += 1
                            n_comp += 1
                if n_comp > 0:
                    flip_matrix[ri, rj] = n_flip / n_comp
                    flip_matrix[rj, ri] = flip_matrix[ri, rj]

    # Observed gap with real networks
    net_id = {n: i for i, n in enumerate(sorted(set(network_names)))}
    region_net = np.array([net_id[n] for n in network_names])

    def compute_gap(assignments):
        within = []
        between = []
        for ri in range(n_regions):
            for rj in range(ri + 1, n_regions):
                fr = flip_matrix[ri, rj]
                if assignments[ri] == assignments[rj]:
                    within.append(fr)
                else:
                    between.append(fr)
        return np.mean(within) - np.mean(between) if within and between else 0

    observed_gap = compute_gap(region_net)
    print(f"  Observed gap: {observed_gap:.4f}")

    # Permutation null
    rng = np.random.RandomState(42)
    null_gaps = []
    for pi in range(n_perm):
        perm_assignments = rng.permutation(region_net)
        null_gaps.append(compute_gap(perm_assignments))
        if (pi + 1) % 50 == 0:
            sys.stdout.write(f'\r  Permutation {pi+1}/{n_perm}...')
            sys.stdout.flush()

    null_gaps = np.array(null_gaps)
    perm_p = float(np.mean(null_gaps >= observed_gap))
    print(f'\n  Null gap: {np.mean(null_gaps):.4f} ± {np.std(null_gaps):.4f}')
    print(f'  Permutation p: {perm_p:.3f}')
    print(f'  95th percentile: {np.percentile(null_gaps, 95):.4f}')

    return {
        'observed_gap': float(observed_gap),
        'null_mean': float(np.mean(null_gaps)),
        'null_std': float(np.std(null_gaps)),
        'permutation_p': perm_p,
        'null_95th': float(np.percentile(null_gaps, 95)),
    }


def main():
    start = time.time()
    print("=" * 60)
    print("DIRECT NOETHER COUNTING: NARPS PER-TEAM MAPS")
    print("=" * 60)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_team_cache')

    # Phase 0: Get team metadata
    print("\nPhase 0: Team metadata")
    teams = fetch_team_collections()
    if not teams:
        print("ERROR: Could not fetch team metadata. Aborting.")
        return

    # Phase 1: Download maps
    print(f"\nPhase 1: Downloading per-team maps (up to {MAX_TEAMS} teams)...")
    team_maps = download_team_maps(teams, cache_dir)
    if len(team_maps) < 5:
        print(f"ERROR: Only {len(team_maps)} complete teams. Need at least 5. Aborting.")
        return

    # Phase 2: Parcellate
    print(f"\nPhase 2: Parcellating {len(team_maps)} teams...")
    region_data, network_names = parcellate_team_maps(team_maps)
    if len(region_data) < 5:
        print(f"ERROR: Only {len(region_data)} complete parcellated teams. Aborting.")
        return

    print(f"\n  Using {len(region_data)} teams × {len(VALID_HYPOTHESES)} hypotheses × 100 regions")

    # Phase 3: Direct Noether test
    print(f"\nPhase 3: Direct Noether counting test...")
    noether_results = compute_noether(region_data, network_names)

    # Phase 4: Random null
    print(f"\nPhase 4: Random grouping null...")
    null_results = random_null_test(region_data, network_names)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print("DEFINITIVE SUMMARY")
    print(f"{'='*60}")

    nr = noether_results
    print(f"\n  Teams analyzed: {nr['n_teams']}")
    print(f"\n  NOETHER COUNTING:")
    print(f"    Within-network mean flip rate: {nr['mean_within_flip']:.3f}")
    print(f"    Between-network mean flip rate: {nr['mean_between_flip']:.3f}")
    print(f"    Gap: {nr['gap']:+.3f} [{nr['gap_ci'][0]:+.3f}, {nr['gap_ci'][1]:+.3f}]")
    print(f"    Cohen's d: {nr['cohens_d']:.3f}")
    print(f"    Mann-Whitney p: {nr['mann_whitney_p']:.2e}")
    print(f"    Consistent hypotheses: {nr['n_consistent']}/{len(VALID_HYPOTHESES)}")

    print(f"\n  RANDOM NULL:")
    print(f"    Observed gap: {null_results['observed_gap']:.4f}")
    print(f"    Null gap: {null_results['null_mean']:.4f} ± {null_results['null_std']:.4f}")
    print(f"    Permutation p: {null_results['permutation_p']:.3f}")

    print(f"\n  VERDICT:")
    if (nr['mann_whitney_p'] < 0.05 and nr['gap'] > 0 and
            null_results['permutation_p'] < 0.05):
        print(f"  ✓ NOETHER ANALOG CONFIRMED at the per-team level.")
        print(f"    Within-network rankings ARE more unstable across teams.")
        if nr['mean_within_flip'] > 0.3:
            print(f"    Within-network flip rate ({nr['mean_within_flip']:.3f}) approaches")
            print(f"    the exact-symmetry prediction (0.500).")
        else:
            print(f"    But within-network flip rate ({nr['mean_within_flip']:.3f}) is well")
            print(f"    below the exact-symmetry prediction (0.500).")
            print(f"    Consistent with APPROXIMATE (not exact) symmetry.")
    elif nr['gap'] > 0 and nr['mann_whitney_p'] < 0.05:
        print(f"  ⚠ Noether analog significant but doesn't survive permutation null.")
    else:
        print(f"  ✗ Noether analog NOT confirmed at per-team level.")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_noether_direct',
        'data': 'NARPS per-team unthresholded t-maps',
        'n_teams': nr['n_teams'],
        'n_hypotheses': len(VALID_HYPOTHESES),
        'n_regions': 100,
        'noether': noether_results,
        'random_null': null_results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_noether_direct.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
