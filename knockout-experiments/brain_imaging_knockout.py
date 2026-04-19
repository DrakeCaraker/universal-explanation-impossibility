#!/usr/bin/env python3
"""
Brain Imaging Knockout: Two critical tests to upgrade from "partial" to "knockout"

Test A: Activation-level control
  Does the Noether result (d=0.43) survive controlling for activation similarity?
  If within-network regions only agree because they have similar activation,
  the effect should vanish after controlling for |overlap_i - overlap_j|.

Test B: Network × Software interaction (product-group Noether)
  Teams grouped by software (SPM/FSL/AFNI). Regions grouped by network.
  Prediction: within-network flip rate premium is modulated by software similarity.
  This decomposes disagreement into structural (network) and methodological (software)
  components — a product-group Noether counting prediction.
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
import urllib.request

EMPTY_HYPOTHESES = {6, 9}
VALID_HYPOTHESES = [1, 2, 3, 4, 5, 7, 8]
N_BOOTSTRAP = 1000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_team_software():
    """Fetch team software from narps_open_pipelines metadata."""
    import re
    url = ('https://raw.githubusercontent.com/Inria-Empenn/narps_open_pipelines/'
           'main/narps_open/data/description/analysis_pipelines_full_descriptions.tsv')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode('utf-8')
    except Exception as e:
        print(f'  Failed to fetch metadata: {e}')
        return {}

    # The TSV has multi-line cells. Find team rows by 4-char alphanumeric ID pattern.
    # Column 6 (index 6) contains the software field.
    team_software = {}
    team_pattern = re.compile(r'^[A-Z0-9]{4}\t')
    for line in content.split('\n'):
        if team_pattern.match(line):
            fields = line.split('\t')
            team_id = fields[0].strip()
            sw_raw = fields[6].strip() if len(fields) > 6 else ''
            # Normalize
            sw_upper = sw_raw.upper()
            if 'SPM' in sw_upper:
                software = 'SPM'
            elif 'FSL' in sw_upper or 'FEAT' in sw_upper:
                software = 'FSL'
            elif 'AFNI' in sw_upper:
                software = 'AFNI'
            elif 'NISTATS' in sw_upper or 'NILEARN' in sw_upper:
                software = 'Python'
            elif sw_raw in ('', 'NA', 'N/A', 'n/a'):
                software = 'Unknown'
            else:
                software = 'Other'
            team_software[team_id] = software

    return team_software


def setup_data():
    """Load parcellated per-team data and network labels."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_team_cache')

    # Load atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels

    network_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        network_names.append(parts[2] if len(parts) >= 3 else 'Unknown')

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')

    # Load overlap maps for activation level
    narps_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_cache')
    overlap_maps = {}
    for hyp in VALID_HYPOTHESES:
        path = os.path.join(narps_cache, f'hypo{hyp}.nii.gz')
        if os.path.exists(path):
            img = nib.load(path)
            resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
            vals = masker.fit_transform(resampled)
            if vals.ndim == 2:
                vals = vals[0]
            overlap_maps[hyp] = vals

    mean_overlap = np.mean([overlap_maps[h] for h in overlap_maps], axis=0)

    # Load per-team parcellated data from cached NIfTI files
    region_data = {}
    team_files = [f for f in os.listdir(cache_dir)
                  if f.startswith('team_') and f.endswith('_unthresh.nii.gz')]
    team_ids_found = set()
    for f in team_files:
        parts = f.split('_')
        if len(parts) >= 2:
            team_ids_found.add(parts[1])

    for team_id in sorted(team_ids_found):
        team_data = {}
        all_valid = True
        for hyp in VALID_HYPOTHESES:
            path = os.path.join(cache_dir, f'team_{team_id}_hypo{hyp}_unthresh.nii.gz')
            if os.path.exists(path):
                try:
                    img = nib.load(path)
                    resampled = image.resample_to_img(img, atlas_img, interpolation='continuous')
                    vals = masker.fit_transform(resampled)
                    if vals.ndim == 2:
                        vals = vals[0]
                    team_data[hyp] = vals
                except Exception:
                    all_valid = False
            else:
                all_valid = False

        if all_valid and len(team_data) == len(VALID_HYPOTHESES):
            region_data[team_id] = team_data

    print(f'  Loaded {len(region_data)} complete teams')
    return region_data, network_names, mean_overlap


# ==========================================================================
# TEST A: Activation-level control on Noether d=0.43
# ==========================================================================

def test_activation_control(region_data, network_names, mean_overlap):
    """Does within-network similarity survive controlling for activation level?"""
    print("\n" + "=" * 60)
    print("TEST A: ACTIVATION-LEVEL CONTROL")
    print("  Does d=0.43 survive controlling for |overlap_i - overlap_j|?")
    print("=" * 60)

    n_regions = 100
    unique_nets = sorted(set(network_names))
    net_id = {n: i for i, n in enumerate(unique_nets)}
    region_net = np.array([net_id[n] for n in network_names])

    # Compute per-hypothesis disagreement from per-team t-statistics
    # Use variance of t-statistics across teams as disagreement measure
    # (more appropriate than overlap-based since we have per-team data)
    team_ids = sorted(region_data.keys())
    n_teams = len(team_ids)

    # Approach: for each region and hypothesis, compute the standard deviation
    # of t-statistics across teams. Normalize by creating a disagreement matrix
    # comparable across hypotheses.
    disagreement_matrix = np.zeros((len(VALID_HYPOTHESES), n_regions))
    for hi, hyp in enumerate(VALID_HYPOTHESES):
        t_vals = np.array([region_data[t][hyp] for t in team_ids])  # (n_teams, n_regions)
        # Use IQR / median as robust disagreement (avoids outlier sensitivity)
        q75 = np.percentile(t_vals, 75, axis=0)
        q25 = np.percentile(t_vals, 25, axis=0)
        median_t = np.median(t_vals, axis=0)
        disagreement_matrix[hi] = (q75 - q25) / (np.abs(median_t) + 0.1)

    mean_disagreement = np.mean(disagreement_matrix, axis=0)

    # For each pair of regions: compute disagreement pattern similarity and activation similarity
    within_disagree_diffs = []
    within_activation_diffs = []
    between_disagree_diffs = []
    between_activation_diffs = []

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            d_diff = np.mean(np.abs(disagreement_matrix[:, i] - disagreement_matrix[:, j]))
            a_diff = abs(mean_overlap[i] - mean_overlap[j])

            if region_net[i] == region_net[j]:
                within_disagree_diffs.append(d_diff)
                within_activation_diffs.append(a_diff)
            else:
                between_disagree_diffs.append(d_diff)
                between_activation_diffs.append(a_diff)

    within_dd = np.array(within_disagree_diffs)
    within_ad = np.array(within_activation_diffs)
    between_dd = np.array(between_disagree_diffs)
    between_ad = np.array(between_activation_diffs)

    # Unstratified result
    raw_d = (np.mean(between_dd) - np.mean(within_dd))
    pooled_std = np.sqrt((np.var(within_dd) * len(within_dd) +
                          np.var(between_dd) * len(between_dd)) /
                         (len(within_dd) + len(between_dd)))
    raw_cohens_d = raw_d / pooled_std if pooled_std > 0 else 0
    U, raw_p = mannwhitneyu(within_dd, between_dd, alternative='less')

    print(f"\n  Unstratified: d = {raw_cohens_d:.3f} (p = {raw_p:.2e})")
    print(f"    Within mean |Δdisagree|: {np.mean(within_dd):.4f}")
    print(f"    Between mean |Δdisagree|: {np.mean(between_dd):.4f}")
    print(f"    Mean within |Δactivation|: {np.mean(within_ad):.4f}")
    print(f"    Mean between |Δactivation|: {np.mean(between_ad):.4f}")

    # Stratified by activation similarity (quintiles)
    all_ad = np.concatenate([within_ad, between_ad])
    quintiles = np.percentile(all_ad, [20, 40, 60, 80])

    print(f"\n  Stratified by activation similarity (quintiles):")
    print(f"  {'Stratum':>12s} {'Within':>8s} {'Between':>8s} {'Gap':>8s} {'d':>8s} {'p':>10s}")
    print("  " + "-" * 60)

    strata_results = []
    strata_labels = ['Q1 (most similar)', 'Q2', 'Q3', 'Q4', 'Q5 (most different)']
    bounds = [(-np.inf, quintiles[0]), (quintiles[0], quintiles[1]),
              (quintiles[1], quintiles[2]), (quintiles[2], quintiles[3]),
              (quintiles[3], np.inf)]

    for si, (lo, hi) in enumerate(bounds):
        w_mask = (within_ad >= lo) & (within_ad < hi) if hi < np.inf else (within_ad >= lo)
        b_mask = (between_ad >= lo) & (between_ad < hi) if hi < np.inf else (between_ad >= lo)

        w_vals = within_dd[w_mask]
        b_vals = between_dd[b_mask]

        if len(w_vals) < 5 or len(b_vals) < 5:
            strata_results.append({'label': strata_labels[si], 'n_w': len(w_vals), 'n_b': len(b_vals)})
            continue

        mean_w = np.mean(w_vals)
        mean_b = np.mean(b_vals)
        gap = mean_b - mean_w
        ps = np.sqrt((np.var(w_vals) * len(w_vals) + np.var(b_vals) * len(b_vals)) /
                     (len(w_vals) + len(b_vals)))
        d_s = gap / ps if ps > 0 else 0
        U_s, p_s = mannwhitneyu(w_vals, b_vals, alternative='less')

        strata_results.append({
            'label': strata_labels[si], 'within': float(mean_w), 'between': float(mean_b),
            'gap': float(gap), 'd': float(d_s), 'p': float(p_s),
            'n_w': len(w_vals), 'n_b': len(b_vals),
        })
        print(f"  {strata_labels[si]:>12s} {mean_w:8.4f} {mean_b:8.4f} {gap:8.4f} {d_s:8.3f} {p_s:10.2e}")

    # Does the gap persist in all strata?
    n_sig = sum(1 for s in strata_results if 'p' in s and s['p'] < 0.05)
    n_positive = sum(1 for s in strata_results if 'gap' in s and s['gap'] > 0)
    print(f"\n  Strata with significant gap: {n_sig}/{len(strata_results)}")
    print(f"  Strata with positive gap: {n_positive}/{len(strata_results)}")

    survives = n_positive >= 4  # at least 4 of 5 quintiles show positive gap
    print(f"\n  VERDICT: Noether effect {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
          f"activation-level control")

    return {
        'raw_d': float(raw_cohens_d), 'raw_p': float(raw_p),
        'strata': strata_results,
        'survives': survives,
        'n_strata_significant': n_sig,
        'n_strata_positive': n_positive,
    }


# ==========================================================================
# TEST B: Network × Software interaction
# ==========================================================================

def test_network_software_interaction(region_data, network_names, team_software):
    """2×2 Noether: network membership × software family."""
    print("\n" + "=" * 60)
    print("TEST B: NETWORK × SOFTWARE INTERACTION")
    print("  Product-group Noether counting prediction")
    print("=" * 60)

    team_ids = sorted(region_data.keys())
    n_teams = len(team_ids)
    n_regions = 100

    # Map teams to software
    team_sw = {}
    sw_counts = {}
    for t in team_ids:
        sw = team_software.get(t, 'Unknown')
        team_sw[t] = sw
        sw_counts[sw] = sw_counts.get(sw, 0) + 1

    print(f"\n  Software distribution ({n_teams} teams):")
    for sw, count in sorted(sw_counts.items(), key=lambda x: -x[1]):
        print(f"    {sw}: {count}")

    # Network assignments
    unique_nets = sorted(set(network_names))
    net_id = {n: i for i, n in enumerate(unique_nets)}
    region_net = np.array([net_id[n] for n in network_names])

    # Precompute t-matrices
    t_matrices = {}
    for hyp in VALID_HYPOTHESES:
        tm = np.zeros((n_teams, n_regions))
        for ti, t in enumerate(team_ids):
            tm[ti] = region_data[t][hyp]
        t_matrices[hyp] = tm

    # 2×2 design: same/diff network × same/diff software
    cells = {
        'same_net_same_sw': [], 'same_net_diff_sw': [],
        'diff_net_same_sw': [], 'diff_net_diff_sw': [],
    }

    print(f"\n  Computing 2×2 flip rates...")
    for hyp in VALID_HYPOTHESES:
        tm = t_matrices[hyp]
        for ri in range(n_regions):
            for rj in range(ri + 1, n_regions):
                same_net = region_net[ri] == region_net[rj]

                # Separate flip rates by software similarity
                same_sw_flips = 0
                same_sw_comp = 0
                diff_sw_flips = 0
                diff_sw_comp = 0

                for t1_idx in range(n_teams):
                    for t2_idx in range(t1_idx + 1, n_teams):
                        d1 = tm[t1_idx, ri] - tm[t1_idx, rj]
                        d2 = tm[t2_idx, ri] - tm[t2_idx, rj]
                        if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                            flipped = np.sign(d1) != np.sign(d2)
                            same_sw = team_sw[team_ids[t1_idx]] == team_sw[team_ids[t2_idx]]
                            if same_sw:
                                same_sw_comp += 1
                                if flipped:
                                    same_sw_flips += 1
                            else:
                                diff_sw_comp += 1
                                if flipped:
                                    diff_sw_flips += 1

                if same_sw_comp > 0:
                    fr_same_sw = same_sw_flips / same_sw_comp
                    if same_net:
                        cells['same_net_same_sw'].append(fr_same_sw)
                    else:
                        cells['diff_net_same_sw'].append(fr_same_sw)

                if diff_sw_comp > 0:
                    fr_diff_sw = diff_sw_flips / diff_sw_comp
                    if same_net:
                        cells['same_net_diff_sw'].append(fr_diff_sw)
                    else:
                        cells['diff_net_diff_sw'].append(fr_diff_sw)

        sys.stdout.write(f'\r  Hyp {hyp} done...')
        sys.stdout.flush()

    print()

    # Compute cell means
    cell_means = {}
    print(f"\n  {'':>20s} {'Same Software':>15s} {'Diff Software':>15s}")
    print("  " + "-" * 52)
    for net_label in ['same_net', 'diff_net']:
        row = []
        for sw_label in ['same_sw', 'diff_sw']:
            key = f'{net_label}_{sw_label}'
            vals = np.array(cells[key])
            mean_v = float(np.mean(vals)) if len(vals) > 0 else 0
            cell_means[key] = mean_v
            row.append(f'{mean_v:.4f} (n={len(vals)})')
        label = 'Same Network' if net_label == 'same_net' else 'Diff Network'
        print(f"  {label:>20s} {row[0]:>15s} {row[1]:>15s}")

    # Main effects and interaction
    net_effect_same_sw = cell_means['same_net_same_sw'] - cell_means['diff_net_same_sw']
    net_effect_diff_sw = cell_means['same_net_diff_sw'] - cell_means['diff_net_diff_sw']
    sw_effect_same_net = cell_means['same_net_diff_sw'] - cell_means['same_net_same_sw']
    sw_effect_diff_net = cell_means['diff_net_diff_sw'] - cell_means['diff_net_same_sw']

    # Network main effect
    all_same_net = np.concatenate([cells['same_net_same_sw'], cells['same_net_diff_sw']])
    all_diff_net = np.concatenate([cells['diff_net_same_sw'], cells['diff_net_diff_sw']])
    net_main = float(np.mean(all_same_net) - np.mean(all_diff_net))
    U_net, p_net = mannwhitneyu(all_same_net, all_diff_net, alternative='greater')

    # Software main effect
    all_same_sw = np.concatenate([cells['same_net_same_sw'], cells['diff_net_same_sw']])
    all_diff_sw = np.concatenate([cells['same_net_diff_sw'], cells['diff_net_diff_sw']])
    sw_main = float(np.mean(all_diff_sw) - np.mean(all_same_sw))
    U_sw, p_sw = mannwhitneyu(all_diff_sw, all_same_sw, alternative='greater')

    # Interaction: does the network effect differ by software?
    interaction = net_effect_diff_sw - net_effect_same_sw

    print(f"\n  Main effects:")
    print(f"    Network (same-net premium): {net_main:+.4f} (p = {p_net:.2e})")
    print(f"    Software (diff-sw premium): {sw_main:+.4f} (p = {p_sw:.2e})")
    print(f"\n  Network effect by software:")
    print(f"    Same software: {net_effect_same_sw:+.4f}")
    print(f"    Diff software: {net_effect_diff_sw:+.4f}")
    print(f"    Interaction: {interaction:+.4f}")

    # Bootstrap CI on interaction
    rng = np.random.RandomState(42)
    boot_interactions = []
    for _ in range(N_BOOTSTRAP):
        sn_ss = rng.choice(cells['same_net_same_sw'],
                           len(cells['same_net_same_sw']), replace=True)
        sn_ds = rng.choice(cells['same_net_diff_sw'],
                           len(cells['same_net_diff_sw']), replace=True)
        dn_ss = rng.choice(cells['diff_net_same_sw'],
                           len(cells['diff_net_same_sw']), replace=True)
        dn_ds = rng.choice(cells['diff_net_diff_sw'],
                           len(cells['diff_net_diff_sw']), replace=True)
        ne_ss = np.mean(sn_ss) - np.mean(dn_ss)
        ne_ds = np.mean(sn_ds) - np.mean(dn_ds)
        boot_interactions.append(ne_ds - ne_ss)

    boot_interactions = np.array(boot_interactions)
    int_ci = [float(np.percentile(boot_interactions, 2.5)),
              float(np.percentile(boot_interactions, 97.5))]

    print(f"    Interaction CI: [{int_ci[0]:+.4f}, {int_ci[1]:+.4f}]")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if p_net < 0.05 and p_sw < 0.05:
        print(f"  ✓ BOTH main effects significant:")
        print(f"    - Network structure predicts flip rates (Noether on regions)")
        print(f"    - Software similarity predicts flip rates (Noether on methods)")
        if int_ci[0] > 0 or int_ci[1] < 0:
            print(f"  ✓ Interaction significant: the two symmetries interact")
            print(f"    This is the PRODUCT-GROUP Noether prediction.")
        else:
            print(f"  ~ Interaction not significant: effects are additive")
            print(f"    Both Noether predictions hold independently.")
    elif p_net < 0.05:
        print(f"  ✓ Network effect significant, software not.")
    elif p_sw < 0.05:
        print(f"  ⚠ Software effect significant, network not.")
    else:
        print(f"  ✗ Neither main effect significant.")

    return {
        'cell_means': cell_means,
        'cell_counts': {k: len(v) for k, v in cells.items()},
        'network_main_effect': net_main, 'network_p': float(p_net),
        'software_main_effect': sw_main, 'software_p': float(p_sw),
        'net_effect_same_sw': float(net_effect_same_sw),
        'net_effect_diff_sw': float(net_effect_diff_sw),
        'interaction': float(interaction),
        'interaction_ci': int_ci,
        'software_distribution': sw_counts,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("BRAIN IMAGING KNOCKOUT TESTS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    region_data, network_names, mean_overlap = setup_data()

    print("\nLoading team software metadata...")
    team_software = load_team_software()
    team_ids = sorted(region_data.keys())
    matched = sum(1 for t in team_ids if t in team_software)
    print(f"  Matched {matched}/{len(team_ids)} teams to software metadata")

    results = {}

    # Test A: Activation-level control
    results['activation_control'] = test_activation_control(
        region_data, network_names, mean_overlap)

    # Test B: Network × Software interaction
    results['network_software'] = test_network_software_interaction(
        region_data, network_names, team_software)

    elapsed = time.time() - start

    # Final verdict
    print(f"\n{'='*60}")
    print("KNOCKOUT VERDICT")
    print(f"{'='*60}")

    ac = results['activation_control']
    ns = results['network_software']

    print(f"\n  Test A (activation control): {'SURVIVES' if ac['survives'] else 'FAILS'}")
    print(f"    {ac['n_strata_positive']}/5 strata show positive gap")
    print(f"    {ac['n_strata_significant']}/5 strata significant")

    print(f"\n  Test B (network × software):")
    print(f"    Network main effect: {ns['network_main_effect']:+.4f} (p={ns['network_p']:.2e})")
    print(f"    Software main effect: {ns['software_main_effect']:+.4f} (p={ns['software_p']:.2e})")
    print(f"    Interaction: {ns['interaction']:+.4f} CI {ns['interaction_ci']}")

    is_knockout = (ac['survives'] and ns['network_p'] < 0.05 and ns['software_p'] < 0.05)
    print(f"\n  KNOCKOUT: {'YES' if is_knockout else 'NO'}")
    if is_knockout:
        print(f"  The Noether structural prediction survives activation control")
        print(f"  AND the product-group decomposition (network × software) holds.")
    else:
        reasons = []
        if not ac['survives']:
            reasons.append("Noether effect doesn't survive activation control")
        if ns['network_p'] >= 0.05:
            reasons.append("Network main effect not significant")
        if ns['software_p'] >= 0.05:
            reasons.append("Software main effect not significant")
        print(f"  Reasons: {'; '.join(reasons)}")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_knockout',
        'results': results,
        'is_knockout': is_knockout,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_knockout.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
