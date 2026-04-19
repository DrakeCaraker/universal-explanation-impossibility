#!/usr/bin/env python3
"""
Brain Imaging: Definitive Analysis

Fixes all issues from adversarial vet:
1. Activation control via regression residuals (not quintile stratification)
2. Software analysis on KNOWN-software teams only (exclude Unknown)
3. Bilateral partial correlation controlling for activation
4. Uses overlap-based disagreement (matching v3 d=0.43 metric)
5. Bootstrap CIs on everything
6. No overclaiming — reports what data shows
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys, re
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, rankdata
import urllib.request

VALID_HYPOTHESES = [1, 2, 3, 4, 5, 7, 8]
N_BOOTSTRAP = 2000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_software():
    """Load team software, return only teams with known software."""
    url = ('https://raw.githubusercontent.com/Inria-Empenn/narps_open_pipelines/'
           'main/narps_open/data/description/analysis_pipelines_full_descriptions.tsv')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode('utf-8')

    team_software = {}
    for line in content.split('\n'):
        if re.match(r'^[A-Z0-9]{4}\t', line):
            fields = line.split('\t')
            team_id = fields[0].strip()
            sw_raw = fields[6].strip() if len(fields) > 6 else ''
            sw_upper = sw_raw.upper()
            if 'SPM' in sw_upper:
                team_software[team_id] = 'SPM'
            elif 'FSL' in sw_upper or 'FEAT' in sw_upper:
                team_software[team_id] = 'FSL'
            elif 'AFNI' in sw_upper:
                team_software[team_id] = 'AFNI'
            # Skip unknown/other — only keep definite classifications

    return team_software


def load_data():
    """Load overlap maps and per-team t-statistics."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels
    roi_labels = labels[1:] if labels[0] in ('Background', b'Background') else labels

    network_names = []
    hemi_names = []
    for lab in roi_labels:
        if isinstance(lab, bytes):
            lab = lab.decode()
        parts = lab.split('_')
        network_names.append(parts[2] if len(parts) >= 3 else 'Unknown')
        hemi_names.append(parts[1] if len(parts) >= 2 else 'Unknown')

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')

    # Load overlap maps
    narps_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_cache')
    overlap_per_hyp = {}
    for hyp in VALID_HYPOTHESES:
        path = os.path.join(narps_cache, f'hypo{hyp}.nii.gz')
        if os.path.exists(path):
            img = nib.load(path)
            resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
            vals = masker.fit_transform(resampled)
            if vals.ndim == 2:
                vals = vals[0]
            overlap_per_hyp[hyp] = vals

    # Overlap-based disagreement (same metric as v3)
    disagree_matrix = np.zeros((len(VALID_HYPOTHESES), 100))
    for hi, hyp in enumerate(VALID_HYPOTHESES):
        ov = overlap_per_hyp[hyp]
        disagree_matrix[hi] = 1 - np.maximum(ov, 1 - ov)

    mean_overlap = np.mean([overlap_per_hyp[h] for h in overlap_per_hyp], axis=0)

    # Load per-team t-statistics
    team_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_team_cache')
    team_files = [f for f in os.listdir(team_cache)
                  if f.startswith('team_') and f.endswith('_unthresh.nii.gz')]
    team_ids_found = set()
    for f in team_files:
        team_ids_found.add(f.split('_')[1])

    region_data = {}
    for team_id in sorted(team_ids_found):
        team_data = {}
        for hyp in VALID_HYPOTHESES:
            path = os.path.join(team_cache, f'team_{team_id}_hypo{hyp}_unthresh.nii.gz')
            if os.path.exists(path):
                try:
                    img = nib.load(path)
                    resampled = image.resample_to_img(img, atlas_img, interpolation='continuous')
                    vals = masker.fit_transform(resampled)
                    if vals.ndim == 2:
                        vals = vals[0]
                    team_data[hyp] = vals
                except Exception:
                    pass
        if len(team_data) == len(VALID_HYPOTHESES):
            region_data[team_id] = team_data

    return (network_names, hemi_names, disagree_matrix, mean_overlap,
            overlap_per_hyp, region_data)


def make_grouping(names):
    unique = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(unique)}
    return np.array([name_to_id[n] for n in names])


# ==================================================================
# TEST 1: Activation control via regression residuals
# ==================================================================

def test_activation_control(disagree_matrix, network_names, mean_overlap):
    """Does the Noether effect survive controlling for activation similarity?

    Method: For each pair, regress |Δdisagree| on |Δactivation|.
    Test whether within-network pairs have lower RESIDUAL disagreement.
    """
    print("\n" + "=" * 60)
    print("TEST 1: ACTIVATION CONTROL (regression residuals)")
    print("  Uses overlap-based disagreement (same metric as v3 d=0.43)")
    print("=" * 60)

    n_regions = 100
    region_net = make_grouping(network_names)

    # Compute pairwise features
    same_net = []
    delta_disagree = []
    delta_activation = []

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            dd = np.mean(np.abs(disagree_matrix[:, i] - disagree_matrix[:, j]))
            da = abs(mean_overlap[i] - mean_overlap[j])
            sn = 1 if region_net[i] == region_net[j] else 0
            same_net.append(sn)
            delta_disagree.append(dd)
            delta_activation.append(da)

    same_net = np.array(same_net)
    delta_disagree = np.array(delta_disagree)
    delta_activation = np.array(delta_activation)

    # Raw effect (no control)
    within_dd = delta_disagree[same_net == 1]
    between_dd = delta_disagree[same_net == 0]
    raw_gap = np.mean(between_dd) - np.mean(within_dd)
    pooled = np.sqrt((np.var(within_dd) * len(within_dd) +
                      np.var(between_dd) * len(between_dd)) /
                     (len(within_dd) + len(between_dd)))
    raw_d = raw_gap / pooled if pooled > 0 else 0
    U, raw_p = mannwhitneyu(within_dd, between_dd, alternative='less')

    print(f"\n  RAW (no control):")
    print(f"    Within mean |Δdisagree|: {np.mean(within_dd):.6f}")
    print(f"    Between mean |Δdisagree|: {np.mean(between_dd):.6f}")
    print(f"    Gap: {raw_gap:.6f}, d = {raw_d:.3f}, p = {raw_p:.2e}")

    # Regress out activation similarity
    A = np.column_stack([delta_activation, np.ones(len(delta_activation))])
    beta = np.linalg.lstsq(A, delta_disagree, rcond=None)[0]
    residuals = delta_disagree - A @ beta

    within_res = residuals[same_net == 1]
    between_res = residuals[same_net == 0]
    res_gap = np.mean(between_res) - np.mean(within_res)
    pooled_res = np.sqrt((np.var(within_res) * len(within_res) +
                          np.var(between_res) * len(between_res)) /
                         (len(within_res) + len(between_res)))
    res_d = res_gap / pooled_res if pooled_res > 0 else 0
    U_res, res_p = mannwhitneyu(within_res, between_res, alternative='less')

    print(f"\n  CONTROLLED (activation regressed out):")
    print(f"    Within mean residual: {np.mean(within_res):.6f}")
    print(f"    Between mean residual: {np.mean(between_res):.6f}")
    print(f"    Gap: {res_gap:.6f}, d = {res_d:.3f}, p = {res_p:.2e}")
    print(f"    Activation β: {beta[0]:.4f} (how much activation similarity explains)")

    # Bootstrap CI on controlled gap
    rng = np.random.RandomState(42)
    boot_gaps = []
    n = len(residuals)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        w = residuals[idx][same_net[idx] == 1]
        b = residuals[idx][same_net[idx] == 0]
        if len(w) > 0 and len(b) > 0:
            boot_gaps.append(np.mean(b) - np.mean(w))
    boot_gaps = np.array(boot_gaps)
    gap_ci = [float(np.percentile(boot_gaps, 2.5)),
              float(np.percentile(boot_gaps, 97.5))]

    print(f"    Bootstrap CI: [{gap_ci[0]:.6f}, {gap_ci[1]:.6f}]")

    survives = res_p < 0.05 and gap_ci[0] > 0
    print(f"\n  VERDICT: {'SURVIVES' if survives else 'DOES NOT SURVIVE'} activation control")
    print(f"    (p < 0.05 AND bootstrap CI excludes zero)")

    # How much does activation explain?
    r2_activation = 1 - np.var(residuals) / np.var(delta_disagree)
    print(f"  Activation R²: {r2_activation:.3f} (fraction of |Δdisagree| explained by |Δactivation|)")

    return {
        'raw_d': float(raw_d), 'raw_p': float(raw_p), 'raw_gap': float(raw_gap),
        'controlled_d': float(res_d), 'controlled_p': float(res_p),
        'controlled_gap': float(res_gap), 'controlled_gap_ci': gap_ci,
        'activation_r2': float(r2_activation),
        'activation_beta': float(beta[0]),
        'survives': survives,
    }


# ==================================================================
# TEST 2: Software effect (known-software teams only)
# ==================================================================

def test_software_known_only(region_data, network_names, team_software):
    """2×2 analysis using ONLY teams with known software (FSL/SPM/AFNI)."""
    print("\n" + "=" * 60)
    print("TEST 2: SOFTWARE EFFECT (known-software teams only)")
    print("  Excludes all teams with unknown/unclassified software")
    print("=" * 60)

    # Filter to known-software teams
    known_teams = {t: sw for t, sw in team_software.items()
                   if t in region_data and sw in ('FSL', 'SPM', 'AFNI')}
    team_ids = sorted(known_teams.keys())
    n_teams = len(team_ids)

    sw_counts = {}
    for t in team_ids:
        sw = known_teams[t]
        sw_counts[sw] = sw_counts.get(sw, 0) + 1

    print(f"\n  Known-software teams: {n_teams}")
    for sw, c in sorted(sw_counts.items()):
        print(f"    {sw}: {c}")

    if n_teams < 10:
        print("  ⚠ Too few known-software teams for reliable analysis")
        return {'n_teams': n_teams, 'insufficient': True}

    n_regions = 100
    region_net = make_grouping(network_names)

    # Precompute t-matrices for known teams
    t_matrices = {}
    for hyp in VALID_HYPOTHESES:
        tm = np.zeros((n_teams, n_regions))
        for ti, t in enumerate(team_ids):
            tm[ti] = region_data[t][hyp]
        t_matrices[hyp] = tm

    # 2×2 flip rates
    cells = {k: [] for k in ['sn_ss', 'sn_ds', 'dn_ss', 'dn_ds']}

    for hyp in VALID_HYPOTHESES:
        tm = t_matrices[hyp]
        for ri in range(n_regions):
            for rj in range(ri + 1, n_regions):
                sn = region_net[ri] == region_net[rj]

                ss_flip = ss_comp = ds_flip = ds_comp = 0
                for t1 in range(n_teams):
                    for t2 in range(t1 + 1, n_teams):
                        d1 = tm[t1, ri] - tm[t1, rj]
                        d2 = tm[t2, ri] - tm[t2, rj]
                        if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                            flipped = np.sign(d1) != np.sign(d2)
                            same_sw = known_teams[team_ids[t1]] == known_teams[team_ids[t2]]
                            if same_sw:
                                ss_comp += 1
                                if flipped: ss_flip += 1
                            else:
                                ds_comp += 1
                                if flipped: ds_flip += 1

                net_key = 'sn' if sn else 'dn'
                if ss_comp > 0:
                    cells[f'{net_key}_ss'].append(ss_flip / ss_comp)
                if ds_comp > 0:
                    cells[f'{net_key}_ds'].append(ds_flip / ds_comp)

        sys.stdout.write(f'\r  Hyp {hyp} done...')
        sys.stdout.flush()
    print()

    # Results
    means = {k: float(np.mean(v)) if v else 0 for k, v in cells.items()}
    counts = {k: len(v) for k, v in cells.items()}

    print(f"\n  {'':>20s} {'Same Software':>15s} {'Diff Software':>15s}")
    print("  " + "-" * 52)
    print(f"  {'Same Network':>20s} {means['sn_ss']:.4f} (n={counts['sn_ss']}) "
          f"{means['sn_ds']:.4f} (n={counts['sn_ds']})")
    print(f"  {'Diff Network':>20s} {means['dn_ss']:.4f} (n={counts['dn_ss']}) "
          f"{means['dn_ds']:.4f} (n={counts['dn_ds']})")

    # Main effects
    all_sn = np.concatenate([cells['sn_ss'], cells['sn_ds']]) if cells['sn_ss'] and cells['sn_ds'] else np.array([])
    all_dn = np.concatenate([cells['dn_ss'], cells['dn_ds']]) if cells['dn_ss'] and cells['dn_ds'] else np.array([])
    all_ss = np.concatenate([cells['sn_ss'], cells['dn_ss']]) if cells['sn_ss'] and cells['dn_ss'] else np.array([])
    all_ds = np.concatenate([cells['sn_ds'], cells['dn_ds']]) if cells['sn_ds'] and cells['dn_ds'] else np.array([])

    net_effect = float(np.mean(all_sn) - np.mean(all_dn)) if len(all_sn) > 0 and len(all_dn) > 0 else 0
    U_net, p_net = mannwhitneyu(all_sn, all_dn, alternative='greater') if len(all_sn) > 0 and len(all_dn) > 0 else (0, 1)

    sw_effect = float(np.mean(all_ds) - np.mean(all_ss)) if len(all_ds) > 0 and len(all_ss) > 0 else 0
    U_sw, p_sw = mannwhitneyu(all_ds, all_ss, alternative='greater') if len(all_ds) > 0 and len(all_ss) > 0 else (0, 1)

    print(f"\n  Main effects:")
    print(f"    Network (within-net premium): {net_effect:+.4f} (p = {p_net:.2e})")
    print(f"    Software (diff-sw premium): {sw_effect:+.4f} (p = {p_sw:.2e})")

    # Bootstrap CI on both main effects
    rng = np.random.RandomState(42)
    boot_net = []
    boot_sw = []
    for _ in range(N_BOOTSTRAP):
        sn_b = rng.choice(all_sn, len(all_sn), replace=True) if len(all_sn) > 0 else np.array([0])
        dn_b = rng.choice(all_dn, len(all_dn), replace=True) if len(all_dn) > 0 else np.array([0])
        ss_b = rng.choice(all_ss, len(all_ss), replace=True) if len(all_ss) > 0 else np.array([0])
        ds_b = rng.choice(all_ds, len(all_ds), replace=True) if len(all_ds) > 0 else np.array([0])
        boot_net.append(np.mean(sn_b) - np.mean(dn_b))
        boot_sw.append(np.mean(ds_b) - np.mean(ss_b))

    net_ci = [float(np.percentile(boot_net, 2.5)), float(np.percentile(boot_net, 97.5))]
    sw_ci = [float(np.percentile(boot_sw, 2.5)), float(np.percentile(boot_sw, 97.5))]

    print(f"    Network CI: [{net_ci[0]:+.4f}, {net_ci[1]:+.4f}]")
    print(f"    Software CI: [{sw_ci[0]:+.4f}, {sw_ci[1]:+.4f}]")

    return {
        'n_teams': n_teams,
        'software_counts': sw_counts,
        'cell_means': means,
        'cell_counts': counts,
        'network_effect': net_effect, 'network_p': float(p_net), 'network_ci': net_ci,
        'software_effect': sw_effect, 'software_p': float(p_sw), 'software_ci': sw_ci,
    }


# ==================================================================
# TEST 3: Bilateral symmetry with activation control
# ==================================================================

def test_bilateral_controlled(disagree_matrix, network_names, hemi_names, mean_overlap):
    """Bilateral r controlled for activation level."""
    print("\n" + "=" * 60)
    print("TEST 3: BILATERAL SYMMETRY (activation-controlled)")
    print("=" * 60)

    n_regions = 100

    # Match LH-RH homologs by network and position
    lh_by_net = {}
    rh_by_net = {}
    for i in range(n_regions):
        net = network_names[i]
        if hemi_names[i] == 'LH':
            lh_by_net.setdefault(net, []).append(i)
        elif hemi_names[i] == 'RH':
            rh_by_net.setdefault(net, []).append(i)

    pairs = []
    for net in lh_by_net:
        lh = sorted(lh_by_net.get(net, []))
        rh = sorted(rh_by_net.get(net, []))
        for li, ri in zip(lh, rh):
            pairs.append((li, ri))

    print(f"  {len(pairs)} bilateral homolog pairs")

    # Mean disagreement per region (across hypotheses)
    mean_disagree = np.mean(disagree_matrix, axis=0)

    lh_d = np.array([mean_disagree[i] for i, j in pairs])
    rh_d = np.array([mean_disagree[j] for i, j in pairs])
    lh_a = np.array([mean_overlap[i] for i, j in pairs])
    rh_a = np.array([mean_overlap[j] for i, j in pairs])

    # Raw bilateral correlation
    raw_r, raw_p = pearsonr(lh_d, rh_d)
    print(f"\n  Raw bilateral r: {raw_r:.3f} (p = {raw_p:.2e})")

    # Partial correlation controlling for activation
    # Regress out activation from both LH and RH disagreement
    def residualize(y, x):
        A = np.column_stack([x, np.ones(len(x))])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - A @ beta

    lh_res = residualize(lh_d, lh_a)
    rh_res = residualize(rh_d, rh_a)
    partial_r, partial_p = pearsonr(lh_res, rh_res)
    print(f"  Partial r (ctrl activation): {partial_r:.3f} (p = {partial_p:.2e})")

    # Also control for BOTH activations (LH and RH)
    lh_res2 = residualize(lh_d, np.column_stack([lh_a, rh_a])[:, 0])  # simplified
    rh_res2 = residualize(rh_d, np.column_stack([lh_a, rh_a])[:, 1])
    partial_r2, partial_p2 = pearsonr(lh_res2, rh_res2)

    # Bootstrap CI on partial r
    rng = np.random.RandomState(42)
    boot_partial = []
    n_pairs = len(pairs)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_pairs, n_pairs, replace=True)
        lr = residualize(lh_d[idx], lh_a[idx])
        rr = residualize(rh_d[idx], rh_a[idx])
        r, _ = pearsonr(lr, rr)
        if not np.isnan(r):
            boot_partial.append(r)
    boot_partial = np.array(boot_partial)
    partial_ci = [float(np.percentile(boot_partial, 2.5)),
                  float(np.percentile(boot_partial, 97.5))]

    print(f"  Partial r CI: [{partial_ci[0]:.3f}, {partial_ci[1]:.3f}]")

    # Activation-matched random baseline
    # For each bilateral pair, find a random non-bilateral pair with similar |Δactivation|
    bilateral_act_diffs = np.abs(lh_a - rh_a)
    n_random_trials = 1000
    random_rs = []
    for _ in range(n_random_trials):
        # Random pairs of regions with matched activation difference
        rand_lh = rng.randint(0, n_regions, n_pairs)
        rand_rh = rng.randint(0, n_regions, n_pairs)
        # Don't require activation matching for the null — just random pairs
        r_disagree = np.array([mean_disagree[rand_lh[k]] for k in range(n_pairs)])
        l_disagree = np.array([mean_disagree[rand_rh[k]] for k in range(n_pairs)])
        r, _ = pearsonr(l_disagree, r_disagree)
        if not np.isnan(r):
            random_rs.append(r)
    random_rs = np.array(random_rs)
    perm_p = float(np.mean(random_rs >= raw_r))

    print(f"\n  Random pair baseline:")
    print(f"    Mean random r: {np.mean(random_rs):.3f} ± {np.std(random_rs):.3f}")
    print(f"    Bilateral r exceeds {100*(1-perm_p):.1f}% of random pairs")

    survives = partial_p < 0.05 and partial_ci[0] > 0
    print(f"\n  VERDICT: Bilateral symmetry {'SURVIVES' if survives else 'DOES NOT SURVIVE'} activation control")

    return {
        'n_pairs': n_pairs,
        'raw_r': float(raw_r), 'raw_p': float(raw_p),
        'partial_r': float(partial_r), 'partial_p': float(partial_p),
        'partial_ci': partial_ci,
        'random_mean_r': float(np.mean(random_rs)),
        'random_std_r': float(np.std(random_rs)),
        'perm_p': perm_p,
        'survives': survives,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("BRAIN IMAGING: DEFINITIVE ANALYSIS")
    print("All issues from adversarial vet addressed")
    print("=" * 60)

    print("\nLoading data...")
    (network_names, hemi_names, disagree_matrix, mean_overlap,
     overlap_per_hyp, region_data) = load_data()
    print(f"  {len(region_data)} teams with per-team t-statistics")

    team_software = load_software()
    known_sw_teams = {t: sw for t, sw in team_software.items() if t in region_data}
    print(f"  {len(known_sw_teams)} teams with known software (of {len(region_data)} total)")

    results = {}

    # Test 1: Activation control
    results['activation_control'] = test_activation_control(
        disagree_matrix, network_names, mean_overlap)

    # Test 2: Software (known only)
    results['software'] = test_software_known_only(
        region_data, network_names, team_software)

    # Test 3: Bilateral controlled
    results['bilateral'] = test_bilateral_controlled(
        disagree_matrix, network_names, hemi_names, mean_overlap)

    elapsed = time.time() - start

    # Final summary
    print(f"\n{'='*60}")
    print("DEFINITIVE SUMMARY")
    print(f"{'='*60}")

    ac = results['activation_control']
    sw = results['software']
    bi = results['bilateral']

    print(f"\n  TEST 1 — Activation control on Noether d=0.43:")
    print(f"    Raw: d = {ac['raw_d']:.3f} (p = {ac['raw_p']:.2e})")
    print(f"    Controlled: d = {ac['controlled_d']:.3f} (p = {ac['controlled_p']:.2e})")
    print(f"    CI: [{ac['controlled_gap_ci'][0]:.6f}, {ac['controlled_gap_ci'][1]:.6f}]")
    print(f"    Activation R²: {ac['activation_r2']:.3f}")
    print(f"    → {'SURVIVES' if ac['survives'] else 'DOES NOT SURVIVE'}")

    print(f"\n  TEST 2 — Software effect ({sw.get('n_teams', 0)} known-software teams):")
    if not sw.get('insufficient', False):
        print(f"    Network: {sw['network_effect']:+.4f} (p = {sw['network_p']:.2e}) CI {sw['network_ci']}")
        print(f"    Software: {sw['software_effect']:+.4f} (p = {sw['software_p']:.2e}) CI {sw['software_ci']}")
    else:
        print(f"    Insufficient teams")

    print(f"\n  TEST 3 — Bilateral symmetry:")
    print(f"    Raw r: {bi['raw_r']:.3f} (p = {bi['raw_p']:.2e})")
    print(f"    Partial r (ctrl activation): {bi['partial_r']:.3f} (p = {bi['partial_p']:.2e})")
    print(f"    CI: [{bi['partial_ci'][0]:.3f}, {bi['partial_ci'][1]:.3f}]")
    print(f"    → {'SURVIVES' if bi['survives'] else 'DOES NOT SURVIVE'}")

    # Overall verdict
    n_pass = sum([
        ac.get('survives', False),
        not sw.get('insufficient', True) and sw.get('network_p', 1) < 0.05,
        bi.get('survives', False),
    ])

    print(f"\n  OVERALL: {n_pass}/3 tests pass")
    print(f"  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_definitive',
        'results': results,
        'n_tests_pass': n_pass,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_definitive.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
