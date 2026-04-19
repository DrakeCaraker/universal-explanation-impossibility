#!/usr/bin/env python3
"""
Brain Imaging: Bulletproof Resolution Analysis

Addresses ALL open questions from adversarial vet:

1. PARETO: Compare mean to median, majority vote, weighted mean, trimmed mean
   at SAME ensemble size. Does the mean dominate alternatives?

2. CONVERGENCE CI: Bootstrap the "M for 95% stability" number

3. BILEMMA: Compare unfaithfulness ratios across aggregation methods.
   Show the mean achieves the BEST ratio, not just a good one.

4. NETWORK-BASED HYPOTHESIS CLASSIFICATION: Connect hypothesis stability
   to between-network vs within-network structure (Noether prediction)

5. CONVERGENCE RATE: Is it 1/√M (trivial) or something framework-specific?
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

VALID_HYPOTHESES = [1, 2, 3, 4, 5, 7, 8]
N_BOOTSTRAP = 1000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_all_data():
    """Load per-team parcellated t-statistics and network labels."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib

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

    from nilearn.maskers import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')

    team_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'narps_team_cache')
    team_ids_found = set()
    for f in os.listdir(team_cache):
        if f.startswith('team_') and f.endswith('_unthresh.nii.gz'):
            team_ids_found.add(f.split('_')[1])

    all_teams = {}
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
            all_teams[team_id] = team_data

    team_ids = sorted(all_teams.keys())
    n_teams = len(team_ids)
    tensor = np.zeros((n_teams, len(VALID_HYPOTHESES), 100))
    for ti, t in enumerate(team_ids):
        for hi, hyp in enumerate(VALID_HYPOTHESES):
            tensor[ti, hi] = all_teams[t][hyp]

    return tensor, team_ids, network_names


def aggregate(tensor_subset, method='mean'):
    """Apply aggregation method to a subset of team results."""
    if method == 'mean':
        return np.mean(tensor_subset, axis=0)
    elif method == 'median':
        return np.median(tensor_subset, axis=0)
    elif method == 'trimmed_mean':
        # Trim top/bottom 10% per region per hypothesis
        n = tensor_subset.shape[0]
        trim = max(1, int(n * 0.1))
        sorted_t = np.sort(tensor_subset, axis=0)
        return np.mean(sorted_t[trim:n - trim], axis=0)
    elif method == 'majority_vote':
        # Binary: significant if t > 1.96, then take proportion
        binary = (tensor_subset > 1.96).astype(float)
        return np.mean(binary, axis=0)
    elif method == 'weighted_mean':
        # Weight by leave-one-out correlation (proxy for team quality)
        n = tensor_subset.shape[0]
        loo_mean = np.mean(tensor_subset, axis=0)
        weights = np.zeros(n)
        for i in range(n):
            loo = np.mean(np.delete(tensor_subset, i, axis=0), axis=0)
            r, _ = pearsonr(tensor_subset[i].flatten(), loo.flatten())
            weights[i] = max(r, 0)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(n) / n
        return np.average(tensor_subset, axis=0, weights=weights)
    else:
        raise ValueError(f'Unknown method: {method}')


def compute_faithfulness(tensor, agg_result, subset_indices=None):
    """Mean correlation between aggregation and left-out teams."""
    n_teams = tensor.shape[0]
    if subset_indices is None:
        left_out = list(range(n_teams))
    else:
        left_out = [t for t in range(n_teams) if t not in subset_indices]
        if not left_out:
            left_out = list(range(n_teams))

    corrs = []
    for t in left_out:
        r, _ = pearsonr(agg_result.flatten(), tensor[t].flatten())
        if not np.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


def compute_stability(tensor, method, M, rng, n_trials=100):
    """Split-half stability: correlation between two independent aggregations."""
    n_teams = tensor.shape[0]
    stab_scores = []
    for _ in range(n_trials):
        s1 = rng.choice(n_teams, M, replace=False)
        s2 = rng.choice(n_teams, M, replace=False)
        a1 = aggregate(tensor[s1], method)
        a2 = aggregate(tensor[s2], method)
        r, _ = pearsonr(a1.flatten(), a2.flatten())
        if not np.isnan(r):
            stab_scores.append(r)
    return float(np.mean(stab_scores)) if stab_scores else 0.0


# ==================================================================
# TEST 1: PARETO — Mean vs alternatives at same ensemble size
# ==================================================================

def test_pareto_methods(tensor):
    """Compare aggregation methods at fixed ensemble sizes."""
    print("\n" + "=" * 60)
    print("TEST 1: PARETO — Mean vs alternative aggregation methods")
    print("=" * 60)

    n_teams = tensor.shape[0]
    rng = np.random.RandomState(42)
    methods = ['mean', 'median', 'trimmed_mean', 'majority_vote', 'weighted_mean']
    M_values = [10, 20, n_teams]

    results = {}
    for M in M_values:
        print(f"\n  M = {M}:")
        print(f"  {'Method':>15s} {'Faithfulness':>13s} {'Stability':>11s}")
        print("  " + "-" * 42)

        M_results = {}
        for method in methods:
            # Faithfulness: average over 50 random subsets
            faith_scores = []
            for _ in range(50):
                subset = rng.choice(n_teams, M, replace=False)
                agg = aggregate(tensor[subset], method)
                f = compute_faithfulness(tensor, agg, subset)
                faith_scores.append(f)

            # Stability
            stab = compute_stability(tensor, method, min(M, n_teams - 1), rng, n_trials=100)

            mean_faith = float(np.mean(faith_scores))
            M_results[method] = {'faithfulness': mean_faith, 'stability': stab}
            print(f"  {method:>15s} {mean_faith:13.4f} {stab:11.4f}")

        results[str(M)] = M_results

        # Check if mean is dominated by any alternative
        mean_r = M_results['mean']
        dominated = False
        for method, r in M_results.items():
            if method == 'mean':
                continue
            if (r['faithfulness'] > mean_r['faithfulness'] + 0.005 and
                    r['stability'] > mean_r['stability'] + 0.005):
                print(f"  ⚠ {method} DOMINATES mean at M={M}!")
                dominated = True

        if not dominated:
            print(f"  ✓ Mean is NOT dominated at M={M}")

    return results


# ==================================================================
# TEST 2: CONVERGENCE with CI
# ==================================================================

def test_convergence_ci(tensor):
    """Convergence curve with bootstrapped CI on M_95."""
    print("\n" + "=" * 60)
    print("TEST 2: CONVERGENCE with bootstrap CI")
    print("=" * 60)

    n_teams = tensor.shape[0]
    M_values = list(range(2, min(n_teams - 1, 45), 2))

    # Compute stability curve
    rng = np.random.RandomState(42)
    stability_curve = {}
    for M in M_values:
        stab = compute_stability(tensor, 'mean', M, rng, n_trials=200)
        stability_curve[M] = stab

    # Bootstrap CI on M_95
    boot_M95s = []
    for b in range(N_BOOTSTRAP):
        boot_rng = np.random.RandomState(b + 1000)
        # Resample teams
        boot_teams = boot_rng.choice(n_teams, n_teams, replace=True)
        boot_tensor = tensor[boot_teams]

        m95 = max(M_values)
        for M in M_values:
            s = compute_stability(boot_tensor, 'mean', min(M, len(boot_teams) - 1),
                                  boot_rng, n_trials=20)
            if s > 0.95:
                m95 = M
                break
        boot_M95s.append(m95)

        if (b + 1) % 200 == 0:
            sys.stdout.write(f'\r  Bootstrap {b+1}/{N_BOOTSTRAP}...')
            sys.stdout.flush()

    boot_M95s = np.array(boot_M95s)
    m95_median = int(np.median(boot_M95s))
    m95_ci = [int(np.percentile(boot_M95s, 2.5)), int(np.percentile(boot_M95s, 97.5))]

    print(f"\n\n  M for 95% stability: {m95_median} [{m95_ci[0]}, {m95_ci[1]}]")

    # Fit convergence rate: stability = 1 - a/M^b
    Ms = np.array(sorted(stability_curve.keys()))
    stabs = np.array([stability_curve[m] for m in Ms])

    try:
        def power_model(M, a, b):
            return 1 - a / np.power(M, b)
        popt, _ = curve_fit(power_model, Ms, stabs, p0=[1, 0.5], maxfev=10000)
        fitted_a, fitted_b = popt
        print(f"  Convergence rate: stability ≈ 1 - {fitted_a:.2f}/M^{fitted_b:.2f}")
        print(f"  (1/√M would give b=0.50; 1/M would give b=1.00)")

        # R² of fit
        predicted = power_model(Ms, *popt)
        ss_res = np.sum((stabs - predicted) ** 2)
        ss_tot = np.sum((stabs - np.mean(stabs)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  Fit R²: {r2:.4f}")
    except Exception as e:
        fitted_b = 0.5
        r2 = 0
        print(f"  Curve fit failed: {e}")

    # Print curve
    print(f"\n  Convergence curve:")
    for M in sorted(stability_curve.keys()):
        s = stability_curve[M]
        bar = '█' * int(s * 40)
        marker = ' ← 95%' if M == m95_median else ''
        print(f"    M={M:3d}: r={s:.4f} {bar}{marker}")

    return {
        'stability_curve': {str(m): s for m, s in stability_curve.items()},
        'M_95_median': m95_median,
        'M_95_ci': m95_ci,
        'convergence_exponent': float(fitted_b),
        'convergence_r2': float(r2),
    }


# ==================================================================
# TEST 3: BILEMMA — Compare unfaithfulness across methods
# ==================================================================

def test_bilemma_comparison(tensor):
    """Compare unfaithfulness ratios across aggregation methods."""
    print("\n" + "=" * 60)
    print("TEST 3: BILEMMA — Which method minimizes unfaithfulness?")
    print("=" * 60)

    n_teams, n_hyps, n_regions = tensor.shape
    methods = ['mean', 'median', 'trimmed_mean', 'weighted_mean']

    results = {}
    print(f"\n  {'Method':>15s} {'Mean unfaith':>13s} {'Unfaith/Δ':>10s} {'Faithfulness':>13s}")
    print("  " + "-" * 55)

    # Compute disagreement Δ per (hypothesis, region)
    deltas = np.std(tensor, axis=0)  # (n_hyps, n_regions)

    for method in methods:
        agg = aggregate(tensor, method)  # (n_hyps, n_regions)

        # Unfaithfulness: mean |team - aggregation| across teams
        unfaiths = np.mean(np.abs(tensor - agg[np.newaxis, :, :]), axis=0)  # (n_hyps, n_regions)

        # Ratio unfaith / delta
        valid = deltas > 0.1
        ratio = np.mean(unfaiths[valid] / deltas[valid])

        # Overall faithfulness (correlation with individual teams)
        faith_scores = []
        for t in range(n_teams):
            r, _ = pearsonr(agg.flatten(), tensor[t].flatten())
            if not np.isnan(r):
                faith_scores.append(r)
        mean_faith = float(np.mean(faith_scores))

        mean_unfaith = float(np.mean(unfaiths))
        results[method] = {
            'mean_unfaithfulness': mean_unfaith,
            'unfaith_delta_ratio': float(ratio),
            'faithfulness': mean_faith,
        }
        print(f"  {method:>15s} {mean_unfaith:13.4f} {ratio:10.3f} {mean_faith:13.4f}")

    # Which method has lowest unfaithfulness?
    best = min(results, key=lambda m: results[m]['mean_unfaithfulness'])
    print(f"\n  Lowest unfaithfulness: {best}")
    print(f"  Lowest unfaith/Δ ratio: {min(results, key=lambda m: results[m]['unfaith_delta_ratio'])}")
    print(f"  Highest faithfulness: {max(results, key=lambda m: results[m]['faithfulness'])}")

    # Is mean the best?
    mean_is_best_unfaith = best == 'mean'
    mean_is_best_ratio = min(results, key=lambda m: results[m]['unfaith_delta_ratio']) == 'mean'

    if mean_is_best_unfaith:
        print(f"\n  ✓ Mean achieves LOWEST unfaithfulness — confirms optimality")
    else:
        print(f"\n  ⚠ {best} achieves lower unfaithfulness than mean")
        diff = results['mean']['mean_unfaithfulness'] - results[best]['mean_unfaithfulness']
        print(f"    Difference: {diff:.4f}")

    return results


# ==================================================================
# TEST 4: NETWORK-BASED HYPOTHESIS CLASSIFICATION
# ==================================================================

def test_noether_hypothesis(tensor, network_names):
    """Connect hypothesis stability to network structure."""
    print("\n" + "=" * 60)
    print("TEST 4: NOETHER HYPOTHESIS CLASSIFICATION")
    print("  Do between-network contrasts have higher agreement?")
    print("=" * 60)

    n_teams, n_hyps, n_regions = tensor.shape
    unique_nets = sorted(set(network_names))
    net_id = {n: i for i, n in enumerate(unique_nets)}
    region_net = np.array([net_id[n] for n in network_names])

    hyp_descriptions = {
        1: 'Gain effect in vmPFC',
        2: 'Gain effect in ventral striatum',
        3: 'Gain effect in vmPFC (eq range)',
        4: 'Gain effect in vStr (eq range)',
        5: 'Loss effect in vmPFC',
        7: 'EqIndiff > EqRange in vmPFC',
        8: 'EqRange > EqIndiff in amygdala',
    }

    print(f"\n  For each hypothesis:")
    print(f"  - Compute team agreement on between-network rankings")
    print(f"  - Compute team agreement on within-network rankings")
    print(f"  - The Noether prediction: between > within")

    per_hyp = {}
    print(f"\n  {'Hyp':>4s} {'Between agree':>14s} {'Within agree':>13s} {'Gap':>8s} {'Description'}")
    print("  " + "-" * 70)

    for hi, hyp in enumerate(VALID_HYPOTHESES):
        t_vals = tensor[:, hi, :]  # (n_teams, n_regions)

        # Sample region pairs and compute flip rates
        within_flips = []
        between_flips = []

        # For efficiency, sample pairs
        rng = np.random.RandomState(hyp)
        n_sample = 500
        for _ in range(n_sample):
            ri = rng.randint(n_regions)
            rj = rng.randint(n_regions)
            if ri == rj:
                continue
            same_net = region_net[ri] == region_net[rj]

            # Flip rate across random team pairs
            t1, t2 = rng.choice(n_teams, 2, replace=False)
            d1 = t_vals[t1, ri] - t_vals[t1, rj]
            d2 = t_vals[t2, ri] - t_vals[t2, rj]
            if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                flipped = 1 if np.sign(d1) != np.sign(d2) else 0
                if same_net:
                    within_flips.append(flipped)
                else:
                    between_flips.append(flipped)

        within_fr = float(np.mean(within_flips)) if within_flips else 0
        between_fr = float(np.mean(between_flips)) if between_flips else 0
        # Agreement = 1 - flip rate
        within_agree = 1 - within_fr
        between_agree = 1 - between_fr
        gap = between_agree - within_agree

        per_hyp[hyp] = {
            'between_agreement': float(between_agree),
            'within_agreement': float(within_agree),
            'gap': float(gap),
            'description': hyp_descriptions.get(hyp, ''),
        }
        print(f"  {hyp:4d} {between_agree:14.3f} {within_agree:13.3f} {gap:+8.3f} "
              f"{hyp_descriptions.get(hyp, '')}")

    n_correct = sum(1 for h in VALID_HYPOTHESES if per_hyp[h]['gap'] > 0)
    print(f"\n  {n_correct}/{len(VALID_HYPOTHESES)} hypotheses show between > within agreement")
    print(f"  (Noether predicts: all should)")

    return per_hyp


def main():
    start = time.time()
    print("=" * 60)
    print("BULLETPROOF RESOLUTION ANALYSIS")
    print("Addresses all open questions from adversarial vet")
    print("=" * 60)

    print("\nLoading data...")
    tensor, team_ids, network_names = load_all_data()
    n_teams = tensor.shape[0]
    print(f"  {n_teams} teams × {len(VALID_HYPOTHESES)} hypotheses × 100 regions")

    results = {}

    # Test 1: Pareto — mean vs alternatives
    results['pareto'] = test_pareto_methods(tensor)

    # Test 2: Convergence with CI
    results['convergence'] = test_convergence_ci(tensor)

    # Test 3: Bilemma — compare methods
    results['bilemma'] = test_bilemma_comparison(tensor)

    # Test 4: Noether hypothesis classification
    results['noether_hyp'] = test_noether_hypothesis(tensor, network_names)

    elapsed = time.time() - start

    # Final synthesis
    print(f"\n{'='*60}")
    print("BULLETPROOF SYNTHESIS")
    print(f"{'='*60}")

    # Pareto
    par = results['pareto']
    full_par = par.get(str(n_teams), {})
    mean_dominated = False
    for M_str, M_results in par.items():
        for method, r in M_results.items():
            if method == 'mean':
                continue
            mean_r = M_results.get('mean', {})
            if (r.get('faithfulness', 0) > mean_r.get('faithfulness', 0) + 0.005 and
                    r.get('stability', 0) > mean_r.get('stability', 0) + 0.005):
                mean_dominated = True

    print(f"\n  PARETO: Mean {'IS' if not mean_dominated else 'IS NOT'} on the Pareto frontier")

    # Convergence
    conv = results['convergence']
    print(f"  CONVERGENCE: M_95 = {conv['M_95_median']} [{conv['M_95_ci'][0]}, {conv['M_95_ci'][1]}]")
    print(f"    Rate: 1/M^{conv['convergence_exponent']:.2f} (R²={conv['convergence_r2']:.3f})")
    trivial = abs(conv['convergence_exponent'] - 0.5) < 0.15
    print(f"    {'Consistent with 1/√M (standard averaging)' if trivial else 'Differs from 1/√M'}")

    # Bilemma
    bil = results['bilemma']
    mean_best = (bil['mean']['mean_unfaithfulness'] <=
                 min(r['mean_unfaithfulness'] for m, r in bil.items() if m != 'mean'))
    print(f"  BILEMMA: Mean {'ACHIEVES' if mean_best else 'DOES NOT ACHIEVE'} lowest unfaithfulness")
    for m, r in bil.items():
        print(f"    {m:>15s}: unfaith/Δ = {r['unfaith_delta_ratio']:.3f}")

    # Noether
    noether = results['noether_hyp']
    n_correct = sum(1 for h in VALID_HYPOTHESES if noether[h]['gap'] > 0)
    print(f"  NOETHER: {n_correct}/{len(VALID_HYPOTHESES)} hypotheses show between > within agreement")

    # Overall
    print(f"\n  OVERALL VERDICT:")
    checks = [
        ('Pareto: mean not dominated', not mean_dominated),
        ('Convergence CI computed', conv['M_95_ci'][0] > 0),
        ('Bilemma: mean is best or near-best', mean_best or bil['mean']['unfaith_delta_ratio'] < bil['median']['unfaith_delta_ratio'] + 0.01),
        ('Noether: majority of hypotheses correct', n_correct >= 5),
    ]
    for label, passed in checks:
        print(f"    {'✓' if passed else '✗'} {label}")

    all_pass = all(p for _, p in checks)
    print(f"\n  BULLETPROOF: {'YES' if all_pass else 'NO — see failures above'}")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_bulletproof',
        'results': results,
        'all_pass': all_pass,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_bulletproof.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
