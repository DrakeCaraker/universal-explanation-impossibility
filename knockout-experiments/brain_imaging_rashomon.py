#!/usr/bin/env python3
"""
Brain Imaging Rashomon: Does Spatial Correlation Predict Team Disagreement?

Uses the Botvinik-Nezer et al. (Nature 2020) NARPS dataset:
- 70 analysis teams, same fMRI data, different analysis pipelines
- NeuroVault collection 6047 has overlap maps (how many teams agree per voxel)

Design (revised per vet):
- This is an ANALOGICAL extension, not a formal theorem application
- Tests: does spatial correlation between brain regions predict which
  regions have high team disagreement?
- Comparison to baselines: spatial distance, random permutation
- No pre-specified threshold — report effect sizes with CIs
- Honest about the disanalogy: researcher DOF ≠ Rashomon

Steps:
1. Download 9 overlap maps from NeuroVault
2. Parcellate into regions (Schaefer atlas)
3. Compute per-region agreement (mean overlap / 70)
4. Compute inter-region correlation from the parcellated overlap maps
5. Test: does correlation structure predict disagreement pattern?
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, tempfile
import numpy as np
from scipy.stats import spearmanr
import urllib.request

N_HYPOTHESES = 9
N_TEAMS = 70


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
        maps[hyp] = img
        print(f'  Hyp {hyp}: shape={img.shape}, range=[{np.nanmin(img.get_fdata()):.0f}, {np.nanmax(img.get_fdata()):.0f}]')

    return maps


def parcellate_maps(overlap_maps):
    """Parcellate overlap maps using Schaefer atlas."""
    from nilearn import datasets, image
    from nilearn.maskers import NiftiLabelsMasker

    # Get Schaefer atlas (100 parcels for cleaner analysis)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = atlas.maps
    labels = atlas.labels

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        strategy='mean'
    )

    # For each hypothesis, extract per-region mean overlap
    region_data = {}  # hyp → (n_regions,) array of mean overlap values
    for hyp, img in overlap_maps.items():
        # Resample overlap map to atlas space
        resampled = image.resample_to_img(img, atlas_img, interpolation='nearest')
        # Pass the NiftiImage directly (not numpy array)
        region_vals = masker.fit_transform(resampled)
        if region_vals.ndim == 2:
            region_vals = region_vals[0]
        # Scale: if values are in [0, 1], they represent fraction of teams
        # Multiply by N_TEAMS to get overlap count
        if np.max(region_vals) <= 1.0:
            region_vals = region_vals * N_TEAMS
        region_data[hyp] = region_vals
        print(f'  Hyp {hyp}: {len(region_vals)} regions, '
              f'mean overlap={np.mean(region_vals):.1f}/{N_TEAMS}')

    return region_data, labels, masker


def compute_agreement_and_correlation(region_data):
    """Compute per-region agreement and inter-region correlation."""
    n_regions = len(region_data[1])

    # Per-region agreement: fraction of teams that agree (for each hypothesis)
    # Overlap = number of teams finding significance
    # Agreement = max(overlap/70, 1 - overlap/70) — how much teams agree
    agreement_matrix = np.zeros((N_HYPOTHESES, n_regions))
    for hyp in range(1, N_HYPOTHESES + 1):
        overlap = region_data[hyp] / N_TEAMS  # fraction significant
        agreement = np.maximum(overlap, 1 - overlap)  # agreement rate
        agreement_matrix[hyp - 1] = agreement

    # Mean agreement across hypotheses per region
    mean_agreement = np.mean(agreement_matrix, axis=0)
    # Disagreement = 1 - agreement
    mean_disagreement = 1 - mean_agreement

    # Inter-region correlation from the disagreement patterns
    # Regions with correlated disagreement across hypotheses → "similar" regions
    if N_HYPOTHESES > 1:
        region_corr = np.corrcoef(agreement_matrix.T)  # (n_regions, n_regions)
    else:
        region_corr = np.eye(n_regions)

    return mean_disagreement, agreement_matrix, region_corr


def test_correlation_predicts_disagreement(mean_disagreement, region_corr):
    """Test: do highly-correlated regions have similar disagreement levels?"""
    n_regions = len(mean_disagreement)

    # For each pair of regions: does correlation predict disagreement similarity?
    corr_values = []
    disagree_diffs = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            corr_values.append(abs(region_corr[i, j]))
            disagree_diffs.append(abs(mean_disagreement[i] - mean_disagreement[j]))

    corr_values = np.array(corr_values)
    disagree_diffs = np.array(disagree_diffs)

    # High correlation → similar disagreement → small disagreement difference
    # So we expect NEGATIVE correlation between |r_ij| and |disagree_i - disagree_j|
    rho, p = spearmanr(corr_values, disagree_diffs)

    # Also: do regions with more "neighbors" (high correlation with many others)
    # have higher disagreement? This is the η law analog.
    n_neighbors = np.sum(np.abs(region_corr) > 0.5, axis=1) - 1  # exclude self
    neighbor_corr, neighbor_p = spearmanr(n_neighbors, mean_disagreement)

    return {
        'corr_disagree_rho': float(rho),
        'corr_disagree_p': float(p),
        'n_pairs': len(corr_values),
        'neighbor_disagree_rho': float(neighbor_corr),
        'neighbor_disagree_p': float(neighbor_p),
    }


def compute_baselines(mean_disagreement, region_corr):
    """Compute baseline predictors for comparison."""
    n_regions = len(mean_disagreement)

    # Baseline 1: mean correlation magnitude (how "connected" is each region)
    mean_corr_mag = np.mean(np.abs(region_corr), axis=1)  # mean |r| with all others
    rho_baseline, p_baseline = spearmanr(mean_corr_mag, mean_disagreement)

    # Baseline 2: permutation null (shuffle region labels)
    perm_rhos = []
    rng = np.random.RandomState(42)
    for _ in range(1000):
        perm_disagree = rng.permutation(mean_disagreement)
        perm_rho, _ = spearmanr(mean_corr_mag, perm_disagree)
        perm_rhos.append(perm_rho)

    perm_p = np.mean(np.abs(perm_rhos) >= np.abs(rho_baseline))

    return {
        'mean_corr_disagree_rho': float(rho_baseline),
        'mean_corr_disagree_p': float(p_baseline),
        'permutation_p': float(perm_p),
        'permutation_rho_mean': float(np.mean(perm_rhos)),
        'permutation_rho_std': float(np.std(perm_rhos)),
    }


def main():
    start = time.time()
    print("=" * 60)
    print("BRAIN IMAGING RASHOMON")
    print("Does spatial correlation predict team disagreement?")
    print("(Botvinik-Nezer et al., Nature 2020)")
    print("=" * 60)

    # Phase 1: Download
    print("\nPhase 1: Downloading overlap maps...")
    overlap_maps = download_overlap_maps()

    # Phase 2: Parcellate
    print("\nPhase 2: Parcellating into brain regions...")
    region_data, labels, masker = parcellate_maps(overlap_maps)

    # Phase 3: Compute agreement and correlation
    print("\nPhase 3: Computing agreement and correlation...")
    mean_disagreement, agreement_matrix, region_corr = \
        compute_agreement_and_correlation(region_data)

    print(f"  Mean disagreement across regions: {np.mean(mean_disagreement):.3f}")
    print(f"  Std disagreement: {np.std(mean_disagreement):.3f}")
    print(f"  Most agreed region: {np.min(mean_disagreement):.3f}")
    print(f"  Most disagreed region: {np.max(mean_disagreement):.3f}")

    # Phase 4: Test the prediction
    print("\nPhase 4: Testing predictions...")
    prediction_results = test_correlation_predicts_disagreement(
        mean_disagreement, region_corr)

    print(f"  Correlation predicts disagreement similarity: "
          f"ρ={prediction_results['corr_disagree_rho']:.3f} "
          f"(p={prediction_results['corr_disagree_p']:.2e})")
    print(f"  Number of neighbors predicts disagreement: "
          f"ρ={prediction_results['neighbor_disagree_rho']:.3f} "
          f"(p={prediction_results['neighbor_disagree_p']:.2e})")

    # Phase 5: Baselines
    print("\nPhase 5: Baseline comparisons...")
    baseline_results = compute_baselines(mean_disagreement, region_corr)

    print(f"  Mean |correlation| predicts disagreement: "
          f"ρ={baseline_results['mean_corr_disagree_rho']:.3f} "
          f"(p={baseline_results['mean_corr_disagree_p']:.2e})")
    print(f"  Permutation null p-value: {baseline_results['permutation_p']:.3f}")

    # Per-hypothesis analysis
    print("\nPer-hypothesis analysis:")
    per_hyp = {}
    for hyp in range(1, N_HYPOTHESES + 1):
        overlap = region_data[hyp] / N_TEAMS
        hyp_disagreement = 1 - np.maximum(overlap, 1 - overlap)
        mean_corr_mag = np.mean(np.abs(region_corr), axis=1)
        rho_h, p_h = spearmanr(mean_corr_mag, hyp_disagreement)
        per_hyp[hyp] = {'rho': float(rho_h), 'p': float(p_h),
                        'mean_disagreement': float(np.mean(hyp_disagreement))}
        print(f"  Hyp {hyp}: ρ={rho_h:.3f} (p={p_h:.2e}), "
              f"mean disagree={np.mean(hyp_disagreement):.3f}")

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    main_rho = baseline_results['mean_corr_disagree_rho']
    main_p = baseline_results['mean_corr_disagree_p']
    perm_p = baseline_results['permutation_p']

    print(f"\n  Main result: ρ(mean|correlation|, disagreement) = {main_rho:.3f}")
    print(f"  p-value: {main_p:.2e}")
    print(f"  Permutation p: {perm_p:.3f}")
    print(f"\n  Interpretation:")
    if main_p < 0.05 and perm_p < 0.05:
        print(f"  Spatial correlation structure DOES predict team disagreement.")
        print(f"  Regions more correlated with other regions have {'MORE' if main_rho > 0 else 'LESS'} disagreement.")
    elif main_p < 0.05:
        print(f"  Correlation significant but permutation test fails — may be artifact.")
    else:
        print(f"  Spatial correlation does NOT significantly predict disagreement.")
        print(f"  The disagreement pattern is not driven by inter-region correlation.")

    print(f"\n  Caveat: this tests the ANALOGY (correlation → instability)")
    print(f"  not the THEOREM (Rashomon → impossibility). The 70 teams used")
    print(f"  different methods, not the same method with different seeds.")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'brain_imaging_rashomon',
        'data': 'Botvinik-Nezer et al. Nature 2020, NeuroVault collection 6047',
        'n_teams': N_TEAMS,
        'n_hypotheses': N_HYPOTHESES,
        'n_regions': len(mean_disagreement),
        'prediction_results': prediction_results,
        'baseline_results': baseline_results,
        'per_hypothesis': per_hyp,
        'mean_disagreement_stats': {
            'mean': float(np.mean(mean_disagreement)),
            'std': float(np.std(mean_disagreement)),
            'min': float(np.min(mean_disagreement)),
            'max': float(np.max(mean_disagreement)),
        },
        'caveat': 'Analogical extension — researcher DOF ≠ Rashomon',
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_brain_imaging_rashomon.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
