#!/usr/bin/env python3
"""
Attribution PCA Eigenspectrum Test

Tests whether the eigenspectrum of SHAP values across multiple models
reveals the symmetry group structure:

Phase 1: SYNTHETIC (known groups)
  - g=3 groups of k=4 features, ρ=0.95
  - 50 models, SHAP PCA
  - Prediction: eigenspectrum gap at g=3 (3 stable directions)
  - η from PCA should match (k-1)/k = 0.75

Phase 2: SYNTHETIC (varying ρ)
  - Same structure at ρ = 0.5, 0.7, 0.9, 0.95, 0.99
  - Does the gap sharpen with increasing ρ?
  - Does η_PCA interpolate between 0 (ρ=0) and (k-1)/k (ρ=1)?

Phase 3: REAL DATA (breast cancer, unknown groups)
  - Does the eigenspectrum show a gap?
  - Does the gap predict which SHAP rankings are stable vs unstable?

Phase 4: NARPS (48 teams × 100 regions)
  - Does the eigenspectrum gap appear near 7 (one per network)?
  - This would mean the team disagreement has exactly the dimensionality
    of the functional network structure — without being told about networks.

Methodology:
  - Non-overlapping SHAP computation (each model trained independently)
  - PCA on the centered SHAP matrix
  - Scree plot analysis for gap detection
  - Comparison to the exact η law prediction
  - Bootstrap CIs on eigenvalue ratios
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_MODELS = 50
N_BOOTSTRAP = 1000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_grouped_data(n, g, k, rho, seed=0):
    """Generate data with g groups of k correlated features."""
    P = g * k
    rng = np.random.RandomState(seed)
    Sigma = np.eye(P)
    for gi in range(g):
        for i in range(k):
            for j in range(k):
                if i != j:
                    Sigma[gi * k + i, gi * k + j] = rho
    L = np.linalg.cholesky(Sigma + np.eye(P) * 1e-8)
    X = rng.randn(n, P) @ L.T
    effects = np.array([1.0, -0.5, 0.3])[:g]
    y_lin = sum(effects[gi] * X[:, gi * k:(gi + 1) * k].mean(axis=1) for gi in range(g))
    y = (y_lin + rng.randn(n) * 0.5 > 0).astype(int)
    groups = {gi * k + fi: gi for gi in range(g) for fi in range(k)}
    return X, y, groups


def train_and_shap(X_train, y_train, X_test, n_models, seed_offset=42):
    """Train n_models and compute SHAP for each."""
    all_shap = []
    for i in range(n_models):
        seed = seed_offset + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, use_label_encoder=False,
            eval_metric='logloss', verbosity=0)
        model.fit(X_train[idx], y_train[idx])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:100])
        if isinstance(sv, list):
            sv = sv[1]
        # Average SHAP across test samples to get one P-vector per model
        all_shap.append(np.mean(sv, axis=0))
    return np.array(all_shap)  # (n_models, P)


def analyze_eigenspectrum(shap_matrix, n_groups_expected=None, label=""):
    """PCA on the SHAP matrix. Returns eigenspectrum analysis."""
    n_models, P = shap_matrix.shape

    # Center the SHAP matrix
    centered = shap_matrix - shap_matrix.mean(axis=0)

    # Covariance matrix of SHAP across models
    cov = np.cov(centered.T)  # (P, P)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical cleanup
    total_var = eigenvalues.sum()

    if total_var < 1e-15:
        print(f"  ⚠ No SHAP variance — all models agree perfectly")
        return None

    # Cumulative explained variance
    cumvar = np.cumsum(eigenvalues) / total_var
    # Eigenvalue ratios (consecutive)
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-15)

    # Find the gap: largest ratio between consecutive eigenvalues
    if len(ratios) > 0:
        gap_idx = np.argmax(ratios[:min(P - 1, 20)])  # look in first 20
        gap_ratio = ratios[gap_idx]
        n_stable_pcs = gap_idx + 1
    else:
        gap_idx = 0
        gap_ratio = 1
        n_stable_pcs = 1

    # η from PCA: fraction of SHAP variance NOT in the top g PCs
    if n_groups_expected is not None:
        g = n_groups_expected
        eta_pca = 1 - cumvar[g - 1] if g <= len(cumvar) else 0
    else:
        g = n_stable_pcs
        eta_pca = 1 - cumvar[g - 1] if g <= len(cumvar) else 0

    # Bootstrap CI on η_PCA
    rng = np.random.RandomState(42)
    boot_etas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_models, n_models, replace=True)
        boot_centered = shap_matrix[idx] - shap_matrix[idx].mean(axis=0)
        boot_cov = np.cov(boot_centered.T)
        boot_eig = np.linalg.eigvalsh(boot_cov)[::-1]
        boot_eig = np.maximum(boot_eig, 0)
        boot_total = boot_eig.sum()
        if boot_total > 1e-15:
            g_use = n_groups_expected if n_groups_expected else n_stable_pcs
            if g_use <= len(boot_eig):
                boot_etas.append(1 - np.sum(boot_eig[:g_use]) / boot_total)
    boot_etas = np.array(boot_etas)
    eta_ci = [float(np.percentile(boot_etas, 2.5)),
              float(np.percentile(boot_etas, 97.5))] if len(boot_etas) > 0 else [0, 0]

    # Print scree plot
    print(f"\n  Eigenspectrum{' (' + label + ')' if label else ''}:")
    print(f"    P={P}, n_models={n_models}")
    n_show = min(P, 15)
    print(f"    {'PC':>4s} {'Eigenvalue':>12s} {'CumVar':>8s} {'Ratio':>8s}")
    print("    " + "-" * 38)
    for i in range(n_show):
        ratio_str = f"{ratios[i]:.1f}" if i < len(ratios) else ""
        marker = " ← GAP" if i == gap_idx else ""
        print(f"    {i + 1:4d} {eigenvalues[i]:12.6f} {cumvar[i]:8.3f} {ratio_str:>8s}{marker}")

    print(f"\n    Gap at PC {gap_idx + 1} → {gap_idx + 2} (ratio={gap_ratio:.1f})")
    print(f"    Stable PCs (above gap): {n_stable_pcs}")
    if n_groups_expected:
        print(f"    Expected (n_groups): {n_groups_expected}")
        print(f"    Match: {'YES' if n_stable_pcs == n_groups_expected else 'NO'}")
    print(f"    η_PCA = {eta_pca:.3f} [{eta_ci[0]:.3f}, {eta_ci[1]:.3f}]")

    return {
        'n_stable_pcs': int(n_stable_pcs),
        'gap_ratio': float(gap_ratio),
        'gap_position': int(gap_idx + 1),
        'eta_pca': float(eta_pca),
        'eta_ci': eta_ci,
        'eigenvalues': eigenvalues[:n_show].tolist(),
        'cumvar': cumvar[:n_show].tolist(),
        'n_groups_expected': n_groups_expected,
        'match': n_stable_pcs == n_groups_expected if n_groups_expected else None,
    }


def validate_stability(shap_matrix, n_stable_pcs):
    """Validate: do features loading on stable PCs have lower flip rates?"""
    n_models, P = shap_matrix.shape
    centered = shap_matrix - shap_matrix.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sort]

    # For each feature: how much of its SHAP variance is in the stable PCs?
    stable_loading = np.sum(eigenvectors[:, :n_stable_pcs] ** 2, axis=1)

    # For each feature: compute flip rate across model pairs
    flip_rates = []
    for fi in range(P):
        flips = 0
        total = 0
        for m1 in range(n_models):
            for m2 in range(m1 + 1, n_models):
                if abs(shap_matrix[m1, fi]) > 1e-10 and abs(shap_matrix[m2, fi]) > 1e-10:
                    if np.sign(shap_matrix[m1, fi]) != np.sign(shap_matrix[m2, fi]):
                        flips += 1
                    total += 1
        flip_rates.append(flips / total if total > 0 else 0)
    flip_rates = np.array(flip_rates)

    # Correlation: stable loading → lower flip rate?
    rho, p = spearmanr(stable_loading, flip_rates)

    print(f"\n  VALIDATION: stable PC loading vs flip rate")
    print(f"    Spearman ρ = {rho:.3f} (p = {p:.2e})")
    print(f"    Expected: NEGATIVE (higher stable loading → lower flip rate)")

    return {
        'loading_flip_correlation': float(rho),
        'loading_flip_p': float(p),
        'stable_loadings': stable_loading.tolist(),
        'flip_rates': flip_rates.tolist(),
    }


# ==================================================================
# PHASE 1: Synthetic with known groups
# ==================================================================

def phase1_synthetic():
    """Test eigenspectrum on synthetic data with known groups."""
    print("\n" + "=" * 60)
    print("PHASE 1: SYNTHETIC DATA (g=3, k=4, ρ=0.95)")
    print("  Prediction: gap at PC 3, η_PCA = 0.75")
    print("=" * 60)

    g, k, rho = 3, 4, 0.95
    P = g * k
    eta_exact = (k - 1) / k

    X, y, groups = generate_grouped_data(2000, g, k, rho)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print(f"\n  Training {N_MODELS} models...")
    shap_matrix = train_and_shap(X_train, y_train, X_test, N_MODELS)

    result = analyze_eigenspectrum(shap_matrix, n_groups_expected=g, label=f"g={g},k={k},ρ={rho}")
    print(f"\n  η exact = {eta_exact:.3f}")
    print(f"  η PCA = {result['eta_pca']:.3f}")
    print(f"  Gap at {result['gap_position']}, expected at {g}")

    validation = validate_stability(shap_matrix, result['n_stable_pcs'])
    result['validation'] = validation
    result['eta_exact'] = float(eta_exact)

    return result


# ==================================================================
# PHASE 2: Varying ρ
# ==================================================================

def phase2_varying_rho():
    """Does the gap sharpen with increasing ρ?"""
    print("\n" + "=" * 60)
    print("PHASE 2: VARYING CORRELATION (ρ)")
    print("  Does the eigenvalue gap sharpen with ρ?")
    print("=" * 60)

    g, k = 3, 4
    rho_values = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    results = {}

    for rho in rho_values:
        X, y, groups = generate_grouped_data(2000, g, k, rho, seed=int(rho * 100))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        shap_matrix = train_and_shap(X_train, y_train, X_test, N_MODELS, seed_offset=42 + int(rho * 100))
        result = analyze_eigenspectrum(shap_matrix, n_groups_expected=g, label=f"ρ={rho}")
        results[str(rho)] = result

    # Summary
    print(f"\n  SUMMARY: η_PCA vs ρ")
    print(f"  {'ρ':>6s} {'η_PCA':>8s} {'η_exact':>8s} {'Gap pos':>8s} {'Gap ratio':>10s} {'Match?':>7s}")
    print("  " + "-" * 55)
    eta_exact = (k - 1) / k
    for rho in rho_values:
        r = results[str(rho)]
        match = '✓' if r['n_stable_pcs'] == g else '✗'
        print(f"  {rho:6.2f} {r['eta_pca']:8.3f} {eta_exact:8.3f} {r['gap_position']:8d} "
              f"{r['gap_ratio']:10.1f} {match:>7s}")

    return results


# ==================================================================
# PHASE 3: Real data (breast cancer)
# ==================================================================

def phase3_real_data():
    """Does the eigenspectrum reveal structure in real data?"""
    print("\n" + "=" * 60)
    print("PHASE 3: BREAST CANCER (unknown groups)")
    print("  Does the eigenspectrum show a gap?")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print(f"\n  Training {N_MODELS} models on {X.shape[1]} features...")
    shap_matrix = train_and_shap(X_train, y_train, X_test, N_MODELS)

    result = analyze_eigenspectrum(shap_matrix, label="Breast Cancer")
    validation = validate_stability(shap_matrix, result['n_stable_pcs'])
    result['validation'] = validation

    # Which features load on stable PCs?
    n_stable = result['n_stable_pcs']
    centered = shap_matrix - shap_matrix.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sort]

    stable_loading = np.sum(eigenvectors[:, :n_stable] ** 2, axis=1)
    top_stable = np.argsort(stable_loading)[::-1][:5]
    top_unstable = np.argsort(stable_loading)[:5]

    print(f"\n  Most STABLE features (high stable-PC loading):")
    for fi in top_stable:
        print(f"    {feature_names[fi]:>25s}: loading={stable_loading[fi]:.3f}, "
              f"flip={validation['flip_rates'][fi]:.3f}")
    print(f"\n  Most UNSTABLE features (low stable-PC loading):")
    for fi in top_unstable:
        print(f"    {feature_names[fi]:>25s}: loading={stable_loading[fi]:.3f}, "
              f"flip={validation['flip_rates'][fi]:.3f}")

    return result


# ==================================================================
# PHASE 4: NARPS
# ==================================================================

def phase4_narps():
    """Does the NARPS eigenspectrum show a gap near 7 (networks)?"""
    print("\n" + "=" * 60)
    print("PHASE 4: NARPS (48 teams × 100 regions × 7 hypotheses)")
    print("  Prediction: gap near 7 (one per functional network)")
    print("=" * 60)

    # Load per-team parcellated data
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib
    from nilearn import image

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
    n_networks = len(set(network_names))

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy='mean')
    valid_hyps = [1, 2, 3, 4, 5, 7, 8]

    team_cache = os.path.join(SCRIPT_DIR, 'narps_team_cache')
    team_ids_found = set()
    for f in os.listdir(team_cache):
        if f.startswith('team_') and f.endswith('_unthresh.nii.gz'):
            team_ids_found.add(f.split('_')[1])

    # Build team × (regions*hypotheses) matrix
    all_teams = {}
    for team_id in sorted(team_ids_found):
        team_vec = []
        complete = True
        for hyp in valid_hyps:
            path = os.path.join(team_cache, f'team_{team_id}_hypo{hyp}_unthresh.nii.gz')
            if os.path.exists(path):
                try:
                    img = nib.load(path)
                    resampled = image.resample_to_img(img, atlas_img, interpolation='continuous')
                    vals = masker.fit_transform(resampled)
                    if vals.ndim == 2:
                        vals = vals[0]
                    team_vec.extend(vals)
                except Exception:
                    complete = False
            else:
                complete = False
        if complete and len(team_vec) == 100 * len(valid_hyps):
            all_teams[team_id] = np.array(team_vec)

    team_ids = sorted(all_teams.keys())
    n_teams = len(team_ids)
    narps_matrix = np.array([all_teams[t] for t in team_ids])  # (n_teams, 700)

    print(f"  {n_teams} teams × {narps_matrix.shape[1]} dimensions")
    print(f"  Expected gap position: {n_networks} (number of functional networks)")

    result = analyze_eigenspectrum(narps_matrix, n_groups_expected=n_networks, label="NARPS")

    print(f"\n  Gap found at: PC {result['gap_position']}")
    print(f"  Expected at: PC {n_networks}")

    return result


def main():
    start = time.time()
    print("=" * 60)
    print("ATTRIBUTION PCA EIGENSPECTRUM TEST")
    print("Does the SHAP eigenspectrum reveal symmetry group structure?")
    print("=" * 60)

    results = {}

    results['phase1_synthetic'] = phase1_synthetic()
    results['phase2_varying_rho'] = phase2_varying_rho()
    results['phase3_breast_cancer'] = phase3_real_data()
    results['phase4_narps'] = phase4_narps()

    elapsed = time.time() - start

    # Synthesis
    print(f"\n{'='*60}")
    print("SYNTHESIS")
    print(f"{'='*60}")

    p1 = results['phase1_synthetic']
    p4 = results['phase4_narps']

    print(f"\n  SYNTHETIC (known groups):")
    print(f"    Gap at PC {p1['gap_position']} (expected: 3)")
    print(f"    η_PCA = {p1['eta_pca']:.3f} (expected: 0.750)")
    print(f"    Loading-flip ρ = {p1['validation']['loading_flip_correlation']:.3f}")

    print(f"\n  NARPS (functional networks):")
    print(f"    Gap at PC {p4['gap_position']} (expected: 7)")
    print(f"    η_PCA = {p4['eta_pca']:.3f}")

    p3 = results['phase3_breast_cancer']
    print(f"\n  BREAST CANCER (unknown groups):")
    print(f"    Gap at PC {p3['gap_position']}")
    print(f"    η_PCA = {p3['eta_pca']:.3f}")
    print(f"    Loading-flip ρ = {p3['validation']['loading_flip_correlation']:.3f}")

    # Does it work?
    synthetic_works = p1['match'] and p1['validation']['loading_flip_p'] < 0.05
    narps_close = abs(p4['gap_position'] - 7) <= 3
    bc_predicts = p3['validation']['loading_flip_p'] < 0.05

    print(f"\n  VERDICT:")
    print(f"    Synthetic gap matches prediction: {'YES' if p1['match'] else 'NO'}")
    print(f"    Stable loading predicts flip rate (synthetic): {'YES' if p1['validation']['loading_flip_p'] < 0.05 else 'NO'}")
    print(f"    NARPS gap near 7: {'YES' if narps_close else 'NO'} (found: {p4['gap_position']})")
    print(f"    Stable loading predicts flip rate (breast cancer): {'YES' if bc_predicts else 'NO'}")

    if synthetic_works and bc_predicts:
        print(f"\n  ✓ The eigenspectrum diagnostic WORKS:")
        print(f"    - Recovers known group structure in synthetic data")
        print(f"    - Predicts attribution stability in real data")
        print(f"    - This is a practical tool for any SHAP user")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'attribution_pca_eigenspectrum',
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(SCRIPT_DIR, 'results_attribution_pca_eigenspectrum.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
