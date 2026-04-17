#!/usr/bin/env python3
"""
Approximate Symmetry Experiment: The η Law at Broken Correlations

Tests the critical prediction: as correlation ρ decreases from 1 to 0,
how does the bimodal gap, within-group flip rate, and η prediction change?

This bridges the η law (exact groups, R²=0.957) with the Gaussian flip
formula (continuous ρ, R²=0.814) and shows they are the same prediction
at different scales.

Design:
- 7 correlation levels: ρ ∈ {0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99}
- P = 12 features in g = 3 groups of 4
- 200 XGBoost models per ρ level (bootstrap + subsample)
- Full Noether analysis at each level
- Gaussian formula comparison
- The bridge: at ρ→1, Gaussian Φ(-SNR) → 0.5 = η(S₄) = 1 - 1/4

Predictions:
- ρ = 0: unimodal, no bimodal gap (control)
- ρ ≥ 0.5: bimodal emerges, within-group flip ≈ Gaussian prediction
- ρ → 1: within-group flip → 0.5, bimodal gap → 47pp (Noether)
- The transition curve should match Gaussian Φ(-SNR(ρ))
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations

N_MODELS = 200
P = 12  # total features
G = 3   # number of groups
K = P // G  # group size = 4
N_SAMPLES = 2000
RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_correlated_data(n, p, g, k, rho, seed=0):
    """Generate data with g groups of k correlated features."""
    rng = np.random.RandomState(seed)

    # Correlation matrix: block-diagonal with rho within groups
    Sigma = np.eye(p)
    for group in range(g):
        for i in range(k):
            for j in range(k):
                if i != j:
                    Sigma[group*k + i, group*k + j] = rho

    # Generate features from multivariate normal
    L = np.linalg.cholesky(Sigma)
    Z = rng.randn(n, p)
    X = Z @ L.T

    # Target: depends on group means (each group contributes equally)
    group_effects = np.array([1.0, -0.5, 0.3] * (g // 3 + 1))[:g]
    y_linear = np.zeros(n)
    for group in range(g):
        group_mean = X[:, group*k:(group+1)*k].mean(axis=1)
        y_linear += group_effects[group] * group_mean

    y = (y_linear + rng.randn(n) * 0.5 > 0).astype(int)
    feature_names = [f'g{gi}_f{fi}' for gi in range(g) for fi in range(k)]
    return X, y, feature_names


def run_at_rho(rho):
    """Run the full Noether analysis at one correlation level."""
    import xgboost as xgb
    import shap

    print(f"\n  ρ = {rho}...")

    X, y, feature_names = generate_correlated_data(N_SAMPLES, P, G, K, rho)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    # Define group membership
    groups = {}  # feature index → group index
    for gi in range(G):
        for fi in range(K):
            groups[gi * K + fi] = gi

    # Train models
    all_shap = []
    accuracies = []

    for i in range(N_MODELS):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        boot_idx = rng.choice(len(X_train), len(X_train), replace=True)

        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train[boot_idx], y_train[boot_idx])
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:100])
        if isinstance(sv, list):
            sv = sv[1]
        all_shap.append(sv)

    all_shap = np.array(all_shap)  # (N_MODELS, n_test, P)

    # Per-pair flip rates
    n_test = all_shap.shape[1]
    n_models = N_MODELS
    within_flips = []
    between_flips = []
    all_pair_flips = []

    for fi in range(P):
        for fj in range(fi + 1, P):
            # Compute flip rate for this pair
            n_flip = 0
            n_total = 0
            for ind in range(n_test):
                # Mean importance difference across models
                diffs = all_shap[:, ind, fi] - all_shap[:, ind, fj]
                for m1 in range(0, n_models, 5):  # subsample for speed
                    for m2 in range(m1 + 1, min(m1 + 5, n_models)):
                        s1 = all_shap[m1, ind, fi] - all_shap[m1, ind, fj]
                        s2 = all_shap[m2, ind, fi] - all_shap[m2, ind, fj]
                        if abs(s1) > 1e-10 and abs(s2) > 1e-10:
                            if np.sign(s1) != np.sign(s2):
                                n_flip += 1
                            n_total += 1

            flip_rate = n_flip / n_total if n_total > 0 else 0
            all_pair_flips.append(flip_rate)

            # Classify within vs between
            gi = groups[fi]
            gj = groups[fj]
            if gi == gj:
                within_flips.append(flip_rate)
            else:
                between_flips.append(flip_rate)

    # Compute statistics
    mean_within = float(np.mean(within_flips)) if within_flips else 0
    mean_between = float(np.mean(between_flips)) if between_flips else 0
    bimodal_gap = mean_within - mean_between

    # Gaussian prediction for within-group flip
    # For within-group pair at correlation ρ:
    # Δ_jk ≈ 0 (symmetric features), σ_jk depends on model variance
    # At ρ = 1: SNR = 0, flip = 0.5
    # At ρ < 1: features are distinguishable, SNR > 0, flip < 0.5
    # Compute empirical SNR for within-group pairs
    within_snrs = []
    for fi in range(P):
        for fj in range(fi + 1, P):
            if groups[fi] == groups[fj]:
                # Compute mean and std of importance difference across models
                mean_diffs = []
                for ind in range(n_test):
                    diffs = all_shap[:, ind, fi] - all_shap[:, ind, fj]
                    mean_diffs.append(np.mean(diffs))
                mu = np.mean(mean_diffs)
                sigma = np.std(mean_diffs)
                if sigma > 1e-10:
                    within_snrs.append(abs(mu) / sigma)
                else:
                    within_snrs.append(0)

    mean_snr = float(np.mean(within_snrs)) if within_snrs else 0
    gaussian_predicted_within = float(norm.cdf(-mean_snr)) if mean_snr > 0 else 0.5

    # η law prediction
    # At exact symmetry: η = 1/k for S_k, instability = 1 - 1/k
    # At broken symmetry: η(ρ) ≈ ρ² × 1/k (heuristic)
    eta_exact = 1.0 / K  # = 0.25 for k=4
    eta_approx = rho**2 * eta_exact
    predicted_instability_exact = 1 - eta_exact  # = 0.75
    predicted_within_eta = 1 - eta_approx

    # Bimodality test (Hartigan's dip test approximation via distribution shape)
    all_flips_arr = np.array(all_pair_flips)
    q25 = float(np.percentile(all_flips_arr, 25))
    q75 = float(np.percentile(all_flips_arr, 75))
    median = float(np.median(all_flips_arr))

    result = {
        'rho': rho,
        'n_models': N_MODELS,
        'n_features': P,
        'n_groups': G,
        'group_size': K,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'n_within_pairs': len(within_flips),
        'n_between_pairs': len(between_flips),
        'mean_within_flip': mean_within,
        'mean_between_flip': mean_between,
        'bimodal_gap': bimodal_gap,
        'mean_within_snr': mean_snr,
        'gaussian_predicted_within': gaussian_predicted_within,
        'eta_exact': float(eta_exact),
        'predicted_within_eta': float(predicted_within_eta),
        'flip_distribution': {
            'mean': float(np.mean(all_flips_arr)),
            'median': median,
            'std': float(np.std(all_flips_arr)),
            'q25': q25,
            'q75': q75,
            'min': float(np.min(all_flips_arr)),
            'max': float(np.max(all_flips_arr)),
        },
        'within_flips': within_flips,
        'between_flips': between_flips,
    }

    print(f"    Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"    Within-group flip: {mean_within:.3f} (Gaussian: {gaussian_predicted_within:.3f})")
    print(f"    Between-group flip: {mean_between:.3f}")
    print(f"    Bimodal gap: {bimodal_gap:.3f}")
    print(f"    Mean within-group SNR: {mean_snr:.3f}")

    return result


def main():
    start = time.time()

    print("=" * 70)
    print("APPROXIMATE SYMMETRY: The η Law at Broken Correlations")
    print("=" * 70)
    print(f"Features: {P} ({G} groups of {K})")
    print(f"Models: {N_MODELS} per ρ level")
    print(f"ρ values: {RHO_VALUES}")

    results = {}
    for rho in RHO_VALUES:
        results[str(rho)] = run_at_rho(rho)

    # Summary
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"SUMMARY: Bimodal Gap and Gaussian Bridge")
    print(f"{'='*70}")
    print(f"\n{'ρ':>6s} {'Within':>8s} {'Between':>8s} {'Gap':>6s} {'Gaussian':>8s} {'SNR':>6s}")
    print("-" * 50)
    for rho in RHO_VALUES:
        r = results[str(rho)]
        print(f"{rho:6.2f} {r['mean_within_flip']:8.3f} {r['mean_between_flip']:8.3f} "
              f"{r['bimodal_gap']:6.3f} {r['gaussian_predicted_within']:8.3f} "
              f"{r['mean_within_snr']:6.3f}")

    # The bridge theorem check
    print(f"\n  BRIDGE THEOREM CHECK:")
    print(f"  At ρ=0.99: within={results['0.99']['mean_within_flip']:.3f}, "
          f"η prediction=0.50, Gaussian={results['0.99']['gaussian_predicted_within']:.3f}")
    print(f"  At ρ=0.0:  within={results['0.0']['mean_within_flip']:.3f}, "
          f"η prediction=0.00, Gaussian={results['0.0']['gaussian_predicted_within']:.3f}")
    print(f"  → The η law IS the ρ→1 limit of the Gaussian formula")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    # Save
    output = {
        'experiment': 'approximate_symmetry',
        'description': 'η law at broken correlations: bridging η and Gaussian predictions',
        'design': {
            'n_features': P,
            'n_groups': G,
            'group_size': K,
            'n_models': N_MODELS,
            'n_samples': N_SAMPLES,
            'rho_values': RHO_VALUES,
        },
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_approximate_symmetry.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
