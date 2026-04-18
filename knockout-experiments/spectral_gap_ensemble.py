#!/usr/bin/env python3
"""
Spectral Gap → DASH Ensemble Size

The spectral gap of the symmetry group G determines how fast DASH converges.
For S_k: λ₁ = (k-1)/k. The flip rate should decay as ~ (1/k)^M with
ensemble size M, where k is the group size.

Prediction: larger groups need MORE models for stability.
- Group size k=2: M ≈ 4 models for 5% flip rate
- Group size k=4: M ≈ 8 models
- Group size k=6: M ≈ 12 models

Test: train ensembles of varying M, measure flip rate, check if the
decay rate matches the spectral gap prediction.
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

N_MODELS_MAX = 50
M_VALUES = [2, 3, 5, 8, 10, 15, 20, 30, 50]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_data(n, p, g, k, rho, seed=0):
    rng = np.random.RandomState(seed)
    Sigma = np.eye(p)
    for gi in range(g):
        for i in range(k):
            for j in range(k):
                if i != j:
                    Sigma[gi*k+i, gi*k+j] = rho
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, p) @ L.T
    effects = np.array([1.0, -0.5, 0.3])[:g]
    y_lin = sum(effects[gi] * X[:, gi*k:(gi+1)*k].mean(axis=1) for gi in range(g))
    y = (y_lin + rng.randn(n)*0.5 > 0).astype(int)
    return X, y


def measure_flip_rate_at_M(all_shap, M, groups, n_bootstrap=20):
    """Compute within-group flip rate using M-model DASH ensembles."""
    n_models, n_test, P = all_shap.shape
    rng = np.random.RandomState(42)

    within_flips = []
    for _ in range(n_bootstrap):
        # Sample M models
        model_idx = rng.choice(n_models, M, replace=False)
        # DASH = ensemble average
        dash = np.mean(all_shap[model_idx], axis=0)  # (n_test, P)

        # For each within-group pair, check if DASH ranking agrees with individual model
        for fi in range(P):
            for fj in range(fi+1, P):
                if groups.get(fi, -1) == groups.get(fj, -2) and groups.get(fi, -1) >= 0:
                    # Check if DASH ranking of (fi, fj) flips across bootstrap samples
                    dash_diff = np.mean(dash[:, fi]) - np.mean(dash[:, fj])

                    # Compare to a DIFFERENT M-model ensemble
                    model_idx2 = rng.choice(n_models, M, replace=False)
                    dash2 = np.mean(all_shap[model_idx2], axis=0)
                    dash2_diff = np.mean(dash2[:, fi]) - np.mean(dash2[:, fj])

                    if abs(dash_diff) > 1e-10 and abs(dash2_diff) > 1e-10:
                        if np.sign(dash_diff) != np.sign(dash2_diff):
                            within_flips.append(1)
                        else:
                            within_flips.append(0)

    return float(np.mean(within_flips)) if within_flips else 0.0


def run_spectral_gap_test(rho, k):
    """Test spectral gap prediction for a specific group size k."""
    g = 3  # 3 groups
    P = g * k
    print(f"\n  ρ={rho}, k={k} (P={P}, G={g} groups)...")

    X, y = generate_data(2000, P, g, k, rho)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    groups = {gi*k+fi: gi for gi in range(g) for fi in range(k)}

    # Train all models
    all_shap = []
    for i in range(N_MODELS_MAX):
        seed = 42 + i
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
        all_shap.append(sv)
    all_shap = np.array(all_shap)

    # Measure flip rate at each M
    flip_rates = {}
    for M in M_VALUES:
        if M <= N_MODELS_MAX:
            fr = measure_flip_rate_at_M(all_shap, M, groups)
            flip_rates[M] = fr
            print(f"    M={M:3d}: flip rate = {fr:.3f}")

    # Spectral gap prediction: flip ~ (1/k)^M
    # More precisely: flip ≈ c · exp(-M · log(k))
    # where c is the base rate and log(k) = -log(1/k) is the rate constant
    spectral_rate = np.log(k)  # theoretical decay rate
    print(f"    Spectral gap prediction: decay rate = ln({k}) = {spectral_rate:.3f}")

    # Fit empirical decay rate: flip = a · exp(-b · M)
    Ms = np.array([M for M in sorted(flip_rates.keys())])
    frs = np.array([flip_rates[M] for M in sorted(flip_rates.keys())])
    valid = frs > 0.001  # only fit where flip rate is measurably > 0
    if valid.sum() >= 3:
        log_frs = np.log(frs[valid] + 1e-10)
        from numpy.polynomial import polynomial as Poly
        coeffs = np.polyfit(Ms[valid], log_frs, 1)
        empirical_rate = -coeffs[0]
        print(f"    Empirical decay rate = {empirical_rate:.3f}")
        print(f"    Ratio (empirical/spectral) = {empirical_rate/spectral_rate:.2f}")
    else:
        empirical_rate = 0
        print(f"    Insufficient data for decay rate fit")

    return {
        'rho': rho,
        'k': k,
        'P': P,
        'spectral_gap': float(1 - 1/k),
        'spectral_rate': float(spectral_rate),
        'empirical_rate': float(empirical_rate),
        'ratio': float(empirical_rate / spectral_rate) if spectral_rate > 0 else 0,
        'flip_rates': {str(M): float(fr) for M, fr in sorted(flip_rates.items())},
    }


def main():
    start = time.time()
    print("=" * 60)
    print("SPECTRAL GAP → DASH ENSEMBLE SIZE")
    print("Does group theory predict how many models you need?")
    print("=" * 60)

    results = {}

    # Test at ρ=0.95 (strong correlation, clear groups)
    for k in [2, 3, 4, 6]:
        key = f'k{k}'
        results[key] = run_spectral_gap_test(0.95, k)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SUMMARY: Spectral Gap Prediction")
    print(f"{'='*60}")
    print(f"\n{'k':>3s} {'Spectral rate':>13s} {'Empirical rate':>14s} {'Ratio':>7s}")
    print("-" * 40)
    for key, r in results.items():
        print(f"{r['k']:3d} {r['spectral_rate']:13.3f} {r['empirical_rate']:14.3f} "
              f"{r['ratio']:7.2f}")

    # Does the spectral gap predict the convergence rate?
    spectral_rates = [r['spectral_rate'] for r in results.values()]
    empirical_rates = [r['empirical_rate'] for r in results.values()]
    if len(spectral_rates) >= 3:
        corr, p_val = spearmanr(spectral_rates, empirical_rates)
    else:
        corr, p_val = 0, 1
    print(f"\n  Correlation (spectral vs empirical rate): {corr:.3f} (p={p_val:.3f})")
    print(f"  If ratio ≈ 1 for all k, the spectral gap is a perfect predictor.")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'spectral_gap_ensemble',
        'results': results,
        'spectral_empirical_correlation': float(corr),
        'p_value': float(p_val),
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_spectral_gap_ensemble.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
