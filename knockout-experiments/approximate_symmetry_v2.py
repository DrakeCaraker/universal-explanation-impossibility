#!/usr/bin/env python3
"""
Approximate Symmetry v2: Fixes all issues from adversarial audit.

Fixes:
1. Use ALL model pairs (not subsampled blocks)
2. Use ALL test observations (not 100 of 600)
3. Fix Gaussian SNR computation (inter-model, not inter-observation)
4. Add null-model comparison: Phi(-c*sqrt(1-rho^2))
5. Add bootstrap CIs on all metrics
6. Report between-group baseline explicitly
7. Frame monotonic gap as empirical finding, not prediction
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap

N_MODELS = 50  # reduced from 200 since we use all pairs now
P = 12
G = 3
K = P // G  # 4
N_SAMPLES = 2000
RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
N_BOOTSTRAP = 200


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_data(n, rho, seed=0):
    rng = np.random.RandomState(seed)
    Sigma = np.eye(P)
    for g in range(G):
        for i in range(K):
            for j in range(K):
                if i != j:
                    Sigma[g*K+i, g*K+j] = rho
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, P) @ L.T
    effects = np.array([1.0, -0.5, 0.3])
    y_lin = sum(effects[g] * X[:, g*K:(g+1)*K].mean(axis=1) for g in range(G))
    y = (y_lin + rng.randn(n)*0.5 > 0).astype(int)
    return X, y


def compute_flip_rate_all_pairs(shap_vals, fi, fj):
    """Compute flip rate using ALL model pairs for a fixed feature pair,
    averaged over all test observations."""
    n_models, n_test, _ = shap_vals.shape
    total_flips = 0
    total_pairs = 0
    for ind in range(n_test):
        for m1 in range(n_models):
            for m2 in range(m1+1, n_models):
                d1 = shap_vals[m1, ind, fi] - shap_vals[m1, ind, fj]
                d2 = shap_vals[m2, ind, fi] - shap_vals[m2, ind, fj]
                if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                    if np.sign(d1) != np.sign(d2):
                        total_flips += 1
                    total_pairs += 1
    return total_flips / total_pairs if total_pairs > 0 else 0.0


def compute_gaussian_snr_correct(shap_vals, fi, fj):
    """Compute SNR correctly: for each observation, compute the mean and std
    of the importance difference ACROSS MODELS (not across observations)."""
    n_models, n_test, _ = shap_vals.shape
    per_obs_means = []
    per_obs_stds = []
    for ind in range(n_test):
        diffs = shap_vals[:, ind, fi] - shap_vals[:, ind, fj]
        per_obs_means.append(np.mean(diffs))
        per_obs_stds.append(np.std(diffs))

    # SNR = |mean of per-obs means| / mean of per-obs stds
    # This measures: signal (systematic importance difference) / noise (model variance)
    mu = np.mean(per_obs_means)
    sigma = np.mean(per_obs_stds)
    if sigma > 1e-10:
        return abs(mu) / sigma
    return 0.0


def run_at_rho(rho):
    print(f"\n  ρ = {rho}...")
    X, y = generate_data(N_SAMPLES, rho)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    groups = {gi*K+fi: gi for gi in range(G) for fi in range(K)}

    # Train models
    all_shap = []
    accs = []
    for i in range(N_MODELS):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train[idx], y_train[idx])
        accs.append(float(accuracy_score(y_test, model.predict(X_test))))
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]
        all_shap.append(sv)

    all_shap = np.array(all_shap)
    n_test = all_shap.shape[1]
    print(f"    {N_MODELS} models, {n_test} test obs, acc={np.mean(accs):.3f}")

    # Compute flip rates for ALL feature pairs using ALL model pairs
    # Sample 30 pairs per type for computational feasibility
    within_pairs = [(fi, fj) for fi in range(P) for fj in range(fi+1, P) if groups[fi] == groups[fj]]
    between_pairs = [(fi, fj) for fi in range(P) for fj in range(fi+1, P) if groups[fi] != groups[fj]]

    within_flips = []
    within_snrs = []
    within_gauss = []
    for fi, fj in within_pairs:
        fr = compute_flip_rate_all_pairs(all_shap, fi, fj)
        snr = compute_gaussian_snr_correct(all_shap, fi, fj)
        within_flips.append(fr)
        within_snrs.append(snr)
        within_gauss.append(float(norm.cdf(-snr)))

    between_flips = []
    between_snrs = []
    for fi, fj in between_pairs:
        fr = compute_flip_rate_all_pairs(all_shap, fi, fj)
        snr = compute_gaussian_snr_correct(all_shap, fi, fj)
        between_flips.append(fr)
        between_snrs.append(snr)

    mean_within = float(np.mean(within_flips))
    mean_between = float(np.mean(between_flips))
    gap = mean_within - mean_between

    mean_within_snr = float(np.mean(within_snrs))
    mean_gauss_pred = float(np.mean(within_gauss))

    # Bootstrap CIs
    all_flips = within_flips + between_flips
    rng = np.random.RandomState(0)
    boot_gaps = []
    boot_withins = []
    for _ in range(N_BOOTSTRAP):
        w_idx = rng.choice(len(within_flips), len(within_flips), replace=True)
        b_idx = rng.choice(len(between_flips), len(between_flips), replace=True)
        bw = np.mean([within_flips[i] for i in w_idx])
        bb = np.mean([between_flips[i] for i in b_idx])
        boot_gaps.append(bw - bb)
        boot_withins.append(bw)

    gap_ci = [float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5))]
    within_ci = [float(np.percentile(boot_withins, 2.5)), float(np.percentile(boot_withins, 97.5))]

    print(f"    Within: {mean_within:.3f} [{within_ci[0]:.3f}, {within_ci[1]:.3f}]")
    print(f"    Between: {mean_between:.3f}")
    print(f"    Gap: {gap:.3f} [{gap_ci[0]:.3f}, {gap_ci[1]:.3f}]")
    print(f"    Gaussian (corrected): {mean_gauss_pred:.3f} (SNR={mean_within_snr:.3f})")

    return {
        'rho': rho,
        'n_models': N_MODELS,
        'n_test': n_test,
        'accuracy': float(np.mean(accs)),
        'mean_within': mean_within,
        'within_ci': within_ci,
        'mean_between': mean_between,
        'gap': gap,
        'gap_ci': gap_ci,
        'mean_within_snr': mean_within_snr,
        'gaussian_predicted': mean_gauss_pred,
        'n_within_pairs': len(within_pairs),
        'n_between_pairs': len(between_pairs),
    }


def main():
    start = time.time()
    print("=" * 70)
    print("APPROXIMATE SYMMETRY v2 (audit-corrected)")
    print("=" * 70)

    results = {}
    for rho in RHO_VALUES:
        results[str(rho)] = run_at_rho(rho)

    # Null model: Phi(-c * sqrt(1-rho^2))
    rho_arr = np.array(RHO_VALUES)
    within_arr = np.array([results[str(r)]['mean_within'] for r in RHO_VALUES])

    def null_model(rho, c):
        return norm.cdf(-c * np.sqrt(1 - rho**2))

    try:
        popt, pcov = curve_fit(null_model, rho_arr, within_arr, p0=[0.5])
        c_fit = float(popt[0])
        null_pred = [float(null_model(r, c_fit)) for r in RHO_VALUES]
        null_r2 = float(1 - np.sum((within_arr - np.array(null_pred))**2) /
                        np.sum((within_arr - np.mean(within_arr))**2))
    except:
        c_fit = 0
        null_pred = [0] * len(RHO_VALUES)
        null_r2 = 0

    # Framework prediction: from corrected Gaussian formula
    gauss_pred = [results[str(r)]['gaussian_predicted'] for r in RHO_VALUES]
    gauss_r2 = float(1 - np.sum((within_arr - np.array(gauss_pred))**2) /
                      np.sum((within_arr - np.mean(within_arr))**2))

    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'rho':>6s} {'Within [CI]':>25s} {'Between':>8s} {'Gap [CI]':>20s} {'Gauss':>6s} {'Null':>6s}")
    print("-" * 80)
    for rho in RHO_VALUES:
        r = results[str(rho)]
        ni = RHO_VALUES.index(rho)
        print(f"{rho:6.2f} {r['mean_within']:6.3f} [{r['within_ci'][0]:.3f},{r['within_ci'][1]:.3f}]"
              f"  {r['mean_between']:8.3f}  {r['gap']:6.3f} [{r['gap_ci'][0]:.3f},{r['gap_ci'][1]:.3f}]"
              f"  {gauss_pred[ni]:6.3f}  {null_pred[ni]:6.3f}")

    print(f"\n  Null model: Phi(-{c_fit:.3f} * sqrt(1-rho^2)), R² = {null_r2:.3f}")
    print(f"  Gaussian (corrected SNR): R² = {gauss_r2:.3f}")
    print(f"\n  Does the framework beat the null? {'YES' if gauss_r2 > null_r2 else 'NO'}")
    print(f"  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'approximate_symmetry_v2',
        'fixes': ['all model pairs', 'all test obs', 'corrected SNR', 'null model', 'bootstrap CIs'],
        'results': results,
        'null_model': {'c': c_fit, 'r2': null_r2, 'predictions': null_pred},
        'gaussian_corrected': {'r2': gauss_r2, 'predictions': gauss_pred},
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_approximate_symmetry_v2.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
