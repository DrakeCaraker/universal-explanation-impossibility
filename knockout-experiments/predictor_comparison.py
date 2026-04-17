#!/usr/bin/env python3
"""
Head-to-Head Predictor Comparison: Which Tool Should Practitioners Use?

Compares ALL available instability predictors on the SAME datasets
with the SAME models and the SAME evaluation metric.

Predictors:
1. Pairwise Pearson |ρ|        — just the raw correlation (trivial baseline)
2. Φ(-c√(1-ρ²))               — null model from approximate symmetry experiment
3. Coverage conflict (minority fraction) — nonparametric, 7 lines
4. Gaussian flip formula Φ(-SNR) — the paper's parametric prediction
5. η-law based (group η for the pair's cluster) — the framework's prediction

Evaluation:
- Per-pair Spearman ρ between predicted and observed flip rates
- This is the SAME metric used in the Ostrowski session (empirical-validation-results.md)
- Higher = better prediction of which pairs are unstable

Datasets: Breast Cancer, California Housing, German Credit (real data)
+ Synthetic at ρ = {0.5, 0.7, 0.9} (controlled data)
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
import shap

N_MODELS = 50
CLUSTER_THRESHOLD = 0.8


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_synthetic(n, p, g, k, rho, seed=0):
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


def run_comparison(X, y, feature_names, dataset_name, task='classification'):
    """Run all 5 predictors on one dataset and compare."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
        stratify=y if task == 'classification' else None
    )

    # Compute correlation matrix on training data
    corr = np.abs(np.corrcoef(X_train.T))
    P = X.shape[1]

    # Train models
    all_shap = []
    for i in range(N_MODELS):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, use_label_encoder=False,
                eval_metric='logloss', verbosity=0)
        else:
            model = xgb.XGBRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbosity=0)
        model.fit(X_train[idx], y_train[idx])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:200])
        if isinstance(sv, list):
            sv = sv[1]
        all_shap.append(sv)

    all_shap = np.array(all_shap)  # (N_MODELS, n_test, P)
    n_test = all_shap.shape[1]

    # Compute per-pair observed flip rates (ground truth)
    # Sample pairs for speed
    pairs = []
    for fi in range(P):
        for fj in range(fi+1, P):
            pairs.append((fi, fj))

    if len(pairs) > 500:
        rng = np.random.RandomState(0)
        pair_idx = rng.choice(len(pairs), 500, replace=False)
        pairs = [pairs[i] for i in pair_idx]

    observed_flips = []
    pred_corr = []       # 1. raw |ρ|
    pred_null = []       # 2. Φ(-1.06√(1-ρ²))
    pred_cc = []         # 3. coverage conflict (minority fraction)
    pred_gaussian = []   # 4. Φ(-SNR)

    for fi, fj in pairs:
        # Observed flip rate
        n_flip = 0
        n_total = 0
        for ind in range(min(n_test, 50)):  # subsample observations for speed
            for m1 in range(N_MODELS):
                for m2 in range(m1+1, N_MODELS):
                    d1 = all_shap[m1, ind, fi] - all_shap[m1, ind, fj]
                    d2 = all_shap[m2, ind, fi] - all_shap[m2, ind, fj]
                    if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                        if np.sign(d1) != np.sign(d2):
                            n_flip += 1
                        n_total += 1
        obs_flip = n_flip / n_total if n_total > 0 else 0
        observed_flips.append(obs_flip)

        # Predictor 1: raw |ρ|
        rho_ij = corr[fi, fj]
        pred_corr.append(rho_ij)

        # Predictor 2: null model Φ(-1.06√(1-ρ²))
        pred_null.append(float(norm.cdf(-1.06 * np.sqrt(1 - rho_ij**2))))

        # Predictor 3: coverage conflict (minority fraction)
        # For each observation, check sign of SHAP difference across models
        n_pos = 0
        n_neg = 0
        for m in range(N_MODELS):
            mean_diff = np.mean(all_shap[m, :min(n_test, 50), fi] -
                               all_shap[m, :min(n_test, 50), fj])
            if mean_diff > 0:
                n_pos += 1
            else:
                n_neg += 1
        minority = min(n_pos, n_neg) / N_MODELS
        pred_cc.append(minority)

        # Predictor 4: Gaussian Φ(-SNR)
        # SNR = |mean importance diff| / std importance diff across models
        model_diffs = []
        for m in range(N_MODELS):
            mean_d = np.mean(all_shap[m, :min(n_test, 50), fi] -
                             all_shap[m, :min(n_test, 50), fj])
            model_diffs.append(mean_d)
        mu = np.mean(model_diffs)
        sigma = np.std(model_diffs)
        snr = abs(mu) / sigma if sigma > 1e-10 else 0
        pred_gaussian.append(float(norm.cdf(-snr)))

    # Compute Spearman ρ for each predictor
    obs = np.array(observed_flips)

    results = {}
    for name, pred in [
        ('raw_correlation', pred_corr),
        ('null_model', pred_null),
        ('coverage_conflict', pred_cc),
        ('gaussian_formula', pred_gaussian),
    ]:
        rho_s, p_val = spearmanr(pred, obs)
        results[name] = {
            'spearman': float(rho_s),
            'p_value': float(p_val),
        }
        print(f"  {name:25s}: Spearman ρ = {rho_s:.3f} (p = {p_val:.2e})")

    # Winner
    winner = max(results, key=lambda k: results[k]['spearman'])
    print(f"\n  WINNER: {winner} (ρ = {results[winner]['spearman']:.3f})")

    return {
        'dataset': dataset_name,
        'n_features': P,
        'n_pairs': len(pairs),
        'n_models': N_MODELS,
        'results': results,
        'winner': winner,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("HEAD-TO-HEAD PREDICTOR COMPARISON")
    print("Which tool should practitioners actually use?")
    print("=" * 60)

    all_results = {}

    # Real datasets
    bc = load_breast_cancer()
    all_results['breast_cancer'] = run_comparison(
        bc.data, bc.target, list(bc.feature_names), 'Breast Cancer')

    cal = fetch_california_housing()
    all_results['california'] = run_comparison(
        cal.data, cal.target, list(cal.feature_names), 'California Housing',
        task='regression')

    # German Credit
    try:
        data = np.loadtxt(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")
        X_gc = data[:, :-1]
        y_gc = (data[:, -1] == 1).astype(int)
        all_results['german_credit'] = run_comparison(
            X_gc, y_gc, [f'f{i}' for i in range(X_gc.shape[1])], 'German Credit')
    except:
        print("\n  German Credit download failed — skipping")

    # Synthetic
    for rho in [0.5, 0.7, 0.9]:
        X, y = generate_synthetic(2000, 12, 3, 4, rho)
        all_results[f'synthetic_rho{rho}'] = run_comparison(
            X, y, [f'f{i}' for i in range(12)], f'Synthetic ρ={rho}')

    # Summary
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SUMMARY: Which predictor wins?")
    print(f"{'='*60}")
    print(f"\n{'Dataset':25s} {'Corr':>7s} {'Null':>7s} {'CC':>7s} {'Gauss':>7s} {'Winner':>15s}")
    print("-" * 75)
    for name, r in all_results.items():
        res = r['results']
        print(f"{name:25s} "
              f"{res['raw_correlation']['spearman']:7.3f} "
              f"{res['null_model']['spearman']:7.3f} "
              f"{res['coverage_conflict']['spearman']:7.3f} "
              f"{res['gaussian_formula']['spearman']:7.3f} "
              f"{r['winner']:>15s}")

    # Count wins
    from collections import Counter
    wins = Counter(r['winner'] for r in all_results.values())
    print(f"\n  Win count: {dict(wins)}")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'predictor_comparison',
        'predictors': ['raw_correlation', 'null_model', 'coverage_conflict', 'gaussian_formula'],
        'metric': 'Spearman rho between predicted and observed per-pair flip rates',
        'results': all_results,
        'win_count': dict(wins),
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_predictor_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
