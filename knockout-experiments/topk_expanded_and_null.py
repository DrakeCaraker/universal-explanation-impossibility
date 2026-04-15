#!/usr/bin/env python3
"""
Top-K Expansion (15 datasets) + Null Model Comparison

Task 1: Expand top-K regulatory gap to 15 bridge datasets (gain-based)
Task 2: Null model — does the inversion appear without Rashomon?
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from scipy.stats import fisher_exact, norm
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def compute_topk_reliability(snrs_by_pair, feature_mean_importance, K_values):
    """Compute reliability at each K level."""
    P = len(feature_mean_importance)
    ranked_features = np.argsort(-feature_mean_importance)

    results = {}
    for K in K_values:
        if K > P:
            continue
        top_k = set(ranked_features[:K].tolist())
        n_reliable = 0
        n_unreliable = 0
        n_marginal = 0
        n_total = 0

        for (j, k), snr in snrs_by_pair.items():
            if j in top_k and k in top_k:
                n_total += 1
                if snr > 2.0:
                    n_reliable += 1
                elif snr < 0.5:
                    n_unreliable += 1
                else:
                    n_marginal += 1

        results[K] = {
            "n_pairs": n_total,
            "n_reliable": n_reliable,
            "n_unreliable": n_unreliable,
            "n_marginal": n_marginal,
            "pct_reliable": round(n_reliable / n_total * 100, 1) if n_total > 0 else 0,
            "pct_unreliable": round(n_unreliable / n_total * 100, 1) if n_total > 0 else 0,
            "gap": round((1 - n_reliable / n_total) * 100, 1) if n_total > 0 else 100,
        }
    return results


def task1_topk_15datasets():
    """Expand top-K to 15 bridge datasets."""
    print("=" * 60)
    print("TASK 1: Top-K Regulatory Gap (15 datasets, gain-based)")
    print("=" * 60)

    bridge = json.load(open(OUT_DIR / 'results_explanation_landscape_bridge_expanded.json'))
    datasets = bridge['per_dataset']

    K_values = [3, 5, 10, 'all']
    all_topk = {}
    aggregates = {K: {"reliable": 0, "unreliable": 0, "total": 0} for K in K_values}

    # We need per-pair SNR data. The bridge JSON has per-dataset summary stats
    # but not per-pair data. We need to recompute from the importance matrices.
    # Since we don't have those saved, we'll use the coverage_conflict_by_threshold
    # to estimate the distribution.
    #
    # Actually, we can compute top-K from the coverage conflict data:
    # The bridge stores coverage_conflict_degree = fraction with SNR < 0.5
    # and reliable_fraction = fraction with SNR > 2.
    # These are for ALL pairs. For top-K, we need to know the SNR distribution
    # among the top features specifically.
    #
    # Since we don't have per-pair data saved, let's recompute for a subset
    # using the same infrastructure.

    # For a proper analysis, we need the actual importance matrices.
    # Let's recompute for all 15 datasets.

    import xgboost as xgb
    from sklearn.datasets import (load_breast_cancer, load_wine, load_iris,
                                   load_digits, fetch_openml, fetch_covtype)
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    def load_all_datasets():
        datasets = []
        # Same datasets as the bridge experiment
        bc = load_breast_cancer()
        datasets.append(("Breast Cancer", bc.data, bc.target, list(bc.feature_names), 2))

        wine = load_wine()
        datasets.append(("Wine", wine.data, wine.target, list(wine.feature_names), 3))

        iris = load_iris()
        datasets.append(("Iris", iris.data, iris.target, list(iris.feature_names), 3))

        digits = load_digits()
        names_d = [f"pixel_{i}" for i in range(64)]
        datasets.append(("Digits", digits.data, digits.target, names_d, 10))

        for oml_name, display_name, nc in [
            ('heart-statlog', 'Heart Disease', 2),
            ('diabetes', 'Diabetes', 2),
            ('credit-g', 'German Credit', 2),
            ('ionosphere', 'Ionosphere', 2),
            ('sonar', 'Sonar', 2),
            ('vehicle', 'Vehicle', 4),
            ('segment', 'Segment', 7),
            ('satimage', 'Satimage', 6),
            ('vowel', 'Vowel', 11),
        ]:
            try:
                X, y = fetch_openml(oml_name, version=1, return_X_y=True, as_frame=True, parser='auto')
                names = list(X.columns)
                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        X[col] = OrdinalEncoder().fit_transform(X[[col]]).ravel()
                X_np = X.values.astype(float)
                mask = ~np.isnan(X_np).any(axis=1)
                X_np = X_np[mask]
                y_np = LabelEncoder().fit_transform(np.array(y)[mask])
                if len(X_np) > 5000:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X_np), size=5000, replace=False)
                    X_np = X_np[idx]
                    y_np = y_np[idx]
                datasets.append((display_name, X_np, y_np, names, nc))
            except Exception as e:
                print(f"  {display_name} failed: {e}")

        # California Housing
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing()
        y_cal = (cal.target > np.median(cal.target)).astype(int)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(cal.data), size=5000, replace=False)
        datasets.append(("California Housing", cal.data[idx], y_cal[idx],
                         list(cal.feature_names), 2))

        # Adult Income
        try:
            X, y = fetch_openml('adult', version=2, return_X_y=True, as_frame=True, parser='auto')
            names = list(X.columns)
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    X[col] = OrdinalEncoder().fit_transform(X[[col]]).ravel()
            X_np = X.values.astype(float)
            mask = ~np.isnan(X_np).any(axis=1)
            X_np = X_np[mask]
            y_np = LabelEncoder().fit_transform(np.array(y)[mask])
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_np), size=min(5000, len(X_np)), replace=False)
            datasets.append(("Adult Income", X_np[idx], y_np[idx], names, 2))
        except Exception:
            pass

        return datasets

    def train_and_get_snrs(X, y, nc, seeds_cal, seeds_val):
        """Train models and compute per-pair SNR."""
        P = X.shape[1]

        def train(seeds):
            imps = np.zeros((len(seeds), P))
            for i, seed in enumerate(seeds):
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(X), size=len(X), replace=True)
                params = dict(n_estimators=100, max_depth=4, random_state=seed,
                              verbosity=0, tree_method='hist')
                if nc > 2:
                    params['objective'] = 'multi:softprob'
                    params['num_class'] = nc
                    params['eval_metric'] = 'mlogloss'
                else:
                    params['eval_metric'] = 'logloss'
                model = xgb.XGBClassifier(**params)
                model.fit(X[idx], y[idx])
                imps[i] = model.feature_importances_
            return imps

        imp_cal = train(seeds_cal)
        mean_imp = np.mean(imp_cal, axis=0)

        pairs = list(combinations(range(P), 2))
        snrs_by_pair = {}
        for j, k in pairs:
            diff = imp_cal[:, j] - imp_cal[:, k]
            mu = np.mean(diff)
            sd = np.std(diff, ddof=1)
            snr = abs(mu) / sd if sd > 1e-12 else 10.0
            snrs_by_pair[(j, k)] = snr

        return snrs_by_pair, mean_imp

    all_datasets = load_all_datasets()
    seeds_cal = list(range(42, 72))
    seeds_val = list(range(142, 172))

    for name, X, y, feat_names, nc in all_datasets:
        print(f"\n  {name} (P={X.shape[1]})...")
        snrs, mean_imp = train_and_get_snrs(X, y, nc, seeds_cal, seeds_val)

        P = X.shape[1]
        k_vals = [k for k in [3, 5, 10] if k <= P] + [P]
        topk = compute_topk_reliability(snrs, mean_imp, k_vals)

        all_topk[name] = {
            "n_features": P,
            "topk": {str(k): v for k, v in topk.items()},
        }

        for k, v in topk.items():
            label = f"K={k}" if k != P else "K=all"
            print(f"    {label:>8s}: {v['pct_reliable']:5.1f}% reliable, "
                  f"{v['pct_unreliable']:5.1f}% unreliable, gap={v['gap']:.0f}%")

            # Aggregate
            ak = k if k != P else 'all'
            if ak not in aggregates:
                aggregates[ak] = {"reliable": 0, "unreliable": 0, "total": 0}
            aggregates[ak]["reliable"] += v["n_reliable"]
            aggregates[ak]["unreliable"] += v["n_unreliable"]
            aggregates[ak]["total"] += v["n_pairs"]

    # Summary
    print(f"\n  AGGREGATE (15 datasets):")
    print(f"  {'K':>6s} {'Reliable':>10s} {'Unreliable':>12s} {'Gap':>8s}")
    for K in [3, 5, 10, 'all']:
        a = aggregates.get(K, aggregates.get(str(K), {"reliable": 0, "total": 1}))
        if a["total"] > 0:
            pct_r = a["reliable"] / a["total"] * 100
            pct_u = a["unreliable"] / a["total"] * 100
            print(f"  {str(K):>6s} {pct_r:>9.1f}% {pct_u:>11.1f}% {100-pct_r:>7.0f}%")

    # Fisher's exact: K=3 reliable vs K=all reliable
    a3 = aggregates.get(3, {"reliable": 0, "total": 0})
    a_all = aggregates.get('all', {"reliable": 0, "total": 0})
    if a3["total"] > 0 and a_all["total"] > 0:
        table = [
            [a3["reliable"], a3["total"] - a3["reliable"]],
            [a_all["reliable"], a_all["total"] - a_all["reliable"]]
        ]
        odds, p_fisher = fisher_exact(table, alternative='less')
        print(f"\n  Fisher's exact (K=3 vs K=all): p = {p_fisher:.4e}")
        print(f"  Inversion significant: {p_fisher < 0.05}")

    return all_topk, aggregates


def task2_null_model():
    """Null model: does the top-K inversion appear without Rashomon?"""
    print(f"\n{'='*60}")
    print("TASK 2: Null Model — Top-K Without Rashomon")
    print("=" * 60)

    # Generate synthetic importance vectors with NO Rashomon
    # Just sampling noise from a fixed importance distribution
    rng = np.random.RandomState(42)
    n_synthetic = 15
    n_models = 30

    null_results = {}
    agg_null = {K: {"reliable": 0, "unreliable": 0, "total": 0} for K in [3, 5, 10, 'all']}

    for ds_idx in range(n_synthetic):
        P = rng.choice([8, 13, 20, 30, 60]) # Vary feature count
        # True importance: features have decreasing importance
        true_imp = np.sort(rng.exponential(scale=1.0, size=P))[::-1]
        # Add noise: each model has importance = true + N(0, σ)
        sigma = 0.3 * np.mean(true_imp)  # 30% noise
        imp_matrix = np.array([true_imp + rng.normal(0, sigma, P) for _ in range(n_models)])
        imp_matrix = np.abs(imp_matrix)  # Importance is positive

        mean_imp = np.mean(imp_matrix, axis=0)
        pairs = list(combinations(range(P), 2))
        snrs = {}
        for j, k in pairs:
            diff = imp_matrix[:, j] - imp_matrix[:, k]
            mu = np.mean(diff)
            sd = np.std(diff, ddof=1)
            snr = abs(mu) / sd if sd > 1e-12 else 10.0
            snrs[(j, k)] = snr

        k_vals = [k for k in [3, 5, 10] if k <= P] + [P]
        topk = compute_topk_reliability(snrs, mean_imp, k_vals)

        name = f"Synthetic_{ds_idx+1}_P{P}"
        null_results[name] = {"P": P, "topk": {str(k): v for k, v in topk.items()}}

        for k, v in topk.items():
            ak = k if k != P else 'all'
            if ak not in agg_null:
                agg_null[ak] = {"reliable": 0, "unreliable": 0, "total": 0}
            agg_null[ak]["reliable"] += v["n_reliable"]
            agg_null[ak]["unreliable"] += v["n_unreliable"]
            agg_null[ak]["total"] += v["n_pairs"]

    print(f"\n  NULL MODEL AGGREGATE (15 synthetic, NO Rashomon):")
    print(f"  {'K':>6s} {'Reliable':>10s} {'Unreliable':>12s} {'Gap':>8s}")
    for K in [3, 5, 10, 'all']:
        a = agg_null.get(K, {"reliable": 0, "total": 1})
        if a["total"] > 0:
            pct_r = a["reliable"] / a["total"] * 100
            pct_u = a["unreliable"] / a["total"] * 100
            print(f"  {str(K):>6s} {pct_r:>9.1f}% {pct_u:>11.1f}% {100-pct_r:>7.0f}%")

    return null_results, agg_null


if __name__ == '__main__':
    print("Top-K Expansion + Null Model Comparison")
    print("=" * 60)

    topk_real, agg_real = task1_topk_15datasets()
    null_results, agg_null = task2_null_model()

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON: Real Data vs Null Model")
    print("=" * 60)
    print(f"  {'K':>6s} {'Real Gap':>10s} {'Null Gap':>10s} {'Difference':>12s}")
    for K in [3, 5, 10, 'all']:
        ar = agg_real.get(K, {"reliable": 0, "total": 1})
        an = agg_null.get(K, {"reliable": 0, "total": 1})
        gap_r = 100 - ar["reliable"] / ar["total"] * 100 if ar["total"] > 0 else 100
        gap_n = 100 - an["reliable"] / an["total"] * 100 if an["total"] > 0 else 100
        diff = gap_r - gap_n
        print(f"  {str(K):>6s} {gap_r:>9.0f}% {gap_n:>9.0f}% {diff:>+11.0f}pp")

    # Save
    output = {
        "task1_topk_15datasets": {"per_dataset": topk_real, "aggregate": {str(k): v for k, v in agg_real.items()}},
        "task2_null_model": {"per_dataset": null_results, "aggregate": {str(k): v for k, v in agg_null.items()}},
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json.dump(output, open(OUT_DIR / 'results_topk_expanded_and_null.json', 'w'), indent=2, cls=NumpyEncoder)
    print(f"\nSaved to results_topk_expanded_and_null.json")
