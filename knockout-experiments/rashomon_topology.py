#!/usr/bin/env python3
"""
Knockout Experiment: Rashomon Topology Predicts Bimodality

Tests whether the bimodal flip distribution corresponds to the Rashomon
set having two connected components in SHAP space.

If models cluster into two groups in SHAP space, and within-cluster
pairs have low flip rates while between-cluster pairs have high flip
rates, then bimodality IS a topological property of the loss landscape.

Design:
- Use existing 50-model SHAP data from approximate_symmetry_v2
- For each ρ level: compute pairwise SHAP distance between all models
- Apply hierarchical clustering / DBSCAN
- Check: do clusters correspond to the bimodal modes?
- Metric: correlation between "same cluster" indicator and flip rate

Also: generate fresh data for cleaner analysis at ρ = 0.9 (where
bimodality is strongest and groups are detected).
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
import xgboost as xgb
import shap


N_MODELS = 50
N_CLUSTERS_TEST = [2, 3, 4, 5]  # test multiple cluster counts


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


def run_topology_analysis(X, y, dataset_name, task='classification'):
    """Train models, compute SHAP, test cluster structure."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
        stratify=y if task == 'classification' else None
    )
    P = X.shape[1]

    # Train models and compute global SHAP importance
    shap_vectors = []  # (N_MODELS, P) — global importance per model
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
        # Global importance = mean |SHAP| per feature
        global_imp = np.mean(np.abs(sv), axis=0)
        shap_vectors.append(global_imp)

    shap_vectors = np.array(shap_vectors)  # (N_MODELS, P)
    print(f"  {N_MODELS} models, {P} features")

    # Pairwise distance between models in SHAP space
    dists = pdist(shap_vectors, metric='cosine')
    dist_matrix = squareform(dists)

    # Compute pairwise flip rates (for a subset of feature pairs)
    # Use SIGN of mean SHAP (not magnitude) for flip computation
    shap_signs = []
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
        sv = explainer.shap_values(X_test[:100])
        if isinstance(sv, list):
            sv = sv[1]
        # Mean SHAP per feature (signed)
        shap_signs.append(np.mean(sv, axis=0))

    shap_signs = np.array(shap_signs)

    # Per-model-pair flip rate (fraction of features where ranking disagrees)
    pairwise_flips = np.zeros((N_MODELS, N_MODELS))
    for m1 in range(N_MODELS):
        for m2 in range(m1+1, N_MODELS):
            n_flip = 0
            n_total = 0
            for fi in range(P):
                for fj in range(fi+1, P):
                    d1 = shap_signs[m1, fi] - shap_signs[m1, fj]
                    d2 = shap_signs[m2, fi] - shap_signs[m2, fj]
                    if abs(d1) > 1e-10 and abs(d2) > 1e-10:
                        if np.sign(d1) != np.sign(d2):
                            n_flip += 1
                        n_total += 1
            fr = n_flip / n_total if n_total > 0 else 0
            pairwise_flips[m1, m2] = fr
            pairwise_flips[m2, m1] = fr

    # Test cluster structure
    best_k = 1
    best_silhouette = -1
    best_correlation = 0
    results_by_k = {}

    for k in N_CLUSTERS_TEST:
        # K-means clustering in SHAP space
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(shap_vectors)

        if k > 1:
            sil = silhouette_score(shap_vectors, labels, metric='cosine')
        else:
            sil = 0

        # Key test: does same-cluster predict low flip rate?
        same_cluster_flips = []
        diff_cluster_flips = []
        for m1 in range(N_MODELS):
            for m2 in range(m1+1, N_MODELS):
                if labels[m1] == labels[m2]:
                    same_cluster_flips.append(pairwise_flips[m1, m2])
                else:
                    diff_cluster_flips.append(pairwise_flips[m1, m2])

        mean_same = float(np.mean(same_cluster_flips)) if same_cluster_flips else 0
        mean_diff = float(np.mean(diff_cluster_flips)) if diff_cluster_flips else 0
        gap = mean_diff - mean_same

        # Correlation between cluster indicator and flip rate
        cluster_indicators = []
        flip_values = []
        for m1 in range(N_MODELS):
            for m2 in range(m1+1, N_MODELS):
                cluster_indicators.append(0 if labels[m1] == labels[m2] else 1)
                flip_values.append(pairwise_flips[m1, m2])

        if len(set(cluster_indicators)) > 1:
            corr, p_val = spearmanr(cluster_indicators, flip_values)
        else:
            corr, p_val = 0, 1

        results_by_k[k] = {
            'silhouette': float(sil),
            'mean_same_cluster_flip': mean_same,
            'mean_diff_cluster_flip': mean_diff,
            'gap': gap,
            'correlation': float(corr),
            'p_value': float(p_val),
            'n_same_pairs': len(same_cluster_flips),
            'n_diff_pairs': len(diff_cluster_flips),
            'cluster_sizes': [int(np.sum(labels == c)) for c in range(k)],
        }

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_correlation = corr

        print(f"  k={k}: silhouette={sil:.3f}, same-cluster flip={mean_same:.3f}, "
              f"diff-cluster flip={mean_diff:.3f}, gap={gap:.3f}, "
              f"ρ(cluster,flip)={corr:.3f} (p={p_val:.2e})")

    # Hierarchical clustering dendrogram analysis
    Z = linkage(shap_vectors, method='ward', metric='euclidean')
    # Find natural number of clusters using gap statistic approximation
    # (look for largest gap in the merge distances)
    merge_dists = Z[:, 2]
    gaps = np.diff(merge_dists)
    if len(gaps) > 0:
        natural_k = len(merge_dists) - np.argmax(gaps[-min(10, len(gaps)):]) + 1
        natural_k = min(natural_k, 10)
    else:
        natural_k = 1

    print(f"\n  Best k by silhouette: {best_k} (silhouette={best_silhouette:.3f})")
    print(f"  Natural k from dendrogram: {natural_k}")
    print(f"  Best correlation(cluster, flip): {best_correlation:.3f}")

    # THE KEY TEST: at the best k, is the gap between same-cluster
    # and diff-cluster flip rates statistically significant?
    best_result = results_by_k[best_k]
    print(f"\n  AT BEST k={best_k}:")
    print(f"    Same-cluster mean flip: {best_result['mean_same_cluster_flip']:.3f}")
    print(f"    Diff-cluster mean flip: {best_result['mean_diff_cluster_flip']:.3f}")
    print(f"    Gap: {best_result['gap']:.3f}")
    print(f"    ρ(cluster, flip): {best_result['correlation']:.3f} (p={best_result['p_value']:.2e})")

    topology_predicts_bimodality = (
        best_result['correlation'] > 0.3 and
        best_result['p_value'] < 0.01 and
        best_result['gap'] > 0.02
    )
    print(f"    TOPOLOGY PREDICTS BIMODALITY: {topology_predicts_bimodality}")

    return {
        'dataset': dataset_name,
        'n_models': N_MODELS,
        'n_features': P,
        'best_k': best_k,
        'best_silhouette': float(best_silhouette),
        'natural_k': int(natural_k),
        'results_by_k': results_by_k,
        'topology_predicts_bimodality': topology_predicts_bimodality,
        'mean_pairwise_flip': float(np.mean(pairwise_flips[np.triu_indices(N_MODELS, k=1)])),
    }


def main():
    start = time.time()
    print("=" * 60)
    print("RASHOMON TOPOLOGY: Do Clusters Predict Bimodality?")
    print("=" * 60)

    results = {}

    # Synthetic at multiple ρ
    for rho in [0.5, 0.7, 0.9]:
        X, y = generate_synthetic(2000, 12, 3, 4, rho)
        results[f'synthetic_{rho}'] = run_topology_analysis(
            X, y, f'Synthetic ρ={rho}')

    # Real data
    bc = load_breast_cancer()
    results['breast_cancer'] = run_topology_analysis(
        bc.data, bc.target, 'Breast Cancer')

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Dataset':25s} {'Best k':>6s} {'Sil':>6s} {'Gap':>6s} {'ρ(clust,flip)':>13s} {'Predicts?':>10s}")
    print("-" * 70)
    for name, r in results.items():
        bk = r['results_by_k'][r['best_k']]
        print(f"{name:25s} {r['best_k']:>6d} {r['best_silhouette']:>6.3f} "
              f"{bk['gap']:>6.3f} {bk['correlation']:>13.3f} "
              f"{'YES' if r['topology_predicts_bimodality'] else 'no':>10s}")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'rashomon_topology',
        'question': 'Do SHAP-space clusters predict flip rate bimodality?',
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_rashomon_topology.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
