#!/usr/bin/env python3
"""
Clinical Decision Reversal v2: Rigorous Per-Individual Analysis

Fixes all issues from adversarial review:
1. Ablation: 4 conditions (seed-only / +subsample / +colsample / +bootstrap+both)
2. SHAP-gap analysis: filter reversals by magnitude threshold
3. Feature clustering: group |r|>0.8 features, report cluster-level reversal
4. Data integrity: explicit logging of download success/failure
5. Multiple model classes: XGBoost + LightGBM + Random Forest
6. Honest headline: report which variation source drives the effect
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from itertools import combinations

N_MODELS = 30
RASHOMON_THRESHOLD = 0.02
GAP_THRESHOLDS = [0.0, 0.01, 0.05, 0.10]  # |SHAP_1| - |SHAP_2| thresholds
CLUSTER_CORR = 0.8  # correlation threshold for feature clustering


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_datasets():
    """Load datasets with explicit integrity logging."""
    datasets = []
    integrity = {}

    # 1. Breast Cancer (sklearn — always available)
    bc = load_breast_cancer()
    datasets.append((bc.data, bc.target, list(bc.feature_names), 'Breast Cancer'))
    integrity['Breast Cancer'] = 'sklearn.datasets.load_breast_cancer (guaranteed)'

    # 2. German Credit
    try:
        data = np.loadtxt(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
            dtype=float
        )
        X_gc = data[:, :-1]
        y_gc = (data[:, -1] == 1).astype(int)
        feature_names_gc = [
            'Account_status', 'Duration_months', 'Credit_history', 'Purpose',
            'Credit_amount', 'Savings', 'Employment_years', 'Installment_rate',
            'Personal_status', 'Other_debtors', 'Residence_years', 'Property',
            'Age', 'Other_plans', 'Housing', 'Existing_credits', 'Job',
            'Dependents', 'Telephone', 'Foreign_worker'
        ] + [f'Feature_{i}' for i in range(20, X_gc.shape[1])]
        datasets.append((X_gc, y_gc, feature_names_gc[:X_gc.shape[1]], 'German Credit'))
        integrity['German Credit'] = f'UCI ML Repository (downloaded, {X_gc.shape[0]} rows, {X_gc.shape[1]} cols)'
    except Exception as e:
        integrity['German Credit'] = f'DOWNLOAD FAILED: {e}. EXCLUDED from analysis.'
        print(f"  WARNING: German Credit download failed: {e}")

    # 3. Heart Disease (OpenML)
    try:
        from sklearn.datasets import fetch_openml
        hd = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
        X_hd = hd.data.astype(float)
        y_hd = (hd.target == '2').astype(int) if hd.target.dtype == object else hd.target.astype(int)
        fn_hd = ['Age', 'Sex', 'Chest_pain', 'Rest_BP', 'Cholesterol',
                 'Fasting_BS', 'Rest_ECG', 'Max_HR', 'Exercise_angina',
                 'ST_depression', 'ST_slope', 'Major_vessels', 'Thal']
        datasets.append((X_hd, y_hd, fn_hd[:X_hd.shape[1]], 'Heart Disease'))
        integrity['Heart Disease'] = f'OpenML heart-statlog (downloaded, {X_hd.shape[0]} rows, {X_hd.shape[1]} cols)'
    except Exception as e:
        integrity['Heart Disease'] = f'DOWNLOAD FAILED: {e}. EXCLUDED from analysis.'
        print(f"  WARNING: Heart Disease download failed: {e}")

    return datasets, integrity


def compute_feature_clusters(X, threshold=CLUSTER_CORR):
    """Cluster features by |correlation| > threshold."""
    corr = np.abs(np.corrcoef(X.T))
    n = corr.shape[0]
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = {i}
        queue = [i]
        while queue:
            curr = queue.pop()
            for j in range(n):
                if j not in visited and j not in cluster and corr[curr, j] > threshold:
                    cluster.add(j)
                    queue.append(j)
        visited.update(cluster)
        clusters.append(sorted(cluster))

    return clusters


def run_condition(X_train, y_train, X_test, y_test, feature_names, condition, model_class='xgboost'):
    """Run one experimental condition and return per-individual results."""
    import shap

    models = []
    accuracies = []

    for i in range(N_MODELS):
        seed = 42 + i

        # Apply condition-specific variation
        X_tr, y_tr = X_train.copy(), y_train.copy()

        if 'bootstrap' in condition:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_tr), len(X_tr), replace=True)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        if model_class == 'xgboost':
            import xgboost as xgb
            params = dict(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=seed, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
            if 'subsample' in condition:
                params['subsample'] = 0.8
            if 'colsample' in condition:
                params['colsample_bytree'] = 0.8
            model = xgb.XGBClassifier(**params)
        elif model_class == 'lightgbm':
            import lightgbm as lgb
            params = dict(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=seed, verbose=-1
            )
            if 'subsample' in condition:
                params['subsample'] = 0.8
                params['subsample_freq'] = 1
            if 'colsample' in condition:
                params['colsample_bytree'] = 0.8
            model = lgb.LGBMClassifier(**params)
        elif model_class == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=seed
            )

        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        models.append(model)

    # Rashomon filter
    best = max(accuracies)
    rash_idx = [i for i, a in enumerate(accuracies) if a >= best - RASHOMON_THRESHOLD]
    rash_models = [models[i] for i in rash_idx]

    if len(rash_models) < 3:
        return None  # Not enough models

    # SHAP
    all_shap = []
    for model in rash_models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]  # binary class 1
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
        all_shap.append(sv)
    all_shap = np.array(all_shap)  # (n_models, n_test, n_features)

    # Per-individual analysis
    n_test, n_features = X_test.shape[0], X_test.shape[1]
    n_models = len(rash_models)

    # Feature clusters
    clusters = compute_feature_clusters(X_train)
    feature_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for fi in cluster:
            feature_to_cluster[fi] = ci

    results = {
        'n_models': N_MODELS,
        'n_rashomon': len(rash_idx),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'accuracy_range': float(max([accuracies[i] for i in rash_idx]) -
                                min([accuracies[i] for i in rash_idx])),
        'n_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
    }

    # Per-threshold reversal rates
    for gap_t in GAP_THRESHOLDS:
        feature_reversals = 0
        cluster_reversals = 0

        for ind in range(n_test):
            # Top-1 feature per model
            top1_per_model = []
            gaps = []
            for m in range(n_models):
                abs_shap = np.abs(all_shap[m, ind, :])
                sorted_idx = np.argsort(abs_shap)[::-1]
                top1_per_model.append(int(sorted_idx[0]))
                gaps.append(abs_shap[sorted_idx[0]] - abs_shap[sorted_idx[1]])

            mean_gap = np.mean(gaps)

            # Feature-level reversal (only count if gap exceeds threshold)
            if mean_gap >= gap_t:
                unique_top1 = len(set(top1_per_model))
                if unique_top1 > 1:
                    feature_reversals += 1

                # Cluster-level reversal
                top1_clusters = [feature_to_cluster.get(f, f) for f in top1_per_model]
                unique_clusters = len(set(top1_clusters))
                if unique_clusters > 1:
                    cluster_reversals += 1

        results[f'feature_reversal_gap{gap_t}'] = feature_reversals
        results[f'cluster_reversal_gap{gap_t}'] = cluster_reversals
        results[f'feature_reversal_rate_gap{gap_t}'] = feature_reversals / n_test
        results[f'cluster_reversal_rate_gap{gap_t}'] = cluster_reversals / n_test

    # SHAP gap distribution
    all_gaps = []
    for ind in range(n_test):
        for m in range(n_models):
            abs_shap = np.abs(all_shap[m, ind, :])
            sorted_vals = np.sort(abs_shap)[::-1]
            all_gaps.append(sorted_vals[0] - sorted_vals[1])
    results['shap_gap_mean'] = float(np.mean(all_gaps))
    results['shap_gap_median'] = float(np.median(all_gaps))
    results['shap_gap_p25'] = float(np.percentile(all_gaps, 25))
    results['shap_gap_p75'] = float(np.percentile(all_gaps, 75))

    return results


def main():
    start = time.time()
    print("=" * 70)
    print("CLINICAL DECISION REVERSAL v2: Rigorous Ablation Study")
    print("=" * 70)

    datasets, integrity = load_datasets()
    print(f"\nData integrity:")
    for name, status in integrity.items():
        print(f"  {name}: {status}")

    conditions = [
        ('seed_only', 'Seed only (default XGBoost)'),
        ('seed+subsample', 'Seed + row subsample (0.8)'),
        ('seed+colsample', 'Seed + col subsample (0.8)'),
        ('seed+bootstrap+subsample+colsample', 'Seed + bootstrap + both subsamples'),
    ]

    model_classes = ['xgboost']
    try:
        import lightgbm
        model_classes.append('lightgbm')
    except ImportError:
        print("\n  LightGBM not available — skipping")

    model_classes.append('random_forest')

    all_results = {}

    for X, y, feature_names, dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"{'='*70}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )

        dataset_results = {}

        for model_class in model_classes:
            for cond_key, cond_name in conditions:
                label = f"{model_class}/{cond_key}"
                print(f"\n  {label}...")

                result = run_condition(
                    X_train, y_train, X_test, y_test, feature_names,
                    cond_key, model_class
                )

                if result is None:
                    print(f"    Skipped (too few Rashomon models)")
                    continue

                dataset_results[label] = result

                # Print key metrics
                fr0 = result['feature_reversal_rate_gap0.0']
                cr0 = result['cluster_reversal_rate_gap0.0']
                fr5 = result['feature_reversal_rate_gap0.05']
                cr5 = result['cluster_reversal_rate_gap0.05']
                print(f"    Rashomon: {result['n_rashomon']}/{result['n_models']}")
                print(f"    Acc: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
                print(f"    Feature reversal (gap≥0):   {fr0:.1%}")
                print(f"    Cluster reversal (gap≥0):   {cr0:.1%}")
                print(f"    Feature reversal (gap≥0.05): {fr5:.1%}")
                print(f"    Cluster reversal (gap≥0.05): {cr5:.1%}")
                print(f"    SHAP gap: median={result['shap_gap_median']:.4f}, "
                      f"mean={result['shap_gap_mean']:.4f}")

        all_results[dataset_name] = dataset_results

    # Summary
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':25s} {'Condition':40s} {'Feat Rev':>10s} {'Clust Rev':>10s} {'Gap≥0.05':>10s}")
    print("-" * 100)
    for dataset_name, dataset_results in all_results.items():
        for label, result in dataset_results.items():
            fr = result['feature_reversal_rate_gap0.0']
            cr = result['cluster_reversal_rate_gap0.0']
            fr5 = result['feature_reversal_rate_gap0.05']
            print(f"{dataset_name:25s} {label:40s} {fr:>9.1%} {cr:>9.1%} {fr5:>9.1%}")

    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.0f}s")

    # Save
    output = {
        'experiment': 'clinical_decision_reversal_v2',
        'description': 'Rigorous ablation: 4 conditions × 3 model classes × 3 datasets',
        'integrity': integrity,
        'conditions': {k: v for k, v in conditions},
        'model_classes': model_classes,
        'gap_thresholds': GAP_THRESHOLDS,
        'cluster_correlation_threshold': CLUSTER_CORR,
        'results': all_results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_clinical_decision_reversal_v2.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
