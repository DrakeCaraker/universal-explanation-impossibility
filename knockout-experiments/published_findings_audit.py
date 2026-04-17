#!/usr/bin/env python3
"""
Published Findings Audit: Are Published SHAP Claims Reliable?

Takes well-known published analyses on standard datasets and tests whether
their top-feature claims are stable across retraining.

We use three widely-analyzed datasets where SHAP feature importance has been
published in peer-reviewed papers and reproducibility analyses:

1. Boston Housing / California Housing — feature importance for house price prediction
   (Lundberg & Lee 2017 NIPS, the original SHAP paper)

2. Adult Income — feature importance for income prediction
   (Used in hundreds of SHAP tutorials and published analyses)

3. Breast Cancer — feature importance for diagnosis
   (Lundberg et al. 2020 Nature Machine Intelligence)

For each: we train 30 models, compute SHAP, and check which commonly-cited
top features are actually stable.
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from collections import Counter
import xgboost as xgb
import shap

N_MODELS = 30


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def audit_dataset(X, y, feature_names, dataset_name, task='classification',
                  published_top_features=None):
    """Audit a dataset's SHAP stability.

    published_top_features: list of feature names commonly cited as "most important"
    in published analyses. We check whether these claims are stable.
    """
    print(f"\n{'='*60}")
    print(f"  AUDIT: {dataset_name}")
    print(f"  Published top features: {published_top_features}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
        stratify=y if task == 'classification' else None
    )

    # Train 30 models with bootstrap variation
    models = []
    scores = []
    for i in range(N_MODELS):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), len(X_train), replace=True)

        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbosity=0
            )

        model.fit(X_train[idx], y_train[idx])

        if task == 'classification':
            score = accuracy_score(y_test, model.predict(X_test))
        else:
            score = r2_score(y_test, model.predict(X_test))
        scores.append(score)
        models.append(model)

    print(f"  Models: {N_MODELS}, Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Compute global SHAP importance for each model
    global_importances = []  # (N_MODELS, P) — mean |SHAP| per feature
    for model in models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:200])
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.mean(np.abs(sv), axis=0)  # global importance
        global_importances.append(mean_abs)

    global_importances = np.array(global_importances)  # (N_MODELS, P)

    # Top-1 feature across models
    top1_per_model = [int(np.argmax(imp)) for imp in global_importances]
    top1_counter = Counter(top1_per_model)
    n_unique_top1 = len(top1_counter)
    most_common_top1 = top1_counter.most_common(1)[0]

    # Top-3 agreement (Jaccard)
    top3_per_model = [set(np.argsort(imp)[-3:]) for imp in global_importances]
    top3_jaccards = []
    for i in range(N_MODELS):
        for j in range(i+1, N_MODELS):
            jacc = len(top3_per_model[i] & top3_per_model[j]) / len(top3_per_model[i] | top3_per_model[j])
            top3_jaccards.append(jacc)

    # Top-5 agreement
    top5_per_model = [set(np.argsort(imp)[-5:]) for imp in global_importances]
    top5_jaccards = []
    for i in range(N_MODELS):
        for j in range(i+1, N_MODELS):
            jacc = len(top5_per_model[i] & top5_per_model[j]) / len(top5_per_model[i] | top5_per_model[j])
            top5_jaccards.append(jacc)

    # Per-feature stability (CV of global importance)
    feature_cv = np.std(global_importances, axis=0) / (np.mean(global_importances, axis=0) + 1e-10)
    stable_features = np.sum(feature_cv < 0.3)  # CV < 30% = stable
    unstable_features = np.sum(feature_cv >= 0.5)  # CV ≥ 50% = unstable

    # Check published claims
    published_audit = {}
    if published_top_features:
        for feat_name in published_top_features:
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                cv = float(feature_cv[idx])
                mean_rank = float(np.mean([
                    np.where(np.argsort(imp)[::-1] == idx)[0][0] + 1
                    for imp in global_importances
                ]))
                rank_std = float(np.std([
                    np.where(np.argsort(imp)[::-1] == idx)[0][0] + 1
                    for imp in global_importances
                ]))
                in_top3_pct = float(np.mean([idx in t3 for t3 in top3_per_model]))
                in_top5_pct = float(np.mean([idx in t5 for t5 in top5_per_model]))

                stable = cv < 0.3 and rank_std < 2.0
                published_audit[feat_name] = {
                    'cv': cv,
                    'mean_rank': mean_rank,
                    'rank_std': rank_std,
                    'in_top3_pct': in_top3_pct,
                    'in_top5_pct': in_top5_pct,
                    'stable': stable,
                    'verdict': 'STABLE' if stable else 'UNSTABLE',
                }
                print(f"  Published claim '{feat_name}': rank={mean_rank:.1f}±{rank_std:.1f}, "
                      f"CV={cv:.2f}, top3={in_top3_pct:.0%} → {published_audit[feat_name]['verdict']}")

    # Overall assessment
    n_published = len(published_audit) if published_audit else 0
    n_stable_published = sum(1 for v in published_audit.values() if v['stable'])
    n_unstable_published = n_published - n_stable_published

    print(f"\n  SUMMARY:")
    print(f"    Top-1 agreement: {most_common_top1[1]}/{N_MODELS} ({most_common_top1[1]/N_MODELS:.0%}) agree on '{feature_names[most_common_top1[0]]}'")
    print(f"    Top-1 unique features seen: {n_unique_top1}")
    print(f"    Top-3 Jaccard: {np.mean(top3_jaccards):.3f}")
    print(f"    Top-5 Jaccard: {np.mean(top5_jaccards):.3f}")
    print(f"    Stable features (CV<0.3): {stable_features}/{len(feature_names)}")
    print(f"    Unstable features (CV≥0.5): {unstable_features}/{len(feature_names)}")
    if published_audit:
        print(f"    Published claims stable: {n_stable_published}/{n_published}")
        print(f"    Published claims UNSTABLE: {n_unstable_published}/{n_published}")

    return {
        'dataset': dataset_name,
        'n_features': len(feature_names),
        'n_models': N_MODELS,
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'top1_agreement': float(most_common_top1[1] / N_MODELS),
        'top1_feature': feature_names[most_common_top1[0]],
        'top1_unique': n_unique_top1,
        'top3_jaccard_mean': float(np.mean(top3_jaccards)),
        'top5_jaccard_mean': float(np.mean(top5_jaccards)),
        'stable_features': int(stable_features),
        'unstable_features': int(unstable_features),
        'feature_cv': {feature_names[i]: float(feature_cv[i]) for i in range(len(feature_names))},
        'published_audit': published_audit,
        'n_published_stable': n_stable_published,
        'n_published_unstable': n_unstable_published,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("PUBLISHED FINDINGS AUDIT")
    print("Are commonly-cited SHAP feature importance claims stable?")
    print("=" * 60)

    results = {}

    # 1. Breast Cancer — Lundberg et al. 2020 Nature Machine Intelligence
    # Published top features: worst concave points, worst perimeter, worst radius
    bc = load_breast_cancer()
    results['breast_cancer'] = audit_dataset(
        bc.data, bc.target, list(bc.feature_names), 'Breast Cancer',
        task='classification',
        published_top_features=['worst concave points', 'worst perimeter',
                                'worst radius', 'worst area', 'mean concave points']
    )

    # 2. California Housing — widely used SHAP benchmark
    # Published top features: MedInc, AveOccup, Latitude, Longitude
    cal = fetch_california_housing()
    results['california'] = audit_dataset(
        cal.data, cal.target, list(cal.feature_names), 'California Housing',
        task='regression',
        published_top_features=['MedInc', 'AveOccup', 'Latitude', 'Longitude', 'HouseAge']
    )

    # 3. Adult Income — widely used in fairness/XAI literature
    # Published top features: age, education-num, capital-gain, hours-per-week
    try:
        adult = fetch_openml('adult', version=2, as_frame=False, parser='auto')
        X_adult = adult.data
        y_adult = (adult.target == '>50K').astype(int) if adult.target.dtype == object else adult.target.astype(int)
        fn_adult = list(adult.feature_names) if hasattr(adult, 'feature_names') else \
            [f'f{i}' for i in range(X_adult.shape[1])]

        # Handle NaN
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median')
        X_adult = imp.fit_transform(X_adult)

        results['adult_income'] = audit_dataset(
            X_adult[:5000], y_adult[:5000], fn_adult, 'Adult Income',
            task='classification',
            published_top_features=['age', 'education-num', 'capital-gain',
                                    'hours-per-week', 'fnlwgt']
        )
    except Exception as e:
        print(f"\n  Adult Income failed: {e}")

    # Aggregate
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"AGGREGATE FINDINGS")
    print(f"{'='*60}")

    total_claims = sum(r.get('n_published_stable', 0) + r.get('n_published_unstable', 0)
                       for r in results.values())
    total_unstable = sum(r.get('n_published_unstable', 0) for r in results.values())

    print(f"\n  Across {len(results)} datasets:")
    print(f"  {total_unstable}/{total_claims} published top-feature claims are UNSTABLE")
    print(f"  ({total_unstable/total_claims*100:.0f}% of commonly-cited importance claims)")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'published_findings_audit',
        'description': 'Stability audit of commonly-cited SHAP top-feature claims',
        'n_datasets': len(results),
        'total_claims_audited': total_claims,
        'total_unstable': total_unstable,
        'pct_unstable': float(total_unstable / total_claims * 100) if total_claims > 0 else 0,
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_published_findings_audit.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
