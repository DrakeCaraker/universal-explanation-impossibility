#!/usr/bin/env python3
"""
Random Merging Control: Is enrichment reduction due to semantic structure or class count?

CRITICAL CONTROL: The abstraction enrichment experiment showed 4/6 datasets with
reduced flip rates when merging classes. But does RANDOM merging work equally well?

For each confirmed dataset, compare:
  A) Semantic merge (the one that worked)
  B) Random merge to same number of classes (10 random partitions, averaged)

If random merging works equally well, the "neutral element" story is just
"fewer classes = less instability" and has no theoretical content.

If semantic merging works BETTER, there's genuine structure.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.datasets import load_wine, load_iris, fetch_covtype, fetch_openml
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEEDS = list(range(42, 72))  # 30 models


def train_models(X, y, seeds, n_classes):
    n = len(seeds)
    P = X.shape[1]
    imp = np.zeros((n, P))
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        params = dict(n_estimators=100, max_depth=4, random_state=seed,
                      verbosity=0, tree_method='hist')
        if n_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
        else:
            params['eval_metric'] = 'logloss'
        model = xgb.XGBClassifier(**params)
        model.fit(X[idx], y[idx])
        imp[i] = model.feature_importances_
    return imp


def compute_flip_rate(imp, max_pairs=500):
    n_models, P = imp.shape
    all_pairs = list(combinations(range(P), 2))
    if len(all_pairs) > max_pairs:
        rng = np.random.RandomState(999)
        pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=max_pairs, replace=False)]
    else:
        pairs = all_pairs

    flip_rates = []
    for j, k in pairs:
        disagree = 0
        total = 0
        for m1 in range(n_models):
            for m2 in range(m1 + 1, n_models):
                if (imp[m1, j] - imp[m1, k]) * (imp[m2, j] - imp[m2, k]) < 0:
                    disagree += 1
                total += 1
        flip_rates.append(disagree / total if total > 0 else 0.0)
    return float(np.mean(flip_rates))


def random_merge(y_orig, n_target_classes, seed):
    """Randomly assign original classes to n_target_classes groups."""
    rng = np.random.RandomState(seed)
    orig_classes = np.unique(y_orig)
    # Random assignment of original classes to target classes
    assignment = rng.randint(0, n_target_classes, size=len(orig_classes))
    # Ensure at least one class per target (retry if degenerate)
    for _ in range(100):
        if len(np.unique(assignment)) == n_target_classes:
            break
        assignment = rng.randint(0, n_target_classes, size=len(orig_classes))
    mapping = dict(zip(orig_classes, assignment))
    return np.array([mapping[c] for c in y_orig])


def run_control(name, X, y_orig, y_semantic, n_orig_classes, n_target_classes, n_random=10):
    """Compare semantic merge vs random merges."""
    print(f"\n{'='*60}")
    print(f"CONTROL: {name}")
    print(f"  {n_orig_classes} classes → {n_target_classes} classes")
    print(f"{'='*60}")
    t0 = time.time()

    # Fine-grained baseline
    print(f"  Training fine-grained ({n_orig_classes} classes)...")
    imp_fine = train_models(X, y_orig, SEEDS, n_orig_classes)
    fine_flip = compute_flip_rate(imp_fine)
    print(f"    Flip rate: {fine_flip:.4f}")

    # Semantic merge
    print(f"  Training semantic merge ({n_target_classes} classes)...")
    imp_semantic = train_models(X, y_semantic, SEEDS, n_target_classes)
    semantic_flip = compute_flip_rate(imp_semantic)
    print(f"    Flip rate: {semantic_flip:.4f}")

    # Random merges
    random_flips = []
    for r_seed in range(n_random):
        y_random = random_merge(y_orig, n_target_classes, seed=1000 + r_seed)
        n_actual = len(np.unique(y_random))
        if n_actual < 2:
            continue
        print(f"  Training random merge {r_seed+1}/{n_random} ({n_actual} classes)...")
        imp_random = train_models(X, y_random, SEEDS, n_actual)
        rf = compute_flip_rate(imp_random)
        random_flips.append(rf)
        print(f"    Flip rate: {rf:.4f}")

    mean_random = float(np.mean(random_flips)) if random_flips else 0.0
    std_random = float(np.std(random_flips)) if random_flips else 0.0

    semantic_reduction = fine_flip - semantic_flip
    random_reduction = fine_flip - mean_random

    # Is semantic merge better than random?
    semantic_better = semantic_flip < mean_random
    # By how much?
    advantage = mean_random - semantic_flip

    elapsed = time.time() - t0

    result = {
        "dataset": name,
        "n_orig_classes": n_orig_classes,
        "n_target_classes": n_target_classes,
        "fine_flip": fine_flip,
        "semantic_flip": semantic_flip,
        "random_flips": random_flips,
        "mean_random_flip": mean_random,
        "std_random_flip": std_random,
        "semantic_reduction_pp": round(semantic_reduction * 100, 1),
        "random_reduction_pp": round(random_reduction * 100, 1),
        "semantic_better_than_random": semantic_better,
        "semantic_advantage_pp": round(advantage * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  RESULT:")
    print(f"    Fine:     {fine_flip:.4f}")
    print(f"    Semantic: {semantic_flip:.4f} (reduction: {semantic_reduction*100:+.1f}pp)")
    print(f"    Random:   {mean_random:.4f} ± {std_random:.4f} (reduction: {random_reduction*100:+.1f}pp)")
    print(f"    Semantic advantage over random: {advantage*100:+.1f}pp")
    print(f"    Semantic better: {semantic_better}")

    return result


if __name__ == '__main__':
    print("Random Merging Control")
    print("=" * 60)
    print("QUESTION: Is enrichment due to semantic structure or class count?\n")

    all_results = {}

    # 1. Wine: 3 → 2 (semantic: cultivar 1+2 vs 3)
    wine = load_wine()
    y_wine_semantic = np.where(wine.target == 2, 1, 0)
    r = run_control("Wine", wine.data, wine.target, y_wine_semantic, 3, 2)
    all_results["Wine"] = r

    # 2. Iris: 3 → 2 (semantic: setosa vs non-setosa)
    iris = load_iris()
    y_iris_semantic = np.where(iris.target == 0, 0, 1)
    r = run_control("Iris", iris.data, iris.target, y_iris_semantic, 3, 2)
    all_results["Iris"] = r

    # 3. Covertype: 7 → 2 (semantic: conifer vs non-conifer)
    cov = fetch_covtype()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(cov.data), size=5000, replace=False)
    X_cov = cov.data[idx]
    y_cov = cov.target[idx] - 1
    superclass_map = {0: 0, 1: 0, 2: 1, 3: 1, 5: 1, 4: 1, 6: 1}
    y_cov_semantic = np.array([superclass_map.get(c, 1) for c in y_cov])
    r = run_control("Covertype", X_cov, y_cov, y_cov_semantic, 7, 2, n_random=5)
    all_results["Covertype"] = r

    # 4. Fashion-MNIST: 10 → 2 (semantic: clothing vs non-clothing)
    X_fm, y_fm_str = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
    y_fm_full = y_fm_str.astype(int)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_fm), size=5000, replace=False)
    X_fm = X_fm[idx].astype(np.float64)
    y_fm = y_fm_full[idx]
    superclass_map_fm = {0: 0, 2: 0, 4: 0, 6: 0, 1: 0, 5: 1, 7: 1, 9: 1, 3: 1, 8: 1}
    y_fm_semantic = np.array([superclass_map_fm[c] for c in y_fm])
    r = run_control("Fashion-MNIST", X_fm, y_fm, y_fm_semantic, 10, 2, n_random=5)
    all_results["Fashion-MNIST"] = r

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Semantic vs Random Merging")
    print("=" * 60)

    for name, r in all_results.items():
        print(f"\n  {name}:")
        print(f"    Semantic reduction: {r['semantic_reduction_pp']:+.1f}pp")
        print(f"    Random reduction:   {r['random_reduction_pp']:+.1f}pp")
        print(f"    Semantic advantage: {r['semantic_advantage_pp']:+.1f}pp")
        print(f"    Semantic better: {r['semantic_better_than_random']}")

    # Save
    json_path = OUT_DIR / 'results_abstraction_random_control.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("Done.")
