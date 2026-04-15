#!/usr/bin/env python3
"""
Balance-Matched Random Merging Control.

The unmatched random control revealed class-balance confounds:
- Fashion-MNIST semantic merge is 50/50 while random merges average ~40/60
- Covertype's most balanced random merge beats semantic

This control constrains random merges to have similar class balance to the
semantic merge (within ±5pp). If semantic still wins under balanced controls,
the structural claim survives.

Only tests datasets with enough classes for meaningful diversity:
- Fashion-MNIST (10 → 2)
- Covertype (7 → 2)
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_covtype, fetch_openml
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEEDS = list(range(42, 72))


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


def balance_matched_random_merge(y_orig, n_target, target_balance, n_attempts=1000, tolerance=0.05):
    """Generate random merges matching target class balance within tolerance."""
    orig_classes = np.unique(y_orig)
    matched_merges = []

    for seed in range(n_attempts):
        rng = np.random.RandomState(seed + 5000)
        assignment = rng.randint(0, n_target, size=len(orig_classes))
        if len(np.unique(assignment)) < n_target:
            continue
        mapping = dict(zip(orig_classes, assignment))
        y_merged = np.array([mapping[c] for c in y_orig])
        balance = min(np.bincount(y_merged)) / len(y_merged)
        if abs(balance - target_balance) <= tolerance:
            matched_merges.append((seed + 5000, y_merged, balance))

        if len(matched_merges) >= 20:
            break

    return matched_merges


def run_balanced_control(name, X, y_orig, y_semantic, n_orig, n_target):
    print(f"\n{'='*60}")
    print(f"BALANCE-MATCHED CONTROL: {name}")
    print(f"  {n_orig} classes → {n_target} classes")
    print(f"{'='*60}")
    t0 = time.time()

    # Semantic merge balance
    sem_balance = min(np.bincount(y_semantic)) / len(y_semantic)
    print(f"  Semantic merge balance: {sem_balance:.3f}")

    # Fine-grained baseline
    print(f"\n  Training fine-grained ({n_orig} classes)...")
    imp_fine = train_models(X, y_orig, SEEDS, n_orig)
    fine_flip = compute_flip_rate(imp_fine)
    print(f"    Flip rate: {fine_flip:.4f}")

    # Semantic merge
    print(f"  Training semantic merge...")
    imp_sem = train_models(X, y_semantic, SEEDS, n_target)
    sem_flip = compute_flip_rate(imp_sem)
    print(f"    Flip rate: {sem_flip:.4f}")

    # Balance-matched random merges
    print(f"  Finding balance-matched random merges (target: {sem_balance:.3f} ± 0.05)...")
    matched = balance_matched_random_merge(y_orig, n_target, sem_balance, n_attempts=2000, tolerance=0.05)
    print(f"  Found {len(matched)} matched merges")

    matched_flips = []
    matched_balances = []
    for seed, y_merged, balance in matched[:10]:  # Use up to 10
        print(f"    Random (seed={seed}, balance={balance:.3f})...")
        imp_r = train_models(X, y_merged, SEEDS, n_target)
        rf = compute_flip_rate(imp_r)
        matched_flips.append(rf)
        matched_balances.append(balance)
        print(f"      Flip rate: {rf:.4f}")

    if not matched_flips:
        print("  ERROR: No balance-matched merges found!")
        return None

    mean_matched = float(np.mean(matched_flips))
    std_matched = float(np.std(matched_flips)) if len(matched_flips) > 1 else 0.0

    semantic_advantage = mean_matched - sem_flip
    semantic_better = sem_flip < mean_matched

    # How many matched randoms does semantic beat?
    n_beaten = sum(1 for rf in matched_flips if sem_flip < rf)

    elapsed = time.time() - t0

    result = {
        "dataset": name,
        "n_orig_classes": n_orig,
        "n_target_classes": n_target,
        "semantic_balance": float(sem_balance),
        "fine_flip": fine_flip,
        "semantic_flip": sem_flip,
        "n_matched_randoms": len(matched_flips),
        "matched_flips": matched_flips,
        "matched_balances": matched_balances,
        "mean_matched_flip": mean_matched,
        "std_matched_flip": std_matched,
        "semantic_advantage_pp": round(semantic_advantage * 100, 1),
        "semantic_better": semantic_better,
        "n_randoms_beaten": n_beaten,
        "beat_rate": n_beaten / len(matched_flips),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  RESULT:")
    print(f"    Fine flip:               {fine_flip:.4f}")
    print(f"    Semantic flip:           {sem_flip:.4f} (balance={sem_balance:.3f})")
    print(f"    Matched random mean:     {mean_matched:.4f} ± {std_matched:.4f}")
    print(f"    Semantic advantage:      {semantic_advantage*100:+.1f}pp")
    print(f"    Semantic beats N/total:  {n_beaten}/{len(matched_flips)}")
    print(f"    Semantic better overall: {semantic_better}")

    return result


if __name__ == '__main__':
    print("Balance-Matched Random Merging Control")
    print("=" * 60)
    print("QUESTION: Does semantic merging beat random AFTER controlling for balance?\n")

    all_results = {}

    # Fashion-MNIST: 10 → 2
    print("Loading Fashion-MNIST...")
    X_fm, y_fm_str = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
    y_fm_full = y_fm_str.astype(int)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_fm), size=5000, replace=False)
    X_fm = X_fm[idx].astype(np.float64)
    y_fm = y_fm_full[idx]
    # Semantic: clothing vs non-clothing
    sup_fm = {0: 0, 2: 0, 4: 0, 6: 0, 1: 0, 5: 1, 7: 1, 9: 1, 3: 1, 8: 1}
    y_fm_sem = np.array([sup_fm[c] for c in y_fm])
    r = run_balanced_control("Fashion-MNIST", X_fm, y_fm, y_fm_sem, 10, 2)
    if r:
        all_results["Fashion-MNIST"] = r

    # Covertype: 7 → 2
    print("\nLoading Covertype...")
    cov = fetch_covtype()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(cov.data), size=5000, replace=False)
    X_cov = cov.data[idx]
    y_cov = cov.target[idx] - 1
    sup_cov = {0: 0, 1: 0, 2: 1, 3: 1, 5: 1, 4: 1, 6: 1}
    y_cov_sem = np.array([sup_cov.get(c, 1) for c in y_cov])
    r = run_balanced_control("Covertype", X_cov, y_cov, y_cov_sem, 7, 2)
    if r:
        all_results["Covertype"] = r

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Balance-Matched Control")
    print("=" * 60)

    for name, r in all_results.items():
        print(f"\n  {name}:")
        print(f"    Semantic advantage: {r['semantic_advantage_pp']:+.1f}pp (balance-matched)")
        print(f"    Semantic beats: {r['n_randoms_beaten']}/{r['n_matched_randoms']} random merges")

    # Save
    json_path = OUT_DIR / 'results_abstraction_balanced_control.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("Done.")
