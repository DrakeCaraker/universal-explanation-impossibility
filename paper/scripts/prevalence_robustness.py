#!/usr/bin/env python3
"""
Prevalence Robustness Check: Attribution Instability Under Different XGBoost Configs

Reviewer question: Does the 60% instability rate hold under different XGBoost
hyperparameters? This script re-runs the instability check for 8 sklearn datasets
plus up to 2 OpenML datasets across 3 XGBoost configurations.

Configurations:
  (a) Original:     n_estimators=50,  max_depth=4, lr=0.1,  sub=0.8, colsub=0.8
  (b) Deep:         n_estimators=100, max_depth=8, lr=0.1,  sub=0.8, colsub=0.8
  (c) Shallow-fast: n_estimators=200, max_depth=2, lr=0.3,  sub=0.8, colsub=0.8

Results saved to paper/results_prevalence_robustness.txt
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# XGBoost config definitions
# ---------------------------------------------------------------------------

CONFIGS = {
    "A_original":     dict(n_estimators=50,  max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8),
    "B_deep":         dict(n_estimators=100, max_depth=8, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8),
    "C_shallow_fast": dict(n_estimators=200, max_depth=2, learning_rate=0.3,
                           subsample=0.8, colsample_bytree=0.8),
}

N_MODELS = 20
FLIP_THRESHOLD = 0.10
CORR_THRESHOLD = 0.5
N_EVAL = 500   # fixed evaluation set size (or smaller if dataset is smaller)

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_sklearn_datasets():
    """Load the 6 primary sklearn datasets."""
    from sklearn.datasets import (
        load_breast_cancer, load_wine, load_diabetes,
        load_iris, load_digits, fetch_california_housing,
    )

    datasets = []

    # Breast Cancer
    try:
        d = load_breast_cancer()
        datasets.append(("breast_cancer", d.data, d.target, "clf"))
        print("  [OK] breast_cancer")
    except Exception as e:
        print(f"  [FAIL] breast_cancer: {e}")

    # Wine
    try:
        d = load_wine()
        datasets.append(("wine", d.data, d.target, "clf"))
        print("  [OK] wine")
    except Exception as e:
        print(f"  [FAIL] wine: {e}")

    # Diabetes (regression)
    try:
        d = load_diabetes()
        datasets.append(("diabetes", d.data, d.target, "reg"))
        print("  [OK] diabetes")
    except Exception as e:
        print(f"  [FAIL] diabetes: {e}")

    # California Housing (subsample to 5000)
    try:
        d = fetch_california_housing()
        X, y = d.data, d.target
        if len(y) > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y), 5000, replace=False)
            X, y = X[idx], y[idx]
        datasets.append(("california_housing", X, y, "reg"))
        print("  [OK] california_housing")
    except Exception as e:
        print(f"  [FAIL] california_housing: {e}")

    # Iris
    try:
        d = load_iris()
        datasets.append(("iris", d.data, d.target, "clf"))
        print("  [OK] iris")
    except Exception as e:
        print(f"  [FAIL] iris: {e}")

    # Digits (binary: class 0 vs 1, subsample to 500)
    try:
        d = load_digits()
        mask = (d.target == 0) | (d.target == 1)
        X_d, y_d = d.data[mask], d.target[mask]
        if len(y_d) > 500:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_d), 500, replace=False)
            X_d, y_d = X_d[idx], y_d[idx]
        datasets.append(("digits_01", X_d, y_d, "clf"))
        print("  [OK] digits_01")
    except Exception as e:
        print(f"  [FAIL] digits_01: {e}")

    return datasets


def load_openml_datasets():
    """Try to load 2 OpenML datasets. Fail silently if unavailable."""
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return []

    specs = [
        ("credit_g",    31,  "clf"),
        ("heart_statlog", 53, "clf"),
    ]

    datasets = []
    for name, did, task in specs:
        try:
            print(f"  Fetching OpenML id={did} ({name})...")
            d = fetch_openml(data_id=did, as_frame=True, parser="auto")
            X = d.data.select_dtypes(include=[np.number])
            if X.shape[1] < 2:
                print(f"  [skip] {name}: too few numeric features")
                continue

            y_raw = np.array(d.target)
            if y_raw.dtype == object or str(y_raw.dtype) == "category":
                y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
            else:
                y_raw = y_raw.astype(float)

            X_arr = X.values
            mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_raw))
            X_arr, y_raw = X_arr[mask], y_raw[mask]

            if len(y_raw) < 50:
                print(f"  [skip] {name}: too few samples after cleaning")
                continue

            if len(y_raw) > 5000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(y_raw), 5000, replace=False)
                X_arr, y_raw = X_arr[idx], y_raw[idx]

            datasets.append((name, X_arr, y_raw, task))
            print(f"  [OK] {name} (n={X_arr.shape[0]}, P={X_arr.shape[1]})")
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def correlated_pairs(X, threshold=CORR_THRESHOLD):
    """Return list of (i, j, |ρ|) for feature pairs above threshold."""
    corr = np.corrcoef(X, rowvar=False)
    P = X.shape[1]
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            r = abs(corr[i, j])
            if not np.isnan(r) and r > threshold:
                pairs.append((i, j, float(r)))
    return pairs


def measure_instability(X, y, task, corr_pairs, config_params, n_models=N_MODELS):
    """
    Train n_models XGBoost models (each on a different 80% bootstrap subsample
    with a different seed) using the given hyperparameter config.

    Evaluate on a fixed N_EVAL-sample set.
    Flip rate for pair (i,j) = min(count(φᵢ>φⱼ), count(φⱼ>φᵢ)) / n_models.

    Returns (has_instability: bool, max_flip_rate: float).
    """
    from xgboost import XGBClassifier, XGBRegressor
    import shap

    n_eval = min(N_EVAL, len(X))
    X_eval = X[:n_eval]

    rankings_list = []

    for seed in range(n_models):
        try:
            rng = np.random.RandomState(seed)
            n = len(y)
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            X_train, y_train = X[idx], y[idx]

            params = dict(random_state=seed, n_jobs=1, verbosity=0,
                          **config_params)

            if task == "clf":
                n_classes = len(np.unique(y_train))
                if n_classes > 2:
                    params["objective"] = "multi:softprob"
                    params["num_class"] = n_classes
                elif n_classes < 2:
                    continue
                model = XGBClassifier(**params)
            else:
                model = XGBRegressor(**params)

            model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_eval)

            # Multiclass: average absolute SHAP across classes
            if isinstance(shap_vals, list):
                shap_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            else:
                shap_vals = np.abs(shap_vals)

            mean_shap = np.mean(shap_vals, axis=0)
            if mean_shap.ndim > 1:
                mean_shap = np.mean(mean_shap, axis=1)
            rankings_list.append(mean_shap)

        except Exception as e:
            # Skip individual seed failures silently
            continue

    if len(rankings_list) < 2:
        return False, 0.0

    n = len(rankings_list)
    has_instability = False
    max_flip_rate = 0.0

    for i, j, rho in corr_pairs:
        if i >= len(rankings_list[0]) or j >= len(rankings_list[0]):
            continue
        count_i = sum(1 for r in rankings_list if r[i] > r[j])
        count_j = sum(1 for r in rankings_list if r[j] > r[i])
        flip_rate = min(count_i, count_j) / n
        if flip_rate > max_flip_rate:
            max_flip_rate = flip_rate
        if flip_rate > FLIP_THRESHOLD:
            has_instability = True

    return has_instability, float(max_flip_rate)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PREVALENCE ROBUSTNESS CHECK")
    print("Reviewer question: Does 60% instability rate hold under different")
    print("XGBoost hyperparameter configurations?")
    print("=" * 70)
    print()

    # Collect datasets
    print("Loading sklearn datasets...")
    datasets = load_sklearn_datasets()
    print(f"  Loaded {len(datasets)} sklearn datasets\n")

    print("Loading OpenML datasets (may time out — will skip if unavailable)...")
    openml_ds = load_openml_datasets()
    datasets.extend(openml_ds)
    print(f"  Loaded {len(openml_ds)} OpenML datasets")
    print(f"  Total datasets: {len(datasets)}\n")

    if len(datasets) == 0:
        print("ERROR: No datasets loaded. Aborting.")
        sys.exit(1)

    # Per-dataset results: {name: {config_key: unstable_bool, ...}}
    results = []

    config_keys = list(CONFIGS.keys())

    for di, (name, X, y, task) in enumerate(datasets):
        P = X.shape[1]
        n = X.shape[0]
        n_pairs_total = P * (P - 1) // 2
        cpairs = correlated_pairs(X)
        n_corr = len(cpairs)

        print(f"[{di+1}/{len(datasets)}] {name}  "
              f"(n={n}, P={P}, {task}, corr_pairs={n_corr}/{n_pairs_total})")

        row = {
            "dataset": name,
            "n_samples": int(n),
            "n_features": int(P),
            "task": task,
            "n_corr_pairs": int(n_corr),
            "n_pairs_total": int(n_pairs_total),
        }

        if n_corr == 0:
            print("    No correlated pairs — all configs: stable (no pairs)")
            for ck in config_keys:
                row[f"unstable_{ck}"] = False
                row[f"flip_rate_{ck}"] = 0.0
            results.append(row)
            continue

        for ck, cfg in CONFIGS.items():
            label = {"A_original": "A (original)", "B_deep": "B (deep)",
                     "C_shallow_fast": "C (shallow-fast)"}[ck]
            try:
                unstable, flip_rate = measure_instability(
                    X, y, task, cpairs, cfg, n_models=N_MODELS
                )
                row[f"unstable_{ck}"] = unstable
                row[f"flip_rate_{ck}"] = round(flip_rate, 4)
                tag = " ** UNSTABLE **" if unstable else ""
                print(f"    Config {label}: flip_rate={flip_rate:.1%}{tag}")
            except Exception as e:
                print(f"    Config {label}: ERROR — {e}")
                row[f"unstable_{ck}"] = None
                row[f"flip_rate_{ck}"] = None

        results.append(row)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    # Header
    col_w = 22
    hdr = (f"{'Dataset':<{col_w}} {'#feat':>5} {'#corr':>6}"
           f"  {'Config A':>10}  {'Config B':>10}  {'Config C':>10}")
    print(hdr)
    print("-" * len(hdr))

    for row in results:
        name = row["dataset"]
        P = row["n_features"]
        nc = row["n_corr_pairs"]

        def fmt(ck):
            v = row.get(f"unstable_{ck}")
            fr = row.get(f"flip_rate_{ck}")
            if v is None:
                return "  ERROR     "
            mark = "YES" if v else " no"
            fr_str = f"({fr:.0%})" if fr is not None else ""
            return f"{mark:>3} {fr_str:>6}"

        line = (f"{name:<{col_w}} {P:>5} {nc:>6}"
                f"  {fmt('A_original'):>12}"
                f"  {fmt('B_deep'):>12}"
                f"  {fmt('C_shallow_fast'):>12}")
        print(line)

    print("-" * len(hdr))

    # Counts — only over datasets that have correlated pairs
    has_corr = [r for r in results if r["n_corr_pairs"] > 0]
    N_corr = len(has_corr)
    N_total = len(results)

    def count_unstable(ck):
        return sum(1 for r in has_corr if r.get(f"unstable_{ck}") is True)

    n_A = count_unstable("A_original")
    n_B = count_unstable("B_deep")
    n_C = count_unstable("C_shallow_fast")

    n_all3 = sum(
        1 for r in has_corr
        if (r.get("unstable_A_original") is True
            and r.get("unstable_B_deep") is True
            and r.get("unstable_C_shallow_fast") is True)
    )
    n_any1 = sum(
        1 for r in has_corr
        if any(r.get(f"unstable_{ck}") is True for ck in config_keys)
    )

    print()
    print("SUMMARY (datasets with ≥1 correlated pair, |ρ|>0.5):")
    print(f"  Total datasets:               {N_total}")
    print(f"  With correlated pairs:        {N_corr}/{N_total}")
    print(f"  Unstable under Config A:      {n_A}/{N_corr} ({100*n_A/N_corr:.0f}%)")
    print(f"  Unstable under Config B:      {n_B}/{N_corr} ({100*n_B/N_corr:.0f}%)")
    print(f"  Unstable under Config C:      {n_C}/{N_corr} ({100*n_C/N_corr:.0f}%)")
    print(f"  Unstable under ALL 3 configs: {n_all3}/{N_corr} ({100*n_all3/N_corr:.0f}%)")
    print(f"  Unstable under ≥1 config:     {n_any1}/{N_corr} ({100*n_any1/N_corr:.0f}%)")
    print()
    print("COMPARISON TO PAPER'S CLAIM:")
    print("  Paper reports: 60% of 37 public datasets show flip rate >10%")
    print(f"  This robustness check (Config A, the original parameters):")
    print(f"    {n_A}/{N_corr} = {100*n_A/N_corr:.0f}% of datasets with correlated pairs")
    print(f"    {n_A}/{N_total} = {100*n_A/N_total:.0f}% of all datasets")
    if N_corr > 0:
        consistent = "CONSISTENT" if (n_all3 / N_corr) >= 0.5 else "LOWER THAN"
        print(f"  All-3-config instability rate ({100*n_all3/N_corr:.0f}%) is "
              f"{consistent} with the 60% claim.")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save text report
    # ---------------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    txt_path = os.path.join(out_dir, "results_prevalence_robustness.txt")
    json_path = os.path.join(out_dir, "results_prevalence_robustness.json")

    with open(txt_path, "w") as f:
        f.write("PREVALENCE ROBUSTNESS CHECK\n")
        f.write("Reviewer question: Does 60% instability rate hold under different\n")
        f.write("XGBoost hyperparameter configurations?\n\n")
        f.write("Configurations:\n")
        f.write("  A (original):     n_estimators=50,  max_depth=4, lr=0.1,  sub=0.8, colsub=0.8\n")
        f.write("  B (deep):         n_estimators=100, max_depth=8, lr=0.1,  sub=0.8, colsub=0.8\n")
        f.write("  C (shallow-fast): n_estimators=200, max_depth=2, lr=0.3,  sub=0.8, colsub=0.8\n\n")
        f.write("Per-dataset: 20 models × 80% bootstrap subsamples, SHAP on fixed 500-sample eval set\n")
        f.write("Instability criterion: any correlated pair (|ρ|>0.5) has flip rate >10%\n\n")

        f.write(f"{'Dataset':<22} {'#feat':>5} {'#corr':>6}  "
                f"{'Config A':>12}  {'Config B':>12}  {'Config C':>12}\n")
        f.write("-" * 80 + "\n")
        for row in results:
            name = row["dataset"]
            P = row["n_features"]
            nc = row["n_corr_pairs"]

            def fmt(ck):
                v = row.get(f"unstable_{ck}")
                fr = row.get(f"flip_rate_{ck}")
                if v is None:
                    return "  ERROR     "
                mark = "YES" if v else " no"
                fr_str = f"({fr:.0%})" if fr is not None else ""
                return f"{mark} {fr_str:>6}"

            f.write(f"{name:<22} {P:>5} {nc:>6}  "
                    f"{fmt('A_original'):>14}  "
                    f"{fmt('B_deep'):>14}  "
                    f"{fmt('C_shallow_fast'):>14}\n")

        f.write("-" * 80 + "\n\n")
        f.write("SUMMARY (datasets with ≥1 correlated pair, |ρ|>0.5):\n")
        f.write(f"  Total datasets:               {N_total}\n")
        f.write(f"  With correlated pairs:        {N_corr}/{N_total}\n")
        f.write(f"  Unstable under Config A:      {n_A}/{N_corr} ({100*n_A/N_corr:.0f}%)\n")
        f.write(f"  Unstable under Config B:      {n_B}/{N_corr} ({100*n_B/N_corr:.0f}%)\n")
        f.write(f"  Unstable under Config C:      {n_C}/{N_corr} ({100*n_C/N_corr:.0f}%)\n")
        f.write(f"  Unstable under ALL 3 configs: {n_all3}/{N_corr} ({100*n_all3/N_corr:.0f}%)\n")
        f.write(f"  Unstable under ≥1 config:     {n_any1}/{N_corr} ({100*n_any1/N_corr:.0f}%)\n\n")
        f.write("COMPARISON TO PAPER'S CLAIM:\n")
        f.write("  Paper reports: 60% of 37 public datasets show flip rate >10%\n")
        f.write(f"  Config A (original parameters): {n_A}/{N_corr} = {100*n_A/N_corr:.0f}% of datasets with corr pairs\n")
        if N_corr > 0:
            consistent = "CONSISTENT" if (n_all3 / N_corr) >= 0.5 else "LOWER THAN"
            f.write(f"  All-3-config instability rate ({100*n_all3/N_corr:.0f}%) is "
                    f"{consistent} with the 60% claim.\n")

    print(f"\nText results saved to: {txt_path}")

    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "n_total": N_total,
                "n_with_corr": N_corr,
                "n_unstable_A": n_A,
                "n_unstable_B": n_B,
                "n_unstable_C": n_C,
                "n_unstable_all3": n_all3,
                "n_unstable_any1": n_any1,
                "pct_unstable_A": round(100 * n_A / N_corr, 1) if N_corr else 0,
                "pct_unstable_B": round(100 * n_B / N_corr, 1) if N_corr else 0,
                "pct_unstable_C": round(100 * n_C / N_corr, 1) if N_corr else 0,
                "pct_unstable_all3": round(100 * n_all3 / N_corr, 1) if N_corr else 0,
                "pct_unstable_any1": round(100 * n_any1 / N_corr, 1) if N_corr else 0,
            },
            "configs": {k: v for k, v in CONFIGS.items()},
            "datasets": results,
        }, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
