#!/usr/bin/env python3
"""
Prevalence Survey: Attribution Instability Under Collinearity

Surveys 30+ public datasets to measure how often the attribution
impossibility theorem applies in practice. For each dataset we check:
  1. Whether correlated feature pairs exist (|ρ| > 0.5)
  2. Whether XGBoost attribution rankings are unstable across seeds
     (flip rate > 10% for correlated pairs)

Results saved to paper/results_prevalence.json.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import traceback
import numpy as np
import pandas as pd
from itertools import combinations

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_sklearn_datasets():
    """Load sklearn toy and real-world datasets."""
    from sklearn.datasets import (
        load_breast_cancer, load_diabetes, load_wine, load_iris,
        load_digits, load_linnerud, fetch_california_housing,
    )

    datasets = []

    # Classification datasets
    for loader, name, task in [
        (load_breast_cancer, "breast_cancer", "clf"),
        (load_wine, "wine", "clf"),
        (load_iris, "iris", "clf"),
        (load_digits, "digits", "clf"),
    ]:
        try:
            d = loader()
            datasets.append((name, d.data, d.target, task))
        except Exception:
            pass

    # Regression datasets
    for loader, name, task in [
        (load_diabetes, "diabetes", "reg"),
        (fetch_california_housing, "california_housing", "reg"),
    ]:
        try:
            d = loader()
            datasets.append((name, d.data, d.target, task))
        except Exception:
            pass

    # Linnerud is multi-output; use first target
    try:
        d = load_linnerud()
        datasets.append(("linnerud", d.data, d.target[:, 0], "reg"))
    except Exception:
        pass

    return datasets


def load_openml_datasets():
    """Load datasets from OpenML by ID."""
    from sklearn.datasets import fetch_openml

    openml_specs = [
        # (name, data_id, task)
        ("heart-statlog", 53, "clf"),
        ("credit-g", 31, "clf"),
        ("default-credit-card", 42477, "clf"),
        ("adult", 1590, "clf"),
        ("us_crime", 42730, "reg"),
        ("kc_house_prices", 41021, "reg"),
        ("bike_sharing", 42712, "reg"),
        ("phoneme", 1489, "clf"),
        ("blood-transfusion", 1464, "clf"),
        ("vehicle", 54, "clf"),
        ("segment", 36, "clf"),
        ("steel-plates-fault", 1504, "clf"),
        ("vowel", 307, "clf"),
        ("scene", 312, "clf"),
        ("analcatdata_authorship", 458, "clf"),
        ("pc1", 1068, "clf"),
        ("kc1", 1067, "clf"),
        ("pc4", 1049, "clf"),
        ("ionosphere", 59, "clf"),
        ("sonar", 40, "clf"),
        ("mfeat-factors", 12, "clf"),
        ("mfeat-fourier", 14, "clf"),
        ("mfeat-karhunen", 16, "clf"),
        ("wilt", 40983, "clf"),
        ("ozone-level-8hr", 1487, "clf"),
        ("spambase", 44, "clf"),
        ("wall-robot-navigation", 1497, "clf"),
        ("electricity", 151, "clf"),
        ("banknote-authentication", 1462, "clf"),
        ("climate-simulation-crashes", 40994, "clf"),
    ]

    datasets = []
    for name, did, task in openml_specs:
        try:
            d = fetch_openml(data_id=did, as_frame=True, parser="auto")
            X = d.data
            y = d.target

            # Drop non-numeric columns
            X = X.select_dtypes(include=[np.number])
            if X.shape[1] < 2:
                continue

            # Handle target
            y_arr = np.array(y)
            if task == "clf":
                # Encode categorical targets
                if y_arr.dtype == object or str(y_arr.dtype) == "category":
                    from sklearn.preprocessing import LabelEncoder
                    y_arr = LabelEncoder().fit_transform(y_arr)
                else:
                    y_arr = y_arr.astype(float)
            else:
                y_arr = y_arr.astype(float)

            # Drop rows with NaN
            mask = ~(np.isnan(X.values).any(axis=1) | np.isnan(y_arr))
            X_clean = X.values[mask]
            y_clean = y_arr[mask]

            if len(y_clean) < 50:
                continue

            # Subsample large datasets for speed
            if len(y_clean) > 10000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(y_clean), 10000, replace=False)
                X_clean = X_clean[idx]
                y_clean = y_clean[idx]

            datasets.append((name, X_clean, y_clean, task))
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def correlated_pairs(X, threshold=0.5):
    """Return list of (i, j, |ρ|) pairs above threshold."""
    corr = np.corrcoef(X, rowvar=False)
    P = X.shape[1]
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            r = abs(corr[i, j])
            if not np.isnan(r) and r > threshold:
                pairs.append((i, j, float(r)))
    return pairs


def measure_instability(X, y, task, corr_pairs, n_models=20):
    """
    Train n_models XGBoost with different seeds, compute mean |SHAP| per
    feature (TreeExplainer, 100 test samples), and check flip rate for
    correlated pairs.

    Each model uses a different random seed AND an 80% bootstrap subsample
    to expose the Rashomon effect (seed alone is insufficient — XGBoost is
    near-deterministic on identical data).

    Flip rate for a pair (j, k) = min(count(phi_j > phi_k), count(phi_k > phi_j)) / n_models.
    If any pair has flip rate > 10%, has_instability = True.
    """
    from xgboost import XGBClassifier, XGBRegressor
    import shap

    rankings_list = []

    # Fixed evaluation set: first 100 samples (or fewer if dataset is small)
    n_eval = min(100, len(X))
    X_eval = X[:n_eval]

    for seed in range(n_models):
        try:
            rng = np.random.RandomState(seed)

            # 80% bootstrap subsample — creates genuinely different models
            n = len(y)
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            X_train, y_train = X[idx], y[idx]

            params = dict(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                random_state=seed, n_jobs=1, verbosity=0,
            )
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

            # For multiclass, average absolute SHAP across classes
            if isinstance(shap_vals, list):
                shap_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            else:
                shap_vals = np.abs(shap_vals)

            # Mean |SHAP| per feature
            mean_shap = np.mean(shap_vals, axis=0)
            if mean_shap.ndim > 1:
                mean_shap = np.mean(mean_shap, axis=1)
            rankings_list.append(mean_shap)
        except Exception as e:
            print(f"      [shap skip] seed {seed}: {e}")
            continue

    if len(rankings_list) < 2:
        return False, 0.0

    # Check flip rate for correlated pairs
    # flip_rate = min(count(phi_j > phi_k), count(phi_k > phi_j)) / n_models
    has_instability = False
    max_flip_rate = 0.0
    n = len(rankings_list)

    for i, j, rho in corr_pairs:
        if i >= len(rankings_list[0]) or j >= len(rankings_list[0]):
            continue
        count_i_wins = sum(1 for r in rankings_list if r[i] > r[j])
        count_j_wins = sum(1 for r in rankings_list if r[j] > r[i])
        flip_rate = min(count_i_wins, count_j_wins) / n if n > 0 else 0
        max_flip_rate = max(max_flip_rate, flip_rate)
        if flip_rate > 0.10:
            has_instability = True

    return has_instability, float(max_flip_rate)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PREVALENCE SURVEY: Attribution Instability Under Collinearity")
    print("=" * 60)
    print()

    # Collect datasets
    print("Loading sklearn datasets...")
    datasets = load_sklearn_datasets()
    print(f"  Loaded {len(datasets)} sklearn datasets")

    print("Loading OpenML datasets...")
    openml_ds = load_openml_datasets()
    datasets.extend(openml_ds)
    print(f"  Loaded {len(openml_ds)} OpenML datasets")
    print(f"  Total: {len(datasets)} datasets")
    print()

    results = []

    for idx, (name, X, y, task) in enumerate(datasets):
        print(f"[{idx+1}/{len(datasets)}] {name}  "
              f"(n={X.shape[0]}, P={X.shape[1]}, {task})")

        try:
            P = X.shape[1]
            n_pairs = P * (P - 1) // 2
            cpairs = correlated_pairs(X, threshold=0.5)
            n_corr = len(cpairs)

            row = {
                "dataset": name,
                "n_samples": int(X.shape[0]),
                "n_features": int(P),
                "task": task,
                "n_pairs": int(n_pairs),
                "n_corr_pairs": n_corr,
                "has_correlated": n_corr > 0,
                "has_instability": False,
                "max_flip_rate": 0.0,
            }

            if n_corr > 0:
                has_inst, max_flip = measure_instability(
                    X, y, task, cpairs, n_models=20
                )
                row["has_instability"] = has_inst
                row["max_flip_rate"] = round(max_flip, 4)
                tag = " ** UNSTABLE **" if has_inst else ""
                print(f"    corr pairs: {n_corr}/{n_pairs}, "
                      f"max flip: {max_flip:.1%}{tag}")
            else:
                print(f"    corr pairs: 0/{n_pairs} — skipping instability check")

            results.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    n_total = len(results)
    n_corr = sum(1 for r in results if r["has_correlated"])
    n_inst = sum(1 for r in results if r["has_instability"])

    print(f"PREVALENCE SURVEY: Attribution Instability")
    print(f"Datasets surveyed:              {n_total}")
    print(f"Datasets with |ρ| > 0.5 pairs:  {n_corr} ({100*n_corr/n_total:.0f}%)")
    print(f"Datasets with flip rate > 10%:  {n_inst} ({100*n_inst/n_total:.0f}%)")
    print("=" * 60)

    # Save results
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results_prevalence.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "n_datasets": n_total,
                "n_with_correlation": n_corr,
                "pct_with_correlation": round(100 * n_corr / n_total, 1) if n_total else 0,
                "n_with_instability": n_inst,
                "pct_with_instability": round(100 * n_inst / n_total, 1) if n_total else 0,
            },
            "datasets": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
