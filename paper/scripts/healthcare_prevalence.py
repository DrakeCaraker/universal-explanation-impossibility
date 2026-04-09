#!/usr/bin/env python3
"""
Healthcare Prevalence Survey: Attribution Instability Under Collinearity

Surveys medical/healthcare datasets to measure how often the attribution
impossibility theorem applies in clinical ML settings. For each dataset:
  1. Whether correlated feature pairs exist (|rho| > 0.5)
  2. Whether XGBoost attribution rankings are unstable across seeds
     (flip rate > 10% for correlated pairs)

Results saved to paper/results_healthcare_prevalence.json.
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

def load_sklearn_medical():
    """Load medical datasets from sklearn."""
    from sklearn.datasets import load_breast_cancer, load_diabetes

    datasets = []
    for loader, name, task in [
        (load_breast_cancer, "breast_cancer", "clf"),
        (load_diabetes, "sklearn_diabetes", "reg"),
    ]:
        try:
            d = loader()
            datasets.append((name, d.data, d.target, task))
            print(f"    loaded {name} ({d.data.shape[0]} x {d.data.shape[1]})")
        except Exception as e:
            print(f"    [skip] {name}: {e}")

    return datasets


def _fetch_openml_dataset(name, did, task):
    """Fetch a single OpenML dataset, returning (name, X, y, task) or None."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    try:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X = d.data
        y = d.target

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] < 2:
            print(f"    [skip] {name}: fewer than 2 numeric features")
            return None

        # Handle target
        y_arr = np.array(y)
        if task == "clf":
            if y_arr.dtype == object or str(y_arr.dtype) == "category":
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
            print(f"    [skip] {name}: only {len(y_clean)} clean rows")
            return None

        # Subsample large datasets for speed
        if len(y_clean) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_clean), 10000, replace=False)
            X_clean = X_clean[idx]
            y_clean = y_clean[idx]

        print(f"    loaded {name} ({X_clean.shape[0]} x {X_clean.shape[1]})")
        return (name, X_clean, y_clean, task)

    except Exception as e:
        print(f"    [skip] {name} (id={did}): {e}")
        return None


def _fetch_openml_by_name(name, task):
    """Fetch an OpenML dataset by name, returning (name, X, y, task) or None."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    try:
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        X = d.data
        y = d.target

        X = X.select_dtypes(include=[np.number])
        if X.shape[1] < 2:
            print(f"    [skip] {name}: fewer than 2 numeric features")
            return None

        y_arr = np.array(y)
        if task == "clf":
            if y_arr.dtype == object or str(y_arr.dtype) == "category":
                y_arr = LabelEncoder().fit_transform(y_arr)
            else:
                y_arr = y_arr.astype(float)
        else:
            y_arr = y_arr.astype(float)

        mask = ~(np.isnan(X.values).any(axis=1) | np.isnan(y_arr))
        X_clean = X.values[mask]
        y_clean = y_arr[mask]

        if len(y_clean) < 50:
            print(f"    [skip] {name}: only {len(y_clean)} clean rows")
            return None

        if len(y_clean) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_clean), 10000, replace=False)
            X_clean = X_clean[idx]
            y_clean = y_clean[idx]

        print(f"    loaded {name} ({X_clean.shape[0]} x {X_clean.shape[1]})")
        return (name, X_clean, y_clean, task)

    except Exception as e:
        print(f"    [skip] {name} (by name): {e}")
        return None


def load_openml_medical():
    """Load healthcare/medical datasets from OpenML by ID and by name."""

    # Datasets by ID
    openml_by_id = [
        ("heart-statlog", 53, "clf"),
        ("qsar-biodeg", 1494, "clf"),
        ("ilpd", 1480, "clf"),
        ("pc1", 1068, "clf"),
        ("parkinsons", 23517, "clf"),
        ("diabetes", 37, "clf"),
        ("analcatdata_dmft", 469, "clf"),
    ]

    # Datasets to try by name
    openml_by_name = [
        ("dermatology", "clf"),
        ("hepatitis", "clf"),
        ("echocardiogram", "clf"),
        ("thyroid", "clf"),
        ("primary-tumor", "clf"),
        ("lymphography", "clf"),
    ]

    datasets = []

    print("  Loading by OpenML ID...")
    for name, did, task in openml_by_id:
        result = _fetch_openml_dataset(name, did, task)
        if result is not None:
            datasets.append(result)

    print("  Loading by OpenML name...")
    for name, task in openml_by_name:
        result = _fetch_openml_by_name(name, task)
        if result is not None:
            datasets.append(result)

    return datasets


def load_openml_by_search():
    """Search OpenML for additional medical/health datasets."""
    datasets = []

    try:
        import openml
        search_tags = ["health", "medical", "clinical", "heart", "cancer",
                       "diabetes", "mortality", "hospital"]
        seen_ids = set()

        for tag in search_tags:
            try:
                listing = openml.datasets.list_datasets(
                    tag=tag, output_format="dataframe"
                )
                if listing is not None and len(listing) > 0:
                    # Pick datasets with reasonable size
                    candidates = listing[
                        (listing["NumberOfInstances"] >= 50) &
                        (listing["NumberOfInstances"] <= 50000) &
                        (listing["NumberOfFeatures"] >= 3) &
                        (listing["NumberOfFeatures"] <= 500)
                    ]
                    # Take up to 3 per tag to avoid excessive downloads
                    for did in candidates.index[:3]:
                        if did in seen_ids:
                            continue
                        seen_ids.add(did)
                        dname = f"openml_{did}"
                        if did in candidates.index:
                            row = candidates.loc[did]
                            if hasattr(row, "name"):
                                dname = str(row["name"])
                        result = _fetch_openml_dataset(dname, int(did), "clf")
                        if result is not None:
                            datasets.append(result)
            except Exception as e:
                print(f"    [skip tag] {tag}: {e}")

    except ImportError:
        print("    openml package not available, skipping tag search")
    except Exception as e:
        print(f"    [skip] OpenML search: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Analysis helpers (same as prevalence_survey.py)
# ---------------------------------------------------------------------------

def correlated_pairs(X, threshold=0.5):
    """Return list of (i, j, |rho|) pairs above threshold."""
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
    to expose the Rashomon effect.

    Flip rate for a pair (j, k) = min(count(phi_j > phi_k), count(phi_k > phi_j)) / n_models.
    If any pair has flip rate > 10%, has_instability = True.
    """
    from xgboost import XGBClassifier, XGBRegressor
    import shap

    rankings_list = []

    # Fixed evaluation set: first 100 samples (or fewer)
    n_eval = min(100, len(X))
    X_eval = X[:n_eval]

    for seed in range(n_models):
        try:
            rng = np.random.RandomState(seed)

            # 80% bootstrap subsample
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
    print("HEALTHCARE PREVALENCE SURVEY")
    print("Attribution Instability Under Collinearity")
    print("=" * 60)
    print()

    datasets = []

    # 1. sklearn medical datasets
    print("Loading sklearn medical datasets...")
    sklearn_ds = load_sklearn_medical()
    datasets.extend(sklearn_ds)
    print(f"  => {len(sklearn_ds)} sklearn datasets\n")

    # 2. OpenML medical datasets by ID and name
    print("Loading OpenML medical datasets (by ID and name)...")
    openml_ds = load_openml_medical()
    datasets.extend(openml_ds)
    print(f"  => {len(openml_ds)} OpenML datasets\n")

    # 3. OpenML tag search for additional medical datasets
    print("Searching OpenML by tag for additional medical datasets...")
    search_ds = load_openml_by_search()
    # Deduplicate by name
    existing_names = {d[0] for d in datasets}
    new_ds = [d for d in search_ds if d[0] not in existing_names]
    datasets.extend(new_ds)
    print(f"  => {len(new_ds)} additional datasets from tag search\n")

    print(f"Total healthcare datasets: {len(datasets)}")
    print()

    # Analyze each dataset
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
                print(f"    corr pairs: 0/{n_pairs} -- skipping instability check")

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

    print("HEALTHCARE PREVALENCE")
    print(f"Datasets: {n_total}")
    if n_total > 0:
        print(f"With |rho|>0.5: {n_corr} ({100*n_corr/n_total:.0f}%)")
        print(f"With instability: {n_inst} ({100*n_inst/n_total:.0f}%)")
    else:
        print("No datasets loaded successfully.")
    print("=" * 60)

    # Save results
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results_healthcare_prevalence.json"
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
