#!/usr/bin/env python3
"""
Extended Prevalence Survey: Attribution Instability Under Collinearity
(OpenML CC-18 + sklearn + OpenML curated)

Extends prevalence_survey.py (37 datasets) to 100+ datasets using the
OpenML CC-18 benchmark suite (suite 99, 72 curated classification datasets)
combined with the original sklearn and hand-picked OpenML datasets.

For each dataset:
  1. Check for correlated feature pairs (|rho| > 0.5)
  2. If correlated pairs exist: train 20 XGBoost models (different seeds,
     80% subsample), compute TreeSHAP, measure flip rates
  3. Flag as unstable if any correlated pair has flip rate > 10%

Results saved to paper/results_prevalence_openml.json.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import signal
import sys
import time
import traceback
import numpy as np
import pandas as pd
from itertools import combinations

import openml


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Model training timed out")


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

    for loader, name, task in [
        (load_breast_cancer, "breast_cancer", "clf"),
        (load_wine, "wine", "clf"),
        (load_iris, "iris", "clf"),
        (load_digits, "digits", "clf"),
    ]:
        try:
            d = loader()
            datasets.append({
                "name": name,
                "X": d.data,
                "y": d.target,
                "task": task,
                "source": "sklearn",
                "domain": "",
            })
        except Exception:
            pass

    for loader, name, task in [
        (load_diabetes, "diabetes", "reg"),
        (fetch_california_housing, "california_housing", "reg"),
    ]:
        try:
            d = loader()
            datasets.append({
                "name": name,
                "X": d.data,
                "y": d.target,
                "task": task,
                "source": "sklearn",
                "domain": "",
            })
        except Exception:
            pass

    try:
        d = load_linnerud()
        datasets.append({
            "name": "linnerud",
            "X": d.data,
            "y": d.target[:, 0],
            "task": "reg",
            "source": "sklearn",
            "domain": "",
        })
    except Exception:
        pass

    return datasets


def load_curated_openml_datasets():
    """Load hand-picked OpenML datasets (same as prevalence_survey.py)."""
    from sklearn.datasets import fetch_openml

    openml_specs = [
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
            X, y = d.data, d.target
            result = _clean_openml_data(X, y, name, task, source="openml-curated")
            if result is not None:
                datasets.append(result)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    return datasets


def load_cc18_datasets(exclude_ids=None):
    """
    Load datasets from the OpenML CC-18 benchmark suite (suite 99).
    72 curated classification datasets.
    """
    exclude_ids = exclude_ids or set()

    print("  Fetching CC-18 suite (OpenML suite 99)...")
    try:
        suite = openml.study.get_suite(99)
        dataset_ids = suite.data
    except Exception as e:
        print(f"  [error] Could not fetch CC-18 suite: {e}")
        print("  Falling back to hardcoded CC-18 dataset IDs...")
        # Hardcoded fallback: CC-18 dataset IDs as of 2025
        dataset_ids = [
            3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37,
            44, 46, 50, 54, 151, 182, 188, 300, 307, 458, 469, 554, 1049,
            1050, 1053, 1063, 1067, 1068, 1461, 1462, 1464, 1467, 1471,
            1475, 1478, 1480, 1485, 1486, 1487, 1489, 1490, 1491, 1492,
            1493, 1494, 1497, 1501, 1504, 1510, 4134, 4534, 6332, 23381,
            23517, 40499, 40668, 40670, 40701, 40900, 40966, 40975, 40978,
            40979, 40981, 40982, 40983, 40984,
        ]

    print(f"  CC-18 contains {len(dataset_ids)} datasets")

    datasets = []
    for did in dataset_ids:
        if did in exclude_ids:
            continue
        try:
            ds = openml.datasets.get_dataset(
                did,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X_df, y_series, _, _ = ds.get_data(
                dataset_format="dataframe",
                target=ds.default_target_attribute,
            )

            # Extract domain from tags
            domain = ""
            if ds.tag:
                tags = ds.tag if isinstance(ds.tag, list) else [ds.tag]
                domain = ", ".join(tags[:3])

            name = ds.name or f"openml-{did}"
            result = _clean_openml_data(
                X_df, y_series, name, "clf",
                source="cc18", domain=domain,
            )
            if result is not None:
                datasets.append(result)

        except Exception as e:
            print(f"  [skip] openml-{did}: {e}")

    return datasets


def _clean_openml_data(X, y, name, task, source="openml", domain=""):
    """
    Clean an OpenML dataset: drop non-numeric columns, handle NaN,
    encode categorical targets, subsample if needed.

    Returns a dict with keys {name, X, y, task, source, domain} or None.
    """
    from sklearn.preprocessing import LabelEncoder

    # Keep only numeric columns
    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=[np.number])
        X_arr = X.values
    else:
        X_arr = np.asarray(X, dtype=float)

    if X_arr.shape[1] < 2:
        return None

    # Encode target
    y_arr = np.asarray(y)
    if task == "clf":
        if y_arr.dtype == object or str(y_arr.dtype) == "category":
            y_arr = LabelEncoder().fit_transform(y_arr)
        else:
            y_arr = y_arr.astype(float)
    else:
        y_arr = y_arr.astype(float)

    # Drop rows with NaN
    nan_rows = np.isnan(X_arr).any(axis=1)
    try:
        nan_y = np.isnan(y_arr.astype(float))
    except (ValueError, TypeError):
        nan_y = np.zeros(len(y_arr), dtype=bool)
    mask = ~(nan_rows | nan_y)
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]

    if len(y_arr) < 50:
        return None

    # Subsample large datasets
    if len(y_arr) > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_arr), 10000, replace=False)
        X_arr = X_arr[idx]
        y_arr = y_arr[idx]

    return {
        "name": name,
        "X": X_arr,
        "y": y_arr,
        "task": task,
        "source": source,
        "domain": domain,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
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


def measure_instability(X, y, task, corr_pairs, n_models=20, timeout_sec=120):
    """
    Train n_models XGBoost with different seeds, compute mean |SHAP| per
    feature (TreeExplainer, 100 eval samples), and check flip rate for
    correlated pairs.

    Each model uses a different random seed AND an 80% bootstrap subsample
    to expose the Rashomon effect.

    Flip rate for a pair (j, k) = min(count(phi_j > phi_k), count(phi_k > phi_j)) / n_models.
    If any pair has flip rate > 10%, has_instability = True.
    """
    from xgboost import XGBClassifier, XGBRegressor
    import shap

    rankings_list = []

    # Fixed evaluation set: first 100 samples
    n_eval = min(100, len(X))
    X_eval = X[:n_eval]

    for seed in range(n_models):
        try:
            # Set per-model timeout via alarm (Unix only)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)

            rng = np.random.RandomState(seed)

            # 80% subsample — creates genuinely different models
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
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
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

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(f"      [timeout] seed {seed}")
            continue
        except Exception as e:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
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
    print("=" * 70)
    print("EXTENDED PREVALENCE SURVEY: Attribution Instability Under Collinearity")
    print("  OpenML CC-18 (suite 99) + sklearn + curated OpenML datasets")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Collect datasets from all sources
    # ------------------------------------------------------------------

    print("Loading sklearn datasets...")
    datasets = load_sklearn_datasets()
    print(f"  Loaded {len(datasets)} sklearn datasets")

    print("Loading curated OpenML datasets...")
    curated = load_curated_openml_datasets()
    print(f"  Loaded {len(curated)} curated OpenML datasets")

    # Track which dataset IDs we already have to avoid duplicates
    curated_openml_ids = {
        53, 31, 42477, 1590, 42730, 41021, 42712, 1489, 1464, 54, 36,
        1504, 307, 312, 458, 1068, 1067, 1049, 59, 40, 12, 14, 16,
        40983, 1487, 44, 1497, 151, 1462, 40994,
    }

    print("Loading CC-18 benchmark suite datasets...")
    cc18 = load_cc18_datasets(exclude_ids=curated_openml_ids)
    print(f"  Loaded {len(cc18)} CC-18 datasets (after dedup)")

    datasets.extend(curated)
    datasets.extend(cc18)

    # Deduplicate by name (prefer earlier source)
    seen_names = set()
    unique_datasets = []
    for ds in datasets:
        if ds["name"] not in seen_names:
            seen_names.add(ds["name"])
            unique_datasets.append(ds)
    datasets = unique_datasets

    print(f"\n  Total unique datasets: {len(datasets)}")
    print()

    # ------------------------------------------------------------------
    # 2. Analyze each dataset
    # ------------------------------------------------------------------

    results = []

    for idx, ds in enumerate(datasets):
        name = ds["name"]
        X = ds["X"]
        y = ds["y"]
        task = ds["task"]
        source = ds["source"]
        domain = ds["domain"]

        print(f"[{idx+1}/{len(datasets)}] {name}  "
              f"(n={X.shape[0]}, P={X.shape[1]}, {task})")

        try:
            P = X.shape[1]
            n_pairs = P * (P - 1) // 2
            cpairs = correlated_pairs(X, threshold=0.5)
            n_corr = len(cpairs)
            max_rho = max((r for _, _, r in cpairs), default=0.0)

            row = {
                "dataset": name,
                "n_samples": int(X.shape[0]),
                "n_features": int(P),
                "task": task,
                "source": source,
                "domain": domain,
                "n_pairs": int(n_pairs),
                "n_corr_pairs": n_corr,
                "has_correlated": n_corr > 0,
                "max_rho": round(max_rho, 4),
                "has_instability": False,
                "max_flip_rate": 0.0,
            }

            if n_corr > 0:
                has_inst, max_flip = measure_instability(
                    X, y, task, cpairs, n_models=20, timeout_sec=120,
                )
                row["has_instability"] = has_inst
                row["max_flip_rate"] = round(max_flip, 4)
                tag = " ** UNSTABLE **" if has_inst else ""
                print(f"    -> corr_pairs: {n_corr}, "
                      f"max_flip: {max_flip:.1%}, "
                      f"max_rho: {max_rho:.3f}{tag}")
            else:
                print(f"    -> corr_pairs: 0/{n_pairs} — skipping instability check")

            results.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    n_total = len(results)
    if n_total == 0:
        print("No datasets processed.")
        return

    n_corr = sum(1 for r in results if r["has_correlated"])
    n_inst = sum(1 for r in results if r["has_instability"])

    # Feature-count breakdown
    low_p = [r for r in results if r["n_features"] < 20]
    mid_p = [r for r in results if 20 <= r["n_features"] < 100]
    high_p = [r for r in results if r["n_features"] >= 100]

    def _stats(subset):
        total = len(subset)
        if total == 0:
            return 0, 0, 0, 0
        c = sum(1 for r in subset if r["has_correlated"])
        i = sum(1 for r in subset if r["has_instability"])
        return total, c, round(100 * c / total, 1), round(100 * i / total, 1)

    print(f"EXTENDED PREVALENCE SURVEY RESULTS")
    print(f"-" * 50)
    print(f"Total datasets surveyed:            {n_total}")
    print(f"Datasets with |rho| > 0.5 pairs:    {n_corr} "
          f"({100*n_corr/n_total:.0f}%)")
    print(f"Datasets with instability (>10%):    {n_inst} "
          f"({100*n_inst/n_total:.0f}%)")
    print()

    print(f"Breakdown by feature count:")
    print(f"  {'Range':<14} {'N':>5} {'Corr':>6} {'%Corr':>7} {'%Inst':>7}")
    print(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*7} {'-'*7}")
    for label, subset in [("P < 20", low_p), ("20 <= P < 100", mid_p), ("P >= 100", high_p)]:
        t, c, pc, pi = _stats(subset)
        print(f"  {label:<14} {t:>5} {c:>6} {pc:>6.1f}% {pi:>6.1f}%")

    print()

    # Top 10 most unstable datasets
    unstable = sorted(
        [r for r in results if r["has_instability"]],
        key=lambda r: r["max_flip_rate"],
        reverse=True,
    )
    if unstable:
        print(f"Top unstable datasets (flip rate > 10%):")
        print(f"  {'Dataset':<35} {'P':>4} {'Corr':>5} {'MaxFlip':>8} {'MaxRho':>7}")
        print(f"  {'-'*35} {'-'*4} {'-'*5} {'-'*8} {'-'*7}")
        for r in unstable[:15]:
            print(f"  {r['dataset']:<35} {r['n_features']:>4} "
                  f"{r['n_corr_pairs']:>5} "
                  f"{r['max_flip_rate']:>7.1%} "
                  f"{r['max_rho']:>7.3f}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results_prevalence_openml.json"
    )

    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "n_datasets": n_total,
                "n_with_correlation": n_corr,
                "pct_with_correlation": round(100 * n_corr / n_total, 1),
                "n_with_instability": n_inst,
                "pct_with_instability": round(100 * n_inst / n_total, 1),
                "breakdown_by_features": {
                    "P_lt_20": {
                        "n": len(low_p),
                        "n_corr": sum(1 for r in low_p if r["has_correlated"]),
                        "n_inst": sum(1 for r in low_p if r["has_instability"]),
                    },
                    "P_20_to_99": {
                        "n": len(mid_p),
                        "n_corr": sum(1 for r in mid_p if r["has_correlated"]),
                        "n_inst": sum(1 for r in mid_p if r["has_instability"]),
                    },
                    "P_gte_100": {
                        "n": len(high_p),
                        "n_corr": sum(1 for r in high_p if r["has_correlated"]),
                        "n_inst": sum(1 for r in high_p if r["has_instability"]),
                    },
                },
            },
            "datasets": results,
        }, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
