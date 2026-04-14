"""
Phase 2 Knockout: Noether Counting + SAGE on 5 Real Datasets with TreeSHAP
===========================================================================
Validates bimodal flip-rate gap on real-world datasets using actual TreeSHAP
importance values (mean |SHAP|).

Datasets:
  1. Breast Cancer Wisconsin (30 features, classification)
  2. Wine (13 features, classification)
  3. Heart Disease (13 features, classification)
  4. California Housing (8 features, regression)
  5. Diabetes Pima (8 features, classification)

KNOCKOUT criterion: Bimodal gap > 20pp on >= 3 of 5 datasets
"""

import sys, json, time, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ===================================================================
# Configuration
# ===================================================================
N_MODELS = 30
N_TEST_SHAP = 200      # test points for SHAP computation
CLUSTER_THRESHOLD = 0.30
BASE_SEED = 42
N_PERMUTATIONS = 5000
XGB_PARAMS = dict(n_estimators=100, max_depth=4, verbosity=0,
                  tree_method="hist")


# ===================================================================
# Dataset loaders
# ===================================================================

def load_heart_disease():
    """Load Heart Disease via fetch_openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("heart-statlog", version=1, as_frame=False,
                        parser="auto")
    X, y = data.data, data.target
    # Encode target to 0/1 if needed
    if y.dtype == object or y.dtype.kind in ('U', 'S'):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    return X, y.astype(float)


def load_diabetes_pima():
    """Load Pima Diabetes via fetch_openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("diabetes", version=1, as_frame=False,
                        parser="auto")
    X, y = data.data, data.target
    if y.dtype == object or y.dtype.kind in ('U', 'S'):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    return X, y.astype(float)


DATASETS = {
    "Breast Cancer": {
        "loader": lambda: (load_breast_cancer().data, load_breast_cancer().target.astype(float)),
        "task": "classification",
    },
    "Wine": {
        "loader": lambda: (load_wine().data, load_wine().target.astype(float)),
        "task": "classification",
    },
    "Heart Disease": {
        "loader": load_heart_disease,
        "task": "classification",
    },
    "California Housing": {
        "loader": lambda: (fetch_california_housing().data,
                           fetch_california_housing().target),
        "task": "regression",
    },
    "Diabetes Pima": {
        "loader": load_diabetes_pima,
        "task": "classification",
    },
}


# ===================================================================
# Core computation
# ===================================================================

def run_dataset(name, cfg):
    """Run full Noether + SAGE analysis on one dataset."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"{'='*60}")

    # Load and split
    X, y = cfg["loader"]()
    task = cfg["task"]
    P = X.shape[1]
    print(f"  Samples={X.shape[0]}, Features={P}, Task={task}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=BASE_SEED
    )

    # Ensure enough test points
    n_test = min(N_TEST_SHAP, X_test.shape[0])
    X_test_shap = X_test[:n_test]

    # Train N_MODELS bootstrap models and compute TreeSHAP importance
    importance_matrix = np.zeros((N_MODELS, P))

    for i in range(N_MODELS):
        rng = np.random.RandomState(BASE_SEED + i)
        idx = rng.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
        X_b, y_b = X_train[idx], y_train[idx]

        if task == "classification":
            model = xgb.XGBClassifier(
                random_state=BASE_SEED + i, use_label_encoder=False,
                eval_metric="logloss", **XGB_PARAMS
            )
        else:
            model = xgb.XGBRegressor(
                random_state=BASE_SEED + i, **XGB_PARAMS
            )

        model.fit(X_b, y_b)

        # TreeSHAP
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test_shap)

        if isinstance(sv, list):
            # Multi-class: list of arrays, one per class
            importance = np.mean(
                [np.mean(np.abs(s), axis=0) for s in sv], axis=0
            )
        elif sv.ndim == 3:
            # Multi-class returned as 3D array (n_samples, n_features, n_classes)
            importance = np.mean(np.abs(sv), axis=(0, 2))
        else:
            importance = np.mean(np.abs(sv), axis=0)

        importance_matrix[i] = importance
        if (i + 1) % 10 == 0:
            print(f"    Models trained: {i+1}/{N_MODELS}")

    # Compute pairwise flip rates
    flip_matrix = np.zeros((P, P))
    n_model_pairs = N_MODELS * (N_MODELS - 1) // 2

    for j in range(P):
        for k in range(j + 1, P):
            flips = 0
            total = 0
            for m1 in range(N_MODELS):
                for m2 in range(m1 + 1, N_MODELS):
                    rank_j_m1 = importance_matrix[m1, j]
                    rank_k_m1 = importance_matrix[m1, k]
                    rank_j_m2 = importance_matrix[m2, j]
                    rank_k_m2 = importance_matrix[m2, k]

                    sign1 = np.sign(rank_j_m1 - rank_k_m1)
                    sign2 = np.sign(rank_j_m2 - rank_k_m2)

                    if sign1 != 0 and sign2 != 0:
                        total += 1
                        if sign1 != sign2:
                            flips += 1

            flip_matrix[j, k] = flips / total if total > 0 else 0.0
            flip_matrix[k, j] = flip_matrix[j, k]

    # SAGE clustering (hierarchical, complete linkage)
    condensed = squareform(flip_matrix, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=CLUSTER_THRESHOLD, criterion="distance")
    n_groups = len(set(labels))

    # Group composition
    groups = {}
    for feat_idx, grp in enumerate(labels):
        grp = int(grp)
        if grp not in groups:
            groups[grp] = []
        groups[grp].append(feat_idx)

    print(f"  SAGE discovered {n_groups} groups")
    for g, feats in sorted(groups.items()):
        print(f"    Group {g}: {len(feats)} features — indices {feats[:8]}{'...' if len(feats)>8 else ''}")

    # Classify pairs as within-group vs between-group
    within_rates = []
    between_rates = []

    for j in range(P):
        for k in range(j + 1, P):
            rate = flip_matrix[j, k]
            if labels[j] == labels[k]:
                within_rates.append(rate)
            else:
                between_rates.append(rate)

    within_mean = np.mean(within_rates) if within_rates else 0.0
    between_mean = np.mean(between_rates) if between_rates else 0.0
    # For discovered groups: within-group = stable (low flip), between = unstable (high flip)
    # Bimodal gap = between_mean - within_mean (positive when separation exists)
    bimodal_gap = between_mean - within_mean

    print(f"  Within-group pairs: {len(within_rates)}, mean flip rate: {within_mean:.4f}")
    print(f"  Between-group pairs: {len(between_rates)}, mean flip rate: {between_mean:.4f}")
    print(f"  Bimodal gap: {bimodal_gap:.4f} ({bimodal_gap*100:.1f}pp)")

    # Permutation test
    all_rates = within_rates + between_rates
    all_labels_bool = [True] * len(within_rates) + [False] * len(between_rates)
    all_rates = np.array(all_rates)
    all_labels_bool = np.array(all_labels_bool)

    observed_diff = bimodal_gap
    rng_perm = np.random.RandomState(BASE_SEED)
    perm_diffs = np.zeros(N_PERMUTATIONS)

    for p_i in range(N_PERMUTATIONS):
        perm = rng_perm.permutation(len(all_labels_bool))
        perm_labels = all_labels_bool[perm]
        w_mean = np.mean(all_rates[perm_labels]) if np.any(perm_labels) else 0
        b_mean = np.mean(all_rates[~perm_labels]) if np.any(~perm_labels) else 0
        perm_diffs[p_i] = w_mean - b_mean

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # SAGE prediction
    eta = n_groups / P
    predicted_instability = 1.0 - eta

    # Observed instability: mean flip rate across ALL pairs
    all_flip_rates = []
    for j in range(P):
        for k in range(j + 1, P):
            all_flip_rates.append(flip_matrix[j, k])
    observed_instability = np.mean(all_flip_rates)

    elapsed = time.time() - t0
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  SAGE eta = g/P = {n_groups}/{P} = {eta:.4f}")
    print(f"  Predicted instability: {predicted_instability:.4f}")
    print(f"  Observed instability: {observed_instability:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "dataset": name,
        "P": P,
        "n_groups": n_groups,
        "group_composition": {str(g): feats for g, feats in groups.items()},
        "n_within_pairs": len(within_rates),
        "n_between_pairs": len(between_rates),
        "within_mean": round(within_mean, 4),
        "between_mean": round(between_mean, 4),
        "bimodal_gap_pp": round(bimodal_gap * 100, 1),
        "p_value": round(p_value, 4),
        "eta": round(eta, 4),
        "predicted_instability": round(predicted_instability, 4),
        "observed_instability": round(observed_instability, 4),
        "elapsed_s": round(elapsed, 1),
        "within_rates": [round(r, 4) for r in within_rates],
        "between_rates": [round(r, 4) for r in between_rates],
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("Phase 2 Knockout: Noether Counting + SAGE on Real Datasets")
    print("=" * 60)
    t_start = time.time()

    all_results = []
    sage_results = []

    for name, cfg in DATASETS.items():
        try:
            result = run_dataset(name, cfg)
            all_results.append(result)
            sage_results.append({
                "dataset": name,
                "P": result["P"],
                "n_groups": result["n_groups"],
                "eta": result["eta"],
                "predicted_instability": result["predicted_instability"],
                "observed_instability": result["observed_instability"],
            })
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Dataset':<22} {'P':>3} {'g':>3} {'Within':>8} {'Between':>8} "
          f"{'Gap(pp)':>8} {'p-val':>7} {'Pred':>6} {'Obs':>6}")
    print("-" * 80)

    n_knockout = 0
    for r in all_results:
        print(f"  {r['dataset']:<22} {r['P']:>3} {r['n_groups']:>3} "
              f"{r['within_mean']:>8.4f} {r['between_mean']:>8.4f} "
              f"{r['bimodal_gap_pp']:>8.1f} {r['p_value']:>7.4f} "
              f"{r['predicted_instability']:>6.4f} {r['observed_instability']:>6.4f}")
        if r['bimodal_gap_pp'] > 20.0:
            n_knockout += 1

    print("-" * 80)

    # SAGE R²
    if len(sage_results) >= 2:
        preds = [s["predicted_instability"] for s in sage_results]
        obs = [s["observed_instability"] for s in sage_results]
        ss_res = sum((p - o) ** 2 for p, o in zip(preds, obs))
        ss_tot = sum((o - np.mean(obs)) ** 2 for o in obs)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"\n  SAGE prediction R² = {r2:.4f}")
    else:
        r2 = float("nan")

    # Verdict
    print(f"\n  Datasets with bimodal gap > 20pp: {n_knockout} / {len(all_results)}")
    if n_knockout >= 3:
        verdict = "KNOCKOUT"
    elif n_knockout >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"  VERDICT: {verdict}")
    print(f"\n  Total time: {time.time() - t_start:.1f}s")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    # Strip large arrays for JSON
    json_results = []
    for r in all_results:
        rj = {k: v for k, v in r.items() if k not in ("within_rates", "between_rates")}
        json_results.append(rj)

    noether_out = {
        "experiment": "Phase 2 Knockout: Noether + SAGE on Real Datasets",
        "n_models": N_MODELS,
        "n_test_shap": N_TEST_SHAP,
        "cluster_threshold": CLUSTER_THRESHOLD,
        "n_permutations": N_PERMUTATIONS,
        "datasets": json_results,
        "n_knockout": n_knockout,
        "verdict": verdict,
    }
    with open(OUT_DIR / "results_noether_real.json", "w") as f:
        json.dump(noether_out, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'results_noether_real.json'}")

    sage_out = {
        "experiment": "SAGE predictions on real datasets",
        "sage_results": sage_results,
        "r_squared": round(r2, 4) if not np.isnan(r2) else None,
    }
    with open(OUT_DIR / "results_sage_real.json", "w") as f:
        json.dump(sage_out, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'results_sage_real.json'}")

    # ---------------------------------------------------------------
    # Multi-panel figure
    # ---------------------------------------------------------------
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 4),
                             squeeze=False)
    axes = axes[0]

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        w_rates = r["within_rates"]
        b_rates = r["between_rates"]
        bins = np.linspace(0, 0.55, 25)

        ax.hist(w_rates, bins=bins, alpha=0.6, color="red",
                label=f"Within (n={len(w_rates)})", edgecolor="darkred")
        ax.hist(b_rates, bins=bins, alpha=0.6, color="blue",
                label=f"Between (n={len(b_rates)})", edgecolor="darkblue")
        ax.set_title(f"{r['dataset']}\ngap={r['bimodal_gap_pp']:.1f}pp, "
                     f"g={r['n_groups']}, p={r['p_value']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Flip Rate")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    fig.suptitle("Phase 2: Noether Bimodal Gap on Real Datasets (TreeSHAP)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = FIG_DIR / "noether_real.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ("KNOCKOUT", "PARTIAL") else 1)
