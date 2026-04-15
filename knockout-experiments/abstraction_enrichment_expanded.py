#!/usr/bin/env python3
"""
Abstraction Enrichment Expanded: Test bilemma enrichment prediction across 5+ datasets.

PREDICTION: Merging fine-grained classes into superclasses (= enrichment = adding neutral
element) reduces explanation instability, as predicted by the bilemma characterization.

Datasets with natural class hierarchies:
1. Wine (3 classes → 2 superclasses) [BASELINE, already confirmed]
2. Iris (3 species: setosa/versicolor/virginica → 2 genera: setosa vs versicolor+virginica)
3. MNIST digits (10 digits → 5 groups: {0,1} {2,3} {4,5} {6,7} {8,9} → 2: even/odd)
4. Optical digits (10 digits, same hierarchy as MNIST)
5. Covertype (7 types → 3 superclasses → 2)
6. Fashion-MNIST (10 categories → 4 superclasses → 2: clothing vs accessories)

For each dataset:
- Train 30 bootstrap XGBoost models at each abstraction level
- Compute pairwise flip rate at each level
- Prediction: flip rate monotonically decreases with abstraction (enrichment)

Uses feature_importances_ (gain-based) for speed.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.datasets import load_wine, load_iris, load_digits, fetch_openml, fetch_covtype
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / 'figures' / 'abstraction_enrichment'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS_CAL = list(range(42, 72))    # 30 calibration models
SEEDS_VAL = list(range(142, 172))  # 30 validation models


def train_bootstrap_models(X, y, seeds, n_classes):
    """Train XGBoost models with bootstrap resampling."""
    n_models = len(seeds)
    P = X.shape[1]
    imp = np.zeros((n_models, P))

    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=len(X), replace=True)
        params = dict(
            n_estimators=100, max_depth=4, random_state=seed,
            verbosity=0, tree_method='hist'
        )
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


def compute_flip_rate(imp):
    """Compute mean pairwise flip rate."""
    n_models, P = imp.shape
    if P > 50:
        # Subsample pairs for speed
        rng = np.random.RandomState(999)
        all_pairs = list(combinations(range(P), 2))
        pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=min(1000, len(all_pairs)), replace=False)]
    else:
        pairs = list(combinations(range(P), 2))

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

    return np.array(flip_rates)


def run_abstraction_experiment(name, X, y_levels, level_names, level_nclasses):
    """
    Run enrichment experiment across multiple abstraction levels.

    y_levels: list of label arrays, from fine-grained (level 0) to coarse (level N)
    level_names: list of descriptive names for each level
    level_nclasses: list of number of classes at each level
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"  X shape: {X.shape}")
    print(f"  Levels: {len(y_levels)}")
    print(f"{'='*70}")
    t0 = time.time()

    results = {"dataset": name, "n_features": int(X.shape[1]), "n_models": len(SEEDS_CAL), "levels": {}}

    for level_idx, (y, lname, nc) in enumerate(zip(y_levels, level_names, level_nclasses)):
        print(f"\n  Level {level_idx}: {lname} ({nc} classes)")
        unique, counts = np.unique(y, return_counts=True)
        print(f"    Classes: {unique.tolist()}, counts: {counts.tolist()}")

        # Train on calibration seeds
        imp_cal = train_bootstrap_models(X, y, SEEDS_CAL, nc)
        flip_rates_cal = compute_flip_rate(imp_cal)

        # Train on validation seeds (OOS)
        imp_val = train_bootstrap_models(X, y, SEEDS_VAL, nc)
        flip_rates_val = compute_flip_rate(imp_val)

        mean_cal = float(np.mean(flip_rates_cal))
        mean_val = float(np.mean(flip_rates_val))

        print(f"    Calibration flip rate: {mean_cal:.4f}")
        print(f"    Validation flip rate:  {mean_val:.4f}")

        results["levels"][f"level_{level_idx}"] = {
            "name": lname,
            "n_classes": nc,
            "calibration_flip_rate": mean_cal,
            "validation_flip_rate": mean_val,
            "cal_median": float(np.median(flip_rates_cal)),
            "cal_std": float(np.std(flip_rates_cal)),
            "val_median": float(np.median(flip_rates_val)),
            "val_std": float(np.std(flip_rates_val)),
        }

    # Check monotonic decrease
    cal_rates = [results["levels"][f"level_{i}"]["calibration_flip_rate"] for i in range(len(y_levels))]
    val_rates = [results["levels"][f"level_{i}"]["validation_flip_rate"] for i in range(len(y_levels))]

    # Enrichment = fine → coarse reduces instability
    cal_reduction = cal_rates[0] - cal_rates[-1]
    val_reduction = val_rates[0] - val_rates[-1]
    cal_monotonic = all(cal_rates[i] >= cal_rates[i+1] for i in range(len(cal_rates)-1))
    val_monotonic = all(val_rates[i] >= val_rates[i+1] for i in range(len(val_rates)-1))

    results["cal_reduction_pp"] = round(cal_reduction * 100, 1)
    results["val_reduction_pp"] = round(val_reduction * 100, 1)
    results["cal_monotonic"] = cal_monotonic
    results["val_monotonic"] = val_monotonic
    results["enrichment_confirmed"] = val_reduction > 0.02  # At least 2pp reduction on validation

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n  RESULT: cal reduction = {results['cal_reduction_pp']}pp, "
          f"val reduction = {results['val_reduction_pp']}pp")
    print(f"  Cal monotonic: {cal_monotonic}, Val monotonic: {val_monotonic}")
    print(f"  Enrichment confirmed (OOS): {results['enrichment_confirmed']}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return results


# =====================================================================
# Dataset Definitions
# =====================================================================

def dataset_wine():
    wine = load_wine()
    X = wine.data
    y0 = wine.target  # 3 classes: 0, 1, 2

    # Level 1: merge class 0+1 vs class 2 (partial enrichment)
    y1 = np.where(y0 == 2, 1, 0)

    return "Wine", X, [y0, y1], \
        ["3-class (cultivar 1/2/3)", "Binary (cultivar 1+2 vs 3)"], [3, 2]


def dataset_iris():
    iris = load_iris()
    X = iris.data
    y0 = iris.target  # 3 classes: setosa(0), versicolor(1), virginica(2)

    # Level 1: merge versicolor+virginica (same genus Iris) vs setosa
    # Biologically: setosa is easily separable; versicolor/virginica are famously confusable
    y1 = np.where(y0 == 0, 0, 1)

    return "Iris", X, [y0, y1], \
        ["3-species (setosa/versicolor/virginica)", "Binary (setosa vs non-setosa)"], [3, 2]


def dataset_digits():
    digits = load_digits()
    X = digits.data  # 8x8 = 64 features
    y0 = digits.target  # 10 classes: 0-9

    # Level 1: 5 groups of visually similar digits
    # {0,6}, {1,7}, {2,3}, {4,9}, {5,8}
    group_map = {0: 0, 6: 0, 1: 1, 7: 1, 2: 2, 3: 2, 4: 3, 9: 3, 5: 4, 8: 4}
    y1 = np.array([group_map[d] for d in y0])

    # Level 2: even vs odd
    y2 = y0 % 2

    return "Digits (8x8)", X, [y0, y1, y2], \
        ["10-class (digits 0-9)", "5-class (visual pairs)", "Binary (even/odd)"], [10, 5, 2]


def dataset_optical_recognition():
    """Optical recognition of handwritten digits from UCI (larger than sklearn digits)."""
    try:
        data = fetch_openml('optdigits', version=1, return_X_y=True, as_frame=False, parser='auto')
        X, y_str = data
        y0 = y_str.astype(int)
    except Exception:
        # Fallback to sklearn digits
        return None

    # Same hierarchy as digits
    group_map = {0: 0, 6: 0, 1: 1, 7: 1, 2: 2, 3: 2, 4: 3, 9: 3, 5: 4, 8: 4}
    y1 = np.array([group_map[d] for d in y0])
    y2 = y0 % 2

    return "OptDigits (UCI)", X, [y0, y1, y2], \
        ["10-class (digits 0-9)", "5-class (visual pairs)", "Binary (even/odd)"], [10, 5, 2]


def dataset_covertype():
    """Forest cover type: 7 types → 3 superclasses → 2."""
    try:
        cov = fetch_covtype()
        X_full = cov.data
        y_full = cov.target  # 1-7

        # Subsample for speed (581k samples is too many)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_full), size=5000, replace=False)
        X = X_full[idx]
        y0 = y_full[idx] - 1  # Make 0-indexed

        # Ecological hierarchy:
        # Spruce/Fir types: 0(Spruce-Fir), 1(Lodgepole Pine) → conifer
        # Mixed: 2(Ponderosa Pine), 3(Cottonwood/Willow), 5(Douglas-fir) → mixed
        # Sparse: 4(Aspen), 6(Krummholz) → deciduous/sparse
        superclass_map = {0: 0, 1: 0, 2: 1, 3: 1, 5: 1, 4: 2, 6: 2}
        y1 = np.array([superclass_map.get(c, 1) for c in y0])

        # Binary: conifer vs non-conifer
        y2 = np.where(y1 == 0, 0, 1)

        return "Covertype", X, [y0, y1, y2], \
            ["7-class (forest types)", "3-class (conifer/mixed/deciduous)", "Binary (conifer/non-conifer)"], \
            [7, 3, 2]
    except Exception as e:
        print(f"  Covertype failed: {e}")
        return None


def dataset_fashion_mnist():
    """Fashion-MNIST: 10 categories → 4 superclasses → 2."""
    try:
        X_raw, y_str = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
        y0_full = y_str.astype(int)

        # Subsample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_raw), size=5000, replace=False)
        X = X_raw[idx].astype(np.float64)
        y0 = y0_full[idx]

        # Fashion hierarchy:
        # Tops: 0(T-shirt), 2(Pullover), 4(Coat), 6(Shirt) → clothing_top
        # Bottoms: 1(Trouser) → clothing_bottom
        # Footwear: 5(Sandal), 7(Sneaker), 9(Ankle boot) → footwear
        # Accessories: 3(Dress), 8(Bag) → accessories
        superclass_map = {0: 0, 2: 0, 4: 0, 6: 0, 1: 1, 5: 2, 7: 2, 9: 2, 3: 3, 8: 3}
        y1 = np.array([superclass_map[c] for c in y0])

        # Binary: clothing (tops+bottoms) vs non-clothing (footwear+accessories)
        y2 = np.where(y1 <= 1, 0, 1)

        return "Fashion-MNIST", X, [y0, y1, y2], \
            ["10-class (items)", "4-class (tops/bottoms/feet/acc)", "Binary (clothing/non-clothing)"], \
            [10, 4, 2]
    except Exception as e:
        print(f"  Fashion-MNIST failed: {e}")
        return None


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("Abstraction Enrichment Expanded: 6 Datasets")
    print("=" * 70)
    print("PREDICTION: Enrichment (class merging) reduces explanation instability")
    print("This tests the bilemma's enrichment mechanism on real data.\n")

    all_results = {}
    dataset_fns = [
        dataset_wine,
        dataset_iris,
        dataset_digits,
        dataset_optical_recognition,
        dataset_covertype,
        dataset_fashion_mnist,
    ]

    for fn in dataset_fns:
        try:
            result = fn()
            if result is None:
                continue
            name, X, y_levels, level_names, level_nclasses = result
            r = run_abstraction_experiment(name, X, y_levels, level_names, level_nclasses)
            all_results[name] = r
        except Exception as e:
            print(f"\n  ERROR on {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Enrichment Across Datasets")
    print("=" * 70)

    confirmed = 0
    total = len(all_results)

    summary_table = []
    for name, r in all_results.items():
        levels = r["levels"]
        n_levels = len(levels)
        fine_rate = levels["level_0"]["validation_flip_rate"]
        coarse_rate = levels[f"level_{n_levels-1}"]["validation_flip_rate"]
        reduction = fine_rate - coarse_rate

        status = "CONFIRMED" if r["enrichment_confirmed"] else "NOT CONFIRMED"
        if r["enrichment_confirmed"]:
            confirmed += 1

        summary_table.append({
            "dataset": name,
            "n_features": r["n_features"],
            "fine_flip": fine_rate,
            "coarse_flip": coarse_rate,
            "reduction_pp": round(reduction * 100, 1),
            "monotonic": r["val_monotonic"],
            "confirmed": r["enrichment_confirmed"],
        })

        print(f"\n  {name}: {status}")
        print(f"    Fine → Coarse: {fine_rate:.4f} → {coarse_rate:.4f} "
              f"({reduction*100:+.1f}pp)")
        print(f"    Monotonic: {r['val_monotonic']}")

    print(f"\n  {'='*40}")
    print(f"  CONFIRMED: {confirmed}/{total} datasets")
    print(f"  {'='*40}")

    # =====================================================================
    # Save results
    # =====================================================================
    output = {
        "experiment": "abstraction_enrichment_expanded",
        "description": "Enrichment (class merging) reduces explanation instability across multiple datasets",
        "prediction": "Bilemma: merging fine-grained classes into superclasses adds neutral element, restoring stability",
        "n_datasets": total,
        "n_confirmed": confirmed,
        "confirmation_rate": confirmed / total if total > 0 else 0,
        "summary_table": summary_table,
        "per_dataset": all_results,
    }

    json_path = OUT_DIR / 'results_abstraction_enrichment_expanded.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # =====================================================================
    # Figures
    # =====================================================================
    pdf_path = FIG_DIR / 'abstraction_enrichment_expanded.pdf'
    print(f"Saving figures to {pdf_path}...")

    with PdfPages(str(pdf_path)) as pdf:
        # Figure 1: Summary bar chart — reduction by dataset
        fig, ax = plt.subplots(figsize=(10, 6))
        datasets = [s["dataset"] for s in summary_table]
        reductions = [s["reduction_pp"] for s in summary_table]
        colors = ['#27ae60' if s["confirmed"] else '#e74c3c' for s in summary_table]

        bars = ax.barh(datasets, reductions, color=colors, edgecolor='black', alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.axvline(2, color='gray', linestyle='--', linewidth=0.8, label='2pp threshold')
        ax.set_xlabel('Flip Rate Reduction (pp)')
        ax.set_title('Enrichment Effect: Fine-Grained → Coarse Classification\n'
                      f'(OOS validation, {confirmed}/{total} confirmed)')
        ax.legend()

        for bar, val in zip(bars, reductions):
            x_pos = bar.get_width() + 0.3 if bar.get_width() > 0 else bar.get_width() - 0.3
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}pp', ha='left' if val > 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Figure 2: Per-dataset level progression
        n_datasets = len(all_results)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)

        for idx, (name, r) in enumerate(all_results.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            levels = r["levels"]
            n_lev = len(levels)
            level_names_plot = [levels[f"level_{i}"]["name"] for i in range(n_lev)]
            cal_rates = [levels[f"level_{i}"]["calibration_flip_rate"] for i in range(n_lev)]
            val_rates = [levels[f"level_{i}"]["validation_flip_rate"] for i in range(n_lev)]

            x = np.arange(n_lev)
            w = 0.35
            ax.bar(x - w/2, cal_rates, w, label='Calibration', color='steelblue', alpha=0.8)
            ax.bar(x + w/2, val_rates, w, label='Validation', color='coral', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{i}" for i in range(n_lev)], fontsize=9)
            ax.set_ylabel('Flip Rate')
            ax.set_title(f'{name}', fontsize=11)
            ax.legend(fontsize=8)
            ax.set_ylim(0, max(max(cal_rates), max(val_rates)) * 1.3 + 0.01)

        # Hide empty subplots
        for idx in range(len(all_results), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle('Flip Rate by Abstraction Level (Each Dataset)', fontsize=13, y=1.02)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Saved: {pdf_path}")
    print("\nDone.")
