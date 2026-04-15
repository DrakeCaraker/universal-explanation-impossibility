#!/usr/bin/env python3
"""
Enrichment Statistical Tests (Task A)

For each of the 6 datasets in the enrichment experiment:
  - Approximate 95% CI on flip rate reduction (fine - coarse)
  - Uses Welch-style CI from stored per-pair statistics

For the balance-matched control (Fashion-MNIST):
  - Permutation p-value: fraction of random controls with flip <= semantic flip

Saves results to results_enrichment_statistical_tests.json
"""

import json
import math
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

BASE_DIR = Path(__file__).parent


def compute_reduction_ci(fine_mean, fine_std, coarse_mean, coarse_std, n_models=30):
    """
    Approximate 95% CI for (fine - coarse) flip rate reduction.

    Uses the per-pair std values from the JSON. Since these are std across
    feature pairs (not across models), we use them as population-level estimates.
    The n_pairs varies by dataset, but we use n_models as the effective sample
    size for the model-level uncertainty.

    CI: (fine - coarse) +/- 1.96 * sqrt(fine_std^2/n + coarse_std^2/n)
    """
    reduction = fine_mean - coarse_mean
    se = math.sqrt(fine_std**2 / n_models + coarse_std**2 / n_models)
    ci_low = reduction - 1.96 * se
    ci_high = reduction + 1.96 * se
    # z-test for reduction > 0
    if se > 0:
        z = reduction / se
        # one-sided p-value (is reduction significantly > 0?)
        from scipy import stats as sp_stats
        p_value = 1.0 - sp_stats.norm.cdf(z)
    else:
        p_value = 0.0 if reduction > 0 else 1.0
    return {
        "reduction_mean": round(reduction, 6),
        "reduction_se": round(se, 6),
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
        "z_statistic": round(z if se > 0 else float('inf'), 4),
        "p_value_one_sided": round(p_value, 6),
        "significant_at_05": p_value < 0.05
    }


def task_a_enrichment_tests():
    """Compute CIs and significance tests for enrichment experiment."""
    with open(BASE_DIR / "results_abstraction_enrichment_expanded.json") as f:
        enrichment = json.load(f)

    results = {}
    for row in enrichment["summary_table"]:
        ds = row["dataset"]
        detail = enrichment["per_dataset"][ds]
        n_models = detail["n_models"]

        # Get fine (level_0) and coarsest level stats
        levels = detail["levels"]
        level_keys = sorted(levels.keys())
        fine_level = levels[level_keys[0]]
        coarse_level = levels[level_keys[-1]]

        fine_mean = fine_level["validation_flip_rate"]
        fine_std = fine_level["val_std"]
        coarse_mean = coarse_level["validation_flip_rate"]
        coarse_std = coarse_level["val_std"]

        ci_result = compute_reduction_ci(
            fine_mean, fine_std, coarse_mean, coarse_std, n_models
        )

        results[ds] = {
            "fine_flip": round(fine_mean, 6),
            "coarse_flip": round(coarse_mean, 6),
            "n_models": n_models,
            "fine_level": fine_level["name"],
            "coarse_level": coarse_level["name"],
            **ci_result,
            "confirmed": row["confirmed"]
        }

    return results


def task_a_balanced_control():
    """Permutation test for balance-matched control."""
    with open(BASE_DIR / "results_abstraction_balanced_control.json") as f:
        control = json.load(f)

    results = {}
    for ds_name, ds_data in control.items():
        semantic_flip = ds_data["semantic_flip"]
        matched_flips = ds_data["matched_flips"]
        n_randoms = len(matched_flips)

        # How many random controls have flip <= semantic?
        n_below = sum(1 for f in matched_flips if f <= semantic_flip)
        # Permutation p-value: fraction of randoms that are as good or better
        p_value = (n_below + 1) / (n_randoms + 1)  # conservative (+1/+1)

        # Also compute: how extreme is semantic vs random distribution?
        mean_random = np.mean(matched_flips)
        std_random = np.std(matched_flips, ddof=1)
        if std_random > 0:
            z_score = (semantic_flip - mean_random) / std_random
        else:
            z_score = float('-inf') if semantic_flip < mean_random else 0.0

        results[ds_name] = {
            "semantic_flip": round(semantic_flip, 6),
            "mean_random_flip": round(float(mean_random), 6),
            "std_random_flip": round(float(std_random), 6),
            "n_random_controls": n_randoms,
            "n_randoms_beaten": ds_data["n_randoms_beaten"],
            "n_randoms_at_or_below_semantic": n_below,
            "permutation_p_value": round(p_value, 4),
            "z_score_vs_randoms": round(float(z_score), 4),
            "semantic_advantage_pp": ds_data["semantic_advantage_pp"],
            "semantic_better": ds_data["semantic_better"],
            "note": (
                f"With {n_randoms} random controls, minimum achievable "
                f"p-value is {1/(n_randoms+1):.4f} (conservative)."
            )
        }

    return results


def main():
    print("=" * 60)
    print("TASK A: Enrichment Statistical Tests")
    print("=" * 60)

    # Part 1: Enrichment CIs
    print("\n--- Part 1: Flip rate reduction CIs ---")
    enrichment_results = task_a_enrichment_tests()

    for ds, r in enrichment_results.items():
        sig = "***" if r["p_value_one_sided"] < 0.001 else (
            "**" if r["p_value_one_sided"] < 0.01 else (
            "*" if r["p_value_one_sided"] < 0.05 else "ns"))
        print(f"\n  {ds}:")
        print(f"    Fine flip:  {r['fine_flip']:.4f}  ({r['fine_level']})")
        print(f"    Coarse flip: {r['coarse_flip']:.4f}  ({r['coarse_level']})")
        print(f"    Reduction:  {r['reduction_mean']:.4f}  "
              f"95% CI [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]")
        print(f"    z = {r['z_statistic']:.3f}, p = {r['p_value_one_sided']:.6f} {sig}")
        print(f"    Confirmed: {r['confirmed']}")

    # Part 2: Balanced control
    print("\n--- Part 2: Balance-matched control ---")
    control_results = task_a_balanced_control()

    for ds, r in control_results.items():
        print(f"\n  {ds}:")
        print(f"    Semantic flip:    {r['semantic_flip']:.4f}")
        print(f"    Mean random flip: {r['mean_random_flip']:.4f} +/- {r['std_random_flip']:.4f}")
        print(f"    Randoms beaten:   {r['n_randoms_beaten']}/{r['n_random_controls']}")
        print(f"    Permutation p:    {r['permutation_p_value']:.4f}")
        print(f"    z-score:          {r['z_score_vs_randoms']:.3f}")
        print(f"    Semantic better:  {r['semantic_better']}")

    # Combine and save
    output = {
        "experiment": "enrichment_statistical_tests",
        "description": "Statistical significance tests for enrichment (class merging) experiment",
        "enrichment_reduction_tests": enrichment_results,
        "balanced_control_tests": control_results,
        "summary": {
            "n_datasets": len(enrichment_results),
            "n_significant_reductions": sum(
                1 for r in enrichment_results.values()
                if r["significant_at_05"]
            ),
            "n_confirmed": sum(
                1 for r in enrichment_results.values() if r["confirmed"]
            ),
            "all_confirmed_are_significant": all(
                r["significant_at_05"]
                for r in enrichment_results.values()
                if r["confirmed"]
            ),
            "fashion_mnist_semantic_p": control_results.get(
                "Fashion-MNIST", {}
            ).get("permutation_p_value"),
        }
    }

    out_path = BASE_DIR / "results_enrichment_statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
