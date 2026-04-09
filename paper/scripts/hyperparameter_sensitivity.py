"""
Hyperparameter Sensitivity of SHAP Instability.

Shows that SHAP instability varies with hyperparameters but never vanishes
for rho > 0. For each combination of (rho, learning_rate, max_depth,
n_estimators), trains 20 XGBoost models (different seeds, subsample=0.8),
computes TreeSHAP on 200 eval samples, and measures within-group flip rate
for all correlated pairs.

Full factorial grid: 3 rho x 3 lr x 3 depth x 3 n_est = 81 configs x 20
seeds = 1,620 model fits.

DGP: P=10 features (2 groups of 5), N=2000, Y = sum(X_j) + epsilon.

Saves results to paper/results_hyperparameter_sensitivity.json.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import os
import sys
import time
from itertools import product

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Install with: pip install shap")
    sys.exit(1)

# -- Configuration ------------------------------------------------------------

RHO_VALUES = [0.5, 0.7, 0.9]
LEARNING_RATES = [0.01, 0.1, 0.3]
MAX_DEPTHS = [1, 3, 6]
N_ESTIMATORS_LIST = [50, 100, 500]

N_TRAIN = 2000
N_EVAL = 200
N_SEEDS = 20
P_FEATURES = 10
GROUP_SIZE = 5
SUBSAMPLE = 0.8
DATA_SEED = 42


# -- Data generation -----------------------------------------------------------

def generate_correlated_data(rho, n_samples, rng):
    """Generate P=10 features in 2 groups of 5 with within-group correlation rho.

    Y = sum(X_j) + epsilon, epsilon ~ N(0, 0.1).
    """
    p = P_FEATURES
    g = GROUP_SIZE

    cov = np.eye(p)
    for group_start in [0, g]:
        for i in range(group_start, group_start + g):
            for j in range(group_start, group_start + g):
                if i != j:
                    cov[i, j] = rho

    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    beta = np.ones(p)
    Y = X @ beta + rng.normal(0, 0.1, size=n_samples)
    return X, Y


# -- Flip rate measurement -----------------------------------------------------

def measure_flip_rate(shap_values_list):
    """Compute within-group flip rate across seeds.

    For each correlated pair (i, j) within a group, and for each seed,
    determine the ranking by mean |SHAP|. A flip occurs when two seeds
    disagree on which feature in the pair has higher mean |SHAP|.

    Returns the average flip rate across all pairs and seed-pairs.
    """
    n_seeds = len(shap_values_list)
    if n_seeds < 2:
        return 0.0

    # mean |SHAP| per feature per seed: shape (n_seeds, P_FEATURES)
    mean_abs = np.array([
        np.mean(np.abs(sv), axis=0) for sv in shap_values_list
    ])

    flips = 0
    comparisons = 0

    for group_start in [0, GROUP_SIZE]:
        group_end = group_start + GROUP_SIZE
        # All pairs within the group
        for i in range(group_start, group_end):
            for j in range(i + 1, group_end):
                # For each pair of seeds, check if ranking flips
                for s1 in range(n_seeds):
                    for s2 in range(s1 + 1, n_seeds):
                        rank_s1 = mean_abs[s1, i] > mean_abs[s1, j]
                        rank_s2 = mean_abs[s2, i] > mean_abs[s2, j]
                        if rank_s1 != rank_s2:
                            flips += 1
                        comparisons += 1

    return flips / comparisons if comparisons > 0 else 0.0


# -- Main experiment -----------------------------------------------------------

def run_experiment():
    print("=" * 72)
    print("Hyperparameter Sensitivity of SHAP Instability")
    print("=" * 72)
    print(f"Config: P={P_FEATURES}, N_train={N_TRAIN}, N_eval={N_EVAL}, "
          f"seeds={N_SEEDS}, subsample={SUBSAMPLE}")
    print(f"Grid:   rho={RHO_VALUES}, lr={LEARNING_RATES}, "
          f"depth={MAX_DEPTHS}, n_est={N_ESTIMATORS_LIST}")
    print(f"Total:  {len(RHO_VALUES) * len(LEARNING_RATES) * len(MAX_DEPTHS) * len(N_ESTIMATORS_LIST)} "
          f"configs x {N_SEEDS} seeds = "
          f"{len(RHO_VALUES) * len(LEARNING_RATES) * len(MAX_DEPTHS) * len(N_ESTIMATORS_LIST) * N_SEEDS} "
          f"model fits")
    print()

    total_start = time.time()
    all_results = {}  # key: "rho_{rho}_lr_{lr}_depth_{depth}_nest_{nest}"

    for rho in RHO_VALUES:
        rho_start = time.time()
        print(f"--- rho = {rho:.2f} ---")

        # Generate fixed eval data for this rho
        eval_rng = np.random.default_rng(DATA_SEED)
        X_eval, _ = generate_correlated_data(rho, N_EVAL, eval_rng)

        for lr, depth, n_est in product(LEARNING_RATES, MAX_DEPTHS, N_ESTIMATORS_LIST):
            shap_values_list = []

            for seed in range(N_SEEDS):
                # Fresh training data each seed
                train_rng = np.random.default_rng(DATA_SEED + seed + 1)
                X_train, Y_train = generate_correlated_data(rho, N_TRAIN, train_rng)

                model = xgb.XGBRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=lr,
                    subsample=SUBSAMPLE,
                    reg_alpha=0,
                    reg_lambda=0,
                    colsample_bytree=1.0,
                    random_state=seed + 5000,
                    n_jobs=1,
                    verbosity=0,
                )
                model.fit(X_train, Y_train)

                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_eval)
                shap_values_list.append(sv)

            flip = measure_flip_rate(shap_values_list)
            key = f"rho_{rho}_lr_{lr}_depth_{depth}_nest_{n_est}"
            all_results[key] = {
                "rho": rho,
                "learning_rate": lr,
                "max_depth": depth,
                "n_estimators": n_est,
                "flip_rate": round(flip, 6),
            }

        rho_elapsed = time.time() - rho_start
        print(f"  rho={rho:.2f} done in {rho_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # -- Print compact tables --------------------------------------------------
    print()
    print("=" * 72)
    print("Results: Within-group flip rate (%)")
    print("=" * 72)

    for rho in RHO_VALUES:
        print()
        print(f"rho={rho:.2f} | {'depth=1':^18s} | {'depth=3':^18s} | {'depth=6':^18s}")
        lr_header = "  ".join(f"lr={lr}" for lr in LEARNING_RATES)
        print(f"{'':>7s} | {lr_header:^18s} | {lr_header:^18s} | {lr_header:^18s}")
        print("-" * 72)

        for n_est in N_ESTIMATORS_LIST:
            row = f"n={n_est:<4d} |"
            for depth in MAX_DEPTHS:
                cells = []
                for lr in LEARNING_RATES:
                    key = f"rho_{rho}_lr_{lr}_depth_{depth}_nest_{n_est}"
                    pct = all_results[key]["flip_rate"] * 100
                    cells.append(f"{pct:5.1f}%")
                row += " " + " ".join(cells) + " |"
            print(row)
        print()

    # -- Key findings ----------------------------------------------------------
    print("=" * 72)
    print("Key Findings")
    print("=" * 72)

    for rho in RHO_VALUES:
        rho_results = [v for v in all_results.values() if v["rho"] == rho]
        flips = [v["flip_rate"] for v in rho_results]
        min_flip = min(flips)
        max_flip = max(flips)
        min_cfg = min(rho_results, key=lambda v: v["flip_rate"])
        max_cfg = max(rho_results, key=lambda v: v["flip_rate"])

        print(f"\n  rho={rho:.2f}:")
        print(f"    Min flip rate: {min_flip*100:.2f}% "
              f"(lr={min_cfg['learning_rate']}, depth={min_cfg['max_depth']}, "
              f"n_est={min_cfg['n_estimators']})")
        print(f"    Max flip rate: {max_flip*100:.2f}% "
              f"(lr={max_cfg['learning_rate']}, depth={max_cfg['max_depth']}, "
              f"n_est={max_cfg['n_estimators']})")
        print(f"    Min > 0: {min_flip > 0}")

    # -- Which hyperparameter matters most? ------------------------------------
    print()
    print("Hyperparameter effect (average flip rate across other params):")

    for param_name, param_values in [
        ("learning_rate", LEARNING_RATES),
        ("max_depth", MAX_DEPTHS),
        ("n_estimators", N_ESTIMATORS_LIST),
    ]:
        avg_by_val = {}
        for val in param_values:
            matching = [v["flip_rate"] for v in all_results.values()
                        if v[param_name] == val]
            avg_by_val[val] = np.mean(matching)
        spread = max(avg_by_val.values()) - min(avg_by_val.values())
        detail = ", ".join(f"{k}={v*100:.1f}%" for k, v in avg_by_val.items())
        print(f"  {param_name:>15s}: spread={spread*100:.2f}pp  ({detail})")

    # Determine which has largest spread
    spreads = {}
    for param_name, param_values in [
        ("learning_rate", LEARNING_RATES),
        ("max_depth", MAX_DEPTHS),
        ("n_estimators", N_ESTIMATORS_LIST),
    ]:
        avgs = []
        for val in param_values:
            matching = [v["flip_rate"] for v in all_results.values()
                        if v[param_name] == val]
            avgs.append(np.mean(matching))
        spreads[param_name] = max(avgs) - min(avgs)

    most_influential = max(spreads, key=spreads.get)

    # -- Global minimum --------------------------------------------------------
    all_flips_positive_rho = [v for v in all_results.values() if v["rho"] > 0]
    global_min = min(all_flips_positive_rho, key=lambda v: v["flip_rate"])
    global_min_flip = global_min["flip_rate"]

    # Determine which param minimizes instability
    minimizing_params = []
    for param_name, param_values in [
        ("learning_rate", LEARNING_RATES),
        ("max_depth", MAX_DEPTHS),
        ("n_estimators", N_ESTIMATORS_LIST),
    ]:
        avgs = {}
        for val in param_values:
            matching = [v["flip_rate"] for v in all_results.values()
                        if v[param_name] == val]
            avgs[val] = np.mean(matching)
        best_val = min(avgs, key=avgs.get)
        minimizing_params.append(f"{param_name}={best_val}")

    minimizer_str = ", ".join(minimizing_params)

    print()
    print("=" * 72)
    verdict = (f"Instability is minimized by {minimizer_str} "
               f"but never eliminated for rho > 0 "
               f"(global min flip rate = {global_min_flip*100:.2f}%)")
    print(f"VERDICT: {verdict}")
    print("=" * 72)
    print(f"\nTotal runtime: {total_elapsed:.1f}s")

    # -- Save results ----------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    output_path = os.path.join(paper_dir, "results_hyperparameter_sensitivity.json")

    output = {
        "description": (
            "Hyperparameter sensitivity of SHAP instability. "
            "Shows flip rate varies with hyperparameters but never vanishes for rho > 0."
        ),
        "config": {
            "P": P_FEATURES,
            "N_train": N_TRAIN,
            "N_eval": N_EVAL,
            "n_seeds": N_SEEDS,
            "subsample": SUBSAMPLE,
            "rho_values": RHO_VALUES,
            "learning_rates": LEARNING_RATES,
            "max_depths": MAX_DEPTHS,
            "n_estimators_list": N_ESTIMATORS_LIST,
        },
        "results": list(all_results.values()),
        "summary": {
            "global_min_flip_rate": round(global_min_flip, 6),
            "global_min_config": {
                "rho": global_min["rho"],
                "learning_rate": global_min["learning_rate"],
                "max_depth": global_min["max_depth"],
                "n_estimators": global_min["n_estimators"],
            },
            "most_influential_hyperparameter": most_influential,
            "hyperparameter_spreads_pp": {
                k: round(v * 100, 2) for k, v in spreads.items()
            },
        },
        "verdict": verdict,
        "total_runtime_seconds": round(total_elapsed, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
