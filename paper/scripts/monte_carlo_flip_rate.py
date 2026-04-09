"""
Monte Carlo Flip Rate Validation — Phi(-SNR) vs. simulation.

Validates the theoretical flip rate formula from FlipRate.lean (S31 Gaussian):
for a pair of symmetric features with correlation rho, the probability that the
minority feature receives higher attribution follows Phi(-SNR), where
    SNR = |E[phi_j - phi_k]| / SD(phi_j - phi_k).

For symmetric features under the Rashomon property, SNR -> 0 as rho -> 1,
so the flip rate -> Phi(0) = 0.5 (coin flip).

Experiment: for each rho, generate correlated Gaussian data (P=10, 2 groups
of 5), train 10,000 independent XGBoost stumps (max_depth=1), compute
TreeSHAP mean |SHAP| for features 0 and 1 (same group), and compare
the empirical flip rate against Phi(-SNR).

Saves results to paper/results_monte_carlo_flip.json.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import os
import sys
import time

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

from scipy.stats import norm

# ── Configuration ─────────────────────────────────────────────────────────────
RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
N_TRAIN = 2000          # training samples
N_EVAL = 200            # evaluation samples for SHAP
N_TRIALS = 10000        # independent models per rho
P_FEATURES = 10         # total features (2 groups of 5)
GROUP_SIZE = 5          # features per correlated group
N_ESTIMATORS = 50       # stumps per model
MAX_DEPTH = 1           # stumps — theory is exact for depth-1
SUBSAMPLE = 0.8         # XGBoost subsampling
DATA_SEED = 42          # fixed data generation seed
MAX_RUNTIME_MIN = 30    # reduce trials if projected runtime exceeds this

# ── Data generation ───────────────────────────────────────────────────────────

def generate_correlated_data(rho, n_samples, rng):
    """Generate P=10 features in 2 groups of 5 with within-group correlation rho.

    Y = sum(X_j) + epsilon, epsilon ~ N(0, 1).
    """
    p = P_FEATURES
    g = GROUP_SIZE

    # Build correlation matrix: block diagonal with rho within groups
    cov = np.eye(p)
    for group_start in [0, g]:
        for i in range(group_start, group_start + g):
            for j in range(group_start, group_start + g):
                if i != j:
                    cov[i, j] = rho

    # Generate multivariate normal
    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    Y = X.sum(axis=1) + rng.standard_normal(n_samples)
    return X, Y


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 72)
    print("Monte Carlo Flip Rate Validation: Phi(-SNR) vs. Simulation")
    print("=" * 72)
    print(f"Config: P={P_FEATURES}, N_train={N_TRAIN}, N_eval={N_EVAL}, "
          f"trials={N_TRIALS}")
    print(f"        n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"subsample={SUBSAMPLE}")
    print()

    # Fixed evaluation data (same across all trials for a given rho)
    results = []
    total_start = time.time()
    n_trials_actual = N_TRIALS

    for rho_idx, rho in enumerate(RHO_VALUES):
        rho_start = time.time()
        print(f"rho = {rho:.2f} ...", end=" ", flush=True)

        # Generate fixed eval data for this rho
        eval_rng = np.random.RandomState(DATA_SEED)
        X_eval, _ = generate_correlated_data(rho, N_EVAL, eval_rng)

        shap_diffs = []  # phi_0 - phi_1 for each trial
        feat0_wins = 0
        feat1_wins = 0

        for trial in range(n_trials_actual):
            # Fresh training data each trial
            train_rng = np.random.RandomState(DATA_SEED + trial + 1)
            X_train, Y_train = generate_correlated_data(rho, N_TRAIN, train_rng)

            # Train XGBoost stump ensemble
            model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                subsample=SUBSAMPLE,
                random_state=trial,
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X_train, Y_train)

            # TreeSHAP on eval set
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_eval)

            # Mean |SHAP| for features 0 and 1
            mean_abs_0 = np.mean(np.abs(shap_values[:, 0]))
            mean_abs_1 = np.mean(np.abs(shap_values[:, 1]))

            shap_diffs.append(mean_abs_0 - mean_abs_1)

            if mean_abs_0 > mean_abs_1:
                feat0_wins += 1
            else:
                feat1_wins += 1

            # After first rho, 20 trials: check runtime and reduce if needed
            if rho_idx == 0 and trial == 19:
                elapsed = time.time() - rho_start
                projected_total = (elapsed / 20) * n_trials_actual * len(RHO_VALUES)
                if projected_total > MAX_RUNTIME_MIN * 60:
                    n_trials_actual = 1000
                    print(f"\n  [Runtime projected {projected_total/60:.0f} min > "
                          f"{MAX_RUNTIME_MIN} min, reducing to {n_trials_actual} "
                          f"trials]")
                    print(f"  rho = {rho:.2f} ...", end=" ", flush=True)

        # Trim to actual trials run
        shap_diffs = shap_diffs[:n_trials_actual]
        total_trials = feat0_wins + feat1_wins
        if total_trials < n_trials_actual:
            total_trials = n_trials_actual

        # Empirical flip rate: minority side / total
        empirical_flip = min(feat0_wins, feat1_wins) / total_trials

        # Empirical SNR from SHAP difference distribution
        diffs_arr = np.array(shap_diffs)
        mean_diff = np.mean(diffs_arr)
        std_diff = np.std(diffs_arr, ddof=1)
        empirical_snr = abs(mean_diff) / std_diff if std_diff > 0 else 0.0

        # Theoretical flip rate
        theoretical_flip = norm.cdf(-empirical_snr)

        rho_elapsed = time.time() - rho_start
        print(f"done ({rho_elapsed:.1f}s) | flip={empirical_flip:.4f}, "
              f"Phi(-SNR)={theoretical_flip:.4f}")

        results.append({
            "rho": rho,
            "empirical_flip_rate": round(empirical_flip, 6),
            "theoretical_flip_rate": round(theoretical_flip, 6),
            "abs_difference": round(abs(empirical_flip - theoretical_flip), 6),
            "empirical_snr": round(empirical_snr, 4),
            "n_trials": total_trials,
            "feat0_wins": feat0_wins,
            "feat1_wins": feat1_wins,
            "mean_shap_diff": round(float(mean_diff), 6),
            "std_shap_diff": round(float(std_diff), 6),
        })

    total_elapsed = time.time() - total_start

    # ── Print comparison table ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    print(f"{'rho':>6} | {'Empirical flip':>14} | {'Phi(-SNR)':>10} | "
          f"{'|Diff|':>8} | {'SNR':>8} | {'Trials':>6}")
    print("-" * 72)
    max_diff = 0.0
    for r in results:
        max_diff = max(max_diff, r["abs_difference"])
        print(f"{r['rho']:6.2f} | {r['empirical_flip_rate']:14.4f} | "
              f"{r['theoretical_flip_rate']:10.4f} | "
              f"{r['abs_difference']:8.4f} | "
              f"{r['empirical_snr']:8.4f} | "
              f"{r['n_trials']:6d}")
    print("-" * 72)
    print(f"Max |difference|: {max_diff:.4f}")
    print(f"Total runtime: {total_elapsed:.1f}s")
    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    if max_diff < 0.05:
        verdict = ("VALIDATED: theoretical formula Phi(-SNR) matches Monte Carlo "
                    "simulation within 5%")
    else:
        verdict = ("WARNING: max |difference| >= 0.05 — theoretical formula may "
                    "need refinement")
    print(verdict)

    # ── Save results ──────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    output_path = os.path.join(paper_dir, "results_monte_carlo_flip.json")

    output = {
        "description": "Monte Carlo validation of flip rate formula Phi(-SNR)",
        "config": {
            "P": P_FEATURES,
            "N_train": N_TRAIN,
            "N_eval": N_EVAL,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
        },
        "results": results,
        "max_abs_difference": round(max_diff, 6),
        "verdict": verdict,
        "total_runtime_seconds": round(total_elapsed, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run_experiment()
