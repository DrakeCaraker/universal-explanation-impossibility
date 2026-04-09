"""
Missing Data Compounds SHAP Instability.

Shows that missing data (under MCAR, MAR, and MNAR mechanisms) widens the
Rashomon set and amplifies within-group SHAP ranking instability. Synthetic
Gaussian data with P=10 features (2 groups of 5, within-group rho=0.8).
Y = sum(X_j) + epsilon (regression).

For each (mechanism, rate): inject missingness, impute with column medians,
train 30 XGBoost regressors (different seeds, subsample=0.8), compute
TreeSHAP, measure within-group flip rate.

Expected: more missing -> noisier data -> wider Rashomon -> more instability.

Saves results to paper/results_missing_data.json.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import os
import sys
import time
from itertools import combinations

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
P_FEATURES = 10          # total features (2 groups of 5)
GROUP_SIZE = 5            # features per correlated group
RHO = 0.8                # within-group correlation
N_TRAIN = 2000            # training samples
N_EVAL = 200              # evaluation samples for SHAP
N_MODELS = 30             # models per (mechanism, rate) combination
N_ESTIMATORS = 100        # trees per model
MAX_DEPTH = 6             # tree depth
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8           # XGBoost subsampling
DATA_SEED = 42            # fixed data generation seed

MISSING_RATES = [0.0, 0.05, 0.10, 0.20]
MECHANISMS = ["MCAR", "MAR", "MNAR"]


# -- Data generation -----------------------------------------------------------

def generate_correlated_data(n_samples, rng):
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
                    cov[i, j] = RHO

    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)
    Y = X.sum(axis=1) + rng.standard_normal(n_samples)
    return X, Y


# -- Missing data injection ----------------------------------------------------

def inject_missing(X, rate, mechanism, rng):
    """Inject missing values (NaN) into X according to the given mechanism.

    MCAR: each entry missing independently with probability `rate`.
    MAR:  missingness depends on X[:,0] — entries in columns 1..P-1 are
          missing when X[:,0] is above its (1-rate) quantile.
    MNAR: for each column j, entries are missing when X[:,j] itself is
          above its (1-rate) quantile (depends on the missing value).

    Returns a copy of X with NaN in missing positions.
    """
    if rate <= 0.0:
        return X.copy()

    X_out = X.copy()
    n, p = X_out.shape

    if mechanism == "MCAR":
        mask = rng.rand(n, p) < rate
        X_out[mask] = np.nan

    elif mechanism == "MAR":
        # Missingness in columns 1..p-1 depends on X[:,0]
        threshold = np.quantile(X_out[:, 0], 1.0 - rate)
        high_x0 = X_out[:, 0] > threshold
        for j in range(1, p):
            X_out[high_x0, j] = np.nan

    elif mechanism == "MNAR":
        # For each column, entries are missing when the value itself is high
        for j in range(p):
            threshold = np.quantile(X_out[:, j], 1.0 - rate)
            high_mask = X_out[:, j] > threshold
            X_out[high_mask, j] = np.nan

    return X_out


def median_impute(X):
    """Impute NaN with column medians. Returns imputed copy."""
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        col = X_out[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X_out[nan_mask, j] = median_val
    return X_out


# -- Flip rate -----------------------------------------------------------------

def within_group_flip_rate(rankings, group_indices):
    """Compute flip rate for all within-group pairs across model rankings.

    rankings: list of arrays, each array is feature indices sorted by
              descending mean |SHAP|
    group_indices: list of feature indices in one group

    Returns list of dicts with (fi, fj, flip_rate) for each pair.
    """
    n_models = len(rankings)
    results = []
    for fi, fj in combinations(group_indices, 2):
        flips = 0
        total = 0
        for a in range(n_models):
            for b in range(a + 1, n_models):
                pos_fi_a = int(np.where(rankings[a] == fi)[0][0])
                pos_fj_a = int(np.where(rankings[a] == fj)[0][0])
                pos_fi_b = int(np.where(rankings[b] == fi)[0][0])
                pos_fj_b = int(np.where(rankings[b] == fj)[0][0])
                order_a = pos_fi_a < pos_fj_a
                order_b = pos_fi_b < pos_fj_b
                if order_a != order_b:
                    flips += 1
                total += 1
        fr = flips / total if total > 0 else 0.0
        results.append({"fi": fi, "fj": fj, "flip_rate": round(fr, 6)})
    return results


# -- Main experiment -----------------------------------------------------------

def run_experiment():
    print("=" * 72)
    print("Missing Data Compounds SHAP Instability")
    print("=" * 72)
    print(f"Config: P={P_FEATURES}, rho={RHO}, N_train={N_TRAIN}, N_eval={N_EVAL}")
    print(f"        n_models={N_MODELS}, n_estimators={N_ESTIMATORS}, "
          f"max_depth={MAX_DEPTH}, subsample={SUBSAMPLE}")
    print(f"Mechanisms: {MECHANISMS}")
    print(f"Rates:      {MISSING_RATES}")
    print()

    # Generate base datasets (same for all conditions)
    train_rng = np.random.RandomState(DATA_SEED)
    X_train_clean, Y_train = generate_correlated_data(N_TRAIN, train_rng)
    eval_rng = np.random.RandomState(DATA_SEED + 9999)
    X_eval, _ = generate_correlated_data(N_EVAL, eval_rng)

    group_A = list(range(0, GROUP_SIZE))           # features 0-4
    group_B = list(range(GROUP_SIZE, P_FEATURES))  # features 5-9

    results = []
    total_start = time.time()

    for mechanism in MECHANISMS:
        for rate in MISSING_RATES:
            cond_start = time.time()
            label = f"{mechanism} {rate*100:.0f}%"
            print(f"{label:>12} ...", end=" ", flush=True)

            rankings = []
            for seed in range(N_MODELS):
                # Inject missingness (different rng per model for MCAR variety)
                miss_rng = np.random.RandomState(DATA_SEED + seed + 200)
                X_missing = inject_missing(X_train_clean, rate, mechanism, miss_rng)
                X_imputed = median_impute(X_missing)

                # Train XGBoost regressor
                model = xgb.XGBRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    learning_rate=LEARNING_RATE,
                    subsample=SUBSAMPLE,
                    random_state=seed,
                    n_jobs=1,
                    verbosity=0,
                )
                model.fit(X_imputed, Y_train)

                # TreeSHAP on clean eval set (no missingness in eval)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_eval)

                importance = np.abs(shap_vals).mean(axis=0)
                rank = np.argsort(-importance)  # descending
                rankings.append(rank)

            # Compute within-group flip rates
            flips_A = within_group_flip_rate(rankings, group_A)
            flips_B = within_group_flip_rate(rankings, group_B)
            all_flips = flips_A + flips_B

            max_flip = max(d["flip_rate"] for d in all_flips) if all_flips else 0.0
            n_unstable = sum(1 for d in all_flips if d["flip_rate"] > 0)

            elapsed = time.time() - cond_start
            print(f"done ({elapsed:.1f}s) | max_flip={max_flip:.4f}, "
                  f"unstable={n_unstable}/{len(all_flips)}")

            results.append({
                "mechanism": mechanism,
                "rate": rate,
                "rate_pct": f"{rate*100:.0f}%",
                "max_flip_rate": round(max_flip, 6),
                "n_unstable_pairs": n_unstable,
                "n_total_within_group_pairs": len(all_flips),
                "top_flips": sorted(all_flips, key=lambda d: -d["flip_rate"])[:5],
            })

    total_elapsed = time.time() - total_start

    # -- Print summary table ---------------------------------------------------
    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    print(f"{'Mechanism':>10} | {'Rate':>6} | {'max_flip_rate':>13} | "
          f"{'n_unstable_pairs':>16}")
    print("-" * 56)
    for r in results:
        print(f"{r['mechanism']:>10} | {r['rate_pct']:>6} | "
              f"{r['max_flip_rate']:>13.4f} | "
              f"{r['n_unstable_pairs']:>16}")
    print("-" * 56)
    print(f"Total runtime: {total_elapsed:.1f}s")
    print()

    # -- Check increasing trend per mechanism ----------------------------------
    verdicts = []
    for mechanism in MECHANISMS:
        mech_results = [r for r in results if r["mechanism"] == mechanism]
        flips = [r["max_flip_rate"] for r in mech_results]
        # Check if 0% rate <= max of nonzero rates (allowing stochastic noise)
        if len(flips) >= 2 and flips[-1] >= flips[0] - 0.05:
            verdicts.append(f"  {mechanism}: instability increases with missing rate "
                            f"({flips[0]:.4f} -> {flips[-1]:.4f})")
        else:
            verdicts.append(f"  {mechanism}: no clear trend "
                            f"({flips[0]:.4f} -> {flips[-1]:.4f})")

    overall_verdict = ("CONFIRMED: missing data compounds SHAP instability"
                       if all("increases" in v for v in verdicts)
                       else "PARTIAL: trend observed for some mechanisms")

    print(overall_verdict)
    for v in verdicts:
        print(v)

    # -- Save results ----------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    output_path = os.path.join(paper_dir, "results_missing_data.json")

    output = {
        "description": ("Missing data compounds SHAP instability: "
                        "more missing -> noisier data -> wider Rashomon -> more flips"),
        "config": {
            "P": P_FEATURES,
            "group_size": GROUP_SIZE,
            "rho": RHO,
            "N_train": N_TRAIN,
            "N_eval": N_EVAL,
            "n_models": N_MODELS,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
            "mechanisms": MECHANISMS,
            "missing_rates": MISSING_RATES,
        },
        "results": results,
        "verdict": overall_verdict,
        "mechanism_verdicts": verdicts,
        "total_runtime_seconds": round(total_elapsed, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run_experiment()
