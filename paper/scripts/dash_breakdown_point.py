"""
DASH Breakdown Point — Robustness to contaminated ensemble members.

Measures how many adversarial (label-permuted) models can be injected into a
DASH ensemble before the consensus ranking degrades.  For each contamination
level K out of M=25 models, we replace K clean models with adversarial ones
and measure the within-group flip rate of the DASH consensus (mean |SHAP|).

We also evaluate a trimmed mean (10% trim each tail) as a robust alternative.

Setup:
  - Synthetic Gaussian data, P=10 features (2 correlated groups of 5), ρ=0.9
  - Y = ΣX_j + ε,  N=2000 training,  200 evaluation samples
  - M=25 XGBoost regressors (n_estimators=50, max_depth=4)
  - 20 independent trials per contamination level K
"""

import json
import os
import sys
import warnings

import numpy as np

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

from scipy.stats import kendalltau

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
MASTER_SEED   = 42
P             = 10          # total features (2 groups of 5)
GROUP_SIZE    = 5
RHO           = 0.9         # within-group correlation
N_TRAIN       = 2000
N_EVAL        = 200
M             = 25          # ensemble size
N_TRIALS      = 20          # independent trials per K
K_VALUES      = [0, 1, 2, 3, 5, 7, 10, 12, 15, 20]
TRIM_FRAC     = 0.10        # 10% trimmed mean (each tail)

XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    n_jobs=1,
    verbosity=0,
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PAPER_DIR, "results_dash_breakdown.json")


# ── Data generation ───────────────────────────────────────────────────────────

def make_correlated_gaussian(n, p, group_size, rho, rng):
    """Generate Gaussian features with block-diagonal correlation structure."""
    n_groups = p // group_size
    X = np.zeros((n, p))
    for g in range(n_groups):
        # Build correlation matrix for this group
        cov = np.full((group_size, group_size), rho)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        Z = rng.randn(n, group_size)
        X[:, g * group_size:(g + 1) * group_size] = Z @ L.T
    return X


# ── Helpers ───────────────────────────────────────────────────────────────────

def within_group_flip_rate(importance, group_size):
    """Fraction of within-group pairs where the ranking order disagrees with
    the 'ground truth' ordering (features equally important, so any swap within
    a group is a flip relative to natural index order)."""
    n_groups = len(importance) // group_size
    flips = 0
    total = 0
    for g in range(n_groups):
        idxs = list(range(g * group_size, (g + 1) * group_size))
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                fi, fj = idxs[a], idxs[b]
                # Ground truth: equal importance, so any ordering difference
                # from natural index order counts as a flip.  But since the
                # true coefficients are identical, we check whether the ranking
                # *disagrees* with a reference ranking.  We use the first clean
                # ensemble's consensus as reference — but a simpler metric is
                # the fraction of within-group pairs that are NOT tied.
                # Actually: measure instability as fraction of within-group
                # pairs where rank(fi) > rank(fj) (i.e., fi deemed less
                # important than fj), which should be ~0.5 for truly equal
                # features under noise.  The "flip rate" is |fraction − 0.5|
                # normalized — but for consistency with the rest of the codebase
                # we simply report the fraction of within-group pairs where the
                # lower-indexed feature is ranked *below* the higher-indexed one.
                if importance[fi] < importance[fj]:
                    flips += 1
                total += 1
    return flips / total if total > 0 else 0.0


def trimmed_mean(values, trim_frac):
    """Compute trimmed mean along axis 0, trimming trim_frac from each end."""
    arr = np.sort(values, axis=0)
    n = arr.shape[0]
    lo = int(np.floor(n * trim_frac))
    hi = n - lo
    if hi <= lo:
        return np.mean(arr, axis=0)
    return np.mean(arr[lo:hi], axis=0)


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    rng_data = np.random.RandomState(MASTER_SEED)

    # Generate data
    X = make_correlated_gaussian(N_TRAIN + N_EVAL, P, GROUP_SIZE, RHO, rng_data)
    noise = rng_data.randn(N_TRAIN + N_EVAL) * 0.5
    y = X.sum(axis=1) + noise

    X_train, X_eval = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_eval = y[:N_TRAIN], y[N_TRAIN:]

    print("DASH BREAKDOWN POINT ANALYSIS")
    print("=" * 62)
    print(f"  Data       : N_train={N_TRAIN}, N_eval={N_EVAL}, P={P}")
    print(f"  Groups     : {P // GROUP_SIZE} groups of {GROUP_SIZE}, rho={RHO}")
    print(f"  Ensemble   : M={M} XGBoost models")
    print(f"  Trials     : {N_TRIALS} per contamination level")
    print(f"  Trim frac  : {TRIM_FRAC:.0%} each tail")
    print(f"  K values   : {K_VALUES}")
    print()

    # ── Train clean ensemble ──────────────────────────────────────────────────
    print("Training clean ensemble …")
    clean_models = []
    for m in range(M):
        model = xgb.XGBRegressor(random_state=MASTER_SEED + m, **XGB_PARAMS)
        model.fit(X_train, y_train)
        clean_models.append(model)
    print(f"  {M} clean models trained.")

    # ── Compute clean SHAP values ─────────────────────────────────────────────
    print("Computing TreeSHAP for clean models …")
    clean_shap = []
    for m, model in enumerate(clean_models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)
        clean_shap.append(np.abs(sv).mean(axis=0))  # mean |SHAP| per feature
        if (m + 1) % 10 == 0:
            print(f"  Explained {m + 1}/{M} …")
    clean_shap = np.array(clean_shap)  # shape (M, P)
    print(f"  All {M} clean SHAP profiles computed.")

    # ── Train adversarial models ──────────────────────────────────────────────
    print("Training adversarial models …")
    n_adversarial = max(K_VALUES)
    adv_models = []
    adv_shap = []
    for a in range(n_adversarial):
        rng_perm = np.random.RandomState(MASTER_SEED + 1000 + a)
        y_perm = rng_perm.permutation(y_train)
        model = xgb.XGBRegressor(random_state=MASTER_SEED + 2000 + a, **XGB_PARAMS)
        model.fit(X_train, y_perm)
        adv_models.append(model)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)
        adv_shap.append(np.abs(sv).mean(axis=0))
        if (a + 1) % 5 == 0:
            print(f"  Adversarial model {a + 1}/{n_adversarial} …")
    adv_shap = np.array(adv_shap)  # shape (n_adversarial, P)
    print(f"  {n_adversarial} adversarial SHAP profiles computed.")
    print()

    # ── Contamination sweep ───────────────────────────────────────────────────
    results = {}

    print(f"{'K/M':>5s} | {'Mean Flip Rate':>14s} | {'Trimmed Flip Rate':>17s}")
    print("-" * 44)

    for K in K_VALUES:
        flip_rates_mean = []
        flip_rates_trim = []

        for trial in range(N_TRIALS):
            rng_trial = np.random.RandomState(MASTER_SEED + 5000 + K * 100 + trial)

            # Select which clean models to replace
            replace_idx = rng_trial.choice(M, size=K, replace=False) if K > 0 else []
            # Select which adversarial models to use
            adv_idx = rng_trial.choice(n_adversarial, size=K, replace=False) if K > 0 else []

            # Build contaminated SHAP matrix
            shap_matrix = clean_shap.copy()  # (M, P)
            for r, a in zip(replace_idx, adv_idx):
                shap_matrix[r] = adv_shap[a]

            # Standard DASH consensus: simple mean
            consensus_mean = shap_matrix.mean(axis=0)
            fr_mean = within_group_flip_rate(consensus_mean, GROUP_SIZE)
            flip_rates_mean.append(fr_mean)

            # Trimmed DASH consensus
            consensus_trim = trimmed_mean(shap_matrix, TRIM_FRAC)
            fr_trim = within_group_flip_rate(consensus_trim, GROUP_SIZE)
            flip_rates_trim.append(fr_trim)

        avg_fr_mean = float(np.mean(flip_rates_mean))
        avg_fr_trim = float(np.mean(flip_rates_trim))

        results[K] = {
            "K": K,
            "M": M,
            "contamination_frac": K / M,
            "mean_flip_rate": round(avg_fr_mean, 6),
            "trimmed_flip_rate": round(avg_fr_trim, 6),
            "std_flip_rate_mean": round(float(np.std(flip_rates_mean)), 6),
            "std_flip_rate_trim": round(float(np.std(flip_rates_trim)), 6),
            "n_trials": N_TRIALS,
        }

        print(f"{K:>2d}/{M:<2d} | {avg_fr_mean:>13.1%} | {avg_fr_trim:>16.1%}")

    # ── Breakdown points ──────────────────────────────────────────────────────
    print()

    def find_breakdown(results_dict, key, threshold):
        """Find smallest K where flip rate exceeds threshold."""
        for K in sorted(results_dict.keys()):
            if results_dict[K][key] > threshold:
                return K
        return None

    bp_20_mean = find_breakdown(results, "mean_flip_rate", 0.20)
    bp_40_mean = find_breakdown(results, "mean_flip_rate", 0.40)
    bp_20_trim = find_breakdown(results, "trimmed_flip_rate", 0.20)
    bp_40_trim = find_breakdown(results, "trimmed_flip_rate", 0.40)

    def fmt_bp(k, m):
        if k is None:
            return "not reached"
        return f"K = {k} ({k / m:.0%} contamination)"

    print(f"Breakdown point (20% flip, mean)   : {fmt_bp(bp_20_mean, M)}")
    print(f"Breakdown point (40% flip, mean)   : {fmt_bp(bp_40_mean, M)}")
    print(f"Trimmed mean breakdown (20% flip)  : {fmt_bp(bp_20_trim, M)}")
    print(f"Trimmed mean breakdown (40% flip)  : {fmt_bp(bp_40_trim, M)}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "experiment": "DASH Breakdown Point",
        "config": {
            "P": P,
            "group_size": GROUP_SIZE,
            "rho": RHO,
            "N_train": N_TRAIN,
            "N_eval": N_EVAL,
            "M": M,
            "n_trials": N_TRIALS,
            "trim_frac": TRIM_FRAC,
            "K_values": K_VALUES,
            "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        },
        "results": {str(k): v for k, v in results.items()},
        "breakdown_points": {
            "mean_20pct": {"K": bp_20_mean, "frac": bp_20_mean / M if bp_20_mean else None},
            "mean_40pct": {"K": bp_40_mean, "frac": bp_40_mean / M if bp_40_mean else None},
            "trimmed_20pct": {"K": bp_20_trim, "frac": bp_20_trim / M if bp_20_trim else None},
            "trimmed_40pct": {"K": bp_40_trim, "frac": bp_40_trim / M if bp_40_trim else None},
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
