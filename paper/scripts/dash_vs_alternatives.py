"""
DASH vs Alternative SHAP Stabilization Methods — Benchmark (LIGHT version).

Compares five approaches to stabilizing feature attribution rankings:
  1. Single Model TreeSHAP (baseline)
  2. DASH (M=25 models, consensus averaging)
  3. Bootstrap SHAP (1 model, 25 bootstrap resamples of test data)
  4. Subsampled SHAP (1 model, 25 random background datasets)
  5. Confidence Interval SHAP (1 model, report only non-overlapping CI pairs)

Key insight validated: DASH dominates on flip rate because it is the only
method that addresses the Rashomon property (multiple models). The others
only address SHAP estimation noise (single model).

DGP: P=20 features, L=4 groups of m=5, rho sweep, N=1000, Y = sum(X_j) + eps(sigma=0.1).

Usage:
  python dash_vs_alternatives.py
"""

# Force line-buffered stdout so output is visible during execution
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import warnings
import numpy as np
import json
import os
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
N_SAMPLES = 1000
P_PER_GROUP = 5
N_GROUPS = 4
P_TOTAL = P_PER_GROUP * N_GROUPS  # 20
NOISE_STD = 0.1
RHO_VALUES = [0.5, 0.7, 0.9]

N_TRIALS = 10           # independent trials per (method, rho)
N_REPS = 20             # repetitions for flip rate measurement
M_DASH = 25             # ensemble size for DASH
N_BOOTSTRAP = 25        # bootstrap/subsample replicates
N_BACKGROUND = 100      # background samples for subsampled SHAP
SHAP_EVAL_SAMPLES = 100 # samples for SHAP evaluation

N_BOOT_CI = 1000        # bootstrap resamples for confidence intervals on flip rates

XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=4,
    subsample=0.8,
    learning_rate=0.1,
    verbosity=0,
    eval_metric="rmse",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results_dash_vs_alternatives.json")


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n, rho, seed):
    """Gaussian data with block-diagonal correlation (L groups of m features)."""
    rng = np.random.default_rng(seed)
    p = P_TOTAL

    cov = np.eye(p)
    for g in range(N_GROUPS):
        s, e = g * P_PER_GROUP, (g + 1) * P_PER_GROUP
        for i in range(s, e):
            for j in range(s, e):
                if i != j:
                    cov[i, j] = rho

    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.ones(p)
    y = X @ beta + rng.normal(0, NOISE_STD, size=n)
    return X, y


# ── Within-group correlated pairs ────────────────────────────────────────────

def within_group_pairs():
    """Return list of (i, j) pairs within the same collinear group."""
    pairs = []
    for g in range(N_GROUPS):
        s = g * P_PER_GROUP
        for i in range(s, s + P_PER_GROUP):
            for j in range(i + 1, s + P_PER_GROUP):
                pairs.append((i, j))
    return pairs


WITHIN_PAIRS = within_group_pairs()
N_WITHIN_PAIRS = len(WITHIN_PAIRS)  # 4 * C(5,2) = 40


# ── Helper: compute mean |SHAP| ranking ──────────────────────────────────────

def mean_abs_shap(model, X_eval):
    """Return mean |SHAP| per feature for a fitted XGBoost model."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_eval)
    return np.mean(np.abs(sv), axis=0)


def train_xgb(X_train, y_train, seed):
    """Train one XGBoost regressor."""
    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model


# ── Method implementations ────────────────────────────────────────────────────

def method_single(X_train, y_train, X_test, seed):
    """Single Model TreeSHAP (baseline)."""
    model = train_xgb(X_train, y_train, seed)
    imp = mean_abs_shap(model, X_test)
    return imp, None  # no tie info


def method_dash(X_train, y_train, X_test, seed):
    """DASH: train M models with different seeds, average mean |SHAP|."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=M_DASH)
    all_imp = []
    for s in seeds:
        model = train_xgb(X_train, y_train, int(s))
        all_imp.append(mean_abs_shap(model, X_test))
    imp = np.mean(all_imp, axis=0)
    return imp, None


def method_bootstrap_shap(X_train, y_train, X_test, seed):
    """Bootstrap SHAP: 1 model, 25 bootstrap resamples of test data."""
    model = train_xgb(X_train, y_train, seed)
    rng = np.random.default_rng(seed + 7777)
    all_imp = []
    explainer = shap.TreeExplainer(model)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(X_test), size=len(X_test), replace=True)
        sv = explainer.shap_values(X_test[idx])
        all_imp.append(np.mean(np.abs(sv), axis=0))
    imp = np.mean(all_imp, axis=0)
    return imp, None


def method_subsampled_shap(X_train, y_train, X_test, seed):
    """Subsampled SHAP: 1 model, 25 different background datasets (100 each)."""
    model = train_xgb(X_train, y_train, seed)
    rng = np.random.default_rng(seed + 8888)
    all_imp = []
    for _ in range(N_BOOTSTRAP):
        bg_idx = rng.choice(len(X_train), size=N_BACKGROUND, replace=False)
        explainer = shap.TreeExplainer(model, X_train[bg_idx])
        sv = explainer.shap_values(X_test)
        all_imp.append(np.mean(np.abs(sv), axis=0))
    imp = np.mean(all_imp, axis=0)
    return imp, None


def method_ci_shap(X_train, y_train, X_test, seed):
    """Confidence Interval SHAP: 1 model, report only non-overlapping CI pairs."""
    model = train_xgb(X_train, y_train, seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    abs_sv = np.abs(sv)  # (n_test, P)

    # Per-feature mean and 95% CI (using SE of the mean)
    means = np.mean(abs_sv, axis=0)
    ses = np.std(abs_sv, axis=0, ddof=1) / np.sqrt(abs_sv.shape[0])
    z = norm.ppf(0.975)
    lo = means - z * ses
    hi = means + z * ses

    # Ties: pairs where CIs overlap
    ties = set()
    for (i, j) in WITHIN_PAIRS:
        if lo[i] <= hi[j] and lo[j] <= hi[i]:
            ties.add((i, j))

    return means, ties


# ── Flip rate computation ─────────────────────────────────────────────────────

def compute_flip_rate(rankings_list, ties_list=None):
    """
    Compute within-group pairwise flip rate across N_REPS repetitions.

    Parameters
    ----------
    rankings_list : list of ndarray, each shape (P,)
        Importance scores from each repetition.
    ties_list : list of set or None
        For CI method, sets of (i,j) pairs that are tied.

    Returns
    -------
    mean_flip_rate : float
    pairs_resolved : float
    """
    n_reps = len(rankings_list)
    total_flips = 0
    total_comparisons = 0
    resolved_counts = []

    for r1 in range(n_reps):
        for r2 in range(r1 + 1, n_reps):
            flips = 0
            compared = 0
            for (i, j) in WITHIN_PAIRS:
                # Skip tied pairs for CI method
                if ties_list is not None:
                    if (i, j) in ties_list[r1] or (i, j) in ties_list[r2]:
                        continue
                compared += 1
                sign1 = np.sign(rankings_list[r1][i] - rankings_list[r1][j])
                sign2 = np.sign(rankings_list[r2][i] - rankings_list[r2][j])
                if sign1 != sign2 and sign1 != 0 and sign2 != 0:
                    flips += 1
            total_flips += flips
            total_comparisons += max(compared, 1)

    mean_flip = total_flips / max(total_comparisons, 1)

    # Pairs resolved: for CI method, count non-tied pairs; else all pairs
    if ties_list is not None:
        for t in ties_list:
            resolved_counts.append(N_WITHIN_PAIRS - len(t))
        pairs_resolved = np.mean(resolved_counts)
    else:
        pairs_resolved = float(N_WITHIN_PAIRS)

    return mean_flip, pairs_resolved


def bootstrap_ci_flip_rate(trial_flips, n_boot=N_BOOT_CI, alpha=0.05):
    """
    Compute bootstrap confidence interval for the mean flip rate.

    Parameters
    ----------
    trial_flips : array-like of float
        Flip rates from each trial.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 for 95% CI).

    Returns
    -------
    ci_lower, ci_upper : float
    """
    arr = np.array(trial_flips)
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[b] = np.mean(sample)
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


# ── Main benchmark ────────────────────────────────────────────────────────────

METHODS = {
    "Single Model": method_single,
    "DASH (M=25)": method_dash,
    "Bootstrap SHAP": method_bootstrap_shap,
    "Subsampled SHAP": method_subsampled_shap,
    "CI SHAP": method_ci_shap,
}


def run_benchmark():
    """Run full benchmark across all methods and rho values."""
    results = {}
    total_start = time.time()
    print("=" * 100)
    print(f"DASH vs Alternative SHAP Stabilization Methods (LIGHT)")
    print(f"P={P_TOTAL}, L={N_GROUPS} groups of m={P_PER_GROUP}, "
          f"N={N_SAMPLES}, {N_TRIALS} trials, {N_REPS} reps/trial, "
          f"n_estimators={XGB_PARAMS['n_estimators']}, SHAP_eval={SHAP_EVAL_SAMPLES}")
    print("=" * 100)

    for rho in RHO_VALUES:
        results[str(rho)] = {}
        print(f"\n{'─' * 100}")
        print(f"  rho = {rho}")
        print(f"{'─' * 100}")
        header = (f"  {'Method':<22s} {'Flip Rate':>10s} {'95% CI':>22s} "
                  f"{'Flip Std':>10s} {'Wall (s)':>10s} {'Pairs Resolved':>15s}")
        print(header)
        print(f"  {'─' * 92}")

        for method_name, method_fn in METHODS.items():
            trial_flips = []
            trial_times = []
            trial_resolved = []

            for trial in range(N_TRIALS):
                data_seed = trial * 1000 + int(rho * 100)
                X, y = generate_data(N_SAMPLES, rho, data_seed)
                n_train = int(0.8 * N_SAMPLES)
                X_train, X_test_full = X[:n_train], X[n_train:]
                y_train, y_test = y[:n_train], y[n_train:]

                # Subsample SHAP eval set for speed
                rng_eval = np.random.default_rng(data_seed + 9999)
                if len(X_test_full) > SHAP_EVAL_SAMPLES:
                    eval_idx = rng_eval.choice(len(X_test_full), size=SHAP_EVAL_SAMPLES, replace=False)
                    X_test = X_test_full[eval_idx]
                else:
                    X_test = X_test_full

                # Collect N_REPS rankings for flip rate
                rankings = []
                ties = [] if method_name == "CI SHAP" else None

                t0 = time.time()
                for rep in range(N_REPS):
                    rep_seed = trial * 10000 + rep
                    imp, tie_set = method_fn(X_train, y_train, X_test, rep_seed)
                    rankings.append(imp)
                    if ties is not None:
                        ties.append(tie_set if tie_set is not None else set())
                wall = time.time() - t0

                flip, resolved = compute_flip_rate(
                    rankings,
                    ties_list=ties if method_name == "CI SHAP" else None,
                )
                trial_flips.append(flip)
                trial_times.append(wall / N_REPS)  # per-run time
                trial_resolved.append(resolved)

                # Progress indicator
                print(f"    [{method_name}] trial {trial+1}/{N_TRIALS} done "
                      f"(flip={flip:.4f}, {wall:.1f}s)")

            mean_flip = np.mean(trial_flips)
            std_flip = np.std(trial_flips, ddof=1)
            mean_time = np.mean(trial_times)
            mean_resolved = np.mean(trial_resolved)

            # Bootstrap CI on the flip rate
            ci_lower, ci_upper = bootstrap_ci_flip_rate(trial_flips)

            results[str(rho)][method_name] = {
                "flip_rate": round(float(mean_flip), 4),
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
                "flip_rate_std": round(float(std_flip), 4),
                "wall_time_s": round(float(mean_time), 3),
                "pairs_resolved": round(float(mean_resolved), 2),
                "n_within_pairs": N_WITHIN_PAIRS,
            }

            ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            print(f"  {method_name:<22s} {mean_flip:>10.4f} {ci_str:>22s} "
                  f"{std_flip:>10.4f} {mean_time:>10.3f} "
                  f"{mean_resolved:>12.1f}/{N_WITHIN_PAIRS}")

    elapsed = time.time() - total_start
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    results["_meta"] = {
        "N_SAMPLES": N_SAMPLES,
        "N_TRIALS": N_TRIALS,
        "N_REPS": N_REPS,
        "n_estimators": XGB_PARAMS["n_estimators"],
        "SHAP_EVAL_SAMPLES": SHAP_EVAL_SAMPLES,
        "rho_values": RHO_VALUES,
        "total_wall_time_s": round(elapsed, 1),
    }
    return results


def print_summary(results):
    """Print which method achieves lowest flip rate at each rho."""
    print(f"\n{'=' * 100}")
    print("SUMMARY: Lowest flip rate at each rho (with 95% bootstrap CI)")
    print(f"{'=' * 100}")
    dash_wins = True
    for rho in RHO_VALUES:
        rho_key = str(rho)
        best_name = None
        best_flip = float("inf")
        for method_name, vals in results[rho_key].items():
            if vals["flip_rate"] < best_flip:
                best_flip = vals["flip_rate"]
                best_name = method_name
        ci_lo = results[rho_key][best_name]["ci_lower"]
        ci_hi = results[rho_key][best_name]["ci_upper"]
        marker = " <-- DASH" if "DASH" in best_name else " <-- NOT DASH!"
        if "DASH" not in best_name:
            dash_wins = False
        print(f"  rho={rho:<5}  ->  {best_name:<22s}  "
              f"(flip rate = {best_flip:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]){marker}")

    print()
    if dash_wins:
        print("RESULT: DASH has the lowest flip rate at every rho value.")
        print("This confirms: DASH addresses the Rashomon property (multiple models),")
        print("while Bootstrap/Subsampled/CI SHAP only address estimation noise")
        print("(single model). At high rho, single-model methods cannot reduce")
        print("flip rates because the instability is structural, not statistical.")
    else:
        print("WARNING: DASH did NOT win at every rho. Investigate further.")

    # Also print all CIs for easy comparison
    print(f"\n{'=' * 100}")
    print("FULL TABLE: flip_rate [ci_lower, ci_upper]")
    print(f"{'=' * 100}")
    print(f"  {'Method':<22s}", end="")
    for rho in RHO_VALUES:
        print(f"  {'rho='+str(rho):>28s}", end="")
    print()
    for method_name in METHODS:
        print(f"  {method_name:<22s}", end="")
        for rho in RHO_VALUES:
            v = results[str(rho)][method_name]
            print(f"  {v['flip_rate']:.4f} [{v['ci_lower']:.4f},{v['ci_upper']:.4f}]", end="")
        print()


def main():
    results = run_benchmark()
    print_summary(results)

    # Save JSON
    out_path = os.path.abspath(RESULTS_PATH)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
