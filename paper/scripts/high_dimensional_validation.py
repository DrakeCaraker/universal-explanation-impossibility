"""
High-dimensional validation of the Attribution Impossibility diagnostics (P > 500 features).

What this validates
-------------------
The paper (Theorem 1 / Trilemma.lean) proves that no feature ranking can be
simultaneously faithful, stable, and complete under collinearity.  The F1
diagnostic — a Z-test on the mean attribution difference across M models —
detects ranking instability.  This script stress-tests the diagnostic at
enterprise scale (P = 500 features, N = 2000 samples) to confirm:

  (a) The Z-statistic is strongly correlated with the empirical flip rate
      (|r| > 0.7 at P=500 is the acceptance threshold).
  (b) The fraction of unstable pairs grows as expected in the correlated blocks.
  (c) Timing scales acceptably for practical deployment.

Data-generating process
-----------------------
50 groups of 10 features each.  Within-group correlation rho = 0.8 via a
latent factor model: X_{g,i} = sqrt(rho) * Z_g + sqrt(1-rho) * eps_{g,i}.
Groups are independent.  Target Y = sum of group means + N(0,1) noise.

Exit code
---------
0  if |r(Z, flip_rate)| > 0.7 at P=500
1  otherwise
"""

import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import pearsonr
import xgboost as xgb
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M = 20                  # number of XGBoost models per trial
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
N_ROWS = 2000           # training + test pool
TEST_SIZE = 400         # held-out test set used for SHAP
N_GROUPS = 50
GROUP_SIZE = 10
RHO = 0.8
FLIP_THRESHOLD = 0.10   # fraction of model-pairs that disagree on ranking
PASS_THRESHOLD = 0.70   # |r| must exceed this at P=500
NUMPY_SEED = 42

SCALE_PS = [100, 200, 500]   # P values for the scaling experiment


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_correlated_data(P: int, N: int, rng: np.random.Generator):
    """
    Generate synthetic data with block-correlated features.

    Parameters
    ----------
    P : int
        Number of features (must be divisible by GROUP_SIZE).
    N : int
        Number of samples.
    rng : numpy Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    X : ndarray, shape (N, P)
    y : ndarray, shape (N,)
    group_ids : list of list of int
        group_ids[g] = list of feature indices belonging to group g.
    """
    n_groups = P // GROUP_SIZE
    Z = rng.standard_normal((N, n_groups))         # latent factors
    eps = rng.standard_normal((N, P))               # idiosyncratic noise

    X = np.empty((N, P), dtype=np.float32)
    group_ids = []
    for g in range(n_groups):
        cols = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
        group_ids.append(cols)
        X[:, cols] = (
            np.sqrt(RHO) * Z[:, g : g + 1]
            + np.sqrt(1.0 - RHO) * eps[:, cols]
        )

    # Target: sum of group means + unit Gaussian noise
    group_means = X.reshape(N, n_groups, GROUP_SIZE).mean(axis=2)
    y = group_means.sum(axis=1) + rng.standard_normal(N)
    return X, y.astype(np.float32), group_ids


# ---------------------------------------------------------------------------
# Model training and SHAP
# ---------------------------------------------------------------------------

def train_models(X_train, y_train, X_test, seeds):
    """
    Train M XGBoost regressors and return mean |SHAP| per feature per model.

    Parameters
    ----------
    X_train, y_train : training arrays
    X_test : test array used for SHAP computation
    seeds : list of int, length M

    Returns
    -------
    shap_matrix : ndarray, shape (M, P)
        Row m = mean absolute SHAP values for model m over X_test.
    """
    M_local = len(seeds)
    P = X_train.shape[1]
    shap_matrix = np.empty((M_local, P), dtype=np.float64)

    for i, seed in enumerate(seeds):
        model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)          # shape (N_test, P)
        shap_matrix[i] = np.mean(np.abs(sv), axis=0)

        if (i + 1) % 5 == 0:
            print(f"    Trained model {i + 1}/{M_local} (seed={seed})")

    return shap_matrix


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(shap_matrix: np.ndarray, group_ids: list):
    """
    Compute F1 Z-statistics and empirical flip rates for all within-group pairs.

    Parameters
    ----------
    shap_matrix : ndarray, shape (M, P)
    group_ids : list of lists

    Returns
    -------
    z_stats : ndarray, shape (n_correlated_pairs,)
    flip_rates : ndarray, shape (n_correlated_pairs,)
    """
    M_local = shap_matrix.shape[0]
    z_list = []
    flip_list = []

    for cols in group_ids:
        for idx_a, j in enumerate(cols):
            for idx_b in range(idx_a + 1, len(cols)):
                k = cols[idx_b]
                phi_j = shap_matrix[:, j]   # (M,)
                phi_k = shap_matrix[:, k]   # (M,)
                diff = phi_j - phi_k

                mu = np.mean(diff)
                se = np.std(diff, ddof=1) / np.sqrt(M_local)
                z = np.abs(mu) / (se + 1e-12)
                z_list.append(z)

                # Flip rate: fraction of model PAIRS that disagree
                ranks_j = phi_j > phi_k    # True = j ranked above k in model m
                disagreements = 0
                n_pairs = 0
                for m1 in range(M_local):
                    for m2 in range(m1 + 1, M_local):
                        n_pairs += 1
                        if ranks_j[m1] != ranks_j[m2]:
                            disagreements += 1
                flip_rate = disagreements / n_pairs if n_pairs > 0 else 0.0
                flip_list.append(flip_rate)

    return np.array(z_list), np.array(flip_list)


def total_pairs(P: int) -> int:
    return P * (P - 1) // 2


def correlated_pairs_count(P: int) -> int:
    n_groups = P // GROUP_SIZE
    return n_groups * (GROUP_SIZE * (GROUP_SIZE - 1) // 2)


# ---------------------------------------------------------------------------
# Single-scale trial
# ---------------------------------------------------------------------------

def run_trial(P: int, rng: np.random.Generator):
    """
    Full pipeline for a single value of P.

    Returns a dict with keys:
        P, num_pairs, num_correlated_pairs, num_unstable,
        F1_correlation, wall_time
    """
    print(f"\n{'='*60}")
    print(f"  P = {P}  |  N = {N_ROWS}  |  M = {M} models")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    # --- Data generation ---
    print("  Generating correlated data...")
    X, y, group_ids = make_correlated_data(P, N_ROWS, rng)
    X_train, y_train = X[:N_ROWS - TEST_SIZE], y[:N_ROWS - TEST_SIZE]
    X_test = X[N_ROWS - TEST_SIZE:]

    # --- Model training ---
    seeds = list(range(M))
    print(f"  Training {M} XGBoost models...")
    shap_matrix = train_models(X_train, y_train, X_test, seeds)

    # --- Diagnostics (within-group pairs only) ---
    print("  Computing F1 Z-statistics and flip rates...")
    z_stats, flip_rates = compute_diagnostics(shap_matrix, group_ids)

    # --- Summary statistics ---
    n_pairs = total_pairs(P)
    n_corr = correlated_pairs_count(P)
    n_unstable = int(np.sum(flip_rates > FLIP_THRESHOLD))

    if len(z_stats) >= 2:
        r, _ = pearsonr(z_stats, flip_rates)
        # F1 Z is positively correlated with instability at low Z (high flip rate
        # when Z is small), so we check the magnitude.
        # Intuition: low |Z| → rankings are indistinguishable → high flip rate.
        # Thus the raw correlation should be negative; we report |r|.
        f1_corr = abs(r)
    else:
        f1_corr = float("nan")

    wall_time = time.perf_counter() - t0

    print(f"  Correlated pairs:  {n_corr}")
    print(f"  Unstable pairs (flip > {FLIP_THRESHOLD:.0%}): {n_unstable}")
    print(f"  F1 |r(Z, flip)|:   {f1_corr:.4f}")
    print(f"  Wall time:         {wall_time:.1f}s")

    return {
        "P": P,
        "num_pairs": n_pairs,
        "num_correlated_pairs": n_corr,
        "num_unstable": n_unstable,
        "F1_correlation": f1_corr,
        "wall_time": wall_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(NUMPY_SEED)

    print("=" * 60)
    print("  Attribution Impossibility — High-Dimensional Validation")
    print(f"  P values: {SCALE_PS}  |  M={M} models  |  rho={RHO}")
    print(f"  Acceptance threshold: |r| > {PASS_THRESHOLD} at P=500")
    print("=" * 60)

    results = []
    for P in SCALE_PS:
        result = run_trial(P, rng)
        results.append(result)

    # --- Summary table ---
    print()
    print("=" * 80)
    print(
        f"{'P':>6}  {'num_pairs':>12}  {'corr_pairs':>12}  "
        f"{'unstable':>10}  {'F1_|r|':>8}  {'wall_time(s)':>13}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['P']:>6}  {r['num_pairs']:>12,}  {r['num_correlated_pairs']:>12,}  "
            f"{r['num_unstable']:>10,}  {r['F1_correlation']:>8.4f}  "
            f"{r['wall_time']:>12.1f}s"
        )
    print("=" * 80)

    # --- Pass/fail at P=500 ---
    result_500 = next((r for r in results if r["P"] == 500), None)
    if result_500 is not None:
        passed = result_500["F1_correlation"] > PASS_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(
            f"\nF1 diagnostic maintains |r| > {PASS_THRESHOLD} at P=500: {status}"
            f"  (|r| = {result_500['F1_correlation']:.4f})"
        )
        sys.exit(0 if passed else 1)
    else:
        print("\nP=500 trial not found in results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
