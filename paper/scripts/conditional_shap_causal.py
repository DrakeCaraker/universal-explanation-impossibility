"""
conditional_shap_causal.py
--------------------------
Experimental validation of the Conditional Attribution Impossibility theorem.

The paper proves:
  - Switching to conditional SHAP does NOT resolve attribution instability
    when features have equal causal effects (beta_j = beta_k).
  - When beta_j != beta_k, conditional SHAP CAN produce stable rankings.

This script tests both scenarios by comparing marginal SHAP (standard
TreeSHAP with tree_path_dependent perturbation) and interventional SHAP
(TreeSHAP with interventional perturbation and background data) across a
range of feature correlations.

NOTE on approximation: XGBoost's interventional SHAP uses a background
dataset to marginalize over the feature distribution, which approximates
(but does not exactly implement) causal/conditional SHAP in the Pearl sense.
The interventional estimator conditions on the causal structure via the
background distribution, making it a reasonable proxy for the causal
attribution regime tested here. Results should be interpreted as empirical
evidence, not a formal proof.

Causal data generating process (DGP):
  Z, epsilon ~ N(0, 1) independently
  X_1 = Z
  X_2 = rho * Z + sqrt(1 - rho^2) * epsilon
  X_3, X_4, X_5 ~ N(0, 1) independently (controls)
  Y = beta_1 * X_1 + beta_2 * X_2 + 0.5 * X_3 + noise,  noise ~ N(0, 0.5)

Scenarios:
  Symmetric:   beta_1 = beta_2 = 1.0  -> equal causal effects -> instability persists
  Asymmetric:  beta_1 = 1.5, beta_2 = 0.5  -> unequal effects -> interventional SHAP stable
  Sweep:       Delta_beta in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} at rho=0.9
               beta_1 = 1.0 + Delta_beta/2, beta_2 = 1.0 - Delta_beta/2

Validation criteria:
  Symmetric,   high rho:   marginal flip rate > 20% AND interventional flip rate > 20%
  Asymmetric,  any rho:    interventional flip rate < 10% (stable)
  Sweep crossover:         identify Delta_beta where marginal > 10% but interventional < 10%
"""

import os
import sys
import warnings
import itertools

import numpy as np
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_MODELS = 20
N_BACKGROUND = 100
SIGMA_NOISE = 0.5
TRAIN_FRAC = 0.8

RHOS = [0.5, 0.7, 0.9, 0.95]

SCENARIOS = {
    "Symmetric":  (1.0, 1.0),
    "Asymmetric": (1.5, 0.5),
}

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    verbosity=0,
)

# Validation thresholds
SYMMETRIC_HIGH_RHO_THRESHOLD = 0.90   # rho >= this is "high" for the symmetric check
SYMMETRIC_MIN_FLIP = 0.20             # both estimators should exceed this
ASYMMETRIC_MAX_INTERVENTIONAL_FLIP = 0.10  # interventional should be below this

# Sweep parameters
SWEEP_DELTA_BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
SWEEP_RHO = 0.9
SWEEP_N_MODELS = 20
SWEEP_MARGINAL_UNSTABLE_THRESHOLD = 0.10   # marginal flip > this -> unstable
SWEEP_INTERVENTIONAL_STABLE_THRESHOLD = 0.10  # interventional flip < this -> stable

# ── Data generation ────────────────────────────────────────────────────────────

def generate_data(rho: float, beta_1: float, beta_2: float,
                  n: int, rng: np.random.Generator) -> tuple:
    """
    Generate (X, y) from the causal DGP.

    X has 5 columns: X_1, X_2 (correlated), X_3, X_4, X_5 (independent controls).
    Corr(X_1, X_2) = rho by construction via the latent variable Z.
    """
    Z = rng.standard_normal(n)
    eps = rng.standard_normal(n)

    X1 = Z
    X2 = rho * Z + np.sqrt(1.0 - rho ** 2) * eps
    X3 = rng.standard_normal(n)
    X4 = rng.standard_normal(n)
    X5 = rng.standard_normal(n)

    X = np.column_stack([X1, X2, X3, X4, X5])
    noise = rng.normal(0.0, SIGMA_NOISE, size=n)
    y = beta_1 * X1 + beta_2 * X2 + 0.5 * X3 + noise
    return X, y


# ── SHAP computation ──────────────────────────────────────────────────────────

def mean_abs_shap_marginal(model: xgb.XGBRegressor,
                           X_eval: np.ndarray) -> np.ndarray:
    """
    Marginal SHAP attributions via tree_path_dependent perturbation.

    Returns mean |SHAP| per feature (shape: n_features,).
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    shap_vals = explainer.shap_values(X_eval)
    return np.abs(shap_vals).mean(axis=0)


def mean_abs_shap_interventional(model: xgb.XGBRegressor,
                                 X_eval: np.ndarray,
                                 X_background: np.ndarray) -> np.ndarray:
    """
    Interventional SHAP attributions.

    Uses a background dataset to marginalize over feature distribution,
    approximating conditional/causal SHAP (see module docstring caveat).

    Returns mean |SHAP| per feature (shape: n_features,).
    """
    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional",
    )
    shap_vals = explainer.shap_values(X_eval)
    return np.abs(shap_vals).mean(axis=0)


# ── Flip rate computation ─────────────────────────────────────────────────────

def compute_flip_rate(rankings: list) -> float:
    """
    Fraction of model pairs where the ranking of feature 0 vs feature 1 reverses.

    rankings: list of bools, True if feature 0 ranked above feature 1.
    """
    n_pairs = N_MODELS * (N_MODELS - 1) // 2
    if n_pairs == 0:
        return 0.0
    n_flips = sum(1 for a, b in itertools.combinations(rankings, 2) if a != b)
    return n_flips / n_pairs


# ── Main simulation ────────────────────────────────────────────────────────────

def run_scenario(scenario_name: str, beta_1: float, beta_2: float) -> dict:
    """
    Run all rho values for one scenario.

    Returns dict keyed by rho with values (marginal_flip, interventional_flip).
    """
    rng_data = np.random.default_rng(42)
    results = {}

    print(f"\n  Scenario: {scenario_name}  (beta_1={beta_1}, beta_2={beta_2})")
    print(f"  {'rho':>6}  {'Marginal flip':>14}  {'Interv. flip':>13}")
    print(f"  {'-'*40}")

    for rho in RHOS:
        marginal_rankings = []
        interventional_rankings = []

        for seed in range(N_MODELS):
            # Use a reproducible but independent seed per (rho, seed) combination
            local_rng = np.random.default_rng(
                rng_data.integers(0, 2 ** 32)
            )
            X, y = generate_data(rho, beta_1, beta_2, N_SAMPLES, local_rng)

            n_train = int(TRAIN_FRAC * N_SAMPLES)
            X_train, y_train = X[:n_train], y[:n_train]
            X_eval = X[n_train:]

            # Background: N_BACKGROUND random samples from training set
            bg_idx = np.random.default_rng(seed).choice(
                n_train, size=N_BACKGROUND, replace=False
            )
            X_bg = X_train[bg_idx]

            model = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
            model.fit(X_train, y_train)

            # Marginal SHAP
            ms_marginal = mean_abs_shap_marginal(model, X_eval)
            marginal_rankings.append(ms_marginal[0] > ms_marginal[1])

            # Interventional SHAP
            ms_interv = mean_abs_shap_interventional(model, X_eval, X_bg)
            interventional_rankings.append(ms_interv[0] > ms_interv[1])

        marg_flip = compute_flip_rate(marginal_rankings)
        interv_flip = compute_flip_rate(interventional_rankings)
        results[rho] = (marg_flip, interv_flip)

        print(f"  {rho:>6.2f}  {marg_flip:>14.3f}  {interv_flip:>13.3f}")

    return results


# ── Theory prediction strings ─────────────────────────────────────────────────

def theory_prediction(scenario_name: str, rho: float) -> str:
    if scenario_name == "Symmetric":
        return "~50% both"
    else:
        if rho >= 0.7:
            return "marginal unstable, interventional stable"
        else:
            return "marginal unstable, interventional stable"


# ── Validation checks ─────────────────────────────────────────────────────────

def run_validation(all_results: dict) -> bool:
    """
    Run all validation checks. Returns True if all pass.
    """
    print("\n=== Validation Checks ===")
    all_passed = True

    # Check 1: Symmetric case, high rho -> both marginal and interventional > 20%
    print("\nCheck 1: Symmetric case at high rho (>= 0.90) — both methods should be unstable (flip rate > 20%)")
    for rho in RHOS:
        if rho >= SYMMETRIC_HIGH_RHO_THRESHOLD:
            marg, interv = all_results["Symmetric"][rho]
            marg_ok = marg > SYMMETRIC_MIN_FLIP
            interv_ok = interv > SYMMETRIC_MIN_FLIP
            status_marg = "PASS" if marg_ok else "FAIL"
            status_interv = "PASS" if interv_ok else "FAIL"
            print(f"  rho={rho:.2f}  marginal={marg:.3f} [{status_marg}]  "
                  f"interventional={interv:.3f} [{status_interv}]")
            if not (marg_ok and interv_ok):
                all_passed = False

    # Check 2: Asymmetric case -> interventional flip rate < 10% for all rho
    print("\nCheck 2: Asymmetric case — interventional SHAP should be stable (flip rate < 10%)")
    for rho in RHOS:
        marg, interv = all_results["Asymmetric"][rho]
        interv_ok = interv < ASYMMETRIC_MAX_INTERVENTIONAL_FLIP
        status = "PASS" if interv_ok else "FAIL"
        print(f"  rho={rho:.2f}  marginal={marg:.3f}  interventional={interv:.3f} [{status}]")
        if not interv_ok:
            all_passed = False

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return all_passed


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(all_results: dict):
    col_w = [12, 6, 15, 21, 42]
    header = (
        f"{'Scenario':<{col_w[0]}} | "
        f"{'rho':<{col_w[1]}} | "
        f"{'Marginal flip':<{col_w[2]}} | "
        f"{'Interventional flip':<{col_w[3]}} | "
        f"{'Theory prediction':<{col_w[4]}}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for scenario_name, (beta_1, beta_2) in SCENARIOS.items():
        for rho in RHOS:
            marg, interv = all_results[scenario_name][rho]
            theory = theory_prediction(scenario_name, rho)
            print(
                f"{scenario_name:<{col_w[0]}} | "
                f"{rho:<{col_w[1]}.2f} | "
                f"{marg:<{col_w[2]}.3f} | "
                f"{interv:<{col_w[3]}.3f} | "
                f"{theory:<{col_w[4]}}"
            )

    print(sep)


# ── Delta-beta sweep ───────────────────────────────────────────────────────────

def run_sweep() -> list:
    """
    Sweep Delta_beta in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} at rho=SWEEP_RHO.

    For each Delta_beta: beta_1 = 1.0 + Delta_beta/2, beta_2 = 1.0 - Delta_beta/2
    (centered at 1.0, so Delta_beta=0 reproduces the symmetric case and
    Delta_beta=1.0 reproduces the asymmetric case).

    Returns list of dicts with keys: delta_beta, beta_1, beta_2,
                                     marginal_flip, interventional_flip.
    """
    rng_master = np.random.default_rng(99)
    rows = []

    print("\n=== Delta-beta Sweep  (rho = {:.2f}, {:d} models per setting) ===".format(
        SWEEP_RHO, SWEEP_N_MODELS))
    print(f"\n  {'Δβ':>5}  {'β₁':>6}  {'β₂':>6}  {'Marginal flip':>14}  {'Interventional flip':>19}")
    print(f"  {'-'*58}")

    for delta_beta in SWEEP_DELTA_BETAS:
        beta_1 = 1.0 + delta_beta / 2.0
        beta_2 = 1.0 - delta_beta / 2.0

        marginal_rankings = []
        interventional_rankings = []

        for seed in range(SWEEP_N_MODELS):
            local_rng = np.random.default_rng(rng_master.integers(0, 2 ** 32))
            X, y = generate_data(SWEEP_RHO, beta_1, beta_2, N_SAMPLES, local_rng)

            n_train = int(TRAIN_FRAC * N_SAMPLES)
            X_train, y_train = X[:n_train], y[:n_train]
            X_eval = X[n_train:]

            bg_idx = np.random.default_rng(seed + 1000).choice(
                n_train, size=N_BACKGROUND, replace=False
            )
            X_bg = X_train[bg_idx]

            model = xgb.XGBRegressor(random_state=seed + 1000, **XGB_PARAMS)
            model.fit(X_train, y_train)

            ms_marginal = mean_abs_shap_marginal(model, X_eval)
            marginal_rankings.append(ms_marginal[0] > ms_marginal[1])

            ms_interv = mean_abs_shap_interventional(model, X_eval, X_bg)
            interventional_rankings.append(ms_interv[0] > ms_interv[1])

        # Use SWEEP_N_MODELS for pair counting
        n_pairs = SWEEP_N_MODELS * (SWEEP_N_MODELS - 1) // 2
        marg_flips = sum(1 for a, b in itertools.combinations(marginal_rankings, 2) if a != b)
        interv_flips = sum(1 for a, b in itertools.combinations(interventional_rankings, 2) if a != b)
        marg_flip = marg_flips / n_pairs if n_pairs > 0 else 0.0
        interv_flip = interv_flips / n_pairs if n_pairs > 0 else 0.0

        print(f"  {delta_beta:>5.2f}  {beta_1:>6.2f}  {beta_2:>6.2f}  "
              f"{marg_flip:>14.3f}  {interv_flip:>19.3f}")

        rows.append(dict(
            delta_beta=delta_beta,
            beta_1=beta_1,
            beta_2=beta_2,
            marginal_flip=marg_flip,
            interventional_flip=interv_flip,
        ))

    return rows


def print_sweep_crossover(sweep_rows: list):
    """
    Identify and print the crossover Delta_beta where marginal SHAP is still
    unstable (flip rate > SWEEP_MARGINAL_UNSTABLE_THRESHOLD) but interventional
    SHAP is already stable (flip rate < SWEEP_INTERVENTIONAL_STABLE_THRESHOLD).
    """
    print("\n--- Sweep crossover analysis ---")
    print(f"  Unstable threshold (marginal):       flip rate > {SWEEP_MARGINAL_UNSTABLE_THRESHOLD:.2f}")
    print(f"  Stable threshold (interventional):   flip rate < {SWEEP_INTERVENTIONAL_STABLE_THRESHOLD:.2f}")
    print()

    crossover_rows = [
        r for r in sweep_rows
        if r["marginal_flip"] > SWEEP_MARGINAL_UNSTABLE_THRESHOLD
        and r["interventional_flip"] < SWEEP_INTERVENTIONAL_STABLE_THRESHOLD
    ]

    if crossover_rows:
        print("  Delta_beta values where marginal is unstable BUT interventional is stable:")
        for r in crossover_rows:
            print(f"    Δβ={r['delta_beta']:.2f}  β₁={r['beta_1']:.2f}  β₂={r['beta_2']:.2f}"
                  f"  marginal={r['marginal_flip']:.3f}  interventional={r['interventional_flip']:.3f}")
        min_crossover = crossover_rows[0]["delta_beta"]
        print(f"\n  Crossover begins at Δβ = {min_crossover:.2f}")
    else:
        # Report the transition point even if there is no clean crossover
        marg_unstable = [r for r in sweep_rows if r["marginal_flip"] > SWEEP_MARGINAL_UNSTABLE_THRESHOLD]
        interv_stable  = [r for r in sweep_rows if r["interventional_flip"] < SWEEP_INTERVENTIONAL_STABLE_THRESHOLD]
        if marg_unstable:
            print(f"  Last Delta_beta with marginal unstable: "
                  f"Δβ={marg_unstable[-1]['delta_beta']:.2f}  "
                  f"(flip={marg_unstable[-1]['marginal_flip']:.3f})")
        if interv_stable:
            print(f"  First Delta_beta with interventional stable: "
                  f"Δβ={interv_stable[0]['delta_beta']:.2f}  "
                  f"(flip={interv_stable[0]['interventional_flip']:.3f})")
        print("  No clean crossover window found at these thresholds.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=== Conditional Attribution Impossibility — Experimental Validation ===")
    print(f"N_SAMPLES={N_SAMPLES}, N_MODELS={N_MODELS}, rho values={RHOS}")
    print(f"Marginal: tree_path_dependent perturbation")
    print(f"Interventional: interventional perturbation with {N_BACKGROUND} background samples")
    print(f"NOTE: Interventional SHAP approximates (but does not exactly implement)")
    print(f"      causal/conditional SHAP — see module docstring for details.")

    all_results = {}
    for scenario_name, (beta_1, beta_2) in SCENARIOS.items():
        all_results[scenario_name] = run_scenario(scenario_name, beta_1, beta_2)

    print_results_table(all_results)

    passed = run_validation(all_results)

    # ── Section 3: Delta-beta sweep ───────────────────────────────────────────
    print("\n" + "=" * 70)
    sweep_rows = run_sweep()
    print_sweep_crossover(sweep_rows)

    sys.exit(0 if passed else 1)
