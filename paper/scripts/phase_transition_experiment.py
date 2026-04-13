"""Phase transition experiment: Rashomon tolerance vs explanation stability.

Tests the cross-domain prediction from the gauge theory experiment:
coupling β controls explanation stability (variance = sech²(β)).
The framework predicts an analogous phase transition in ML —
as Rashomon tolerance ε increases, explanation stability undergoes
a transition from stable to unstable. The critical tolerance ε_c
should decrease with collinearity ρ.

Predictions:
  (a) Stability decreases monotonically with ε (dose-response)
  (b) Transition is sharper at higher ρ (larger Rashomon sets)
  (c) Critical ε_c is smaller at higher ρ
  (d) Curve shape is sigmoid-like (analogous to sech²(β))
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
STYLE_FILE = SCRIPT_DIR / "publication_style.mplstyle"
RESULTS_FILE = PAPER_DIR / "results_phase_transition.json"
FIGURE_FILE = PAPER_DIR / "figures" / "phase_transition.pdf"

sys.path.insert(0, str(SCRIPT_DIR))
from provenance import stamp

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_TRAIN = 500
N_TEST = 200
D = 5
TRUE_BETA = np.array([3.0, 2.0, 1.0, 0.5, 0.2])
NOISE_STD = 1.0

COLLINEARITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_MODELS = 200
LAMBDA_VALUES = np.logspace(-2, 1, 20)  # 0.01 to 10
EPSILON_VALUES = np.logspace(np.log10(0.01), np.log10(2.0), 20)
STABILITY_THRESHOLD = 0.5  # for marking critical ε

N_SHAP_POINTS = 100  # test points for SHAP averaging


def make_correlated_X(n: int, d: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Generate n samples of d features with pairwise correlation rho."""
    cov = np.full((d, d), rho) + np.eye(d) * (1 - rho)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, d))
    return Z @ L.T


def compute_linear_shap(coefs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Linear SHAP = coefficient * feature value, averaged over points.

    Returns feature importance = mean |coef_j * x_j| for each feature j.
    """
    # Shape: (n_points, d)
    shap_vals = np.abs(coefs[np.newaxis, :] * X)
    # Average over points -> (d,)
    return shap_vals.mean(axis=0)


def run_experiment():
    """Run the full phase transition experiment."""
    results = {
        "metadata": {
            "seed": SEED,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "d": D,
            "true_beta": TRUE_BETA.tolist(),
            "noise_std": NOISE_STD,
            "n_models": N_MODELS,
            "n_shap_points": N_SHAP_POINTS,
            "stability_threshold": STABILITY_THRESHOLD,
        },
        "epsilon_values": EPSILON_VALUES.tolist(),
        "collinearity_levels": COLLINEARITY_LEVELS,
        "curves": {},
        "critical_epsilon": {},
        "rashomon_sizes": {},
        "predictions": {},
    }

    for rho in COLLINEARITY_LEVELS:
        print(f"\n{'='*60}")
        print(f"Collinearity rho = {rho}")
        print(f"{'='*60}")

        # Generate data
        X_train = make_correlated_X(N_TRAIN, D, rho, rng)
        X_test = make_correlated_X(N_TEST, D, rho, rng)
        y_train = X_train @ TRUE_BETA + rng.normal(0, NOISE_STD, N_TRAIN)
        y_test = X_test @ TRUE_BETA + rng.normal(0, NOISE_STD, N_TEST)

        # SHAP evaluation points (subset of test)
        X_shap = X_test[:N_SHAP_POINTS]

        # -----------------------------------------------------------------
        # Build pool of candidate models:
        # Vary regularization AND feature subsets
        # -----------------------------------------------------------------
        model_pool = []  # list of (mse_test, coefs_full)
        feature_subsets = []

        # All subsets of size >= 2
        for size in range(2, D + 1):
            for subset in combinations(range(D), size):
                feature_subsets.append(list(subset))

        for lam in LAMBDA_VALUES:
            for subset in feature_subsets:
                model = Ridge(alpha=lam, fit_intercept=True)
                model.fit(X_train[:, subset], y_train)
                y_pred = model.predict(X_test[:, subset])
                mse = np.mean((y_test - y_pred) ** 2)

                # Expand coefficients to full d-dimensional vector
                coefs_full = np.zeros(D)
                for i, feat_idx in enumerate(subset):
                    coefs_full[feat_idx] = model.coef_[i]

                model_pool.append((mse, coefs_full))

        # Sort by MSE
        model_pool.sort(key=lambda x: x[0])
        best_mse = model_pool[0][0]
        print(f"  Pool size: {len(model_pool)}, Best MSE: {best_mse:.4f}")

        # -----------------------------------------------------------------
        # Sweep epsilon: compute Rashomon set + stability at each level
        # -----------------------------------------------------------------
        stabilities = []
        rashomon_sizes = []

        for eps in EPSILON_VALUES:
            # Rashomon set: models within eps of best
            rashomon_set = [
                (mse, coefs) for mse, coefs in model_pool
                if mse <= best_mse + eps
            ]
            rash_size = len(rashomon_set)
            rashomon_sizes.append(rash_size)

            if rash_size < 2:
                stabilities.append(1.0)
                continue

            # Compute SHAP importances for each Rashomon model
            shap_importances = []
            for _, coefs in rashomon_set:
                imp = compute_linear_shap(coefs, X_shap)
                shap_importances.append(imp)

            # Pairwise Spearman rank correlations
            pairwise_rhos = []
            n_models_rash = len(shap_importances)
            # Cap comparisons to avoid combinatorial explosion
            max_pairs = 500
            if n_models_rash * (n_models_rash - 1) // 2 > max_pairs:
                # Sample pairs
                indices = list(range(n_models_rash))
                for _ in range(max_pairs):
                    i, j = rng.choice(indices, size=2, replace=False)
                    corr, _ = spearmanr(shap_importances[i], shap_importances[j])
                    if not np.isnan(corr):
                        pairwise_rhos.append(corr)
            else:
                for i in range(n_models_rash):
                    for j in range(i + 1, n_models_rash):
                        corr, _ = spearmanr(shap_importances[i], shap_importances[j])
                        if not np.isnan(corr):
                            pairwise_rhos.append(corr)

            stability = np.mean(pairwise_rhos) if pairwise_rhos else 1.0
            stabilities.append(stability)

        # Find critical epsilon (where stability drops below threshold)
        critical_eps = None
        for i, (eps, stab) in enumerate(zip(EPSILON_VALUES, stabilities)):
            if stab < STABILITY_THRESHOLD:
                critical_eps = float(eps)
                break

        rho_key = f"rho_{rho}"
        results["curves"][rho_key] = stabilities
        results["critical_epsilon"][rho_key] = critical_eps
        results["rashomon_sizes"][rho_key] = rashomon_sizes

        print(f"  Stability range: [{min(stabilities):.3f}, {max(stabilities):.3f}]")
        print(f"  Critical epsilon: {critical_eps}")
        print(f"  Rashomon sizes: {rashomon_sizes[0]} -> {rashomon_sizes[-1]}")

    # -------------------------------------------------------------------
    # Evaluate predictions
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATING PREDICTIONS")
    print(f"{'='*60}")

    # (a) Monotonicity: stability decreases with epsilon
    monotonic_results = {}
    for rho in COLLINEARITY_LEVELS:
        rho_key = f"rho_{rho}"
        curve = results["curves"][rho_key]
        # Check if generally decreasing (allow small violations)
        diffs = np.diff(curve)
        frac_decreasing = np.mean(diffs <= 0.01)  # allow tiny increases
        monotonic_results[rho_key] = {
            "fraction_decreasing": float(frac_decreasing),
            "passes": bool(frac_decreasing > 0.7),
        }
    results["predictions"]["monotonicity"] = monotonic_results

    # (b) Sharper transition at higher rho
    # Measure sharpness as max |d(stability)/d(log_epsilon)|
    sharpness = {}
    for rho in COLLINEARITY_LEVELS:
        rho_key = f"rho_{rho}"
        curve = np.array(results["curves"][rho_key])
        log_eps = np.log10(EPSILON_VALUES)
        grad = np.abs(np.gradient(curve, log_eps))
        sharpness[rho_key] = float(np.max(grad))
    results["predictions"]["sharpness"] = sharpness
    sharpness_increases = all(
        sharpness[f"rho_{COLLINEARITY_LEVELS[i]}"]
        <= sharpness[f"rho_{COLLINEARITY_LEVELS[i+1]}"] + 0.05  # small tolerance
        for i in range(len(COLLINEARITY_LEVELS) - 1)
    )
    results["predictions"]["sharpness_increases_with_rho"] = bool(sharpness_increases)

    # (c) Critical epsilon decreases with rho
    crits = [results["critical_epsilon"][f"rho_{rho}"] for rho in COLLINEARITY_LEVELS]
    # Replace None with inf for comparison
    crits_compare = [c if c is not None else float("inf") for c in crits]
    eps_c_decreases = all(
        crits_compare[i] >= crits_compare[i + 1]
        for i in range(len(crits_compare) - 1)
    )
    results["predictions"]["critical_eps_decreases"] = bool(eps_c_decreases)
    results["predictions"]["critical_eps_values"] = {
        f"rho_{rho}": crits[i] for i, rho in enumerate(COLLINEARITY_LEVELS)
    }

    # (d) Sigmoid-like shape: check via fitting logistic
    # Simple test: curve should have inflection point
    sigmoid_like = {}
    for rho in COLLINEARITY_LEVELS:
        rho_key = f"rho_{rho}"
        curve = np.array(results["curves"][rho_key])
        # Second derivative: does it change sign? (inflection point)
        log_eps = np.log10(EPSILON_VALUES)
        first_deriv = np.gradient(curve, log_eps)
        second_deriv = np.gradient(first_deriv, log_eps)
        sign_changes = np.sum(np.diff(np.sign(second_deriv)) != 0)
        sigmoid_like[rho_key] = {
            "inflection_sign_changes": int(sign_changes),
            "is_sigmoid_like": bool(sign_changes >= 1),
        }
    results["predictions"]["sigmoid_shape"] = sigmoid_like

    # Overall verdict
    all_mono = all(v["passes"] for v in monotonic_results.values())
    all_sigmoid = all(v["is_sigmoid_like"] for v in sigmoid_like.values())
    results["predictions"]["overall"] = {
        "phase_transition_exists": bool(all_mono),
        "eps_c_decreases_with_rho": bool(eps_c_decreases),
        "sharper_at_higher_rho": bool(sharpness_increases),
        "sigmoid_shape": bool(all_sigmoid),
    }

    for key, val in results["predictions"]["overall"].items():
        status = "CONFIRMED" if val else "NOT CONFIRMED"
        print(f"  {key}: {status}")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, rho in enumerate(COLLINEARITY_LEVELS):
        rho_key = f"rho_{rho}"
        curve = results["curves"][rho_key]
        crit = results["critical_epsilon"][rho_key]
        ax.plot(
            EPSILON_VALUES,
            curve,
            color=colors[i],
            label=rf"$\rho = {rho}$",
            marker="o",
            markersize=2.5,
        )
        # Mark critical epsilon
        if crit is not None:
            # Find the stability at critical epsilon
            idx = np.argmin(np.abs(EPSILON_VALUES - crit))
            ax.axvline(
                crit,
                color=colors[i],
                linestyle=":",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.plot(
                crit,
                curve[idx],
                marker="v",
                color=colors[i],
                markersize=5,
                zorder=5,
            )

    ax.axhline(
        STABILITY_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.text(
        EPSILON_VALUES[-1] * 0.7,
        STABILITY_THRESHOLD + 0.03,
        r"$\tau = 0.5$",
        fontsize=7,
        color="gray",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Rashomon tolerance $\varepsilon$")
    ax.set_ylabel(r"Stability (mean Spearman $\rho$)")
    ax.set_title("Phase Transition in Explanation Stability")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_ylim(-0.15, 1.05)

    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGURE_FILE))
    print(f"Figure saved to {FIGURE_FILE}")
    stamp(str(FIGURE_FILE), __file__)
    stamp(str(RESULTS_FILE), __file__)

    plt.close(fig)
    return results


if __name__ == "__main__":
    results = run_experiment()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    overall = results["predictions"]["overall"]
    for k, v in overall.items():
        tag = "YES" if v else "NO"
        print(f"  {k}: {tag}")
