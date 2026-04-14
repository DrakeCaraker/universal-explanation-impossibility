"""
Overparameterization Phase Transition Experiment
=================================================
Tests whether explanation instability (flip rate) undergoes a sharp phase
transition at the overparameterization ratio r* = dim(Theta)/dim(Y) ~ 1.

Part A: Linear (Ridge) models with exact control of r
Part B: Tree (XGBoost) models for practical relevance
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not available, skipping Part B")

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

RNG_SEED = 42
N_MODELS = 100


# ── Utility functions ────────────────────────────────────────────────────────

def generate_data(n_obs, n_features, noise_std=1.0, seed=0):
    """Synthetic regression with sparse true signal in first 5 features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_obs, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]
    y = X @ true_coef + noise_std * rng.randn(n_obs)
    return X, y, true_coef


def pairwise_flip_rate(importances_matrix):
    """
    Given (n_models, n_features) matrix of importances,
    compute mean pairwise ranking flip rate.

    For each pair of models, for each pair of features (i,j),
    a flip occurs if model_a ranks i>j but model_b ranks j>i.
    """
    n_models, n_feat = importances_matrix.shape
    # Subsample pairs for speed if needed
    n_pairs = min(500, n_models * (n_models - 1) // 2)
    rng = np.random.RandomState(99)

    flips_total = 0
    comparisons_total = 0

    indices = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            indices.append((i, j))
    indices = np.array(indices)
    if len(indices) > n_pairs:
        chosen = rng.choice(len(indices), size=n_pairs, replace=False)
        indices = indices[chosen]

    # For efficiency, subsample feature pairs if n_feat is large
    max_feat_pairs = 500
    if n_feat > 50:
        feat_indices_i = rng.randint(0, n_feat, size=max_feat_pairs)
        feat_indices_j = rng.randint(0, n_feat, size=max_feat_pairs)
        mask = feat_indices_i != feat_indices_j
        feat_indices_i = feat_indices_i[mask]
        feat_indices_j = feat_indices_j[mask]
    else:
        fi, fj = np.triu_indices(n_feat, k=1)
        feat_indices_i, feat_indices_j = fi, fj

    for (a, b) in indices:
        imp_a = importances_matrix[a]
        imp_b = importances_matrix[b]
        diff_a = imp_a[feat_indices_i] - imp_a[feat_indices_j]
        diff_b = imp_b[feat_indices_i] - imp_b[feat_indices_j]
        flips = np.sum(diff_a * diff_b < 0)
        ties = np.sum((diff_a == 0) | (diff_b == 0))
        flips_total += flips
        comparisons_total += len(feat_indices_i) - ties

    if comparisons_total == 0:
        return 0.0
    return flips_total / comparisons_total


def sigmoid(r, L, k, r_star):
    """Sigmoidal model for flip rate."""
    return L / (1.0 + np.exp(-k * (r - r_star)))


def fit_sigmoid(r_vals, flip_vals):
    """Fit sigmoid and return params + R^2."""
    try:
        # Initial guesses
        L0 = np.max(flip_vals)
        r_star0 = r_vals[np.argmin(np.abs(flip_vals - L0 / 2))]
        k0 = 5.0
        popt, _ = curve_fit(
            sigmoid, r_vals, flip_vals,
            p0=[L0, k0, r_star0],
            bounds=([0, 0.01, 0], [1.0, 200, 10]),
            maxfev=10000,
        )
        predicted = sigmoid(r_vals, *popt)
        ss_res = np.sum((flip_vals - predicted) ** 2)
        ss_tot = np.sum((flip_vals - np.mean(flip_vals)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return dict(L=popt[0], k=popt[1], r_star=popt[2], R2=r_sq)
    except Exception as e:
        return dict(L=np.nan, k=np.nan, r_star=np.nan, R2=np.nan, error=str(e))


def bootstrap_r_star(r_vals, flip_vals, n_boot=1000):
    """Bootstrap 95% CI on r_star."""
    rng = np.random.RandomState(123)
    r_stars = []
    n = len(r_vals)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        res = fit_sigmoid(r_vals[idx], flip_vals[idx])
        if not np.isnan(res["r_star"]):
            r_stars.append(res["r_star"])
    if len(r_stars) < 10:
        return (np.nan, np.nan)
    r_stars = np.array(r_stars)
    return (np.percentile(r_stars, 2.5), np.percentile(r_stars, 97.5))


# ── Part A: Linear (Ridge) models ────────────────────────────────────────────

def run_linear_experiment(n_obs, n_features_list, noise_std=1.0):
    """Run Part A for a given n_obs."""
    print(f"\n{'='*60}")
    print(f"Part A — Linear Ridge models (n_obs={n_obs})")
    print(f"{'='*60}")

    r_vals = []
    flip_rates = []
    test_mses = []

    for idx, n_feat in enumerate(n_features_list):
        r = n_feat / n_obs
        r_vals.append(r)

        X, y, true_coef = generate_data(n_obs * 2, n_feat, noise_std, seed=RNG_SEED)
        X_test, y_test = X[n_obs:], y[n_obs:]
        X_pool, y_pool = X[:n_obs], y[:n_obs]

        importances = np.zeros((N_MODELS, n_feat))
        mses = []

        for i in range(N_MODELS):
            rng_i = np.random.RandomState(RNG_SEED + i)
            boot_idx = rng_i.choice(n_obs, size=n_obs, replace=True)
            X_b, y_b = X_pool[boot_idx], y_pool[boot_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_b, y_b)
            importances[i] = np.abs(model.coef_)
            mses.append(mean_squared_error(y_test, model.predict(X_test)))

        fr = pairwise_flip_rate(importances)
        flip_rates.append(fr)
        test_mses.append(np.mean(mses))

        if idx % 5 == 0 or idx == len(n_features_list) - 1:
            print(f"  r={r:.3f} (p={n_feat:4d}) | flip_rate={fr:.4f} | test_MSE={np.mean(mses):.3f}")

    return np.array(r_vals), np.array(flip_rates), np.array(test_mses)


# ── Part B: Tree (XGBoost) models ────────────────────────────────────────────

def run_tree_experiment(n_obs=200, noise_std=1.0):
    """Run Part B with XGBoost."""
    if not HAS_XGB:
        return None, None, None

    print(f"\n{'='*60}")
    print(f"Part B — XGBoost tree models (n_obs={n_obs})")
    print(f"{'='*60}")

    max_features_list = np.unique(np.geomspace(5, 200, 20).astype(int))
    r_vals = []
    flip_rates = []
    test_mses = []

    for idx, n_feat in enumerate(max_features_list):
        r = n_feat / n_obs
        r_vals.append(r)

        X, y, _ = generate_data(n_obs * 2, n_feat, noise_std, seed=RNG_SEED)
        X_test, y_test = X[n_obs:], y[n_obs:]
        X_pool, y_pool = X[:n_obs], y[:n_obs]

        importances = np.zeros((N_MODELS, n_feat))
        mses = []

        for i in range(N_MODELS):
            rng_i = np.random.RandomState(RNG_SEED + i)
            boot_idx = rng_i.choice(n_obs, size=n_obs, replace=True)
            X_b, y_b = X_pool[boot_idx], y_pool[boot_idx]

            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=RNG_SEED + i,
                verbosity=0,
                n_jobs=1,
            )
            model.fit(X_b, y_b)
            importances[i] = model.feature_importances_
            mses.append(mean_squared_error(y_test, model.predict(X_test)))

        fr = pairwise_flip_rate(importances)
        flip_rates.append(fr)
        test_mses.append(np.mean(mses))

        if idx % 4 == 0 or idx == len(max_features_list) - 1:
            print(f"  r={r:.3f} (p={n_feat:4d}) | flip_rate={fr:.4f} | test_MSE={np.mean(mses):.3f}")

    return np.array(r_vals), np.array(flip_rates), np.array(test_mses)


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Overparameterization Phase Transition in Explanation Stability", fontsize=14)

    # --- Part A: n_obs=200 ---
    ax1 = axes[0, 0]
    ax1b = ax1.twinx()
    r_a = results["linear_200"]["r"]
    fr_a = results["linear_200"]["flip_rate"]
    mse_a = results["linear_200"]["test_mse"]

    ax1.plot(r_a, fr_a, "o-", color="C0", markersize=3, label="Flip rate")
    # Overlay sigmoid fit
    sig_a = results["linear_200"]["sigmoid"]
    if not np.isnan(sig_a["r_star"]):
        r_fine = np.linspace(min(r_a), max(r_a), 200)
        ax1.plot(r_fine, sigmoid(r_fine, sig_a["L"], sig_a["k"], sig_a["r_star"]),
                 "--", color="C0", alpha=0.7, label=f"Sigmoid (r*={sig_a['r_star']:.2f})")
        ax1.axvline(sig_a["r_star"], color="gray", ls=":", alpha=0.5)
    ax1b.plot(r_a, mse_a, "s-", color="C1", markersize=3, alpha=0.7, label="Test MSE")
    ax1.axvline(1.0, color="red", ls="--", alpha=0.4, label="r=1")
    ax1.set_xlabel("r = p / n")
    ax1.set_ylabel("Flip Rate", color="C0")
    ax1b.set_ylabel("Test MSE", color="C1")
    ax1.set_title(f"Linear Ridge (n=200) | r*={sig_a['r_star']:.3f}, R²={sig_a['R2']:.3f}")
    ax1.set_xscale("log")
    ax1.legend(loc="upper left", fontsize=8)
    ax1b.legend(loc="upper right", fontsize=8)

    # --- Part A: n_obs=500 ---
    ax2 = axes[0, 1]
    ax2b = ax2.twinx()
    r_a5 = results["linear_500"]["r"]
    fr_a5 = results["linear_500"]["flip_rate"]
    mse_a5 = results["linear_500"]["test_mse"]
    sig_a5 = results["linear_500"]["sigmoid"]

    ax2.plot(r_a5, fr_a5, "o-", color="C0", markersize=3, label="Flip rate")
    if not np.isnan(sig_a5["r_star"]):
        r_fine = np.linspace(min(r_a5), max(r_a5), 200)
        ax2.plot(r_fine, sigmoid(r_fine, sig_a5["L"], sig_a5["k"], sig_a5["r_star"]),
                 "--", color="C0", alpha=0.7, label=f"Sigmoid (r*={sig_a5['r_star']:.2f})")
        ax2.axvline(sig_a5["r_star"], color="gray", ls=":", alpha=0.5)
    ax2b.plot(r_a5, mse_a5, "s-", color="C1", markersize=3, alpha=0.7, label="Test MSE")
    ax2.axvline(1.0, color="red", ls="--", alpha=0.4, label="r=1")
    ax2.set_xlabel("r = p / n")
    ax2.set_ylabel("Flip Rate", color="C0")
    ax2b.set_ylabel("Test MSE", color="C1")
    ax2.set_title(f"Linear Ridge (n=500) | r*={sig_a5['r_star']:.3f}, R²={sig_a5['R2']:.3f}")
    ax2.set_xscale("log")
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    # --- Sharpness comparison ---
    ax3 = axes[1, 0]
    ax3.plot(r_a, fr_a, "o-", color="C0", markersize=3, alpha=0.7, label=f"n=200 (k={sig_a['k']:.1f})")
    ax3.plot(r_a5, fr_a5, "s-", color="C2", markersize=3, alpha=0.7, label=f"n=500 (k={sig_a5['k']:.1f})")
    ax3.axvline(1.0, color="red", ls="--", alpha=0.4)
    ax3.set_xlabel("r = p / n")
    ax3.set_ylabel("Flip Rate")
    ax3.set_title("Sharpness Comparison: n=200 vs n=500")
    ax3.set_xscale("log")
    ax3.legend(fontsize=9)

    # --- Part B: Trees ---
    ax4 = axes[1, 1]
    if "tree" in results and results["tree"]["r"] is not None:
        r_t = results["tree"]["r"]
        fr_t = results["tree"]["flip_rate"]
        sig_t = results["tree"]["sigmoid"]
        ax4.plot(r_t, fr_t, "o-", color="C3", markersize=3, label="Flip rate (XGBoost)")
        if not np.isnan(sig_t["r_star"]):
            r_fine = np.linspace(min(r_t), max(r_t), 200)
            ax4.plot(r_fine, sigmoid(r_fine, sig_t["L"], sig_t["k"], sig_t["r_star"]),
                     "--", color="C3", alpha=0.7, label=f"Sigmoid (r*={sig_t['r_star']:.2f})")
            ax4.axvline(sig_t["r_star"], color="gray", ls=":", alpha=0.5)
        ax4.axvline(1.0, color="red", ls="--", alpha=0.4, label="r=1")
        ax4.set_xlabel("r = p / n")
        ax4.set_ylabel("Flip Rate")
        ax4.set_title(f"XGBoost (n=200) | r*={sig_t['r_star']:.3f}, R²={sig_t['R2']:.3f}")
        ax4.set_xscale("log")
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, "XGBoost not available", transform=ax4.transAxes, ha="center")

    plt.tight_layout()
    fig_path = FIG_DIR / "phase_transition_r.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    results = {}

    # Feature counts: 25 log-spaced from 10 to 1000
    n_features_list_200 = np.unique(np.geomspace(10, 1000, 25).astype(int))
    n_features_list_500 = np.unique(np.geomspace(25, 2500, 25).astype(int))  # same r range for n=500

    # Part A: n_obs=200
    r200, fr200, mse200 = run_linear_experiment(200, n_features_list_200)
    sig200 = fit_sigmoid(r200, fr200)
    ci200 = bootstrap_r_star(r200, fr200)
    print(f"\n  Sigmoid fit: L={sig200['L']:.4f}, k={sig200['k']:.2f}, r*={sig200['r_star']:.4f}, R²={sig200['R2']:.4f}")
    print(f"  Bootstrap 95% CI on r*: [{ci200[0]:.3f}, {ci200[1]:.3f}]")

    results["linear_200"] = {
        "r": r200.tolist(), "flip_rate": fr200.tolist(), "test_mse": mse200.tolist(),
        "sigmoid": {k: float(v) if not isinstance(v, str) else v for k, v in sig200.items()},
        "r_star_ci95": [float(ci200[0]), float(ci200[1])],
    }

    # Part A: n_obs=500
    r500, fr500, mse500 = run_linear_experiment(500, n_features_list_500)
    sig500 = fit_sigmoid(r500, fr500)
    ci500 = bootstrap_r_star(r500, fr500)
    print(f"\n  Sigmoid fit: L={sig500['L']:.4f}, k={sig500['k']:.2f}, r*={sig500['r_star']:.4f}, R²={sig500['R2']:.4f}")
    print(f"  Bootstrap 95% CI on r*: [{ci500[0]:.3f}, {ci500[1]:.3f}]")

    results["linear_500"] = {
        "r": r500.tolist(), "flip_rate": fr500.tolist(), "test_mse": mse500.tolist(),
        "sigmoid": {k: float(v) if not isinstance(v, str) else v for k, v in sig500.items()},
        "r_star_ci95": [float(ci500[0]), float(ci500[1])],
    }

    # Sharpness comparison
    sharper = sig500["k"] > sig200["k"]
    print(f"\n  Sharpness comparison: k(n=200)={sig200['k']:.2f}, k(n=500)={sig500['k']:.2f}")
    print(f"  Transition sharper with larger n? {sharper}")

    # Part B: Trees
    r_t, fr_t, mse_t = run_tree_experiment(n_obs=200)
    if r_t is not None:
        sig_t = fit_sigmoid(r_t, fr_t)
        ci_t = bootstrap_r_star(r_t, fr_t)
        print(f"\n  Sigmoid fit: L={sig_t['L']:.4f}, k={sig_t['k']:.2f}, r*={sig_t['r_star']:.4f}, R²={sig_t['R2']:.4f}")
        print(f"  Bootstrap 95% CI on r*: [{ci_t[0]:.3f}, {ci_t[1]:.3f}]")
        results["tree"] = {
            "r": r_t.tolist(), "flip_rate": fr_t.tolist(), "test_mse": mse_t.tolist(),
            "sigmoid": {k: float(v) if not isinstance(v, str) else v for k, v in sig_t.items()},
            "r_star_ci95": [float(ci_t[0]), float(ci_t[1])],
        }
    else:
        results["tree"] = {"r": None, "flip_rate": None, "sigmoid": {}}

    # Tests
    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTS")
    print(f"{'='*60}")

    in_range_200 = 0.7 <= sig200["r_star"] <= 1.5
    print(f"  Linear r* in [0.7, 1.5]? {in_range_200} (r*={sig200['r_star']:.3f})")

    in_range_500 = 0.7 <= sig500["r_star"] <= 1.5
    print(f"  Linear (n=500) r* in [0.7, 1.5]? {in_range_500} (r*={sig500['r_star']:.3f})")

    print(f"  Transition sharper with larger n? {sharper} (k200={sig200['k']:.1f}, k500={sig500['k']:.1f})")

    if r_t is not None:
        in_range_t = 0.7 <= sig_t["r_star"] <= 1.5
        print(f"  Tree r* in [0.7, 1.5]? {in_range_t} (r*={sig_t['r_star']:.3f})")

    results["hypothesis_tests"] = {
        "linear_r_star_in_range": bool(in_range_200),
        "sharper_with_larger_n": bool(sharper),
        "tree_r_star_in_range": bool(in_range_t) if r_t is not None else None,
    }

    elapsed = time.time() - t0
    results["runtime_seconds"] = round(elapsed, 1)
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Save results
    res_path = OUT_DIR / "results_phase_transition_r.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {res_path}")

    # Plot
    make_figure(results)


if __name__ == "__main__":
    main()
