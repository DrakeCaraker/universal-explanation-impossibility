"""
Noether Correspondence Counting Experiment
===========================================
Tests the prediction: for P features in g correlation groups,
exactly g(g-1)/2 independent group-level ranking facts are stable,
while all within-group comparisons are unstable (~50% flip rate).

Setup: P=12 features, g=3 groups of 4, ρ_within=0.9, ρ_between=0.1
       True β: Group1=[3,3,3,3], Group2=[1,1,1,1], Group3=[0.5,0.5,0.5,0.5]
"""

import sys, json, time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add experiment_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper" / "scripts"))
from experiment_utils import set_all_seeds, percentile_ci

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
P = 12                  # total features
G = 3                   # number of groups
GROUP_SIZE = P // G     # 4 features per group
RHO_WITHIN = 0.9
RHO_BETWEEN = 0.1
BETAS = np.array([3.0]*4 + [1.0]*4 + [0.5]*4)
N_TRAIN = 500
N_TEST = 200
NOISE_STD = 1.0
N_MODELS = 200
SEED = 42
FLIP_THRESHOLD = 0.05  # pairs with flip rate < this are "stable"

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Group membership
# ---------------------------------------------------------------------------
groups = np.array([i // GROUP_SIZE for i in range(P)])  # [0,0,0,0, 1,1,1,1, 2,2,2,2]

def pair_type(i, j):
    """Return 'within' or 'between' for feature pair (i, j)."""
    return "within" if groups[i] == groups[j] else "between"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(n, seed):
    """Generate correlated Gaussian features with linear response."""
    rng = np.random.default_rng(seed)

    # Build correlation matrix: block structure
    Sigma = np.full((P, P), RHO_BETWEEN)
    for g in range(G):
        idx = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        Sigma[idx, idx] = RHO_WITHIN
    np.fill_diagonal(Sigma, 1.0)

    # Cholesky decomposition for correlated features
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, P))
    X = Z @ L.T

    # Response
    y = X @ BETAS + rng.normal(0, NOISE_STD, n)
    return X, y

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment():
    set_all_seeds(SEED)
    print("=" * 70)
    print("Noether Correspondence Counting Experiment")
    print("=" * 70)

    # Generate data
    X_all, y_all = generate_data(N_TRAIN + N_TEST, seed=SEED)
    X_train, X_test = X_all[:N_TRAIN], X_all[N_TRAIN:]
    y_train, y_test = y_all[:N_TRAIN], y_all[N_TRAIN:]

    print(f"\nData: {N_TRAIN} train, {N_TEST} test, {P} features, {G} groups")
    print(f"Correlation: ρ_within={RHO_WITHIN}, ρ_between={RHO_BETWEEN}")
    print(f"True β: {BETAS.tolist()}")

    # Train Rashomon set
    # Each model gets a bootstrap resample of the training data PLUS
    # XGBoost internal subsampling to create genuine feature-importance variability.
    print(f"\nTraining {N_MODELS} XGBoost models (bootstrap + subsample)...")
    importances = np.zeros((N_MODELS, P))
    r2_scores = []
    for i in range(N_MODELS):
        rng_i = np.random.default_rng(SEED + i)
        # Bootstrap resample of training data
        boot_idx = rng_i.choice(N_TRAIN, size=N_TRAIN, replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=SEED + i,
            verbosity=0,
        )
        model.fit(X_boot, y_boot)
        importances[i] = model.feature_importances_
        r2_scores.append(model.score(X_test, y_test))

    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    print(f"R² = {r2_mean:.4f} ± {r2_std:.4f}")

    # Compute rankings (rank 1 = most important)
    rankings = np.zeros_like(importances, dtype=int)
    for i in range(N_MODELS):
        rankings[i] = stats.rankdata(-importances[i], method="average").astype(int)

    # -------------------------------------------------------------------
    # Pairwise flip rates
    # -------------------------------------------------------------------
    print(f"\nComputing pairwise flip rates for {P*(P-1)//2} pairs...")
    n_model_pairs = N_MODELS * (N_MODELS - 1) // 2

    within_rates = []
    between_rates = []
    all_pair_info = []

    for j in range(P):
        for k in range(j + 1, P):
            # Vectorised: for each model, does feature j have higher importance than k?
            j_gt_k = importances[:, j] > importances[:, k]  # shape (N_MODELS,)

            # Count disagreements across all model pairs
            n_true = j_gt_k.sum()
            n_false = N_MODELS - n_true
            # Number of discordant pairs = n_true * n_false
            flips = int(n_true * n_false)
            rate = flips / n_model_pairs

            pt = pair_type(j, k)
            info = {
                "i": j, "j": k,
                "feature_i": f"X{j}", "feature_j": f"X{k}",
                "group_i": int(groups[j]), "group_j": int(groups[k]),
                "type": pt,
                "flip_rate": round(rate, 6),
            }
            all_pair_info.append(info)

            if pt == "within":
                within_rates.append(rate)
            else:
                between_rates.append(rate)

    within_rates = np.array(within_rates)
    between_rates = np.array(between_rates)
    all_rates = np.array([p["flip_rate"] for p in all_pair_info])

    # -------------------------------------------------------------------
    # Bootstrap CIs
    # -------------------------------------------------------------------
    set_all_seeds(SEED + 999)  # separate seed for bootstrap
    within_ci = percentile_ci(within_rates, n_boot=5000)
    between_ci = percentile_ci(between_rates, n_boot=5000)

    # -------------------------------------------------------------------
    # Stability counts
    # -------------------------------------------------------------------
    n_within = len(within_rates)
    n_between = len(between_rates)
    n_within_stable = int((within_rates < FLIP_THRESHOLD).sum())
    n_within_unstable = n_within - n_within_stable
    n_between_stable = int((between_rates < FLIP_THRESHOLD).sum())
    n_between_unstable = n_between - n_between_stable
    total_stable = n_within_stable + n_between_stable

    # Expected from Noether correspondence
    expected_between_stable = n_between  # all 18 between-group pairs
    # But there are only g(g-1)/2 = 3 INDEPENDENT group-level facts
    n_independent_group_facts = G * (G - 1) // 2

    # -------------------------------------------------------------------
    # Hartigan's dip test for bimodality
    # -------------------------------------------------------------------
    def hartigan_dip_test(data, n_boot=1000):
        """
        Approximate Hartigan's dip test using the empirical CDF.
        The dip statistic measures the maximum deviation from unimodality.
        """
        sorted_data = np.sort(data)
        n = len(sorted_data)
        # Empirical CDF
        ecdf = np.arange(1, n + 1) / n

        # Greatest convex minorant and least concave majorant
        # Simplified: measure max gap between sorted data quantiles
        # and a uniform distribution (unimodal reference)
        uniform_cdf = np.linspace(0, 1, n)

        # Dip = max distance between ECDF and best-fitting unimodal CDF
        # Approximation: use the maximum of |ECDF - uniform_on_range|
        data_range = sorted_data[-1] - sorted_data[0]
        if data_range == 0:
            return 0.0, 1.0
        normalized = (sorted_data - sorted_data[0]) / data_range
        dip_stat = np.max(np.abs(ecdf - normalized))

        # Bootstrap p-value under uniformity
        rng = np.random.default_rng(SEED + 7777)
        dip_boots = []
        for _ in range(n_boot):
            boot = np.sort(rng.uniform(0, 1, n))
            boot_ecdf = np.arange(1, n + 1) / n
            boot_dip = np.max(np.abs(boot_ecdf - boot))
            dip_boots.append(boot_dip)
        p_value = np.mean(np.array(dip_boots) >= dip_stat)
        return float(dip_stat), float(p_value)

    dip_stat, dip_p = hartigan_dip_test(all_rates)

    # Also test with KS 2-sample: within vs between distributions
    ks_stat, ks_p = stats.ks_2samp(within_rates, between_rates)

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nPair counts: {n_within} within-group, {n_between} between-group, {n_within + n_between} total")
    print(f"  Expected: {G * (GROUP_SIZE * (GROUP_SIZE-1) // 2)} within, "
          f"{P*(P-1)//2 - G * (GROUP_SIZE * (GROUP_SIZE-1) // 2)} between")

    print(f"\nMean flip rates:")
    print(f"  Within-group:  {within_ci[1]:.4f}  95% CI [{within_ci[0]:.4f}, {within_ci[2]:.4f}]")
    print(f"  Between-group: {between_ci[1]:.4f}  95% CI [{between_ci[0]:.4f}, {between_ci[2]:.4f}]")

    print(f"\nStability (flip rate < {FLIP_THRESHOLD}):")
    print(f"  Within-group:  {n_within_stable}/{n_within} stable  ({n_within_unstable} unstable)")
    print(f"  Between-group: {n_between_stable}/{n_between} stable ({n_between_unstable} unstable)")
    print(f"  Total stable pairs: {total_stable}/{n_within + n_between}")

    print(f"\nNoether prediction:")
    print(f"  Independent group-level facts: g(g-1)/2 = {n_independent_group_facts}")
    print(f"  Between-group pairs stable: {n_between_stable}/{n_between} "
          f"(expected: {expected_between_stable}/{n_between})")
    print(f"  Within-group pairs unstable: {n_within_unstable}/{n_within} "
          f"(expected: {n_within}/{n_within})")

    print(f"\nBimodality tests:")
    print(f"  Dip test: statistic={dip_stat:.4f}, p={dip_p:.4f}")
    print(f"  KS 2-sample (within vs between): statistic={ks_stat:.4f}, p={ks_p:.2e}")

    # Per-pair detail for within-group
    print(f"\nWithin-group pair flip rates:")
    for p in all_pair_info:
        if p["type"] == "within":
            print(f"  {p['feature_i']} vs {p['feature_j']} (group {p['group_i']}): {p['flip_rate']:.4f}")

    # Per-pair detail for between-group
    print(f"\nBetween-group pair flip rates (summary):")
    bg_by_groups = {}
    for p in all_pair_info:
        if p["type"] == "between":
            key = (min(p["group_i"], p["group_j"]), max(p["group_i"], p["group_j"]))
            bg_by_groups.setdefault(key, []).append(p["flip_rate"])
    for (g1, g2), rates in sorted(bg_by_groups.items()):
        print(f"  Group {g1} vs Group {g2}: mean={np.mean(rates):.4f}, "
              f"range=[{min(rates):.4f}, {max(rates):.4f}], "
              f"all_stable={all(r < FLIP_THRESHOLD for r in rates)}")

    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Histogram
    ax = axes[0]
    bins = np.linspace(0, 0.55, 30)
    ax.hist(within_rates, bins=bins, alpha=0.7, color="#e74c3c", label="Within-group", edgecolor="white")
    ax.hist(between_rates, bins=bins, alpha=0.7, color="#2980b9", label="Between-group", edgecolor="white")
    ax.axvline(FLIP_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"Threshold ({FLIP_THRESHOLD})")
    ax.set_xlabel("Pairwise Flip Rate", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("A. Distribution of Pairwise Flip Rates", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Annotate
    ax.annotate(f"Within: mean={within_ci[1]:.3f}",
                xy=(within_ci[1], 0), xytext=(within_ci[1] + 0.02, ax.get_ylim()[1] * 0.8),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=9, color="#e74c3c")

    # Panel B: Sorted flip rates colored by type
    ax = axes[1]
    sorted_pairs = sorted(all_pair_info, key=lambda x: x["flip_rate"])
    colors = ["#2980b9" if p["type"] == "between" else "#e74c3c" for p in sorted_pairs]
    rates_sorted = [p["flip_rate"] for p in sorted_pairs]
    ax.bar(range(len(sorted_pairs)), rates_sorted, color=colors, width=1.0, edgecolor="none")
    ax.axhline(FLIP_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Feature Pair (sorted by flip rate)", fontsize=12)
    ax.set_ylabel("Flip Rate", fontsize=12)
    ax.set_title("B. Noether Counting: Stable vs Unstable Pairs", fontsize=13, fontweight="bold")

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2980b9", label=f"Between-group ({n_between_stable}/{n_between} stable)"),
        Patch(facecolor="#e74c3c", label=f"Within-group ({n_within_stable}/{n_within} stable)"),
    ], fontsize=9, loc="upper left")

    # Annotation: g(g-1)/2
    ax.text(0.98, 0.95, f"Independent group facts:\n$g(g-1)/2 = {n_independent_group_facts}$",
            transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    fig.tight_layout()
    fig_path = FIG_DIR / "noether_counting.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved figure: {fig_path}")
    plt.close(fig)

    # -------------------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------------------
    results = {
        "experiment": "noether_counting",
        "config": {
            "P": P, "G": G, "group_size": GROUP_SIZE,
            "rho_within": RHO_WITHIN, "rho_between": RHO_BETWEEN,
            "betas": BETAS.tolist(),
            "n_train": N_TRAIN, "n_test": N_TEST,
            "noise_std": NOISE_STD,
            "n_models": N_MODELS, "seed": SEED,
            "flip_threshold": FLIP_THRESHOLD,
        },
        "model_performance": {
            "r2_mean": round(r2_mean, 4),
            "r2_std": round(r2_std, 4),
        },
        "flip_rates": {
            "within_group": {
                "mean": round(within_ci[1], 6),
                "ci_lower": round(within_ci[0], 6),
                "ci_upper": round(within_ci[2], 6),
                "n_pairs": n_within,
                "n_stable": n_within_stable,
                "n_unstable": n_within_unstable,
                "all_rates": [round(r, 6) for r in sorted(within_rates)],
            },
            "between_group": {
                "mean": round(between_ci[1], 6),
                "ci_lower": round(between_ci[0], 6),
                "ci_upper": round(between_ci[2], 6),
                "n_pairs": n_between,
                "n_stable": n_between_stable,
                "n_unstable": n_between_unstable,
                "all_rates": [round(r, 6) for r in sorted(between_rates)],
            },
        },
        "noether_prediction": {
            "g_g_minus_1_over_2": n_independent_group_facts,
            "between_group_all_stable": bool(n_between_stable == n_between),
            "within_group_all_unstable": bool(n_within_unstable == n_within),
            "prediction_confirmed": bool(
                n_between_stable == n_between and n_within_unstable == n_within
            ),
        },
        "bimodality": {
            "dip_statistic": round(dip_stat, 6),
            "dip_p_value": round(dip_p, 6),
            "ks_statistic": round(ks_stat, 6),
            "ks_p_value": float(f"{ks_p:.6e}"),
        },
        "between_group_detail": {
            f"group_{g1}_vs_{g2}": {
                "mean_flip_rate": round(np.mean(rates), 6),
                "min_flip_rate": round(min(rates), 6),
                "max_flip_rate": round(max(rates), 6),
                "all_stable": all(r < FLIP_THRESHOLD for r in rates),
                "n_pairs": len(rates),
            }
            for (g1, g2), rates in sorted(bg_by_groups.items())
        },
        "pair_details": all_pair_info,
        "_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = OUT_DIR / "results_noether_counting.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    # -------------------------------------------------------------------
    # Summary verdict
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if results["noether_prediction"]["prediction_confirmed"]:
        print("CONFIRMED: Noether correspondence prediction holds.")
        print(f"  - All {n_between} between-group pairs are stable (flip rate < {FLIP_THRESHOLD})")
        print(f"  - All {n_within} within-group pairs are unstable (flip rate >= {FLIP_THRESHOLD})")
        print(f"  - {n_independent_group_facts} independent group-level ranking facts")
    else:
        print("PARTIAL or FAILED:")
        print(f"  - Between-group stable: {n_between_stable}/{n_between}")
        print(f"  - Within-group unstable: {n_within_unstable}/{n_within}")

    return results


if __name__ == "__main__":
    results = run_experiment()
