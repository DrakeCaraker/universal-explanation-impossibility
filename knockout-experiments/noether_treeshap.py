"""
Phase 1 Knockout: Noether Counting + SAGE with TreeSHAP
========================================================
Validates that bimodal flip-rate gap persists when using TreeSHAP
importance (mean |SHAP|) instead of gain-based feature_importances_.

Task 1: Noether counting over rho_within in {0.50, 0.70, 0.99}
Task 2: SAGE on Breast Cancer with TreeSHAP vs gain-based importance

GO/NO-GO: SHAP bimodal gap > 30pp at rho=0.70
"""

import sys, json, time, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# Add experiment_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper" / "scripts"))
from experiment_utils import set_all_seeds, percentile_ci

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ===================================================================
# Task 1: Noether Counting with TreeSHAP
# ===================================================================

# Configuration
P = 12
G = 3
GROUP_SIZE = P // G
BETAS = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
RHO_WITHIN_VALUES = [0.50, 0.70, 0.99]
RHO_BETWEEN = 0.0
N_TRAIN = 500
N_TEST = 200
NOISE_STD = 1.0
N_MODELS = 50
SEED = 42

groups = np.array([i // GROUP_SIZE for i in range(P)])


def pair_type(i, j):
    return "within" if groups[i] == groups[j] else "between"


def generate_data(n, rho_within, seed):
    """Generate correlated Gaussian features with linear response."""
    rng = np.random.default_rng(seed)
    Sigma = np.full((P, P), RHO_BETWEEN)
    for g in range(G):
        idx = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        Sigma[idx, idx] = rho_within
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, P))
    X = Z @ L.T
    y = X @ BETAS + rng.normal(0, NOISE_STD, n)
    return X, y


def compute_flip_rates_from_importance(importance_matrix):
    """Compute pairwise flip rates from an (N_MODELS, P) importance matrix."""
    M, P_dim = importance_matrix.shape
    n_model_pairs = M * (M - 1) // 2
    within_rates = []
    between_rates = []
    all_pair_info = []

    for j in range(P_dim):
        for k in range(j + 1, P_dim):
            j_gt_k = importance_matrix[:, j] > importance_matrix[:, k]
            n_true = j_gt_k.sum()
            n_false = M - n_true
            flips = int(n_true * n_false)
            rate = flips / n_model_pairs if n_model_pairs > 0 else 0.0

            pt = pair_type(j, k)
            all_pair_info.append({
                "i": j, "j": k, "type": pt, "flip_rate": round(rate, 6),
                "group_i": int(groups[j]), "group_j": int(groups[k]),
            })
            if pt == "within":
                within_rates.append(rate)
            else:
                between_rates.append(rate)

    return np.array(within_rates), np.array(between_rates), all_pair_info


def run_noether_treeshap():
    print("=" * 70)
    print("TASK 1: Noether Counting with TreeSHAP")
    print("=" * 70)

    all_results = {}

    for rho_w in RHO_WITHIN_VALUES:
        print(f"\n{'─'*60}")
        print(f"ρ_within = {rho_w}")
        print(f"{'─'*60}")

        set_all_seeds(SEED)
        X_all, y_all = generate_data(N_TRAIN + N_TEST, rho_w, seed=SEED)
        X_train, X_test = X_all[:N_TRAIN], X_all[N_TRAIN:]
        y_train, y_test = y_all[:N_TRAIN], y_all[N_TRAIN:]

        # Storage for both importance types
        shap_importances = np.zeros((N_MODELS, P))
        gain_importances = np.zeros((N_MODELS, P))
        r2_scores = []

        print(f"  Training {N_MODELS} XGBoost models...")
        for i in range(N_MODELS):
            rng_i = np.random.default_rng(SEED + i)
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

            # Gain-based importance
            gain_importances[i] = model.feature_importances_

            # TreeSHAP importance
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:200])
            if isinstance(shap_values, list):
                importance = np.mean(
                    [np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0
                )
            else:
                importance = np.mean(np.abs(shap_values), axis=0)
            shap_importances[i] = importance

            r2_scores.append(model.score(X_test, y_test))

        r2_mean = np.mean(r2_scores)
        r2_std = np.std(r2_scores)
        print(f"  R² = {r2_mean:.4f} ± {r2_std:.4f}")

        # Compute flip rates for both methods
        print("  Computing flip rates...")
        shap_within, shap_between, shap_pairs = compute_flip_rates_from_importance(
            shap_importances
        )
        gain_within, gain_between, gain_pairs = compute_flip_rates_from_importance(
            gain_importances
        )

        # Bootstrap CIs
        set_all_seeds(SEED + 999)
        shap_within_ci = percentile_ci(shap_within, n_boot=5000)
        shap_between_ci = percentile_ci(shap_between, n_boot=5000)
        gain_within_ci = percentile_ci(gain_within, n_boot=5000)
        gain_between_ci = percentile_ci(gain_between, n_boot=5000)

        # Bimodal gap = within_mean - between_mean
        shap_gap = shap_within_ci[1] - shap_between_ci[1]
        gain_gap = gain_within_ci[1] - gain_between_ci[1]

        # Permutation test for SHAP bimodality
        all_shap_rates = np.concatenate([shap_within, shap_between])
        all_labels = np.array(
            ["within"] * len(shap_within) + ["between"] * len(shap_between)
        )
        observed_diff = np.mean(shap_within) - np.mean(shap_between)
        n_perm = 10000
        rng_perm = np.random.default_rng(SEED + 7777)
        perm_diffs = np.zeros(n_perm)
        n_within = len(shap_within)
        for p in range(n_perm):
            perm = rng_perm.permutation(len(all_shap_rates))
            perm_diffs[p] = (
                np.mean(all_shap_rates[perm[:n_within]])
                - np.mean(all_shap_rates[perm[n_within:]])
            )
        perm_p = np.mean(perm_diffs >= observed_diff)

        # Print results
        print(f"\n  SHAP-based flip rates:")
        print(f"    Within mean:  {shap_within_ci[1]:.4f}  "
              f"CI [{shap_within_ci[0]:.4f}, {shap_within_ci[2]:.4f}]")
        print(f"    Between mean: {shap_between_ci[1]:.4f}  "
              f"CI [{shap_between_ci[0]:.4f}, {shap_between_ci[2]:.4f}]")
        print(f"    Gap:          {shap_gap:.4f} ({shap_gap*100:.1f}pp)")
        print(f"    Permutation p: {perm_p:.4f}")

        print(f"\n  Gain-based flip rates:")
        print(f"    Within mean:  {gain_within_ci[1]:.4f}  "
              f"CI [{gain_within_ci[0]:.4f}, {gain_within_ci[2]:.4f}]")
        print(f"    Between mean: {gain_between_ci[1]:.4f}  "
              f"CI [{gain_between_ci[0]:.4f}, {gain_between_ci[2]:.4f}]")
        print(f"    Gap:          {gain_gap:.4f} ({gain_gap*100:.1f}pp)")

        rho_key = f"rho_{rho_w}"
        all_results[rho_key] = {
            "rho_within": rho_w,
            "r2_mean": round(r2_mean, 4),
            "r2_std": round(r2_std, 4),
            "shap": {
                "within_mean": round(shap_within_ci[1], 6),
                "within_ci": [round(shap_within_ci[0], 6), round(shap_within_ci[2], 6)],
                "between_mean": round(shap_between_ci[1], 6),
                "between_ci": [round(shap_between_ci[0], 6), round(shap_between_ci[2], 6)],
                "gap_pp": round(shap_gap * 100, 2),
                "permutation_p": round(float(perm_p), 6),
                "within_rates": [round(r, 6) for r in sorted(shap_within)],
                "between_rates": [round(r, 6) for r in sorted(shap_between)],
            },
            "gain": {
                "within_mean": round(gain_within_ci[1], 6),
                "within_ci": [round(gain_within_ci[0], 6), round(gain_within_ci[2], 6)],
                "between_mean": round(gain_between_ci[1], 6),
                "between_ci": [round(gain_between_ci[0], 6), round(gain_between_ci[2], 6)],
                "gap_pp": round(gain_gap * 100, 2),
            },
        }

    # GO/NO-GO decision
    rho70 = all_results["rho_0.7"]
    go_decision = rho70["shap"]["gap_pp"] > 30.0
    all_results["go_nogo"] = {
        "criterion": "SHAP bimodal gap > 30pp at rho=0.70",
        "observed_gap_pp": rho70["shap"]["gap_pp"],
        "decision": "GO" if go_decision else "NO-GO",
    }

    print(f"\n{'='*70}")
    print(f"GO/NO-GO DECISION")
    print(f"{'='*70}")
    print(f"  Criterion: SHAP bimodal gap > 30pp at ρ=0.70")
    print(f"  Observed gap: {rho70['shap']['gap_pp']:.1f}pp")
    print(f"  Decision: {'GO' if go_decision else 'NO-GO'}")

    return all_results


# ===================================================================
# Task 2: SAGE with TreeSHAP on Breast Cancer
# ===================================================================

def compute_flip_rate_matrix(importance_matrix):
    """Compute P x P flip-rate matrix from importance matrix."""
    M, P_dim = importance_matrix.shape
    n_pairs = M * (M - 1) // 2
    F = np.zeros((P_dim, P_dim))
    for j in range(P_dim):
        for k in range(j + 1, P_dim):
            j_gt_k = importance_matrix[:, j] > importance_matrix[:, k]
            n_true = j_gt_k.sum()
            n_false = M - n_true
            F[j, k] = int(n_true * n_false) / n_pairs
            F[k, j] = F[j, k]
    return F


def discover_groups(F, threshold=0.30):
    """Hierarchical clustering on flip-rate matrix."""
    P_dim = F.shape[0]
    if P_dim < 2:
        return np.array([1]), 1, [1]
    dist = 1.0 - F
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    n_groups = len(set(labels))
    sizes = sorted([int(np.sum(labels == g)) for g in set(labels)], reverse=True)
    return labels, n_groups, sizes


def run_sage_treeshap():
    print(f"\n\n{'='*70}")
    print("TASK 2: SAGE with TreeSHAP on Breast Cancer")
    print(f"{'='*70}")

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    P_dim = X.shape[1]
    M = 50
    THRESHOLD = 0.30

    print(f"  n_samples={X.shape[0]}, n_features={P_dim}")
    print(f"  Training {M} XGBoost models with TreeSHAP...")

    shap_importance_matrix = np.zeros((M, P_dim))
    gain_importance_matrix = np.zeros((M, P_dim))

    for i in range(M):
        rng = np.random.RandomState(SEED + i)
        idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        X_b, y_b = X[idx], y[idx]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=SEED + i,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_b, y_b)

        # Gain
        gain_importance_matrix[i] = model.feature_importances_

        # TreeSHAP
        explainer = shap.TreeExplainer(model)
        # Use a fixed subsample for SHAP computation
        shap_sample = X[:200]
        shap_values = explainer.shap_values(shap_sample)
        if isinstance(shap_values, list):
            importance = np.mean(
                [np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0
            )
        else:
            importance = np.mean(np.abs(shap_values), axis=0)
        shap_importance_matrix[i] = importance

    # SAGE clustering for SHAP
    print("  Computing SHAP flip-rate matrix...")
    F_shap = compute_flip_rate_matrix(shap_importance_matrix)
    labels_shap, n_groups_shap, sizes_shap = discover_groups(F_shap, THRESHOLD)

    # SAGE clustering for gain
    print("  Computing gain flip-rate matrix...")
    F_gain = compute_flip_rate_matrix(gain_importance_matrix)
    labels_gain, n_groups_gain, sizes_gain = discover_groups(F_gain, THRESHOLD)

    # Group composition for SHAP
    shap_groups = {}
    for g_id in sorted(set(labels_shap)):
        members = [feature_names[i] for i in range(P_dim) if labels_shap[i] == g_id]
        shap_groups[f"group_{g_id}"] = members

    # Group composition for gain
    gain_groups = {}
    for g_id in sorted(set(labels_gain)):
        members = [feature_names[i] for i in range(P_dim) if labels_gain[i] == g_id]
        gain_groups[f"group_{g_id}"] = members

    # Predicted vs observed instability
    eta_shap = n_groups_shap / P_dim
    instability_pred_shap = 1.0 - eta_shap
    upper_shap = F_shap[np.triu_indices(P_dim, k=1)]
    instability_obs_shap = float(np.mean(upper_shap))

    eta_gain = n_groups_gain / P_dim
    instability_pred_gain = 1.0 - eta_gain
    upper_gain = F_gain[np.triu_indices(P_dim, k=1)]
    instability_obs_gain = float(np.mean(upper_gain))

    print(f"\n  SHAP-based SAGE:")
    print(f"    Groups: {n_groups_shap}, sizes: {sizes_shap}")
    print(f"    eta = {eta_shap:.4f}")
    print(f"    Predicted instability: {instability_pred_shap:.4f}")
    print(f"    Observed instability:  {instability_obs_shap:.4f}")
    print(f"    |error|: {abs(instability_pred_shap - instability_obs_shap):.4f}")

    print(f"\n  Gain-based SAGE:")
    print(f"    Groups: {n_groups_gain}, sizes: {sizes_gain}")
    print(f"    eta = {eta_gain:.4f}")
    print(f"    Predicted instability: {instability_pred_gain:.4f}")
    print(f"    Observed instability:  {instability_obs_gain:.4f}")
    print(f"    |error|: {abs(instability_pred_gain - instability_obs_gain):.4f}")

    # Print SHAP group composition
    print(f"\n  SHAP group composition:")
    for gname, members in shap_groups.items():
        print(f"    {gname} ({len(members)}): {', '.join(members[:5])}"
              + (f"... +{len(members)-5}" if len(members) > 5 else ""))

    print(f"\n  Gain group composition:")
    for gname, members in gain_groups.items():
        print(f"    {gname} ({len(members)}): {', '.join(members[:5])}"
              + (f"... +{len(members)-5}" if len(members) > 5 else ""))

    # Do SHAP and gain discover the same groups?
    # Compute Adjusted Rand Index
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_shap, labels_gain)
    print(f"\n  Adjusted Rand Index (SHAP vs Gain groups): {ari:.4f}")

    sage_results = {
        "dataset": "Breast Cancer Wisconsin",
        "n_features": int(P_dim),
        "n_models": M,
        "threshold": THRESHOLD,
        "shap_sage": {
            "n_groups": int(n_groups_shap),
            "group_sizes": sizes_shap,
            "groups": {k: list(v) for k, v in shap_groups.items()},
            "eta": round(eta_shap, 6),
            "instability_predicted": round(instability_pred_shap, 6),
            "instability_observed": round(instability_obs_shap, 6),
            "error": round(abs(instability_pred_shap - instability_obs_shap), 6),
        },
        "gain_sage": {
            "n_groups": int(n_groups_gain),
            "group_sizes": sizes_gain,
            "groups": {k: list(v) for k, v in gain_groups.items()},
            "eta": round(eta_gain, 6),
            "instability_predicted": round(instability_pred_gain, 6),
            "instability_observed": round(instability_obs_gain, 6),
            "error": round(abs(instability_pred_gain - instability_obs_gain), 6),
        },
        "adjusted_rand_index": round(ari, 6),
        "_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return sage_results


# ===================================================================
# Figure
# ===================================================================

def make_figure(noether_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, rho_w in enumerate(RHO_WITHIN_VALUES):
        ax = axes[idx]
        rho_key = f"rho_{rho_w}"
        res = noether_results[rho_key]

        shap_within = np.array(res["shap"]["within_rates"])
        shap_between = np.array(res["shap"]["between_rates"])

        bins = np.linspace(0, 0.55, 25)
        ax.hist(shap_within, bins=bins, alpha=0.7, color="#e74c3c",
                label="Within-group", edgecolor="white")
        ax.hist(shap_between, bins=bins, alpha=0.7, color="#2980b9",
                label="Between-group", edgecolor="white")
        ax.axvline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.5)

        gap = res["shap"]["gap_pp"]
        ax.set_title(
            f"$\\rho_{{within}}={rho_w}$\n"
            f"Gap = {gap:.1f}pp",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Pairwise Flip Rate (TreeSHAP)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)

        # Annotate means
        ax.axvline(res["shap"]["within_mean"], color="#e74c3c",
                    linestyle=":", linewidth=1.5, alpha=0.8)
        ax.axvline(res["shap"]["between_mean"], color="#2980b9",
                    linestyle=":", linewidth=1.5, alpha=0.8)

    fig.suptitle(
        "Noether Counting with TreeSHAP: Bimodal Gap Persists",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "noether_treeshap.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved figure: {fig_path}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    t0 = time.time()

    # Task 1
    noether_results = run_noether_treeshap()

    # Task 2
    sage_results = run_sage_treeshap()

    # Save results
    noether_results["_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    noether_results["experiment"] = "noether_treeshap"
    noether_results["config"] = {
        "P": P, "G": G, "group_size": GROUP_SIZE,
        "betas": BETAS.tolist(),
        "rho_within_values": RHO_WITHIN_VALUES,
        "rho_between": RHO_BETWEEN,
        "n_train": N_TRAIN, "n_test": N_TEST,
        "noise_std": NOISE_STD,
        "n_models": N_MODELS, "seed": SEED,
    }

    noether_path = OUT_DIR / "results_noether_treeshap.json"
    with open(noether_path, "w") as f:
        json.dump(noether_results, f, indent=2)
    print(f"\nSaved: {noether_path}")

    sage_path = OUT_DIR / "results_sage_treeshap.json"
    with open(sage_path, "w") as f:
        json.dump(sage_results, f, indent=2)
    print(f"Saved: {sage_path}")

    # Figure
    make_figure(noether_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
