"""
Uncertainty Principle Experiment
================================
Measures the (faithfulness alpha, stability sigma, decisiveness delta) triple
for multiple explanation methods across multiple instances, testing the bound
alpha + sigma + delta <= 2 + f(mu_R).

Instances:
  1. Feature Attribution (XGBoost on correlated features)
  2. Model Selection (Rashomon set on synthetic binary classification)
  3. Linear Attribution (Ridge regression, clean theory)

Methods per instance:
  a) Single-model
  b) Ensemble average (DASH proxy)
  c) Constant method
  d) Top-k partial ranking
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost is required. Install with: pip install xgboost")

SEED = 42
np.random.seed(SEED)

RESULTS_PATH = Path(__file__).parent / "results_uncertainty_principle.json"
FIGURE_PATH = Path(__file__).parent / "figures" / "uncertainty_principle.pdf"
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def generate_correlated_features(n, p_per_group=4, n_groups=2, rho_within=0.9,
                                 rho_between=0.1, seed=42):
    """Generate features with block correlation structure."""
    rng = np.random.RandomState(seed)
    p = p_per_group * n_groups
    cov = np.full((p, p), rho_between)
    for g in range(n_groups):
        s = g * p_per_group
        e = s + p_per_group
        cov[s:e, s:e] = rho_within
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    return X


def flip_rate(ranking_a, ranking_b):
    """Fraction of pairwise comparisons that flip between two rankings."""
    n = len(ranking_a)
    flips = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = np.sign(ranking_a[i] - ranking_a[j])
            sign_b = np.sign(ranking_b[i] - ranking_b[j])
            if sign_a != 0 and sign_b != 0:
                total += 1
                if sign_a != sign_b:
                    flips += 1
    return flips / max(total, 1)


def safe_spearman(a, b, mask=None):
    """Spearman correlation, optionally restricted to a subset."""
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(a) < 3:
        return 0.0
    corr, _ = spearmanr(a, b)
    return 0.0 if np.isnan(corr) else float(corr)


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    means = [np.mean(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lo), float(hi)


def count_tied_pairs(imp, p, atol=1e-6):
    """Count tied pairs in an importance vector."""
    n_tied = 0
    for i in range(p):
        for j in range(i + 1, p):
            if np.isclose(imp[i], imp[j], atol=atol):
                n_tied += 1
    return n_tied


# ---------------------------------------------------------------------------
# Instance 1: Feature Attribution (XGBoost)
# ---------------------------------------------------------------------------

def run_instance1_xgboost():
    """Feature attribution on XGBoost with correlated features.

    Key: train models on BOOTSTRAP RESAMPLES of training data so that
    feature_importances_ genuinely differ across models (Rashomon effect).
    Alpha = Spearman correlation with ground truth |beta|.
    """
    print("  Instance 1: Feature Attribution (XGBoost)")
    p_per_group, n_groups = 4, 2
    p = p_per_group * n_groups
    beta_true = np.array([3, 3, 3, 3, 1, 1, 1, 1], dtype=float)
    ground_truth = np.abs(beta_true)  # true feature importances
    n_train, n_test = 500, 200
    noise_std = 1.0
    n_models = 100

    # Generate data
    X_train = generate_correlated_features(n_train, p_per_group, n_groups, seed=SEED)
    X_test = generate_correlated_features(n_test, p_per_group, n_groups, seed=SEED + 1)
    rng = np.random.RandomState(SEED)
    y_train = X_train @ beta_true + rng.normal(0, noise_std, n_train)

    # Train models on bootstrap resamples (creates genuine Rashomon variability)
    rng_boot = np.random.RandomState(SEED + 10)
    importances_list = []
    for i in range(n_models):
        idx = rng_boot.choice(n_train, size=n_train, replace=True)
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED + i, verbosity=0, n_jobs=1,
        )
        model.fit(X_train[idx], y_train[idx])
        importances_list.append(model.feature_importances_)
    importances = np.array(importances_list)  # (n_models, p)
    n_pairs = p * (p - 1) // 2

    # --- Method a: Single-model ---
    single_imp = importances[0]
    alpha_single = safe_spearman(single_imp, ground_truth)
    flip_rates_s = [flip_rate(single_imp, importances[j]) for j in range(1, n_models)]
    sigma_single = 1.0 - np.mean(flip_rates_s)
    delta_single = 1.0 - count_tied_pairs(single_imp, p) / n_pairs

    # Bootstrap CIs for alpha
    alphas_single_boot = [safe_spearman(importances[j], ground_truth) for j in range(n_models)]
    alpha_s_mean, alpha_s_lo, alpha_s_hi = bootstrap_ci(alphas_single_boot)
    # Bootstrap CIs for sigma
    sigma_s_mean, sigma_s_lo, sigma_s_hi = bootstrap_ci(1.0 - np.array(flip_rates_s))

    method_a = {
        "alpha": float(alpha_single), "alpha_ci": [alpha_s_lo, alpha_s_hi],
        "sigma": float(sigma_single), "sigma_ci": [sigma_s_lo, sigma_s_hi],
        "delta": float(delta_single), "delta_ci": [float(delta_single), float(delta_single)],
    }

    # --- Method b: Ensemble average (DASH proxy) ---
    ensemble_imp = np.mean(importances, axis=0)
    alpha_ens = safe_spearman(ensemble_imp, ground_truth)
    # Alpha CIs: bootstrap over which models are in the ensemble
    n_boot_ens = 50
    rng_b2 = np.random.RandomState(SEED + 100)
    alpha_ens_samples = []
    ensemble_boots = []
    for _ in range(n_boot_ens):
        idx = rng_b2.choice(n_models, size=n_models // 2, replace=True)
        ens_b = np.mean(importances[idx], axis=0)
        ensemble_boots.append(ens_b)
        alpha_ens_samples.append(safe_spearman(ens_b, ground_truth))
    alpha_e_mean, alpha_e_lo, alpha_e_hi = bootstrap_ci(alpha_ens_samples)
    # Sigma: flip rate between bootstrap ensembles
    flip_rates_ens = []
    for i_ in range(len(ensemble_boots)):
        for j_ in range(i_ + 1, min(i_ + 20, len(ensemble_boots))):
            flip_rates_ens.append(flip_rate(ensemble_boots[i_], ensemble_boots[j_]))
    sigma_ens = 1.0 - np.mean(flip_rates_ens)
    sigma_e_mean, sigma_e_lo, sigma_e_hi = bootstrap_ci(1.0 - np.array(flip_rates_ens))
    # Delta
    delta_ens = 1.0 - count_tied_pairs(ensemble_imp, p, atol=1e-6) / n_pairs

    method_b = {
        "alpha": float(alpha_ens), "alpha_ci": [alpha_e_lo, alpha_e_hi],
        "sigma": float(sigma_ens), "sigma_ci": [sigma_e_lo, sigma_e_hi],
        "delta": float(delta_ens), "delta_ci": [float(delta_ens), float(delta_ens)],
    }

    # --- Method c: Constant ---
    method_c = {
        "alpha": 0.0, "alpha_ci": [0.0, 0.0],
        "sigma": 1.0, "sigma_ci": [1.0, 1.0],
        "delta": 0.0, "delta_ci": [0.0, 0.0],
    }

    # --- Method d: Top-k partial ---
    top_k = 4
    ensemble_rank = np.argsort(-ensemble_imp)
    top_features = set(ensemble_rank[:top_k])
    mask_top = np.array([i in top_features for i in range(p)])

    # Alpha on ranked features only
    alphas_topk = [safe_spearman(ensemble_imp, importances[j], mask=mask_top)
                   for j in range(n_models)]
    alpha_t_mean, alpha_t_lo, alpha_t_hi = bootstrap_ci(alphas_topk)
    # Also measure against ground truth on top features
    alpha_topk_gt = safe_spearman(ensemble_imp, ground_truth, mask=mask_top)

    # Sigma: stability of top-4 set identity
    top4_sets = [set(np.argsort(-eb)[:top_k]) for eb in ensemble_boots]
    jaccard_vals = []
    for i_ in range(len(top4_sets)):
        for j_ in range(i_ + 1, len(top4_sets)):
            inter = len(top4_sets[i_] & top4_sets[j_])
            union = len(top4_sets[i_] | top4_sets[j_])
            jaccard_vals.append(inter / union)
    sigma_topk = float(np.mean(jaccard_vals))
    sigma_t_mean, sigma_t_lo, sigma_t_hi = bootstrap_ci(jaccard_vals)

    # Delta: fraction of pairs ranked
    n_ranked_pairs = top_k * (top_k - 1) // 2 + top_k * (p - top_k)
    delta_topk = n_ranked_pairs / n_pairs

    method_d = {
        "alpha": float(alpha_topk_gt), "alpha_ci": [alpha_t_lo, alpha_t_hi],
        "sigma": float(sigma_topk), "sigma_ci": [sigma_t_lo, sigma_t_hi],
        "delta": float(delta_topk), "delta_ci": [float(delta_topk), float(delta_topk)],
    }

    return {
        "single_model": method_a,
        "ensemble_average": method_b,
        "constant": method_c,
        "top_k_partial": method_d,
    }


# ---------------------------------------------------------------------------
# Instance 2: Model Selection (Rashomon set)
# ---------------------------------------------------------------------------

def run_instance2_model_selection():
    """Model selection on synthetic binary classification."""
    print("  Instance 2: Model Selection (Rashomon set)")
    rng = np.random.RandomState(SEED + 200)
    n_train, n_test = 800, 200
    p = 10

    # Synthetic binary classification
    X_all = rng.randn(n_train + n_test, p)
    beta = rng.randn(p)
    logits = X_all @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y_all = (rng.rand(n_train + n_test) < prob).astype(int)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    n_models = 50
    models = []
    test_accs = []
    for i in range(n_models):
        model = xgb.XGBClassifier(
            n_estimators=50 + i * 2, max_depth=3 + (i % 4),
            learning_rate=0.05 + 0.005 * (i % 10),
            random_state=SEED + 300 + i, verbosity=0,
            eval_metric="logloss", n_jobs=1,
        )
        model.fit(X_train, y_train)
        models.append(model)
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))
    test_accs = np.array(test_accs)
    best_model_idx = np.argmax(test_accs)
    n_pairs_ms = n_models * (n_models - 1) // 2

    # --- Method a: Single split selection ---
    n_splits = 30
    rng2 = np.random.RandomState(SEED + 400)
    split_rankings = []  # full ranking per split
    split_bests = []
    for s in range(n_splits):
        perm = rng2.permutation(n_train)
        val_idx = perm[:n_train // 5]
        val_accs = np.array([accuracy_score(y_train[val_idx], m.predict(X_train[val_idx]))
                             for m in models])
        split_rankings.append(val_accs)
        split_bests.append(np.argmax(val_accs))

    # Alpha: Spearman between single-split ranking and test accuracy (use first split)
    alpha_ss = safe_spearman(split_rankings[0], test_accs)
    # Bootstrap CIs across splits
    alphas_ss_all = [safe_spearman(sr, test_accs) for sr in split_rankings]
    alpha_ss_mean, alpha_ss_lo, alpha_ss_hi = bootstrap_ci(alphas_ss_all)
    # Sigma: agreement of best model selection across splits
    counts = Counter(split_bests)
    sigma_ss = counts.most_common(1)[0][1] / n_splits
    # Also measure ranking flip rate between splits
    flip_rates_ss = []
    for i_ in range(min(n_splits, 15)):
        for j_ in range(i_ + 1, min(n_splits, 15)):
            flip_rates_ss.append(flip_rate(split_rankings[i_], split_rankings[j_]))
    sigma_ss_flip = 1.0 - np.mean(flip_rates_ss)
    sigma_ss_combined = (sigma_ss + sigma_ss_flip) / 2  # blend both stability measures
    # Delta: 1.0 (ranks all models)
    delta_ss = 1.0

    method_a = {
        "alpha": float(alpha_ss), "alpha_ci": [alpha_ss_lo, alpha_ss_hi],
        "sigma": float(sigma_ss_combined),
        "sigma_ci": [float(sigma_ss_combined), float(sigma_ss_combined)],
        "delta": float(delta_ss), "delta_ci": [1.0, 1.0],
    }

    # --- Method b: Cross-validation ---
    cv_scores = []
    for i, m in enumerate(models):
        m_cv = xgb.XGBClassifier(
            n_estimators=50 + i * 2, max_depth=3 + (i % 4),
            learning_rate=0.05 + 0.005 * (i % 10),
            random_state=SEED + 300 + i, verbosity=0,
            eval_metric="logloss", n_jobs=1,
        )
        scores = cross_val_score(m_cv, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores.append(np.mean(scores))
    cv_scores = np.array(cv_scores)

    alpha_cv = safe_spearman(cv_scores, test_accs)
    # Bootstrap stability: resample within CV variance
    n_boot_cv = 50
    rng3 = np.random.RandomState(SEED + 500)
    cv_boot_rankings = []
    for _ in range(n_boot_cv):
        # Simulate CV variance by adding noise proportional to CV std
        noise = rng3.normal(0, 0.005, n_models)
        cv_boot_rankings.append(cv_scores + noise)
    flip_rates_cv = []
    for i_ in range(min(n_boot_cv, 20)):
        for j_ in range(i_ + 1, min(n_boot_cv, 20)):
            flip_rates_cv.append(flip_rate(cv_boot_rankings[i_], cv_boot_rankings[j_]))
    sigma_cv = 1.0 - np.mean(flip_rates_cv)
    delta_cv = 1.0

    method_b = {
        "alpha": float(alpha_cv), "alpha_ci": [float(alpha_cv), float(alpha_cv)],
        "sigma": float(sigma_cv), "sigma_ci": [float(sigma_cv), float(sigma_cv)],
        "delta": float(delta_cv), "delta_ci": [1.0, 1.0],
    }

    # --- Method c: Ensemble average ---
    ensemble_preds = np.column_stack([m.predict(X_test) for m in models])
    ensemble_vote = (np.mean(ensemble_preds, axis=1) > 0.5).astype(int)
    # "Importance" = agreement with ensemble
    model_importances = np.array([
        np.mean(ensemble_preds[:, i] == ensemble_vote) for i in range(n_models)
    ])
    alpha_ens = safe_spearman(model_importances, test_accs)
    # Stability: ensemble is very stable by design
    # Measure by subsampling models
    rng4 = np.random.RandomState(SEED + 600)
    ens_boot_importances = []
    for _ in range(30):
        idx = rng4.choice(n_models, size=n_models // 2, replace=True)
        sub_preds = ensemble_preds[:, idx]
        sub_vote = (np.mean(sub_preds, axis=1) > 0.5).astype(int)
        sub_imp = np.array([np.mean(ensemble_preds[:, i] == sub_vote) for i in range(n_models)])
        ens_boot_importances.append(sub_imp)
    flip_rates_ens_ms = []
    for i_ in range(len(ens_boot_importances)):
        for j_ in range(i_ + 1, min(i_ + 10, len(ens_boot_importances))):
            flip_rates_ens_ms.append(flip_rate(ens_boot_importances[i_], ens_boot_importances[j_]))
    sigma_ens_ms = 1.0 - np.mean(flip_rates_ens_ms)
    # Delta: many models may tie in agreement score
    n_tied_ens = count_tied_pairs(model_importances, n_models, atol=1e-6)
    delta_ens = 1.0 - n_tied_ens / n_pairs_ms

    method_c_ms = {
        "alpha": float(alpha_ens),
        "alpha_ci": [float(alpha_ens), float(alpha_ens)],
        "sigma": float(sigma_ens_ms),
        "sigma_ci": [float(sigma_ens_ms), float(sigma_ens_ms)],
        "delta": float(delta_ens), "delta_ci": [float(delta_ens), float(delta_ens)],
    }

    # --- Method d: Random selection ---
    method_d_ms = {
        "alpha": 0.0, "alpha_ci": [0.0, 0.0],
        "sigma": 1.0 / n_models, "sigma_ci": [1.0 / n_models, 1.0 / n_models],
        "delta": 0.0, "delta_ci": [0.0, 0.0],
    }

    return {
        "single_split": method_a,
        "cross_validation": method_b,
        "ensemble_average": method_c_ms,
        "random_selection": method_d_ms,
    }


# ---------------------------------------------------------------------------
# Instance 3: Linear Attribution (Ridge)
# ---------------------------------------------------------------------------

def run_instance3_linear():
    """Feature attribution on Ridge regression (exact SHAP = coefficients x features)."""
    print("  Instance 3: Linear Attribution (Ridge)")
    p_per_group, n_groups = 4, 2
    p = p_per_group * n_groups
    beta_true = np.array([3, 3, 3, 3, 1, 1, 1, 1], dtype=float)
    ground_truth = np.abs(beta_true)
    n_train, n_test = 500, 200
    noise_std = 1.0
    n_models = 100

    X_train = generate_correlated_features(n_train, p_per_group, n_groups, seed=SEED + 600)
    rng = np.random.RandomState(SEED + 602)
    y_train = X_train @ beta_true + rng.normal(0, noise_std, n_train)

    # Train on bootstrap resamples
    rng_boot = np.random.RandomState(SEED + 700)
    coefs_list = []
    for _ in range(n_models):
        idx = rng_boot.choice(n_train, size=n_train, replace=True)
        model = Ridge(alpha=1.0)
        model.fit(X_train[idx], y_train[idx])
        coefs_list.append(np.abs(model.coef_))
    coefs = np.array(coefs_list)
    n_pairs = p * (p - 1) // 2

    # --- Method a: Single-model ---
    single_coef = coefs[0]
    alpha_single = safe_spearman(single_coef, ground_truth)
    flip_rates_s = [flip_rate(single_coef, coefs[j]) for j in range(1, n_models)]
    sigma_single = 1.0 - np.mean(flip_rates_s)
    delta_single = 1.0 - count_tied_pairs(single_coef, p, atol=1e-8) / n_pairs

    # CIs
    alphas_s_boot = [safe_spearman(coefs[j], ground_truth) for j in range(n_models)]
    alpha_s_mean, alpha_s_lo, alpha_s_hi = bootstrap_ci(alphas_s_boot)
    sigma_s_mean, sigma_s_lo, sigma_s_hi = bootstrap_ci(1.0 - np.array(flip_rates_s))

    method_a = {
        "alpha": float(alpha_single), "alpha_ci": [alpha_s_lo, alpha_s_hi],
        "sigma": float(sigma_single), "sigma_ci": [sigma_s_lo, sigma_s_hi],
        "delta": float(delta_single), "delta_ci": [float(delta_single), float(delta_single)],
    }

    # --- Method b: Ensemble average ---
    ensemble_coef = np.mean(coefs, axis=0)
    alpha_ens = safe_spearman(ensemble_coef, ground_truth)

    n_boot_ens = 50
    rng_b2 = np.random.RandomState(SEED + 800)
    ensemble_boots = []
    alpha_ens_samples = []
    for _ in range(n_boot_ens):
        idx = rng_b2.choice(n_models, size=n_models // 2, replace=True)
        ens_b = np.mean(coefs[idx], axis=0)
        ensemble_boots.append(ens_b)
        alpha_ens_samples.append(safe_spearman(ens_b, ground_truth))
    alpha_e_mean, alpha_e_lo, alpha_e_hi = bootstrap_ci(alpha_ens_samples)

    flip_rates_ens = []
    for i_ in range(len(ensemble_boots)):
        for j_ in range(i_ + 1, min(i_ + 20, len(ensemble_boots))):
            flip_rates_ens.append(flip_rate(ensemble_boots[i_], ensemble_boots[j_]))
    sigma_ens = 1.0 - np.mean(flip_rates_ens)
    sigma_e_mean, sigma_e_lo, sigma_e_hi = bootstrap_ci(1.0 - np.array(flip_rates_ens))

    delta_ens = 1.0 - count_tied_pairs(ensemble_coef, p, atol=1e-6) / n_pairs

    method_b = {
        "alpha": float(alpha_ens), "alpha_ci": [alpha_e_lo, alpha_e_hi],
        "sigma": float(sigma_ens), "sigma_ci": [sigma_e_lo, sigma_e_hi],
        "delta": float(delta_ens), "delta_ci": [float(delta_ens), float(delta_ens)],
    }

    # --- Method c: Constant ---
    method_c = {
        "alpha": 0.0, "alpha_ci": [0.0, 0.0],
        "sigma": 1.0, "sigma_ci": [1.0, 1.0],
        "delta": 0.0, "delta_ci": [0.0, 0.0],
    }

    # --- Method d: Top-k partial ---
    top_k = 4
    ensemble_rank = np.argsort(-ensemble_coef)
    top_features = set(ensemble_rank[:top_k])
    mask_top = np.array([i in top_features for i in range(p)])

    alpha_topk_gt = safe_spearman(ensemble_coef, ground_truth, mask=mask_top)
    alphas_topk = [safe_spearman(ensemble_coef, coefs[j], mask=mask_top)
                   for j in range(n_models)]
    alpha_t_mean, alpha_t_lo, alpha_t_hi = bootstrap_ci(alphas_topk)

    top4_sets = [set(np.argsort(-eb)[:top_k]) for eb in ensemble_boots]
    jaccard_vals = []
    for i_ in range(len(top4_sets)):
        for j_ in range(i_ + 1, len(top4_sets)):
            inter = len(top4_sets[i_] & top4_sets[j_])
            union = len(top4_sets[i_] | top4_sets[j_])
            jaccard_vals.append(inter / union)
    sigma_topk = float(np.mean(jaccard_vals))
    sigma_t_mean, sigma_t_lo, sigma_t_hi = bootstrap_ci(jaccard_vals)

    n_ranked_pairs = top_k * (top_k - 1) // 2 + top_k * (p - top_k)
    delta_topk = n_ranked_pairs / n_pairs

    method_d = {
        "alpha": float(alpha_topk_gt), "alpha_ci": [alpha_t_lo, alpha_t_hi],
        "sigma": float(sigma_topk), "sigma_ci": [sigma_t_lo, sigma_t_hi],
        "delta": float(delta_topk), "delta_ci": [float(delta_topk), float(delta_topk)],
    }

    return {
        "single_model": method_a,
        "ensemble_average": method_b,
        "constant": method_c,
        "top_k_partial": method_d,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results):
    """Compute bound, Pareto-optimality, etc."""
    all_points = []
    for instance_name, methods in results.items():
        for method_name, vals in methods.items():
            a, s, d = vals["alpha"], vals["sigma"], vals["delta"]
            all_points.append({
                "instance": instance_name,
                "method": method_name,
                "alpha": a, "sigma": s, "delta": d,
                "sum": a + s + d,
            })

    max_sum = max(pt["sum"] for pt in all_points)
    max_pt = max(all_points, key=lambda pt: pt["sum"])
    bound_holds_2 = all(pt["sum"] <= 2.0 + 1e-9 for pt in all_points)

    # Pareto optimality: not dominated in all three components
    pareto = []
    for i, pt in enumerate(all_points):
        dominated = False
        for j, other in enumerate(all_points):
            if i == j:
                continue
            if (other["alpha"] >= pt["alpha"] and
                other["sigma"] >= pt["sigma"] and
                other["delta"] >= pt["delta"] and
                (other["alpha"] > pt["alpha"] or
                 other["sigma"] > pt["sigma"] or
                 other["delta"] > pt["delta"])):
                dominated = True
                break
        if not dominated:
            pareto.append(pt)

    return all_points, max_sum, max_pt, bound_holds_2, pareto


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_points, max_sum, bound_holds, pareto):
    """Create 2D projections of (alpha, sigma, delta) triples."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    instance_colors = {
        "instance1_xgboost": "#1f77b4",
        "instance2_model_selection": "#ff7f0e",
        "instance3_linear": "#2ca02c",
    }
    instance_labels = {
        "instance1_xgboost": "XGBoost Attribution",
        "instance2_model_selection": "Model Selection",
        "instance3_linear": "Linear Attribution",
    }
    method_markers = {
        "single_model": "o", "ensemble_average": "s", "constant": "^",
        "top_k_partial": "D", "single_split": "o", "cross_validation": "P",
        "random_selection": "X",
    }

    pareto_labels = set((pt["instance"], pt["method"]) for pt in pareto)

    pairs = [
        ("alpha", "sigma", r"$\alpha$ (Faithfulness)", r"$\sigma$ (Stability)"),
        ("alpha", "delta", r"$\alpha$ (Faithfulness)", r"$\delta$ (Decisiveness)"),
        ("sigma", "delta", r"$\sigma$ (Stability)", r"$\delta$ (Decisiveness)"),
    ]

    for ax, (xk, yk, xlabel, ylabel) in zip(axes, pairs):
        for pt in all_points:
            color = instance_colors.get(pt["instance"], "gray")
            marker = method_markers.get(pt["method"], "x")
            is_pareto = (pt["instance"], pt["method"]) in pareto_labels
            edgecolor = "red" if is_pareto else color
            linewidth = 2.5 if is_pareto else 1.0
            size = 120 if is_pareto else 70

            ax.scatter(pt[xk], pt[yk], c=color, marker=marker, s=size,
                       edgecolors=edgecolor, linewidths=linewidth, zorder=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-0.1, 1.15)

        # Bound line: x + y = 2 (projected from 3D; since third component in [0,1])
        # The tightest 2D projection is x + y <= 2 - min(third_component)
        # Show x + y = 2 as dashed and x + y = 1 as another reference
        x_line = np.linspace(-0.1, 1.15, 100)
        ax.plot(x_line, 2.0 - x_line, "r--", alpha=0.5, label=r"$x + y = 2$")
        ax.fill_between(x_line, np.maximum(2.0 - x_line, -0.1), 1.15,
                        alpha=0.05, color="red",
                        where=(2.0 - x_line <= 1.15))
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = []
    for inst, color in instance_colors.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=color, markersize=8,
                                      label=instance_labels[inst]))
    # Method markers
    for mname, marker in [("single", "o"), ("ensemble", "s"),
                          ("constant/random", "^"), ("partial", "D"), ("CV", "P")]:
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                                      markerfacecolor="gray", markersize=8,
                                      label=mname))
    legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor="gray", markersize=10,
                                  markeredgecolor="red", markeredgewidth=2.5,
                                  label="Pareto-optimal"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 1.03))

    bound_str = "HOLDS" if bound_holds else "VIOLATED"
    fig.suptitle(
        r"Uncertainty Principle: $\alpha + \sigma + \delta$ "
        f"empirical max = {max_sum:.3f}  |  Bound <= 2: {bound_str}",
        fontsize=13, y=1.10,
    )

    plt.tight_layout()
    fig.savefig(str(FIGURE_PATH), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure saved to {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("UNCERTAINTY PRINCIPLE EXPERIMENT")
    print("Testing bound: alpha + sigma + delta <= 2 + f(mu_R)")
    print("=" * 70)

    results = {}
    print("\nRunning instances...")
    results["instance1_xgboost"] = run_instance1_xgboost()
    results["instance2_model_selection"] = run_instance2_model_selection()
    results["instance3_linear"] = run_instance3_linear()

    all_points, max_sum, max_pt, bound_holds, pareto = analyze_results(results)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: (alpha, sigma, delta) TRIPLES")
    print("=" * 70)
    for instance_name, methods in results.items():
        print(f"\n  {instance_name}:")
        for method_name, vals in methods.items():
            a, s, d = vals["alpha"], vals["sigma"], vals["delta"]
            total = a + s + d
            ci_str = ""
            if "alpha_ci" in vals and vals["alpha_ci"][0] != vals["alpha_ci"][1]:
                ci_str = f"  alpha_CI=[{vals['alpha_ci'][0]:.3f}, {vals['alpha_ci'][1]:.3f}]"
            print(f"    {method_name:25s}  alpha={a:.4f}  sigma={s:.4f}  "
                  f"delta={d:.4f}  SUM={total:.4f}{ci_str}")

    print(f"\n{'=' * 70}")
    print(f"EMPIRICAL MAXIMUM of alpha + sigma + delta = {max_sum:.4f}")
    print(f"  Achieved by: {max_pt['instance']} / {max_pt['method']}")
    print(f"  Bound alpha + sigma + delta <= 2 : "
          f"{'HOLDS' if bound_holds else '*** VIOLATED ***'}")
    print(f"{'=' * 70}")

    print(f"\nPARETO-OPTIMAL METHODS ({len(pareto)}):")
    for pt in pareto:
        print(f"  {pt['instance']:35s} {pt['method']:25s}  "
              f"({pt['alpha']:.4f}, {pt['sigma']:.4f}, {pt['delta']:.4f})  "
              f"sum={pt['sum']:.4f}")

    # Check bound with detailed analysis
    print(f"\nBOUND ANALYSIS:")
    for pt in sorted(all_points, key=lambda x: -x["sum"]):
        flag = " ** MAX" if pt["sum"] == max_sum else ""
        flag += " [PARETO]" if (pt["instance"], pt["method"]) in \
                set((p_["instance"], p_["method"]) for p_ in pareto) else ""
        print(f"  {pt['sum']:.4f}  {pt['instance']:35s} {pt['method']:25s}{flag}")

    # Save
    output = {
        "triples": {},
        "analysis": {
            "empirical_max_sum": max_sum,
            "max_point": max_pt,
            "bound_holds_at_2": bound_holds,
            "pareto_optimal": pareto,
            "all_points": all_points,
        },
    }
    # Convert triples for JSON (ensure all values are serializable)
    for inst, methods in results.items():
        output["triples"][inst] = {}
        for meth, vals in methods.items():
            output["triples"][inst][meth] = {
                k: (list(v) if isinstance(v, np.ndarray) else v)
                for k, v in vals.items()
            }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")

    # Plot
    plot_results(all_points, max_sum, bound_holds, pareto)

    print("\nDone.")


if __name__ == "__main__":
    main()
