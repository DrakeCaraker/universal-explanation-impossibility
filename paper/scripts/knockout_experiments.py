#!/usr/bin/env python3
"""
Knockout experiments: Three cross-domain predictions from the universal
explanation impossibility framework.

Experiment 1: Area law for attribution stability (QCD analogy)
Experiment 2: Double descent as Rashomon phase transition
Experiment 3: Confusion matrix from Rashomon overlap

All experiments use sklearn and numpy only.
"""

import json
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, pairwise_distances

# Use experiment utils from this project
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    PAPER_DIR,
    FIGURES_DIR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Experiment 1: Area Law for Attribution Stability
# ---------------------------------------------------------------------------

def experiment_area_law(seed=42):
    """
    Prediction: stability of k-feature joint attributions decays
    EXPONENTIALLY with k (area law), not polynomially or linearly.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Area Law for Attribution Stability")
    print("=" * 60)
    set_all_seeds(seed)

    n_samples, n_features = 500, 10
    rho = 0.7

    # Generate correlated features (Rashomon-inducing)
    mean = np.zeros(n_features)
    cov = np.full((n_features, n_features), rho)
    np.fill_diagonal(cov, 1.0)
    X = np.random.multivariate_normal(mean, cov, size=n_samples)

    true_beta = np.random.randn(n_features)
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    # Build Rashomon set: 100 ridge models with different feature subsets and lambdas
    n_models = 100
    lambdas = np.logspace(-2, 2, 10)
    models = []
    for i in range(n_models):
        rng = np.random.RandomState(seed + i)
        # Random subset of 7-10 features
        n_use = rng.randint(7, n_features + 1)
        feat_idx = np.sort(rng.choice(n_features, n_use, replace=False))
        lam = rng.choice(lambdas)
        m = Ridge(alpha=lam)
        m.fit(X[:, feat_idx], y)
        models.append((m, feat_idx))

    # For each k, measure stability of k-feature joint importance
    ks = list(range(1, 9))
    stabilities = []
    n_subsets = 50

    for k in ks:
        # Sample random k-subsets of features
        all_feats = list(range(n_features))
        if k <= 4:
            # enumerate more subsets for small k
            possible = list(combinations(all_feats, k))
            chosen = [possible[i] for i in np.random.choice(len(possible), min(n_subsets, len(possible)), replace=False)]
        else:
            chosen = [tuple(sorted(np.random.choice(all_feats, k, replace=False))) for _ in range(n_subsets)]

        # For each k-subset, compute importance for each model
        importances = np.zeros((n_models, len(chosen)))
        for mi, (model, feat_idx) in enumerate(models):
            for si, subset in enumerate(chosen):
                # k-feature importance: change in prediction variance when
                # those k features are permuted
                X_perm = X.copy()
                for f in subset:
                    X_perm[:, f] = np.random.permutation(X_perm[:, f])
                # Only use features available to this model
                pred_orig = model.predict(X[:, feat_idx])
                pred_perm = model.predict(X_perm[:, feat_idx])
                importances[mi, si] = np.var(pred_orig) - np.var(pred_perm)

        # Stability: mean pairwise Spearman correlation across models
        corrs = []
        for i in range(n_models):
            for j in range(i + 1, min(i + 20, n_models)):  # subsample pairs
                r, _ = stats.spearmanr(importances[i], importances[j])
                if not np.isnan(r):
                    corrs.append(r)
        stabilities.append(np.mean(corrs) if corrs else 0.0)
        print(f"  k={k}: stability={stabilities[-1]:.4f} ({len(corrs)} pairs)")

    ks_arr = np.array(ks, dtype=float)
    stab_arr = np.array(stabilities)

    # Fit three models
    # 1. Exponential: stability = A * exp(-sigma * k) + C
    def exp_model(k, A, sigma, C):
        return A * np.exp(-sigma * k) + C

    # 2. Power law: stability = A * k^(-beta) + C
    def power_model(k, A, beta, C):
        return A * np.power(k, -beta) + C

    # 3. Linear: stability = A - B * k
    def linear_model(k, A, B):
        return A - B * k

    results = {}
    n_data = len(ks_arr)

    # Fit exponential
    try:
        popt_exp, pcov_exp = curve_fit(exp_model, ks_arr, stab_arr,
                                        p0=[1.0, 0.3, 0.0], maxfev=10000)
        resid_exp = stab_arr - exp_model(ks_arr, *popt_exp)
        ss_exp = np.sum(resid_exp ** 2)
        k_exp = 3  # number of parameters
        aic_exp = n_data * np.log(ss_exp / n_data + 1e-30) + 2 * k_exp
        bic_exp = n_data * np.log(ss_exp / n_data + 1e-30) + k_exp * np.log(n_data)
        results["exponential"] = {
            "params": {"A": popt_exp[0], "sigma": popt_exp[1], "C": popt_exp[2]},
            "SS_residual": ss_exp,
            "AIC": aic_exp,
            "BIC": bic_exp,
        }
        print(f"  Exponential: A={popt_exp[0]:.4f}, sigma={popt_exp[1]:.4f}, C={popt_exp[2]:.4f}, AIC={aic_exp:.2f}")
    except Exception as e:
        print(f"  Exponential fit failed: {e}")
        results["exponential"] = {"error": str(e)}

    # Fit power law
    try:
        popt_pow, pcov_pow = curve_fit(power_model, ks_arr, stab_arr,
                                        p0=[1.0, 0.5, 0.0], maxfev=10000)
        resid_pow = stab_arr - power_model(ks_arr, *popt_pow)
        ss_pow = np.sum(resid_pow ** 2)
        k_pow = 3
        aic_pow = n_data * np.log(ss_pow / n_data + 1e-30) + 2 * k_pow
        bic_pow = n_data * np.log(ss_pow / n_data + 1e-30) + k_pow * np.log(n_data)
        results["power_law"] = {
            "params": {"A": popt_pow[0], "beta": popt_pow[1], "C": popt_pow[2]},
            "SS_residual": ss_pow,
            "AIC": aic_pow,
            "BIC": bic_pow,
        }
        print(f"  Power law:   A={popt_pow[0]:.4f}, beta={popt_pow[1]:.4f}, C={popt_pow[2]:.4f}, AIC={aic_pow:.2f}")
    except Exception as e:
        print(f"  Power law fit failed: {e}")
        results["power_law"] = {"error": str(e)}

    # Fit linear
    try:
        popt_lin, pcov_lin = curve_fit(linear_model, ks_arr, stab_arr,
                                        p0=[1.0, 0.1], maxfev=10000)
        resid_lin = stab_arr - linear_model(ks_arr, *popt_lin)
        ss_lin = np.sum(resid_lin ** 2)
        k_lin = 2
        aic_lin = n_data * np.log(ss_lin / n_data + 1e-30) + 2 * k_lin
        bic_lin = n_data * np.log(ss_lin / n_data + 1e-30) + k_lin * np.log(n_data)
        results["linear"] = {
            "params": {"A": popt_lin[0], "B": popt_lin[1]},
            "SS_residual": ss_lin,
            "AIC": aic_lin,
            "BIC": bic_lin,
        }
        print(f"  Linear:      A={popt_lin[0]:.4f}, B={popt_lin[1]:.4f}, AIC={aic_lin:.2f}")
    except Exception as e:
        print(f"  Linear fit failed: {e}")
        results["linear"] = {"error": str(e)}

    # Determine best model
    aics = {}
    bics = {}
    for name in ["exponential", "power_law", "linear"]:
        if "AIC" in results.get(name, {}):
            aics[name] = results[name]["AIC"]
            bics[name] = results[name]["BIC"]

    best_aic = min(aics, key=aics.get) if aics else "none"
    best_bic = min(bics, key=bics.get) if bics else "none"
    results["best_model_AIC"] = best_aic
    results["best_model_BIC"] = best_bic

    prediction_holds = best_aic == "exponential" or best_bic == "exponential"
    results["prediction_holds"] = prediction_holds

    print(f"\n  Best model (AIC): {best_aic}")
    print(f"  Best model (BIC): {best_bic}")
    print(f"  PREDICTION HOLDS: {prediction_holds}")
    if "sigma" in results.get("exponential", {}).get("params", {}):
        print(f"  String tension sigma = {results['exponential']['params']['sigma']:.4f}")

    results["ks"] = ks
    results["stabilities"] = stabilities

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Double Descent as Rashomon Phase Transition
# ---------------------------------------------------------------------------

def experiment_double_descent(seed=42):
    """
    Prediction: explanation instability peaks at the interpolation threshold,
    where the Rashomon set is maximized.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Double Descent as Rashomon Phase Transition")
    print("=" * 60)
    set_all_seeds(seed)

    n_train, n_test = 100, 50
    true_dim = 5

    # Generate data from a degree-5 polynomial
    X_raw = np.random.randn(n_train + n_test, 1) * 2
    true_coefs = np.random.randn(true_dim + 1) * 0.5
    y_all = np.zeros(n_train + n_test)
    for d in range(true_dim + 1):
        y_all += true_coefs[d] * X_raw[:, 0] ** d
    y_all += np.random.randn(n_train + n_test) * 0.3

    X_train, X_test = X_raw[:n_train], X_raw[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    degrees = list(range(1, 31))
    n_perturbations = 50
    noise_scale = 0.05

    test_errors = []
    instabilities = []
    rashomon_sizes = []

    for deg in degrees:
        coef_vectors = []
        train_losses = []

        for p in range(n_perturbations):
            rng = np.random.RandomState(seed + p)
            # Small perturbation to training data
            X_pert = X_train + rng.randn(*X_train.shape) * noise_scale
            y_pert = y_train + rng.randn(*y_train.shape) * noise_scale

            pipe = make_pipeline(
                PolynomialFeatures(deg, include_bias=False),
                Ridge(alpha=1e-6)
            )
            try:
                pipe.fit(X_pert, y_pert)
                coefs = pipe.named_steps["ridge"].coef_
                coef_vectors.append(coefs)
                pred_train = pipe.predict(X_pert)
                train_losses.append(np.mean((pred_train - y_pert) ** 2))
            except Exception:
                continue

        if len(coef_vectors) < 10:
            test_errors.append(np.nan)
            instabilities.append(np.nan)
            rashomon_sizes.append(np.nan)
            continue

        # Test error (from first model)
        pipe0 = make_pipeline(
            PolynomialFeatures(deg, include_bias=False),
            Ridge(alpha=1e-6)
        )
        pipe0.fit(X_train, y_train)
        pred_test = pipe0.predict(X_test)
        te = np.mean((pred_test - y_test) ** 2)
        test_errors.append(te)

        # Instability: mean pairwise cosine distance of coefficient vectors
        # Pad to same length
        max_len = max(len(c) for c in coef_vectors)
        padded = np.zeros((len(coef_vectors), max_len))
        for i, c in enumerate(coef_vectors):
            padded[i, :len(c)] = c

        # Normalize to avoid scale issues
        norms = np.linalg.norm(padded, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        padded_normed = padded / norms
        dists = pairwise_distances(padded_normed, metric="cosine")
        instab = np.mean(dists[np.triu_indices(len(coef_vectors), k=1)])
        instabilities.append(instab)

        # Rashomon set size: models within 10% of best training loss
        train_losses = np.array(train_losses[:len(coef_vectors)])
        best_loss = np.min(train_losses)
        threshold = best_loss * 1.1 + 1e-10
        rash_size = np.sum(train_losses <= threshold) / len(train_losses)
        rashomon_sizes.append(rash_size)

        print(f"  degree={deg:2d}: test_err={te:.4f}, instability={instab:.4f}, rashomon_frac={rash_size:.2f}")

    # Find peaks
    te_arr = np.array(test_errors)
    inst_arr = np.array(instabilities)
    rash_arr = np.array(rashomon_sizes)

    valid = ~np.isnan(te_arr) & ~np.isnan(inst_arr)
    if valid.any():
        te_peak = np.array(degrees)[valid][np.argmax(te_arr[valid])]
        inst_peak = np.array(degrees)[valid][np.argmax(inst_arr[valid])]
        rash_peak = np.array(degrees)[valid][np.argmax(rash_arr[valid])]
    else:
        te_peak = inst_peak = rash_peak = -1

    # Interpolation threshold: approximately n_train / 1 = 100 for degree d
    # (polynomial of degree d has d features)
    interp_threshold = n_train  # degree where #params ~ n_samples

    # Prediction: instability peak near test error peak
    peak_distance = abs(te_peak - inst_peak)
    prediction_holds = peak_distance <= 5  # within 5 degrees

    results = {
        "degrees": degrees,
        "test_errors": [float(x) if not np.isnan(x) else None for x in test_errors],
        "instabilities": [float(x) if not np.isnan(x) else None for x in instabilities],
        "rashomon_sizes": [float(x) if not np.isnan(x) else None for x in rashomon_sizes],
        "test_error_peak_degree": int(te_peak),
        "instability_peak_degree": int(inst_peak),
        "rashomon_peak_degree": int(rash_peak),
        "interpolation_threshold": interp_threshold,
        "peak_distance": int(peak_distance),
        "prediction_holds": prediction_holds,
    }

    print(f"\n  Test error peak at degree:   {te_peak}")
    print(f"  Instability peak at degree:  {inst_peak}")
    print(f"  Rashomon size peak at degree: {rash_peak}")
    print(f"  Peak distance: {peak_distance}")
    print(f"  PREDICTION HOLDS: {prediction_holds}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Confusion Matrix from Rashomon Overlap
# ---------------------------------------------------------------------------

def experiment_confusion_rashomon(seed=42):
    """
    Prediction: Rashomon overlap is a better predictor of confusion
    than feature similarity.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Confusion Matrix from Rashomon Overlap")
    print("=" * 60)
    set_all_seeds(seed)

    # Load digits 0-4
    digits = load_digits()
    mask = digits.target < 5
    X, y = digits.data[mask], digits.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    classes = sorted(np.unique(y))
    n_classes = len(classes)

    print(f"  Classes: {classes}, Train: {len(X_train)}, Test: {len(X_test)}")

    # Train 200 random forest models (Rashomon set)
    n_models = 200
    models = []
    for i in range(n_models):
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=8,
            random_state=seed + i,
            max_features="sqrt"
        )
        rf.fit(X_train, y_train)
        models.append(rf)

    # Compute predictions from all models on test set
    all_preds = np.array([m.predict(X_test) for m in models])  # (n_models, n_test)

    # 1. Rashomon overlap for each pair (i, j)
    rashomon_overlap = np.zeros((n_classes, n_classes))
    for ci, i in enumerate(classes):
        for cj, j in enumerate(classes):
            if i == j:
                continue
            # For samples of true class i, fraction of models predicting class j
            mask_i = y_test == i
            if mask_i.sum() == 0:
                continue
            preds_on_i = all_preds[:, mask_i]  # (n_models, n_samples_of_class_i)
            frac_j = np.mean(preds_on_i == j)
            rashomon_overlap[ci, cj] = frac_j

    # 2. Feature similarity (cosine distance between class centroids)
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    centroid_dists = pairwise_distances(centroids, metric="cosine")
    # Convert distance to similarity (closer = more confusable)
    feature_similarity = 1.0 - centroid_dists

    # 3. Actual confusion matrix (from first model)
    y_pred_single = models[0].predict(X_test)
    cm = confusion_matrix(y_test, y_pred_single, labels=classes)
    # Normalize rows
    cm_normed = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Extract off-diagonal entries for correlation
    off_diag_indices = [(i, j) for i in range(n_classes) for j in range(n_classes) if i != j]

    confusion_vals = np.array([cm_normed[i, j] for i, j in off_diag_indices])
    rashomon_vals = np.array([rashomon_overlap[i, j] for i, j in off_diag_indices])
    feature_vals = np.array([feature_similarity[i, j] for i, j in off_diag_indices])

    # Spearman correlations
    r_rashomon, p_rashomon = stats.spearmanr(confusion_vals, rashomon_vals)
    r_feature, p_feature = stats.spearmanr(confusion_vals, feature_vals)

    prediction_holds = abs(r_rashomon) > abs(r_feature)

    results = {
        "rashomon_overlap_matrix": rashomon_overlap.tolist(),
        "feature_similarity_matrix": feature_similarity.tolist(),
        "confusion_matrix_normed": cm_normed.tolist(),
        "spearman_rashomon_vs_confusion": {"r": float(r_rashomon), "p": float(p_rashomon)},
        "spearman_feature_vs_confusion": {"r": float(r_feature), "p": float(p_feature)},
        "prediction_holds": prediction_holds,
        "classes": [int(c) for c in classes],
    }

    print(f"\n  Spearman(Rashomon overlap, confusion):   r={r_rashomon:.4f}, p={p_rashomon:.4e}")
    print(f"  Spearman(Feature similarity, confusion): r={r_feature:.4f}, p={p_feature:.4e}")
    print(f"  PREDICTION HOLDS (|r_rashomon| > |r_feature|): {prediction_holds}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(res1, res2, res3):
    """Create 3-panel figure."""
    load_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # --- Panel A: Area Law ---
    ax = axes[0]
    ks = np.array(res1["ks"])
    stab = np.array(res1["stabilities"])
    ax.plot(ks, stab, "o-", color="#0072B2", markersize=4, label="Data")

    k_fine = np.linspace(1, 8, 100)
    if "params" in res1.get("exponential", {}):
        p = res1["exponential"]["params"]
        ax.plot(k_fine, p["A"] * np.exp(-p["sigma"] * k_fine) + p["C"],
                "--", color="#D55E00", label=f'Exp ($\\sigma$={p["sigma"]:.2f})')
    if "params" in res1.get("power_law", {}):
        p = res1["power_law"]["params"]
        ax.plot(k_fine, p["A"] * k_fine ** (-p["beta"]) + p["C"],
                ":", color="#009E73", label=f'Power ($\\beta$={p["beta"]:.2f})')
    if "params" in res1.get("linear", {}):
        p = res1["linear"]["params"]
        ax.plot(k_fine, p["A"] - p["B"] * k_fine,
                "-.", color="#CC79A7", label="Linear")

    ax.set_xlabel("Feature subset size $k$")
    ax.set_ylabel("Attribution stability")
    ax.set_title("A: Area law decay")
    ax.legend(fontsize=6)

    # --- Panel B: Double Descent ---
    ax = axes[1]
    degrees = np.array(res2["degrees"])
    te = np.array([x if x is not None else np.nan for x in res2["test_errors"]])
    inst = np.array([x if x is not None else np.nan for x in res2["instabilities"]])
    rash = np.array([x if x is not None else np.nan for x in res2["rashomon_sizes"]])

    # Normalize for joint plotting
    def norm01(arr):
        v = arr[~np.isnan(arr)]
        if len(v) == 0 or v.max() == v.min():
            return arr * 0
        return (arr - v.min()) / (v.max() - v.min())

    ax.plot(degrees, norm01(te), "-", color="#0072B2", label="Test error")
    ax.plot(degrees, norm01(inst), "-", color="#D55E00", label="Instability")
    ax.plot(degrees, norm01(rash), "-", color="#009E73", label="Rashomon frac")

    ax.axvline(res2["interpolation_threshold"], color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Polynomial degree $d$")
    ax.set_ylabel("Normalized value")
    ax.set_title("B: Phase transition")
    ax.legend(fontsize=6)

    # --- Panel C: Confusion vs Rashomon ---
    ax = axes[2]
    cm_normed = np.array(res3["confusion_matrix_normed"])
    rash_ov = np.array(res3["rashomon_overlap_matrix"])
    feat_sim = np.array(res3["feature_similarity_matrix"])
    n_classes = cm_normed.shape[0]
    off_diag = [(i, j) for i in range(n_classes) for j in range(n_classes) if i != j]

    confusion_vals = np.array([cm_normed[i, j] for i, j in off_diag])
    rashomon_vals = np.array([rash_ov[i, j] for i, j in off_diag])
    feature_vals = np.array([feat_sim[i, j] for i, j in off_diag])

    ax.scatter(rashomon_vals, confusion_vals, s=12, color="#D55E00", alpha=0.7,
               label=f'Rashomon ($r$={res3["spearman_rashomon_vs_confusion"]["r"]:.2f})')
    ax.scatter(feature_vals, confusion_vals, s=12, color="#0072B2", alpha=0.7, marker="^",
               label=f'Feature sim ($r$={res3["spearman_feature_vs_confusion"]["r"]:.2f})')
    ax.set_xlabel("Predictor value")
    ax.set_ylabel("Confusion rate")
    ax.set_title("C: Rashomon predicts confusion")
    ax.legend(fontsize=6)

    fig.tight_layout()
    save_figure(fig, "knockout_experiments")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    res1 = experiment_area_law()
    res2 = experiment_double_descent()
    res3 = experiment_confusion_rashomon()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exp 1 (Area law):       prediction holds = {res1['prediction_holds']}")
    print(f"  Exp 2 (Phase transition): prediction holds = {res2['prediction_holds']}")
    print(f"  Exp 3 (Confusion):       prediction holds = {res3['prediction_holds']}")

    all_results = {
        "experiment_1_area_law": res1,
        "experiment_2_double_descent": res2,
        "experiment_3_confusion_rashomon": res3,
        "summary": {
            "area_law_holds": res1["prediction_holds"],
            "phase_transition_holds": res2["prediction_holds"],
            "confusion_holds": res3["prediction_holds"],
            "total_predictions_confirmed": sum([
                res1["prediction_holds"],
                res2["prediction_holds"],
                res3["prediction_holds"],
            ]),
        },
    }

    save_results(all_results, "knockout_experiments")
    make_figure(res1, res2, res3)

    return all_results


if __name__ == "__main__":
    main()
