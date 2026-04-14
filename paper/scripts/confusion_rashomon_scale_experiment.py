#!/usr/bin/env python3
"""
Scaling experiment: Does Rashomon overlap predict confusion better than
feature similarity across progressively larger datasets and model types?

Baseline (from knockout_experiments.py):
  Digits 0-4 (5 classes, 10 pairs): Rashomon r=0.678 (p=0.001), Feature r=0.053 (p=0.823)

This tests:
  Dataset 1: Full digits (10 classes, 45 pairs)
  Dataset 2: Fashion-MNIST (10 classes, 45 pairs, 784 features)
  Dataset 3: 20 Newsgroups (20 classes, 190 pairs)

Model types: Random Forest, Gradient Boosted Trees, Logistic Regression
"""

import json
import sys
import warnings
import time
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, pairwise_distances
from sklearn.preprocessing import StandardScaler

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
# Partial correlation
# ---------------------------------------------------------------------------

def partial_corr_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y controlling for z.
    Uses the formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    where all r values are Spearman rank correlations.
    Returns (r_partial, p_value) using a t-test for significance.
    """
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)

    numer = r_xy - r_xz * r_yz
    denom = np.sqrt(max(1e-15, (1 - r_xz**2) * (1 - r_yz**2)))
    r_partial = numer / denom
    # Clip to valid range
    r_partial = np.clip(r_partial, -1.0, 1.0)

    # t-test: df = n - 3 for partial correlation
    n = len(x)
    df = n - 3
    if df <= 0:
        return r_partial, 1.0
    t_stat = r_partial * np.sqrt(df / max(1e-15, 1 - r_partial**2))
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)
    return float(r_partial), float(p_val)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_confusion_rashomon(X_train, X_test, y_train, y_test, classes,
                                model_class, model_kwargs, n_models=100,
                                seed=42, dataset_name="", model_name=""):
    """
    Train n_models classifiers, compute Rashomon overlap, feature similarity,
    actual confusion, and all correlations.
    """
    tag = f"[{dataset_name} / {model_name}]"
    n_classes = len(classes)
    print(f"\n  {tag} Training {n_models} models...")

    # Train models
    models = []
    for i in range(n_models):
        kwargs = dict(model_kwargs)
        kwargs["random_state"] = seed + i
        m = model_class(**kwargs)
        m.fit(X_train, y_train)
        models.append(m)

    # Predictions from all models
    all_preds = np.array([m.predict(X_test) for m in models])  # (n_models, n_test)

    # Average confusion matrix across all Rashomon models
    cm_sum = np.zeros((n_classes, n_classes))
    for i_model in range(n_models):
        cm_i = confusion_matrix(y_test, all_preds[i_model], labels=classes)
        cm_sum += cm_i
    cm_avg = cm_sum / n_models
    # Row-normalize
    row_sums = cm_avg.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normed = cm_avg / row_sums

    # Rashomon overlap: for each pair (i, j), mean fraction of models
    # predicting j on true-i samples, plus predicting i on true-j samples
    rashomon_overlap = {}
    for ci, i in enumerate(classes):
        for cj, j in enumerate(classes):
            if i == j:
                continue
            mask_i = y_test == i
            mask_j = y_test == j
            frac_ij = 0.0
            if mask_i.sum() > 0:
                frac_ij = np.mean(all_preds[:, mask_i] == j)
            frac_ji = 0.0
            if mask_j.sum() > 0:
                frac_ji = np.mean(all_preds[:, mask_j] == i)
            rashomon_overlap[(ci, cj)] = (frac_ij + frac_ji) / 2

    # Feature similarity: cosine similarity of class centroids
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    centroid_dists = pairwise_distances(centroids, metric="cosine")
    feature_similarity = 1.0 - centroid_dists

    # Extract off-diagonal pairs (upper triangle to avoid double-counting)
    pairs = list(combinations(range(n_classes), 2))
    n_pairs = len(pairs)

    confusion_vals = np.array([(cm_normed[i, j] + cm_normed[j, i]) / 2 for i, j in pairs])
    rashomon_vals = np.array([rashomon_overlap[(i, j)] for i, j in pairs])
    feature_vals = np.array([feature_similarity[i, j] for i, j in pairs])

    # Spearman correlations
    r_rash, p_rash = stats.spearmanr(confusion_vals, rashomon_vals)
    r_feat, p_feat = stats.spearmanr(confusion_vals, feature_vals)

    # Partial correlation: Rashomon overlap predicting confusion, controlling for feature similarity
    r_partial, p_partial = partial_corr_spearman(confusion_vals, rashomon_vals, feature_vals)

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "n_classes": n_classes,
        "n_pairs": n_pairs,
        "n_models": n_models,
        "spearman_rashomon": {"r": round(float(r_rash), 4), "p": float(p_rash)},
        "spearman_feature": {"r": round(float(r_feat), 4), "p": float(p_feat)},
        "partial_corr_rashomon_controlling_feature": {
            "r": round(float(r_partial), 4), "p": float(p_partial)
        },
        "rashomon_wins": abs(r_rash) > abs(r_feat),
    }

    print(f"  {tag} {n_pairs} class pairs")
    print(f"  {tag} Spearman(Rashomon, confusion):   r={r_rash:.4f}, p={p_rash:.2e}")
    print(f"  {tag} Spearman(Feature, confusion):    r={r_feat:.4f}, p={p_feat:.2e}")
    print(f"  {tag} Partial(Rashomon|Feature):        r={r_partial:.4f}, p={p_partial:.2e}")
    print(f"  {tag} RASHOMON WINS: {abs(r_rash) > abs(r_feat)}")

    return result, confusion_vals, rashomon_vals, feature_vals


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_full_digits(seed=42):
    """Full sklearn digits: 10 classes, 64 features."""
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    classes = sorted(np.unique(y))
    return X_train, X_test, y_train, y_test, classes


def load_fashion_mnist(seed=42):
    """Fashion-MNIST: 10 classes, 784 features. Use fetch_openml."""
    from sklearn.datasets import fetch_openml
    print("  Fetching Fashion-MNIST from OpenML...")
    fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    X, y = fmnist.data.astype(np.float32), fmnist.target.astype(int)
    # Subsample to 10k for speed (still much larger than digits)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(10000, len(X)), replace=False)
    X, y = X[idx], y[idx]
    # Scale
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    classes = sorted(np.unique(y))
    return X_train, X_test, y_train, y_test, classes


def load_newsgroups(seed=42):
    """20 Newsgroups: 20 classes, TF-IDF features."""
    print("  Fetching 20 Newsgroups...")
    train_data = fetch_20newsgroups(subset="train", random_state=seed, remove=("headers", "footers", "quotes"))
    test_data = fetch_20newsgroups(subset="test", random_state=seed, remove=("headers", "footers", "quotes"))

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_data.data).toarray()
    X_test = vectorizer.transform(test_data.data).toarray()
    y_train = train_data.target
    y_test = test_data.target
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    return X_train, X_test, y_train, y_test, classes


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "RandomForest": (
        RandomForestClassifier,
        {"n_estimators": 50, "max_depth": 10, "max_features": "sqrt"}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier,
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8}
    ),
    "LogisticRegression": (
        LogisticRegression,
        {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "multi_class": "multinomial"}
    ),
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    seed = 42
    set_all_seeds(seed)
    n_models = 100

    all_results = {}
    # Store arrays for the figure
    figure_data = {}

    print("=" * 70)
    print("SCALING EXPERIMENT: Rashomon Overlap vs Feature Similarity")
    print("=" * 70)

    # --- Dataset 1: Full Digits ---
    print("\n" + "-" * 60)
    print("DATASET 1: Full Digits (10 classes, 45 pairs)")
    print("-" * 60)
    X_tr, X_te, y_tr, y_te, classes = load_full_digits(seed)
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Features: {X_tr.shape[1]}")
    for mname, (mclass, mkwargs) in MODEL_CONFIGS.items():
        res, c_vals, r_vals, f_vals = analyze_confusion_rashomon(
            X_tr, X_te, y_tr, y_te, classes, mclass, mkwargs,
            n_models=n_models, seed=seed,
            dataset_name="Full Digits", model_name=mname
        )
        all_results[f"full_digits_{mname}"] = res
        figure_data[f"full_digits_{mname}"] = (c_vals, r_vals, f_vals)

    # --- Dataset 2: Fashion-MNIST ---
    print("\n" + "-" * 60)
    print("DATASET 2: Fashion-MNIST (10 classes, 45 pairs)")
    print("-" * 60)
    try:
        X_tr, X_te, y_tr, y_te, classes = load_fashion_mnist(seed)
        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Features: {X_tr.shape[1]}")
        for mname, (mclass, mkwargs) in MODEL_CONFIGS.items():
            res, c_vals, r_vals, f_vals = analyze_confusion_rashomon(
                X_tr, X_te, y_tr, y_te, classes, mclass, mkwargs,
                n_models=n_models, seed=seed,
                dataset_name="Fashion-MNIST", model_name=mname
            )
            all_results[f"fashion_mnist_{mname}"] = res
            figure_data[f"fashion_mnist_{mname}"] = (c_vals, r_vals, f_vals)
    except Exception as e:
        print(f"  Fashion-MNIST FAILED: {e}")
        all_results["fashion_mnist_error"] = str(e)

    # --- Dataset 3: 20 Newsgroups ---
    print("\n" + "-" * 60)
    print("DATASET 3: 20 Newsgroups (20 classes, 190 pairs)")
    print("-" * 60)
    X_tr, X_te, y_tr, y_te, classes = load_newsgroups(seed)
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Features: {X_tr.shape[1]}")
    for mname, (mclass, mkwargs) in MODEL_CONFIGS.items():
        res, c_vals, r_vals, f_vals = analyze_confusion_rashomon(
            X_tr, X_te, y_tr, y_te, classes, mclass, mkwargs,
            n_models=n_models, seed=seed,
            dataset_name="20 Newsgroups", model_name=mname
        )
        all_results[f"newsgroups_{mname}"] = res
        figure_data[f"newsgroups_{mname}"] = (c_vals, r_vals, f_vals)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Key':<40s} {'r_Rash':>8s} {'p_Rash':>10s} {'r_Feat':>8s} {'p_Feat':>10s} {'r_Part':>8s} {'p_Part':>10s} {'Wins?':>6s}")
    print("-" * 100)
    for key, res in all_results.items():
        if isinstance(res, str):
            print(f"  {key}: {res}")
            continue
        rr = res["spearman_rashomon"]
        rf = res["spearman_feature"]
        pc = res["partial_corr_rashomon_controlling_feature"]
        print(f"  {key:<38s} {rr['r']:>8.4f} {rr['p']:>10.2e} {rf['r']:>8.4f} {rf['p']:>10.2e} {pc['r']:>8.4f} {pc['p']:>10.2e} {'YES' if res['rashomon_wins'] else 'NO':>6s}")

    # --- Save results ---
    save_results(all_results, "confusion_rashomon_scale")

    # --- Figure ---
    make_figure(all_results, figure_data)

    return all_results


def make_figure(all_results, figure_data):
    """
    Panel A: Scatter of Rashomon overlap vs confusion for 20 Newsgroups (190 points)
    Panel B: Bar chart comparing Spearman rho across all datasets
    Panel C: Partial correlation (controlling for feature similarity) across datasets
    """
    load_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # --- Panel A: 20 Newsgroups scatter ---
    ax = axes[0]
    # Find the newsgroups + RandomForest result for primary scatter
    ng_key = "newsgroups_RandomForest"
    if ng_key in figure_data:
        c_vals, r_vals, f_vals = figure_data[ng_key]
        ax.scatter(r_vals, c_vals, s=8, color="#D55E00", alpha=0.5, label="Rashomon overlap", zorder=3)
        ax.scatter(f_vals, c_vals, s=8, color="#0072B2", alpha=0.5, marker="^", label="Feature similarity", zorder=2)
        # Regression lines
        if len(r_vals) > 2:
            slope_r, intercept_r, _, _, _ = stats.linregress(r_vals, c_vals)
            x_fit = np.linspace(r_vals.min(), r_vals.max(), 50)
            ax.plot(x_fit, slope_r * x_fit + intercept_r, "--", color="#D55E00", lw=1.2, alpha=0.8)
        if len(f_vals) > 2:
            slope_f, intercept_f, _, _, _ = stats.linregress(f_vals, c_vals)
            x_fit = np.linspace(f_vals.min(), f_vals.max(), 50)
            ax.plot(x_fit, slope_f * x_fit + intercept_f, "--", color="#0072B2", lw=1.2, alpha=0.8)
        res = all_results[ng_key]
        rr = res["spearman_rashomon"]["r"]
        rf = res["spearman_feature"]["r"]
        ax.set_title(f"A: 20 Newsgroups (190 pairs)\nRashomon $\\rho$={rr:.3f}, Feature $\\rho$={rf:.3f}", fontsize=8)
    ax.set_xlabel("Predictor value")
    ax.set_ylabel("Confusion rate")
    ax.legend(fontsize=6, loc="upper left")

    # --- Panel B: Bar chart of Spearman rho across datasets ---
    ax = axes[1]
    # Group by dataset
    dataset_order = [
        ("Full Digits", "full_digits"),
        ("Fashion-MNIST", "fashion_mnist"),
        ("20 Newsgroups", "newsgroups"),
    ]
    model_names = ["RandomForest", "GradientBoosting", "LogisticRegression"]
    model_short = ["RF", "GBT", "LR"]
    colors_rash = ["#D55E00", "#E69F00", "#CC79A7"]
    colors_feat = ["#0072B2", "#56B4E9", "#009E73"]

    x_pos = []
    x_labels = []
    rho_rash_list = []
    rho_feat_list = []
    pos = 0
    group_centers = []
    for ds_label, ds_prefix in dataset_order:
        group_start = pos
        for mi, mname in enumerate(model_names):
            key = f"{ds_prefix}_{mname}"
            if key in all_results and isinstance(all_results[key], dict):
                rho_rash_list.append(all_results[key]["spearman_rashomon"]["r"])
                rho_feat_list.append(all_results[key]["spearman_feature"]["r"])
            else:
                rho_rash_list.append(0)
                rho_feat_list.append(0)
            x_pos.append(pos)
            x_labels.append(model_short[mi])
            pos += 1
        group_centers.append((group_start + pos - 1) / 2)
        pos += 0.5  # gap between groups

    x_pos = np.array(x_pos)
    bar_w = 0.35
    ax.bar(x_pos - bar_w/2, rho_rash_list, bar_w, color="#D55E00", alpha=0.8, label="Rashomon overlap")
    ax.bar(x_pos + bar_w/2, rho_feat_list, bar_w, color="#0072B2", alpha=0.8, label="Feature similarity")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=6)
    # Add dataset labels
    for gc, (ds_label, _) in zip(group_centers, dataset_order):
        ax.text(gc, -0.08, ds_label, ha="center", va="top", fontsize=6,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("B: Rashomon vs Feature similarity", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- Panel C: Partial correlation across datasets ---
    ax = axes[2]
    partial_vals = []
    partial_ps = []
    bar_labels = []
    bar_colors = []
    color_map = {"RandomForest": "#D55E00", "GradientBoosting": "#E69F00", "LogisticRegression": "#CC79A7"}

    pos = 0
    x_pos2 = []
    group_centers2 = []
    for ds_label, ds_prefix in dataset_order:
        group_start = pos
        for mi, mname in enumerate(model_names):
            key = f"{ds_prefix}_{mname}"
            if key in all_results and isinstance(all_results[key], dict):
                pc = all_results[key]["partial_corr_rashomon_controlling_feature"]
                partial_vals.append(pc["r"])
                partial_ps.append(pc["p"])
            else:
                partial_vals.append(0)
                partial_ps.append(1.0)
            bar_labels.append(model_short[mi])
            bar_colors.append(color_map[mname])
            x_pos2.append(pos)
            pos += 1
        group_centers2.append((group_start + pos - 1) / 2)
        pos += 0.5

    x_pos2 = np.array(x_pos2)
    bars = ax.bar(x_pos2, partial_vals, 0.6, color=bar_colors, alpha=0.8)

    # Add significance stars
    for i, (v, p) in enumerate(zip(partial_vals, partial_ps)):
        star = ""
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        if star:
            ax.text(x_pos2[i], v + 0.02, star, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x_pos2)
    ax.set_xticklabels(bar_labels, fontsize=6)
    for gc, (ds_label, _) in zip(group_centers2, dataset_order):
        ax.text(gc, -0.08, ds_label, ha="center", va="top", fontsize=6,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Partial Spearman $\\rho$\n(controlling feature sim.)")
    ax.set_title("C: Partial correlation", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    save_figure(fig, "confusion_rashomon_scale")


if __name__ == "__main__":
    results = main()
