"""
Counterfactual Explanation Instability Experiment (Task 1B)

Research question: Do equivalent-accuracy models produce contradictory
counterfactual explanations?

Dataset: German Credit (UCI via fetch_openml('credit-g'))
Models: 20 XGBoost classifiers with identical hyperparameters, different seeds
Method: Greedy feature-importance-ordered perturbation toward positive centroid

NEGATIVE CONTROL:
- Synthetic data with moderate class separation (class_sep=0.5, no redundant features)
- subsample=1.0 means training is deterministic → models are nearly identical
- CFs are found (boundary is reachable) and stable (low flip rate because all models agree)
- Expected: CF found rate >0%, direction flip rate <5%

RESOLUTION TEST:
- Consensus counterfactual = feature-wise median of CFs across all 20 models
- Measure: fraction of individual models' CFs consistent with the consensus
  (same direction on top features) — expected >80%
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── import experiment utilities ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from experiment_utils import (
    set_all_seeds, load_publication_style, save_figure, save_results,
    percentile_ci
)

from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

PAPER_DIR = SCRIPT_DIR.parent
SECTIONS_DIR = PAPER_DIR / "sections"
SECTIONS_DIR.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_german_credit():
    """Load German Credit, one-hot encode categoricals, standardize numerics."""
    print("Loading German Credit dataset …")
    data = fetch_openml('credit-g', version=1, as_frame=True)
    X_raw = data.data.copy()
    y_raw = data.target.copy()

    # Encode target: 'good' → 0, 'bad' → 1  (positive class = bad credit)
    y = (y_raw == 'bad').astype(int).values

    cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Handle missing values
    for c in num_cols:
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna(X_raw[c].mode()[0])

    # One-hot encode categoricals
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)
    feature_names = X_enc.columns.tolist()
    X_arr = X_enc.values.astype(float)

    # Standardize numerics only
    num_idx_set = set()
    for n in num_cols:
        for i, f in enumerate(feature_names):
            if f == n:
                num_idx_set.add(i)

    scaler = StandardScaler()
    X_arr[:, list(num_idx_set)] = scaler.fit_transform(X_arr[:, list(num_idx_set)])

    print(f"  Samples: {X_arr.shape[0]}, Features: {X_arr.shape[1]}")
    print(f"  Categorical → one-hot: {len(cat_cols)} cols → {X_arr.shape[1] - len(num_cols)} binary cols")
    return X_arr, y, feature_names


def train_rashomon_xgb(X_train, y_train, n_models=20, subsample=0.8):
    """Train n_models XGBoost classifiers with different seeds."""
    models = []
    for i in range(n_models):
        m = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            subsample=subsample,
            random_state=42 + i,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        m.fit(X_train, y_train)
        models.append(m)
    return models


def verify_auc_equivalence(models, X_test, y_test, tol=0.03):
    """Verify all models achieve equivalent AUC (within tol)."""
    aucs = []
    for m in models:
        prob = m.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, prob))
    aucs = np.array(aucs)
    auc_range = aucs.max() - aucs.min()
    print(f"  AUC range: [{aucs.min():.4f}, {aucs.max():.4f}] (spread={auc_range:.4f})")
    assert auc_range <= tol, f"AUC spread {auc_range:.4f} exceeds tolerance {tol}"
    return aucs


def get_gain_importance(model, n_features):
    """Return normalized gain-based feature importance (length = n_features)."""
    scores = model.get_booster().get_score(importance_type='gain')
    imp = np.zeros(n_features)
    for fname, val in scores.items():
        idx = int(fname.replace('f', ''))
        imp[idx] = val
    total = imp.sum()
    if total > 0:
        imp /= total
    return imp


def greedy_counterfactual(x0, model, feature_stds, pos_centroid, n_features,
                           max_steps=50, delta_scale=0.1):
    """
    Find a counterfactual by greedy perturbation.

    Iterates through features in importance order, applying repeated small
    perturbations toward the positive-class centroid.  Allows multiple passes
    over features (cycling) up to max_steps total perturbation steps.

    Returns:
        x_cf       : counterfactual point
        cf_dir     : unit-direction vector (x_cf - x0), nonzero entries only
        found      : bool — True if prediction flipped
    """
    imp = get_gain_importance(model, n_features)
    feature_order = np.argsort(imp)[::-1]

    x = x0.copy()
    orig_pred = model.predict(x.reshape(1, -1))[0]

    for step in range(max_steps):
        feat = feature_order[step % n_features]
        delta = delta_scale * feature_stds[feat]
        diff = pos_centroid[feat] - x[feat]
        direction = np.sign(diff) if diff != 0 else 1.0
        x[feat] += direction * delta

        new_pred = model.predict(x.reshape(1, -1))[0]
        if new_pred != orig_pred:
            cf_diff = x - x0
            cf_dir = np.sign(cf_diff)  # +1 / -1 / 0
            return x.copy(), cf_dir, True

    cf_diff = x - x0
    cf_dir = np.sign(cf_diff)
    return x.copy(), cf_dir, False


def _pair_disagree(da, db):
    """True if the two direction recommendations disagree."""
    if da == 0 and db == 0:
        return False
    if da != 0 and db != 0 and np.sign(da) == np.sign(db):
        return False
    return True


def compute_direction_flip_rate(directions, found_flags, n_features, n_queries, n_models):
    """Compute per-feature direction flip rates and overall mean."""
    flip_rates_per_feature = np.zeros(n_features)
    all_pair_flips_per_feature = [[] for _ in range(n_features)]
    for fi in range(n_features):
        pair_flips = []
        for qi in range(n_queries):
            d = directions[qi, :, fi]
            found_qi = found_flags[qi]
            for ma in range(n_models):
                for mb in range(ma + 1, n_models):
                    if not (found_qi[ma] or found_qi[mb]):
                        continue
                    pair_flips.append(int(_pair_disagree(d[ma], d[mb])))
        flip_rates_per_feature[fi] = np.mean(pair_flips) if pair_flips else 0.0
        all_pair_flips_per_feature[fi] = pair_flips

    flip_ci_lo = np.zeros(n_features)
    flip_ci_hi = np.zeros(n_features)
    for fi in range(n_features):
        pair_flips = all_pair_flips_per_feature[fi]
        if len(pair_flips) >= 2:
            lo, _, hi = percentile_ci(pair_flips, n_boot=500)
        else:
            lo, hi = 0.0, 0.0
        flip_ci_lo[fi] = lo
        flip_ci_hi[fi] = hi

    overall_flip = float(flip_rates_per_feature.mean())
    lo_f, _, hi_f = percentile_ci(flip_rates_per_feature.tolist(), n_boot=500)
    return flip_rates_per_feature, flip_ci_lo, flip_ci_hi, overall_flip, lo_f, hi_f


# ── NEGATIVE CONTROL ─────────────────────────────────────────────────────────

def run_negative_control_cf():
    """
    Negative control: synthetic data with moderate class separation (class_sep=1.0).
    No redundant features.  subsample=1.0 means training is deterministic across seeds,
    so models are nearly identical.  The decision boundary is close enough that greedy
    perturbation can find counterfactuals.  CFs should be FOUND and STABLE (low flip
    rate because all models agree on the boundary).
    Expected: CF found rate >0%, direction flip rate <5%.
    """
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROL: Moderate-separation synthetic data (class_sep=1.0, subsample=1.0)")
    print("=" * 60)

    # Generate data with some overlap — boundary is reachable by greedy perturbation
    X_ctrl, y_ctrl = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=0.5,          # moderate separation → CFs findable; boundary stable
        random_state=42,
    )
    feature_names_ctrl = [f"feat_{i}" for i in range(10)]

    scaler_ctrl = StandardScaler()
    X_ctrl = scaler_ctrl.fit_transform(X_ctrl)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ctrl, y_ctrl, test_size=0.2, random_state=42, stratify=y_ctrl
    )

    print(f"  Synthetic data: {X_ctrl.shape[0]} samples, {X_ctrl.shape[1]} features")

    # Train 20 models — subsample=1.0 for maximum consistency
    models_ctrl = train_rashomon_xgb(X_tr, y_tr, n_models=20, subsample=1.0)

    # AUC check
    aucs_ctrl = np.array([roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
                          for m in models_ctrl])
    print(f"  AUC range: [{aucs_ctrl.min():.4f}, {aucs_ctrl.max():.4f}] "
          f"(spread={aucs_ctrl.max() - aucs_ctrl.min():.4f})")

    # Select query points (class 1 predicted by majority)
    test_preds_ctrl = np.array([m.predict(X_te) for m in models_ctrl])
    majority_pred_ctrl = (test_preds_ctrl.mean(axis=0) >= 0.5).astype(int)
    bad_idx_ctrl = np.where(majority_pred_ctrl == 1)[0]
    rng_ctrl = np.random.RandomState(42)
    n_q_ctrl = min(30, len(bad_idx_ctrl))
    query_idx_ctrl = rng_ctrl.choice(bad_idx_ctrl, size=n_q_ctrl, replace=False)
    print(f"  Query points: {n_q_ctrl}")

    # Compute CFs
    n_features_ctrl = X_ctrl.shape[1]
    pos_centroid_ctrl = X_tr[y_tr == 1].mean(axis=0)
    feature_stds_ctrl = X_tr.std(axis=0)
    feature_stds_ctrl[feature_stds_ctrl == 0] = 1.0

    n_models_ctrl = len(models_ctrl)
    directions_ctrl = np.zeros((n_q_ctrl, n_models_ctrl, n_features_ctrl))
    found_ctrl = np.zeros((n_q_ctrl, n_models_ctrl), dtype=bool)

    for qi, tidx in enumerate(query_idx_ctrl):
        x0 = X_te[tidx]
        for mi, model in enumerate(models_ctrl):
            x_cf, dirs, found = greedy_counterfactual(
                x0, model, feature_stds_ctrl, pos_centroid_ctrl,
                n_features_ctrl, max_steps=50, delta_scale=0.1
            )
            directions_ctrl[qi, mi] = dirs
            found_ctrl[qi, mi] = found

    cf_found_rate_ctrl = found_ctrl.mean()
    print(f"  CF found rate: {cf_found_rate_ctrl:.3f}")

    _, _, _, nc_flip_rate, nc_lo, nc_hi = compute_direction_flip_rate(
        directions_ctrl, found_ctrl, n_features_ctrl, n_q_ctrl, n_models_ctrl
    )

    print(f"  Direction flip rate: {nc_flip_rate:.3f} [{nc_lo:.3f}, {nc_hi:.3f}]")
    print(f"  Expected: CF found rate >0%, flip rate <0.05 (deterministic training → stable boundary)")

    return {
        "description": "make_classification class_sep=0.5, subsample=1.0",
        "n_models": n_models_ctrl,
        "n_queries": n_q_ctrl,
        "auc_range": float(aucs_ctrl.max() - aucs_ctrl.min()),
        "auc_min": float(aucs_ctrl.min()),
        "auc_max": float(aucs_ctrl.max()),
        "cf_found_rate": float(cf_found_rate_ctrl),
        "direction_flip_rate": float(nc_flip_rate),
        "direction_flip_rate_ci_lo": float(nc_lo),
        "direction_flip_rate_ci_hi": float(nc_hi),
        "interpretation": "Expected CF found rate >0% and flip rate <5%; validates instability is from Rashomon ambiguity",
    }


# ── RESOLUTION TEST ───────────────────────────────────────────────────────────

def run_resolution_test_cf(models, directions, found_flags, n_queries, n_features, feature_names):
    """
    Resolution test: consensus counterfactual = feature-wise median direction.

    For each query, compute the median direction across all 20 models.
    Measure: fraction of individual model CFs that are consistent with the consensus
    (same direction on TOP-5 features by average importance).

    Expected: >80% consistency.
    """
    print("\n" + "=" * 60)
    print("RESOLUTION TEST: Consensus counterfactual (feature-wise median)")
    print("=" * 60)

    n_models = len(models)

    # Average feature importance across models
    avg_imp = np.array([get_gain_importance(m, n_features) for m in models]).mean(axis=0)
    top5_idx = np.argsort(avg_imp)[::-1][:5]
    print(f"  Top-5 features by avg importance: {[feature_names[i] for i in top5_idx]}")

    # Consensus: feature-wise median direction across models for each query
    consistency_per_query = []
    for qi in range(n_queries):
        found_qi = found_flags[qi]
        if found_qi.sum() < 2:
            continue

        # Only use models that found a CF
        found_models = np.where(found_qi)[0]
        dir_qi = directions[qi][found_models]   # (n_found, n_features)

        # Median direction (for top 5 features)
        consensus_dir = np.median(dir_qi[:, top5_idx], axis=0)  # (5,)
        consensus_sign = np.sign(consensus_dir)  # may be 0 if tied

        # For each individual model, count agreement on non-zero consensus features
        n_consistent = 0
        n_models_found = len(found_models)
        for mi_local in range(n_models_found):
            ind_dir = dir_qi[mi_local, top5_idx]
            # Agreement on feature f: both push same nonzero direction
            agree_count = 0
            total_nz = 0
            for f in range(len(top5_idx)):
                if consensus_sign[f] != 0:
                    total_nz += 1
                    if ind_dir[f] != 0 and np.sign(ind_dir[f]) == consensus_sign[f]:
                        agree_count += 1
            if total_nz > 0:
                consistency = agree_count / total_nz
            else:
                consistency = 1.0
            if consistency >= 0.6:   # "consistent" = agree on majority of top features
                n_consistent += 1

        consistency_per_query.append(n_consistent / n_models_found)

    consistency_arr = np.array(consistency_per_query)
    mean_consistency = float(np.mean(consistency_arr)) if len(consistency_arr) > 0 else 0.0
    lo_c, _, hi_c = percentile_ci(consistency_arr.tolist(), n_boot=500)

    print(f"  Mean consistency with consensus: {mean_consistency:.3f} [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"  Fraction >80%: {mean_consistency:.1%}  (expected >80%)")

    return {
        "description": "feature-wise median consensus CF; consistency = same direction on top-5 features",
        "top5_feature_names": [feature_names[i] for i in top5_idx.tolist()],
        "mean_consistency_with_consensus": float(mean_consistency),
        "consistency_ci_lo": float(lo_c),
        "consistency_ci_hi": float(hi_c),
        "interpretation": "Expected >80% of individual CFs agree with consensus direction",
    }


# ── main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    set_all_seeds(42)
    load_publication_style()

    # 1. Data
    X, y, feature_names = load_german_credit()
    n_features = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Positive-class centroid and feature stds (from training data)
    pos_mask = y_train == 1
    pos_centroid = X_train[pos_mask].mean(axis=0)
    feature_stds = X_train.std(axis=0)
    feature_stds[feature_stds == 0] = 1.0

    # ─── POSITIVE TEST ──────────────────────────────────────────────────────
    print("=" * 60)
    print("POSITIVE TEST: German Credit Rashomon set (20 XGBoost models)")
    print("=" * 60)

    print("\nTraining 20 XGBoost classifiers …")
    models = train_rashomon_xgb(X_train, y_train, n_models=20)
    print("Verifying AUC equivalence …")
    aucs = verify_auc_equivalence(models, X_test, y_test, tol=0.03)
    n_models = len(models)

    # Select 50 test points
    print("\nSelecting 50 test query points …")
    test_preds = np.array([m.predict(X_test) for m in models])
    majority_pred = (test_preds.mean(axis=0) >= 0.5).astype(int)
    bad_credit_idx = np.where(majority_pred == 1)[0]
    rng = np.random.RandomState(42)
    if len(bad_credit_idx) >= 50:
        query_idx = rng.choice(bad_credit_idx, size=50, replace=False)
    else:
        query_idx = bad_credit_idx
        print(f"  Warning: only {len(bad_credit_idx)} bad-credit test points found")

    n_queries = len(query_idx)
    print(f"  Selected {n_queries} query points from {len(bad_credit_idx)} bad-credit predictions")

    # Compute counterfactuals
    print("\nComputing counterfactuals …")
    cfs = np.zeros((n_queries, n_models, n_features))
    directions = np.zeros((n_queries, n_models, n_features))
    found_flags = np.zeros((n_queries, n_models), dtype=bool)

    for qi, tidx in enumerate(query_idx):
        x0 = X_test[tidx]
        for mi, model in enumerate(models):
            x_cf, dirs, found = greedy_counterfactual(
                x0, model, feature_stds, pos_centroid, n_features,
                max_steps=50, delta_scale=0.1
            )
            cfs[qi, mi] = x_cf
            directions[qi, mi] = dirs
            found_flags[qi, mi] = found

    cf_found_rate = found_flags.mean()
    print(f"  CF found rate: {cf_found_rate:.3f}")

    # Metrics
    print("\nComputing metrics …")

    flip_rates_per_feature, flip_ci_lo, flip_ci_hi, overall_flip, lo_f, hi_f = \
        compute_direction_flip_rate(directions, found_flags, n_features, n_queries, n_models)

    # Cross-model recourse validity
    validity_per_query = []
    for qi in range(n_queries):
        x0 = X_test[query_idx[qi]]
        vals = []
        for mi in range(n_models):
            if not found_flags[qi, mi]:
                continue
            x_cf = cfs[qi, mi]
            for mj in range(n_models):
                if mi == mj:
                    continue
                pred_cf = models[mj].predict(x_cf.reshape(1, -1))[0]
                orig_pred_mj = models[mj].predict(x0.reshape(1, -1))[0]
                vals.append(int(pred_cf != orig_pred_mj))
        validity_per_query.append(np.mean(vals) if vals else np.nan)

    validity_per_query = np.array(validity_per_query)
    valid_mask = ~np.isnan(validity_per_query)
    mean_validity = float(validity_per_query[valid_mask].mean())
    lo_v, _, hi_v = percentile_ci(validity_per_query[valid_mask].tolist(), n_boot=500)
    print(f"  Cross-model recourse validity: {mean_validity:.3f} [{lo_v:.3f}, {hi_v:.3f}]")

    # CF distances
    cf_distances_per_query = []
    for qi in range(n_queries):
        x0 = X_test[query_idx[qi]]
        dists = []
        for mi in range(n_models):
            if found_flags[qi, mi]:
                dists.append(np.linalg.norm(cfs[qi, mi] - x0))
        cf_distances_per_query.append(np.mean(dists) if dists else np.nan)
    cf_distances_per_query = np.array(cf_distances_per_query)

    cv_per_query = []
    for qi in range(n_queries):
        x0 = X_test[query_idx[qi]]
        dists = [np.linalg.norm(cfs[qi, mi] - x0)
                 for mi in range(n_models) if found_flags[qi, mi]]
        if len(dists) >= 2:
            cv = np.std(dists) / (np.mean(dists) + 1e-9)
        else:
            cv = np.nan
        cv_per_query.append(cv)
    cv_per_query = np.array(cv_per_query)
    cv_valid = cv_per_query[~np.isnan(cv_per_query)]
    lo_cv, mean_cv, hi_cv = percentile_ci(cv_valid.tolist(), n_boot=500)
    print(f"  Distance CV: {mean_cv:.3f} [{lo_cv:.3f}, {hi_cv:.3f}]")

    avg_imp = np.array([get_gain_importance(m, n_features) for m in models]).mean(axis=0)
    top_feature = int(np.argmax(avg_imp))
    consensus_per_query = []
    for qi in range(n_queries):
        d = directions[qi, :, top_feature]
        nonzero = d[d != 0]
        if len(nonzero) > 0:
            consensus = max((nonzero > 0).mean(), (nonzero < 0).mean())
        else:
            consensus = 0.5
        consensus_per_query.append(consensus)
    consensus_per_query = np.array(consensus_per_query)

    print(f"  Overall direction flip rate: {overall_flip:.3f} [{lo_f:.3f}, {hi_f:.3f}]")

    # ─── NEGATIVE CONTROL ───────────────────────────────────────────────────
    nc_results = run_negative_control_cf()

    # ─── RESOLUTION TEST ────────────────────────────────────────────────────
    res_results = run_resolution_test_cf(
        models, directions, found_flags, n_queries, n_features, feature_names
    )

    # ─── Figure ─────────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    top10_idx = np.argsort(flip_rates_per_feature)[::-1][:10]
    top10_names = [feature_names[i] for i in top10_idx]
    top10_rates = flip_rates_per_feature[top10_idx]
    top10_lo = flip_ci_lo[top10_idx]
    top10_hi = flip_ci_hi[top10_idx]
    err_lo = top10_rates - top10_lo
    err_hi = top10_hi - top10_rates

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    # Left: bar chart of direction flip rate (positive test)
    ax = axes[0]
    colors_bar = ['#D55E00' if r > 0.3 else '#0072B2' for r in top10_rates]
    bars = ax.barh(range(10), top10_rates[::-1], color=colors_bar[::-1],
                   xerr=[err_lo[::-1], err_hi[::-1]],
                   error_kw=dict(elinewidth=0.8, capsize=2, ecolor='#555555'),
                   height=0.65)
    ax.axvline(0.5, color='#555555', linestyle='--', linewidth=0.8, label='Coin flip (50%)')
    ax.set_yticks(range(10))
    short_names = []
    for n in top10_names[::-1]:
        if len(n) > 22:
            n = n[:19] + '...'
        short_names.append(n)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Direction flip rate')
    ax.set_title('(a) CF direction instability\n(top 10 features, positive test)', fontsize=9)
    ax.set_xlim(0, max(0.65, top10_rates.max() + 0.12))
    ax.legend(fontsize=7, loc='lower right')

    # Middle: scatter cross-model validity vs CF distance
    ax2 = axes[1]
    dist_nonan = cf_distances_per_query.copy()
    val_nonan = validity_per_query.copy()
    cons_nonan = consensus_per_query.copy()
    mask = ~(np.isnan(dist_nonan) | np.isnan(val_nonan))
    d_plot = dist_nonan[mask]
    v_plot = val_nonan[mask]
    c_plot = cons_nonan[mask]

    sc = ax2.scatter(d_plot, v_plot, c=c_plot, cmap='RdYlGn',
                     vmin=0.5, vmax=1.0, s=18, alpha=0.75, linewidths=0.3,
                     edgecolors='#333333')
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label('Consensus level', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    if len(d_plot) >= 3:
        slope, intercept, r, p, _ = linregress(d_plot, v_plot)
        x_line = np.linspace(d_plot.min(), d_plot.max(), 100)
        y_line = slope * x_line + intercept
        ax2.plot(x_line, y_line, color='#555555', linewidth=0.9, linestyle='--',
                 label=f'r={r:.2f}')
        ax2.legend(fontsize=7)

    ax2.set_xlabel('Mean CF distance $\\|x\\prime - x_0\\|$')
    ax2.set_ylabel('Cross-model recourse validity')
    ax2.set_title('(b) Validity vs CF distance\n(color = direction consensus)', fontsize=9)

    # Right: comparison of positive / neg control / resolution
    ax3 = axes[2]
    bar_labels_comp = ['Positive\n(Rashomon)', 'Neg. Control\n(class_sep=0.5)', 'Resolution\n(consensus)']
    bar_vals_comp = [overall_flip, nc_results['direction_flip_rate'],
                     1.0 - res_results['mean_consistency_with_consensus']]
    bar_colors_comp = ['#D55E00', '#009E73', '#0072B2']
    bar_errs_lo_comp = [overall_flip - lo_f,
                        nc_results['direction_flip_rate'] - nc_results['direction_flip_rate_ci_lo'],
                        0.0]
    bar_errs_hi_comp = [hi_f - overall_flip,
                        nc_results['direction_flip_rate_ci_hi'] - nc_results['direction_flip_rate'],
                        0.0]

    xs3 = np.arange(len(bar_labels_comp))
    bars3 = ax3.bar(xs3, bar_vals_comp, color=bar_colors_comp, alpha=0.8, width=0.55,
                    yerr=[bar_errs_lo_comp, bar_errs_hi_comp],
                    error_kw=dict(elinewidth=1.0, capsize=3, ecolor='#333333'))
    ax3.set_xticks(xs3)
    ax3.set_xticklabels(bar_labels_comp, fontsize=7)
    ax3.set_ylabel('Direction flip rate (or 1-consistency)')
    ax3.set_title('(c) Positive / Control /\nResolution comparison', fontsize=9)
    ax3.set_ylim(0, 1.05)
    ax3.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars3, bar_vals_comp):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.tight_layout(pad=1.0)
    save_figure(fig, 'counterfactual_instability')

    # ─── LaTeX table ─────────────────────────────────────────────────────────
    print("Writing LaTeX table …")
    tex_path = SECTIONS_DIR / 'table_counterfactual.tex'

    lo_v_pct = lo_v * 100
    hi_v_pct = hi_v * 100
    mean_v_pct = mean_validity * 100

    tex = r"""\begin{table}[t]
\centering
\caption{Counterfactual explanation instability on German Credit (20 equivalent XGBoost models, AUC within 0.03).
  \emph{Negative control}: synthetic data with class\_sep=0.5 and subsample=1.0 --- deterministic training means models are nearly identical; CFs are found and stable, expected flip rate $<5\%$.
  \emph{Resolution}: consensus CF (feature-wise median direction) --- fraction of individual CFs consistent with consensus.
  All 95\% bootstrap CIs from 500 resamples.}
\label{tab:counterfactual-instability}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{llcc}
\toprule
Test & Metric & Mean & 95\% CI \\
\midrule
"""
    tex += f"\\multirow{{5}}{{*}}{{Positive (Rashomon)}} & Mean AUC & {aucs.mean():.4f} & [{aucs.min():.4f}, {aucs.max():.4f}] \\\\\n"
    tex += f"& CF found rate & {cf_found_rate:.3f} & --- \\\\\n"
    tex += f"& Direction flip rate & {overall_flip:.3f} & [{lo_f:.3f}, {hi_f:.3f}] \\\\\n"
    tex += f"& Cross-model validity & {mean_v_pct:.1f}\\% & [{lo_v_pct:.1f}\\%, {hi_v_pct:.1f}\\%] \\\\\n"
    tex += f"& Distance CV & {mean_cv:.3f} & [{lo_cv:.3f}, {hi_cv:.3f}] \\\\\n"
    tex += r"\midrule" + "\n"
    tex += (f"Neg.\\ control (class\\_sep=0.5) & CF found rate & "
            f"{nc_results['cf_found_rate']:.3f} & --- \\\\\n")
    tex += (f"& Direction flip rate & "
            f"{nc_results['direction_flip_rate']:.3f} & "
            f"[{nc_results['direction_flip_rate_ci_lo']:.3f}, {nc_results['direction_flip_rate_ci_hi']:.3f}] \\\\\n")
    tex += r"\midrule" + "\n"
    tex += (f"Resolution (consensus CF) & Consistency with consensus & "
            f"{res_results['mean_consistency_with_consensus']:.3f} & "
            f"[{res_results['consistency_ci_lo']:.3f}, {res_results['consistency_ci_hi']:.3f}] \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    tex_path.write_text(tex)
    print(f"Saved table: {tex_path}")

    # ─── Save JSON results ────────────────────────────────────────────────────
    results = {
        'n_models': n_models,
        'n_queries': n_queries,
        'auc_mean': float(aucs.mean()),
        'auc_min': float(aucs.min()),
        'auc_max': float(aucs.max()),
        'auc_spread': float(aucs.max() - aucs.min()),
        'cf_found_rate': float(cf_found_rate),
        'direction_flip_rate': {
            'mean': float(overall_flip),
            'ci_lo': float(lo_f),
            'ci_hi': float(hi_f),
        },
        'cross_model_validity': {
            'mean': float(mean_validity),
            'ci_lo': float(lo_v),
            'ci_hi': float(hi_v),
        },
        'distance_cv': {
            'mean': float(mean_cv),
            'ci_lo': float(lo_cv),
            'ci_hi': float(hi_cv),
        },
        'top10_flip_rates': {
            feature_names[i]: float(flip_rates_per_feature[i])
            for i in top10_idx
        },
        'negative_control': nc_results,
        'resolution_test': res_results,
    }
    save_results(results, 'counterfactual_instability')

    # ─── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("POSITIVE TEST (German Credit, Rashomon set):")
    print(f"  Models: {n_models} XGBoost (AUC {aucs.min():.4f}–{aucs.max():.4f})")
    print(f"  Query points: {n_queries} (bad credit by majority)")
    print(f"  CF found rate: {cf_found_rate:.1%}")
    print(f"  Direction flip rate: {overall_flip:.3f} [{lo_f:.3f}, {hi_f:.3f}]")
    print(f"  Cross-model validity: {mean_validity:.1%} [{lo_v:.1%}, {hi_v:.1%}]")
    print(f"  Distance CV: {mean_cv:.3f} [{lo_cv:.3f}, {hi_cv:.3f}]")
    print()
    print("NEGATIVE CONTROL (class_sep=0.5, subsample=1.0):")
    print(f"  CF found rate: {nc_results['cf_found_rate']:.3f}  (expected >0%)")
    print(f"  Direction flip rate: {nc_results['direction_flip_rate']:.3f} "
          f"[{nc_results['direction_flip_rate_ci_lo']:.3f}, {nc_results['direction_flip_rate_ci_hi']:.3f}]  (expected <0.05)")
    print()
    print("RESOLUTION TEST (consensus CF):")
    print(f"  Consistency with consensus: {res_results['mean_consistency_with_consensus']:.3f} "
          f"[{res_results['consistency_ci_lo']:.3f}, {res_results['consistency_ci_hi']:.3f}]")
    print(f"  Expected: >0.80")
    print()
    print("Top 5 most unstable features:")
    for rank, fi in enumerate(top10_idx[:5]):
        print(f"  {rank+1}. {feature_names[fi]}: flip rate = {flip_rates_per_feature[fi]:.3f}")
    print("=" * 60)
    return results


if __name__ == '__main__':
    run_experiment()
