#!/usr/bin/env python3
"""
Quantitative Bilemma Test: Does unfaith₁ + unfaith₂ ≥ Δ - δ Hold?

The quantitative bilemma (proved in Lean) predicts: for any stable method E,
when two models disagree on a feature's importance by Δ, the method must be
unfaithful to at least one by ≥ (Δ-δ)/2.

This is a prediction the null model CANNOT make. The null model predicts flip
rates from correlation. The quantitative bilemma predicts faithfulness LOSS
from the incompatibility gap Δ — a different quantity entirely.

Test:
1. Train 50 models, compute SHAP
2. DASH = ensemble average (the stable method)
3. For each feature j and each model pair (m₁, m₂):
   - Δ_j = |SHAP_m₁(j) - SHAP_m₂(j)|  (incompatibility gap)
   - unfaith_m₁ = |DASH(j) - SHAP_m₁(j)|  (faithfulness loss at m₁)
   - unfaith_m₂ = |DASH(j) - SHAP_m₂(j)|  (faithfulness loss at m₂)
   - bound_holds = (unfaith_m₁ + unfaith_m₂ ≥ Δ - δ)
   - tightness = (unfaith_m₁ + unfaith_m₂) / max(Δ - δ, 1e-10)
4. Report: bound satisfaction rate (should be ~100% — it's a theorem)
5. Report: tightness distribution (closer to 1 = more informative)
6. Report: tightness vs within/between group (within should be tighter)
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap

N_MODELS = 50
CLUSTER_THRESHOLD = 0.8


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def generate_synthetic(n, p, g, k, rho, seed=0):
    rng = np.random.RandomState(seed)
    Sigma = np.eye(p)
    for gi in range(g):
        for i in range(k):
            for j in range(k):
                if i != j:
                    Sigma[gi*k+i, gi*k+j] = rho
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, p) @ L.T
    effects = np.array([1.0, -0.5, 0.3])[:g]
    y_lin = sum(effects[gi] * X[:, gi*k:(gi+1)*k].mean(axis=1) for gi in range(g))
    y = (y_lin + rng.randn(n)*0.5 > 0).astype(int)
    return X, y


def compute_groups(X_train, threshold=CLUSTER_THRESHOLD):
    """Cluster features by |correlation| > threshold."""
    corr = np.abs(np.corrcoef(X_train.T))
    P = corr.shape[0]
    visited = set()
    clusters = []
    for i in range(P):
        if i in visited:
            continue
        cluster = {i}
        queue = [i]
        while queue:
            curr = queue.pop()
            for j in range(P):
                if j not in visited and j not in cluster and corr[curr, j] > threshold:
                    cluster.add(j)
                    queue.append(j)
        visited.update(cluster)
        clusters.append(sorted(cluster))

    feature_to_group = {}
    for ci, cluster in enumerate(clusters):
        for fi in cluster:
            feature_to_group[fi] = ci
    return clusters, feature_to_group


def run_quantitative_bilemma(X, y, dataset_name, task='classification'):
    """Test the quantitative bilemma on one dataset."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
        stratify=y if task == 'classification' else None
    )

    clusters, f2g = compute_groups(X_train)
    P = X.shape[1]
    print(f"  Groups: {len(clusters)} (sizes: {[len(c) for c in clusters]})")

    # Train models
    all_shap = []
    for i in range(N_MODELS):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, use_label_encoder=False,
                eval_metric='logloss', verbosity=0)
        else:
            model = xgb.XGBRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbosity=0)
        model.fit(X_train[idx], y_train[idx])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:200])
        if isinstance(sv, list):
            sv = sv[1]
        all_shap.append(sv)

    all_shap = np.array(all_shap)  # (N_MODELS, n_test, P)
    n_test = all_shap.shape[1]

    # DASH = ensemble average (the stable method)
    dash_shap = np.mean(all_shap, axis=0)  # (n_test, P)

    # Estimate δ (DASH stability tolerance) via bootstrap
    n_bootstrap = 20
    rng = np.random.RandomState(0)
    dash_variations = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(N_MODELS, N_MODELS, replace=True)
        dash_boot = np.mean(all_shap[boot_idx], axis=0)
        dash_variations.append(np.abs(dash_boot - dash_shap).mean())
    delta = float(np.mean(dash_variations))
    print(f"  DASH stability δ = {delta:.4f}")

    # Per-feature, per-model-pair analysis
    # Sample model pairs for speed
    rng = np.random.RandomState(42)
    n_pairs = min(500, N_MODELS * (N_MODELS - 1) // 2)
    model_pairs = []
    seen = set()
    while len(model_pairs) < n_pairs:
        m1, m2 = rng.choice(N_MODELS, 2, replace=False)
        key = (min(m1, m2), max(m1, m2))
        if key not in seen:
            seen.add(key)
            model_pairs.append(key)

    within_tightness = []
    between_tightness = []
    bound_violations = 0
    total_checks = 0
    all_deltas_within = []
    all_deltas_between = []
    all_unfaith_sums_within = []
    all_unfaith_sums_between = []

    for j in range(P):
        for m1, m2 in model_pairs:
            # Per-observation average SHAP for this feature
            shap_m1_j = np.mean(np.abs(all_shap[m1, :, j]))
            shap_m2_j = np.mean(np.abs(all_shap[m2, :, j]))
            dash_j = np.mean(np.abs(dash_shap[:, j]))

            # Incompatibility gap
            big_delta = abs(shap_m1_j - shap_m2_j)

            # Faithfulness loss
            unfaith_m1 = abs(dash_j - shap_m1_j)
            unfaith_m2 = abs(dash_j - shap_m2_j)
            unfaith_sum = unfaith_m1 + unfaith_m2

            # Bound check: unfaith_sum ≥ Δ - δ
            bound_value = big_delta - delta
            if bound_value > 0:
                total_checks += 1
                holds = unfaith_sum >= bound_value - 1e-10  # numerical tolerance
                if not holds:
                    bound_violations += 1

                tightness = unfaith_sum / bound_value if bound_value > 1e-10 else float('inf')

                # Within vs between group
                if f2g.get(j, -1) == f2g.get(j, -1):  # same feature, always same group
                    # Compare with partner features
                    pass

                # Store for analysis
                gi = f2g.get(j, -1)
                # Check if this feature has within-group partners
                has_partner = any(f2g.get(jj, -2) == gi and jj != j for jj in range(P))

                if has_partner and big_delta > 0.01:
                    within_tightness.append(tightness)
                    all_deltas_within.append(big_delta)
                    all_unfaith_sums_within.append(unfaith_sum)
                elif big_delta > 0.01:
                    between_tightness.append(tightness)
                    all_deltas_between.append(big_delta)
                    all_unfaith_sums_between.append(unfaith_sum)

    # Also compute per-feature-PAIR analysis (within vs between groups)
    within_pair_deltas = []
    within_pair_unfaith = []
    between_pair_deltas = []
    between_pair_unfaith = []

    for fi in range(P):
        for fj in range(fi+1, P):
            for m1, m2 in model_pairs[:50]:  # subsample for speed
                # Ranking-level gap
                rank_m1 = np.mean(all_shap[m1, :, fi]) - np.mean(all_shap[m1, :, fj])
                rank_m2 = np.mean(all_shap[m2, :, fi]) - np.mean(all_shap[m2, :, fj])
                pair_delta = abs(rank_m1 - rank_m2)

                dash_rank = np.mean(dash_shap[:, fi]) - np.mean(dash_shap[:, fj])
                pair_unfaith_m1 = abs(dash_rank - rank_m1)
                pair_unfaith_m2 = abs(dash_rank - rank_m2)

                if pair_delta > 0.001:
                    if f2g.get(fi, -1) == f2g.get(fj, -2):
                        within_pair_deltas.append(pair_delta)
                        within_pair_unfaith.append(pair_unfaith_m1 + pair_unfaith_m2)
                    else:
                        between_pair_deltas.append(pair_delta)
                        between_pair_unfaith.append(pair_unfaith_m1 + pair_unfaith_m2)

    # Correlation between Δ and unfaithfulness
    if within_pair_deltas and between_pair_deltas:
        within_rho, within_p = spearmanr(within_pair_deltas, within_pair_unfaith)
        between_rho, between_p = spearmanr(between_pair_deltas, between_pair_unfaith)
        all_deltas_combined = within_pair_deltas + between_pair_deltas
        all_unfaith_combined = within_pair_unfaith + between_pair_unfaith
        combined_rho, combined_p = spearmanr(all_deltas_combined, all_unfaith_combined)
    else:
        within_rho = between_rho = combined_rho = 0
        within_p = between_p = combined_p = 1

    bound_rate = 1 - (bound_violations / total_checks) if total_checks > 0 else 1.0

    print(f"\n  RESULTS:")
    print(f"    Bound satisfaction: {bound_rate:.1%} ({total_checks - bound_violations}/{total_checks})")
    print(f"    Mean tightness (all): {np.mean(within_tightness + between_tightness):.2f}" if (within_tightness + between_tightness) else "    No valid tightness values")
    print(f"    Δ-unfaithfulness correlation (within pairs): Spearman {within_rho:.3f} (p={within_p:.2e})")
    print(f"    Δ-unfaithfulness correlation (between pairs): Spearman {between_rho:.3f} (p={between_p:.2e})")
    print(f"    Δ-unfaithfulness correlation (combined): Spearman {combined_rho:.3f} (p={combined_p:.2e})")
    print(f"    Within-pair mean Δ: {np.mean(within_pair_deltas):.4f}" if within_pair_deltas else "")
    print(f"    Between-pair mean Δ: {np.mean(between_pair_deltas):.4f}" if between_pair_deltas else "")

    return {
        'dataset': dataset_name,
        'n_features': P,
        'n_groups': len(clusters),
        'n_models': N_MODELS,
        'delta_stability': delta,
        'bound_satisfaction_rate': float(bound_rate),
        'bound_violations': bound_violations,
        'total_checks': total_checks,
        'correlation_within': {'spearman': float(within_rho), 'p': float(within_p),
                               'n': len(within_pair_deltas)},
        'correlation_between': {'spearman': float(between_rho), 'p': float(between_p),
                                'n': len(between_pair_deltas)},
        'correlation_combined': {'spearman': float(combined_rho), 'p': float(combined_p),
                                 'n': len(all_deltas_combined) if within_pair_deltas else 0},
        'mean_delta_within': float(np.mean(within_pair_deltas)) if within_pair_deltas else 0,
        'mean_delta_between': float(np.mean(between_pair_deltas)) if between_pair_deltas else 0,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("QUANTITATIVE BILEMMA TEST")
    print("Does Δ predict faithfulness loss? (null model can't do this)")
    print("=" * 60)

    results = {}

    # Synthetic
    for rho in [0.5, 0.7, 0.9]:
        X, y = generate_synthetic(2000, 12, 3, 4, rho)
        results[f'synthetic_{rho}'] = run_quantitative_bilemma(
            X, y, f'Synthetic ρ={rho}')

    # Real
    bc = load_breast_cancer()
    results['breast_cancer'] = run_quantitative_bilemma(
        bc.data, bc.target, 'Breast Cancer')

    cal = fetch_california_housing()
    results['california'] = run_quantitative_bilemma(
        cal.data, cal.target, 'California Housing', task='regression')

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SUMMARY: Does Δ predict faithfulness loss?")
    print(f"{'='*60}")
    print(f"\n{'Dataset':25s} {'Bound OK':>10s} {'ρ(Δ,unfaith)':>12s} {'p-value':>10s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:25s} {r['bound_satisfaction_rate']:>9.1%} "
              f"{r['correlation_combined']['spearman']:>12.3f} "
              f"{r['correlation_combined']['p']:>10.2e}")

    print(f"\n  The null model Φ(-c√(1-ρ²)) predicts flip rates from correlation.")
    print(f"  The quantitative bilemma predicts faithfulness loss from Δ.")
    print(f"  If ρ(Δ, unfaithfulness) is high, the framework adds predictive")
    print(f"  power that the null model fundamentally cannot provide.")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'quantitative_bilemma_test',
        'description': 'Tests whether incompatibility gap Δ predicts faithfulness loss',
        'null_model_comparison': 'Null model predicts flip rates from ρ; cannot predict faithfulness loss from Δ',
        'results': results,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_quantitative_bilemma_test.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
