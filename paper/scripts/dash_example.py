"""
DASH: Minimal reference implementation.

Demonstrates the paper's practical recommendation (§5):
  Train M models → compute SHAP → average → stable attributions.

Usage:
  python dash_example.py

Requires: pip install xgboost shap scikit-learn numpy

For the full DASH pipeline (population generation, epsilon-filtering,
deduplication, consensus), see the companion dash-shap repository.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap


def dash_consensus(X, y, M=25, n_estimators=100, max_depth=6,
                   learning_rate=0.1, n_background=200):
    """Train M models and return consensus (averaged) SHAP values.

    Parameters
    ----------
    X, y : array-like
        Training data.
    M : int
        Ensemble size. M=25 gives <1% flip rate on synthetic data.
    n_background : int
        Number of background samples for TreeSHAP.

    Returns
    -------
    consensus_shap : ndarray, shape (n_background, n_features)
        Averaged absolute SHAP values across M models.
    all_shap : ndarray, shape (M, n_features)
        Per-model mean |SHAP| for each feature.
    """
    all_shap = []
    for seed in range(M):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed + 1000,
            verbosity=0, eval_metric='logloss',
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test[:n_background])
        all_shap.append(np.mean(np.abs(sv), axis=0))

    all_shap = np.array(all_shap)  # (M, P)
    consensus = np.mean(all_shap, axis=0)  # (P,)
    return consensus, all_shap


def diagnose_instability(all_shap, feature_names=None, threshold=0.05):
    """Check which feature pairs flip rankings across models.

    Parameters
    ----------
    all_shap : ndarray, shape (M, P)
        Per-model mean |SHAP| for each feature.
    threshold : float
        Minimum flip rate to report.

    Returns
    -------
    unstable_pairs : list of (feature_i, feature_j, flip_rate, correlation)
    """
    M, P = all_shap.shape
    unstable = []
    for j in range(P):
        for k in range(j + 1, P):
            jw = np.sum(all_shap[:, j] > all_shap[:, k])
            kw = np.sum(all_shap[:, k] > all_shap[:, j])
            total = jw + kw
            if total > 0:
                flip_rate = min(jw, kw) / total
                if flip_rate >= threshold:
                    j_name = feature_names[j] if feature_names is not None else f"f{j}"
                    k_name = feature_names[k] if feature_names is not None else f"f{k}"
                    unstable.append((j_name, k_name, flip_rate))
    return sorted(unstable, key=lambda x: -x[2])


if __name__ == "__main__":
    # Load public dataset
    data = load_breast_cancer()
    X, y, names = data.data, data.target, list(data.feature_names)

    print("=" * 60)
    print("DASH Example: Breast Cancer (Wisconsin)")
    print("=" * 60)

    # Step 1: Single model — check instability
    print("\n1. Training 25 models to diagnose instability...")
    _, single_shap = dash_consensus(X, y, M=25)

    unstable = diagnose_instability(single_shap, names, threshold=0.1)
    print(f"   Found {len(unstable)} unstable pairs (flip rate >= 10%):")
    for j_name, k_name, flip in unstable[:5]:
        print(f"   {j_name:25s} <-> {k_name:25s}: {flip:.1%} flip rate")

    # Step 2: DASH consensus — stable attributions
    print("\n2. DASH consensus (M=25):")
    consensus, all_shap = dash_consensus(X, y, M=25)
    ranking = np.argsort(-consensus)
    print("   Top 5 features (stable ranking):")
    for i, idx in enumerate(ranking[:5]):
        print(f"   {i+1}. {names[idx]:25s} (mean |SHAP| = {consensus[idx]:.4f})")

    # Step 3: Compare stability
    print("\n3. Stability comparison:")
    single_ranking = np.argsort(-single_shap[0])  # first model's ranking
    dash_ranking = np.argsort(-consensus)
    agreement = np.sum(single_ranking[:5] == dash_ranking[:5])
    print(f"   Single model top-5 agrees with DASH top-5 on {agreement}/5 features")
    print(f"   Recommendation: {'Rankings are stable' if len(unstable) == 0 else 'Use DASH (M>=25) for stable rankings'}")
