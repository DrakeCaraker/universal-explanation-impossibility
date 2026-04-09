"""Fixed Test Set Experiment: Isolate model instability from evaluation-set variation."""
import json, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

MASTER_SEED = 42
N_MODELS = 50

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=MASTER_SEED
)

shap_arrays = []
for seed in range(N_MODELS):
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=seed, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list): sv = sv[1]
    shap_arrays.append(np.mean(np.abs(sv), axis=0))

shap_matrix = np.array(shap_arrays)
P = shap_matrix.shape[1]

pairs = []
for j in range(P):
    for k in range(j + 1, P):
        n_flips = n_compare = 0
        for a in range(N_MODELS):
            for b in range(a + 1, N_MODELS):
                sa = np.sign(shap_matrix[a, j] - shap_matrix[a, k])
                sb = np.sign(shap_matrix[b, j] - shap_matrix[b, k])
                if sa != 0 and sb != 0:
                    n_compare += 1
                    if sa != sb: n_flips += 1
        flip_rate = n_flips / n_compare if n_compare > 0 else 0.0
        if flip_rate > 0.01:
            pairs.append({
                'j': int(j), 'k': int(k),
                'feature_j': data.feature_names[j],
                'feature_k': data.feature_names[k],
                'flip_rate': round(flip_rate, 4),
                'n_comparisons': n_compare
            })

unstable = [p for p in pairs if p['flip_rate'] > 0.10]
max_flip = max(p['flip_rate'] for p in pairs) if pairs else 0.0

results = {
    'experiment': 'fixed_testset_isolation',
    'n_models': N_MODELS,
    'dataset': 'breast_cancer',
    'test_set': 'FIXED (seed=42, 20% holdout)',
    'training_variation': 'subsample=0.8, varying random_state only',
    'n_features': int(P),
    'n_total_pairs': int(P * (P - 1) // 2),
    'n_unstable_pairs': len(unstable),
    'pct_unstable': round(100 * len(unstable) / (P * (P - 1) // 2), 1),
    'max_flip_rate': round(max_flip, 4),
    'comparison_varying_testset': {'n_unstable': 162, 'source': 'results_cross_implementation.json'},
    'top_5_unstable': sorted(unstable, key=lambda x: -x['flip_rate'])[:5],
}

with open('paper/results_fixed_testset.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: results[k] for k in ['n_unstable_pairs', 'pct_unstable', 'max_flip_rate']}, indent=2))
