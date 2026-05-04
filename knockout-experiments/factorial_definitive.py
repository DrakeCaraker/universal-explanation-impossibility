"""
DEFINITIVE FACTORIAL VALIDATION
Tests ALL 7 tools × 6 model/method configs × 15 random datasets = 630 tests.
Single unified experiment proving theory-agnosticity.

Tools tested:
1. SAGE directional (within > between gap)
2. DASH variance reduction (ensemble flip < individual flip)
3. DASH accuracy preservation (ensemble acc >= mean - 1std)
4. Gaussian Phi(-SNR) per-pair R²
5. Coverage conflict (fraction SNR<0.5 vs mean flip)
6. Bimodal gap (within-between separation > 0.10)
7. Capacity exceedance (mean flip > 0.05)

Models × Methods:
- XGBoost × TreeSHAP
- XGBoost × gain
- Random Forest × TreeSHAP
- Random Forest × gain
- Logistic Regression × |coef|
- Neural Network × permutation

Statistics:
- Per-tool: success rate, Wilson CI
- Chi-square: model effect (4 models)
- Chi-square: SHAP vs non-SHAP
- Cochran's Q: all configs equivalent? (repeated measures)
- Negative control: random grouping
"""
import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from pmlb import fetch_data, classification_dataset_names
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy import stats
import json, time
import warnings
warnings.filterwarnings('ignore')


def get_treeshap(clf, X_sample):
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        return np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    elif sv.ndim == 3:
        return np.abs(sv).mean(axis=(0, 2))
    else:
        return np.abs(sv).mean(axis=0)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 1)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def compute_all_tools(imp_a, imp_b, P, acc_individual, acc_dash):
    """Given importance arrays from split A and B, compute all 7 tool outcomes."""
    M_a, M_b = imp_a.shape[0], imp_b.shape[0]
    results = {}

    # --- Flip matrix on A (for SAGE group discovery) ---
    n_mp_a = M_a * (M_a - 1) / 2
    flip_a = np.zeros((P, P))
    for i in range(M_a):
        for j in range(i+1, M_a):
            for a in range(P):
                for b in range(a+1, P):
                    if (imp_a[i,a] > imp_a[i,b]) != (imp_a[j,a] > imp_a[j,b]):
                        flip_a[a,b] += 1/n_mp_a
                        flip_a[b,a] += 1/n_mp_a

    # --- Flip matrix on B (for testing) ---
    n_mp_b = M_b * (M_b - 1) / 2
    flip_b = np.zeros((P, P))
    for i in range(M_b):
        for j in range(i+1, M_b):
            for a in range(P):
                for b in range(a+1, P):
                    if (imp_b[i,a] > imp_b[i,b]) != (imp_b[j,a] > imp_b[j,b]):
                        flip_b[a,b] += 1/n_mp_b
                        flip_b[b,a] += 1/n_mp_b

    mean_flip = np.mean(flip_b[np.triu_indices(P, k=1)])

    # --- Tool 7: Capacity exceedance ---
    results['capacity_exceedance'] = mean_flip > 0.05

    # --- SAGE clustering on A ---
    dist = 1.0 - flip_a
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    try:
        Z = linkage(squareform(dist), method='ward')
        labels = fcluster(Z, t=0.6, criterion='distance')
    except:
        labels = np.arange(P)
    n_groups = len(np.unique(labels))

    # --- Tool 1: SAGE directional ---
    within = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if labels[a]==labels[b]]
    between = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if labels[a]!=labels[b]]
    if n_groups < P and within and between:
        sage_gap = np.mean(within) - np.mean(between)
        results['sage_positive'] = sage_gap > 0
        results['sage_gap'] = sage_gap
    else:
        results['sage_positive'] = None

    # --- Tool 6: Bimodal gap ---
    if within and between:
        results['bimodal_gap'] = np.mean(within) - np.mean(between) > 0.10
    else:
        results['bimodal_gap'] = None

    # --- Tool 2: DASH variance reduction ---
    # DASH = average importances; check if DASH flip rate < individual
    dash_imp = np.mean(np.vstack([imp_a, imp_b]), axis=0)
    # Bootstrap DASH flip rate
    all_imp = np.vstack([imp_a, imp_b])
    M_all = all_imp.shape[0]
    n_dash_flips = 0
    n_dash_total = 0
    np.random.seed(42)
    for _ in range(30):
        idx1 = np.random.choice(M_all, M_all, replace=True)
        idx2 = np.random.choice(M_all, M_all, replace=True)
        d1 = np.mean(all_imp[idx1], axis=0)
        d2 = np.mean(all_imp[idx2], axis=0)
        for a in range(min(P, 20)):
            for b in range(a+1, min(P, 20)):
                if (d1[a] > d1[b]) != (d2[a] > d2[b]):
                    n_dash_flips += 1
                n_dash_total += 1
    dash_flip = n_dash_flips / n_dash_total if n_dash_total > 0 else 0
    results['dash_reduces_variance'] = dash_flip < mean_flip

    # --- Tool 3: DASH accuracy ---
    results['dash_preserves_accuracy'] = acc_dash >= np.mean(acc_individual) - np.std(acc_individual)

    # --- Tool 4: Gaussian Phi(-SNR) ---
    # Use first half of B as calibration, second half as validation
    M_cal = M_b // 2
    imp_cal = imp_b[:M_cal]
    imp_val = imp_b[M_cal:]
    predicted, observed = [], []
    pairs = [(a,b) for a in range(min(P,30)) for b in range(a+1, min(P,30))]
    for a, b in pairs[:150]:
        diffs = imp_cal[:, a] - imp_cal[:, b]
        delta = np.mean(diffs)
        sigma = np.std(diffs, ddof=1)
        if sigma > 1e-10:
            pred = 2 * stats.norm.cdf(delta/sigma) * stats.norm.cdf(-delta/sigma)
            nf = sum(1 for i in range(M_cal, M_b) for j in range(i+1, M_b)
                     if (imp_b[i,a]>imp_b[i,b]) != (imp_b[j,a]>imp_b[j,b]))
            n_val_pairs = (M_b - M_cal) * (M_b - M_cal - 1) / 2
            obs = nf / n_val_pairs if n_val_pairs > 0 else 0
            predicted.append(pred)
            observed.append(obs)
    if len(predicted) >= 10:
        ss_res = np.sum((np.array(observed) - np.array(predicted))**2)
        ss_tot = np.sum((np.array(observed) - np.mean(observed))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -999
        results['gaussian_r2'] = r2
        results['gaussian_pass'] = r2 > 0.50  # relaxed threshold for M=20
    else:
        results['gaussian_pass'] = None

    # --- Tool 5: Coverage conflict ---
    n_low_snr = 0
    n_pairs = 0
    all_imp_full = np.vstack([imp_a, imp_b])
    for a, b in pairs[:150]:
        diffs = all_imp_full[:, a] - all_imp_full[:, b]
        delta = np.mean(diffs)
        sigma = np.std(diffs, ddof=1)
        if sigma > 1e-10:
            snr = abs(delta) / sigma
            if snr < 0.5:
                n_low_snr += 1
        n_pairs += 1
    results['coverage_conflict'] = n_low_snr / n_pairs if n_pairs > 0 else 0
    results['mean_flip'] = mean_flip

    # Negative control: random grouping
    rand_labels = np.random.randint(0, max(2, n_groups), size=P)
    w_r = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if rand_labels[a]==rand_labels[b]]
    b_r = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if rand_labels[a]!=rand_labels[b]]
    results['random_gap'] = (np.mean(w_r)-np.mean(b_r)) if w_r and b_r else 0

    return results


# ============================================================================
# PHASE 1: DATASET SELECTION (random, P>=11)
# ============================================================================
print('='*70)
print('DEFINITIVE FACTORIAL: 7 tools × 6 configs × 15 datasets')
print('='*70)

np.random.seed(999)  # never used before
candidates = []
for name in classification_dataset_names:
    try:
        X, y = fetch_data(name, return_X_y=True)
        N, P = X.shape
        if 11 <= P <= 50 and 300 <= N <= 3000:
            le = LabelEncoder()
            yt = le.fit_transform(y)
            if len(np.unique(yt)) >= 2 and min(np.bincount(yt)) >= 5:
                candidates.append((name, N, P))
    except:
        continue

selected = [candidates[i] for i in np.random.choice(len(candidates), 15, replace=False)]
print(f'Selected {len(selected)} datasets (seed=999):')
for name, N, P in selected:
    print(f'  {name}: N={N}, P={P}')

# ============================================================================
# PHASE 2: FACTORIAL EXECUTION
# ============================================================================
print()
print('='*70)
print('PHASE 2: RUNNING FACTORIAL')
print('='*70)

configs = [
    ('XGBoost', 'TreeSHAP'),
    ('XGBoost', 'gain'),
    ('RF', 'TreeSHAP'),
    ('RF', 'gain'),
    ('LR', 'coef'),
    ('NN', 'permutation'),
]

M = 20
n_shap = 50
t0 = time.time()
all_results = []

for ds_idx, (ds_name, N_ds, P_ds) in enumerate(selected):
    try:
        X, y = fetch_data(ds_name, return_X_y=True)
        le = LabelEncoder(); y = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.random.seed(42)
        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X, y, X_scaled = X[idx], y[idx], X_scaled[idx]

        N, P = X.shape
        half = N // 2
        X_a, y_a, Xs_a = X[:half], y[:half], X_scaled[:half]
        X_b, y_b, Xs_b = X[half:], y[half:], X_scaled[half:]

        for model_name, method_name in configs:
            try:
                imp_a_list, imp_b_list = [], []
                acc_ind = []
                preds_all = []

                for seed in range(M):
                    if model_name == 'XGBoost':
                        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, subsample=0.8,
                                                 colsample_bytree=0.5, random_state=seed,
                                                 eval_metric='mlogloss', verbosity=0)
                        clf.fit(X_a, y_a)
                        if method_name == 'TreeSHAP':
                            s = X_a[np.random.choice(len(X_a), min(n_shap, len(X_a)), replace=False)]
                            imp_a_list.append(get_treeshap(clf, s))
                        else:
                            imp_a_list.append(clf.feature_importances_)
                        acc_ind.append(accuracy_score(y_b, clf.predict(X_b)))
                        preds_all.append(clf.predict_proba(X_b))

                        clf2 = xgb.XGBClassifier(n_estimators=100, max_depth=4, subsample=0.8,
                                                  colsample_bytree=0.5, random_state=seed+100,
                                                  eval_metric='mlogloss', verbosity=0)
                        clf2.fit(X_b, y_b)
                        if method_name == 'TreeSHAP':
                            s = X_b[np.random.choice(len(X_b), min(n_shap, len(X_b)), replace=False)]
                            imp_b_list.append(get_treeshap(clf2, s))
                        else:
                            imp_b_list.append(clf2.feature_importances_)

                    elif model_name == 'RF':
                        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                                     max_features='sqrt', random_state=seed)
                        clf.fit(X_a, y_a)
                        if method_name == 'TreeSHAP':
                            s = X_a[np.random.choice(len(X_a), min(n_shap, len(X_a)), replace=False)]
                            imp_a_list.append(get_treeshap(clf, s))
                        else:
                            imp_a_list.append(clf.feature_importances_)
                        acc_ind.append(accuracy_score(y_b, clf.predict(X_b)))
                        preds_all.append(clf.predict_proba(X_b))

                        clf2 = RandomForestClassifier(n_estimators=100, max_depth=8,
                                                      max_features='sqrt', random_state=seed+100)
                        clf2.fit(X_b, y_b)
                        if method_name == 'TreeSHAP':
                            s = X_b[np.random.choice(len(X_b), min(n_shap, len(X_b)), replace=False)]
                            imp_b_list.append(get_treeshap(clf2, s))
                        else:
                            imp_b_list.append(clf2.feature_importances_)

                    elif model_name == 'LR':
                        np.random.seed(seed)
                        boot = np.random.choice(half, half, replace=True)
                        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                        clf.fit(Xs_a[boot], y_a[boot])
                        coef = np.mean(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 and clf.coef_.shape[0] > 1 else np.abs(clf.coef_.ravel())
                        imp_a_list.append(coef)
                        acc_ind.append(accuracy_score(y_b, clf.predict(Xs_b)))
                        preds_all.append(clf.predict_proba(Xs_b))

                        np.random.seed(seed+100)
                        boot2 = np.random.choice(len(Xs_b), len(Xs_b), replace=True)
                        clf2 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed+100)
                        clf2.fit(Xs_b[boot2], y_b[boot2])
                        coef2 = np.mean(np.abs(clf2.coef_), axis=0) if clf2.coef_.ndim > 1 and clf2.coef_.shape[0] > 1 else np.abs(clf2.coef_.ravel())
                        imp_b_list.append(coef2)

                    elif model_name == 'NN':
                        clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300,
                                            random_state=seed, early_stopping=True)
                        clf.fit(Xs_a, y_a)
                        perm = permutation_importance(clf, Xs_a, y_a, n_repeats=3, random_state=seed)
                        imp_a_list.append(perm.importances_mean)
                        acc_ind.append(accuracy_score(y_b, clf.predict(Xs_b)))
                        preds_all.append(clf.predict_proba(Xs_b))

                        clf2 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300,
                                             random_state=seed+100, early_stopping=True)
                        clf2.fit(Xs_b, y_b)
                        perm2 = permutation_importance(clf2, Xs_b, y_b, n_repeats=3, random_state=seed+100)
                        imp_b_list.append(perm2.importances_mean)

                imp_a = np.array(imp_a_list)
                imp_b = np.array(imp_b_list)

                # DASH accuracy
                avg_proba = np.mean(preds_all, axis=0)
                dash_pred = np.argmax(avg_proba, axis=1)
                acc_dash = accuracy_score(y_b, dash_pred)

                # Compute all 7 tools
                tool_results = compute_all_tools(imp_a, imp_b, P, acc_ind, acc_dash)
                tool_results['dataset'] = ds_name
                tool_results['model'] = model_name
                tool_results['method'] = method_name
                tool_results['P'] = P
                all_results.append(tool_results)

            except Exception as e:
                all_results.append({'dataset': ds_name, 'model': model_name,
                                    'method': method_name, 'error': str(e)[:60]})

        elapsed = time.time() - t0
        valid_sage = [r for r in all_results if r.get('sage_positive') is not None]
        sage_ok = sum(1 for r in valid_sage if r['sage_positive'])
        print(f'  [{ds_idx+1}/15] {ds_name} done | {elapsed:.0f}s | SAGE: {sage_ok}/{len(valid_sage)}')

    except Exception as e:
        print(f'  {ds_name}: DATASET ERROR - {str(e)[:60]}')

# ============================================================================
# PHASE 3: STATISTICS
# ============================================================================
print()
print('='*70)
print('PHASE 3: UNIFIED STATISTICS')
print('='*70)

valid = [r for r in all_results if 'error' not in r]
errors = [r for r in all_results if 'error' in r]
print(f'  Valid tests: {len(valid)}, Errors: {len(errors)}')

# Per-tool success rates
tools = [
    ('SAGE directional', 'sage_positive'),
    ('DASH variance reduction', 'dash_reduces_variance'),
    ('DASH accuracy preservation', 'dash_preserves_accuracy'),
    ('Gaussian R²>0.50', 'gaussian_pass'),
    ('Bimodal gap > 0.10', 'bimodal_gap'),
    ('Capacity exceedance', 'capacity_exceedance'),
]

print()
print(f'  {"TOOL":<30} {"RATE":>8} {"CI":>15}')
print(f'  {"-"*55}')
tool_stats = {}
for tool_name, key in tools:
    applicable = [r for r in valid if r.get(key) is not None]
    positive = sum(1 for r in applicable if r[key])
    n = len(applicable)
    ci = wilson_ci(positive, n)
    rate = positive/n*100 if n > 0 else 0
    print(f'  {tool_name:<30} {positive}/{n:>3} ({rate:.0f}%) [{ci[0]*100:.0f}-{ci[1]*100:.0f}%]')
    tool_stats[tool_name] = {'k': positive, 'n': n, 'rate': round(rate, 1), 'ci': [round(ci[0]*100,1), round(ci[1]*100,1)]}

# Coverage conflict (correlation, not binary)
cc_data = [(r['coverage_conflict'], r['mean_flip']) for r in valid if r.get('mean_flip', 0) > 0]
if len(cc_data) >= 5:
    cc_v, mf_v = zip(*cc_data)
    rho_cc, p_cc = stats.spearmanr(cc_v, mf_v)
    print(f'  {"Coverage conflict rho":<30} {rho_cc:.3f}    p={p_cc:.2e}')
    tool_stats['coverage_conflict'] = {'rho': round(rho_cc, 3), 'p': float(p_cc)}

# --- Model effect (chi-square) ---
print()
print('  MODEL EFFECT (Chi-square on SAGE):')
sage_valid = [r for r in valid if r.get('sage_positive') is not None]
from scipy.stats import chi2_contingency
models = ['XGBoost', 'RF', 'LR', 'NN']
table = []
for model in models:
    sub = [r for r in sage_valid if r['model'] == model]
    pos = sum(1 for r in sub if r['sage_positive'])
    table.append([pos, len(sub) - pos])
    print(f'    {model:10s}: {pos}/{len(sub)}')
table_arr = np.array(table)
if table_arr.min() >= 0 and table_arr.sum() > 0:
    chi2, p_model, dof, _ = chi2_contingency(table_arr)
    print(f'    Chi-square: chi2={chi2:.2f}, p={p_model:.4f}, dof={dof}')
    cramers_v = np.sqrt(chi2 / (table_arr.sum() * (min(table_arr.shape) - 1)))
    print(f'    Cramers V = {cramers_v:.3f} (effect size)')

# --- SHAP vs non-SHAP ---
print()
print('  SHAP vs NON-SHAP:')
shap_tests = [r for r in sage_valid if r['method'] == 'TreeSHAP']
non_shap = [r for r in sage_valid if r['method'] != 'TreeSHAP']
s_pos = sum(1 for r in shap_tests if r['sage_positive'])
ns_pos = sum(1 for r in non_shap if r['sage_positive'])
print(f'    SHAP: {s_pos}/{len(shap_tests)}, Non-SHAP: {ns_pos}/{len(non_shap)}')
table2 = np.array([[s_pos, len(shap_tests)-s_pos], [ns_pos, len(non_shap)-ns_pos]])
if table2.min() >= 0 and table2.sum() > 0:
    chi2_2, p_shap, _, _ = chi2_contingency(table2)
    print(f'    Chi-square: chi2={chi2_2:.2f}, p={p_shap:.4f}')

# --- Cochran's Q (repeated measures) ---
print()
print('  COCHRAN Q (are all configs equivalent?):')
# Build binary matrix: datasets × configs
datasets_tested = sorted(set(r['dataset'] for r in sage_valid))
config_names = [f'{m}-{mt}' for m, mt in configs]
Q_matrix = np.full((len(datasets_tested), len(config_names)), np.nan)
for r in sage_valid:
    i = datasets_tested.index(r['dataset'])
    j = config_names.index(f'{r["model"]}-{r["method"]}')
    Q_matrix[i, j] = 1 if r['sage_positive'] else 0

# Remove rows/cols with NaN
mask = ~np.isnan(Q_matrix).any(axis=1)
Q_clean = Q_matrix[mask].astype(int)
if Q_clean.shape[0] >= 3:
    # Cochran's Q test
    k = Q_clean.shape[1]
    n_subj = Q_clean.shape[0]
    T = Q_clean.sum(axis=1)
    C = Q_clean.sum(axis=0)
    N_total = Q_clean.sum()
    Q_stat = (k-1) * (k * np.sum(C**2) - N_total**2) / (k * N_total - np.sum(T**2))
    p_cochran = 1 - stats.chi2.cdf(Q_stat, k-1)
    print(f'    Q={Q_stat:.2f}, p={p_cochran:.4f}, k={k} configs, n={n_subj} datasets')
    print(f'    {"EQUIVALENT (theory-agnostic)" if p_cochran > 0.05 else "DIFFERENT (model/method matters)"}')

# --- Negative control ---
print()
sage_gaps = [r.get('sage_gap', 0) for r in valid if r.get('sage_gap') is not None]
rand_gaps = [r.get('random_gap', 0) for r in valid if r.get('random_gap') is not None]
if sage_gaps and rand_gaps and len(sage_gaps) == len(rand_gaps):
    stat_w, p_nc = stats.wilcoxon(sage_gaps, rand_gaps, alternative='greater')
    print(f'  NEGATIVE CONTROL (paired Wilcoxon, SAGE > random):')
    print(f'    SAGE mean gap: {np.mean(sage_gaps):.4f}')
    print(f'    Random mean gap: {np.mean(rand_gaps):.4f}')
    print(f'    p = {p_nc:.2e}')

# ============================================================================
# SAVE
# ============================================================================
elapsed = time.time() - t0
output = {
    'description': 'Definitive factorial: 7 tools × 6 configs × 15 random datasets',
    'datasets': [name for name, _, _ in selected],
    'configs': [f'{m}-{mt}' for m, mt in configs],
    'n_valid': len(valid),
    'n_errors': len(errors),
    'tool_stats': tool_stats,
    'elapsed_seconds': round(elapsed, 1),
    'per_test': all_results,
}
with open('knockout-experiments/results_factorial_definitive.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f'\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)')
print('  Saved: results_factorial_definitive.json')
