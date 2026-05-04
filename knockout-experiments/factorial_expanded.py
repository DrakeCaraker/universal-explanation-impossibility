"""
EXPANDED FACTORIAL — Addresses adversarial peer review:
- 30 datasets (25 clf balanced across P + 5 regression)
- M=30 (not 20)
- P ranges: 11-20, 21-50, >50 all included
- Proper LabelEncoder on all datasets
- Bimodal gap at thresholds {0.00, 0.05, 0.10}
- All XGBoost errors fixed
"""
import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names
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
    if n == 0: return (0, 1)
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    s = z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return (max(0,c-s), min(1,c+s))

def compute_tools(imp_a, imp_b, P, acc_ind, acc_dash, is_clf=True):
    """Compute all tools from importance matrices."""
    M_a, M_b = imp_a.shape[0], imp_b.shape[0]
    n_mp_a = M_a*(M_a-1)/2
    n_mp_b = M_b*(M_b-1)/2

    # Flip matrices
    flip_a = np.zeros((P, P))
    for i in range(M_a):
        for j in range(i+1, M_a):
            for a in range(P):
                for b in range(a+1, P):
                    if (imp_a[i,a]>imp_a[i,b]) != (imp_a[j,a]>imp_a[j,b]):
                        flip_a[a,b] += 1/n_mp_a; flip_a[b,a] += 1/n_mp_a

    flip_b = np.zeros((P, P))
    for i in range(M_b):
        for j in range(i+1, M_b):
            for a in range(P):
                for b in range(a+1, P):
                    if (imp_b[i,a]>imp_b[i,b]) != (imp_b[j,a]>imp_b[j,b]):
                        flip_b[a,b] += 1/n_mp_b; flip_b[b,a] += 1/n_mp_b

    mean_flip = np.mean(flip_b[np.triu_indices(P, k=1)])
    res = {'mean_flip': float(mean_flip), 'capacity_exceedance': mean_flip > 0.05}

    # SAGE
    dist = 1.0 - flip_a; dist = (dist+dist.T)/2
    np.fill_diagonal(dist, 0); dist = np.clip(dist, 0, None)
    try:
        Z = linkage(squareform(dist), method='ward')
        labels = fcluster(Z, t=0.6, criterion='distance')
    except:
        labels = np.arange(P)
    n_groups = len(np.unique(labels))

    within = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if labels[a]==labels[b]]
    between = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if labels[a]!=labels[b]]
    if n_groups < P and within and between:
        gap = np.mean(within) - np.mean(between)
        res['sage_positive'] = bool(gap > 0)
        res['sage_gap'] = float(gap)
        res['bimodal_000'] = bool(gap > 0.00)
        res['bimodal_005'] = bool(gap > 0.05)
        res['bimodal_010'] = bool(gap > 0.10)
    else:
        res['sage_positive'] = None

    # DASH variance reduction
    all_imp = np.vstack([imp_a, imp_b])
    M_all = all_imp.shape[0]
    np.random.seed(42)
    n_df, n_dt = 0, 0
    for _ in range(30):
        i1 = np.random.choice(M_all, M_all, replace=True)
        i2 = np.random.choice(M_all, M_all, replace=True)
        d1, d2 = np.mean(all_imp[i1], axis=0), np.mean(all_imp[i2], axis=0)
        for a in range(min(P,20)):
            for b in range(a+1, min(P,20)):
                if (d1[a]>d1[b]) != (d2[a]>d2[b]): n_df += 1
                n_dt += 1
    dash_flip = n_df/n_dt if n_dt > 0 else 0
    res['dash_reduces_variance'] = bool(dash_flip < mean_flip)

    # DASH accuracy
    mean_ind = np.mean(acc_ind)
    std_ind = np.std(acc_ind)
    res['dash_preserves_accuracy'] = bool(acc_dash >= mean_ind - std_ind)

    # Gaussian (cal=first half of B, val=second half)
    M_cal = M_b // 2
    pairs = [(a,b) for a in range(min(P,25)) for b in range(a+1, min(P,25))][:100]
    pred, obs = [], []
    for a, b in pairs:
        diffs = imp_b[:M_cal, a] - imp_b[:M_cal, b]
        delta, sigma = np.mean(diffs), np.std(diffs, ddof=1)
        if sigma > 1e-10:
            p_f = 2*stats.norm.cdf(delta/sigma)*stats.norm.cdf(-delta/sigma)
            nf = sum(1 for i in range(M_cal, M_b) for j in range(i+1, M_b)
                     if (imp_b[i,a]>imp_b[i,b]) != (imp_b[j,a]>imp_b[j,b]))
            n_vp = (M_b-M_cal)*(M_b-M_cal-1)/2
            pred.append(p_f); obs.append(nf/n_vp if n_vp>0 else 0)
    if len(pred) >= 10:
        ss_r = np.sum((np.array(obs)-np.array(pred))**2)
        ss_t = np.sum((np.array(obs)-np.mean(obs))**2)
        r2 = 1-ss_r/ss_t if ss_t > 0 else -999
        res['gaussian_r2'] = float(r2)
        res['gaussian_pass'] = bool(r2 > 0.50)
    else:
        res['gaussian_pass'] = None

    # Coverage conflict
    n_low, n_tot = 0, 0
    for a, b in pairs:
        diffs = all_imp[:, a] - all_imp[:, b]
        d, s = np.mean(diffs), np.std(diffs, ddof=1)
        if s > 1e-10:
            if abs(d)/s < 0.5: n_low += 1
        n_tot += 1
    res['coverage_conflict'] = float(n_low/n_tot) if n_tot > 0 else 0

    # Random control
    rl = np.random.randint(0, max(2, n_groups), size=P)
    wr = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if rl[a]==rl[b]]
    br = [flip_b[a,b] for a in range(P) for b in range(a+1,P) if rl[a]!=rl[b]]
    res['random_gap'] = float(np.mean(wr)-np.mean(br)) if wr and br else 0.0

    return res

# ============================================================================
print('='*70)
print('EXPANDED FACTORIAL: 30 datasets × 6 configs × M=30')
print('='*70)

# Select 25 clf (balanced) + 5 regression
np.random.seed(777)
clf_all = []
for name in classification_dataset_names:
    try:
        X, y = fetch_data(name, return_X_y=True)
        N, P = X.shape
        if P >= 11 and 200 <= N <= 5000:
            le = LabelEncoder(); yt = le.fit_transform(y)
            if len(np.unique(yt)) >= 2 and min(np.bincount(yt)) >= 3:
                clf_all.append((name, N, P))
    except:
        continue

# Stratified by P
small = [(n,N,P) for n,N,P in clf_all if 11<=P<=20]
medium = [(n,N,P) for n,N,P in clf_all if 21<=P<=50]
large = [(n,N,P) for n,N,P in clf_all if P>50]

n_small = min(8, len(small))
n_med = min(9, len(medium))
n_large = min(8, len(large))

selected_clf = (
    [small[i] for i in np.random.choice(len(small), n_small, replace=False)] +
    [medium[i] for i in np.random.choice(len(medium), n_med, replace=False)] +
    [large[i] for i in np.random.choice(len(large), n_large, replace=False)]
)

# Regression
reg_all = []
for name in regression_dataset_names:
    try:
        X, y = fetch_data(name, return_X_y=True)
        N, P = X.shape
        if P >= 11 and 200 <= N <= 5000:
            reg_all.append((name, N, P))
    except:
        continue
selected_reg = [reg_all[i] for i in np.random.choice(len(reg_all), min(5, len(reg_all)), replace=False)]

all_selected = [(n,N,P,'clf') for n,N,P in selected_clf] + [(n,N,P,'reg') for n,N,P in selected_reg]
print(f'  Classification: {len(selected_clf)} (small:{n_small}, med:{n_med}, large:{n_large})')
print(f'  Regression: {len(selected_reg)}')
print(f'  Total: {len(all_selected)}')

configs = [
    ('XGBoost', 'TreeSHAP'),
    ('XGBoost', 'gain'),
    ('RF', 'TreeSHAP'),
    ('RF', 'gain'),
    ('LR', 'coef'),
    ('NN', 'permutation'),
]

M = 30
n_shap = 50
t0 = time.time()
all_results = []

for ds_idx, (ds_name, N_ds, P_ds, task) in enumerate(all_selected):
    if time.time() - t0 > 5400:
        print(f'  TIME LIMIT at {ds_idx}')
        break
    try:
        X, y = fetch_data(ds_name, return_X_y=True)
        le = LabelEncoder()
        if task == 'clf':
            y = le.fit_transform(y)
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
                imp_a_list, imp_b_list, acc_ind, preds_all = [], [], [], []

                for seed in range(M):
                    if model_name == 'XGBoost':
                        if task == 'clf':
                            c = xgb.XGBClassifier(n_estimators=100, max_depth=4, subsample=0.8,
                                colsample_bytree=0.5, random_state=seed, eval_metric='mlogloss', verbosity=0)
                        else:
                            c = xgb.XGBRegressor(n_estimators=100, max_depth=4, subsample=0.8,
                                colsample_bytree=0.5, random_state=seed, eval_metric='rmse', verbosity=0)
                        c.fit(X_a, y_a)
                        if method_name == 'TreeSHAP':
                            s = X_a[np.random.choice(len(X_a), min(n_shap,len(X_a)), replace=False)]
                            imp_a_list.append(get_treeshap(c, s))
                        else:
                            imp_a_list.append(c.feature_importances_)
                        if task == 'clf':
                            acc_ind.append(accuracy_score(y_b, c.predict(X_b)))
                            preds_all.append(c.predict_proba(X_b))
                        else:
                            acc_ind.append(-mean_squared_error(y_b, c.predict(X_b)))
                            preds_all.append(c.predict(X_b))

                        if task == 'clf':
                            c2 = xgb.XGBClassifier(n_estimators=100, max_depth=4, subsample=0.8,
                                colsample_bytree=0.5, random_state=seed+100, eval_metric='mlogloss', verbosity=0)
                        else:
                            c2 = xgb.XGBRegressor(n_estimators=100, max_depth=4, subsample=0.8,
                                colsample_bytree=0.5, random_state=seed+100, eval_metric='rmse', verbosity=0)
                        c2.fit(X_b, y_b)
                        if method_name == 'TreeSHAP':
                            s = X_b[np.random.choice(len(X_b), min(n_shap,len(X_b)), replace=False)]
                            imp_b_list.append(get_treeshap(c2, s))
                        else:
                            imp_b_list.append(c2.feature_importances_)

                    elif model_name == 'RF':
                        if task == 'clf':
                            c = RandomForestClassifier(n_estimators=100, max_depth=8, max_features='sqrt', random_state=seed)
                        else:
                            from sklearn.ensemble import RandomForestRegressor
                            c = RandomForestRegressor(n_estimators=100, max_depth=8, max_features='sqrt', random_state=seed)
                        c.fit(X_a, y_a)
                        if method_name == 'TreeSHAP':
                            s = X_a[np.random.choice(len(X_a), min(n_shap,len(X_a)), replace=False)]
                            imp_a_list.append(get_treeshap(c, s))
                        else:
                            imp_a_list.append(c.feature_importances_)
                        if task == 'clf':
                            acc_ind.append(accuracy_score(y_b, c.predict(X_b)))
                            preds_all.append(c.predict_proba(X_b))
                        else:
                            acc_ind.append(-mean_squared_error(y_b, c.predict(X_b)))
                            preds_all.append(c.predict(X_b))

                        if task == 'clf':
                            c2 = RandomForestClassifier(n_estimators=100, max_depth=8, max_features='sqrt', random_state=seed+100)
                        else:
                            c2 = RandomForestRegressor(n_estimators=100, max_depth=8, max_features='sqrt', random_state=seed+100)
                        c2.fit(X_b, y_b)
                        if method_name == 'TreeSHAP':
                            s = X_b[np.random.choice(len(X_b), min(n_shap,len(X_b)), replace=False)]
                            imp_b_list.append(get_treeshap(c2, s))
                        else:
                            imp_b_list.append(c2.feature_importances_)

                    elif model_name == 'LR':
                        np.random.seed(seed)
                        boot = np.random.choice(half, half, replace=True)
                        if task == 'clf':
                            c = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                            c.fit(Xs_a[boot], y_a[boot])
                            coef = np.mean(np.abs(c.coef_), axis=0) if c.coef_.shape[0]>1 else np.abs(c.coef_.ravel())
                            acc_ind.append(accuracy_score(y_b, c.predict(Xs_b)))
                            preds_all.append(c.predict_proba(Xs_b))
                        else:
                            c = Ridge(alpha=1.0)
                            c.fit(Xs_a[boot], y_a[boot])
                            coef = np.abs(c.coef_.ravel())
                            acc_ind.append(-mean_squared_error(y_b, c.predict(Xs_b)))
                            preds_all.append(c.predict(Xs_b))
                        imp_a_list.append(coef)

                        np.random.seed(seed+100)
                        boot2 = np.random.choice(len(Xs_b), len(Xs_b), replace=True)
                        if task == 'clf':
                            c2 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed+100)
                            c2.fit(Xs_b[boot2], y_b[boot2])
                            coef2 = np.mean(np.abs(c2.coef_), axis=0) if c2.coef_.shape[0]>1 else np.abs(c2.coef_.ravel())
                        else:
                            c2 = Ridge(alpha=1.0)
                            c2.fit(Xs_b[boot2], y_b[boot2])
                            coef2 = np.abs(c2.coef_.ravel())
                        imp_b_list.append(coef2)

                    elif model_name == 'NN':
                        if task == 'clf':
                            c = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=seed, early_stopping=True)
                        else:
                            c = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=seed, early_stopping=True)
                        c.fit(Xs_a, y_a)
                        perm = permutation_importance(c, Xs_a, y_a, n_repeats=3, random_state=seed)
                        imp_a_list.append(perm.importances_mean)
                        if task == 'clf':
                            acc_ind.append(accuracy_score(y_b, c.predict(Xs_b)))
                            preds_all.append(c.predict_proba(Xs_b))
                        else:
                            acc_ind.append(-mean_squared_error(y_b, c.predict(Xs_b)))
                            preds_all.append(c.predict(Xs_b))

                        if task == 'clf':
                            c2 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=seed+100, early_stopping=True)
                        else:
                            c2 = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=seed+100, early_stopping=True)
                        c2.fit(Xs_b, y_b)
                        perm2 = permutation_importance(c2, Xs_b, y_b, n_repeats=3, random_state=seed+100)
                        imp_b_list.append(perm2.importances_mean)

                imp_a = np.array(imp_a_list)
                imp_b = np.array(imp_b_list)

                # DASH ensemble prediction
                if task == 'clf':
                    avg_proba = np.mean(preds_all, axis=0)
                    dash_pred = np.argmax(avg_proba, axis=1)
                    acc_dash = accuracy_score(y_b, dash_pred)
                else:
                    acc_dash = -mean_squared_error(y_b, np.mean(preds_all, axis=0))

                tr = compute_tools(imp_a, imp_b, P, acc_ind, acc_dash, task=='clf')
                tr['dataset'] = ds_name; tr['model'] = model_name
                tr['method'] = method_name; tr['P'] = P; tr['task'] = task
                all_results.append(tr)
            except Exception as e:
                all_results.append({'dataset':ds_name,'model':model_name,'method':method_name,'error':str(e)[:60]})

        elapsed = time.time() - t0
        v = [r for r in all_results if 'error' not in r and r.get('sage_positive') is not None]
        sp = sum(1 for r in v if r['sage_positive'])
        print(f'  [{ds_idx+1}/{len(all_selected)}] {ds_name} (P={P_ds},{task}) | {elapsed:.0f}s | SAGE {sp}/{len(v)}')
    except Exception as e:
        print(f'  {ds_name}: ERROR - {str(e)[:60]}')

# ============================================================================
print('\n' + '='*70)
print('RESULTS')
print('='*70)

valid = [r for r in all_results if 'error' not in r]
errors = [r for r in all_results if 'error' in r]
print(f'Valid: {len(valid)}, Errors: {len(errors)}')

# Per-tool
for name, key in [('SAGE directional','sage_positive'),('DASH var','dash_reduces_variance'),
                   ('DASH acc','dash_preserves_accuracy'),('Gaussian R2>0.50','gaussian_pass'),
                   ('Bimodal>0.00','bimodal_000'),('Bimodal>0.05','bimodal_005'),('Bimodal>0.10','bimodal_010'),
                   ('Capacity','capacity_exceedance')]:
    app = [r for r in valid if r.get(key) is not None]
    pos = sum(1 for r in app if r[key])
    ci = wilson_ci(pos, len(app))
    print(f'  {name:<25} {pos:>3}/{len(app):<3} ({pos/len(app)*100:.0f}%) [{ci[0]*100:.0f}-{ci[1]*100:.0f}%]')

# Coverage conflict
cc = [(r['coverage_conflict'], r['mean_flip']) for r in valid if r.get('mean_flip',0)>0]
if cc:
    cv, mf = zip(*cc)
    rho, p = stats.spearmanr(cv, mf)
    print(f'  {"CC rho":<25} {rho:.3f}       p={p:.2e}')

# Model effect
print('\n  MODEL EFFECT (SAGE):')
sage_v = [r for r in valid if r.get('sage_positive') is not None]
table = []
for m in ['XGBoost','RF','LR','NN']:
    sub = [r for r in sage_v if r['model']==m]
    pos = sum(1 for r in sub if r['sage_positive'])
    table.append([pos, len(sub)-pos])
    print(f'    {m:10}: {pos}/{len(sub)}')
from scipy.stats import chi2_contingency
try:
    chi2, p_m, dof, _ = chi2_contingency(np.array(table))
    v_c = np.sqrt(chi2/(sum(sum(r) for r in table)*(min(len(table),2)-1)))
    print(f'    Chi2={chi2:.2f}, p={p_m:.4f}, Cramers V={v_c:.3f}')
except:
    print('    Chi2: could not compute')

# Cochran Q
print('\n  COCHRAN Q:')
datasets_all = sorted(set(r['dataset'] for r in sage_v))
cfg_names = [f'{m}-{mt}' for m,mt in configs]
Q_mat = np.full((len(datasets_all), len(cfg_names)), np.nan)
for r in sage_v:
    try:
        i = datasets_all.index(r['dataset'])
        j = cfg_names.index(f"{r['model']}-{r['method']}")
        Q_mat[i,j] = 1.0 if r['sage_positive'] else 0.0
    except: pass
mask = ~np.isnan(Q_mat).any(axis=1)
Q_c = Q_mat[mask]
if Q_c.shape[0] >= 3:
    k = Q_c.shape[1]; n = Q_c.shape[0]
    T = Q_c.sum(axis=1); C = Q_c.sum(axis=0); Nt = Q_c.sum()
    Q_s = (k-1)*(k*np.sum(C**2)-Nt**2)/(k*Nt-np.sum(T**2)) if (k*Nt-np.sum(T**2))>0 else 0
    p_q = 1-stats.chi2.cdf(Q_s, k-1)
    print(f'    Q={Q_s:.2f}, p={p_q:.4f}, n={n} datasets, k={k} configs')
    print(f'    {"EQUIVALENT" if p_q>0.05 else "DIFFERENT"}')

# Negative control
sg = [r['sage_gap'] for r in valid if r.get('sage_gap') is not None]
rg = [r['random_gap'] for r in valid if r.get('random_gap') is not None and r.get('sage_gap') is not None]
if sg and rg and len(sg)==len(rg):
    _, p_nc = stats.wilcoxon(sg, rg, alternative='greater')
    print(f'\n  NEGATIVE CONTROL: SAGE gap mean={np.mean(sg):.4f}, random={np.mean(rg):.4f}, p={p_nc:.2e}')

# By task
print('\n  BY TASK:')
for t in ['clf','reg']:
    sub = [r for r in sage_v if r.get('task')==t]
    if sub:
        pos = sum(1 for r in sub if r['sage_positive'])
        print(f'    {t}: {pos}/{len(sub)}')

elapsed = time.time() - t0
print(f'\n  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)')

with open('knockout-experiments/results_factorial_expanded.json', 'w') as f:
    json.dump({'n_valid':len(valid),'n_errors':len(errors),'per_test':all_results,
               'datasets':[n for n,_,_,_ in all_selected]}, f, indent=2, default=str)
print('  Saved: results_factorial_expanded.json')
