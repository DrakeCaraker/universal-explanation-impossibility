#!/usr/bin/env python
"""
Rescue attempt for the falsified dimension-estimator: replace the naive eigengap
with a PERMUTATION-NULL (parallel-analysis) rule that, by construction, keeps only
ensemble-covariance directions whose eigenvalue exceeds a column-permuted null.
This MUST pass the null control (independent features -> no collapse, ghat~p).

Tests: (A) null independent, (B) synthetic groups exact, (C) 4 real datasets.
A capacity estimator is only credible if it passes A. If it still fails A or
gives ghat=1 on real data, the from-data DIMENSION claim stays falsified (the
per-pair SNR predictor, validated separately, is unaffected).
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from numpy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing

def dim_parallel(coefs, B=200, seed=0, pct=95):
    r = np.random.default_rng(seed); M, p = coefs.shape
    w_obs = np.sort(eigh(np.cov(coefs, rowvar=False))[0])[::-1]
    null = np.zeros((B, p))
    for b in range(B):
        perm = np.column_stack([r.permutation(coefs[:, j]) for j in range(p)])
        null[b] = np.sort(eigh(np.cov(perm, rowvar=False))[0])[::-1]
    thr = np.percentile(null, pct, axis=0)
    k_unstable = int((w_obs > thr).sum())         # significant large eigs = unstable dirs
    return p - k_unstable                          # ghat = dim V^G

def ridge_ens(X, y, M=80, seed=0):
    r = np.random.default_rng(seed); n = X.shape[0]
    return np.array([Ridge(1.0).fit(X[i], y[i]).coef_ for i in (r.integers(0, n, n) for _ in range(M))])

rng = np.random.default_rng(3); out = {}

# A. NULL
gh = []
for s in range(15):
    p = int(rng.integers(6, 13)); n = 500; r = np.random.default_rng(s)
    X = (lambda z: (z-z.mean(0))/z.std(0))(r.standard_normal((n, p)))
    y = X @ r.standard_normal(p) + 0.4*r.standard_normal(n)
    gh.append((dim_parallel(ridge_ens(X, y, 80, s), seed=s), p))
out["A_null"] = {"mean_ghat_over_p": round(float(np.mean([g/p for g, p in gh])), 3),
                 "frac_no_collapse(ghat>=p-1)": round(float(np.mean([g >= p-1 for g, p in gh])), 3)}

# B. SYNTHETIC groups exact
def synth(g, size, n, seed):
    r = np.random.default_rng(seed); p = g*size; Z = r.standard_normal((n, g)); X = np.zeros((n, p)); c = 0
    for gi in range(g):
        for _ in range(size):
            X[:, c] = np.sqrt(0.9)*Z[:, gi] + np.sqrt(0.1)*r.standard_normal(n); c += 1
    y = Z @ r.standard_normal(g) + 0.4*r.standard_normal(n)
    return (X-X.mean(0))/(X.std(0)+1e-12), y
hits = []
for s in range(12):
    X, y = synth(4, 3, 600, s); hits.append(dim_parallel(ridge_ens(X, y, 80, s), seed=s))
out["B_synth_g4"] = {"ghat_values": hits, "recovery_rate_eq4": round(float(np.mean([h == 4 for h in hits])), 3)}

# C. REAL
def rf_ens(X, y, clf, M=40, seed=0):
    r = np.random.default_rng(seed); n = X.shape[0]; Mdl = RandomForestClassifier if clf else RandomForestRegressor
    return np.array([Mdl(n_estimators=60, max_depth=6, random_state=int(r.integers(1e9))).fit(X[i], y[i]).feature_importances_
                     for i in (r.integers(0, n, n) for _ in range(M))])
real = []
for name, loader, clf in [("breast_cancer", load_breast_cancer, True), ("wine", load_wine, True),
                          ("diabetes", load_diabetes, False), ("california", fetch_california_housing, False)]:
    d = loader(); X = np.asarray(d.data, float); y = np.asarray(d.target)
    if name == "california": X, y = X[:1200], y[:1200]
    X = (X-X.mean(0))/(X.std(0)+1e-12); p = X.shape[1]
    gh = dim_parallel(rf_ens(X, y, clf, 40, 1), seed=1, B=150)
    real.append({"dataset": name, "p": p, "ghat_parallel": int(gh)})
out["C_real"] = real
print(json.dumps(out, indent=2))
json.dump(out, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_dimfix_results.json', 'w'), indent=2)
