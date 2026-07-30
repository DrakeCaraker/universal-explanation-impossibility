#!/usr/bin/env python
"""
Two clean strengthenings, verified:
 S1 Cantelli guarantee: a flip is the MINORITY tail P(D<0) (when E D>0), so by the ONE-SIDED
    Cantelli inequality flip <= 1/(1+SNR^2) -- strictly tighter than Chebyshev's 1/SNR^2, never
    vacuous. Verify observed flip <= 1/(1+SNR^2) for all pairs, and the STABLE-band bound.
 S2 Combined capacity upper bound: feature-geometry catches correlation-Rashomon; the certificate
    catches target-symmetry Rashomon (low-correlation but UNRELIABLE pairs). Combine:
       extra_unstable = #nodes - #components  of the (|corr|<0.3 AND SNR<0.5) graph
       capacity_combined = feature_geo_upper - extra_unstable
    This TIGHTENS the upper bound where feature-geometry alone is loose. Test on the target-symmetry
    stress (should drop 8 -> 7) and on real datasets.
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, load_iris, fetch_california_housing

def ens(Mdl, X, y, M, seed):
    r = np.random.default_rng(seed); n = X.shape[0]
    return np.array([Mdl(random_state=int(r.integers(1e9))).fit(X[i], y[i]).feature_importances_
                     for i in (r.integers(0, n, n) for _ in range(M))])
def pairs(co):
    M, p = co.shape; out = []
    for j in range(p):
        for k in range(j+1, p):
            D = co[:, j]-co[:, k]; mu = D.mean(); sd = D.std()+1e-12; snr = abs(mu)/sd
            flip = float(min((D > 0).mean(), (D < 0).mean()))
            out.append((j, k, snr, flip))
    return out

DS = [("breast_cancer", load_breast_cancer, True), ("wine", load_wine, True),
      ("diabetes", load_diabetes, False), ("iris", load_iris, True),
      ("california", fetch_california_housing, False)]
out = {}

# S1: Cantelli vs Chebyshev, pooled
cant_ok = cheb_ok = tot = 0; stable_flips = []
for name, loader, clf in DS:
    d = loader(); X = np.asarray(d.data, float); y = np.asarray(d.target)
    if name == "california": X, y = X[:1000], y[:1000]
    X = (X-X.mean(0))/(X.std(0)+1e-12)
    Mdl = (lambda **k: RandomForestClassifier(n_estimators=50, max_depth=6, **k)) if clf else (lambda **k: RandomForestRegressor(n_estimators=50, max_depth=6, **k))
    for j, k, snr, flip in pairs(ens(Mdl, X, y, 40, 1)):
        tot += 1
        cant = min(0.5, 1.0/(1.0+snr**2)); cheb = min(1.0, 1.0/snr**2)
        cant_ok += int(flip <= cant + 1e-9); cheb_ok += int(flip <= cheb + 1e-9)
        if snr >= 2: stable_flips.append(flip)
out["S1_cantelli"] = {
    "cantelli_holds_frac (want 1.0)": round(cant_ok/tot, 4),
    "chebyshev_holds_frac": round(cheb_ok/tot, 4),
    "cantelli_bound_at_SNR2": round(1/(1+4), 4), "chebyshev_bound_at_SNR2": round(1/4, 4),
    "STABLE_band_max_flip (<= cantelli 0.20)": round(float(max(stable_flips)), 4), "n_pairs": tot}

# S2: combined capacity bound
def corr_effrank_hi(X):
    lam = np.sort(np.clip(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)), 0, None))[::-1]
    return int((lam >= 0.5).sum())
def extra_unstable(X, co):
    p = X.shape[1]; C = np.abs(np.corrcoef(X, rowvar=False))
    # graph of low-corr, UNRELIABLE pairs
    parent = list(range(p))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    nodes = set()
    for j, k, snr, flip in pairs(co):
        if C[j, k] < 0.3 and snr < 0.5:
            nodes.add(j); nodes.add(k); parent[find(j)] = find(k)
    if not nodes: return 0
    comps = len({find(x) for x in nodes})
    return len(nodes) - comps    # independent swap directions

# target-symmetry stress
r = np.random.default_rng(0); n = 800; p = 8
Xn = r.standard_normal((n, p)); y = (Xn[:,0]+Xn[:,1]) + 0.6*Xn[:,2] + 0.3*Xn[:,3] + 0.3*r.standard_normal(n)
Xn = (Xn-Xn.mean(0))/Xn.std(0)
co = ens(lambda **k: RandomForestRegressor(n_estimators=60, max_depth=6, **k), Xn, y, 60, 2)
hi = corr_effrank_hi(Xn); ex = extra_unstable(Xn, co)
out["S2_combined_stress"] = {"feature_geo_upper (loose, =p)": hi, "certificate_extra_unstable": ex,
                             "combined_upper (tight, want 7)": hi-ex}
real = []
for name, loader, clf in DS:
    d = loader(); X = np.asarray(d.data, float); y = np.asarray(d.target)
    if name == "california": X, y = X[:1000], y[:1000]
    X = (X-X.mean(0))/(X.std(0)+1e-12); p = X.shape[1]
    Mdl = (lambda **k: RandomForestClassifier(n_estimators=50, max_depth=6, **k)) if clf else (lambda **k: RandomForestRegressor(n_estimators=50, max_depth=6, **k))
    co = ens(Mdl, X, y, 40, 3); hi = corr_effrank_hi(X); ex = extra_unstable(X, co)
    real.append({"dataset": name, "p": p, "feature_geo_upper": hi, "extra_unstable": ex, "combined_upper": hi-ex})
out["S2_combined_real"] = real
print(json.dumps(out, indent=2))
json.dump(out, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/strengthen_results.json','w'), indent=2)
