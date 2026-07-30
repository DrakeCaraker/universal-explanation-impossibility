#!/usr/bin/env python
"""
Capacity BOUNDS from feature geometry (not the noisy ensemble).

Theory: attribution is unstable exactly along the near-null-space of the design X
(collinear directions where coefficients move at equal loss). So the stable capacity
dim(V^G) = effective rank of the feature covariance. This is a DATA-ONLY, model-
agnostic quantity computed from the noise-free feature geometry, and it PASSES THE
NULL by construction (independent features -> full-rank correlation -> capacity = p).

Deliverable: a two-sided bracket
   capacity in [ #{lambda_i >= tau_hi},  #{lambda_i >= tau_lo} ]
on the eigenvalues of the feature CORRELATION matrix (trace = p; independent -> all 1),
plus the parameter-free Roy effective rank exp(spectral entropy) as a point estimate.
The bracket TIGHTENS under a clean spectral gap (exact symmetry) and widens on smooth
spectra (real approximate symmetry) -- i.e. it is honest about what it can and can't pin.

Credible iff: NULL -> bracket ~ [p,p]; SYNTH(g=4) -> bracket contains 4 and is tight
under exact symmetry; REAL -> non-degenerate interval. We also check the bound is an
UPPER bound on the ensemble-observed stable count (consistency with the instability side).
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from numpy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing

def corr_eigs(X):
    C = np.corrcoef(X, rowvar=False)
    return np.sort(np.clip(eigh(C)[0], 0, None))[::-1]   # descending, >=0

def roy_effrank(lam):
    p = lam / (lam.sum() + 1e-12); p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))          # exp(spectral entropy)

def bracket(lam, tau_lo=0.5, tau_hi=0.9):
    return int((lam >= tau_hi).sum()), int((lam >= tau_lo).sum())

def stable_rank(lam):    # ||.||_F^2 / ||.||_2^2  (another param-free eff rank)
    return float((lam.sum()) / (lam.max() + 1e-12)) if lam.max() > 0 else 0.0

rng = np.random.default_rng(2); out = {}

# NULL: independent features -> capacity should be ~ p
res = []
for s in range(12):
    p = int(rng.integers(6, 12)); r = np.random.default_rng(s)
    X = r.standard_normal((600, p))
    lam = corr_eigs(X); lo, hi = bracket(lam)
    res.append((roy_effrank(lam)/p, lo/p, hi/p))
a = np.array(res)
out["NULL(want~1.0)"] = {"roy_over_p": round(float(a[:,0].mean()),3),
                         "bracket_lo_over_p": round(float(a[:,1].mean()),3),
                         "bracket_hi_over_p": round(float(a[:,2].mean()),3)}

# SYNTH g=4 (exact + approx)
def synth(g, size, het, n, seed):
    r = np.random.default_rng(seed); p = g*size; Z = r.standard_normal((n, g)); X = np.zeros((n, p)); c = 0
    for gi in range(g):
        for _ in range(size):
            load = 1 + het*r.standard_normal(); rho = np.clip(0.9 + het*0.05*r.standard_normal(), 0.5, 0.99)
            X[:, c] = load*(np.sqrt(rho)*Z[:, gi] + np.sqrt(1-rho)*r.standard_normal(n)); c += 1
    return X
for het in [0.0, 0.3, 0.6]:
    roys, los, his = [], [], []
    for s in range(12):
        lam = corr_eigs(synth(4, 3, het, 600, s)); lo, hi = bracket(lam)
        roys.append(roy_effrank(lam)); los.append(lo); his.append(hi)
    out[f"SYNTH_g4_het{het}(want 4)"] = {"roy": round(float(np.mean(roys)),2),
                                         "bracket": [round(float(np.mean(los)),1), round(float(np.mean(his)),1)]}

# REAL: report bracket + roy; check UPPER-bound consistency vs ensemble stable count
def rf_coefs(X, y, clf, M, seed):
    r = np.random.default_rng(seed); n = X.shape[0]; Mdl = RandomForestClassifier if clf else RandomForestRegressor
    return np.array([Mdl(n_estimators=50, max_depth=6, random_state=int(r.integers(1e9))).fit(X[i], y[i]).feature_importances_
                     for i in (r.integers(0, n, n) for _ in range(M))])
real = []
for name, loader, clf in [("breast_cancer", load_breast_cancer, True), ("wine", load_wine, True),
                          ("diabetes", load_diabetes, False), ("california", fetch_california_housing, False)]:
    d = loader(); X = np.asarray(d.data, float); y = np.asarray(d.target)
    if name == "california": X, y = X[:1000], y[:1000]
    Xs = (X - X.mean(0))/(X.std(0)+1e-12); p = Xs.shape[1]
    lam = corr_eigs(Xs); lo, hi = bracket(lam)
    # ensemble stable count: features with high per-feature attribution stability (SNR of its own importance>2)
    co = rf_coefs(Xs, y, clf, 30, 1)
    snr_feat = np.abs(co.mean(0)) / (co.std(0) + 1e-12)
    ens_stable = int((snr_feat > 2).sum())   # a rough ensemble-side count of clearly-stable features
    real.append({"dataset": name, "p": p, "bracket_capacity": [lo, hi],
                 "roy_effrank": round(roy_effrank(lam), 2), "stable_rank": round(stable_rank(lam), 2),
                 "ensemble_high_snr_features": ens_stable})
out["REAL"] = real
print(json.dumps(out, indent=2))
json.dump(out, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_bounds_results.json','w'), indent=2)
