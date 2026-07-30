#!/usr/bin/env python
"""
Adversarial hardening of the empirical-V^G estimator. Each block tries to FALSIFY
the claim "the stable subspace V^G is recoverable from a Rashomon ensemble without
knowing G, and its SNR predicts per-pair flip."  We report the numbers and let
them fall where they fall.

Blocks:
 A. NULL control: independent features (no symmetry). The estimator MUST NOT
    hallucinate a low-dim V^G — capacity should be ~ full (ghat ~ p).
 B. APPROXIMATE symmetry: within-group features not exactly exchangeable
    (heterogeneous loadings). Degradation curve of dim-recovery vs exactness.
 C. REAL data + REAL nonlinear models: bootstrap RandomForest ensembles, native
    impurity attribution, on 4 sklearn datasets. Does ensemble SNR predict observed
    per-pair flip (honest real-data metric)? Does estimated V^G match correlation
    groups? (breast_cancer also has known measurement-triplet groups.)
 D. ROBUSTNESS: small M, small n, p>n, and an alternate dim-selection rule.
"""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.linalg import eigh, svd
from scipy.stats import norm, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import (load_diabetes, load_breast_cancer, load_wine,
                              fetch_california_housing)

def estimate_dim(coefs, rule="ratio"):
    C = np.cov(coefs, rowvar=False); w = np.sort(eigh(C)[0])[::-1]
    total = w.sum() + 1e-12; wn = w/total; p = len(wn)
    if rule == "ratio":
        gaps = [(wn[k-1]/max(wn[k],1e-15), k) for k in range(1, p)]
        k_unstable = max(gaps)[1]
    else:  # variance-threshold: dirs holding >1% of ensemble variance are 'unstable'
        k_unstable = int((wn > 0.01).sum())
    return p - k_unstable, w

def snr_flip(coefs):
    M, p = coefs.shape; A = np.abs(coefs); pred, obs = [], []
    for j in range(p):
        for k in range(j+1, p):
            d = coefs[:, j]-coefs[:, k]; snr = abs(d.mean())/(d.std()+1e-12)
            pred.append(norm.cdf(-snr))
            g = (A[:, j] > A[:, k]).mean(); obs.append(min(g, 1-g))
    return np.array(pred), np.array(obs)

def align(A, B):
    if A.shape[1]==0 or B.shape[1]==0: return 0.0
    Qa,_=np.linalg.qr(A); Qb,_=np.linalg.qr(B)
    return float(np.mean(np.clip(svd(Qa.T@Qb, compute_uv=False),0,1)))

rng = np.random.default_rng(11)
report = {}

# ---------- A. NULL CONTROL ----------
null_ghats = []
for s in range(20):
    p = int(rng.integers(6, 14)); n = 500
    r = np.random.default_rng(s)
    X = r.standard_normal((n, p))                 # INDEPENDENT features
    X = (X-X.mean(0))/X.std(0)
    beta = r.standard_normal(p)                    # distinct importances, no symmetry
    y = X@beta + 0.4*r.standard_normal(n)
    coefs = np.array([Ridge(1.0).fit(X[i], y[i]).coef_
                      for i in (r.integers(0,n,n) for _ in range(80))])
    ghat,_ = estimate_dim(coefs)
    null_ghats.append((ghat, p))
frac_full = np.mean([g>=p-1 for g,p in null_ghats])   # should be ~1: no spurious collapse
report["A_null"] = {"mean_ghat_over_p": round(float(np.mean([g/p for g,p in null_ghats])),3),
                    "frac_reporting_no_collapse(ghat>=p-1)": round(float(frac_full),3),
                    "verdict": "PASS (no hallucinated V^G)" if frac_full>=0.8 else "FAIL (hallucinates structure)"}

# ---------- B. APPROXIMATE SYMMETRY ----------
def approx_dataset(g, size, het, n, seed):
    r=np.random.default_rng(seed); p=g*size; groups=np.repeat(np.arange(g),size)
    Z=r.standard_normal((n,g)); X=np.zeros((n,p)); col=0
    for gi in range(g):
        for _ in range(size):
            load = 1.0 + het*r.standard_normal()          # het=0 exact exchangeable; het>0 breaks it
            rho = np.clip(0.9 + het*0.05*r.standard_normal(),0.5,0.99)
            X[:,col]=load*(np.sqrt(rho)*Z[:,gi]+np.sqrt(1-rho)*r.standard_normal(n)); col+=1
    y=Z@r.standard_normal(g)+0.4*r.standard_normal(n); X=(X-X.mean(0))/(X.std(0)+1e-12)
    Vg=np.zeros((p,g))
    for gi in range(g): idx=np.where(groups==gi)[0]; Vg[idx,gi]=1/np.sqrt(len(idx))
    return X,y,groups,Vg,p
curve=[]
for het in [0.0,0.1,0.25,0.5,1.0]:
    hits=[]; als=[]
    for s in range(12):
        X,y,groups,Vg,p=approx_dataset(4,3,het,600,int(s+het*100))
        coefs=np.array([Ridge(1.0).fit(X[i],y[i]).coef_ for i in (np.random.default_rng(s).integers(0,600,600) for _ in range(80))])
        ghat,w=estimate_dim(coefs)
        C=np.cov(coefs,rowvar=False); Vg_hat=eigh(C)[1][:, :ghat]  # smallest-eigenvalue eigenvecs = stable V^G
        hits.append(int(ghat==4)); als.append(align(Vg_hat,Vg))
    curve.append({"het":het,"dim_recovery":round(float(np.mean(hits)),2),"subspace_align":round(float(np.mean(als)),3)})
report["B_approx_symmetry"]=curve

# ---------- C. REAL DATA + REAL MODELS ----------
def rf_ensemble(X,y,clf,M=50,seed=0):
    r=np.random.default_rng(seed); n=X.shape[0]; imps=[]
    Model=RandomForestClassifier if clf else RandomForestRegressor
    for _ in range(M):
        idx=r.integers(0,n,n)
        m=Model(n_estimators=60,max_depth=6,random_state=int(r.integers(1e9))).fit(X[idx],y[idx])
        imps.append(m.feature_importances_)
    return np.array(imps)
real=[]
datasets=[("breast_cancer",load_breast_cancer,True),("wine",load_wine,True),
          ("diabetes",load_diabetes,False),("california",fetch_california_housing,False)]
for name,loader,clf in datasets:
    d=loader(); X=np.asarray(d.data,float); y=np.asarray(d.target)
    if name=="california": X=X[:1500]; y=y[:1500]
    X=(X-X.mean(0))/(X.std(0)+1e-12); p=X.shape[1]
    imps=rf_ensemble(X,y,clf,M=50,seed=1)
    pred,obs=snr_flip(imps)
    rho_snr=spearmanr(pred,obs)[0]
    # baseline |corr| per pair
    cor=np.corrcoef(X,rowvar=False); base=[abs(cor[j,k]) for j in range(p) for k in range(j+1,p)]
    rho_base=spearmanr(base,obs)[0]
    # estimated V^G vs correlation-cluster groups
    Z=linkage(np.abs(cor), method="average")
    lab=fcluster(Z, t=0.6, criterion="distance"); gcorr=len(set(lab))
    ghat,w=estimate_dim(imps); C=np.cov(imps,rowvar=False); Vhat=eigh(C)[1][:, :ghat]
    Vg_corr=np.zeros((p,gcorr))
    for i,c in enumerate(sorted(set(lab))):
        idx=np.where(lab==c)[0]; Vg_corr[idx,i]=1/np.sqrt(len(idx))
    real.append({"dataset":name,"p":p,"snr_pred_flip_spearman":round(float(rho_snr),3),
                 "corr_baseline_spearman":round(float(rho_base),3),
                 "ghat":int(ghat),"g_corr_clusters":int(gcorr),
                 "Vhat_vs_corrgroups_align":round(float(align(Vhat,Vg_corr)),3)})
report["C_real_data"]=real

# ---------- D. ROBUSTNESS ----------
rob={}
Xb,yb,gb,Vgb,pb=approx_dataset(4,3,0.0,600,7)
for tag,(M,n) in {"M20":(20,600),"M100":(100,600),"n100":(80,100),"pgtn":(80,20)}.items():
    Xs,ys=Xb[:n],yb[:n]
    coefs=np.array([Ridge(1.0).fit(Xs[i],ys[i]).coef_ for i in (np.random.default_rng(k).integers(0,n,n) for k in range(M))])
    g1,_=estimate_dim(coefs,"ratio"); g2,_=estimate_dim(coefs,"thresh")
    rob[tag]={"ghat_ratio":int(g1),"ghat_thresh":int(g2),"true_g":4}
report["D_robustness"]=rob

print(json.dumps(report, indent=2))
json.dump(report, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_harden_results.json','w'), indent=2)
