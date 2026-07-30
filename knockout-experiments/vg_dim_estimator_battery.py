#!/usr/bin/env python
"""
Head-to-head battery for a BETTER capacity-dimension estimator. capacity = p - (eff
rank of the across-model covariance C). Candidates:
  N  naive eigengap (baseline; known to fail)
  D  data-level decorrelated null: shuffle each FEATURE column, refit ensemble ->
     C_null (bootstrap noise w/o redundancy); k_unstable = #{ sorted lambda_real[i]
     > p95(sorted lambda_null[i]) }.  (Proper parallel analysis.)
  S  split-half eigenvector reproducibility: real unstable dirs reproduce across an
     ensemble split; k_unstable = largest k with top-k subspace alignment >= 0.9.
  P  continuous participation-ratio capacity: eff_unstable = (Σλ)²/Σλ² of C.
A candidate is credible only if it PASSES the null (capacity ~ p on independent
features) AND recovers synthetic dim (capacity ~ g) AND is non-degenerate on real.
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from numpy.linalg import eigh, svd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing

def cov_eigs(C): return np.sort(eigh(C)[0])[::-1]
def align(A,B):
    if A.shape[1]==0 or B.shape[1]==0: return 1.0
    Qa,_=np.linalg.qr(A); Qb,_=np.linalg.qr(B)
    return float(np.mean(np.clip(svd(Qa.T@Qb,compute_uv=False),0,1)))

def ridge_coefs(X,y,M,seed):
    r=np.random.default_rng(seed); n=X.shape[0]
    return np.array([Ridge(1.0).fit(X[i],y[i]).coef_ for i in (r.integers(0,n,n) for _ in range(M))])
def rf_coefs(X,y,clf,M,seed):
    r=np.random.default_rng(seed); n=X.shape[0]; Mdl=RandomForestClassifier if clf else RandomForestRegressor
    return np.array([Mdl(n_estimators=50,max_depth=6,random_state=int(r.integers(1e9))).fit(X[i],y[i]).feature_importances_
                     for i in (r.integers(0,n,n) for _ in range(M))])

def cap_naive(C):
    w=cov_eigs(C); wn=w/(w.sum()+1e-12); p=len(wn)
    k=max((wn[i-1]/max(wn[i],1e-15),i) for i in range(1,p))[1]; return p-k
def cap_decorr(coefs, Xfit, yfit, fitfn, seed, reps=8):
    C=np.cov(coefs,rowvar=False); wr=cov_eigs(C); p=len(wr)
    nulls=[]
    for b in range(reps):
        rb=np.random.default_rng(seed*100+b)
        Xs=np.column_stack([rb.permutation(Xfit[:,j]) for j in range(Xfit.shape[1])])
        nulls.append(cov_eigs(np.cov(fitfn(Xs,yfit,coefs.shape[0],seed*100+b),rowvar=False)))
    thr=np.percentile(np.array(nulls),95,axis=0)
    return p-int((wr>thr).sum())
def cap_splithalf(coefs):
    M,p=coefs.shape; h=M//2; C1=np.cov(coefs[:h],rowvar=False); C2=np.cov(coefs[h:],rowvar=False)
    V1=eigh(C1)[1][:,::-1]; V2=eigh(C2)[1][:,::-1]  # descending
    k_unstable=0
    for k in range(1,p+1):
        if align(V1[:,:k],V2[:,:k])>=0.9: k_unstable=k
        else: break
    return p-k_unstable
def cap_pr(C):
    w=cov_eigs(C); w=w[w>0]; return len(cov_eigs(C))-float((w.sum()**2)/(np.sum(w**2)+1e-12))

rng=np.random.default_rng(5); out={}

# NULL
res={"N":[],"D":[],"S":[],"P":[]}
for s in range(12):
    p=int(rng.integers(6,12)); n=500; r=np.random.default_rng(s)
    X=(lambda z:(z-z.mean(0))/z.std(0))(r.standard_normal((n,p)))
    y=X@r.standard_normal(p)+0.4*r.standard_normal(n)
    co=ridge_coefs(X,y,60,s); C=np.cov(co,rowvar=False)
    res["N"].append(cap_naive(C)/p); res["S"].append(cap_splithalf(co)/p)
    res["D"].append(cap_decorr(co,X,y,ridge_coefs,s)/p); res["P"].append(cap_pr(C)/p)
out["NULL_capacity_over_p"]={k:round(float(np.mean(v)),3) for k,v in res.items()}

# SYNTH (exact + approx)
def synth(g,size,het,n,seed):
    r=np.random.default_rng(seed); p=g*size; Z=r.standard_normal((n,g)); X=np.zeros((n,p)); c=0
    for gi in range(g):
        for _ in range(size):
            load=1+het*r.standard_normal(); rho=np.clip(0.9+het*0.05*r.standard_normal(),0.5,0.99)
            X[:,c]=load*(np.sqrt(rho)*Z[:,gi]+np.sqrt(1-rho)*r.standard_normal(n)); c+=1
    y=Z@r.standard_normal(g)+0.4*r.standard_normal(n); return (X-X.mean(0))/(X.std(0)+1e-12),y
for het in [0.0,0.3]:
    res={"N":[],"D":[],"S":[],"P":[]}
    for s in range(10):
        X,y=synth(4,3,het,600,s); co=ridge_coefs(X,y,60,s); C=np.cov(co,rowvar=False)
        res["N"].append(cap_naive(C)); res["S"].append(cap_splithalf(co))
        res["D"].append(cap_decorr(co,X,y,ridge_coefs,s)); res["P"].append(cap_pr(C))
    out[f"SYNTH_g4_het{het}_capacity(true=4)"]={k:round(float(np.mean(v)),2) for k,v in res.items()}

# REAL
real=[]
for name,loader,clf in [("breast_cancer",load_breast_cancer,True),("wine",load_wine,True),
                        ("diabetes",load_diabetes,False),("california",fetch_california_housing,False)]:
    d=loader(); X=np.asarray(d.data,float); y=np.asarray(d.target)
    if name=="california": X,y=X[:1000],y[:1000]
    X=(X-X.mean(0))/(X.std(0)+1e-12); p=X.shape[1]
    co=rf_coefs(X,y,clf,30,1); C=np.cov(co,rowvar=False)
    real.append({"dataset":name,"p":p,"N":cap_naive(C),"D":cap_decorr(co,X,y,lambda a,b,m,sd:rf_coefs(a,b,clf,m,sd),1,reps=4),
                 "S":cap_splithalf(co),"P":round(cap_pr(C),2)})
out["REAL_capacity"]=real
print(json.dumps(out,indent=2))
json.dump(out,open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_battery_results.json','w'),indent=2)
