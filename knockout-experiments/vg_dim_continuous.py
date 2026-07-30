#!/usr/bin/env python
"""
Final principled attempt: NULL-CORRECTED CONTINUOUS capacity. Combines the only two
things that individually worked -- the data-level decorrelated null (null-safe) and a
continuous effective-rank (dodges the integer/degeneracy obstruction).

  eff_unstable = PR(C_real) - PR(C_null)           # excess effective rank over bootstrap noise
  capacity_cont = p - max(0, eff_unstable)         # continuous
where PR(C) = (Σλ)²/Σλ² and C_null = across-model cov of an ensemble refit on
column-shuffled features (redundancy destroyed -> pure bootstrap noise).

Credible iff: NULL -> capacity ~ p; SYNTH(g=4) -> capacity ~ 4; REAL -> non-degenerate,
in (0,p), and correlates with the per-pair instability actually present.
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from numpy.linalg import eigh
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, fetch_california_housing

def PR(C):
    w = np.sort(eigh(C)[0])[::-1]; w = w[w > 1e-12]
    return float((w.sum()**2)/(np.sum(w**2)+1e-12))
def ridge_coefs(X,y,M,seed):
    r=np.random.default_rng(seed); n=X.shape[0]
    return np.array([Ridge(1.0).fit(X[i],y[i]).coef_ for i in (r.integers(0,n,n) for _ in range(M))])
def rf_coefs(X,y,clf,M,seed):
    r=np.random.default_rng(seed); n=X.shape[0]; Mdl=RandomForestClassifier if clf else RandomForestRegressor
    return np.array([Mdl(n_estimators=50,max_depth=6,random_state=int(r.integers(1e9))).fit(X[i],y[i]).feature_importances_
                     for i in (r.integers(0,n,n) for _ in range(M))])
def cap_cont(coefs, X, y, fitfn, seed, reps=6):
    p = coefs.shape[1]; pr_real = PR(np.cov(coefs, rowvar=False)); pr_nulls=[]
    for b in range(reps):
        rb=np.random.default_rng(seed*97+b)
        Xs=np.column_stack([rb.permutation(X[:,j]) for j in range(X.shape[1])])
        pr_nulls.append(PR(np.cov(fitfn(Xs,y,coefs.shape[0],seed*97+b), rowvar=False)))
    eff_unstable = max(0.0, pr_real - float(np.mean(pr_nulls)))
    return p - eff_unstable, pr_real, float(np.mean(pr_nulls))

rng=np.random.default_rng(9); out={}

# NULL
caps=[]
for s in range(12):
    p=int(rng.integers(6,12)); n=500; r=np.random.default_rng(s)
    X=(lambda z:(z-z.mean(0))/z.std(0))(r.standard_normal((n,p)))
    y=X@r.standard_normal(p)+0.4*r.standard_normal(n)
    co=ridge_coefs(X,y,60,s); c,_,_=cap_cont(co,X,y,ridge_coefs,s); caps.append(c/p)
out["NULL_capacity_over_p"]=round(float(np.mean(caps)),3)

# SYNTH
def synth(g,size,het,n,seed):
    r=np.random.default_rng(seed); p=g*size; Z=r.standard_normal((n,g)); X=np.zeros((n,p)); c=0
    for gi in range(g):
        for _ in range(size):
            load=1+het*r.standard_normal(); rho=np.clip(0.9+het*0.05*r.standard_normal(),0.5,0.99)
            X[:,c]=load*(np.sqrt(rho)*Z[:,gi]+np.sqrt(1-rho)*r.standard_normal(n)); c+=1
    y=Z@r.standard_normal(g)+0.4*r.standard_normal(n); return (X-X.mean(0))/(X.std(0)+1e-12),y
for het in [0.0,0.3,0.6]:
    cs=[]
    for s in range(10):
        X,y=synth(4,3,het,600,s); co=ridge_coefs(X,y,60,s); c,_,_=cap_cont(co,X,y,ridge_coefs,s); cs.append(c)
    out[f"SYNTH_g4_het{het}_capacity(true=4)"]=round(float(np.mean(cs)),2)

# REAL (+ does capacity's implied instability track the per-pair flip?)
real=[]
for name,loader,clf in [("breast_cancer",load_breast_cancer,True),("wine",load_wine,True),
                        ("diabetes",load_diabetes,False),("california",fetch_california_housing,False)]:
    d=loader(); X=np.asarray(d.data,float); y=np.asarray(d.target)
    if name=="california": X,y=X[:1000],y[:1000]
    X=(X-X.mean(0))/(X.std(0)+1e-12); p=X.shape[1]
    co=rf_coefs(X,y,clf,30,1); c,prr,prn=cap_cont(co,X,y,lambda a,b,m,sd:rf_coefs(a,b,clf,m,sd),1,reps=3)
    # per-pair observed mean flip as a scalar instability check
    A=np.abs(co); flips=[min((A[:,j]>A[:,k]).mean(),1-(A[:,j]>A[:,k]).mean()) for j in range(p) for k in range(j+1,p)]
    real.append({"dataset":name,"p":p,"capacity_cont":round(float(c),2),
                 "eta_cont":round(float(1-c/p),3),"mean_pair_flip":round(float(np.mean(flips)),3),
                 "PR_real":round(prr,2),"PR_null":round(prn,2)})
out["REAL"]=real
print(json.dumps(out,indent=2))
json.dump(out,open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_cont_results.json','w'),indent=2)
