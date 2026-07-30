#!/usr/bin/env python
"""
Solidify the feature-geometry capacity bound (adversarial, before believing it):
 1. tau-robustness: is the synthetic bracket=[4,4] an artifact of tau in [0.5,0.9], or
    does the count plateau at g over a WIDE tau range? (plateau => real gap, not tuning)
 2. subspace recovery: does the LARGE-eigenvalue feature subspace align with the TRUE
    V^G (group directions), and the SMALL-eigenvalue subspace with true (V^G)^perp?
    (recovering the subspace, not just the dimension)
 3. does the bound UPPER-bound the ensemble-observed instability? on synthetic, the #
    of genuinely unstable ensemble directions should be <= p - capacity_bound.
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from numpy.linalg import eigh, svd
from sklearn.linear_model import Ridge

def corr_eigs_vecs(X):
    C = np.corrcoef(X, rowvar=False); w, V = eigh(C)
    idx = np.argsort(w)[::-1]; return np.clip(w[idx],0,None), V[:, idx]
def align(A, B):
    if A.shape[1]==0 or B.shape[1]==0: return 1.0
    Qa,_=np.linalg.qr(A); Qb,_=np.linalg.qr(B)
    return float(np.mean(np.clip(svd(Qa.T@Qb, compute_uv=False),0,1)))

def synth(g, size, het, n, seed):
    r=np.random.default_rng(seed); p=g*size; Z=r.standard_normal((n,g)); X=np.zeros((n,p)); groups=np.repeat(np.arange(g),size); c=0
    for gi in range(g):
        for _ in range(size):
            load=1+het*r.standard_normal(); rho=np.clip(0.9+het*0.05*r.standard_normal(),0.5,0.99)
            X[:,c]=load*(np.sqrt(rho)*Z[:,gi]+np.sqrt(1-rho)*r.standard_normal(n)); c+=1
    Vg=np.zeros((p,g))
    for gi in range(g): idx=np.where(groups==gi)[0]; Vg[idx,gi]=1/np.sqrt(len(idx))
    # true (V^G)^perp = within-group difference directions
    Vperp=[]
    for gi in range(g):
        idx=np.where(groups==gi)[0]
        for k in range(1,len(idx)):
            v=np.zeros(p); v[idx[0]]=1; v[idx[k]]=-1; Vperp.append(v/np.linalg.norm(v))
    return X, groups, Vg, np.array(Vperp).T

out={}
# 1. tau-robustness (het=0.3, g=4)
taus=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
counts={t:[] for t in taus}
for s in range(15):
    X,_,_,_=synth(4,3,0.3,600,s); lam,_=corr_eigs_vecs(X)
    for t in taus: counts[t].append(int((lam>=t).sum()))
out["tau_robustness_synth_g4"]={str(t):round(float(np.mean(v)),2) for t,v in counts.items()}

# 2. subspace recovery (het=0..0.6)
sub={}
for het in [0.0,0.3,0.6]:
    aG,aP=[],[]
    for s in range(12):
        X,groups,Vg,Vperp=synth(4,3,het,600,s); lam,Vvec=corr_eigs_vecs(X)
        ghat=int((lam>=0.7).sum())                 # bracket point
        Vg_hat=Vvec[:, :ghat]                        # large-eig subspace = estimated V^G
        Vperp_hat=Vvec[:, ghat:]                      # small-eig subspace = estimated (V^G)^perp
        aG.append(align(Vg_hat,Vg)); aP.append(align(Vperp_hat,Vperp))
    sub[f"het{het}"]={"align_stable_vs_trueVG":round(float(np.mean(aG)),3),
                      "align_unstable_vs_trueVperp":round(float(np.mean(aP)),3)}
out["subspace_recovery"]=sub

# 3. upper-bound consistency: ensemble unstable count <= p - capacity_bound
cons=[]
for s in range(12):
    X,groups,Vg,Vperp=synth(4,3,0.3,600,s); n=X.shape[0]; p=X.shape[1]
    Xs=(X-X.mean(0))/(X.std(0)+1e-12); r=np.random.default_rng(s)
    y=(Xs@r.standard_normal(p))+0.4*r.standard_normal(n)
    co=np.array([Ridge(1.0).fit(Xs[i],y[i]).coef_ for i in (r.integers(0,n,n) for _ in range(80))])
    ens_var=np.sort(eigh(np.cov(co,rowvar=False))[0])[::-1]
    ens_unstable=int((ens_var>0.1*ens_var.max()).sum())   # rough ensemble unstable count
    lam,_=corr_eigs_vecs(X); cap_bound=int((lam>=0.7).sum())
    cons.append((ens_unstable, p-cap_bound))                # want ens_unstable ~ p-cap_bound (=8)
out["consistency_ens_unstable_vs_(p-cap)"]={"mean_ens_unstable":round(float(np.mean([a for a,_ in cons])),2),
                                            "mean_p_minus_cap":round(float(np.mean([b for _,b in cons])),2)}
print(json.dumps(out,indent=2))
json.dump(out,open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_bounds_confirm_results.json','w'),indent=2)
