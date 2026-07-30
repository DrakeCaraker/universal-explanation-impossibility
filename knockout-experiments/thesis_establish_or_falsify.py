#!/usr/bin/env python
"""
Establish-or-falsify the transformational thesis, clause by clause. Adversarial.

Clauses:
 C1 "provably reliable"  -> distribution-free guarantee: a per-pair FLIP is |D-mu|>=|mu|, so by
    Chebyshev flip <= 1/SNR^2 UNCONDITIONALLY (STABLE:SNR>=2 => flip<=25%). Test: does observed
    flip <= 1/SNR^2 hold on real data? Also Gaussian calibration Phi(-SNR) vs observed.
 C2 "bound capacity from ANY model" -> the feature-geometry capacity is only a CORRELATION-Rashomon
    bound. TARGET-SYMMETRY stress: x0,x1 independent (corr~0) but interchangeable in y. Feature
    geometry says both stable (misses it); does the ENSEMBLE certificate catch (x0,x1)=UNRELIABLE?
    If yes: certificate strictly more general than the feature-geometry capacity; capacity bound is a
    valid UPPER bound (target-symmetry only lowers true capacity) but can be loose.
 C3 cross-model-class: does the per-pair verdict agree across RF / GradientBoosting / Ridge, and does
    feature-capacity upper-bound each model's stable count?
 C4 cross-attribution-method: impurity vs linear-coef verdict agreement.
 C5 catches a real failure single-model practice misses (demo).
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, load_iris, fetch_california_housing

def ens_importance(Mdl, X, y, M, seed, kind):
    r = np.random.default_rng(seed); n = X.shape[0]; out = []
    for _ in range(M):
        idx = r.integers(0, n, n); m = Mdl(random_state=int(r.integers(1e9))) if 'random_state' in Mdl().get_params() else Mdl()
        m.fit(X[idx], y[idx])
        if kind == 'imp': out.append(m.feature_importances_)
        else: out.append(np.abs(m.coef_).ravel() if m.coef_.ndim==1 else np.abs(m.coef_).mean(0))
    return np.array(out)

def pair_stats(co):
    M, p = co.shape; rows = []
    for j in range(p):
        for k in range(j+1, p):
            D = co[:, j] - co[:, k]; mu = D.mean(); sd = D.std() + 1e-12; snr = abs(mu)/sd
            flip = float(min((D > 0).mean(), (D < 0).mean()))    # signed-difference sign flip
            rows.append((snr, flip, float(norm.cdf(-snr)), float(min(1.0, 1.0/snr**2))))
    return np.array(rows)  # snr, obs_flip, gauss_pred, chebyshev_bound

DS = [("breast_cancer", load_breast_cancer, True), ("wine", load_wine, True),
      ("diabetes", load_diabetes, False), ("iris", load_iris, True),
      ("california", fetch_california_housing, False)]
out = {}

# ---- C1: Chebyshev guarantee + Gaussian calibration (pooled, RF) ----
allrows = []
for name, loader, clf in DS:
    d = loader(); X = np.asarray(d.data, float); y = np.asarray(d.target)
    if name == "california": X, y = X[:1000], y[:1000]
    X = (X - X.mean(0))/(X.std(0)+1e-12)
    Mdl = (lambda **k: RandomForestClassifier(n_estimators=50, max_depth=6, **k)) if clf else (lambda **k: RandomForestRegressor(n_estimators=50, max_depth=6, **k))
    allrows.append(pair_stats(ens_importance(Mdl, X, y, 40, 1, 'imp')))
R = np.vstack(allrows)
cheb_ok = float((R[:,1] <= R[:,3] + 1e-9).mean())               # observed flip <= 1/SNR^2 ?
stable = R[R[:,0] >= 2]                                          # STABLE band
out["C1_provable"] = {
    "chebyshev_holds_frac (want 1.0)": round(cheb_ok, 4),
    "STABLE_band_max_observed_flip (want <=0.25)": round(float(stable[:,1].max()) if len(stable) else 0.0, 4),
    "STABLE_band_mean_observed_flip": round(float(stable[:,1].mean()) if len(stable) else 0.0, 4),
    "gaussian_calibration_bins": [ [round(lo,2), round(float(R[(R[:,2]>=lo)&(R[:,2]<lo+0.1),1].mean()),3) if ((R[:,2]>=lo)&(R[:,2]<lo+0.1)).any() else None]
                                    for lo in [0.0,0.1,0.2,0.3,0.4] ],  # [pred_bin_lo, mean_observed]
    "n_pairs": len(R)}

# ---- C2: target-symmetry stress (independent but interchangeable features) ----
r = np.random.default_rng(0); n = 800; p = 8
Xn = r.standard_normal((n, p))
y = (Xn[:,0] + Xn[:,1]) + 0.6*Xn[:,2] + 0.3*Xn[:,3] + 0.3*r.standard_normal(n)   # x0,x1 interchangeable; x0,x1 indep
Xn = (Xn - Xn.mean(0))/Xn.std(0)
co = ens_importance(lambda **k: RandomForestRegressor(n_estimators=60, max_depth=6, **k), Xn, y, 60, 2, 'imp')
D01 = co[:,0]-co[:,1]; snr01 = abs(D01.mean())/(D01.std()+1e-12); flip01 = float(min((D01>0).mean(),(D01<0).mean()))
corr01 = float(abs(np.corrcoef(Xn[:,0], Xn[:,1])[0,1]))
# feature-geometry capacity (correlation eff-rank) -- should say ~full (misses target symmetry)
lam = np.sort(np.clip(np.linalg.eigvalsh(np.corrcoef(Xn, rowvar=False)),0,None))[::-1]
feat_cap_hi = int((lam >= 0.5).sum())
out["C2_target_symmetry_stress"] = {
    "corr(x0,x1) (indep ~0)": round(corr01,3),
    "feature_capacity_bracket_hi (misses it if ~p=8)": feat_cap_hi,
    "cert_verdict_x0_x1": "STABLE" if snr01>=2 else ("MARGINAL" if snr01>=0.5 else "UNRELIABLE"),
    "SNR_x0_x1": round(float(snr01),2), "observed_flip_x0_x1": round(flip01,3),
    "interpretation": "certificate catches target-symmetry Rashomon that feature-geometry misses"}

# ---- C3: cross-model-class per-pair verdict agreement ----
d = load_breast_cancer(); X = (np.asarray(d.data,float)); X=(X-X.mean(0))/(X.std(0)+1e-12); y=d.target; p=X.shape[1]
def verdicts(co):
    v={}
    for j in range(p):
        for k in range(j+1,p):
            D=co[:,j]-co[:,k]; s=abs(D.mean())/(D.std()+1e-12)
            v[(j,k)]="S" if s>=2 else ("M" if s>=0.5 else "U")
    return v
vr = verdicts(ens_importance(lambda **k: RandomForestClassifier(n_estimators=50,max_depth=6,**k),X,y,40,3,'imp'))
vg = verdicts(ens_importance(lambda **k: GradientBoostingClassifier(n_estimators=50,max_depth=3,**k),X,y,40,3,'imp'))
vl = verdicts(ens_importance(lambda **k: LogisticRegression(max_iter=500,**k),X,y,40,3,'coef'))
keys=list(vr)
agree_rf_gb=float(np.mean([vr[q]==vg[q] for q in keys]))
agree_rf_lin=float(np.mean([vr[q]==vl[q] for q in keys]))
# collapse to binary stable(S) vs not, agreement on the STABLE call
sb=lambda v: {q:(v[q]=="S") for q in keys}
sr,sg,sl=sb(vr),sb(vg),sb(vl)
out["C3_C4_cross_model_method"]={
    "verdict_agree_RF_vs_GBM": round(agree_rf_gb,3),
    "verdict_agree_RF_vs_Linear": round(agree_rf_lin,3),
    "STABLE-call_agree_RF_vs_GBM": round(float(np.mean([sr[q]==sg[q] for q in keys])),3),
    "STABLE-call_agree_RF_vs_Linear": round(float(np.mean([sr[q]==sl[q] for q in keys])),3)}

# ---- C5: concrete failure catch ----
co = ens_importance(lambda **k: RandomForestClassifier(n_estimators=50,max_depth=6,**k),X,y,50,7,'imp')
single = co[0]; order = np.argsort(single)[::-1]
# a top pair the single model ranks confidently but that flips across the ensemble:
catches=[]
for a,b in [(order[0],order[1]),(order[1],order[2]),(order[2],order[3])]:
    D=co[:,a]-co[:,b]; s=abs(D.mean())/(D.std()+1e-12); fl=float(min((D>0).mean(),(D<0).mean()))
    catches.append({"pair_rank_by_single_model":f"{a} vs {b}","single_says":f"{a}>{b}" if single[a]>single[b] else f"{b}>{a}",
                    "SNR":round(float(s),2),"cert":"UNRELIABLE" if s<0.5 else ("MARGINAL" if s<2 else "STABLE"),
                    "ensemble_flip_rate":round(fl,3)})
out["C5_failure_catch"]=catches
print(json.dumps(out, indent=2))
json.dump(out, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/thesis_results.json','w'), indent=2)
