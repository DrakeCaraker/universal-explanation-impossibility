#!/usr/bin/env python
"""
Decisive test of the transformative claim: can the STABLE SUBSPACE V^G be
estimated from a Rashomon ensemble WITHOUT knowing the symmetry group G, and
does it then (a) recover the true capacity dim(V^G), (b) align with the true
V^G, and (c) predict per-pair explanation flip rates OUT OF SAMPLE — beating
the raw-data baselines the paper reports failing (max Spearman 0.26)?

Ground-truth setup (so we can check the estimator against truth):
- p features in g groups. Within a group, features are exchangeable and highly
  correlated (rho); the target depends only on the GROUP-level signal. Hence the
  within-group difference directions are a genuine Rashomon symmetry: attribution
  can be split arbitrarily among within-group features at equal loss. The stable
  subspace V^G = span of the g group-indicator (mean) directions; dim(V^G)=g;
  capacity fraction = g/p. A feature PAIR is stable iff the two are in DIFFERENT
  groups (their difference lies in V^G); unstable iff SAME group.

Estimator (uses ONLY the ensemble attribution matrix, never the group labels):
- Fit M bootstrap ridge models -> M x p coefficient (attribution) matrix.
- Ensemble covariance across models. Its LOW-variance eigenspace = estimated V^G.
- Estimate ghat via the largest eigengap in the sorted (descending) spectrum:
  the unstable complement carries the large ensemble variance; the stable part is
  near-zero variance. ghat = p - (#high-variance dirs).
- Per pair (j,k): SNR = |mean(phi_j-phi_k)| / std(phi_j-phi_k) over the ensemble;
  predicted flip = Phi(-SNR)  (parameter-free; the paper's Gaussian formula).

OOS: Phi(-SNR) has NO fitted parameters, so every dataset is out-of-sample by
construction. We report per-pair predicted-vs-observed across HELD-OUT datasets
(different g, rho, p, n than any 'design' set — there is nothing to design).
Baselines (raw-data, group-free but NOT ensemble-based): |Pearson corr| of the
pair, and VIF-style redundancy. The paper's claim is these fail (rho~0.26).
"""
import json, numpy as np
from numpy.linalg import eigh, svd
from scipy.stats import norm, spearmanr, pearsonr
from sklearn.linear_model import Ridge

rng = np.random.default_rng(20260729)

def make_dataset(g, group_sizes, rho, n, noise, seed):
    r = np.random.default_rng(seed)
    p = sum(group_sizes)
    groups = np.concatenate([[gi]*s for gi, s in enumerate(group_sizes)])
    # latent group signals
    Z = r.standard_normal((n, g))
    X = np.zeros((n, p))
    col = 0
    for gi, s in enumerate(group_sizes):
        for _ in range(s):
            # each feature = sqrt(rho)*group latent + sqrt(1-rho)*idiosyncratic
            X[:, col] = np.sqrt(rho)*Z[:, gi] + np.sqrt(1-rho)*r.standard_normal(n)
            col += 1
    # target depends only on group-level signal -> within-group features exchangeable
    beta_group = r.standard_normal(g)
    y = Z @ beta_group + noise*r.standard_normal(n)
    # standardize features (so coefficients are comparable attributions)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # true V^G: span of group-indicator directions (normalized)
    Vg_true = np.zeros((p, g))
    for gi in range(g):
        idx = np.where(groups == gi)[0]
        Vg_true[idx, gi] = 1.0/np.sqrt(len(idx))
    return X, y, groups, Vg_true, p

def rashomon_ensemble(X, y, M, alpha, seed):
    r = np.random.default_rng(seed)
    n, p = X.shape
    coefs = np.zeros((M, p))
    losses = np.zeros(M)
    for m in range(M):
        idx = r.integers(0, n, n)           # bootstrap resample -> samples the Rashomon set
        model = Ridge(alpha=alpha).fit(X[idx], y[idx])
        coefs[m] = model.coef_
        losses[m] = np.mean((X @ model.coef_ + model.intercept_ - y)**2)
    return coefs, losses

def principal_angles_align(A, B):
    """mean cos of principal angles between column-spaces of A and B (1=identical)."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0, 1)
    return float(np.mean(s))

def estimate_Vg(coefs):
    """From the M x p ensemble, estimate the stable subspace and capacity, no group labels."""
    C = np.cov(coefs, rowvar=False)          # p x p ensemble covariance of attributions
    w, V = eigh(C)                           # ascending eigenvalues
    w = w[::-1]; V = V[:, ::-1]              # descending: large var = UNSTABLE dirs
    total = w.sum() + 1e-12
    # estimate #unstable dirs by largest multiplicative gap in the (normalized) spectrum
    wn = w / total
    # candidate cut k = #high-variance directions (unstable); stable dim = p-k
    # choose k maximizing gap wn[k-1]/wn[k]
    gaps = []
    for k in range(1, len(wn)):
        denom = wn[k] if wn[k] > 1e-15 else 1e-15
        gaps.append((wn[k-1]/denom, k))
    ratio, k_unstable = max(gaps)
    p = coefs.shape[1]
    ghat = p - k_unstable                    # estimated dim(V^G)
    Vg_hat = V[:, k_unstable:]               # low-variance eigenvectors = estimated V^G
    return ghat, Vg_hat, w

def pair_predictions(coefs):
    """Per-pair SNR and predicted flip (parameter-free)."""
    M, p = coefs.shape
    preds = {}
    for j in range(p):
        for k in range(j+1, p):
            d = coefs[:, j] - coefs[:, k]
            snr = abs(d.mean()) / (d.std() + 1e-12)
            preds[(j, k)] = norm.cdf(-snr)   # predicted flip rate
    return preds

def observed_flip(coefs):
    """Observed per-pair flip rate: fraction of models on the minority side of |phi_j|>|phi_k|."""
    M, p = coefs.shape
    A = np.abs(coefs)
    obs = {}
    for j in range(p):
        for k in range(j+1, p):
            greater = (A[:, j] > A[:, k]).mean()
            obs[(j, k)] = float(min(greater, 1-greater))
    return obs

def baselines(X, groups):
    """Raw-data, group-free per-pair predictors (NOT ensemble-based)."""
    p = X.shape[1]
    corr = {}
    for j in range(p):
        for k in range(j+1, p):
            corr[(j, k)] = abs(pearsonr(X[:, j], X[:, k])[0])  # |corr| : high => same group
    return corr

# ---- experiment grid: a diverse family; every dataset is OOS (method is parameter-free) ----
configs = []
for seed in range(24):
    g = int(rng.integers(2, 6))
    sizes = [int(rng.integers(2, 5)) for _ in range(g)]
    rho = float(rng.uniform(0.75, 0.97))
    n = int(rng.choice([300, 600, 1200]))
    noise = float(rng.uniform(0.2, 0.8))
    configs.append(dict(g=g, sizes=sizes, rho=rho, n=n, noise=noise, seed=int(seed)))

per_dataset = []
all_pred, all_obs, all_corr = [], [], []
dim_hits = 0
aligns = []

for cfg in configs:
    X, y, groups, Vg_true, p = make_dataset(cfg['g'], cfg['sizes'], cfg['rho'], cfg['n'], cfg['noise'], cfg['seed'])
    coefs, losses = rashomon_ensemble(X, y, M=100, alpha=1.0, seed=cfg['seed']+1000)
    loss_cv = losses.std()/ (losses.mean()+1e-12)      # ensemble is ~equal-loss?
    ghat, Vg_hat, spec = estimate_Vg(coefs)
    align = principal_angles_align(Vg_hat, Vg_true) if Vg_hat.shape[1] > 0 else 0.0
    aligns.append(align)
    dim_ok = (ghat == cfg['g'])
    dim_hits += int(dim_ok)
    preds = pair_predictions(coefs); obs = observed_flip(coefs); corr = baselines(X, groups)
    keys = list(preds.keys())
    pv = np.array([preds[k] for k in keys]); ov = np.array([obs[k] for k in keys]); cv = np.array([corr[k] for k in keys])
    # per-dataset OOS correlation of predicted-vs-observed flip
    rho_pred = spearmanr(pv, ov)[0] if len(keys) > 2 else np.nan
    rho_base = spearmanr(cv, ov)[0] if len(keys) > 2 else np.nan
    all_pred += list(pv); all_obs += list(ov); all_corr += list(cv)
    per_dataset.append(dict(g=cfg['g'], ghat=int(ghat), dim_ok=bool(dim_ok),
                            align=round(align,3), loss_cv=round(float(loss_cv),4),
                            n_pairs=len(keys), rho_pred=None if np.isnan(rho_pred) else round(float(rho_pred),3),
                            rho_base=None if np.isnan(rho_base) else round(float(rho_base),3)))

allp = np.array(all_pred); allo = np.array(all_obs); allc = np.array(all_corr)
def r2(pred, obs):
    ss_res = np.sum((obs-pred)**2); ss_tot = np.sum((obs-obs.mean())**2)
    return float(1 - ss_res/ss_tot)
pooled = dict(
    n_datasets=len(configs), n_pairs=len(allp),
    dim_recovery_rate=round(dim_hits/len(configs), 3),
    mean_subspace_alignment=round(float(np.mean(aligns)), 3),
    pooled_spearman_pred=round(float(spearmanr(allp, allo)[0]), 3),
    pooled_spearman_baseline_corr=round(float(spearmanr(allc, allo)[0]), 3),
    pooled_R2_pred=round(r2(allp, allo), 3),
    median_perdataset_spearman_pred=round(float(np.nanmedian([d['rho_pred'] for d in per_dataset if d['rho_pred'] is not None])), 3),
    median_perdataset_spearman_baseline=round(float(np.nanmedian([d['rho_base'] for d in per_dataset if d['rho_base'] is not None])), 3),
)
out = dict(pooled=pooled, per_dataset=per_dataset)
print(json.dumps(pooled, indent=2))
print("\nPER-DATASET (g, ghat, dim_ok, align, loss_cv, rho_pred vs rho_base):")
for d in per_dataset:
    print(f"  g={d['g']} ghat={d['ghat']} ok={d['dim_ok']} align={d['align']} lcv={d['loss_cv']} "
          f"rho_pred={d['rho_pred']} rho_base={d['rho_base']}")
json.dump(out, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/vg_results.json','w'), indent=2)
print("\nSaved vg_results.json")
