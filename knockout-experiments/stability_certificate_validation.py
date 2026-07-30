#!/usr/bin/env python
"""
#4 Certification tool + validation.

A per-query STABILITY CERTIFICATE derived from the impossibility theory: given a
model + a pairwise explanation query (is feature j more important than k?), fit a
Rashomon ensemble, compute the ensemble SNR = |mean(phi_j-phi_k)|/std, and return
    SNR >= 2   -> STABLE     (claim is safe to report)
    0.5<=SNR<2 -> MARGINAL   (report with caveat)
    SNR < 0.5  -> UNRELIABLE (provably unstable; do NOT report a direction)
The bands come from the Gaussian flip law flip=Phi(-SNR): SNR=2 -> flip~2.3%,
SNR=0.5 -> flip~31%.  This is the missing tri-band wrapper (dash-shap has
compute_snr but no banding / single-query entrypoint).

VALIDATION (the honest part): does the certificate mean what it says?  We check,
on held-out synthetic datasets with ground-truth groups, that (a) observed flip
rate is monotone decreasing across UNRELIABLE->MARGINAL->STABLE, (b) STABLE-
certified pairs are almost never truly within-group (precision), (c) UNRELIABLE-
certified pairs are almost always truly within-group.
"""
import json, numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

def make_dataset(g, sizes, rho, n, noise, seed):
    r = np.random.default_rng(seed); p = sum(sizes)
    groups = np.concatenate([[gi]*s for gi, s in enumerate(sizes)])
    Z = r.standard_normal((n, g)); X = np.zeros((n, p)); col = 0
    for gi, s in enumerate(sizes):
        for _ in range(s):
            X[:, col] = np.sqrt(rho)*Z[:, gi] + np.sqrt(1-rho)*r.standard_normal(n); col += 1
    y = Z @ r.standard_normal(g) + noise*r.standard_normal(n)
    X = (X - X.mean(0))/(X.std(0)+1e-12)
    return X, y, groups, p

def ensemble(X, y, M, seed):
    r = np.random.default_rng(seed); n = X.shape[0]
    return np.array([Ridge(alpha=1.0).fit(X[idx], y[idx]).coef_
                     for idx in (r.integers(0, n, n) for _ in range(M))])

def certify(coefs, j, k):
    d = coefs[:, j] - coefs[:, k]; snr = abs(d.mean())/(d.std()+1e-12)
    band = "STABLE" if snr >= 2 else ("MARGINAL" if snr >= 0.5 else "UNRELIABLE")
    return snr, band, float(norm.cdf(-snr))

rng = np.random.default_rng(7)
bands = {"STABLE": [], "MARGINAL": [], "UNRELIABLE": []}
stable_within = 0; stable_total = 0; unrel_within = 0; unrel_total = 0
for s in range(30):                       # 30 held-out datasets (nothing is fit; parameter-free)
    g = int(rng.integers(2, 6)); sizes = [int(rng.integers(2, 5)) for _ in range(g)]
    rho = float(rng.uniform(0.75, 0.97)); n = int(rng.choice([300, 600]))
    X, y, groups, p = make_dataset(g, sizes, rho, n, float(rng.uniform(0.2, 0.7)), s)
    C = ensemble(X, y, 100, s+500); A = np.abs(C)
    for j in range(p):
        for k in range(j+1, p):
            snr, band, _ = certify(C, j, k)
            greater = (A[:, j] > A[:, k]).mean(); obs_flip = float(min(greater, 1-greater))
            within = (groups[j] == groups[k])
            bands[band].append(obs_flip)
            if band == "STABLE": stable_total += 1; stable_within += int(within)
            if band == "UNRELIABLE": unrel_total += 1; unrel_within += int(within)

summary = {
    "mean_observed_flip_by_band": {b: round(float(np.mean(v)), 4) for b, v in bands.items() if v},
    "n_by_band": {b: len(v) for b, v in bands.items()},
    "monotone_unreliable>marginal>stable": (
        bool(np.mean(bands["UNRELIABLE"]) > np.mean(bands["MARGINAL"]) > np.mean(bands["STABLE"])
        if all(bands.values()) else None)),
    "STABLE_precision_truly_between_group": round(1 - stable_within/max(stable_total,1), 4),
    "UNRELIABLE_precision_truly_within_group": round(unrel_within/max(unrel_total,1), 4),
}
print(json.dumps(summary, indent=2))
json.dump(summary, open('/private/tmp/claude-501/-Users-drakecaraker/36f6d484-0188-469a-981d-34e4cbf2e2bc/scratchpad/cert_results.json','w'), indent=2)

# demo of the single-query API on one model+query
X, y, groups, p = make_dataset(3, [3,3,2], 0.9, 400, 0.4, 999)
C = ensemble(X, y, 100, 12345)
print("\nSingle-query demo (same-group 0 vs 1, diff-group 0 vs 6):")
for (j,k) in [(0,1),(0,6)]:
    snr, band, flip = certify(C, j, k)
    print(f"  feature {j} vs {k}: SNR={snr:.2f} -> {band}  (predicted flip {flip:.1%}, "
          f"truly {'within' if groups[j]==groups[k] else 'between'}-group)")
