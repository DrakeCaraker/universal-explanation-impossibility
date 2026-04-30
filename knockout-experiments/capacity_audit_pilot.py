"""
Explanation Capacity Audit — Pilot Study
==========================================
Runs the full audit pipeline on standard ML datasets across multiple domains.

For each dataset:
1. Trains 50 XGBoost models with stochastic regularisation (different seeds)
2. Computes TreeSHAP importance rankings per model
3. Identifies correlation groups via SAGE (hierarchical clustering on |ρ|)
4. Computes explanation capacity C = g (number of groups)
5. Measures pairwise flip rates (between-group vs within-group)
6. Reports exceedance: how many ranking claims exceed capacity

Pre-registered settings (from explanation-capacity-audit-preregistration.md):
- Primary correlation threshold: ρ* = 0.70
- Sensitivity thresholds: {0.50, 0.60, 0.70, 0.80, 0.90}
- Number of models: M = 50
- XGBoost: n_estimators=100, max_depth=6, subsample=0.8, colsample_bytree=0.5
- Falsification test: Mann-Whitney U, within > between, α = 0.005
"""

import json, time, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_wine,
    fetch_california_housing,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─── Configuration ───────────────────────────────────────────────────
M = 50                    # models per dataset
PRIMARY_THR = 0.70        # primary correlation threshold
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MAX_SHAP_SAMPLES = 200    # cap SHAP evaluation for speed
OUT = Path(__file__).resolve().parent

XGB_CLF = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0, eval_metric='logloss')
XGB_REG = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0)


# ═══════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════

def load_datasets():
    ds = {}

    # ── sklearn built-ins ──
    d = load_breast_cancer()
    ds["Breast_Cancer"] = dict(X=d.data, y=d.target,
        names=list(d.feature_names), task="clf", domain="Clinical")

    d = fetch_california_housing()
    ds["California_Housing"] = dict(X=d.data, y=d.target,
        names=list(d.feature_names), task="reg", domain="Economics")

    d = load_wine()
    ds["Wine"] = dict(X=d.data, y=(d.target == 0).astype(int),
        names=list(d.feature_names), task="clf", domain="Food_Science")

    d = load_diabetes()
    ds["Diabetes"] = dict(X=d.data, y=d.target,
        names=list(d.feature_names), task="reg", domain="Clinical")

    # ── Synthetic positive control: 3 groups of 4, ρ=0.99 ──
    # Matches noether_treeshap.py: all features have equal weight within group
    # (exchangeability requires equal marginal effect on y)
    rng = np.random.default_rng(42)
    n = 500
    P_syn, G_syn, GS = 12, 3, 4
    betas = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)  # same within group
    Sigma = np.zeros((P_syn, P_syn))
    for g in range(G_syn):
        sl = slice(g*GS, (g+1)*GS)
        Sigma[sl, sl] = 0.99
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    X_syn = rng.standard_normal((n, P_syn)) @ L.T
    y_syn = (X_syn @ betas + rng.normal(0, 1.0, n) > np.median(X_syn @ betas)).astype(int)
    ds["Synth_3groups_rho99"] = dict(
        X=X_syn, y=y_syn,
        names=[f"g{g}_f{f}" for g in range(3) for f in range(4)],
        task="clf", domain="Synthetic",
        _true_groups=[0]*4 + [1]*4 + [2]*4)

    # ── Synthetic negative control: 12 independent features ──
    rng2 = np.random.default_rng(43)
    X_ind = rng2.standard_normal((500, 12))
    betas_ind = np.array([3.0, -2.0, 1.5, -1.0] + [0.5]*4 + [0.1]*4)
    y_ind = (X_ind @ betas_ind + rng2.normal(0, 1.0, 500) > 0).astype(int)
    ds["Synth_Independent"] = dict(
        X=X_ind, y=y_ind,
        names=[f"indep_{i}" for i in range(12)],
        task="clf", domain="Synthetic")

    # ── Synthetic moderate correlation: 4 groups of 3, ρ=0.70 ──
    rng3 = np.random.default_rng(44)
    P_m, G_m, GS_m = 12, 4, 3
    betas_m = np.array([4.0]*3 + [2.0]*3 + [1.0]*3 + [0.3]*3)
    Sigma_m = np.zeros((P_m, P_m))
    for g in range(G_m):
        sl = slice(g*GS_m, (g+1)*GS_m)
        Sigma_m[sl, sl] = 0.70
    np.fill_diagonal(Sigma_m, 1.0)
    L_m = np.linalg.cholesky(Sigma_m)
    X_mod = rng3.standard_normal((n, P_m)) @ L_m.T
    y_mod = (X_mod @ betas_m + rng3.normal(0, 1.0, n) > np.median(X_mod @ betas_m)).astype(int)
    ds["Synth_4groups_rho70"] = dict(
        X=X_mod, y=y_mod,
        names=[f"g{g}_f{f}" for g in range(4) for f in range(3)],
        task="clf", domain="Synthetic",
        _true_groups=[0]*3 + [1]*3 + [2]*3 + [3]*3)

    # ── Gene expression from OpenML (if available) ──
    try:
        from sklearn.datasets import fetch_openml
        ap = fetch_openml(data_id=1137, as_frame=False, parser='auto')
        X_ap = ap.data.astype(float)
        y_ap = LabelEncoder().fit_transform(ap.target)
        var = np.var(X_ap, axis=0)
        top = np.argsort(var)[-50:]
        ds["AP_Colon_Kidney"] = dict(
            X=X_ap[:, top], y=y_ap,
            names=[f"gene_{i}" for i in range(50)],
            task="clf", domain="Genomics")
        print("  ✓ Loaded AP_Colon_Kidney from OpenML")
    except Exception as e:
        print(f"  [skip] AP_Colon_Kidney: {e}")

    # ── Try Heart Disease ──
    try:
        from sklearn.datasets import fetch_openml
        hd = fetch_openml(data_id=53, as_frame=False, parser='auto')
        X_hd = np.nan_to_num(hd.data.astype(float))
        y_hd = LabelEncoder().fit_transform(hd.target)
        y_hd = (y_hd > 0).astype(int)
        ds["Heart_Disease"] = dict(
            X=X_hd, y=y_hd,
            names=[f"feat_{i}" for i in range(X_hd.shape[1])],
            task="clf", domain="Clinical")
        print("  ✓ Loaded Heart Disease from OpenML")
    except Exception as e:
        print(f"  [skip] Heart Disease: {e}")

    return ds


# ═══════════════════════════════════════════════════════════════════════
# SAGE: Group Identification
# ═══════════════════════════════════════════════════════════════════════

def sage_groups(X, threshold):
    """Hierarchical clustering on |Spearman ρ|; cut at 1 − threshold."""
    rho = spearmanr(X).statistic
    if np.ndim(rho) == 0:          # only 2 features
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    rho = np.abs(rho)
    np.fill_diagonal(rho, 1.0)
    rho = np.clip(rho, 0, 1)
    dist = 1 - rho
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method='average')
    return fcluster(Z, t=1 - threshold, criterion='distance')


# ═══════════════════════════════════════════════════════════════════════
# Core Audit Pipeline
# ═══════════════════════════════════════════════════════════════════════

def audit(name, data):
    X, y, task = data["X"], data["y"], data["task"]
    P = X.shape[1]
    t0 = time.time()

    # ── 1. Train M models, collect importances ──
    imps = np.zeros((M, P))
    accs = []
    for s in range(M):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        if task == "clf":
            mdl = XGBClassifier(**XGB_CLF, random_state=s)
        else:
            mdl = XGBRegressor(**XGB_REG, random_state=s)
        mdl.fit(Xtr, ytr)
        accs.append(float(mdl.score(Xte, yte)))

        if HAS_SHAP:
            ex = shap.TreeExplainer(mdl)
            sv = ex.shap_values(Xte[:MAX_SHAP_SAMPLES])
            if isinstance(sv, list):
                sv = sv[1]
            imps[s] = np.mean(np.abs(sv), axis=0)
        else:
            imps[s] = mdl.feature_importances_

    # ── 2. Inter-model agreement ──
    rhos = []
    for i in range(M):
        for j in range(i + 1, M):
            r, _ = spearmanr(imps[i], imps[j])
            if np.isfinite(r):
                rhos.append(r)
    mean_rho = float(np.mean(rhos)) if rhos else float('nan')

    # ── 3. Pairwise flip rates ──
    flips = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            wins_i = np.sum(imps[:, i] > imps[:, j])
            fr = min(wins_i, M - wins_i) / M
            flips[i, j] = flips[j, i] = fr

    # ── 4. SAGE + capacity at each threshold ──
    thr_results = {}
    for thr in THRESHOLDS:
        grp = sage_groups(X, thr)
        g = int(len(np.unique(grp)))
        C = g

        between, within = [], []
        for i in range(P):
            for j in range(i + 1, P):
                (within if grp[i] == grp[j] else between).append(flips[i, j])

        # Mann-Whitney
        if len(within) >= 2 and len(between) >= 2:
            _, pval = mannwhitneyu(within, between, alternative='greater')
            pval = float(pval)
        else:
            pval = None

        total_pairs = P * (P - 1) // 2
        stable_pairs = g * (g - 1) // 2

        thr_results[f"{thr:.2f}"] = dict(
            n_groups=g, capacity=C,
            features_ranked=P,
            exceedance_ratio=round((P - C) / C, 2) if C > 0 else None,
            exceeds=bool(P > C),
            stable_pairs=stable_pairs, total_pairs=total_pairs,
            pct_unstable=round(100 * (total_pairs - stable_pairs) / total_pairs, 1)
                if total_pairs > 0 else 0,
            within_flip=round(float(np.mean(within)), 4) if within else None,
            between_flip=round(float(np.mean(between)), 4) if between else None,
            n_within=len(within), n_between=len(between),
            mw_p=round(pval, 10) if pval is not None else None,
            group_sizes=sorted([int(np.sum(grp == gid))
                                for gid in np.unique(grp)], reverse=True),
        )

    return dict(
        dataset=name, domain=data["domain"],
        n_samples=int(X.shape[0]), n_features=P, n_models=M,
        mean_acc=round(float(np.mean(accs)), 4),
        acc_range=[round(float(min(accs)), 4), round(float(max(accs)), 4)],
        mean_rho=round(mean_rho, 4),
        elapsed=round(time.time() - t0, 1),
        thresholds=thr_results,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  EXPLANATION CAPACITY AUDIT — PILOT STUDY")
    print("  Pre-registered: ρ*=0.70, M=50, XGBoost(subsample=0.8, colsample=0.5)")
    print(f"  SHAP: {'TreeSHAP' if HAS_SHAP else 'native importance (fallback)'}")
    print("=" * 78 + "\n")

    datasets = load_datasets()
    print(f"\n  {len(datasets)} datasets loaded. Beginning audit...\n")

    results = []
    for name, data in datasets.items():
        P = data["X"].shape[1]
        print(f"  [{name}] P={P}, domain={data['domain']} ... ", end="", flush=True)
        r = audit(name, data)
        results.append(r)
        p = r["thresholds"][f"{PRIMARY_THR:.2f}"]
        print(f"g={p['n_groups']}, C={p['capacity']}, "
              f"within={p['within_flip']}, between={p['between_flip']}, "
              f"p={p['mw_p']}, {r['elapsed']}s")

    # ═══════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════
    T = f"{PRIMARY_THR:.2f}"

    print("\n" + "=" * 78)
    print(f"  RESULTS  (primary threshold ρ* = {PRIMARY_THR})")
    print("=" * 78)

    hdr = (f"{'Dataset':<24} {'Domain':<14} {'P':>3} {'g':>3} {'C':>3} "
           f"{'Exc':>5} {'Within':>7} {'Between':>7}  {'p-value':>11} {'ρ̄':>5}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    n_exc = 0
    w_all, b_all = [], []
    for r in results:
        p = r["thresholds"][T]
        tag = " ***" if p["exceeds"] else ""
        n_exc += int(p["exceeds"])
        w_s = f"{p['within_flip']:.3f}" if p['within_flip'] is not None else "  — "
        b_s = f"{p['between_flip']:.3f}" if p['between_flip'] is not None else "  — "
        p_s = f"{p['mw_p']:.2e}" if p['mw_p'] is not None else "     — "
        print(f"{r['dataset']:<24} {r['domain']:<14} {r['n_features']:>3} "
              f"{p['n_groups']:>3} {p['capacity']:>3} "
              f"{'YES':>5}" if p['exceeds'] else f"{'no':>5}",
              end="")
        print(f" {w_s:>7} {b_s:>7}  {p_s:>11} {r['mean_rho']:>5.3f}{tag}")
        if p['within_flip'] is not None: w_all.append(p['within_flip'])
        if p['between_flip'] is not None: b_all.append(p['between_flip'])

    print("-" * len(hdr))

    # ── Summary statistics ──
    pct = 100 * n_exc / len(results)
    print(f"\n  Datasets exceeding capacity: {n_exc}/{len(results)} ({pct:.0f}%)")

    if w_all and b_all:
        print(f"  Mean within-group flip rate:  {np.mean(w_all):.4f}  (predicted: ~0.50)")
        print(f"  Mean between-group flip rate: {np.mean(b_all):.4f}  (predicted: ~0.00)")
        gap = np.mean(w_all) - np.mean(b_all)
        print(f"  Bimodal gap:                  {gap:.4f}")

    # ── Pairwise instability summary ──
    print(f"\n  Pairwise stability breakdown (ρ* = {PRIMARY_THR}):")
    total_s, total_t = 0, 0
    for r in results:
        p = r["thresholds"][T]
        total_s += p["stable_pairs"]
        total_t += p["total_pairs"]
    print(f"  Total pairwise claims:  {total_t}")
    print(f"  Stable (between-group): {total_s} ({100*total_s/total_t:.1f}%)")
    print(f"  Unstable (within-group): {total_t - total_s} ({100*(total_t-total_s)/total_t:.1f}%)")

    # ── Sensitivity ──
    print(f"\n  Sensitivity analysis: exceedance count by threshold")
    print(f"  {'ρ*':<6}", end="")
    for r in results:
        print(f" {r['dataset'][:10]:>11}", end="")
    print()
    for thr in THRESHOLDS:
        tk = f"{thr:.2f}"
        marker = " ◀" if thr == PRIMARY_THR else ""
        print(f"  {thr:<6.2f}", end="")
        for r in results:
            p = r["thresholds"][tk]
            print(f" {'g='+str(p['n_groups']):>5}/C={p['capacity']:<3}", end="")
        print(marker)

    # ── Falsification verdict ──
    print(f"\n  FALSIFICATION TEST (between-group vs within-group flip rates):")
    n_confirmed = 0
    for r in results:
        p = r["thresholds"][T]
        if (p['within_flip'] is not None and p['between_flip'] is not None
                and p['mw_p'] is not None):
            sig = p['mw_p'] < 0.005
            correct_dir = (p['within_flip'] > p['between_flip'])
            confirmed = sig and correct_dir
            if confirmed:
                n_confirmed += 1
            status = "CONFIRMED" if confirmed else ("direction ok, p>" if correct_dir else "FAILED")
            print(f"    {r['dataset']:<24} within={p['within_flip']:.3f}  "
                  f"between={p['between_flip']:.3f}  p={p['mw_p']:.2e}  {status}")
    print(f"\n  Confirmed: {n_confirmed}/{len(results)} datasets")

    # ── Save ──
    out_path = OUT / "results_capacity_audit_pilot.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
