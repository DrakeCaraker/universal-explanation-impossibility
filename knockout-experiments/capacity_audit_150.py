"""
Explanation Capacity Audit — 150+ Dataset Cross-Domain Validation
==================================================================
Comprehensive audit spanning 150+ datasets from sklearn, OpenML, and PMLB
across 40+ scientific domains.

Key framing (from peer review):
- The prediction is DIRECTIONAL: within-group flip rate > between-group flip rate
- The prediction is NOT that within-group = 0.50 (that's exact exchangeability only)
- The headline metric is ZERO significant reversals across all datasets
- "Over-specified" not "broken": group-level conclusions may be correct,
  feature-level specificity is unsupported

Pre-registered: ρ*=0.70, M=50, XGBoost(sub=0.8, col=0.5), TreeSHAP, α=0.005
"""

import json, time, traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
import shap

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
M = 50
PRIMARY_THR = 0.70
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MAX_SHAP = 200
MAX_SAMPLES = 3000
MIN_FEATURES = 8
TOP_K = 50
OUT = Path(__file__).resolve().parent

XGB_CLF = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0, eval_metric='logloss')
XGB_REG = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0)

# Domain assignment by keyword patterns in dataset names
DOMAIN_PATTERNS = [
    ('Genomics', ['gene', 'colon', 'kidney', 'breast', 'lung', 'endometrium',
                  'leukemia', 'tumor', 'cancer_w', 'dna', 'splice', 'promoter']),
    ('Oncology', ['leukemia', 'tumor', 'lymphoma']),
    ('Clinical', ['heart', 'diabetes', 'hepatitis', 'liver', 'thyroid',
                  'dermatology', 'spectf', 'parkinsons', 'arrhythmia',
                  'breast_cancer', 'wdbc', 'wpbc', 'haberman', 'saheart',
                  'pima', 'cleveland']),
    ('Neuroscience', ['mice', 'brain', 'eeg']),
    ('Drug_Discovery', ['bioresponse', 'qsar', 'biodeg', 'drug']),
    ('Chemistry', ['qsar', 'biodeg', 'molecular', 'analcatdata_cyyoung']),
    ('Physics', ['ionosphere', 'magic', 'particle', 'boone', 'higgs',
                 'electron', 'gamma']),
    ('Remote_Sensing', ['satellite', 'segment', 'landsat']),
    ('Computer_Vision', ['digit', 'mnist', 'optdigit', 'pendigit', 'texture',
                         'mfeat', 'letter']),
    ('Manufacturing', ['steel', 'cylinder', 'machine']),
    ('Environmental', ['ozone', 'pollution', 'air', 'cloud', 'wind']),
    ('Finance', ['credit', 'bank', 'german']),
    ('Software_Eng', ['kc2', 'jm1', 'pc1', 'pc3', 'pc4', 'mc1', 'mc2',
                      'cm1', 'mw1', 'kc1']),
    ('Ecology', ['abalone', 'yeast', 'ecoli']),
    ('Economics', ['house', 'california', 'analcatdata_election', 'adult']),
    ('Food_Science', ['wine', 'tecator', 'vineyard']),
    ('Wearables', ['har', 'activity', 'gesture']),
    ('NLP', ['spam', 'cnae', 'collins', 'authorship', 'dexter']),
    ('Vehicle_Eng', ['vehicle', 'auto']),
    ('Robotics', ['wall', 'robot']),
    ('Energy', ['electricity', 'energy', 'power', 'solar']),
    ('Agriculture', ['soybean', 'seeds', 'flare']),
    ('Geophysics', ['sonar', 'seismic']),
    ('Signal_Processing', ['waveform', 'twonorm', 'ringnorm', 'phoneme']),
    ('Social_Science', ['vote', 'labor', 'census', 'election']),
    ('Telecommunications', ['churn', 'connect']),
]


def assign_domain(name):
    name_lower = name.lower()
    for domain, keywords in DOMAIN_PATTERNS:
        for kw in keywords:
            if kw in name_lower:
                return domain
    return "Other"


# ═══════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════

def prep(X, y, task, top_k=None):
    X = np.array(X, dtype=float)
    y = np.array(y)
    # Drop all-NaN columns
    good = ~np.all(np.isnan(X), axis=0)
    X = X[:, good]
    # Impute NaN with median
    for c in range(X.shape[1]):
        m = np.isnan(X[:, c])
        if m.any():
            X[m, c] = np.nanmedian(X[:, c])
    # Drop zero-variance
    v = np.var(X, axis=0)
    X = X[:, v > 1e-12]
    # Top-k by variance
    if top_k and X.shape[1] > top_k:
        X = X[:, np.argsort(np.var(X, axis=0))[-top_k:]]
    # Encode target
    if task == "clf":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        if len(np.unique(y)) > 2:
            mode = np.bincount(y).argmax()
            y = (y == mode).astype(int)
    # Subsample
    if X.shape[0] > MAX_SAMPLES:
        idx = np.random.RandomState(0).choice(X.shape[0], MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════

def load_all():
    ds = {}
    seen_names = set()

    def add(name, X, y, task, domain=None):
        if name in seen_names:
            return
        try:
            X, y = prep(X, y, task, top_k=TOP_K)
            if X.shape[1] >= MIN_FEATURES:
                d = domain or assign_domain(name)
                ds[name] = dict(X=X, y=y, task=task, domain=d)
                seen_names.add(name)
                return True
        except Exception:
            pass
        return False

    # ── 1. SKLEARN ────────────────────────────────────────────
    from sklearn.datasets import (
        load_breast_cancer, load_diabetes, load_wine,
        fetch_california_housing, load_digits)

    d = load_breast_cancer()
    add("Breast_Cancer", d.data, d.target, "clf", "Clinical")
    d = fetch_california_housing()
    add("California_Housing", d.data, d.target, "reg", "Economics")
    d = load_wine()
    add("Wine", d.data, (d.target == 0).astype(int), "clf", "Food_Science")
    d = load_diabetes()
    add("Diabetes", d.data, d.target, "reg", "Clinical")
    d = load_digits()
    add("Digits", d.data, d.target, "clf", "Computer_Vision")

    print(f"  sklearn: {len(ds)} datasets")

    # ── 2. OPENML ─────────────────────────────────────────────
    openml_ids = [
        # Genomics
        (1137, "clf", "Genomics"), (1138, "clf", "Genomics"),
        (1166, "clf", "Genomics"), (1167, "clf", "Genomics"),
        (1104, "clf", "Oncology"),
        # Clinical
        (53, "clf", "Clinical"), (37, "clf", "Clinical"),
        (337, "clf", "Clinical"), (1510, "clf", "Clinical"),
        (1511, "clf", "Clinical"), (1488, "clf", "Neurology"),
        (35, "clf", "Dermatology"), (5, "clf", "Cardiology"),
        # CS / NLP
        (44, "clf", "NLP"), (1468, "clf", "Text_Mining"),
        (458, "clf", "Stylometry"), (40971, "clf", "NLP"),
        # Software
        (1063, "clf", "Software_Eng"), (1049, "clf", "Software_Eng"),
        (1053, "clf", "Software_Eng"), (1050, "clf", "Software_Eng"),
        (1056, "clf", "Software_Eng"),
        # Manufacturing / Materials
        (40982, "clf", "Manufacturing"), (40499, "clf", "Materials"),
        # Remote Sensing
        (182, "clf", "Remote_Sensing"), (40984, "clf", "Remote_Sensing"),
        # Physics
        (59, "clf", "Physics"), (1120, "clf", "Astrophysics"),
        (41150, "clf", "Particle_Physics"),
        # Environment / Climate
        (40994, "clf", "Climate"), (1487, "clf", "Environmental"),
        # Drug Discovery / Chemistry
        (4134, "clf", "Drug_Discovery"), (1494, "clf", "Chemistry"),
        # Neuroscience
        (40966, "clf", "Neuroscience"),
        # Other domains
        (294, "clf", "Veterinary"), (181, "clf", "Microbiology"),
        (1478, "clf", "Wearables"), (4538, "clf", "HCI"),
        (54, "clf", "Vehicle_Eng"), (41, "clf", "Forensics"),
        (40, "clf", "Geophysics"), (151, "clf", "Energy"),
        (574, "reg", "Geography"), (1497, "clf", "Robotics"),
        (505, "reg", "Food_Spectroscopy"), (806, "clf", "Census"),
        (1485, "clf", "Synthetic_ML"), (1038, "clf", "Synthetic_ML"),
        (41142, "clf", "AutoML"),
        # Pattern Recognition
        (12, "clf", "Pattern_Recognition"), (14, "clf", "Pattern_Recognition"),
        (28, "clf", "Computer_Vision"), (32, "clf", "Computer_Vision"),
        # Additional high-value from the 3116 available
        (4532, "clf", "Particle_Physics"),   # higgs
        (1593, "clf", "Vehicle_Sensing"),    # SensIT Vehicle
        (23517, "clf", "Finance"),           # numerai
        (43429, "clf", "Particle_Physics"),  # CERN electron
        (43455, "clf", "Astronomy"),         # SDSS
        (44231, "clf", "Telecommunications"),# mobile churn
        (42343, "clf", "Marketing"),         # KDD98
        (41168, "clf", "Benchmark"),         # jannis
        (41169, "clf", "Benchmark"),         # helena
        (46302, "clf", "Cybersecurity"),     # UNSW_NB15
        (43039, "clf", "Cybersecurity"),     # internet-firewall
        (43079, "clf", "Software_Quality"),  # code smells
        (44975, "reg", "Wave_Energy"),       # wave energy
        (44974, "reg", "Video_Tech"),        # video transcoding
        (43733, "clf", "Epidemiology"),      # Covid-19
        (45548, "clf", "E_Commerce"),        # Otto Group
        (43617, "clf", "Healthcare_Admin"),  # medical appointment
        (1591, "clf", "Board_Games"),        # connect-4
        (46908, "clf", "Truck_Maint"),       # APS Failure
    ]

    from sklearn.datasets import fetch_openml
    n_oml = 0
    for did, task, domain in openml_ids:
        try:
            d = fetch_openml(data_id=did, as_frame=False, parser='auto')
            name = f"OML_{did}_{d['details']['name'] if 'details' in d else str(did)}"
            # Shorten name
            name = str(d.get('details', {}).get('name', f'oml_{did}'))[:30]
            name = name.replace(' ', '_').replace('-', '_')
            if add(name, d.data, d.target, task, domain):
                n_oml += 1
        except Exception:
            pass
    print(f"  OpenML: {n_oml} additional datasets")

    # ── 3. PMLB ───────────────────────────────────────────────
    try:
        import pmlb
        pmlb_clf = pmlb.classification_dataset_names
        pmlb_reg = pmlb.regression_dataset_names
        n_pmlb = 0
        for name in sorted(pmlb_clf + pmlb_reg):
            if n_pmlb >= 80:  # cap at 80 PMLB datasets
                break
            # Skip names we likely already have
            if any(s in name.lower() for s in ['breast', 'iris', 'wine', 'diabetes']):
                if name.lower() in [n.lower() for n in seen_names]:
                    continue
            try:
                df = pmlb.fetch_data(name, return_X_y=False,
                                     local_cache_dir='/tmp/pmlb_cache')
                X_p = df.iloc[:, :-1].values.astype(float)
                y_p = df.iloc[:, -1].values
                task = "clf" if name in pmlb_clf else "reg"
                pname = f"PMLB_{name}"
                if add(pname, X_p, y_p, task):
                    n_pmlb += 1
            except Exception:
                pass
        print(f"  PMLB: {n_pmlb} additional datasets")
    except ImportError:
        print("  PMLB: not available")

    # ── 4. SYNTHETIC CONTROLS ─────────────────────────────────
    def corr_groups(ng, gs, rho, seed, betas):
        rng = np.random.default_rng(seed)
        P = ng * gs
        b = np.concatenate([np.full(gs, bl) for bl in betas])
        S = np.zeros((P, P))
        for g in range(ng):
            sl = slice(g*gs, (g+1)*gs)
            S[sl, sl] = rho
        np.fill_diagonal(S, 1.0)
        L = np.linalg.cholesky(S)
        X = rng.standard_normal((500, P)) @ L.T
        sc = X @ b + rng.normal(0, 1, 500)
        return X, (sc > np.median(sc)).astype(int)

    X, y = corr_groups(3, 4, 0.99, 42, [5, 2, 0.5])
    add("Synth_3g_rho99", X, y, "clf", "Synthetic_Control")
    X, y = corr_groups(4, 3, 0.70, 44, [4, 2, 1, 0.3])
    add("Synth_4g_rho70", X, y, "clf", "Synthetic_Control")
    X, y = corr_groups(5, 3, 0.50, 45, [3, 2, 1.5, 1, 0.5])
    add("Synth_5g_rho50", X, y, "clf", "Synthetic_Control")
    X, y = corr_groups(2, 10, 0.95, 46, [3, 1])
    add("Synth_2g_rho95", X, y, "clf", "Synthetic_Control")

    rng = np.random.default_rng(43)
    Xi = rng.standard_normal((500, 15))
    yi = (Xi @ np.array([3,-2,1.5,-1,.8,-.6,.4,-.2,.1,0,0,0,0,0,0])
          + rng.normal(0,1,500) > 0).astype(int)
    add("Synth_Independent", Xi, yi, "clf", "Synthetic_Control")

    print(f"  Synthetic: 5 controls")
    return ds


# ═══════════════════════════════════════════════════════════════════════
# SAGE + Audit Core (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════════════

def sage_groups(X, thr):
    rho = spearmanr(X).statistic
    if np.ndim(rho) == 0:
        rho = np.array([[1, abs(float(rho))], [abs(float(rho)), 1]])
    rho = np.abs(np.nan_to_num(rho, nan=0))
    np.fill_diagonal(rho, 1)
    rho = np.clip(rho, 0, 1)
    d = 1 - rho; d = (d + d.T)/2; np.fill_diagonal(d, 0); d = np.clip(d, 0, 2)
    return fcluster(linkage(squareform(d, checks=False), 'average'),
                    t=1-thr, criterion='distance')


def audit_one(name, data):
    X, y, task = data["X"], data["y"], data["task"]
    P = X.shape[1]
    t0 = time.time()

    imps = np.zeros((M, P))
    accs = []
    for s in range(M):
        strat = y if task == "clf" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=s, stratify=strat)
        mdl = (XGBClassifier(**XGB_CLF, random_state=s) if task == "clf"
               else XGBRegressor(**XGB_REG, random_state=s))
        mdl.fit(Xtr, ytr)
        accs.append(float(mdl.score(Xte, yte)))
        sv = shap.TreeExplainer(mdl).shap_values(Xte[:MAX_SHAP])
        if isinstance(sv, list): sv = sv[1]
        imps[s] = np.mean(np.abs(sv), axis=0)

    # Inter-model ρ (sampled for speed)
    rhos = []
    for i in range(0, M, 3):
        for j in range(i+1, min(i+4, M)):
            r, _ = spearmanr(imps[i], imps[j])
            if np.isfinite(r): rhos.append(r)

    # Pairwise flips
    flips = np.zeros((P, P))
    for i in range(P):
        for j in range(i+1, P):
            w = np.sum(imps[:, i] > imps[:, j])
            flips[i,j] = flips[j,i] = min(w, M-w)/M

    thr_res = {}
    for thr in THRESHOLDS:
        grp = sage_groups(X, thr)
        g = int(len(np.unique(grp)))
        bw, wi = [], []
        for i in range(P):
            for j in range(i+1, P):
                (wi if grp[i]==grp[j] else bw).append(flips[i,j])

        pval, cd = None, None
        if len(wi) >= 2 and len(bw) >= 2:
            _, pval = mannwhitneyu(wi, bw, alternative='greater')
            pval = float(pval)
            ps = np.sqrt((np.var(wi)+np.var(bw))/2)
            cd = (np.mean(wi)-np.mean(bw))/ps if ps > 0 else 0

        tp = P*(P-1)//2
        sp = g*(g-1)//2
        thr_res[f"{thr:.2f}"] = dict(
            g=g, C=g, P=P, exceed=bool(P>g),
            exc_ratio=round((P-g)/g,2) if g>0 else None,
            stable=sp, total=tp,
            pct_unstable=round(100*(tp-sp)/tp,1) if tp>0 else 0,
            within_flip=round(float(np.mean(wi)),4) if wi else None,
            between_flip=round(float(np.mean(bw)),4) if bw else None,
            n_w=len(wi), n_b=len(bw),
            mw_p=pval,
            cohens_d=round(cd,3) if cd is not None else None,
        )

    return dict(
        dataset=name, domain=data["domain"],
        N=int(X.shape[0]), P=P, M=M, task=task,
        acc_mean=round(float(np.mean(accs)),4),
        rho_mean=round(float(np.mean(rhos)),4) if rhos else None,
        elapsed=round(time.time()-t0,1),
        thresholds=thr_res,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  EXPLANATION CAPACITY AUDIT — 150+ DATASET VALIDATION")
    print("  ρ*=0.70 | M=50 | XGBoost(sub=0.8,col=0.5) | TreeSHAP")
    print("=" * 80 + "\n")

    datasets = load_all()
    domains = set(d["domain"] for d in datasets.values())
    print(f"\n  TOTAL: {len(datasets)} datasets across {len(domains)} domains\n")
    print("-" * 80)

    results = []
    for i, (name, data) in enumerate(datasets.items(), 1):
        P = data["X"].shape[1]
        print(f"  [{i}/{len(datasets)}] {name[:32]:<33} P={P:<3} "
              f"{data['domain'][:16]:<17} ", end="", flush=True)
        try:
            r = audit_one(name, data)
            results.append(r)
            p = r["thresholds"][f"{PRIMARY_THR:.2f}"]
            mw = p['mw_p']
            ps = '<1e-15' if mw is not None and mw < 1e-15 else (
                 f'{mw:.1e}' if mw is not None else 'N/A')
            wf = p['within_flip'] or 0
            bf = p['between_flip'] or 0
            print(f"g={p['g']:<3} w={wf:.3f} b={bf:.3f} p={ps:<9} {r['elapsed']:.0f}s")
        except Exception as e:
            print(f"FAIL: {str(e)[:40]}")

        # Checkpoint every 20
        if i % 20 == 0:
            with open(OUT / "results_audit_150_checkpoint.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  --- checkpoint: {len(results)} results saved ---")

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════
    T = f"{PRIMARY_THR:.2f}"

    # Collect stats
    n_exc = sum(1 for r in results if r["thresholds"][T]["exceed"])
    n_with_data = 0
    n_dir = n_sig = n_sig_rev = 0
    all_w, all_b, all_d = [], [], []
    tot_stable = tot_pairs = 0
    domain_stats = defaultdict(lambda: [0, 0])  # [exceed, total]

    for r in results:
        p = r["thresholds"][T]
        if p["exceed"]: domain_stats[r["domain"]][0] += 1
        domain_stats[r["domain"]][1] += 1
        tot_stable += p["stable"]
        tot_pairs += p["total"]
        wf, bf, mw, cd = p["within_flip"], p["between_flip"], p["mw_p"], p["cohens_d"]
        if wf is not None and bf is not None:
            n_with_data += 1
            all_w.append(wf); all_b.append(bf)
            if cd is not None: all_d.append(cd)
            if wf > bf: n_dir += 1
            if mw is not None and mw < 0.005:
                if wf > bf: n_sig += 1
                else: n_sig_rev += 1

    print("\n" + "=" * 80)
    print(f"  RESULTS: {len(results)} datasets, {len(set(r['domain'] for r in results))} domains")
    print("=" * 80)

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  EXCEEDANCE                                                        │
  │  Datasets exceeding capacity:      {n_exc}/{len(results)} ({100*n_exc/len(results):.0f}%){' '*24}│
  │  Total pairwise claims:            {tot_pairs:<40}│
  │  Stable (between-group):           {tot_stable} ({100*tot_stable/tot_pairs:.1f}%){' '*24}│
  │  Unstable (within-group):          {tot_pairs-tot_stable} ({100*(tot_pairs-tot_stable)/tot_pairs:.1f}%){' '*23}│
  │                                                                    │
  │  DIRECTIONAL PREDICTION (within > between flip rate)               │
  │  Datasets with measurable pairs:   {n_with_data:<40}│
  │  Correct direction:                {n_dir}/{n_with_data} ({100*n_dir/n_with_data:.0f}%){' '*27}│
  │  Significant (p<0.005):            {n_sig}/{n_with_data} ({100*n_sig/n_with_data:.0f}%){' '*27}│
  │  *** Significant REVERSALS:        {n_sig_rev}/{n_with_data} ***{' '*30}│
  │                                                                    │
  │  EFFECT SIZES                                                      │
  │  Mean within-group flip rate:      {np.mean(all_w):.4f}{' '*31}│
  │  Mean between-group flip rate:     {np.mean(all_b):.4f}{' '*31}│
  │  Median Cohen's d:                 {np.median(all_d):.3f}{' '*32}│
  └────────────────────────────────────────────────────────────────────┘""")

    # Domain breakdown
    print(f"\n  Exceedance by domain (ρ* = {PRIMARY_THR}):")
    for dom in sorted(domain_stats.keys()):
        e, t = domain_stats[dom]
        pct = 100*e/t if t > 0 else 0
        bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
        print(f"    {dom:<24} {e:>2}/{t:<2} ({pct:5.1f}%) {bar}")

    # Sensitivity
    print(f"\n  Exceedance by threshold:")
    for thr in THRESHOLDS:
        tk = f"{thr:.2f}"
        ne = sum(1 for r in results if r["thresholds"][tk]["exceed"])
        pct = 100*ne/len(results)
        m = " ◀ primary" if thr == PRIMARY_THR else ""
        print(f"    ρ*={thr:.2f}  {ne:>3}/{len(results)} ({pct:5.1f}%){m}")

    # Top 15 strongest confirmations
    confirmed = []
    for r in results:
        p = r["thresholds"][T]
        if (p["within_flip"] is not None and p["between_flip"] is not None
                and p["mw_p"] is not None and p["mw_p"] < 0.005
                and p["within_flip"] > p["between_flip"]):
            confirmed.append((r["dataset"], r["domain"], p["cohens_d"] or 0,
                              p["mw_p"], p["within_flip"], p["between_flip"]))
    confirmed.sort(key=lambda x: -x[2])

    print(f"\n  Top 15 strongest confirmations (by Cohen's d):")
    for name, dom, d, p, w, b in confirmed[:15]:
        print(f"    {name[:28]:<29} {dom[:16]:<17} d={d:.2f}  w={w:.3f}  "
              f"b={b:.3f}  p={p:.1e}")

    # Save
    out_path = OUT / "results_capacity_audit_150.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
