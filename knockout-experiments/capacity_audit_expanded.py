"""
Explanation Capacity Audit — Expanded Cross-Domain Study
==========================================================
Comprehensive audit across 25+ datasets spanning 10+ scientific domains.
Tests whether standard ML explanation practices exceed the explanation capacity.

Methodology (pre-registered in explanation-capacity-audit-preregistration.md):
- M=50 XGBoost models per dataset, subsample=0.8, colsample_bytree=0.5
- TreeSHAP importance (mean |SHAP|)
- SAGE group identification (hierarchical clustering on |Spearman ρ|)
- Primary threshold ρ*=0.70, sensitivity at {0.50, 0.60, 0.80, 0.90}
- Falsification: within-group vs between-group flip rates, Mann-Whitney U

Domains covered:
  Genomics, Clinical/Medical, Finance/Economics, Environmental/Climate,
  Food Science, Computer Science, Manufacturing, Remote Sensing,
  Social Science, Drug Discovery, Synthetic Controls
"""

import json, time, sys, traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
import shap

# ═══════════════════════════════════════════════════════════════════════
# Configuration (pre-registered, do not modify after data collection)
# ═══════════════════════════════════════════════════════════════════════
M = 50
PRIMARY_THR = 0.70
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MAX_SHAP = 200          # max test samples for SHAP
MAX_SAMPLES = 3000      # subsample large datasets
MIN_FEATURES = 8        # exclude datasets with < 8 features
TOP_K_FEATURES = 50     # for high-dim datasets, use top-k by variance

XGB_CLF = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0, eval_metric='logloss')
XGB_REG = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0)

OUT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════════

def _prep(X, y, task, top_k=None):
    """Standardise a dataset: drop NaN, encode labels, subsample, top-k."""
    X = np.array(X, dtype=float)
    y = np.array(y)

    # Drop columns that are all NaN
    good_cols = ~np.all(np.isnan(X), axis=0)
    X = X[:, good_cols]

    # Impute remaining NaN with column median
    for c in range(X.shape[1]):
        mask = np.isnan(X[:, c])
        if mask.any():
            X[mask, c] = np.nanmedian(X[:, c])

    # Drop zero-variance columns
    var = np.var(X, axis=0)
    nz = var > 1e-12
    X = X[:, nz]

    # Top-k by variance for high-dimensional data
    if top_k and X.shape[1] > top_k:
        idx = np.argsort(np.var(X, axis=0))[-top_k:]
        X = X[:, idx]

    # Encode target
    if task == "clf":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        # Convert multiclass to binary (largest class vs rest)
        if len(np.unique(y)) > 2:
            mode = np.bincount(y).argmax()
            y = (y == mode).astype(int)

    # Subsample if too large
    if X.shape[0] > MAX_SAMPLES:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    return X, y


def load_all():
    """Load datasets from sklearn, OpenML, and synthetic generators."""
    from sklearn.datasets import (
        load_breast_cancer, load_diabetes, load_wine,
        fetch_california_housing, load_digits,
    )
    ds = {}

    # ─── SKLEARN BUILT-INS ────────────────────────────────────────
    def add_sklearn(name, loader, task, domain):
        try:
            d = loader()
            X, y = _prep(d.data, d.target, task)
            if X.shape[1] >= MIN_FEATURES:
                names = (list(d.feature_names) if hasattr(d, 'feature_names')
                         else [f"f{i}" for i in range(X.shape[1])])
                ds[name] = dict(X=X, y=y, task=task, domain=domain,
                                names=names[:X.shape[1]])
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    add_sklearn("Breast_Cancer", load_breast_cancer, "clf", "Clinical")
    add_sklearn("California_Housing", fetch_california_housing, "reg", "Economics")
    add_sklearn("Wine", load_wine, "clf", "Food_Science")
    add_sklearn("Diabetes", load_diabetes, "reg", "Clinical")
    add_sklearn("Digits", load_digits, "clf", "Computer_Vision")

    # ─── OPENML DATASETS ─────────────────────────────────────────
    openml_registry = [
        # (name, data_id, task, domain, top_k)
        # Genomics
        ("AP_Colon_Kidney",      1137, "clf", "Genomics",         50),
        ("AP_Endometrium_Colon", 1138, "clf", "Genomics",         50),
        ("AP_Breast_Colon",      1166, "clf", "Genomics",         50),
        ("AP_Breast_Lung",       1167, "clf", "Genomics",         50),
        # Clinical
        ("Heart_Disease",          53, "clf", "Clinical",        None),
        ("Pima_Diabetes",         37,  "clf", "Clinical",        None),
        # Finance / Economics
        ("German_Credit",         31,  "clf", "Finance",         None),
        ("Adult_Income",        1590,  "clf", "Social_Science",  None),
        ("Bank_Marketing",      1461,  "clf", "Economics",       None),
        # Computer Science
        ("Spambase",              44,  "clf", "CS_NLP",            50),
        ("KC2_SoftwareDefect",  1063,  "clf", "Software_Eng",    None),
        # Manufacturing
        ("Steel_Plates_Fault",  40982, "clf", "Manufacturing",   None),
        # Remote Sensing
        ("Satellite_Image",       182, "clf", "Remote_Sensing",  None),
        ("Segment",             40984, "clf", "Remote_Sensing",  None),
        # Environmental
        ("Climate_Crashes",     40994, "clf", "Climate",         None),
        # Drug Discovery
        ("Bioresponse",          4134, "clf", "Drug_Discovery",    50),
        # Additional clinical
        ("SPECTF_Heart",          337, "clf", "Clinical",        None),
        # Physics
        ("Ionosphere",            59,  "clf", "Physics",         None),
        # Biodegradation
        ("QSAR_Biodeg",         1494,  "clf", "Chemistry",       None),
        # ── Wave 2: 29 additional datasets across 18 new domains ──
        # Neuroscience
        ("MiceProtein",         40966, "clf", "Neuroscience",      50),
        # Environmental
        ("Ozone_Level",          1487, "clf", "Environmental",     50),
        # Software Engineering (additional)
        ("PC4_Software",         1049, "clf", "Software_Eng",    None),
        ("JM1_Software",         1053, "clf", "Software_Eng",    None),
        # Handwriting / Pattern Recognition
        ("Handwriting_Factors",    12, "clf", "Pattern_Recog",     50),
        ("Handwriting_Fourier",    14, "clf", "Pattern_Recog",     50),
        # Vehicle Engineering
        ("Vehicle_Silhouette",     54, "clf", "Vehicle_Eng",     None),
        # Forensics
        ("Glass_Forensic",         41, "clf", "Forensics",       None),
        # Geophysics
        ("Sonar",                  40, "clf", "Geophysics",      None),
        # Computer Vision (additional)
        ("OptDigits",              28, "clf", "Computer_Vision",   50),
        ("PenDigits",              32, "clf", "Computer_Vision",  None),
        # Materials Science
        ("Texture",             40499, "clf", "Materials",       None),
        # Wearables / Human Activity
        ("HAR_Activity",         1478, "clf", "Wearables",         50),
        # Text Mining
        ("CNAE9_Text",           1468, "clf", "Text_Mining",       50),
        # Human-Computer Interaction
        ("GesturePhase",         4538, "clf", "HCI",             None),
        # Clinical (additional)
        ("WDBC",                 1510, "clf", "Clinical",        None),
        ("WPBC_Prognosis",       1511, "clf", "Clinical",        None),
        # Neurology
        ("Parkinsons",           1488, "clf", "Neurology",       None),
        # Dermatology
        ("Dermatology",            35, "clf", "Dermatology",     None),
        # Veterinary
        ("Horse_Colic",           294, "clf", "Veterinary",      None),
        # Microbiology
        ("Yeast_Protein",         181, "clf", "Microbiology",    None),
        # Cardiology
        ("Arrhythmia",              5, "clf", "Cardiology",        50),
        # Synthetic ML Benchmarks
        ("Madelon",              1485, "clf", "Synthetic_ML",      50),
        ("GINA_Agnostic",        1038, "clf", "Synthetic_ML",      50),
        ("Christine_AutoML",    41142, "clf", "AutoML",            50),
        # Energy
        ("Electricity",           151, "clf", "Energy",          None),
        # Food Science (spectroscopy)
        ("Tecator_Meat",          505, "reg", "Food_Spectro",      50),
        # Stylometry / Authorship
        ("Authorship",            458, "clf", "Stylometry",        50),
        # NLP
        ("Collins_NLP",         40971, "clf", "NLP",             None),
        # ── Wave 3: additional domains ──
        # Oncology (gene expression)
        ("Leukemia",             1104, "clf", "Oncology",          50),
        # Census
        ("Microaggregation",      806, "clf", "Census",            50),
        # Astrophysics
        ("MAGIC_Gamma",          1120, "clf", "Astrophysics",    None),
        # Particle Physics
        ("MiniBooNE",           41150, "clf", "Particle_Physics",  50),
        # Geography / Housing
        ("House_16H",             574, "reg", "Geography",       None),
        # Software Eng (additional)
        ("PC3_Software",         1050, "clf", "Software_Eng",    None),
        ("MC1_Software",         1056, "clf", "Software_Eng",    None),
        # Robotics
        ("WallRobot_Nav",        1497, "clf", "Robotics",        None),
    ]

    try:
        from sklearn.datasets import fetch_openml
        for name, did, task, domain, top_k in openml_registry:
            try:
                d = fetch_openml(data_id=did, as_frame=False, parser='auto')
                X, y = _prep(d.data, d.target, task, top_k=top_k)
                if X.shape[1] >= MIN_FEATURES:
                    ds[name] = dict(
                        X=X, y=y, task=task, domain=domain,
                        names=[f"f{i}" for i in range(X.shape[1])])
                    print(f"  ✓ {name} (P={X.shape[1]}, N={X.shape[0]})")
                else:
                    print(f"  [skip] {name}: only {X.shape[1]} features after prep")
            except Exception as e:
                print(f"  [skip] {name}: {e}")
    except ImportError:
        print("  [warn] fetch_openml unavailable; skipping OpenML datasets")

    # ─── SYNTHETIC CONTROLS ───────────────────────────────────────
    def make_corr_groups(n_groups, group_size, rho, seed, beta_levels):
        """Generate features with exact within-group correlation and equal betas."""
        rng = np.random.default_rng(seed)
        P = n_groups * group_size
        betas = np.concatenate([np.full(group_size, b) for b in beta_levels])
        Sigma = np.zeros((P, P))
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            Sigma[sl, sl] = rho
        np.fill_diagonal(Sigma, 1.0)
        L = np.linalg.cholesky(Sigma)
        X = rng.standard_normal((500, P)) @ L.T
        score = X @ betas + rng.normal(0, 1.0, 500)
        y = (score > np.median(score)).astype(int)
        return X, y

    # Positive control: 3 groups × 4, ρ=0.99, strong signal separation
    X_s, y_s = make_corr_groups(3, 4, 0.99, 42, [5.0, 2.0, 0.5])
    ds["Synth_3g_rho99"] = dict(
        X=X_s, y=y_s, task="clf", domain="Synthetic",
        names=[f"g{g}_f{f}" for g in range(3) for f in range(4)])

    # Moderate control: 4 groups × 3, ρ=0.70
    X_m, y_m = make_corr_groups(4, 3, 0.70, 44, [4.0, 2.0, 1.0, 0.3])
    ds["Synth_4g_rho70"] = dict(
        X=X_m, y=y_m, task="clf", domain="Synthetic",
        names=[f"g{g}_f{f}" for g in range(4) for f in range(3)])

    # Weak control: 5 groups × 3, ρ=0.50
    X_w, y_w = make_corr_groups(5, 3, 0.50, 45, [3.0, 2.0, 1.5, 1.0, 0.5])
    ds["Synth_5g_rho50"] = dict(
        X=X_w, y=y_w, task="clf", domain="Synthetic",
        names=[f"g{g}_f{f}" for g in range(5) for f in range(3)])

    # Negative control: 15 independent features
    rng_n = np.random.default_rng(43)
    X_n = rng_n.standard_normal((500, 15))
    b_n = np.array([3, -2, 1.5, -1, 0.8, -0.6, 0.4, -0.2, 0.1, 0, 0, 0, 0, 0, 0])
    y_n = (X_n @ b_n + rng_n.normal(0, 1, 500) > 0).astype(int)
    ds["Synth_Independent"] = dict(
        X=X_n, y=y_n, task="clf", domain="Synthetic",
        names=[f"indep_{i}" for i in range(15)])

    # Large-group control: 2 groups × 10, ρ=0.95
    X_lg, y_lg = make_corr_groups(2, 10, 0.95, 46, [3.0, 1.0])
    ds["Synth_2g_large_rho95"] = dict(
        X=X_lg, y=y_lg, task="clf", domain="Synthetic",
        names=[f"g{g}_f{f}" for g in range(2) for f in range(10)])

    return ds


# ═══════════════════════════════════════════════════════════════════════
# SAGE + Audit Core
# ═══════════════════════════════════════════════════════════════════════

def sage_groups(X, threshold):
    rho = spearmanr(X).statistic
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, abs(float(rho))], [abs(float(rho)), 1.0]])
    rho = np.abs(rho)
    np.fill_diagonal(rho, 1.0)
    rho = np.nan_to_num(rho, nan=0.0)
    rho = np.clip(rho, 0, 1)
    dist = 1 - rho
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method='average')
    return fcluster(Z, t=1 - threshold, criterion='distance')


def audit_one(name, data):
    X, y, task = data["X"], data["y"], data["task"]
    P = X.shape[1]
    t0 = time.time()

    # ── Train M models ──
    imps = np.zeros((M, P))
    accs = []
    for s in range(M):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=s, stratify=y if task == "clf" else None)
        if task == "clf":
            mdl = XGBClassifier(**XGB_CLF, random_state=s)
        else:
            mdl = XGBRegressor(**XGB_REG, random_state=s)
        mdl.fit(Xtr, ytr)
        accs.append(float(mdl.score(Xte, yte)))
        ex = shap.TreeExplainer(mdl)
        sv = ex.shap_values(Xte[:MAX_SHAP])
        if isinstance(sv, list):
            sv = sv[1]
        imps[s] = np.mean(np.abs(sv), axis=0)

    # ── Inter-model Spearman ρ ──
    rho_pairs = []
    for i in range(0, M, 2):  # sample pairs for speed
        for j in range(i + 1, min(i + 5, M)):
            r, _ = spearmanr(imps[i], imps[j])
            if np.isfinite(r):
                rho_pairs.append(r)
    mean_rho = float(np.mean(rho_pairs)) if rho_pairs else float('nan')

    # ── Pairwise flip rates ──
    flips = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            wins = np.sum(imps[:, i] > imps[:, j])
            flips[i, j] = flips[j, i] = min(wins, M - wins) / M

    # ── SAGE + metrics per threshold ──
    thr_res = {}
    for thr in THRESHOLDS:
        grp = sage_groups(X, thr)
        g = int(len(np.unique(grp)))

        between, within = [], []
        for i in range(P):
            for j in range(i + 1, P):
                (within if grp[i] == grp[j] else between).append(flips[i, j])

        if len(within) >= 2 and len(between) >= 2:
            _, pval = mannwhitneyu(within, between, alternative='greater')
            pval = float(pval)
            # Cohen's d
            pooled_std = np.sqrt((np.var(within) + np.var(between)) / 2)
            cohens_d = ((np.mean(within) - np.mean(between)) / pooled_std
                        if pooled_std > 0 else 0)
        else:
            pval, cohens_d = None, None

        total_p = P * (P - 1) // 2
        stable_p = g * (g - 1) // 2

        thr_res[f"{thr:.2f}"] = dict(
            g=g, C=g, P=P,
            exceed=bool(P > g),
            exc_ratio=round((P - g) / g, 2) if g > 0 else None,
            stable=stable_p, total=total_p,
            pct_unstable=round(100 * (total_p - stable_p) / total_p, 1)
                if total_p > 0 else 0,
            within_flip=round(float(np.mean(within)), 4) if within else None,
            between_flip=round(float(np.mean(between)), 4) if between else None,
            n_w=len(within), n_b=len(between),
            mw_p=pval, cohens_d=round(cohens_d, 3) if cohens_d is not None else None,
            sizes=sorted([int(np.sum(grp == gid)) for gid in np.unique(grp)],
                         reverse=True),
        )

    return dict(
        dataset=name, domain=data["domain"],
        N=int(X.shape[0]), P=P, M=M, task=task,
        acc_mean=round(float(np.mean(accs)), 4),
        acc_range=[round(float(min(accs)), 4), round(float(max(accs)), 4)],
        rho_mean=round(mean_rho, 4),
        elapsed=round(time.time() - t0, 1),
        thresholds=thr_res,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  EXPLANATION CAPACITY AUDIT — EXPANDED CROSS-DOMAIN STUDY")
    print("  Pre-registered: ρ*=0.70 | M=50 | XGBoost(sub=0.8, col=0.5) | TreeSHAP")
    print("=" * 80 + "\n")

    datasets = load_all()
    N_ds = len(datasets)
    print(f"\n  {N_ds} datasets loaded across {len(set(d['domain'] for d in datasets.values()))} domains\n")
    print("-" * 80)

    results = []
    for i, (name, data) in enumerate(datasets.items(), 1):
        P = data["X"].shape[1]
        print(f"  [{i}/{N_ds}] {name:<28} P={P:<3} N={data['X'].shape[0]:<5} "
              f"{data['domain']:<18} ", end="", flush=True)
        try:
            r = audit_one(name, data)
            results.append(r)
            p = r["thresholds"][f"{PRIMARY_THR:.2f}"]
            mw = p['mw_p']
            p_str = ('<1e-15' if mw is not None and mw < 1e-15
                     else (f'{mw:.1e}' if mw is not None else 'N/A'))
            wf = p['within_flip'] or 0
            bf = p['between_flip'] or 0
            print(f"g={p['g']:<3} C={p['C']:<3} "
                  f"w={wf:.3f} b={bf:.3f} p={p_str:<9} "
                  f"{r['elapsed']:.0f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

        # Save intermediate results
        if i % 5 == 0:
            with open(OUT / "results_audit_expanded_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════════
    T = f"{PRIMARY_THR:.2f}"

    print("\n" + "=" * 80)
    print(f"  AGGREGATE RESULTS  (ρ* = {PRIMARY_THR}, {len(results)} datasets)")
    print("=" * 80)

    # ── Main table ──
    hdr = (f"  {'Dataset':<28} {'Domain':<18} {'P':>3} {'g':>3} {'C':>3} "
           f"{'Exc':>4} {'%Unst':>6} {'Within':>7} {'Betw':>7} {'d':>6} {'p':>10}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    n_exc, n_dir, n_sig = 0, 0, 0
    all_w, all_b, all_d = [], [], []
    total_stable, total_pairs = 0, 0
    domain_exc = {}

    for r in results:
        p = r["thresholds"][T]
        n_exc += int(p["exceed"])
        total_stable += p["stable"]
        total_pairs += p["total"]

        dom = r["domain"]
        domain_exc.setdefault(dom, [0, 0])
        domain_exc[dom][1] += 1
        if p["exceed"]:
            domain_exc[dom][0] += 1

        w_s = f"{p['within_flip']:.3f}" if p['within_flip'] is not None else "  —  "
        b_s = f"{p['between_flip']:.3f}" if p['between_flip'] is not None else "  —  "
        d_s = f"{p['cohens_d']:.2f}" if p['cohens_d'] is not None else "  — "
        p_s = (f"{p['mw_p']:.1e}" if p['mw_p'] is not None and p['mw_p'] >= 1e-15
               else ("<1e-15" if p['mw_p'] is not None else "    —"))
        exc_s = "YES" if p["exceed"] else " no"

        if p["within_flip"] is not None and p["between_flip"] is not None:
            all_w.append(p["within_flip"])
            all_b.append(p["between_flip"])
            if p["within_flip"] > p["between_flip"]:
                n_dir += 1
            if p["mw_p"] is not None and p["mw_p"] < 0.005:
                n_sig += 1
            if p["cohens_d"] is not None:
                all_d.append(p["cohens_d"])

        print(f"  {r['dataset']:<28} {r['domain']:<18} {r['P']:>3} "
              f"{p['g']:>3} {p['C']:>3} {exc_s:>4} {p['pct_unstable']:>5.1f}% "
              f"{w_s:>7} {b_s:>7} {d_s:>6} {p_s:>10}")

    print("  " + "-" * (len(hdr) - 2))

    # ── Summary ──
    n_with_pairs = len(all_w)
    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  EXCEEDANCE                                                  │")
    print(f"  │  Datasets exceeding capacity:   {n_exc}/{len(results)}"
          f" ({100*n_exc/len(results):.0f}%){' ' * 25}│")
    print(f"  │  Total pairwise claims:         {total_pairs:<30}│")
    print(f"  │  Stable (between-group):        {total_stable} "
          f"({100*total_stable/total_pairs:.1f}%){' ' * 22}│")
    print(f"  │  Unstable (within-group):       {total_pairs - total_stable} "
          f"({100*(total_pairs-total_stable)/total_pairs:.1f}%){' ' * 21}│")
    print(f"  │                                                              │")
    print(f"  │  FALSIFICATION TEST                                          │")
    print(f"  │  Mean within-group flip rate:    {np.mean(all_w):.4f}  "
          f"(predicted: ~0.50)       │")
    print(f"  │  Mean between-group flip rate:   {np.mean(all_b):.4f}  "
          f"(predicted: ~0.00)       │")
    print(f"  │  Correct direction (within>between): {n_dir}/{n_with_pairs}"
          f" ({100*n_dir/n_with_pairs:.0f}%){' ' * 15}│")
    print(f"  │  Significant (p<0.005):         {n_sig}/{n_with_pairs}"
          f" ({100*n_sig/n_with_pairs:.0f}%){' ' * 20}│")
    if all_d:
        print(f"  │  Median Cohen's d:              {np.median(all_d):.2f}"
              f"{' ' * 31}│")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    # ── By domain ──
    print(f"\n  Exceedance by domain:")
    for dom in sorted(domain_exc.keys()):
        exc, tot = domain_exc[dom]
        pct = 100 * exc / tot if tot > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {dom:<20} {exc}/{tot} ({pct:5.1f}%) {bar}")

    # ── Sensitivity ──
    print(f"\n  Exceedance prevalence by threshold:")
    for thr in THRESHOLDS:
        tk = f"{thr:.2f}"
        n_e = sum(1 for r in results if r["thresholds"][tk]["exceed"])
        pct = 100 * n_e / len(results)
        marker = " ◀ primary" if thr == PRIMARY_THR else ""
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    ρ*={thr:.2f}  {n_e:>2}/{len(results)} ({pct:5.1f}%) {bar}{marker}")

    # ── Detailed falsification ──
    print(f"\n  Falsification test details (α=0.005, within > between):")
    for r in results:
        p = r["thresholds"][T]
        wf = p.get("within_flip")
        bf = p.get("between_flip")
        mwp = p.get("mw_p")
        cd = p.get("cohens_d") or 0
        if wf is not None and bf is not None:
            sig = mwp is not None and mwp < 0.005
            right_dir = wf > bf
            if sig and right_dir:
                status = "✓ CONFIRMED"
            elif right_dir:
                status = "~ direction"
            else:
                status = "✗ reversed"
            p_s = f"{mwp:.1e}" if mwp is not None else "N/A"
            print(f"    {r['dataset']:<28} w={wf:.3f} "
                  f"b={bf:.3f} d={cd:.2f} p={p_s} {status}")

    # ── Save ──
    out_path = OUT / "results_capacity_audit_expanded.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
