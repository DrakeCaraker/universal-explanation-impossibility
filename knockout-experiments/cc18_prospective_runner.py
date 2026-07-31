#!/usr/bin/env python
"""
FROZEN — OSF Prospective Round 3: certificate validation on OpenML-CC18.

*** DO NOT RUN BEFORE OSF REGISTRATION. ***
The OSF registration cites the commit hash at which this file was frozen.
Protocol: (1) review + freeze this script; (2) commit to the public repo under
knockout-experiments/; (3) cite that commit hash in the OSF registration text
(see OSF_PREREGISTRATION_DRAFT.md); (4) wait >= 48 h; (5) run exactly once.
Any post-registration change requires a public commit + disclosure + full restart.

Pipeline is byte-identical in logic to large_scale_certificate_validation.py
(PMLB retrospective study, main repo @ b37e1fe): M=60 RandomForests
(n_estimators=50, max_depth=6), impurity importances, 30/30 A/B split,
SNR bands STABLE>=2.0 / MARGINAL>=0.5, predicted flip Phi(-SNR_A),
observed flip on B, Cantelli bound 1/(1+SNR^2). Only the data source
(OpenML-CC18 instead of PMLB), NA-row handling, ordinal encoding of
categoricals, and the PMLB-deduplication step are new — each pre-specified in
the registration. Emits a P1–P5 pass/fail table against the pre-registered
predictions; the report ships whatever the outcome.
"""
import os, re, json, csv, math, hashlib, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm, spearmanr

SCRATCH = os.environ.get("CERT_SCRATCH", os.path.expanduser("~/.cache/cc18_prospective"))
os.makedirs(SCRATCH, exist_ok=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cc18_prospective.json")

M = 60                    # ensemble size; split 30/30 (A predicts, B observes)
HALF = M // 2
N_MIN, N_MAX = 200, 5000  # primary envelope (pre-registered fallback: n<=10000, p<=50
P_MIN, P_MAX = 4, 30      #   if <15 datasets survive; results then labeled 'widened')
STABLE, MARGINAL = 2.0, 0.5
EPS = 1e-12
MASTER_SEED = 12345       # identical to the PMLB study
CC18_SUITE_ID = 99        # OpenML-CC18 curated suite

# ---- Pre-registered predictions (thresholds frozen at registration) ----
PREREG = {
    "P1_stable_transfer_per_dataset_min": 0.95,   # in >=90% of datasets w/ >=5 STABLE pairs
    "P1_dataset_frac_min": 0.90,
    "P2_pooled_stable_meanB_max": 0.05,
    "P2_worst_dataset_stable_meanB_max": 0.20,
    "P3_oos_spearman_median_min": 0.60,
    "P3_beats_corr_frac_min": 0.85,
    "P4_pooled_ECE_max": 0.05,
    "P5_topk_certified_repro_min": 0.90,          # underpowered if <10 STABLE boundaries
    "P5_topk_uncertified_repro_max": 0.60,
    "P5_min_certified_instances": 10,
    "TOPK_K": 5,
}


def normalize_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def canonical_matrix_hash(X):
    """Order-invariant content hash: columns sorted by (mean, std), rows
    lexicographically sorted, values rounded to 6 decimals."""
    Xr = np.round(np.asarray(X, dtype=float), 6)
    col_key = list(zip(np.nanmean(Xr, 0).tolist(), np.nanstd(Xr, 0).tolist()))
    order = sorted(range(Xr.shape[1]), key=lambda j: col_key[j])
    Xc = Xr[:, order]
    Xc = Xc[np.lexsort(Xc.T[::-1])]
    return hashlib.sha256(Xc.tobytes()).hexdigest()


def pmlb_exclusion_sets():
    """Names of ALL PMLB datasets + canonical hashes of the 136 previously
    selected ones (same filters). PMLB access here is not 'peeking' — PMLB is
    the already-published retrospective source; CC-18 data is only touched
    inside run_dataset, after registration."""
    import pmlb
    from pmlb import fetch_data
    stats = os.path.join(os.path.dirname(pmlb.__file__), "all_summary_stats.tsv")
    with open(stats) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = {normalize_name(r["dataset"]) for r in rows}
    hashes = set()
    cache = os.path.join(SCRATCH, "pmlb_cache")
    os.makedirs(cache, exist_ok=True)
    for r in rows:
        n, k = int(r["n_instances"]), int(r["n_features"])
        if N_MIN <= n <= N_MAX and P_MIN <= k <= P_MAX:
            try:
                df = fetch_data(r["dataset"], local_cache_dir=cache)
                hashes.add(canonical_matrix_hash(df.drop(columns=["target"]).to_numpy(dtype=float)))
            except Exception:
                pass  # unfetchable PMLB dataset cannot collide; name check still applies
    return names, hashes


def select_cc18():
    import openml
    suite = openml.study.get_suite(CC18_SUITE_ID)
    out = []
    for tid in suite.tasks:
        try:
            task = openml.tasks.get_task(tid, download_data=False,
                                         download_qualities=False)
            ds = openml.datasets.get_dataset(task.dataset_id, download_data=False,
                                             download_qualities=False)
            out.append((tid, task.dataset_id, ds.name))
        except Exception as e:
            out.append((tid, None, f"metadata_error:{type(e).__name__}"))
    return out


def load_cc18_dataset(dataset_id):
    """Load, drop NA rows, ordinal-encode categoricals, return X (float), y."""
    import openml
    ds = openml.datasets.get_dataset(dataset_id)
    X, y, cat, _ = ds.get_data(target=ds.default_target_attribute,
                               dataset_format="dataframe")
    keep = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[keep], y[keep]
    for c, is_cat in zip(X.columns, cat):
        if is_cat or X[c].dtype.name in ("category", "object"):
            X[c] = X[c].astype("category").cat.codes.astype(float)
    return X.to_numpy(dtype=float), y.astype("category").cat.codes.to_numpy()


def rf_ensemble(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    imps = []
    for _ in range(M):
        idx = rng.integers(0, n, n)
        m = RandomForestClassifier(n_estimators=50, max_depth=6,
                                   random_state=int(rng.integers(1e9)), n_jobs=1)
        m.fit(X[idx], y[idx])
        imps.append(m.feature_importances_)
    return np.asarray(imps)  # M x p


def run_dataset(job):
    tid, did, name, pmlb_names, pmlb_hashes = job
    try:
        if did is None:
            return {"dataset": name, "task_id": tid, "error": name}
        if normalize_name(name) in pmlb_names:
            return {"dataset": name, "task_id": tid, "skipped": "dedup_name_matches_pmlb"}
        X, y = load_cc18_dataset(did)
        n, p = X.shape
        if not (N_MIN <= n <= N_MAX and P_MIN <= p <= P_MAX):
            return {"dataset": name, "task_id": tid, "skipped": f"size n={n} p={p}"}
        if canonical_matrix_hash(X) in pmlb_hashes:
            return {"dataset": name, "task_id": tid, "skipped": "dedup_content_matches_pmlb"}
        sd = X.std(0)
        X = X[:, sd > EPS]
        p = X.shape[1]
        if p < P_MIN:
            return {"dataset": name, "task_id": tid, "skipped": "too_few_nonconstant_features"}
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        _, cnt = np.unique(y, return_counts=True)
        if len(cnt) < 2 or cnt.min() < 5:
            return {"dataset": name, "task_id": tid, "skipped": "degenerate_target"}

        imps = rf_ensemble(X, y, seed=MASTER_SEED)
        A, B, full = imps[:HALF], imps[HALF:], imps
        corr = np.corrcoef(X, rowvar=False)
        single = full[0]

        rows = []
        for j in range(p):
            for k in range(j + 1, p):
                dA, dB, dF = A[:, j] - A[:, k], B[:, j] - B[:, k], full[:, j] - full[:, k]
                muA, sdA = dA.mean(), dA.std() + EPS
                if muA == 0:
                    continue
                snrA = abs(muA) / sdA
                pred_gauss = float(norm.cdf(-snrA))
                obs_B = float(np.mean(np.sign(dB) != np.sign(muA)))
                muF, sdF = dF.mean(), dF.std() + EPS
                snrF = abs(muF) / sdF
                obs_F = float(np.mean(np.sign(dF) != np.sign(muF)))
                cant_F = 1.0 / (1.0 + snrF ** 2)
                band = "STABLE" if snrF >= STABLE else ("MARGINAL" if snrF >= MARGINAL else "UNRELIABLE")
                bandA = "STABLE" if snrA >= STABLE else ("MARGINAL" if snrA >= MARGINAL else "UNRELIABLE")
                rows.append((pred_gauss, obs_B, snrF, obs_F, cant_F, band, bandA,
                             abs(corr[j, k]), abs(single[j] - single[k])))
        if len(rows) < 5:
            return {"dataset": name, "task_id": tid, "skipped": "too_few_pairs"}
        rows = np.array(rows, dtype=object)
        pred, obsB = rows[:, 0].astype(float), rows[:, 1].astype(float)
        obsF, cantF = rows[:, 3].astype(float), rows[:, 4].astype(float)
        band, bandA = rows[:, 5], rows[:, 6]
        acorr, gap = rows[:, 7].astype(float), rows[:, 8].astype(float)

        def sp(a, b):
            if np.std(a) < EPS or np.std(b) < EPS:
                return float("nan")
            return float(spearmanr(a, b).correlation)

        oos_cert, oos_corr, oos_gap = sp(pred, obsB), sp(acorr, obsB), sp(-gap, obsB)
        stA = bandA == "STABLE"
        stable_transfer = float(np.mean(obsB[stA] <= 0.20)) if stA.any() else float("nan")
        stable_meanB = float(obsB[stA].mean()) if stA.any() else float("nan")
        unrel_A = bandA == "UNRELIABLE"
        unrel_meanB = float(obsB[unrel_A].mean()) if unrel_A.any() else float("nan")

        # Certified top-k (P5): boundary STABLE on A <=> the pair (rank k vs k+1
        # by A-mean importance) has SNR_A >= 2. Reproduction = top-k sets by
        # A-mean and B-mean importances are identical.
        K = PREREG["TOPK_K"]
        topk = None
        if p > K:
            mA, mB = A.mean(0), B.mean(0)
            oA = np.argsort(-mA)
            dbound = A[:, oA[K - 1]] - A[:, oA[K]]
            snr_bound = abs(dbound.mean()) / (dbound.std() + EPS)
            same = set(oA[:K].tolist()) == set(np.argsort(-mB)[:K].tolist())
            topk = {"boundary_snrA": float(snr_bound),
                    "certified": bool(snr_bound >= STABLE), "reproduced": bool(same)}

        def band_stat(bn):
            msk = band == bn
            return [int(msk.sum()), float(obsF[msk].mean()) if msk.any() else float("nan"),
                    float(obsF[msk].max()) if msk.any() else float("nan")]
        cant_hold = float(np.mean(obsF <= cantF + 1e-9))

        return {
            "dataset": name, "task_id": tid, "n": int(n), "p": int(p), "n_pairs": int(len(rows)),
            "oos_spearman_cert": oos_cert, "oos_spearman_corr": oos_corr, "oos_spearman_gap": oos_gap,
            "stable_transfer_frac": stable_transfer, "stable_meanB": stable_meanB,
            "unreliable_meanB": unrel_meanB, "n_stableA": int(stA.sum()),
            "band_STABLE": band_stat("STABLE"), "band_MARGINAL": band_stat("MARGINAL"),
            "band_UNRELIABLE": band_stat("UNRELIABLE"), "cantelli_hold_frac": cant_hold,
            "topk": topk,
            "_calib_pred": pred.tolist(), "_calib_obs": obsB.tolist(),
        }
    except Exception as e:
        return {"dataset": name, "task_id": tid, "error": f"{type(e).__name__}: {e}"}


def evaluate_prereg(ok, summary):
    """Mechanical pass/fail against the frozen P1–P5 thresholds."""
    P = PREREG
    verdicts = {}
    elig = [r for r in ok if r["n_stableA"] >= 5 and not math.isnan(r["stable_transfer_frac"])]
    if elig:
        frac = float(np.mean([r["stable_transfer_frac"] >= P["P1_stable_transfer_per_dataset_min"]
                              for r in elig]))
        verdicts["P1"] = {"value": round(frac, 3), "n_eligible": len(elig),
                          "pass": bool(frac >= P["P1_dataset_frac_min"])}
    else:
        verdicts["P1"] = {"pass": None, "note": "no eligible datasets"}
    sm = [r["stable_meanB"] for r in ok if not math.isnan(r.get("stable_meanB", float("nan")))]
    ns = [r["n_stableA"] for r in ok if not math.isnan(r.get("stable_meanB", float("nan")))]
    if sm:
        pooled = float(np.average(sm, weights=ns))
        verdicts["P2"] = {"pooled": round(pooled, 4), "worst": round(max(sm), 4),
                          "pass": bool(pooled <= P["P2_pooled_stable_meanB_max"]
                                       and max(sm) <= P["P2_worst_dataset_stable_meanB_max"])}
    else:
        verdicts["P2"] = {"pass": None, "note": "no STABLE calls"}
    med = summary["OOS_spearman_certificate"].get("median")
    beat = summary["cert_beats_corr_frac"]
    verdicts["P3"] = {"median": med, "beats_corr_frac": beat,
                      "pass": (None if med is None or beat is None
                               else bool(med >= P["P3_oos_spearman_median_min"]
                                         and beat >= P["P3_beats_corr_frac_min"]))}
    ece = summary["calibration_ECE"]
    verdicts["P4"] = {"ECE": ece,
                      "pass": bool(ece <= P["P4_pooled_ECE_max"]) if ok else None}
    tk = [r["topk"] for r in ok if r.get("topk")]
    cert = [t for t in tk if t["certified"]]
    unc = [t for t in tk if not t["certified"]]
    if len(cert) >= P["P5_min_certified_instances"]:
        cr = float(np.mean([t["reproduced"] for t in cert]))
        ur = float(np.mean([t["reproduced"] for t in unc])) if unc else float("nan")
        verdicts["P5"] = {"certified_repro": round(cr, 3), "n_certified": len(cert),
                          "uncertified_repro": (None if math.isnan(ur) else round(ur, 3)),
                          "pass": bool(cr >= P["P5_topk_certified_repro_min"]
                                       and (math.isnan(ur) or ur <= P["P5_topk_uncertified_repro_max"]))}
    else:
        verdicts["P5"] = {"pass": None, "n_certified": len(cert),
                          "note": "underpowered (<10 certified boundaries); reported, not scored"}
    return verdicts


def aggregate(results):
    ok = [r for r in results if "oos_spearman_cert" in r]
    sk = [r for r in results if "skipped" in r]
    er = [r for r in results if "error" in r]

    def dist(key):
        v = np.array([r[key] for r in ok if isinstance(r.get(key), (int, float))
                      and not math.isnan(r.get(key))])
        if len(v) == 0:
            return {}
        return {"median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3),
                "q25": round(float(np.percentile(v, 25)), 3), "q75": round(float(np.percentile(v, 75)), 3),
                "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "n": int(len(v))}

    Pv = np.concatenate([np.array(r["_calib_pred"]) for r in ok]) if ok else np.array([])
    Ob = np.concatenate([np.array(r["_calib_obs"]) for r in ok]) if ok else np.array([])
    bins = np.linspace(0, 0.5, 6)
    calib, ece, tot = [], 0.0, max(len(Pv), 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Pv >= lo) & (Pv < hi)
        if m.sum() > 0:
            pm, om = float(Pv[m].mean()), float(Ob[m].mean())
            calib.append([round(pm, 3), round(om, 3), int(m.sum())])
            ece += (m.sum() / tot) * abs(pm - om)

    beat_corr = np.mean([r["oos_spearman_cert"] > r["oos_spearman_corr"] for r in ok
                         if not math.isnan(r["oos_spearman_cert"])
                         and not math.isnan(r["oos_spearman_corr"])]) if ok else float("nan")

    summary = {
        "n_tasks": len(results), "n_ok": len(ok), "n_skipped": len(sk), "n_error": len(er),
        "n_dedup_excluded": len([r for r in sk if str(r.get("skipped", "")).startswith("dedup")]),
        "total_pairs": int(sum(r["n_pairs"] for r in ok)),
        "OOS_spearman_certificate": dist("oos_spearman_cert"),
        "OOS_spearman_baseline_corr": dist("oos_spearman_corr"),
        "OOS_spearman_baseline_singlemodel_gap": dist("oos_spearman_gap"),
        "cert_beats_corr_frac": round(float(beat_corr), 3) if not math.isnan(beat_corr) else None,
        "STABLE_transfer_frac_per_dataset": dist("stable_transfer_frac"),
        "STABLE_meanB_flip_per_dataset": dist("stable_meanB"),
        "UNRELIABLE_meanB_flip_per_dataset": dist("unreliable_meanB"),
        "calibration_OOS_bins_[pred,obs,n]": calib, "calibration_ECE": round(ece, 4),
        "cantelli_hold_frac_insample_pooled":
            round(float(np.mean([r["cantelli_hold_frac"] for r in ok])), 4) if ok else None,
        "skipped": [{k: r[k] for k in ("dataset", "skipped")} for r in sk],
        "errors": [{k: r[k] for k in ("dataset", "error")} for r in er],
    }
    summary["PREREG_VERDICTS"] = evaluate_prereg(ok, summary)
    summary["PREREG_THRESHOLDS"] = PREREG
    return summary


def main():
    print("computing PMLB exclusion sets (names + content hashes)...")
    pmlb_names, pmlb_hashes = pmlb_exclusion_sets()
    print(f"  {len(pmlb_names)} PMLB names, {len(pmlb_hashes)} content hashes")
    sel = select_cc18()
    print(f"CC-18 suite: {len(sel)} tasks")
    jobs = [(tid, did, name, pmlb_names, pmlb_hashes) for tid, did, name in sel]
    workers = max(1, (os.cpu_count() or 4) - 2)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_dataset, j): j[2] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            results.append(fu.result())
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done")
    summary = aggregate(results)
    detail = [{k: v for k, v in r.items() if not k.startswith("_calib")} for r in results]
    json.dump({"summary": summary, "detail": detail}, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
