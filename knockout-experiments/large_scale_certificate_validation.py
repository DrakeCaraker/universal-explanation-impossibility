#!/usr/bin/env python
"""
Large-scale validation of the per-query stability certificate (roadmap item #2).

The Cantelli flip bound is now a Lean THEOREM on the finite ensemble
(CertificateGuarantee.lean), so the in-ensemble guarantee `flip <= 1/(1+SNR^2)`
holds 100% by construction -- a sanity check, not a discovery. The scientific
question at scale is OUT-OF-SAMPLE TRANSFER: does a certificate verdict computed
on one independent Rashomon ensemble predict the instability of a *different*
ensemble it was never fit on? We answer this on 100+ real PMLB datasets with a
split-ensemble design.

Design (per dataset):
  - Fit M bootstrap RandomForests (classifier/regressor by task); attribution =
    impurity importance. Split into ensemble A (predict) and B (observe).
  - For every feature pair (j,k), attribution difference D = imp_j - imp_k:
      * On A: SNR_A = |mean_A|/std_A; predicted flip = Phi(-SNR_A) (Gaussian),
        Cantelli bound = min(1/2, 1/(1+SNR_A^2)).
      * On B (out of sample): observed flip = fraction of B whose sign disagrees
        with A's claimed direction sign(mean_A).
      * On the full ensemble (deployment/in-sample): band (STABLE/MARGINAL/
        UNRELIABLE) and the Cantelli/observed check the theorem guarantees.
      * Group-free baselines that predict flip from data alone: |corr(X_j,X_k)|
        and single-model importance-gap -- the paper's <=0.26 from-data ceiling.

Headline metrics (adversarial, full distributions reported):
  1. OOS Spearman(predicted_A, observed_B) per dataset vs baselines.
  2. STABLE transfer: among pairs called STABLE on A, fraction that stay stable
     (observed_B <= 20%) on the independent ensemble B  <-- the real "precision".
  3. Calibration of Phi(-SNR): pooled reliability diagram + ECE (OOS).
  4. In-sample Cantelli hold fraction (theorem sanity, must be ~1.0).
  5. Where it fails: datasets with low OOS Spearman / STABLE over-promise.
"""
import os, json, csv, warnings, math
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm, spearmanr

SCRATCH = "/private/tmp/claude-501/-Users-drakecaraker/cfe4166d-73b6-4488-868f-379c98db298c/scratchpad"
CACHE = os.path.join(SCRATCH, "pmlb_cache")
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_large_scale_certificate.json")

M = 60                    # ensemble size; split 30/30 (A predicts, B observes)
HALF = M // 2
N_MIN, N_MAX = 200, 5000  # dataset row envelope
P_MIN, P_MAX = 4, 30      # feature envelope (keeps all-pairs tractable, no sampling)
STABLE, MARGINAL = 2.0, 0.5
EPS = 1e-12


def select_datasets():
    import pmlb
    p = os.path.join(os.path.dirname(pmlb.__file__), "all_summary_stats.tsv")
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sel = []
    for r in rows:
        n, k = int(r["n_instances"]), int(r["n_features"])
        if N_MIN <= n <= N_MAX and P_MIN <= k <= P_MAX:
            sel.append((r["dataset"], r["task"]))
    return sorted(sel)


def rf_ensemble(X, y, task, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    Mk = (RandomForestClassifier if task == "classification" else RandomForestRegressor)
    imps = []
    for _ in range(M):
        idx = rng.integers(0, n, n)
        m = Mk(n_estimators=50, max_depth=6, random_state=int(rng.integers(1e9)), n_jobs=1)
        m.fit(X[idx], y[idx])
        imps.append(m.feature_importances_)
    return np.asarray(imps)  # M x p


def run_dataset(name_task):
    name, task = name_task
    try:
        from pmlb import fetch_data
        df = fetch_data(name, local_cache_dir=CACHE)
        y = df["target"].to_numpy()
        X = df.drop(columns=["target"]).to_numpy(dtype=float)
        n, p = X.shape
        if not (N_MIN <= n <= N_MAX and P_MIN <= p <= P_MAX):
            return {"dataset": name, "skipped": "size_after_load"}
        # standardize; drop constant columns
        sd = X.std(0)
        keep = sd > EPS
        X = X[:, keep]
        p = X.shape[1]
        if p < P_MIN:
            return {"dataset": name, "skipped": "too_few_nonconstant_features"}
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        if task == "classification":
            # need >=2 classes with enough support for bootstrap
            _, cnt = np.unique(y, return_counts=True)
            if len(cnt) < 2 or cnt.min() < 5:
                return {"dataset": name, "skipped": "degenerate_target"}

        imps = rf_ensemble(X, y, task, seed=12345)
        A, B, full = imps[:HALF], imps[HALF:], imps
        corr = np.corrcoef(X, rowvar=False)
        single = full[0]  # a single model's importances (single-model practice)

        rows = []
        for j in range(p):
            for k in range(j + 1, p):
                dA, dB, dF = A[:, j] - A[:, k], B[:, j] - B[:, k], full[:, j] - full[:, k]
                muA, sdA = dA.mean(), dA.std() + EPS
                if muA == 0:
                    continue
                snrA = abs(muA) / sdA
                pred_gauss = float(norm.cdf(-snrA))
                # OOS observed flip on B: disagreement with A's claimed direction
                obs_B = float(np.mean(np.sign(dB) != np.sign(muA)))
                # in-sample (deployment) quantities on full ensemble
                muF, sdF = dF.mean(), dF.std() + EPS
                snrF = abs(muF) / sdF
                obs_F = float(np.mean(np.sign(dF) != np.sign(muF)))
                # TRUE Cantelli bound (uncapped) = variance/(variance+mean^2) = 1/(1+SNR^2);
                # this is exactly what cantelli_lower_tail proves, so obs_F <= cant_F must hold.
                cant_F = 1.0 / (1.0 + snrF ** 2)
                band = "STABLE" if snrF >= STABLE else ("MARGINAL" if snrF >= MARGINAL else "UNRELIABLE")
                bandA = "STABLE" if snrA >= STABLE else ("MARGINAL" if snrA >= MARGINAL else "UNRELIABLE")
                rows.append((pred_gauss, obs_B, snrF, obs_F, cant_F, band, bandA,
                             abs(corr[j, k]), abs(single[j] - single[k])))
        if len(rows) < 5:
            return {"dataset": name, "skipped": "too_few_pairs"}
        rows = np.array(rows, dtype=object)
        pred = rows[:, 0].astype(float)
        obsB = rows[:, 1].astype(float)
        snrF = rows[:, 2].astype(float)
        obsF = rows[:, 3].astype(float)
        cantF = rows[:, 4].astype(float)
        band = rows[:, 5]
        bandA = rows[:, 6]
        acorr = rows[:, 7].astype(float)
        gap = rows[:, 8].astype(float)

        def sp(a, b):
            if np.std(a) < EPS or np.std(b) < EPS:
                return float("nan")
            return float(spearmanr(a, b).correlation)

        # OOS ranking: certificate vs group-free baselines (predict obs_B)
        oos_cert = sp(pred, obsB)              # higher pred flip -> more flip
        oos_corr = sp(acorr, obsB)             # higher |corr| -> more flip
        oos_gap = sp(-gap, obsB)               # smaller single-model gap -> more flip
        # STABLE transfer (the real precision): STABLE on A -> stays <=20% on B
        stA = bandA == "STABLE"
        stable_transfer = float(np.mean(obsB[stA] <= 0.20)) if stA.any() else float("nan")
        stable_meanB = float(obsB[stA].mean()) if stA.any() else float("nan")
        unrel_A = bandA == "UNRELIABLE"
        unrel_meanB = float(obsB[unrel_A].mean()) if unrel_A.any() else float("nan")
        # in-sample band table (deployment)
        def band_stat(bn):
            msk = band == bn
            return [int(msk.sum()), float(obsF[msk].mean()) if msk.any() else float("nan"),
                    float(obsF[msk].max()) if msk.any() else float("nan")]
        # theorem sanity: in-sample Cantelli holds (uncapped bound); and how often the
        # claim's mean sits on the minority side (obs>0.5) -- those are the pairs a naive
        # 0.5-capped bound would wrongly flag, but the true Cantelli bound still covers.
        cant_hold = float(np.mean(obsF <= cantF + 1e-9))
        mean_minority = float(np.mean(obsF > 0.5))

        return {
            "dataset": name, "task": task, "n": int(n), "p": int(p), "n_pairs": int(len(rows)),
            "oos_spearman_cert": oos_cert, "oos_spearman_corr": oos_corr, "oos_spearman_gap": oos_gap,
            "stable_transfer_frac": stable_transfer, "stable_meanB": stable_meanB,
            "unreliable_meanB": unrel_meanB, "n_stableA": int(stA.sum()),
            "band_STABLE": band_stat("STABLE"), "band_MARGINAL": band_stat("MARGINAL"),
            "band_UNRELIABLE": band_stat("UNRELIABLE"), "cantelli_hold_frac": cant_hold,
            "mean_minority_frac": mean_minority,
            # calibration payload (OOS): predicted (A) vs observed (B)
            "_calib_pred": pred.tolist(), "_calib_obs": obsB.tolist(),
        }
    except Exception as e:
        return {"dataset": name, "error": f"{type(e).__name__}: {e}"}


def aggregate(results):
    ok = [r for r in results if "oos_spearman_cert" in r]
    sk = [r for r in results if "skipped" in r]
    er = [r for r in results if "error" in r]

    def dist(key, arr=None):
        v = np.array([r[key] for r in ok if isinstance(r.get(key), (int, float))
                      and not math.isnan(r.get(key))]) if arr is None else np.array(arr)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return {}
        return {"median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3),
                "q25": round(float(np.percentile(v, 25)), 3), "q75": round(float(np.percentile(v, 75)), 3),
                "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "n": int(len(v))}

    # pooled calibration (OOS)
    P = np.concatenate([np.array(r["_calib_pred"]) for r in ok]) if ok else np.array([])
    Ob = np.concatenate([np.array(r["_calib_obs"]) for r in ok]) if ok else np.array([])
    bins = np.linspace(0, 0.5, 6)
    calib, ece, tot = [], 0.0, len(P)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum() > 0:
            pm, om = float(P[m].mean()), float(Ob[m].mean())
            calib.append([round(pm, 3), round(om, 3), int(m.sum())])
            ece += (m.sum() / tot) * abs(pm - om)

    # pooled band table (in-sample deployment)
    def pooled_band(bn):
        cnt = sum(r["band_" + bn][0] for r in ok)
        wm = sum(r["band_" + bn][0] * r["band_" + bn][1] for r in ok
                 if not math.isnan(r["band_" + bn][1]))
        mx = max((r["band_" + bn][2] for r in ok if not math.isnan(r["band_" + bn][2])), default=float("nan"))
        return {"n_pairs": int(cnt), "mean_flip": round(wm / cnt, 4) if cnt else None,
                "max_flip": round(mx, 4)}

    # certificate beats baseline?
    beat_corr = np.mean([r["oos_spearman_cert"] > r["oos_spearman_corr"]
                         for r in ok if not math.isnan(r["oos_spearman_cert"])
                         and not math.isnan(r["oos_spearman_corr"])])
    beat_gap = np.mean([r["oos_spearman_cert"] > r["oos_spearman_gap"]
                        for r in ok if not math.isnan(r["oos_spearman_cert"])
                        and not math.isnan(r["oos_spearman_gap"])])
    cert_vals = [r["oos_spearman_cert"] for r in ok if not math.isnan(r["oos_spearman_cert"])]
    frac_beat_026 = float(np.mean(np.array(cert_vals) > 0.26)) if cert_vals else float("nan")

    # failures
    fails = sorted([{"dataset": r["dataset"], "oos_spearman_cert": round(r["oos_spearman_cert"], 3),
                     "stable_transfer": (None if math.isnan(r["stable_transfer_frac"])
                                         else round(r["stable_transfer_frac"], 3))}
                    for r in ok if not math.isnan(r["oos_spearman_cert"])
                    and r["oos_spearman_cert"] < 0.2],
                   key=lambda d: d["oos_spearman_cert"])

    summary = {
        "n_datasets_selected": len(results), "n_ok": len(ok), "n_skipped": len(sk), "n_error": len(er),
        "total_pairs": int(sum(r["n_pairs"] for r in ok)),
        "OOS_spearman_certificate": dist("oos_spearman_cert"),
        "OOS_spearman_baseline_corr": dist("oos_spearman_corr"),
        "OOS_spearman_baseline_singlemodel_gap": dist("oos_spearman_gap"),
        "cert_beats_corr_frac": round(float(beat_corr), 3),
        "cert_beats_gap_frac": round(float(beat_gap), 3),
        "cert_frac_datasets_above_0.26_ceiling": round(frac_beat_026, 3),
        "STABLE_transfer_frac_per_dataset": dist("stable_transfer_frac"),
        "STABLE_meanB_flip_per_dataset": dist("stable_meanB"),
        "UNRELIABLE_meanB_flip_per_dataset": dist("unreliable_meanB"),
        "calibration_OOS_bins_[pred,obs,n]": calib, "calibration_ECE": round(ece, 4),
        "band_table_insample": {"STABLE": pooled_band("STABLE"),
                                "MARGINAL": pooled_band("MARGINAL"),
                                "UNRELIABLE": pooled_band("UNRELIABLE")},
        "cantelli_hold_frac_insample_pooled": round(float(np.mean([r["cantelli_hold_frac"] for r in ok])), 4),
        "pooled_mean_is_minority_frac": round(float(np.mean([r["mean_minority_frac"] for r in ok])), 4),
        "n_failure_datasets_oos<0.2": len(fails), "failure_datasets": fails[:20],
        "errors_sample": [r for r in er][:8], "skipped_reasons_sample": [r for r in sk][:8],
    }
    return summary


def main():
    sel = select_datasets()
    print(f"selected {len(sel)} datasets (200<=n<=5000, 4<=p<=30)")
    workers = max(1, (os.cpu_count() or 4) - 2)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_dataset, nt): nt[0] for nt in sel}
        done = 0
        for fu in as_completed(futs):
            results.append(fu.result())
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(sel)} done")
    summary = aggregate(results)
    # strip calib payloads from per-dataset before dumping detail
    detail = [{k: v for k, v in r.items() if not k.startswith("_calib")} for r in results]
    json.dump({"summary": summary, "detail": detail}, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
