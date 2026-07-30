#!/usr/bin/env python
"""
Instability decomposition: model-invariant vs model-specific (roadmap item #4).

The recurring caveat is that certificate reliability is PER-MODEL (~0.65 STABLE-call
agreement across model classes). This experiment decomposes a pairwise claim's
instability into (a) a WITHIN-CLASS part (same model class, different bootstrap --
what the single-class certificate measures) and (b) a BETWEEN-CLASS part (the
mean attribution difference disagreeing across model classes), and builds a stricter
MODEL-AGNOSTIC certificate that requires within-class STABLE in every class AND sign
agreement across classes. We then test, leave-one-class-out, whether the model-
agnostic certificate transfers to a HELD-OUT model class better than a single-class
(RF-only) certificate does -- turning the per-model caveat into a decomposition plus
an actionable stricter guarantee.

Model classes (diverse: bagging / boosting / linear / randomized-trees):
  RF, GradientBoosting, ExtraTrees, Linear(|coef|).
Held-out test class for transfer: Linear (the most different from the trees).
"""
import os, json, csv, warnings, math
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-drakecaraker/cfe4166d-73b6-4488-868f-379c98db298c/scratchpad"
CACHE = os.path.join(SCRATCH, "pmlb_cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_instability_decomposition.json")

M = 40
N_MIN, N_MAX = 200, 2500      # tighter row cap (4 model classes => 4x fits)
P_MIN, P_MAX = 4, 25
MAX_DATASETS = 40
STABLE = 2.0
EPS = 1e-12


def select():
    import pmlb
    p = os.path.join(os.path.dirname(pmlb.__file__), "all_summary_stats.tsv")
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sel = [(r["dataset"], r["task"]) for r in rows
           if N_MIN <= int(r["n_instances"]) <= N_MAX and P_MIN <= int(r["n_features"]) <= P_MAX]
    return sorted(sel)[:MAX_DATASETS]


def importances(kind, task, X, y, seed):
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    out = []
    for _ in range(M):
        idx = rng.integers(0, n, n)
        Xi, yi = X[idx], y[idx]
        rs = int(rng.integers(1e9))
        if kind == "RF":
            m = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(
                n_estimators=50, max_depth=6, random_state=rs, n_jobs=1)
            m.fit(Xi, yi); imp = m.feature_importances_
        elif kind == "ET":
            m = (ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor)(
                n_estimators=50, max_depth=6, random_state=rs, n_jobs=1)
            m.fit(Xi, yi); imp = m.feature_importances_
        elif kind == "GBM":
            m = (GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor)(
                n_estimators=40, max_depth=3, subsample=0.8, random_state=rs)
            m.fit(Xi, yi); imp = m.feature_importances_
        else:  # Linear: |coef| on standardized X
            if task == "classification":
                m = LogisticRegression(max_iter=200, random_state=rs)
                m.fit(Xi, yi); C = np.abs(m.coef_); imp = C.mean(0) if C.ndim > 1 else C.ravel()
            else:
                m = Ridge(alpha=1.0, random_state=rs)
                m.fit(Xi, yi); imp = np.abs(m.coef_).ravel()
            s = imp.sum()
            imp = imp / s if s > 0 else imp
        out.append(imp)
    return np.asarray(out)


def pair_stats(imp):
    M_, p = imp.shape
    res = {}
    for j in range(p):
        for k in range(j + 1, p):
            D = imp[:, j] - imp[:, k]
            mu, sd = D.mean(), D.std() + EPS
            res[(j, k)] = (mu, abs(mu) / sd)  # (mean, snr)
    return res


def run(name_task):
    name, task = name_task
    try:
        from pmlb import fetch_data
        df = fetch_data(name, local_cache_dir=CACHE)
        y = df["target"].to_numpy()
        X = df.drop(columns=["target"]).to_numpy(dtype=float)
        n, p = X.shape
        if not (N_MIN <= n <= N_MAX and P_MIN <= p <= P_MAX):
            return {"dataset": name, "skipped": "size"}
        keep = X.std(0) > EPS
        X = X[:, keep]; p = X.shape[1]
        if p < P_MIN:
            return {"dataset": name, "skipped": "constcols"}
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        if task == "classification":
            _, c = np.unique(y, return_counts=True)
            if len(c) < 2 or c.min() < 5:
                return {"dataset": name, "skipped": "target"}
        classes = ["RF", "GBM", "ET", "Linear"]
        stats = {c: pair_stats(importances(c, task, X, y, seed=7)) for c in classes}
        pairs = list(stats["RF"].keys())

        # variance decomposition per pair: between-class (var of class means) vs
        # within-class (avg class ensemble variance). ICC-like ratio.
        btw, wth, agree_all, n_pairs = 0.0, 0.0, 0, 0
        # transfer: predict-with {RF,GBM,ET}, test on held-out Linear
        rf_stable = magn_stable = 0
        rf_holds = magn_holds = 0  # of stable-called, holds on Linear (STABLE + sign agree)
        for pr in pairs:
            means = np.array([stats[c][pr][0] for c in classes])
            snrs = {c: stats[c][pr][1] for c in classes}
            # within-class "spread" proxy: 1/snr (relative); between: std of class means
            btw += float(np.std(means))
            wth += float(np.mean([abs(means[i]) / (snrs[classes[i]] + EPS) for i in range(len(classes))]))
            signs = np.sign(means)
            if np.all(signs == signs[0]) and signs[0] != 0:
                agree_all += 1
            n_pairs += 1
            # single-class RF certificate
            rf_ok = snrs["RF"] >= STABLE
            # model-agnostic over the PREDICT classes {RF,GBM,ET}
            pred_classes = ["RF", "GBM", "ET"]
            magn_ok = all(snrs[c] >= STABLE for c in pred_classes) and \
                len(set(np.sign([stats[c][pr][0] for c in pred_classes]))) == 1
            # held-out Linear "stable+consistent": Linear snr>=STABLE and sign matches RF's claim
            lin_ok = (snrs["Linear"] >= STABLE) and (np.sign(stats["Linear"][pr][0]) == np.sign(stats["RF"][pr][0]))
            if rf_ok:
                rf_stable += 1; rf_holds += int(lin_ok)
            if magn_ok:
                magn_stable += 1; magn_holds += int(lin_ok)
        return {
            "dataset": name, "task": task, "n": int(n), "p": int(p), "n_pairs": n_pairs,
            "between_class_meanspread": round(btw / n_pairs, 5),
            "within_class_meanspread": round(wth / n_pairs, 5),
            "sign_agree_all4_frac": round(agree_all / n_pairs, 4),
            "rf_stable_count": rf_stable, "rf_transfer_to_linear": (round(rf_holds / rf_stable, 4) if rf_stable else None),
            "magn_stable_count": magn_stable, "magn_transfer_to_linear": (round(magn_holds / magn_stable, 4) if magn_stable else None),
        }
    except Exception as e:
        return {"dataset": name, "error": f"{type(e).__name__}: {e}"}


def main():
    sel = select()
    print(f"decomposition on {len(sel)} datasets (4 model classes each)")
    workers = max(1, (os.cpu_count() or 4) - 2)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, nt): nt[0] for nt in sel}
        done = 0
        for fu in as_completed(futs):
            res.append(fu.result()); done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(sel)}")
    ok = [r for r in res if "n_pairs" in r]

    def med(key, cond=lambda r: True):
        v = [r[key] for r in ok if cond(r) and isinstance(r.get(key), (int, float))
             and not (isinstance(r.get(key), float) and math.isnan(r[key]))]
        return round(float(np.median(v)), 4) if v else None

    rf_tr = [r["rf_transfer_to_linear"] for r in ok if r["rf_transfer_to_linear"] is not None]
    mg_tr = [r["magn_transfer_to_linear"] for r in ok if r["magn_transfer_to_linear"] is not None]
    summary = {
        "n_ok": len(ok), "n_datasets": len(res),
        "median_sign_agree_all4": med("sign_agree_all4_frac"),
        "median_between_class_spread": med("between_class_meanspread"),
        "median_within_class_spread": med("within_class_meanspread"),
        "RF_only_STABLE_transfer_to_heldout_linear_median": round(float(np.median(rf_tr)), 4) if rf_tr else None,
        "RF_only_STABLE_transfer_mean": round(float(np.mean(rf_tr)), 4) if rf_tr else None,
        "MODEL_AGNOSTIC_STABLE_transfer_to_heldout_linear_median": round(float(np.median(mg_tr)), 4) if mg_tr else None,
        "MODEL_AGNOSTIC_STABLE_transfer_mean": round(float(np.mean(mg_tr)), 4) if mg_tr else None,
        "interpretation": ("If MODEL_AGNOSTIC transfer > RF_only transfer, requiring cross-class "
                           "agreement genuinely buys held-out-model reliability; the per-model caveat "
                           "is then decomposable and fixable by a stricter certificate."),
        "errors": [r for r in res if "error" in r][:6],
        "skipped": [r for r in res if "skipped" in r][:6],
    }
    json.dump({"summary": summary, "detail": ok}, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
