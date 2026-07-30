#!/usr/bin/env python
"""
Simultaneous (family-wise) guarantee + certified top-k (roadmap items #3, #6).

#3 Simultaneous: the per-query certificate bounds ONE claim's flip. To trust K
claims AT ONCE, a union (Bonferroni) bound gives
    P(any of the K STABLE claims flips) <= sum_k 1/(1+SNR_k^2)  (<= K/5 if all STABLE).
This mirrors the Lean `simultaneous_cantelli`. We validate it OUT OF SAMPLE: form the
STABLE set on ensemble A, then measure on independent ensemble B the fraction of
ensemble members on which AT LEAST ONE STABLE-A claim flips, and check it stays under
the Bonferroni bound (family-wise coverage).

#6 Certified top-k: the top-k feature set is "certified" if the boundary comparison
(k-th vs (k+1)-th by mean importance) is STABLE on A. We test whether a certified
top-k SET survives retraining (exact-match / Jaccard on independent ensemble B) more
reliably than an uncertified one -- a guarantee on the headline "top features" claim.
"""
import os, json, csv, warnings, math
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-drakecaraker/cfe4166d-73b6-4488-868f-379c98db298c/scratchpad"
CACHE = os.path.join(SCRATCH, "pmlb_cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_simultaneous_topk.json")

M = 60
HALF = M // 2
N_MIN, N_MAX = 200, 5000
P_MIN, P_MAX = 4, 30
STABLE = 2.0
EPS = 1e-12


def select():
    import pmlb
    p = os.path.join(os.path.dirname(pmlb.__file__), "all_summary_stats.tsv")
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return sorted((r["dataset"], r["task"]) for r in rows
                  if N_MIN <= int(r["n_instances"]) <= N_MAX and P_MIN <= int(r["n_features"]) <= P_MAX)


def ensemble(task, X, y, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    rng = np.random.default_rng(seed); n = X.shape[0]
    Mk = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return np.asarray([Mk(n_estimators=50, max_depth=6, random_state=int(rng.integers(1e9)), n_jobs=1)
                       .fit(X[rng.integers(0, n, n)], y[rng.integers(0, n, n)]).feature_importances_
                       for _ in range(M)])


def run(name_task):
    name, task = name_task
    try:
        from pmlb import fetch_data
        df = fetch_data(name, local_cache_dir=CACHE)
        y = df["target"].to_numpy(); X = df.drop(columns=["target"]).to_numpy(dtype=float)
        n, p = X.shape
        if not (N_MIN <= n <= N_MAX and P_MIN <= p <= P_MAX):
            return {"dataset": name, "skipped": "size"}
        keep = X.std(0) > EPS; X = X[:, keep]; p = X.shape[1]
        if p < P_MIN:
            return {"dataset": name, "skipped": "constcols"}
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        if task == "classification":
            _, c = np.unique(y, return_counts=True)
            if len(c) < 2 or c.min() < 5:
                return {"dataset": name, "skipped": "target"}
        # need bootstrap idx consistent across the two feature draws; refit with fixed seed
        imp = ensemble(task, X, y, seed=2024)
        A, B = imp[:HALF], imp[HALF:]

        # ---- #3 simultaneous ----
        stable_pairs, bonf = [], 0.0
        for j in range(p):
            for k in range(j + 1, p):
                dA = A[:, j] - A[:, k]; muA, sdA = dA.mean(), dA.std() + EPS
                if muA == 0:
                    continue
                snrA = abs(muA) / sdA
                if snrA >= STABLE:
                    stable_pairs.append((j, k, np.sign(muA)))
                    bonf += 1.0 / (1.0 + snrA ** 2)
        bonf = min(1.0, bonf)
        # OOS any-flip on B: fraction of B members where >=1 stable-A claim flips
        if stable_pairs:
            flipmat = np.zeros(B.shape[0], dtype=bool)
            for (j, k, s) in stable_pairs:
                dB = B[:, j] - B[:, k]
                flipmat |= (np.sign(dB) != s)
            any_flip_B = float(flipmat.mean())
            simult_holds = bool(any_flip_B <= bonf + 1e-9)
        else:
            any_flip_B, simult_holds = None, None

        # ---- #6 certified top-k ----
        meanA = A.mean(0)
        order = np.argsort(-meanA)
        topk_res = {}
        for k in (3, 5):
            if p <= k:
                continue
            kth, k1 = order[k - 1], order[k]      # boundary pair
            dA = A[:, kth] - A[:, k1]; muA, sdA = dA.mean(), dA.std() + EPS
            snr_boundary = abs(muA) / sdA
            certified = bool(snr_boundary >= STABLE)
            setA = set(order[:k].tolist())
            setB = set(np.argsort(-B.mean(0))[:k].tolist())
            exact = bool(setA == setB)
            jac = len(setA & setB) / len(setA | setB)
            topk_res[f"k{k}"] = {"boundary_snr": round(float(snr_boundary), 3),
                                 "certified": certified, "exact_match_B": exact,
                                 "jaccard_B": round(float(jac), 3)}
        return {"dataset": name, "task": task, "n": int(n), "p": int(p),
                "n_stable_pairs": len(stable_pairs), "bonferroni_bound": round(float(bonf), 4),
                "any_flip_B": (None if any_flip_B is None else round(any_flip_B, 4)),
                "simultaneous_holds": simult_holds, "topk": topk_res}
    except Exception as e:
        return {"dataset": name, "error": f"{type(e).__name__}: {e}"}


def main():
    sel = select()
    print(f"simultaneous+topk on {len(sel)} datasets")
    workers = max(1, (os.cpu_count() or 4) - 2)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, nt): nt[0] for nt in sel}
        done = 0
        for fu in as_completed(futs):
            res.append(fu.result()); done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(sel)}")
    ok = [r for r in res if "n_stable_pairs" in r]

    # #3 aggregate
    sim = [r for r in ok if r["simultaneous_holds"] is not None]
    simult_hold_frac = round(float(np.mean([r["simultaneous_holds"] for r in sim])), 4) if sim else None
    # conservativeness gap: bound - observed (how much slack)
    gaps = [r["bonferroni_bound"] - r["any_flip_B"] for r in sim]
    # #6 aggregate: certified vs uncertified exact-match rate
    def topk_rates(k):
        cert_ex, cert_n, unc_ex, unc_n = 0, 0, 0, 0
        for r in ok:
            t = r["topk"].get(f"k{k}")
            if not t:
                continue
            if t["certified"]:
                cert_n += 1; cert_ex += int(t["exact_match_B"])
            else:
                unc_n += 1; unc_ex += int(t["exact_match_B"])
        return {"certified_exactmatch": (round(cert_ex / cert_n, 4) if cert_n else None), "certified_n": cert_n,
                "uncertified_exactmatch": (round(unc_ex / unc_n, 4) if unc_n else None), "uncertified_n": unc_n}

    summary = {
        "n_ok": len(ok), "n_datasets": len(res),
        "simultaneous_bonferroni_holds_frac": simult_hold_frac,
        "simultaneous_n_datasets": len(sim),
        "median_bound_minus_observed_slack": round(float(np.median(gaps)), 4) if gaps else None,
        "median_any_flip_B": round(float(np.median([r["any_flip_B"] for r in sim])), 4) if sim else None,
        "topk_k3": topk_rates(3), "topk_k5": topk_rates(5),
        "errors": [r for r in res if "error" in r][:6], "skipped": [r for r in res if "skipped" in r][:6],
    }
    json.dump({"summary": summary, "detail": ok}, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
