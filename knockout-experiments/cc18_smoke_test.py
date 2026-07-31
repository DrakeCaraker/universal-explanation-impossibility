#!/usr/bin/env python
"""Pre-registration smoke test for cc18_prospective_runner.py.

Verifies the pipeline MECHANICS (run_dataset -> aggregate -> evaluate_prereg)
on synthetic data only. The CC-18 loader is monkeypatched out, and the test
asserts `openml` is never imported — so running this accesses NO CC-18 data
and does not violate the OSF protocol (registration must precede any CC-18
download). Safe to run before, during, or after registration.
"""
import sys
import json
import numpy as np

import cc18_prospective_runner as R


def synthetic_loader(dataset_id):
    rng = np.random.default_rng(777 + dataset_id)
    n, p = 400, 8
    X = rng.normal(size=(n, p))
    # planted structure: 0,1 informative; 2,3 exact duplicates (a Rashomon pair)
    X[:, 3] = X[:, 2]
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def main():
    assert "openml" not in sys.modules
    R.load_cc18_dataset = synthetic_loader

    jobs = [(1000 + i, i, f"synthetic_{i}", set(), set()) for i in range(4)]
    jobs.append((2000, 99, "dupname", {"dupname"}, set()))  # dedup-by-name path
    results = [R.run_dataset(j) for j in jobs]
    assert "openml" not in sys.modules, "CC-18 loader must never be touched"

    ok = [r for r in results if "oos_spearman_cert" in r]
    skipped = [r for r in results if "skipped" in r]
    errors = [r for r in results if "error" in r]
    assert len(ok) == 4, f"expected 4 ok datasets, got {len(ok)}; errors={errors}"
    assert any(r.get("skipped") == "dedup_name_matches_pmlb" for r in skipped)
    for r in ok:
        # in-sample Cantelli holds by theorem (cantelli_lower_tail); any
        # violation here means the pipeline computes the bound wrong
        assert r["cantelli_hold_frac"] == 1.0, r

    summary = R.aggregate(results)
    verdicts = summary["PREREG_VERDICTS"]
    assert set(verdicts) == {"P1", "P2", "P3", "P4", "P5"}
    for pid, v in verdicts.items():
        assert v["pass"] in (True, False, None), (pid, v)

    print(json.dumps(verdicts, indent=2))
    print("SMOKE PASS — pipeline mechanics verified; openml never imported, "
          "no CC-18 data accessed")


if __name__ == "__main__":
    main()
