#!/usr/bin/env python3
"""
run_all_universal_experiments.py
---------------------------------
Runs all four universal explanation impossibility experiments sequentially.

Experiment order matches Section 4 (Instances) of universal_impossibility.tex:
  1. Attention map instability (DistilBERT, Task 1A)
  2. Counterfactual explanation instability (XGBoost, Task 1B)
  3. Concept probe instability / TCAV (neural nets, Task 1C)
  4. Model selection instability (Rashomon multiplicity, Task 1D)

Each script saves its own JSON results and (where applicable) LaTeX table
fragments under paper/results_*.json and paper/sections/table_*.tex.

Usage (from repo root):
    python paper/scripts/run_all_universal_experiments.py

Or with verbose output piped to a log:
    python paper/scripts/run_all_universal_experiments.py 2>&1 | tee paper/run_all.log
"""

import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate scripts directory relative to this file
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name": "Attention Map Instability (Instance 2: DistilBERT)",
        "script": SCRIPTS_DIR / "attention_instability_experiment.py",
        "outputs": [
            "paper/results_attention_instability.json",
            "paper/figures/attention_instability.pdf",
            "paper/sections/table_attention.tex",
        ],
    },
    {
        "name": "Counterfactual Instability (Instance 3: XGBoost / German Credit)",
        "script": SCRIPTS_DIR / "counterfactual_instability_experiment.py",
        "outputs": [
            "paper/results_counterfactual_instability.json",
            "paper/figures/counterfactual_instability.pdf",
            "paper/sections/table_counterfactual.tex",
        ],
    },
    {
        "name": "Concept Probe Instability (Instance 4: TCAV / MNIST)",
        "script": SCRIPTS_DIR / "concept_probe_instability_experiment.py",
        "outputs": [
            "paper/results_concept_probe_instability.json",
            "paper/sections/table_concept.tex",
        ],
    },
    {
        "name": "Model Selection Instability (Instance 6: Rashomon multiplicity)",
        "script": SCRIPTS_DIR / "model_selection_instability_experiment.py",
        "outputs": [
            "paper/results_model_selection_instability.json",
            "paper/sections/table_model_selection.tex",
        ],
    },
]


def run_experiment(info: dict, idx: int, total: int) -> bool:
    """Run a single experiment script. Return True on success."""
    name = info["name"]
    script = info["script"]

    print()
    print("=" * 70)
    print(f"[{idx}/{total}] {name}")
    print(f"  Script: {script.name}")
    print("=" * 70)

    if not script.exists():
        print(f"ERROR: script not found: {script}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPTS_DIR.parent.parent),  # repo root
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\nOK — completed in {elapsed:.1f}s")
    print("Expected outputs:")
    for out in info["outputs"]:
        print(f"  {out}")
    return True


def main():
    print()
    print("Universal Explanation Impossibility — Experiment Runner")
    print("========================================================")
    print(f"Running {len(EXPERIMENTS)} experiments sequentially.")
    print()

    wall_start = time.time()
    results = []

    for i, exp in enumerate(EXPERIMENTS, start=1):
        ok = run_experiment(exp, i, len(EXPERIMENTS))
        results.append((exp["name"], ok))

    wall_elapsed = time.time() - wall_start

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_ok = sum(ok for _, ok in results)
    for name, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}")
    print()
    print(f"  {n_ok}/{len(results)} experiments succeeded in {wall_elapsed:.1f}s total")

    if n_ok < len(results):
        print()
        print("Some experiments failed. Check output above for details.")
        print("Common causes:")
        print("  - Missing dependencies: pip install -r paper/scripts/requirements.txt")
        print("  - transformers/torch not installed: needed for attention experiment")
        print("    (pip install transformers torch torchvision)")
        print("  - openml not installed: pip install openml")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
