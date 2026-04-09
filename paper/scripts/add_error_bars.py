"""
Post-processing script: add bootstrap confidence intervals to flip rate estimates
and run a subsample=1.0 control experiment.

PART 1: For each results JSON with flip rates, compute 95% CIs
PART 2: Run subsample=1.0 control (deterministic training -> no Rashomon -> no instability)
PART 3: Print summary table with all experiments and their CIs

Self-contained. Requires: numpy, scipy, json, os.
Optional (Part 2 only): xgboost, shap.
"""

import json
import os
import glob
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

PAPER_DIR = Path(__file__).resolve().parent.parent
RESULTS_GLOB = str(PAPER_DIR / "results_*.json")
Z_95 = 1.96  # z for 95% CI


# =============================================================================
# PART 1: Confidence Intervals
# =============================================================================

def wilson_score_ci(p, n, z=Z_95):
    """Wilson score interval for a proportion p from n observations."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    """Bootstrap percentile CI on the mean of `values`."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0]))
    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def process_prevalence_openml(filepath):
    """Add Wilson CI to the 68% prevalence estimate (n=77 datasets)."""
    with open(filepath) as f:
        data = json.load(f)

    summary = data["summary"]
    n = summary["n_datasets"]
    k = summary["n_with_instability"]
    p = k / n

    lo, hi = wilson_score_ci(p, n)
    summary["prevalence_ci_lower"] = round(lo, 4)
    summary["prevalence_ci_upper"] = round(hi, 4)
    summary["prevalence_ci_method"] = "Wilson score"

    # Also add per-breakdown CIs
    for key, breakdown in summary.get("breakdown_by_features", {}).items():
        bn = breakdown["n"]
        bk = breakdown["n_inst"]
        if bn > 0:
            bp = bk / bn
            blo, bhi = wilson_score_ci(bp, bn)
            breakdown["instability_pct"] = round(100 * bp, 1)
            breakdown["ci_lower"] = round(blo, 4)
            breakdown["ci_upper"] = round(bhi, 4)

    # Per-dataset CIs on flip rates: these are proportions over seed-pairs
    # Each dataset used n_seeds seeds -> C(n_seeds, 2) pairwise comparisons
    # The config doesn't store n_seeds, but the flip_rate is already an average
    # Use Wilson with effective n from the survey setup
    for ds in data.get("datasets", []):
        if "max_flip_rate" in ds:
            fr = ds["max_flip_rate"]
            # Typically 20 seeds -> 190 seed-pairs, but per-dataset we only
            # have summary. Use n_effective = 190 (conservative).
            n_eff = 190
            lo_ds, hi_ds = wilson_score_ci(fr, n_eff)
            ds["flip_rate_ci_lower"] = round(lo_ds, 4)
            ds["flip_rate_ci_upper"] = round(hi_ds, 4)

    return data


def process_hyperparameter_sensitivity(filepath):
    """Add Wilson CI to each flip rate (20 seeds -> 190 seed-pairs)."""
    with open(filepath) as f:
        data = json.load(f)

    n_seeds = data["config"].get("n_seeds", 20)
    # Number of pairwise seed comparisons
    n_pairs = n_seeds * (n_seeds - 1) // 2
    # Within each config, flip_rate is averaged over pairs and seed-pairs
    # Groups: 2 groups of 5 -> C(5,2)=10 within-group pairs -> 20 total
    n_feature_pairs = 20
    # Total comparisons per config: n_feature_pairs * n_seed_pairs
    n_eff = n_feature_pairs * n_pairs  # 20 * 190 = 3800

    all_flip_rates = []
    for row in data["results"]:
        fr = row["flip_rate"]
        all_flip_rates.append(fr)
        lo, hi = wilson_score_ci(fr, n_eff)
        row["ci_lower"] = round(lo, 4)
        row["ci_upper"] = round(hi, 4)

    # Add CI on the global minimum
    min_fr = data["summary"]["global_min_flip_rate"]
    lo_min, hi_min = wilson_score_ci(min_fr, n_eff)
    data["summary"]["global_min_ci_lower"] = round(lo_min, 4)
    data["summary"]["global_min_ci_upper"] = round(hi_min, 4)
    data["summary"]["ci_method"] = "Wilson score (n_eff = n_feature_pairs * C(n_seeds, 2))"

    return data


def process_causal_dag(filepath):
    """Add Wilson CI to conditional vs marginal flip rates."""
    with open(filepath) as f:
        data = json.load(f)

    n_models = data["config"].get("n_models", 50)
    n_pairs = n_models * (n_models - 1) // 2  # C(50, 2) = 1225

    for row in data["rows"]:
        for key in ["marginal_flip_rate", "conditional_flip_rate"]:
            if key in row:
                fr = row[key]
                lo, hi = wilson_score_ci(fr, n_pairs)
                row[f"{key}_ci_lower"] = round(lo, 4)
                row[f"{key}_ci_upper"] = round(hi, 4)

    data["ci_method"] = "Wilson score (n_eff = C(n_models, 2))"
    return data


def process_generic_flip_rate(filepath):
    """Generic handler: add Wilson CI to any JSON with flip_rate fields.

    Heuristic for n_eff:
    - If config has n_seeds: use C(n_seeds, 2) * 20 (assuming 2 groups of 5)
    - If config has n_models: use C(n_models, 2)
    - Otherwise: use n_eff = 190 (conservative default: 20 seeds)
    """
    with open(filepath) as f:
        data = json.load(f)

    # Already has CIs
    if "ci_lower" in json.dumps(data):
        return data

    # Determine effective sample size
    config = data.get("config", {})
    n_seeds = config.get("n_seeds", config.get("n_models", 20))
    n_seed_pairs = n_seeds * (n_seeds - 1) // 2
    n_feature_pairs = 20  # default: 2 groups of 5
    n_eff = max(n_seed_pairs, 30)  # at least 30 for Wilson to be reasonable

    def add_ci_recursive(obj):
        """Walk the data structure and add CIs next to flip_rate fields."""
        if isinstance(obj, dict):
            keys_to_add = {}
            for k, v in obj.items():
                if "flip_rate" in k and isinstance(v, (int, float)) and 0 <= v <= 1:
                    lo, hi = wilson_score_ci(v, n_eff)
                    keys_to_add[f"{k}_ci_lower"] = round(lo, 4)
                    keys_to_add[f"{k}_ci_upper"] = round(hi, 4)
                elif isinstance(v, (dict, list)):
                    add_ci_recursive(v)
            obj.update(keys_to_add)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    add_ci_recursive(item)

    add_ci_recursive(data)
    if "ci_method" not in data:
        data["ci_method"] = f"Wilson score (n_eff={n_eff}, heuristic)"
    return data


def add_error_bars_to_all():
    """PART 1 main: add CIs to all result JSONs with flip rates."""
    print("=" * 72)
    print("PART 1: Adding Confidence Intervals to Experiment Results")
    print("=" * 72)
    print()

    files_processed = []
    files_skipped = []

    for filepath in sorted(glob.glob(RESULTS_GLOB)):
        fname = os.path.basename(filepath)

        # Skip broken JSON
        try:
            with open(filepath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  SKIP (invalid JSON): {fname}")
            files_skipped.append(fname)
            continue

        # Skip files without flip_rate
        raw_str = json.dumps(raw)
        if "flip_rate" not in raw_str and "instability" not in raw_str:
            print(f"  SKIP (no flip rates): {fname}")
            files_skipped.append(fname)
            continue

        # Back up original
        bak_path = filepath + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(filepath, bak_path)

        # Dispatch to specialized handler
        if "prevalence_openml" in fname:
            result = process_prevalence_openml(filepath)
        elif "hyperparameter_sensitivity" in fname:
            result = process_hyperparameter_sensitivity(filepath)
        elif "causal_dag" in fname:
            result = process_causal_dag(filepath)
        else:
            result = process_generic_flip_rate(filepath)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        files_processed.append(fname)
        print(f"  OK: {fname}")

    print()
    print(f"Processed: {len(files_processed)} files")
    print(f"Skipped:   {len(files_skipped)} files")
    return files_processed


# =============================================================================
# PART 2: Subsample=1.0 Control Experiment
# =============================================================================

def run_subsample_control():
    """Run XGBoost with subsample=1.0 (deterministic) to show flip rate -> 0."""
    print()
    print("=" * 72)
    print("PART 2: Subsample=1.0 Control Experiment")
    print("=" * 72)

    try:
        import xgboost as xgb
        import shap
    except ImportError:
        print("  xgboost or shap not installed. Skipping experiment.")
        print("  Install with: pip install xgboost shap")
        return None

    # Configuration (matches hyperparameter_sensitivity.py)
    P = 10
    GROUP_SIZE = 5
    N_TRAIN = 2000
    N_EVAL = 200
    N_SEEDS = 20
    RHO = 0.9
    DATA_SEED = 42

    print(f"  Config: P={P}, N_train={N_TRAIN}, N_eval={N_EVAL}, "
          f"n_seeds={N_SEEDS}, rho={RHO}")
    print(f"  XGBoost: subsample=1.0, colsample_bytree=1.0 (deterministic)")
    print()

    t0 = time.time()

    # Generate data (fixed seed for reproducibility)
    rng = np.random.default_rng(DATA_SEED)
    cov = np.eye(P)
    for group_start in [0, GROUP_SIZE]:
        for i in range(group_start, group_start + GROUP_SIZE):
            for j in range(group_start, group_start + GROUP_SIZE):
                if i != j:
                    cov[i, j] = RHO

    X_train = rng.multivariate_normal(np.zeros(P), cov, size=N_TRAIN)
    beta = np.ones(P)
    Y_train = X_train @ beta + rng.normal(0, 0.1, size=N_TRAIN)

    X_eval = rng.multivariate_normal(np.zeros(P), cov, size=N_EVAL)

    # Train N_SEEDS models with subsample=1.0 (only seed varies)
    shap_values_list = []
    for seed in range(N_SEEDS):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, Y_train)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)
        shap_values_list.append(sv)
        if (seed + 1) % 5 == 0:
            print(f"    Trained seed {seed + 1}/{N_SEEDS}")

    # Compute flip rate (same logic as hyperparameter_sensitivity.py)
    mean_abs = np.array([
        np.mean(np.abs(sv), axis=0) for sv in shap_values_list
    ])

    flips = 0
    comparisons = 0
    per_pair_flips = []

    for group_start in [0, GROUP_SIZE]:
        group_end = group_start + GROUP_SIZE
        for i in range(group_start, group_end):
            for j in range(i + 1, group_end):
                pair_flips = 0
                pair_comps = 0
                for s1 in range(N_SEEDS):
                    for s2 in range(s1 + 1, N_SEEDS):
                        rank_s1 = mean_abs[s1, i] > mean_abs[s1, j]
                        rank_s2 = mean_abs[s2, i] > mean_abs[s2, j]
                        if rank_s1 != rank_s2:
                            flips += 1
                            pair_flips += 1
                        comparisons += 1
                        pair_comps += 1
                per_pair_flips.append(pair_flips / pair_comps if pair_comps > 0 else 0)

    flip_rate = flips / comparisons if comparisons > 0 else 0.0

    # Also run with subsample=0.8 as the "stochastic" control
    print()
    print("  Running stochastic control (subsample=0.8) for comparison...")
    shap_values_stoch = []
    for seed in range(N_SEEDS):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, Y_train)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval)
        shap_values_stoch.append(sv)

    mean_abs_stoch = np.array([
        np.mean(np.abs(sv), axis=0) for sv in shap_values_stoch
    ])

    flips_stoch = 0
    comparisons_stoch = 0
    for group_start in [0, GROUP_SIZE]:
        group_end = group_start + GROUP_SIZE
        for i in range(group_start, group_end):
            for j in range(i + 1, group_end):
                for s1 in range(N_SEEDS):
                    for s2 in range(s1 + 1, N_SEEDS):
                        rank_s1 = mean_abs_stoch[s1, i] > mean_abs_stoch[s1, j]
                        rank_s2 = mean_abs_stoch[s2, i] > mean_abs_stoch[s2, j]
                        if rank_s1 != rank_s2:
                            flips_stoch += 1
                        comparisons_stoch += 1

    flip_rate_stoch = flips_stoch / comparisons_stoch if comparisons_stoch > 0 else 0.0

    elapsed = time.time() - t0

    # Wilson CIs
    n_eff = comparisons
    lo_det, hi_det = wilson_score_ci(flip_rate, n_eff)
    lo_stoch, hi_stoch = wilson_score_ci(flip_rate_stoch, comparisons_stoch)

    result = {
        "description": (
            "Subsample=1.0 control: deterministic XGBoost training eliminates "
            "stochasticity, removing the Rashomon set, which should eliminate "
            "flip rate instability. Compared against subsample=0.8 (stochastic)."
        ),
        "config": {
            "P": P,
            "group_size": GROUP_SIZE,
            "N_train": N_TRAIN,
            "N_eval": N_EVAL,
            "n_seeds": N_SEEDS,
            "rho": RHO,
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        "deterministic": {
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "flip_rate": round(flip_rate, 6),
            "ci_lower": round(lo_det, 4),
            "ci_upper": round(hi_det, 4),
            "n_comparisons": comparisons,
            "per_pair_flip_rates": [round(x, 4) for x in per_pair_flips],
        },
        "stochastic": {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "flip_rate": round(flip_rate_stoch, 6),
            "ci_lower": round(lo_stoch, 4),
            "ci_upper": round(hi_stoch, 4),
            "n_comparisons": comparisons_stoch,
        },
        "verdict": (
            f"Deterministic flip rate = {flip_rate:.4f} "
            f"(CI: [{lo_det:.4f}, {hi_det:.4f}]), "
            f"stochastic flip rate = {flip_rate_stoch:.4f} "
            f"(CI: [{lo_stoch:.4f}, {hi_stoch:.4f}]). "
            f"{'Mechanism confirmed: stochastic training creates Rashomon set.' if flip_rate < 0.05 and flip_rate_stoch > 0.10 else 'See per-pair rates for detail.'}"
        ),
        "runtime_seconds": round(elapsed, 1),
        "ci_method": "Wilson score",
    }

    out_path = str(PAPER_DIR / "results_subsample_control.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"  Deterministic (subsample=1.0): flip rate = {flip_rate:.4f} "
          f"  95% CI: [{lo_det:.4f}, {hi_det:.4f}]")
    print(f"  Stochastic   (subsample=0.8): flip rate = {flip_rate_stoch:.4f} "
          f"  95% CI: [{lo_stoch:.4f}, {hi_stoch:.4f}]")
    print(f"  Saved to: {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# =============================================================================
# PART 3: Summary Table
# =============================================================================

def print_summary_table():
    """Print a table of all experiments with confidence intervals."""
    print()
    print("=" * 72)
    print("PART 3: Summary Table")
    print("=" * 72)
    print()

    rows = []

    # --- Prevalence (OpenML) ---
    try:
        with open(str(PAPER_DIR / "results_prevalence_openml.json")) as f:
            d = json.load(f)
        s = d["summary"]
        rows.append((
            "Prevalence (77 ds)",
            "% with instability",
            f"{s['pct_with_instability']:.0f}%",
            f"[{s.get('prevalence_ci_lower', '?'):.0%}, "
            f"{s.get('prevalence_ci_upper', '?'):.0%}]"
            if isinstance(s.get('prevalence_ci_lower'), (int, float))
            else "[not computed]"
        ))
    except Exception:
        pass

    # --- Hyperparameter sensitivity ---
    try:
        with open(str(PAPER_DIR / "results_hyperparameter_sensitivity.json")) as f:
            d = json.load(f)
        s = d["summary"]
        rows.append((
            "Hyperparam sensitivity",
            "min flip rate",
            f"{s['global_min_flip_rate']:.1%}",
            f"[{s.get('global_min_ci_lower', '?'):.1%}, "
            f"{s.get('global_min_ci_upper', '?'):.1%}]"
            if isinstance(s.get('global_min_ci_lower'), (int, float))
            else "[not computed]"
        ))
    except Exception:
        pass

    # --- Causal DAG ---
    try:
        with open(str(PAPER_DIR / "results_causal_dag.json")) as f:
            d = json.load(f)
        # delta_beta=0 row
        row0 = d["rows"][0]
        rows.append((
            "Causal DAG (db=0)",
            "marginal flip",
            f"{row0['marginal_flip_rate']:.1%}",
            f"[{row0.get('marginal_flip_rate_ci_lower', '?'):.1%}, "
            f"{row0.get('marginal_flip_rate_ci_upper', '?'):.1%}]"
            if isinstance(row0.get('marginal_flip_rate_ci_lower'), (int, float))
            else "[not computed]"
        ))
        rows.append((
            "Causal DAG (db=0)",
            "conditional flip",
            f"{row0['conditional_flip_rate']:.1%}",
            f"[{row0.get('conditional_flip_rate_ci_lower', '?'):.1%}, "
            f"{row0.get('conditional_flip_rate_ci_upper', '?'):.1%}]"
            if isinstance(row0.get('conditional_flip_rate_ci_lower'), (int, float))
            else "[not computed]"
        ))
    except Exception:
        pass

    # --- Subsample control ---
    try:
        with open(str(PAPER_DIR / "results_subsample_control.json")) as f:
            d = json.load(f)
        det = d["deterministic"]
        sto = d["stochastic"]
        rows.append((
            "Subsample=1.0 ctrl",
            "flip (deterministic)",
            f"{det['flip_rate']:.1%}",
            f"[{det['ci_lower']:.1%}, {det['ci_upper']:.1%}]"
        ))
        rows.append((
            "Subsample=0.8 ctrl",
            "flip (stochastic)",
            f"{sto['flip_rate']:.1%}",
            f"[{sto['ci_lower']:.1%}, {sto['ci_upper']:.1%}]"
        ))
    except Exception:
        pass

    # --- Other experiments with flip rates ---
    other_files = [
        ("results_class_imbalance.json", "Class imbalance"),
        ("results_missing_data.json", "Missing data"),
        ("results_longitudinal.json", "Longitudinal"),
        ("results_adversarial.json", "Adversarial"),
        ("results_dash_breakdown.json", "DASH breakdown"),
        ("results_nn_shap.json", "NN SHAP"),
        ("results_nlp_token.json", "NLP token"),
        ("results_llm_attention.json", "LLM attention"),
        ("results_timeseries.json", "Time series"),
        ("results_pdp_comparison.json", "PDP comparison"),
        ("results_sage_comparison.json", "SAGE comparison"),
        ("results_prevalence.json", "Prevalence (simple)"),
        ("results_prevalence_robustness.json", "Prevalence robust"),
        ("results_healthcare_prevalence.json", "Healthcare prev"),
    ]

    for fname, label in other_files:
        try:
            with open(str(PAPER_DIR / fname)) as f:
                d = json.load(f)
            # Find the first flip_rate in the data
            s = json.dumps(d)
            if "flip_rate" not in s:
                continue

            # Try to extract a headline flip_rate
            flip_val = None
            ci_lo = None
            ci_hi = None

            # Check summary
            if isinstance(d.get("summary"), dict):
                for k, v in d["summary"].items():
                    if "flip_rate" in k and isinstance(v, (int, float)) and ci_lo is None:
                        flip_val = v
                        ci_lo = d["summary"].get(f"{k}_ci_lower")
                        ci_hi = d["summary"].get(f"{k}_ci_upper")

            # Check results list
            if flip_val is None and isinstance(d.get("results"), list) and d["results"]:
                first = d["results"][0]
                if isinstance(first, dict):
                    for k, v in first.items():
                        if "flip_rate" in k and isinstance(v, (int, float)):
                            flip_val = v
                            ci_lo = first.get(f"{k}_ci_lower")
                            ci_hi = first.get(f"{k}_ci_upper")
                            break

            # Check rows list
            if flip_val is None and isinstance(d.get("rows"), list) and d["rows"]:
                first = d["rows"][0]
                if isinstance(first, dict):
                    for k, v in first.items():
                        if "flip_rate" in k and isinstance(v, (int, float)):
                            flip_val = v
                            ci_lo = first.get(f"{k}_ci_lower")
                            ci_hi = first.get(f"{k}_ci_upper")
                            break

            if flip_val is not None:
                ci_str = (
                    f"[{ci_lo:.1%}, {ci_hi:.1%}]"
                    if isinstance(ci_lo, (int, float)) and isinstance(ci_hi, (int, float))
                    else "[not computed]"
                )
                rows.append((
                    label,
                    "flip rate",
                    f"{flip_val:.1%}",
                    ci_str,
                ))
        except Exception:
            continue

    # Print table
    if not rows:
        print("  No experiments found with CI data.")
        return

    col_widths = [
        max(len(r[0]) for r in rows),
        max(len(r[1]) for r in rows),
        max(len(r[2]) for r in rows),
        max(len(r[3]) for r in rows),
    ]
    header = (
        f"{'Experiment':<{col_widths[0]}} | "
        f"{'Key Metric':<{col_widths[1]}} | "
        f"{'Point Est':>{col_widths[2]}} | "
        f"{'95% CI':<{col_widths[3]}}"
    )
    sep = "-" * col_widths[0] + "-+-" + "-" * col_widths[1] + "-+-" + \
          "-" * col_widths[2] + "-+-" + "-" * col_widths[3]

    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r[0]:<{col_widths[0]}} | "
            f"{r[1]:<{col_widths[1]}} | "
            f"{r[2]:>{col_widths[2]}} | "
            f"{r[3]:<{col_widths[3]}}"
        )

    print()
    print(f"Total rows: {len(rows)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    os.chdir(PAPER_DIR)

    # PART 1: Add error bars
    add_error_bars_to_all()

    # PART 2: Subsample control
    run_subsample_control()

    # PART 3: Summary table
    print_summary_table()

    print()
    print("Done.")
