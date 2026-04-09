"""
causal_dag_experiment.py
------------------------
Validates the conditional SHAP impossibility and escape condition using a
known causal DAG.

Known DAG:
  X1 -> Y, X2 -> Y, X1 <-> X2 (correlated via shared latent)

Data generating process:
  X1, X2 ~ bivariate Gaussian with correlation rho = 0.8
  Y = beta1 * X1 + beta2 * X2 + epsilon,  epsilon ~ N(0, 0.1)

We sweep Delta_beta = |beta1 - beta2| from 0 to 1 in steps of 0.1, with
  beta1 = 1 + Delta_beta / 2,  beta2 = 1 - Delta_beta / 2
so the average coefficient stays at 1.

For each Delta_beta we:
  1. Generate N=2000 samples
  2. Train 50 XGBoost models (different seeds, subsample=0.8)
  3. Compute marginal SHAP (TreeSHAP, tree_path_dependent) -- expected UNSTABLE
     for small Delta_beta
  4. Compute conditional SHAP (TreeSHAP, interventional with background data) --
     expected UNSTABLE at Delta_beta=0, stable for large Delta_beta
  5. Compute decorrelated SHAP (residualize X2 on X1, retrain, TreeSHAP) --
     expected ALWAYS stable (changes the problem, not a valid comparison)
  6. Measure flip rate for the (X1, X2) pair under all three methods

Expected results:
  - Marginal SHAP: unstable for Delta_beta < ~0.2, stable for large Delta_beta
  - Conditional SHAP: unstable when Delta_beta = 0 (impossible to escape),
    stable when Delta_beta > 0
  - Decorrelated: always stable (changes the problem — NOT a valid SHAP method)
  - Validates: conditional SHAP escapes impossibility ONLY when causal effects
    differ

Output:
  - Table: Delta_beta | marginal_flip | conditional_flip | decorrelated_flip
  - Verdict string summarizing the resolution thresholds
  - Saved to paper/results_causal_dag.json
"""

import os
import sys
import json
import warnings
import itertools

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Install with: pip install shap")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

N_SAMPLES = 2000
N_MODELS = 50
N_EVAL = 200
RHO = 0.8
SIGMA_NOISE = 0.1
TRAIN_FRAC = 0.8
DELTA_BETAS = [round(d * 0.1, 1) for d in range(11)]  # 0.0, 0.1, ..., 1.0

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    n_jobs=1,
    verbosity=0,
)

# Threshold: flip rate below this is considered "stable"
STABLE_THRESHOLD = 0.10

# ── Check for interventional SHAP support ────────────────────────────────────

_INTERVENTIONAL_AVAILABLE = True
try:
    # Test that TreeExplainer accepts feature_perturbation="interventional"
    # by inspecting the constructor signature (avoids training a throwaway model)
    import inspect
    sig = inspect.signature(shap.TreeExplainer.__init__)
    if "feature_perturbation" not in sig.parameters:
        _INTERVENTIONAL_AVAILABLE = False
except Exception:
    _INTERVENTIONAL_AVAILABLE = False

if not _INTERVENTIONAL_AVAILABLE:
    print("WARNING: shap.TreeExplainer does not support feature_perturbation='interventional'.")
    print("         Conditional SHAP will fall back to decorrelated proxy (known to be inaccurate).")
    print("         Upgrade shap: pip install --upgrade shap>=0.41")

# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(delta_beta: float, rng: np.random.Generator) -> tuple:
    """
    Generate (X, y) from the causal DGP.

    X1, X2 ~ bivariate Gaussian with Corr(X1, X2) = RHO.
    Y = beta1 * X1 + beta2 * X2 + epsilon.
    """
    beta1 = 1.0 + delta_beta / 2.0
    beta2 = 1.0 - delta_beta / 2.0

    # Bivariate Gaussian via latent variable
    Z = rng.standard_normal(N_SAMPLES)
    eps_x = rng.standard_normal(N_SAMPLES)
    X1 = Z
    X2 = RHO * Z + np.sqrt(1.0 - RHO ** 2) * eps_x

    noise = rng.normal(0.0, SIGMA_NOISE, size=N_SAMPLES)
    y = beta1 * X1 + beta2 * X2 + noise

    return X1, X2, y, beta1, beta2


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def mean_abs_shap_marginal(model: xgb.XGBRegressor, X_eval: np.ndarray) -> np.ndarray:
    """
    Marginal SHAP via tree_path_dependent perturbation.
    Returns mean |SHAP| per feature.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    shap_vals = explainer.shap_values(X_eval)
    return np.abs(shap_vals).mean(axis=0)


def mean_abs_shap_conditional(
    model: xgb.XGBRegressor,
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Conditional SHAP via interventional perturbation with background data.

    This uses the SAME model as marginal but changes how SHAP handles
    feature dependencies: interventional mode conditions on observed
    feature values from the background dataset rather than following
    tree paths.

    Returns mean |SHAP| per feature.
    """
    explainer = shap.TreeExplainer(
        model,
        X_train,
        feature_perturbation="interventional",
    )
    shap_vals = explainer.shap_values(X_eval)
    return np.abs(shap_vals).mean(axis=0)


def mean_abs_shap_decorrelated(
    model: xgb.XGBRegressor, X_eval: np.ndarray
) -> np.ndarray:
    """
    Decorrelated SHAP: train a SEPARATE model on residualized features,
    then compute TreeSHAP.  This changes the effective coefficients and
    is NOT a valid conditional SHAP method — included for contrast only.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    shap_vals = explainer.shap_values(X_eval)
    return np.abs(shap_vals).mean(axis=0)


# ── Flip rate ─────────────────────────────────────────────────────────────────

def compute_flip_rate(rankings: list) -> float:
    """
    Fraction of model pairs where feature 0 vs feature 1 ranking reverses.
    """
    n_pairs = len(rankings) * (len(rankings) - 1) // 2
    if n_pairs == 0:
        return 0.0
    n_flips = sum(1 for a, b in itertools.combinations(rankings, 2) if a != b)
    return n_flips / n_pairs


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment() -> list:
    """
    Sweep Delta_beta and compute marginal, conditional, and decorrelated flip rates.
    Returns list of row dicts.
    """
    rows = []

    use_interventional = _INTERVENTIONAL_AVAILABLE

    header = (f"{'Delta_beta':>10}  {'beta1':>6}  {'beta2':>6}  "
              f"{'Marginal':>10}  {'Conditional':>12}  {'Decorrelated':>12}")
    print(f"\n{header}")
    print(f"  {'-' * 70}")

    for delta_beta in DELTA_BETAS:
        beta1 = 1.0 + delta_beta / 2.0
        beta2 = 1.0 - delta_beta / 2.0

        marginal_rankings = []
        conditional_rankings = []
        decorrelated_rankings = []

        for seed in range(N_MODELS):
            rng = np.random.default_rng(seed * 1000 + int(delta_beta * 100))
            X1, X2, y, _, _ = generate_data(delta_beta, rng)

            n_train = int(TRAIN_FRAC * N_SAMPLES)

            # ── Original features: [X1, X2] ──────────────────────────────
            X_orig = np.column_stack([X1, X2])
            X_train_orig = X_orig[:n_train]
            y_train = y[:n_train]
            X_eval_orig = X_orig[n_train : n_train + N_EVAL]

            model = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
            model.fit(X_train_orig, y_train)

            # ── Marginal SHAP (tree_path_dependent) ──────────────────────
            ms_m = mean_abs_shap_marginal(model, X_eval_orig)
            marginal_rankings.append(ms_m[0] > ms_m[1])

            # ── Conditional SHAP (interventional, same model) ────────────
            if use_interventional:
                try:
                    ms_c = mean_abs_shap_conditional(
                        model, X_train_orig, X_eval_orig
                    )
                    conditional_rankings.append(ms_c[0] > ms_c[1])
                except Exception as exc:
                    # If interventional fails at runtime, fall back for this
                    # entire run and warn once
                    if use_interventional:
                        print(f"\nWARNING: interventional SHAP failed ({exc}). "
                              f"Falling back to decorrelated proxy.")
                        use_interventional = False
                    # Use marginal as placeholder for this seed (will be
                    # overwritten below if we fall back to decorrelated)
                    conditional_rankings.append(ms_m[0] > ms_m[1])

            if not use_interventional:
                # Fallback: use decorrelated as proxy (known inaccurate)
                X2_resid = X2 - RHO * X1
                X_fb = np.column_stack([X1, X2_resid])
                X_train_fb = X_fb[:n_train]
                X_eval_fb = X_fb[n_train : n_train + N_EVAL]
                model_fb = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
                model_fb.fit(X_train_fb, y_train)
                ms_fb = mean_abs_shap_decorrelated(model_fb, X_eval_fb)
                conditional_rankings.append(ms_fb[0] > ms_fb[1])

            # ── Decorrelated SHAP (residualize, retrain — contrast) ──────
            X2_resid = X2 - RHO * X1
            X_decor = np.column_stack([X1, X2_resid])
            X_train_d = X_decor[:n_train]
            X_eval_d = X_decor[n_train : n_train + N_EVAL]

            model_d = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
            model_d.fit(X_train_d, y_train)

            ms_d = mean_abs_shap_decorrelated(model_d, X_eval_d)
            decorrelated_rankings.append(ms_d[0] > ms_d[1])

        marg_flip = compute_flip_rate(marginal_rankings)
        cond_flip = compute_flip_rate(conditional_rankings)
        decor_flip = compute_flip_rate(decorrelated_rankings)

        print(f"  {delta_beta:>8.1f}  {beta1:>6.2f}  {beta2:>6.2f}  "
              f"{marg_flip:>10.3f}  {cond_flip:>12.3f}  {decor_flip:>12.3f}")

        rows.append(dict(
            delta_beta=delta_beta,
            beta1=beta1,
            beta2=beta2,
            marginal_flip_rate=round(marg_flip, 4),
            conditional_flip_rate=round(cond_flip, 4),
            decorrelated_flip_rate=round(decor_flip, 4),
        ))

    return rows


# ── Verdict ───────────────────────────────────────────────────────────────────

def compute_verdict(rows: list) -> str:
    """
    Summarize the resolution thresholds for marginal, conditional, and
    decorrelated SHAP.
    """
    cond_stable = [r for r in rows
                   if r["conditional_flip_rate"] < STABLE_THRESHOLD
                   and r["delta_beta"] > 0]
    marg_stable = [r for r in rows
                   if r["marginal_flip_rate"] < STABLE_THRESHOLD]
    decor_stable = [r for r in rows
                    if r["decorrelated_flip_rate"] < STABLE_THRESHOLD]

    if cond_stable:
        cond_threshold = min(r["delta_beta"] for r in cond_stable)
    else:
        cond_threshold = None

    if marg_stable:
        marg_threshold = min(r["delta_beta"] for r in marg_stable)
    else:
        marg_threshold = None

    parts = []

    # -- Conditional verdict --
    zero_row = [r for r in rows if r["delta_beta"] == 0.0]
    if zero_row and zero_row[0]["conditional_flip_rate"] >= STABLE_THRESHOLD:
        parts.append(f"Conditional SHAP is UNSTABLE at Delta_beta=0 "
                     f"(flip={zero_row[0]['conditional_flip_rate']:.2%}) — "
                     f"impossibility holds for equal causal effects")
    if cond_threshold is not None:
        parts.append(f"conditional resolves at Delta_beta >= {cond_threshold:.1f}")
    else:
        parts.append("conditional does not resolve at any tested Delta_beta")

    # -- Marginal verdict --
    if marg_threshold is not None:
        parts.append(f"marginal requires Delta_beta >= {marg_threshold:.1f}")
    else:
        parts.append("marginal does not resolve at any tested Delta_beta")

    # -- Decorrelated verdict --
    if decor_stable and min(r["delta_beta"] for r in decor_stable) == 0.0:
        parts.append("decorrelated is ALWAYS stable (changes the problem, not a valid fix)")
    elif decor_stable:
        decor_threshold = min(r["delta_beta"] for r in decor_stable)
        parts.append(f"decorrelated resolves at Delta_beta >= {decor_threshold:.1f}")
    else:
        parts.append("decorrelated does not resolve at any tested Delta_beta")

    return "; ".join(parts)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PAPER_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_PATH = os.path.join(PAPER_DIR, "results_causal_dag.json")

    print("=== Causal DAG Experiment — Conditional SHAP Impossibility Validation ===")
    print(f"N_SAMPLES={N_SAMPLES}, N_MODELS={N_MODELS}, N_EVAL={N_EVAL}, rho={RHO}")
    print(f"sigma_noise={SIGMA_NOISE}, n_jobs=1")
    print(f"Marginal:     TreeSHAP (tree_path_dependent) — standard")
    print(f"Conditional:  TreeSHAP (interventional with background data) — "
          f"{'available' if _INTERVENTIONAL_AVAILABLE else 'UNAVAILABLE, using fallback'}")
    print(f"Decorrelated: residualized X2|X1 then TreeSHAP — contrast only (changes problem)")

    rows = run_experiment()

    verdict = compute_verdict(rows)
    print(f"\n=== Verdict ===")
    print(f"  {verdict}")

    # ── Check Delta_beta = 0 specifically ─────────────────────────────────────
    zero_row = [r for r in rows if r["delta_beta"] == 0.0]
    if zero_row:
        r = zero_row[0]
        print(f"\n  At Delta_beta = 0 (equal effects):")
        print(f"    Marginal flip rate:     {r['marginal_flip_rate']:.4f}")
        print(f"    Conditional flip rate:  {r['conditional_flip_rate']:.4f}")
        print(f"    Decorrelated flip rate: {r['decorrelated_flip_rate']:.4f}")
        if r["conditional_flip_rate"] >= STABLE_THRESHOLD:
            print(f"    -> Impossibility holds: conditional SHAP cannot escape when "
                  f"causal effects are equal")
        if r["decorrelated_flip_rate"] < STABLE_THRESHOLD:
            print(f"    -> Decorrelated is stable but INVALID: residualization changes "
                  f"effective coefficients")

    # ── Save results ──────────────────────────────────────────────────────────
    output = dict(
        config=dict(
            n_samples=N_SAMPLES,
            n_models=N_MODELS,
            n_eval=N_EVAL,
            rho=RHO,
            sigma_noise=SIGMA_NOISE,
            stable_threshold=STABLE_THRESHOLD,
            methods=["marginal (tree_path_dependent)",
                     "conditional (interventional with background)",
                     "decorrelated (residualized, contrast only)"],
            interventional_available=_INTERVENTIONAL_AVAILABLE,
        ),
        rows=rows,
        verdict=verdict,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
