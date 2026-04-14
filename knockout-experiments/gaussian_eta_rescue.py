#!/usr/bin/env python3
"""
Phase 2: Rescue the universal eta plot using Gaussian-predicted mean instability.

For the 7 "well-characterized" instances (SHAP, codon x3, stat mech, concept probe,
parser), keep the theoretical eta prediction -- these already had R^2 = 0.957.

For the 9 "approximate" instances, attempt to compute a Gaussian-predicted mean
instability from the raw experimental data instead of the group-theoretic
dim(V^G)/dim(V) prediction.

Gaussian flip rate formula:
  predicted_flip(Delta, sigma) = 2 * Phi(Delta / sigma) * Phi(-Delta / sigma)
where Delta = |mean_diff| across models, sigma = std_diff across models.

The key test: what R^2 do we get across all 16 domains when we use
Gaussian-corrected predictions for the approximate instances?
"""

import json
import numpy as np
from scipy.stats import norm
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BASE_DIR / "paper"
KO_DIR = Path(__file__).resolve().parent
OUT_DIR = KO_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_FILE = PAPER_DIR / "scripts" / "publication_style.mplstyle"


def load_paper(name):
    p = PAPER_DIR / name
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


def load_ko(name):
    p = KO_DIR / name
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load existing universal eta data
# ---------------------------------------------------------------------------
eta_data = load_ko("results_universal_eta.json")
points_original = eta_data["points"]

# Build a lookup by domain name
orig_by_domain = {p["domain"]: p for p in points_original}

# ---------------------------------------------------------------------------
# Define well-characterized vs approximate instances
# ---------------------------------------------------------------------------
WELL_CHARACTERIZED = [
    "Attribution (SHAP, $S_2$)",
    "Codon ($S_2$)",
    "Codon ($S_4$)",
    "Codon ($S_6$)",
    r"Stat mech ($S_{252}$, $N$=10)",
    r"Concept probe (TCAV, $O(64)$)",
    "Parser (PP-attach, $S_2$)",
]

APPROXIMATE = [
    r"Attention (argmax, $S_6$)",
    r"Counterfactual (direction, $\mathbb{Z}_2$)",
    r"Model selection ($S_{11}$ winners)",
    r"GradCAM (peak, $1{-}\mathrm{IoU}$)",
    r"LLM citation (token, $S_3$)",
    r"Gauge lattice ($\mathbb{Z}_2^{L^2}$, $\beta$=1)",
    r"Linear solver ($d$=20, $m$=10)",
    r"Phase retrieval ($U(1)^N$, $N$=128)",
    r"Causal discovery ($\mathbb{Z}_2^4$, Asia)",
]


def gaussian_flip_rate(delta, sigma):
    """Gaussian flip rate: P(sign flip) = 2 * Phi(d/s) * Phi(-d/s)."""
    if sigma <= 0 or np.isnan(sigma) or np.isnan(delta):
        return None
    snr = abs(delta) / sigma
    return 2 * norm.cdf(snr) * norm.cdf(-snr)


# ---------------------------------------------------------------------------
# Attempt Gaussian prediction for each approximate instance
# ---------------------------------------------------------------------------
print("=" * 72)
print("GAUSSIAN ETA RESCUE: per-domain analysis")
print("=" * 72)

gaussian_predictions = {}

# ---- Attention (argmax, S_6) ----
# We have 10 perturbed models, each giving attention argmax per sentence.
# The flip rate is 0.60. The group prediction is 5/6 = 0.833.
# We do NOT have per-feature importance vectors to compute SNR.
# We only have a summary flip rate. We cannot compute Gaussian prediction.
# BUT: the attention flip rate IS itself the downstream observable. The issue
# is the GROUP prediction overshoots. Without per-pair importance data for
# a continuous Gaussian model, we keep the original or use the observed as-is.
print("\n[Attention] No per-feature importance data for Gaussian model.")
print("  Original pred=0.833, obs=0.600. Cannot improve via Gaussian formula.")
gaussian_predictions["Attention (argmax, $S_6$)"] = None

# ---- Counterfactual (direction, Z_2) ----
# We have top-10 feature flip rates. Each is a per-feature disagreement rate.
# The overall direction_flip_rate.mean = 0.235, top-10 mean ~ 0.367.
# The group prediction is 0.5 (Z_2). The top-10 flip rates are individual
# feature-level flip rates, not a Gaussian SNR computation.
# BUT we can treat each feature's flip rate as an "observed per-pair rate"
# and compute: if the flip rates are Bernoulli draws, the mean flip rate
# across features IS the Gaussian prediction when SNR varies per feature.
# Actually: the top-10 flip rates give us mean(flip_rate) for the most
# ambiguous features. This is already the "Gaussian-predicted" instability
# for those features (the flip rate IS the instability).
# The observed instability used in the original plot was the top-10 mean.
cf = load_paper("results_counterfactual_instability.json")
if cf:
    top10_rates = list(cf["top10_flip_rates"].values())
    mean_top10 = np.mean(top10_rates)
    overall_flip = cf["direction_flip_rate"]["mean"]
    print(f"\n[Counterfactual] top-10 flip rates: mean={mean_top10:.4f}")
    print(f"  Overall direction flip rate: {overall_flip:.4f}")
    print(f"  Original pred=0.500, obs={orig_by_domain[APPROXIMATE[1]]['observed_instability']:.4f}")
    print("  Cannot compute Gaussian SNR without per-pair importance vectors.")
gaussian_predictions[APPROXIMATE[1]] = None

# ---- Model selection (S_11 winners) ----
# We have per-model AUC means and per-split winners. The group prediction is
# 1 - 1/11 = 0.909. The observed is 0.80.
# Can we compute Gaussian prediction? We have 50 model AUCs and 20 splits.
# The "AUC gap" between models could give us SNR. But this is model selection
# instability (which model wins), not a continuous importance metric.
ms = load_paper("results_model_selection_instability.json")
if ms:
    auc_per_model = np.array(ms["mean_auc_per_model"])
    auc_std = np.std(auc_per_model)
    auc_spread = ms["mean_auc_spread_per_split"]
    # The instability comes from auc_spread / auc_std being small (models are close)
    # Gaussian prediction: P(different winner) ~ 1 - expected_margin / noise
    # With 11 unique winners out of 50, the "effective group" is S_11.
    # The per-split AUC spread (~0.02) vs overall AUC std (~0.002) gives SNR.
    # Actually the key: on each split, the gap between #1 and #2 model is tiny.
    # If gap ~ N(delta, sigma), flip rate = 2*Phi(delta/sigma)*Phi(-delta/sigma).
    # But we don't have per-split #1-#2 gap data directly. Skip.
    print(f"\n[Model selection] AUC std across models: {auc_std:.6f}")
    print(f"  Mean AUC spread per split: {auc_spread:.5f}")
    print("  Cannot compute Gaussian prediction without per-split gap data.")
gaussian_predictions[APPROXIMATE[2]] = None

# ---- GradCAM (peak, 1-IoU) ----
# pred=0.046, obs=0.096. The group prediction used IoU-based computation.
# We have IoU data but not per-pixel importance vectors for Gaussian formula.
gc = load_paper("results_gradcam_instability.json")
if gc:
    print(f"\n[GradCAM] IoU={gc['positive']['mean_iou']:.4f}, flip={gc['positive']['flip_rate']:.4f}")
    print("  Cannot compute Gaussian prediction without per-pixel data.")
gaussian_predictions[APPROXIMATE[3]] = None

# ---- LLM citation (token, S_3) ----
# pred=0.667, obs=0.345. Group prediction is 1-1/3 = 0.667.
# Only have summary flip rate and Jaccard. No per-token importance.
llm = load_paper("results_llm_explanation_instability.json")
if llm:
    print(f"\n[LLM citation] flip_rate={llm['positive_test']['flip_rate_ci'][1]:.4f}")
    print("  Cannot compute Gaussian prediction without per-token data.")
gaussian_predictions[APPROXIMATE[4]] = None

# ---- Gauge lattice (Z_2^{L^2}, beta=1) ----
# pred=0.5, obs=0.411. The pred used sech^2(beta) fraction.
# We have plaquette variance at beta=1: analytic = 0.001640.
# The gauge prediction is actually analytical. The variance fraction
# Var(P)/Var_max gives the instability fraction.
# At beta=1: Var(P) = sech^2(1) / L^2 = 0.4200 / 256 = 0.001640
# Var_max = sech^2(0) / L^2 = 1/256 = 0.003906
# fraction = 0.001640 / 0.003906 = 0.4200 = sech^2(1)
gauge = load_paper("results_gauge_lattice.json")
if gauge:
    beta_idx = gauge["beta_values"].index(1.0)
    obs_var = gauge["plaquette_variance"][beta_idx]
    max_var = gauge["plaquette_variance"][0]  # beta=0.1 is closest to 0
    # Actually max variance is at beta->0, which is sech^2(0)/L^2 = 1/L^2
    L = gauge["lattice_size"]
    analytic_max_var = 1.0 / (L * L)  # sech^2(0) = 1
    analytic_var_b1 = gauge["analytic_plaquette_variance"][beta_idx]
    variance_fraction = analytic_var_b1 / analytic_max_var
    print(f"\n[Gauge lattice] analytic_var(beta=1)={analytic_var_b1:.6f}")
    print(f"  analytic_max_var={analytic_max_var:.6f}")
    print(f"  variance_fraction = {variance_fraction:.4f}")
    print(f"  sech^2(1) = {1/np.cosh(1)**2:.4f}")
    print(f"  Original pred=0.500, obs={orig_by_domain[APPROXIMATE[5]]['observed_instability']:.4f}")
    # The variance fraction IS the Gaussian-predicted instability:
    # it measures what fraction of the "explanation freedom" is exercised.
    # Use variance_fraction as the Gaussian-corrected prediction.
    gaussian_predictions[APPROXIMATE[5]] = variance_fraction

# ---- Linear solver (d=20, m=10) ----
# pred=0.5 (null_dim/d = 10/20), obs=0.697 (normalized RMSD).
# The RMSD data shows a clear dose-response curve with d.
# At d=20 (null_dim=10), the mean RMSD is 0.0914 (from per_d["20"]).
# The normalization used RMSD(d=11) as reference... but the observed
# instability of 0.697 is already normalized.
# The group prediction of 0.5 is too low. Can we get a better Gaussian prediction?
# The linear solver has analytic theory: RMSD ~ sqrt(null_dim/d) for pseudoinverse
# vs random_null. For d=20, m=10: null_dim=10, RMSD ~ sqrt(10/20) = 0.707.
# That's actually very close to the observed 0.697!
ls = load_paper("results_linear_solver.json")
if ls:
    d20_data = ls["per_d"]["20"]
    print(f"\n[Linear solver] d=20 mean RMSD: {d20_data['mean_rmsd']:.6f}")
    # The RMS disagreement for Gaussian vectors in a d-dim null space:
    # E[RMSD^2] = 2*(1 - null_dim/d) for unit-norm solutions... no.
    # For the pseudoinverse vs random_null component:
    # The random_null adds a component in the null space with norm ~ sqrt(null_dim/d).
    # The expected RMSD between pseudoinverse and random = sqrt(2) * sigma_null.
    # More precisely: for underdetermined Ax=b with null_dim = d-m,
    # solutions differ by null-space vectors. The RMSD between two
    # independent projections is sqrt(2 * null_dim / d) for unit-variance.
    # At d=20, m=10: sqrt(2*10/20) = 1.0. But observed is 0.091...
    # The normalization matters. The observed 0.697 was already normalized.
    # Actually, looking at results_universal_eta.json:
    # observed_instability = 0.6972, and notes say "RMSD normalized by RMSD(d=11)".
    # So the normalization is: obs = RMSD(d=20) / RMSD(d=11, m=10) where d=11
    # means null_dim=1 (minimal underdetermination).
    #
    # Gaussian theory: RMSD ~ sqrt(null_dim) for fixed m, so:
    # RMSD(d=20) / RMSD(d=11) = sqrt(10) / sqrt(1) = 3.16
    # But observed ratio is 0.697... that doesn't match. The normalization
    # must be different. Let's compute from raw data:
    d11_rmsd = ls["per_d"]["11"]["mean_rmsd"]  # d-m=1
    d20_rmsd = d20_data["mean_rmsd"]
    raw_ratio = d20_rmsd / d11_rmsd if d11_rmsd > 0 else None
    print(f"  RMSD(d=11)={d11_rmsd:.6f}, ratio d20/d11 = {raw_ratio:.4f}")
    # OK, 0.0914 / 0.1098 = 0.832. But observed_instability is 0.697.
    # The normalization in universal_eta_synthesis.py was different.
    # Actually the note says "RMSD normalized by RMSD(d=11)". So observed=0.697
    # means... Actually let me just leave this as-is. The Gaussian formula
    # requires per-pair importance vectors, which we don't have for linear solvers.
    print("  Cannot compute Gaussian SNR prediction; this is RMSD-based.")
gaussian_predictions[APPROXIMATE[6]] = None

# ---- Phase retrieval (U(1)^N, N=128) ----
# pred=0.5, obs=0.936. The group prediction 0.5 is from heuristic.
# The RMSD data is available per-reconstruction, but we can't compute
# Gaussian flip rates without per-element importance.
# However, the RMSD^2/2 normalization gives the fraction of phase disorder.
# The observed 0.936 suggests near-random phases (U(1) symmetry is large).
# A better analytic prediction: for N phases each uniform on [0, 2*pi),
# the expected RMSD^2 = 2 (normalized). So RMSD^2/2 ~ 1.0 for fully random.
# The positive control constrains phases, reducing RMSD.
# At N=128: general RMSD = 1.368, so RMSD^2/2 = 0.936.
# Better prediction: the "effective degrees of freedom" ratio.
# For phase retrieval with magnitude-only constraints, the phase freedom
# is approximately N-1 out of N (only global phase is constrained).
# So predicted instability = (N-1)/N = 127/128 = 0.992.
pr = load_paper("results_phase_retrieval.json")
if pr:
    rmsd_128 = pr["per_length"]["128"]["general"]["mean_pairwise_rmsd"]
    rmsd_sq_norm = rmsd_128**2 / 2
    predicted_dof = 127 / 128  # (N-1)/N
    print(f"\n[Phase retrieval] RMSD(N=128)={rmsd_128:.4f}, RMSD^2/2={rmsd_sq_norm:.4f}")
    print(f"  DOF prediction (N-1)/N = {predicted_dof:.4f}")
    print(f"  Original pred=0.500, obs={orig_by_domain[APPROXIMATE[7]]['observed_instability']:.4f}")
    # Use (N-1)/N as the improved analytic prediction
    gaussian_predictions[APPROXIMATE[7]] = predicted_dof

# ---- Causal discovery (Z_2^4, Asia) ----
# pred=0.5 (4/8 reversible edges), obs=0.710 (100-seed mean disagreement).
# The observed comes from 100-seed agreement data. The group prediction
# counts reversible edges but doesn't account for finite-sample effects.
# The 100-seed per_seed_agreements for small-N are available.
# Mean small-N agreement ~ 0.29, so disagreement ~ 0.71.
# The "Gaussian-corrected" prediction: with 3 methods and 8 edges,
# the finite-sample effects inflate disagreement beyond 4/8.
# Method: treat each edge's orientation as a Bernoulli(p) where p
# depends on the signal-to-noise ratio. For reversible edges p~0.5.
# For v-structure edges (4), the methods should agree (p~1.0 at large N).
# At small N, even v-structure edges may disagree. The 100-seed data
# shows mean agreement = 0.29 for small N (1000 samples).
cd = load_paper("results_causal_discovery_exp.json")
if cd:
    small_agreements = cd["statistical_test"]["n_small_per_seed_agreements"]
    mean_small_agreement = np.mean(small_agreements)
    mean_small_disagreement = 1 - mean_small_agreement
    large_agreements = cd["statistical_test"]["n_large_per_seed_agreements"]
    mean_large_agreement = np.mean(large_agreements)

    # At large N, agreement = 0.621 (reversible edges still cause disagreement).
    # The structural disagreement floor = 1 - mean_large_agreement
    structural_disagreement = 1 - mean_large_agreement
    # At small N, additional noise pushes disagreement higher.
    # The "Gaussian prediction" could use: structural + noise component
    # But we don't have the per-edge SNR data to compute this.
    print(f"\n[Causal discovery] mean small-N agreement: {mean_small_agreement:.4f}")
    print(f"  mean large-N agreement: {mean_large_agreement:.4f}")
    print(f"  structural disagreement floor: {structural_disagreement:.4f}")
    print(f"  Original pred=0.500, obs={orig_by_domain[APPROXIMATE[8]]['observed_instability']:.4f}")
    # Use the large-N structural disagreement as a better "theory" prediction
    # (this removes finite-sample noise and isolates the Markov equivalence effect)
    # Actually: the large-N disagreement = 0.375 = 3/8 (undirected edges stay
    # undirected). The small-N disagreement of 0.71 includes both structural
    # and finite-sample noise. Cannot cleanly separate with Gaussian formula.
gaussian_predictions[APPROXIMATE[8]] = None


# ---------------------------------------------------------------------------
# Alternative approach: use variance-based corrections where possible
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("BUILDING CORRECTED PREDICTION SET")
print("=" * 72)

# For instances where we CAN'T compute Gaussian predictions, try a different
# strategy: use the "attenuation factor" approach.
# The group-theoretic prediction eta = 1 - dim(V^G)/dim(V) assumes ALL symmetry
# orbits are equally populated. In practice, some configurations are more likely
# than others, attenuating the instability.
# The attenuation factor alpha is: observed / predicted (from well-characterized set).
# For the well-characterized set, alpha ~ 0.85-0.95 (slight attenuation from
# non-uniform orbit weights).

# Strategy: for each approximate instance, check if the ratio obs/pred suggests
# a systematic bias we can correct. But this is post-hoc curve fitting!

# THE HONEST APPROACH: only change predictions where we have principled Gaussian
# corrections. For the rest, keep the original prediction.

corrected_points = []
corrections_made = {}

for p in points_original:
    domain = p["domain"]
    new_p = dict(p)  # copy

    if domain in WELL_CHARACTERIZED:
        # Keep as-is
        new_p["correction"] = "none (well-characterized)"
        corrected_points.append(new_p)
        continue

    # Check if we have a Gaussian correction
    gauss_pred = gaussian_predictions.get(domain)
    if gauss_pred is not None:
        old_pred = p["predicted_instability"]
        new_p["predicted_instability"] = gauss_pred
        new_p["correction"] = f"gaussian_corrected (was {old_pred:.4f})"
        corrections_made[domain] = {
            "old_pred": old_pred,
            "new_pred": gauss_pred,
            "observed": p["observed_instability"],
        }
        print(f"  {domain}: pred {old_pred:.4f} -> {gauss_pred:.4f} (obs={p['observed_instability']:.4f})")
    else:
        new_p["correction"] = "none (no Gaussian data available)"
        corrected_points.append(new_p)
        continue

    corrected_points.append(new_p)


# ---------------------------------------------------------------------------
# Compute R^2 for original and corrected sets
# ---------------------------------------------------------------------------
def compute_r2(points_list):
    x = np.array([p["predicted_instability"] for p in points_list])
    y = np.array([p["observed_instability"] for p in points_list])
    # OLS fit
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = result[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2, slope, intercept, x, y


r2_orig, slope_orig, intercept_orig, x_orig, y_orig = compute_r2(points_original)
r2_corr, slope_corr, intercept_corr, x_corr, y_corr = compute_r2(corrected_points)

print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)
print(f"Original R^2 (all 16):   {r2_orig:.4f}  (slope={slope_orig:.3f}, intercept={intercept_orig:.3f})")
print(f"Corrected R^2 (all 16):  {r2_corr:.4f}  (slope={slope_corr:.3f}, intercept={intercept_corr:.3f})")
print(f"Improvement:             {r2_corr - r2_orig:+.4f}")


# Also compute R^2 for well-characterized subset only
wc_points = [p for p in corrected_points if p["domain"] in WELL_CHARACTERIZED]
r2_wc, slope_wc, intercept_wc, _, _ = compute_r2(wc_points)
print(f"\nWell-characterized R^2:  {r2_wc:.4f}  (n={len(wc_points)})")

# And for approximate subset only
approx_points = [p for p in corrected_points if p["domain"] in APPROXIMATE]
r2_approx, slope_approx, intercept_approx, _, _ = compute_r2(approx_points)
print(f"Approximate R^2:         {r2_approx:.4f}  (n={len(approx_points)})")

# Per-point residuals
print("\nPer-point residuals (corrected):")
print(f"  {'Domain':<48s} {'pred':>6s} {'obs':>6s} {'resid':>7s}")
for p in sorted(corrected_points, key=lambda p: abs(p["observed_instability"] - p["predicted_instability"]), reverse=True):
    pred = p["predicted_instability"]
    obs = p["observed_instability"]
    resid = obs - pred
    marker = " ***" if abs(resid) > 0.20 else ""
    print(f"  {p['domain']:<48s} {pred:>6.3f} {obs:>6.3f} {resid:>+7.3f}{marker}")

# Identify the worst outliers
worst = sorted(corrected_points, key=lambda p: abs(p["observed_instability"] - p["predicted_instability"]), reverse=True)
print(f"\nWorst outlier: {worst[0]['domain']}")
print(f"  pred={worst[0]['predicted_instability']:.4f}, obs={worst[0]['observed_instability']:.4f}")
print(f"  residual={worst[0]['observed_instability'] - worst[0]['predicted_instability']:+.4f}")


# ---------------------------------------------------------------------------
# Additional analysis: What R^2 would we get if we had perfect predictions
# for the 2 Gaussian-corrected instances?
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("COUNTERFACTUAL ANALYSIS")
print("=" * 72)

# What if we set pred=obs for the worst outliers?
# This tells us the "ceiling" R^2 if we could perfectly predict those instances.
for n_fix in [1, 2, 3, 5]:
    fixed_points = list(corrected_points)
    for i, w in enumerate(worst[:n_fix]):
        idx = next(j for j, p in enumerate(fixed_points) if p["domain"] == w["domain"])
        fixed_points[idx] = dict(fixed_points[idx])
        fixed_points[idx]["predicted_instability"] = fixed_points[idx]["observed_instability"]
    r2_fix, _, _, _, _ = compute_r2(fixed_points)
    domains_fixed = ", ".join(w["domain"][:20] for w in worst[:n_fix])
    print(f"  Fix {n_fix} worst outlier(s): R^2 = {r2_fix:.4f}  [{domains_fixed}]")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if STYLE_FILE.exists():
    plt.style.use(str(STYLE_FILE))

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Color coding
WC_COLOR = '#2166AC'  # blue
APPROX_COLOR = '#B2182B'  # red
CORRECTED_COLOR = '#4DAF4A'  # green

for ax_idx, (ax, pts, title, r2_val, slope_val, intercept_val) in enumerate([
    (axes[0], points_original, f"Original ($R^2={r2_orig:.3f}$)", r2_orig, slope_orig, intercept_orig),
    (axes[1], corrected_points, f"Gaussian-corrected ($R^2={r2_corr:.3f}$)", r2_corr, slope_corr, intercept_corr),
]):
    # Plot diagonal
    ax.plot([0, 1.05], [0, 1.05], 'k--', alpha=0.3, lw=1, label='$y = x$')

    # Plot regression line
    xx = np.linspace(0, 1.05, 100)
    ax.plot(xx, slope_val * xx + intercept_val, 'k-', alpha=0.5, lw=1.5,
            label=f'OLS: $y = {slope_val:.2f}x + {intercept_val:.2f}$')

    for p in pts:
        is_wc = p["domain"] in WELL_CHARACTERIZED
        color = WC_COLOR if is_wc else APPROX_COLOR
        marker = 'o' if is_wc else 's'
        size = 60 if is_wc else 50

        # If this is the corrected panel and this point was corrected, use green
        if ax_idx == 1 and p.get("correction", "").startswith("gaussian"):
            color = CORRECTED_COLOR
            marker = 'D'
            size = 70

        ax.scatter(p["predicted_instability"], p["observed_instability"],
                   c=color, marker=marker, s=size, zorder=5, edgecolors='k', linewidths=0.5)

        # Add error bars if available
        if p["ci_lo"] is not None and p["ci_hi"] is not None:
            ax.errorbar(p["predicted_instability"], p["observed_instability"],
                        yerr=[[p["observed_instability"] - p["ci_lo"]],
                              [p["ci_hi"] - p["observed_instability"]]],
                        fmt='none', ecolor=color, alpha=0.4, capsize=2)

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Predicted instability ($\\hat{\\eta}$)")
    ax.set_ylabel("Observed instability")
    ax.set_title(title)
    ax.set_aspect('equal')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=WC_COLOR,
               markersize=8, markeredgecolor='k', markeredgewidth=0.5,
               label='Well-characterized'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=APPROX_COLOR,
               markersize=8, markeredgecolor='k', markeredgewidth=0.5,
               label='Approximate'),
    ]
    if ax_idx == 1:
        handles.append(
            Line2D([0], [0], marker='D', color='w', markerfacecolor=CORRECTED_COLOR,
                   markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                   label='Gaussian-corrected')
        )
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "gaussian_eta_rescue.pdf", bbox_inches='tight', dpi=150)
print(f"\nFigure saved: {OUT_DIR / 'gaussian_eta_rescue.pdf'}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "description": "Phase 2: Gaussian eta rescue attempt",
    "approach": "Replace group-theoretic predictions with Gaussian-predicted mean instability where per-pair data is available",
    "original_R2": r2_orig,
    "corrected_R2": r2_corr,
    "improvement": r2_corr - r2_orig,
    "well_characterized_R2": r2_wc,
    "approximate_R2": r2_approx,
    "n_corrections_made": len(corrections_made),
    "corrections": corrections_made,
    "points": [
        {
            "domain": p["domain"],
            "predicted_instability": p["predicted_instability"],
            "observed_instability": p["observed_instability"],
            "ci_lo": p.get("ci_lo"),
            "ci_hi": p.get("ci_hi"),
            "correction": p.get("correction", "none"),
        }
        for p in corrected_points
    ],
    "diagnosis": {
        "worst_outliers": [
            {
                "domain": w["domain"],
                "predicted": w["predicted_instability"],
                "observed": w["observed_instability"],
                "residual": w["observed_instability"] - w["predicted_instability"],
            }
            for w in worst[:5]
        ],
        "problem_summary": (
            "The Gaussian formula requires per-pair importance vectors (Delta, sigma per feature). "
            "Most approximate instances only provide SUMMARY flip rates, not the raw importance data "
            "needed to compute SNR. Only gauge lattice (variance fraction) and phase retrieval "
            "(DOF fraction) could be corrected with principled predictions. "
            "The core issue: 7 of 9 approximate instances lack the per-pair importance data "
            "needed for Gaussian correction. The group-theoretic prediction remains the only "
            "available framework for these domains."
        ),
        "rescue_verdict": "PARTIAL" if r2_corr > r2_orig else "FAILED",
    },
}

with open(KO_DIR / "results_gaussian_eta_rescue.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {KO_DIR / 'results_gaussian_eta_rescue.json'}")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
print(f"Original R^2:  {r2_orig:.4f}")
print(f"Corrected R^2: {r2_corr:.4f}")
print(f"Change:        {r2_corr - r2_orig:+.4f}")
if r2_corr > 0.7:
    print("RESCUE SUCCEEDED: R^2 > 0.70")
elif r2_corr > r2_orig:
    print("PARTIAL IMPROVEMENT: R^2 increased but still below target")
else:
    print("RESCUE FAILED: Gaussian correction did not improve R^2")
print(f"\nThe fundamental limitation: {7} of {9} approximate instances lack per-pair")
print("importance data. The Gaussian formula requires (Delta, sigma) per feature,")
print("but these experiments only recorded summary statistics (mean flip rate, RMSD).")
print("\nTo fully rescue the universal eta plot, future experiments should record")
print("per-pair importance vectors, enabling Gaussian flip rate prediction.")
