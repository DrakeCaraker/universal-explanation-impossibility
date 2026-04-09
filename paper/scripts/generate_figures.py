"""
generate_figures.py — Generate publication-quality figures for the DASH Impossibility paper.

Produces three PDFs:
  paper/figures/ratio_divergence.pdf   — Figure 1a: Attribution ratio vs rho
  paper/figures/shap_instability.pdf   — Figure 1b: Ranking instability vs rho
  paper/figures/dash_resolution.pdf    — Figure 1c: DASH convergence

Usage:
  python paper/scripts/generate_figures.py
  (run from the dash-impossibility-lean repo root)

Data source: ~/ds_projects/dash-shap/results/tables/synthetic_linear_sweep.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.figsize': (3.2, 2.4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.4,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'axes.formatter.use_mathtext': True,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.expanduser(
    "~/ds_projects/dash-shap/results/tables/synthetic_linear_sweep.json"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH) as f:
    raw = json.load(f)

RHO_KEYS = ["0.0", "0.5", "0.7", "0.9", "0.95"]
RHOS = [float(r) for r in RHO_KEYS]

SB = "Single Best"
DASH = "DASH (MaxMin)"


# ---------------------------------------------------------------------------
# Helper: extract scalar metrics across rho values
# ---------------------------------------------------------------------------

def get_metric(method, key, se_key=None):
    vals, ses = [], []
    for rk in RHO_KEYS:
        vals.append(raw[rk][method][key])
        if se_key:
            ses.append(raw[rk][method][se_key])
    return (np.array(vals), np.array(ses)) if se_key else np.array(vals)


# ---------------------------------------------------------------------------
# Figure 1a: Attribution ratio vs rho
# ---------------------------------------------------------------------------

def fig_ratio_divergence():
    rho_dense = np.linspace(0.0, 0.98, 300)
    theory = 1.0 / (1.0 - rho_dense ** 2)

    # Load within-group validation results if available
    val_path = os.path.join(os.path.dirname(__file__), "..", "results_validation.json")
    has_validation = os.path.exists(val_path)

    fig, ax = plt.subplots()

    ax.plot(rho_dense, theory, color="#d62728", linestyle="--", lw=1.0,
            label=r"Idealized ($\alpha{=}1$): $1/(1-\rho^2)$", zorder=2)

    # Corrected curve: 1/(1 - alpha * rho^2) with alpha = 2/pi (fitted ≈ 0.60)
    ALPHA_FIT = 2.0 / np.pi  # 0.637; fitted value is 0.60
    corrected = 1.0 / (1.0 - ALPHA_FIT * rho_dense ** 2)
    ax.plot(rho_dense, corrected, color="#2ca02c", linestyle="-", lw=1.4,
            label=r"Corrected ($\alpha{=}2/\pi$): $1/(1-\alpha\rho^2)$", zorder=3)

    if has_validation:
        with open(val_path) as vf:
            val = json.load(vf)
        # Extract within-group split count ratios from validation experiments
        val_rhos, val_means, val_ses = [], [], []
        for key, data in val.items():
            if key.startswith("depth1_rho"):
                val_rhos.append(data["rho"])
                val_means.append(data["split_ratio_mean"])
                val_ses.append(data["split_ratio_se"])
        # Sort by rho
        order = np.argsort(val_rhos)
        val_rhos = np.array(val_rhos)[order]
        val_means = np.array(val_means)[order]
        val_ses = np.array(val_ses)[order]

        ax.errorbar(val_rhos, val_means, yerr=val_ses, fmt="s", color="#1f77b4",
                    markersize=4, capsize=3, lw=1.0,
                    label="Within-group ratio (XGBoost stumps)", zorder=4)
    else:
        # Fallback: use dash-shap data (global top-1/top-2, less precise)
        emp_means, emp_ses = [], []
        for rk, rho_f in zip(RHO_KEYS, RHOS):
            imp = np.array(raw[rk][SB]["imp_runs"])
            seed_ratios = []
            for row in imp:
                sorted_row = np.sort(row)[::-1]
                if sorted_row[1] > 1e-8:
                    seed_ratios.append(sorted_row[0] / sorted_row[1])
            emp_means.append(np.mean(seed_ratios))
            emp_ses.append(np.std(seed_ratios) / np.sqrt(len(seed_ratios)))
        ax.errorbar(RHOS, np.array(emp_means), yerr=np.array(emp_ses),
                    fmt="o", color="#1f77b4", markersize=4, capsize=3, lw=1.0,
                    label="Empirical (mean ± SE)", zorder=4)

    ax.set_xlabel(r"Feature correlation $\rho$")
    ax.set_ylabel("Within-group attribution ratio")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(bottom=0.8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=6))
    ax.grid(True, linewidth=0.3, alpha=0.25, color='#cccccc')
    ax.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ratio_divergence.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 1b: Ranking instability vs rho
# ---------------------------------------------------------------------------

def fig_shap_instability():
    # instability = 1 - stability
    sb_instab, sb_se = get_metric(SB, "stability", "stability_se")
    sb_instab = 1.0 - sb_instab
    # SE of (1-x) is the same as SE of x
    dash_instab, dash_se = get_metric(DASH, "stability", "stability_se")
    dash_instab = 1.0 - dash_instab

    fig, ax = plt.subplots()

    ax.errorbar(RHOS, sb_instab, yerr=sb_se, fmt="s-", color="#d62728",
                markersize=4, capsize=3, lw=1.2, label="Single model")
    ax.errorbar(RHOS, dash_instab, yerr=dash_se, fmt="o-", color="#1f77b4",
                markersize=4, capsize=3, lw=1.2, label="DASH ensemble")

    ax.set_xlabel(r"Feature correlation $\rho$")
    ax.set_ylabel(r"Ranking instability ($1 - S$)")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linewidth=0.3, alpha=0.25, color='#cccccc')
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "shap_instability.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 1c: DASH convergence
# ---------------------------------------------------------------------------

def fig_dash_resolution():
    # Use imp_runs from the highest-correlation setting (rho=0.95) to show
    # convergence most dramatically. Each row is a seed; columns are
    # individual ensemble member importances. We compute the coefficient of
    # variation of the running mean across outer seeds as M grows.
    rho_key = "0.95"
    imp_sb = np.array(raw[rho_key][SB]["imp_runs"])     # (50, 50)
    imp_dash = np.array(raw[rho_key][DASH]["imp_runs"])  # (50, 50)

    Ms = np.array([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50])
    Ms = Ms[Ms <= imp_sb.shape[1]]

    def convergence_cv(imp, Ms):
        """Coefficient of variation of running-mean importance across outer seeds."""
        cvs = []
        for M in Ms:
            ensemble_means = imp[:, :M].mean(axis=1)
            cv = ensemble_means.std() / (ensemble_means.mean() + 1e-12)
            cvs.append(cv)
        return np.array(cvs)

    cv_sb = convergence_cv(imp_sb, Ms)
    cv_dash = convergence_cv(imp_dash, Ms)

    # Theoretical O(1/M) curve, anchored to single-model value
    theory_scale = cv_sb[0]
    theory_curve = theory_scale / np.sqrt(Ms)

    fig, ax = plt.subplots()

    ax.plot(Ms, theory_curve, color="#7f7f7f", linestyle=":", lw=1.2,
            label=r"Theory: $O(1/\sqrt{M})$", zorder=2)
    ax.plot(Ms, cv_sb, "s-", color="#d62728", markersize=4, lw=1.2,
            label="Single model", zorder=3)
    ax.plot(Ms, cv_dash, "o-", color="#1f77b4", markersize=4, lw=1.2,
            label="DASH ensemble", zorder=3)

    ax.axhline(0.01, color="#2ca02c", linestyle="--", lw=0.9,
               label=r"1\% threshold", zorder=2)

    ax.set_xlabel(r"Ensemble size $M$")
    ax.set_ylabel("Attribution CV (within-group)")
    ax.set_xlim(0, Ms.max() + 2)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.3, alpha=0.25, color='#cccccc')
    ax.legend(loc="upper right", fontsize=6.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dash_resolution.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_ratio_divergence()
    fig_shap_instability()
    fig_dash_resolution()
    print("All figures generated.")
