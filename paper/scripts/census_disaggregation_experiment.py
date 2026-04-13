"""
Task 1.4: Computer Science — Census Disaggregation (Synthetic Fallback)

Demonstrates that disaggregating a coarse aggregate (state total) into
fine-grained components (county populations) is fundamentally underspecified:
the more counties a state has, the higher the KL-divergence between
Dirichlet samples and the true distribution.

Uses synthetic data: 50 "states" with 1–100 "counties" each.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.special import rel_entr

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
    PAPER_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SEED             = 42
N_STATES         = 50
MIN_COUNTIES     = 1
MAX_COUNTIES     = 100
TOTAL_POP        = 1_000_000
N_DIRICHLET_SAMP = 100
DIRICHLET_ALPHA  = 1.0      # symmetric (uniform) prior — maximum entropy
EPS              = 1e-12    # numerical guard for KL

# ── Reproducibility ────────────────────────────────────────────────────────────
set_all_seeds(SEED)
load_publication_style()

# ── 1. Generate synthetic states ───────────────────────────────────────────────
rng = np.random.default_rng(SEED)

# Spread county counts from 1 to MAX_COUNTIES across N_STATES.
# Ensure at least one state has exactly 1 county (control).
county_counts = np.sort(
    rng.integers(MIN_COUNTIES, MAX_COUNTIES + 1, size=N_STATES)
)
county_counts[0] = 1  # guarantee the n=1 control point

states = []
for i, n_counties in enumerate(county_counts):
    # True county populations ~ Dirichlet(alpha) × TOTAL_POP
    alpha_vec   = np.full(n_counties, DIRICHLET_ALPHA)
    true_shares = rng.dirichlet(alpha_vec)
    true_pops   = true_shares * TOTAL_POP          # float counts
    state_total = true_pops.sum()                  # == TOTAL_POP by construction

    states.append({
        'state_id':    i,
        'n_counties':  int(n_counties),
        'true_shares': true_shares,
        'state_total': float(state_total),
    })

# ── 2. Disaggregation experiment ───────────────────────────────────────────────
# For each state: given ONLY the state total, generate N_DIRICHLET_SAMP
# Dirichlet samples and compute KL-divergence from the true distribution.

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) — KL from sample q to truth p."""
    p = np.clip(p, EPS, None);  p = p / p.sum()
    q = np.clip(q, EPS, None);  q = q / q.sum()
    return float(np.sum(rel_entr(p, q)))   # rel_entr(a,b) = a*log(a/b)

results_per_state = []
for s in states:
    n  = s['n_counties']
    p  = s['true_shares']

    if n == 1:
        # Only one valid disaggregation: the whole state → KL = 0
        kl_values = [0.0] * N_DIRICHLET_SAMP
    else:
        alpha_vec = np.full(n, DIRICHLET_ALPHA)
        samples   = rng.dirichlet(alpha_vec, size=N_DIRICHLET_SAMP)
        kl_values = [kl_divergence(p, q) for q in samples]

    mean_kl = float(np.mean(kl_values))
    results_per_state.append({
        'state_id':   s['state_id'],
        'n_counties': n,
        'mean_kl':    mean_kl,
        'std_kl':     float(np.std(kl_values)),
        'kl_values':  [float(v) for v in kl_values],
    })

# ── 3. Summary statistics ──────────────────────────────────────────────────────
n_vals  = np.array([r['n_counties'] for r in results_per_state])
kl_vals = np.array([r['mean_kl']    for r in results_per_state])

r_pearson, p_value = pearsonr(n_vals, kl_vals)

# Group by n_counties for display
unique_n = np.unique(n_vals)
group_mean = np.array([kl_vals[n_vals == n].mean() for n in unique_n])

# Fit simple OLS trend line (log-linear: KL ~ a + b*log(n))
log_n = np.log(np.maximum(n_vals, 1))
coeffs = np.polyfit(log_n, kl_vals, 1)
trend_logn = np.linspace(log_n.min(), log_n.max(), 200)
trend_kl   = np.polyval(coeffs, trend_logn)
trend_n    = np.exp(trend_logn)

# Overall CI on KL for n > 1
kl_noncontrol = kl_vals[n_vals > 1]
ci_lo, ci_mean, ci_hi = percentile_ci(kl_noncontrol, n_boot=2000)

print(f"States: {N_STATES}")
print(f"County counts range: {n_vals.min()} – {n_vals.max()}")
print(f"KL range: {kl_vals.min():.4f} – {kl_vals.max():.4f}")
print(f"Pearson r (n_counties vs KL): {r_pearson:.4f}  p={p_value:.4e}")
print(f"Mean KL (n>1): {ci_mean:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"n=1 control: KL = {kl_vals[n_vals == 1][0]:.4f}  (expected 0)")

# ── 4. Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# All states (non-control)
mask_ctrl  = n_vals == 1
mask_other = ~mask_ctrl

ax.scatter(
    n_vals[mask_other], kl_vals[mask_other],
    color='steelblue', alpha=0.65, s=40, zorder=3,
    label='Synthetic states'
)

# Highlight control point (n=1, KL=0)
ax.scatter(
    n_vals[mask_ctrl], kl_vals[mask_ctrl],
    color='firebrick', s=120, zorder=5, marker='*',
    label=r'Control: $n=1$ county (KL $= 0$)'
)

# Log-linear trend line
ax.plot(
    trend_n, trend_kl,
    color='darkorange', linewidth=2, zorder=4,
    label=f'Log-linear fit  ($r={r_pearson:.2f}$)'
)

ax.set_xlabel('Number of counties (disaggregation dimensionality)', fontsize=12)
ax.set_ylabel('Mean KL-divergence (sample ‖ truth)', fontsize=12)
ax.set_title(
    'Disaggregation Impossibility: KL-Divergence vs County Count\n'
    '(Synthetic census data, 50 states, 100 Dirichlet samples each)',
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=-0.05)

# Annotation
ax.annotate(
    f'Pearson $r = {r_pearson:.2f}$\n$p = {p_value:.2e}$',
    xy=(0.62, 0.15), xycoords='axes fraction',
    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)
)

plt.tight_layout()
save_figure(fig, 'census_disaggregation')

# ── 5. LaTeX table ────────────────────────────────────────────────────────────
# Bin states into groups and show mean KL per bin
bins      = [1, 2, 10, 25, 50, 75, 100]
bin_labels = ['$n=1$', '$2$–$9$', '$10$–$24$', '$25$–$49$', '$50$–$74$', '$75$–$100$']
bin_means, bin_stds, bin_ns = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask  = (n_vals >= lo) & (n_vals < hi)
    if lo == 1:
        mask = n_vals == 1
    vals  = kl_vals[mask]
    bin_means.append(vals.mean() if len(vals) > 0 else float('nan'))
    bin_stds.append(vals.std()  if len(vals) > 0 else float('nan'))
    bin_ns.append(len(vals))

sections_dir = PAPER_DIR / 'sections'
sections_dir.mkdir(exist_ok=True)
tex_path = sections_dir / 'table_census.tex'

with open(tex_path, 'w') as f:
    f.write(r"""\begin{table}[ht]
\centering
\caption{%
  Mean KL-divergence from Dirichlet-sampled disaggregations to the
  true county distribution, grouped by number of counties.
  The $n=1$ control has KL$\,{=}\,0$ (unique disaggregation).
  Higher county count ↔ higher KL, confirming the impossibility scaling.
  Synthetic data: 50 states, 100 samples each, $\alpha=1$ Dirichlet prior.
}
\label{tab:census_disaggregation}
\begin{tabular}{lrrr}
\toprule
Counties & States & Mean KL & Std KL \\
\midrule
""")
    for label, mn, sd, cnt in zip(bin_labels, bin_means, bin_stds, bin_ns):
        if cnt == 0:
            continue
        mn_str = f'{mn:.4f}' if not np.isnan(mn) else '--'
        sd_str = f'{sd:.4f}' if not np.isnan(sd) else '--'
        f.write(f'{label} & {cnt} & {mn_str} & {sd_str} \\\\\n')
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print(f"Saved LaTeX table: {tex_path}")

# ── 6. Results JSON ────────────────────────────────────────────────────────────
summary = {
    'experiment':         'census_disaggregation_synthetic',
    'n_states':           N_STATES,
    'n_dirichlet_samples': N_DIRICHLET_SAMP,
    'dirichlet_alpha':    DIRICHLET_ALPHA,
    'total_pop':          TOTAL_POP,
    'county_count_range': [int(n_vals.min()), int(n_vals.max())],
    'kl_range':           [float(kl_vals.min()), float(kl_vals.max())],
    'pearson_r':          float(r_pearson),
    'pearson_p':          float(p_value),
    'mean_kl_noncontrol_ci': {
        'lo':   float(ci_lo),
        'mean': float(ci_mean),
        'hi':   float(ci_hi),
    },
    'control_n1_kl':      float(kl_vals[n_vals == 1][0]),
    'log_linear_fit':     {'slope': float(coeffs[0]), 'intercept': float(coeffs[1])},
    'per_state':          results_per_state,
}
save_results(summary, 'census_disagg')
print("\nDone.")
