"""
Gauge Lattice Experiment (Task 1.2 — Physics)

ℤ₂ lattice gauge theory on N×N grids (N=4,6,8,10).
Demonstrates gauge-variant vs gauge-invariant structure:
  - Generate a base config; apply 500 random gauge transformations to create a
    gauge orbit (all configs with the same plaquette values)
  - Compute link variance within each orbit (gauge-variant, should be high)
  - Compute plaquette variance within each orbit (gauge-invariant, should be 0)
  - Metric: mean within-orbit link variance vs lattice size

Control: 2×2 lattice — verify gauge-invariant quantities have zero within-orbit
plaquette variance (by construction).

Mathematical structure:
  - Each link e ∈ {0,1} (ℤ₂)
  - Plaquette p = link1 XOR link2 XOR link3 XOR link4  (product mod 2)
  - Under gauge transformation g at site x: flip all links touching x
  - Plaquette values are invariant under all gauge transformations
  - Link values are gauge-variant (non-physical)
  - Gauge orbit size = 2^(N²) for periodic boundary (N² independent gauge DOF)

Output:
  - paper/figures/gauge_lattice.pdf
  - paper/results_gauge_lattice.json
"""

import sys
import os
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from experiment_utils import (
    set_all_seeds, save_results, load_publication_style, save_figure
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── reproducibility ────────────────────────────────────────────────────────────
set_all_seeds(42)

print("=" * 68)
print("Gauge Lattice Experiment (Task 1.2 — Physics)")
print("ℤ₂ Lattice Gauge Theory — Gauge-Variant vs Gauge-Invariant")
print("=" * 68)
print()


# ── Core functions ─────────────────────────────────────────────────────────────

def generate_config(N, rng):
    """Generate a random ℤ₂ gauge configuration on an N×N lattice.

    Returns:
        h_links: (N, N) array — h_links[i,j] = link from site (i,j) to (i, (j+1)%N)
        v_links: (N, N) array — v_links[i,j] = link from site (i,j) to ((i+1)%N, j)
    """
    h_links = rng.integers(0, 2, size=(N, N))
    v_links = rng.integers(0, 2, size=(N, N))
    return h_links, v_links


def compute_plaquettes(h_links, v_links):
    """Compute all N×N plaquette values (ℤ₂ XOR around each square).

    Plaquette at (i,j) = h_links[i,j] XOR v_links[i,(j+1)%N]
                         XOR h_links[(i+1)%N,j] XOR v_links[i,j]
    """
    return (
        h_links
        ^ np.roll(v_links, -1, axis=1)
        ^ np.roll(h_links, -1, axis=0)
        ^ v_links
    )


def apply_gauge_transform(h_links, v_links, gauge_field):
    """Apply a ℤ₂ gauge transformation to a config.

    gauge_field: (N, N) boolean/int array, gauge_field[i,j] = 1 means flip
    all links at site (i,j).

    For each site (i,j) with gauge_field[i,j]=1:
      flip h_links[i,j]    (horizontal link going right)
      flip h_links[i,(j-1)%N]  (horizontal link coming from left)
      flip v_links[i,j]    (vertical link going down)
      flip v_links[(i-1)%N,j]  (vertical link coming from above)
    """
    N = h_links.shape[0]
    h_new = h_links.copy()
    v_new = v_links.copy()
    for i in range(N):
        for j in range(N):
            if gauge_field[i, j]:
                h_new[i, j] ^= 1           # right link
                h_new[i, (j - 1) % N] ^= 1  # left link
                v_new[i, j] ^= 1           # down link
                v_new[(i - 1) % N, j] ^= 1  # up link
    return h_new, v_new


def sample_gauge_orbit(h_base, v_base, n_samples, rng):
    """Generate n_samples gauge-equivalent configs by applying random gauge transforms.

    Each sample is obtained by applying a freshly drawn random gauge field to
    the base config. All samples share the same plaquette values (same orbit).
    """
    N = h_base.shape[0]
    link_samples = []
    plaq_samples = []

    for _ in range(n_samples):
        g = rng.integers(0, 2, size=(N, N))
        h_t, v_t = apply_gauge_transform(h_base, v_base, g)
        link_samples.append(np.concatenate([h_t.ravel(), v_t.ravel()]))
        plaq_samples.append(compute_plaquettes(h_t, v_t).ravel())

    links = np.array(link_samples, dtype=float)     # (n_samples, 2N²)
    plaqs = np.array(plaq_samples, dtype=float)     # (n_samples, N²)
    return links, plaqs


def run_lattice_experiment(N, n_base_configs, n_orbit_samples, rng):
    """Run the gauge experiment for a single lattice size N.

    For each of n_base_configs random base configs:
      - Sample n_orbit_samples gauge-equivalent configs (same plaquette)
      - Compute within-orbit link variance and plaquette variance

    Returns:
        mean_link_var: mean within-orbit link variance (should be ~0.25)
        mean_plaq_var: mean within-orbit plaquette variance (should be 0)
    """
    link_vars = []
    plaq_vars = []

    for _ in range(n_base_configs):
        h_base, v_base = generate_config(N, rng)
        links, plaqs = sample_gauge_orbit(h_base, v_base, n_orbit_samples, rng)

        # Within-orbit variance per position, then averaged
        link_vars.append(links.var(axis=0).mean())
        plaq_vars.append(plaqs.var(axis=0).mean())

    return float(np.mean(link_vars)), float(np.mean(plaq_vars))


def control_2x2(rng, n_base=20, n_orbit=500):
    """2×2 control: verify within-orbit plaquette variance = 0."""
    N = 2
    plaq_vars = []

    for _ in range(n_base):
        h_base, v_base = generate_config(N, rng)
        _, plaqs = sample_gauge_orbit(h_base, v_base, n_orbit, rng)
        plaq_vars.append(plaqs.var(axis=0).mean())

    return float(max(plaq_vars)), float(np.mean(plaq_vars))


# ── Main experiment ────────────────────────────────────────────────────────────

LATTICE_SIZES = [4, 6, 8, 10]
N_BASE_CONFIGS = 20    # base configs to average over (reduces noise)
N_ORBIT_SAMPLES = 500  # gauge-equivalent configs per base (forms the orbit)
rng = np.random.default_rng(42)

print(f"Lattice sizes: {LATTICE_SIZES}")
print(f"Base configs per size: {N_BASE_CONFIGS}")
print(f"Orbit samples per base config: {N_ORBIT_SAMPLES}")
print()

results_by_size = {}
mean_link_vars = []
mean_plaq_vars = []

for N in LATTICE_SIZES:
    print(f"  N={N}: sampling gauge orbits on {N}×{N} lattice "
          f"({2*N*N} links, {N*N} plaquettes per config)...")
    mean_lv, mean_pv = run_lattice_experiment(N, N_BASE_CONFIGS, N_ORBIT_SAMPLES, rng)
    print(f"    → mean within-orbit link variance    = {mean_lv:.5f}  (gauge-VARIANT)")
    print(f"       mean within-orbit plaquette var   = {mean_pv:.2e}  (gauge-INVARIANT, ≈0)")
    mean_link_vars.append(mean_lv)
    mean_plaq_vars.append(mean_pv)
    results_by_size[N] = {
        "n_links_per_config": 2 * N * N,
        "n_plaquettes": N * N,
        "n_base_configs": N_BASE_CONFIGS,
        "n_orbit_samples": N_ORBIT_SAMPLES,
        "mean_within_orbit_link_variance": mean_lv,
        "mean_within_orbit_plaquette_variance": mean_pv,
    }

print()
print("Control: 2×2 lattice — verifying within-orbit plaquette variance = 0")
ctrl_max_pv, ctrl_mean_pv = control_2x2(rng)
print(f"  → max within-orbit plaquette variance = {ctrl_max_pv:.2e}")
print(f"     mean within-orbit plaquette variance = {ctrl_mean_pv:.2e}")
assert ctrl_max_pv < 1e-12, f"Control FAILED: plaquette variance = {ctrl_max_pv}"
print("  Control PASSED: gauge-invariant quantity exactly preserved (var = 0)")
print()


# ── Figure ─────────────────────────────────────────────────────────────────────
load_publication_style()

fig, ax = plt.subplots(figsize=(4.5, 3.2))

ax.plot(LATTICE_SIZES, mean_link_vars, 'o-', color='#0072B2', linewidth=1.5,
        markersize=6, markerfacecolor='white', markeredgewidth=1.5,
        label='Link variance (gauge-variant)')

ax.plot(LATTICE_SIZES, mean_plaq_vars, 's--', color='#D55E00', linewidth=1.5,
        markersize=6, markerfacecolor='white', markeredgewidth=1.5,
        label='Plaquette variance (gauge-invariant, $\\approx 0$)')

# Theoretical upper bound for link variance under uniform gauge sampling
ax.axhline(0.25, color='gray', linewidth=0.8, linestyle=':',
           label='Bernoulli(0.5) baseline (var $= 0.25$)')

ax.set_xlabel('Lattice size $N$')
ax.set_ylabel('Mean within-orbit variance')
ax.set_title(r'$\mathbb{Z}_2$ Gauge Theory: Gauge-Variant vs Gauge-Invariant',
             fontsize=9)
ax.set_xticks(LATTICE_SIZES)
ax.legend(fontsize=7)

# Annotate the key message
mid_N = LATTICE_SIZES[1]
mid_lv = mean_link_vars[1]
ax.annotate(
    'Same plaquette config\n(gauge-invariant observable),\ndifferent link values\n(gauge-variant)',
    xy=(mid_N, mid_lv),
    xytext=(mid_N + 1.5, mid_lv - 0.04),
    fontsize=6.5,
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
    ha='left',
)

fig.tight_layout()
save_figure(fig, 'gauge_lattice')
print()


# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "experiment": "gauge_lattice",
    "description": (
        "Z2 lattice gauge theory: within-orbit link variance vs lattice size. "
        "Gauge orbits sampled by applying random gauge transformations to base configs. "
        "All orbit members share the same plaquette values (gauge-invariant), "
        "but differ in link values (gauge-variant)."
    ),
    "lattice_sizes": LATTICE_SIZES,
    "n_base_configs": N_BASE_CONFIGS,
    "n_orbit_samples": N_ORBIT_SAMPLES,
    "mean_within_orbit_link_variance": mean_link_vars,
    "mean_within_orbit_plaquette_variance": mean_plaq_vars,
    "control_2x2": {
        "max_within_orbit_plaquette_variance": ctrl_max_pv,
        "mean_within_orbit_plaquette_variance": ctrl_mean_pv,
        "status": "PASSED"
    },
    "by_size": results_by_size,
    "interpretation": (
        "Within each gauge orbit (same plaquette configuration), link values vary "
        "substantially across gauge-equivalent configs — approaching the Bernoulli(0.5) "
        "baseline as N grows (more gauge freedom). Plaquette values are exactly "
        "preserved (variance = 0) — they are the physical, gauge-invariant observables. "
        "This is a direct physical instance of the Rashomon property: many link "
        "configurations (explanations) are consistent with the same plaquette "
        "configuration (observable). No single link configuration is 'the truth'."
    ),
}
save_results(results, 'gauge_lattice')

print()
print("=" * 68)
print("Gauge Lattice Experiment COMPLETE")
for N, lv, pv in zip(LATTICE_SIZES, mean_link_vars, mean_plaq_vars):
    print(f"  N={N:2d}: link var = {lv:.4f} (gauge-variant),  "
          f"plaq var = {pv:.2e} (gauge-invariant)")
print(f"  Control (2×2) plaquette variance: {ctrl_max_pv:.2e} (≈0 ✓)")
print("=" * 68)
