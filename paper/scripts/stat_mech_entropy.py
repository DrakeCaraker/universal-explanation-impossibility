"""
Statistical Mechanics Entropy Computation (Task 1.3 — Mathematical Observation)

This is a MATHEMATICAL OBSERVATION, not an experiment.

For N = 10, 20, 50 binary spins:
  - For each macrostate k = 0..N: Ω(k) = C(N, k)
  - Rashomon entropy: S_R(k) = ln Ω(k)       [= Boltzmann entropy with k_B=1]
  - Max achievable faithfulness: 1/Ω(k)

Two-panel figure:
  Left:  S_R vs macrostate k  (bell curves for N=10,20,50)
  Right: max faithfulness (1/Ω) vs S_R  (log scale, exponential decay)
  Annotation: "S_R = k_B ln Ω (Boltzmann entropy)"

Theoretical connection:
  The Rashomon entropy S_R(k) counts the number of configurations (explanations)
  consistent with the macrostate k. The maximum achievable faithfulness — i.e.,
  the probability that a uniformly randomly chosen explanation is the true one —
  is exactly 1/Ω(k). This decays exponentially in S_R: faithfulness = exp(-S_R).

Output:
  - paper/figures/stat_mech_entropy.pdf
  - paper/results_stat_mech_entropy.json
"""

import sys
import os
import numpy as np
from math import comb, log

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from experiment_utils import (
    set_all_seeds, save_results, load_publication_style, save_figure
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── reproducibility (deterministic, but set for consistency) ───────────────────
set_all_seeds(42)

print("=" * 68)
print("Stat Mech Entropy (Task 1.3) — Mathematical Observation")
print("Rashomon Entropy = Boltzmann Entropy for Binary Spin Systems")
print("=" * 68)
print()

# ── Computation ────────────────────────────────────────────────────────────────

N_VALUES = [10, 20, 50]

# Colors: colorblind-safe palette matching publication_style
COLORS = {10: '#0072B2', 20: '#D55E00', 50: '#009E73'}
LINESTYLES = {10: '-', 20: '--', 50: '-.'}

data_by_N = {}

for N in N_VALUES:
    ks = list(range(N + 1))
    omegas = [comb(N, k) for k in ks]
    # S_R(k) = ln Ω(k); Ω(0) = 1 → S_R(0) = 0 (fine)
    S_R = [log(om) if om > 0 else 0.0 for om in omegas]
    # max faithfulness = 1/Ω(k); at k=0: 1/1 = 1
    faithfulness = [1.0 / om for om in omegas]

    data_by_N[N] = {
        "k": ks,
        "omega": omegas,
        "S_R": S_R,
        "max_faithfulness": faithfulness,
    }

    k_mid = N // 2
    print(f"  N={N:2d}: Ω(N/2) = C({N},{k_mid}) = {omegas[k_mid]:,}")
    print(f"         S_R(N/2) = ln Ω = {S_R[k_mid]:.3f}")
    print(f"         max_faithfulness(N/2) = 1/Ω = {faithfulness[k_mid]:.2e}")
    print()

# ── Figure ─────────────────────────────────────────────────────────────────────
load_publication_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

# ── Panel 1: S_R vs macrostate k ──────────────────────────────────────────────
for N in N_VALUES:
    d = data_by_N[N]
    # Normalize k to fraction k/N for comparability
    k_frac = [k / N for k in d["k"]]
    ax1.plot(k_frac, d["S_R"],
             color=COLORS[N], linestyle=LINESTYLES[N], linewidth=1.5,
             label=f'$N={N}$')

ax1.set_xlabel('Macrostate fraction $k/N$')
ax1.set_ylabel(r'Rashomon entropy $S_R(k) = \ln \Omega(k)$')
ax1.set_title('Rashomon Entropy vs Macrostate', fontsize=9)
ax1.legend(fontsize=7)

# Annotate Boltzmann connection
ax1.annotate(
    r'$S_R = k_B \ln \Omega$' + '\n(Boltzmann entropy)',
    xy=(0.5, data_by_N[50]["S_R"][25]),
    xytext=(0.68, data_by_N[50]["S_R"][25] * 0.55),
    fontsize=7,
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
    ha='center',
)

# ── Panel 2: max faithfulness vs S_R (log scale) ─────────────────────────────
for N in N_VALUES:
    d = data_by_N[N]
    S_R_arr = np.array(d["S_R"])
    faith_arr = np.array(d["max_faithfulness"])

    # Sort by S_R for clean curve
    order = np.argsort(S_R_arr)
    S_R_sorted = S_R_arr[order]
    faith_sorted = faith_arr[order]

    # Filter S_R > 0 for log-scale (k=0 and k=N give S_R=0, faith=1)
    mask = S_R_sorted > 0
    ax2.semilogy(S_R_sorted[mask], faith_sorted[mask],
                 color=COLORS[N], linestyle=LINESTYLES[N], linewidth=1.5,
                 label=f'$N={N}$')

# Overlay theoretical curve: faithfulness = exp(-S_R)
S_theory = np.linspace(0, max(data_by_N[50]["S_R"]), 300)
ax2.semilogy(S_theory, np.exp(-S_theory),
             color='black', linewidth=0.8, linestyle=':', alpha=0.7,
             label=r'Theory: $e^{-S_R}$')

ax2.set_xlabel(r'Rashomon entropy $S_R$')
ax2.set_ylabel(r'Max achievable faithfulness $1/\Omega$')
ax2.set_title('Faithfulness Decays Exponentially\nwith Rashomon Entropy', fontsize=9)
ax2.legend(fontsize=7)

# Annotate exponential decay
S_mid = data_by_N[20]["S_R"][10]
f_mid = data_by_N[20]["max_faithfulness"][10]
ax2.annotate(
    r'Faithfulness $\propto e^{-S_R}$',
    xy=(S_mid, f_mid),
    xytext=(S_mid + 4, f_mid * 50),
    fontsize=7,
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
    ha='left',
)

fig.tight_layout()
save_figure(fig, 'stat_mech_entropy')
print()


# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "observation": "stat_mech_entropy",
    "type": "Mathematical Observation (not an experiment)",
    "description": (
        "Rashomon entropy S_R(k) = ln C(N,k) = Boltzmann entropy. "
        "Max achievable faithfulness = 1/C(N,k) decays exponentially in S_R. "
        "For N binary spins in macrostate k: Omega(k) = C(N,k) microstates, "
        "each equally consistent with the macrostate (Rashomon property)."
    ),
    "N_values": N_VALUES,
    "key_values": {
        str(N): {
            "k_at_max_entropy": N // 2,
            "omega_at_max": data_by_N[N]["omega"][N // 2],
            "S_R_at_max": data_by_N[N]["S_R"][N // 2],
            "max_faithfulness_at_max_entropy": data_by_N[N]["max_faithfulness"][N // 2],
        }
        for N in N_VALUES
    },
    "interpretation": (
        "The Rashomon entropy of a binary spin system is identical to the "
        "Boltzmann entropy S = k_B ln Omega. This is not a coincidence: both "
        "count the degeneracy — the number of microstates (or explanations) "
        "consistent with the macrostate (or observable). The maximum achievable "
        "faithfulness of any single explanation is 1/Omega = exp(-S_R), which "
        "decays exponentially in the entropy. Near maximum entropy (k ≈ N/2), "
        "faithfulness approaches zero: no single explanation can be faithful. "
        "This gives a sharp thermodynamic lower bound on explanation quality."
    ),
}
save_results(results, 'stat_mech_entropy')

print()
print("=" * 68)
print("Stat Mech Entropy Observation COMPLETE")
for N in N_VALUES:
    k_mid = N // 2
    d = data_by_N[N]
    print(f"  N={N:2d}: S_R(N/2) = {d['S_R'][k_mid]:.3f}, "
          f"max_faithfulness = {d['max_faithfulness'][k_mid]:.2e}")
print("=" * 68)
