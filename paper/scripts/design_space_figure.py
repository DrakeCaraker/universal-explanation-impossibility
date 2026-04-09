"""
Generate the signature figure: the Attribution Design Space.

Two families plotted in (Unfaithfulness U, Ranking Stability S) space:
- Family A (single-model): U = 1/2, S ≤ 1 - m³/P³
- Family B (ensemble/DASH): U = 0, S = 1 - O(1/M)
- Infeasible region: U = 0, S = 1, C = complete
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import os

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'figure.figsize': (5.5, 4.5), 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

# Parameters
m, P = 5, 20
s_bound = 1 - (m**3) / (P**3)  # ≈ 0.984

# Infeasible region: shade the "ideal" corner
# The ideal is S=1, U=0, C=complete — which is infeasible
ax.fill_between([0, 0.05], [0.99, 0.99], [1.01, 1.01],
                color='#ffcccc', alpha=0.5, zorder=0)
ax.annotate('Infeasible: $(S{=}1, U{=}0, C{=}\\mathrm{complete})$',
            xy=(0.01, 1.005), fontsize=7.5, color='#cc0000', ha='left', va='bottom')

# Family B (ensemble/DASH): U = 0, S increases with M
M_values = [1, 2, 5, 10, 25, 50, 100, 500]
# S_M = 1 - sigma^2/(M * Delta^2), approximate with sigma/Delta = 0.5
sigma_over_delta = 0.5
S_B = [1 - sigma_over_delta**2 / M for M in M_values]

ax.plot([0]*len(M_values), S_B, 'o-', color='#1f77b4', markersize=8,
        linewidth=2, zorder=5, label='Family $\\mathcal{B}$ (DASH)')

# Label specific M values (only M=1, M=5, and collapsed top cluster)
for M, s in zip(M_values, S_B):
    if M == 1:
        ax.annotate(f'$M = 1$', xy=(0, s), xytext=(0.06, s),
                    fontsize=8, color='#1f77b4',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))
    elif M == 5:
        ax.annotate(f'$M = 5$', xy=(0, s), xytext=(0.06, s),
                    fontsize=8, color='#1f77b4',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))
# Single label for the top cluster (M >= 25)
ax.annotate(r'$M \geq 25$', xy=(0, S_B[4]), xytext=(0.06, 0.975),
            fontsize=8, color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))

# Family A (single-model): U = 1/2, S ≤ s_bound
# Show as a vertical bar at U = 0.5
s_a_values = np.linspace(0.5, s_bound, 20)
ax.fill_betweenx(s_a_values, 0.47, 0.53, color='#d62728', alpha=0.3, zorder=3)
ax.plot([0.5]*5, np.linspace(0.6, s_bound, 5), 's', color='#d62728',
        markersize=6, zorder=5, label='Family $\\mathcal{A}$ (single-model)')
ax.annotate(f'$S \\leq 1 - m^3/P^3$\n$= {s_bound:.3f}$',
            xy=(0.5, s_bound), xytext=(0.35, s_bound + 0.003),
            fontsize=8, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))

# Annotations for the two modes
ax.text(0.52, 0.55, 'Faithful + Complete\nbut Unstable\n($U = 1/2$)',
        fontsize=9, color='#d62728', style='italic', ha='left')
ax.text(0.01, 0.62, 'Stable + Faithful\nbut Ties within groups\n($U = 0$)',
        fontsize=9, color='#1f77b4', style='italic', ha='left')

# Arrow showing the practitioner choice
ax.annotate('', xy=(0.05, 0.82), xytext=(0.45, 0.82),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, ls='--'))
ax.text(0.25, 0.835, 'No third option', fontsize=10, color='gray',
        ha='center', weight='bold')

ax.set_xlabel('Within-group Unfaithfulness $U$', fontsize=11)
ax.set_ylabel('Ranking Stability $S$ (expected Spearman)', fontsize=11)
ax.set_xlim(-0.05, 0.6)
ax.set_ylim(0.45, 1.02)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, lw=0.3, alpha=0.2, color='#cccccc')
ax.set_title('The Attribution Design Space', fontsize=12, weight='bold')

fig.tight_layout()
out = os.path.join(OUT_DIR, "design_space.pdf")
fig.savefig(out)
print(f"Saved {out}")
