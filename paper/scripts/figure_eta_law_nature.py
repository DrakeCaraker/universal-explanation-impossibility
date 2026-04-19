#!/usr/bin/env python3
"""
Publication-quality η law scatter for Nature.

Extended Data Figure 1: The universal η law.
7 well-characterized domains, R² = 0.957, zero free parameters.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress

# Nature-quality settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3

# Load data
eta = json.load(open('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/results_universal_eta.json'))
points = eta['points']

well_characterized = {
    'Attribution (SHAP, $S_2$)',
    'Concept probe (TCAV, $O(64)$)',
    'Model selection ($S_{11}$ winners)',
    'Codon ($S_2$)',
    'Codon ($S_4$)',
    'Codon ($S_6$)',
    'Stat mech ($S_{252}$, $N$=10)',
}

wc_x, wc_y, wc_labels = [], [], []
other_x, other_y, other_labels = [], [], []

for p in points:
    x = p['predicted_instability']
    y = p['observed_instability']
    name = p['domain']
    if name in well_characterized:
        wc_x.append(x)
        wc_y.append(y)
        # Clean labels for figure
        label = name.split('(')[0].strip()
        if 'Codon' in name:
            k = name.split('$S_')[1].split('$')[0] if '$S_' in name else '?'
            label = f'Codon ($S_{{{k}}}$)'
        elif 'Stat mech' in name:
            label = 'Stat. mech.'
        elif 'Concept' in name:
            label = 'Concept probe'
        elif 'Model' in name:
            label = 'Model selection'
        elif 'Attribution' in name:
            label = 'Attribution'
        wc_labels.append(label)
    else:
        other_x.append(x)
        other_y.append(y)
        other_labels.append(name.split('(')[0].strip())

wc_x, wc_y = np.array(wc_x), np.array(wc_y)
other_x, other_y = np.array(other_x), np.array(other_y)

slope, intercept, r, p_val, se = linregress(wc_x, wc_y)

# Figure: single column Nature width (89mm ≈ 3.5in)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.3))

# Diagonal reference
ax.plot([0, 1.02], [0, 1.02], color='#cccccc', linewidth=0.8, linestyle='--', zorder=1)

# Regression line
x_line = np.linspace(0, 1.02, 100)
ax.plot(x_line, slope * x_line + intercept, color='#2166ac', linewidth=1.0,
        alpha=0.6, zorder=2)

# Fill the confidence band (approximate)
y_pred = slope * wc_x + intercept
residuals = wc_y - y_pred
se_fit = np.sqrt(np.sum(residuals**2) / (len(wc_x) - 2))
ax.fill_between(x_line, slope * x_line + intercept - 1.96 * se_fit,
                slope * x_line + intercept + 1.96 * se_fit,
                color='#2166ac', alpha=0.08, zorder=1)

# Plot other (open circles, grey) — background
ax.scatter(other_x, other_y, facecolors='none', edgecolors='#aaaaaa',
           s=35, linewidths=0.8, zorder=3)

# Plot well-characterized (filled circles, blue) — foreground
ax.scatter(wc_x, wc_y, c='#2166ac', s=45, zorder=5, edgecolors='white',
           linewidths=0.5)

# Labels for well-characterized points
label_offsets = {
    'Attribution': (0.03, -0.06),
    'Concept probe': (-0.28, -0.04),
    'Model selection': (-0.02, 0.04),
    'Codon ($S_{2}$)': (0.03, -0.05),
    'Codon ($S_{4}$)': (0.03, 0.03),
    'Codon ($S_{6}$)': (0.03, -0.06),
    'Stat. mech.': (-0.20, -0.04),
}

for x, y, label in zip(wc_x, wc_y, wc_labels):
    dx, dy = label_offsets.get(label, (0.03, 0.02))
    ax.annotate(label, (x, y), (x + dx, y + dy),
                fontsize=6.5, color='#2166ac',
                arrowprops=dict(arrowstyle='-', color='#2166ac', alpha=0.3, lw=0.4)
                if abs(dx) > 0.1 or abs(dy) > 0.05 else None)

# Annotations
ax.text(0.05, 0.92, f'$R^2 = {r**2:.3f}$\nslope $= {slope:.2f}$\n$n = 7$, $p = {p_val:.1e}$',
        transform=ax.transAxes, fontsize=7, color='#2166ac',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2166ac',
                  alpha=0.8, linewidth=0.5))

# Grey annotation for approximate group points
ax.text(0.55, 0.15, f'Open: unknown/\napprox. group ($n=9$)',
        transform=ax.transAxes, fontsize=6, color='#999999',
        verticalalignment='top')

ax.set_xlabel(r'Predicted instability: $\eta = 1 - \dim(V^G)/\dim(V)$', fontsize=8)
ax.set_ylabel('Observed instability rate', fontsize=8)
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(-0.03, 1.05)
ax.set_aspect('equal')

# Clean ticks
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

plt.tight_layout()

out_path = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/paper/figures/eta_law_scatter.pdf'
fig.savefig(out_path, dpi=600, bbox_inches='tight')
print(f'Saved η law scatter to {out_path}')
print(f'R²={r**2:.4f}, slope={slope:.3f}, intercept={intercept:.3f}, p={p_val:.2e}')
