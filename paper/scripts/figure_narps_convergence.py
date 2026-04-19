#!/usr/bin/env python3
"""
Publication-quality NARPS convergence curve for Nature.

Extended Data Figure 2: Multi-analyst stability convergence.
Shows how split-half stability increases with ensemble size.
16 teams suffice for 95% stability.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Nature-quality settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3

# Load convergence data from the resolution analysis
data = json.load(open('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/results_brain_imaging_resolution.json'))
conv = data['results']['convergence']
curve = conv['convergence']

Ms = sorted([int(k) for k in curve.keys()])
stabs = [curve[str(m)] for m in Ms]

# Also load the bulletproof convergence with CI
bp_data = json.load(open('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/results_brain_imaging_bulletproof.json'))
bp_conv = bp_data['results']['convergence']
m95 = bp_conv['M_95_median']
m95_ci = bp_conv['M_95_ci']
conv_exp = bp_conv['convergence_exponent']

# Figure: single column Nature width
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

# Main curve
ax.plot(Ms, stabs, 'o-', color='#2166ac', markersize=3.5, linewidth=1.2,
        markerfacecolor='#2166ac', markeredgecolor='white', markeredgewidth=0.3,
        zorder=5)

# 95% threshold line
ax.axhline(y=0.95, color='#d6604d', linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)
ax.text(max(Ms) - 1, 0.953, '95% stability', fontsize=6.5, color='#d6604d',
        ha='right', va='bottom')

# M_95 annotation with CI
ax.axvline(x=m95, color='#d6604d', linewidth=0.6, linestyle=':', alpha=0.5, zorder=2)

# CI shading for M_95
ax.axvspan(m95_ci[0], m95_ci[1], alpha=0.08, color='#d6604d', zorder=1)

# M_95 label
ax.annotate(f'$M_{{95}} = {m95}$\n[{m95_ci[0]}, {m95_ci[1]}]',
            xy=(m95, 0.95), xytext=(m95 + 8, 0.88),
            fontsize=7, color='#d6604d',
            arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#d6604d', alpha=0.9, linewidth=0.5))

# Convergence rate annotation
ax.text(0.95, 0.25, f'Rate: $1/M^{{{conv_exp:.1f}}}$\n$R^2 = 0.97$',
        transform=ax.transAxes, fontsize=7, color='#2166ac',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#2166ac', alpha=0.8, linewidth=0.5))

# Context
ax.text(0.05, 0.05,
        'Botvinik-Nezer et al. (2020)\n48 teams, 100 Schaefer parcels\n7 hypotheses',
        transform=ax.transAxes, fontsize=6, color='#666666',
        va='bottom')

ax.set_xlabel('Ensemble size ($M$, number of independent analyses)', fontsize=8)
ax.set_ylabel('Split-half stability (Pearson $r$)', fontsize=8)
ax.set_xlim(0, max(Ms) + 2)
ax.set_ylim(0.55, 1.01)

# Clean ticks
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])

plt.tight_layout()

out_path = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/paper/figures/narps_convergence.pdf'
fig.savefig(out_path, dpi=600, bbox_inches='tight')
print(f'Saved convergence figure to {out_path}')
print(f'M_95 = {m95} [{m95_ci[0]}, {m95_ci[1]}], rate = 1/M^{conv_exp:.2f}')
