#!/usr/bin/env python3
"""
Figure 1: Gene expression alternation — the biological hook.

Panel a: Top-gene identity across 50 seeds (TSPAN8 vs CEACAM5)
Panel b: SHAP importance scatter (TSPAN8 vs CEACAM5, anti-correlated)
Panel c: The impossibility trilemma with resolution arrow
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

# Load gene expression data
with open('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/'
          'knockout-experiments/results_gene_expression_replication.json') as f:
    ge = json.load(f)

ck = ge['ap_colon_kidney_colsample']
top1_dist = ck['top1_distribution']
top1_frac = ck['top1_fractions']

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4),
                          gridspec_kw={'width_ratios': [2, 2, 1.5]})

# --- Panel a: Top gene across seeds ---
ax = axes[0]
seeds = list(range(50))
colors = []
gene_names = []
for s in seeds:
    # We don't have per-seed data in the JSON, so simulate from the distribution
    pass

# Use the distribution to create a visual
n_tspan8 = top1_dist.get('TSPAN8', 0)
n_ceacam5 = top1_dist.get('CEACAM5', 0)
n_other = 50 - n_tspan8 - n_ceacam5

# Bar chart of top-gene frequency
genes = ['TSPAN8', 'CEACAM5', 'Other']
counts = [n_tspan8, n_ceacam5, n_other]
colors_bar = ['#c44e52', '#4c72b0', '#cccccc']

bars = ax.bar(genes, counts, color=colors_bar, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Seeds (out of 50)', fontsize=8)
ax.set_title('a', fontsize=10, fontweight='bold', loc='left', pad=8)
ax.text(0, n_tspan8 + 1, f'{n_tspan8/50*100:.0f}%', ha='center', fontsize=7,
        fontweight='bold', color='#c44e52')
ax.text(1, n_ceacam5 + 1, f'{n_ceacam5/50*100:.0f}%', ha='center', fontsize=7,
        fontweight='bold', color='#4c72b0')
ax.set_ylim(0, 55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotation
ax.text(0.5, 0.95, 'AP_Colon_Kidney\n$\\rho$ = 0.858',
        transform=ax.transAxes, fontsize=6.5, ha='center', va='top',
        color='#666666')

# --- Panel b: Pathway comparison ---
ax = axes[1]

# Show the two genes with their GO terms
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('b', fontsize=10, fontweight='bold', loc='left', pad=8)

# TSPAN8 box
rect1 = plt.Rectangle((0.3, 5.5), 4.2, 4, facecolor='#c44e52', alpha=0.15,
                        edgecolor='#c44e52', linewidth=1)
ax.add_patch(rect1)
ax.text(2.4, 9.0, 'TSPAN8', fontsize=8, fontweight='bold', ha='center',
        color='#c44e52')
ax.text(2.4, 8.2, 'Tetraspanin', fontsize=6.5, ha='center', color='#666666')
ax.text(2.4, 7.4, 'Integrin binding', fontsize=6, ha='center', color='#888888')
ax.text(2.4, 6.8, 'Cell motility', fontsize=6, ha='center', color='#888888')
ax.text(2.4, 6.2, 'Invasion/metastasis', fontsize=6, ha='center', color='#888888')

# CEACAM5 box
rect2 = plt.Rectangle((5.5, 5.5), 4.2, 4, facecolor='#4c72b0', alpha=0.15,
                        edgecolor='#4c72b0', linewidth=1)
ax.add_patch(rect2)
ax.text(7.6, 9.0, 'CEACAM5', fontsize=8, fontweight='bold', ha='center',
        color='#4c72b0')
ax.text(7.6, 8.2, 'Cell adhesion', fontsize=6.5, ha='center', color='#666666')
ax.text(7.6, 7.4, 'Immune signaling', fontsize=6, ha='center', color='#888888')
ax.text(7.6, 6.8, 'Apoptosis regulation', fontsize=6, ha='center', color='#888888')
ax.text(7.6, 6.2, 'Clinical CEA marker', fontsize=6, ha='center', color='#888888')

# Zero overlap annotation
ax.text(5.0, 5.0, 'GO BP overlap: 0 terms', fontsize=7, ha='center',
        fontweight='bold', color='#333333')

# Arrow showing alternation
ax.annotate('', xy=(5.3, 7.5), xytext=(4.7, 7.5),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
ax.text(5.0, 7.8, 'Random\nseed', fontsize=6, ha='center', color='#333333')

# Resolution box at bottom
rect3 = plt.Rectangle((1.5, 0.5), 7, 2.5, facecolor='#55a868', alpha=0.15,
                        edgecolor='#55a868', linewidth=1, linestyle='--')
ax.add_patch(rect3)
ax.text(5.0, 2.3, 'DASH resolution', fontsize=7, fontweight='bold',
        ha='center', color='#55a868')
ax.text(5.0, 1.5, 'Reports both genes in top-2', fontsize=6,
        ha='center', color='#55a868')
ax.text(5.0, 0.9, 'Provably optimal', fontsize=6, ha='center',
        color='#55a868', style='italic')

ax.annotate('', xy=(5.0, 3.2), xytext=(5.0, 5.0),
            arrowprops=dict(arrowstyle='->', color='#55a868', lw=1.2))

# --- Panel c: Trilemma ---
ax = axes[2]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.8, 1.3)
ax.axis('off')
ax.set_title('c', fontsize=10, fontweight='bold', loc='left', pad=8)
ax.set_aspect('equal')

# Triangle vertices
verts = np.array([[0, 1.1], [-1.0, -0.5], [1.0, -0.5]])
triangle = plt.Polygon(verts, fill=False, edgecolor='#333333', linewidth=1.2)
ax.add_patch(triangle)

# Labels at vertices
ax.text(0, 1.25, 'Faithful', fontsize=7, ha='center', fontweight='bold')
ax.text(-1.15, -0.65, 'Stable', fontsize=7, ha='center', fontweight='bold')
ax.text(1.15, -0.65, 'Decisive', fontsize=7, ha='center', fontweight='bold')

# Center X
ax.text(0, 0.15, 'X', fontsize=16, ha='center', va='center',
        color='#c44e52', fontweight='bold')
ax.text(0, -0.15, 'Impossible', fontsize=6, ha='center', color='#c44e52')

# Edge labels (achievable pairs)
ax.text(-0.6, 0.45, 'F+S ok', fontsize=5.5, ha='center', color='#55a868',
        rotation=35)
ax.text(0.6, 0.45, 'F+D ok', fontsize=5.5, ha='center', color='#55a868',
        rotation=-35)
ax.text(0, -0.55, 'S+D ok', fontsize=5.5, ha='center', color='#55a868')

plt.tight_layout()

out_path = '/Users/drake.caraker/ds_projects/universal-explanation-impossibility/paper/figures/figure1_gene_alternation.pdf'
fig.savefig(out_path, dpi=600, bbox_inches='tight')
print(f'Saved Figure 1 to {out_path}')
