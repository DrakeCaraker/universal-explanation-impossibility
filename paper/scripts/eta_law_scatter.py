import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
eta = json.load(open('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/knockout-experiments/results_universal_eta.json'))
points = eta['points']

# Well-characterized domains (from pre-registration)
well_characterized = {
    'Attribution (SHAP, $S_2$)',
    'Concept probe (TCAV, $O(64)$)',
    'Model selection ($S_{11}$ winners)',
    'Codon ($S_2$)',
    'Codon ($S_4$)',
    'Codon ($S_6$)',
    'Stat mech ($S_{252}$, $N$=10)',
}

# Separate into two groups
wc_x, wc_y, wc_labels = [], [], []
other_x, other_y, other_labels = [], [], []

for p in points:
    x = p['predicted_instability']
    y = p['observed_instability']
    name = p['domain']

    if name in well_characterized:
        wc_x.append(x)
        wc_y.append(y)
        wc_labels.append(name.split('(')[0].strip())
    else:
        other_x.append(x)
        other_y.append(y)
        other_labels.append(name.split('(')[0].strip())

# Linear regression on well-characterized
slope, intercept, r, p_val, se = linregress(wc_x, wc_y)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

# y=x reference line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect prediction')

# Regression line (well-characterized only)
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, slope * x_line + intercept, 'b-', alpha=0.5, linewidth=1.5,
        label=f'Fit (7 points): $R^2 = {r**2:.3f}$')

# Plot well-characterized (filled circles, blue)
ax.scatter(wc_x, wc_y, c='#2166ac', s=80, zorder=5, edgecolors='white',
           linewidths=0.5, label=f'Known group ($n=7$, $R^2={r**2:.3f}$)')

# Plot other (open circles, grey)
ax.scatter(other_x, other_y, c='none', s=60, zorder=4, edgecolors='#999999',
           linewidths=1.5, label=f'Unknown/approx. group ($n=9$)')

# Labels for well-characterized points
for x, y, label in zip(wc_x, wc_y, wc_labels):
    # Offset labels to avoid overlap
    offset_x, offset_y = 0.02, 0.02
    if 'Stat mech' in label:
        offset_x, offset_y = -0.15, -0.04
    elif 'Attribution' in label:
        offset_x, offset_y = 0.03, -0.04
    elif 'Model' in label:
        offset_x, offset_y = -0.25, 0.02
    ax.annotate(label, (x, y), (x + offset_x, y + offset_y),
                fontsize=7, color='#2166ac', alpha=0.8)

# Labels for a few notable other points
for x, y, label in zip(other_x, other_y, other_labels):
    if any(k in label for k in ['Phase', 'Causal', 'Linear', 'GradCAM']):
        ax.annotate(label, (x, y), (x + 0.02, y + 0.02),
                    fontsize=6.5, color='#999999', alpha=0.7)

ax.set_xlabel('Predicted instability: $\\eta = \\dim(V^G)/\\dim(V)$', fontsize=11)
ax.set_ylabel('Observed instability rate', fontsize=11)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='lower right')
ax.set_title('Universal $\\eta$ law across 16 domain instances', fontsize=12)

plt.tight_layout()
fig.savefig('/Users/drake.caraker/ds_projects/universal-explanation-impossibility/paper/figures/eta_law_scatter.pdf',
            dpi=300, bbox_inches='tight')
print(f'Saved. R^2={r**2:.4f}, slope={slope:.3f}, intercept={intercept:.3f}, p={p_val:.2e}')
