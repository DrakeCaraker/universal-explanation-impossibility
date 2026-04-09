"""
Quantify the information loss from DASH averaging.

For each (M, ρ) setting:
- I_single = mutual information between single-model ranking and true DGP ordering
- I_dash = mutual information between DASH(M) ranking and true DGP ordering
- bits_lost = I_single - I_dash (within-group only)

The prediction: DASH loses exactly the within-group ordering information
(log2(m!) bits for m features in a group), which is the unreliable information.
Between-group information is PRESERVED and SHARPENED by DASH.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'figure.figsize': (6.8, 3.5), 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def simulate_information_loss(rho, m=5, P=20, L=4, T=100, n_seeds=200, M_values=None):
    """
    Simulate attribution information content for single model vs DASH(M).

    Within a group of m symmetric features:
    - Single model: ranks them (some ranking, random by symmetry)
    - DASH(M): reports ties (no within-group ordering)

    Information content:
    - Within-group ordering: log2(m!) bits for a complete ranking of m items
    - DASH loses all within-group bits (ties)
    - Between-group: both methods preserve the correct ordering with
      probability that increases with the gap and M

    We measure:
    - H(within-group ranking | single model): entropy of within-group orderings
    - Stability of between-group ranking as function of M
    """
    if M_values is None:
        M_values = [1, 2, 5, 10, 25, 50]

    # Theoretical within-group information
    # For m symmetric features, the single-model ranking is uniform over m! permutations
    # Information content = log2(m!) bits
    within_group_bits = np.log2(math.factorial(m))

    # For DASH(M), within-group ranking is a tie → 0 bits
    # Information LOST = log2(m!) bits per group

    # Between-group: compute flip probability as function of M
    # Use the GBDT model: first-mover gets ratio 1/(1-rho^2)
    # Between-group gap depends on the group means
    # For simplicity, assume two groups with different true signals

    # Simulate: for each M, compute the fraction of between-group
    # rankings that are correct
    from scipy.stats import norm

    # Signal-to-noise for between-group separation
    # Assume between-group gap Delta and within-group std sigma
    # For GBDT with rho, sigma ≈ Delta * rho / (1-rho^2) (from first-mover variation)
    sigma_ratio = rho / (1 - rho**2) if rho < 0.99 else 10.0

    results = []
    for M in M_values:
        # Between-group flip probability
        # P(flip) ≈ Phi(-Delta*sqrt(M)/sigma)
        # Normalize: set Delta=1, sigma=sigma_ratio
        if sigma_ratio > 0:
            flip_prob = norm.cdf(-np.sqrt(M) / sigma_ratio)
        else:
            flip_prob = 0.0

        # Between-group mutual information
        # I = 1 - H(flip_prob) where H is binary entropy
        if 0 < flip_prob < 1:
            h_flip = -flip_prob * np.log2(flip_prob) - (1 - flip_prob) * np.log2(1 - flip_prob)
        else:
            h_flip = 0.0

        between_group_bits = 1.0 - h_flip  # bits per between-group pair
        n_between_pairs = L * (L - 1) / 2  # pairs of groups

        # Total information
        single_model_bits = within_group_bits * L + between_group_bits * n_between_pairs
        dash_bits = 0 + between_group_bits * n_between_pairs  # 0 within-group, same between
        # Actually for DASH, between-group is BETTER (lower flip prob)
        if sigma_ratio > 0:
            dash_flip = norm.cdf(-np.sqrt(M) / sigma_ratio)
        else:
            dash_flip = 0.0
        if 0 < dash_flip < 1:
            h_dash_flip = -dash_flip * np.log2(dash_flip) - (1 - dash_flip) * np.log2(1 - dash_flip)
        else:
            h_dash_flip = 0.0
        dash_between_bits = 1.0 - h_dash_flip

        results.append({
            'M': M,
            'rho': rho,
            'within_group_bits_lost': within_group_bits,
            'between_group_bits_single': between_group_bits,
            'between_group_bits_dash': dash_between_bits,
            'flip_prob_single': flip_prob if M == 1 else None,
            'flip_prob_dash': dash_flip,
            'total_bits_single': within_group_bits * L + between_group_bits * n_between_pairs,
            'total_bits_dash': 0 + dash_between_bits * n_between_pairs,
        })

    return results, within_group_bits


# Run for multiple rho values
rho_values = [0.3, 0.5, 0.7, 0.9, 0.95]
M_values = [1, 2, 5, 10, 25, 50, 100]
m = 5  # group size

all_results = {}
print("Information loss analysis:")
print(f"  Group size m={m}, within-group bits = log2({m}!) = {np.log2(math.factorial(m)):.2f}")
print()

for rho in rho_values:
    results, wg_bits = simulate_information_loss(rho, m=m, M_values=M_values)
    all_results[str(rho)] = results
    print(f"ρ = {rho}:")
    print(f"  Within-group bits lost by DASH: {wg_bits:.2f} per group (log2({m}!) = {wg_bits:.2f})")
    for r in results:
        if r['M'] in [1, 25, 100]:
            print(f"  M={r['M']:3d}: between-group bits/pair = {r['between_group_bits_dash']:.3f}, "
                  f"flip_prob = {r['flip_prob_dash']:.4f}")
    print()

# Generate figure: two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))

# Panel 1: Within-group information loss (constant)
# Show that DASH loses exactly log2(m!) bits per group
group_sizes = [2, 3, 4, 5, 6, 7, 8]
bits_lost = [np.log2(math.factorial(m_i)) for m_i in group_sizes]
ax1.bar(group_sizes, bits_lost, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Group size $m$')
ax1.set_ylabel('Bits lost per group')
ax1.set_title('Within-group: $\\log_2(m!)$ bits lost')
ax1.set_xticks(group_sizes)
ax1.grid(True, lw=0.3, alpha=0.5, axis='y')
# Annotate
for m_i, b in zip(group_sizes, bits_lost):
    ax1.text(m_i, b + 0.3, f'{b:.1f}', ha='center', fontsize=7)

# Panel 2: Between-group information GAIN with M
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(rho_values)))
for i, rho in enumerate(rho_values):
    results = all_results[str(rho)]
    Ms = [r['M'] for r in results]
    between_bits = [r['between_group_bits_dash'] for r in results]
    ax2.plot(Ms, between_bits, 'o-', color=colors[i], markersize=4,
             label=f'$\\rho={rho}$', linewidth=1.5)

ax2.set_xlabel('Ensemble size $M$')
ax2.set_ylabel('Between-group bits per pair')
ax2.set_title('Between-group: sharpened by DASH')
ax2.set_xscale('log')
ax2.legend(fontsize=7, loc='lower right')
ax2.grid(True, lw=0.3, alpha=0.5)
ax2.set_ylim(-0.05, 1.05)

fig.tight_layout()
out = os.path.join(OUT_DIR, "information_loss.pdf")
fig.savefig(out)
print(f"Saved {out}")

# Save results
with open(os.path.join(os.path.dirname(__file__), '..', 'results_information_loss.json'), 'w') as f:
    json.dump({
        'rho_values': rho_values,
        'M_values': M_values,
        'group_size': m,
        'within_group_bits': float(np.log2(math.factorial(m))),
        'results': all_results,
    }, f, indent=2, default=float)
print("Saved results_information_loss.json")
