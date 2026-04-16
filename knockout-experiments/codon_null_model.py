#!/usr/bin/env python3
"""
Codon null model: Is rho=1.0 (degeneracy vs usage entropy) trivial?

Under uniform codon usage within each degeneracy class, entropy = log2(d).
Since log2 is monotone, Spearman rho between degeneracy rank and entropy rank
is guaranteed to be 1.0 by construction.

This script:
1. Shows the null (uniform) model gives rho=1.0
2. Shows ANY monotone entropy-degeneracy relationship gives rho=1.0
3. Concludes: rho=1.0 is a calibration check, not independent validation
"""

import numpy as np
from scipy.stats import spearmanr

# Standard genetic code: amino acid degeneracies
# Met(1), Trp(1), Asn(2), Asp(2), Cys(2), Gln(2), Glu(2), His(2), Lys(2),
# Phe(2), Tyr(2), Ile(3), Ala(4), Gly(4), Pro(4), Thr(4), Val(4),
# Arg(6), Leu(6), Ser(6)
amino_acids = {
    'Met': 1, 'Trp': 1,
    'Asn': 2, 'Asp': 2, 'Cys': 2, 'Gln': 2, 'Glu': 2,
    'His': 2, 'Lys': 2, 'Phe': 2, 'Tyr': 2,
    'Ile': 3,
    'Ala': 4, 'Gly': 4, 'Pro': 4, 'Thr': 4, 'Val': 4,
    'Arg': 6, 'Leu': 6, 'Ser': 6
}

degeneracies = list(amino_acids.values())
names = list(amino_acids.keys())

# Null model 1: Uniform usage => entropy = log2(d)
uniform_entropy = [np.log2(d) if d > 1 else 0.0 for d in degeneracies]
rho_uniform, p_uniform = spearmanr(degeneracies, uniform_entropy)

print("=" * 60)
print("CODON NULL MODEL ANALYSIS")
print("=" * 60)
print(f"\nNull model (uniform usage): Spearman rho = {rho_uniform:.4f}")
print(f"  p-value = {p_uniform:.2e}")
print(f"  Conclusion: rho=1.0 is GUARANTEED under uniform usage")

# Null model 2: Any monotone function of d
# Even extreme non-uniform usage maintains rho=1.0 as long as
# mean entropy is monotone in d (which it is for any reasonable usage pattern)
print("\nNull model (any monotone function):")
for func_name, func in [
    ("sqrt(d)", lambda d: np.sqrt(d)),
    ("d^2", lambda d: d**2),
    ("log10(d)", lambda d: np.log10(d) if d > 1 else 0),
    ("d + noise(0.01)", lambda d: d + np.random.RandomState(42).normal(0, 0.01)),
]:
    vals = [func(d) for d in degeneracies]
    rho, _ = spearmanr(degeneracies, vals)
    print(f"  f(d) = {func_name}: Spearman rho = {rho:.4f}")

# How could rho < 1.0?
# Only if a lower-degeneracy amino acid has HIGHER mean entropy than a
# higher-degeneracy one. This requires anti-selection: organisms preferring
# rare codons more for high-degeneracy amino acids.
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The Spearman rho=1.0 between degeneracy and usage entropy is guaranteed
by construction for ANY monotone entropy-degeneracy relationship.
Since entropy is bounded by log2(d) and observed usage shows strong
codon bias (Sharp & Li 1987), the mean entropy per degeneracy class
is monotonically increasing. Therefore:

  rho = 1.0 is a CALIBRATION CHECK (the framework correctly identifies
  a known case of maximal redundancy), NOT an independent validation.

The constructive Rashomon witness (zero axioms, explicit codons via
`decide`) is the primary validation of the codon instance.
""")
