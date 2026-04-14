# Pre-Registration: Nature Knockout Predictions

**Date committed**: 2026-04-14
**Framework**: Universal Explanation Impossibility (Lean 4 verified)
**Core theorem**: No explanation is faithful, stable, and decisive under the Rashomon property.

## Tautology Firewall

For every prediction below:
- **PREDICTOR**: Derived from group structure (character theory, dim(V^G)/dim(V))
- **PREDICTED**: A DOWNSTREAM OBSERVABLE (flip rate, reconstruction error, practitioner disagreement)
- **NOT**: Re-measurement of the defining quantity itself
- **NULL**: A simpler competing explanation that doesn't invoke representation theory

---

## Prediction 1: Universal Information-Loss Law

**Statement**: The instability rate (fraction of explanation queries that flip under equivalent configurations) equals 1 - η, where η = dim(V^G)/dim(V).

**Quantitative predictions**:

| Domain | Group G | dim(V) | dim(V^G) | η | Predicted instability (1-η) | Observable |
|--------|---------|--------|----------|---|---------------------------|------------|
| Attribution (2-group) | S₂ | 2 | 1 | 0.500 | 0.500 | SHAP rank flip rate |
| Attribution (3-group) | S₃ | 3 | 1 | 0.333 | 0.667 | SHAP rank flip rate |
| Attribution (4-group) | S₄ | 4 | 1 | 0.250 | 0.750 | SHAP rank flip rate |
| Attribution (6-group) | S₆ | 6 | 1 | 0.167 | 0.833 | SHAP rank flip rate |
| Attention (n heads) | S_n | n | 1 | 1/n | 1-1/n | Argmax flip rate |
| Concept probes | O(d) | d | ~1 | ~0 | ~1.0 | Cosine distance |
| Causal (MEC=m) | S_m | m | 1 | 1/m | 1-1/m | Edge flip rate |
| Model selection (k) | S_k | k | 1 | 1/k | 1-1/k | Best-model flip rate |
| Genetic code (k-fold) | S_k | k | 1 | 1/k | (k-1)/k | Normalized entropy |
| Gauge (Z₂, L×L) | Z₂^(L²-1) | 2L² | L²+1 | (L²+1)/(2L²) | (L²-1)/(2L²) | Link variance fraction |
| Stat mech (N,k) | S_Ω | Ω=C(N,k) | 1 | 1/Ω | 1-1/Ω | Microstate error |
| Linear solver (d null) | ℝ^d | n | n-d | (n-d)/n | d/n | Solver RMSD (normalized) |
| Phase retrieval (n) | Z₂×Z_n×Z₂ | 4n | ~1 | ~0 | ~1 | Reconstruction RMSD |
| Parser (k parses) | S_k | k | 1 | 1/k | 1-1/k | Parse disagreement rate |

**Success criterion**: R² > 0.90 on the universal plot (predicted vs observed), slope ∈ [0.8, 1.2], intercept ∈ [-0.1, 0.1].

**Null hypotheses**:
- H0_1: Instability is uniform across domains (no structure) → poor R²
- H0_2: Instability depends on sample size, not group structure → partial correlation test
- H0_3: Instability depends on model complexity, not group structure → partial correlation test

---

## Prediction 2: Phase Transition at Overparameterization Threshold

**Statement**: Explanation instability undergoes a sharp sigmoidal transition at r* = dim(Θ)/dim(Y) ≈ 1.

**Quantitative predictions**:
- For r < 0.5: flip rate < 0.05 (stable regime)
- For r > 2.0: flip rate > 0.40 (unstable regime)
- Transition midpoint r* ∈ [0.7, 1.5]
- Transition width narrows with dim(Y) (1/k → 0 scaling)

**Distinction from interpolation threshold**: We measure EXPLANATION stability (SHAP flip rate), not generalization error. Both may transition near r≈1 but with different curves.

**Success criterion**: Sigmoidal fit R² > 0.95, r* ∈ [0.7, 1.5] in ≥2 independent model classes.

**Null hypothesis**: Instability increases smoothly/linearly with r (no phase transition).

---

## Prediction 3: Explanation Uncertainty Principle

**Statement**: For any explanation method E, the triple (α, σ, δ) satisfies α + σ + δ ≤ 2 + f(μ_R) where f → 0 as Rashomon measure → 1.

**Operationalization**:
- α = faithfulness = mean Spearman(E(θ), explain(θ)) across configurations
- σ = stability = 1 - mean flip rate
- δ = decisiveness = fraction of pairs with strict ranking

**Success criterion**: All empirical triples satisfy the bound. At least one method per instance achieves >90% of the bound (tightness).

**Null hypothesis**: No structured bound exists; (α, σ, δ) fills the full unit cube.

---

## Prediction 4: Noether Counting

**Statement**: For P features in g correlation groups, exactly g(g-1)/2 independent group-level ranking facts are stable. All within-group pairwise comparisons flip at ~50%.

**Quantitative prediction**: For P=12, g=3 groups of 4:
- 33 between-group pairs: flip rate < 5%
- 33 within-group pairs: flip rate ∈ [40%, 60%]
- Bimodal distribution with gap at ~20%

**Success criterion**: Bimodal distribution with between-group mean < 10% and within-group mean > 40%.

**Null hypothesis**: Flip rates are continuous and depend on effect size, not group membership.

---

## Prediction 5: Interpretability Ceiling

**Statement**: For n hidden units with S_n symmetry, neuron-level agreement ≤ 1/n, subspace agreement ≈ 1.

**Already tested** (results in results_interpretability_ceiling.json):
- frac_stable = 0.0 for all n ∈ {4, 8, 16, 32, 64, 128} ✓
- mean_spearman_corr ≈ 0 for all n ✓
- invariant_variance ≈ 0 (mean activation perfectly stable) ✓
- per_neuron_variance > 0 (individual neurons unstable) ✓

**Status**: CONFIRMED. Include in synthesis.

---

## Prediction 6: Molecular Evolution (Extrapolation)

**Statement**: Synonymous substitution rates should correlate with degeneracy group order. Character theory predicts the gauge-variant fraction = (k-1)/k determines the neutral substitution rate.

**This is an EXTRAPOLATION**, not a proven theorem.

**Novel test beyond neutral theory**: Control for the number of 1-mutation synonymous neighbors and test whether S_k group structure explains residual variance.

**Success criterion**: Partial R² > 0.05 for group structure after controlling for neighbor count, p < 0.01.

**Null hypothesis**: Simple neighbor-count model explains dS fully.

---

## Well-Characterized Group Criterion

A group assignment is **well-characterized** if:

1. The symmetry is **exact by construction** (part of the domain's mathematical definition, not an empirical approximation), AND
2. The instability metric **directly measures the flip/disagreement rate** (not a proxy).

Under this criterion, the well-characterized instances are:

- **Attribution S₂** (exact by `FlipRate.lean`)
- **Concept probe O(d)** (rotation invariance of probe directions)
- **Model selection S_k** (exact count of winners)
- **Codon S₂/S₄/S₆** (exact by genetic code table)
- **Stat mech S_Ω** (exact by combinatorics)

The all-16-instance R² is the primary result of the universal η analysis. The 7-instance R² (restricted to well-characterized groups) is a **pre-specified subset analysis** defined by this criterion, not a post hoc selection.
