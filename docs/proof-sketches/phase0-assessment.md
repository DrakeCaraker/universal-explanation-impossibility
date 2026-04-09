# Phase 0 Assessment — Universal Explanation Impossibility

## Summary

The abstract `ExplanationSystem` framework compiles and the universal
impossibility theorem (`explanation_impossibility`) is proved with **zero
axiom dependencies**. Two non-attribution instances (attention maps,
counterfactual explanations) instantiate the framework and inherit the
impossibility with zero new proof content.

## Lean State After Phase 0

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Files | 54 | 57 | +3 |
| Theorems + lemmas | 305 | 308 | +3 |
| Axioms | 16 | 24 | +8 |
| Sorry | 0 | 0 | 0 |

New axioms (8 total):
- 3 type axioms per instance (Config, Explanation/Map, Observable) × 2 instances = 6
- 1 system axiom per instance (attentionSystem, cfSystem) × 2 = 2
- ExplanationSystem itself adds 0 axioms (it's a structure, not axiomatized)
- explanation_impossibility adds 0 axioms (pure logic)

## Proof Structure Comparison

| | Attribution | Attention | Counterfactual |
|---|-----------|-----------|----------------|
| Θ (config) | Model parameters | NN weights | Model parameters |
| Y (observable) | Predictions | Input→output function | Predictions on test set |
| H (explanation) | Feature rankings | Token distributions | Nearest contrastive example |
| Incompatible | Opposite ranking on some pair | Different argmax token | Different counterfactual direction |
| Rashomon source | Collinearity | D'Amour et al. 2020 | Diverse decision boundaries |
| Proof steps | 5 (same) | 5 (same) | 5 (same) |
| New axioms beyond Rashomon | 0 | 0 | 0 |

## Key Findings

1. **The proof structure is identical across all three instances.** The only
   thing that changes is what Θ, H, Y, and "incompatible" mean. The 5-step
   contradiction (Rashomon → faithful → decisive → stable → contradiction)
   is completely generic.

2. **"Decisive" works uniformly.** The definition `∀ θ₁ θ₂, incompatible (E θ₁) (E θ₂) → E θ₁ ≠ E θ₂`
   does not need per-instance specialization. It captures exactly the right
   property: if the explanation method distinguishes incompatible explanations,
   it must produce different outputs for them.

3. **The Rashomon property is the only domain-specific content.** Each instance
   axiomatizes exactly one thing: that the Rashomon property holds for its
   explanation type. The impossibility follows mechanically.

4. **Zero axiom dependencies for the abstract theorem.** The core theorem
   `explanation_impossibility` depends on no axioms at all — not even `propext`.
   It is pure constructive logic.

## Remaining Instances for Phase 1

| Instance | Θ | H | Rashomon source | Difficulty |
|----------|---|---|-----------------|------------|
| Concept probes (TCAV) | NN weights | Concept directions | Equivalent models, different representations | New |
| Causal discovery | DAGs | DAGs | Markov equivalence | Port from CausalDiscovery.lean |
| Model selection | Model class | Selected model | Rashomon set | Port from ModelSelection.lean |

All three follow the same pattern. The porting tasks are mechanical.

## GO/NO-GO Recommendation

**GO.**

The universal impossibility is real. The abstract theorem is proved with zero
axioms. The proof structure transfers identically to non-attribution explanation
types. The Rashomon property is the unique input; everything else is inherited.

Phase 1 should proceed: instantiate all 6 explanation types, then prove the
universal design space dichotomy.
