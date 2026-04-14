/-- NOTE: This file contains documentation and commentary only — no theorems,
    lemmas, or axioms. Random forests serve as a contrast case; formal
    verification is left to future work. This file does not contribute to
    the 417-theorem count. -/

/-
  Random forests: contrast case. The Attribution Impossibility may hold weakly,
  but the mechanism differs fundamentally from sequential methods.

  Key difference: trees are trained INDEPENDENTLY (no shared residuals).
  Feature j being selected in tree t does NOT reduce feature k's signal
  in tree t+1. There is no cumulative first-mover advantage.

  Consequences:
  - The Rashomon property may hold (different bootstrap samples →
    different feature utilization), but attribution differences are
    O(1/√T) from sampling noise, not O(ρ²T) from cumulative bias.
  - The trilemma applies (if Rashomon holds) but the equity violation
    is BOUNDED — the max/min ratio → 1 as T → ∞ (law of large numbers).
  - This makes RF inherently more equitable than GBDT under collinearity.

  This contrast STRENGTHENS the impossibility result: the trilemma
  DISCRIMINATES between model classes. Sequential methods (GBDT, Lasso, NN)
  have provably worse equity violations than parallel methods (RF, bagging).

  No formal proofs in this file — this is a scope discussion.
  The formal contribution is: the trilemma's severity depends on the
  optimization architecture, and ensemble independence is the resolution.
-/
import UniversalImpossibility.Iterative

set_option autoImplicit false

namespace UniversalImpossibility.RandomForest

/-!
## Why the trilemma is weak for random forests

### The mechanism difference

Sequential methods (GBDT, Lasso, NN):
  - Tree/iteration t's choices affect tree/iteration t+1 (shared residuals)
  - The dominant feature accumulates advantage: O(ρ² · T) cumulative bias
  - Attribution ratio = 1/(1-ρ²) → ∞ as ρ → 1

Parallel methods (Random Forest, Bagging):
  - Tree t is independent of tree t' (different bootstrap sample)
  - No cumulative advantage: each tree's feature selection is independent
  - Attribution difference = O(1/√T) sampling noise → 0 as T → ∞

### Does the Rashomon property hold for RF?

Possibly, but weakly. Different bootstrap samples may prefer different
features, satisfying `RashimonProperty`. But the magnitude of the
preference reversal is bounded:
  - In GBDT: the dominant feature gets T/(2-ρ²) splits vs (1-ρ²)T/(2-ρ²)
    for the other → ratio = 1/(1-ρ²), divergent
  - In RF: expected splits are T/m per feature (uniform by independence),
    with O(√T) deviation → ratio ≈ 1 + O(1/√T), convergent

### Implication for the research program

The Attribution Impossibility's QUALITATIVE conclusion (impossible to be
faithful + stable + complete) may hold for RF under finite T, but its
PRACTICAL significance vanishes with more trees. For GBDT, more trees
makes the violation WORSE (cumulative bias). For RF, more trees makes
it BETTER (law of large numbers).

This is why DASH (which breaks sequential dependence) resolves the
impossibility: it converts a sequential ensemble (growing violation)
into an independent ensemble (shrinking violation).
-/

end UniversalImpossibility.RandomForest
