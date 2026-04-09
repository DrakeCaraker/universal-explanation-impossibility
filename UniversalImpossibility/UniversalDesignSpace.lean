import UniversalImpossibility.ExplanationSystem

/-!
# Universal Design Space Dichotomy

For any explanation system with the Rashomon property, achievable methods
fall into exactly two families:
- Family A: faithful + decisive, unstable (individual-model explanations)
- Family B: stable, with symmetry-induced ties (aggregated)

The dichotomy theorem shows that stable methods cannot be both faithful
and decisive. This is a direct corollary of the explanation impossibility.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

/-- Universal Design Space Dichotomy.

For any explanation system with the Rashomon property, achievable methods
fall into exactly two families:
- Family A: faithful + decisive, unstable (individual-model explanations)
- Family B: stable, with symmetry-induced ties (aggregated)

The dichotomy: stable methods cannot be both faithful and decisive. -/
theorem universal_design_space_dichotomy (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hstable : stable S E) :
    ¬(faithful S E ∧ decisive S E) := by
  intro ⟨hf, hd⟩
  exact explanation_impossibility S E hf hstable hd
