/-
  Relaxation path convergence and Design Space corollaries.

  S38: Theorem 1 (impossibility) as a Design Space corollary
  S40: Both relaxation paths converge to ties

  Supplement: §Previous results as corollaries
-/
import UniversalImpossibility.UnfaithfulBound

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- S38: The Attribution Impossibility is the statement that the point
    (S=1, U=0, C=complete) lies outside the achievable set.
    This is just attribution_impossibility restated. -/
theorem impossibility_as_design_space_corollary
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h_faithful

/-- S40: Path convergence.
    The "drop faithfulness" path: the optimal stable ranking assigns ties.
    The "drop completeness" path: DASH assigns ties.
    Both converge to: ties for within-group pairs.

    Formally: under Rashomon, any ranking that avoids unfaithfulness
    must be a tie — and DASH is a tie (consensus_equity). -/
theorem relaxation_paths_converge
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    -- The ranking avoids unfaithfulness (the "drop faithfulness" path solution)
    (h_no_unfaith : ¬ ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f)) :
    -- Then it must be a tie (same as DASH)
    ¬ ranking j k ∧ ¬ ranking k j :=
  unfaithfulness_free_implies_tie fs hrash ℓ j k hj hk hjk ranking h_no_unfaith

/-- Complete + Rashomon → unfaithful (contrapositively: faithful → incomplete).
    This is the "complete rankings are necessarily unfaithful" half. -/
theorem complete_implies_unfaithful
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (hcomp : ranking j k ∨ ranking k j) :
    ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f) :=
  stable_complete_unfaithful fs hrash ℓ j k hj hk hjk ranking hcomp

end UniversalImpossibility
