/-
  Design Space Theorem — Step 3: Exhaustiveness.

  Every attribution method falls into Family A (faithful+complete+unstable)
  or Family B (stable+ties+U=0). No third option exists.

  This closes the "half-formalized" gap in DesignSpace.lean.

  Supplement: §The Attribution Design Space Theorem, Step 3
-/
import UniversalImpossibility.UnfaithfulBound
import UniversalImpossibility.DesignSpace

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Step 3, Case 2: Any method that avoids unfaithfulness for a symmetric
    pair must report a tie (neither j ≻ k nor k ≻ j).

    This is a direct application of unfaithfulness_free_implies_tie. -/
theorem design_space_exhaustiveness
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop) :
    -- Either the ranking is complete (and therefore unfaithful to some model)
    -- Or the ranking has a tie for (j,k) (and is never unfaithful for this pair)
    (ranking j k ∨ ranking k j) ∧
      (∃ f : Model, (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
                     (ranking k j ∧ attribution fs j f > attribution fs k f))
    ∨
    (¬ ranking j k ∧ ¬ ranking k j) := by
  by_cases hcomp : ranking j k ∨ ranking k j
  · -- Complete case: the ranking decides j vs k, so it's unfaithful to some model
    left
    exact ⟨hcomp, stable_complete_unfaithful fs hrash ℓ j k hj hk hjk ranking hcomp⟩
  · -- Tie case: the ranking doesn't decide j vs k
    right
    push Not at hcomp
    exact hcomp

/-- The full Design Space dichotomy: every ranking for a Rashomon pair
    is EITHER complete-and-unfaithful (Family A) OR a tie (Family B).
    There is no ranking that is complete AND avoids unfaithfulness. -/
theorem no_complete_faithful_ranking
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (hcomp : ranking j k ∨ ranking k j) :
    ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f) :=
  stable_complete_unfaithful fs hrash ℓ j k hj hk hjk ranking hcomp

/-- Combining: a ranking either has ties (Family B) or unfaithfulness (Family A).
    This is the Design Space exhaustiveness. -/
theorem family_a_or_family_b
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop) :
    -- Family A: complete + unfaithful to some model
    ((ranking j k ∨ ranking k j) ∧
     ∃ f : Model, (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
                   (ranking k j ∧ attribution fs j f > attribution fs k f))
    ∨
    -- Family B: tie (not complete for this pair)
    (¬ ranking j k ∧ ¬ ranking k j) := by
  by_cases hcomp : ranking j k ∨ ranking k j
  · -- Family A: ranking is complete, so some model witnesses unfaithfulness
    left
    exact ⟨hcomp, stable_complete_unfaithful fs hrash ℓ j k hj hk hjk ranking hcomp⟩
  · -- Family B: ranking has a tie
    right
    push Not at hcomp
    exact hcomp

end UniversalImpossibility
