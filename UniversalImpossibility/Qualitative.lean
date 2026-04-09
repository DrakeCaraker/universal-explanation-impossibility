/-
  Qualitative impossibility: the core result from just two axioms.

  1. FirstMoverSurjective — each feature can be first-mover
  2. FirstMoverDominates — first-mover has strictly higher attribution

  No split counts, no proportionality, no numTrees. This is the minimal
  axiom set for the impossibility theorem.

  The quantitative axioms (split-count formulas, proportionality) imply
  the qualitative ones, so this strictly generalizes the existing results.
-/
import UniversalImpossibility.Trilemma
import UniversalImpossibility.General

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Qualitative dominance: first-mover has strictly higher attribution
    than any other feature in its group. Implied by the split-count
    axioms + proportionality, but can stand alone. -/
def FirstMoverDominates : Prop :=
  ∀ (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ →
    firstMover fs f = j → j ≠ k →
    attribution fs k f < attribution fs j f

/-- Qualitative surjectivity: each feature in a group can be first-mover. -/
def FirstMoverSurjective : Prop :=
  ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
    j ∈ fs.group ℓ → ∃ f : Model, firstMover fs f = j

/-- From dominance + surjectivity alone, the Rashomon property follows.
    This is the minimal path to impossibility. -/
theorem rashomon_from_qualitative
    (hdom : FirstMoverDominates fs) (hsurj : FirstMoverSurjective fs) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := hsurj ℓ j hj
  obtain ⟨f', hfm'⟩ := hsurj ℓ k hk
  exact ⟨f, f',
    hdom f j k ℓ hj hk hfm hjk,
    hdom f' k j ℓ hk hj hfm' (Ne.symm hjk)⟩

/-- The impossibility from just two qualitative axioms.
    No split counts, no proportionality, no numTrees needed. -/
theorem impossibility_qualitative
    (hdom : FirstMoverDominates fs) (hsurj : FirstMoverSurjective fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs (rashomon_from_qualitative fs hdom hsurj)
    ℓ j k hj hk hjk ranking h_faithful

/-- The quantitative axioms imply qualitative dominance. -/
theorem quantitative_implies_dominates : FirstMoverDominates fs :=
  fun f j k ℓ hj hk hfm hjk =>
    attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk

/-- The surjectivity axiom is already qualitative. -/
theorem axiom_implies_surjective : FirstMoverSurjective fs :=
  firstMover_surjective fs

end UniversalImpossibility
