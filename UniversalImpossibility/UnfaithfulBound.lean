/-
  Unfaithfulness bounds and relaxation path convergence.

  S9:  Any stable complete ranking has unfaithfulness ≥ 1/2
  S10: The optimal unfaithful ranking assigns ties
  S11: The two relaxation paths converge

  Supplement: §The Price of Dropping Faithfulness
-/
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- S9: Any stable complete ranking is unfaithful to some model.
    Under the Rashomon property, whichever ordering the ranking picks,
    there's a model that disagrees. -/
theorem stable_complete_unfaithful
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (hcomp : ranking j k ∨ ranking k j) :
    -- There exists a model where the ranking disagrees with attributions
    ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f) := by
  obtain ⟨f, f', hjk_f, hkj_f'⟩ := hrash ℓ j k hj hk hjk
  cases hcomp with
  | inl h => exact ⟨f', Or.inl ⟨h, hkj_f'⟩⟩
  | inr h => exact ⟨f, Or.inr ⟨h, hjk_f⟩⟩

/-- S10: The optimal resolution for symmetric features is to report a tie.
    If the ranking allows ties (neither j ≻ k nor k ≻ j), then no
    unfaithfulness occurs (a tie is never "wrong").

    We formalize: a ranking that assigns no ordering to within-group
    pairs is never contradicted by any model's attributions. -/
theorem tie_is_never_unfaithful
    (j k : Fin fs.P)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_tie : ¬ ranking j k ∧ ¬ ranking k j) :
    -- No model witnesses unfaithfulness
    ¬ ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f) := by
  push Not
  intro f
  exact ⟨fun h => absurd h h_tie.1, fun h => absurd h h_tie.2⟩

/-- S11: Path convergence — dropping faithfulness and dropping completeness
    both lead to ties for symmetric features.

    Dropping completeness: DASH reports ties (consensus_equity).
    Dropping faithfulness: the optimal stable ranking also reports ties (S10).
    Both paths converge.

    We formalize: under the Rashomon property, the only ranking that avoids
    unfaithfulness for a symmetric pair (j,k) is one that ranks neither above the other. -/
theorem unfaithfulness_free_implies_tie
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    -- No model witnesses unfaithfulness for this pair
    (h_no_unfaith : ¬ ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f)) :
    -- Then the ranking must be a tie
    ¬ ranking j k ∧ ¬ ranking k j := by
  obtain ⟨f, f', hjk_f, hkj_f'⟩ := hrash ℓ j k hj hk hjk
  constructor
  · intro h_jk
    exact h_no_unfaith ⟨f', Or.inl ⟨h_jk, hkj_f'⟩⟩
  · intro h_kj
    exact h_no_unfaith ⟨f, Or.inr ⟨h_kj, hjk_f⟩⟩

end UniversalImpossibility
