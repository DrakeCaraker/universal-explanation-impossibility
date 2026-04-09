/-
  Model Selection Design Space.

  Mirrors the Attribution Design Space (DesignSpaceFull.lean) for
  model selection. Under the Model Rashomon Property, every model
  ranking falls into Family A (complete + unfaithful) or Family B (tie).

  Supplement: §Model Selection under Model Multiplicity
-/
import UniversalImpossibility.ModelSelection

set_option autoImplicit false

namespace UniversalImpossibility

variable (CandidateModel EvalInstance : Type) (quality : CandidateModel → EvalInstance → ℝ)

/-- Any stable complete model ranking is unfaithful to some evaluation instance.
    Mirrors stable_complete_unfaithful for model selection. -/
theorem model_stable_complete_unfaithful
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (ranking : CandidateModel → CandidateModel → Prop)
    (hcomp : ranking m₁ m₂ ∨ ranking m₂ m₁) :
    ∃ d : EvalInstance,
      (ranking m₁ m₂ ∧ quality m₂ d > quality m₁ d) ∨
      (ranking m₂ m₁ ∧ quality m₁ d > quality m₂ d) := by
  obtain ⟨d, d', h1, h2⟩ := hrash m₁ m₂ hne
  cases hcomp with
  | inl h => exact ⟨d', Or.inl ⟨h, h2⟩⟩
  | inr h => exact ⟨d, Or.inr ⟨h, h1⟩⟩

/-- Under Model Rashomon, the only ranking avoiding unfaithfulness is a tie. -/
theorem model_unfaithfulness_free_implies_tie
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (ranking : CandidateModel → CandidateModel → Prop)
    (h_no_unfaith : ¬ ∃ d : EvalInstance,
      (ranking m₁ m₂ ∧ quality m₂ d > quality m₁ d) ∨
      (ranking m₂ m₁ ∧ quality m₁ d > quality m₂ d)) :
    ¬ ranking m₁ m₂ ∧ ¬ ranking m₂ m₁ := by
  obtain ⟨d, d', h1, h2⟩ := hrash m₁ m₂ hne
  constructor
  · intro hrank
    exact h_no_unfaith ⟨d', Or.inl ⟨hrank, h2⟩⟩
  · intro hrank
    exact h_no_unfaith ⟨d, Or.inr ⟨hrank, h1⟩⟩

/-- Model Selection Design Space: every ranking is Family A or Family B. -/
theorem model_family_a_or_family_b
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (ranking : CandidateModel → CandidateModel → Prop) :
    -- Family A: complete + unfaithful
    ((ranking m₁ m₂ ∨ ranking m₂ m₁) ∧
     ∃ d : EvalInstance,
      (ranking m₁ m₂ ∧ quality m₂ d > quality m₁ d) ∨
      (ranking m₂ m₁ ∧ quality m₁ d > quality m₂ d))
    ∨
    -- Family B: tie
    (¬ ranking m₁ m₂ ∧ ¬ ranking m₂ m₁) := by
  by_cases hcomp : ranking m₁ m₂ ∨ ranking m₂ m₁
  · exact Or.inl ⟨hcomp, model_stable_complete_unfaithful CandidateModel EvalInstance quality hrash m₁ m₂ hne ranking hcomp⟩
  · push Not at hcomp
    exact Or.inr hcomp

end UniversalImpossibility
