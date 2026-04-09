/-
  Model Selection Impossibility.

  Formalizes the impossibility of simultaneously faithful, stable, and
  complete model rankings when multiple near-optimal models exist.
  This is structurally identical to the Attribution Impossibility
  (Trilemma.lean) — the proof is a direct analogue.

  Supplement: §Model Selection under Model Multiplicity
-/
import UniversalImpossibility.Defs

set_option autoImplicit false

namespace UniversalImpossibility

-- We work with an abstract type of "candidate models" being ranked
-- and an abstract type of "evaluation instances" (e.g., validation splits)
variable (CandidateModel : Type) (EvalInstance : Type)

-- Quality of a candidate model on an evaluation instance
variable (quality : CandidateModel → EvalInstance → ℝ)

/-! ### The Model Rashomon Property -/

/-- The Model Rashomon Property: for any two distinct candidate models,
    there exist evaluation instances ranking them in opposite orders.
    This holds whenever the model class contains near-optimal models
    whose relative performance varies across evaluation datasets —
    a consequence of model multiplicity in the Rashomon set. -/
def ModelRashimonProperty : Prop :=
  ∀ (m₁ m₂ : CandidateModel), m₁ ≠ m₂ →
    ∃ d d' : EvalInstance,
      quality m₁ d > quality m₂ d ∧
      quality m₂ d' > quality m₁ d'

/-! ### The Model Selection Impossibility -/

/-- **The Model Selection Impossibility (biconditional version).**
    No evaluation-independent ranking can faithfully represent all
    evaluation instances' model quality orderings when near-optimal
    models exist.

    A ranking that is:
    • **Faithful** — reflects each evaluation's quality ordering
    • **Stable** — the same relation regardless of which evaluation is used
    necessarily FAILS to be:
    • **Complete** — some model pairs cannot be decided

    This is a formal impossibility: assuming faithfulness for all
    evaluation instances and a fixed (stable) ranking derives `False`.

    The resolution parallels the attribution case: use partial orders
    where near-equivalent models are incomparable, or use ensemble
    evaluation (DASH) where near-equivalent models are tied. -/
theorem model_selection_impossibility
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (ranking : CandidateModel → CandidateModel → Prop)
    (h_faithful : ∀ d : EvalInstance,
      ranking m₁ m₂ ↔ quality m₁ d > quality m₂ d) :
    False := by
  obtain ⟨d, d', h1, h2⟩ := hrash m₁ m₂ hne
  have hrank : ranking m₁ m₂ := (h_faithful d).mpr h1
  have hcontra : quality m₁ d' > quality m₂ d' := (h_faithful d').mp hrank
  linarith

/-! ### Implication-only version -/

/-- **The Model Selection Impossibility (weak faithfulness version).**
    The impossibility holds with implication-only faithfulness
    rather than the biconditional. This requires antisymmetry of the
    ranking (a standard property of strict orders).

    The result: given faithfulness (→) and antisymmetry, completeness
    is impossible — the ranking cannot decide every model pair. -/
theorem model_selection_impossibility_weak
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (ranking : CandidateModel → CandidateModel → Prop)
    -- Faithfulness (implication only):
    (h_faithful_12 : ∀ d : EvalInstance,
      quality m₁ d > quality m₂ d → ranking m₁ m₂)
    (h_faithful_21 : ∀ d : EvalInstance,
      quality m₂ d > quality m₁ d → ranking m₂ m₁)
    -- Antisymmetry (standard for strict orders):
    (h_antisym : ¬ (ranking m₁ m₂ ∧ ranking m₂ m₁)) :
    -- Completeness is impossible:
    ¬ (ranking m₁ m₂ ∨ ranking m₂ m₁) := by
  intro hcomp
  obtain ⟨d, d', h1, h2⟩ := hrash m₁ m₂ hne
  cases hcomp with
  | inl h12_rank =>
    exact h_antisym ⟨h12_rank, h_faithful_21 d' h2⟩
  | inr h21_rank =>
    exact h_antisym ⟨h_faithful_12 d h1, h21_rank⟩

end UniversalImpossibility
