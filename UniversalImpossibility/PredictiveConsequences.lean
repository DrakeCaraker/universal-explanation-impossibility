import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility
import Mathlib.Tactic.ByContra
import Mathlib.Tactic.Push

/-!
# Predictive Consequences

Quantifiable structural predictions for maximally incompatible systems.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

/-- Stability forces unfaithfulness somewhere. -/
theorem stable_implies_unfaithful
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hs : stable S E) :
    ∃ θ, S.incompatible (E θ) (S.explain θ) := by
  by_contra hall
  push Not at hall
  exact bilemma S hmax E hall hs

/-- On any Rashomon pair, a stable E is unfaithful at ≥ 1 of 2 witnesses. -/
theorem rashomon_unfaithfulness
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hs : stable S E)
    (θ₁ θ₂ : Θ) (hobs : S.observe θ₁ = S.observe θ₂)
    (hinc : S.incompatible (S.explain θ₁) (S.explain θ₂)) :
    S.incompatible (E θ₁) (S.explain θ₁) ∨
    S.incompatible (E θ₂) (S.explain θ₂) := by
  have heq := hs θ₁ θ₂ hobs
  by_cases h : S.incompatible (E θ₁) (S.explain θ₁)
  · left; exact h
  · right
    have heq_exp := hmax _ _ h
    have : E θ₂ = S.explain θ₁ := heq.symm.trans heq_exp
    rw [this]; exact hinc

/-- The only faithful explanation is explain itself. -/
theorem faithful_is_explain
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hf : faithful S E) :
    ∀ θ, E θ = S.explain θ :=
  faithful_eq_explain_of_maxIncompat S hmax E hf

/-- Faithful explanations are unique (both = explain). -/
theorem faithful_unique
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E₁ E₂ : Θ → H) (hf₁ : faithful S E₁) (hf₂ : faithful S E₂) :
    ∀ θ, E₁ θ = E₂ θ := by
  intro θ
  rw [faithful_is_explain S hmax E₁ hf₁ θ,
      faithful_is_explain S hmax E₂ hf₂ θ]

/-- The all-or-nothing theorem: either E = explain or E is unfaithful. -/
theorem all_or_nothing
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) :
    (faithful S E ∧ (∀ θ, E θ = S.explain θ)) ∨
    (∃ θ, S.incompatible (E θ) (S.explain θ)) := by
  by_cases hf : faithful S E
  · left; exact ⟨hf, faithful_is_explain S hmax E hf⟩
  · right
    simp only [faithful, not_forall, Classical.not_not] at hf
    exact hf
