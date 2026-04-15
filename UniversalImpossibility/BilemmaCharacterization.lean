import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility

/-!
# Bilemma Characterization

The bilemma holds iff no neutral element exists (for maximally incompatible
systems). Maximally incompatible ⇒ no neutral element ⇒ necessary for bilemma.
The converse of the first implication is false.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

/-- A neutral element is compatible with all native explanations. -/
def hasNeutralElement (S : ExplanationSystem Θ H Y) : Prop :=
  ∃ (c : H), ∀ (θ : Θ), ¬S.incompatible c (S.explain θ)

/-- Neutral element → F+S achievable. -/
theorem neutral_implies_faithful_stable (S : ExplanationSystem Θ H Y)
    (h : hasNeutralElement S) :
    ∃ (E : Θ → H), faithful S E ∧ stable S E :=
  let ⟨c, hc⟩ := h
  ⟨fun _ => c, fun θ => hc θ, fun _ _ _ => rfl⟩

/-- Bilemma → no neutral element. -/
theorem bilemma_implies_no_neutral (S : ExplanationSystem Θ H Y)
    (hbilemma : ∀ (E : Θ → H), faithful S E → stable S E → False) :
    ¬hasNeutralElement S := by
  intro ⟨c, hc⟩
  have ⟨E, hf, hs⟩ := neutral_implies_faithful_stable S ⟨c, hc⟩
  exact hbilemma E hf hs

/-- Maximally incompatible → no neutral element. -/
theorem maxIncompat_implies_no_neutral (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S) :
    ¬hasNeutralElement S := by
  intro ⟨c, hc⟩
  obtain ⟨θ₁, θ₂, _, hinc⟩ := S.rashomon
  have h1 := hmax c (S.explain θ₁) (hc θ₁)
  have h2 := hmax c (S.explain θ₂) (hc θ₂)
  have : S.explain θ₁ = S.explain θ₂ := h1.symm.trans h2
  rw [this] at hinc
  exact S.incompatible_irrefl _ hinc
