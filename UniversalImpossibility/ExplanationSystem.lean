/-!
# Abstract Explanation System

The universal impossibility framework. A system S : Θ → Y is explained by
E : Θ → H. The Rashomon property says equivalent configurations can produce
incompatible explanations.
-/

set_option autoImplicit false

/-- An explanation system with configuration space Θ, explanation space H,
    and observable space Y. -/
structure ExplanationSystem (Θ : Type) (H : Type) (Y : Type) where
  /-- The observation map: what the system produces. -/
  observe : Θ → Y
  /-- The explanation map: how we interpret a configuration. -/
  explain : Θ → H
  /-- Incompatibility relation on explanations. -/
  incompatible : H → H → Prop
  /-- The Rashomon property: equivalent configurations produce incompatible
      explanations. -/
  rashomon : ∃ θ₁ θ₂ : Θ,
    observe θ₁ = observe θ₂ ∧ incompatible (explain θ₁) (explain θ₂)

variable {Θ : Type} {H : Type} {Y : Type}

/-- Faithfulness: the explanation reflects the configuration's structure.
    Modeled as: E agrees with the system's native explanation map. -/
def faithful (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  E = S.explain

/-- Stability: the explanation factors through the observable map.
    Equivalent configurations get the same explanation. -/
def stable (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ → E θ₁ = E θ₂

/-- Decisiveness: E resolves incompatible explanations — it never maps
    two configurations to incompatible outputs (it commits). When combined
    with faithfulness, this means E = explain, which inherits the
    incompatibility from the Rashomon property. -/
def decisive (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), S.incompatible (E θ₁) (E θ₂) → E θ₁ ≠ E θ₂

/-- The Universal Explanation Impossibility.

No explanation of a system with the Rashomon property can be simultaneously
faithful, stable, and decisive.

Proof structure:
1. Rashomon gives θ₁, θ₂ with same output but incompatible explanations
2. Faithfulness: E = explain, so E(θ₁) and E(θ₂) are incompatible
3. Decisiveness: incompatible → E(θ₁) ≠ E(θ₂)
4. Stability: same output → E(θ₁) = E(θ₂)
5. Contradiction -/
theorem explanation_impossibility (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hf : faithful S E) (hs : stable S E) (hd : decisive S E) :
    False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := S.rashomon
  have heq : E θ₁ = E θ₂ := hs θ₁ θ₂ hobs
  unfold faithful at hf
  subst hf
  exact absurd heq (hd θ₁ θ₂ hinc)
