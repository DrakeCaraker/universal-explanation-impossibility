import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Necessity of the Rashomon Property

The Rashomon property is the exact boundary between possibility and
impossibility for fully specified systems.

- Sufficiency: Rashomon → impossibility (Theorem 1, ExplanationSystem.lean)
- Necessity: ¬Rashomon → possibility (this file)
-/

variable {Θ : Type} {H : Type} {Y : Type}

/-- An explanation setup WITHOUT the bundled Rashomon property. -/
structure ExplanationSetup (Θ : Type) (H : Type) (Y : Type) where
  observe : Θ → Y
  explain : Θ → H
  incompatible : H → H → Prop
  incompatible_irrefl : ∀ (h : H), ¬incompatible h h

/-- The Rashomon property as an external proposition. -/
def hasRashomon (S : ExplanationSetup Θ H Y) : Prop :=
  ∃ θ₁ θ₂ : Θ, S.observe θ₁ = S.observe θ₂ ∧
    S.incompatible (S.explain θ₁) (S.explain θ₂)

/-- Faithful for setup. -/
def faithfulS (S : ExplanationSetup Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ : Θ), ¬S.incompatible (E θ) (S.explain θ)

/-- Stable for setup. -/
def stableS (S : ExplanationSetup Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ → E θ₁ = E θ₂

/-- Decisive for setup. -/
def decisiveS (S : ExplanationSetup Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ : Θ) (h : H), S.incompatible (S.explain θ) h → S.incompatible (E θ) h

/-- Sufficiency: Rashomon → impossibility (for the unbundled version). -/
theorem impossibility_from_rashomon (S : ExplanationSetup Θ H Y)
    (hr : hasRashomon S) (E : Θ → H)
    (hf : faithfulS S E) (hs : stableS S E) (hd : decisiveS S E) : False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := hr
  have h1 := hd θ₁ (S.explain θ₂) hinc
  have h2 := hs θ₁ θ₂ hobs
  rw [h2] at h1
  exact hf θ₂ h1

/-- Necessity (fully specified case): If observe is injective,
    then E = explain satisfies all three properties. -/
theorem fully_specified_possibility (S : ExplanationSetup Θ H Y)
    (h_inj : ∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ → θ₁ = θ₂) :
    faithfulS S S.explain ∧ stableS S S.explain ∧ decisiveS S S.explain := by
  refine ⟨?_, ?_, ?_⟩
  · intro θ; exact S.incompatible_irrefl _
  · intro θ₁ θ₂ hobs
    have := h_inj θ₁ θ₂ hobs
    subst this; rfl
  · intro θ h hinc; exact hinc

/-- Contrapositive: existence of faithful+stable+decisive E
    implies the Rashomon property does NOT hold. -/
theorem no_rashomon_from_all_three (S : ExplanationSetup Θ H Y)
    (E : Θ → H) (hf : faithfulS S E) (hs : stableS S E) (hd : decisiveS S E) :
    ¬hasRashomon S :=
  fun hr => impossibility_from_rashomon S hr E hf hs hd
