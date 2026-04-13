import UniversalImpossibility.Necessity
import Mathlib.Tactic.ByContra

set_option autoImplicit false

/-!
# Biconditional Necessity: Rashomon ↔ Impossibility

## Main Result

When `incompatible = (· ≠ ·)` (the standard case used by all 8 derived instances),
the Rashomon property is the **exact boundary** between possibility and impossibility:

  ¬hasRashomon S ↔ ∃ E, faithfulS S E ∧ stableS S E ∧ decisiveS S E

This is stronger than the one-direction results in `Necessity.lean`:
- Forward (already proved): Rashomon → impossibility
- Backward (new, this file): ¬Rashomon → achievability **when incomp = (≠)**

## Why the general biconditional is FALSE

For a general incompatibility relation, ¬Rashomon does NOT imply achievability.
Counterexample: 6 configurations, 3 observations, 6 explanations.

  fibers: y₁={a,b}, y₂={c,d}, y₃={e,f}
  explain: a→h₁, b→h₂, c→h₃, d→h₄, e→h₅, f→h₆
  incomp: h₁⊥h₃, h₂⊥h₅, h₄⊥h₆ (plus symmetric closure)

Within each fiber, no incompatibility → ¬Rashomon holds.
But no stable E exists that is both faithful and decisive:
- Decisiveness requires E(y₁) ⊥ h₃ (from h₁⊥h₃) AND E(y₁) ⊥ h₅ (from h₂⊥h₅)
- Only h₁ is ⊥h₃, and only h₂ is ⊥h₅ — no single value satisfies both.

The gap: ¬Rashomon says within-fiber explanations are pairwise COMPATIBLE.
Stability requires E to be CONSTANT on fibers. Decisiveness requires E to
INHERIT all incompatibilities. When incomp is sparse, these jointly over-constrain.

When incomp = (≠), ¬Rashomon becomes "explain factors through observe"
(compatible = equal), which makes E = explain automatically stable.
-/

variable {Θ : Type} {H : Type} {Y : Type}

/-! ## The Biconditional for incomp = (≠) -/

/-- When ¬Rashomon holds AND explain factors through observe
    (which is automatic when incomp = (≠)),
    E = explain achieves all three properties. -/
theorem possibility_from_factoring (S : ExplanationSetup Θ H Y)
    (h_factor : ∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ →
      S.explain θ₁ = S.explain θ₂) :
    faithfulS S S.explain ∧ stableS S S.explain ∧ decisiveS S S.explain := by
  refine ⟨?_, ?_, ?_⟩
  · -- Faithful: ¬incomp(explain θ, explain θ) = irreflexivity
    intro θ; exact S.incompatible_irrefl _
  · -- Stable: follows from the factoring hypothesis
    intro θ₁ θ₂ hobs; exact h_factor θ₁ θ₂ hobs
  · -- Decisive: explain(θ) ⊥ h → explain(θ) ⊥ h (trivial)
    intro θ h hinc; exact hinc

/-- When incomp = (≠), ¬Rashomon is equivalent to "explain factors through observe."
    This is because ¬(h₁ ≠ h₂) is the same as h₁ = h₂. -/
theorem not_rashomon_iff_factoring_neq (S : ExplanationSetup Θ H Y)
    (h_neq : ∀ (h₁ h₂ : H), S.incompatible h₁ h₂ ↔ h₁ ≠ h₂) :
    ¬hasRashomon S ↔
    (∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ → S.explain θ₁ = S.explain θ₂) := by
  constructor
  · -- (→) ¬Rashomon → factoring
    intro h_nr θ₁ θ₂ hobs
    by_contra h_ne
    apply h_nr
    exact ⟨θ₁, θ₂, hobs, (h_neq _ _).mpr h_ne⟩
  · -- (←) factoring → ¬Rashomon
    intro h_factor h_rash
    obtain ⟨θ₁, θ₂, hobs, hinc⟩ := h_rash
    have h_eq := h_factor θ₁ θ₂ hobs
    rw [h_eq] at hinc
    exact S.incompatible_irrefl _ hinc

/-- **The Biconditional (for incomp = ≠).**

    The Rashomon property is the exact boundary:
    ¬Rashomon ↔ ∃ E with all three properties.

    This is the strongest form of the necessity theorem. -/
theorem rashomon_biconditional_neq (S : ExplanationSetup Θ H Y)
    (h_neq : ∀ (h₁ h₂ : H), S.incompatible h₁ h₂ ↔ h₁ ≠ h₂) :
    ¬hasRashomon S ↔
    (∃ E : Θ → H, faithfulS S E ∧ stableS S E ∧ decisiveS S E) := by
  constructor
  · -- (→) ¬Rashomon → achievable: E = explain works
    intro h_nr
    have h_factor := (not_rashomon_iff_factoring_neq S h_neq).mp h_nr
    exact ⟨S.explain, possibility_from_factoring S h_factor⟩
  · -- (←) achievable → ¬Rashomon: contrapositive of impossibility
    intro ⟨E, hf, hs, hd⟩
    exact no_rashomon_from_all_three S E hf hs hd
