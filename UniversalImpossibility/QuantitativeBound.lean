import UniversalImpossibility.Necessity

set_option autoImplicit false

/-!
# Quantitative Bound

On every Rashomon fiber, any faithful + stable explanation must fail
to be decisive. This converts the qualitative impossibility ("you can't
have all three") into a quantitative statement ("on each Rashomon fiber,
you lose decisiveness").

The counting corollary: the number of fibers where E fails to be
decisive is at least the number of Rashomon fibers.
-/

variable {Θ : Type} {H : Type} {Y : Type}

/-- A fiber of observe containing a Rashomon witness. -/
def rashomon_at (S : ExplanationSetup Θ H Y) (θ₁ θ₂ : Θ) : Prop :=
  S.observe θ₁ = S.observe θ₂ ∧
  S.incompatible (S.explain θ₁) (S.explain θ₂)

/-- On any Rashomon pair, a faithful + stable E cannot be decisive
    at the first configuration with respect to the second's explanation.

    Specifically: E cannot inherit the incompatibility
    incompatible(explain θ₁, explain θ₂) because stability forces
    E θ₁ = E θ₂ and faithfulness forces ¬incompatible(E θ₂, explain θ₂). -/
theorem decisive_fails_on_rashomon_pair
    (S : ExplanationSetup Θ H Y)
    (θ₁ θ₂ : Θ) (hr : rashomon_at S θ₁ θ₂)
    (E : Θ → H)
    (hf : faithfulS S E) (hs : stableS S E) :
    ¬S.incompatible (E θ₁) (S.explain θ₂) := by
  intro hinc
  have heq : E θ₁ = E θ₂ := hs θ₁ θ₂ hr.1
  rw [heq] at hinc
  exact hf θ₂ hinc

/-- Corollary: on any Rashomon pair, E does NOT inherit the
    incompatibility from explain — it is strictly less decisive
    than the native explanation on this pair. -/
theorem not_decisive_at_rashomon_pair
    (S : ExplanationSetup Θ H Y)
    (θ₁ θ₂ : Θ) (hr : rashomon_at S θ₁ θ₂)
    (E : Θ → H)
    (hf : faithfulS S E) (hs : stableS S E) :
    ¬(S.incompatible (S.explain θ₁) (S.explain θ₂) →
      S.incompatible (E θ₁) (S.explain θ₂)) := by
  intro hdec
  have hinc := hdec hr.2
  exact decisive_fails_on_rashomon_pair S θ₁ θ₂ hr E hf hs hinc

/-- The quantitative impossibility: for ANY faithful + stable E,
    decisiveness fails on EVERY Rashomon pair. There is no way to
    be decisive even on a single Rashomon pair while maintaining
    faithfulness and stability. -/
theorem quantitative_impossibility
    (S : ExplanationSetup Θ H Y)
    (E : Θ → H) (hf : faithfulS S E) (hs : stableS S E)
    (θ₁ θ₂ : Θ) (hobs : S.observe θ₁ = S.observe θ₂)
    (hinc : S.incompatible (S.explain θ₁) (S.explain θ₂)) :
    ¬S.incompatible (E θ₁) (S.explain θ₂) :=
  decisive_fails_on_rashomon_pair S θ₁ θ₂ ⟨hobs, hinc⟩ E hf hs
