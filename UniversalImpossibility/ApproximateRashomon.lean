import UniversalImpossibility.ExplanationSystem

/-!
# Approximate Rashomon Property

Extension to handle approximate observational equivalence.
When observations are approximately (not exactly) equal, the
impossibility still holds under ε-stability.

This bridges the gap between the exact framework (synthetic data,
bimodal gaps) and real-world applications (approximate symmetry,
continuous flip rates via the Gaussian formula).
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

/-- An approximate explanation system adds a "nearness" relation on Y
    representing approximate observational equivalence. -/
structure ApproxExplanationSystem (Θ : Type) (H : Type) (Y : Type)
    extends ExplanationSystem Θ H Y where
  /-- Approximate observational equivalence. -/
  near : Y → Y → Prop
  /-- near is reflexive (every observation is near itself). -/
  near_refl : ∀ y, near y y
  /-- Exact equality implies nearness. -/
  eq_implies_near : ∀ y₁ y₂, y₁ = y₂ → near y₁ y₂

/-- ε-Stability: the explanation is constant on approximately-equivalent
    observations. Stronger than exact stability. -/
def approxStable (S : ApproxExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ θ₁ θ₂, S.near (S.observe θ₁) (S.observe θ₂) → E θ₁ = E θ₂

/-- ε-Stability implies exact stability (since equal implies near). -/
theorem approxStable_implies_stable (S : ApproxExplanationSystem Θ H Y)
    (E : Θ → H) (has : approxStable S E) :
    stable S.toExplanationSystem E :=
  fun θ₁ θ₂ heq => has θ₁ θ₂ (S.eq_implies_near _ _ heq)

/-- The approximate impossibility: if the Rashomon property holds for the
    underlying exact system, then F + approxStable + D is impossible.
    This follows immediately from the exact impossibility + the fact that
    approxStable implies stable. -/
theorem approx_impossibility (S : ApproxExplanationSystem Θ H Y)
    (E : Θ → H)
    (hf : faithful S.toExplanationSystem E)
    (has : approxStable S E)
    (hd : decisive S.toExplanationSystem E) : False :=
  explanation_impossibility S.toExplanationSystem E hf
    (approxStable_implies_stable S E has) hd

/-- The approximate Rashomon property: approximately-equivalent observations
    with incompatible explanations. -/
def approxRashomon (S : ApproxExplanationSystem Θ H Y) : Prop :=
  ∃ θ₁ θ₂, S.near (S.observe θ₁) (S.observe θ₂) ∧
    S.incompatible (S.explain θ₁) (S.explain θ₂)

/-- If approximate Rashomon holds, then no E can be faithful +
    ε-stable + decisive. This is the approximate trilemma. -/
theorem approx_trilemma (S : ApproxExplanationSystem Θ H Y)
    (hrash : approxRashomon S)
    (E : Θ → H)
    (hf : faithful S.toExplanationSystem E)
    (has : approxStable S E)
    (hd : decisive S.toExplanationSystem E) : False := by
  obtain ⟨θ₁, θ₂, hnear, hinc⟩ := hrash
  have h1 : S.incompatible (E θ₁) (S.explain θ₂) := hd θ₁ (S.explain θ₂) hinc
  have h2 : E θ₁ = E θ₂ := has θ₁ θ₂ hnear
  rw [h2] at h1
  exact hf θ₂ h1

/-- Exact Rashomon implies approximate Rashomon (since equal implies near). -/
theorem exact_implies_approx_rashomon (S : ApproxExplanationSystem Θ H Y) :
    approxRashomon S := by
  obtain ⟨θ₁, θ₂, heq, hinc⟩ := S.rashomon
  exact ⟨θ₁, θ₂, S.eq_implies_near _ _ heq, hinc⟩
