/-!
# Abstract Explanation System

The universal impossibility framework. An explanation system S : Θ → Y
is explained by E : Θ → H. The Rashomon property says observationally
equivalent configurations can produce incompatible explanations.

## The Trilemma

No explanation E can simultaneously be:
- **Faithful**: E(θ) never contradicts the system's native explanation
- **Stable**: E(θ₁) = E(θ₂) whenever observe(θ₁) = observe(θ₂)
- **Decisive**: E(θ) commits to every distinction the native explanation makes

Each pair is achievable:
- Faithful + stable: abstain on controversial distinctions (ties)
- Faithful + decisive: use the native explanation (unstable under retraining)
- Stable + decisive: pick an arbitrary fixed explanation (unfaithful)
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
  /-- Incompatibility is irreflexive: no explanation contradicts itself. -/
  incompatible_irrefl : ∀ (h : H), ¬incompatible h h
  /-- The Rashomon property: equivalent configurations produce incompatible
      explanations. -/
  rashomon : ∃ θ₁ θ₂ : Θ,
    observe θ₁ = observe θ₂ ∧ incompatible (explain θ₁) (explain θ₂)

variable {Θ : Type} {H : Type} {Y : Type}

/-- Faithful: E never contradicts the native explanation.
    E(θ) may be less specific than explain(θ) but never disagrees with it. -/
def faithful (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ : Θ), ¬S.incompatible (E θ) (S.explain θ)

/-- Stability: the explanation factors through the observable map.
    Equivalent configurations get the same explanation. -/
def stable (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), S.observe θ₁ = S.observe θ₂ → E θ₁ = E θ₂

/-- Decisive: E inherits every incompatibility of the native explanation.
    Whatever the true explanation rules out, E also rules out.
    E is at least as committal as explain. -/
def decisive (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ : Θ) (h : H), S.incompatible (S.explain θ) h → S.incompatible (E θ) h

/-- The Universal Explanation Impossibility.

No explanation of a system with the Rashomon property can be simultaneously
faithful, stable, and decisive.

Proof structure:
1. Rashomon gives θ₁, θ₂ with same output but incompatible explanations
2. Decisiveness at θ₁: incompatible(explain θ₁, explain θ₂) → incompatible(E θ₁, explain θ₂)
3. Stability: observe θ₁ = observe θ₂ → E θ₁ = E θ₂, so incompatible(E θ₂, explain θ₂)
4. Faithfulness at θ₂: ¬incompatible(E θ₂, explain θ₂)
5. Contradiction between 3 and 4 -/
theorem explanation_impossibility (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hf : faithful S E) (hs : stable S E) (hd : decisive S E) :
    False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := S.rashomon
  have h1 : S.incompatible (E θ₁) (S.explain θ₂) := hd θ₁ (S.explain θ₂) hinc
  have h2 : E θ₁ = E θ₂ := hs θ₁ θ₂ hobs
  rw [h2] at h1
  exact hf θ₂ h1

-- ============================================================================
-- Tightness: each pair of properties is achievable
-- ============================================================================

/-- **Tightness 1: Faithful + Decisive (dropping Stable).**
    E = explain is always faithful and decisive. It is not stable in general
    because the Rashomon property means explain disagrees on some
    observationally equivalent pair. -/
theorem tightness_faithful_decisive (S : ExplanationSystem Θ H Y) :
    faithful S S.explain ∧ decisive S S.explain := by
  constructor
  · intro θ
    exact S.incompatible_irrefl (S.explain θ)
  · intro θ h hinc
    exact hinc

/-- **Tightness 2: Faithful + Stable (dropping Decisive).**
    If there exists a "neutral" element c ∈ H that is not incompatible with
    any native explanation, then the constant function E(θ) = c is faithful
    and stable (but not decisive, since it abstains on all distinctions). -/
theorem tightness_faithful_stable (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ (θ : Θ), ¬S.incompatible c (S.explain θ)) :
    faithful S (fun _ => c) ∧ stable S (fun _ => c) := by
  constructor
  · intro θ
    exact hc θ
  · intro _ _ _
    rfl

/-- **Tightness 3: Stable + Decisive (dropping Faithful).**
    If there exists a "maximally committal" element c ∈ H that inherits every
    incompatibility of every native explanation, then the constant function
    E(θ) = c is stable and decisive (but not faithful, since c may contradict
    some explain(θ)). -/
theorem tightness_stable_decisive (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ (θ : Θ) (h : H), S.incompatible (S.explain θ) h → S.incompatible c h) :
    stable S (fun _ => c) ∧ decisive S (fun _ => c) := by
  constructor
  · intro _ _ _
    rfl
  · intro θ h hinc
    exact hc θ h hinc
