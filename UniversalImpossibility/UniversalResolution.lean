import UniversalImpossibility.ExplanationSystem
import Mathlib.GroupTheory.GroupAction.Defs

/-!
# G-Invariant Resolution Framework

Abstract framework showing that symmetry-respecting resolutions of the
explanation impossibility are automatically stable.  If a group G acts on the
configuration space Θ, preserves observables, and acts transitively on each
observe-equivalence class (fiber), then any G-invariant resolution R satisfies
stability: observe θ₁ = observe θ₂ → R θ₁ = R θ₂.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type} {G : Type}

/-- A symmetry structure on an explanation system: a group G acting on Θ that
    preserves observables and acts transitively on each observe-fiber. -/
structure HasSymmetry (S : ExplanationSystem Θ H Y) (G : Type)
    [Group G] [MulAction G Θ] where
  /-- G preserves observables: g • θ has the same observable as θ. -/
  observe_invariant : ∀ (g : G) (θ : Θ), S.observe (g • θ) = S.observe θ
  /-- G acts transitively on observe-fibers: any two configurations with the
      same observable are related by some group element. -/
  fiber_transitive : ∀ (θ₁ θ₂ : Θ),
    S.observe θ₁ = S.observe θ₂ → ∃ (g : G), g • θ₁ = θ₂

/-- A resolution R is G-invariant if it is constant on G-orbits. -/
def gInvariant (R : Θ → H) (G : Type) [Group G] [MulAction G Θ] : Prop :=
  ∀ (g : G) (θ : Θ), R (g • θ) = R θ

/-- If R is G-invariant and G acts transitively on observe-fibers, then R is
    stable (i.e., R factors through the observable map).

    Proof:
    1. Given θ₁ θ₂ with observe θ₁ = observe θ₂
    2. By fiber transitivity, ∃ g such that g • θ₁ = θ₂
    3. By G-invariance, R (g • θ₁) = R θ₁
    4. Since g • θ₁ = θ₂, we get R θ₂ = R θ₁ -/
theorem gInvariant_stable
    (S : ExplanationSystem Θ H Y) (R : Θ → H)
    [Group G] [MulAction G Θ]
    (sym : HasSymmetry S G)
    (hInv : gInvariant R G) :
    stable S R := by
  intro θ₁ θ₂ hobs
  obtain ⟨g, hg⟩ := sym.fiber_transitive θ₁ θ₂ hobs
  have h := hInv g θ₁
  rw [hg] at h
  exact h.symm
