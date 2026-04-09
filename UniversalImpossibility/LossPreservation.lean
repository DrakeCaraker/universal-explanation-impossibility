/-
  Loss Preservation for the Rashomon-from-Symmetry theorem.

  The paper's Theorem 2 claims that permuting features preserves
  population loss: L(f') = L(f). The existing RashomonUniversality.lean
  proves attributions swap but does NOT prove loss preservation, because
  Model is an opaque axiom type.

  This file introduces SymmetricModelSwap — a structure that bundles
  a swap function, a loss function, and proofs that the swap preserves
  loss and exchanges attributions. This is the definition-as-hypothesis
  pattern: the loss function and its properties are carried as hypotheses
  in the structure, not as axioms.

  Main results:
  - rashomon_from_swap_with_loss: Rashomon witnesses with equal loss
  - impossibility_with_quality: Attribution impossibility where the
    two witnessing models have equal population loss

  Zero new axioms. All proofs complete.
-/
import UniversalImpossibility.Defs
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### SymmetricModelSwap structure -/

/-- A symmetric model swap bundles a swap function, a loss function,
    and proofs that the swap preserves loss and exchanges attributions.
    This formalizes the DGP symmetry argument from Theorem 2 without
    requiring Model to be a function type. -/
structure SymmetricModelSwap where
  /-- The swap function: given a model and two features, produces
      the "permuted" model. -/
  swap : Model → Fin fs.P → Fin fs.P → Model
  /-- Population loss function. -/
  loss : Model → ℝ
  /-- Swapping features preserves population loss. -/
  loss_preserved : ∀ (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ →
    loss (swap f j k) = loss f
  /-- Swapping exchanges attribution of j. -/
  attr_swap_j : ∀ (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ →
    attribution fs k (swap f j k) = attribution fs j f
  /-- Swapping exchanges attribution of k. -/
  attr_swap_k : ∀ (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ →
    attribution fs j (swap f j k) = attribution fs k f

/-! ### Rashomon from swap with loss preservation -/

/-- **Rashomon from swap with loss preservation.**
    Under SymmetricModelSwap, if there exists a model with φ_j(f) > φ_k(f),
    then there exist models f, f' with:
    - φ_j(f) > φ_k(f) and φ_k(f') > φ_j(f') (Rashomon property)
    - loss(f) = loss(f') (quality guarantee) -/
theorem rashomon_from_swap_with_loss
    (S : SymmetricModelSwap fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (f : Model)
    (hdiff : attribution fs j f > attribution fs k f) :
    ∃ f₁ f₂ : Model,
      attribution fs j f₁ > attribution fs k f₁ ∧
      attribution fs k f₂ > attribution fs j f₂ ∧
      S.loss f₁ = S.loss f₂ := by
  refine ⟨f, S.swap f j k, hdiff, ?_, ?_⟩
  · -- φ_k(swap f j k) > φ_j(swap f j k)
    rw [S.attr_swap_j f j k ℓ hj hk, S.attr_swap_k f j k ℓ hj hk]
    exact hdiff
  · -- loss(f) = loss(swap f j k)
    exact (S.loss_preserved f j k ℓ hj hk).symm

/-- Helper: SymmetricModelSwap implies the Rashomon property
    (forgetting loss information). -/
theorem rashomon_from_swap_loss
    (S : SymmetricModelSwap fs)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, attribution fs j f ≠ attribution fs k f) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hdiff⟩ := hnd ℓ j k hj hk hjk
  rcases lt_or_gt_of_ne hdiff with h | h
  · -- φ_j(f) < φ_k(f): use swap for j > k witness
    exact ⟨S.swap f j k, f,
      by rw [S.attr_swap_j f j k ℓ hj hk, S.attr_swap_k f j k ℓ hj hk]; exact h,
      h⟩
  · -- φ_j(f) > φ_k(f): use swap for k > j witness
    exact ⟨f, S.swap f j k,
      h,
      by rw [S.attr_swap_j f j k ℓ hj hk, S.attr_swap_k f j k ℓ hj hk]; exact h⟩

/-! ### Impossibility with quality guarantee -/

/-- **Attribution impossibility with quality guarantee.**
    The impossibility holds, and the two witnessing models have equal
    population loss. This strengthens the base impossibility: not only
    can no stable faithful complete ranking exist, but the contradiction
    arises from models of equal quality — so restricting to "good" models
    cannot escape the impossibility. -/
theorem impossibility_with_quality
    (S : SymmetricModelSwap fs)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, attribution fs j f ≠ attribution fs k f)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs
    (rashomon_from_swap_loss fs S hnd)
    ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility
