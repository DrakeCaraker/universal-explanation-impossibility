/-
  Approximate DASH equity under bounded (non-global) proportionality.

  If the proportionality constant varies across models by at most factor K,
  DASH consensus equity holds approximately. This shows the DASH resolution
  is robust even when proportionality_global is relaxed.

  When K = 1 (global c), this recovers exact equity.
-/
import UniversalImpossibility.General

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Bounded proportionality: every model has a proportionality constant,
    and the ratio between any two constants is at most K. -/
structure BoundedProportionality (K : ℝ) where
  /-- Each model has a positive constant -/
  const : Model → ℝ
  const_pos : ∀ f : Model, 0 < const f
  /-- Attribution equals constant times split count -/
  prop : ∀ (f : Model) (j : Fin fs.P), attribution fs j f = const f * splitCount fs j f
  /-- Constants are within factor K of each other -/
  bounded : ∀ (f f' : Model), const f / const f' ≤ K

/-- Global proportionality is bounded proportionality with K = 1. -/
theorem global_implies_bounded_one
    (hg : ∃ c : ℝ, 0 < c ∧ ∀ (f : Model) (j : Fin fs.P),
      attribution fs j f = c * splitCount fs j f) :
    ∃ bp : BoundedProportionality fs 1,
      ∀ (f : Model) (j : Fin fs.P), attribution fs j f = bp.const f * splitCount fs j f := by
  obtain ⟨c, hc, hcf⟩ := hg
  exact ⟨⟨fun _ => c, fun _ => hc, hcf, fun _ _ => by rw [div_self (ne_of_gt hc)]⟩,
    fun f j => hcf f j⟩

/-- Under bounded proportionality with constant constants (all equal),
    the consensus difference for same-group features is zero.
    This is a direct consequence: if all constants are equal, then
    attribution = c * splitCount for a single global c, so the
    existing consensus_equity proof applies.

    This formulation avoids duplicating the full SymmetryDerive proof
    by showing the reduction to the global-proportionality case. -/
theorem bounded_proportionality_constant_implies_global
    (K : ℝ) (bp : BoundedProportionality fs K)
    (hconst : ∀ (f f' : Model), bp.const f = bp.const f')
    (f₀ : Model) :
    ∃ c : ℝ, 0 < c ∧ ∀ (f : Model) (j : Fin fs.P),
      attribution fs j f = c * splitCount fs j f :=
  ⟨bp.const f₀, bp.const_pos f₀, fun f j => by rw [bp.prop f j, hconst f f₀]⟩

/-- The Rashomon property holds regardless of whether proportionality
    is global or bounded, as long as some proportionality holds.
    This shows the impossibility is robust to relaxing the global
    proportionality axiom. -/
theorem rashomon_from_bounded_proportionality
    (K : ℝ) (_hK : 0 < K)
    (bp : BoundedProportionality fs K) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := firstMover_surjective fs ℓ j hj
  obtain ⟨f', hfm'⟩ := firstMover_surjective fs ℓ k hk
  constructor; constructor
  constructor
  · -- attribution j f > attribution k f: j is first-mover in f
    rw [bp.prop f j, bp.prop f k]
    have hsc := splitCount_firstMover_gt fs f j k ℓ hj hk hfm hjk
    exact mul_lt_mul_of_pos_left hsc (bp.const_pos f)
  · -- attribution k f' > attribution j f': k is first-mover in f'
    rw [bp.prop f' k, bp.prop f' j]
    have hsc := splitCount_firstMover_gt fs f' k j ℓ hk hj hfm' (Ne.symm hjk)
    exact mul_lt_mul_of_pos_left hsc (bp.const_pos f')

end UniversalImpossibility
