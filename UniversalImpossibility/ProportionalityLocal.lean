/-
  Local (per-model) proportionality suffices for the core impossibility.

  This file shows that gbdt_rashomon and gbdt_impossibility can be proved
  from ProportionalityLocal (per-model c) instead of proportionality_global.
  The DASH consensus equity (Corollary.lean) genuinely requires global c,
  but the impossibility itself does not.

  Confirmed by `#print axioms`: gbdt_impossibility depends on
  proportionality_global only because attribution_proportional derives
  from it. The proof only uses the per-model consequence.
-/
import UniversalImpossibility.General

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Per-model proportionality: each model has its own positive constant.
    Strictly weaker than proportionality_global. -/
def ProportionalityLocal : Prop :=
  ∀ (f : Model), ∃ c : ℝ, 0 < c ∧ ∀ (j : Fin fs.P),
    attribution fs j f = c * splitCount fs j f

/-- Global proportionality implies local. -/
theorem proportionality_global_implies_local
    (hg : ∃ c : ℝ, 0 < c ∧ ∀ (f : Model) (j : Fin fs.P),
      attribution fs j f = c * splitCount fs j f) :
    ProportionalityLocal fs := by
  intro f
  obtain ⟨c, hc, hcf⟩ := hg
  exact ⟨c, hc, fun j => hcf f j⟩

/-- First-mover dominates using only local proportionality. -/
theorem attribution_firstMover_gt_local
    (hprop : ProportionalityLocal fs)
    (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm : firstMover fs f = j) (hjk : j ≠ k) :
    attribution fs k f < attribution fs j f := by
  obtain ⟨c, hc_pos, hc_eq⟩ := hprop f
  rw [hc_eq j, hc_eq k]
  exact mul_lt_mul_of_pos_left (splitCount_firstMover_gt fs f j k ℓ hj hk hfm hjk) hc_pos

/-- GBDT satisfies Rashomon using only local proportionality. -/
theorem gbdt_rashomon_local (hprop : ProportionalityLocal fs) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := firstMover_surjective fs ℓ j hj
  obtain ⟨f', hfm'⟩ := firstMover_surjective fs ℓ k hk
  exact ⟨f, f',
    attribution_firstMover_gt_local fs hprop f j k ℓ hj hk hfm hjk,
    attribution_firstMover_gt_local fs hprop f' k j ℓ hk hj hfm' (Ne.symm hjk)⟩

/-- GBDT impossibility using only local proportionality.
    `#print axioms` for this theorem will show NO proportionality_global. -/
theorem gbdt_impossibility_local (hprop : ProportionalityLocal fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs (gbdt_rashomon_local fs hprop) ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility
