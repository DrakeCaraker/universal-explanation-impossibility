/-
  Abstract iterative optimizer framework. Any iterative optimization
  process under feature collinearity produces a "dominant feature" per
  group that accumulates more attribution through iteration.

  Instances: GBDT (first-mover), Lasso (selected feature), NN (captured feature).
-/
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Abstract Iterative Optimizer -/

/-- An iterative optimizer is characterized by a dominant-feature function
    satisfying dominance (higher attribution) and surjectivity (each feature
    can be dominant under some initialization). -/
structure IterativeOptimizer where
  /-- The dominant feature for a given model -/
  dominant : Model → Fin fs.P
  /-- The dominant feature has strictly higher attribution than all
      other features in the same group -/
  dominant_gt : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
    dominant f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ dominant f →
    attribution fs k f < attribution fs (dominant f) f
  /-- Every feature in every group can be dominant -/
  dominant_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
    j ∈ fs.group ℓ → ∃ f : Model, dominant f = j

/-! ### Iterative Optimizer → Rashomon Property → Impossibility -/

/-- Any iterative optimizer satisfies the Rashomon property. -/
theorem iterative_rashomon (opt : IterativeOptimizer fs) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := opt.dominant_surjective ℓ j hj
  obtain ⟨f', hfm'⟩ := opt.dominant_surjective ℓ k hk
  refine ⟨f, f', ?_, ?_⟩
  · -- j is dominant in f, so attribution j f > attribution k f
    have h_dom_in : opt.dominant f ∈ fs.group ℓ := by rw [hfm]; exact hj
    have h := opt.dominant_gt f ℓ k h_dom_in hk (by rw [hfm]; exact Ne.symm hjk)
    rwa [hfm] at h
  · -- k is dominant in f', so attribution k f' > attribution j f'
    have h_dom_in : opt.dominant f' ∈ fs.group ℓ := by rw [hfm']; exact hk
    have h := opt.dominant_gt f' ℓ j h_dom_in hj (by rw [hfm']; exact hjk)
    rwa [hfm'] at h

/-- The Attribution Impossibility holds for any iterative optimizer. -/
theorem iterative_impossibility (opt : IterativeOptimizer fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs (iterative_rashomon fs opt) ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility
