/-
  Lasso / sparse regression: the Attribution Impossibility holds with
  INFINITE attribution ratio. Under collinearity, Lasso selects
  one feature per correlated group and zeros out the rest.

  Properties are theorem hypotheses, not global axioms.
-/
import UniversalImpossibility.Iterative

set_option autoImplicit false

namespace UniversalImpossibility.Lasso

variable (fs : FeatureSpace)

/-- Lasso inherits the Attribution Impossibility. The selected feature gets
    positive attribution; all other same-group features get zero.
    The attribution ratio is literally infinite (positive vs zero). -/
theorem lasso_impossibility
    (selected : Model → Fin fs.P)
    (selected_pos : ∀ (f : Model), 0 < attribution fs (selected f) f)
    (non_selected_zero : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      selected f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ selected f →
      attribution fs k f = 0)
    (selected_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
      j ∈ fs.group ℓ → ∃ f : Model, selected f = j)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False := by
  have opt : IterativeOptimizer fs := {
    dominant := selected
    dominant_gt := by
      intro f ℓ' k' h_sel_in hk' hne
      rw [non_selected_zero f ℓ' k' h_sel_in hk' hne]
      exact selected_pos f
    dominant_surjective := selected_surjective
  }
  exact iterative_impossibility fs opt ℓ j k hj hk hjk ranking h_faithful

/-- The Lasso attribution ratio is infinite: selected feature has
    positive attribution while others have zero. -/
theorem lasso_ratio_infinite
    (selected : Model → Fin fs.P)
    (selected_pos : ∀ (f : Model), 0 < attribution fs (selected f) f)
    (non_selected_zero : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      selected f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ selected f →
      attribution fs k f = 0)
    (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P)
    (h_sel_in : selected f ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hne : k ≠ selected f) :
    attribution fs k f = 0 ∧ 0 < attribution fs (selected f) f :=
  ⟨non_selected_zero f ℓ k h_sel_in hk hne, selected_pos f⟩

end UniversalImpossibility.Lasso
