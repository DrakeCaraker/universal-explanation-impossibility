/-
  Neural networks: the Attribution Impossibility holds when initialization-dependent
  feature utilization produces a dominant feature per correlated group.

  Properties are theorem hypotheses, not global axioms. The classical
  justification (initialization symmetry breaking, gradient flow path
  dependence) is Paper 4 in the research program.
-/
import UniversalImpossibility.Iterative

set_option autoImplicit false

namespace UniversalImpossibility.NeuralNet

variable (fs : FeatureSpace)

/-- Neural networks inherit the Attribution Impossibility when initialization
    symmetry-breaking produces a dominant feature per correlated group. -/
theorem nn_impossibility
    (captured : Model → Fin fs.P)
    (captured_gt : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      captured f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ captured f →
      attribution fs k f < attribution fs (captured f) f)
    (captured_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
      j ∈ fs.group ℓ → ∃ f : Model, captured f = j)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False := by
  have opt : IterativeOptimizer fs := {
    dominant := captured
    dominant_gt := captured_gt
    dominant_surjective := captured_surjective
  }
  exact iterative_impossibility fs opt ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility.NeuralNet
