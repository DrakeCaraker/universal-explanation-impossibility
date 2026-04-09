import UniversalImpossibility.ExplanationSystem

/-!
# Counterfactual Explanation Impossibility

No counterfactual explanation method can simultaneously be faithful (report the
actual nearest counterfactual for the model), stable (give the same
counterfactual for observationally equivalent models), and decisive (distinguish
incompatible counterfactual directions).

This is a direct instantiation of the universal explanation impossibility
framework (ExplanationSystem.lean). The Rashomon property for decision
boundaries is well-documented (Semenova et al. 2022, Fisher et al. 2019).
-/

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Counterfactual explanation types -/

/-- Configuration space: model parameters defining a decision boundary. -/
axiom CFConfig : Type

/-- Explanation space: counterfactual examples (points in input space). -/
axiom CFExplanation : Type

/-- Observable space: predictions on the test set. -/
axiom CFObservable : Type

/-! ### Counterfactual explanation system -/

/-- The counterfactual explanation system.
    - `observe`: maps model parameters to predictions on the test set
    - `explain`: maps model parameters to the nearest counterfactual example
    - `incompatible`: two counterfactuals point in different directions
    - `rashomon`: two models with identical test predictions but different
      nearest counterfactuals (different decision boundaries near query point) -/
axiom cfSystem : ExplanationSystem CFConfig CFExplanation CFObservable

/-! ### Counterfactual impossibility -/

/-- **Counterfactual Explanation Impossibility.**
    No counterfactual explanation method E can simultaneously be:
    - **Faithful**: E reports the actual nearest counterfactual for each model
    - **Stable**: E gives the same counterfactual for equivalent models
    - **Decisive**: E distinguishes incompatible counterfactual directions

    Proof: direct application of the universal explanation impossibility. -/
theorem counterfactual_impossibility
    (E : CFConfig → CFExplanation)
    (hf : faithful cfSystem E)
    (hs : stable cfSystem E)
    (hd : decisive cfSystem E) :
    False :=
  explanation_impossibility cfSystem E hf hs hd

end UniversalImpossibility
