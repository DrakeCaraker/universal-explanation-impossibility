import UniversalImpossibility.ExplanationSystem

/-!
# Causal Discovery Explanation Impossibility

No causal explanation method can simultaneously be faithful (report the actual
DAG generating the observed conditional independence structure), stable (give
the same DAG for observationally equivalent data-generating processes), and
decisive (distinguish incompatible causal structures).

This is a direct instantiation of the universal explanation impossibility
framework (ExplanationSystem.lean). The Rashomon property for causal
structures follows from Markov equivalence: multiple DAGs encode the same
conditional independence relations, so observationally equivalent
configurations can produce incompatible causal explanations.
-/

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Causal discovery types -/

/-- Configuration space: DAGs (directed acyclic graphs) representing
    possible causal structures over a fixed set of variables. -/
axiom CausalConfig : Type

/-- Explanation space: DAGs returned as causal explanations
    (the oriented causal structure the method commits to). -/
axiom CausalExplanation : Type

/-- Observable space: conditional independence relations observable from
    the joint distribution (equivalently, the CPDAG / Markov equivalence
    class skeleton). -/
axiom CausalObservable : Type

/-! ### Causal explanation system -/

/-- The causal explanation system.
    - `observe`: maps a DAG to its induced conditional independence structure
    - `explain`: maps a DAG to the causal explanation a method would return
    - `incompatible`: two causal explanations disagree on at least one edge
      orientation (they assert opposite directions for some X → Y / Y → X)
    - `rashomon`: two DAGs in the same Markov equivalence class (identical
      observables) whose causal explanations are incompatible — the
      classically known fact that a CPDAG with ≥2 member DAGs witnesses this -/
axiom causalSystem : ExplanationSystem CausalConfig CausalExplanation CausalObservable

/-! ### Causal impossibility -/

/-- **Causal Discovery Explanation Impossibility.**
    No causal explanation method E can simultaneously be:
    - **Faithful**: E reports the actual generating DAG for each configuration
    - **Stable**: E gives the same explanation for observationally equivalent
      configurations (same conditional independence structure)
    - **Decisive**: E distinguishes incompatible causal structures (commits to
      an orientation rather than returning the CPDAG)

    Proof: direct application of the universal explanation impossibility.
    The Rashomon witness is any CPDAG containing ≥2 member DAGs with
    incompatible edge orientations. -/
theorem causal_instance_impossibility
    (E : CausalConfig → CausalExplanation)
    (hf : faithful causalSystem E)
    (hs : stable causalSystem E)
    (hd : decisive causalSystem E) :
    False :=
  explanation_impossibility causalSystem E hf hs hd

end UniversalImpossibility
