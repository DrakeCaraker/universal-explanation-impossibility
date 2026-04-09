import UniversalImpossibility.ExplanationSystem

/-!
# Mechanistic Interpretability Instance

Neural network circuit explanations as an instance of ExplanationSystem.
The Rashomon property holds because equivalent networks (same I/O behavior)
admit multiple valid circuit decompositions.

Empirical evidence: Meloux et al. (ICLR 2025) found 85 distinct valid
circuits with zero circuit error for a simple XOR task, each admitting
an average of 535.8 valid interpretations.
-/

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Mechanistic interpretability types -/

/-- Configuration space: neural network weight configurations.
    Each θ ∈ CircuitConfig is a trained network (architecture + weights). -/
axiom CircuitConfig : Type

/-- Explanation space: circuit decompositions.
    Each h ∈ CircuitExplanation is a subgraph of the computational graph
    together with an interpretation of which components implement which
    sub-computations (e.g., "heads 1.3 and 2.1 form an induction circuit"). -/
axiom CircuitExplanation : Type

/-- Observable space: input–output functions.
    Each y ∈ CircuitObservable is the function computed by the network
    on the domain of interest. -/
axiom CircuitObservable : Type

/-! ### Circuit explanation system -/

/-- The mechanistic interpretability explanation system.
    - `observe`: maps a network to its input–output function
    - `explain`: maps a network to its circuit decomposition
    - `incompatible`: two circuit explanations disagree on which components
      are responsible for a computation (they attribute the same behavior
      to disjoint or contradictory subgraphs)
    - `rashomon`: two networks with identical I/O behavior whose circuit
      decompositions are incompatible — witnessed empirically by the
      85 distinct valid circuits found by Meloux et al. (ICLR 2025) -/
axiom circuitSystem : ExplanationSystem CircuitConfig CircuitExplanation CircuitObservable

/-! ### Mechanistic interpretability impossibility -/

/-- **Mechanistic Interpretability Explanation Impossibility.**
    No circuit explanation method E can simultaneously be:
    - **Faithful**: E reports the actual circuit decomposition for each network
    - **Stable**: E gives the same circuit for observationally equivalent
      networks (same input–output function)
    - **Decisive**: E commits to a single circuit decomposition rather than
      returning the space of valid circuits

    Proof: direct application of the universal explanation impossibility.
    The Rashomon witness is any pair of networks with the same I/O map
    but incompatible circuit decompositions, as documented by
    Meloux et al. (2025) for XOR networks. -/
theorem mech_interp_impossibility
    (E : CircuitConfig → CircuitExplanation)
    (hf : faithful circuitSystem E)
    (hs : stable circuitSystem E)
    (hd : decisive circuitSystem E) : False :=
  explanation_impossibility circuitSystem E hf hs hd

end UniversalImpossibility
