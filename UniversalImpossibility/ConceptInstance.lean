import UniversalImpossibility.ExplanationSystem

/-!
# Concept Probe (TCAV) Impossibility Instance

Instantiates the universal explanation impossibility framework for concept probes
(Testing with Concept Activation Vectors). Concept activation directions in
representation space cannot simultaneously be faithful to the model's actual
internal encoding, stable across equivalent models, and decisive about which
direction represents a concept.
-/

set_option autoImplicit false

/-- Model weight configurations (defines internal representations). -/
axiom ConceptConfig : Type

/-- Concept activation vectors: directions in representation space (ℝᶜ). -/
axiom ConceptExplanation : Type

/-- Observable behavior: input→output predictions (the function the model computes). -/
axiom ConceptObservable : Type

/-- The concept probe explanation system.
    - observe(θ) = f_θ (the input-output function)
    - explain(θ) = v_θ (the concept activation direction for a given concept)
    - incompatible(v₁, v₂) = directions differ (not related by rotation in concept subspace)
    - Rashomon: ∃ θ₁ θ₂ with same predictions but different concept directions
      (Bolukbasi et al. 2016; Kim et al. 2018; Ravfogel et al. 2020) -/
axiom conceptSystem : ExplanationSystem ConceptConfig ConceptExplanation ConceptObservable

/-- Concept probe impossibility: no explanation of concept directions can be simultaneously
    faithful, stable, and decisive. Direct application of the universal impossibility
    theorem to the concept probe system. -/
theorem concept_impossibility
    (E : ConceptConfig → ConceptExplanation)
    (hf : faithful conceptSystem E)
    (hs : stable conceptSystem E)
    (hd : decisive conceptSystem E) :
    False :=
  explanation_impossibility conceptSystem E hf hs hd
