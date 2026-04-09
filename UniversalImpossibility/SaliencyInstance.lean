import UniversalImpossibility.ExplanationSystem

/-!
# Saliency Map Impossibility Instance

Instantiates the universal explanation impossibility framework for saliency maps.
GradCAM / saliency maps as spatial heatmaps over image pixels cannot simultaneously
be faithful to the model's gradient-based explanation, stable across observationally
equivalent models, and decisive about which pixel region drives the prediction.
-/

set_option autoImplicit false

/-- Neural network weight configurations (parameters) for vision models. -/
axiom SaliencyConfig : Type

/-- Saliency maps: spatial heatmaps assigning relevance scores to image pixels. -/
axiom SaliencyMap : Type

/-- Observable behavior: input→output mappings (the function the network computes). -/
axiom SaliencyObservable : Type

/-- The saliency explanation system.
    - observe(θ) = f_θ (the function the network computes on inputs)
    - explain(θ) = s_θ (the saliency heatmap, e.g. GradCAM, for a given input)
    - incompatible(s₁, s₂) = the argmax pixel region differs (different most-salient patch)
    - Rashomon: ∃ θ₁ θ₂ with same predictions but different maximally-salient regions
      (Kindermans et al., 2019; Adebayo et al., 2018) -/
axiom saliencySystem : ExplanationSystem SaliencyConfig SaliencyMap SaliencyObservable

/-- Saliency map impossibility: no explanation of a saliency map can be simultaneously
    faithful, stable, and decisive. Direct application of the universal impossibility
    theorem to the saliency system. -/
theorem saliency_impossibility
    (E : SaliencyConfig → SaliencyMap)
    (hf : faithful saliencySystem E)
    (hs : stable saliencySystem E)
    (hd : decisive saliencySystem E) :
    False :=
  explanation_impossibility saliencySystem E hf hs hd
