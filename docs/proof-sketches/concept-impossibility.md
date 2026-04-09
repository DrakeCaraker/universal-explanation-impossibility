# Concept Probe Impossibility — Proof Sketch

## Setup
- Θ = model weights (defines internal representations)
- Y = predictions (the input-output function)
- H = ℝᶜ (concept activation vectors — directions in representation space)
- observe(θ) = the function f_θ : X → Y
- explain(θ) = concept direction v_θ ∈ ℝᶜ (the linear probe direction for a concept like "gender")

## Rashomon Property
Equivalent models (same predictions) can encode concepts in different directions in representation space. This is well-documented:
- Bolukbasi et al. (2016): gender direction varies across architectures
- TCAV (Kim et al. 2018): concept directions are representation-dependent
- Ravfogel et al. (2020): iterative nullspace projection shows concepts spread across multiple directions

∃ θ₁, θ₂ such that:
- f_{θ₁} = f_{θ₂} (same predictions)
- v_{θ₁} ≠ v_{θ₂} (different concept directions)

Incompatibility: the concept probe directions differ (not related by rotation within the concept subspace).

## Trilemma
- Faithful: report the actual concept direction for THIS model's representation
- Stable: same concept direction for equivalent models
- Decisive: commit to a specific concept direction

## Proof
1. Rashomon: ∃ θ₁, θ₂ equivalent with different concept directions
2. Faithful E reports v_{θ₁} for θ₁ and v_{θ₂} for θ₂
3. Decisive: E distinguishes them (different directions)
4. Stable: E(θ₁) = E(θ₂) since observe(θ₁) = observe(θ₂)
5. Contradiction.

## Verdict: GO
Same 4-step structure. Rashomon property is empirically robust.

## Axioms needed: Rashomon property only.
