# Attention Map Impossibility — Proof Sketch

## Setup
- Θ = neural network weight configurations (parameters)
- Y = input→output mappings (the function the network computes)
- H = Δ^T (probability simplex over T tokens — attention distributions)
- observe(θ) = the function f_θ : X → Y
- explain(θ) = attention distribution α_θ ∈ Δ^T for a given input x

## Rashomon Property
By D'Amour et al. (2020): ∃ θ₁, θ₂ such that:
- f_{θ₁} ≈ f_{θ₂} (same test loss, equivalent predictions)
- α_{θ₁} ≠ α_{θ₂} (different attention patterns)

Incompatibility: α_{θ₁} and α_{θ₂} assign maximum attention to different tokens (argmax differs).

## Trilemma
- Faithful: report the model's actual attention α_θ
- Stable: same attention for equivalent models
- Decisive: commit to which token is most attended

## Proof
1. Rashomon: ∃ θ₁, θ₂ equivalent with different max-attended tokens
2. Faithful E reports α_{θ₁} for θ₁ and α_{θ₂} for θ₂
3. Decisive: E distinguishes them (different argmax)
4. Stable: E(θ₁) = E(θ₂) since observe(θ₁) = observe(θ₂)
5. Contradiction.

## Verdict: GO
Same 4-step structure. No additional axioms needed beyond Rashomon.

## Axioms needed: Rashomon property only.
