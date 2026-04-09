# Counterfactual Explanation Impossibility — Proof Sketch

## Setup
- Θ = model parameters (defines a decision boundary)
- Y = predictions on test set
- H = X (input space — the nearest contrastive example)
- observe(θ) = predictions f_θ on test set
- explain(θ) = nearest x' such that f_θ(x') ≠ f_θ(x) for query x

## Rashomon Property
Two models with identical predictions on the test set but different decision boundaries near query point x:
- Model 1: boundary passes above x → nearest counterfactual is x'₁
- Model 2: boundary passes below x → nearest counterfactual is x'₂
- x'₁ ≠ x'₂

Incompatibility: the counterfactuals point in different directions (different recommended actions for the individual).

## Trilemma
- Faithful: report the actual nearest counterfactual for THIS model
- Stable: same counterfactual for equivalent models
- Decisive: commit to a specific counterfactual direction

## Proof
1. Rashomon: ∃ θ₁, θ₂ equivalent with different nearest counterfactuals
2. Faithful E reports x'₁ for θ₁ and x'₂ for θ₂
3. Decisive: E distinguishes (different directions)
4. Stable: E(θ₁) = E(θ₂) since observe(θ₁) = observe(θ₂)
5. Contradiction.

## Verdict: GO
Same 4-step structure. Rashomon property for decision boundaries is well-documented (Semenova et al. 2022, Fisher et al. 2019).

## Axioms needed: Rashomon property only.
