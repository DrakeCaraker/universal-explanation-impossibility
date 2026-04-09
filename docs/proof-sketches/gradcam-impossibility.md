# GradCAM / Saliency Map Impossibility — Proof Sketch

## Setup
- Θ = CNN weight configurations (e.g., ResNet parameters)
- Y = class predictions (the classification function)
- H = ℝ^(W×H) (spatial heatmaps — per-pixel importance scores)
- observe(θ) = the classification function f_θ : images → labels
- explain(θ) = GradCAM heatmap for a given input image and target class

## Rashomon Property
Equivalent CNNs (same predictions on test set) produce different GradCAM heatmaps because:
- GradCAM depends on the GRADIENT of the class score w.r.t. the last convolutional layer activations
- Different weight configurations achieve the same classification via different internal feature maps
- The spatial distribution of "where the model looks" varies across equivalent models

Cite: Adebayo et al. (2018) showed saliency maps can be insensitive to model parameters (some methods fail sanity checks). Hooker et al. (2019) showed interpretation methods disagree on feature importance ordering. Selvaraju et al. (2017) introduced GradCAM.

## Incompatibility
Two heatmaps are incompatible if they highlight different spatial regions: the top-20% intensity regions have low IoU (intersection over union).

## Trilemma
- Faithful: report the actual GradCAM heatmap for THIS model
- Stable: same heatmap for equivalent models
- Decisive: commit to a specific spatial region (no averaging/blurring)

## Proof
1. Rashomon: ∃ θ₁, θ₂ equivalent with incompatible heatmaps
2. Decisive: E inherits the incompatibility
3. Stable: E(θ₁) = E(θ₂)
4. Faithful: E(θ₂) is not incompatible with explain(θ₂) — contradiction with step 2+3
5. QED

## Verdict: GO
Same 4-step structure. The Rashomon property for CNN internal representations is well-documented.

## Axioms needed: Rashomon property only.
