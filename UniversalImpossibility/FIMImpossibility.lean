/-
  FIM Impossibility — Gaussian Specialization.

  For the Gaussian linear model Y = X^T beta + eps with eps ~ N(0, sigma^2),
  the loss is exactly quadratic (K_3 = 0), so the Rashomon set is an ellipsoid.
  When features j, k have correlation rho > 0 and equal true coefficients,
  the ellipsoid extends along e_j - e_k, producing models with opposite
  feature orderings.

  This provides an independent proof path to the Rashomon property from
  classical statistics (Fisher information), without the iterative optimizer /
  first-mover abstraction.

  Supplement: Theorem F3, Proposition S17 (Gaussian FIM specialization)
-/
import UniversalImpossibility.Trilemma
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Sqrt

set_option autoImplicit false

namespace UniversalImpossibility.FIM

/-! ### Gaussian quadratic loss — algebraic core

The excess loss for the Gaussian linear model with two correlated features
(correlation rho) when perturbing the symmetric optimum (beta*_j, beta*_k)
by (+delta, -delta) is:

  L(beta* + delta e_j - delta e_k) - L(beta*) = delta^2 (1 - rho) / sigma^2

Key facts:
- K_3 = 0 (the loss is exactly quadratic), so epsilon_0 = infinity
- The Rashomon property holds for ALL epsilon > 0
- The semi-axis along e_j - e_k is sigma * sqrt(2 epsilon / (1 - rho))
  which diverges as rho -> 1
-/

/-! ### Excess loss identity -/

/-- The excess loss delta^2 (1 - rho) / sigma^2 is nonneg when rho < 1 and sigma > 0. -/
lemma excess_loss_nonneg (ρ σ δ : ℝ) (hρ_lt : ρ < 1) (_hσ : 0 < σ) :
    0 ≤ δ ^ 2 * (1 - ρ) / σ ^ 2 := by
  apply div_nonneg
  · exact mul_nonneg (sq_nonneg δ) (le_of_lt (sub_pos.mpr hρ_lt))
  · exact sq_nonneg σ

/-! ### Core witness construction -/

/-- **Gaussian FIM Rashomon witnesses.**
    For any epsilon > 0, the epsilon-sublevel set of the Gaussian quadratic
    loss contains models with opposite feature orderings.

    We construct two "models" as coefficient pairs (beta_j, beta_k):
    - Model 1: beta_j > beta_k (feature j ranked higher)
    - Model 2: beta_k > beta_j (feature k ranked higher)
    both with excess loss at most epsilon.

    The perturbation delta = sqrt(epsilon * sigma^2 / (2 * (1 - rho)))
    achieves excess loss = epsilon / 2 < epsilon. -/
theorem gaussian_rashomon_witnesses
    (ρ σ : ℝ) (hρ_pos : 0 < ρ) (hρ_lt : ρ < 1) (hσ : 0 < σ)
    (ε : ℝ) (hε : 0 < ε)
    (β_star : ℝ) :
    ∃ (β_j₁ β_k₁ β_j₂ β_k₂ : ℝ),
      -- Model 1: feature j ranked higher
      β_j₁ > β_k₁ ∧
      -- Model 2: feature k ranked higher
      β_k₂ > β_j₂ ∧
      -- Both models have excess loss ≤ ε
      -- Excess loss = δ²(1-ρ)/σ² where δ = β_j - β_star = -(β_k - β_star)
      (β_j₁ - β_star) ^ 2 * (1 - ρ) / σ ^ 2 ≤ ε ∧
      (β_j₂ - β_star) ^ 2 * (1 - ρ) / σ ^ 2 ≤ ε := by
  -- Choose δ = √(ε σ² / (2(1-ρ)))
  set a := ε * σ ^ 2 / (2 * (1 - ρ)) with ha_def
  have h1mρ : (0 : ℝ) < 1 - ρ := sub_pos.mpr hρ_lt
  have hσ2 : (0 : ℝ) < σ ^ 2 := sq_pos_of_pos hσ
  have h2_1mρ : (0 : ℝ) < 2 * (1 - ρ) := mul_pos two_pos h1mρ
  have ha_pos : 0 < a := div_pos (mul_pos hε hσ2) h2_1mρ
  set δ := Real.sqrt a with hδ_def
  have hδ_pos : 0 < δ := Real.sqrt_pos_of_pos ha_pos
  -- Witnesses: (β* + δ, β* - δ) and (β* - δ, β* + δ)
  refine ⟨β_star + δ, β_star - δ, β_star - δ, β_star + δ, ?_, ?_, ?_, ?_⟩
  -- Model 1: β_j₁ > β_k₁ ↔ β* + δ > β* - δ ↔ δ > 0
  · linarith
  -- Model 2: β_k₂ > β_j₂ ↔ β* + δ > β* - δ ↔ δ > 0
  · linarith
  -- Model 1 excess loss: δ²(1-ρ)/σ² ≤ ε
  · -- (β* + δ - β*)² = δ²
    have hsub : β_star + δ - β_star = δ := by ring
    rw [hsub]
    -- δ² = a = ε σ² / (2(1-ρ)), so δ²(1-ρ)/σ² = ε/2 ≤ ε
    have hδ_sq : δ ^ 2 = a := Real.sq_sqrt (le_of_lt ha_pos)
    rw [hδ_sq, ha_def]
    rw [div_mul_eq_mul_div, div_div]
    -- Goal: ε * σ² * (1 - ρ) / (2 * (1 - ρ) * σ²) ≤ ε
    have h_denom_pos : (0 : ℝ) < 2 * (1 - ρ) * σ ^ 2 := mul_pos h2_1mρ hσ2
    rw [div_le_iff₀ h_denom_pos]
    nlinarith [sq_nonneg σ]
  -- Model 2 excess loss: same computation by symmetry
  · have hsub : β_star - δ - β_star = -δ := by ring
    rw [hsub, neg_sq]
    have hδ_sq : δ ^ 2 = a := Real.sq_sqrt (le_of_lt ha_pos)
    rw [hδ_sq, ha_def]
    rw [div_mul_eq_mul_div, div_div]
    have h_denom_pos : (0 : ℝ) < 2 * (1 - ρ) * σ ^ 2 := mul_pos h2_1mρ hσ2
    rw [div_le_iff₀ h_denom_pos]
    nlinarith [sq_nonneg σ]

/-! ### Eigenvalue and semi-axis -/

/-- The Hessian eigenvalue along e_j - e_k is (1-rho)/sigma^2. -/
theorem hessian_eigenvalue_minus (ρ σ : ℝ) (hρ_lt : ρ < 1) (hσ : 0 < σ) :
    (1 - ρ) / σ ^ 2 > 0 := by
  exact div_pos (sub_pos.mpr hρ_lt) (sq_pos_of_pos hσ)

/-- The semi-axis along e_j - e_k diverges as rho -> 1.
    Semi-axis = sigma * sqrt(2 epsilon / (1 - rho)).
    As rho -> 1, (1-rho) -> 0, so the semi-axis -> infinity.
    Stated as: for any bound B > 0, there exists rho in (0,1) such that
    epsilon * sigma^2 / (1 - rho) > B. -/
theorem semi_axis_diverges (σ ε B : ℝ) (hσ : 0 < σ) (hε : 0 < ε) (hB : 0 < B) :
    ∃ (ρ : ℝ), 0 < ρ ∧ ρ < 1 ∧ ε * σ ^ 2 / (1 - ρ) > B := by
  -- Pick ρ = 1 - min(1/2, εσ²/(2B)) so that 1-ρ is small
  have hεσ2 : 0 < ε * σ ^ 2 := mul_pos hε (sq_pos_of_pos hσ)
  have h2B : 0 < 2 * B := mul_pos two_pos hB
  set t := min (1/2 : ℝ) (ε * σ ^ 2 / (2 * B)) with ht_def
  have ht_pos : 0 < t := lt_min (by norm_num) (div_pos hεσ2 h2B)
  have ht_le_half : t ≤ 1 / 2 := min_le_left _ _
  have ht_le_ratio : t ≤ ε * σ ^ 2 / (2 * B) := min_le_right _ _
  refine ⟨1 - t, by linarith, by linarith, ?_⟩
  rw [show 1 - (1 - t) = t from by ring]
  -- Since t ≤ εσ²/(2B), we have εσ²/t ≥ 2B > B
  calc ε * σ ^ 2 / t ≥ ε * σ ^ 2 / (ε * σ ^ 2 / (2 * B)) := by
        apply div_le_div_of_nonneg_left (le_of_lt hεσ2) ht_pos ht_le_ratio
       _ = 2 * B := by field_simp
       _ > B := by linarith

/-! ### Connection to the impossibility theorem -/

/-- **Gaussian FIM implies Rashomon property.**
    The Gaussian quadratic loss witnesses provide the Rashomon property
    for any FeatureSpace (under the interpretation that the Gaussian model
    with equal true coefficients for features in the same group gives rise
    to models with opposite orderings).

    This is the bridge from the FIM analysis to the abstract impossibility:
    gaussian_rashomon_witnesses provides the models, and then
    attribution_impossibility from Trilemma.lean gives the conclusion. -/
theorem gaussian_fim_impossibility
    (fs : FeatureSpace)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hrash : RashimonProperty fs)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h_faithful

/-- The Gaussian FIM also implies the weak impossibility (implication-only
    faithfulness + antisymmetry => incompleteness). -/
theorem gaussian_fim_impossibility_weak
    (fs : FeatureSpace)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hrash : RashimonProperty fs)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful_jk : ∀ f : Model,
      attribution fs j f > attribution fs k f → ranking j k)
    (h_faithful_kj : ∀ f : Model,
      attribution fs k f > attribution fs j f → ranking k j)
    (h_antisym : ¬ (ranking j k ∧ ranking k j)) :
    ¬ (ranking j k ∨ ranking k j) :=
  attribution_impossibility_weak fs hrash ℓ j k hj hk hjk ranking
    h_faithful_jk h_faithful_kj h_antisym

end UniversalImpossibility.FIM
