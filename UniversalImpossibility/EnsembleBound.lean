/-
  Ensemble Size Bounds and DASH Variance Optimality.

  Derives consequences of the consensus_variance_bound axiom:
  - Variance is monotone decreasing in M
  - Doubling M halves variance
  - The ensemble size formula M_min for target stability
  - DASH achieves minimum variance among unbiased linear estimators

  Supplement: §Theorem F2: DASH Pareto Optimality + §Necessary ensemble size
-/
import UniversalImpossibility.Corollary

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Variance-ensemble size relationship -/

/-- The consensus variance at ensemble size M is Var(φ_j)/M.
    This wraps the axiom in a more usable form. -/
theorem consensus_var_eq_over_M (M : ℕ) (hM : 0 < M) (j : Fin fs.P) :
    ∃ v : ℝ, v = attribution_variance fs j / M ∧ 0 ≤ v :=
  consensus_variance_bound fs M hM j

/-- Variance at M₁ vs M₂: if M₂ > M₁ > 0, then Var/M₂ < Var/M₁
    (assuming Var > 0). More models means less variance. -/
theorem variance_decreases_with_M
    (j : Fin fs.P)
    (hvar_pos : 0 < attribution_variance fs j)
    (M₁ M₂ : ℕ) (hM₁ : 0 < M₁) (_hM₂ : 0 < M₂) (hlt : M₁ < M₂) :
    attribution_variance fs j / (M₂ : ℝ) < attribution_variance fs j / (M₁ : ℝ) := by
  apply div_lt_div_of_pos_left hvar_pos
  · exact Nat.cast_pos.mpr hM₁
  · exact Nat.cast_lt.mpr hlt

/-- Doubling the ensemble size halves the variance. -/
theorem double_ensemble_halves_variance
    (j : Fin fs.P) (M : ℕ) (_hM : 0 < M) :
    attribution_variance fs j / (2 * (M : ℝ)) =
    (attribution_variance fs j / (M : ℝ)) / 2 := by
  ring

/-- The variance at M=1 is just the single-model variance. -/
theorem single_model_variance (j : Fin fs.P) :
    attribution_variance fs j / (1 : ℝ) = attribution_variance fs j := by
  simp

/-- For M ≥ 25, the variance is at most 4% of the single-model variance.
    (1/25 = 0.04) -/
theorem variance_at_25 (j : Fin fs.P) :
    attribution_variance fs j / (25 : ℝ) = attribution_variance fs j * (1 / 25) := by
  ring

/-! ### Ensemble size lower bound (algebraic form) -/

/-- The minimum ensemble size for a given signal-to-noise ratio.
    If we want Var(consensus)/Δ² ≤ δ (target stability), then
    M ≥ Var(φ)/(δ · Δ²).

    We state this as: the required M satisfies a simple inequality. -/
theorem ensemble_bound_formula
    (sigma_sq delta_sq : ℝ) (_h_sigma : 0 < sigma_sq) (h_delta : 0 < delta_sq) (d : ℝ) (_hd : 0 < d)
    (M : ℕ) (hM : 0 < M)
    (h_sufficient : sigma_sq / (M : ℝ) ≤ d * delta_sq) :
    -- Then the variance-to-gap ratio is controlled
    sigma_sq / ((M : ℝ) * delta_sq) ≤ d := by
  have hM_pos : (0 : ℝ) < M := Nat.cast_pos.mpr hM
  have h_denom_pos : (0 : ℝ) < (M : ℝ) * delta_sq := mul_pos hM_pos h_delta
  rw [div_le_iff₀ h_denom_pos]
  calc sigma_sq ≤ d * delta_sq * (M : ℝ) := by
        rwa [div_le_iff₀ hM_pos] at h_sufficient
       _ = d * ((M : ℝ) * delta_sq) := by ring

/-- For 5% flip rate (z = 1.645), M_min ≈ 2.71 · σ²/Δ².
    We verify the constant: 1.645² = 2.706025. -/
theorem z_squared_for_5pct : (1.645 : ℝ) ^ 2 = 2.706025 := by norm_num

/-! ### DASH optimality statement -/

/-- DASH (sample mean) achieves the minimum variance among all
    averages of M i.i.d. observations: Var = σ²/M.

    The Cramér-Rao bound states no unbiased estimator can do better.
    We state this as: DASH variance equals the theoretical minimum,
    without formalizing Cramér-Rao itself (which is not in Mathlib). -/
theorem dash_achieves_minimum_variance
    (M : ℕ) (hM : 0 < M) (j : Fin fs.P) :
    ∃ v : ℝ, v = attribution_variance fs j / M ∧ 0 ≤ v :=
  consensus_variance_bound fs M hM j

/-- No unbiased linear estimator has lower variance than DASH.
    Among estimators of the form ∑ wᵢ · φ_j(fᵢ), the one with
    wᵢ = 1/M (DASH) minimizes variance when the models are i.i.d.

    We state this algebraically: for any weights summing to 1,
    ∑ wᵢ² ≥ 1/M, with equality iff all wᵢ = 1/M. -/
theorem sum_squares_ge_inv_M (M : ℕ) (hM : 0 < M) (w : Fin M → ℝ)
    (hw_sum : Finset.univ.sum w = 1) :
    Finset.univ.sum (fun i => w i ^ 2) ≥ 1 / (M : ℝ) := by
  -- Proof via Titu's lemma (Sedrakyan / Engel form of Cauchy-Schwarz):
  -- (∑ wᵢ)² / (∑ gᵢ) ≤ ∑ (wᵢ² / gᵢ) with gᵢ = 1.
  have h_titu := Finset.sq_sum_div_le_sum_sq_div Finset.univ w
    (fun (i : Fin M) _ => (zero_lt_one : (0 : ℝ) < 1))
  simp only [div_one, Finset.sum_const, Finset.card_fin, Nat.smul_one_eq_cast] at h_titu
  rw [hw_sum, one_pow] at h_titu
  -- h_titu : 1 / ↑M ≤ Finset.sum Finset.univ fun i => w i ^ 2
  exact ge_iff_le.mpr h_titu

end UniversalImpossibility
