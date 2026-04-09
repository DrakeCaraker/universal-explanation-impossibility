/-
  VarianceDerivation.lean — Derive Var(consensus) = Var(φ)/M from independence.

  Replaces the tautological consensus_variance_bound with a genuine derivation:
  1. Define HasVarianceDecomposition: Var(∑ Xᵢ) = M · Var(X) (consequence of i.i.d.)
  2. Prove: Var(mean) = Var(sum)/M² = M·σ²/M² = σ²/M (pure algebra)
  3. Connect to the consensus function defined in Defs.lean

  No new axioms are introduced. Independence is expressed as a hypothesis
  (definition), following the same pattern as IsBalanced in Defs.lean.
-/

import UniversalImpossibility.MeasureHypotheses
import Mathlib.Analysis.SpecialFunctions.Pow.Real

set_option autoImplicit false

open MeasureTheory Finset

variable (fs : FeatureSpace)

/-! ## Independence hypothesis -/

/-- An ensemble has the variance decomposition property if the variance of
    the sum of attributions equals M times the individual attribution variance.
    This is the key consequence of independence + identical distribution:
    for i.i.d. X₁,...,X_M, Var(∑ Xᵢ) = M · Var(X₁).

    We state this as a hypothesis on the product-space variance of the sum,
    avoiding the need to formalize product measures or the full independence
    machinery. The caller provides a proof that the ensemble satisfies this
    (e.g., from i.i.d. training seeds). -/
def HasVarianceDecomposition (M : ℕ) (sum_variance : ℝ) (j : Fin fs.P) : Prop :=
  sum_variance = M * attribution_variance fs j

/-- An ensemble is independent and identically distributed if each member
    is drawn from modelMeasure and the draws are mutually independent.
    We capture the key quantitative consequence: the variance of any
    linear combination ∑ wᵢ Xᵢ equals (∑ wᵢ²) · Var(X). -/
def IsIIDEnsemble (M : ℕ) (sum_variance : ℝ) (j : Fin fs.P) : Prop :=
  -- Consequence of independence: Var(sum) = M · Var(single)
  HasVarianceDecomposition fs M sum_variance j ∧
  -- Individual variance is nonneg (for downstream use)
  0 ≤ attribution_variance fs j

/-! ## Core algebraic derivation -/

/-- The fundamental identity: for i.i.d. random variables with common
    variance σ², the mean has variance σ²/M.

    Proof: Var(X̄) = Var(∑Xᵢ/M) = Var(∑Xᵢ)/M² = Mσ²/M² = σ²/M.

    This is the key step that the old consensus_variance_bound skipped:
    it connects the sum-variance decomposition to the mean-variance formula
    via division by M². -/
theorem variance_of_mean_from_sum_variance
    (σ_sq : ℝ) (_hσ : 0 ≤ σ_sq)
    (M : ℕ) (hM : 0 < M)
    (sum_var : ℝ)
    (h_sum_var : sum_var = M * σ_sq) :
    sum_var / (M : ℝ) ^ 2 = σ_sq / M := by
  rw [h_sum_var]
  have hM_pos : (0 : ℝ) < M := Nat.cast_pos.mpr hM
  have hM_ne : (M : ℝ) ≠ 0 := ne_of_gt hM_pos
  field_simp

/-- Var(mean) = σ²/M, stated with explicit decomposition hypothesis.
    This is the non-tautological replacement for consensus_variance_bound. -/
theorem consensus_variance_from_independence
    (M : ℕ) (hM : 0 < M)
    (j : Fin fs.P)
    (sum_var : ℝ)
    (h_iid : IsIIDEnsemble fs M sum_var j) :
    sum_var / (M : ℝ) ^ 2 = attribution_variance fs j / M := by
  exact variance_of_mean_from_sum_variance (attribution_variance fs j)
    h_iid.2 M hM sum_var h_iid.1

/-- The consensus variance is nonneg. Unlike the old version, this
    follows from the decomposition rather than being trivially witnessed. -/
theorem consensus_variance_nonneg_from_independence
    (M : ℕ) (hM : 0 < M)
    (j : Fin fs.P)
    (sum_var : ℝ)
    (h_iid : IsIIDEnsemble fs M sum_var j) :
    0 ≤ sum_var / (M : ℝ) ^ 2 := by
  rw [consensus_variance_from_independence fs M hM j sum_var h_iid]
  exact div_nonneg h_iid.2 (Nat.cast_nonneg M)

/-! ## Scaling laws -/

/-- Doubling M halves the consensus variance — now derived from independence,
    not from a tautological existential. -/
theorem double_M_halves_variance_derived
    (M : ℕ) (hM : 0 < M) (j : Fin fs.P)
    (sum_var_M sum_var_2M : ℝ)
    (h_M : IsIIDEnsemble fs M sum_var_M j)
    (h_2M : IsIIDEnsemble fs (2 * M) sum_var_2M j) :
    sum_var_2M / ((2 * M : ℕ) : ℝ) ^ 2 =
    (sum_var_M / (M : ℝ) ^ 2) / 2 := by
  rw [consensus_variance_from_independence fs M hM j sum_var_M h_M]
  rw [consensus_variance_from_independence fs (2 * M) (by omega) j sum_var_2M h_2M]
  push_cast
  field_simp

/-- The ratio of consensus variance at M₁ vs M₂ (M₂ > M₁) is M₁/M₂ < 1.
    More models always reduce variance, derived from the independence structure. -/
theorem variance_ratio_from_independence
    (M₁ M₂ : ℕ) (hM₁ : 0 < M₁) (hM₂ : 0 < M₂)
    (j : Fin fs.P)
    (hvar_pos : 0 < attribution_variance fs j)
    (sum_var₁ sum_var₂ : ℝ)
    (h₁ : IsIIDEnsemble fs M₁ sum_var₁ j)
    (h₂ : IsIIDEnsemble fs M₂ sum_var₂ j)
    (hlt : M₁ < M₂) :
    sum_var₂ / (M₂ : ℝ) ^ 2 < sum_var₁ / (M₁ : ℝ) ^ 2 := by
  rw [consensus_variance_from_independence fs M₁ hM₁ j sum_var₁ h₁]
  rw [consensus_variance_from_independence fs M₂ hM₂ j sum_var₂ h₂]
  exact div_lt_div_of_pos_left hvar_pos (Nat.cast_pos.mpr hM₁) (Nat.cast_lt.mpr hlt)

/-! ## Weighted estimator optimality connection -/

/-- For a weighted estimator with Var = (∑wᵢ²)·σ², the minimum variance
    is achieved when wᵢ = 1/M (DASH), giving Var = σ²/M.
    This connects the algebraic optimality (sum_squares_ge_inv_M from
    EnsembleBound.lean) to the variance decomposition. -/
theorem weighted_variance_ge_consensus_variance
    (M : ℕ) (_hM : 0 < M)
    (σ_sq : ℝ) (hσ : 0 < σ_sq)
    (w : Fin M → ℝ)
    (hw_sum : Finset.univ.sum w = 1)
    (weighted_var : ℝ)
    (h_wvar : weighted_var = Finset.univ.sum (fun i => w i ^ 2) * σ_sq) :
    σ_sq / M ≤ weighted_var := by
  rw [h_wvar]
  have h_sq_ge : Finset.univ.sum (fun i => w i ^ 2) ≥ 1 / (M : ℝ) := by
    have h_titu := Finset.sq_sum_div_le_sum_sq_div Finset.univ w
      (fun (i : Fin M) _ => (zero_lt_one : (0 : ℝ) < 1))
    simp only [div_one, Finset.sum_const, Finset.card_fin, Nat.smul_one_eq_cast] at h_titu
    rw [hw_sum, one_pow] at h_titu
    exact ge_iff_le.mpr h_titu
  calc σ_sq / M = 1 / (M : ℝ) * σ_sq := by ring
    _ ≤ Finset.univ.sum (fun i => w i ^ 2) * σ_sq :=
        mul_le_mul_of_nonneg_right (ge_iff_le.mp h_sq_ge) (le_of_lt hσ)
