/-
  Binary Quantizer Capture Fraction.

  The binary quantizer (decision stump) maps X ~ N(0, sigma^2) to
  plus-or-minus E[|X|] = plus-or-minus sigma * sqrt(2/pi), each with probability 1/2.

  The quantized variance is therefore (sigma * sqrt(2/pi))^2 = (2/pi) sigma^2,
  giving a capture fraction of 2/pi ~ 0.6366.

  This matches the paper's fitted alpha ~ 0.60 for stumps (Section 4).

  Supplement: Signal capture fraction (alpha = 2/pi)
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Real.Pi.Bounds

set_option autoImplicit false

namespace UniversalImpossibility

open Real

/-! ### Binary Quantizer: capture fraction = 2/pi -/

/-- Equivalent formulation: the quantized variance equals (2/pi) * sigma^2.
    This holds for all real sigma (not just positive). -/
theorem binary_quantizer_variance
    (σ : ℝ) :
    (σ * Real.sqrt (2 / Real.pi)) ^ 2 = 2 / Real.pi * σ ^ 2 := by
  have h_nn : (0 : ℝ) ≤ 2 / Real.pi := div_nonneg (by norm_num) (le_of_lt Real.pi_pos)
  rw [mul_pow, Real.sq_sqrt h_nn]
  ring

/-- The binary quantizer capture fraction is 2/pi.

    If X ~ N(0, sigma^2), the binary quantizer outputs +v or -v where
    v = sigma * sqrt(2/pi) = E[|X|].  The quantized signal Q takes
    values +v or -v each with probability 1/2, so Var(Q) = v^2.

    This theorem proves: v^2 / sigma^2 = 2/pi, i.e., the quantizer
    captures exactly fraction 2/pi of the original variance. -/
theorem binary_quantizer_fraction
    (σ : ℝ) (hσ : 0 < σ) :
    (σ * Real.sqrt (2 / Real.pi)) ^ 2 / σ ^ 2 = 2 / Real.pi := by
  rw [binary_quantizer_variance σ]
  field_simp

/-- The capture fraction is strictly between 0 and 1 (a proper fraction). -/
theorem two_over_pi_in_unit_interval :
    (0 : ℝ) < 2 / Real.pi ∧ 2 / Real.pi < 1 := by
  constructor
  · exact div_pos (by norm_num) Real.pi_pos
  · rw [div_lt_one Real.pi_pos]
    linarith [Real.pi_gt_three]

/-- 2/pi is at most 2/3 (since pi > 3). -/
theorem two_over_pi_le_two_thirds :
    2 / Real.pi ≤ 2 / 3 := by
  apply div_le_div_of_nonneg_left (by norm_num : (0:ℝ) ≤ 2) (by norm_num : (0:ℝ) < 3) (le_of_lt Real.pi_gt_three)

/-- 2/pi exceeds 1/2 (since pi < 4). -/
theorem two_over_pi_gt_half :
    1 / 2 < 2 / Real.pi := by
  -- Suffices to show pi * 1 < 2 * 2, i.e., pi < 4
  rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 2) Real.pi_pos]
  linarith [Real.pi_lt_four]

/-- The capture fraction 2/pi is strictly less than 1, confirming that
    binary quantization loses signal (the impossibility gap). -/
theorem quantizer_signal_loss
    (σ : ℝ) (hσ : 0 < σ) :
    (σ * Real.sqrt (2 / Real.pi)) ^ 2 < σ ^ 2 := by
  have hσ2 : (0 : ℝ) < σ ^ 2 := sq_pos_of_pos hσ
  calc (σ * Real.sqrt (2 / Real.pi)) ^ 2
      = 2 / Real.pi * σ ^ 2 := binary_quantizer_variance σ
    _ < 1 * σ ^ 2 := by
        apply mul_lt_mul_of_pos_right _ hσ2
        exact two_over_pi_in_unit_interval.2
    _ = σ ^ 2 := one_mul _

/-- The capture fraction 2/pi exceeds 1/2, so the binary quantizer
    retains more than half the variance. -/
theorem quantizer_better_than_half
    (σ : ℝ) (hσ : 0 < σ) :
    1 / 2 * σ ^ 2 < (σ * Real.sqrt (2 / Real.pi)) ^ 2 := by
  have hσ2 : (0 : ℝ) < σ ^ 2 := sq_pos_of_pos hσ
  calc 1 / 2 * σ ^ 2
      < 2 / Real.pi * σ ^ 2 := by
        apply mul_lt_mul_of_pos_right _ hσ2
        exact two_over_pi_gt_half
    _ = (σ * Real.sqrt (2 / Real.pi)) ^ 2 := (binary_quantizer_variance σ).symm

end UniversalImpossibility
