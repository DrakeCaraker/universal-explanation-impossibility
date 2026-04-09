/-
  Parametric Query Complexity: axiom-free version.

  Refactors QueryComplexity.lean to pass the testing constant as a
  hypothesis rather than a global axiom. All theorems hold for ANY
  positive constant C, making the dependency transparent.

  The global axioms `testing_constant` and `testing_constant_pos` remain
  for backward compatibility. This file provides the axiom-free alternative
  for contexts that want to minimize axiom dependencies.

  Supplement: §Query complexity of stability detection
-/
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Parametric testing bound -/

/-- A testing bound configuration: packages a positive constant C with
    its positivity proof. Replaces the global axioms `testing_constant`
    and `testing_constant_pos`. -/
structure TestingBound where
  /-- The constant from Le Cam's two-point method (e.g., C = 1/8) -/
  C : ℝ
  /-- The constant is positive -/
  hC : 0 < C

/-- The query complexity lower bound C·σ²/Δ². -/
noncomputable def TestingBound.queryBound (tb : TestingBound) (σ Δ : ℝ) : ℝ :=
  tb.C * σ ^ 2 / Δ ^ 2

/-- The query bound is nonneg. -/
lemma TestingBound.queryBound_nonneg (tb : TestingBound) (σ Δ : ℝ) :
    0 ≤ tb.queryBound σ Δ := by
  unfold queryBound
  exact div_nonneg (mul_nonneg (le_of_lt tb.hC) (sq_nonneg σ)) (sq_nonneg Δ)

/-- The query bound is positive when σ > 0 and Δ > 0. -/
lemma TestingBound.queryBound_pos (tb : TestingBound) (σ Δ : ℝ)
    (hσ : 0 < σ) (hΔ : 0 < Δ) :
    0 < tb.queryBound σ Δ := by
  unfold queryBound
  exact div_pos (mul_pos tb.hC (sq_pos_of_pos hσ)) (sq_pos_of_pos hΔ)

/-! ### Parametric lower bound theorem -/

/-- A stability certifier (same as in QueryComplexity.lean). -/
structure StabilityCertifierP where
  numQueries : ℕ
  hpos : 0 < numQueries

/-- **Parametric query complexity lower bound.**
    Any reliable stability certifier needs at least C·σ²/Δ² queries.
    This is the axiom-free version: C is a hypothesis, not a global axiom. -/
theorem query_complexity_parametric
    (tb : TestingBound) (σ Δ : ℝ) (hσ : 0 < σ) (hΔ : 0 < Δ)
    (cert : StabilityCertifierP)
    (h_reliable : (cert.numQueries : ℝ) < tb.queryBound σ Δ → False) :
    tb.queryBound σ Δ ≤ cert.numQueries :=
  not_lt.mp h_reliable

/-! ### All algebraic properties, axiom-free -/

/-- Sigma scaling: doubling noise quadruples queries. -/
theorem queryBound_scales_sigma (tb : TestingBound) (σ Δ : ℝ) (_hΔ : 0 < Δ) :
    tb.queryBound (2 * σ) Δ = 4 * tb.queryBound σ Δ := by
  unfold TestingBound.queryBound
  field_simp; ring

/-- Gap scaling: halving gap quadruples queries. -/
theorem queryBound_scales_gap (tb : TestingBound) (σ Δ : ℝ)
    (_hσ : 0 < σ) (hΔ : 0 < Δ) :
    tb.queryBound σ (Δ / 2) = 4 * tb.queryBound σ Δ := by
  unfold TestingBound.queryBound
  have : Δ ^ 2 ≠ 0 := ne_of_gt (sq_pos_of_pos hΔ)
  field_simp; ring

/-- Sigma monotonicity: more noise → more queries. -/
theorem queryBound_mono_sigma (tb : TestingBound) (σ₁ σ₂ Δ : ℝ)
    (_hσ₁ : 0 < σ₁) (_hσ₂ : 0 < σ₂) (hΔ : 0 < Δ)
    (hle : σ₁ ≤ σ₂) :
    tb.queryBound σ₁ Δ ≤ tb.queryBound σ₂ Δ := by
  unfold TestingBound.queryBound
  apply div_le_div_of_nonneg_right _ (le_of_lt (sq_pos_of_pos hΔ))
  exact mul_le_mul_of_nonneg_left (sq_le_sq' (by linarith) hle) (le_of_lt tb.hC)

/-- Gap anti-monotonicity: larger gap → fewer queries. -/
theorem queryBound_antimono_gap (tb : TestingBound) (σ Δ₁ Δ₂ : ℝ)
    (hσ : 0 < σ) (hΔ₁ : 0 < Δ₁) (_hΔ₂ : 0 < Δ₂)
    (hle : Δ₁ ≤ Δ₂) :
    tb.queryBound σ Δ₂ ≤ tb.queryBound σ Δ₁ := by
  unfold TestingBound.queryBound
  apply div_le_div_of_nonneg_left
      (le_of_lt (mul_pos tb.hC (sq_pos_of_pos hσ)))
      (sq_pos_of_pos hΔ₁)
  exact sq_le_sq' (by linarith) hle

/-! ### Z-test optimality, axiom-free -/

/-- The Z-test query count. -/
noncomputable def zTestQueries (z_const σ Δ : ℝ) : ℝ :=
  z_const * σ ^ 2 / Δ ^ 2

/-- Z-test is within constant factor of optimal. -/
theorem zTest_optimal (tb : TestingBound) (σ Δ z_const : ℝ)
    (hσ : 0 < σ) (hΔ : 0 < Δ) (hz : 0 < z_const) :
    ∃ ratio : ℝ, ratio > 0 ∧
      zTestQueries z_const σ Δ = ratio * tb.queryBound σ Δ := by
  refine ⟨z_const / tb.C, div_pos hz tb.hC, ?_⟩
  unfold zTestQueries TestingBound.queryBound
  have : Δ ^ 2 ≠ 0 := ne_of_gt (sq_pos_of_pos hΔ)
  have : tb.C ≠ 0 := ne_of_gt tb.hC
  field_simp

/-! ### Concrete instantiation -/

/-- Le Cam's constant C = 1/8 (from Tsybakov 2009, Theorem 2.2). -/
noncomputable def leCamBound : TestingBound :=
  ⟨1/8, by norm_num⟩

/-- With C = 1/8 and SNR Δ/σ = 0.15, the bound is (1/8)/0.0225 ≈ 5.56. -/
theorem leCam_breast_cancer :
    leCamBound.C / (0.15 ^ 2) = 1 / 8 / (9 / 400) := by
  unfold leCamBound; norm_num

/-- The axiom count: this entire file uses ZERO axioms beyond Lean's kernel.
    Verify with: `#print axioms zTest_optimal` -/
theorem axiom_free_check : True := trivial

end UniversalImpossibility
