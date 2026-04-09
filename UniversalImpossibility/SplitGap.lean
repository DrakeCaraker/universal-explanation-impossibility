/-
  Split gap lemma: the difference in split counts between the first-mover
  and any other feature in the same group is at least ½ρ²T.

  Pure algebra from Axioms 2 and 3.
-/
import UniversalImpossibility.Defs

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Helper lemmas: ρ² bounds and denominator positivity -/

/-- ρ² < 1 when ρ ∈ (0,1). -/
lemma rho_sq_lt_one : fs.ρ ^ 2 < 1 := by
  have : fs.ρ * fs.ρ < 1 :=
    calc fs.ρ * fs.ρ
        < fs.ρ * 1 := mul_lt_mul_of_pos_left fs.hρ_lt_one fs.hρ_pos
      _ = fs.ρ := mul_one fs.ρ
      _ < 1 := fs.hρ_lt_one
  linarith [sq fs.ρ]

/-- 1 - ρ² > 0 when ρ ∈ (0,1). -/
lemma one_minus_rho_sq_pos : (0 : ℝ) < 1 - fs.ρ ^ 2 :=
  sub_pos.mpr (rho_sq_lt_one fs)

/-- 2 - ρ² > 0 when ρ ∈ (0,1). -/
lemma denom_pos : (0 : ℝ) < 2 - fs.ρ ^ 2 := by
  linarith [rho_sq_lt_one fs]

/-- 2 - ρ² ≠ 0 when ρ ∈ (0,1). -/
lemma denom_ne_zero : (2 : ℝ) - fs.ρ ^ 2 ≠ 0 :=
  ne_of_gt (denom_pos fs)

/-! ### Split gap: exact value and lower bound -/

/-- Exact split-count gap: n_{j₁} - n_{j₂} = ρ²T / (2 - ρ²).
    Direct consequence of Axioms 2 and 3. -/
theorem split_gap_exact (f : Model) (j₁ j₂ : Fin fs.P) (ℓ : Fin fs.L)
    (hj₁ : j₁ ∈ fs.group ℓ) (hj₂ : j₂ ∈ fs.group ℓ)
    (hfm : firstMover fs f = j₁) (hne : firstMover fs f ≠ j₂) :
    splitCount fs j₁ f - splitCount fs j₂ f =
      fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2) := by
  have hfm_grp : firstMover fs f ∈ fs.group ℓ := by rw [hfm]; exact hj₁
  rw [splitCount_firstMover fs f j₁ hfm,
      splitCount_nonFirstMover fs f j₂ ℓ hj₂ hne hfm_grp]
  have := denom_ne_zero fs
  field_simp
  ring

/-- The gap ρ²T/(2-ρ²) is at least ½ρ²T, since 2 - ρ² ≤ 2. -/
theorem split_gap_ge_half :
    fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2) ≥
      1 / 2 * (fs.ρ ^ 2 * (numTrees : ℝ)) := by
  rw [ge_iff_le, le_div_iff₀ (denom_pos fs)]
  nlinarith [sq_nonneg fs.ρ, Nat.cast_nonneg (α := ℝ) numTrees,
             mul_nonneg (sq_nonneg fs.ρ) (mul_nonneg (sq_nonneg fs.ρ)
               (Nat.cast_nonneg (α := ℝ) numTrees))]

/-- Combined: the split gap between first-mover and non-first-mover
    in the same group is at least ½ρ²T. -/
theorem split_gap_lower_bound (f : Model) (j₁ j₂ : Fin fs.P) (ℓ : Fin fs.L)
    (hj₁ : j₁ ∈ fs.group ℓ) (hj₂ : j₂ ∈ fs.group ℓ)
    (hfm : firstMover fs f = j₁) (hne : firstMover fs f ≠ j₂) :
    splitCount fs j₁ f - splitCount fs j₂ f ≥
      1 / 2 * (fs.ρ ^ 2 * (numTrees : ℝ)) := by
  rw [split_gap_exact fs f j₁ j₂ ℓ hj₁ hj₂ hfm hne]
  exact split_gap_ge_half fs

end UniversalImpossibility
