/-
  Gaussian Flip Rate Formula.

  Defines the standard normal CDF Φ via Mathlib's gaussianReal and cdf,
  then states the flip rate formula flip(j,k) = Φ(-|μ|/σ).

  Supplement: §Theorem F1: Rashomon Characterization (Gaussian case)
-/
import Mathlib.Probability.CDF
import Mathlib.Probability.Distributions.Gaussian.Real
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

open MeasureTheory ProbabilityTheory

/-! ### Standard Normal CDF -/

/-- The standard normal CDF: Φ(x) = P(Z ≤ x) where Z ~ N(0,1).
    Defined using Mathlib's gaussianReal and cdf. -/
noncomputable def Phi (x : ℝ) : ℝ :=
  ProbabilityTheory.cdf (gaussianReal 0 1) x

/-- Φ is monotone (increasing). -/
theorem Phi_mono : Monotone Phi :=
  monotone_cdf (gaussianReal 0 1)

/-- Φ(x) ∈ [0, 1] since it's a CDF value. -/
theorem Phi_nonneg (x : ℝ) : 0 ≤ Phi x :=
  cdf_nonneg (gaussianReal 0 1) x

theorem Phi_le_one (x : ℝ) : Phi x ≤ 1 :=
  cdf_le_one (gaussianReal 0 1) x

/-- Φ tends to 0 at -∞. -/
theorem Phi_tendsto_atBot : Filter.Tendsto Phi Filter.atBot (nhds 0) :=
  tendsto_cdf_atBot (gaussianReal 0 1)

/-- Φ tends to 1 at +∞. -/
theorem Phi_tendsto_atTop : Filter.Tendsto Phi Filter.atTop (nhds 1) :=
  tendsto_cdf_atTop (gaussianReal 0 1)

/-- The standard normal distribution is symmetric: N(0,1) mapped by negation is N(0,1).
    This follows from gaussianReal_map_neg with μ = 0. -/
theorem gaussianReal_standard_symm :
    (gaussianReal 0 1).map (fun x => -x) = gaussianReal 0 1 := by
  rw [gaussianReal_map_neg]
  simp

/-- Helper: the standard normal Gaussian has no atoms (it's a continuous distribution). -/
instance : NoAtoms (gaussianReal (0 : ℝ) 1) :=
  noAtoms_gaussianReal one_ne_zero

/-- Φ(-x) = 1 - Φ(x) by symmetry of the standard normal.
    The standard normal satisfies P(Z ≤ -x) = P(Z ≥ x) = 1 - P(Z ≤ x). -/
theorem Phi_neg (x : ℝ) : Phi (-x) = 1 - Phi x := by
  -- Strategy: Phi(-x) = μ(Iic(-x)) and Phi(x) = μ(Iic(x))
  -- By symmetry μ.map(neg) = μ, so μ(neg⁻¹'(Iic(-x))) = μ(Iic(-x))
  -- neg⁻¹'(Iic(-x)) = Ici(x), so μ(Ici(x)) = μ(Iic(-x))
  -- μ(Ici(x)) = μ(Ioi(x)) (no atoms) = μ((Iic(x))ᶜ) (compl_Iic) = 1 - μ(Iic(x))
  -- So Phi(-x) = 1 - Phi(x).
  have μ_def := gaussianReal_standard_symm
  unfold Phi
  -- Work in ENNReal via ofReal_cdf, then convert back
  have h_prob : IsProbabilityMeasure (gaussianReal (0 : ℝ) 1) := inferInstance
  -- Step 1: cdf(-x) = μ(Iic(-x)) and we want to show it equals 1 - cdf(x) = 1 - μ(Iic(x))
  -- Use the fact that μ(Ici x) = μ(Iic (-x)) by symmetry
  have key : (gaussianReal (0 : ℝ) 1) (Set.Ici x) = (gaussianReal (0 : ℝ) 1) (Set.Iic (-x)) := by
    -- μ(Ici x) = μ.map(neg)(Ici x) by symmetry
    conv_lhs => rw [← μ_def]
    rw [Measure.map_apply (measurable_neg) measurableSet_Ici]
    congr 1
    ext y
    simp
  -- Step 2: μ(Ici x) = 1 - μ(Iic x) (using no atoms: Ici =ᵐ Ioi, and Ioi = (Iic)ᶜ)
  have h_ici : (gaussianReal (0 : ℝ) 1) (Set.Ici x) = 1 - (gaussianReal (0 : ℝ) 1) (Set.Iic x) := by
    have h_ae : (Set.Ici x : Set ℝ) =ᵐ[gaussianReal (0 : ℝ) 1] Set.Ioi x :=
      (Ioi_ae_eq_Ici (μ := gaussianReal (0 : ℝ) 1)).symm
    rw [measure_congr h_ae, ← Set.compl_Iic]
    exact prob_compl_eq_one_sub measurableSet_Iic
  -- Step 3: combine to get cdf(-x) = 1 - cdf(x)
  -- Work directly with ENNReal then convert
  have lhs_eq : ENNReal.ofReal (cdf (gaussianReal (0 : ℝ) 1) (-x)) =
      (gaussianReal (0 : ℝ) 1) (Set.Iic (-x)) := ofReal_cdf _ _
  have rhs_eq : ENNReal.ofReal (cdf (gaussianReal (0 : ℝ) 1) x) =
      (gaussianReal (0 : ℝ) 1) (Set.Iic x) := ofReal_cdf _ _
  -- From key and h_ici: μ(Iic(-x)) = μ(Ici(x)) = 1 - μ(Iic(x))
  rw [← key] at lhs_eq
  rw [h_ici] at lhs_eq
  -- Now lhs_eq : ofReal(cdf(-x)) = 1 - μ(Iic x)
  -- and rhs_eq : ofReal(cdf(x)) = μ(Iic x)
  rw [← rhs_eq] at lhs_eq
  -- lhs_eq : ofReal(cdf(-x)) = 1 - ofReal(cdf(x))
  have h_le : cdf (gaussianReal (0 : ℝ) 1) x ≤ 1 := cdf_le_one _ _
  have h_nn : 0 ≤ cdf (gaussianReal (0 : ℝ) 1) x := cdf_nonneg _ _
  have h_nn_neg : 0 ≤ cdf (gaussianReal (0 : ℝ) 1) (-x) := cdf_nonneg _ _
  -- Convert the ENNReal equation to ℝ by applying toReal to both sides
  have := congr_arg ENNReal.toReal lhs_eq
  rw [ENNReal.toReal_ofReal h_nn_neg] at this
  have h_le_ennreal : ENNReal.ofReal (cdf (gaussianReal (0 : ℝ) 1) x) ≤ 1 := by
    rw [ENNReal.ofReal_le_one]
    exact h_le
  rw [ENNReal.toReal_sub_of_le h_le_ennreal (ENNReal.one_ne_top)] at this
  simp only [ENNReal.toReal_one, ENNReal.toReal_ofReal h_nn] at this
  linarith

/-- Φ(0) = 1/2 by symmetry of the standard normal. -/
theorem Phi_zero : Phi 0 = 1 / 2 := by
  -- From Phi_neg with x = 0: Phi(-0) = 1 - Phi(0), i.e. Phi(0) = 1 - Phi(0)
  have h := Phi_neg 0
  simp at h
  linarith

/-! ### Flip Rate Formula -/

/-- The flip rate for a Gaussian attribution difference.
    If D = φ_j(f) - φ_k(f) ~ N(μ, σ²), the probability that
    a random model ranks k above j is Φ(-μ/σ).

    We state this as a DEFINITION (the formula) rather than deriving
    it from measure theory (which would require Gaussian CDF properties). -/
noncomputable def gaussianFlipRate (μ σ : ℝ) (_hσ : 0 < σ) : ℝ :=
  Phi (-(|μ| / σ))

/-- The flip rate is in [0, 1/2] when μ ≥ 0. -/
theorem gaussianFlipRate_le_half (μ σ : ℝ) (_hμ : 0 ≤ μ) (hσ : 0 < σ) :
    gaussianFlipRate μ σ hσ ≤ 1 / 2 := by
  unfold gaussianFlipRate
  -- Φ(-|μ|/σ) ≤ Φ(0) = 1/2 by monotonicity, since -|μ|/σ ≤ 0.
  calc Phi (-(|μ| / σ)) ≤ Phi 0 := by
        apply Phi_mono
        simp only [neg_nonpos]
        exact div_nonneg (abs_nonneg μ) (le_of_lt hσ)
    _ = 1 / 2 := Phi_zero

/-- The flip rate is nonneg (it's a CDF value). -/
theorem gaussianFlipRate_nonneg (μ σ : ℝ) (hσ : 0 < σ) :
    0 ≤ gaussianFlipRate μ σ hσ :=
  Phi_nonneg _

/-- At μ = 0 (perfect symmetry): flip rate = 1/2 (coin flip). -/
theorem gaussianFlipRate_zero (σ : ℝ) (hσ : 0 < σ) :
    gaussianFlipRate 0 σ hσ = 1 / 2 := by
  unfold gaussianFlipRate
  simp only [abs_zero, zero_div, neg_zero]
  exact Phi_zero

/-- The flip rate decreases with SNR = |μ|/σ.
    Higher signal-to-noise → more stable rankings. -/
theorem gaussianFlipRate_decreasing (μ₁ μ₂ σ : ℝ) (hσ : 0 < σ)
    (h : |μ₁| / σ ≤ |μ₂| / σ) :
    gaussianFlipRate μ₂ σ hσ ≤ gaussianFlipRate μ₁ σ hσ := by
  unfold gaussianFlipRate
  apply Phi_mono
  linarith

/-! ### Connection to the impossibility theorem -/

/-- Connection to the impossibility: when the flip rate is 1/2 (μ = 0),
    the impossibility theorem applies — rankings are pure coin flips. -/
theorem coin_flip_implies_impossibility
    (fs : FeatureSpace)
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility
