/-
  Robustness of the flip rate: Lipschitz continuity.

  The flip rate function flip(ρ) = Φ(-SNR(ρ)) is Lipschitz-continuous in ρ.
  This means small perturbations in the correlation parameter ρ cause only
  small perturbations in the flip rate — the instability phenomenon degrades
  gracefully rather than exhibiting discontinuous jumps.

  Key insight: Φ is Lipschitz with constant 1/√(2π) (the maximum of the
  standard normal PDF), and SNR is smooth in ρ. Their composition is
  therefore Lipschitz by the chain rule for Lipschitz functions.

  Supplement: §S31 Robustness / Lipschitz continuity
-/
import UniversalImpossibility.GaussianFlipRate

set_option autoImplicit false

namespace UniversalImpossibility

open MeasureTheory ProbabilityTheory

/-! ### Lipschitz constants and abstract framework -/

/-- A function f : ℝ → ℝ is C-Lipschitz on all of ℝ. -/
def IsLipschitzWith (C : ℝ) (f : ℝ → ℝ) : Prop :=
  0 ≤ C ∧ ∀ x y : ℝ, |f x - f y| ≤ C * |x - y|

/-- The Lipschitz constant for the standard normal CDF.
    Equal to 1/√(2π), the maximum of the standard normal PDF. -/
noncomputable def C_Phi : ℝ := 1 / Real.sqrt (2 * Real.pi)

/-- C_Phi is positive (1/√(2π) > 0). -/
theorem C_Phi_pos : 0 < C_Phi := by
  unfold C_Phi
  apply div_pos one_pos
  apply Real.sqrt_pos_of_pos
  apply mul_pos two_pos Real.pi_pos

/-! ### Phi is Lipschitz -/

/-- The flip rate as a function of SNR alone: flip(s) = Φ(-s).

    If Φ is C_Phi-Lipschitz (hypothesis — the full proof requires bounding
    the Gaussian PDF, which needs Radon-Nikodym + exp bounds not yet
    available for this specific application in Mathlib), then the flip
    rate inherits Lipschitz continuity.

    We state theorems with the Lipschitz property of Φ as an explicit
    hypothesis, making the dependency transparent. -/
noncomputable def flipOfSNR (s : ℝ) : ℝ := Phi (-s)

/-- flipOfSNR is just Phi composed with negation. -/
theorem flipOfSNR_eq (s : ℝ) : flipOfSNR s = Phi (-s) := rfl

/-! ### Lipschitz chain rule -/

/-- The identity function is 1-Lipschitz. -/
theorem isLipschitzWith_id : IsLipschitzWith 1 id := by
  constructor
  · linarith
  · intro x y
    simp [one_mul]

/-- Negation is 1-Lipschitz. -/
theorem isLipschitzWith_neg : IsLipschitzWith 1 (fun x : ℝ => -x) := by
  constructor
  · linarith
  · intro x y
    show |(-x) - (-y)| ≤ 1 * |x - y|
    rw [one_mul, neg_sub_neg]
    exact le_of_eq (abs_sub_comm y x)

/-- Composition of Lipschitz functions is Lipschitz (chain rule). -/
theorem isLipschitzWith_comp {C₁ C₂ : ℝ} {f g : ℝ → ℝ}
    (hf : IsLipschitzWith C₁ f) (hg : IsLipschitzWith C₂ g) :
    IsLipschitzWith (C₁ * C₂) (f ∘ g) := by
  obtain ⟨hC₁, hf_lip⟩ := hf
  obtain ⟨hC₂, hg_lip⟩ := hg
  constructor
  · exact mul_nonneg hC₁ hC₂
  · intro x y
    simp only [Function.comp]
    calc |f (g x) - f (g y)|
        ≤ C₁ * |g x - g y| := hf_lip (g x) (g y)
      _ ≤ C₁ * (C₂ * |x - y|) := by
          apply mul_le_mul_of_nonneg_left (hg_lip x y) hC₁
      _ = C₁ * C₂ * |x - y| := by ring

/-- A constant function is 0-Lipschitz. -/
theorem isLipschitzWith_const (c : ℝ) : IsLipschitzWith 0 (fun _ : ℝ => c) := by
  constructor
  · linarith
  · intro x y
    simp

/-! ### Flip rate is Lipschitz in SNR -/

/-- If Φ is C_Phi-Lipschitz, then flipOfSNR is C_Phi-Lipschitz.
    This follows because flipOfSNR = Φ ∘ (- ·) and negation is 1-Lipschitz. -/
theorem flipOfSNR_lipschitz
    (hPhi : IsLipschitzWith C_Phi Phi) :
    IsLipschitzWith C_Phi flipOfSNR := by
  -- flipOfSNR = Phi ∘ (- ·), negation is 1-Lipschitz
  have h := isLipschitzWith_comp hPhi isLipschitzWith_neg
  -- C_Phi * 1 = C_Phi
  rw [mul_one] at h
  exact h

/-! ### Flip rate is Lipschitz in ρ -/

/-- The flip rate as a function of ρ, given an SNR function.
    In general, SNR(ρ) depends on the model class. -/
noncomputable def flipOfRho (snr : ℝ → ℝ) (ρ : ℝ) : ℝ :=
  flipOfSNR (snr ρ)

/-- If Φ is C_Phi-Lipschitz and SNR is L-Lipschitz in ρ,
    then the flip rate is (C_Phi * L)-Lipschitz in ρ.
    This is the main robustness theorem. -/
theorem flipOfRho_lipschitz
    (snr : ℝ → ℝ)
    (L : ℝ)
    (hPhi : IsLipschitzWith C_Phi Phi)
    (hsnr : IsLipschitzWith L snr) :
    IsLipschitzWith (C_Phi * L) (flipOfRho snr) := by
  -- flipOfRho snr = flipOfSNR ∘ snr = (Phi ∘ neg) ∘ snr
  -- Phi ∘ neg is C_Phi-Lipschitz, snr is L-Lipschitz
  have h_flip := flipOfSNR_lipschitz hPhi
  exact isLipschitzWith_comp h_flip hsnr

/-! ### Corollary: ε-δ robustness -/

/-- Small perturbations in ρ cause small perturbations in the flip rate.
    If |ρ₁ - ρ₂| < ε, then |flip(ρ₁) - flip(ρ₂)| ≤ (C_Phi * L) * ε. -/
theorem flip_rate_robust
    (snr : ℝ → ℝ)
    (L : ℝ)
    (hPhi : IsLipschitzWith C_Phi Phi)
    (hsnr : IsLipschitzWith L snr)
    (ρ₁ ρ₂ ε : ℝ)
    (_hε : 0 ≤ ε)
    (hclose : |ρ₁ - ρ₂| ≤ ε) :
    |flipOfRho snr ρ₁ - flipOfRho snr ρ₂| ≤ C_Phi * L * ε := by
  obtain ⟨hCL, hflip_lip⟩ := flipOfRho_lipschitz snr L hPhi hsnr
  calc |flipOfRho snr ρ₁ - flipOfRho snr ρ₂|
      ≤ C_Phi * L * |ρ₁ - ρ₂| := hflip_lip ρ₁ ρ₂
    _ ≤ C_Phi * L * ε := by
        apply mul_le_mul_of_nonneg_left hclose hCL

/-- Specialized version: the flip rate perturbation vanishes as ε → 0.
    This establishes continuity as a special case of Lipschitz. -/
theorem flip_rate_continuous_at
    (snr : ℝ → ℝ)
    (L : ℝ)
    (hPhi : IsLipschitzWith C_Phi Phi)
    (hsnr : IsLipschitzWith L snr)
    (ρ₀ : ℝ) :
    ∀ δ : ℝ, 0 < δ → ∃ ε : ℝ, 0 < ε ∧
      ∀ ρ : ℝ, |ρ - ρ₀| < ε → |flipOfRho snr ρ - flipOfRho snr ρ₀| < δ := by
  intro δ hδ
  obtain ⟨hCL, hflip_lip⟩ := flipOfRho_lipschitz snr L hPhi hsnr
  by_cases hCL_zero : C_Phi * L = 0
  · -- If the Lipschitz constant is 0, the function is constant on its range
    refine ⟨1, one_pos, fun ρ hρ => ?_⟩
    have h := hflip_lip ρ ρ₀
    rw [hCL_zero, zero_mul] at h
    linarith [abs_nonneg (flipOfRho snr ρ - flipOfRho snr ρ₀)]
  · -- Choose ε = δ / (C_Phi * L)
    have hCL_pos : 0 < C_Phi * L := by
      rcases lt_or_eq_of_le hCL with h | h
      · exact h
      · exact absurd h.symm hCL_zero
    refine ⟨δ / (C_Phi * L), div_pos hδ hCL_pos, fun ρ hρ => ?_⟩
    calc |flipOfRho snr ρ - flipOfRho snr ρ₀|
        ≤ C_Phi * L * |ρ - ρ₀| := hflip_lip ρ ρ₀
      _ < C_Phi * L * (δ / (C_Phi * L)) := by
          apply mul_lt_mul_of_pos_left hρ hCL_pos
      _ = δ := mul_div_cancel₀ δ (ne_of_gt hCL_pos)

/-! ### Closer correlations yield tighter flip rate bounds -/

/-- If ρ₁ is closer to ρ₀ than ρ₂ is, the Lipschitz bound on the flip
    rate perturbation for ρ₁ is no larger than for ρ₂. -/
theorem flip_perturbation_bound_monotone
    (snr : ℝ → ℝ)
    (L : ℝ)
    (hPhi : IsLipschitzWith C_Phi Phi)
    (hsnr : IsLipschitzWith L snr)
    (ρ₀ ρ₁ ρ₂ : ℝ)
    (hclose : |ρ₁ - ρ₀| ≤ |ρ₂ - ρ₀|) :
    |flipOfRho snr ρ₁ - flipOfRho snr ρ₀| ≤ C_Phi * L * |ρ₂ - ρ₀| := by
  obtain ⟨hCL, hflip_lip⟩ := flipOfRho_lipschitz snr L hPhi hsnr
  calc |flipOfRho snr ρ₁ - flipOfRho snr ρ₀|
      ≤ C_Phi * L * |ρ₁ - ρ₀| := hflip_lip ρ₁ ρ₀
    _ ≤ C_Phi * L * |ρ₂ - ρ₀| :=
        mul_le_mul_of_nonneg_left hclose hCL

end UniversalImpossibility
