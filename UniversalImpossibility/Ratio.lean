/-
  Attribution ratio lemma: the split-count ratio between first-mover
  and non-first-mover in the same group equals 1/(1-ρ²), and this
  diverges as ρ → 1.

  Algebraic consequence of Axioms 2 and 3; the limit is real analysis.
-/
import UniversalImpossibility.SplitGap

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Split-count ratio -/

/-- Split-count ratio between first-mover and non-first-mover = 1/(1-ρ²).
    This is the within-group inequity created by sequential boosting. -/
theorem splitCount_ratio (f : Model) (j₁ j₂ : Fin fs.P) (ℓ : Fin fs.L)
    (hj₁ : j₁ ∈ fs.group ℓ) (hj₂ : j₂ ∈ fs.group ℓ)
    (hfm : firstMover fs f = j₁) (hne : firstMover fs f ≠ j₂) :
    splitCount fs j₁ f / splitCount fs j₂ f =
      1 / (1 - fs.ρ ^ 2) := by
  have hfm_grp : firstMover fs f ∈ fs.group ℓ := by rw [hfm]; exact hj₁
  rw [splitCount_firstMover fs f j₁ hfm,
      splitCount_nonFirstMover fs f j₂ ℓ hj₂ hne hfm_grp]
  have h1 := denom_ne_zero fs
  have h2 : (1 : ℝ) - fs.ρ ^ 2 ≠ 0 := ne_of_gt (one_minus_rho_sq_pos fs)
  have h3 : (↑numTrees : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr numTrees_pos)
  field_simp

/-- Attribution ratio between first-mover and non-first-mover = 1/(1-ρ²).
    Follows from splitCount_ratio + strengthened Axiom 4 (model-wide c). -/
theorem attribution_ratio (f : Model) (j₁ j₂ : Fin fs.P) (ℓ : Fin fs.L)
    (hj₁ : j₁ ∈ fs.group ℓ) (hj₂ : j₂ ∈ fs.group ℓ)
    (hfm : firstMover fs f = j₁) (hne : firstMover fs f ≠ j₂) :
    attribution fs j₁ f / attribution fs j₂ f = 1 / (1 - fs.ρ ^ 2) := by
  obtain ⟨c, hc_pos, hc_eq⟩ := attribution_proportional fs f
  rw [hc_eq j₁, hc_eq j₂]
  rw [mul_div_mul_left _ _ (ne_of_gt hc_pos)]
  exact splitCount_ratio fs f j₁ j₂ ℓ hj₁ hj₂ hfm hne

/-! ### Ratio divergence -/

/-- As ρ → 1⁻, the ratio 1/(1-ρ²) → +∞ (Theorem 10(i)).
    Factor as 1/((1-ρ)(1+ρ)); as ρ → 1, (1+ρ) → 2 and 1/(1-ρ) → ∞. -/
theorem ratio_tendsto_atTop :
    Filter.Tendsto (fun ρ : ℝ => 1 / (1 - ρ ^ 2))
      (nhdsWithin 1 (Set.Iio 1)) Filter.atTop := by
  simp_rw [one_div]
  apply Filter.Tendsto.comp tendsto_inv_nhdsGT_zero
  -- Goal: Tendsto (fun ρ => 1 - ρ ^ 2) (𝓝[<] 1) (𝓝[>] 0)
  apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
  · -- Tendsto (fun ρ => 1 - ρ ^ 2) (𝓝[<] 1) (𝓝 0)
    have : Filter.Tendsto (fun ρ : ℝ => 1 - ρ ^ 2) (nhds 1) (nhds 0) := by
      have h : (fun ρ : ℝ => 1 - ρ ^ 2) = (fun ρ => 1 - ρ ^ 2) := rfl
      have : (1 : ℝ) - (1 : ℝ) ^ 2 = 0 := by ring
      rw [← this]
      exact ((continuous_const.sub (continuous_pow 2)).tendsto 1)
    exact this.mono_left nhdsWithin_le_nhds
  · -- ∀ᶠ ρ in 𝓝[<] 1, 1 - ρ ^ 2 ∈ Set.Ioi 0
    filter_upwards [Ioo_mem_nhdsLT (show (0 : ℝ) < 1 by norm_num)] with ρ hρ
    simp only [Set.mem_Ioi, Set.mem_Ioo] at *
    nlinarith [sq_nonneg ρ]

end UniversalImpossibility
