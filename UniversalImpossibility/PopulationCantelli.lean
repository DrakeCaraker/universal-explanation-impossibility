/-
  PopulationCantelli.lean — Measure-theoretic (population) form of Cantelli's inequality.

  Cantelli's one-sided inequality bounds the upper tail of a real random variable's
  deviation from its mean by  v / (v + λ²), where  v = Var[X]  is the variance and
  λ > 0.  This is the population (measure-theoretic) analogue of the sample-based
  Cantelli guarantee.

  Mathlib provides Chebyshev's inequality
  (`ProbabilityTheory.meas_ge_le_variance_div_sq`) but NOT Cantelli.  This file fills
  that gap, using the standard "optimize over the shift u" argument and reusing:

    * Markov's inequality           `MeasureTheory.meas_ge_le_lintegral_div`
    * variance as an integral       `ProbabilityTheory.variance_eq_integral`
    * `ENNReal.ofReal`/lintegral    `MeasureTheory.ofReal_integral_eq_lintegral_ofReal`

  from `Mathlib/Probability/Moments/Variance.lean` and the measure-theory integral
  library, rather than re-deriving Markov.

  Proof sketch (upper tail).  For any shift u ≥ 0,
      {λ ≤ X - m}  ⊆  {(λ+u)² ≤ (X - m + u)²}
  (on the event, X - m + u ≥ λ + u ≥ 0, and squaring is monotone on nonnegatives).
  Markov applied to the nonnegative variable (X - m + u)² gives
      μ{(λ+u)² ≤ (X - m + u)²} ≤ 𝔼[(X - m + u)²] / (λ+u)².
  Expanding, 𝔼[(X - m + u)²] = v + u² (the cross term 2u·𝔼[X - m] vanishes).
  Setting u = v/λ collapses (v + u²)/(λ + u)² to v/(v + λ²).

  No new axioms are introduced; the results use only Lean core axioms
  (propext, Classical.choice, Quot.sound).
-/

import Mathlib.Probability.Moments.Variance

set_option autoImplicit false

open MeasureTheory ProbabilityTheory
open scoped ENNReal ProbabilityTheory

namespace UniversalImpossibility.PopulationCantelli

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} {X : Ω → ℝ}

/-- **Cantelli's inequality, upper tail (measure-theoretic / population form).**

For a probability measure `μ` and a real random variable `X ∈ L²(μ)` with mean
`m = μ[X]` and variance `v = variance X μ`, for every `λ > 0`
    `μ {ω | λ ≤ X ω - m} ≤ ENNReal.ofReal (v / (v + λ²))`. -/
theorem cantelli_upper [IsProbabilityMeasure μ] (hX : MemLp X 2 μ) {lam : ℝ}
    (hlam : 0 < lam) :
    μ {ω | lam ≤ X ω - μ[X]} ≤
      ENNReal.ofReal (variance X μ / (variance X μ + lam ^ 2)) := by
  -- variance is nonnegative
  have hvnn : 0 ≤ variance X μ := variance_nonneg X μ
  -- integrability of `X`, of the centered variable, and of its square
  have hX_int : Integrable X μ := hX.integrable one_le_two
  have hXm : MemLp (fun ω => X ω - μ[X]) 2 μ := hX.sub (memLp_const _)
  have hXm_int : Integrable (fun ω => X ω - μ[X]) μ := hXm.integrable one_le_two
  have hXmsq_int : Integrable (fun ω => (X ω - μ[X]) ^ 2) μ := hXm.integrable_sq
  -- the mean of the centered variable is zero
  have hXmean : ∫ ω, (X ω - μ[X]) ∂μ = 0 := by
    rw [integral_sub hX_int (integrable_const _), integral_const, probReal_univ, one_smul,
      sub_self]
  -- variance equals the integral of the square of the centered variable
  have hv_int : variance X μ = ∫ ω, (X ω - μ[X]) ^ 2 ∂μ :=
    variance_eq_integral hX.aemeasurable
  -- ===== key inequality for an arbitrary nonnegative shift `u` =====
  have key : ∀ u : ℝ, 0 ≤ u →
      μ {ω | lam ≤ X ω - μ[X]} ≤
        ENNReal.ofReal ((variance X μ + u ^ 2) / (lam + u) ^ 2) := by
    intro u hu
    have hlu : 0 < lam + u := by linarith
    have hlu2 : 0 < (lam + u) ^ 2 := by positivity
    -- integrability of the shifted square
    have hXmu : MemLp (fun ω => X ω - μ[X] + u) 2 μ := hXm.add (memLp_const _)
    have hfi : Integrable (fun ω => (X ω - μ[X] + u) ^ 2) μ := hXmu.integrable_sq
    -- expectation of the shifted square is v + u²
    have hexp : ∫ ω, (X ω - μ[X] + u) ^ 2 ∂μ = variance X μ + u ^ 2 := by
      have hrw : (fun ω => (X ω - μ[X] + u) ^ 2)
          = (fun ω => (X ω - μ[X]) ^ 2 + (2 * u) * (X ω - μ[X]) + u ^ 2) := by
        funext ω; ring
      have h1 : Integrable (fun ω => (X ω - μ[X]) ^ 2 + (2 * u) * (X ω - μ[X])) μ :=
        hXmsq_int.add (hXm_int.const_mul (2 * u))
      rw [hrw,
        integral_add h1 (integrable_const _),
        integral_add hXmsq_int (hXm_int.const_mul (2 * u)),
        integral_const_mul, hXmean, integral_const, ← hv_int, probReal_univ]
      simp
    -- `ENNReal.ofReal`-valued shifted square is a.e.-measurable
    have hg_meas :
        AEMeasurable (fun ω => ENNReal.ofReal ((X ω - μ[X] + u) ^ 2)) μ :=
      ENNReal.measurable_ofReal.comp_aemeasurable hfi.aestronglyMeasurable.aemeasurable
    -- its Lebesgue integral equals ofReal of the (real) expectation
    have hlint : ∫⁻ ω, ENNReal.ofReal ((X ω - μ[X] + u) ^ 2) ∂μ
        = ENNReal.ofReal (variance X μ + u ^ 2) := by
      rw [← ofReal_integral_eq_lintegral_ofReal hfi (ae_of_all _ fun ω => sq_nonneg _), hexp]
    -- the upper-tail event is contained in the level set of the shifted square
    have hsub : {ω | lam ≤ X ω - μ[X]}
        ⊆ {ω | ENNReal.ofReal ((lam + u) ^ 2)
              ≤ ENNReal.ofReal ((X ω - μ[X] + u) ^ 2)} := by
      intro ω hω
      simp only [Set.mem_setOf_eq] at hω ⊢
      apply ENNReal.ofReal_le_ofReal
      have h2 : lam + u ≤ X ω - μ[X] + u := by linarith
      exact pow_le_pow_left₀ hlu.le h2 2
    -- assemble: monotonicity + Markov + rewrite the bound
    calc
      μ {ω | lam ≤ X ω - μ[X]}
          ≤ μ {ω | ENNReal.ofReal ((lam + u) ^ 2)
                ≤ ENNReal.ofReal ((X ω - μ[X] + u) ^ 2)} := measure_mono hsub
      _ ≤ (∫⁻ ω, ENNReal.ofReal ((X ω - μ[X] + u) ^ 2) ∂μ)
              / ENNReal.ofReal ((lam + u) ^ 2) :=
            meas_ge_le_lintegral_div hg_meas
              (ENNReal.ofReal_ne_zero_iff.mpr hlu2) ENNReal.ofReal_ne_top
      _ = ENNReal.ofReal (variance X μ + u ^ 2) / ENNReal.ofReal ((lam + u) ^ 2) := by
            rw [hlint]
      _ = ENNReal.ofReal ((variance X μ + u ^ 2) / (lam + u) ^ 2) :=
            (ENNReal.ofReal_div_of_pos hlu2).symm
  -- ===== optimize: choose u = v/λ =====
  have hmain := key (variance X μ / lam) (div_nonneg hvnn hlam.le)
  -- algebra: (v + (v/λ)²)/(λ + v/λ)² = v/(v + λ²)
  have hlam' : lam ≠ 0 := hlam.ne'
  have hden : (0 : ℝ) < variance X μ + lam ^ 2 :=
    add_pos_of_nonneg_of_pos hvnn (by positivity)
  have hbase : lam + variance X μ / lam ≠ 0 :=
    (add_pos_of_pos_of_nonneg hlam (div_nonneg hvnn hlam.le)).ne'
  -- optimality identity: at u = v/λ the cross term (v - u·λ)² vanishes
  have hu0 : variance X μ / lam * lam = variance X μ := div_mul_cancel₀ _ hlam'
  have cross :
      (variance X μ + (variance X μ / lam) ^ 2) * (variance X μ + lam ^ 2)
        = variance X μ * (lam + variance X μ / lam) ^ 2 := by
    have hsq : (variance X μ - variance X μ / lam * lam) ^ 2 = 0 := by rw [hu0]; ring
    nlinarith [hsq]
  have hid :
      (variance X μ + (variance X μ / lam) ^ 2) / (lam + variance X μ / lam) ^ 2
        = variance X μ / (variance X μ + lam ^ 2) := by
    rw [div_eq_iff (pow_ne_zero 2 hbase), div_mul_eq_mul_div, eq_div_iff hden.ne']
    exact cross
  rwa [hid] at hmain

/-- **Cantelli's inequality, lower tail (measure-theoretic / population form).**

Symmetric counterpart of `cantelli_upper`, obtained by applying it to `-X`:
for `λ > 0`,
    `μ {ω | X ω - m ≤ -λ} ≤ ENNReal.ofReal (v / (v + λ²))`. -/
theorem cantelli_lower [IsProbabilityMeasure μ] (hX : MemLp X 2 μ) {lam : ℝ}
    (hlam : 0 < lam) :
    μ {ω | X ω - μ[X] ≤ -lam} ≤
      ENNReal.ofReal (variance X μ / (variance X μ + lam ^ 2)) := by
  -- apply the upper-tail bound to `-X`
  have hY := cantelli_upper (X := fun ω => -X ω) hX.neg hlam
  rw [variance_fun_neg] at hY
  -- the mean of `-X` is `-μ[X]`
  have hne : (μ[fun ω => -X ω]) = -μ[X] := integral_neg X
  -- the level sets coincide
  have hset : {ω | X ω - μ[X] ≤ -lam}
      = {ω | lam ≤ (fun ω => -X ω) ω - μ[fun ω => -X ω]} := by
    ext ω
    simp only [Set.mem_setOf_eq, hne]
    constructor <;> intro h <;> linarith
  rw [hset]
  exact hY

end UniversalImpossibility.PopulationCantelli
