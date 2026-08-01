/-
  ScoreRegularityNEF.lean — M2 of the score-regularity frontier campaign:
  the natural-exponential-family (tilted) discharge of `CramerRaoScoreProperty`.

  CONTEXT. `ParetoGlobal.lean` isolates the analytic core of the Cramér–Rao
  step as the named hypothesis `CramerRaoScoreProperty μ unbiased I` and
  discharges it only for the single-sample Gaussian location model with the
  SINGLETON class {id} — where the unit-covariance condition closes by
  covariance algebra alone, with no differentiation under the integral sign.
  M1 (`ScoreRegularity.lean`) supplied the one missing analytic lemma:
  `hasDerivAt_integral_mul_exp`, differentiation under the integral for
  t ↦ ∫ T·exp(t·X) dμ with a general estimator `T`.

  WHAT THIS FILE PROVES (M2). Following SCORE_REGULARITY_SCOPING.md §3.2
  (decomposition B), the parametric family is run THROUGH the base measure μ
  as the exponential tilt μ_t = μ.tilted (t·X·), t near 0:

  1. `unbiasedNEF X μ` — the regular unbiased-estimator class for the tilt
     parameter: square-integrable `T` with the M1 integrability conditions
     near 0 and unit-rate unbiasedness under tilting,
     ∫ T dμ_t = ∫ T dμ + t for t near 0. This is the honest formalization of
     "regular unbiased estimator of the natural parameter": unbiasedness is
     the estimand condition, the integrability fields are exactly the
     regularity that every classical Cramér–Rao proof assumes.

  2. `cramerRaoScoreProperty_tilted` — the MAIN theorem: for ANY probability
     measure μ and statistic X with exponential moments near 0
     (`0 ∈ interior (integrableExpSet X μ)`) and `Var[X; μ] = I`, the score
     property `CramerRaoScoreProperty μ (unbiasedNEF X μ) I` holds with
     score witness S = X − E[X]. No positivity hypothesis on I is needed.
     This discharges the named hypothesis for EVERY natural exponential
     family at once (Gaussian, Poisson, Bernoulli, exponential, Gamma with
     known shape, …), for the full regular unbiased class — not a singleton.
     The heart (O4) is machine-checked differentiation under the integral:
     d/dt [∫ T·exp(tX) dμ / ∫ exp(tX) dμ] at 0 equals 1 by unbiasedness,
     which after evaluating the quotient rule at t = 0 is exactly
     ∫ T·X dμ − (∫ T dμ)(∫ X dμ) = cov(T, X) = cov(T, S) = 1.

  3. Sanity check: the Gaussian instance is RE-DERIVED through the new
     theorem (`cramerRaoScoreProperty_gaussianReal_via_tilted`), with the
     normalized statistic X(x) = (x−m)/v, integrableExpSet = ℝ, tilt moving
     the mean at unit rate, and Fisher information Var[X] = 1/v. Non-vacuity
     is proved, not assumed: the identity estimator provably belongs to the
     class (`id_mem_unbiasedNEF_gaussianReal`). The existing
     `cramerRaoScoreProperty_gaussianReal` in ParetoGlobal.lean is untouched.

  WHAT THIS FILE DOES NOT PROVE. It does NOT instantiate the M-sample
  exchangeable Gaussian model into `dash_mvue`/`dash_global_min_variance` —
  that is M3 (forthcoming): it requires the multivariate Gaussian joint
  measure, Σ_exch positive semidefiniteness, and membership of the DASH
  aggregate in the class. Families WITHOUT exponential moments near 0
  (e.g. Cauchy-type location families) are not covered by this route; for
  them `CramerRaoScoreProperty` remains exactly the named hypothesis it was,
  in the repo's Sard pattern.

  Zero new axioms. Zero sorry.
-/

import UniversalImpossibility.ScoreRegularity
import UniversalImpossibility.ParetoGlobal
import Mathlib.Probability.Moments.Tilted

set_option autoImplicit false

open MeasureTheory ProbabilityTheory Real Filter
open scoped Topology NNReal

namespace UniversalImpossibility.ScoreRegularityNEF

/-! ## §1 The regular unbiased-estimator class of a natural exponential family -/

/-- **The regular unbiased estimators of the tilt parameter.** For a statistic
    `X` and base measure `μ`, the exponential tilt `μ_t = μ.tilted (t·X·)`
    is a one-parameter family through `μ = μ_0`. `unbiasedNEF X μ` is the
    class of estimators `T` that are:

    * square-integrable (`MemLp T 2 μ` — finite variance, the minimal
      requirement for a Cramér–Rao statement);
    * regular: `T·exp(tX)` and `T·X·exp(tX)` are integrable for `t` near 0 —
      exactly the hypotheses of M1's `hasDerivAt_integral_mul_exp`, i.e. the
      domination conditions every classical proof of Cramér–Rao assumes when
      it differentiates under the integral sign;
    * unbiased at unit rate: `∫ T dμ_t = ∫ T dμ + t` for `t` near 0 — the
      tilt moves the estimand at unit speed and `T` tracks it exactly.

    Non-vacuity is proved, not assumed: for the Gaussian instance below the
    identity estimator provably belongs (`id_mem_unbiasedNEF_gaussianReal`). -/
def unbiasedNEF {Ω : Type*} [MeasurableSpace Ω] (X : Ω → ℝ) (μ : Measure Ω) :
    Set (Ω → ℝ) :=
  {T | MemLp T 2 μ ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), Integrable (fun ω => T ω * exp (t * X ω)) μ) ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), Integrable (fun ω => T ω * X ω * exp (t * X ω)) μ) ∧
    ∀ᶠ t in 𝓝 (0 : ℝ),
      ∫ ω, T ω ∂(μ.tilted (fun ω => t * X ω)) = (∫ ω, T ω ∂μ) + t}

/-! ## §2 The NEF discharge of the score property (M2 main theorem) -/

/-- **The Cramér–Rao score property holds for every natural exponential
    family (M2).** For a probability measure `μ` and a statistic `X` with
    exponential moments in a neighbourhood of 0 and variance `I`, the score
    `S = X − E[X]` witnesses `CramerRaoScoreProperty μ (unbiasedNEF X μ) I`:

    * O1 `MemLp S 2 μ` — from `memLp_of_mem_interior_integrableExpSet`;
    * O2 `E[S] = 0` — by construction;
    * O3 `Var[S] = Var[X] = I` — shift invariance of the variance;
    * O4 `cov(T, S) = 1` for every `T` in the class — the heart: by M1 the
      numerator `t ↦ ∫ T·exp(tX) dμ` and (via `hasDerivAt_mgf`) the
      denominator `t ↦ ∫ exp(tX) dμ` of `∫ T dμ_t` are differentiable at 0;
      unit-rate unbiasedness forces the quotient's derivative to be 1, and
      evaluating the quotient rule at `t = 0` (where the mgf is 1) yields
      `∫ T·X dμ − (∫ T dμ)(∫ X dμ) = 1`, i.e. `cov(T, X) = cov(T, S) = 1`.

    No positivity of `I` is required for the score property itself; positivity
    enters only downstream in `cramer_rao_bound_of_score`. -/
theorem cramerRaoScoreProperty_tilted
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {I : ℝ}
    (h0 : (0 : ℝ) ∈ interior (integrableExpSet X μ))
    (hI : Var[X; μ] = I) :
    ParetoGlobal.CramerRaoScoreProperty μ (unbiasedNEF X μ) I := by
  have hX2 : MemLp X 2 μ := memLp_of_mem_interior_integrableExpSet h0 2
  have hXint : Integrable X μ := hX2.integrable one_le_two
  have hXm : AEStronglyMeasurable X μ :=
    (aemeasurable_of_mem_interior_integrableExpSet h0).aestronglyMeasurable
  refine ⟨fun ω => X ω - ∫ x, X x ∂μ, hX2.sub (memLp_const _), ?_, ?_, ?_⟩
  · -- O2: the score has mean zero
    rw [integral_sub hXint (integrable_const _), integral_const, probReal_univ,
      one_smul, sub_self]
  · -- O3: the score's variance is the Fisher information Var[X] = I
    rw [variance_sub_const hXm, hI]
  · -- O4: unit covariance against every regular unbiased estimator
    rintro T ⟨hT2, hTint, hTXint, hunb⟩ _
    -- Numerator derivative at 0, from M1
    have hN : HasDerivAt (fun t => ∫ ω, T ω * exp (t * X ω) ∂μ)
        (∫ ω, T ω * X ω ∂μ) 0 := by
      simpa using ScoreRegularity.hasDerivAt_integral_mul_exp hTint hTXint
    -- Denominator (mgf) derivative at 0
    have hD : HasDerivAt (mgf X μ) (∫ ω, X ω ∂μ) 0 := by
      simpa using hasDerivAt_mgf h0
    have hDne : mgf X μ 0 ≠ 0 := by rw [mgf_zero]; exact one_ne_zero
    -- Quotient rule at 0, simplified using mgf(0) = 1
    have hq := hN.div hD hDne
    have hq' : HasDerivAt (fun t => (∫ ω, T ω * exp (t * X ω) ∂μ) / mgf X μ t)
        ((∫ ω, T ω * X ω ∂μ) - (∫ ω, T ω ∂μ) * ∫ ω, X ω ∂μ) 0 := by
      convert hq using 1
      rw [mgf_zero]
      simp
    -- The quotient IS the tilted expectation
    have heq : ∀ t : ℝ, ∫ ω, T ω ∂(μ.tilted (fun ω => t * X ω))
        = (∫ ω, T ω * exp (t * X ω) ∂μ) / mgf X μ t := by
      intro t
      rw [integral_tilted_mul_eq_mgf, ← integral_div]
      congr 1
      funext ω
      rw [smul_eq_mul]
      ring
    -- Unbiasedness: the quotient agrees with c + t near 0, so its derivative is 1
    have hev : (fun t : ℝ => (∫ ω, T ω ∂μ) + t)
        =ᶠ[𝓝 (0 : ℝ)] fun t => (∫ ω, T ω * exp (t * X ω) ∂μ) / mgf X μ t := by
      filter_upwards [hunb] with t ht
      rw [← ht]
      exact heq t
    have hlin : HasDerivAt (fun t : ℝ => (∫ ω, T ω ∂μ) + t)
        ((∫ ω, T ω * X ω ∂μ) - (∫ ω, T ω ∂μ) * ∫ ω, X ω ∂μ) 0 :=
      hq'.congr_of_eventuallyEq hev
    have hone : HasDerivAt (fun t : ℝ => (∫ ω, T ω ∂μ) + t) 1 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_add (∫ ω, T ω ∂μ)
    have hkey : (∫ ω, T ω * X ω ∂μ) - (∫ ω, T ω ∂μ) * ∫ ω, X ω ∂μ = 1 :=
      hlin.unique hone
    -- Conclude: cov(T, S) = cov(T, X) = E[TX] − E[T]E[X] = 1
    have hScov : cov[T, X; μ] = 1 := by
      rw [covariance_eq_sub hT2 hX2]
      exact hkey
    calc cov[T, fun ω => X ω - ∫ x, X x ∂μ; μ]
        = cov[T, X; μ] := covariance_sub_const_right hXint _
      _ = 1 := hScov

/-! ## §3 Sanity check: the Gaussian instance through the NEF theorem

  For `N(m, v)` the normalized statistic `X(x) = (x−m)/v` has
  `μ.map X = N(0, 1/v)`, hence all exponential moments
  (`integrableExpSet X μ = ℝ`), cgf `t ↦ t²/(2v)`, and tilted mean identity
  `E_{μ_t}[X] = t/v` — so the tilt moves the mean of the ORIGINAL observable
  at exactly unit rate: `E_{μ_t}[id] = m + t`. The identity estimator
  therefore belongs to `unbiasedNEF X μ` (non-vacuity), and the score
  property with Fisher information `Var[X] = 1/v` re-derives the Gaussian
  discharge of ParetoGlobal.lean through the general NEF theorem. -/

section GaussianSanityCheck

/-- The normalized Gaussian statistic `x ↦ (x−m)/v` pushes `N(m, v)` to
    `N(0, 1/v)`. -/
lemma map_normalized_gaussianReal (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    (gaussianReal m v).map (fun x => (x - m) / (v : ℝ)) = gaussianReal 0 v⁻¹ := by
  have hvR : ((v : ℝ)) ≠ 0 := NNReal.coe_ne_zero.mpr hv
  have hcomp : (fun x : ℝ => (x - m) / (v : ℝ))
      = (fun y : ℝ => (v : ℝ)⁻¹ * y) ∘ (fun x : ℝ => x + -m) := by
    funext x
    simp [Function.comp, div_eq_inv_mul, sub_eq_add_neg]
  rw [hcomp, ← Measure.map_map (by fun_prop) (by fun_prop),
    gaussianReal_map_add_const, add_neg_cancel, gaussianReal_map_const_mul,
    mul_zero]
  congr 1
  ext
  push_cast
  rw [sq, mul_assoc, inv_mul_cancel₀ hvR, mul_one]

/-- The normalized Gaussian statistic has ALL exponential moments. -/
lemma integrableExpSet_normalized_gaussianReal (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    integrableExpSet (fun x => (x - m) / (v : ℝ)) (gaussianReal m v) = Set.univ := by
  have hvR : ((v : ℝ)) ≠ 0 := NNReal.coe_ne_zero.mpr hv
  refine Set.eq_univ_of_forall fun t => ?_
  have h := (integrable_exp_mul_gaussianReal (μ := m) (v := v) (t / (v : ℝ))).const_mul
    (exp (-(t * m / (v : ℝ))))
  refine h.congr (ae_of_all _ fun x => ?_)
  dsimp only
  rw [← exp_add]
  congr 1
  field_simp
  ring

/-- **Non-vacuity of the class.** The identity estimator is a regular
    unbiased estimator of the tilt parameter of the normalized Gaussian
    family: tilting `N(m, v)` by `t·(x−m)/v` shifts the mean to `m + t`. -/
lemma id_mem_unbiasedNEF_gaussianReal (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    (fun x : ℝ => x)
      ∈ unbiasedNEF (fun x => (x - m) / (v : ℝ)) (gaussianReal m v) := by
  have hvR : ((v : ℝ)) ≠ 0 := NNReal.coe_ne_zero.mpr hv
  set X : ℝ → ℝ := fun x => (x - m) / (v : ℝ) with hXdef
  -- exponential moments everywhere
  have hti : ∀ t : ℝ, t ∈ interior (integrableExpSet X (gaussianReal m v)) := by
    intro t
    rw [hXdef, integrableExpSet_normalized_gaussianReal m v hv, interior_univ]
    exact Set.mem_univ t
  have hexp : ∀ t : ℝ, Integrable (fun x => exp (t * X x)) (gaussianReal m v) :=
    fun t => integrable_of_mem_integrableExpSet (interior_subset (hti t))
  have hX1 : ∀ t : ℝ, Integrable (fun x => X x * exp (t * X x)) (gaussianReal m v) := by
    intro t
    simpa using integrable_pow_mul_exp_of_mem_interior_integrableExpSet (hti t) 1
  have hX2 : ∀ t : ℝ, Integrable (fun x => X x ^ 2 * exp (t * X x)) (gaussianReal m v) :=
    fun t => integrable_pow_mul_exp_of_mem_interior_integrableExpSet (hti t) 2
  -- the algebraic pivot: v·X(x) = x − m
  have hxx : ∀ x : ℝ, (v : ℝ) * X x = x - m := by
    intro x
    simp only [hXdef]
    field_simp
  -- tilted mean of X via the cgf t²/(2v)
  have hcgf : cgf X (gaussianReal m v) = fun s : ℝ => (v : ℝ)⁻¹ * s ^ 2 / 2 := by
    funext s
    rw [hXdef, cgf_gaussianReal (map_normalized_gaussianReal m v hv) s]
    push_cast
    ring
  have hmean : ∀ t : ℝ,
      ∫ x, X x ∂((gaussianReal m v).tilted (fun x => t * X x)) = t / (v : ℝ) := by
    intro t
    have h1 : ∫ x, X x ∂((gaussianReal m v).tilted (fun x => t * X x))
        = deriv (cgf X (gaussianReal m v)) t := integral_tilted_mul_self (hti t)
    have h2 := ((hasDerivAt_pow 2 t).const_mul ((v : ℝ)⁻¹)).div_const 2
    rw [h1, hcgf, h2.deriv]
    push_cast
    field_simp
  refine ⟨by simpa using memLp_id_gaussianReal (μ := m) (v := v) 2, ?_, ?_, ?_⟩
  · -- integrability of id·exp(tX), for all t
    refine Eventually.of_forall fun t => ?_
    refine (((hexp t).const_mul m).add ((hX1 t).const_mul (v : ℝ))).congr
      (ae_of_all _ fun x => ?_)
    simp only [Pi.add_apply]
    linear_combination exp (t * X x) * hxx x
  · -- integrability of id·X·exp(tX), for all t
    refine Eventually.of_forall fun t => ?_
    refine (((hX1 t).const_mul m).add ((hX2 t).const_mul (v : ℝ))).congr
      (ae_of_all _ fun x => ?_)
    simp only [Pi.add_apply]
    linear_combination X x * exp (t * X x) * hxx x
  · -- unit-rate unbiasedness: E_{μ_t}[id] = m + t
    refine Eventually.of_forall fun t => ?_
    haveI hprob : IsProbabilityMeasure ((gaussianReal m v).tilted (fun x => t * X x)) :=
      isProbabilityMeasure_tilted (hexp t)
    have hXint_t : Integrable X ((gaussianReal m v).tilted (fun x => t * X x)) :=
      memLp_one_iff_integrable.mp (memLp_tilted_mul (hti t) 1)
    have hfun : (fun x : ℝ => x) = fun x => m + (v : ℝ) * X x := by
      funext x
      rw [hxx x]
      ring
    rw [integral_id_gaussianReal]
    calc ∫ x, x ∂((gaussianReal m v).tilted (fun x => t * X x))
        = ∫ x, (m + (v : ℝ) * X x) ∂((gaussianReal m v).tilted (fun x => t * X x)) := by
          rw [hfun]
      _ = m + (v : ℝ) * (t / (v : ℝ)) := by
          rw [integral_add (integrable_const m) (hXint_t.const_mul (v : ℝ)),
            integral_const_mul, hmean t, integral_const]
          simp
      _ = m + t := by field_simp

/-- **The Gaussian discharge, re-derived through the NEF theorem.** For
    `N(m, v)`, `v ≠ 0`, the score property holds with the FULL regular
    unbiased class `unbiasedNEF ((·−m)/v) (N(m,v))` — which provably contains
    the identity estimator — and Fisher information `1/v`. This reproduces
    (and strengthens from a singleton class to the regular class) the
    conclusion of `cramerRaoScoreProperty_gaussianReal`, as a machine-checked
    sanity check that `cramerRaoScoreProperty_tilted` says what it should. -/
theorem cramerRaoScoreProperty_gaussianReal_via_tilted (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    ParetoGlobal.CramerRaoScoreProperty (gaussianReal m v)
      (unbiasedNEF (fun x => (x - m) / (v : ℝ)) (gaussianReal m v)) (1 / (v : ℝ)) := by
  refine cramerRaoScoreProperty_tilted ?_ ?_
  · -- exponential moments near 0
    rw [integrableExpSet_normalized_gaussianReal m v hv, interior_univ]
    exact Set.mem_univ 0
  · -- Fisher information: Var[(x−m)/v] = 1/v
    have hXmeas : AEMeasurable (fun x : ℝ => (x - m) / (v : ℝ)) (gaussianReal m v) := by
      fun_prop
    rw [← variance_id_map hXmeas, map_normalized_gaussianReal m v hv,
      variance_id_gaussianReal]
    push_cast
    rw [one_div]

end GaussianSanityCheck

end UniversalImpossibility.ScoreRegularityNEF
