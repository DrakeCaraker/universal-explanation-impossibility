/-
  ScoreRegularityExchangeable.lean — M3 of the score-regularity frontier
  campaign: the M-sample exchangeable-Gaussian discharge of
  `CramerRaoScoreProperty`, wired end-to-end into `dash_mvue`.

  CONTEXT. `ParetoGlobal.lean` isolates the analytic core of the global
  Cramér–Rao step as the named hypothesis `CramerRaoScoreProperty` (Sard
  pattern) and, before this campaign, discharged it only for the single-sample
  Gaussian location model with the singleton class {id}. M1
  (`ScoreRegularity.lean`) supplied differentiation under the integral for
  exponential tilts; M2 (`ScoreRegularityNEF.lean`) turned it into
  `cramerRaoScoreProperty_tilted`, discharging the property for EVERY natural
  exponential family through the tilt `μ_t = μ.tilted (t·X·)` with the regular
  unbiased class `unbiasedNEF X μ`. This file (M3) instantiates M2 at the
  genuine M-sample model of `dash_mvue`.

  THE MODEL (SCORE_REGULARITY_SCOPING.md §3.2, decomposition B; O6).
  `μ = multivariateGaussian (θ₀ • 𝟙) Σ_exch` on `EuclideanSpace ℝ (Fin n)`,
  with the exchangeable covariance
  `Σ_exch = σ²((1−ρ)·1 + ρ·J)`, i.e. `Σ_exch i i = σ²`, `Σ_exch i j = ρσ²`
  for `i ≠ j`, `0 ≤ ρ ≤ 1`, `σ > 0`. Because `Σ_exch · 𝟙 = σ²(1−ρ+ρn)·𝟙`, the
  score direction `Σ_exch⁻¹𝟙` is the EXPLICIT multiple `c·𝟙`,
  `c = 1/(σ²(1−ρ+ρn))` — no matrix inversion. The sufficient statistic is the
  continuous linear functional `X(x) = c·∑ᵢ xᵢ`.

  WHAT THIS FILE PROVES.
  * (a) `exchCov_posSemidef` — `Σ_exch.PosSemidef` via the direct quadratic
    form `σ²((1−ρ)Σxᵢ² + ρ(Σxᵢ)²) ≥ 0`; and `isExchangeableEnsemble_coords`
    — the coordinate ensemble `x ↦ xᵢ` is `IsExchangeableEnsemble` via
    `variance_eval_multivariateGaussian` / `covariance_eval_multivariateGaussian`.
  * (b) `variance_statFun` — `Var[X; μ] = exchFisherInfo n σ ρ`, reusing the
    repo's Bienaymé theorem `variance_weighted_sum_exchangeable`; and
    `integrableExpSet_statFun` — `integrableExpSet X μ = ℝ` (a continuous linear
    functional of a Gaussian pushes forward to `gaussianReal`, which has all
    exponential moments). Hence `cramerRaoScoreProperty_exchangeableGaussian`:
    `CramerRaoScoreProperty μ (unbiasedNEF X μ) (exchFisherInfo n σ ρ)` by M2.
  * (c) `dashAggregate_mem_unbiasedNEF` — the DASH equal-weight aggregate
    `X̄(x) = ∑ᵢ (1/n)·xᵢ` provably belongs to `unbiasedNEF X μ`: it is the scalar
    multiple `(1/I)·X`, its exponential integrability is inherited from `X`, and
    unit-rate unbiasedness `E_{μ_t}[X̄] = θ₀ + t` follows from the tilted-mean
    identity `E_{μ_t}[X] = deriv (cgf X μ) t = I·θ₀ + I·t`, where `cgf X μ` is
    computed by identifying the pushforward `μ.map X = gaussianReal (I·θ₀) I`.

  THE END-TO-END THEOREM. `dash_mvue_exchangeableGaussian` combines the score
  property (b) with the membership (c) through `dash_mvue`: for the genuine
  M-sample exchangeable Gaussian model, the DASH aggregate is (i) a regular
  unbiased estimator of the natural parameter, (ii) attains the Cramér–Rao
  bound `Var[X̄; μ] = 1/exchFisherInfo`, and (iii) has variance no larger than
  that of EVERY square-integrable estimator in the regular unbiased class —
  arbitrary nonlinear, Bayesian or adversarial estimators included. This
  discharges the `CramerRaoScoreProperty` hypothesis of `dash_mvue`/
  `dash_global_min_variance` for the concrete model that the monograph reported
  as argued-but-unformalized.

  This upgrades the ρ-side hypotheses of the concrete model to `0 ≤ ρ ≤ 1`
  (the range on which `Σ_exch` is positive semidefinite): `ρ ≤ 1` is needed
  only for `Σ_exch.PosSemidef`, exactly as one expects for a genuine covariance.

  Zero new axioms. Zero sorry. Every declaration here depends only on Lean-core
  axioms (`propext`, `Classical.choice`, `Quot.sound`) — it stays entirely in
  the Tier-A spine (it never touches the GBDT layer).
-/

import UniversalImpossibility.ScoreRegularityNEF
import UniversalImpossibility.ParetoGlobal
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Moments.Tilted

set_option autoImplicit false

open MeasureTheory ProbabilityTheory Real Filter Matrix
open scoped Topology NNReal BigOperators

namespace UniversalImpossibility.ScoreRegularityExchangeable

open UniversalImpossibility.ParetoGlobal UniversalImpossibility.GaussMarkovDASH
open UniversalImpossibility.ScoreRegularityNEF

/-! ## §1 The exchangeable Gaussian model and its sufficient statistic -/

/-- The exchangeable covariance matrix `Σ_exch = σ²((1−ρ)·1 + ρ·J)`:
    `σ²` on the diagonal, `ρσ²` off the diagonal. -/
noncomputable def exchCov (n : ℕ) (σ ρ : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => if i = j then σ ^ 2 else ρ * σ ^ 2

/-- The common mean vector `θ₀ · 𝟙 ∈ EuclideanSpace ℝ (Fin n)`. -/
noncomputable def meanVec (n : ℕ) (θ₀ : ℝ) : EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 (fun _ => θ₀)

/-- The explicit score-normalisation constant `c = 1/(σ²(1−ρ+ρn))`, i.e. the
    multiple with `Σ_exch⁻¹𝟙 = c·𝟙`. -/
noncomputable def statCoef (n : ℕ) (σ ρ : ℝ) : ℝ := 1 / (σ ^ 2 * (1 - ρ + ρ * n))

/-- The sufficient statistic `X(x) = c·∑ᵢ xᵢ`, packaged as a continuous linear
    functional (a `StrongDual`) so its Gaussian pushforward is available. -/
noncomputable def statDual (n : ℕ) (σ ρ : ℝ) : StrongDual ℝ (EuclideanSpace ℝ (Fin n)) :=
  (statCoef n σ ρ) • ∑ i, EuclideanSpace.proj i

/-- The sufficient statistic `X(x) = c·∑ᵢ xᵢ` as a plain function. -/
noncomputable def statFun (n : ℕ) (σ ρ : ℝ) : EuclideanSpace ℝ (Fin n) → ℝ :=
  fun x => (statCoef n σ ρ) * ∑ i, x i

/-- The `StrongDual` packaging and the plain function agree. -/
lemma statDual_coe (n : ℕ) (σ ρ : ℝ) : ⇑(statDual n σ ρ) = statFun n σ ρ := by
  funext x; simp [statDual, statFun, Finset.mul_sum]

/-! ## §2 Sub-milestone (a): positive semidefiniteness and ensemble structure -/

/-- **`Σ_exch` is positive semidefinite** for `0 ≤ ρ ≤ 1`. Proved by the direct
    quadratic-form identity
    `xᵀ Σ_exch x = σ²((1−ρ)·∑xᵢ² + ρ·(∑xᵢ)²) ≥ 0`. -/
theorem exchCov_posSemidef (n : ℕ) (σ ρ : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    (exchCov n σ ρ).PosSemidef := by
  rw [Matrix.posSemidef_iff_dotProduct_mulVec]
  refine ⟨?_, ?_⟩
  · ext i j
    simp only [exchCov, Matrix.conjTranspose_apply, Matrix.of_apply, star_trivial]
    by_cases h : i = j
    · simp [h]
    · simp [h, eq_comm]
  · intro x
    have key : ∑ i, x i * (exchCov n σ ρ *ᵥ x) i
        = σ ^ 2 * ((1 - ρ) * (∑ i, (x i) ^ 2) + ρ * (∑ i, x i) ^ 2) := by
      simp only [exchCov, Matrix.mulVec, Matrix.of_apply, dotProduct]
      have hsplit : ∀ i : Fin n, ∑ j, (if i = j then σ ^ 2 else ρ * σ ^ 2) * x j
          = ρ * σ ^ 2 * (∑ j, x j) + (1 - ρ) * σ ^ 2 * x i := by
        intro i
        have hpt : ∀ j, (if i = j then σ ^ 2 else ρ * σ ^ 2) * x j
            = ρ * σ ^ 2 * x j + (if i = j then (1 - ρ) * σ ^ 2 * x j else 0) := by
          intro j; by_cases h : i = j
          · simp only [h, if_true]; ring
          · simp [h]
        simp_rw [hpt, Finset.sum_add_distrib, ← Finset.mul_sum, Fintype.sum_ite_eq]
      simp_rw [hsplit]
      rw [Finset.sum_congr rfl (fun i _ =>
        show x i * (ρ * σ ^ 2 * (∑ j, x j) + (1 - ρ) * σ ^ 2 * x i)
          = (ρ * σ ^ 2 * (∑ j, x j)) * x i + ((1 - ρ) * σ ^ 2) * (x i) ^ 2 by ring)]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      ring
    rw [show (star x ⬝ᵥ exchCov n σ ρ *ᵥ x) = ∑ i, x i * (exchCov n σ ρ *ᵥ x) i by
      simp [dotProduct, star_trivial], key]
    have h1 : (0:ℝ) ≤ σ ^ 2 := sq_nonneg σ
    have h2 : (0:ℝ) ≤ 1 - ρ := by linarith
    positivity

/-- **The coordinate ensemble is exchangeable.** The evaluation maps
    `x ↦ xᵢ` under `multivariateGaussian (θ₀•𝟙) Σ_exch` form an
    `IsExchangeableEnsemble` with variance `σ²` and pairwise covariance `ρσ²`,
    from the covariance-matrix reading of `Σ_exch`. -/
theorem isExchangeableEnsemble_coords (n : ℕ) (σ ρ θ₀ : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    IsExchangeableEnsemble (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      (fun i => fun x => x i) σ ρ := by
  refine ⟨?_, ?_, ?_⟩
  · intro i
    have hmp := measurePreserving_eval_multivariateGaussian (μ := meanVec n θ₀)
      (S := exchCov n σ ρ) (exchCov_posSemidef n σ ρ hρ0 hρ1) (i := i)
    have hm := (memLp_id_gaussianReal (μ := (meanVec n θ₀) i)
      (v := (exchCov n σ ρ i i).toNNReal) 2)
    rw [← hmp.map_eq] at hm
    exact (memLp_map_measure_iff (by fun_prop) (by fun_prop)).mp hm
  · intro i
    rw [variance_eval_multivariateGaussian (exchCov_posSemidef n σ ρ hρ0 hρ1)]
    simp [exchCov]
  · intro i j hij
    rw [covariance_eval_multivariateGaussian (exchCov_posSemidef n σ ρ hρ0 hρ1)]
    simp [exchCov, hij]

/-! ## §3 Sub-milestone (b): Fisher information, mean, and exponential moments -/

/-- `c·n = exchFisherInfo n σ ρ`: the normalisation constant times the sample
    size is the Fisher information. -/
theorem statCoef_mul_cast (n : ℕ) (σ ρ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) :
    statCoef n σ ρ * (n : ℝ) = exchFisherInfo n σ ρ := by
  have hd : (0:ℝ) < 1 - ρ + ρ * n := exch_denom_pos hn hρ0
  have hD : σ ^ 2 * (1 - ρ + ρ * (n:ℝ)) ≠ 0 := ne_of_gt (mul_pos (by positivity) hd)
  rw [statCoef, exchFisherInfo]; field_simp

/-- **Sub-milestone (b): `Var[X; μ] = exchFisherInfo n σ ρ`.** The statistic
    `X = ∑ᵢ c·xᵢ` is a weighted aggregate of the exchangeable coordinate
    ensemble, so its variance is the repo's Bienaymé form
    `exchVar σ ρ (fun _ => c)`, which the explicit `c` collapses to the Fisher
    information `n/(σ²(1−ρ+ρn))`. -/
theorem variance_statFun (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    (hρ1 : ρ ≤ 1) :
    Var[statFun n σ ρ; multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)]
      = exchFisherInfo n σ ρ := by
  have hfun : statFun n σ ρ = fun x => ∑ i, (statCoef n σ ρ) * x i := by
    funext x; simp [statFun, Finset.mul_sum]
  rw [hfun, variance_weighted_sum_exchangeable (isExchangeableEnsemble_coords n σ ρ θ₀ hρ0 hρ1)
    (fun _ => statCoef n σ ρ), exchVar]
  have hd : (0:ℝ) < 1 - ρ + ρ * n := exch_denom_pos hn hρ0
  have hD : σ ^ 2 * (1 - ρ + ρ * (n:ℝ)) ≠ 0 := ne_of_gt (mul_pos (by positivity) hd)
  have hc1 : statCoef n σ ρ * (σ ^ 2 * (1 - ρ + ρ * (n:ℝ))) = 1 := by
    rw [statCoef, one_div, inv_mul_cancel₀ hD]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [exchFisherInfo, eq_div_iff hD]
  linear_combination ((n:ℝ) * (statCoef n σ ρ * (σ ^ 2 * (1 - ρ + ρ * (n:ℝ))) + 1)) * hc1

/-- The mean of the sufficient statistic: `E_μ[X] = I·θ₀` (with
    `I = exchFisherInfo`). A continuous linear functional commutes with the
    Gaussian mean `∫ x dμ = θ₀·𝟙`. -/
theorem integral_statFun (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) :
    ∫ x, statFun n σ ρ x ∂(multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      = exchFisherInfo n σ ρ * θ₀ := by
  have h1 : ∫ x, statFun n σ ρ x ∂(multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      = (statDual n σ ρ) (∫ x, x ∂(multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))) := by
    rw [← statDual_coe n σ ρ]; exact IsGaussian.integral_dual (statDual n σ ρ)
  rw [h1, integral_id_multivariateGaussian, congrFun (statDual_coe n σ ρ) (meanVec n θ₀)]
  simp only [statFun, meanVec, PiLp.toLp_apply, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  rw [← statCoef_mul_cast n σ ρ hn hσ hρ0]; ring

/-- The pushforward of `μ` under the sufficient statistic is the mean-shifted
    real Gaussian `gaussianReal (I·θ₀) I` (`I = exchFisherInfo`). This is the
    tilt-route substitute for the (absent) multivariate Lebesgue density. -/
theorem map_statFun (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)).map (statFun n σ ρ)
      = gaussianReal (exchFisherInfo n σ ρ * θ₀) (exchFisherInfo n σ ρ).toNNReal := by
  have hmapL := IsGaussian.map_eq_gaussianReal
    (μ := multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)) (statDual n σ ρ)
  rw [statDual_coe n σ ρ, integral_statFun n σ ρ θ₀ hn hσ hρ0,
    variance_statFun n σ ρ θ₀ hn hσ hρ0 hρ1] at hmapL
  exact hmapL

/-- **The sufficient statistic has all exponential moments**: `integrableExpSet
    X μ = ℝ`, since `μ.map X` is a real Gaussian. -/
theorem integrableExpSet_statFun (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    (hρ1 : ρ ≤ 1) :
    integrableExpSet (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      = Set.univ := by
  ext t
  simp only [Set.mem_univ, iff_true, integrableExpSet, Set.mem_setOf_eq]
  have hf : AEMeasurable (statFun n σ ρ)
      (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)) := by
    rw [← statDual_coe n σ ρ]; exact (statDual n σ ρ).continuous.aemeasurable
  rw [show (fun x => exp (t * statFun n σ ρ x)) = (fun y => exp (t * y)) ∘ (statFun n σ ρ) from rfl]
  rw [← integrable_map_measure (by fun_prop) hf, map_statFun n σ ρ θ₀ hn hσ hρ0 hρ1]
  exact integrable_exp_mul_gaussianReal t

/-- `0` — and indeed every `t` — lies in the interior of the exponential-moment
    set of `X`. -/
theorem mem_interior_integrableExpSet (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    (hρ1 : ρ ≤ 1) (t : ℝ) :
    t ∈ interior (integrableExpSet (statFun n σ ρ)
      (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))) := by
  rw [integrableExpSet_statFun n σ ρ θ₀ hn hσ hρ0 hρ1, interior_univ]; exact Set.mem_univ t

/-! ## §4 Sub-milestone (c): the tilted mean and DASH-aggregate membership -/

/-- **The tilted-mean identity.** `E_{μ_t}[X] = deriv (cgf X μ) t = I·θ₀ + I·t`,
    computed from `cgf X μ s = (I·θ₀)·s + I·s²/2` (the cgf of the mean-shifted
    Gaussian `μ.map X`). This is the exact-rate statement making the DASH
    aggregate an unbiased estimator of `θ` under the tilt. -/
theorem tilted_mean_statFun (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    (hρ1 : ρ ≤ 1) (t : ℝ) :
    ∫ x, statFun n σ ρ x ∂((multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)).tilted
        (fun x => t * statFun n σ ρ x))
      = exchFisherInfo n σ ρ * θ₀ + exchFisherInfo n σ ρ * t := by
  set I := exchFisherInfo n σ ρ
  have hIpos : 0 < I := exchFisherInfo_pos hn hσ hρ0
  have hcgf : cgf (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      = fun s => (I * θ₀) * s + I * s ^ 2 / 2 := by
    funext s
    rw [cgf_gaussianReal (map_statFun n σ ρ θ₀ hn hσ hρ0 hρ1) s, Real.coe_toNNReal I hIpos.le]
  have hderiv : deriv (cgf (statFun n σ ρ)
      (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))) t = I * θ₀ + I * t := by
    rw [hcgf]
    have h1 : HasDerivAt (fun s : ℝ => (I * θ₀) * s) (I * θ₀) t := by
      simpa using (hasDerivAt_id t).const_mul (I * θ₀)
    have h2 : HasDerivAt (fun s : ℝ => I * s ^ 2 / 2) (I * t) t := by
      have := ((hasDerivAt_pow 2 t).const_mul I).div_const 2
      convert this using 1; push_cast; ring
    exact (h1.add h2).deriv
  rw [integral_tilted_mul_self (mem_interior_integrableExpSet n σ ρ θ₀ hn hσ hρ0 hρ1 t), hderiv]

/-- **Sub-milestone (c): the DASH aggregate belongs to the regular unbiased
    class.** The equal-weight mean `X̄(x) = ∑ᵢ (1/n)·xᵢ` is the scalar multiple
    `(1/I)·X`; it is square-integrable, its `X̄·exp(tX)` and `X̄·X·exp(tX)` are
    integrable near `0`, and it is unbiased at unit rate:
    `E_{μ_t}[X̄] = θ₀ + t = E_μ[X̄] + t`. Hence `X̄ ∈ unbiasedNEF X μ`. -/
theorem dashAggregate_mem_unbiasedNEF (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    (fun x : EuclideanSpace ℝ (Fin n) => ∑ i, (1 / (n:ℝ)) * x i)
      ∈ unbiasedNEF (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)) := by
  set μ := multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)
  set I := exchFisherInfo n σ ρ
  have hIpos : 0 < I := exchFisherInfo_pos hn hσ hρ0
  have hIne : I ≠ 0 := ne_of_gt hIpos
  have hn0 : (n:ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hcoef : (1:ℝ)/n = (1/I) * statCoef n σ ρ := by
    rw [mul_comm, mul_one_div, eq_comm, div_eq_div_iff hIne hn0, one_mul]
    exact statCoef_mul_cast n σ ρ hn hσ hρ0
  have hbar : ∀ x : EuclideanSpace ℝ (Fin n),
      (∑ i, (1 / (n:ℝ)) * x i) = (1 / I) * statFun n σ ρ x := by
    intro x; rw [statFun, ← Finset.mul_sum, ← mul_assoc, ← hcoef]
  have hmemX : MemLp (statFun n σ ρ) 2 μ :=
    memLp_of_mem_interior_integrableExpSet (by
      rw [integrableExpSet_statFun n σ ρ θ₀ hn hσ hρ0 hρ1, interior_univ]
      exact Set.mem_univ 0) 2
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hfe : (fun x : EuclideanSpace ℝ (Fin n) => ∑ i, (1 / (n:ℝ)) * x i)
        = fun x => (1 / I) * statFun n σ ρ x := funext hbar
    rw [hfe]; exact hmemX.const_mul (1 / I)
  · refine Eventually.of_forall fun t => ?_
    have hX1 := integrable_pow_mul_exp_of_mem_interior_integrableExpSet
      (mem_interior_integrableExpSet n σ ρ θ₀ hn hσ hρ0 hρ1 t) 1
    refine (hX1.const_mul (1 / I)).congr (ae_of_all _ fun x => ?_)
    simp only [hbar, pow_one]; ring
  · refine Eventually.of_forall fun t => ?_
    have hX2 := integrable_pow_mul_exp_of_mem_interior_integrableExpSet
      (mem_interior_integrableExpSet n σ ρ θ₀ hn hσ hρ0 hρ1 t) 2
    refine (hX2.const_mul (1 / I)).congr (ae_of_all _ fun x => ?_)
    simp only [hbar]; ring
  · refine Eventually.of_forall fun t => ?_
    have hintdmu : ∫ x, (∑ i, (1 / (n:ℝ)) * x i) ∂μ = θ₀ := by
      simp_rw [hbar]
      rw [integral_const_mul, integral_statFun n σ ρ θ₀ hn hσ hρ0, one_div,
        inv_mul_cancel_left₀ hIne]
    have htint : ∫ x, (∑ i, (1 / (n:ℝ)) * x i) ∂(μ.tilted (fun x => t * statFun n σ ρ x))
        = θ₀ + t := by
      simp_rw [hbar]
      rw [integral_const_mul, tilted_mean_statFun n σ ρ θ₀ hn hσ hρ0 hρ1 t, ← mul_add, one_div,
        inv_mul_cancel_left₀ hIne]
    rw [htint, hintdmu]

/-! ## §5 The score-property instance and the end-to-end MVUE theorem -/

/-- **The Cramér–Rao score property for the M-sample exchangeable Gaussian
    model (O6).** Instantiating M2's `cramerRaoScoreProperty_tilted` at the
    sufficient statistic `X = c·∑ᵢ xᵢ`: for `0 < σ`, `0 ≤ ρ ≤ 1`, `n ≥ 1`,
    `CramerRaoScoreProperty μ (unbiasedNEF X μ) (exchFisherInfo n σ ρ)` holds
    for the genuine joint measure `μ = multivariateGaussian (θ₀•𝟙) Σ_exch` —
    the discharge that `ParetoGlobal.lean` reported as the open frontier item.
    Depends only on Lean-core axioms. -/
theorem cramerRaoScoreProperty_exchangeableGaussian (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    CramerRaoScoreProperty (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
      (unbiasedNEF (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)))
      (exchFisherInfo n σ ρ) := by
  refine cramerRaoScoreProperty_tilted ?_ (variance_statFun n σ ρ θ₀ hn hσ hρ0 hρ1)
  rw [integrableExpSet_statFun n σ ρ θ₀ hn hσ hρ0 hρ1, interior_univ]; exact Set.mem_univ 0

/-- **DASH is the minimum-variance unbiased estimator for the exchangeable
    Gaussian ensemble (M3 capstone).** For `μ = multivariateGaussian (θ₀•𝟙)
    Σ_exch` with `0 < σ`, `0 ≤ ρ ≤ 1`, `n ≥ 1`, the DASH equal-weight aggregate
    `X̄(x) = ∑ᵢ (1/n)·xᵢ`:

    1. is a regular unbiased estimator of the natural parameter
       (`X̄ ∈ unbiasedNEF X μ`);
    2. attains the Cramér–Rao bound: `Var[X̄; μ] = 1/exchFisherInfo n σ ρ`;
    3. has variance no larger than that of EVERY square-integrable estimator in
       the regular unbiased class — arbitrary nonlinear/Bayesian/adversarial
       estimators included.

    This wires the concrete-model score property into `dash_mvue`, discharging
    end-to-end the `CramerRaoScoreProperty` hypothesis that
    `dash_mvue`/`dash_global_min_variance` carried as the campaign's target.
    Depends only on Lean-core axioms (it stays entirely in the Tier-A spine and
    never touches the GBDT layer). -/
theorem dash_mvue_exchangeableGaussian (n : ℕ) (σ ρ θ₀ : ℝ) (hn : 0 < n) (hσ : 0 < σ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    (fun x : EuclideanSpace ℝ (Fin n) => ∑ i, (1 / (n:ℝ)) * x i)
        ∈ unbiasedNEF (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ))
    ∧ Var[fun x : EuclideanSpace ℝ (Fin n) => ∑ i, (1 / (n:ℝ)) * x i;
        multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)] = 1 / exchFisherInfo n σ ρ
    ∧ ∀ T ∈ unbiasedNEF (statFun n σ ρ) (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)),
        MemLp T 2 (multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)) →
        Var[fun x : EuclideanSpace ℝ (Fin n) => ∑ i, (1 / (n:ℝ)) * x i;
            multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)]
          ≤ Var[T; multivariateGaussian (meanVec n θ₀) (exchCov n σ ρ)] := by
  have hmvue := dash_mvue hn hσ hρ0 (isExchangeableEnsemble_coords n σ ρ θ₀ hρ0 hρ1)
    (cramerRaoScoreProperty_exchangeableGaussian n σ ρ θ₀ hn hσ hρ0 hρ1)
  exact ⟨dashAggregate_mem_unbiasedNEF n σ ρ θ₀ hn hσ hρ0 hρ1, hmvue.1, hmvue.2⟩

end UniversalImpossibility.ScoreRegularityExchangeable
