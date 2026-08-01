/-
  ParetoGlobal.lean — Global Pareto optimality of DASH: reducing the Cramér–Rao step
  to one named hypothesis, with the entire surrounding argument machine-checked.

  THE GAP THIS FILE CLOSES (honestly). The monograph states that DASH's *global*
  Pareto optimality "remains argued, resting on an unformalized Cramér–Rao step"
  (proof-status inventory). Concretely, the Lean state before this file was:

  - WITHIN-GROUP pairs: fully machine-checked (`dash_within_group_dominance` in
    ParetoOptimality.lean) — any committed ranking has disagreement exactly 1/2,
    ties have 0.
  - BETWEEN-GROUP pairs: machine-checked only for the LINEAR unbiased aggregator
    class (`dash_min_variance` in GaussMarkovDASH.lean, a Gauss–Markov statement).
    The extension to ALL unbiased estimators — the Cramér–Rao step — was argued
    in prose only.

  WHAT THIS FILE DOES. Following the repo's established Sard pattern
  (`SardProperty` in UbiquityDimensional.lean), the analytic core that Mathlib
  cannot currently express (score functions, Fisher information regularity,
  differentiation under the integral sign) is isolated as ONE named hypothesis,
  `CramerRaoScoreProperty`, and everything else is machine-checked:

  1. PROVED OUTRIGHT (pure Mathlib, no repo axioms, no named hypothesis):
     - `covariance_sq_le_variance_mul_variance` — the Cauchy–Schwarz inequality
       for covariance, cov(T,S)² ≤ Var(T)·Var(S). (Not in Mathlib at this pin;
       proved here via the quadratic-discriminant argument.)
     - `abstract_cramer_rao` — the information inequality: if cov(T,S) = 1 and
       Var(S) > 0 then Var(T) ≥ 1/Var(S). This is the entire *inequality*
       content of Cramér–Rao; what it does not supply is the existence of a
       score S with Var(S) = Fisher information and unit covariance against
       every unbiased estimator — that is regularity, not inequality.
     - `variance_weighted_sum_exchangeable` — Bienaymé for exchangeable
       ensembles: the abstract quadratic form `exchVar` of GaussMarkovDASH.lean
       is the GENUINE variance of the weighted aggregate of random variables
       with the exchangeable second-moment structure. (This upgrades the
       "closed form taken as definition" caveat of GaussMarkovDASH.lean to a
       theorem.)
     - `dash_variance_eq_inv_fisher` — the DASH aggregate variance
       σ²((1−ρ)/n + ρ) is EXACTLY the reciprocal of the exchangeable-model
       Fisher information n/(σ²(1−ρ+ρn)). So conditional on the score
       hypothesis, DASH does not merely satisfy the Cramér–Rao bound — it
       attains it (MVUE).
     - `cramerRaoScoreProperty_gaussianReal` — the named hypothesis is
       DISCHARGED outright for the single-sample Gaussian location model
       N(m, v): the classical score S(x) = (x−m)/v is exhibited, its mean-zero,
       variance-1/v (= Fisher information), and unit-covariance properties are
       proved from Mathlib's `gaussianReal` API. This parallels
       `sardProperty_of_continuousLinearMap`: the hypothesis is exactly
       classical content, satisfiable, and not vacuous.

  2. NAMED HYPOTHESIS (the honest residual, Sard pattern):
     - `CramerRaoScoreProperty μ unbiased I` — "there is a score S ∈ L² with
       E[S] = 0, Var(S) = I, and cov(T, S) = 1 for every T in the unbiased
       class." For the M-model exchangeable Gaussian ensemble this is the
       standard regularity computation (score of the multivariate Gaussian
       location family + differentiation under the integral sign); it needs
       Gaussian-family calculus not available in Mathlib at this pin. The
       single-sample case IS proved here (see above), which pins down the
       hypothesis's mathematical meaning exactly.

  3. MACHINE-CHECKED REDUCTION (conditional only on the named hypothesis):
     - `cramer_rao_bound_of_score` — score property ⟹ Var(T) ≥ 1/I for every
       unbiased T.
     - `dash_global_min_variance` — score property at I = exchFisherInfo ⟹
       DASH's variance ≤ Var(T) for EVERY unbiased estimator T, not just
       linear aggregators. This is the Cramér–Rao step.
     - `dash_mvue` — DASH attains the bound: it is minimum-variance unbiased.
     - `dash_global_pareto_optimal` — the two-coordinate global statement:
       against ANY competing method (a within-group ranking plus an unbiased
       between-group estimator), DASH is coordinatewise no worse, and strictly
       better within-group whenever the competitor commits.
     - `no_unbiased_method_dominates_dash` — hence no faithful method strictly
       Pareto-dominates DASH.

  AXIOM STRATIFICATION. Sections 1–2 and the reductions through `dash_mvue`
  are axiom-clean (propext, Classical.choice, Quot.sound only). The two
  combined theorems (`dash_global_pareto_optimal`,
  `no_unbiased_method_dominates_dash`) DELIBERATELY build on the quantitative
  GBDT layer for their within-group coordinate, hence additionally depend on
  the two bundled axioms `gbdtWorld`/`gbdtAxioms` — the same footing as
  `dash_within_group_dominance`, which they extend.

  A SMALL HONEST FINDING. The Cramér–Rao side needs only 0 ≤ ρ (with n ≥ 1),
  not ρ < 1: the linear Gauss–Markov theorem `dash_min_variance` requires
  ρ < 1 for the strictness of its Cauchy–Schwarz step, but the information
  bound and the attainment identity do not. The global theorems therefore
  carry ρ < 1 only where the within-group/GBDT layer needs its hypotheses.

  WHAT REMAINS UNFORMALIZED (and why). Discharging `CramerRaoScoreProperty`
  for the M-sample exchangeable Gaussian model requires: the product/joint
  Gaussian measure with correlation structure, its score (a linear functional
  of the observation), and differentiation under the integral sign for the
  θ-family to obtain cov(T, S) = 1 from unbiasedness. Mathlib at this pin has
  none of Fisher information, score calculus, or a Cramér–Rao development;
  building them is a Mathlib-scale project, not a repo-scale one. The
  single-sample Gaussian discharge proved here is the honest certificate that
  the named hypothesis says exactly what classical statistics says.

  Zero new axioms. Zero sorry.
-/

import UniversalImpossibility.ParetoOptimality
import UniversalImpossibility.GaussMarkovDASH
import Mathlib.Probability.Moments.Covariance
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Algebra.QuadraticDiscriminant

set_option autoImplicit false

open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace UniversalImpossibility.ParetoGlobal

/-! ## §1 The Cauchy–Schwarz core of Cramér–Rao (proved outright) -/

/-- **Cauchy–Schwarz for covariance.** For square-integrable `T`, `S` over a
    finite measure, `cov(T,S)² ≤ Var(T)·Var(S)`. Proved by the classical
    discriminant argument: `t ↦ Var(T + t·S)` is a nonnegative quadratic in `t`,
    so its discriminant is nonpositive. -/
theorem covariance_sq_le_variance_mul_variance
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {T S : Ω → ℝ} (hT : MemLp T 2 μ) (hS : MemLp S 2 μ) :
    cov[T, S; μ] ^ 2 ≤ Var[T; μ] * Var[S; μ] := by
  have key : ∀ t : ℝ,
      0 ≤ Var[S; μ] * (t * t) + (2 * cov[T, S; μ]) * t + Var[T; μ] := by
    intro t
    have h0 : 0 ≤ Var[T + t • S; μ] := variance_nonneg _ μ
    rw [variance_add hT (hS.const_smul t), covariance_smul_right,
      variance_smul] at h0
    nlinarith [h0]
  have hd := discrim_le_zero key
  simp only [discrim] at hd
  nlinarith [hd]

/-- **The abstract Cramér–Rao inequality.** If an estimator `T` has unit
    covariance with a "score" `S` of positive variance, then
    `Var(T) ≥ 1 / Var(S)`. This is the entire inequality content of the
    Cramér–Rao bound; the statistical content (that such a score exists with
    `Var(S) = ` Fisher information and unit covariance against every unbiased
    estimator) is isolated as `CramerRaoScoreProperty` below. -/
theorem abstract_cramer_rao
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {T S : Ω → ℝ} (hT : MemLp T 2 μ) (hS : MemLp S 2 μ)
    (hSvar : 0 < Var[S; μ]) (hcov : cov[T, S; μ] = 1) :
    1 / Var[S; μ] ≤ Var[T; μ] := by
  have h := covariance_sq_le_variance_mul_variance hT hS
  rw [hcov, one_pow] at h
  rw [div_le_iff₀ hSvar]
  linarith [h]

/-! ## §2 The named hypothesis (Sard pattern) and its machine-checked reduction -/

/-- **The Cramér–Rao score property** — the analytic core of the Cramér–Rao
    step, isolated as a single named hypothesis in the style of `SardProperty`.

    `CramerRaoScoreProperty μ unbiased I` says: the statistical experiment
    carried by `μ` admits a *score* — a square-integrable, mean-zero random
    variable `S` whose variance is the Fisher information `I` and whose
    covariance with every (square-integrable) estimator in the unbiased class
    is `1`. The last condition is exactly what differentiation under the
    integral sign yields from unbiasedness (`d/dθ E_θ[T] = cov(T, S) = 1`);
    the first two are the standard score identities `E[S] = 0`,
    `Var(S) = I(θ)`. NOT proved here in general (Mathlib has no Fisher
    information or score calculus at this pin), but proved outright for the
    single-sample Gaussian location model below
    (`cramerRaoScoreProperty_gaussianReal`). -/
def CramerRaoScoreProperty
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (unbiased : Set (Ω → ℝ)) (I : ℝ) : Prop :=
  ∃ S : Ω → ℝ, MemLp S 2 μ ∧ (∫ ω, S ω ∂μ) = 0 ∧ Var[S; μ] = I ∧
    ∀ T ∈ unbiased, MemLp T 2 μ → cov[T, S; μ] = 1

/-- **The Cramér–Rao bound, conditional on the score property.** Every
    square-integrable estimator in the unbiased class has variance at least
    `1/I`. The reduction from the named hypothesis to the bound is entirely
    machine-checked. -/
theorem cramer_rao_bound_of_score
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {unbiased : Set (Ω → ℝ)} {I : ℝ}
    (hscore : CramerRaoScoreProperty μ unbiased I) (hI : 0 < I)
    {T : Ω → ℝ} (hTu : T ∈ unbiased) (hT : MemLp T 2 μ) :
    1 / I ≤ Var[T; μ] := by
  obtain ⟨S, hS2, _hS0, hSvar, hScov⟩ := hscore
  have hpos : 0 < Var[S; μ] := by rw [hSvar]; exact hI
  have h := abstract_cramer_rao hT hS2 hpos (hScov T hTu hT)
  rwa [hSvar] at h

/-! ## §3 Discharging the score property: the Gaussian location model

  Parallel to `sardProperty_of_continuousLinearMap` (which discharged
  `SardProperty` outright for linear maps), we discharge
  `CramerRaoScoreProperty` outright for the single-sample Gaussian location
  model `N(m, v)`: the classical score `S(x) = (x − m)/v` is exhibited and all
  four defining conditions are proved from Mathlib's `gaussianReal` API. The
  Fisher information is `1/v`, and the identity estimator attains the bound —
  the classical statement that the sample is efficient for a Gaussian mean. -/

section GaussianDischarge

/-- **The score property holds outright for the Gaussian location model.**
    For `N(m, v)` with `v ≠ 0`, the score `S(x) = (x − m)/v` witnesses
    `CramerRaoScoreProperty` for the class containing the identity estimator,
    with Fisher information `I = 1/v`. Every condition — square-integrability,
    mean zero, `Var(S) = 1/v`, and `cov(id, S) = 1` — is machine-checked. -/
theorem cramerRaoScoreProperty_gaussianReal (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    CramerRaoScoreProperty (gaussianReal m v)
      ({fun x : ℝ => x} : Set (ℝ → ℝ)) (1 / (v : ℝ)) := by
  have hvpos : 0 < v := pos_iff_ne_zero.mpr hv
  have hv0 : (0 : ℝ) < (v : ℝ) := NNReal.coe_pos.mpr hvpos
  have hvne : (v : ℝ) ≠ 0 := ne_of_gt hv0
  -- The identity estimator is square-integrable and integrable
  have hid2 : MemLp (fun x : ℝ => x) 2 (gaussianReal m v) := by
    simpa using memLp_id_gaussianReal (μ := m) (v := v) 2
  have hint : Integrable (fun x : ℝ => x) (gaussianReal m v) :=
    hid2.integrable one_le_two
  -- The centered variable and the score are square-integrable
  have hsub : MemLp (fun x : ℝ => x - m) 2 (gaussianReal m v) :=
    hid2.sub (memLp_const m)
  have hS2 : MemLp (fun x : ℝ => (x - m) / (v : ℝ)) 2 (gaussianReal m v) := by
    simpa [div_eq_inv_mul] using hsub.const_mul ((v : ℝ)⁻¹)
  -- The score has mean zero
  have hS0 : (∫ x, (x - m) / (v : ℝ) ∂gaussianReal m v) = 0 := by
    rw [integral_div, integral_sub hint (integrable_const m)]
    simp [integral_id_gaussianReal]
  -- Var(id) = v, hence Var(id − m) = v
  have hidvar : Var[fun x : ℝ => x; gaussianReal m v] = (v : ℝ) :=
    variance_id_gaussianReal
  have ham : AEStronglyMeasurable (fun x : ℝ => x) (gaussianReal m v) :=
    aestronglyMeasurable_id
  have hsubvar : Var[fun x : ℝ => x - m; gaussianReal m v] = (v : ℝ) := by
    rw [variance_sub_const ham m]
    exact hidvar
  -- Var(score) = 1/v — the Fisher information of the Gaussian location model
  have hSvar : Var[fun x : ℝ => (x - m) / (v : ℝ); gaussianReal m v]
      = 1 / (v : ℝ) := by
    have h1 : (fun x : ℝ => (x - m) / (v : ℝ))
        = fun x : ℝ => ((v : ℝ)⁻¹) * (x - m) := by
      funext x; rw [div_eq_inv_mul]
    rw [h1, variance_const_mul, hsubvar, pow_two, mul_assoc,
      inv_mul_cancel₀ hvne, mul_one, one_div]
  -- cov(id, score) = 1 — the differentiation-under-the-integral identity,
  -- provable outright here because the estimator is the identity
  have hcovid : cov[(fun x : ℝ => x), fun x : ℝ => (x - m) / (v : ℝ);
      gaussianReal m v] = 1 := by
    have h1 : cov[(fun x : ℝ => x), fun x : ℝ => (x - m) / (v : ℝ);
        gaussianReal m v]
        = cov[(fun x : ℝ => x), fun x : ℝ => x - m; gaussianReal m v] / (v : ℝ) :=
      covariance_fun_div_right (v : ℝ)
    have h2 : cov[(fun x : ℝ => x), fun x : ℝ => x - m; gaussianReal m v]
        = cov[(fun x : ℝ => x), (fun x : ℝ => x); gaussianReal m v] :=
      covariance_sub_const_right hint m
    have h3 : cov[(fun x : ℝ => x), (fun x : ℝ => x); gaussianReal m v]
        = (v : ℝ) := by
      rw [covariance_self ham.aemeasurable]
      exact hidvar
    rw [h1, h2, h3]
    exact div_self hvne
  exact ⟨fun x => (x - m) / (v : ℝ), hS2, hS0, hSvar, by
    intro T hTmem _hT2
    rw [Set.mem_singleton_iff] at hTmem
    subst hTmem
    exact hcovid⟩

/-- **The Gaussian Cramér–Rao bound is attained.** The identity estimator's
    variance equals the reciprocal Fisher information `1/(1/v) = v`: the sample
    is an efficient estimator of a Gaussian mean. Together with
    `cramer_rao_bound_of_score` this reproduces the classical statement
    end-to-end for the single-sample model, entirely inside Lean. -/
theorem gaussianReal_cramer_rao_attained (m : ℝ) (v : ℝ≥0) :
    Var[fun x : ℝ => x; gaussianReal m v] = 1 / (1 / (v : ℝ)) := by
  rw [one_div_one_div]
  exact variance_id_gaussianReal

end GaussianDischarge

/-! ## §4 Exchangeable ensembles: `exchVar` is a genuine variance (Bienaymé) -/

/-- An ensemble of `n` estimators is *exchangeable* (in second moments) when
    each member is square-integrable with variance `σ²` and each distinct pair
    has covariance `ρσ²`. This is the moment structure of the DASH ensemble:
    `M` models retrained from i.i.d. seeds on the same data. Stated as a
    hypothesis-definition, like `IsBalanced` and `IsDGPSymmetric`. -/
def IsExchangeableEnsemble
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {n : ℕ}
    (X : Fin n → Ω → ℝ) (σ ρ : ℝ) : Prop :=
  (∀ i, MemLp (X i) 2 μ) ∧
  (∀ i, Var[X i; μ] = σ ^ 2) ∧
  (∀ i j, i ≠ j → cov[X i, X j; μ] = ρ * σ ^ 2)

/-- **Bienaymé for exchangeable ensembles.** The variance of the weighted
    aggregate `∑ᵢ wᵢ Xᵢ` of an exchangeable ensemble equals the abstract
    quadratic form `exchVar σ ρ w` of GaussMarkovDASH.lean. This discharges the
    "closed form taken as the definition" caveat of that file: `exchVar` is the
    genuine `ProbabilityTheory.variance` of the aggregate, so the Gauss–Markov
    optimality proved there is about real variances of real random variables. -/
theorem variance_weighted_sum_exchangeable
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} {X : Fin n → Ω → ℝ} {σ ρ : ℝ}
    (hX : IsExchangeableEnsemble μ X σ ρ) (w : Fin n → ℝ) :
    Var[fun ω => ∑ i, w i * X i ω; μ] = GaussMarkovDASH.exchVar σ ρ w := by
  obtain ⟨hmem, hvar, hcov⟩ := hX
  have hmem' : ∀ i, MemLp (fun ω => w i * X i ω) 2 μ :=
    fun i => (hmem i).const_mul (w i)
  have hbien := variance_fun_sum (μ := μ) (X := fun i ω => w i * X i ω) hmem'
  refine hbien.trans ?_
  -- Each covariance term: wᵢwⱼ·ρσ² off-diagonal, wᵢwⱼ·σ² on the diagonal
  have hterm : ∀ i j : Fin n,
      cov[fun ω => w i * X i ω, fun ω => w j * X j ω; μ]
        = (ρ * σ ^ 2) * (w i * w j)
          + (if i = j then ((1 - ρ) * σ ^ 2) * (w i * w j) else 0) := by
    intro i j
    rw [covariance_const_mul_left, covariance_const_mul_right]
    by_cases hij : i = j
    · subst hij
      rw [covariance_self (hmem i).aestronglyMeasurable.aemeasurable, hvar i,
        if_pos rfl]
      ring
    · rw [hcov i j hij, if_neg hij]
      ring
  have hdiag : ∀ i : Fin n,
      (∑ j, if i = j then ((1 - ρ) * σ ^ 2) * (w i * w j) else 0)
        = ((1 - ρ) * σ ^ 2) * (w i * w i) := by
    intro i
    exact Fintype.sum_ite_eq i (fun j => ((1 - ρ) * σ ^ 2) * (w i * w j))
  calc ∑ i, ∑ j, cov[fun ω => w i * X i ω, fun ω => w j * X j ω; μ]
      = ∑ i, ∑ j, ((ρ * σ ^ 2) * (w i * w j)
          + if i = j then ((1 - ρ) * σ ^ 2) * (w i * w j) else 0) :=
        Finset.sum_congr rfl fun i _ =>
          Finset.sum_congr rfl fun j _ => hterm i j
    _ = (∑ i, ∑ j, (ρ * σ ^ 2) * (w i * w j))
          + ∑ i, ∑ j, (if i = j then ((1 - ρ) * σ ^ 2) * (w i * w j) else 0) := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => Finset.sum_add_distrib
    _ = (ρ * σ ^ 2) * (∑ i, ∑ j, w i * w j)
          + ((1 - ρ) * σ ^ 2) * ∑ i, w i * w i := by
        congr 1
        · rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
        · rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => hdiag i
    _ = GaussMarkovDASH.exchVar σ ρ w := by
        simp only [GaussMarkovDASH.exchVar]
        rw [pow_two (∑ i, w i), Fintype.sum_mul_sum]
        simp_rw [pow_two]
        ring

/-! ## §5 The exchangeable Fisher information and the attainment identity -/

/-- The Fisher information of the exchangeable Gaussian location model: `n`
    observations, each with variance `σ²`, pairwise correlation `ρ`, common
    mean `θ`. The classical computation gives
    `I(θ) = 𝟙ᵀΣ⁻¹𝟙 = n / (σ²(1 − ρ + ρn))`; this definition packages that
    value, and `CramerRaoScoreProperty` instantiated at it asserts exactly the
    classical score regularity for this model. For `ρ = 0` it reduces to the
    i.i.d. value `n/σ²`. -/
noncomputable def exchFisherInfo (n : ℕ) (σ ρ : ℝ) : ℝ :=
  (n : ℝ) / (σ ^ 2 * (1 - ρ + ρ * n))

/-- The denominator `1 − ρ + ρn = 1 + ρ(n−1)` is positive for `ρ ≥ 0`, `n ≥ 1`.
    (Note: `ρ < 1` is NOT needed on the Cramér–Rao side.) -/
theorem exch_denom_pos {n : ℕ} (hn : 0 < n) {ρ : ℝ} (hρ0 : 0 ≤ ρ) :
    0 < 1 - ρ + ρ * n := by
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  nlinarith [mul_nonneg hρ0 (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1)]

/-- The exchangeable Fisher information is positive. -/
theorem exchFisherInfo_pos {n : ℕ} (hn : 0 < n) {σ ρ : ℝ} (hσ : 0 < σ)
    (hρ0 : 0 ≤ ρ) : 0 < exchFisherInfo n σ ρ := by
  have hd := exch_denom_pos hn hρ0
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  exact div_pos hn' (mul_pos (by positivity) hd)

/-- **DASH attains the Cramér–Rao bound: the attainment identity.** The
    equal-weight (DASH) aggregate variance `σ²((1−ρ)/n + ρ)` is EXACTLY the
    reciprocal of the exchangeable Fisher information `n/(σ²(1−ρ+ρn))`. Pure
    algebra, fully proved. This is the reason the Cramér–Rao step closes the
    global optimality claim: conditional on the score property, no unbiased
    estimator can undercut `1/I`, and DASH sits exactly at `1/I`. -/
theorem dash_variance_eq_inv_fisher {n : ℕ} (hn : 0 < n) {σ ρ : ℝ}
    (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) :
    GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ))
      = 1 / exchFisherInfo n σ ρ := by
  rw [GaussMarkovDASH.exchVar_equal hn]
  unfold exchFisherInfo
  have hd : (0 : ℝ) < 1 - ρ + ρ * n := exch_denom_pos hn hρ0
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hσ' : σ ≠ 0 := ne_of_gt hσ
  field_simp

/-! ## §6 The Cramér–Rao step: DASH minimum variance over ALL unbiased methods -/

/-- **The Cramér–Rao step (global DASH minimum variance).** Conditional on the
    score property for the exchangeable model, the DASH equal-weight aggregate
    variance lower-bounds the variance of EVERY square-integrable estimator in
    the unbiased class — arbitrary nonlinear, Bayesian, or adversarial
    estimators included. This extends `dash_min_variance` (GaussMarkovDASH.lean)
    from the linear aggregator class to the full unbiased class, which is
    precisely the step the monograph reported as argued-but-unformalized. The
    reduction is entirely machine-checked; only `CramerRaoScoreProperty` (the
    score regularity of the exchangeable Gaussian model) is assumed. -/
theorem dash_global_min_variance
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} (hn : 0 < n) {σ ρ : ℝ} (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    {unbiased : Set (Ω → ℝ)}
    (hscore : CramerRaoScoreProperty μ unbiased (exchFisherInfo n σ ρ))
    {T : Ω → ℝ} (hTu : T ∈ unbiased) (hT : MemLp T 2 μ) :
    GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) ≤ Var[T; μ] := by
  rw [dash_variance_eq_inv_fisher hn hσ hρ0]
  exact cramer_rao_bound_of_score hscore (exchFisherInfo_pos hn hσ hρ0) hTu hT

/-- **DASH is the minimum-variance unbiased estimator (MVUE), conditional on
    the score property.** For a genuine exchangeable ensemble `X`, the DASH
    aggregate `(1/n)∑ᵢ Xᵢ` (i) has variance exactly `1/I` — it attains the
    Cramér–Rao bound — and (ii) its variance lower-bounds that of every
    square-integrable estimator in the unbiased class. Both parts are
    machine-checked given the one named hypothesis. -/
theorem dash_mvue
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} (hn : 0 < n) {σ ρ : ℝ} (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    {X : Fin n → Ω → ℝ} (hX : IsExchangeableEnsemble μ X σ ρ)
    {unbiased : Set (Ω → ℝ)}
    (hscore : CramerRaoScoreProperty μ unbiased (exchFisherInfo n σ ρ)) :
    Var[fun ω => ∑ i, (1 / (n : ℝ)) * X i ω; μ] = 1 / exchFisherInfo n σ ρ
    ∧ ∀ T ∈ unbiased, MemLp T 2 μ →
        Var[fun ω => ∑ i, (1 / (n : ℝ)) * X i ω; μ] ≤ Var[T; μ] := by
  have hbridge : Var[fun ω => ∑ i, (1 / (n : ℝ)) * X i ω; μ]
      = GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) :=
    variance_weighted_sum_exchangeable hX _
  constructor
  · rw [hbridge, dash_variance_eq_inv_fisher hn hσ hρ0]
  · intro T hTu hT
    rw [hbridge]
    exact dash_global_min_variance hn hσ hρ0 hscore hTu hT

/-! ## §7 Global Pareto optimality (builds on the quantitative GBDT layer)

  The two theorems below combine the between-group Cramér–Rao step (§6) with
  the within-group dominance already machine-checked in ParetoOptimality.lean.
  The within-group coordinate lives on the GBDT model measure, so — like
  `dash_within_group_dominance` itself — these theorems deliberately depend on
  the two bundled axioms `gbdtWorld`/`gbdtAxioms` in addition to Lean core. -/

/-- **Global Pareto optimality of DASH** (two-coordinate form, conditional on
    the DGP hypotheses within-group and the score property between-group).

    A competing method is a within-group ranking `ranking` together with a
    between-group gap estimator `T` in the unbiased (faithful) class. Against
    every such method, DASH — which ties within-group and estimates
    between-group gaps by the equal-weight consensus — satisfies:

    1. (within-group) DASH's disagreement measure ≤ the competitor's;
    2. (between-group) DASH's aggregate variance ≤ `Var(T)`;
    3. (strictness) if the competitor COMMITS on the within-group pair, DASH
       is strictly better there (0 < 1/2 by `dash_within_group_dominance`).

    So DASH is coordinatewise minimal over faithful methods: the global Pareto
    claim, with the Cramér–Rao step carried by `CramerRaoScoreProperty` and
    everything else machine-checked. Biased (unfaithful) between-group
    estimators remain outside the claim, exactly as the monograph's remark
    states — that exclusion is the content of "faithful", not a gap. -/
theorem dash_global_pareto_optimal
    {fs : FeatureSpace}
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} (hn : 0 < n) {σ ρ : ℝ} (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    {unbiased : Set (Ω → ℝ)}
    (hscore : CramerRaoScoreProperty μ unbiased (exchFisherInfo n σ ρ))
    -- the competing method
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_antisym : ¬ (ranking j k ∧ ranking k j))
    {T : Ω → ℝ} (hTu : T ∈ unbiased) (hT : MemLp T 2 μ)
    -- DASH ties the within-group pair
    (ranking_dash : Fin fs.P → Fin fs.P → Prop)
    (h_dash_tie : ¬ ranking_dash j k ∧ ¬ ranking_dash k j) :
    -- 1. within-group: DASH's disagreement ≤ competitor's
    (modelMeasure
        ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
         {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
      ≤ modelMeasure
        ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
         {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f}))
    -- 2. between-group: DASH's variance ≤ competitor's
    ∧ GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) ≤ Var[T; μ]
    -- 3. strictness within-group whenever the competitor commits
    ∧ ((ranking j k ∨ ranking k j) →
        modelMeasure
          ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
        < modelMeasure
          ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [tie_disagreement_zero j k ranking_dash h_dash_tie]
    exact zero_le _
  · exact dash_global_min_variance hn hσ hρ0 hscore hTu hT
  · intro h_commit
    rw [tie_disagreement_zero j k ranking_dash h_dash_tie]
    exact dash_within_group_dominance hprob hmeas hsym hnd ℓ j k hj hk hjk
      ranking h_antisym h_commit

/-- **No faithful method strictly Pareto-dominates DASH.** Strict domination
    would require the competitor to be no worse in both coordinates and
    strictly better in one; but DASH sits at the within-group floor (measure 0)
    and, conditional on the score property, at the between-group Cramér–Rao
    floor, so neither strict improvement is possible. This is the global
    Pareto-optimality statement in its no-domination form. -/
theorem no_unbiased_method_dominates_dash
    {fs : FeatureSpace}
    (j k : Fin fs.P)
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {n : ℕ} (hn : 0 < n) {σ ρ : ℝ} (hσ : 0 < σ) (hρ0 : 0 ≤ ρ)
    {unbiased : Set (Ω → ℝ)}
    (hscore : CramerRaoScoreProperty μ unbiased (exchFisherInfo n σ ρ))
    (ranking : Fin fs.P → Fin fs.P → Prop)
    {T : Ω → ℝ} (hTu : T ∈ unbiased) (hT : MemLp T 2 μ)
    (ranking_dash : Fin fs.P → Fin fs.P → Prop)
    (h_dash_tie : ¬ ranking_dash j k ∧ ¬ ranking_dash k j) :
    ¬ ((modelMeasure
          ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
        ≤ modelMeasure
          ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
        ∧ Var[T; μ] ≤ GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)))
      ∧ (modelMeasure
          ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
        < modelMeasure
          ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
           {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
        ∨ Var[T; μ]
          < GaussMarkovDASH.exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)))) := by
  rintro ⟨⟨_h1, _h2⟩, h3 | h3⟩
  · -- Strictly better within-group than DASH is impossible: DASH is at 0
    rw [tie_disagreement_zero j k ranking_dash h_dash_tie] at h3
    simp at h3
  · -- Strictly better between-group is impossible: the Cramér–Rao floor
    have hcr := dash_global_min_variance hn hσ hρ0 hscore hTu hT
    linarith

/-! ## Remark: exact proof-status accounting for the global Pareto claim

  Machine-checked unconditionally (Lean core axioms only):
  `covariance_sq_le_variance_mul_variance`, `abstract_cramer_rao`,
  `cramer_rao_bound_of_score` (reduction), `cramerRaoScoreProperty_gaussianReal`
  (discharge, single-sample Gaussian), `gaussianReal_cramer_rao_attained`,
  `variance_weighted_sum_exchangeable`, `exch_denom_pos`, `exchFisherInfo_pos`,
  `dash_variance_eq_inv_fisher`, `dash_global_min_variance`, `dash_mvue`.

  Machine-checked on the quantitative GBDT layer (adds `gbdtWorld`,
  `gbdtAxioms`, matching `dash_within_group_dominance`):
  `dash_global_pareto_optimal`, `no_unbiased_method_dominates_dash`.

  The single remaining unformalized ingredient is `CramerRaoScoreProperty`
  instantiated at the M-sample exchangeable Gaussian model — the existence of
  its score with variance `exchFisherInfo` and unit covariance against every
  unbiased estimator. That is the classical regularity computation
  (multivariate Gaussian score + differentiation under the integral), proved
  here in the single-sample case and standard in the literature for the
  general case; formalizing it requires a Fisher-information development that
  Mathlib does not yet contain. -/

end UniversalImpossibility.ParetoGlobal
