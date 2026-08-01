/-
  ScoreRegularity.lean — M1 of the score-regularity frontier campaign:
  differentiation under the integral sign for exponential tilts.

  CONTEXT. `ParetoGlobal.lean` isolates the analytic core of the Cramér–Rao
  step as the named hypothesis `CramerRaoScoreProperty` (Sard pattern) and
  discharges it for the single-sample Gaussian location model only. The open
  frontier item of the program is the discharge for general regular families.
  The campaign plan (SCORE_REGULARITY_SCOPING.md, decomposition B) runs the
  family through the base measure as the exponential tilt μ_t = μ.tilted (t*X·)
  and needs exactly ONE analytic lemma that Mathlib does not supply at this
  pin: differentiation under the integral sign for t ↦ ∫ T·exp(t·X) dμ with a
  GENERAL estimator `T` in place of the powers `X^n` that Mathlib's
  `hasDerivAt_integral_pow_mul_exp_real` (MGFAnalytic.lean) handles.

  WHAT THIS FILE PROVES (M1). `hasDerivAt_integral_mul_exp`:
  if `T·exp(t·X)` and `T·X·exp(t·X)` are integrable for all `t` near `t₀`, then
      d/dt ∫ T·exp(t·X) dμ  =  ∫ T·X·exp(t₀·X) dμ   at t₀.
  Proof: Mathlib's `hasDerivAt_integral_of_dominated_loc_of_deriv_le`
  (ParametricIntegral.lean) with the dominating envelope
  |T·X|·(exp(a·X) + exp(b·X)) on the interval [a,b] = [t₀−δ/2, t₀+δ/2];
  endpoint integrability comes from the hTX hypothesis. This is the skeleton
  of Mathlib's own `hasDerivAt_integral_pow_mul_exp` (ComplexMGF.lean) with
  `X^n` replaced by `T` under hypothesis, carried out over ℝ directly.

  DEVIATION FROM THE SCOPED STATEMENT (documented honestly). The scoping
  report's §3.2 shape carried two further hypotheses:
  `t₀ ∈ interior (integrableExpSet X μ)` and `AEStronglyMeasurable T μ`.
  Both turned out to be unnecessary: an integrable function is a.e. strongly
  measurable, so the two eventual-integrability hypotheses already carry all
  the measurability the dominated-derivative theorem needs. The lemma proved
  here is therefore STRICTLY MORE GENERAL than the scoped shape, and remains
  directly dischargeable in the NEF setting of M2 (X with exponential moments
  near t₀, T in an L²-with-exponential-envelope class: there hT/hTX follow
  from Cauchy–Schwarz and the interior-membership toolkit of
  IntegrableExpMul.lean).

  WHAT THIS FILE DOES NOT PROVE. It does NOT discharge
  `CramerRaoScoreProperty` for any new family. The general
  natural-exponential-family discharge (M2, `cramerRaoScoreProperty_tilted`)
  and the M-sample exchangeable Gaussian instantiation (M3) build on this
  lemma and are forthcoming; until they land, the general-family score
  property remains exactly as open as `ParetoGlobal.lean` states.

  Zero new axioms. Zero sorry.
-/

import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

set_option autoImplicit false

open MeasureTheory Real Filter
open scoped Topology

namespace UniversalImpossibility.ScoreRegularity

/-- On an interval `[a, b]`, the exponential `exp (t * x)` is dominated by the
    sum of its endpoint values `exp (a * x) + exp (b * x)`, uniformly in the
    sign of `x`. The elementary envelope inequality behind the domination
    argument. -/
lemma exp_mul_le_exp_add_exp {a b t x : ℝ} (hat : a ≤ t) (htb : t ≤ b) :
    exp (t * x) ≤ exp (a * x) + exp (b * x) := by
  rcases le_total 0 x with hx | hx
  · calc exp (t * x) ≤ exp (b * x) :=
          exp_le_exp.mpr (mul_le_mul_of_nonneg_right htb hx)
    _ ≤ exp (a * x) + exp (b * x) := le_add_of_nonneg_left (exp_pos _).le
  · calc exp (t * x) ≤ exp (a * x) :=
          exp_le_exp.mpr (mul_le_mul_of_nonpos_right hat hx)
    _ ≤ exp (a * x) + exp (b * x) := le_add_of_nonneg_right (exp_pos _).le

/-- **Differentiation under the integral sign for exponential tilts (M1).**
    If `T·exp(t·X)` and `T·X·exp(t·X)` are integrable for all `t` in a
    neighbourhood of `t₀`, then `t ↦ ∫ T·exp(t·X) dμ` is differentiable at
    `t₀` with derivative `∫ T·X·exp(t₀·X) dμ`. For an unbiased estimator `T`
    of the tilt parameter this is exactly the identity
    `d/dt E_{μ_t}[T]·(normalisation) = E[T·X·exp(t₀X)]` from which M2 will
    extract `cov(T, S) = 1` for the NEF score `S = X − E[X]`. -/
theorem hasDerivAt_integral_mul_exp
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}
    {T X : Ω → ℝ} {t₀ : ℝ}
    (hT : ∀ᶠ t in 𝓝 t₀, Integrable (fun ω => T ω * exp (t * X ω)) μ)
    (hTX : ∀ᶠ t in 𝓝 t₀, Integrable (fun ω => T ω * X ω * exp (t * X ω)) μ) :
    HasDerivAt (fun t => ∫ ω, T ω * exp (t * X ω) ∂μ)
      (∫ ω, T ω * X ω * exp (t₀ * X ω) ∂μ) t₀ := by
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.mp hTX
  -- Endpoint integrability of the derivative integrand at t₀ ± δ/2.
  have ha : Integrable (fun ω => T ω * X ω * exp ((t₀ - δ / 2) * X ω)) μ :=
    hball (by rw [Real.dist_eq, show t₀ - δ / 2 - t₀ = -(δ / 2) by ring,
      abs_neg, abs_of_pos (half_pos hδ)]; linarith)
  have hb : Integrable (fun ω => T ω * X ω * exp ((t₀ + δ / 2) * X ω)) μ :=
    hball (by rw [Real.dist_eq, show t₀ + δ / 2 - t₀ = δ / 2 by ring,
      abs_of_pos (half_pos hδ)]; linarith)
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (F := fun t ω => T ω * exp (t * X ω))
      (F' := fun t ω => T ω * X ω * exp (t * X ω))
      (bound := fun ω => |T ω * X ω * exp ((t₀ - δ / 2) * X ω)|
        + |T ω * X ω * exp ((t₀ + δ / 2) * X ω)|)
      (Metric.ball_mem_nhds t₀ (half_pos hδ)) ?_ ?_ ?_ ?_ ?_ ?_).2
  · -- a.e. strong measurability of t ↦ T·exp(t·X), eventually near t₀:
    -- integrability already carries it.
    exact hT.mono fun t ht => ht.aestronglyMeasurable
  · -- integrability at the point t₀ itself
    exact hT.self_of_nhds
  · -- a.e. strong measurability of the derivative integrand at t₀
    exact hTX.self_of_nhds.aestronglyMeasurable
  · -- uniform domination on the ball by the endpoint envelope
    refine ae_of_all _ fun ω t ht => ?_
    rw [Metric.mem_ball, Real.dist_eq, abs_lt] at ht
    calc ‖T ω * X ω * exp (t * X ω)‖
        = |T ω * X ω| * exp (t * X ω) := by
          rw [Real.norm_eq_abs, abs_mul, Real.abs_exp]
      _ ≤ |T ω * X ω| * (exp ((t₀ - δ / 2) * X ω) + exp ((t₀ + δ / 2) * X ω)) :=
          mul_le_mul_of_nonneg_left
            (exp_mul_le_exp_add_exp (by linarith [ht.1]) (by linarith [ht.2]))
            (abs_nonneg _)
      _ = |T ω * X ω * exp ((t₀ - δ / 2) * X ω)|
          + |T ω * X ω * exp ((t₀ + δ / 2) * X ω)| := by
          simp [mul_add, abs_mul, Real.abs_exp]
  · -- the envelope is integrable: it is |hTX at a| + |hTX at b|
    exact ha.abs.add hb.abs
  · -- pointwise differentiability in t, everywhere on the ball
    refine ae_of_all _ fun ω t _ => ?_
    have h : HasDerivAt (fun t' : ℝ => T ω * exp (t' * X ω))
        (T ω * (exp (t * X ω) * X ω)) t :=
      ((hasDerivAt_mul_const (X ω)).exp).const_mul (T ω)
    show HasDerivAt (fun t' : ℝ => T ω * exp (t' * X ω))
      (T ω * X ω * exp (t * X ω)) t
    rw [show T ω * X ω * exp (t * X ω) = T ω * (exp (t * X ω) * X ω) by ring]
    exact h

end UniversalImpossibility.ScoreRegularity
