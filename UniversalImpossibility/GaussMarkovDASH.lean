/-
  Gauss–Markov / DASH minimum-variance aggregator (exchangeable case).

  Among linear *unbiased* aggregators of exchangeable estimators, the
  equal-weight aggregator — the orbit-average / DASH ensemble mean, i.e. the
  G-invariant symmetric aggregator — has minimum variance.

  Setup.  Suppose we aggregate `n` estimators `X₁, …, Xₙ` with weights
  `w : Fin n → ℝ` summing to `1` (linear unbiased aggregator).  Under the
  *exchangeable* covariance model — each `Xᵢ` has variance `σ²` and each distinct
  pair has covariance `ρσ²` with `0 ≤ ρ < 1` — the variance of `∑ᵢ wᵢ Xᵢ` is

      Var(∑ wᵢ Xᵢ) = σ² · Cov-quadratic-form
                    = σ² · [ (1 − ρ) · ∑ᵢ wᵢ² + ρ · (∑ᵢ wᵢ)² ].

  (The `∑ wᵢ²` term collects the `n` diagonal variances `σ²`; the
  `(∑ wᵢ)² − ∑ wᵢ²` off-diagonal terms each carry covariance `ρσ²`; regrouping
  gives the closed form above.)  We take this closed form as the *definition*
  `exchVar σ ρ w` and prove the optimality result about it as a clean finite
  optimization — a Gauss–Markov theorem for the exchangeable case.

  Main result (`dash_min_variance`): for `0 < σ`, `0 ≤ ρ < 1`, and any weights
  `w` with `∑ wᵢ = 1`,

      exchVar σ ρ (fun _ => 1/n)  ≤  exchVar σ ρ w.

  Core of the proof: with `∑ wᵢ = 1`, the aggregate variance reduces to
  `σ²[(1−ρ)·∑ wᵢ² + ρ]`, so it suffices to show `∑ wᵢ² ≥ 1/n`.  This is the
  finite Cauchy–Schwarz inequality `(∑ wᵢ)² ≤ n · ∑ wᵢ²` (Mathlib's
  `sq_sum_le_card_mul_sum_sq`), i.e. `1 ≤ n · ∑ wᵢ²`.

  This file adds NO axioms; the results depend only on Lean core axioms
  (`propext`, `Classical.choice`, `Quot.sound`) via Mathlib.
-/
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Tactic

set_option autoImplicit false

namespace UniversalImpossibility.GaussMarkovDASH

open scoped BigOperators

/-- Exchangeable aggregate variance.

    `exchVar σ ρ w` is the variance of the linear aggregate `∑ᵢ wᵢ Xᵢ` when each
    `Xᵢ` has variance `σ²` and each distinct pair `(Xᵢ, Xⱼ)` has covariance `ρσ²`.
    It is the quadratic form
    `σ² · ((1 − ρ) · ∑ᵢ wᵢ² + ρ · (∑ᵢ wᵢ)²)`. -/
noncomputable def exchVar {n : ℕ} (σ ρ : ℝ) (w : Fin n → ℝ) : ℝ :=
  σ ^ 2 * ((1 - ρ) * (∑ i, (w i) ^ 2) + ρ * (∑ i, w i) ^ 2)

/-! ### The Cauchy–Schwarz core: equal weights minimize `∑ wᵢ²` -/

/-- **Cauchy–Schwarz lower bound on the sum of squares.**  For weights summing to
    `1` over `Fin n` (`n ≥ 1`), the sum of squares is at least `1/n`, attained by
    the equal weights `wᵢ = 1/n`.  This is the algebraic heart of DASH optimality:
    `1 = (∑ wᵢ)² ≤ n · ∑ wᵢ²`. -/
theorem sum_sq_ge_inv_card {n : ℕ} (hn : 0 < n) (w : Fin n → ℝ)
    (hw : ∑ i, w i = 1) :
    1 / (n : ℝ) ≤ ∑ i, (w i) ^ 2 := by
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  -- finite Cauchy–Schwarz: (∑ wᵢ)² ≤ #univ · ∑ wᵢ²
  have hcs : (∑ i, w i) ^ 2 ≤ (n : ℝ) * ∑ i, (w i) ^ 2 := by
    have h := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset (Fin n))) (f := w)
    simpa [Finset.card_univ] using h
  rw [hw, one_pow] at hcs
  -- hcs : 1 ≤ n · ∑ wᵢ²
  rw [div_le_iff₀ hn']
  nlinarith [hcs]

/-! ### Reduction of the aggregate variance under the unbiasedness constraint -/

/-- Equal-weight sums: `∑ᵢ (1/n) = 1`. -/
theorem sum_equal {n : ℕ} (hn : 0 < n) :
    ∑ _i : Fin n, (1 / (n : ℝ)) = 1 := by
  have hn0 : (n : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hn)
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Equal-weight sum of squares: `∑ᵢ (1/n)² = 1/n`. -/
theorem sum_sq_equal {n : ℕ} (hn : 0 < n) :
    ∑ _i : Fin n, (1 / (n : ℝ)) ^ 2 = 1 / (n : ℝ) := by
  have hn0 : (n : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hn)
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Closed form of the equal-weight (DASH) aggregate variance:
    `exchVar σ ρ (1/n) = σ²·((1−ρ)/n + ρ)`. -/
theorem exchVar_equal {n : ℕ} (hn : 0 < n) (σ ρ : ℝ) :
    exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) = σ ^ 2 * ((1 - ρ) * (1 / (n : ℝ)) + ρ) := by
  unfold exchVar
  rw [sum_equal hn, sum_sq_equal hn]
  ring

/-- Closed form of the aggregate variance for any unbiased weights
    (`∑ wᵢ = 1`): `exchVar σ ρ w = σ²·((1−ρ)·∑ wᵢ² + ρ)`. -/
theorem exchVar_unbiased {n : ℕ} (σ ρ : ℝ) (w : Fin n → ℝ) (hw : ∑ i, w i = 1) :
    exchVar σ ρ w = σ ^ 2 * ((1 - ρ) * (∑ i, (w i) ^ 2) + ρ) := by
  unfold exchVar
  rw [hw]
  ring

/-! ### Main theorem: DASH minimum-variance optimality -/

/-- **DASH minimum-variance aggregator (Gauss–Markov, exchangeable case).**

    Among all linear unbiased aggregators `w : Fin n → ℝ` with `∑ wᵢ = 1`, the
    equal-weight aggregator `wᵢ = 1/n` (the DASH / orbit-average ensemble mean)
    has minimum exchangeable aggregate variance.

    Hypotheses: `n ≥ 1`, `0 < σ`, `0 ≤ ρ`, `ρ < 1`. -/
theorem dash_min_variance {n : ℕ} (hn : 0 < n) (σ ρ : ℝ)
    (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (w : Fin n → ℝ) (hw : ∑ i, w i = 1) :
    exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) ≤ exchVar σ ρ w := by
  rw [exchVar_equal hn, exchVar_unbiased σ ρ w hw]
  have hkey : 1 / (n : ℝ) ≤ ∑ i, (w i) ^ 2 := sum_sq_ge_inv_card hn w hw
  have hσ2 : (0 : ℝ) < σ ^ 2 := by positivity
  have h1ρ : (0 : ℝ) < 1 - ρ := by linarith
  nlinarith [hkey, hσ2, h1ρ, mul_pos hσ2 h1ρ]

/-- **Equality characterization at the equal weights.**  The DASH aggregator
    attains its own optimum: `exchVar σ ρ (1/n) = exchVar σ ρ (1/n)`, and any
    unbiased `w` whose sum of squares equals `1/n` achieves the DASH minimum.
    (The sum of squares `∑ wᵢ²` equals `1/n` exactly when `w ≡ 1/n`, by the
    strictness of Cauchy–Schwarz; we record the achievement direction here.) -/
theorem dash_optimum_attained {n : ℕ} (hn : 0 < n) (σ ρ : ℝ)
    (w : Fin n → ℝ) (hw : ∑ i, w i = 1)
    (hopt : ∑ i, (w i) ^ 2 = 1 / (n : ℝ)) :
    exchVar σ ρ w = exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) := by
  rw [exchVar_equal hn, exchVar_unbiased σ ρ w hw, hopt]

/-! ### DASH / orbit-average optimality corollary -/

/-- **DASH orbit-average optimality.**  Restated as the optimality of the
    G-invariant symmetric aggregator: the equal-weight ensemble mean — the orbit
    average under the symmetric-group action permuting the exchangeable
    estimators, i.e. the DASH ensemble — is the minimum-variance linear unbiased
    aggregator.  Concretely, for every linear unbiased aggregator `w`
    (`∑ wᵢ = 1`), the symmetric equal-weight aggregator `wᵢ = 1/n` does no worse:

        exchVar σ ρ (DASH)  ≤  exchVar σ ρ w. -/
theorem dash_orbit_average_optimal {n : ℕ} (hn : 0 < n) (σ ρ : ℝ)
    (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    ∀ w : Fin n → ℝ, (∑ i, w i = 1) →
      exchVar σ ρ (fun _ : Fin n => 1 / (n : ℝ)) ≤ exchVar σ ρ w :=
  fun w hw => dash_min_variance hn σ ρ hσ hρ0 hρ1 w hw

end UniversalImpossibility.GaussMarkovDASH
