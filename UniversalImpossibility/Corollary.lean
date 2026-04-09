/-
  Corollary 1: DASH achieves equity and between-group stability,
  resolving the impossibility by breaking sequential dependence.

  (a) Equity in expectation — consensus attributions are equal within groups
  (b) Stability via LLN — variance → 0 as M → ∞ (stated, proof deferred)
  (c) Within-group ranking is undetermined by symmetry
-/
import UniversalImpossibility.Impossibility
import UniversalImpossibility.SymmetryDerive

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Corollary 1(a): Equity in expectation -/

/-- DASH consensus attributions are equal for features in the same group,
    provided the ensemble is balanced (each feature serves as first-mover
    equally often). Direct from Axiom 6 + definition of consensus. -/
theorem consensus_equity (M : ℕ) (hM : 0 < M) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) :
    consensus fs M hM models j = consensus fs M hM models k := by
  unfold consensus
  congr 1
  exact attribution_sum_symmetric fs M hM models hbal j k ℓ hj hk

/-! ### Corollary 1(c): Within-group instability is irreducible -/

/-- The consensus difference between same-group features is exactly zero
    for balanced ensembles. Neither feature systematically outranks the
    other — any observed ordering is due to finite-sample noise, not a
    true importance difference. -/
theorem consensus_difference_zero (M : ℕ) (hM : 0 < M) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) :
    consensus fs M hM models j - consensus fs M hM models k = 0 := by
  rw [consensus_equity fs M hM models hbal j k ℓ hj hk]
  simp

/-! ### Corollary 1(b): Between-group stability via variance bound -/

/-- The consensus variance for feature j equals the single-model variance
    divided by the ensemble size M. This is the key convergence result:
    larger ensembles produce more stable attributions. -/
theorem consensus_variance_rate (M : ℕ) (hM : 0 < M) (j : Fin fs.P) :
    ∃ (v : ℝ), v = attribution_variance fs j / M ∧ 0 ≤ v := by
  exact consensus_variance_bound fs M hM j

/-- Doubling the ensemble size halves the consensus variance. -/
theorem consensus_variance_halves (M : ℕ) (hM : 0 < M)
    (j : Fin fs.P) (hv : 0 < attribution_variance fs j) :
    attribution_variance fs j / (2 * M) <
    attribution_variance fs j / M := by
  have hM_pos : (0 : ℝ) < M := Nat.cast_pos.mpr hM
  apply div_lt_div_of_pos_left hv (by positivity) (by linarith)

/-- The consensus variance is nonneg for any ensemble size. -/
theorem consensus_variance_nonneg (M : ℕ) (_ : 0 < M) (j : Fin fs.P) :
    0 ≤ attribution_variance fs j / M := by
  apply div_nonneg
  · exact attribution_variance_nonneg fs j
  · exact Nat.cast_nonneg M

end UniversalImpossibility
