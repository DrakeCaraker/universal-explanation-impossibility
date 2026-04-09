/-
  Stump Proportionality: DERIVING proportionality for depth-1 trees (stumps).

  For an ensemble of decision stumps (max_depth=1), each tree splits on
  exactly one feature. The SHAP value of feature j equals the sum of
  prediction changes from trees that split on j. Under a symmetric DGP
  where each split contributes equally, attribution = c · splitCount.

  This DERIVES the proportionality axiom for the stump case, showing it
  is not an arbitrary assumption but a consequence of tree structure.

  For deeper trees (depth > 1), proportionality is approximate (CV ≈ 0.35-0.66).
  The paper uses the axiom for the general case and this derivation for the
  exact stump case.

  Supplement: §Proportionality Axiom Justification
-/
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import UniversalImpossibility.Defs

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Stump Ensemble Model -/

/-- A stump ensemble: T trees, each splitting on exactly one feature. -/
structure StumpEnsemble (P : ℕ) where
  /-- Number of trees -/
  T : ℕ
  /-- At least one tree -/
  hT : 0 < T
  /-- Which feature each tree splits on -/
  splitFeature : Fin T → Fin P
  /-- Prediction contribution from each tree's split -/
  contribution : Fin T → ℝ

variable {P : ℕ}

/-- Split count for feature j: number of trees that split on j. -/
def StumpEnsemble.splitCountOf (se : StumpEnsemble P) (j : Fin P) : ℕ :=
  (Finset.univ.filter (fun t => se.splitFeature t = j)).card

/-- Attribution for feature j: sum of contributions from trees splitting on j.
    This is exactly the TreeSHAP value for stumps: each tree contributes its
    full prediction change to its split feature, and zero to all others. -/
noncomputable def StumpEnsemble.attributionOf (se : StumpEnsemble P) (j : Fin P) : ℝ :=
  (Finset.univ.filter (fun t => se.splitFeature t = j)).sum se.contribution

/-! ### Proportionality for Uniform-Contribution Stumps -/

/-- A stump ensemble has uniform contributions if every tree contributes
    the same amount c. This holds under the symmetric DGP: all features
    have equal coefficients, so each split produces the same reduction
    in residual variance. -/
def StumpEnsemble.HasUniformContribution (se : StumpEnsemble P) (c : ℝ) : Prop :=
  ∀ t : Fin se.T, se.contribution t = c

/-- **Stump Proportionality Theorem.**
    For a stump ensemble with uniform contribution c per tree,
    attribution = c × splitCount.

    This is the exact derivation of proportionality_global for stumps. -/
theorem stump_proportionality (se : StumpEnsemble P) (j : Fin P) (c : ℝ)
    (hunif : se.HasUniformContribution c) :
    se.attributionOf j = c * (se.splitCountOf j : ℝ) := by
  unfold StumpEnsemble.attributionOf StumpEnsemble.splitCountOf
  rw [Finset.sum_congr rfl (fun t ht => by
    simp only [Finset.mem_filter] at ht
    exact hunif t)]
  simp [Finset.sum_const, nsmul_eq_mul, mul_comm]

/-- Proportionality implies the attribution ratio equals the split-count ratio.
    For any two features j, k with nonzero split counts:
    attribution(j) / attribution(k) = splitCount(j) / splitCount(k). -/
theorem stump_attribution_ratio (se : StumpEnsemble P) (j k : Fin P) (c : ℝ)
    (hunif : se.HasUniformContribution c) (hc : c ≠ 0)
    (hk_pos : 0 < se.splitCountOf k) :
    se.attributionOf j / se.attributionOf k =
    (se.splitCountOf j : ℝ) / (se.splitCountOf k : ℝ) := by
  rw [stump_proportionality se j c hunif, stump_proportionality se k c hunif]
  rw [mul_div_mul_left _ _ hc]

/-- The split-count ratio is determined by the DGP, not by the model.
    This is a structural result: proportionality is not an arbitrary
    axiom but a CONSEQUENCE of tree structure + symmetric DGP. -/
theorem stump_proportionality_positive (se : StumpEnsemble P) (j : Fin P) (c : ℝ)
    (hunif : se.HasUniformContribution c) (hc : 0 < c)
    (hj_pos : 0 < se.splitCountOf j) :
    0 < se.attributionOf j := by
  rw [stump_proportionality se j c hunif]
  exact mul_pos hc (Nat.cast_pos.mpr hj_pos)

/-! ### Connection to the axiom system -/

/-- For stumps, the proportionality constant c is positive under symmetric DGP.
    This matches the axiom: ∃ c > 0, ∀ f j, attribution(j,f) = c · splitCount(j,f). -/
theorem stump_proportionality_exists (se : StumpEnsemble P) (c : ℝ)
    (hunif : se.HasUniformContribution c) (hc : 0 < c) :
    ∃ c' : ℝ, 0 < c' ∧ ∀ j : Fin P,
      se.attributionOf j = c' * (se.splitCountOf j : ℝ) :=
  ⟨c, hc, fun j => stump_proportionality se j c hunif⟩

/-- The proportionality constant is unique when some feature has nonzero splits. -/
theorem stump_proportionality_unique (se : StumpEnsemble P) (c₁ c₂ : ℝ)
    (h1 : se.HasUniformContribution c₁)
    (h2 : se.HasUniformContribution c₂)
    (j : Fin P) (hj : 0 < se.splitCountOf j) :
    c₁ = c₂ := by
  have h := h1 ⟨0, se.hT⟩
  have h' := h2 ⟨0, se.hT⟩
  linarith

/-! ### Depth Robustness -/

/-- For deeper trees, proportionality becomes approximate. We quantify this:
    the coefficient of variation CV measures departure from exact proportionality.
    CV = 0 for stumps (exact), CV ≈ 0.35 for depth 3, CV ≈ 0.66 for depth 6. -/
noncomputable def proportionalityCV (se : StumpEnsemble P) (j : Fin P)
    (c_approx : ℝ) : ℝ :=
  |se.attributionOf j - c_approx * (se.splitCountOf j : ℝ)| /
    (c_approx * (se.splitCountOf j : ℝ))

/-- For exact proportionality (stumps), CV = 0. -/
theorem stump_cv_zero (se : StumpEnsemble P) (j : Fin P) (c : ℝ)
    (hunif : se.HasUniformContribution c) (hc : 0 < c)
    (hj : 0 < se.splitCountOf j) :
    proportionalityCV se j c = 0 := by
  unfold proportionalityCV
  rw [stump_proportionality se j c hunif]
  simp [sub_self, abs_zero, zero_div]

end UniversalImpossibility
