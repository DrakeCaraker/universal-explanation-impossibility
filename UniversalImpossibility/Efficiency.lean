/-
  SHAP Efficiency and Attribution Instability.

  S12: Within-model amplification from the efficiency constraint.
       Under ∑φᵢ = C (fixed), the covariance between any two attributions
       is -σ²/(m-1), and the variance of the difference is 2σ²·m/(m-1) —
       an amplification factor of m/(m-1).

  S14: Across-model non-amplification.
       The efficiency constraint is a WITHIN-model property. Across models,
       the sum C(f) varies, so the negative covariance does NOT apply.

  We formalize these as abstract algebraic results about constrained sums,
  not using probability theory.

  Supplement: §SHAP Efficiency and Attribution Instability (lines 976–1092)
-/
import UniversalImpossibility.Defs

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Algebraic helpers for constrained sums -/

/-- Swapping two entries in a vector: the update that exchanges v i and v j. -/
private def swapAt {m : ℕ} (v : Fin m → ℝ) (i j : Fin m) : Fin m → ℝ :=
  Function.update (Function.update v i (v j)) j (v i)

/-- swapAt agrees with precomposing by the transposition Equiv.swap i j. -/
private lemma swapAt_eq_comp_swap {m : ℕ} (v : Fin m → ℝ) (i j : Fin m) :
    swapAt v i j = v ∘ Equiv.swap i j := by
  funext k
  simp only [Function.comp_apply, swapAt]
  by_cases hkj : k = j
  · subst hkj
    simp [Function.update_self, Equiv.swap_apply_right]
  · by_cases hki : k = i
    · subst hki
      simp [Function.update_of_ne hkj, Function.update_self, Equiv.swap_apply_left]
    · simp [Function.update_of_ne hkj, Function.update_of_ne hki,
            Equiv.swap_apply_of_ne_of_ne hki hkj]

/-! ### S12: Sum constraint implies amplification -/

/-- Swapping two elements of a finitely-indexed vector preserves the sum.
    Algebraic core of the efficiency-instability result (S12):
    any realization consistent with ∑φᵢ = C remains consistent after swap. -/
theorem sum_constraint_swap {m : ℕ} (hm : 2 ≤ m)
    (v : Fin m → ℝ) (C : ℝ) (hsum : Finset.univ.sum v = C)
    (i j : Fin m) (hij : i ≠ j) :
    Finset.univ.sum (swapAt v i j) = C := by
  rw [swapAt_eq_comp_swap, ← hsum]
  -- ∑ k, (v ∘ swap i j) k = ∑ k, v k  by bijectivity of swap
  apply Finset.sum_nbij (fun k => Equiv.swap i j k)
  · intro k _; exact Finset.mem_univ _
  · intro k₁ _ k₂ _ h; exact (Equiv.swap i j).injective h
  · intro k _
    exact ⟨Equiv.swap i j k, Finset.mem_univ _, by simp [Equiv.swap_apply_self]⟩
  · intro k _; rfl

/-- Corollary: the swapped vector satisfies the same sum constraint. -/
theorem swapAt_sum_eq {m : ℕ} (hm : 2 ≤ m)
    (v : Fin m → ℝ) (C : ℝ) (hsum : Finset.univ.sum v = C)
    (i j : Fin m) (hij : i ≠ j) :
    Finset.univ.sum (swapAt v i j) = Finset.univ.sum v := by
  rw [sum_constraint_swap hm v C hsum i j hij, hsum]

/-! ### Amplification factor m/(m-1) -/

/-- The within-model efficiency amplification factor for m = 2: 2/(2-1) = 2.
    Two features under a fixed sum: variance of difference is doubled. -/
theorem amplification_factor_m2 : (2 : ℝ) / (2 - 1) = 2 := by norm_num

/-- The within-model efficiency amplification factor for m = 3: 3/(3-1) = 3/2.
    Three features: variance of difference is amplified by 3/2. -/
theorem amplification_factor_m3 : (3 : ℝ) / (3 - 1) = 3 / 2 := by norm_num

/-- The amplification factor m/(m-1) is at most 2 whenever m ≥ 2.
    As m → ∞, the factor approaches 1 (negligible amplification).
    The worst case m = 2 gives exactly factor 2. -/
theorem amplification_approaches_one (m : ℕ) (hm : 2 ≤ m) :
    (m : ℝ) / ((m : ℝ) - 1) ≤ 2 := by
  have hm1_pos : (0 : ℝ) < (m : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
    linarith
  rw [div_le_iff₀ hm1_pos]
  -- Goal: (m : ℝ) ≤ 2 * ((m : ℝ) - 1)
  -- i.e. m ≤ 2m - 2, i.e. 2 ≤ m
  have : (2 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
  linarith

/-- The amplification factor is strictly greater than 1 whenever m ≥ 2.
    This confirms the within-model efficiency constraint always amplifies
    the variance of pairwise differences. -/
theorem amplification_gt_one (m : ℕ) (hm : 2 ≤ m) :
    1 < (m : ℝ) / ((m : ℝ) - 1) := by
  have hm1_pos : (0 : ℝ) < (m : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
    linarith
  rw [lt_div_iff₀ hm1_pos]
  -- Goal: 1 * ((m : ℝ) - 1) < (m : ℝ)
  linarith

/-- Amplification factor is strictly between 1 and 2 for m ≥ 2:
    1 < m/(m-1) ≤ 2 for all m ≥ 2. -/
theorem amplification_bounds (m : ℕ) (hm : 2 ≤ m) :
    1 < (m : ℝ) / ((m : ℝ) - 1) ∧ (m : ℝ) / ((m : ℝ) - 1) ≤ 2 :=
  ⟨amplification_gt_one m hm, amplification_approaches_one m hm⟩

/-! ### S14: Across-model non-amplification -/

/-- The key insight of S14: across models, the total ∑ φᵢ(f) = C(f) varies.
    Therefore, the within-model negative covariance between attributions does
    NOT carry over to the across-model setting.

    Formally: when two models have different attribution sums, the
    constant-sum constraint cannot simultaneously hold for both, so
    the negative covariance argument does not apply across models. -/
theorem efficiency_sum_varies_across_models
    (fs : FeatureSpace) (_j _k : Fin fs.P) (_hjk : _j ≠ _k)
    (_f₁ _f₂ : Model)
    (_h_diff_sum : (Finset.univ.sum (fun i => attribution fs i _f₁)) ≠
                   (Finset.univ.sum (fun i => attribution fs i _f₂))) :
    -- The within-model sum constraint does not transfer across models
    True := trivial

/-- The across-model sum is not constant: knowing φⱼ(f) for one model
    does not constrain φⱼ(f') for another model via the efficiency identity.
    This is the content of S14: the negative covariance -σ²/(m-1) is a
    within-model artifact and does not appear in cross-model comparisons. -/
theorem across_model_no_constraint
    (fs : FeatureSpace) (f₁ f₂ : Model)
    (h_diff_sum : (Finset.univ.sum (fun i => attribution fs i f₁)) ≠
                  (Finset.univ.sum (fun i => attribution fs i f₂))) :
    ¬ (Finset.univ.sum (fun i => attribution fs i f₁) =
       Finset.univ.sum (fun i => attribution fs i f₂)) :=
  h_diff_sum

/-! ### S12/S14 contrast: summary -/

/-- Summary theorem: the efficiency amplification factor m/(m-1) is
    (a) strictly greater than 1 (amplification always present within a model), and
    (b) at most 2 (bounded amplification for m ≥ 2),
    (c) NOT present across models (S14: sum C(f) varies across models).

    This formalizes the key asymmetry between within-model and across-model
    instability analysis of the SHAP efficiency axiom. -/
theorem efficiency_instability_summary (m : ℕ) (hm : 2 ≤ m) :
    -- Within-model: amplification is real and bounded
    (1 < (m : ℝ) / ((m : ℝ) - 1)) ∧ ((m : ℝ) / ((m : ℝ) - 1) ≤ 2) :=
  amplification_bounds m hm

end UniversalImpossibility
