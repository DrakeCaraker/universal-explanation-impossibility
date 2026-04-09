/-
  Derivation of attribution_sum_symmetric from the split-count axioms
  and the balanced-ensemble property.

  This lives in a separate file to break the circular dependency:
  Defs.lean ← SplitGap.lean ← SymmetryDerive.lean
-/
import UniversalImpossibility.SplitGap

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Helper: split-count difference for each model, classified by first-mover -/

/-- For each model, the split-count difference n_j - n_k depends only on
    whether the first-mover is j, k, or neither. -/
private lemma splitCount_diff_cases (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    (firstMover fs f = j → splitCount fs j f - splitCount fs k f =
      fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2)) ∧
    (firstMover fs f = k → splitCount fs j f - splitCount fs k f =
      -(fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2))) ∧
    (firstMover fs f ≠ j → firstMover fs f ≠ k →
      splitCount fs j f - splitCount fs k f = 0) := by
  refine ⟨fun hfmj => ?_, fun hfmk => ?_, fun hfmj hfmk => ?_⟩
  · have hne : firstMover fs f ≠ k := by rw [hfmj]; exact hjk
    exact split_gap_exact fs f j k ℓ hj hk hfmj hne
  · have hne : firstMover fs f ≠ j := by rw [hfmk]; exact hjk.symm
    have : splitCount fs k f - splitCount fs j f =
      fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2) :=
      split_gap_exact fs f k j ℓ hk hj hfmk hne
    linarith
  · have := splitCount_eq_of_not_firstMover_j_or_k fs f j k ℓ hj hk hfmj hfmk
    linarith

/-- Attribution symmetry for balanced ensembles.
    For a balanced ensemble, ∑ᵢ φ_j(fᵢ) = ∑ᵢ φ_k(fᵢ) for features j, k in the same group.

    Derived from proportionality_global + split-count structure + balance. -/
theorem attribution_sum_symmetric (M : ℕ) (_hM : 0 < M) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (j k : Fin fs.P) (ℓ : Fin fs.L) (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) :
    Finset.univ.sum (fun i => attribution fs j (models i)) =
    Finset.univ.sum (fun i => attribution fs k (models i)) := by
  -- Trivial case: j = k
  by_cases hjk : j = k
  · subst hjk; rfl
  -- Factor out the global proportionality constant
  obtain ⟨c, hc_pos, hc_eq⟩ := proportionality_global fs
  -- Rewrite attributions as c * splitCount
  have attr_j : ∀ i, attribution fs j (models i) = c * splitCount fs j (models i) :=
    fun i => hc_eq (models i) j
  have attr_k : ∀ i, attribution fs k (models i) = c * splitCount fs k (models i) :=
    fun i => hc_eq (models i) k
  conv_lhs => arg 2; ext i; rw [attr_j i]
  conv_rhs => arg 2; ext i; rw [attr_k i]
  -- Factor out c from both sums
  simp_rw [← Finset.mul_sum]
  congr 1
  -- Now show: ∑ splitCount j = ∑ splitCount k
  -- Equivalently: ∑ (splitCount j - splitCount k) = 0
  suffices h : Finset.univ.sum (fun i => splitCount fs j (models i) - splitCount fs k (models i)) = 0 by
    linarith [Finset.sum_sub_distrib (s := Finset.univ)
      (f := fun i => splitCount fs j (models i))
      (g := fun i => splitCount fs k (models i))]
  -- Classify each term by first-mover
  -- Let gap = ρ²T/(2-ρ²)
  set gap := fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2) with hgap_def
  -- Partition Finset.univ into three parts: fm=j, fm=k, fm∉{j,k}
  -- For fm=j: difference = gap
  -- For fm=k: difference = -gap
  -- For fm∉{j,k}: difference = 0
  have h_term : ∀ i ∈ Finset.univ,
      splitCount fs j (models i) - splitCount fs k (models i) =
      if firstMover fs (models i) = j then gap
      else if firstMover fs (models i) = k then -gap
      else 0 := by
    intro i _
    obtain ⟨hfmj, hfmk, hfm_neither⟩ := splitCount_diff_cases fs (models i) j k ℓ hj hk hjk
    split_ifs with h1 h2
    · exact hfmj h1
    · exact hfmk h2
    · exact sub_eq_zero.mpr (splitCount_eq_of_not_firstMover_j_or_k fs (models i) j k ℓ hj hk h1 h2)
  rw [Finset.sum_congr rfl h_term]
  -- Now sum the if-then-else
  -- Split: ∑(if p then f else g) = ∑_{filter p} f + ∑_{filter ¬p} g
  rw [Finset.sum_ite, Finset.sum_ite]
  -- The constant sums simplify
  simp only [Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
  -- Now we have: card_j * gap + card_k * (-gap) = 0
  -- where card_j = |{i : fm(models i) = j}| and card_k = |{i : fm(models i) = k}|
  -- By IsBalanced, card_j = card_k
  have hbal_jk := hbal ℓ j k hj hk
  -- Convert the Nat equality to ℝ
  have : (Finset.univ.filter (fun i => firstMover fs (models i) = j)).card =
         (Finset.univ.filter (fun i => firstMover fs (models i) = k)).card := hbal_jk
  -- The nested filter: {¬(fm=j)}.filter(fm=k) = {fm=k}
  -- because fm=k → fm≠j (since j≠k)
  have filter_nested :
      (Finset.univ.filter (fun x => ¬firstMover fs (models x) = j)).filter
        (fun x => firstMover fs (models x) = k) =
      Finset.univ.filter (fun x => firstMover fs (models x) = k) := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · exact fun ⟨_, h⟩ => h
    · intro h; exact ⟨fun hj_eq => hjk (hj_eq.symm.trans h), h⟩
  rw [filter_nested]
  have h_cast : ((Finset.univ.filter (fun i => firstMover fs (models i) = j)).card : ℝ) =
                ((Finset.univ.filter (fun i => firstMover fs (models i) = k)).card : ℝ) := by
    exact_mod_cast this
  rw [h_cast]
  ring

end UniversalImpossibility
