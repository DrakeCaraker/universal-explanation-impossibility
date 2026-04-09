/-
  Spearman rank correlation: definition from scratch and key properties.

  Replaces the axiomatized `spearman` and `spearman_bound` in Defs.lean
  with a concrete definition using midranks and Σd².
-/
import UniversalImpossibility.Defs
import UniversalImpossibility.General
import Mathlib.Data.Finset.Card

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ## Midrank definition -/

/-- Number of elements strictly below v(j) -/
noncomputable def countBelow (v : Fin fs.P → ℝ) (j : Fin fs.P) : ℕ :=
  (Finset.univ.filter (fun i => v i < v j)).card

/-- Number of elements equal to v(j) (including j itself) -/
noncomputable def countEqual (v : Fin fs.P → ℝ) (j : Fin fs.P) : ℕ :=
  (Finset.univ.filter (fun i => v i = v j)).card

/-- Midrank of element j in vector v.
    If v(j) has B elements below it and E elements equal to it,
    midrank = B + (E + 1) / 2. -/
noncomputable def midrank (v : Fin fs.P → ℝ) (j : Fin fs.P) : ℝ :=
  (countBelow fs v j : ℝ) + ((countEqual fs v j : ℝ) + 1) / 2

/-- Sum of squared rank differences between two vectors -/
noncomputable def sumSqRankDiff (v w : Fin fs.P → ℝ) : ℝ :=
  Finset.univ.sum (fun j => (midrank fs v j - midrank fs w j) ^ 2)

/-- Spearman rank correlation coefficient -/
noncomputable def spearmanCorr (v w : Fin fs.P → ℝ) : ℝ :=
  1 - 6 * sumSqRankDiff fs v w / ((fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1))

/-! ## Key property: countEqual is always ≥ 1 -/

/-- Every element is equal to itself, so countEqual ≥ 1 -/
lemma countEqual_pos (v : Fin fs.P → ℝ) (j : Fin fs.P) :
    1 ≤ countEqual fs v j := by
  unfold countEqual
  have : j ∈ Finset.univ.filter (fun i => v i = v j) := by
    simp [Finset.mem_filter]
  exact Finset.one_le_card.mpr ⟨j, this⟩

/-! ## Key lemma: v(j) > v(k) implies midrank(v, j) > midrank(v, k) -/

/-- If v(j) > v(k), then every element ≤ v(k) is strictly below v(j).
    Specifically, countBelow(v, j) ≥ countBelow(v, k) + countEqual(v, k). -/
lemma countBelow_of_gt (v : Fin fs.P → ℝ) (j k : Fin fs.P) (h : v k < v j) :
    countBelow fs v k + countEqual fs v k ≤ countBelow fs v j := by
  unfold countBelow countEqual
  -- The set {i : v(i) < v(k)} ∪ {i : v(i) = v(k)} ⊆ {i : v(i) < v(j)}
  have hsub : Finset.univ.filter (fun i => v i < v k) ∪
              Finset.univ.filter (fun i => v i = v k) ⊆
              Finset.univ.filter (fun i => v i < v j) := by
    intro i hi
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    rcases hi with hlt | heq
    · exact lt_trans hlt h
    · rw [heq]; exact h
  have hdisj : Disjoint (Finset.univ.filter (fun i => v i < v k))
                         (Finset.univ.filter (fun i => v i = v k)) := by
    rw [Finset.disjoint_filter]
    intro i _ hlt heq
    exact absurd heq (ne_of_lt hlt)
  calc (Finset.univ.filter (fun i => v i < v k)).card +
       (Finset.univ.filter (fun i => v i = v k)).card
      = (Finset.univ.filter (fun i => v i < v k) ∪
         Finset.univ.filter (fun i => v i = v k)).card := by
        rw [Finset.card_union_of_disjoint hdisj]
    _ ≤ (Finset.univ.filter (fun i => v i < v j)).card :=
        Finset.card_le_card hsub

/-- If v(j) > v(k), then midrank(v, j) - midrank(v, k) ≥ 1/2.
    This follows from countBelow(j) ≥ countBelow(k) + countEqual(k). -/
lemma midrank_strict_mono (v : Fin fs.P → ℝ) (j k : Fin fs.P) (h : v k < v j) :
    midrank fs v k + 1 / 2 ≤ midrank fs v j := by
  unfold midrank
  have hcb := countBelow_of_gt fs v j k h
  have hce := countEqual_pos fs v j
  -- countBelow(j) ≥ countBelow(k) + countEqual(k)
  -- midrank(j) = countBelow(j) + (countEqual(j) + 1) / 2
  --            ≥ countBelow(k) + countEqual(k) + 1/2   (since countEqual(j) ≥ 1)
  -- midrank(k) = countBelow(k) + (countEqual(k) + 1) / 2
  --            = countBelow(k) + countEqual(k)/2 + 1/2
  -- midrank(j) - midrank(k) ≥ countEqual(k) - countEqual(k)/2 = countEqual(k)/2 ≥ 1/2
  have hcek := countEqual_pos fs v k
  have h1 : (countBelow fs v j : ℝ) ≥ (countBelow fs v k : ℝ) + (countEqual fs v k : ℝ) := by
    exact_mod_cast hcb
  have h2 : (countEqual fs v j : ℝ) ≥ 1 := by exact_mod_cast hce
  have h3 : (countEqual fs v k : ℝ) ≥ 1 := by exact_mod_cast hcek
  linarith

/-! ## Σd² > 0 when first-movers differ -/

/-- The sum of squared rank differences is nonneg (each term is a square) -/
lemma sumSqRankDiff_nonneg (v w : Fin fs.P → ℝ) :
    0 ≤ sumSqRankDiff fs v w := by
  unfold sumSqRankDiff
  exact Finset.sum_nonneg (fun j _ => sq_nonneg _)

/-- When two features have reversed orderings in v and w (v(j) > v(k)
    but w(k) > w(j)), the sum of squared rank differences is at least 1/2. -/
lemma sumSqRankDiff_pos_of_reversal (v w : Fin fs.P → ℝ) (j k : Fin fs.P)
    (hvjk : v k < v j) (hwkj : w j < w k) :
    1 / 2 ≤ sumSqRankDiff fs v w := by
  unfold sumSqRankDiff
  -- d_j = midrank(v, j) - midrank(w, j)
  -- d_k = midrank(v, k) - midrank(w, k)
  -- From hvjk: midrank(v, j) ≥ midrank(v, k) + 1/2
  -- From hwkj: midrank(w, k) ≥ midrank(w, j) + 1/2
  -- So d_j - d_k = (midrank(v,j) - midrank(v,k)) - (midrank(w,j) - midrank(w,k))
  --             ≥ 1/2 + 1/2 = 1
  -- Since d_j ≠ d_k, d_j² + d_k² ≥ (d_j - d_k)²/2 ≥ 1/2
  have hv := midrank_strict_mono fs v j k hvjk
  have hw := midrank_strict_mono fs w k j hwkj
  -- Use: sum ≥ term_j + term_k ≥ (d_j - d_k)²/2
  -- Actually let's just bound Σ ≥ d_j² + d_k²
  have hj_mem : j ∈ Finset.univ (α := Fin fs.P) := Finset.mem_univ j
  have hk_mem : k ∈ Finset.univ (α := Fin fs.P) := Finset.mem_univ k
  -- We need j ≠ k
  have hjk : j ≠ k := by
    intro heq; subst heq; exact absurd hvjk (not_lt.mpr le_rfl)
  -- Σd² ≥ d_j² + d_k²
  have hpair : (midrank fs v j - midrank fs w j) ^ 2 +
               (midrank fs v k - midrank fs w k) ^ 2 ≤
               Finset.univ.sum (fun i => (midrank fs v i - midrank fs w i) ^ 2) := by
    have hsub : {j, k} ⊆ Finset.univ (α := Fin fs.P) := Finset.subset_univ _
    have hpair_sum := Finset.sum_le_sum_of_subset_of_nonneg hsub
      (fun (i : Fin fs.P) (_ : i ∈ Finset.univ) (_ : i ∉ ({j, k} : Finset (Fin fs.P))) =>
        sq_nonneg (midrank fs v i - midrank fs w i))
    rw [Finset.sum_pair hjk] at hpair_sum
    exact hpair_sum
  -- d_j - d_k ≥ 1
  set dj := midrank fs v j - midrank fs w j
  set dk := midrank fs v k - midrank fs w k
  have hdiff : dj - dk ≥ 1 := by linarith
  -- d_j² + d_k² ≥ (d_j - d_k)²/2
  have hsq : dj ^ 2 + dk ^ 2 ≥ (dj - dk) ^ 2 / 2 := by nlinarith [sq_nonneg (dj + dk)]
  -- (d_j - d_k)² ≥ 1
  have hdiffsq : (dj - dk) ^ 2 ≥ 1 := by nlinarith
  linarith

/-! ## Spearman < 1 when attributions reverse -/

/-- The Spearman correlation is strictly less than 1 when two features
    have reversed rankings between the two vectors. -/
theorem spearmanCorr_lt_one_of_reversal (v w : Fin fs.P → ℝ) (j k : Fin fs.P)
    (hvjk : v k < v j) (hwkj : w j < w k) (hP : 2 ≤ fs.P) :
    spearmanCorr fs v w < 1 := by
  unfold spearmanCorr
  have hsd := sumSqRankDiff_pos_of_reversal fs v w j k hvjk hwkj
  have hP_pos : (0 : ℝ) < (fs.P : ℝ) := Nat.cast_pos.mpr fs.hP
  have hP2 : (1 : ℝ) < (fs.P : ℝ) := by exact_mod_cast (show 1 < fs.P by omega)
  have hPsq : (0 : ℝ) < (fs.P : ℝ) ^ 2 - 1 := by nlinarith
  have hdenom_pos : (0 : ℝ) < (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) :=
    mul_pos hP_pos hPsq
  have hsd_pos : (0 : ℝ) < sumSqRankDiff fs v w := by linarith
  have : (0 : ℝ) < 6 * sumSqRankDiff fs v w / ((fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1)) :=
    div_pos (mul_pos (by norm_num : (0:ℝ) < 6) hsd_pos) hdenom_pos
  linarith

/-! ## Quantitative bound: spearmanCorr ≤ 1 - 3/(P³ - P) -/

/-- When first-movers differ in the same group, Spearman ≤ 1 - 3/(P³ - P).
    This is the DERIVED stability bound that replaces the axiomatized spearman_bound. -/
theorem spearmanCorr_bound (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k)
    (hP : 2 ≤ fs.P) :
    spearmanCorr fs (fun i => attribution fs i f) (fun i => attribution fs i f') ≤
      1 - 3 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) := by
  unfold spearmanCorr
  -- We have attribution reversal: attribution j f > attribution k f,
  -- attribution k f' > attribution j f'
  have hrev := attribution_reversal fs f f' j k ℓ hj hk hfm hfm' hjk
  have hsd := sumSqRankDiff_pos_of_reversal fs
    (fun i => attribution fs i f) (fun i => attribution fs i f')
    j k hrev.1 hrev.2
  -- sumSqRankDiff ≥ 1/2
  -- spearmanCorr = 1 - 6 * Σd² / (P(P²-1))
  --             ≤ 1 - 6 * (1/2) / (P(P²-1))
  --             = 1 - 3 / (P(P²-1))
  --             = 1 - 3 / (P³ - P)
  have hP_pos : (0 : ℝ) < (fs.P : ℝ) := Nat.cast_pos.mpr fs.hP
  have hP2 : (1 : ℝ) < (fs.P : ℝ) := by exact_mod_cast (show 1 < fs.P by omega)
  have hPsq : (0 : ℝ) < (fs.P : ℝ) ^ 2 - 1 := by nlinarith
  have hdenom_pos : (0 : ℝ) < (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) :=
    mul_pos hP_pos hPsq
  have hdenom_ne : (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) ≠ 0 := ne_of_gt hdenom_pos
  -- P * (P² - 1) = P³ - P
  have hfactor : (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) = (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by ring
  rw [hfactor]
  have hdenom_pos' : (0 : ℝ) < (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by linarith [hfactor]
  -- 1 - 6 * Σd² / (P³ - P) ≤ 1 - 3 / (P³ - P)
  -- ⟺ 3 / (P³ - P) ≤ 6 * Σd² / (P³ - P)
  -- ⟺ 3 ≤ 6 * Σd²  (since denominator > 0)
  -- ⟺ 1/2 ≤ Σd²  ✓
  suffices h : 3 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) ≤
               6 * sumSqRankDiff fs (fun i => attribution fs i f)
                 (fun i => attribution fs i f') /
               ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) by linarith
  rw [div_le_div_iff_of_pos_right hdenom_pos']
  linarith

/-! ## Cross-group stability consequences

  With `splitCount_crossGroup_stable` (Defs.lean), we derive:
  1. Features outside group ℓ have identical attributions in models f and f'
     when both first-movers are in group ℓ.
  2. Non-first-mover features in group ℓ all have the same attribution,
     so countEqual ≥ m-1 for any non-first-mover in the group.
  3. The midrank gap between the first-mover and non-first-movers is ≥ (m-1)/2.
  4. Σd² ≥ (m-1)²/2, giving spearmanCorr ≤ 1 - 3(m-1)²/(P³-P).
-/

/-- Features outside group ℓ have identical attributions when both
    first-movers are in group ℓ. Follows from splitCount_crossGroup_stable
    and proportionality_global. -/
theorem attribution_eq_outside_group (f f' : Model) (i : Fin fs.P) (ℓ : Fin fs.L)
    (hi : i ∉ fs.group ℓ)
    (hfm : firstMover fs f ∈ fs.group ℓ)
    (hfm' : firstMover fs f' ∈ fs.group ℓ) :
    attribution fs i f = attribution fs i f' := by
  obtain ⟨c, _, hcf⟩ := proportionality_global fs
  rw [hcf f i, hcf f' i]
  congr 1
  exact splitCount_crossGroup_stable fs f f' i ℓ hi hfm hfm'

/-- Non-first-mover features in the same group all have the same attribution.
    If j and k are both in group ℓ and neither is the first-mover of f,
    then attr(j,f) = attr(k,f). -/
theorem attribution_eq_non_firstMover (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfmj : firstMover fs f ≠ j) (hfmk : firstMover fs f ≠ k) :
    attribution fs j f = attribution fs k f := by
  obtain ⟨c, _, hcf⟩ := proportionality_global fs
  rw [hcf f j, hcf f k]
  congr 1
  exact splitCount_eq_of_not_firstMover_j_or_k fs f j k ℓ hj hk hfmj hfmk

/-- In model f with first-mover j ∈ group ℓ, every other member k of group ℓ
    has the same attribution as the non-first-mover value. Combined with
    crossGroup_stable, ONLY j and the first-mover of f' differ between models. -/
theorem attribution_swap_structure (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k) :
    -- In model f: k is a non-first-mover with same attribution as all other
    -- non-first-movers in the group. In model f': j is a non-first-mover
    -- with attribution equal to k's attribution in model f.
    attribution fs k f = attribution fs j f' := by
  obtain ⟨c, _, hcf⟩ := proportionality_global fs
  rw [hcf f k, hcf f' j]
  congr 1
  -- splitCount k f = (1-ρ²)T/(2-ρ²) since k is non-first-mover in group ℓ
  rw [splitCount_nonFirstMover fs f k ℓ hk (by rw [hfm]; exact hjk) (by rw [hfm]; exact hj)]
  -- splitCount j f' = (1-ρ²)T/(2-ρ²) since j is non-first-mover in group ℓ
  rw [splitCount_nonFirstMover fs f' j ℓ hj (by rw [hfm']; exact hjk.symm) (by rw [hfm']; exact hk)]

/-- For any non-first-mover i in group ℓ of model f, the attribution equals
    that of any other non-first-mover in group ℓ (including across models,
    when both models have their first-mover in group ℓ). Concretely:
    if firstMover f = j and i ∈ group ℓ with i ≠ j, then
    attr(i,f) = attr(i,f') for any f' with firstMover f' ∈ group ℓ and
    firstMover f' ≠ i. -/
theorem attribution_non_fm_stable (f f' : Model) (i : Fin fs.P) (ℓ : Fin fs.L)
    (hi : i ∈ fs.group ℓ)
    (hfm_i : firstMover fs f ≠ i)
    (hfm'_i : firstMover fs f' ≠ i)
    (hfm : firstMover fs f ∈ fs.group ℓ)
    (hfm' : firstMover fs f' ∈ fs.group ℓ) :
    attribution fs i f = attribution fs i f' := by
  obtain ⟨c, _, hcf⟩ := proportionality_global fs
  rw [hcf f i, hcf f' i]
  congr 1
  rw [splitCount_nonFirstMover fs f i ℓ hi hfm_i hfm,
      splitCount_nonFirstMover fs f' i ℓ hi hfm'_i hfm']

/-! ## Stronger Σd² lower bound using cross-group stability

  With crossGroup_stable, when firstMover f = j and firstMover f' = k (both
  in group ℓ), the attribution vectors v(i) = attr(i,f) and w(i) = attr(i,f')
  satisfy:
  - v(i) = w(i) for all i ∉ {j, k}  (outside group: crossGroup_stable;
    inside group non-fm: same split count formula)
  - v(j) = w(k) = HIGH, v(k) = w(j) = LOW  (j,k swap roles)

  When two positions swap values in a vector:
  - The midranks of ALL other positions are unchanged (the set of values
    below/equal to any other position is just permuted).
  - The swapped positions exchange midranks: midrank(v,j) = midrank(w,k),
    midrank(v,k) = midrank(w,j).
  - Σd² = 2 * (midrank(v,j) - midrank(v,k))².

  The midrank gap midrank(v,j) - midrank(v,k) ≥ (m-1)/2 because k is tied
  with (m-2) other non-first-movers, so countEqual(v,k) ≥ m-1, and the
  existing proof of `midrank_strict_mono` actually gives gap ≥ countEqual(k)/2.

  Formalizing the full "swap preserves other midranks" argument requires
  proving that for all i ∉ {j,k}, countBelow and countEqual are preserved —
  a finset cardinality argument involving filter-set bijections. The
  `countEqual_ge_groupSize_minus_one` lemma below captures the key algebraic
  insight about the within-group tie structure.
-/

/-- In model f with first-mover j in group ℓ, the number of features with
    attribution equal to attr(k,f) (where k ≠ j is in the group) is at
    least m-1 (all non-first-movers in the group are tied). -/
lemma countEqual_ge_groupSize_minus_one (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) :
    fs.groupSize ℓ - 1 ≤
      countEqual fs (fun i => attribution fs i f) k := by
  unfold countEqual
  -- Every member of group ℓ other than j has the same attribution as k
  -- So {i : attr(i,f) = attr(k,f)} ⊇ group ℓ \ {j}
  have hsub : (fs.group ℓ).erase j ⊆
      Finset.univ.filter (fun i => attribution fs i f = attribution fs k f) := by
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have hi_in : i ∈ fs.group ℓ := Finset.mem_of_mem_erase hi
    have hi_ne_j : i ≠ j := Finset.ne_of_mem_erase hi
    -- i is not the first-mover (since firstMover = j and i ≠ j)
    have hfm_ne_i : firstMover fs f ≠ i := by rw [hfm]; exact hi_ne_j.symm
    -- k is not the first-mover
    have hfm_ne_k : firstMover fs f ≠ k := by rw [hfm]; exact hjk
    exact attribution_eq_non_firstMover fs f i k ℓ hi_in hk hfm_ne_i hfm_ne_k
  have hcard_erase : ((fs.group ℓ).erase j).card = fs.groupSize ℓ - 1 := by
    unfold FeatureSpace.groupSize
    exact Finset.card_erase_of_mem hj
  calc fs.groupSize ℓ - 1
      = ((fs.group ℓ).erase j).card := hcard_erase.symm
    _ ≤ (Finset.univ.filter (fun i =>
          attribution fs i f = attribution fs k f)).card :=
        Finset.card_le_card hsub

/-- Strengthened midrank gap: when the first-mover j has strictly higher
    attribution than k (a non-first-mover in a group of size m),
    midrank(v,j) - midrank(v,k) ≥ (m-1)/2. -/
lemma midrank_gap_ge_half_groupSize (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) :
    let v := fun i => attribution fs i f
    ((fs.groupSize ℓ : ℝ) - 1) / 2 ≤ midrank fs v j - midrank fs v k := by
  intro v
  -- We know v(k) < v(j) from the first-mover structure
  have hvjk : v k < v j := attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk
  -- From countBelow_of_gt: countBelow(j) ≥ countBelow(k) + countEqual(k)
  have hcb := countBelow_of_gt fs v j k hvjk
  -- countEqual(v, k) ≥ m-1
  have hce_k := countEqual_ge_groupSize_minus_one fs f j k ℓ hj hk hjk hfm
  -- countEqual(v, j) ≥ 1
  have hce_j := countEqual_pos fs v j
  unfold midrank
  have h1 : (countBelow fs v j : ℝ) ≥ (countBelow fs v k : ℝ) + (countEqual fs v k : ℝ) := by
    exact_mod_cast hcb
  have h2 : (countEqual fs v j : ℝ) ≥ 1 := by exact_mod_cast hce_j
  have h3 : (countEqual fs v k : ℝ) ≥ (fs.groupSize ℓ : ℝ) - 1 := by
    have hle : (fs.groupSize ℓ - 1 : ℕ) ≤ countEqual fs v k := hce_k
    -- groupSize ≥ 2 (from group_size_ge_two, noting groupSize = group.card)
    have hgs : 2 ≤ fs.groupSize ℓ := fs.group_size_ge_two ℓ
    -- groupSize ≥ 1, so Nat subtraction is well-behaved
    have hge1 : 1 ≤ fs.groupSize ℓ := Nat.one_le_iff_ne_zero.mpr (by intro h; simp [h] at hgs)
    -- groupSize = (groupSize - 1) + 1 ≤ countEqual + 1
    have hce_ge : fs.groupSize ℓ ≤ countEqual fs v k + 1 :=
      calc fs.groupSize ℓ = (fs.groupSize ℓ - 1) + 1 := (Nat.sub_add_cancel hge1).symm
        _ ≤ countEqual fs v k + 1 := Nat.add_le_add_right hle 1
    -- Cast to ℝ
    have hcast : (fs.groupSize ℓ : ℝ) ≤ (countEqual fs v k : ℝ) + 1 := by exact_mod_cast hce_ge
    linarith
  linarith

/-- Strengthened Σd² bound: when first-movers differ within a group of size m,
    Σd² ≥ (m-1)²/2 (instead of just 1/2 from the generic reversal bound).
    This bound is derived using crossGroup_stable. -/
theorem sumSqRankDiff_ge_sq_groupSize (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k) :
    ((fs.groupSize ℓ : ℝ) - 1) ^ 2 / 2 ≤
      sumSqRankDiff fs (fun i => attribution fs i f) (fun i => attribution fs i f') := by
  set v := fun i => attribution fs i f
  set w := fun i => attribution fs i f'
  -- Attribution reversal gives v(k) < v(j) and w(j) < w(k)
  have hrev := attribution_reversal fs f f' j k ℓ hj hk hfm hfm' hjk
  -- Midrank gaps ≥ (m-1)/2 in each model
  have hgap_v := midrank_gap_ge_half_groupSize fs f j k ℓ hj hk hjk hfm
  have hgap_w := midrank_gap_ge_half_groupSize fs f' k j ℓ hk hj hjk.symm hfm'
  -- d_j = midrank(v,j) - midrank(w,j), d_k = midrank(v,k) - midrank(w,k)
  -- d_j - d_k = (midrank(v,j) - midrank(v,k)) - (midrank(w,j) - midrank(w,k))
  --           = (midrank(v,j) - midrank(v,k)) + (midrank(w,k) - midrank(w,j))
  --           ≥ (m-1)/2 + (m-1)/2 = m-1
  set dj := midrank fs v j - midrank fs w j
  set dk := midrank fs v k - midrank fs w k
  have hdiff : dj - dk ≥ (fs.groupSize ℓ : ℝ) - 1 := by
    show midrank fs v j - midrank fs w j - (midrank fs v k - midrank fs w k) ≥ _
    -- Rearrange: (midrank(v,j) - midrank(v,k)) + (midrank(w,k) - midrank(w,j)) ≥ m-1
    have := hgap_v
    have := hgap_w
    linarith
  -- Σd² ≥ d_j² + d_k² ≥ (d_j - d_k)²/2 ≥ (m-1)²/2
  unfold sumSqRankDiff
  have hpair : dj ^ 2 + dk ^ 2 ≤
      Finset.univ.sum (fun i => (midrank fs v i - midrank fs w i) ^ 2) := by
    have hsub : {j, k} ⊆ Finset.univ (α := Fin fs.P) := Finset.subset_univ _
    have hne : j ≠ k := hjk
    have hpair_sum := Finset.sum_le_sum_of_subset_of_nonneg hsub
      (fun (i : Fin fs.P) (_ : i ∈ Finset.univ) (_ : i ∉ ({j, k} : Finset (Fin fs.P))) =>
        sq_nonneg (midrank fs v i - midrank fs w i))
    rw [Finset.sum_pair hne] at hpair_sum
    exact hpair_sum
  have hsq : dj ^ 2 + dk ^ 2 ≥ (dj - dk) ^ 2 / 2 := by nlinarith [sq_nonneg (dj + dk)]
  have hm := fs.group_size_ge_two ℓ
  have hm_cast : (fs.groupSize ℓ : ℝ) ≥ 2 := by
    unfold FeatureSpace.groupSize
    exact_mod_cast hm
  have hdiffsq : (dj - dk) ^ 2 ≥ ((fs.groupSize ℓ : ℝ) - 1) ^ 2 := by nlinarith
  linarith

/-- Strengthened Spearman bound using cross-group stability:
    spearmanCorr ≤ 1 - 3(m-1)²/(P³-P), where m = groupSize ℓ.
    This is strictly tighter than the generic 1 - 3/(P³-P) bound
    since m ≥ 2 implies (m-1)² ≥ 1. -/
theorem spearmanCorr_bound_groupSize (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k)
    (hP : 2 ≤ fs.P) :
    spearmanCorr fs (fun i => attribution fs i f) (fun i => attribution fs i f') ≤
      1 - 3 * ((fs.groupSize ℓ : ℝ) - 1) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) := by
  unfold spearmanCorr
  have hsd := sumSqRankDiff_ge_sq_groupSize fs f f' j k ℓ hj hk hjk hfm hfm'
  have hP_pos : (0 : ℝ) < (fs.P : ℝ) := Nat.cast_pos.mpr fs.hP
  have hP2 : (1 : ℝ) < (fs.P : ℝ) := by exact_mod_cast (show 1 < fs.P by omega)
  have hPsq : (0 : ℝ) < (fs.P : ℝ) ^ 2 - 1 := by nlinarith
  have hdenom_pos : (0 : ℝ) < (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) :=
    mul_pos hP_pos hPsq
  have hfactor : (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) = (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by ring
  rw [hfactor]
  have hdenom_pos' : (0 : ℝ) < (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by linarith [hfactor]
  suffices h : 3 * ((fs.groupSize ℓ : ℝ) - 1) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) ≤
               6 * sumSqRankDiff fs (fun i => attribution fs i f)
                 (fun i => attribution fs i f') /
               ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) by linarith
  rw [div_le_div_iff_of_pos_right hdenom_pos']
  linarith

/-! ## Spearman bound (derived from split-count structure)

  The classical combinatorial argument gives Spearman ≤ 1 - m³/P³. Our formal
  derivation achieves the slightly weaker bound 1 - 3(m-1)²/(P³-P), which
  suffices for all downstream results. The gap is a "swap preserves midranks"
  finset cardinality argument — formalizable but tedious (~250 lines).

  The derived bound is used everywhere the classical bound was previously
  axiomatized. The key insight is the same: instability scales with (m/P)². -/

/-- Spearman instability bound (derived, formerly axiomatized):
    When two models have different first-movers within the same group,
    their Spearman correlation is bounded away from 1.

    The bound 3(m-1)²/(P³-P) is fully derived from the split-count axioms
    and cross-group stability. It is weaker than the classical m³/P³ bound
    but requires no additional axioms. -/
theorem spearman_instability_bound (f f' : Model) (ℓ : Fin fs.L)
    (hfm_grp : firstMover fs f ∈ fs.group ℓ)
    (hfm'_grp : firstMover fs f' ∈ fs.group ℓ)
    (hdiff : firstMover fs f ≠ firstMover fs f')
    (hP : 2 ≤ fs.P) :
    spearmanCorr fs (fun j => attribution fs j f) (fun j => attribution fs j f') ≤
      1 - 3 * ((fs.groupSize ℓ : ℝ) - 1) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) := by
  exact spearmanCorr_bound_groupSize fs f f' (firstMover fs f) (firstMover fs f')
    ℓ hfm_grp hfm'_grp hdiff rfl rfl hP

/-! ## Tighter Spearman bound: m²/2 instead of (m-1)²/2

  The key observation: the midrank gap between the first-mover j and a
  non-first-mover k is at least m/2 (not just (m-1)/2). This is because:
  - countEqual(v,k) ≥ m-1 (all non-first-movers in the group are tied)
  - countEqual(v,j) ≥ 1 (every element is equal to itself)
  - The midrank gap ≥ (countEqual(k) + countEqual(j))/2 ≥ ((m-1) + 1)/2 = m/2

  The existing `midrank_gap_ge_half_groupSize` already has all the hypotheses
  for this tighter bound — it just states (m-1)/2 instead of m/2. -/

/-- Tighter midrank gap: midrank(v,j) - midrank(v,k) ≥ m/2 (not just (m-1)/2).
    The improvement comes from combining countEqual(k) ≥ m-1 with countEqual(j) ≥ 1,
    giving (countEqual(k) + countEqual(j))/2 ≥ m/2. -/
lemma midrank_gap_ge_half_groupSize_tight (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) :
    let v := fun i => attribution fs i f
    (fs.groupSize ℓ : ℝ) / 2 ≤ midrank fs v j - midrank fs v k := by
  intro v
  have hvjk : v k < v j := attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk
  have hcb := countBelow_of_gt fs v j k hvjk
  have hce_k := countEqual_ge_groupSize_minus_one fs f j k ℓ hj hk hjk hfm
  have hce_j := countEqual_pos fs v j
  unfold midrank
  have h1 : (countBelow fs v j : ℝ) ≥ (countBelow fs v k : ℝ) + (countEqual fs v k : ℝ) := by
    exact_mod_cast hcb
  have h2 : (countEqual fs v j : ℝ) ≥ 1 := by exact_mod_cast hce_j
  have h3 : (countEqual fs v k : ℝ) ≥ (fs.groupSize ℓ : ℝ) - 1 := by
    have hle : (fs.groupSize ℓ - 1 : ℕ) ≤ countEqual fs v k := hce_k
    have hgs : 2 ≤ fs.groupSize ℓ := fs.group_size_ge_two ℓ
    have hge1 : 1 ≤ fs.groupSize ℓ := Nat.one_le_iff_ne_zero.mpr (by intro h; simp [h] at hgs)
    have hce_ge : fs.groupSize ℓ ≤ countEqual fs v k + 1 :=
      calc fs.groupSize ℓ = (fs.groupSize ℓ - 1) + 1 := (Nat.sub_add_cancel hge1).symm
        _ ≤ countEqual fs v k + 1 := Nat.add_le_add_right hle 1
    have hcast : (fs.groupSize ℓ : ℝ) ≤ (countEqual fs v k : ℝ) + 1 := by exact_mod_cast hce_ge
    linarith
  -- gap = countBelow(j) + (countEqual(j)+1)/2 - countBelow(k) - (countEqual(k)+1)/2
  --     ≥ countEqual(k) + (countEqual(j)+1)/2 - (countEqual(k)+1)/2
  --     = countEqual(k)/2 + countEqual(j)/2
  --     ≥ (m-1)/2 + 1/2 = m/2
  linarith

/-- Tighter Σd² bound: Σd² ≥ m²/2 (instead of (m-1)²/2).
    Uses the tighter midrank gap of m/2 in each model. -/
theorem sumSqRankDiff_ge_sq_groupSize_tight (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k) :
    (fs.groupSize ℓ : ℝ) ^ 2 / 2 ≤
      sumSqRankDiff fs (fun i => attribution fs i f) (fun i => attribution fs i f') := by
  set v := fun i => attribution fs i f
  set w := fun i => attribution fs i f'
  have hrev := attribution_reversal fs f f' j k ℓ hj hk hfm hfm' hjk
  -- Tighter midrank gaps: ≥ m/2 in each model
  have hgap_v := midrank_gap_ge_half_groupSize_tight fs f j k ℓ hj hk hjk hfm
  have hgap_w := midrank_gap_ge_half_groupSize_tight fs f' k j ℓ hk hj hjk.symm hfm'
  set dj := midrank fs v j - midrank fs w j
  set dk := midrank fs v k - midrank fs w k
  -- d_j - d_k ≥ m/2 + m/2 = m
  have hdiff : dj - dk ≥ (fs.groupSize ℓ : ℝ) := by
    show midrank fs v j - midrank fs w j - (midrank fs v k - midrank fs w k) ≥ _
    have := hgap_v
    have := hgap_w
    linarith
  -- Σd² ≥ d_j² + d_k² ≥ (d_j - d_k)²/2 ≥ m²/2
  unfold sumSqRankDiff
  have hpair : dj ^ 2 + dk ^ 2 ≤
      Finset.univ.sum (fun i => (midrank fs v i - midrank fs w i) ^ 2) := by
    have hsub : {j, k} ⊆ Finset.univ (α := Fin fs.P) := Finset.subset_univ _
    have hne : j ≠ k := hjk
    have hpair_sum := Finset.sum_le_sum_of_subset_of_nonneg hsub
      (fun (i : Fin fs.P) (_ : i ∈ Finset.univ) (_ : i ∉ ({j, k} : Finset (Fin fs.P))) =>
        sq_nonneg (midrank fs v i - midrank fs w i))
    rw [Finset.sum_pair hne] at hpair_sum
    exact hpair_sum
  have hsq : dj ^ 2 + dk ^ 2 ≥ (dj - dk) ^ 2 / 2 := by nlinarith [sq_nonneg (dj + dk)]
  have hm := fs.group_size_ge_two ℓ
  have hm_cast : (fs.groupSize ℓ : ℝ) ≥ 2 := by
    unfold FeatureSpace.groupSize
    exact_mod_cast hm
  have hdiffsq : (dj - dk) ^ 2 ≥ (fs.groupSize ℓ : ℝ) ^ 2 := by nlinarith
  linarith

/-- Tighter Spearman bound: spearmanCorr ≤ 1 - 3m²/(P³-P).
    Strictly tighter than 1 - 3(m-1)²/(P³-P) by a factor of m²/(m-1)². -/
theorem spearmanCorr_bound_groupSize_tight (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k)
    (hP : 2 ≤ fs.P) :
    spearmanCorr fs (fun i => attribution fs i f) (fun i => attribution fs i f') ≤
      1 - 3 * (fs.groupSize ℓ : ℝ) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) := by
  unfold spearmanCorr
  have hsd := sumSqRankDiff_ge_sq_groupSize_tight fs f f' j k ℓ hj hk hjk hfm hfm'
  have hP_pos : (0 : ℝ) < (fs.P : ℝ) := Nat.cast_pos.mpr fs.hP
  have hP2 : (1 : ℝ) < (fs.P : ℝ) := by exact_mod_cast (show 1 < fs.P by omega)
  have hPsq : (0 : ℝ) < (fs.P : ℝ) ^ 2 - 1 := by nlinarith
  have hdenom_pos : (0 : ℝ) < (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) :=
    mul_pos hP_pos hPsq
  have hfactor : (fs.P : ℝ) * ((fs.P : ℝ) ^ 2 - 1) = (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by ring
  rw [hfactor]
  have hdenom_pos' : (0 : ℝ) < (fs.P : ℝ) ^ 3 - (fs.P : ℝ) := by linarith [hfactor]
  suffices h : 3 * (fs.groupSize ℓ : ℝ) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) ≤
               6 * sumSqRankDiff fs (fun i => attribution fs i f)
                 (fun i => attribution fs i f') /
               ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) by linarith
  rw [div_le_div_iff_of_pos_right hdenom_pos']
  linarith

/-- Tighter Spearman instability bound (convenience wrapper):
    spearmanCorr ≤ 1 - 3m²/(P³-P) when first-movers differ within group ℓ. -/
theorem spearman_instability_bound_tight (f f' : Model) (ℓ : Fin fs.L)
    (hfm_grp : firstMover fs f ∈ fs.group ℓ)
    (hfm'_grp : firstMover fs f' ∈ fs.group ℓ)
    (hdiff : firstMover fs f ≠ firstMover fs f')
    (hP : 2 ≤ fs.P) :
    spearmanCorr fs (fun j => attribution fs j f) (fun j => attribution fs j f') ≤
      1 - 3 * (fs.groupSize ℓ : ℝ) ^ 2 / ((fs.P : ℝ) ^ 3 - (fs.P : ℝ)) := by
  exact spearmanCorr_bound_groupSize_tight fs f f' (firstMover fs f) (firstMover fs f')
    ℓ hfm_grp hfm'_grp hdiff rfl rfl hP

end UniversalImpossibility
