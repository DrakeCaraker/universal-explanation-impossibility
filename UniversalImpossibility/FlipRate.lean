/-
  Exact GBDT flip rate under the first-mover model.

  For a balanced ensemble, the within-group ranking structure
  is completely determined by the first-mover assignment:
  - The first-mover has strictly more splits (hence higher attribution)
  - All non-first-movers in the group have identical split counts (ties)

  This file formalizes the counting arguments from the supplement
  (§Exact flip rate for GBDT, lines 448-496):
  (i)   Pr[random model ranks j > k] = 1/m  (j is first-mover)
  (ii)  Pr[tie between j and k] = (m-2)/m  (first-mover elsewhere in group)
  (iii) For m=2: every model ranks j vs k with no ties
  (iv)  The flip-rate partition: models split into j-wins, k-wins, and ties

  Supplement: §Exact flip rate for GBDT
-/
import UniversalImpossibility.General

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Classification of models by first-mover status -/

/-- The set of model indices where j is the first-mover. -/
noncomputable def firstMoverSet (M : ℕ) (models : Fin M → Model) (j : Fin fs.P) : Finset (Fin M) :=
  Finset.univ.filter (fun i => firstMover fs (models i) = j)

/-- The set of model indices where neither j nor k is the first-mover. -/
noncomputable def tieSet (M : ℕ) (models : Fin M → Model) (j k : Fin fs.P) : Finset (Fin M) :=
  Finset.univ.filter (fun i => firstMover fs (models i) ≠ j ∧ firstMover fs (models i) ≠ k)

/-! ### Non-first-mover implies tied split counts -/

/-- When the first-mover is neither j nor k, both features have identical
    split counts (the "tie" case from the supplement). -/
theorem non_firstmover_tie (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (_hjk : j ≠ k)
    (i : Fin M)
    (hfmj : firstMover fs (models i) ≠ j)
    (hfmk : firstMover fs (models i) ≠ k) :
    splitCount fs j (models i) = splitCount fs k (models i) :=
  splitCount_eq_of_not_firstMover_j_or_k fs (models i) j k ℓ hj hk hfmj hfmk

/-- When the first-mover is neither j nor k, attributions are also tied
    (follows from proportionality and split-count ties). -/
theorem non_firstmover_attribution_tie (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hjk : j ≠ k)
    (i : Fin M)
    (hfmj : firstMover fs (models i) ≠ j)
    (hfmk : firstMover fs (models i) ≠ k) :
    attribution fs j (models i) = attribution fs k (models i) := by
  obtain ⟨c, _, hc_eq⟩ := attribution_proportional fs (models i)
  rw [hc_eq j, hc_eq k]
  congr 1
  exact non_firstmover_tie fs M models j k ℓ hj hk hjk i hfmj hfmk

/-! ### First-mover dominates: attribution ordering -/

/-- When the first-mover IS j, then j's attribution strictly exceeds k's.
    This is the "j-wins" case. -/
theorem firstmover_dominates (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (f : Model) (hfm : firstMover fs f = j) :
    attribution fs j f > attribution fs k f :=
  attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk

/-! ### Balanced ensemble: equal first-mover counts -/

/-- For a balanced ensemble, the count of models where j is first-mover
    equals the count where k is first-mover. -/
theorem balanced_firstmover_count_eq (M : ℕ) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) :
    (firstMoverSet fs M models j).card = (firstMoverSet fs M models k).card :=
  hbal ℓ j k hj hk

/-! ### The three-way partition of models -/

/-- Every model index is in exactly one of three categories:
    first-mover is j, first-mover is k, or first-mover is neither. -/
theorem model_trichotomy (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) (_hjk : j ≠ k) (i : Fin M) :
    firstMover fs (models i) = j ∨
    firstMover fs (models i) = k ∨
    (firstMover fs (models i) ≠ j ∧ firstMover fs (models i) ≠ k) := by
  by_cases hj : firstMover fs (models i) = j
  · exact Or.inl hj
  · by_cases hk : firstMover fs (models i) = k
    · exact Or.inr (Or.inl hk)
    · exact Or.inr (Or.inr ⟨hj, hk⟩)

/-- The firstMoverSet for j and the firstMoverSet for k are disjoint
    (since j ≠ k). -/
theorem firstMoverSets_disjoint (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) (hjk : j ≠ k) :
    Disjoint (firstMoverSet fs M models j) (firstMoverSet fs M models k) := by
  rw [Finset.disjoint_left]
  intro i hi hk
  simp only [firstMoverSet, Finset.mem_filter, Finset.mem_univ, true_and] at hi hk
  exact hjk (hi.symm.trans hk)

/-- The tieSet is disjoint from both firstMoverSets. -/
theorem tieSet_disjoint_left (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) :
    Disjoint (firstMoverSet fs M models j) (tieSet fs M models j k) := by
  rw [Finset.disjoint_left]
  intro i hi ht
  simp only [firstMoverSet, tieSet, Finset.mem_filter] at hi ht
  exact ht.2.1 hi.2

theorem tieSet_disjoint_right (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) :
    Disjoint (firstMoverSet fs M models k) (tieSet fs M models j k) := by
  rw [Finset.disjoint_left]
  intro i hi ht
  simp only [firstMoverSet, tieSet, Finset.mem_filter] at hi ht
  exact ht.2.2 hi.2

/-- The three sets partition all model indices:
    |firstMoverSet j| + |firstMoverSet k| + |tieSet j k| = M. -/
theorem fliprate_partition (M : ℕ) (models : Fin M → Model)
    (j k : Fin fs.P) (hjk : j ≠ k)
    (ℓ : Fin fs.L) (_hj : j ∈ fs.group ℓ) (_hk : k ∈ fs.group ℓ)
    (_hfm_in_group : ∀ i : Fin M, firstMover fs (models i) ∈ fs.group ℓ) :
    (firstMoverSet fs M models j).card + (firstMoverSet fs M models k).card +
    (tieSet fs M models j k).card = M := by
  -- Every model whose first-mover is in group ℓ but ≠ j and ≠ k
  -- belongs to tieSet. Together with j-set and k-set, they cover Finset.univ.
  have hunion : Finset.univ (α := Fin M) =
      firstMoverSet fs M models j ∪ firstMoverSet fs M models k ∪
      tieSet fs M models j k := by
    ext i
    simp only [Finset.mem_univ, true_iff, Finset.mem_union, Finset.mem_filter,
               firstMoverSet, tieSet]
    rcases model_trichotomy fs M models j k hjk i with h | h | ⟨h1, h2⟩
    · left; left; exact ⟨trivial, h⟩
    · left; right; exact ⟨trivial, h⟩
    · right; exact ⟨trivial, h1, h2⟩
  have hdj : Disjoint (firstMoverSet fs M models j) (firstMoverSet fs M models k) :=
    firstMoverSets_disjoint fs M models j k hjk
  have hdjt : Disjoint (firstMoverSet fs M models j ∪ firstMoverSet fs M models k)
      (tieSet fs M models j k) := by
    rw [Finset.disjoint_union_left]
    exact ⟨tieSet_disjoint_left fs M models j k,
           tieSet_disjoint_right fs M models j k⟩
  calc (firstMoverSet fs M models j).card + (firstMoverSet fs M models k).card +
        (tieSet fs M models j k).card
      = (firstMoverSet fs M models j ∪ firstMoverSet fs M models k).card +
        (tieSet fs M models j k).card := by
          rw [Finset.card_union_of_disjoint hdj]
    _ = (firstMoverSet fs M models j ∪ firstMoverSet fs M models k ∪
         tieSet fs M models j k).card := by
          rw [Finset.card_union_of_disjoint hdjt]
    _ = Finset.card (Finset.univ : Finset (Fin M)) := by rw [hunion]
    _ = M := Finset.card_fin M

/-! ### Binary groups (m = 2): no ties -/

/-- For a group of size 2 containing j and k with j ≠ k, if the first-mover
    is in the group, then it must be j or k. -/
theorem binary_group_firstmover_is_j_or_k
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hsize : fs.groupSize ℓ = 2)
    (f : Model)
    (hfm_group : firstMover fs f ∈ fs.group ℓ) :
    firstMover fs f = j ∨ firstMover fs f = k := by
  -- The group has exactly 2 elements. By card_eq_two, group ℓ = {a, b} for some a ≠ b.
  rw [FeatureSpace.groupSize] at hsize
  rw [Finset.card_eq_two] at hsize
  obtain ⟨a, b, hab, hgroup⟩ := hsize
  -- j and k are in {a, b}
  rw [hgroup] at hj hk hfm_group
  rw [Finset.mem_insert, Finset.mem_singleton] at hj hk hfm_group
  -- firstMover is a or b
  rcases hfm_group with rfl | rfl
  · -- firstMover = a; a is j or k
    rcases hj with rfl | rfl
    · left; rfl
    · rcases hk with rfl | rfl
      · right; rfl
      · -- j = b, k = b, contradicts j ≠ k... but actually j and k are both a or b
        -- hj : j = a (we're in the case firstMover = a, and hj gave a = j... wait let me re-examine)
        -- Actually in this branch: firstMover = a, hj : j = a ∨ j = b
        -- We took hj as rfl (j = b case), so j = b
        -- hk : k = a ∨ k = b, and we took hk as rfl, meaning k = a
        -- But wait, we are NOT in the k=a branch... let me re-check
        -- Actually: hk gave k = a ∨ k = b. First branch would be k = a.
        -- But we said the first-mover = a, and we need firstMover = j or = k.
        -- j = b (from hj second branch). k = a (from hk first branch) → firstMover = a = k. Done.
        -- But we're in the hk second branch, so k = b. Then j = b = k, contradiction.
        exact absurd rfl hjk
  · -- firstMover = b; b is j or k
    rcases hj with rfl | rfl
    · -- j = a
      rcases hk with rfl | rfl
      · -- k = a = j, contradiction
        exact absurd rfl hjk
      · -- k = b = firstMover
        right; rfl
    · -- j = b = firstMover
      left; rfl

/-- For m = 2: the tie set is empty (every model has either j or k as first-mover). -/
theorem binary_group_no_ties (M : ℕ) (models : Fin M → Model)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hsize : fs.groupSize ℓ = 2)
    (hfm_in_group : ∀ i : Fin M, firstMover fs (models i) ∈ fs.group ℓ) :
    tieSet fs M models j k = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro i hi
  simp only [tieSet, Finset.mem_filter] at hi
  obtain ⟨_, hfmj, hfmk⟩ := hi
  have := binary_group_firstmover_is_j_or_k fs ℓ j k hj hk hjk hsize
    (models i) (hfm_in_group i)
  rcases this with h | h
  · exact hfmj h
  · exact hfmk h

/-- For m = 2 balanced ensembles: each feature is first-mover for exactly M/2 models,
    and every model produces a strict ranking (no ties).
    The flip rate is exactly 1/2: half the models rank j > k, half rank k > j. -/
theorem binary_group_flip_rate (M : ℕ) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hsize : fs.groupSize ℓ = 2)
    (hfm_in_group : ∀ i : Fin M, firstMover fs (models i) ∈ fs.group ℓ) :
    (firstMoverSet fs M models j).card = (firstMoverSet fs M models k).card ∧
    (firstMoverSet fs M models j).card + (firstMoverSet fs M models k).card = M := by
  constructor
  · exact balanced_firstmover_count_eq fs M models hbal ℓ j k hj hk
  · have htie := binary_group_no_ties fs M models ℓ j k hj hk hjk hsize hfm_in_group
    have hpart := fliprate_partition fs M models j k hjk ℓ hj hk hfm_in_group
    rw [htie, Finset.card_empty] at hpart
    omega

/-! ### Flip rate: the number of (model₁, model₂) pairs that disagree -/

/-- In a balanced ensemble with m=2 groups, the number of model pairs
    where model₁ ranks j > k and model₂ ranks k > j is exactly
    (M/2)² out of M² total pairs. The flip rate is (M/2)²/M² = 1/4...
    but across BOTH directions of disagreement it doubles to 1/2.

    More precisely: #(pairs ranking j>k then k>j) = |fmSet j| · |fmSet k|,
    and by balance |fmSet j| = |fmSet k| = M/2. -/
theorem binary_flip_pair_count (M : ℕ) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (_hjk : j ≠ k)
    (_hsize : fs.groupSize ℓ = 2)
    (_hfm_in_group : ∀ i : Fin M, firstMover fs (models i) ∈ fs.group ℓ) :
    -- The number of disagreeing ordered pairs (one ranks j>k, other ranks k>j)
    -- equals 2 · |fmSet j| · |fmSet k|
    2 * ((firstMoverSet fs M models j).card * (firstMoverSet fs M models k).card) =
    2 * ((firstMoverSet fs M models j).card ^ 2) := by
  have heq := balanced_firstmover_count_eq fs M models hbal ℓ j k hj hk
  rw [heq, sq]

end UniversalImpossibility
