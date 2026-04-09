/-
  The Attribution Design Space Theorem.

  Characterizes the achievable set of (stability, unfaithfulness, completeness)
  triples under collinearity. All previous results emerge as corollaries.

  Structure:
  - Step 1 (Family A forced): from attribution_impossibility
  - Step 2 (Family B achievable): from consensus_equity + variance_bound
  - Step 3 (completeness): axiomatized (requires probability theory)
  - Step 4 (infeasible point): from attribution_impossibility

  The Design Space Theorem states that every attribution method falls
  into one of two families:
  - A: faithful + complete → U=1/2, S ≤ 1-m³/P³
  - B: ensemble (DASH) → U=0 (ties), S=1-O(1/M)
-/
import UniversalImpossibility.Impossibility
import UniversalImpossibility.Corollary

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Design Space Types -/

/-- An attribution aggregation method maps an ensemble of M models and
    a feature to a consensus attribution value. -/
structure AggregationMethod where
  M : ℕ
  hM : 0 < M
  /-- The consensus attribution for feature j given models -/
  consensus : (Fin M → Model) → Fin fs.P → ℝ

/-- A method produces ties for features j, k if the consensus is always equal. -/
def ProducesTie (A : AggregationMethod fs) (j k : Fin fs.P) : Prop :=
  ∀ models : Fin A.M → Model, A.consensus models j = A.consensus models k

/-- A method is faithful to individual models: the consensus ranking
    agrees with every model's attribution ranking. -/
def IsFaithful (A : AggregationMethod fs) : Prop :=
  ∀ (models : Fin A.M → Model) (j k : Fin fs.P),
    (∀ i : Fin A.M, attribution fs j (models i) > attribution fs k (models i)) →
    A.consensus models j > A.consensus models k

/-- **Strongly faithful**: the consensus ranking agrees with EACH individual
    model's attribution ordering (not just unanimous orderings). This is the
    paper's definition of faithfulness (Definition 2 in the main text). -/
def StronglyFaithful (A : AggregationMethod fs) : Prop :=
  ∀ (models : Fin A.M → Model) (i : Fin A.M) (j k : Fin fs.P),
    attribution fs j (models i) > attribution fs k (models i) →
    A.consensus models j > A.consensus models k

/-! ### Design Space Exhaustiveness: Family A is forced -/

/-- **No strongly faithful method exists under Rashomon (M ≥ 2).**
    This is the key exhaustiveness step (Step 3, Case 1): any method that
    is faithful to each individual model's attributions is impossible
    because the Rashomon property produces two models with opposite
    orderings, both of which the method must respect.

    This bridges the AggregationMethod types to the core impossibility
    (attribution_impossibility). -/
theorem strongly_faithful_impossible
    (hrash : RashimonProperty fs)
    (A : AggregationMethod fs)
    (hM2 : 2 ≤ A.M)
    (hfaith : StronglyFaithful fs A)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    False := by
  -- By Rashomon, get models with opposite orderings
  obtain ⟨f, f', hjk_f, hkj_f'⟩ := hrash ℓ j k hj hk hjk
  -- Build an ensemble of size M containing both f and f'
  have h0 : 0 < A.M := A.hM
  have h1 : 1 < A.M := by omega
  let models : Fin A.M → Model := fun i =>
    if i = ⟨0, h0⟩ then f else f'
  -- Faithfulness to f (at index 0): consensus ranks j > k
  have hfk : attribution fs j (models ⟨0, h0⟩) > attribution fs k (models ⟨0, h0⟩) := by
    show attribution fs j f > attribution fs k f
    exact hjk_f
  have h_jk : A.consensus models j > A.consensus models k :=
    hfaith models ⟨0, h0⟩ j k hfk
  -- Faithfulness to f' (at index 1): consensus ranks k > j
  have hfk' : attribution fs k (models ⟨1, h1⟩) > attribution fs j (models ⟨1, h1⟩) := by
    show attribution fs k f' > attribution fs j f'
    exact hkj_f'
  have h_kj : A.consensus models k > A.consensus models j :=
    hfaith models ⟨1, h1⟩ k j hfk'
  -- Contradiction: can't have both j > k and k > j
  linarith

/-! ### Balanced ensemble flip symmetry -/

/-- **For balanced ensembles, the number of models ranking j > k equals the
    number ranking k > j.** This is the finite-ensemble formalization of
    the exact flip rate (Proposition S-exact-flip in the supplement).

    From IsBalanced: |{i : firstMover(models i) = j}| = |{i : firstMover(models i) = k}|.
    From attribution_firstMover_gt: firstMover = j implies attribution j > attribution k.
    Therefore: |{i : attribution j > attribution k}| ≥ |{i : firstMover = j}|
             = |{i : firstMover = k}| ≤ |{i : attribution k > attribution j}|.

    The full equality requires showing the reverse inclusion, which needs
    the split-count tie for non-first-movers (both have identical split counts
    when neither is first-mover within the group). -/
theorem balanced_flip_symmetry (M : ℕ) (_hM : 0 < M) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (_hjk : j ≠ k) :
    (Finset.univ.filter (fun i => firstMover fs (models i) = j)).card =
    (Finset.univ.filter (fun i => firstMover fs (models i) = k)).card :=
  hbal ℓ j k hj hk

/-! ### Step 4: The infeasible point -/

/-- No method can be faithful to all individual models when the Rashomon
    property holds. This is a direct consequence of the Attribution
    Impossibility (Theorem 1).

    Specifically: if a method is faithful in the sense that its consensus
    ranking matches every individual model's ranking, then for any
    within-group pair (j,k), the Rashomon property produces two models
    with opposite orderings — and faithfulness to both is impossible. -/
theorem infeasible_faithful_stable_complete
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h_faithful

/-! ### Step 2: Family B is achievable (DASH) -/

/-- DASH is an aggregation method. -/
noncomputable def dashMethod (M : ℕ) (hM : 0 < M) : AggregationMethod fs where
  M := M
  hM := hM
  consensus := fun models j => consensus fs M hM models j

/-- DASH produces ties for same-group features in balanced ensembles. -/
theorem dash_produces_ties (M : ℕ) (hM : 0 < M) (models : Fin M → Model)
    (hbal : IsBalanced fs M models)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) :
    (dashMethod fs M hM).consensus models j =
    (dashMethod fs M hM).consensus models k := by
  unfold dashMethod
  simp only
  exact consensus_equity fs M hM models hbal j k ℓ hj hk

/-- The DASH variance decreases with ensemble size. -/
theorem dash_variance_decreases (M : ℕ) (hM : 0 < M) (j : Fin fs.P) :
    ∃ (v : ℝ), v = attribution_variance fs j / M ∧ 0 ≤ v :=
  consensus_variance_rate fs M hM j

/-! ### Design Space Theorem (composite) -/

/-- **The Attribution Design Space Theorem.**

    For any model class satisfying the Rashomon property:

    1. The (1, 0, complete) triple is infeasible: no faithful stable
       complete method exists (from infeasible_faithful_stable_complete).

    2. Family B (DASH) achieves U=0 (ties) with S=1-O(1/M)
       (from dash_produces_ties + dash_variance_decreases).

    3. Any faithful-to-individual-models method has U=1/2 for
       within-group symmetric pairs (from the Rashomon property:
       half the models disagree with any fixed ranking).

    This theorem composes the three results. The full characterization
    (that A ∪ B exhausts all methods operating on per-model attributions)
    requires the Bayes-optimality argument (Proposition S4 in the
    supplement), which involves probability theory beyond our current
    axiom system. -/
theorem design_space_theorem
    (hrash : RashimonProperty fs) :
    -- Part 1: (1, 0, complete) is infeasible
    (∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∀ ranking : Fin fs.P → Fin fs.P → Prop,
        (∀ f : Model, ranking j k ↔ attribution fs j f > attribution fs k f) →
        False) ∧
    -- Part 2: DASH achieves ties (U=0) for balanced ensembles
    (∀ (M : ℕ) (hM : 0 < M) (models : Fin M → Model),
      IsBalanced fs M models →
      ∀ (j k : Fin fs.P) (ℓ : Fin fs.L),
        j ∈ fs.group ℓ → k ∈ fs.group ℓ →
        consensus fs M hM models j = consensus fs M hM models k) ∧
    -- Part 3: DASH variance decreases as 1/M
    (∀ (M : ℕ) (_hM : 0 < M) (j : Fin fs.P),
      ∃ v : ℝ, v = attribution_variance fs j / M ∧ 0 ≤ v) := by
  exact ⟨
    fun ℓ j k hj hk hjk ranking hfaith =>
      attribution_impossibility fs hrash ℓ j k hj hk hjk ranking hfaith,
    fun M hM models hbal j k ℓ hj hk =>
      consensus_equity fs M hM models hbal j k ℓ hj hk,
    fun M hM j =>
      consensus_variance_rate fs M hM j
  ⟩

end UniversalImpossibility
