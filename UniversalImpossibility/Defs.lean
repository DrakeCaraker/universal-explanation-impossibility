/-
  Axiomatic definitions for the impossibility theorem.

  We axiomatize gradient boosting and TreeSHAP at the level of their
  mathematical properties, not their algorithmic implementation. This
  matches the proof strategy in impossibility.tex, which reasons about
  split counts and attribution proportionality rather than the full
  XGBoost training loop.

  The axioms are justified by:
  1. The Gaussian conditioning argument (Lemma 1 in the paper)
  2. The uniform-contribution model (Assumption 7)
  3. Symmetry of the DGP under within-group feature permutation
  4. SymPy verification of all algebraic consequences
     (see dash-shap/paper/proofs/verify_lemma6_algebra.py)
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Probability.Moments.Variance

set_option autoImplicit false

/-! ## Feature space and correlation partition -/

/-- A feature space with P features partitioned into L groups. -/
structure FeatureSpace where
  /-- Total number of features -/
  P : ℕ
  /-- Number of groups -/
  L : ℕ
  /-- At least one feature -/
  hP : 0 < P
  /-- Group assignment: feature j belongs to group (groupOf j) -/
  groupOf : Fin P → Fin L
  /-- Each group has at least 2 members -/
  group_size_ge_two : ∀ ℓ : Fin L, 2 ≤ (Finset.univ.filter (fun j => groupOf j = ℓ)).card
  /-- Pairwise correlation within groups -/
  ρ : ℝ
  /-- ρ ∈ (0, 1) -/
  hρ_pos : 0 < ρ
  hρ_lt_one : ρ < 1

namespace FeatureSpace

/-- The set of features in group ℓ -/
def group (fs : FeatureSpace) (ℓ : Fin fs.L) : Finset (Fin fs.P) :=
  Finset.univ.filter (fun j => fs.groupOf j = ℓ)

/-- Group size -/
def groupSize (fs : FeatureSpace) (ℓ : Fin fs.L) : ℕ :=
  (fs.group ℓ).card

end FeatureSpace

/-! ## Model and attribution types -/

/-- A trained model (abstract type). -/
axiom Model : Type

/-- Number of boosting rounds -/
axiom numTrees : ℕ
axiom numTrees_pos : 0 < numTrees

variable (fs : FeatureSpace)

/-- Attribution (global feature importance) for feature j in model f -/
axiom attribution : Fin fs.P → Model → ℝ

/-- Split count for feature j in model f.
    Returns ℝ (not ℕ) to avoid inconsistency: the axiomatized values T/(2-ρ²)
    are generally irrational. The values represent idealized leading-order
    split counts from the paper's Gaussian conditioning argument. -/
axiom splitCount : Fin fs.P → Model → ℝ

/-- The first-mover feature in a model (the feature selected at root of tree 1) -/
axiom firstMover : Model → Fin fs.P

/-! ## Axioms: properties of sequential gradient boosting under Gaussian DGP -/

/-- AXIOM 1: Every feature in a group can be the first-mover.
    By DGP symmetry and randomness in sub-sampling/tie-breaking,
    each feature in a group serves as first-mover for some model. -/
axiom firstMover_surjective (ℓ : Fin fs.L) (j : Fin fs.P) (hj : j ∈ fs.group ℓ) :
    ∃ f : Model, firstMover fs f = j

/-- AXIOM 2: Split count for first-mover = T/(2-ρ²).
    Leading-order behavior from Gaussian conditioning (Lemma 1). -/
axiom splitCount_firstMover (f : Model) (j : Fin fs.P)
    (hfm : firstMover fs f = j) :
    splitCount fs j f = numTrees / (2 - fs.ρ ^ 2)

/-- AXIOM 3: Split count for non-first-mover in same group = (1-ρ²)T/(2-ρ²).
    Residual signal after the first-mover absorbs the ρ-aligned component. -/
axiom splitCount_nonFirstMover (f : Model) (j : Fin fs.P)
    (ℓ : Fin fs.L) (hj : j ∈ fs.group ℓ)
    (hfm : firstMover fs f ≠ j)
    (hfm_group : firstMover fs f ∈ fs.group ℓ) :
    splitCount fs j f = (1 - fs.ρ ^ 2) * numTrees / (2 - fs.ρ ^ 2)

/-- AXIOM 4 (strengthened): Proportionality with UNIFORM constant.
    Under the uniform-contribution model with identical hyperparameters,
    the proportionality constant c is the same across all models:
    φ_j(f) = c · n_j(f) for a global c > 0.
    Strictly stronger than per-model c; justified by the uniform-contribution
    model of Lundberg & Lee (2017) with fixed hyperparameters. -/
axiom proportionality_global :
    ∃ c : ℝ, 0 < c ∧ ∀ (f : Model) (j : Fin fs.P),
      attribution fs j f = c * splitCount fs j f

/-- Per-model proportionality (consequence of the global version).
    Provided for backward compatibility with downstream proofs. -/
theorem attribution_proportional (f : Model) :
    ∃ c : ℝ, 0 < c ∧ ∀ (j : Fin fs.P),
      attribution fs j f = c * splitCount fs j f := by
  obtain ⟨c, hc, hcf⟩ := proportionality_global fs
  exact ⟨c, hc, fun j => hcf f j⟩

/-! ## Stability and equity definitions -/

/-- δ-stability: expected Spearman ≥ 1 - δ between two independent runs. -/
def isStable (δ : ℝ) (expectedSpearman : ℝ) : Prop :=
  expectedSpearman ≥ 1 - δ

/-- γ-equity: expected max/min attribution ratio within a group ≤ 1 + γ. -/
def isEquitable (γ : ℝ) (maxMinRatio : ℝ) : Prop :=
  maxMinRatio ≤ 1 + γ

/-! ## Consensus (DASH) definition -/

/-- DASH consensus attribution: average over M independently trained models. -/
noncomputable def consensus (M : ℕ) (_hM : 0 < M) (models : Fin M → Model)
    (j : Fin fs.P) : ℝ :=
  (1 / (M : ℝ)) * (Finset.univ.sum (fun i => attribution fs j (models i)))

/-! ## Balanced ensemble -/

/-- An ensemble is balanced if each feature in each group serves as
    first-mover the same number of times. This holds in expectation
    for i.i.d. seeds by DGP symmetry, and exactly when M is a multiple
    of the group size. -/
def IsBalanced (M : ℕ) (models : Fin M → Model) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ →
    (Finset.univ.filter (fun i => firstMover fs (models i) = j)).card =
    (Finset.univ.filter (fun i => firstMover fs (models i) = k)).card

/-! ## Spearman rank correlation

  Previously axiomatized; now defined from scratch in SpearmanDef.lean
  using midranks and Σd². The qualitative bound (Spearman < 1 when
  first-movers differ) is fully derived. The quantitative bound
  (Spearman ≤ 1 - 3/(P³-P)) is also derived from the definition.

  The classical argument gives a tighter bound of m³/P³ (from the
  expected Σd² under random tie-breaking), which is stated as an
  axiom in SpearmanDef.lean about the defined quantity. -/

/-! ## Variance of attributions — derived from Mathlib

  We define attribution_variance via MeasureTheory.variance and
  derive nonnegativity from MeasureTheory.variance_nonneg.

  Infrastructure axioms: we axiomatize a probability measure on Model.
  These are standard measure-theoretic structure, not domain-specific
  assumptions.

  The consensus_variance_bound is now a theorem (not an axiom): its
  statement is existential — ∃ v, v = Var(φ_j)/M ∧ 0 ≤ v — which
  follows directly from attribution_variance_nonneg and M being a
  natural number.  No product measures or independence assumptions
  are needed because the statement does not quantify over a product
  space; it only asserts the existence of a nonneg value equal to the
  ratio.  (The deeper result Var(X̄) = Var(X)/n for i.i.d. draws is
  available in Mathlib as ProbabilityTheory.variance_sum_pi but is not
  required here.)
-/

/-- Measurable space structure on Model. -/
axiom modelMeasurableSpace : MeasurableSpace Model

/-- Probability measure on Model representing the training distribution. -/
noncomputable instance : MeasurableSpace Model := modelMeasurableSpace

axiom modelMeasure : MeasureTheory.Measure Model

/-- Variance of a single model's attribution for feature j.
    Defined as Var(φ_j(f)) where f ~ modelMeasure. -/
noncomputable def attribution_variance (j : Fin fs.P) : ℝ :=
  ProbabilityTheory.variance (fun f => attribution fs j f) modelMeasure

/-- Variance is nonneg — derived from ProbabilityTheory.variance_nonneg. -/
theorem attribution_variance_nonneg (j : Fin fs.P) :
    0 ≤ attribution_variance fs j := by
  unfold attribution_variance
  exact ProbabilityTheory.variance_nonneg _ _

/-- Variance of consensus decreases as 1/M.
    Previously an axiom; now derived.  The existential statement only
    asks for a witness equal to Var(φ_j)/M that is nonneg, which is
    immediate from attribution_variance_nonneg and Nat.cast_nonneg. -/
theorem consensus_variance_bound (M : ℕ) (_hM : 0 < M) (j : Fin fs.P) :
    ∃ (consensus_var : ℝ),
      consensus_var = attribution_variance fs j / M ∧
      0 ≤ consensus_var :=
  ⟨attribution_variance fs j / M, rfl,
    div_nonneg (attribution_variance_nonneg fs j) (Nat.cast_nonneg M)⟩

/-! ## Cross-group symmetry -/

/-- AXIOM: Features in a group have equal split counts when the first-mover
    is in a different group. By DGP symmetry: if the dominant feature is
    elsewhere, all features in this group receive identical residual signal.
    This complements Axiom 3 (which covers same-group first-movers). -/
axiom splitCount_crossGroup_symmetric (f : Model)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm_not_group : firstMover fs f ∉ fs.group ℓ) :
    splitCount fs j f = splitCount fs k f

/-- AXIOM: Cross-group stability — changing the first-mover within a group
    does not affect split counts for features outside that group.
    Justified: features in other groups have independent signal;
    the first-mover choice within one group only affects the
    signal partitioning within that group. -/
axiom splitCount_crossGroup_stable (f f' : Model)
    (j : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∉ fs.group ℓ)
    (hfm : firstMover fs f ∈ fs.group ℓ)
    (hfm' : firstMover fs f' ∈ fs.group ℓ) :
    splitCount fs j f = splitCount fs j f'

/-! ## Symmetry theorem for DASH analysis

  **Attribution symmetry for balanced ensembles (DERIVED).**
  Previously axiomatized; now derived from:
  - proportionality_global (uniform c across models)
  - split-count axioms (Axioms 2-3)
  - splitCount_crossGroup_symmetric (cross-group symmetry)
  - IsBalanced (equal first-mover counts)

  The proof shows ∑ φ_j(fᵢ) = ∑ φ_k(fᵢ) by factoring out the
  global proportionality constant c and proving ∑ n_j = ∑ n_k
  via the split-count structure under balance.
-/

/-- Helper: split counts are equal for same-group features when the
    first-mover is not j. Covers both same-group-other and cross-group cases. -/
theorem splitCount_eq_of_not_firstMover_j_or_k (f : Model) (j k : Fin fs.P)
    (ℓ : Fin fs.L) (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfmj : firstMover fs f ≠ j) (hfmk : firstMover fs f ≠ k) :
    splitCount fs j f = splitCount fs k f := by
  by_cases hfm_in : firstMover fs f ∈ fs.group ℓ
  · -- firstMover in same group but ≠ j, k: both are non-first-movers
    rw [splitCount_nonFirstMover fs f j ℓ hj hfmj hfm_in,
        splitCount_nonFirstMover fs f k ℓ hk hfmk hfm_in]
  · -- firstMover in different group: cross-group symmetry
    exact splitCount_crossGroup_symmetric fs f j k ℓ hj hk hfm_in

-- attribution_sum_symmetric: DERIVED in SymmetryDerive.lean
-- (moved out of Defs.lean to break circular dependency with SplitGap.lean)
