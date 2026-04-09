/-
  BayesOptimalTie.lean — Bayes-optimality of ties for symmetric features.

  Under DGP symmetry + non-degeneracy, any committed ranking (j > k or k > j)
  has disagreement probability exactly 1/2, while a tie has disagreement
  probability 0. Therefore ties strictly dominate commitments, and the
  Bayes-optimal ranking for symmetric features is a tie.

  This closes the Design Space Step 3 exhaustiveness gap by upgrading
  the existential result (some model disagrees) to a quantitative one
  (exactly half the models disagree).

  Zero new axioms. All proofs complete.

  Supplement: Design Space Theorem, Step 3 (substantive version)
-/

import UniversalImpossibility.UnfaithfulQuantitative
import UniversalImpossibility.DesignSpaceFull

set_option autoImplicit false

open MeasureTheory

namespace UniversalImpossibility

variable {fs : FeatureSpace}

/-- If you commit to ranking j > k for symmetric features, the probability
    the ranking disagrees with the model is exactly 1/2.

    Disagreement means: the ranking says j > k, but the model has φ_k > φ_j.
    By attribution_prob_half (with j and k swapped), this event has measure 1/2. -/
theorem committed_ranking_error_half
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_commit_jk : ranking j k) :
    modelMeasure {f : Model | attribution fs k f > attribution fs j f}
    = 1 / 2 :=
  stable_ranking_unfaithfulness_half hprob hmeas hsym hnd ℓ j k hj hk hjk ranking h_commit_jk

/-- A tie has zero disagreement probability.

    If the ranking commits to neither j > k nor k > j, then the set of
    models where the ranking "disagrees" is empty: there is no model f
    such that (ranking j k AND φ_k(f) > φ_j(f)) OR (ranking k j AND φ_j(f) > φ_k(f)).

    We measure the disagreement set and show it has measure 0. -/
theorem tie_disagreement_zero
    (j k : Fin fs.P)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_tie : ¬ ranking j k ∧ ¬ ranking k j) :
    modelMeasure
      ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
    = 0 := by
  have h1 : {f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} = ∅ := by
    ext f; simp [h_tie.1]
  have h2 : {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f} = ∅ := by
    ext f; simp [h_tie.2]
  rw [h1, h2, Set.empty_union, measure_empty]

/-- Ties strictly dominate commitments for symmetric features.

    A commitment has disagreement probability 1/2 > 0 = disagreement
    probability of a tie. This is the quantitative upgrade of the
    existential result in DesignSpaceFull.lean. -/
theorem tie_dominates_commitment
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking_commit : Fin fs.P → Fin fs.P → Prop)
    (h_commit : ranking_commit j k)
    (ranking_tie : Fin fs.P → Fin fs.P → Prop)
    (h_tie : ¬ ranking_tie j k ∧ ¬ ranking_tie k j) :
    -- The tie's disagreement (0) is strictly less than the commitment's (1/2)
    modelMeasure
      ({f : Model | ranking_tie j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking_tie k j ∧ attribution fs j f > attribution fs k f})
    <
    modelMeasure {f : Model | attribution fs k f > attribution fs j f} := by
  rw [tie_disagreement_zero j k ranking_tie h_tie]
  rw [committed_ranking_error_half hprob hmeas hsym hnd ℓ j k hj hk hjk ranking_commit h_commit]
  norm_num

/-- Design Space Step 3 (substantive version): For any ranking of symmetric
    features, either it commits (and disagrees with exactly half the models)
    or it ties (and disagrees with zero models). Since 0 < 1/2, the
    Bayes-optimal ranking is a tie.

    This upgrades `design_space_exhaustiveness` from "some model disagrees"
    to "exactly half the models disagree, so ties are Bayes-optimal". -/
theorem design_space_step3_substantive
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop) :
    -- Either the ranking commits j > k and has error rate 1/2
    (ranking j k ∧
     modelMeasure {f : Model | attribution fs k f > attribution fs j f} = 1 / 2)
    ∨
    -- Or the ranking commits k > j and has error rate 1/2
    (ranking k j ∧
     modelMeasure {f : Model | attribution fs j f > attribution fs k f} = 1 / 2)
    ∨
    -- Or the ranking ties and has error rate 0
    (¬ ranking j k ∧ ¬ ranking k j ∧
     modelMeasure
       ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
        {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
     = 0) := by
  by_cases hjk_rank : ranking j k
  · -- Case 1: ranking commits j > k, error = 1/2
    left
    exact ⟨hjk_rank,
      committed_ranking_error_half hprob hmeas hsym hnd ℓ j k hj hk hjk ranking hjk_rank⟩
  · by_cases hkj_rank : ranking k j
    · -- Case 2: ranking commits k > j, error = 1/2
      right; left
      exact ⟨hkj_rank,
        committed_ranking_error_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)
          ranking hkj_rank⟩
    · -- Case 3: tie, error = 0
      right; right
      exact ⟨hjk_rank, hkj_rank,
        tie_disagreement_zero j k ranking ⟨hjk_rank, hkj_rank⟩⟩

end UniversalImpossibility
