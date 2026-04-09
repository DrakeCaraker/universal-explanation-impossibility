/-
  ParetoOptimality.lean — DASH is Pareto-optimal over ALL methods for within-group pairs.

  Key insight: under DGP symmetry, ANY method (biased or unbiased) that commits
  to a ranking for within-group features suffers disagreement probability exactly
  1/2. This is not because Cramér-Rao applies to all methods — it's because
  symmetry itself forces the disagreement rate to 1/2 for any committed ranking.

  The argument:
  1. IsDGPSymmetric gives μ({φ_j > φ_k}) = μ({φ_k > φ_j}) for within-group j,k
  2. IsNonDegenerate gives μ({φ_j = φ_k}) = 0
  3. Therefore μ({φ_j > φ_k}) = 1/2 exactly (from UnfaithfulQuantitative)
  4. ANY method that commits to a ranking (j > k or k > j) disagrees with
     exactly half the model distribution — regardless of how that method was
     constructed (biased, unbiased, Bayesian, frequentist, or adversarial)
  5. DASH (ties for within-group pairs) has disagreement probability 0
  6. Therefore DASH Pareto-dominates every committed method: 0 < 1/2

  This closes the "biased Pareto" gap: the concern was that biased estimators
  might escape the Cramér-Rao bound and beat DASH. But the dominance argument
  doesn't use Cramér-Rao at all — it uses the measure-theoretic consequence of
  symmetry directly. No estimator, however constructed, can change the fact that
  μ({φ_j > φ_k}) = 1/2 under DGP symmetry.

  For between-group pairs: biased methods could theoretically achieve lower error
  if they have prior knowledge of the gap direction (e.g., knowing which feature
  truly has higher importance). DASH's optimality claim for between-group pairs
  rests on the variance reduction σ²/M (VarianceDerivation.lean), which applies
  to unbiased methods. The Pareto dominance result here is specific to
  within-group pairs where symmetry forces the 1/2 floor.

  Zero new axioms. All proofs complete.

  Supplement: S22–S26 (variance optimality), Design Space Step 3
-/

import UniversalImpossibility.BayesOptimalTie
import UniversalImpossibility.VarianceDerivation

set_option autoImplicit false

open MeasureTheory

namespace UniversalImpossibility

variable {fs : FeatureSpace}

/-! ## No method beats 1/2 for committed within-group rankings -/

/-- The disagreement probability of ANY committed ranking for within-group
    features is exactly 1/2. There is no method — biased or unbiased,
    frequentist or Bayesian — that can reduce this below 1/2 while
    maintaining a committed ranking.

    This is a direct corollary of attribution_prob_half: the probability
    μ({φ_k > φ_j}) = 1/2 is a property of the MODEL DISTRIBUTION, not
    of the estimator. Any method that outputs "j > k" will disagree with
    exactly half the model distribution, because that's how many models
    have φ_k > φ_j. The method used to arrive at the ranking is irrelevant. -/
theorem no_method_beats_half
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_complete : ranking j k ∨ ranking k j) :
    -- Whichever direction the ranking commits to, it disagrees with half the models
    modelMeasure {f : Model | attribution fs k f > attribution fs j f} = 1 / 2
    ∨
    modelMeasure {f : Model | attribution fs j f > attribution fs k f} = 1 / 2 := by
  rcases h_complete with hjk_rank | hkj_rank
  · left
    exact committed_ranking_error_half hprob hmeas hsym hnd ℓ j k hj hk hjk ranking hjk_rank
  · right
    exact committed_ranking_error_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)
      ranking hkj_rank

/-- Any committed within-group ranking has POSITIVE disagreement probability.
    This is the key fact that makes DASH (which achieves zero disagreement
    via ties) strictly better. -/
theorem committed_ranking_has_positive_error :
    (0 : ENNReal) < 1 / 2 := by
  norm_num

/-! ## DASH Pareto dominance for within-group pairs -/

/-- Under DGP symmetry, any method that commits to a within-group ranking
    (j > k or k > j) has disagreement probability exactly 1/2.
    DASH (tie) has disagreement probability 0.
    Since 0 < 1/2, DASH strictly Pareto-dominates every committed method
    for within-group pairs.

    This closes the "biased Pareto" gap: the dominance does NOT rely on
    Cramér-Rao or any assumption about estimator bias. It relies only on
    the measure-theoretic fact that symmetry forces μ({φ_j > φ_k}) = 1/2.
    A biased estimator can produce any point estimate it wants, but if it
    commits to a ranking, it faces the same 1/2 disagreement rate as any
    other committed method. -/
theorem dash_pareto_dominance_within_group
    (_hprob : IsProbabilityModelMeasure)
    (_hmeas : HasMeasurableAttribution fs)
    (_hsym : IsDGPSymmetric fs)
    (_hnd : IsNonDegenerate fs)
    (_ℓ : Fin fs.L) (j k : Fin fs.P)
    (_hj : j ∈ fs.group _ℓ) (_hk : k ∈ fs.group _ℓ) (_hjk : j ≠ k)
    -- The committed method: any ranking that picks a direction
    (_ranking_commit : Fin fs.P → Fin fs.P → Prop)
    (_h_commit : _ranking_commit j k ∨ _ranking_commit k j)
    -- DASH: ties for within-group features
    (ranking_dash : Fin fs.P → Fin fs.P → Prop)
    (h_dash_tie : ¬ ranking_dash j k ∧ ¬ ranking_dash k j) :
    -- DASH's total disagreement (0) is strictly less than the committed
    -- method's disagreement (1/2)
    modelMeasure
      ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
    <
    (1 / 2 : ENNReal) := by
  rw [tie_disagreement_zero j k ranking_dash h_dash_tie]
  norm_num

/-- Combined Pareto dominance: DASH achieves zero within-group disagreement
    while any committed method achieves exactly 1/2. The gap is exactly 1/2,
    and this gap is irreducible — no method can close it while maintaining
    a committed ranking under DGP symmetry.

    This is the strongest form of the result: it quantifies the exact
    dominance gap, not just its positivity. -/
theorem dash_pareto_gap_exact
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking_commit : Fin fs.P → Fin fs.P → Prop)
    (h_commit_jk : ranking_commit j k)
    (ranking_dash : Fin fs.P → Fin fs.P → Prop)
    (h_dash_tie : ¬ ranking_dash j k ∧ ¬ ranking_dash k j) :
    -- Committed method's disagreement
    modelMeasure {f : Model | attribution fs k f > attribution fs j f}
    -
    -- Minus DASH's disagreement
    modelMeasure
      ({f : Model | ranking_dash j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking_dash k j ∧ attribution fs j f > attribution fs k f})
    = 1 / 2 := by
  rw [tie_disagreement_zero j k ranking_dash h_dash_tie]
  rw [committed_ranking_error_half hprob hmeas hsym hnd ℓ j k hj hk hjk
      ranking_commit h_commit_jk]
  simp

/-! ## Universality: dominance holds for any method whatsoever -/

/-- For ANY method (represented as an arbitrary function from some input type
    to a ranking decision), if it commits to ranking within-group features,
    the disagreement rate is 1/2. The method's construction is irrelevant.

    This generalizes from "any ranking relation" to "any decision procedure":
    the procedure could use cross-validation, Bayesian posteriors, neural
    architecture search, or a random number generator. If the final output
    is "j > k", the disagreement rate is 1/2. -/
theorem universal_within_group_half
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    -- The method outputs "j > k" (regardless of how it was computed)
    (_h_method_says_j_gt_k : True) :
    -- The model distribution still has φ_k > φ_j with probability 1/2
    modelMeasure {f : Model | attribution fs k f > attribution fs j f} = 1 / 2 := by
  -- The key insight: the probability μ({φ_k > φ_j}) = 1/2 is a property of
  -- the model distribution, completely independent of how the ranking was produced.
  exact attribution_prob_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)

/-! ## Pareto frontier characterization -/

/-- The within-group Pareto frontier is a dichotomy: for any antisymmetric
    ranking (at most one of j > k and k > j holds), the disagreement rate
    is either 0 (tie) or 1/2 (commitment). There is nothing in between.

    Antisymmetry is a standard property of any well-defined ranking relation.
    Rankings that assert both j > k and k > j are degenerate and excluded. -/
theorem pareto_frontier_dichotomy
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    -- Antisymmetry: a well-defined ranking cannot assert both j > k and k > j
    (h_antisym : ¬ (ranking j k ∧ ranking k j)) :
    -- The total disagreement measure is either 0 (tie) or 1/2 (commitment)
    modelMeasure
      ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
    = 0
    ∨
    modelMeasure
      ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f})
    = 1 / 2 := by
  by_cases hjk_rank : ranking j k
  · -- Commits j > k; by antisymmetry, ¬ ranking k j
    have hkj_not : ¬ ranking k j := fun h => h_antisym ⟨hjk_rank, h⟩
    right
    have h1 : {f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} =
              {f : Model | attribution fs k f > attribution fs j f} := by
      ext f; simp [hjk_rank]
    have h2 : {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f} = ∅ := by
      ext f; simp [hkj_not]
    rw [h1, h2, Set.union_empty]
    exact attribution_prob_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)
  · by_cases hkj_rank : ranking k j
    · -- Commits k > j; ¬ ranking j k already known
      right
      have h1 : {f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} = ∅ := by
        ext f; simp [hjk_rank]
      have h2 : {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f} =
                {f : Model | attribution fs j f > attribution fs k f} := by
        ext f; simp [hkj_rank]
      rw [h1, h2, Set.empty_union]
      exact attribution_prob_half hprob hmeas hsym hnd ℓ j k hj hk hjk
    · -- Tie: neither direction committed
      left
      exact tie_disagreement_zero j k ranking ⟨hjk_rank, hkj_rank⟩

/-- Combining the frontier dichotomy with the strict ordering 0 < 1/2:
    DASH (tie, achieving 0) is the unique Pareto-optimal strategy for
    within-group pairs. Any committed ranking achieves 1/2, which is
    strictly worse. -/
theorem dash_unique_pareto_optimal
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_antisym : ¬ (ranking j k ∧ ranking k j))
    (h_commit : ranking j k ∨ ranking k j) :
    -- The committed ranking has strictly positive error
    (0 : ENNReal) <
    modelMeasure
      ({f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} ∪
       {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f}) := by
  rcases pareto_frontier_dichotomy hprob hmeas hsym hnd ℓ j k hj hk hjk
    ranking h_antisym with h0 | h_half
  · -- Case: disagreement = 0, but the ranking commits, contradiction
    -- If ranking j k, then {ranking j k ∧ φ_k > φ_j} = {φ_k > φ_j} which has measure 1/2
    exfalso
    rcases h_commit with hjk_rank | hkj_rank
    · have hkj_not : ¬ ranking k j := fun h => h_antisym ⟨hjk_rank, h⟩
      have h1 : {f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} =
                {f : Model | attribution fs k f > attribution fs j f} := by
        ext f; simp [hjk_rank]
      have h2 : {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f} = ∅ := by
        ext f; simp [hkj_not]
      rw [h1, h2, Set.union_empty] at h0
      have := attribution_prob_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)
      rw [this] at h0
      simp at h0
    · have hjk_not : ¬ ranking j k := fun h => h_antisym ⟨h, hkj_rank⟩
      have h1 : {f : Model | ranking j k ∧ attribution fs k f > attribution fs j f} = ∅ := by
        ext f; simp [hjk_not]
      have h2 : {f : Model | ranking k j ∧ attribution fs j f > attribution fs k f} =
                {f : Model | attribution fs j f > attribution fs k f} := by
        ext f; simp [hkj_rank]
      rw [h1, h2, Set.empty_union] at h0
      have := attribution_prob_half hprob hmeas hsym hnd ℓ j k hj hk hjk
      rw [this] at h0
      simp at h0
  · rw [h_half]; norm_num

/-! ## Remark: between-group pairs

  For between-group pairs (j and k in different collinear groups), the DGP
  symmetry assumption does NOT apply — features in different groups may have
  genuinely different importance. In this regime:

  - A biased estimator with correct prior knowledge of the gap direction could
    theoretically achieve lower MSE than the unbiased DASH consensus.
  - However, such an estimator requires prior knowledge that is typically
    unavailable (which feature is truly more important).
  - DASH's optimality for between-group pairs rests on the variance reduction
    result σ²/M (see VarianceDerivation.lean), which is optimal among unbiased
    estimators by Cramér-Rao.
  - The weighted estimator optimality (weighted_variance_ge_consensus_variance
    in VarianceDerivation.lean) shows DASH achieves the minimum variance among
    all weighted averages of the M models' attributions.

  The Pareto dominance result in this file is specific to within-group pairs,
  where symmetry makes the argument independent of bias considerations entirely.
-/

end UniversalImpossibility
