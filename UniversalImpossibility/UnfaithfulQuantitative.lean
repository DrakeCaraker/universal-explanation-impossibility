/-
  UnfaithfulQuantitative.lean — Pr(unfaithfulness) = 1/2

  Under DGP symmetry and non-degeneracy, the probability that a
  random model ranks feature j above feature k is exactly 1/2.
  Any stable, complete ranking therefore disagrees with exactly
  half the models.

  This strengthens the existential result in UnfaithfulBound.lean
  (which only proves ∃ one disagreeing model) to a quantitative
  probability statement.

  Uses definitions from MeasureHypotheses.lean as hypotheses (not axioms).
-/

import UniversalImpossibility.MeasureHypotheses

open MeasureTheory

variable {fs : FeatureSpace}

/-- Core result: under DGP symmetry + non-degeneracy + probability measure,
    the probability that φ_j(f) > φ_k(f) is exactly 1/2.

    Proof: The events A = {φ_j > φ_k}, B = {φ_k > φ_j}, C = {φ_j = φ_k}
    partition the model space. By DGP symmetry μ(A) = μ(B).
    By non-degeneracy μ(C) = 0. By probability measure μ(A∪B∪C) = 1.
    Since A,B,C are disjoint: μ(A) + μ(B) + μ(C) = 1.
    Substituting: 2μ(A) = 1, so μ(A) = 1/2. -/
theorem attribution_prob_half
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    modelMeasure {f : Model | attribution fs j f > attribution fs k f}
    = 1 / 2 := by
  -- Let A = {φ_j > φ_k}, B = {φ_k > φ_j}, C = {φ_j = φ_k}
  let A := {f : Model | attribution fs j f > attribution fs k f}
  let B := {f : Model | attribution fs k f > attribution fs j f}
  let C := {f : Model | attribution fs j f = attribution fs k f}
  -- Step 1: μ(A) = μ(B) by DGP symmetry
  have hAB : modelMeasure A = modelMeasure B := hsym ℓ j k hj hk hjk
  -- Step 2: μ(C) = 0 by non-degeneracy
  have hC0 : modelMeasure C = 0 := hnd ℓ j k hj hk hjk
  -- Step 3: Measurability
  have hmj : Measurable (attribution fs j) := hmeas j
  have hmk : Measurable (attribution fs k) := hmeas k
  have hmsA : MeasurableSet A := measurableSet_lt hmk hmj
  have hmsB : MeasurableSet B := measurableSet_lt hmj hmk
  have hmsC : MeasurableSet C := by
    show MeasurableSet {f | attribution fs j f = attribution fs k f}
    have heq : {f : Model | attribution fs j f = attribution fs k f} =
               (fun f => attribution fs j f - attribution fs k f) ⁻¹' {(0 : ℝ)} := by
      ext f; simp [sub_eq_zero]
    rw [heq]
    exact (hmj.sub hmk) (measurableSet_singleton 0)
  -- Step 4: Aᶜ = B ∪ C (by trichotomy on reals)
  have hAc : Aᶜ = B ∪ C := by
    ext f
    simp only [Set.mem_compl_iff, Set.mem_union]
    change ¬(attribution fs k f < attribution fs j f) ↔
           (attribution fs k f > attribution fs j f ∨
            attribution fs j f = attribution fs k f)
    constructor
    · intro h
      rcases eq_or_lt_of_le (not_lt.mp h) with h1 | h1
      · right; exact h1
      · left; exact h1
    · rintro (h | h)
      · exact not_lt.mpr (le_of_lt h)
      · exact not_lt.mpr (le_of_eq h)
  -- Step 5: B and C are disjoint
  have hBC_disj : Disjoint B C := by
    rw [Set.disjoint_left]
    intro f (hfB : attribution fs k f > attribution fs j f)
               (hfC : attribution fs j f = attribution fs k f)
    linarith
  -- Step 6: μ(Aᶜ) = μ(B) + μ(C) (disjoint union)
  have hAc_meas : modelMeasure Aᶜ = modelMeasure B + modelMeasure C := by
    rw [hAc]
    exact measure_union hBC_disj hmsC
  -- Step 7: μ(A) + μ(Aᶜ) = 1
  have hprob' : IsProbabilityMeasure modelMeasure := hprob
  have htotal : modelMeasure A + modelMeasure Aᶜ = 1 := by
    rw [measure_add_measure_compl hmsA, measure_univ]
  -- Step 8: Combine: μ(A) + μ(A) = 1
  have hAA : modelMeasure A + modelMeasure A = 1 := by
    rw [← htotal, hAc_meas, hAB, hC0, add_zero]
  -- Step 9: μ(A) = 1/2 from 2 * μ(A) = 1
  rw [← two_mul] at hAA
  exact (ENNReal.eq_div_iff two_ne_zero ENNReal.ofNat_ne_top).mpr hAA

/-- Any stable, complete ranking disagrees with exactly half the models.
    This is the quantitative strengthening of stable_complete_unfaithful
    from UnfaithfulBound.lean. -/
theorem stable_ranking_unfaithfulness_half
    (hprob : IsProbabilityModelMeasure)
    (hmeas : HasMeasurableAttribution fs)
    (hsym : IsDGPSymmetric fs)
    (hnd : IsNonDegenerate fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_ranking_jk : ranking j k) :  -- the ranking fixes j > k
    modelMeasure {f : Model | attribution fs k f > attribution fs j f}
    = 1 / 2 := by
  -- The models where k > j are exactly the models where the ranking
  -- is unfaithful. By attribution_prob_half applied with j and k swapped
  -- (using the symmetry of the setup), μ({φ_k > φ_j}) = 1/2.
  have := attribution_prob_half hprob hmeas hsym hnd ℓ k j hk hj (Ne.symm hjk)
  -- IsDGPSymmetric gives μ({φ_k > φ_j}) = μ({φ_j > φ_k})
  -- And attribution_prob_half gives μ({φ_k > φ_j}) = 1/2
  exact this
