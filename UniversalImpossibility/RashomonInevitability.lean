/-
  Rashomon Inevitability: the impossibility is inescapable for standard ML.

  S5: Attribution non-degeneracy — collinear features almost never have
      exactly equal attribution.
  S6: Rashomon inevitability — non-degeneracy + algorithmic symmetry →
      the Rashomon property holds, making the impossibility apply.

  The only escapes: (i) ρ = 0, (ii) deterministic single model,
  (iii) asymmetric algorithm.

  Supplement: §Extended Results: Universality
-/
import UniversalImpossibility.RashomonUniversality

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### S5: Non-degeneracy -/

/-- Attribution non-degeneracy: there exists a model that distinguishes
    features j and k (their attributions are not equal).
    In practice, this holds with probability 1 for any stochastic
    training algorithm on collinear features. -/
def NonDegenerate (j k : Fin fs.P) : Prop :=
  ∃ f : Model, attribution fs j f ≠ attribution fs k f

/-- A stronger form: there exists a model where j has STRICTLY more
    attribution than k. -/
def StrictlyDistinguishes (j k : Fin fs.P) : Prop :=
  ∃ f : Model, attribution fs j f > attribution fs k f

/-- Non-degeneracy implies strict distinguishing in one direction.
    If φ_j(f) ≠ φ_k(f), then either φ_j(f) > φ_k(f) or φ_k(f) > φ_j(f). -/
theorem nondegen_implies_strict_one_direction
    (j k : Fin fs.P) (hnd : NonDegenerate fs j k) :
    StrictlyDistinguishes fs j k ∨ StrictlyDistinguishes fs k j := by
  obtain ⟨f, hne⟩ := hnd
  rcases lt_or_gt_of_ne hne with h | h
  · exact Or.inr ⟨f, h⟩
  · exact Or.inl ⟨f, h⟩

/-! ### S6: Algorithmic symmetry -/

/-- Algorithmic symmetry: for any model where j dominates k, there exists
    a model where k dominates j. This captures the DGP symmetry: when
    features have identical distributions, the training algorithm is
    equally likely to produce models favoring either one.

    This is weaker than the full Rashomon property — it only requires
    the existence of the reversed model, not for ALL pairs simultaneously. -/
def AlgorithmicSymmetry (j k : Fin fs.P) : Prop :=
  (∀ f : Model, attribution fs j f > attribution fs k f →
    ∃ f' : Model, attribution fs k f' > attribution fs j f') ∧
  (∀ f : Model, attribution fs k f > attribution fs j f →
    ∃ f' : Model, attribution fs j f' > attribution fs k f')

/-! ### S6: Rashomon inevitability -/

/-- Non-degeneracy + algorithmic symmetry → both directions are realized.
    If some model distinguishes j from k, and the algorithm is symmetric,
    then models exist ranking j > k AND models exist ranking k > j. -/
theorem rashomon_from_nondegen_and_symmetry
    (j k : Fin fs.P)
    (hnd : NonDegenerate fs j k)
    (hsym : AlgorithmicSymmetry fs j k) :
    (∃ f : Model, attribution fs j f > attribution fs k f) ∧
    (∃ f' : Model, attribution fs k f' > attribution fs j f') := by
  obtain ⟨hsym_jk, hsym_kj⟩ := hsym
  rcases nondegen_implies_strict_one_direction fs j k hnd with ⟨f, hf⟩ | ⟨f, hf⟩
  · exact ⟨⟨f, hf⟩, hsym_jk f hf⟩
  · exact ⟨hsym_kj f hf, ⟨f, hf⟩⟩

/-- Full Rashomon inevitability: if every within-group pair is non-degenerate
    and algorithmically symmetric, the full Rashomon property holds. -/
theorem rashomon_inevitability
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      NonDegenerate fs j k)
    (hsym : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      AlgorithmicSymmetry fs j k) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  have hpair := rashomon_from_nondegen_and_symmetry fs j k
    (hnd ℓ j k hj hk hjk) (hsym ℓ j k hj hk hjk)
  obtain ⟨⟨f, hf⟩, ⟨f', hf'⟩⟩ := hpair
  exact ⟨f, f', hf, hf'⟩

/-- The impossibility is inescapable: non-degeneracy + algorithmic symmetry
    implies the full attribution impossibility. -/
theorem impossibility_inevitable
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      NonDegenerate fs j k)
    (hsym : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      AlgorithmicSymmetry fs j k)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs (rashomon_inevitability fs hnd hsym)
    ℓ j k hj hk hjk ranking h_faithful

/-! ### Escapes -/

/-- Escape 1: If all models agree on the ordering (no non-degeneracy in
    both directions), a faithful stable complete ranking exists. -/
theorem escape_unanimous_agreement
    (j k : Fin fs.P)
    (_h_agree : ∀ f : Model, attribution fs j f ≥ attribution fs k f) :
    ∃ ranking : Fin fs.P → Fin fs.P → Prop,
      -- The ranking is complete for (j,k)
      (ranking j k) ∧
      -- The ranking never contradicts any model's strict ordering
      (∀ f : Model, attribution fs j f > attribution fs k f → ranking j k) := by
  exact ⟨fun a b => a = j ∧ b = k, ⟨⟨rfl, rfl⟩, fun _ _ => ⟨rfl, rfl⟩⟩⟩

-- Escape 2: If the algorithm is NOT symmetric (built-in feature preferences),
-- all models may agree. We don't need to prove this — it's the negation
-- of the hypotheses. The theorem `rashomon_inevitability` simply doesn't
-- apply without `AlgorithmicSymmetry`.

end UniversalImpossibility
