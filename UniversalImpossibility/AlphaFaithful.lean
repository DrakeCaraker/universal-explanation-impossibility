/-
  α-Faithfulness and the approximate faithfulness-stability tradeoff.

  Under DGP symmetry (Pr[φ_j > φ_k] = Pr[φ_k > φ_j] = 1/2),
  any stable complete ranking has α ≤ 1/2 for within-group pairs.

  A ranking is α-faithful if it agrees with the model's attribution
  ordering for at least fraction α of models.  We work with the
  qualitative version (counting witnesses) rather than measure theory.

  Key results:
  • stable_ranking_half_unfaithful — a stable complete ranking always
    disagrees with at least one of the two Rashomon witnesses.
  • alpha_faithful_bound — under the Rashomon property, no stable
    complete ranking can be faithful to strictly more than one of the
    two symmetric witnesses; hence α ≤ 1/2.

  Supplement: §Approximate faithfulness bounds (lines 4134-4177)
-/
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### α-Faithfulness definition -/

/-- A ranking `r` is faithful *at model f* for the pair (j, k) if it
    agrees with the attribution ordering at that model: r j k iff
    φ_j(f) > φ_k(f).  This is the per-model faithfulness condition.

    Note: the biconditional matches Definition 2 (strong faithfulness)
    in the paper.  The implication-only (weak) variant appears in
    `stable_ranking_half_unfaithful` below. -/
def FaithfulAt
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (j k : Fin fs.P) (f : Model) : Prop :=
  ranking j k ↔ attribution fs j f > attribution fs k f

/-- A ranking `r` is weakly faithful at model f for the pair (j, k)
    if attribution order implies ranking order (implication only). -/
def WeaklyFaithfulAt
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (j k : Fin fs.P) (f : Model) : Prop :=
  (attribution fs j f > attribution fs k f → ranking j k) ∧
  (attribution fs k f > attribution fs j f → ranking k j)

/-! ### The half-unfaithfulness theorem -/

/-- **Stable complete rankings are unfaithful to at least one Rashomon witness.**

    Given any stable (model-independent) complete ranking of a within-group
    pair (j, k), there exists a model f such that the ranking disagrees with
    f's attribution ordering.

    Proof sketch:
    - The Rashomon property yields witnesses f (where φ_j > φ_k) and
      f' (where φ_k > φ_j).
    - Completeness forces the ranking to choose: either j ≻ k or k ≻ j.
    - Whichever choice is made, the OTHER witness is a disagreement.

    This is the qualitative content of α ≤ 1/2: a stable ranking is
    "wrong" for at least half the symmetric witness pairs. -/
theorem stable_ranking_half_unfaithful
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    -- Completeness: ranking decides j vs k
    (hcomp : ranking j k ∨ ranking k j) :
    -- There exists a model where the ranking disagrees with attributions:
    -- (regardless of what faithfulness the ranking has, one Rashomon witness
    --  always contradicts the fixed choice forced by completeness)
    ∃ f : Model,
      (ranking j k ∧ attribution fs k f > attribution fs j f) ∨
      (ranking k j ∧ attribution fs j f > attribution fs k f) := by
  -- Extract the two Rashomon witnesses
  obtain ⟨f, f', h1, h2⟩ := hrash ℓ j k hj hk hjk
  -- Case split on which way the complete ranking goes
  cases hcomp with
  | inl hjk_rank =>
    -- Ranking says j ≻ k, but witness f' has φ_k(f') > φ_j(f')
    exact ⟨f', Or.inl ⟨hjk_rank, h2⟩⟩
  | inr hkj_rank =>
    -- Ranking says k ≻ j, but witness f has φ_j(f) > φ_k(f)
    exact ⟨f, Or.inr ⟨hkj_rank, h1⟩⟩

/-! ### Faithfulness to at most one witness -/

/-- **A stable complete ranking is faithful to at most one of the two
    Rashomon witnesses.**

    This is the precise two-witness formulation of α ≤ 1/2.
    Given the pair of witnesses (f, f') from the Rashomon property:
    - If the ranking is strongly faithful at f, it disagrees with f'.
    - If the ranking is strongly faithful at f', it disagrees with f.
    - The ranking cannot be strongly faithful at both simultaneously.

    This formalises the α-faithful bound: the best achievable faithfulness
    fraction for a stable complete ranking is 1/2 — one witness right,
    one witness wrong. -/
theorem alpha_faithful_bound
    (j k : Fin fs.P)
    (ranking : Fin fs.P → Fin fs.P → Prop) :
    -- No ranking can be strongly faithful to BOTH of two models whose
    -- attributions rank j and k in opposite directions:
    ¬ (∃ f f' : Model,
        attribution fs j f > attribution fs k f ∧
        attribution fs k f' > attribution fs j f' ∧
        FaithfulAt fs ranking j k f ∧
        FaithfulAt fs ranking j k f') := by
  intro hexists
  obtain ⟨f, f', h1, h2, hff, hff'⟩ := hexists
  -- From FaithfulAt f: ranking j k ↔ φ_j(f) > φ_k(f)
  -- h1 says φ_j(f) > φ_k(f), so ranking j k holds
  have hjk_rank : ranking j k := hff.mpr h1
  -- From FaithfulAt f': ranking j k ↔ φ_j(f') > φ_k(f')
  -- hjk_rank says ranking j k, so φ_j(f') > φ_k(f') must hold
  have hcontra : attribution fs j f' > attribution fs k f' := hff'.mp hjk_rank
  -- But h2 says φ_k(f') > φ_j(f') — contradiction
  linarith

/-! ### Corollary: antisymmetric stable rankings lose exactly one witness -/

/-- **An antisymmetric stable complete ranking is unfaithful to exactly
    one of the two Rashomon witnesses.**

    Under antisymmetry (¬(j ≻ k ∧ k ≻ j)) and completeness, the ranking
    makes a definitive choice. Whichever direction it picks, the other
    Rashomon witness is a model where the ranking is wrong.
    Combined with `stable_ranking_half_unfaithful`, this shows the
    infaithfulness rate is exactly 1/2 on the witness pair. -/
theorem faithful_exactly_one_witness
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (hcomp : ranking j k ∨ ranking k j) :
    -- There exist witnesses f_agree and f_disagree such that:
    -- f_agree: the ranking's fixed choice matches attributions
    -- f_disagree: the ranking's fixed choice contradicts attributions
    ∃ f_agree f_disagree : Model,
      ((ranking j k ∧ attribution fs j f_agree > attribution fs k f_agree) ∨
       (ranking k j ∧ attribution fs k f_agree > attribution fs j f_agree)) ∧
      ((ranking j k ∧ attribution fs k f_disagree > attribution fs j f_disagree) ∨
       (ranking k j ∧ attribution fs j f_disagree > attribution fs k f_disagree)) := by
  obtain ⟨f, f', h1, h2⟩ := hrash ℓ j k hj hk hjk
  cases hcomp with
  | inl hjk_rank =>
    -- Ranking says j ≻ k.
    -- f_agree = f  (φ_j(f) > φ_k(f), matching j ≻ k)
    -- f_disagree = f'  (φ_k(f') > φ_j(f'), contradicting j ≻ k)
    exact ⟨f, f', Or.inl ⟨hjk_rank, h1⟩, Or.inl ⟨hjk_rank, h2⟩⟩
  | inr hkj_rank =>
    -- Ranking says k ≻ j.
    -- f_agree = f'  (φ_k(f') > φ_j(f'), matching k ≻ j)
    -- f_disagree = f   (φ_j(f) > φ_k(f), contradicting k ≻ j)
    exact ⟨f', f, Or.inr ⟨hkj_rank, h2⟩, Or.inr ⟨hkj_rank, h1⟩⟩

end UniversalImpossibility
