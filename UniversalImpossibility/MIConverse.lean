/-
  The MI-boundary CONVERSE: machine-checked reduction + falsification.

  `mi_sufficient_for_impossibility` (MutualInformation.lean) machine-checks the
  FORWARD direction only: MI > 0 ⇒ impossibility, conditional on the bridge
  hypothesis `hdep_implies_diff`. The CONVERSE claimed in that file's header —
  "MI = 0 (features independent) ⇒ stable ranking exists (escape)" — was argued
  in prose but never machine-checked: `stable_ranking_from_agreement` assumes
  pointwise attribution agreement directly and never connects I = 0 to it.

  This file closes the converse honestly, in three parts:

  1. FALSIFICATION (the naked converse is FALSE). The unconditional statement
     "MI = 0 ⇒ a faithful stable ranking exists" fails, and not merely by an
     adversarial decoupling of the MI assignment from the attributions: it
     fails for the semantic reason the monograph itself concedes (capacity
     "captures correlation-driven Rashomon only — it can miss target-symmetry
     instability (independent but interchangeable features)"). We exhibit an
     explicit product distribution on two fair bits whose mutual information is
     COMPUTED to be 0 via the Shannon formula (`MIindep_eq_zero`, with
     independence itself verified as `q_eq_product`), paired with the
     feature-swap Rashomon attribution pair of `MIWitness` — the configuration
     forced by an exchangeable target such as Y = X₁ XOR X₂, whose feature-swap
     symmetry exists REGARDLESS of feature dependence. No faithful stable
     ranking exists (`no_faithful_ranking_swap_pair`), yet MI = 0
     (`independent_bits_mi_zero`). Conclusion: `mi_converse_fails_without_bridge`.
     Rashomon structure can come from target symmetry, which feature-MI does
     not see; MI = 0 is therefore NOT sufficient for possibility.

  2. REDUCTION (one named bridge, mirroring the Sard pattern). The converse
     becomes a theorem under exactly ONE named hypothesis, the mirror of
     `hdep_implies_diff`:

       `hindep_implies_agree` : MI-independence of (j,k) ⇒ all models agree
         on the strict attribution comparison for (j,k).

     Under it, a faithful stable decisive ranking exists
     (`mi_independent_implies_possibility`). The bridge is exactly what is
     missing and nothing more: `faithful_ranking_exists_iff_agreement` shows
     model-agreement is NECESSARY and SUFFICIENT for a faithful ranking, so no
     weaker abstract hypothesis could suffice; and part 1 shows the bridge is
     not abstractly dischargeable (`mirror_bridge_not_dischargeable`).

  3. THE FULL BOUNDARY, CONDITIONAL. Combining both bridges yields the first
     machine-checked exact-boundary biconditional
     (`mi_exact_boundary_biconditional`): a faithful stable ranking for (j,k)
     exists IFF I(X_j;X_k) = 0 — conditional on `hdep_implies_diff` (forward)
     and `hindep_implies_agree` (converse), both stated as explicit named
     hypotheses. Equivalently `mi_dependent_iff_impossibility`: impossibility
     iff MI > 0. As with `MIWitness`, we also exhibit a concrete model in which
     the mirror bridge is a THEOREM (`agree_bridge_holds`) and possibility
     holds unconditionally (`concrete_mi_possibility`).

  Status classification: the prose converse is REFUTED as an unconditional
  statement and REDUCED to one named bridge hypothesis under which it (and the
  full biconditional) is machine-checked. No `sorry`, no new axioms.

  Possibility is stated exactly as the negation of the impossibility target in
  MutualInformation.lean: `∃ ranking, ∀ f, ranking j k ↔ attribution j f >
  attribution k f`. The ranking is a single Prop (hence stable across models by
  construction) and the constructed witnesses commit to a definite verdict
  (hence decisive); faithfulness is the per-model iff.
-/
import UniversalImpossibility.MIWitness

set_option autoImplicit false

namespace UniversalImpossibility
namespace MIConverse

/-! ### Part 0: Possibility, and its exact characterization -/

section Abstract

variable (fs : FeatureSpaceMI)

/-- A faithful stable ranking exists for the pair (j,k). This is precisely the
    negation of the impossibility conclusion in `impossibility_from_mi`
    (MutualInformation.lean): the ranking is model-independent (stable), the
    per-model iff is faithfulness, and the witnesses constructed below commit
    to a definite verdict (decisive). Parametric in the model type `M`, as in
    `MIWitness.impossibility_from_mi_general`. -/
def FaithfulRankingExists {M : Type} (attribution : Fin fs.P → M → ℝ)
    (j k : Fin fs.P) : Prop :=
  ∃ ranking : Fin fs.P → Fin fs.P → Prop,
    ∀ f : M, ranking j k ↔ attribution j f > attribution k f

/-- **Exact characterization of pair-possibility.** A faithful stable ranking
    for (j,k) exists iff all models agree on the strict attribution comparison.
    This pins down exactly what any converse bridge must supply: nothing weaker
    than model-agreement can yield possibility, and nothing stronger is
    needed. -/
theorem faithful_ranking_exists_iff_agreement {M : Type}
    (attribution : Fin fs.P → M → ℝ) (j k : Fin fs.P) :
    FaithfulRankingExists fs attribution j k ↔
      ∀ f g : M, (attribution j f > attribution k f ↔
                   attribution j g > attribution k g) := by
  constructor
  · rintro ⟨ranking, hfaith⟩ f g
    exact (hfaith f).symm.trans (hfaith g)
  · intro hagree
    by_cases hex : ∃ f : M, attribution j f > attribution k f
    · obtain ⟨f₀, hf₀⟩ := hex
      exact ⟨fun _ _ => True,
        fun f => ⟨fun _ => (hagree f₀ f).mp hf₀, fun _ => trivial⟩⟩
    · push Not at hex
      exact ⟨fun _ _ => False,
        fun f => ⟨False.elim, fun h => absurd h (not_lt.mpr (hex f))⟩⟩

/-! ### Part 2: The conditional converse (one named mirror bridge) -/

/-- **The machine-checked converse, conditional on ONE named bridge.**
    If MI-independent features are attribution-order-agreed across models
    (`hindep_implies_agree`, the exact mirror of `hdep_implies_diff`), then
    MI = 0 for a pair implies a faithful stable ranking exists for that pair.
    This is the escape direction that MutualInformation.lean's header claimed
    and never machine-checked. -/
theorem mi_independent_implies_possibility {M : Type}
    (attribution : Fin fs.P → M → ℝ)
    (mia : MutualInfoAssignment fs)
    (hindep_implies_agree : ∀ j k : Fin fs.P,
      MIIndependent fs mia j k →
      ∀ f g : M, (attribution j f > attribution k f ↔
                   attribution j g > attribution k g))
    (j k : Fin fs.P)
    (hindep : MIIndependent fs mia j k) :
    FaithfulRankingExists fs attribution j k :=
  (faithful_ranking_exists_iff_agreement fs attribution j k).mpr
    (hindep_implies_agree j k hindep)

/-- The forward impossibility, parametric in both the feature space and the
    model type (the library `impossibility_from_mi` is `Model`-specialized and
    `MIWitness.impossibility_from_mi_general` is `fs`-specialized; this is the
    common generalization, needed for the biconditional below). Same four-step
    proof; stated as the negation of `FaithfulRankingExists`. -/
theorem impossibility_from_mi_general {M : Type}
    (attribution : Fin fs.P → M → ℝ)
    (swap : Fin fs.P → Fin fs.P → M → M)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : M, attribution j f ≠ attribution k f)
    (j k : Fin fs.P)
    (hdep : MIDependent fs mia j k) :
    ¬ FaithfulRankingExists fs attribution j k := by
  rintro ⟨ranking, hfaith⟩
  obtain ⟨f, hdiff⟩ := hdep_implies_diff j k hdep
  rcases lt_or_gt_of_ne hdiff with h | h
  · have h1 : attribution j (swap j k f) > attribution k (swap j k f) := by
      rw [hsym_j, hsym_k]; exact h
    have hrank : ranking j k := (hfaith (swap j k f)).mpr h1
    have hcon : attribution j f > attribution k f := (hfaith f).mp hrank
    linarith
  · have hrank : ranking j k := (hfaith f).mpr h
    have h2 : attribution j (swap j k f) > attribution k (swap j k f) :=
      (hfaith (swap j k f)).mp hrank
    rw [hsym_j, hsym_k] at h2
    linarith

/-! ### Part 3: The exact boundary, conditional on the two named bridges -/

/-- **The MI exact-boundary biconditional (conditional).** Under a symmetric
    DGP and the two named bridge hypotheses — `hdep_implies_diff` (forward,
    as in `mi_sufficient_for_impossibility`) and `hindep_implies_agree`
    (converse, this file) — a faithful stable ranking for (j,k) exists
    IF AND ONLY IF I(X_j; X_k) = 0.

    This is the statement `mi_is_exact_boundary` named but did not prove:
    the boundary is now a machine-checked equivalence, with every unproved
    ingredient isolated as an explicit hypothesis. -/
theorem mi_exact_boundary_biconditional {M : Type}
    (attribution : Fin fs.P → M → ℝ)
    (swap : Fin fs.P → Fin fs.P → M → M)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : M, attribution j f ≠ attribution k f)
    (hindep_implies_agree : ∀ j k : Fin fs.P,
      MIIndependent fs mia j k →
      ∀ f g : M, (attribution j f > attribution k f ↔
                   attribution j g > attribution k g))
    (j k : Fin fs.P) :
    FaithfulRankingExists fs attribution j k ↔ MIIndependent fs mia j k := by
  constructor
  · intro hposs
    by_contra hnindep
    exact impossibility_from_mi_general fs attribution swap hsym_j hsym_k mia
      hdep_implies_diff j k
      ((mi_dependent_iff_not_independent fs mia j k).mpr hnindep) hposs
  · exact mi_independent_implies_possibility fs attribution mia
      hindep_implies_agree j k

/-- Equivalent dependent form: impossibility iff MI > 0 (same hypotheses). -/
theorem mi_dependent_iff_impossibility {M : Type}
    (attribution : Fin fs.P → M → ℝ)
    (swap : Fin fs.P → Fin fs.P → M → M)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : M, attribution j f ≠ attribution k f)
    (hindep_implies_agree : ∀ j k : Fin fs.P,
      MIIndependent fs mia j k →
      ∀ f g : M, (attribution j f > attribution k f ↔
                   attribution j g > attribution k g))
    (j k : Fin fs.P) :
    MIDependent fs mia j k ↔ ¬ FaithfulRankingExists fs attribution j k :=
  (mi_dependent_iff_not_independent fs mia j k).trans
    (not_congr (mi_exact_boundary_biconditional fs attribution swap hsym_j
      hsym_k mia hdep_implies_diff hindep_implies_agree j k)).symm

end Abstract

/-! ### Part 1: Falsification — the naked converse is FALSE

An explicit product distribution on two fair bits: MI is COMPUTED to be 0 via
the Shannon formula (independence is manifest: the joint equals the product of
marginals, `q_eq_product`). Paired with the feature-swap Rashomon attribution
of `MIWitness` — the configuration forced by an exchangeable target such as
Y = X₁ XOR X₂, whose feature-swap symmetry does not require feature
dependence — the pair (0,1) is MI-independent yet admits NO faithful stable
ranking. This is the target-symmetry instability the monograph acknowledges
("independent but interchangeable features"): MI measures feature-feature
dependence and is blind to target symmetry, so MI = 0 cannot imply escape. -/

/-- Joint pmf on `Bool × Bool`: the uniform product distribution
    `q(x,y) = 1/4` (two INDEPENDENT fair bits) — the independence counterpart
    of `MIWitness.p` (two perfectly-correlated fair bits). -/
noncomputable def q (_x _y : Bool) : ℝ := 1/4

/-- Marginal of the first coordinate (`= 1/2` for each value). -/
noncomputable def qX (x : Bool) : ℝ := q x true + q x false

/-- Marginal of the second coordinate (`= 1/2` for each value). -/
noncomputable def qY (y : Bool) : ℝ := q true y + q false y

theorem qX_eq (x : Bool) : qX x = 1/2 := by cases x <;> norm_num [qX, q]
theorem qY_eq (y : Bool) : qY y = 1/2 := by cases y <;> norm_num [qY, q]

/-- Independence is manifest: the joint pmf factorizes as the product of its
    marginals — the counterexample's MI = 0 is semantic, not stipulated. -/
theorem q_eq_product (x y : Bool) : q x y = qX x * qY y := by
  rw [qX_eq, qY_eq]; norm_num [q]

/-- Shannon mutual information of `q`, from the definition (same formula as
    `MIWitness.MI`). -/
noncomputable def MIindep : ℝ :=
  ∑ x : Bool, ∑ y : Bool, q x y * Real.log (q x y / (qX x * qY y))

/-- The mutual information of two independent fair bits is `0`: every cell
    contributes `(1/4) · log 1 = 0`. COMPUTED from the distribution via the
    Shannon formula, mirroring `MIWitness.MI_eq_log_two`. -/
theorem MIindep_eq_zero : MIindep = 0 := by
  simp only [MIindep, Fintype.sum_bool, q, qX, qY]
  norm_num

/-- The MI assignment induced by the independent-bits distribution on the
    two-feature space of `MIWitness`: every pair carries the COMPUTED value
    `MIindep` (= 0). -/
noncomputable def miaIndep : MutualInfoAssignment MIWitness.fs where
  mi := fun _ _ => MIindep
  mi_nonneg := fun _ _ => le_of_eq MIindep_eq_zero.symm
  mi_symm := fun _ _ => rfl
  mi_self_nonneg := fun _ => le_of_eq MIindep_eq_zero.symm

/-- The pair (0,1) is MI-independent: I(X₀; X₁) = 0, computed from the
    product distribution. -/
theorem independent_bits_mi_zero : MIIndependent MIWitness.fs miaIndep 0 1 :=
  MIindep_eq_zero

/-- The feature-swap Rashomon pair of `MIWitness` admits NO faithful stable
    ranking of features 0 and 1: model `false` ranks 0 strictly above 1, model
    `true` ranks 1 strictly above 0, so model-agreement fails and
    `faithful_ranking_exists_iff_agreement` forbids any faithful ranking.
    The two models are related by the feature swap — a symmetry an
    exchangeable target (e.g. Y = X₁ XOR X₂) provides even when the features
    are independent. -/
theorem no_faithful_ranking_swap_pair :
    ¬ FaithfulRankingExists MIWitness.fs MIWitness.attribution 0 1 := by
  rw [faithful_ranking_exists_iff_agreement]
  intro hagree
  have h0 : MIWitness.attribution 0 false > MIWitness.attribution 1 false := by
    norm_num [MIWitness.attribution]
  have h1 := (hagree false true).mp h0
  norm_num [MIWitness.attribution] at h1

/-- **FALSIFICATION of the unconditional converse.** The claim
    "MI = 0 ⇒ a faithful stable ranking exists" (MutualInformation.lean,
    header, 'Key results') is FALSE: here is a pair with computed mutual
    information 0 (independent fair bits) and no faithful stable ranking
    (feature-swap Rashomon from target symmetry). The converse therefore
    genuinely REQUIRES the mirror bridge `hindep_implies_agree`; it is not
    abstractly dischargeable, exactly as `hdep_implies_diff` is not in the
    forward direction. -/
theorem mi_converse_fails_without_bridge :
    MIIndependent MIWitness.fs miaIndep 0 1 ∧
      ¬ FaithfulRankingExists MIWitness.fs MIWitness.attribution 0 1 :=
  ⟨independent_bits_mi_zero, no_faithful_ranking_swap_pair⟩

/-- The mirror bridge itself fails in this model — direct evidence that
    `hindep_implies_agree` cannot be discharged in the abstract framework
    (the exact analogue of the degenerate-attribution remark for
    `hdep_implies_diff`). -/
theorem mirror_bridge_not_dischargeable :
    ¬ (∀ j k : Fin MIWitness.fs.P,
        MIIndependent MIWitness.fs miaIndep j k →
        ∀ f g : Bool,
          (MIWitness.attribution j f > MIWitness.attribution k f ↔
           MIWitness.attribution j g > MIWitness.attribution k g)) :=
  fun hbridge =>
    no_faithful_ranking_swap_pair
      ((faithful_ranking_exists_iff_agreement MIWitness.fs
          MIWitness.attribution 0 1).mpr
        (hbridge 0 1 independent_bits_mi_zero))

/-! ### Part 3 (witness): a concrete model where the mirror bridge is a theorem

Mirroring `MIWitness.concrete_mi_impossibility` for the forward direction:
when the features are independent AND no target symmetry interchanges them,
every model assigns the same attributions, the mirror bridge holds as a
theorem, and possibility is unconditional. -/

/-- Attribution for the no-symmetry case: feature 0 carries all importance in
    EVERY model (no observationally-equivalent model can shift importance to
    feature 1 — independence removes the correlation channel and the target is
    not exchangeable). Model-invariant by construction. -/
noncomputable def attributionAgree : Fin MIWitness.fs.P → Bool → ℝ :=
  fun j _ => if j = 0 then 1 else 0

/-- **The mirror bridge, discharged.** In this concrete model, MI-independence
    implies all models agree on the strict attribution comparison — proved,
    not assumed (the attribution is model-invariant). -/
theorem agree_bridge_holds : ∀ j k : Fin MIWitness.fs.P,
    MIIndependent MIWitness.fs miaIndep j k →
    ∀ f g : Bool,
      (attributionAgree j f > attributionAgree k f ↔
       attributionAgree j g > attributionAgree k g) :=
  fun _ _ _ _ _ => Iff.rfl

/-- **Unconditional concrete possibility.** With the mirror bridge discharged,
    the escape holds with no remaining hypotheses: the independent-bits model
    with the non-exchangeable attribution admits a faithful stable ranking of
    the pair (0,1). Together with `MIWitness.concrete_mi_impossibility`
    (MI = log 2 > 0 ⇒ impossibility, unconditional), the two concrete models
    realize BOTH sides of the MI boundary with all bridges as theorems. -/
theorem concrete_mi_possibility :
    FaithfulRankingExists MIWitness.fs attributionAgree 0 1 :=
  mi_independent_implies_possibility MIWitness.fs attributionAgree miaIndep
    agree_bridge_holds 0 1 independent_bits_mi_zero

end MIConverse
end UniversalImpossibility

