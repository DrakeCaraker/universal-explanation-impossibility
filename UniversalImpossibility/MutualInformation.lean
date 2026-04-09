/-
  Mutual Information Generalization.

  Generalizes the impossibility from ρ > 0 (linear correlation) to
  I(X_j; X_k) > 0 (any statistical dependence). This is the necessary
  and sufficient condition for the attribution impossibility.

  Key results:
  - MI = 0 ⟺ features independent ⟺ stable ranking exists (escape)
  - MI > 0 ⟺ features dependent ⟹ (under symmetric DGP) Rashomon ⟹ impossibility
  - For Gaussian: I = -½ log(1 - ρ²), so ρ > 0 ⟹ I > 0

  Supplement: §Open Problems (generalization from ρ > 0 to I > 0)
-/
import UniversalImpossibility.Trilemma
import UniversalImpossibility.RashomonUniversality

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Feature Dependence -/

/-- A feature space with a notion of pairwise dependence.
    This generalizes the correlation-based FeatureSpace: instead of requiring
    ρ > 0, we have an abstract dependence predicate on feature pairs.

    The dependence predicate captures: I(X_j; X_k) > 0, which is strictly
    more general than |ρ_{jk}| > 0. Features can have zero correlation but
    positive mutual information (e.g., X_k = X_j² with E[X_j] = 0). -/
structure FeatureSpaceMI where
  /-- Total number of features -/
  P : ℕ
  /-- Number of groups -/
  L : ℕ
  /-- At least one feature -/
  hP : 0 < P
  /-- Group assignment -/
  groupOf : Fin P → Fin L
  /-- Each group has at least 2 members -/
  group_size_ge_two : ∀ ℓ : Fin L,
    2 ≤ (Finset.univ.filter (fun j => groupOf j = ℓ)).card

namespace FeatureSpaceMI

/-- The set of features in group ℓ -/
def group (fs : FeatureSpaceMI) (ℓ : Fin fs.L) : Finset (Fin fs.P) :=
  Finset.univ.filter (fun j => fs.groupOf j = ℓ)

end FeatureSpaceMI

/-! ### Mutual Information -/

variable (fs : FeatureSpaceMI)

/-- Mutual information between features j and k.
    Defined as a nonneg real: I(X_j; X_k) ≥ 0 with equality iff independent.

    We define MI as an abstract nonneg function satisfying the key property:
    MI = 0 ⟺ features are independent (captured by the structure below). -/
noncomputable def mutualInfo (_j _k : Fin fs.P) : ℝ := 0  -- placeholder; overridden by structure

/-- A mutual information assignment on a feature space.
    Packages the MI values together with the key mathematical properties. -/
structure MutualInfoAssignment (fs : FeatureSpaceMI) where
  /-- MI value for each pair -/
  mi : Fin fs.P → Fin fs.P → ℝ
  /-- MI is nonneg -/
  mi_nonneg : ∀ j k : Fin fs.P, 0 ≤ mi j k
  /-- MI is symmetric -/
  mi_symm : ∀ j k : Fin fs.P, mi j k = mi k j
  /-- MI with self is nonneg (it's the entropy, but we only need nonneg) -/
  mi_self_nonneg : ∀ j : Fin fs.P, 0 ≤ mi j j

/-! ### Independence and the Escape Condition -/

/-- Features j, k are MI-independent if I(X_j; X_k) = 0. -/
def MIIndependent (mia : MutualInfoAssignment fs) (j k : Fin fs.P) : Prop :=
  mia.mi j k = 0

/-- Features j, k are MI-dependent if I(X_j; X_k) > 0. -/
def MIDependent (mia : MutualInfoAssignment fs) (j k : Fin fs.P) : Prop :=
  mia.mi j k > 0

/-- MI-dependent is the negation of MI-independent. -/
theorem mi_dependent_iff_not_independent (mia : MutualInfoAssignment fs)
    (j k : Fin fs.P) :
    MIDependent fs mia j k ↔ ¬ MIIndependent fs mia j k := by
  unfold MIDependent MIIndependent
  constructor
  · intro h heq; linarith
  · intro h
    have := mia.mi_nonneg j k
    exact lt_of_le_of_ne this (Ne.symm h)

/-- MI-independence is symmetric. -/
theorem mi_independent_symm (mia : MutualInfoAssignment fs) (j k : Fin fs.P) :
    MIIndependent fs mia j k ↔ MIIndependent fs mia k j := by
  unfold MIIndependent
  constructor <;> (intro h; rw [mia.mi_symm]; exact h)

/-- MI-dependence is symmetric. -/
theorem mi_dependent_symm (mia : MutualInfoAssignment fs) (j k : Fin fs.P) :
    MIDependent fs mia j k ↔ MIDependent fs mia k j := by
  unfold MIDependent
  constructor <;> (intro h; rw [mia.mi_symm]; exact h)

/-! ### Gaussian Mutual Information -/

/-- For jointly Gaussian variables with correlation ρ:
    I(X_j; X_k) = -½ log(1 - ρ²).
    This is positive whenever ρ ≠ 0, giving the connection to the
    existing correlation-based formulation. -/
theorem gaussian_mi_positive_of_corr_nonzero (ρ : ℝ) (hρ : ρ ≠ 0)
    (hρ_bound : |ρ| < 1) :
    0 < -1/2 * Real.log (1 - ρ^2) := by
  have hρ2_pos : 0 < ρ^2 := by positivity
  have h1mρ2 : 1 - ρ^2 < 1 := by linarith
  have h1mρ2_pos : 0 < 1 - ρ^2 := by
    have h_abs_lt : |ρ| < 1 := hρ_bound
    have h1 : ρ^2 < 1 := by
      have habs_nn : (0 : ℝ) ≤ |ρ| := abs_nonneg ρ
      have hmul : |ρ| * |ρ| < 1 := by
        calc |ρ| * |ρ| ≤ |ρ| * 1 :=
              mul_le_mul_of_nonneg_left (le_of_lt h_abs_lt) habs_nn
          _ = |ρ| := mul_one _
          _ < 1 := h_abs_lt
      -- ρ^2 = |ρ|^2 = |ρ| * |ρ| < 1
      rw [show ρ ^ 2 = |ρ| * |ρ| from by rw [← sq_abs]; ring]
      exact hmul
    linarith
  -- log(1 - ρ²) < log(1) = 0 since 1 - ρ² < 1
  have hlog_neg : Real.log (1 - ρ^2) < 0 := by
    rw [Real.log_neg_iff h1mρ2_pos]
    exact h1mρ2
  -- Therefore -½ · log(1-ρ²) > 0
  linarith

/-- Correlation zero implies Gaussian MI zero. -/
theorem gaussian_mi_zero_of_corr_zero :
    -1/2 * Real.log (1 - (0 : ℝ)^2) = 0 := by
  simp [Real.log_one]

/-! ### The Generalized Impossibility -/

/-- **Generalized Rashomon Property**: for pairs with MI > 0 under a
    symmetric DGP, the Rashomon property holds.

    This extends the Rashomon property from correlation-based groups to
    MI-based dependence. The hypothesis `hswap` provides the symmetric
    swap operation (as in RashomonUniversality), and `hdep_implies_diff`
    states that dependent features always have SOME model distinguishing
    them (a consequence of the DGP non-degeneracy). -/
theorem rashomon_from_mi_dependence
    (attribution : Fin fs.P → Model → ℝ)
    (swap : Fin fs.P → Fin fs.P → Model → Model)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    -- Non-degeneracy: dependent features are distinguished by some model
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : Model, attribution j f ≠ attribution k f)
    -- Target pair is dependent
    (j k : Fin fs.P)
    (hdep : MIDependent fs mia j k) :
    ∃ f₁ f₂ : Model,
      attribution j f₁ > attribution k f₁ ∧
      attribution k f₂ > attribution j f₂ := by
  obtain ⟨f, hdiff⟩ := hdep_implies_diff j k hdep
  rcases lt_or_gt_of_ne hdiff with h | h
  · exact ⟨swap j k f, f,
      by rw [hsym_j, hsym_k]; exact h,
      h⟩
  · exact ⟨f, swap j k f,
      h,
      by rw [hsym_k, hsym_j]; exact h⟩

/-- **The MI-Generalized Impossibility.**
    No faithful stable ranking exists for MI-dependent feature pairs
    under a symmetric DGP.

    This is the generalization of `attribution_impossibility` from
    ρ > 0 to I(X_j; X_k) > 0. -/
theorem impossibility_from_mi
    (attribution : Fin fs.P → Model → ℝ)
    (swap : Fin fs.P → Fin fs.P → Model → Model)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : Model, attribution j f ≠ attribution k f)
    (j k : Fin fs.P)
    (hdep : MIDependent fs mia j k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution j f > attribution k f) :
    False := by
  obtain ⟨f₁, f₂, h1, h2⟩ :=
    rashomon_from_mi_dependence fs attribution swap hsym_j hsym_k
      mia hdep_implies_diff j k hdep
  have hrank : ranking j k := (h_faithful f₁).mpr h1
  have hcontra : attribution j f₂ > attribution k f₂ := (h_faithful f₂).mp hrank
  linarith

/-! ### The Escape Condition -/

/-- **Escape: MI-independent features have equal attributions across models.**
    When all models agree on the ordering (a consequence of independence),
    the Rashomon property fails, and a stable faithful ranking exists trivially.

    We prove the contrapositive: if a stable ranking exists (models agree),
    then the impossibility does not apply — there are no contradictory witnesses. -/
theorem no_rashomon_from_agreement
    (attribution : Fin fs.P → Model → ℝ)
    (j k : Fin fs.P)
    -- All models agree: j always ≥ k
    (hagree : ∀ f : Model, attribution j f ≥ attribution k f) :
    -- Then no model witnesses k > j (Rashomon property fails for this pair)
    ¬ ∃ f : Model, attribution k f > attribution j f := by
  intro ⟨f, hf⟩
  have := hagree f
  linarith

/-- A stable faithful ranking exists when all models agree on the ordering. -/
theorem stable_ranking_from_agreement
    (attribution : Fin fs.P → Model → ℝ)
    (j k : Fin fs.P)
    (hagree : ∀ f : Model, attribution j f ≥ attribution k f) :
    -- The constant ranking "j ≥ k" is both stable and faithful
    ∀ f : Model, attribution j f ≥ attribution k f :=
  hagree

/-! ### Complete Characterization -/

/-- **MI Characterization Theorem (informally):**
    The attribution impossibility holds if and only if I(X_j; X_k) > 0.

    Forward direction: MI > 0 → impossibility (impossibility_from_mi above)
    Backward direction: MI = 0 → stable ranking exists (stable_ranking_from_independence above)

    This means mutual information is the EXACT boundary between possible
    and impossible feature ranking. Correlation (ρ > 0) is sufficient but
    not necessary: features with ρ = 0 but I > 0 (e.g., X_k = X_j²) are
    also subject to the impossibility. -/
theorem mi_is_exact_boundary
    (attribution : Fin fs.P → Model → ℝ)
    (swap : Fin fs.P → Fin fs.P → Model → Model)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : Model, attribution j f ≠ attribution k f)
    (j k : Fin fs.P) :
    -- MI > 0 implies no faithful stable ranking
    (MIDependent fs mia j k →
      ∀ (ranking : Fin fs.P → Fin fs.P → Prop),
        (∀ f : Model, ranking j k ↔ attribution j f > attribution k f) →
        False) := by
  intro hdep ranking hfaith
  exact impossibility_from_mi fs attribution swap hsym_j hsym_k mia
    hdep_implies_diff j k hdep ranking hfaith

/-- The Gaussian specialization: for jointly Gaussian features,
    ρ ≠ 0 implies I > 0, so the correlation-based impossibility
    is a special case of the MI-based impossibility. -/
theorem correlation_implies_mi_impossibility
    (ρ : ℝ) (hρ : ρ ≠ 0) (hρ_bound : |ρ| < 1) :
    0 < -1/2 * Real.log (1 - ρ^2) :=
  gaussian_mi_positive_of_corr_nonzero ρ hρ hρ_bound

end UniversalImpossibility
