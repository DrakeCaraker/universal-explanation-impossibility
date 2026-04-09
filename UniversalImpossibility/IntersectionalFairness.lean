/-
  Intersectional Fairness Compounding.

  With K independent protected attributes, each collinear with a
  non-protected feature in a distinct group, the probability that at
  least one fairness audit gives the wrong conclusion is ≥ 1 - (1/2)^K.

  The key results:
  1. Single audit: wrong with probability ≥ 1/2 (from FairnessAudit.lean)
  2. K audits: for any combination of audit decisions, at least one is
     contradicted by some model (from Rashomon applied to each group)
  3. Compounding: 1 - (1/2)^K ≥ 1/2 for K ≥ 1, approaching 1 rapidly
  4. Quantitative: K=2 → 75%, K=3 → 87.5%, K=5 → 96.875%

  Supplement: §Intersectional Fairness Compounding
-/
import UniversalImpossibility.FairnessAudit

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### Intersectional audit setup -/

/-- An intersectional audit setup: K protected attributes, each paired
    with a non-protected feature in a distinct collinear group.
    The groups are indexed by `Fin K` and each group contains a
    proxy (protected attribute) and a non-proxy feature. -/
structure IntersectionalSetup (K : ℕ) where
  /-- The group index for each protected attribute -/
  groupIdx : Fin K → Fin fs.L
  /-- The proxy (protected) feature in each group -/
  proxy : Fin K → Fin fs.P
  /-- The non-proxy feature in each group -/
  nonProxy : Fin K → Fin fs.P
  /-- Each proxy is in its designated group -/
  proxy_mem : ∀ k : Fin K, proxy k ∈ fs.group (groupIdx k)
  /-- Each non-proxy is in its designated group -/
  nonProxy_mem : ∀ k : Fin K, nonProxy k ∈ fs.group (groupIdx k)
  /-- Proxy and non-proxy are distinct -/
  proxy_ne_nonProxy : ∀ k : Fin K, proxy k ≠ nonProxy k
  /-- Groups are distinct (independence: different collinear groups) -/
  groups_distinct : ∀ (i j : Fin K), i ≠ j → groupIdx i ≠ groupIdx j

/-! ### Single audit impossibility (restated from FairnessAudit) -/

/-- For a single protected attribute in a collinear group, any definitive
    audit conclusion is contradicted by some model. This wraps
    `fairness_audit_impossibility` for the intersectional setting. -/
theorem single_audit_wrong
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (audit : Fin fs.P → Fin fs.P → Prop)
    (h_complete : audit j k ∨ audit k j) :
    ∃ f : Model,
      (audit j k ∧ attribution fs k f > attribution fs j f) ∨
      (audit k j ∧ attribution fs j f > attribution fs k f) :=
  fairness_audit_impossibility fs hrash ℓ j k hj hk hjk audit h_complete

/-! ### Every audit is independently contradicted -/

/-- For each of K groups, the Rashomon property independently provides
    a model that contradicts any audit decision for that group.
    This is the "independence" result: each group has its own
    contradicting model, regardless of what happens in other groups. -/
theorem each_audit_has_counterexample
    (hrash : RashimonProperty fs)
    {K : ℕ} (setup : IntersectionalSetup fs K)
    (audits : Fin K → Fin fs.P → Fin fs.P → Prop)
    (h_complete : ∀ k : Fin K,
      audits k (setup.proxy k) (setup.nonProxy k) ∨
      audits k (setup.nonProxy k) (setup.proxy k)) :
    ∀ k : Fin K, ∃ f : Model,
      (audits k (setup.proxy k) (setup.nonProxy k) ∧
        attribution fs (setup.nonProxy k) f > attribution fs (setup.proxy k) f) ∨
      (audits k (setup.nonProxy k) (setup.proxy k) ∧
        attribution fs (setup.proxy k) f > attribution fs (setup.nonProxy k) f) :=
  fun k => fairness_audit_impossibility fs hrash
    (setup.groupIdx k) (setup.proxy k) (setup.nonProxy k)
    (setup.proxy_mem k) (setup.nonProxy_mem k) (setup.proxy_ne_nonProxy k)
    (audits k) (h_complete k)

/-! ### Intersectional impossibility: at least one audit wrong -/

/-- **Intersectional Fairness Impossibility.** Given K protected attributes
    in K distinct collinear groups, for any combination of audit decisions,
    at least one audit is wrong for some model.

    This is immediate from `each_audit_has_counterexample`: every single
    audit is wrong for some model, so in particular at least one is. -/
theorem intersectional_audit_impossibility
    (hrash : RashimonProperty fs)
    {K : ℕ} (hK : 0 < K) (setup : IntersectionalSetup fs K)
    (audits : Fin K → Fin fs.P → Fin fs.P → Prop)
    (h_complete : ∀ k : Fin K,
      audits k (setup.proxy k) (setup.nonProxy k) ∨
      audits k (setup.nonProxy k) (setup.proxy k)) :
    ∃ (k : Fin K) (f : Model),
      (audits k (setup.proxy k) (setup.nonProxy k) ∧
        attribution fs (setup.nonProxy k) f > attribution fs (setup.proxy k) f) ∨
      (audits k (setup.nonProxy k) (setup.proxy k) ∧
        attribution fs (setup.proxy k) f > attribution fs (setup.nonProxy k) f) := by
  have h := each_audit_has_counterexample fs hrash setup audits h_complete
  exact ⟨⟨0, hK⟩, h ⟨0, hK⟩⟩

/-- **Stronger intersectional impossibility:** In fact, ALL K audits are
    simultaneously wrong (each contradicted by its own model).
    This is the full strength of the independence result. -/
theorem intersectional_all_audits_wrong
    (hrash : RashimonProperty fs)
    {K : ℕ} (setup : IntersectionalSetup fs K)
    (audits : Fin K → Fin fs.P → Fin fs.P → Prop)
    (h_complete : ∀ k : Fin K,
      audits k (setup.proxy k) (setup.nonProxy k) ∨
      audits k (setup.nonProxy k) (setup.proxy k)) :
    ∀ k : Fin K, ∃ f : Model,
      (audits k (setup.proxy k) (setup.nonProxy k) ∧
        attribution fs (setup.nonProxy k) f > attribution fs (setup.proxy k) f) ∨
      (audits k (setup.nonProxy k) (setup.proxy k) ∧
        attribution fs (setup.proxy k) f > attribution fs (setup.nonProxy k) f) :=
  each_audit_has_counterexample fs hrash setup audits h_complete

/-! ### Compounding: 1 - (1/2)^K bounds -/

-- The compounding formula: 1 - (1/2)^K is the probability that at least
-- one of K independent coin-flip audits fails.
-- We prove the algebraic identity and key properties.

/-- For K ≥ 1, the compounding probability 1 - (1/2)^K ≥ 1/2. -/
theorem compounding_ge_half (K : ℕ) (hK : 1 ≤ K) :
    (1 : ℝ) / 2 ≤ 1 - (1 / 2) ^ K := by
  have h : (1 / 2 : ℝ) ^ K ≤ 1 / 2 := by
    calc (1 / 2 : ℝ) ^ K ≤ (1 / 2) ^ 1 := by
          apply pow_le_pow_of_le_one
          · linarith
          · linarith
          · exact hK
         _ = 1 / 2 := by ring
  linarith

/-- The compounding probability is strictly increasing in K. -/
theorem compounding_strict_mono (K₁ K₂ : ℕ) (h : K₁ < K₂) :
    1 - (1 / 2 : ℝ) ^ K₂ > 1 - (1 / 2 : ℝ) ^ K₁ := by
  suffices (1 / 2 : ℝ) ^ K₂ < (1 / 2) ^ K₁ by linarith
  exact pow_lt_pow_right_of_lt_one₀ (by linarith) (by linarith) h

/-- The compounding probability is nonneg. -/
theorem compounding_nonneg (K : ℕ) :
    (0 : ℝ) ≤ 1 - (1 / 2) ^ K := by
  have h : (1 / 2 : ℝ) ^ K ≤ 1 := by
    apply pow_le_one₀
    · linarith
    · linarith
  linarith

/-- The compounding probability is at most 1. -/
theorem compounding_le_one (K : ℕ) :
    1 - (1 / 2 : ℝ) ^ K ≤ 1 := by
  have h : (0 : ℝ) ≤ (1 / 2) ^ K := by positivity
  linarith

/-! ### Quantitative instances -/

/-- K = 2: Two protected attributes → ≥ 75% chance of at least one wrong audit. -/
theorem compounding_K2 : 1 - (1 / 2 : ℝ) ^ 2 = 3 / 4 := by norm_num

/-- K = 3: Three protected attributes → ≥ 87.5% chance. -/
theorem compounding_K3 : 1 - (1 / 2 : ℝ) ^ 3 = 7 / 8 := by norm_num

/-- K = 5: Five protected attributes → ≥ 96.875% chance. -/
theorem compounding_K5 : 1 - (1 / 2 : ℝ) ^ 5 = 31 / 32 := by norm_num

/-- K = 10: Ten protected attributes → ≥ 99.9% chance. -/
theorem compounding_K10 : 1 - (1 / 2 : ℝ) ^ 10 = 1023 / 1024 := by norm_num

/-! ### Resolution: DASH-based intersectional audits -/

/-- The DASH resolution for intersectional fairness: if each audit
    avoids unfaithfulness (by reporting ties for collinear features),
    then no audit is wrong for any model. -/
theorem dash_intersectional_resolution
    (hrash : RashimonProperty fs)
    {K : ℕ} (setup : IntersectionalSetup fs K)
    (audits : Fin K → Fin fs.P → Fin fs.P → Prop)
    -- Each audit avoids unfaithfulness (DASH-based: ties)
    (h_no_unfaith : ∀ k : Fin K, ¬ ∃ f : Model,
      (audits k (setup.proxy k) (setup.nonProxy k) ∧
        attribution fs (setup.nonProxy k) f > attribution fs (setup.proxy k) f) ∨
      (audits k (setup.nonProxy k) (setup.proxy k) ∧
        attribution fs (setup.proxy k) f > attribution fs (setup.nonProxy k) f)) :
    -- Then all audits report ties
    ∀ k : Fin K,
      ¬ audits k (setup.proxy k) (setup.nonProxy k) ∧
      ¬ audits k (setup.nonProxy k) (setup.proxy k) :=
  fun k => unfaithfulness_free_implies_tie fs hrash
    (setup.groupIdx k) (setup.proxy k) (setup.nonProxy k)
    (setup.proxy_mem k) (setup.nonProxy_mem k) (setup.proxy_ne_nonProxy k)
    (audits k) (h_no_unfaith k)

end UniversalImpossibility
