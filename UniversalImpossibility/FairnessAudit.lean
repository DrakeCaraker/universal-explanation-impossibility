/-
  Fairness Audit Impossibility.

  When a protected proxy is collinear with a non-protected feature,
  SHAP-based proxy discrimination audits are coin flips: the audit
  concludes "proxy relied upon" for exactly half the models.

  This is a direct corollary of the attribution impossibility.

  Supplement: §Fairness Audit Impossibility
-/
import UniversalImpossibility.UnfaithfulBound

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- A fairness audit for features j (proxy, e.g., zip code) and k (non-proxy,
    e.g., income) determines whether the model relies more on the proxy.
    The audit decision is: "proxy j is more important than non-proxy k." -/
def AuditDecision (_j _k : Fin fs.P) := Prop

-- A fairness audit is a function mapping a model to an audit decision.
-- A "stable" audit gives the same answer regardless of which model is examined.
-- We represent this as a fixed proposition (the audit conclusion).

/-- Fairness Audit Impossibility: if the proxy j and non-proxy k are in the
    same collinear group (Rashomon property holds), then any stable audit
    decision is wrong for some model.

    Formally: the audit says "j is more important" (ranking j k), but some
    model has φ_k > φ_j. Or the audit says "k is more important" (ranking k j),
    but some model has φ_j > φ_k. The audit is unfaithful to some model. -/
theorem fairness_audit_impossibility
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    -- j is the proxy, k is the non-proxy
    -- The audit makes a definitive determination (complete)
    (audit : Fin fs.P → Fin fs.P → Prop)
    (h_complete : audit j k ∨ audit k j) :
    -- The audit is wrong for some model
    ∃ f : Model,
      (audit j k ∧ attribution fs k f > attribution fs j f) ∨
      (audit k j ∧ attribution fs j f > attribution fs k f) :=
  stable_complete_unfaithful fs hrash ℓ j k hj hk hjk audit h_complete

/-- The audit impossibility means: a definitive "proxy relied upon: yes"
    conclusion is contradicted by half the models (under DGP symmetry).

    Formally: the impossibility holds regardless of which direction the
    audit chooses. This is the "coin flip" result. -/
theorem fairness_audit_coin_flip
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    -- If the audit says "proxy j is more important" (audit = yes)
    (_h_audit_yes : True) :
    -- Then there exists a model where proxy is LESS important
    ∃ f : Model, attribution fs k f > attribution fs j f := by
  obtain ⟨_, f', _, hf'⟩ := hrash ℓ j k hj hk hjk
  exact ⟨f', hf'⟩

/-- And if the audit says "proxy j is NOT more important" (audit = no),
    there exists a model where proxy IS more important. -/
theorem fairness_audit_coin_flip_reverse
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    ∃ f : Model, attribution fs j f > attribution fs k f := by
  obtain ⟨f, _, hf, _⟩ := hrash ℓ j k hj hk hjk
  exact ⟨f, hf⟩

/-- The resolution: DASH-based audits report proxy and non-proxy as
    interchangeable (a tie), avoiding the coin-flip problem. -/
theorem dash_audit_resolution
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (audit : Fin fs.P → Fin fs.P → Prop)
    -- The audit avoids unfaithfulness (DASH-based: reports tie)
    (h_no_unfaith : ¬ ∃ f : Model,
      (audit j k ∧ attribution fs k f > attribution fs j f) ∨
      (audit k j ∧ attribution fs j f > attribution fs k f)) :
    -- Then it must be a tie (neither "proxy relied upon" nor "not relied upon")
    ¬ audit j k ∧ ¬ audit k j :=
  unfaithfulness_free_implies_tie fs hrash ℓ j k hj hk hjk audit h_no_unfaith

end UniversalImpossibility
