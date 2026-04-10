import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Query-Relative Impossibility

The impossibility is not monolithic — it applies per query.
For each question q about the system, the impossibility holds
if and only if the Rashomon set produces incompatible answers
to q. Questions where the Rashomon set agrees admit faithful,
stable, decisive answers.

This replaces "you can't explain" with "here is the exact set
of questions you cannot answer stably."
-/

variable {Θ : Type} {H : Type} {Y : Type} {Q : Type}

/-- A query-specific incompatibility relation. For each query q,
    incomp_q h₁ h₂ means the two explanations give contradictory
    answers to question q. -/
structure QueryIncompatibility (H : Type) (Q : Type) where
  incomp : Q → H → H → Prop
  incomp_irrefl : ∀ (q : Q) (h : H), ¬incomp q h h

/-- The Rashomon property holds for query q if equivalent
    configurations give incompatible answers to q. -/
def rashomon_at_query (observe : Θ → Y) (explain : Θ → H)
    (qi : QueryIncompatibility H Q) (q : Q) : Prop :=
  ∃ θ₁ θ₂ : Θ, observe θ₁ = observe θ₂ ∧
    qi.incomp q (explain θ₁) (explain θ₂)

/-- Query-specific faithfulness: E never contradicts explain
    on question q. -/
def q_faithful (explain : Θ → H) (qi : QueryIncompatibility H Q)
    (q : Q) (E : Θ → H) : Prop :=
  ∀ (θ : Θ), ¬qi.incomp q (E θ) (explain θ)

/-- Query-specific decisiveness: E inherits all q-incompatibilities
    from explain. -/
def q_decisive (explain : Θ → H) (qi : QueryIncompatibility H Q)
    (q : Q) (E : Θ → H) : Prop :=
  ∀ (θ : Θ) (h : H), qi.incomp q (explain θ) h →
    qi.incomp q (E θ) h

/-- Query-Relative Impossibility.

    For each query q where the Rashomon property holds, no
    explanation can be q-faithful, stable, and q-decisive.
    The proof is identical to the universal impossibility —
    the same four steps, applied to the query-specific
    incompatibility. -/
theorem query_impossibility
    (observe : Θ → Y) (explain : Θ → H)
    (qi : QueryIncompatibility H Q) (q : Q)
    (hrash : rashomon_at_query observe explain qi q)
    (E : Θ → H)
    (hf : q_faithful explain qi q E)
    (hs : ∀ (θ₁ θ₂ : Θ), observe θ₁ = observe θ₂ → E θ₁ = E θ₂)
    (hd : q_decisive explain qi q E) : False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := hrash
  have h1 := hd θ₁ (explain θ₂) hinc
  have h2 := hs θ₁ θ₂ hobs
  rw [h2] at h1
  exact hf θ₂ h1

/-- Corollary: queries where Rashomon does NOT hold are
    answerable. If no equivalent pair gives incompatible
    answers to q, then E = explain is q-faithful, stable
    (when observe is injective), and q-decisive. -/
theorem query_possibility
    (observe : Θ → Y) (explain : Θ → H)
    (qi : QueryIncompatibility H Q) (q : Q)
    (_h_no_rash : ¬rashomon_at_query observe explain qi q)
    (h_inj : ∀ (θ₁ θ₂ : Θ), observe θ₁ = observe θ₂ → θ₁ = θ₂) :
    q_faithful explain qi q explain ∧
    (∀ (θ₁ θ₂ : Θ), observe θ₁ = observe θ₂ → explain θ₁ = explain θ₂) ∧
    q_decisive explain qi q explain := by
  refine ⟨?_, ?_, ?_⟩
  · intro θ; exact qi.incomp_irrefl q (explain θ)
  · intro θ₁ θ₂ hobs
    have := h_inj θ₁ θ₂ hobs
    subst this; rfl
  · intro θ h hinc; exact hinc
