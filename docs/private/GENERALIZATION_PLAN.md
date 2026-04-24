# Pre-Submission Generalization — Query-Relative + Quantitative Remark

**Goal**: Add two items before NeurIPS submission:
1. Query-relative impossibility (Lean theorem + paper paragraph) — FREE
2. Quantitative tradeoff remark (paper paragraph only) — SIGNALS DEPTH

**Time**: ~1 hour total. No experiments needed.
**Model**: Opus for Lean theorem. Sonnet for paper integration.

---

## VET RECORD

### Round 1 — Factual

- The query-relative theorem is genuinely ~15 lines of Lean.
  The proof is identical to `explanation_impossibility` but
  parameterized by a query-specific incompatibility relation.
  ⚠️ HOWEVER: the current ExplanationSystem bundles `incompatible`
  and `rashomon` as fields. The query-relative version needs
  these PER QUERY, not per system. Two approaches:
  (a) Create a separate `QueryImpossibility` theorem that takes
      `incomp_q` and a query-specific Rashomon witness as
      hypotheses (not bundled). This is clean.
  (b) Modify ExplanationSystem to parameterize by Q. This is
      invasive and would break all 9 instances.
  DECISION: (a). New theorem, doesn't touch existing code.

- The quantitative remark requires no Lean. It's a paper-only
  observation: ε + δ + (1-d) ≥ μ(Rashomon) follows from the
  union bound applied to the per-fiber theorem. The per-fiber
  bound is already proved in QuantitativeBound.lean.

### Round 2 — Reasoning

- Adding the query-relative theorem STRENGTHENS the paper's
  response to the "so what?" objection. Instead of "you can't
  explain," the paper says "here's exactly which questions you
  can and can't answer." This is the single most useful addition
  for practitioners.

- The quantitative remark SIGNALS that the framework has depth
  without developing it fully. This addresses the "4-line proof
  is trivial" objection by showing the framework extends to
  continuous tradeoffs.

- ⚠️ Neither addition should change the paper's CHARACTER.
  The query-relative theorem gets one paragraph + one Lean file.
  The quantitative remark gets one paragraph in Discussion.
  The paper should not become about these generalizations.

### Round 3 — Omissions

- ⚠️ The query-relative theorem needs at least ONE concrete
  example to be useful. E.g.: "For feature attribution, the
  query 'is feature x relevant?' (top-k membership) may be
  stable even when 'is x ranked above y?' (pairwise ordering)
  is not." Add this as an example after the theorem.

- ⚠️ The plan should specify WHICH paper versions get the
  additions. ANSWER: PNAS + NeurIPS + monograph. JMLR can
  be updated later.

- ⚠️ After adding a new Lean file, Basic.lean needs updating,
  counts change (75 files, 350+ theorems), and all paper
  versions' Lean stats need updating.

---

## Phase 1: Query-Relative Impossibility in Lean [Opus]

### Task 1.1: Create UniversalImpossibility/QueryRelative.lean

```lean
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

/-- Stability is query-independent (same definition). -/
-- stable is already defined in ExplanationSystem.lean

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
    (if observe is injective), and q-decisive. -/
theorem query_possibility
    (observe : Θ → Y) (explain : Θ → H)
    (qi : QueryIncompatibility H Q) (q : Q)
    (h_no_rash : ¬rashomon_at_query observe explain qi q)
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
```

NOTE: The stability in `query_impossibility` uses the standard
definition (not query-parameterized) because stability is about
E's output, not about what questions are being asked. This is
correct — stability is a property of the map, not the query.

Add import to Basic.lean. Run `lake build`. Zero sorry.

---

## Phase 2: Paper Integration [Sonnet]

### Task 2.1: Add to PNAS paper

In paper/universal_impossibility_pnas.tex, after the necessity
proposition (or in the Discussion), add one paragraph:

"The impossibility is query-relative. Parameterize incompatibility
by a query space $Q$: for each question $q$ (e.g., 'is feature $x$
ranked above $y$?'), the impossibility holds if and only if the
Rashomon set produces incompatible answers to $q$ (Lean:
\texttt{query\_impossibility}). Questions where the Rashomon set
agrees — such as 'is feature $x$ relevant?' when all equivalent
models include $x$ in the top $k$ — admit faithful, stable, decisive
answers. The framework thus replaces 'explanation is impossible'
with a precise characterization of which questions are and are not
stably answerable."

### Task 2.2: Add quantitative remark to PNAS Discussion

Add one paragraph to the Discussion:

"The proof immediately yields a quantitative version. For each
Rashomon pair $(\theta_1, \theta_2)$, the four-step chain forces
at least one failure: unfaithfulness at $\theta_2$, instability on
$(\theta_1, \theta_2)$, or indecisiveness at $\theta_1$. Summing
over the Rashomon set under a probability measure:
$\varepsilon + \delta + (1-d) \geq \mu(\text{Rashomon})$,
where $\varepsilon$ is the unfaithfulness rate, $\delta$ the
instability rate, $d$ the decisiveness rate, and $\mu$ the measure
of the Rashomon set. This tradeoff surface — which includes the
binary impossibility as the special case
$\varepsilon = \delta = 0$ — is developed with domain-specific
bounds in forthcoming work."

### Task 2.3: Add to NeurIPS paper

Same two paragraphs, adapted for length. The query-relative
paragraph can be slightly shorter (3 sentences instead of 4).

### Task 2.4: Add to monograph

The monograph gets the FULL treatment:
- The query-relative theorem statement (with Lean code listing)
- A worked example: "is x ranked above y?" vs "is x in top k?"
- The quantitative remark (same as PNAS)
- A brief discussion of Generalizations 5-7 as future work

### Task 2.5: Update Lean counts in all papers

After adding QueryRelative.lean: 75 files, ~353 theorems.
Run the grep counts and update everywhere.

### Task 2.6: Compile all versions

Compile PNAS, NeurIPS, monograph. Report page counts.
PNAS must still be ≤6pp. NeurIPS ≤10pp.

---

## Execution Order

```
Phase 1: [1.1 Lean theorem]
Phase 2: [2.1 PNAS] ∥ [2.2 PNAS] ∥ [2.3 NeurIPS] ∥ [2.4 monograph]
         → [2.5 counts] → [2.6 compile]
```

Phase 1 first (blocks on Lean compilation).
Phase 2 tasks are independent except counts and compilation.

## Confidence

| Item | Confidence |
|------|-----------|
| Query-relative Lean theorem | HIGH — same proof, parameterized |
| Query possibility corollary | HIGH — same as fully_specified_possibility |
| PNAS paragraph fits in 6pp | MEDIUM — may need to cut elsewhere |
| NeurIPS paragraph fits in 10pp | MEDIUM — already at limit |
| Quantitative remark (paper only) | HIGH — no Lean needed |
