# Universal Explanation Impossibility — Legendary Paper Plan

**Goal**: Execute the 6 items that elevate this from "very good PNAS
paper" to "foundational result that gets used daily."

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Current state**: 73 files, 346 theorems, 72 axioms, 0 sorry.
9 instances, 7 experiments, 5 paper versions. All audited.
**Model assignment**: Opus for proofs/theory/writing. Sonnet for
experiments/scaffolding/mechanical.

---

## VET RECORD (3 rounds on this plan)

### Round 1 — Factual

- The quantitative bound (Item 1) is the hardest task. It requires
  new Lean theorems relating Rashomon set structure to minimum
  unfaithfulness. The abstract framework may not have enough
  structure for a closed-form bound without additional axioms.
  ⚠️ REVISED: Include a FALLBACK — if the abstract bound doesn't
  close, prove it for the attribution instance only (where 1/(1-ρ²)
  is already established) and state the abstract version as a
  conjecture.

- Full retraining (Item 2) requires GPU for 20 DistilBERT fine-tuning
  runs on SST-2. On MPS (Apple Silicon, 16GB), this is feasible but
  slow (~15-30 min per run = 5-10 hours total). ⚠️ REVISED: Use
  a smaller task (100 training sentences, 3 epochs, freeze all but
  last 2 transformer layers) to keep wall time under 2 hours.

- The info-theoretic proof (Item 4) doesn't need Lean — a rigorous
  paper proof suffices. But the DPI argument requires MEASURE-THEORETIC
  definitions of faithfulness/stability/decisiveness, which may not
  map cleanly to the algebraic definitions. ⚠️ REVISED: State the
  info-theoretic version as a PARALLEL formulation, not as a proof
  of the same theorem. The algebraic and info-theoretic versions are
  complementary perspectives, not the same result in different language.

### Round 2 — Reasoning

- The quantitative bound should be stated carefully. "Minimum price
  of stability" means: among all stable E, what is the minimum
  distance from E to explain? This requires a METRIC on H, which
  the abstract framework doesn't have. ⚠️ REVISED: For the abstract
  framework, count the number of fibers where E must "abstain"
  (fail to be decisive). For specific instances, use domain-specific
  metrics (ranking distance for attributions, IoU for saliency, etc.).

- The full retraining experiment could WEAKEN the paper if the flip
  rate is low. This is an honest risk. ⚠️ REVISED: Run the experiment
  BEFORE writing about it. If flip rate < 10%, report honestly and
  discuss: "The Rashomon property for attention under standard
  fine-tuning is modest, consistent with recent evidence that
  pretrained representations exhibit approximate linear mode
  connectivity." This is honest and interesting — it distinguishes
  instances where the impossibility bites hard (attributions,
  concepts) from those where it's theoretically present but
  practically mild (attention).

- The regulatory application should cite SPECIFIC paragraphs of
  SR 11-7, not just the document name. ⚠️ REVISED: Reference
  SR 11-7 Section III.A (conceptual soundness), Section III.B
  (outcomes analysis), Section IV (effective challenge).

### Round 3 — Omissions

- ⚠️ The plan doesn't address REMOVING the weak cross-domain claims
  (QM, legal, cryptography) from the existing paper files. Added as
  Task 3.2.

- ⚠️ The plan doesn't update the co-author message draft with the
  vet corrections. Not in scope — user can handle separately.

- ⚠️ The plan doesn't address the LLM experiment renaming (from
  "LLM Self-Explanation" to "Attention-Based Citation" or similar).
  Added as Task 3.3.

- ⚠️ The plan doesn't specify which paper versions to update. For
  efficiency: update PNAS version only for Items 1-6. Propagate to
  other versions in a final integration step.

---

## Phase 1: Quantitative Bound [Opus]

The single hardest and highest-impact addition.

### Task 1.1: Abstract Counting Bound in Lean

**File**: `UniversalImpossibility/QuantitativeBound.lean`

The simplest quantitative version: count the number of observe-fibers
where ANY stable E must fail to be decisive.

Setup: An ExplanationSetup (unbundled, from Necessity.lean) with
FINITELY many configurations.

```
-- A "Rashomon fiber" is a fiber of observe where incompatible
-- explanations exist.
def rashomon_fiber (S : ExplanationSetup Θ H Y) (y : Y) : Prop :=
  ∃ θ₁ θ₂ : Θ, S.observe θ₁ = y ∧ S.observe θ₂ = y ∧
    S.incompatible (S.explain θ₁) (S.explain θ₂)

-- On every Rashomon fiber, a stable + faithful E must fail to
-- be decisive (it cannot inherit all incompatibilities).
theorem stable_faithful_not_decisive_on_rashomon_fiber
    (S : ExplanationSetup Θ H Y)
    (y : Y) (hr : rashomon_fiber S y)
    (E : Θ → H) (hs : stableS S E) (hf : faithfulS S E) :
    ¬(∀ (θ : Θ) (h : H), S.observe θ = y →
      S.incompatible (S.explain θ) h → S.incompatible (E θ) h)
```

Proof sketch: On a Rashomon fiber, ∃ θ₁ θ₂ with same observe,
incompatible explains. Stable → E θ₁ = E θ₂. Call this value e.
Faithful at θ₁: ¬incompatible(e, explain θ₁).
Faithful at θ₂: ¬incompatible(e, explain θ₂).
If E were decisive at θ₁: incompatible(explain θ₁, explain θ₂) →
incompatible(e, explain θ₂). But faithful says ¬incompatible(e, explain θ₂).
Contradiction. So E is not decisive at θ₁.

This is essentially the impossibility theorem applied per-fiber,
showing that EACH Rashomon fiber forces a decisiveness failure.

The COUNTING version (for finite types): the number of fibers where
E fails to be decisive is at least the number of Rashomon fibers.
This may require Fintype assumptions.

If the Lean proof is too complex with finiteness, state the per-fiber
theorem (which is clean) and note the counting corollary in the paper.

Run `lake build`. Zero sorry.

### Task 1.2: Attribution-Specific Bound [Opus]

**File**: Add to paper (not Lean — paper proof)

For the attribution instance with collinearity ρ:
- The attribution ratio between collinear features diverges as 1/(1-ρ²)
- Any stable attribution must assign EQUAL importance to symmetric features
  (by the G-invariant resolution)
- The minimum unfaithfulness = |true_ratio - 1| where true_ratio = 1/(1-ρ²)

This gives: min unfaithfulness ≥ ρ²/(1-ρ²) → ∞ as ρ → 1.

State in the paper as: "For GBDT attributions with feature correlation ρ,
the minimum unfaithfulness of any stable explanation method is ρ²/(1-ρ²),
achieved uniquely by the DASH consensus. This bound is proved in the
companion paper [cite dash-impossibility-lean]."

This connects the abstract impossibility to a concrete, computable number.

---

## Phase 2: Full Retraining Experiment [Sonnet]

### Task 2.1: Full Fine-Tuning Attention Experiment

**File**: `paper/scripts/attention_full_retraining_experiment.py`

Design:
1. Load DistilBERT-base-uncased
2. Freeze all layers EXCEPT the last 2 transformer layers + classification head
3. Fine-tune on 200 sentiment sentences (100 positive, 100 negative)
   for 5 epochs with 20 different random seeds
4. Use Adam lr=2e-5, batch_size=16 (standard BERT fine-tuning)
5. Verify all 20 models achieve >85% accuracy on 40 held-out sentences
6. Compute attention rollout on 50 test sentences
7. Measure argmax flip rate, Kendall tau, prediction agreement
8. Compute 95% bootstrap CIs

Device: CPU (to avoid MPS hook issues with attention extraction).
Estimated time: ~2 hours (20 models × ~6 min each).

Output:
- `paper/results_attention_full_retraining.json`
- Console comparison table: perturbation vs head-only vs full retraining

**Key**: Run this BEFORE writing about it. Report honestly regardless
of outcome:
- If flip rate > 20%: "Full retraining confirms substantial attention
  instability under the Rashomon property."
- If flip rate 5-20%: "Full retraining shows moderate attention
  instability, less severe than perturbation but practically significant."
- If flip rate < 5%: "Under standard fine-tuning, the Rashomon property
  for attention is mild. The impossibility is theoretically present but
  practically modest for attention, unlike attributions and concept probes
  where instability is severe."

All three outcomes are scientifically interesting and strengthen the paper
through honesty.

---

## Phase 3: Honest Cross-Domain Tiering [Opus]

### Task 3.1: Rewrite "Implications Beyond ML" Section

Edit `paper/sections/beyond_ml.tex` (or the relevant section in
`paper/universal_impossibility_pnas.tex`).

Replace the flat list with tiered presentation:

**Tier 1 — Formal instances** (1 paragraph):
"The theorem applies directly to any domain where the ExplanationSystem
structure is instantiated. Beyond the nine ML instances, the database
view update problem (Bancilhon & Spyratos 1981) and Rashomon sets in
statistics (Fisher et al. 2019) share the exact mathematical structure."

**Tier 2 — Substantive applications** (1 paragraph):
"The degeneracy principle in neuroscience (Marder & Goaillard 2006),
partial identification in econometrics (Manski 2003), and differential
diagnosis in medicine exhibit the same structure: multiple configurations
produce identical observables. Formalizing each as an ExplanationSystem
and verifying the Rashomon property would yield the impossibility as a
corollary. We leave these formalizations to domain specialists."

**Tier 3 — Suggestive connections** (1 paragraph):
"The Quine-Duhem underdetermination thesis in philosophy of science
is the informal precursor to our result. Our theorem can be read as a
formalization of a core consequence of underdetermination: no method of
choosing between empirically equivalent theories can simultaneously be
faithful to a specific theory, stable across equivalent theories, and
decisive about theoretical commitments. The G-invariant resolution
formalizes what structural realists (Worrall 1989) have argued
informally: the stable content of a theory is its structure, not its
interpretation. We note, however, that the philosophical setting
involves normative and ontological dimensions that our descriptive
framework does not capture."

### Task 3.2: Remove Weak Claims

In ALL paper versions, remove or downgrade:
- Quantum mechanics interpretation (remove from tables and lists)
- Legal interpretation (remove from tables, keep as 1-sentence aside)
- Cryptography (remove entirely)

### Task 3.3: Rename LLM Experiment

In all paper versions and experiment files:
- "LLM Self-Explanation Instability" → "Token Citation Instability"
  or "Attention-Based Explanation Instability"
- Update the instance name in the cross-instance table
- Update the Lean file docstring (but keep the Lean theorem name
  `llm_explanation_impossibility` since renaming would break references)

---

## Phase 4: Information-Theoretic Formulation [Opus]

### Task 4.1: Write the Information-Theoretic Section

**File**: Add as a new subsection in the Resolution section of the
PNAS paper, or as a standalone section.

~0.5 pages. NOT a Lean proof — a rigorous paper proof.

**Setup**: Let (Θ, H, Y) be an explanation system with probability
measure P over Θ. Define:
- Stability ↔ E is a function of observe(Θ) only ↔ I(E; Θ | Y) = 0
  (E carries no information about Θ beyond what Y provides)
- Faithfulness ↔ I(E; explain(Θ)) is high (E captures information
  about the native explanation)
- Decisiveness ↔ H(E | Θ) is low (E is determined by the configuration)

**The DPI argument**:
Since E is a function of Θ (any explanation takes Θ as input):
  Θ → E → (anything)
By the data processing inequality:
  I(E; explain(Θ)) ≤ I(Θ; explain(Θ)) = H(explain(Θ))

If E must also factor through Y (stability):
  E = g(Y) for some function g
  I(E; explain(Θ)) = I(g(Y); explain(Θ)) ≤ I(Y; explain(Θ))

When H(explain(Θ) | Y) > 0 (the system is underspecified — explain
carries more information about Θ than observe does):
  I(Y; explain(Θ)) < H(explain(Θ))

So any stable E has strictly less information about the native
explanation than the native explanation carries about itself. The
information gap H(explain(Θ) | Y) is the fundamental limit.

**State as Proposition**: "The information-theoretic cost of stability
is at least H(explain(Θ) | Y), the conditional entropy of the native
explanation given the observables. This quantity is positive whenever
the Rashomon property holds with positive probability."

**Relationship to algebraic theorem**: "The algebraic impossibility
(Theorem 1) is a qualitative consequence: when H(explain(Θ) | Y) > 0,
no deterministic E can simultaneously maximize I(E; explain(Θ))
(faithfulness), achieve I(E; Θ | Y) = 0 (stability), and minimize
H(E | Θ) (decisiveness). The information-theoretic formulation
additionally provides a quantitative bound on the minimum information
loss."

Mark as "argued (paper proof)" in proof status transparency.

---

## Phase 5: Worked Regulatory Application [Opus]

### Task 5.1: Write SR 11-7 Analysis

**File**: Replace the current regulatory_analysis.tex with a
FOCUSED analysis of ONE framework (OCC SR 11-7) done rigorously.

~0.75 pages.

**Structure**:

**Paragraph 1 — The three SR 11-7 requirements**:
Quote the EXACT language from each section:
- Section III.A (Conceptual Soundness): "[The model's] theoretical
  construct, ... assumptions, and limitations should be assessed."
  → Maps to FAITHFUL: the explanation must reflect the model's
  actual structure and assumptions.
- Section III.B (Outcomes Analysis): "comparing the model's outputs
  with the corresponding realized outcomes" across "model versions."
  → Maps to STABLE: the explanation must be consistent across
  model versions that produce equivalent outcomes.
- Section IV (Effective Challenge): "critical analysis by objective,
  informed parties" requiring "specific findings."
  → Maps to DECISIVE: the challenge must produce specific,
  actionable findings, not hedged non-answers.

**Paragraph 2 — The impossibility**:
"These three requirements, as stated, cannot be simultaneously
satisfied for any banking model with the Rashomon property — i.e.,
any model where multiple parameter configurations achieve equivalent
predictive performance. For models trained with bagging, boosting,
or stochastic gradient descent, the Rashomon property is generic
(Section 6, Ubiquity). Our Theorem 1 applies directly."

**Paragraph 3 — The resolution**:
"Compliant model documentation under the trilemma should:
(a) State which property is prioritized for each model use case.
(b) Use ensemble explanations (DASH) for the stable + faithful
tradeoff, documenting where decisiveness is sacrificed.
(c) Report explanation uncertainty bounds (confidence intervals
on feature importance across the Rashomon set) alongside point
estimates.
(d) Document the Rashomon set size and the resulting explanation
variance as part of the model risk assessment."

**Paragraph 4 — Broader regulatory note (brief)**:
"The same analysis applies to EU AI Act Article 13 (transparency)
and NIST AI RMF MAP 1.5 (interpretability). In each case, the
regulatory language implicitly assumes all three properties are
jointly achievable. Our theorem shows they are not. Regulations
should be reformulated as explicit tradeoffs: specify which
property is required, which may be relaxed, and what resolution
method is acceptable."

Cite: occ2011sr117, euaiact2024. Add a bib entry for the specific
SR 11-7 text if possible.

---

## Phase 6: Framing Artifacts [Sonnet + Opus]

### Task 6.1: Perfect the One-Sentence Version [Opus]

The one-sentence version appears in:
- Abstract (first sentence)
- Significance statement (first sentence)
- Introduction (first paragraph)
- Conclusion (last paragraph)

It should be IDENTICAL in all four locations:

"No explanation of an underspecified system can simultaneously be
faithful, stable, and decisive."

Or the practitioner version:

"Faithful, stable, decisive — pick two."

Verify this sentence (or close variant) appears in all four locations
across ALL paper versions.

### Task 6.2: Perfect the Practitioner Decision Table [Opus]

**File**: Add to escape_routes.tex or create a new table.

| Your goal | Sacrifice | Method | Example tools |
|-----------|-----------|--------|---------------|
| Debug THIS model | Stability | Report model's own explanation | SHAP, GradCAM, circuit analysis |
| Monitor across retrains | Decisiveness | Aggregate over Rashomon set | DASH, CPDAG, ensemble probes |
| Regulatory template | Faithfulness | Fixed rule-based explanation | Predefined feature lists |
| Academic comparison | Stability | Per-model explanation + error bars | Standard XAI + uncertainty |
| Safety audit | Decisiveness | Conservative bounds | Worst-case over Rashomon set |

This table should be in the Discussion section of every paper version.

### Task 6.3: Verify Trilemma Diagram [Sonnet]

Check paper/figures/trilemma_universal.tex:
- Remove QM, legal, cryptography from the instance labels
- Verify remaining labels match the paper's instance list
- Recompile: pdflatex trilemma_universal.tex

---

## Phase 7: Integration + Final Verification [Sonnet + Opus]

### Task 7.1: Update PNAS Paper [Opus]

Integrate all new content into `paper/universal_impossibility_pnas.tex`:
- Add quantitative bound (per-fiber theorem statement)
- Add attribution-specific bound (ρ²/(1-ρ²) reference)
- Add full retraining result to the cross-instance table
- Replace cross-domain section with tiered version
- Replace regulatory section with focused SR 11-7 analysis
- Add information-theoretic proposition
- Add practitioner decision table
- Update Lean counts if changed
- Verify one-sentence version in all 4 locations

Compile. Report page count (must be ≤6).

### Task 7.2: Update Main Paper [Sonnet]

Apply the same changes to `paper/universal_impossibility.tex`.
Compile. Report page count.

### Task 7.3: Lean Build + Counts [Sonnet]

```bash
lake build
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

All must match paper. Sorry must be 0.

### Task 7.4: Final Vet [Opus]

Re-run the audit checklist from AUDIT_PROMPT_FINAL.md:
- Every number traces to source
- Every claim is tagged PROVED/DERIVED/ARGUED/EMPIRICAL
- No overclaiming (all fixes from previous audit still in place)
- Cross-domain claims are tiered, not flat
- Weak claims removed
- One-sentence version consistent across all locations

---

## Execution Order

```
Phase 1: [1.1 Lean bound] ∥ [1.2 attribution bound (paper only)]
Phase 2: [2.1 full retraining experiment]
Phase 3: [3.1 ∥ 3.2 ∥ 3.3]  (tiering, removals, rename — parallel)
Phase 4: [4.1 info-theoretic section]
Phase 5: [5.1 SR 11-7 analysis]
Phase 6: [6.1 ∥ 6.2 ∥ 6.3]  (framing artifacts — parallel)
Phase 7: [7.1] → [7.2] → [7.3] → [7.4]  (integration — sequential)
```

Phases 1-2 are the critical path (hardest items).
Phases 3-6 are independent and can run in any order after Phase 2.
Phase 7 depends on all previous phases.

## Confidence

| Phase | Confidence | Risk | Mitigation |
|-------|-----------|------|------------|
| 1.1 Abstract bound | MEDIUM | May need new axioms | Fallback: per-fiber theorem only |
| 1.2 Attribution bound | HIGH | Already established | Reference companion paper |
| 2.1 Full retraining | HIGH | Low flip rate possible | Report honestly either way |
| 3.x Cross-domain tiering | HIGH | Presentation only | Straightforward |
| 4.1 Info-theoretic | MEDIUM | DPI argument needs care | Paper proof, not Lean |
| 5.1 Regulatory | HIGH | Interpretive but careful | Quote exact SR 11-7 language |
| 6.x Framing | HIGH | Mechanical | Verification only |
| 7.x Integration | HIGH | Depends on prior phases | Final gate |
