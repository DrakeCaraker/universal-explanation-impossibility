# Adversarial Peer Review — Round 2

**Context**: Round 1 found 16 issues (1 FATAL, 15 MAJOR). All were
fixed. This round looks for issues the first round MISSED and
problems INTRODUCED by the fixes.

**Principle**: After 16 edits by different agents, the paper may
now contain contradictions, orphaned references, inconsistent
claims, or hedging that undermines the argument. A revision can
create new problems.

**Model**: Opus for all reviews.

---

## Review 6: The Post-Revision Auditor

**Goal**: Check that the 16 fixes are correctly applied and haven't
introduced new problems.

### Consistency between revised claims
- [ ] Does the abstract still accurately summarize the paper?
  After removing "Pareto-optimal," "first formalization,"
  "exact boundary," etc., is the abstract still compelling?
  Or has it become a series of hedges?
- [ ] Does the significance statement still justify PNAS-level
  impact? After all the downgrades, is the claim strong enough?
- [ ] Are there any sentences that now CONTRADICT each other?
  (e.g., one place says "structural analogy" and another
  still says "instantiation")
- [ ] Do all cross-references still resolve after edits?
- [ ] Are theorem/proposition numbers still sequential?
- [ ] Are Table 1 and Table 2 internally consistent with
  the text around them?

### Hedging audit
- [ ] Count the number of "to our knowledge" qualifiers.
  More than 3 in a 6-page paper is excessive hedging.
- [ ] Count the number of "suggestive," "reminiscent,"
  "analogous." Too many weakens the paper to the point
  where the reader asks "what ARE you claiming?"
- [ ] Is there a clear, unhedged statement of the main
  contribution? The paper must have at least ONE sentence
  that says what it proves without qualification.

### Orphan check
- [ ] Are there any citations added to the bib that are
  never cited in the paper? (laudan1991empirical,
  dorling1979bayesian, howson2006scientific)
- [ ] Are there any \ref or \cite that no longer resolve?
- [ ] Do footnotes in the cross-instance table compile
  correctly?

---

## Review 7: The Line-by-Line Lean Auditor

**Goal**: Read the 5 core Lean files character by character.
The first review checked structure; this one checks SUBSTANCE.

### ExplanationSystem.lean
- [ ] Read the ENTIRE file (should be ~120 lines).
- [ ] For each definition: is the Lean code EXACTLY what
  the paper says? Check variable names, quantifier order,
  implication direction.
- [ ] The `faithful` definition: the paper says "E(θ) never
  contradicts explain(θ)." The Lean says `¬S.incompatible
  (E θ) (S.explain θ)`. Is the argument ORDER correct?
  Does `incompatible a b` mean "a contradicts b" or
  "a and b are mutually contradictory"? If incompatible
  is symmetric, order doesn't matter. If not, the direction
  matters. Is incompatibility SYMMETRIC in the formalization?
  (It's not required to be — only irreflexive.)
- [ ] The `decisive` definition: `S.incompatible (S.explain θ) h
  → S.incompatible (E θ) h`. The first argument is
  `S.explain θ`, not `E θ`. Is this the intended direction?
  If `incompatible` is not symmetric, this says: "if explain(θ)
  contradicts h, then E(θ) also contradicts h" — E inherits
  explain's left-incompatibilities. What about right-
  incompatibilities? Is there a hidden asymmetry?
- [ ] The proof of `explanation_impossibility`: trace each step.
  Line 72: `hd θ₁ (S.explain θ₂) hinc` — this calls decisive
  at θ₁ with h = explain(θ₂). This gives
  `incompatible(E θ₁, explain θ₂)`. But decisive says
  `incompatible(explain θ₁, h) → incompatible(E θ₁, h)`.
  So hinc must be `incompatible(explain θ₁, explain θ₂)`.
  Check: does the Rashomon property give us
  `incompatible(explain θ₁, explain θ₂)` or
  `incompatible(explain θ₂, explain θ₁)`? If incompatible
  is not symmetric, one direction may not give the other.
  IS THIS A BUG?
- [ ] The tightness proof for `tightness_faithful_decisive`:
  faithful requires `¬incompatible(explain θ, explain θ)`.
  This is `incompatible_irrefl (explain θ)`. Correct.
  BUT: this uses irreflexivity on the FIRST argument
  position. The definition is `¬incompatible (E θ) (explain θ)`
  — first argument is E(θ), second is explain(θ). When
  E = explain, this becomes `¬incompatible (explain θ) (explain θ)`.
  Irreflexivity gives `¬incompatible h h` for any h. So
  `¬incompatible (explain θ) (explain θ)` follows. CORRECT.

### Necessity.lean
- [ ] Read the ENTIRE file.
- [ ] `fully_specified_possibility`: the stability step uses
  `h_inj θ₁ θ₂ hobs` to get `θ₁ = θ₂`, then `subst` + `rfl`.
  This is correct ONLY if `h_inj` gives `θ₁ = θ₂` (not
  `θ₂ = θ₁`). Check the quantifier direction.
- [ ] `no_rashomon_from_all_three`: does this genuinely prove
  the CONTRAPOSITIVE, or is it just the impossibility theorem
  restated? If it's `¬(∃ rashomon pair)` derived from
  `∀ E, ¬(F ∧ S ∧ D)`, that's not quite the contrapositive.
  The contrapositive would be: `(∃ E, F ∧ S ∧ D) → ¬Rashomon`.
  Check which one is proved.

### MarkovEquivalence.lean
- [ ] Read the ENTIRE file.
- [ ] Verify: `chain = ⟨true, false, true, false, false, false⟩`.
  Is edge01=true (0→1), edge12=true (1→2)? So chain is
  0→1→2. Correct.
- [ ] Verify: `fork = ⟨false, true, true, false, false, false⟩`.
  Is edge10=true (1→0), edge12=true (1→2)? So fork is
  0←1→2. Correct.
- [ ] The incompatibility relation `fun g₁ g₂ => g₁ ≠ g₂`:
  this IS symmetric (a ≠ b iff b ≠ a). So the asymmetry
  concern from ExplanationSystem.lean is moot for THIS
  instance. But for OTHER instances where incompatibility
  might not be symmetric, the direction matters.

### QuantitativeBound.lean
- [ ] Read the ENTIRE file.
- [ ] `decisive_fails_on_rashomon_pair`: the proof does
  `hd θ₁ (S.explain θ₂) hinc` — same pattern as the main
  theorem. Check that `hinc` has the right type.

### AxiomSubstitution.lean
- [ ] Read the results table at the top.
- [ ] `decisive_implies_complete`: does the proof correctly
  show that current decisive + faithful → complete?
- [ ] `surjective_consistent`: read the counterexample.
  Is the Bool/Bool/Unit system correctly constructed?
  Does the asymmetric incompatibility actually break
  the impossibility?

---

## Review 8: PNAS Format Compliance

**Goal**: Check every PNAS formatting requirement.

### Mandatory PNAS elements
- [ ] Classification line (e.g., "Physical Sciences /
  Applied Mathematics" or "Computer Sciences"). Present?
- [ ] Keywords. Present?
- [ ] Data availability statement. Present?
- [ ] Code availability statement. Present?
- [ ] Competing interests declaration. Present?
- [ ] Author contributions statement. Present?
- [ ] ORCID for each author. Present?
- [ ] Word count: ≤4500 main text words?
- [ ] References: PNAS style (numbered, not author-year)?
  NOTE: PNAS uses numbered references, not natbib.
  If the paper uses \citep/\citet, this is WRONG for PNAS.
- [ ] Figures: ≤6? At print quality (300 DPI)?
- [ ] SI structured with "SI Appendix" labels?

### Formatting check
- [ ] Title: ≤150 characters?
- [ ] Abstract: ≤250 words?
- [ ] Significance statement: ≤120 words?
- [ ] Two-column format?
- [ ] Font size ≤9pt for main text?

---

## Review 9: SI Consistency

**Goal**: Verify the SI is consistent with the revised main text.

- [ ] Every claim in the main text that says "see SI" has
  a corresponding section in the SI
- [ ] SI theorem counts match main text
- [ ] SI experimental results match main text table numbers
- [ ] SI does not contain claims that have been downgraded
  in the main text (e.g., old "Pareto-optimal" language,
  old "first formalization" language)
- [ ] The SI practitioner decision table is present (moved
  from main text in the PNAS trimming)
- [ ] SI citations are in the bibliography

---

## Review 10: PNAS Editor Simulation

**Goal**: Would a PNAS editor send this to review or desk-reject?

### Scope assessment
- [ ] Is this clearly a PNAS paper (broad scientific interest)
  or an ML-specialist paper that belongs at NeurIPS?
- [ ] Does the cross-domain framing (philosophy, neuroscience,
  economics) feel genuine or bolted-on?
- [ ] Would a biologist or physicist reading the abstract
  understand the contribution?
- [ ] Is the significance statement compelling enough to
  justify PNAS over a specialist journal?

### Editor red flags
- [ ] Is the paper too hedged? (Multiple "to our knowledge,"
  "suggestive," "illustrative" qualifiers)
- [ ] Is the result too narrow for PNAS? (It's fundamentally
  an ML theory paper)
- [ ] Is the result too simple for PNAS? (4-line proof)
- [ ] Does the paper have the wrong tone? (PNAS expects
  scientific gravitas, not ML conference energy)

### Verdict
- [ ] Would this be sent to review? YES / NO / UNCERTAIN
- [ ] If YES, which reviewers would the editor choose?
  (This determines which of our 5 reviewer archetypes
  we'd actually face)

---

## Meta-Review 2

After Reviews 6-10:

1. List ALL new findings (FATAL / MAJOR / MINOR / STYLISTIC)
2. For each MAJOR: is it fixable?
3. The single most important remaining issue
4. Updated recommendation: Strong Accept / Accept / Weak Accept /
   Weak Reject / Reject
5. Is the paper ready for submission? YES / NO / CONDITIONAL
