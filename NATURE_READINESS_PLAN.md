# Nature Readiness — Full Re-Vet + Brief Communication

**Goal**: (1) Re-vet the entire monograph with the 8 derived
instances, fixing anything that would prevent Nature acceptance.
(2) Write the Nature Brief Communication (~1500 words).
(3) Rebuild the arXiv package with all fixes.

**The Nature bar**: A result that changes how scientists across
multiple fields think about a fundamental question. The 8-field
derivation is the hook. The machine-checked verification is the
differentiator. The resolution (G-invariant projection) is the
constructive contribution.

---

## Phase 1: Re-Vet the Monograph [Opus]

### Task 1.1: Verify ALL 8 derived instances

For EACH derived Lean file, verify:
- The file compiles with zero sorry
- #print axioms shows zero behavioral axioms
- The Rashomon witness is CORRECT for the domain
  (not just a generic two-element example)
- The incompatible relation is appropriate for the domain
- The observe map correctly captures the domain's "observable"
- The paper section accurately describes the Lean code

Files to check:
1. LinearSystem.lean — Ax=b, solutions (1,1) and (0,2)
2. GeneticCode.lean — UCU and UCC both encode Serine
3. GaugeTheory.lean — triangle graph, ℤ₂ edges, holonomy
4. StatisticalMechanics.lean — coins (H,T) and (T,H)
5. SyntacticAmbiguity.lean — left-attach vs right-attach
6. PhaseProblem.lean — signals (1,0) and (0,1), same energy
7. ViewUpdate.lean — rows (true,true) and (true,false)
8. MarkovEquivalence.lean — chain vs fork DAGs

For EACH: is the domain label honest? A 2-coin system labeled
"statistical mechanics" and a 2-element signal labeled "the
phase problem" are toy examples. The paper must be clear these
are MINIMAL WITNESSES of the domain-specific phenomenon, not
complete formalizations of the domain.

### Task 1.2: Verify ALL numbers in the monograph

Run the Lean count commands. Read every results JSON. Check
every number in the monograph matches its source:
- 82 files, 377 theorems, 72 axioms, 0 sorry
- All experiment numbers (attention 19.9%, counterfactual
  23.5%, concept 0.90, model selection 80%, GradCAM 9.6%,
  token citation 34.5%)
- "eight scientific domains" count
- "nine ML instances" count

### Task 1.3: Check for stale content

After adding 6 new derived instances, the monograph may have:
- Old "seven" references that should be "eight"
- Old instance counts that should be updated
- Old Lean counts (75/351 should be 82/377)
- Abstract or intro that doesn't mention all 8 domains
- Cross-instance table missing the new rows
- Unification table incomplete

### Task 1.4: Check for internal contradictions

- Does the abstract match the body?
- Does the instance list in the intro match the actual sections?
- Do the Lean appendix listings match the actual files?
  (Previous reviews found mismatches here — verify the new
  instance listings are correct or present)
- Is the theorem numbering consistent (shared counter)?
- Do all \ref and \cite resolve?

### Task 1.5: Nature-specific checks

- Is the title "The Limits of Explanation" appropriate for Nature?
- Does the abstract work for a general science audience?
  (A biologist, a physicist, and a computer scientist must
  all understand it)
- Are there any ML-specific jargon terms that would alienate
  non-ML readers? (SHAP, GradCAM, DistilBERT — these need
  brief glosses or should be in SI only)
- Is the resolution (G-invariant projection) explained in
  terms a non-ML scientist understands?
- Are all 8 domain connections accurately stated?
  (No overclaiming, no underclaiming)

---

## Phase 2: Fix Everything Found [Opus/Sonnet]

Fix all issues from Phase 1. Prioritize:
- FATAL: any factual error, any Lean mismatch
- MAJOR: any stale count, any internal contradiction
- MINOR: jargon, framing

---

## Phase 3: Write Nature Brief Communication [Opus]

### Task 3.1: Create paper/nature_brief_communication.tex

~1500 words. 2 figures max. All details in SI (the monograph).

**Structure**:

**Title**: "The Limits of Explanation"

**Abstract** (150 words max):
State the theorem, the 8 derivations, the resolution, the
Lean verification. One sentence per element.

**Main text** (~1200 words):

Paragraph 1 — The problem (4-5 sentences):
Across science, we explain systems by interpreting their
internal structure. But when multiple configurations produce
identical observable output — the Rashomon property — any
interpretation faces a fundamental constraint. We prove this
constraint is an impossibility.

Paragraph 2 — The theorem (4-5 sentences):
Define the three properties (in plain English, not formalism).
State the impossibility. State it's verified in Lean 4
(82 files, 377 theorems, 0 unproved goals).

Paragraph 3 — The eight derivations (8-10 sentences):
One sentence per domain. Each sentence names the domain,
the Rashomon source, and the concrete witness. End with:
"Each derivation requires zero shared axioms — only the
domain's own mathematical structure."

Figure 1: The unification table (all 8 domains, their
witnesses, same-observable check).

Paragraph 4 — The resolution (3-4 sentences):
The G-invariant projection. What practitioners in each
domain already do (CPDAGs, gauge-invariant observables,
microcanonical ensemble, minimum-norm solutions). The
theorem proves these are optimal.

Paragraph 5 — Implications (3-4 sentences):
The impossibility is the exact boundary (necessity).
The query-relative version characterizes which questions
are answerable. The result connects phenomena across
eight sciences that were previously studied independently.

Figure 2: The trilemma triangle diagram.

**Methods** (300 words):
Lean 4 proof assistant. Mathlib library. 82 files.
Each derived instance uses decidable computation (decide
or native_decide). The resolution framework uses Mathlib's
group action infrastructure. All code available at [URL].

**References**: ~20-30, spanning all 8 domains.

### Task 3.2: Create the Nature SI

The SI for Nature is the monograph itself. Create a brief
cover page:

paper/nature_si.tex:
"Supporting Information for 'The Limits of Explanation.'
The complete monograph, including all proofs, all 17 instances
(8 derived + 9 ML), all experiments, and the Arrow comparison,
is provided as the Supporting Information document."

Then append or reference the monograph PDF.

---

## Phase 4: Update PNAS Version for 8 Instances [Sonnet]

The PNAS 6pp version currently has 7 derived instances. Update:
- Abstract: "eight scientific domains"
- Select 4 for main text: gauge, genetics, stat mech, causal
  (physics × 2, biology, statistics — maximum breadth in 4)
- Rest in SI
- Update Lean counts
- Compile. Must be ≤6pp.

---

## Phase 5: Rebuild + Verify + Commit [Sonnet]

### 5.1: Full Lean build
lake build. 82 files, 377 theorems, 72 axioms, 0 sorry.

### 5.2: Compile ALL paper versions
Monograph (61pp), Nature Brief Comm, Nature SI, PNAS (6pp),
PNAS SI, NeurIPS (10pp), JMLR (31pp).

### 5.3: Rebuild arXiv package

### 5.4: Commit and push

---

## Execution Order

```
Phase 1: [1.1 ∥ 1.2 ∥ 1.3 ∥ 1.4 ∥ 1.5]  (all parallel)
Phase 2: fix everything found
Phase 3: [3.1 Nature BC ∥ 3.2 Nature SI]  (parallel)
Phase 4: update PNAS
Phase 5: [5.1] → [5.2] → [5.3] → [5.4]
```

## Confidence

| Phase | Confidence |
|-------|-----------|
| Re-vet (Phase 1) | HIGH — systematic checklist |
| Fixes (Phase 2) | HIGH — depends on findings |
| Nature BC (Phase 3) | HIGH — 1500 words, clear structure |
| PNAS update (Phase 4) | HIGH — mechanical |
| Verify + commit (Phase 5) | HIGH — scripted |
