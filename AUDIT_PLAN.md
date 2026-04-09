# Universal Explanation Impossibility — Full Rigor Audit

**Goal**: Every assumption, axiom, proof, experiment, and claim
undergoes rigorous examination. The paper must sail through
PNAS peer review from any reviewer archetype (theorist, empiricist,
philosopher, Lean expert, domain skeptic).

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Lean state**: 73 files, 346 theorems, 72 axioms, 0 sorry
**Paper**: 5 versions (PNAS 6pp, monograph 39pp, JMLR 30pp, NeurIPS 10pp, base 31pp)

**Model assignment**: Opus for all audit tasks (requires judgment).
Sonnet for mechanical fixes identified by audit.

**Critical rule**: If the audit finds a FUNDAMENTAL issue (wrong
definition, invalid proof, flawed experiment design), STOP and
fix it before continuing the audit. Do not accumulate unfixed
fundamental issues.

---

## VET RECORD (3 rounds on this plan itself)

### Round 1 — Factual Accuracy of the Plan

1. **72 axioms**: The plan must account for ALL axioms. They
   fall into categories: type declarations (Model, numTrees, etc.),
   behavioral (firstMover_surjective, etc.), infrastructure
   (modelMeasurableSpace, etc.), instance types (AttentionConfig,
   etc.), instance systems (attentionSystem, etc.). Each category
   needs different justification.

2. **346 theorems**: Too many for individual review. The plan
   uses a STRATIFIED SAMPLING strategy: audit ALL theorems in
   the universal framework files (ExplanationSystem, Necessity,
   MarkovEquivalence, AxiomSubstitution, tightness, instances)
   plus SPOT-CHECK 10 theorems from the attribution-specific files.

3. **7 experiments**: attention (perturbation + retraining),
   counterfactual, concept probe, model selection, GradCAM,
   LLM explanation. Each needs methodology + statistics + code
   audit.

4. ⚠️ **Missing from initial draft**: The plan must also audit
   the PNAS-specific version (not just the main paper) since
   that's the primary submission target. Added as Phase 5B.

5. ⚠️ **Missing**: Code-paper correspondence — does each
   experiment script actually produce the numbers cited in the
   paper? Added as Phase 3C.

### Round 2 — Reasoning Quality of the Plan

1. **Audit ordering is correct**: definitions before proofs,
   proofs before claims, experiments independently.

2. **Fundamental issue protocol**: The plan correctly gates on
   fundamental issues. If a definition is wrong (like the
   earlier faithful = E = explain bug), nothing downstream is
   valid.

3. ⚠️ **Potential overclaim in the plan**: "sail through peer
   review" is overclaiming. Revised to: "withstand rigorous
   peer review from any reviewer archetype."

4. The plan's pass/fail criteria are concrete and measurable
   for Lean items (compilation, #print axioms) but need to be
   more precise for paper claims (what counts as "justified"?).
   Added: each paper claim tagged as PROVED/DERIVED/ARGUED/EMPIRICAL.

### Round 3 — Omissions in the Plan

1. ⚠️ **Self-plagiarism check**: The universal paper shares
   Lean code with the attribution paper. PNAS may flag this.
   Added: verify that the papers have different tex content
   and the shared Lean code is properly attributed.

2. ⚠️ **Meloux et al. verification**: We STILL cannot
   independently verify this citation. The plan must flag this
   and provide a fallback (cite SAE seed paper instead, or
   soften to "recent evidence suggests").

3. ⚠️ **The attention perturbation vs retraining gap**: 60% vs
   2.8% is a large discrepancy. The paper must explain this
   clearly. The audit must verify the explanation is scientifically
   sound.

4. ⚠️ **GradCAM weak result**: 9.6% peak flip rate is below
   the 20% practical significance threshold stated in the
   validation plan. The paper must either justify this as still
   meaningful or flag it as a limitation.

5. ⚠️ **Tightness conditional witnesses**: tightness_faithful_stable
   and tightness_stable_decisive require existence of "neutral"
   and "committal" elements. Are these guaranteed to exist for
   each instance? If not, tightness is conditional.

---

## Phase 1: LEAN FORMALIZATION AUDIT

### Task 1.1: Definition Correctness [Opus]

Read UniversalImpossibility/ExplanationSystem.lean.
For EACH definition, answer:

**ExplanationSystem structure:**
- [ ] `observe : Θ → Y` — Does this correctly model "what the system produces"?
- [ ] `explain : Θ → H` — Does this correctly model "the native explanation"?
- [ ] `incompatible : H → H → Prop` — Is this the right notion of disagreement?
- [ ] `incompatible_irrefl` — Is irreflexivity justified? Could two genuinely different explanations both map to the same H value?
- [ ] `rashomon` — Is the existential the right strength? Should it be universal (for ALL observe-equivalent pairs) or existential (at least one)?

**faithful:**
- [ ] `∀ θ, ¬S.incompatible (E θ) (S.explain θ)` — Does "E never contradicts explain" correctly capture faithfulness?
- [ ] Is this too weak? Should faithful require E to AGREE with explain, not just not-contradict?
- [ ] Is this too strong? Are there reasonable explanation methods that are "faithful" but would fail this definition?
- [ ] For each of the 9 instances: does this definition match what "faithful" means in that domain?

**stable:**
- [ ] `∀ θ₁ θ₂, observe θ₁ = observe θ₂ → E θ₁ = E θ₂` — Is this correct?
- [ ] Should stability be approximate (E θ₁ ≈ E θ₂) rather than exact?
- [ ] The paper claims stability means "factors through observe." Is `E θ₁ = E θ₂` exactly this?

**decisive:**
- [ ] `∀ θ h, S.incompatible (S.explain θ) h → S.incompatible (E θ) h` — Does "E inherits all incompatibilities" correctly capture decisiveness?
- [ ] Is this the right direction? Should it be `incompatible (E θ) h → incompatible (explain θ) h` (the converse)?
- [ ] The axiom substitution test showed this is the UNIQUE calibration. Verify this is correctly reported.

**PASS CRITERIA**: Each definition is justified by (a) informal motivation from practice, (b) the axiom substitution test showing it's uniquely optimal, and (c) the tightness proof showing it's not vacuous.

### Task 1.2: Axiom Audit [Opus]

Run: `grep -n "^axiom" UniversalImpossibility/*.lean`

For EACH axiom, classify and justify:

| Classification | Justification required |
|---------------|----------------------|
| Type declaration | Minimal — just declares a type exists |
| System axiom | The Rashomon property is bundled — justified by empirical evidence for each instance |
| Behavioral axiom | Must be justified by domain knowledge. CAN it be derived instead? |
| Infrastructure | Mathlib connection — justified by mathematical convention |

**SPECIAL ATTENTION to instance axioms:**
For each instance file (Attention, Counterfactual, Concept, Causal,
ModelSelection, GradCAM, LLMExplanation, MechInterp, Attribution):
- [ ] The axiomatized system bundles `incompatible_irrefl`. Is this correct for this domain? Could two genuinely different explanations be equal in H?
- [ ] The axiomatized Rashomon property: is the empirical evidence cited in the proof sketch sufficient to justify this axiom?

**EXCEPTION — MarkovEquivalence.lean:**
- [ ] Verify that `causalSystemDerived` has ZERO axioms in its dependency chain (the Rashomon property is derived)
- [ ] Run: `#print axioms causal_impossibility_derived` — must show only structural axioms (propext, etc.), no behavioral axioms
- [ ] Verify the chain/fork CI computation is correct: does `ciFromDAG chain = ciFromDAG fork` actually hold by computation?

**PASS CRITERIA**: Every axiom is classified. Every behavioral axiom is justified. The causal instance has zero behavioral axioms.

### Task 1.3: Proof Audit [Opus]

For EACH theorem in the universal framework files, verify:

**ExplanationSystem.lean:**
- [ ] `explanation_impossibility`: Run `#print axioms`. Verify zero behavioral axioms. Read the proof term. Verify it uses all three properties (decisive at line X, stable at line Y, faithful at line Z).
- [ ] `tightness_faithful_decisive`: Verify E = explain actually satisfies both. Check: faithful by irreflexivity (is this correct?), decisive by identity (trivially).
- [ ] `tightness_faithful_stable`: Verify the neutral-element witness. Is the hypothesis `∀ θ, ¬incompatible c (explain θ)` achievable for each instance?
- [ ] `tightness_stable_decisive`: Verify the committal-element witness. Is the hypothesis `∀ θ h, incompatible (explain θ) h → incompatible c h` achievable for each instance?

**Necessity.lean:**
- [ ] `impossibility_from_rashomon`: Same proof structure as the bundled version? Verify.
- [ ] `fully_specified_possibility`: Verify the proof. The stable step uses `h_inj θ₁ θ₂ hobs` to get `θ₁ = θ₂`, then `subst` + `rfl`. Is this correct?
- [ ] `no_rashomon_from_all_three`: This should be trivial (contradiction with impossibility). Verify.

**AxiomSubstitution.lean:**
- [ ] For each alternative formalization: is the claimed result (impossibility yes/no, tightness yes/no) actually what the Lean code proves?
- [ ] `no_tightness_stable_complete`: This says stable + complete is ALREADY impossible. Verify the proof. Does this correctly show the "complete" formalization is too strong?
- [ ] `surjective_consistent`: This gives a concrete counterexample. Verify the counterexample is valid.

**MarkovEquivalence.lean:**
- [ ] `chain_fork_same_ci`: Verified by `decide`. But is `ciFromDAG` correctly computing d-separation? The formula `!g.edge02 && !g.edge20 && (g.edge01 || g.edge10) && (g.edge12 || g.edge21)` — is this actually d-separation for 3 nodes?
- [ ] Is the simplified CI formula correct for ALL 3-node DAGs, or only for the chain and fork? A PNAS reviewer might construct a counterexample DAG.
- [ ] The incompatibility relation is `g₁ ≠ g₂`. Is "different DAGs are incompatible" the right notion? Should incompatibility be "different edge orientations" rather than "different DAGs"?

**PASS CRITERIA**: Every proof in the universal framework is verified. Every `#print axioms` output is documented. The Markov equivalence formalization is correct.

### Task 1.4: Instance Theorem Audit [Opus]

For EACH of the 9 instance files:
- [ ] The theorem applies `explanation_impossibility` to the instance system
- [ ] The proof compiles (already verified by `lake build`)
- [ ] Run `#print axioms <instance_impossibility>` and document the axiom dependencies

**PASS CRITERIA**: All 9 instance theorems compile and have documented axiom dependencies.

---

## Phase 2: MATHEMATICAL RIGOR AUDIT

### Task 2.1: Proof Structure Essentiality [Opus]

Verify that the proof GENUINELY requires all three properties.
This is the most important mathematical check.

Method: For each property, attempt to prove the impossibility
WITHOUT that property. If the proof goes through, the property
is redundant (like the original faithful = E = explain bug).

- [ ] Try: `theorem without_faithful (S) (E) (hs : stable S E) (hd : decisive S E) : False` — This should NOT be provable. Verify by attempting and failing.
- [ ] Try: `theorem without_stable (S) (E) (hf : faithful S E) (hd : decisive S E) : False` — This should NOT be provable.
- [ ] Try: `theorem without_decisive (S) (E) (hf : faithful S E) (hs : stable S E) : False` — This should NOT be provable.

If ANY of these is provable, the trilemma has collapsed.

**PASS CRITERIA**: All three "without" attempts fail (ideally with a concrete counterexample showing why).

### Task 2.2: Tightness Witness Existence [Opus]

The tightness proofs are CONDITIONAL: they assume the existence
of neutral/committal elements. For a PNAS reviewer, we need to
show these elements exist for at least the key instances.

For each tightness theorem:
- [ ] `tightness_faithful_stable`: Does a "neutral" element exist for the attribution instance? (Yes: the trivial ranking where all features are tied.) For attention? (Yes: uniform attention.) For causal? (Yes: the empty graph or fully connected graph.)
- [ ] `tightness_stable_decisive`: Does a "committal" element exist? This is harder. For attribution: any total ranking. For causal: any DAG. For attention: any attention distribution with a unique argmax.

Document which instances have concrete witnesses and which are
conditional.

**PASS CRITERIA**: At least 3 instances have concrete witnesses for both tightness theorems. Others are flagged as conditional.

### Task 2.3: Necessity Scope [Opus]

The necessity theorem (`fully_specified_possibility`) requires
`observe` to be injective. This is a strong condition.

- [ ] Is the paper clear that necessity holds for FULLY SPECIFIED systems only?
- [ ] Does the paper discuss what happens for partially specified systems WITHOUT Rashomon?
- [ ] Is there a concrete example of a partially specified system without Rashomon where all three ARE achievable? (This would strengthen the claim.)
- [ ] Is there a concrete example where they are NOT achievable even without Rashomon? (This would show the necessity is incomplete.)

**PASS CRITERIA**: The paper clearly states the scope of necessity and doesn't overclaim.

---

## Phase 3: EXPERIMENTAL METHODOLOGY AUDIT

### Task 3.1: Design Audit [Opus]

For EACH of the 7 experiments, answer:

1. **Does it measure what it claims?**
   - Attention (perturbation): Claims to show Rashomon → instability. But perturbation ≠ retraining. Is this a valid proxy?
   - Attention (retraining): Claims to show head-only retraining is stable. But the backbone is frozen — this doesn't test the Rashomon property for the full model.
   - Counterfactual: Claims to show direction instability. Is "greedy perturbation toward positive centroid" a standard CF method? Would DiCE give different results?
   - Concept probe: Claims to show representation instability. Is "curved vs angular" a meaningful concept? Would a less arbitrary concept give different results?
   - Model selection: Claims to show best-model instability. Is subsample=0.8 sufficient to create a genuine Rashomon set?
   - GradCAM: Claims to show saliency instability. σ=0.0005 gives 9.6% — is this below practical significance?
   - LLM explanation: Claims to show citation instability. But it reuses the attention infrastructure — is this genuinely about "explanations" or just "attention with extra steps"?

2. **Negative control validity:**
   - Attention: single-token inputs (2% flip). Valid — trivially no choice. ✓
   - Counterfactual: separable data (0% flip). Valid — boundary is unique. ✓
   - Concept probe: LogReg (1.00 cos). Valid — convex, no representation. ✓
   - Model selection: identical models (0% flip). Valid — deterministic. ✓
   - GradCAM: identical weights (1.00 IoU). Valid — deterministic. ✓
   - LLM: top-quartile consensus sentences (1.00). Is this a valid control or circular reasoning (selecting sentences where models agree, then noting they agree)?

3. **Resolution test validity:**
   - Do the resolution tests actually demonstrate that G-invariant aggregation improves stability?
   - Attention: +15.2pp (individual-vs-avg > pairwise). ✓
   - Counterfactual: 92.7% consensus. ✓
   - Concept: +17.9pp avg CAV. ✓
   - Model selection: ensemble reduces variance. ✓
   - GradCAM: +1.5% IoU. Marginal. Flag as weak evidence.

### Task 3.2: Statistical Audit [Opus]

For EACH experiment:
- [ ] Are confidence intervals computed correctly (bootstrap with B=2000)?
- [ ] Are the CIs for the RIGHT quantity (the mean, not a single observation)?
- [ ] Is the sample size sufficient for the claimed precision?
- [ ] Are multiple comparisons addressed? (6 experiments, each with 3 tests = 18 tests. Bonferroni correction would set α = 0.05/18 ≈ 0.003.)
- [ ] Are seeds recorded and reported?
- [ ] Are package versions pinned?

### Task 3.3: Code-Paper Correspondence [Sonnet]

For EACH experiment, verify that the numbers in the paper
EXACTLY match the numbers in the results JSON file:
- [ ] Read paper/results_attention_instability.json → check paper cites 60.0%
- [ ] Read paper/results_counterfactual_instability.json → check paper cites 23.5%
- [ ] Read paper/results_concept_probe_instability.json → check paper cites 0.90
- [ ] Read paper/results_model_selection_instability.json → check paper cites 80%
- [ ] Read paper/results_gradcam_instability.json → check paper cites 9.6%
- [ ] Read paper/results_llm_explanation_instability.json → check paper cites 34.5%
- [ ] Read paper/results_attention_retraining.json → check paper cites 2.8%

**PASS CRITERIA**: Every number in the paper traces to a results file. No stale numbers.

---

## Phase 4: CLAIMS & LITERATURE AUDIT

### Task 4.1: Novelty Claim Verification [Opus]

For EACH of the 5 novelty claims:

1. "First formal theorem from Quine-Duhem"
   - [ ] Verified against literature review (Phase 1.1 of POSITIONING_PLAN.md): Okasha applied Arrow but proved no new theorem. Van Fraassen, Stanford, Norton — all verbal.
   - [ ] Search risk: recent (2024-2026) philosophy preprints? The plan flags this as unverifiable.
   - [ ] Soften to: "To our knowledge, the first..." in the paper.

2. "First unification of domain-specific impossibilities"
   - [ ] Verified: Bilodeau (Shapley-specific), Chouldechova (fairness-specific), Verma & Pearl (causal-specific). No prior unification.
   - [ ] Does Rao (2025) attempt unification? MUST CHECK.

3. "First Hunt-Stein ↔ XAI connection"
   - [ ] Literature search found no prior connection.
   - [ ] Soften to: "To our knowledge..."

4. "First proof-assistant XAI impossibility"
   - [ ] Nipkow did Arrow in Isabelle. No XAI impossibility in any prover.
   - [ ] Verified.

5. "Rashomon as exact boundary"
   - [ ] Verified: necessity + sufficiency. But necessity is scoped to fully specified systems.
   - [ ] Ensure the paper doesn't overclaim beyond this scope.

### Task 4.2: Citation Accuracy [Sonnet]

For EACH citation in the PNAS paper:
- [ ] Does the cited paper actually say what we claim it says?
- [ ] Special attention: Meloux et al. (2025) — we cannot independently verify the numbers (85 circuits, 535.8 interpretations). Flag in limitations or soften to "Meloux et al. report..."
- [ ] Okasha (2011) — verify he applied Arrow, did NOT prove new theorem
- [ ] Marder & Goaillard (2006) — verify the degeneracy claim

### Task 4.3: Overclaiming Audit [Opus]

Read the PNAS abstract and significance statement word by word.
Flag EVERY sentence that could be challenged:

- "fundamental tradeoff" — is it fundamental or specific to this formalization?
- "formalizes the century-old Quine-Duhem underdetermination thesis" — does it FORMALIZE Quine-Duhem or formalize SOMETHING LIKE Quine-Duhem?
- "every major explanation paradigm" — are there paradigms we missed?
- "uniquely optimal" — is the resolution UNIQUE or just Pareto-optimal?
- "mechanically verified, eliminating logical gaps" — does Lean verification eliminate ALL logical gaps, or only gaps in the formal proof (not in the formalization choices)?

For each flag: either fix the language or add a qualification.

---

## Phase 5: PAPER PRESENTATION AUDIT

### Task 5.1: PNAS Version Specific [Opus]

- [ ] Does the abstract fit in 150 words?
- [ ] Does the significance statement fit in 120 words?
- [ ] Is the main text ≤ 4500 words?
- [ ] Are there ≤ 6 figures?
- [ ] Does the paper open domain-general (not ML-specific)?
- [ ] Is the theorem stated accessibly for a non-ML audience?
- [ ] Are the 4 selected instances the right choice for cross-domain breadth?
- [ ] Does the SI contain everything a reviewer would want to check?

### Task 5.2: Cross-Reference Audit [Sonnet]

- [ ] Every \ref resolves
- [ ] Every \cite resolves
- [ ] Every theorem name in the paper matches a Lean theorem name
- [ ] Every Lean file name in the paper matches an actual file
- [ ] The Lean counts in the paper match grep output

### Task 5.3: Figure Audit [Sonnet]

- [ ] Trilemma diagram: are the labels correct? Do the escape routes match the paper text?
- [ ] All experiment figures: do they match the results JSONs?
- [ ] Are figures readable at print size?
- [ ] Do figures use colorblind-safe palettes?

---

## Phase 6: REMEDIATION

### Task 6.1: Fix All Issues Found [Opus + Sonnet]

For each issue found in Phases 1-5:
- FUNDAMENTAL: fix immediately, re-audit affected downstream items
- SERIOUS: fix before submission
- MINOR: fix or flag as known limitation
- STYLISTIC: fix if time allows

### Task 6.2: Update All Paper Versions [Sonnet]

After fixes, propagate changes to ALL 5 paper versions.
Recompile all. Verify page counts.

---

## Phase 7: FINAL VERIFICATION

### Task 7.1: Clean Lean Build [Sonnet]

```bash
cd ~/ds_projects/universal-explanation-impossibility
rm -rf .lake/build  # clean build cache
lake build 2>&1 | tee build_log.txt
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
```

All counts match paper. Sorry = 0. Zero errors.

### Task 7.2: Experiment Reproducibility [Sonnet]

```bash
python paper/scripts/run_all_universal_experiments.py
```

All experiments complete without error. All results match paper.

### Task 7.3: Paper Compilation [Sonnet]

Compile all 5 versions. Zero LaTeX errors. Zero undefined references.
Zero undefined citations.

### Task 7.4: Final Vet [Opus]

Run full /vet protocol on the PNAS version:
- Round 1: every number cross-checked against source
- Round 2: every claim tagged PROVED/DERIVED/ARGUED/EMPIRICAL
- Round 3: what would a hostile PNAS reviewer ask?

**ANY LOW confidence finding blocks submission.**

---

## EXECUTION ORDER

```
Phase 1: [1.1] → [1.2] → [1.3] → [1.4]  (sequential — each depends on previous)
Phase 2: [2.1] → [2.2] → [2.3]           (sequential)
Phase 3: [3.1 ∥ 3.2 ∥ 3.3]              (parallel — independent)
Phase 4: [4.1 ∥ 4.2 ∥ 4.3]              (parallel — independent)
Phase 5: [5.1 ∥ 5.2 ∥ 5.3]              (parallel — independent)
Phase 6: [6.1] → [6.2]                    (sequential — fix then propagate)
Phase 7: [7.1] → [7.2] → [7.3] → [7.4]  (sequential — final gate)
```

Phases 1-2 are the critical path (if definitions are wrong,
everything downstream is invalid).
Phases 3-5 can run in parallel after Phase 2 passes.
Phase 6 incorporates all findings.
Phase 7 is the final gate.

---

## SUBMISSION-BLOCKING CRITERIA

The paper CANNOT be submitted if ANY of these fail:

- [ ] `explanation_impossibility` has non-zero behavioral axiom deps
- [ ] Any `sorry` exists in any Lean file
- [ ] The trilemma is degenerate (any "without_X" proof succeeds)
- [ ] Any negative control shows flip rate > 10%
- [ ] Any number in the paper doesn't match its source
- [ ] Any novelty claim is contradicted by found prior art
- [ ] The PNAS version exceeds format limits
- [ ] `lake build` fails on clean build
- [ ] The Markov equivalence CI computation is incorrect
- [ ] The paper overclaims beyond what the proofs support
