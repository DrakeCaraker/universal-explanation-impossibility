# Verification Audit

**Project:** The Attribution Impossibility — Lean 4 Formalization
**Date:** April 1, 2026
**Status:** Pre-submission audit record

This document records what has been verified (by the LLM during development), what remains to be verified by humans, the complete ranked checklist of verification items, and the risk assessment. It is the authoritative record of the project's epistemic state.

---

## 1. Executive Summary

### What the LLM verified

The following were verified by `#print axioms` in Lean 4 and by inspection of the Lean source:

- **`attribution_impossibility` uses zero domain axioms.** The axiom trace contains only `Model`, `attribution`, `propext`, `Classical.choice`, and `Quot.sound` — the first two are type declarations, and the last three are the standard logical foundations of Lean 4. No GBDT-specific, measure-theoretic, or attribution-specific axioms appear.
- **`attribution_impossibility_weak` uses zero domain axioms.** Same axiom trace as above.
- **`strongly_faithful_impossible` uses zero domain axioms.** Same axiom trace.
- **`attribution_sum_symmetric` is derived, not axiomatized.** It is a theorem in `SymmetryDerive.lean` and depends on `proportionality_global`, `splitCount_firstMover`, `splitCount_nonFirstMover`, `splitCount_crossGroup_symmetric`, and `numTrees_pos` — exactly the expected domain axioms, no more.
- **`design_space_theorem` uses the expected axioms.** It depends on the above theorems and the domain axioms for variance and Spearman bounds.
- **Declaration count: 49.** Verified by `grep -c "^theorem\|^lemma" DASHImpossibility/*.lean`.
- **Axiom count: 15.** Verified by `grep -c "^axiom" DASHImpossibility/*.lean`.
- **Sorry count: 0.** Verified by `grep -r "sorry" DASHImpossibility/*.lean` returning no matches.
- **Cross-references: no undefined references.** The project builds successfully with `lake build` (~2500 jobs).

### What humans must verify

The LLM can certify that the deductions are correct given the axioms. It cannot certify that the axioms accurately describe the real world. The critical human verification items are:

1. Do the 7 domain axioms accurately describe gradient boosting under Gaussian data?
2. Does the informal theorem statement in the paper match the formal Lean statement?
3. Do the experimental results support the empirical claims?
4. Is the related work coverage complete?
5. Is the proof status labeling (proved / derived / argued / empirical) accurate throughout?

### Key risks

| Risk | Probability | Impact | Priority |
|------|-------------|--------|----------|
| An axiom is empirically implausible | Medium | High | Critical |
| The informal statement overstates the formal result | Low | High | Critical |
| A concurrent paper proves a similar result | Low | High | High |
| Experimental CV claims are cherry-picked | Low | Medium | Medium |
| NeurIPS format noncompliance | Low | Low | Low |

**The #1 risk** is discussed in detail in Section 5.

---

## 2. Self-Verification Results

### 2.1 Axiom dependency traces (from `#print axioms`)

```
attribution_impossibility
  axioms: Model, attribution, propext, Classical.choice, Quot.sound

attribution_impossibility_weak
  axioms: Model, attribution, propext, Classical.choice, Quot.sound

strongly_faithful_impossible
  axioms: Model, attribution, propext, Classical.choice, Quot.sound
```

**Interpretation:** `Model` and `attribution` are type declarations (they introduce types and functions without asserting properties). `propext`, `Classical.choice`, and `Quot.sound` are the standard axioms of Lean 4's foundation — they are present in every Lean 4 proof. The core impossibility theorems therefore have **zero domain-specific axiom dependencies**.

```
attribution_sum_symmetric
  axioms: Model, attribution, splitCount, firstMover,
          numTrees, numTrees_pos,
          proportionality_global,
          splitCount_firstMover, splitCount_nonFirstMover,
          splitCount_crossGroup_symmetric,
          propext, Classical.choice, Quot.sound, Quot.lift
```

**Interpretation:** This theorem uses exactly the expected domain axioms. It was formerly axiomatized and is now derived — the derivation is in `DASHImpossibility/SymmetryDerive.lean`.

```
design_space_theorem
  axioms: (all of the above) + modelMeasurableSpace, modelMeasure,
          consensus_variance_bound, spearman_classical_bound
```

**Interpretation:** The design space theorem uses all domain axioms, as expected. The measure-infrastructure axioms (`modelMeasurableSpace`, `modelMeasure`) are standard mathematical structure. `consensus_variance_bound` encodes the 1/M variance result. `spearman_classical_bound` encodes the Spearman stability bound.

### 2.2 Count verification

| Item | Expected | Verified |
|------|----------|----------|
| Total declarations | 49 | 49 |
| Theorems + lemmas | 49 | 49 |
| Axioms | 15 | 15 |
| `sorry` occurrences | 0 | 0 |
| Files | 15 | 15 |

### 2.3 Build verification

`lake build` completes successfully (~2500 jobs). No type errors, no universe inconsistencies, no unresolved `sorry`. All imports resolve.

### 2.4 Cross-reference verification

All `#check` references within the codebase resolve. The import graph is:
```
Basic.lean
  └── DesignSpace.lean
        ├── Impossibility.lean
        │     ├── Ratio.lean → SplitGap.lean → Defs.lean
        │     ├── General.lean → Trilemma.lean → Defs.lean
        │     └── SpearmanDef.lean → Defs.lean
        └── Corollary.lean
              ├── Impossibility.lean (above)
              └── SymmetryDerive.lean → SplitGap.lean → Defs.lean
```

No circular dependencies. No undefined references.

---

## 3. Complete Human Verification Checklist

Items are ranked by **Priority** (1 = most critical). Each item includes: what to check, who should do it, how long it takes, what a red flag looks like, and what the consequence is if it is wrong.

---

### CRITICAL (do before any submission)

**[V1] Domain axiom empirical plausibility**
- **Priority:** 1
- **What to check:** Read the 7 domain axioms in `DASHImpossibility/Defs.lean` (lines 82–215). For each, verify that the mathematical property is a plausible idealization of the described behavior. Pay special attention to: (a) `proportionality_global` — this asserts a uniform proportionality constant `c` across *all* models, not just per-model. This is stronger than per-model proportionality and justified by the uniform-contribution model of Lundberg & Lee (2017). (b) `splitCount_firstMover` and `splitCount_nonFirstMover` — these encode the Gaussian conditioning argument. The exact values T/(2-ρ²) and (1-ρ²)T/(2-ρ²) should follow from the argument in Lemma 1 of the supplement.
- **Who:** Bryan Arnold
- **Time:** 2 hours
- **Red flag:** Any axiom that is false for standard XGBoost with default hyperparameters, or that requires unrealistic assumptions not disclosed in the paper.
- **Consequence if wrong:** The GBDT instantiation is invalid. The core impossibility (Level 0) still holds, but the quantitative claims (ratio = 1/(1-ρ²), specific Spearman bounds) require revised axioms.

**[V2] Informal-formal theorem match**
- **Priority:** 2
- **What to check:** Read the informal statement of Theorem 1 in the paper. Then read the formal Lean statement of `attribution_impossibility` in `DASHImpossibility/Trilemma.lean` (lines 63–74). Do they match? Specifically: (a) Is the faithfulness condition in the paper a biconditional or an implication? (The Lean proof provides both versions: `attribution_impossibility` uses a biconditional, `attribution_impossibility_weak` uses implication + antisymmetry.) (b) Does the paper correctly describe what "stable" means in the formal proof? (In the Lean proof, stability is captured by having a *single fixed ranking* — there is no explicit quantification over runs.) (c) Does the paper correctly describe the Rashomon property?
- **Who:** Bryan Arnold or David Rhoads
- **Time:** 1 hour
- **Red flag:** The informal statement claims something the formal proof does not establish, or the formal proof has a stronger precondition than described.
- **Consequence if wrong:** The paper's headline claim is overstated. Requires revising either the paper text or the theorem statement.

**[V3] Proof status labeling**
- **Priority:** 3
- **What to check:** Read the "Proof status transparency" paragraph in the paper. Then audit 10 specific claims in the body of the paper: (a) Are all claims labeled "proved" traceable to Lean theorems without domain-axiom dependencies? (b) Are all claims labeled "derived" traceable to Lean theorems that use domain axioms? (c) Are all claims labeled "argued" in the supplement with a written proof? (d) Are all claims labeled "empirical" backed by experiment results in `paper/results_*.json`?
- **Who:** David Rhoads
- **Time:** 1.5 hours
- **Red flag:** Any claim in the paper body labeled "proved" or "Lean-verified" that actually relies on domain axioms or has only an argued proof.
- **Consequence if wrong:** The paper's epistemic integrity claim is false, which is a serious problem for a paper that makes formal verification a selling point.

**[V4] Core proof correctness (informal)**
- **Priority:** 4
- **What to check:** Read the informal proof of Theorem 1 in the paper (or supplement). Does the argument make sense? The core idea is: by the Rashomon property, there exist models f and f' that rank j above k and k above j, respectively. A faithful-and-stable ranking would have to rank j above k (from f) and k above j (from f') simultaneously — a contradiction. Verify that this argument is correctly stated and that the paper does not use any additional assumptions not present in the Lean proof.
- **Who:** Bryan Arnold
- **Time:** 45 minutes
- **Red flag:** The informal argument uses an assumption not in the Lean proof, or the argument is logically invalid.
- **Consequence if wrong:** The paper contains an incorrect proof. This is the worst possible outcome.

---

### HIGH (do before final submission)

**[V5] Experiment reproducibility**
- **Priority:** 5
- **What to check:** Run at least 3 experiment scripts from `paper/scripts/`. Compare results to the corresponding `results_*.json` or `results_*.txt` files. Suggested scripts: `results_validation.json` (the main validation), `results_f1_f5.json` (the F1/F5 figure), and `results_cross_implementation.json` (cross-implementation stability). If results differ by more than ±5% from reported values, flag it.
- **Who:** David Rhoads
- **Time:** 3 hours
- **Red flag:** Any result that differs materially (>10%) from what is reported in the paper, or a script that fails to run.
- **Consequence if wrong:** Experimental claims in the paper are not reproducible. Requires rerunning and potentially revising claims.

**[V6] Axiom count and declaration count consistency**
- **Priority:** 6
- **What to check:** Run `grep -c "^axiom" DASHImpossibility/*.lean` (should total 15) and `grep -c "^theorem\|^lemma" DASHImpossibility/*.lean` (should total 49). Compare to what the paper states. Also check `grep -r "sorry" DASHImpossibility/*.lean` returns no output.
- **Who:** Anyone
- **Time:** 10 minutes
- **Red flag:** Any discrepancy between code counts and paper claims.
- **Consequence if wrong:** The paper's count claims are wrong. Easy to fix, but embarrassing.

**[V7] Related work completeness**
- **Priority:** 7
- **What to check:** Read the related work section. Verify that the following are cited and correctly described: Lundberg & Lee (2017) SHAP; Arrow (1951) impossibility; Bilodeau et al. (2024) limits of post-hoc explanations; Laberge et al. (2023) perturbation stability; Rudin (2024) interpretable models; Huang et al. (2024) SHAP under collinearity. Search for any 2024–2026 papers on feature attribution stability or SHAP impossibility that might be concurrent work.
- **Who:** Bryan Arnold
- **Time:** 1 hour
- **Red flag:** A paper that proves a closely related theorem that is either not cited or incorrectly characterized.
- **Consequence if wrong:** Reviewers will flag missing citations; in the worst case, there is a concurrent result that reduces novelty.

**[V8] DASH experiment validity**
- **Priority:** 8
- **What to check:** Verify that the DASH ensemble experiment is implemented correctly. Specifically: (a) Is the ensemble constructed with i.i.d. random seeds? (b) Is the ensemble size M varied, and is the Spearman correlation measured against a ground truth? (c) Is the comparison between single-model and DASH attributions fair (same datasets, same features)?
- **Who:** David Rhoads
- **Time:** 1 hour
- **Red flag:** The DASH ensemble uses non-independent models, or the comparison is unfair.
- **Consequence if wrong:** The empirical resolution claim is overstated.

**[V9] Corollary 1 and Design Space Theorem correctness**
- **Priority:** 9
- **What to check:** Read `DASHImpossibility/Corollary.lean` and `DASHImpossibility/DesignSpace.lean`. Verify that `consensus_equity` and `design_space_theorem` are stated and proved correctly. Pay attention to the `IsBalanced` condition — the equity result holds exactly for balanced ensembles, and approximately (in expectation) for i.i.d. seeded ensembles. Make sure the paper correctly states which version holds.
- **Who:** Bryan Arnold
- **Time:** 45 minutes
- **Red flag:** The paper claims DASH achieves exact equity without the balanced condition.
- **Consequence if wrong:** The resolution claim is overstated. Requires a caveat in the paper.

**[V10] Extension claims validity**
- **Priority:** 10
- **What to check:** Read Section 6 (extensions: fairness, causal, LLM). For each extension: (a) Is the impossibility claim correctly derived from the core theorem? (b) Are the limitations of the extension clearly stated? (c) Are the empirical extension results (LLM attention, fairness audit) correctly described?
- **Who:** Bryan Arnold
- **Time:** 1 hour
- **Red flag:** An extension claims the core theorem applies in a setting where the Rashomon property has not been verified.
- **Consequence if wrong:** Extension claims are unsupported. Can be fixed by weakening language to "suggests" or "motivates".

---

### MEDIUM (important but not blocking)

**[V11] Spearman definition correctness**
- **Priority:** 11
- **What to check:** Read `DASHImpossibility/SpearmanDef.lean`. The Spearman correlation is defined from midranks using the Σd² formula. Verify that this definition matches the standard Spearman formula for the case of ties. The midrank definition (countBelow + (countEqual + 1)/2) is standard, but verify the Σd² formula is correct.
- **Who:** Bryan Arnold
- **Time:** 30 minutes
- **Red flag:** The Lean definition of Spearman differs from the standard definition in a material way.

**[V12] Split-count algebra**
- **Priority:** 12
- **What to check:** Read `DASHImpossibility/SplitGap.lean`. The algebraic results `split_gap_exact` and `split_gap_ge_half` are pure algebra. Spot-check the computation: if first-mover split count = T/(2-ρ²) and non-first-mover split count = (1-ρ²)T/(2-ρ²), then the gap is T/(2-ρ²) - (1-ρ²)T/(2-ρ²) = ρ²T/(2-ρ²). Verify this is what the Lean proof establishes.
- **Who:** Anyone
- **Time:** 20 minutes
- **Red flag:** Algebraic error in the gap formula.

**[V13] Ratio divergence**
- **Priority:** 13
- **What to check:** Read `DASHImpossibility/Ratio.lean`. The theorem `ratio_tendsto_atTop` asserts that 1/(1-ρ²) → +∞ as ρ → 1⁻. This is a standard calculus result. Verify the proof strategy makes sense: it factors 1-ρ² = (1-ρ)(1+ρ) and uses the known limit of 1/(1-ρ) → ∞.
- **Who:** Anyone
- **Time:** 15 minutes
- **Red flag:** The limit proof has an error in the nhds filter argument.

**[V14] SymPy algebra verification**
- **Priority:** 14
- **What to check:** The SymPy script `verify_lemma6_algebra.py` in `dash-shap/paper/proofs/` verifies the algebraic consequences of the split-count axioms. Run this script and verify it outputs "VERIFIED" for all checks.
- **Who:** David Rhoads
- **Time:** 15 minutes
- **Red flag:** Any SymPy check fails.

**[V15] Figure accuracy**
- **Priority:** 15
- **What to check:** For each figure in `paper/figures/`, verify that it accurately represents the experimental data. Check figure labels, axis ranges, and captions. Specifically: the ratio figure (should show 1/(1-ρ²) curve), the instability figure (should show CV vs ρ), and the DASH resolution figure (should show variance decreasing with M).
- **Who:** David Rhoads
- **Time:** 45 minutes
- **Red flag:** A figure caption claims something different from what the figure shows.

**[V16] NeuralNet impossibility**
- **Priority:** 16
- **What to check:** Read `DASHImpossibility/NeuralNet.lean`. The `nn_impossibility` theorem is conditional on "captured feature" — a feature that has higher attribution than other same-group features. Verify that this condition is correctly described in the paper (it should say the impossibility is conditional, not unconditional).
- **Who:** Bryan Arnold
- **Time:** 20 minutes
- **Red flag:** The paper claims neural net impossibility without stating the conditioning.

**[V17] Lasso impossibility**
- **Priority:** 17
- **What to check:** Read `DASHImpossibility/Lasso.lean`. The `lasso_impossibility` theorem takes the Lasso properties as hypotheses (not global axioms). Verify that the paper correctly describes this as an instantiation of the general framework, not as a separate theorem with its own axioms.
- **Who:** Anyone
- **Time:** 15 minutes
- **Red flag:** The paper counts Lasso properties as part of the global axiom count.

**[V18] Iterative optimizer framework**
- **Priority:** 18
- **What to check:** Read `DASHImpossibility/Iterative.lean`. The `IterativeOptimizer` structure generalizes both GBDT (via firstMover) and Lasso (via selected feature). Verify that the paper describes this generalization correctly and that the theorem `iterative_impossibility` is cited appropriately.
- **Who:** Bryan Arnold
- **Time:** 20 minutes
- **Red flag:** The paper describes the framework as GBDT-specific when it is general.

---

### LOW (polish, not blocking)

**[V19] Abstract accuracy**
- **Priority:** 19
- **What to check:** Does the abstract accurately describe all main results? Does it correctly characterize the Lean formalization (zero domain axioms for the core theorem)?
- **Who:** All co-authors
- **Time:** 15 minutes

**[V20] Introduction claims**
- **Priority:** 20
- **What to check:** Does the introduction claim anything that is not supported by the results? Specifically: "provably optimal" for DASH — is this the right characterization? DASH achieves zero unfaithfulness in the design space, but "optimal" should be caveated.
- **Who:** Bryan Arnold
- **Time:** 20 minutes

**[V21] Supplement proof completeness**
- **Priority:** 21
- **What to check:** Are all claims labeled "argued" in the main text covered by a complete proof in the supplement? Are the supplement proofs self-contained?
- **Who:** Drake Caraker
- **Time:** 1 hour

**[V22] References formatting**
- **Priority:** 22
- **What to check:** Are all 23 citations correctly formatted in NeurIPS style? Are DOIs or arXiv links provided where available?
- **Who:** Anyone
- **Time:** 20 minutes

**[V23] Code comments accuracy**
- **Priority:** 23
- **What to check:** Do the comments in the Lean files accurately describe what the code does? Are the axiom justification comments accurate?
- **Who:** Drake Caraker
- **Time:** 30 minutes

**[V24] Datasets and reproducibility**
- **Priority:** 24
- **What to check:** Are the 11 datasets used in experiments publicly available? Are download instructions provided or can the reader replicate the data collection?
- **Who:** David Rhoads
- **Time:** 20 minutes

**[V25] Page count**
- **Priority:** 25
- **What to check:** Run `pdfinfo paper/main.pdf` and `pdfinfo paper/supplement.pdf`. Verify main text is ≤ 9 pages plus references plus checklist.
- **Who:** Drake Caraker
- **Time:** 5 minutes

**[V26] Author information completeness**
- **Priority:** 26
- **What to check:** Are all author names, affiliations, and email addresses complete and accurate?
- **Who:** All co-authors
- **Time:** 5 minutes

**[V27] License**
- **Priority:** 27
- **What to check:** Is the repository license appropriate? Is the paper licensed correctly for open access?
- **Who:** Drake Caraker
- **Time:** 5 minutes

**[V28] arXiv metadata**
- **Priority:** 28
- **What to check:** Before posting to arXiv: verify title, abstract, author list, and subject category (cs.LG + cs.AI, with cs.LO for logic).
- **Who:** Drake Caraker
- **Time:** 10 minutes

**[V29] NeurIPS style file**
- **Priority:** 29
- **What to check:** Replace the placeholder `neurips_2026.sty` with the official NeurIPS 2026 style file when it becomes available.
- **Who:** Drake Caraker
- **Time:** 10 minutes

**[V30] Competing interests**
- **Priority:** 30
- **What to check:** NeurIPS requires a competing interests statement. All co-authors should confirm they have no competing interests.
- **Who:** All co-authors
- **Time:** 5 minutes

**[V31] Lean toolchain version**
- **Priority:** 31
- **What to check:** Run `cat lean-toolchain`. Verify the Lean version is a recent stable release and that the project builds cleanly with this version.
- **Who:** Drake Caraker
- **Time:** 5 minutes

**[V32] Mathlib version**
- **Priority:** 32
- **What to check:** Check `lake-manifest.json` for the Mathlib commit hash. Verify that the Mathlib version used has no known issues that affect our imports (`ProbabilityTheory.variance`, `Finset.*`).
- **Who:** Drake Caraker
- **Time:** 10 minutes

---

## 4. Risk Assessment

| Risk | Probability | Impact | P×I | Mitigation |
|------|-------------|--------|-----|------------|
| Domain axiom is empirically false | 0.15 | 5 | 0.75 | Human review of each axiom (V1) |
| Informal statement overstates formal result | 0.10 | 5 | 0.50 | Informal-formal match check (V2) |
| Concurrent paper with similar result | 0.10 | 4 | 0.40 | Related work search (V7) |
| Experiment results not reproducible | 0.10 | 4 | 0.40 | Reproducibility check (V5) |
| Proof status mislabeling | 0.20 | 3 | 0.60 | Proof status audit (V3) |
| Extension claims unsupported | 0.15 | 3 | 0.45 | Extension validity check (V10) |
| Page limit exceeded at NeurIPS | 0.20 | 2 | 0.40 | Page count check (V25) |
| Missing citation found in review | 0.30 | 2 | 0.60 | Related work search (V7) |
| SymPy verification fails | 0.05 | 4 | 0.20 | Run verify script (V14) |
| Build failure on reviewer machine | 0.10 | 2 | 0.20 | Document dependencies in README |

---

## 5. The #1 Risk: Axioms Certify Deduction, Not Premises

**This is the most important limitation of the Lean formalization, and it must be clearly communicated in the paper.**

The `#print axioms` verification shows that `attribution_impossibility` depends on zero domain-specific axioms. This is a precise, verifiable, and genuine result. What it means is:

*If* the Rashomon property holds for a given model class, *then* no faithful-stable-complete ranking exists. The logic is airtight.

What it does not mean is:

- That the Rashomon property holds for GBDT specifically. (This is argued in the supplement from the Gaussian conditioning model, and confirmed empirically.)
- That the domain axioms for the GBDT instantiation (`splitCount_firstMover`, `proportionality_global`, etc.) accurately describe real XGBoost behavior. These are mathematical idealizations.
- That the empirical CV of 0.35–0.66 is not dataset-specific. It is measured on 11 datasets; other datasets may show different values.

**The honest statement:** The Lean proof verifies the deductive structure. The empirical experiments verify that the instability is large and consistent across implementations. The domain axioms connect the two, and their plausibility is the primary human verification target.

**CV = 0.35–0.66:** This is high enough to be practically significant. A CV > 0.3 is generally considered high variability. The range reflects variation across datasets, not within a dataset. The lowest CV (0.35) still corresponds to substantial instability.

---

## 6. Non-Negotiables Before Submission

These 5 items must be completed before any submission. No exceptions.

1. **[V1] Axiom empirical plausibility** — at least one domain expert must have read and approved the 7 domain axioms.
2. **[V2] Informal-formal match** — someone other than Drake must have verified that Theorem 1 in the paper matches `attribution_impossibility` in Lean.
3. **[V3] Proof status labeling** — every "proved" claim in the paper must trace to a Lean theorem without `sorry`.
4. **[V6] Count consistency** — the axiom count, theorem count, and sorry count in the paper must match the code.
5. **[V10] NeurIPS format compliance** — official style file, correct page count, complete author information.
