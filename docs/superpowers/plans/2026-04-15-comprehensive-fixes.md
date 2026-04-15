# Comprehensive Fix Plan: All Open Issues

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address every open issue from peer review (12 reviewers), vet analysis, and CORRECTIONS.md before Nature/arXiv submission.

**Architecture:** Four phases by type. Phase 1 (factual fixes, 30 min) corrects errors. Phase 2 (empirical expansion, 2 hours) closes data gaps. Phase 3 (paper text, 1.5 hours) addresses framing. Phase 4 (acknowledge, 15 min) documents limitations.

---

## Status of Existing Corrections (C1-C35, L1-L8)

| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| C1 | Decisive definition mismatch | **NEEDS VERIFY** | Informal text fixed; Methods section needs check |
| C2 | Stale Lean counts | **FIXED** | 104/463 everywhere |
| C3 | Gauge lattice 200× error | **FIXED** | Spatial averaging applied |
| C4 | Kruskal-Wallis p-value | **FIXED** | Both values reported |
| C5 | Flyspeck comparison | **FIXED** | Removed from all files |
| C6 | Attention retraining | **FIXED** | 19.9% from 20 retrained models |
| C7 | Noether moderate ρ | **FIXED** | Sensitivity: 50pp across ρ=0.5-0.99 |
| C8 | Pre-specify η criterion | **FIXED** | PRE_REGISTRATION.md documents criterion |
| C9 | Mann-Whitney non-independence | **NEEDS FIX** | Permutation test needed |
| C10 | Power analyses | **DEFER** | Would require significant compute |
| C11 | Restructure paper | **PARTIAL** | Nature article restructured; monograph not |
| C12 | "Trivially true" defense | **FIXED** | 7 mentions in monograph |
| C13 | Group assignment justification | **FIXED** | Monograph §8.7 has per-instance justification |
| C14 | "Resolutions are backwards" | **PARTIAL** | Nature uses "independently developed" framing |
| C15 | Pareto-optimality in Lean | **DONE** | ParetoOptimality.lean exists |
| C16 | Qualify "Noether" terminology | **PARTIAL** | Monograph uses "invariance counting" |
| C17 | Rashomon set literature | **DONE** | Fisher VIC, Laberge, Marx cited |
| C18 | Cite Selbst & Barocas | **NEEDS CHECK** |  |
| C19 | EU AI Act specific articles | **NEEDS CHECK** |  |
| C20-C35 | Various | Mixed | See below |
| L1-L8 | Limitations | **ACKNOWLEDGE** | Document in Discussion |

---

## Phase 1: Factual Fixes (30 min)

### Task 1: Fix SI FDR discrepancy (62% → 57%)

**Source:** Vet Round 1 — SI line 74 claims 62% but validated number is 57.08%

**Files:**
- Modify: `paper/supplementary_information.tex:74`

- [ ] **Step 1: Update SI text**

Change "reduces false discovery rate from 62%" to "reduces false discovery rate from 57%"

- [ ] **Step 2: Verify no other 62% references**

```bash
grep -rn "62%" paper/*.tex | grep -i "fdr\|false.*discovery\|unreliable"
```

- [ ] **Step 3: Commit**

### Task 2: Fix Noether gap in RESULTS_SYNTHESIS.md (47pp → 50pp)

**Source:** Vet Round 1 — Sensitivity analysis shows 50pp, not 47pp

**Files:**
- Modify: `knockout-experiments/RESULTS_SYNTHESIS.md`

- [ ] **Step 1: Update all 47pp references to "~50pp (47pp initial, 50pp across ρ sensitivity)"**

- [ ] **Step 2: Commit**

### Task 3: Verify C1 decisive definition in Nature Methods

**Source:** CORRECTIONS.md C1 + Vet Round 3

**Files:**
- Read: `paper/nature_article.tex` Methods section

- [ ] **Step 1: Find the formal definition of decisive in Methods**
- [ ] **Step 2: Verify it matches Lean (pointwise: ∀ θ h, incomp(explain(θ),h) → incomp(E(θ),h))**
- [ ] **Step 3: If pairwise, fix to pointwise. If pointwise, mark C1 as RESOLVED**
- [ ] **Step 4: Commit**

### Task 4: Add Bilodeau et al. comparison

**Source:** Vet Round 3 — missing from related work

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex` Related Work section

- [ ] **Step 1: Add explicit comparison paragraph**

"Bilodeau et al. (2024) prove an impossibility for additive feature attributions under collinearity. Our framework generalizes this to arbitrary explanation types: the `AttributionInstance` in our Lean formalization recovers their result as a special case, while the abstract `ExplanationSystem` extends it to 8 additional ML methods and 8 scientific domains."

- [ ] **Step 2: Verify Bilodeau is in references.bib**
- [ ] **Step 3: Commit**

---

## Phase 2: Empirical Expansion (2 hours)

### Task 5: Expand landscape bridge to 15+ datasets

**Source:** Reviewers A1, D2, F2 — n=7 is insufficient for ρ=0.929 claim

**Files:**
- Modify: `knockout-experiments/explanation_landscape_bridge.py`

- [ ] **Step 1: Add 8+ more datasets**

Add: CalHousing (regression→binarize), Adult (Census), Ionosphere, Sonar, Parkinsons, Vehicle, Segment, Satimage. Use fetch_openml or sklearn built-ins. Target: 15+ total.

- [ ] **Step 2: Run expanded experiment**

- [ ] **Step 3: Report ρ and p-value with n≥15**

Pass criterion: ρ > 0.7 with p < 0.01. If fails, report honestly.

- [ ] **Step 4: LOO sensitivity — drop each dataset and recompute ρ**

Report min/max ρ across leave-one-out. If any single dataset drives the correlation, flag it.

- [ ] **Step 5: Commit**

### Task 6: Bridge sensitivity to SNR threshold

**Source:** Reviewer F2

**Files:**
- Modify: `knockout-experiments/explanation_landscape_bridge.py`

- [ ] **Step 1: Compute coverage conflict degree at thresholds 0.3, 0.5, 1.0, 2.0**
- [ ] **Step 2: Report ρ at each threshold**
- [ ] **Step 3: If ρ is robust (>0.7) across thresholds, result is genuine**
- [ ] **Step 4: Commit**

### Task 7: Statistical tests for enrichment

**Source:** Reviewer D1

**Files:**
- Create: `knockout-experiments/enrichment_statistical_tests.py`

- [ ] **Step 1: For each dataset in the enrichment experiment, compute:**
  - Bootstrap 95% CI on flip rate reduction (fine → coarse)
  - Permutation test p-value (null: enrichment has no effect)

- [ ] **Step 2: For the balance-matched control:**
  - One-sample t-test: semantic flip rate vs mean of random flip rates
  - Report p-value and effect size (Cohen's d)

- [ ] **Step 3: Commit**

### Task 8: Permutation test for Noether counting (C9)

**Source:** CORRECTIONS.md C9 — Mann-Whitney non-independence

**Files:**
- Create: `knockout-experiments/noether_permutation_test.py`

- [ ] **Step 1: Implement permutation test**

Permute the within/between group labels of features 10,000 times. Compute separation gap for each permutation. Report permutation p-value.

- [ ] **Step 2: Compare to existing Mann-Whitney p-value (2.7e-13)**
- [ ] **Step 3: Commit**

---

## Phase 3: Paper Text Fixes (1.5 hours)

### Task 9: "What does formalization buy?" argument (A1)

**Files:**
- Modify: `paper/nature_article.tex` or `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: Add paragraph in Introduction or Discussion**

"The Lean formalization serves three purposes beyond the informal proof: (1) it eliminates the definitional mismatch risk — the precise definitions of faithful, stable, and decisive are mechanically checked to be the ones that make the proof work and the tightness witnesses valid; (2) it verifies that each of the 23 domain instances actually satisfies the ExplanationSystem axioms, catching structural errors that informal checking misses (e.g., incompatible must be irreflexive); (3) the constructive Rashomon witnesses in 8 scientific domains are computed, not assumed — the Lean kernel confirms the computation."

- [ ] **Step 2: Commit**

### Task 10: Enrichment structural claim — be explicit about limits (A1)

- [ ] **Step 1: In enrichment discussion, add:**

"Enrichment reliably reduces instability (4/6 datasets, 6.7–18.8pp OOS). Balance-matched random controls confirm structural advantage on Fashion-MNIST (+9.6pp, 10/10 random controls beaten) but not Covertype (-0.6pp after balance matching). The structural claim — that semantically motivated merging outperforms arbitrary merging — has support on one robustly controlled dataset. The qualitative claim — that enrichment reduces instability — is confirmed broadly."

- [ ] **Step 2: Commit**

### Task 11: Highlight generalized bilemma (A2)

- [ ] **Step 1: In Nature article bilemma section, add:**

"The bilemma generalizes: F+S is impossible whenever any Rashomon pair has empty compatible-intersection (Theorem X), a condition strictly weaker than maximal incompatibility. The complete characterization: bilemma ↔ no fiber-compatible element (for single-fiber systems)."

- [ ] **Step 2: Commit**

### Task 12: η law group ID as fundamental limitation (A2, D2)

- [ ] **Step 1: In Discussion, add paragraph:**

"The η law's practical reach is limited by group identification. For 7/16 instances with well-characterized groups (exact symmetry by construction), R²=0.957. For the remaining 9 with approximate groups, R²=0.25. The group identification problem — determining the correct symmetry group from data — is fundamental, not merely technical. SAGE sidesteps it heuristically, but a formal algorithm for group discovery from model ensembles remains open."

- [ ] **Step 2: Commit**

### Task 13: Coverage conflict as conceptual naming (B2, F1)

- [ ] **Step 1: In the coverage conflict discussion, add:**

"The biconditional `coverage_conflict ↔ ¬neutral_element` is a definitional equivalence: both express that no element of H is compatible with all explain-values. The value of naming this 'coverage conflict' is conceptual, not mathematical — it identifies the algebraic feature that powers the bilemma and clarifies why binary H strengthens the impossibility (maximum coverage) while larger H weakens it (room for neutral elements)."

- [ ] **Step 2: Commit**

### Task 14: Landscape gap documentation (B1, F1)

- [ ] **Step 1: In ExplanationLandscape.lean, add doc-comment:**

"Note: coverage conflict proves no neutral element exists (landscape_bottom_no_fs), but does NOT directly prove the bilemma. The bilemma additionally requires a specific Rashomon pair with empty compatible-intersection (generalized_bilemma). Coverage conflict is a global property; the bilemma condition is per-pair. Global coverage conflict implies per-pair coverage conflict at every Rashomon pair if maximal incompatibility holds (maxIncompat_landscape_chain), but not in general."

- [ ] **Step 2: Commit**

### Task 15: SNR = coverage conflict is modeling assumption (A2)

- [ ] **Step 1: In monograph bridge discussion, add:**

"The connection between coverage conflict (a discrete algebraic property) and SNR (a continuous statistical quantity) rests on the modeling assumption that importance differences are approximately Gaussian. Under this assumption, SNR < τ operationalizes 'no compatible direction exists for this pair.' The assumption is validated by Shapiro-Wilk tests (60-100% of pairs pass) but is not a theorem."

- [ ] **Step 2: Commit**

### Task 16: Tone down abstract framework generality (B2)

- [ ] **Step 1: In the abstract impossibility discussion, change:**

"a general theory of impossibility" → "an abstract framework that captures both Arrow's theorem and the bilemma as instances, suggesting a broader pattern that further investigation may confirm"

- [ ] **Step 2: Commit**

### Task 17: Frame the "discovery" clearly for Nature (C3)

- [ ] **Step 1: Ensure the Nature article's first paragraph states:**

"We prove that explanation instability in machine learning is not a deficiency of current methods but a mathematical necessity: no explanation method, present or future, can be simultaneously faithful, stable, and decisive when the underlying system is underspecified. This impossibility holds with zero axiom dependencies and applies uniformly across nine explanation types and eight scientific domains."

- [ ] **Step 2: Commit**

### Task 18: Scientific revolutions as conjecture (C1)

- [ ] **Step 1: In the scientific revolutions section, add:**

"We state this as a conjecture, not a theorem. The pattern is supported by five formalized instances but counterexamples exist (Bohmian mechanics resolves the measurement problem without enrichment). The conjecture should be restricted to transitions that preserve the configuration space."

- [ ] **Step 2: Commit**

### Task 19: Physics instances as structural analogies (C1)

- [ ] **Step 1: In the cross-domain section, add:**

"These instances are structural analogies: the Rashomon property maps onto gauge freedom, microstate degeneracy, codon degeneracy, etc. The formalization verifies the structural correspondence but does not produce new domain-specific results. The value is unification, not discovery within any single domain."

- [ ] **Step 2: Commit**

### Task 20: Document Gaussian assumption failures (D1)

- [ ] **Step 1: In the Gaussian flip discussion, add:**

"The formula assumes Gaussian-distributed importance differences. This holds for tree-based methods (60-100% of pairs pass Shapiro-Wilk) but may fail for coefficient-based methods: Ridge regression on California Housing gives R²=-1.96 due to heavy-tailed coefficient distributions. The Gaussianity diagnostic (Shapiro-Wilk per pair) should be run before applying the formula."

- [ ] **Step 2: Commit**

### Task 21: Model selection stochasticity vs Rashomon (D2)

- [ ] **Step 1: In model selection instance, clarify:**

"Bootstrap resampling produces models that differ in both training data and stochastic initialization. The 80% best-model flip rate conflates data-level Rashomon multiplicity (genuinely different optimal models on different subsets) with seed-level stochasticity (the same optimal structure with different initializations). The impossibility theorem applies regardless — both sources of multiplicity generate the Rashomon property — but the practical interpretation differs."

- [ ] **Step 2: Commit**

### Task 22: SAGE LOO-CV R² reporting (A1)

- [ ] **Step 1: Wherever SAGE R²=0.92 is reported, add LOO-CV:**

"SAGE grouping predicts per-pair instability with R²=0.92 (in-sample) and R²=0.69 (leave-one-out cross-validation across 8 datasets)."

- [ ] **Step 2: Commit**

### Task 23: Attention 19.9% vs 60% clarification (general readers)

- [ ] **Step 1: In the attention instance, clarify:**

"Attention argmax flip rate is 19.9% under full retraining (20 independently converged DistilBERT models, the canonical Rashomon setup) and 60% under weight perturbation (dropout masks on a single model, a supplementary proxy). The 19.9% is the primary result."

- [ ] **Step 2: Commit**

---

## Phase 4: Acknowledge as Limitations (15 min)

### Task 24: Update CORRECTIONS.md with status

- [ ] **Step 1: Add status column to all C1-C35, L1-L8**
- [ ] **Step 2: Mark: FIXED, DONE, PARTIAL, NEEDS FIX, DEFER, ACKNOWLEDGE**
- [ ] **Step 3: Add new issues from this session's review (N1-N8):**

```
N1. Bridge validated on only 7 datasets (expanding in Task 5)
N2. Fairness instance not empirically validated
N3. Abstract impossibility framework has only 2 instances
N4. Coverage conflict is definitional equivalence
N5. Landscape gap: coverage conflict ≠ bilemma directly
N6. No SHAP-based enrichment validation
N7. Multiple testing not addressed across experiments
N8. Only one cal/val split (no k-fold)
```

- [ ] **Step 4: Commit**

---

## Execution Order

```
Phase 1 (sequential, 30 min):
  Task 1 (SI FDR fix) — 2 min
  Task 2 (Noether 47→50pp) — 2 min
  Task 3 (C1 verify) — 5 min
  Task 4 (Bilodeau comparison) — 10 min

Phase 2 (parallel where possible, 2 hours):
  Task 5 (bridge expansion) — 45 min, agent
  Task 6 (SNR sensitivity) — included in Task 5
  Task 7 (enrichment stats) — 30 min, agent
  Task 8 (Noether permutation) — 20 min, agent

Phase 3 (sequential, 1.5 hours):
  Tasks 9-23 — text edits, 5-10 min each

Phase 4 (sequential, 15 min):
  Task 24 — corrections tracker update
```

**Total estimated: ~4.5 hours**
