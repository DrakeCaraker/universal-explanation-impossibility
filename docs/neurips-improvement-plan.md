# Phased Implementation Plan: Addressing All Weaknesses, Gaps, and Open Questions

**Generated**: 2026-04-01
**Context**: Based on rigorous 5-reviewer panel assessment of "The Attribution Impossibility"
**Deadline**: NeurIPS 2026 abstract May 4, paper May 6

**Repo snapshot**: 15 Lean files, 15 axioms, 49 declarations, 0 sorry. 13-page main + 65-page supplement. 26 scripts. 23 references.

---

## Classification Key

- **(N)** — Must be in NeurIPS draft
- **(S)** — Supplement only
- **(J)** — Deferred to JMLR full version
- **(X)** — Not worth doing (justification provided)

---

## Issues Tracked (33 total)

### A. Consensus weaknesses (all 5 reviewers flagged)
1. Core impossibility is mathematically trivial — risk of "so what?" at NeurIPS
2. Quantitative bounds rely on strong axioms (proportionality CV 0.35–0.66, equicorrelation, balanced ensembles)
3. Conditional SHAP / causal attribution underaddressed (one paragraph)
4. Symmetric Bayes dichotomy underdeveloped (3 instances, no general theorem)
5. Axiom consistency unverified (no concrete model constructed)

### B. Per-reviewer issues
6. [R1] Design Space exhaustiveness only half-formalized in Lean
7. [R2] Berry-Esseen bounds are decorative — acknowledge or justify
8. [R2] FIM impossibility repackages non-identifiability — differentiate
9. [R2] NTK extension is conjectural — either prove or remove
10. [R3] Lasso/NN treatments are sketches, not full analyses
11. [R3] Didn't deeply engage Janzing et al. (2020) on causal feature relevance
12. [R4] Intersectional fairness not addressed
13. [R4] "Reduction template" for dichotomy stated without proof
14. [R5] Missing enterprise-scale validation (P > 500)
15. [R5] M=25 prohibitive for real-time — needs practical guidance
16. [R5] Gap between theoretical assumptions and real-world messiness

### C. Open questions from vet
17. Can a concrete Model (e.g., Fin 2) satisfying all 15 axioms be constructed?
18. Generalization from ρ > 0 to I(X_j; X_k) > 0 (mutual information)
19. Local vs. global SHAP — does the impossibility hold for instance-level?
20. DASH breakdown point under model contamination

### D. Data gaps
21. Enterprise-scale validation (P > 500 features)
22. Non-tree-model SHAP validation (neural networks)
23. True causal structure experiment (conditional SHAP with known DAG)

### E. Vet omissions
24. Prevalence survey methodology: 32% power at 10% threshold — underreported
25. D'Amour et al. (2022) underspecification: insufficient engagement
26. Scaling to high-dimensional settings: O(P²) pairwise diagnostic cost
27. Reproducibility crisis connection: underemphasized in abstract/intro
28. LLM attention instability: preliminary results need deeper treatment or honest scoping

### F. Venue and positioning
29. NeurIPS "so what?" defense: position as Design Space Theorem + diagnostic toolkit, not just impossibility
30. NeurIPS 13 → 9 page restructure (deferred to week of April 28)
31. JMLR full-length preparation strategy
32. Strongest skeptical criticism: "78 pages to prove what practitioners already knew" — preemptive defense
33. arXiv preprint timing (before May 4 abstract deadline)

---

## Phase 1: Axiom Consistency & Theoretical Foundations

**Priority**: CRITICAL — claims about the axiom system's soundness depend on this.

### 1A. Construct Concrete Model Satisfying All 15 Axioms

**Addresses**: #5, #17
**Classification**: **(S)** — construction in supplement; one-sentence claim in main text
**Dependencies**: None
**Effort**: 3–4 hours

**What**: A Python script (`paper/scripts/axiom_consistency_model.py`) that constructs a fully explicit numerical model satisfying all 15 axioms simultaneously, and a supplement section documenting it.

**How**: Construct for P=4, L=2, m=2, ρ=0.5, T=100:
- **Model type**: Fin 4 (four models: f₀, f₁, f₂, f₃)
- **Features**: {0,1} in group 0, {2,3} in group 1
- **firstMover mapping**: f₀→0, f₁→1, f₂→2, f₃→3

Verify each axiom numerically:

| Axiom | Verification |
|-------|-------------|
| `numTrees_pos` | T=100 > 0 |
| `firstMover_surjective` | Each feature j has a model fⱼ with fm(fⱼ)=j |
| `splitCount_firstMover` | n_{fm}(f) = 100/(2-0.25) = 57.14... |
| `splitCount_nonFirstMover` | n_k(f) = 0.75·100/1.75 = 42.86... |
| `proportionality_global` | Set c=1, φ_j(f) = 1·n_j(f) |
| `splitCount_crossGroup_symmetric` | When fm∈group0, all group1 features get equal n |
| `consensus_variance_bound` | Compute Var(φ_j) across 4 models, verify Var(consensus)=Var(φ_j)/M |
| `spearman_classical_bound` | Compute Spearman for f₀ vs f₁, verify ≤ 1-m³/P³ |
| `modelMeasurableSpace/modelMeasure` | Uniform measure on Fin 4 |

**Script output**: Table of all 15 axioms with computed values and PASS/FAIL.

**Supplement addition** (~0.5 pages): New section "Axiom Consistency: Concrete Model" after the current axiom table.

**Main text addition**: One sentence in the formalization paragraph: "The axiom system is consistent: we construct an explicit model (P=4, L=2, Fin 4 models) satisfying all 15 axioms simultaneously (supplement §X)."

**Also update**: `docs/self-verification-report.md` with consistency result.

**Risk**: The `consensus_variance_bound` requires i.i.d. models. For Fin 4 with deterministic uniform measure, this is a direct computation. If too small, increase to Fin 8.

**Verification**: `python3 paper/scripts/axiom_consistency_model.py` prints all 15 checks with PASS.

### 1B. Mutual Information Generalization Discussion

**Addresses**: #18
**Classification**: **(J)** — deferred to JMLR. Add one sentence to Open Problems.
**Effort**: 0.5 hours

**What**: Add to the "Open problems" paragraph (main.tex ~line 571):

> "Generalizing from linear correlation (ρ > 0) to nonlinear dependence (I(X_j; X_k) > 0) via mutual information remains open; the Rashomon property should hold whenever the joint distribution admits models ranking features in opposite orders, but the quantitative bounds (which rely on Gaussian conditioning) would require a different derivation."

---

## Phase 2: Paper Text — Framing, Positioning, Gap-Filling

**Priority**: HIGH — directly affects NeurIPS reviewers' first impression.

### 2A. Reframe Abstract and Introduction Around Design Space Theorem

**Addresses**: #1, #29, #32
**Classification**: **(N)**
**Dependencies**: None
**Effort**: 2–3 hours

**What**: Restructure the abstract and introduction so the Design Space Theorem is the headline contribution, with the impossibility as its foundation.

**Abstract revision** — lead with the design space characterization:
- Current lead: "Every feature attribution ranking of collinear features is either unstable or honest about uncertainty"
- Revised lead: "We characterize the complete attribution design space under feature collinearity: exactly two families of methods exist, and we prove DASH ensemble averaging is Pareto-optimal."
- Then: impossibility as base case → quantitative bounds → diagnostics → formalization
- End with: "The framework provides both a theoretical guarantee (no method outside these two families exists) and practical tools (a diagnostic workflow and the optimal ensemble size formula)."

**Introduction revision** — add a "Preemptive defense" paragraph after contributions:
> "The core impossibility proof is deliberately simple—a four-line contradiction from the Rashomon property. Like Arrow's original argument, the mathematical contribution lies not in proof complexity but in identifying the right abstraction and characterizing the complete achievable set. The quantitative depth comes from the architecture-specific bounds and the Pareto optimality proof."

**Risk**: Reframing may require cutting other intro text to stay within NeurIPS limits.

### 2B. Expand Conditional SHAP Treatment in Main Text

**Addresses**: #3, #11
**Classification**: **(N)** — 2-3 sentences in Discussion; **(S)** — already have full supplement section
**Dependencies**: Read existing supplement §Conditional (lines 2711-2797)
**Effort**: 1.5 hours

**What**: The supplement already contains a Conditional Attribution Impossibility theorem, escape conditions, quantitative thresholds, and a simulation. The main text has only one sentence. Expand to a proper paragraph in Discussion.

**Main text addition**:
> "The conditional impossibility (supplement) requires two conditions: equal causal effects (β_j = β_k) and symmetric causal position. When causal effects differ, conditional SHAP can resolve the pair by appealing to the causal graph—this is the precise escape condition. Table SX in the supplement gives quantitative thresholds: for ρ = 0.9, a causal effect difference |Δβ| ≥ 0.5 eliminates instability. This connects to Janzing et al. (2020): causal feature relevance quantification can break the symmetry that drives our impossibility, but only when the causal structure provides genuine asymmetry."

**Also add D'Amour engagement** (#25): One sentence to the Rashomon paragraph:
> "The Rashomon effect is an instance of the broader underspecification phenomenon identified by D'Amour et al. (2022): many model configurations achieve near-optimal performance, and the specific configuration chosen depends on arbitrary training randomness."

### 2C. Acknowledge Berry-Esseen Role Honestly

**Addresses**: #7
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Add to supplement Theorem F1 discussion:
> "**Remark (Role of Berry-Esseen).** The Berry-Esseen bounds provide a completeness guarantee: they bound the worst-case error for non-Gaussian attribution distributions. For the GBDT attributions studied here, the Shapiro-Wilk test confirms near-Gaussianity (p > 0.10 for 412/435 pairs), so the Berry-Esseen correction is negligible in practice. The bounds would become relevant for heavy-tailed attribution distributions (e.g., deep networks) or bimodal distributions (Lasso)."

### 2D. Differentiate FIM Impossibility from Standard Non-Identifiability

**Addresses**: #8
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Add paragraph to supplement §F3 after the proof of Theorem F3:
> "**Relationship to classical non-identifiability.** The near-singularity of the Fisher information under collinearity is textbook: Var(β̂_j - β̂_k) ≥ σ²/(n(1-ρ)) diverges as ρ → 1 (Lehmann & Romano, 2005). Our contribution is reframing this as an impossibility theorem about feature rankings rather than a variance bound on parameter estimates. The reframing has two consequences: (i) it establishes that the ranking problem is qualitatively different from the estimation problem, and (ii) it connects to the Design Space Theorem, showing that the Fisher information rank deficiency determines the boundary between Family A and Family B."

### 2E. Scope NTK Extension Honestly

**Addresses**: #9
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Replace the last sentence of the NTK remark in supplement with:
> "This extension is conjectural: a rigorous proof requires (i) uniform NTK approximation bounds across the Rashomon set, (ii) a spectral gap argument relating feature correlation to NTK eigenvalue separation, and (iii) control of the finite-width correction. None of these are available in the current literature. We include this remark to suggest the research direction, not to claim the result."

### 2F. Strengthen Lasso Treatment

**Addresses**: #10
**Classification**: **(N)** — one additional sentence in main text §3.2
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Add to main text Lasso section:
> "Formally, Lasso is an iterative optimizer with d(f) = the selected feature and φ_{d(f)} > 0, φ_k = 0 for k ≠ d(f). The impossibility applies to any model class with hard feature selection, including stepwise regression and ℓ₀-penalized methods."

### 2G. Real-Time Guidance and Practical Assumptions

**Addresses**: #15, #16, #26
**Classification**: **(N)**
**Dependencies**: None
**Effort**: 1 hour

**What**: Add a "Practitioner Guidance" paragraph to Discussion:
> "**Computational cost.** DASH requires M model trainings; at M=25, this is 25× the cost of a single model. For batch pipelines (daily/weekly model refreshes), this is feasible. For real-time SHAP explanations, M=25 is prohibitive; use the F5 single-model diagnostic to flag unstable pairs, then report affected features with an instability caveat. For high-dimensional feature spaces (P > 500), the pairwise F1 diagnostic scales as O(P²); the correlation-based grouping step prunes this to O(G·m²) where G is the number of correlated groups. The equicorrelation assumption simplifies the axioms but is not required for the diagnostics: F1 and F5 operate on observed attribution arrays and make no distributional assumptions. The proportionality axiom (CV ≈ 0.35–0.66 depending on tree depth) affects the quantitative ratio 1/(1-ρ²), not the qualitative impossibility."

### 2H. Prevalence Survey Power and Reproducibility Crisis

**Addresses**: #24, #27
**Classification**: **(N)**
**Dependencies**: None
**Effort**: 0.5 hours

**What**:
- In abstract, add "(conservative: survey power 32% at the 10% flip-rate threshold)" after "approximately 60%"
- In Discussion broader implications paragraph, add: "The low survey power means the true prevalence is likely higher. Published studies reporting single-model SHAP rankings as definitive findings may be contributing to the reproducibility crisis in ML-based discovery."

### 2I. Intersectional Fairness Acknowledgment

**Addresses**: #12
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Add one paragraph to supplement fairness section:
> "**Intersectional considerations.** When multiple protected attributes are each proxied by collinear non-protected features, the impossibility applies independently to each proxy pair. The worst case is additive: if K protected attributes each have unstable proxies, the probability that the audit correctly identifies all K proxy reliance directions is (1/2)^K, exponentially vanishing in K. A full analysis of intersectional instability is left to future work."

### 2J. Preemptive Defense Against "Already Known"

**Addresses**: #32
**Classification**: **(N)** — incorporated into 2A
**Dependencies**: 2A
**Effort**: included in 2A

Add to Related Work, after the Laberge comparison:
> "The qualitative intuition that correlated features have unstable rankings is practitioner folklore; our contribution is making it precise (exactly two families), quantitative (architecture-discriminating bounds), provably optimal (DASH Pareto optimality with tight ensemble size bounds), and machine-verified (Lean 4 with zero axiom dependencies for the core impossibility)."

---

## Phase 3: New Experiments

**Priority**: MEDIUM — strengthens the empirical case but not strictly required.

### 3A. Enterprise-Scale Validation (P > 500)

**Addresses**: #14, #21, #26
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 3 hours

**What**: Script `paper/scripts/high_dimensional_validation.py`

**How**:
- Generate synthetic high-dimensional data: P=500 features in 50 groups of 10, within-group ρ=0.8
- Train 20 XGBoost models, compute SHAP, measure instability
- Key measurements: unstable pair count, F1 diagnostic performance, wall-clock time for full workflow
- Expected: F1 maintains |r|>0.8; wall-clock <30 min on Apple Silicon

**Supplement addition**: New subsection "High-Dimensional Scalability" with results table and timing.

### 3B. Conditional SHAP with Known Causal Structure

**Addresses**: #23
**Classification**: **(S)**
**Dependencies**: Phase 2B (text discussion)
**Effort**: 3 hours

**What**: Script `paper/scripts/conditional_shap_causal.py`

**How**:
- Generate data from known causal DAG: X₁ → Y, X₂ → Y, X₁ ↔ X₂ (correlated)
- Two scenarios: β₁ = β₂ (symmetric) and β₁ ≠ β₂ (asymmetric)
- Train 20 XGBoost models per scenario
- Compute both marginal SHAP and interventional SHAP approximation
- Expected: symmetric case both unstable; asymmetric case conditional SHAP stable

### 3C. Neural Network SHAP Validation

**Addresses**: #22
**Classification**: **(S)** if time permits; **(J)** if not
**Dependencies**: None
**Effort**: 3-4 hours

**What**: Script `paper/scripts/nn_shap_validation.py`

**How**:
- sklearn MLPRegressor or small PyTorch MLP on Breast Cancer
- 20 models with different seeds, KernelSHAP
- Measure flip rate for correlated pairs

**Risk**: KernelSHAP is slow (~10 min/model). Budget 20 models × 10 min.

**Fallback**: If time-constrained, classify as (J) and note: "Neural network SHAP validation is deferred; the LLM attention instability results provide preliminary evidence of generality."

### 3D. LLM Attention Instability — Honest Scoping

**Addresses**: #28
**Classification**: **(S)** — text edit only
**Dependencies**: None
**Effort**: 0.5 hours

**What**: Add scoping paragraph to existing LLM section:
> "**Scope.** The attention-based instability reported here (14.5% of adjacent token pairs under fine-tuning) is preliminary evidence, not a formal extension of the impossibility theorem. Attention weights are not SHAP values and do not satisfy the proportionality axiom. We report this as suggestive evidence for the generality of the phenomenon."

---

## Phase 4: Theoretical Extensions

**Priority**: MEDIUM-LOW for NeurIPS; HIGH for JMLR.

### 4A. Symmetric Bayes Dichotomy Generalization

**Addresses**: #4, #13
**Classification**: **(J)** — deferred entirely
**Effort**: 0 hours now; 20+ hours for JMLR

**Why (J)**: A general theorem about group actions on hypothesis spaces requires defining "G-symmetric decision problem" formally and proving the two-family decomposition. This is a separate paper's worth of work.

**What to do now**: Add one sentence to Discussion: "A general theorem proving that G-invariance always yields a two-family decomposition is the natural next step; our three instances provide the test cases."

### 4B. Local vs. Global SHAP Analysis

**Addresses**: #19
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 1.5 hours

**What**: New supplement subsection "Local vs. Global Attribution Instability."

**Argument**: For local SHAP at a fixed data point x, the impossibility still holds. The flip rate for local attributions is at least as high as for global (averaging over data points can only reduce variance). One-paragraph proof sketch + remark.

### 4C. DASH Breakdown Point

**Addresses**: #20
**Classification**: **(S)**
**Dependencies**: None
**Effort**: 1 hour

**What**: New supplement subsection "DASH Robustness and Breakdown Point."

- DASH (sample mean) has breakdown point 0
- Median has breakdown point 1/2 but ARE = 2/π (already in supplement)
- Recommend trimmed mean (5% trim) for production: ARE 0.992, breakdown point 0.05
- One paragraph + comparison table

### 4D. Design Space Exhaustiveness — Optimal-Unfaithful Case

**Addresses**: #6
**Classification**: **(J)** for Lean; already proved in supplement
**Dependencies**: None
**Effort**: 0 hours now; 8-15 hours for Lean version

---

## Phase 5: Lean Formalization Extensions

**Priority**: LOW for NeurIPS. All deferred.

### 5A. Axiom Consistency Model in Lean

**Addresses**: #17 (Lean version)
**Classification**: **(J)**
**Effort**: 0 hours now; 6-10 hours for JMLR

Python construction (Phase 1A) suffices for NeurIPS.

### 5B. Design Space Exhaustiveness in Lean

**Addresses**: #6
**Classification**: **(J)**
**Effort**: 0 hours now; 15-25 hours for JMLR

Requires measure-theoretic expected loss in Lean/Mathlib.

---

## Phase 6: Venue Preparation

### 6A. NeurIPS 13 → 9 Page Restructure

**Addresses**: #30
**Classification**: **(N)**
**Dependencies**: Phases 1-2 complete
**Effort**: 5-8 hours
**Timing**: Week of April 28

**Specific cuts**:
1. Cut NN section (§3.3) → supplement. Save ~0.3 pages.
2. Cut RF section (§3.4) → supplement. Save ~0.3 pages.
3. Compress Discussion to ~1.2 pages. Save ~0.8 pages.
4. Compress Related Work into tighter format. Save ~0.3 pages.
5. Move Axiom 3 details to supplement. Save ~0.3 pages.
6. Move Table 1 to supplement. Save ~0.3 pages.
7. Inline three experimental figures as single 2-column figure. Save ~0.2 pages.
8. Move full Design Space theorem statement to supplement, keep informal + figure. Save ~0.5 pages.
9. Line-by-line prose tightening. Save ~0.5 pages.

**Total savings target**: ~3.5 pages. Net with ~1 page new text: 13 - 3.5 + 1 = ~10.5. Further cutting needed.

**Risk**: Over-cutting loses clarity. Have Bryan/David read the 9-page version.

### 6B. arXiv Preprint

**Addresses**: #33
**Classification**: **(N)**
**Dependencies**: Phases 1-2 complete
**Timing**: April 30 – May 2
**Effort**: 2 hours

**Steps**:
1. Switch to `\usepackage[preprint]{neurips_2026}`
2. Uncomment author block
3. Fill in repository URL
4. Compile both PDFs
5. Upload to arXiv under cs.LG + cs.AI + stat.ML
6. Switch back to anonymous for NeurIPS submission

### 6C. JMLR Roadmap Document

**Addresses**: #31
**Classification**: documentation only
**Effort**: 1 hour

**What**: Create `docs/jmlr-roadmap.md` listing all (J) items:
1. Axiom consistency model in Lean (5A)
2. Design Space exhaustiveness in Lean (5B)
3. Symmetric Bayes dichotomy generalization (4A)
4. Neural network SHAP validation (3C if deferred)
5. Mutual information generalization (1B)
6. Consensus variance bound from Mathlib's IndepFun.variance_sum
7. Full 35-40 page restructure integrating supplement into main text

**Target**: Q4 2026 submission.

### 6D. Post-Completion Housekeeping

- Update CLAUDE.md if any axiom/declaration counts change
- Update docs/self-verification-report.md with axiom consistency result
- Re-run NeurIPS checklist to ensure all abstract claims match paper content
- Replace `neurips_2026.sty` placeholder with official style file when available

---

## Items Classified as (X) — Not Worth Doing

| Issue | Why Not |
|-------|---------|
| #12 full intersectional analysis | Separate research direction; paragraph acknowledgment (2I) is sufficient |
| #18 MI generalization proof | Requires non-Gaussian Rashomon set theory that doesn't exist; sentence in Open Problems is honest |
| #13 reduction template formal proof | Core of a follow-up paper; requires category theory/abstract algebra framework |

---

## Critical Path (Minimum for May 4)

```
Week 1 (Apr 1-7):
  1A  Axiom consistency script + supplement text     [3.5 hrs]
  2A  Reframe abstract/intro around Design Space     [2.5 hrs]
  2B  Expand conditional SHAP + Janzing/D'Amour      [1.5 hrs]
  2G  Practitioner guidance paragraph                 [1 hr]
  2H  Prevalence power + reproducibility crisis       [0.5 hrs]
  2J  Preemptive "already known" defense              [incl. in 2A]

Week 2 (Apr 8-14):
  2C-2F  Supplement text fixes (Berry-Esseen, FIM,    [2.5 hrs]
         NTK, Lasso)
  2I  Intersectional fairness acknowledgment           [0.5 hrs]
  3A  High-dimensional validation script               [3 hrs]
  3B  Conditional SHAP causal experiment               [3 hrs]

Week 3 (Apr 15-21):
  3C  Neural network validation (if time)              [3-4 hrs]
  3D  LLM scoping paragraph                            [0.5 hrs]
  4B  Local vs global SHAP analysis                    [1.5 hrs]
  4C  DASH breakdown point                             [1 hr]

Week 4 (Apr 22-28):
  6A  NeurIPS 13→9 restructure                         [5-8 hrs]

Apr 30-May 2:
  6B  arXiv preprint                                   [2 hrs]

May 4: Abstract deadline
May 6: Paper deadline
```

**Total**: ~32-38 hours across 5 weeks.

---

## Triage List (What Gets Cut If Time Runs Out)

In order of what to sacrifice first (least → most impact):

1. **3C** Neural network SHAP validation → defer to (J)
2. **4C** DASH breakdown point → defer to (J)
3. **4B** Local vs global SHAP → defer with one-sentence note
4. **3D** LLM scoping paragraph → minor polish, skip if tight
5. **2I** Intersectional fairness → one sentence instead of paragraph
6. **2C-2E** Supplement fixes (Berry-Esseen, FIM, NTK) → supplement is less scrutinized
7. **3B** Conditional SHAP experiment → already have theoretical result + threshold simulation
8. **3A** High-dimensional validation → already have P=126; add sentence about scaling instead

**Do NOT cut**: 1A (axiom consistency), 2A (reframing), 2B (conditional SHAP text), 2G (practitioner guidance), 2H (prevalence/reproducibility), 6A (restructure), 6B (arXiv).

---

## Co-Author Review Items (Bryan Arnold, David Rhoads)

Items requiring human judgment that Claude cannot provide:

| Item | What They Need to Do | Why Claude Can't |
|------|---------------------|------------------|
| **Lean ↔ paper alignment** | For each numbered theorem in main.tex, verify the Lean statement matches the English statement | Requires understanding both mathematical intent and Lean encoding |
| **Axiom plausibility** | Read the 7 domain-specific axioms and assess whether they are reasonable idealizations of real GBDT behavior | Requires domain expertise judgment |
| **Novelty vs. Laberge 2023** | Read Laberge et al. (2023) in full and verify this paper's impossibility is genuinely new | Requires reading a 50-page paper for subtle overlaps |
| **Experimental design** | Verify the 50-seed, 6-ρ-level synthetic design is sufficient | Requires judgment about subfield expectations |
| **Axiom consistency model** | Cross-check the Python output against the Lean definitions | Requires careful mathematical verification |
| **9-page version readability** | Read post-restructure NeurIPS draft and flag clarity losses | Requires reading as a naive reader |
| **Regulatory claims** | Verify EU AI Act Art. 13(3)(b)(ii) and SR 11-7 citations are correctly interpreted | Requires legal/regulatory expertise |
| **Abstract accuracy** | Confirm every claim in the abstract is supported by a specific result | Requires holistic reading |
| **Reproduce 3 key experiments** | Run `validate_ratio.py`, `cross_implementation_validation.py`, `snr_calibration.py` on a different machine | Requires different hardware environment |

**Timeline**: Send co-author guide + verification checklist by April 7. Request feedback by April 21.

---

## Confidence Ratings

| Plan Element | Confidence | Justification |
|-------------|-----------|---------------|
| Phase 1A (axiom consistency) will work | HIGH | Simple finite construction; well-defined verification |
| Phase 2A-2J text edits will fit in 9 pages | MEDIUM | Depends on restructure success |
| Phase 3A-3B experiments will confirm theory | HIGH | Previous 11-dataset validation gives confidence |
| Phase 3C (NN SHAP) will work within budget | MEDIUM | KernelSHAP is slow |
| Phase 6A restructure achieves 9 pages | MEDIUM | Aggressive cuts required |
| Total effort ≤ 38 hours | MEDIUM | Could be 40+ if experiments hit snags |
| NeurIPS acceptance after plan | MEDIUM | Estimate: 40-55% → 50-65% |
