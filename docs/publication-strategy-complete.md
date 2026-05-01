# Complete Publication Strategy: The Limits of Explanation

**One-shot writing guide for every paper in the research program.**
All material references point to specific files across 4 repositories.

**Last verified: 2026-05-01**
**Total Lean: 1,359 theorems (519 + 482 + 358), 5 axioms (2 + 1 + 2), 0 sorry across 3 repos**

**URGENT: NeurIPS 2026 abstract deadline May 4 (3 days). Paper deadline May 6.**

---

## Executive Summary

### Papers to publish (in order of submission)

| # | Paper | Venue | Status | Timeline |
|---|-------|-------|--------|----------|
| 1 | **The Limits of Explanation** | Nature | Ready | Submit now (rolling) |
| 2 | **The Attribution Impossibility** | NeurIPS 2026 | Ready (1-page cut needed) | Abstract May 4, Paper May 6 |
| 3 | **Universal monograph** | arXiv | Ready | Post same week as Nature |
| 4 | **Attribution monograph** | arXiv | Ready | Post same week as NeurIPS |
| 5 | **First-Mover Bias** (DASH method) | TMLR | Under review | Awaiting decision |
| 6 | **Spacetime and the Bilemma** | Foundations of Physics | Ready (21/21 Accept) | Submit now |
| 7 | **The Attribution Impossibility** (full) | JMLR | Ready | After NeurIPS decision |

### Additional papers to extract (recommended)

| # | Paper | Venue | Material location | Effort |
|---|-------|-------|-------------------|--------|
| 8 | **The MI Exact Boundary** | ICML or AISTATS | `MutualInformation.lean` + `MIQuantitativeBridge.lean` + drug discovery | 2-3 weeks |
| 9 | **Impossibility Classification** | J. Economic Theory or Social Choice | Ostrowski `UnifiedMetaTheorem.lean` + all 23 instances | 3-4 weeks |
| 10 | **Circuit Stability in Neural Networks** | ICLR or AI Safety venue | MI v2 experiments + TinyStories + GPT-2 | 2-3 weeks |
| 11 | **The Explanation Capacity Audit** | NeurIPS Datasets & Benchmarks | 149-dataset audit + SAGE | 2 weeks |
| 12 | **NARPS Reanalysis** | Nature Neuroscience or NeuroImage | Brain imaging experiments | 3-4 weeks |
| 13 | **SAGE: Automatic Stability Auditing** | TMLR or JMLR (Software) | SAGE algorithm + 131-dataset MI comparison | 2 weeks |

---

## Paper 1: The Limits of Explanation (Nature)

### Framing

A Godel/Shannon/Arrow-tier discovery: the fundamental limits of what observation can explain about underspecified systems. Not an ML paper — a science paper about the structure of explanation itself. Eight scientific communities independently converged on the same resolution over a century; this paper explains why.

### What goes in

- The abstract theorem (`explanation_impossibility`, 4 lines, zero axioms)
- The bilemma strengthening
- Four domain instances (genomics, mechanistic interpretability, causal inference, neuroimaging)
- The η law (R² = 0.957, zero free parameters)
- The tightness classification (23 theorems, 14+ domains — Extended Data Table 2)
- The constructive resolution (orbit average, proved Pareto-optimal)
- The Shannon parallel (capacity, coding theorem, over-explanation penalty)
- The Langlands connection (GL(n) collapsed tightness, character as resolution)
- Implementation pipeline (SAGE, DASH — 1 paragraph in Discussion + SI)
- The 149-dataset audit (1 paragraph: "75% exceed capacity, Wilcoxon p=5.1×10⁻¹¹, 27:1")
- Navier-Stokes numerical tightness classification (1 paragraph)
- Formal verification statement (519 + 357 + 482 = 1,358 theorems, 0 sorry)

### What does NOT go in (to avoid dual submission with NeurIPS)

- GBDT-specific mechanism (first-mover bias, split-count model)
- DASH algorithm details (training, filtering, diversity selection)
- Model-class-specific bounds (ratio 1/(1-ρ²), Lasso ∞, RF O(1/√T))
- Empirical benchmarks vs baselines
- The F5→F1 stability API
- Drug discovery MI clustering details (cite only)

### Specific material references

| Content | Source file | Repo |
|---------|------------|------|
| Core theorem | `ExplanationSystem.lean:explanation_impossibility` | universal |
| Bilemma | `MaximalIncompatibility.lean:bilemma` | universal |
| Tightness classification | `UnifiedMetaTheorem.lean` (23 instances) | ostrowski |
| Gene expression | `results_gene_expression_replication.json` | universal |
| MI v2 circuits | `results_mi_v2_final_validation.json` | universal |
| NARPS convergence | `results_brain_imaging_bulletproof.json` | universal |
| Causal inference | `CausalInstanceConstructive.lean` | universal |
| η law | `results_universal_eta.json` | universal |
| Audit summary | `results_audit_150_final.json` | universal |
| GL(n) boundary | `GLnLanglands.lean` | ostrowski |
| Navier-Stokes | `NavierStokesImpossibility.lean` + `results/ns3d_bulletproof.json` | ostrowski |
| Enrichment stack | `EnrichmentStack.lean` (parametric, 0 axioms) | universal |
| Pareto optimality | `ParetoOptimality.lean:dash_unique_pareto_optimal` | universal |
| MI boundary | `MutualInformation.lean:mi_is_exact_boundary` | universal |

### Key numbers (all verified 2026-04-30)

- 519 theorems (universal), 482 (ostrowski), 357 (attribution) = **1,358 total**
- R² = 0.957 (7 well-characterised groups), holdout R² = 0.24 (9 approximate)
- TSPAN8 92% / CEACAM5 6%, ρ = 0.858
- ρ = 0.518 → 0.929 (MI G-invariant projection), Jaccard = 2.2%
- M₉₅ = 16 [10, 22] for NARPS
- 75% exceed capacity (111/149 at ρ*=0.70), Wilcoxon p = 5.09 × 10⁻¹¹
- 23 impossibility theorems, 14+ domains, 3 collapsed tightness

### Word limit and structure

~2,500 words. Structure: Abstract → Introduction → Results (4 subsections: impossibility, tightness, 4 instances, η law) → Discussion (structural severity, Shannon parallel, Langlands, implementation, audit, outlook) → Methods.

### File

`universal-explanation-impossibility/paper/nature_article.tex` (ready)

---

## Paper 2: The Attribution Impossibility (NeurIPS 2026)

### Framing

An ML deep dive: the attribution impossibility as a concrete, quantitative result for feature importance methods. Two levels (input-level SHAP, component-level activation patching), one impossibility, one resolution. The MI boundary is the headline: mutual information is the exact condition, and the Δ/2 quantitative floor is unreachable.

### What goes in

- The attribution-specific impossibility (zero axioms, but framed for ML audience)
- The bilemma for SHAP sign, feature selection, counterfactual direction
- GBDT mechanism: first-mover bias, split-count model, ratio 1/(1-ρ²) → ∞
- Lasso (ratio = ∞), neural net (conditional), random forest (bounded O(1/√T))
- The MI exact boundary (`mi_is_exact_boundary`) — mutual information, not correlation
- The Δ/2 quantitative floor (`mi_quantitative_unfaithfulness`)
- DASH resolution: proved Pareto-optimal, variance σ²/M
- Design space theorem: Family A / Family B, exhaustive
- Component-level: TinyStories (ρ = 0.565 → 0.972), GPT-2 boundary test
- Drug discovery: Pearson fails (0%), MI recovers (19%), actual (23%)
- Prevalence: 68% of 77 datasets show instability
- The nonparametric minority-fraction diagnostic (7 lines, outperforms Gaussian 2×)
- Gaussian flip formula: Φ(-SNR), R² = 0.946-0.980 per dataset
- Coverage conflict: Spearman 0.961 across 15 datasets

### What does NOT go in (different from Nature)

- Cross-domain instances (Arrow, quantum, gauge, linguistics, etc.)
- The tightness classification of 23 impossibilities
- The Langlands connection
- The enrichment stack (physics levels)
- The η law in its cross-domain form
- Neuroimaging (NARPS)
- Navier-Stokes

### Specific material references

| Content | Source file | Repo |
|---------|------------|------|
| Core attribution theorem | `Trilemma.lean:attribution_impossibility` | dash-impossibility-lean |
| GBDT bounds | `General.lean`, `Ratio.lean`, `SplitGap.lean` | dash-impossibility-lean |
| MI boundary | `MutualInformation.lean:mi_is_exact_boundary` | dash-impossibility-lean |
| MI bridge | `MIQuantitativeBridge.lean` | universal |
| Bilemma | `Bilemma.lean`, `BeyondBinary.lean` | dash-impossibility-lean |
| TinyStories | `docs/tinystories-results-reference.json` | dash-impossibility-lean |
| Drug discovery | `results_drug_discovery_mi_clustering.json` | universal |
| Prevalence | 68% of 77 datasets (experiment in dash-impossibility-lean) | dash-impossibility-lean |
| DASH optimality | `ParetoOptimality.lean` | dash-impossibility-lean |
| Gaussian flip | `GaussianFlipRate.lean` + `results_open_questions_capstone.json` | universal |
| Coverage conflict | `results_explanation_landscape_bridge_expanded.json` | universal |

### Dual-submission differentiation from Nature

| Aspect | Nature | NeurIPS |
|--------|--------|---------|
| Scope | Any explanation system | Feature attribution specifically |
| Instances | 4 high-stakes domains + 14 cross-domain | GBDT/Lasso/NN/RF + TinyStories + drug discovery |
| Theory depth | Abstract (zero axioms) | Quantitative (ratio formula, flip formula, MI bridge) |
| Resolution | G-invariant projection (abstract) | DASH (concrete algorithm + benchmarks) |
| Audience | Scientists across all domains | ML researchers and practitioners |
| Cross-domain | 23 impossibilities classified | None |
| Lean | Referenced | 357 theorems cited |

The papers share the CORE THEOREM (unavoidable — it's the same mathematical result) but differ in: scope (universal vs attribution-specific), depth (abstract vs quantitative), instances (cross-domain vs ML-specific), resolution (abstract projection vs DASH algorithm), and audience.

### Page limit and structure

9 pages + references + appendix. Structure: Introduction → Background → The Attribution Impossibility (input-level + component-level) → The MI Boundary → Quantitative Bounds → The Resolution (DASH) → Experiments → Discussion.

### File

`dash-impossibility-lean/paper/main.tex` (ready, needs 1-page cut)

---

## Paper 3: Universal Monograph (arXiv)

### Framing

The definitive technical reference. Everything in one place. ~4,500 lines. All proofs, all experiments, all instances, all bounds. This is the "Supplementary Information" for the entire research program.

### File

`universal-explanation-impossibility/paper/universal_impossibility_monograph.tex` (ready)

---

## Paper 4: Attribution Monograph (arXiv)

### Framing

The definitive attribution-specific reference. 83 pages. All GBDT proofs, all empirical validation, all sensitivity analyses.

### File

`dash-impossibility-lean/paper/main_definitive.tex` (ready)

---

## Paper 5: First-Mover Bias / DASH Method (TMLR)

### Framing

The METHOD paper: how DASH works, why it works, benchmarks against 11 baselines. This is NOT the impossibility paper — it's the engineering paper.

### What goes in

- The first-mover bias mechanism (GBDT-specific)
- DASH pipeline: population → filter → diversify → consensus → diagnostics
- F5→F1→DASH progressive API
- Benchmarks: 18 experiments, 50 reps, M=200, K=30
- Crossed ANOVA: 60% reduction in model-selection noise
- Colsample ablation: low colsample isolates the mechanism
- 11 baseline comparisons
- FSI diagnostic, IS plots
- Real-world: Breast Cancer (+0.549 stability), Superconductor (+0.124)

### Dual-submission differentiation

This paper describes the SOFTWARE METHOD. It doesn't prove impossibility theorems or characterize design spaces. It benchmarks an algorithm. Completely different from Nature and NeurIPS.

### File

`dash-shap/paper/draft_v7_tmlr.tex` (under review) + `draft_v8_reviewer_response.tex`

---

## Paper 6: Spacetime and the Bilemma (Foundations of Physics)

### Framing

The physics companion: applying the bilemma to quantum measurement, black hole information, and spacetime emergence. The enrichment stack as a pattern that mirrors Gödel's incompleteness.

### What goes in

- Ostrowski's classification bridged to Mathlib
- The bilemma applied to physics (wave-particle, information fate, spacetime status)
- Three levels of enrichment, each predicted by the previous
- Tightness classification for physics impossibilities (Bell, KS, Penrose-Hawking)
- Conditional tightness (Navier-Stokes, circuit complexity)
- The Langlands boundary (n=1 full, n≥2 collapsed)
- The Freund-Witten zeta connection
- Arrow and May proved from scratch

### File

`ostrowski-impossibility/paper/fop-submission.tex` (submission-ready, 21/21 Accept)

---

## Paper 7: Attribution Impossibility Full (JMLR)

### Framing

The full journal version of the NeurIPS paper, with all proofs, all experiments, extended discussion. Submitted after NeurIPS decision (either accepted = extended version, or rejected = standalone submission).

### File

`dash-impossibility-lean/paper/main_jmlr.tex` (ready)

---

## Recommended Additional Papers (8-13)

### Paper 8: The MI Exact Boundary (ICML or AISTATS)

**Headline**: Mutual information — not correlation — is the necessary and sufficient condition for the explanation impossibility. MI > 0 → unfaithfulness ≥ Δ/2. Standard diagnostics (Pearson, VIF) miss nonlinear dependence entirely.

**Content**: `mi_is_exact_boundary` theorem + drug discovery case study (Pearson 0%, MI 19%, actual 23%) + 131-dataset MI-vs-correlation comparison (ARI = 0.84) + synthetic X₂ = X₁² example (MI = 1.91, |ρ| = 0.08) + conformal SHAP intervals.

**Sources**: `universal/MutualInformation.lean`, `universal/MIQuantitativeBridge.lean`, `universal/results_drug_discovery_mi_clustering.json`, `universal/results_mi_reaudit.json`, `dash-shap/theory_bridge/mi_only_dependence_test.py`

### Paper 9: The Impossibility Classification (Journal of Economic Theory)

**Headline**: A structural classification of 23 impossibility theorems from 14+ domains by tightness type. Full vs collapsed vs blocked. The explanation bilemma is one of only 3 collapsed instances — structurally more severe than Arrow, Gödel, or Bell.

**Content**: The 2×2 tightness table + all 23 instances with proofs + obstruction taxonomy + conditional tightness for NS and circuit complexity.

**Sources**: `ostrowski/UnifiedMetaTheorem.lean`, `ostrowski/QuantumContextuality.lean`, `ostrowski/NavierStokesImpossibility.lean`, `ostrowski/CircuitComplexity.lean`, `ostrowski/DiophantineImpossibility.lean`, all cross-domain Lean files in both repos.

### Paper 10: Circuit Stability in Neural Networks (ICLR or AI Safety)

**Headline**: Ten transformers computing the same function discover circuits that overlap by 2.2%. The G-invariant projection lifts agreement from ρ = 0.52 to ρ = 0.93. Safety cases should be built on equivalence classes, not specific circuits.

**Content**: MI v2 (10 transformers, modular addition) + TinyStories (2 scales, 7/7 predictions confirmed) + GPT-2 boundary test + MI adversarial audit (alignment lift 1.92×, deeper than permutation symmetry).

**Sources**: `universal/results_mi_v2_final_validation.json`, `universal/results_mech_interp_definitive_v2.json`, `universal/results_mi_audit.json`, `dash-impossibility-lean/docs/tinystories-results-reference.json`, `dash-impossibility-lean/experiments/` (GPU scripts)

### Paper 11: The Explanation Capacity Audit (NeurIPS Datasets & Benchmarks)

**Headline**: 75% of 149 standard ML datasets exceed their explanation capacity. The directional prediction (within > between) holds at p = 5 × 10⁻¹¹ across 110 datasets. SAGE reduces false discovery rate from 57% to <5%.

**Content**: The full 149-dataset audit + null model rejection + SAGE algorithm + conformal intervals + 131-dataset MI comparison + the Gaussian flip formula (R² = 0.95-0.98).

**Sources**: `universal/results_audit_150_final.json`, `universal/results_audit_strengthening.json`, `universal/results_open_questions_capstone.json`, `universal/results_mi_reaudit.json`, `universal/paper/supplementary_information.tex` (SAGE section)

### Paper 12: NARPS Reanalysis (Nature Neuroscience)

**Headline**: The disagreement among 70 neuroimaging teams analyzing the same fMRI data is an instance of the explanation impossibility. 16 independent analyses suffice for 95% stability. Network membership predicts disagreement (d = 0.32).

**Content**: NARPS reanalysis + convergence prescription + network-based grouping + the η law failure (brain regions lack exact exchangeability) + comparison with Silberzahn (29 teams) and Breznau (73 teams).

**Sources**: `universal/results_brain_imaging_bulletproof.json`, `universal/results_multi_analyst_bulletproof.json`

### Paper 13: SAGE Algorithm (TMLR Software)

**Headline**: Automatic stability auditing for feature importance in 7 lines of code. SAGE identifies which ranking claims are reliable and which are structurally unreliable, reducing false discovery rate from 57% to <5%.

**Content**: SAGE algorithm + comparison vs correlation/MI baselines + SAGE gap 0.25 vs correlation 0.05 + worked examples + pip-installable implementation.

**Sources**: `universal/results_sage_audit.json`, `universal/results_sage_baseline_comparison.json`, `universal/paper/supplementary_information.tex`, `dash-shap/dash_shap/core/diagnostics.py`

---

## Dual Submission Matrix

Which content appears in which paper:

| Content | Nature | NeurIPS | TMLR | FoP | JMLR | MI | Classification | Circuits | Audit | NARPS | SAGE |
|---------|--------|---------|------|-----|------|-----|----------------|----------|-------|-------|------|
| Core theorem | ✓ | ✓ | — | ✓ | ✓ | ✓ | — | — | — | — | — |
| Bilemma | ✓ | ✓ | — | ✓ | ✓ | — | ✓ | — | — | — | — |
| GBDT mechanism | — | ✓ | ✓ | — | ✓ | — | — | — | — | — | — |
| DASH method | brief | brief | **✓** | — | brief | — | — | — | — | — | — |
| MI boundary | ✓ | ✓ | — | — | ✓ | **✓** | — | — | — | — | — |
| Tightness (23) | ✓ | — | — | ✓ | — | — | **✓** | — | — | — | — |
| Gene expression | **✓** | brief | — | — | — | — | — | — | — | — | — |
| MI circuits | **✓** | ✓ | — | — | — | — | — | **✓** | — | — | — |
| NARPS | **✓** | — | — | — | — | — | — | — | — | **✓** | — |
| η law | **✓** | — | — | — | — | — | — | — | ✓ | — | — |
| 149-dataset audit | brief | — | — | — | — | — | — | — | **✓** | — | — |
| SAGE algorithm | SI | — | — | — | — | — | — | — | ✓ | — | **✓** |
| Drug discovery | brief | ✓ | — | — | ✓ | ✓ | — | — | — | — | — |
| Physics stack | brief | — | — | **✓** | — | — | ✓ | — | — | — | — |
| Langlands | ✓ | — | — | ✓ | — | — | ✓ | — | — | — | — |
| NS tightness | brief | — | — | ✓ | — | — | ✓ | — | — | — | — |

**✓** = primary content, **brief** = mentioned/cited, **—** = absent

Each paper has a UNIQUE primary contribution:
- Nature: cross-domain breadth + 4 instances + η law
- NeurIPS: attribution-specific depth + MI boundary + GBDT bounds + component-level
- TMLR: DASH method + benchmarks + engineering
- FoP: physics enrichment + Langlands + conditional tightness
- MI paper: MI as exact boundary + drug discovery
- Classification: 23 impossibilities by structural type
- Circuits: neural network interpretability stability
- Audit: 149 datasets at scale
- NARPS: neuroimaging reanalysis
- SAGE: automatic diagnostic tool

---

## Naming Conventions (mandatory across all papers)

Per `universal-explanation-impossibility/docs/naming-conventions.md`:

| Concept | Canonical Name | Do NOT use |
|---------|---------------|------------|
| F+S+D impossible | The explanation trilemma | — |
| F+S impossible (binary) | The explanation bilemma | — |
| C = dim(V^G) | Explanation capacity | — |
| η = 1 − C/dim(V) | Explanation loss rate | — |
| Capacity predicts instability | Explanation Capacity Theorem | "law", "eta law" |
| unfaith₁+unfaith₂ ≥ Δ−δ | Explanation uncertainty bound | "tradeoff bound" |
| Orbit average / DASH | The stable projection | "the explanation code" |
| g(g−1)/2 stable facts | Stable fact count | "Noether counting law" |
| ‖v−Rv‖²+‖Rv‖²=‖v‖² | Explanatory information loss | "Pythagorean decomposition" |
| ‖w‖ ≤ ‖u−w‖ | Over-explanation penalty | "beyond-capacity penalty" |
| 4-part theorem | Explanation Stability Theorem | "Explanation Coding Theorem" |

---

## Verified Numbers Reference (all cross-checked 2026-04-30)

### Lean

| Repo | Theorems | Axioms | Files | Sorry |
|------|----------|--------|-------|-------|
| universal-explanation-impossibility | 519 | 2 | 102 | 0 |
| ostrowski-impossibility | 482 | 1 | 38 | 0 |
| dash-impossibility-lean | 358 | 2 | 58 | 0 |
| **Total** | **1,359** | **5** | **198** | **0** |

**TMLR contingency:** If the DASH method paper (Paper 5) is rejected, its engineering content can fold into the JMLR attribution paper (Paper 7), creating a combined theory+method submission. The JMLR paper is already written and would absorb the benchmarks with minimal editing.

### Key empirical numbers

| Claim | Value | Source |
|-------|-------|--------|
| η law R² (7 well-characterised) | 0.957 | `universal/results_universal_eta.json` |
| η law R² (16 all) | 0.60 | `universal/results_universal_eta.json` |
| η law holdout R² (9 approximate) | 0.24 | `universal/results_eta_law_oos_gof.json` |
| Exceedance (ρ*=0.70) | 75% (111/149) | `universal/results_audit_150_final.json` |
| Cross-dataset Wilcoxon | p = 5.09 × 10⁻¹¹ | Computed from audit data |
| Real-world only Wilcoxon | p = 1.7 × 10⁻⁶ (59 datasets) | Computed from audit data |
| Family-level Wilcoxon | p = 2.5 × 10⁻⁸ (72 families) | `universal/results_audit_strengthening.json` |
| Block bootstrap CI | [0.036, 0.069] | `universal/results_audit_strengthening.json` |
| Confirmation-to-reversal | 27:1 at p<0.005 | Computed from audit data |
| Bonferroni | 22:0 at p<0.05/149 | Computed from audit data |
| TSPAN8/CEACAM5 | 92%/6%, ρ=0.858 | `universal/results_gene_expression_replication.json` |
| MI circuits (raw → G-inv) | ρ=0.518 → 0.929 | `universal/results_mi_v2_final_validation.json` |
| Fourier Jaccard | 2.2% | `universal/results_mech_interp_definitive_v2.json` |
| NARPS M₉₅ | 16 [10, 22] | `universal/results_brain_imaging_bulletproof.json` |
| Drug discovery (Pearson/MI/actual) | 0%/19%/23% | `universal/results_drug_discovery_mi_clustering.json` |
| MI vs corr ARI (131 datasets) | 0.84 mean, identical 77% | `universal/results_mi_reaudit.json` |
| Gaussian flip per-pair R² | 0.946–0.980 | `universal/results_open_questions_capstone.json` |
| SAGE vs correlation gap | 0.25 vs 0.05 | `universal/results_open_questions_final.json` |
| Null model OLS (14 datasets) | coef=0.037, CI [0.014,0.045] | `universal/results_open_questions_final.json` |
| DASH stability (ρ=0.9) | 0.977 vs SB 0.958 | `dash-shap/results/tables/` |
| DASH Breast Cancer improvement | +0.549 stability | `dash-shap/results/tables/` |
| FoP peer review | 21/21 Accept (Round 4) | `ostrowski/docs/` |

---

## Submission Checklist

Before submitting any paper:
1. Run `lake build` in the relevant Lean repo — must pass with 0 errors
2. Run the verification block: theorem count, axiom count, file count, sorry count
3. Cross-check every number in the paper against its source JSON
4. Verify naming conventions compliance
5. Check dual submission matrix — no overlapping primary content
6. Run `prepare_arxiv.sh` for arXiv versions (uncomments authors)
7. Verify figures regenerate from scripts in `paper/scripts/`
