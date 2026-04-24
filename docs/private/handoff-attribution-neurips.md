# Handoff: Universal Impossibility → Attribution NeurIPS Session

**Date:** 2026-04-17
**From:** universal-explanation-impossibility repo (Nature / peer review session)
**To:** dash-impossibility-lean repo (NeurIPS 2026 submission)

## Deadline

**NeurIPS 2026: Abstract May 4, Paper May 6.** 17 days from today.

## Current State of the Attribution Paper

The DASH repo has two paper versions:

| File | Format | Lines | Status |
|------|--------|-------|--------|
| `paper/main.tex` | NeurIPS (10pp) | 504 | **STALE** — missing all new results |
| `paper/main_definitive.tex` | Full (JMLR-length) | 3,996 | **Up to date** — has everything |
| `paper/supplement.tex` | Supplement | 4,724 | Partially updated |

**The task:** Extract a compelling 10-page NeurIPS paper from `main_definitive.tex`, updating the stale `main.tex`.

## What's New Since main.tex Was Last Updated

### New Theory (in main_definitive.tex, not in main.tex)

1. **Quantitative bilemma** (sec:quantitative-bilemma):
   For (ε,δ)-stable E on an ε-Rashomon pair with gap Δ:
   unfaith(θ₁) + unfaith(θ₂) ≥ Δ - δ
   Extends the binary bilemma to continuous SHAP magnitudes.
   Proved in Lean (ApproximateBilemma.lean). Testable.

2. **Compatibility complex** (paragraph in bilemma section):
   Tightness classification = topology of simplicial complex K on H.
   Discrete K → bilemma. Contractible K → F+S achievable.
   NOT sheaf-theoretic contextuality — simpler obstruction (empty stalks).
   Framework sits BELOW Abramsky-Brandenburger in obstruction hierarchy.

3. **Coverage conflict as nonparametric diagnostic** (sec:nonparametric-flip):
   Minority fraction = min(n_pos, n_neg)/M. Seven lines. No assumptions.
   Spearman 0.92–0.98 vs Gaussian 0.46–0.89 on 5 datasets.
   Complete theory-to-practice pipeline: theorem → diagnostic → tool → resolution.

4. **Fairness impossibilities as future instances** (remark in sec:fairness):
   Chouldechova/KMR as AbstractImpossibility instances — noted as future work.

### New Experiments (results JSONs in universal repo, scripts in dash-shap)

5. **Bimodality validation** (sec:bimodality-validation):
   Hartigan dip test: p < 0.002 under ρ ≥ 0.5, control passes (ρ=0, p=0.373).
   Second mode at ~0.27 (not 0.50). Permutation-validated.

6. **Coverage conflict empirical comparison**:
   CC precision 0.74–0.86 at 10% threshold.
   CC Spearman 0.96 vs Gaussian 0.46 on California Housing.

7. **Clinical decision reversal** (universal repo: results_clinical_decision_reversal_v2.json):
   45% of German Credit applicants receive different explanation categories.
   Ablation: 4 conditions × 3 model classes × 3 datasets.
   Narrative: Applicant #91 gets 6 different "most important feature" across seeds.

### New Lean State

8. **Axiom reduction: 16 → 6 axioms** (from Ostrowski session):
   - All ML instances now constructive (Bool/Unit, zero axioms)
   - Arrow proved from scratch
   - Total: 357 theorems, 6 axioms, 0 sorry

9. **Arrow is now proved** (not axiomatized):
   Direction theorem (coverage conflict anti-monotone vs Arrow monotone) is a proved structural observation.

### What's Already in main_definitive.tex

All 9 items above are in `main_definitive.tex`. The supplement (`supplement.tex`) has most of the extended content. The NeurIPS `main.tex` has NONE of them.

## What the NeurIPS Paper Should Contain (10 pages)

### The Story (in order)

1. **Opening** (0.5 pages): The SHAP flip anecdote. 68% of datasets affected. The question: is this fixable?

2. **The impossibility** (1 page): Three desiderata (faithful, stable, complete). The Rashomon property. The 4-line proof. Zero axioms. Tightness (each pair achievable).

3. **The design space** (1 page): Exactly two families (Family A: single-model, 50% flip; Family B: DASH ensemble, reports ties). Pareto-optimal. No escape via shrinkage (James-Stein paragraph).

4. **The quantitative bilemma** (0.5 pages): Extension to continuous H. The Δ-δ bound. Practical interpretation: if models disagree by Δ, you're unfaithful by ≥ (Δ-δ)/2.

5. **The diagnostic pipeline** (1.5 pages): THE headline for NeurIPS reviewers.
   - Coverage conflict: exact condition for bilemma to bind
   - Minority fraction: nonparametric flip predictor (7 lines, Figure)
   - Comparison: Spearman 0.96 vs Gaussian 0.46 on real data (Table)
   - Bimodality: dip test p < 0.002, permutation-validated (Table)

6. **Clinical decision reversal** (1 page): 45% of German Credit applicants. The Applicant #91 narrative. DASH resolves to 92% agreement. Ablation across model classes.

7. **DASH resolution** (1 page): Pareto-optimal. O(M) cost. Between-group stable, within-group tied. DASH ≠ changing predictions.

8. **Lean verification** (0.5 pages): 357 theorems, 6 axioms, 0 sorry. Core impossibility: zero axioms. All instances constructive.

9. **Discussion + limitations** (1 page): Binary limitation (SHAP sign IS binary). Group identification bottleneck. Regulatory implications (one paragraph). Fairness connection (one paragraph).

10. **Related work** (0.5 pages): Bilodeau et al. 2024 (complementary). Slack et al. (adversarial, different mechanism). Krishna et al. (disagreement, different framing).

### What Goes in the Supplement

Everything in `main_definitive.tex` that doesn't fit in 10 pages:
- All proofs (extended)
- Quantitative bounds by model class (GBDT ratio, Lasso, neural net, random forest)
- Symmetric Bayes Dichotomy (general two-families theorem)
- Conditional attribution impossibility
- Fairness audit impossibility (full section)
- Extended experimental results (all 15+ subsections)
- Regulatory mapping (EU AI Act full analysis)
- Topological analysis (permutohedron)
- Arrow comparison
- SymPy verification
- Lean theorem listings

## Key Peer Review Findings to Address

From 29 simulated reviewers + Red Team on the universal paper (many apply to attribution):

1. **"The core theorem is elementary."** Defense: the contribution is the framework + quantitative predictions + practical tools, not the proof depth. The NeurIPS paper should lead with the diagnostic tool, not the theorem.

2. **"DASH Pareto-optimality is within unbiased linear."** Already qualified in the paper. James-Stein paragraph acknowledges biased alternatives.

3. **"The 68% prevalence is from benchmarks."** Add caveat: "benchmark-representative tabular datasets." The clinical reversal (45% on German Credit) is the stronger per-individual result.

4. **"NeurIPS reviewers demand tools."** The coverage conflict diagnostic IS the tool. Seven lines. No assumptions. Outperforms the parametric alternative. This should be the paper's selling point.

5. **"Acknowledge the binary limitation."** The SHAP sign question IS binary. The quantitative bilemma extends to continuous H. Both are in the paper.

## What to Read First

1. `paper/main_definitive.tex` — the source of truth (3,996 lines)
2. `paper/main.tex` — the stale NeurIPS version (504 lines) — this is what needs rewriting
3. This handoff document
4. `../universal-explanation-impossibility/docs/publication-strategy.md` — the overall strategy
5. `../universal-explanation-impossibility/knockout-experiments/results_clinical_decision_reversal_v2.json` — the 45% result

## File Locations

| File | Content |
|------|---------|
| `paper/main_definitive.tex` | Source of truth (all new results) |
| `paper/main.tex` | NeurIPS 10pp version (STALE — needs rewrite) |
| `paper/supplement.tex` | Supplement (partially updated) |
| `paper/main_jmlr.tex` | JMLR version (fallback venue) |
| `paper/references.bib` | Shared bibliography |
| `DASHImpossibility/*.lean` | 58 Lean files |
| `paper/figures/` | All figures |
| `../universal-explanation-impossibility/knockout-experiments/results_clinical_decision_reversal_v2.json` | Clinical reversal data |
| `../universal-explanation-impossibility/knockout-experiments/results_drug_discovery_prospective.json` | Drug discovery (for context) |
| `../ostrowski-impossibility/docs/empirical-validation-results.md` | Bimodality + CC validation data |
| `../ostrowski-impossibility/docs/sheaf-contextuality-analysis.md` | Compatibility complex analysis |

## The NeurIPS Pitch (one paragraph)

Feature importance explanations — the backbone of explainable AI — are provably unreliable under collinearity. We prove that no SHAP ranking can simultaneously be faithful, stable, and complete (the Attribution Impossibility), characterize the complete design space (exactly two achievable families), and provide a nonparametric diagnostic that identifies unstable features in seven lines of code, outperforming the Gaussian formula by 2× on real data. A per-individual analysis shows 45% of loan applicants receive different explanation categories under standard model regularization. The entire framework is verified in Lean 4 (357 theorems, 6 axioms, 0 sorry), and the constructive resolution (DASH ensemble averaging) is proved Pareto-optimal.
