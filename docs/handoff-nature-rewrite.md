# Handoff: Nature Paper Rewrite Session

**Date:** 2026-04-20
**Status:** COMPLETED. The rewrite was executed in the session of 2026-04-20/21. The Nature paper (`paper/nature_article.tex`, 729 lines) now incorporates all four instances, the tightness classification of 20 impossibilities, the recursive resolution paragraph, and the Duhem-Quine/Shannon/structural realism framing. The content below is the original handoff; retained for provenance.

---

*Original handoff (superseded):*

## The Current State

`paper/nature_article.tex` (780 lines) is STALE. It predates:
- Gene expression pathway divergence (TSPAN8 vs CEACAM5)
- MI v2 results (10 grokked transformers, G-invariant resolution 0.518→0.929)
- Brain imaging NARPS reanalysis (d=0.32, 16 teams for 95% stability)
- GO enrichment (zero pathway overlap for TSPAN8/CEACAM5)
- Two modes of alternation discovery
- Gene expression replication (AP_Endometrium_Colon, AP_Breast_Colon)

The current draft has the right theorem and methods sections but the Results and Discussion need rewriting around four concrete instances.

## The New Structure (from handoff-nature-universal.md)

1. **The theorem** (1 page): ExplanationSystem, Rashomon, trilemma (4-line proof), bilemma. Zero axioms.
2. **Tightness classification** (1 page): Neutral element ↔ F+S achievable. 2×2 table.
3. **Instance 1: Genomics** (1.5 pages): TSPAN8 vs CEACAM5 (80/20 alternation, ρ=0.858). GO enrichment (zero BP overlap). Replication on AP_Endometrium_Colon (78/22). COL3A1/COL1A1 (72/28, same ECM pathway = Mode 2). DASH resolves. Positive control: AP_Breast_Lung (98% stable).
4. **Instance 2: AI Safety** (1.5 pages): MI v2. 10 grokked transformers, Fourier Jaccard=0.022, component ρ=0.518. G-invariant projection (S₄×S₄) recovers ρ=0.929. Noether within heads: flip=0.500, between head/MLP: flip=0.000. MLP1 universally dominant (CV=0.027).
5. **Instance 3: Causal Inference** (0.5 page): Markov equivalence. CPDAG = neutral element resolution.
6. **Instance 4: Neuroimaging** (0.5 page): NARPS 48 teams. Orbit averaging near-optimal (within 3%). 16 teams for 95% stability. Network structure predicts disagreement (d=0.32 after activation control).
7. **Resolution** (1 page): DASH, equivalence classes, CPDAG, Pitman estimator.
8. **Discussion** (1 page): Lean verification, limitations, implications.

## Key Data Files (all in knockout-experiments/)

### Gene Expression
- `results_gene_expression_replication.json` — 4 datasets
  - AP_Colon_Kidney: TSPAN8 dominant (100% under standard params, 92% under colsample=0.5), CEACAM5 6%, ρ=0.858
  - AP_Endometrium_Colon: TSPAN8 78%, CEACAM5 22%, ρ=0.781
  - AP_Breast_Colon: COL3A1 72%, COL1A1 28% (default XGBoost, no column subsampling)
  - AP_Breast_Lung: SFTPB 98% (positive control, stable)
- `results_go_enrichment.json` — TSPAN8 (tetraspanin, integrin binding) vs CEACAM5 (cell adhesion, immune signaling), zero BP overlap. COL3A1/COL1A1 share 3 BP terms (both collagens).
- `results_gene_expression_validation.json` — SHAP sensitivity + sample overlap check

### Mechanistic Interpretability (MI v2)
- `results_mech_interp_definitive_v2.json` — 10 models, patching results
- `results_mi_v2_comprehensive.json` — Controls: step_50 ρ=0.300, step_500 ρ=0.668, step_50k ρ=0.518. post_vs_random p=0.0002, d=0.76. determinism_r=0.9998
- `results_mi_v2_final_validation.json` — G-invariant decomposition:
  - Full 10-dim: ρ=0.518
  - G-invariant 4-dim (S₄×S₄): ρ=0.929 (Pearson 0.967)
  - Excl MLP1 3-dim: ρ=0.822
  - Noether: within-group mean flip=0.500, between=0.227, gap=0.273, p=2.4e-5
  - MLP1 universally dominant (CV=0.027)

### Brain Imaging (NARPS)
- `results_brain_imaging_definitive.json` — Noether d=0.32 after activation control
- `results_brain_imaging_resolution.json` — Convergence: M_95=16 [10,22], Pareto frontier, bilemma ρ=0.962
- `results_brain_imaging_bulletproof.json` — Methods comparison, convergence CI

### Cross-Domain Multi-Analyst
- `results_multi_analyst_bulletproof.json` — Silberzahn (29 teams, real OSF data), Breznau (71 teams, real GitHub data)
- `results_noether_cross_domain.json` — Noether: 1/3 domains confirmed (NARPS only)

## Key Numbers for the Paper

| Claim | Value | Source |
|-------|-------|--------|
| Gene: TSPAN8 dominant fraction | 92% (colsample) or 100% (standard) | results_gene_expression_replication.json |
| Gene: CEACAM5 fraction | 6-8% | same |
| Gene: feature correlation | ρ = 0.858 | same |
| Gene: replication TSPAN8/CEACAM5 | 78/22 on AP_Endometrium_Colon | same |
| Gene: COL3A1/COL1A1 | 72/28 on AP_Breast_Colon | same |
| Gene: GO BP overlap (TSPAN8/CEACAM5) | 0 terms | results_go_enrichment.json |
| MI: component Spearman (raw) | 0.518 | results_mi_v2_final_validation.json |
| MI: G-invariant Spearman | 0.929 | same |
| MI: excl MLP1 Spearman | 0.822 | same |
| MI: within-layer head flip rate | 0.500 | same |
| MI: head-vs-MLP flip rate | 0.000 (between group) | same |
| MI: Noether gap | 0.273, p=2.4e-5 | same |
| MI: MLP1 CV | 0.027 | same |
| MI: Fourier Jaccard | 0.022 | same |
| MI: post_vs_random p | 0.0002, d=0.76 | results_mi_v2_comprehensive.json |
| NARPS: Noether d (controlled) | 0.32 | results_brain_imaging_definitive.json |
| NARPS: M_95 | 16 [10, 22] | results_brain_imaging_bulletproof.json |
| NARPS: orbit averaging within 3% | all methods | results_brain_imaging_resolution.json |
| η law R² | 0.957 | results_universal_eta.json |
| Noether gap | 50pp (synthetic) | results_noether_counting.json |
| Lean (universal): 100 files, 491 theorems, 25 axioms, 0 sorry | | CLAUDE.md |
| Lean (attribution): 58 files, 357 theorems, 6 axioms, 0 sorry | | dash CLAUDE.md |

## Opening Paragraph Drafts

Three options at `paper/nature_opening_drafts.md`. Recommendation: Option A (gene expression hook) for opening, B and C as second/third paragraphs. Read this file before writing.

## What Does NOT Go in the Nature Paper

- Diagnostic pipeline (minority fraction, Z-test, variance budget) → NeurIPS paper
- Ranking lottery on Breast Cancer → cite NeurIPS
- 1/(1-ρ²) GBDT ratio → too ML-specific
- SBD, split-count axioms → too technical
- Attribution PCA eigenspectrum (failed)
- Cross-domain Noether on Silberzahn/Breznau (1/3 confirmed, inconclusive)
- Ostrowski/physics material → FoP paper

## What Goes in Methods/Supplement

- Full proofs
- Quantitative bounds by model class
- All experimental details
- Extended gene expression (all 4 datasets + SHAP sensitivity)
- Extended MI analysis (per-model circuits, Fourier details)
- NARPS full progression (v1-v3, bilateral, approximate η)
- Lean theorem cross-reference
- Arrow comparison

## Constraints

- Do NOT duplicate NeurIPS paper content
- DO reference NeurIPS paper for SHAP-specific depth
- Nature main text should be ~3000 words (8-10 pages with figures)
- The monograph at `paper/universal_impossibility_monograph.tex` (4102 lines) is the comprehensive reference
- Verify all Lean counts before committing
