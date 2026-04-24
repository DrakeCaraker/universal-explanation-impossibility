# Handoff: Attribution Impossibility NeurIPS Paper — Final Session

**Date:** 2026-04-19
**Deadline:** Abstract May 4, Paper May 6 (15/17 days)
**Status:** Paper rewritten Apr 17. Needs final validation, compilation, and checklist.

## What This Paper Is

The Attribution Impossibility paper proves that no SHAP ranking can simultaneously be faithful, stable, and complete under collinearity. It is the **ML-specific companion** to the universal impossibility theorem (Nature track). It targets NeurIPS 2026.

**Title:** "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity"

**Authors:** Drake Caraker, Bryan Arnold, David Rhoads

## Repository

`/Users/drake.caraker/ds_projects/dash-impossibility-lean/`

Key files:
- `paper/main.tex` — NeurIPS 10-page submission (332 lines, rewritten Apr 17)
- `paper/main_definitive.tex` — 66-page definitive reference (source of truth)
- `paper/supplement.tex` — 79-page NeurIPS supplement
- `paper/references.bib` — 49 references
- `DASHImpossibility/*.lean` — 58 Lean files, 357 theorems, 6 axioms, 0 sorry

## What's Done

- [x] Complete rewrite of main.tex from definitive version (Apr 17)
- [x] All new results integrated: quantitative bilemma, compatibility complex, coverage conflict diagnostic, fairness impossibilities, bimodality validation, clinical decision reversal (45%), axiom reduction (16→6), Arrow proof
- [x] Supplement complete (79 pages)
- [x] All Lean proofs verified (357/6/0)
- [x] Figures generated (12 in paper/figures/)

## What Needs to Be Done

### Critical (before May 4 abstract deadline)

1. **Compile the PDF.** Run `pdflatex + bibtex` on main.tex. Fix any compilation errors. Verify it renders correctly at 10 pages.

2. **Verify theorem/axiom counts match the text.** Run:
   ```bash
   cd /Users/drake.caraker/ds_projects/dash-impossibility-lean
   grep -c "^theorem\|^lemma" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
   grep -c "^axiom" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
   ```
   Should give 357 theorems, 6 axioms. The paper text must match.

3. **Submit the abstract.** Extract from main.tex abstract section.

### Important (before May 6 paper deadline)

4. **NeurIPS checklist.** Fill in the mandatory "NeurIPS Paper Checklist" section at the end of main.tex. This is a required part of NeurIPS submissions.

5. **Anonymization check.** Verify neurips_2026.sty is active and the author block is hidden for review. Check that no self-identifying references remain in the paper text.

6. **Cross-reference check.** Verify all \ref{} and \cite{} resolve. Verify all 12 figures are referenced in the text.

7. **Supplement compilation.** Compile supplement.tex separately. Verify it stands alone.

### Nice to Have

8. **Update from universal repo session (Apr 19).** The brain imaging results, multi-analyst cross-domain analysis, and the "what the framework adds beyond CLT" framing from this session could inform the Discussion section. Specifically:
   - The irreducibility proof is the framework's unique contribution (not better prediction)
   - The NARPS convergence prescription (16 teams for 95% stability) demonstrates practical value
   - The honest negative (Noether doesn't generalize to scalar studies) sharpens the boundary

9. **MI v2 results.** If the mechanistic interpretability experiment finishes (10 grokked transformers, activation patching), the results could be mentioned in the Discussion as a connection to the universal framework. But this is NOT critical for NeurIPS — the attribution paper stands on its own.

## Key Results to Verify in the Paper

| Claim | Source | Value |
|-------|--------|-------|
| SHAP flip prevalence | results_prevalence.json | 68% of 77 datasets |
| Clinical decision reversal | results_clinical_decision_reversal_v2.json | 45% of German Credit applicants |
| Coverage conflict Spearman | empirical-validation-results.md (ostrowski) | 0.92–0.98 across 5 datasets |
| Gaussian formula Spearman | same | 0.46–0.89 on real data |
| Bimodality dip test | same | p < 0.002 |
| NN SHAP instability | results in dash repo | 87% unstable |
| Lean verification | DASHImpossibility/*.lean | 357 theorems, 6 axioms, 0 sorry |
| DASH overhead | paper text | O(M) cost |

## Relationship to Other Papers

| Paper | Venue | Relationship |
|-------|-------|-------------|
| Universal impossibility (Nature) | Nature | This paper is an ML-specific instance; references the universal theorem |
| Universal monograph (arXiv) | arXiv | Contains the full technical detail; this paper extracts the attribution-specific content |
| Physics companion (arXiv) | arXiv (hep-th) | Independent; connects to spacetime geometry via Ostrowski |

**Do NOT dual-submit** the attribution paper and the universal paper to the same venue. They share substantial content. The attribution paper goes to NeurIPS; the universal paper goes to Nature.

## The Pitch (for the abstract)

Feature importance explanations are provably unreliable under collinearity. We prove that no SHAP ranking can simultaneously be faithful, stable, and complete (the Attribution Impossibility), characterize the complete design space (exactly two achievable families), and provide a nonparametric diagnostic that identifies unstable features in seven lines of code, outperforming the Gaussian formula by 2× on real data. A per-individual analysis shows 45% of loan applicants receive different explanation categories under standard model regularization. The entire framework is verified in Lean 4 (357 theorems, 6 axioms, 0 sorry), and the constructive resolution (DASH ensemble averaging) is proved Pareto-optimal.
