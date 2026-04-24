# Session Prompt: Attribution Impossibility NeurIPS Submission

**Copy everything below this line into the new session.**

---

I need to finalize the Attribution Impossibility paper for NeurIPS 2026 submission. Abstract deadline May 4, paper deadline May 6.

## Context

The paper proves that no SHAP ranking can simultaneously be faithful, stable, and complete under collinearity. It's the ML-specific companion to a universal impossibility theorem (separate Nature submission). The paper has been fully rewritten as of April 17.

## Repository

`/Users/drake.caraker/ds_projects/dash-impossibility-lean/`

## What Exists

- `paper/main.tex` — 10-page NeurIPS submission (rewritten Apr 17)
- `paper/main_definitive.tex` — 66-page definitive reference (source of truth)
- `paper/supplement.tex` — 79-page supplement
- `paper/references.bib` — 49 references
- `DASHImpossibility/*.lean` — 58 Lean files, 357 theorems, 6 axioms, 0 sorry
- 12 figures in `paper/figures/`

Key results: 68% of 77 datasets show SHAP flip prevalence. 45% of German Credit applicants receive different explanation categories. Coverage conflict diagnostic achieves Spearman 0.92–0.98, outperforming Gaussian 2× on real data. Bimodality dip p < 0.002. Full Lean verification.

## What I Need You To Do

1. **Read the handoff doc** at `/Users/drake.caraker/ds_projects/universal-explanation-impossibility/docs/handoff-attribution-neurips-final.md` for full context.

2. **Read CLAUDE.md** in the dash-impossibility-lean repo for conventions.

3. **Compile the PDF.** Fix any compilation errors. Verify it renders at 10 pages with neurips_2026.sty.

4. **Verify all numbers.** Cross-check every claim in main.tex against source data (Lean files, result JSONs, definitive paper). Run the verification commands in CLAUDE.md.

5. **Fill in the NeurIPS checklist** at the end of main.tex.

6. **Anonymization check.** Ensure no self-identifying information for blind review.

7. **Cross-reference check.** All \ref{} and \cite{} resolve. All figures referenced.

8. **Read the companion universal paper** at `/Users/drake.caraker/ds_projects/universal-explanation-impossibility/paper/nature_article.tex` to ensure consistent framing. The attribution paper should reference but not duplicate the universal theorem.

9. **Final review.** Read the complete main.tex critically. Flag any claims that aren't supported by the data, any missing references, any unclear exposition. Fix issues.

10. **Prepare submission artifacts.** Ensure main.tex + supplement.tex + figures + references.bib form a complete, compilable submission package.

## Key Constraints

- Do NOT change the Lean proofs. They are verified and stable.
- Do NOT add results from the universal repo's brain imaging experiments — those belong in the Nature paper, not this one.
- The paper must compile with neurips_2026.sty (NeurIPS 2026 official style).
- The paper is DIFFERENT from the universal impossibility paper. It focuses specifically on feature attribution (SHAP, LIME, permutation importance), not the general meta-theorem. Do not expand scope.
- Verify counts: 357 theorems, 6 axioms, 0 sorry. If these have changed, update the paper text.

## Deadline Urgency

Abstract: May 4 (15 days). Paper: May 6 (17 days). The paper is substantively complete — this is a validation and polishing session, not a content creation session. Be thorough but efficient.
