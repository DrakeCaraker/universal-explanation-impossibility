# Research Roadmap — The Explanation Impossibility Program

Last updated: 2026-04-11

This document consolidates the multi-paper research program around the
Universal Explanation Impossibility theorem. It covers venue strategy,
paper sequencing, and the full arc from the current submissions through
the cross-science follow-up.

---

## Paper inventory

| # | Paper | Status | Target venue | Format |
|---|-------|--------|-------------|--------|
| 1 | **Universal Impossibility** ("The Limits of Explanation") | Draft complete, 5 formats | JMLR (primary) | 30+ pages |
| 2 | **Attribution Impossibility** (companion) | Draft complete | NeurIPS 2026 (10pp) + JMLR (54pp) | 10pp / 54pp |
| 3 | **Cross-Science Impossibility** ("One Theorem, Twelve Names") | Draft started | Science / Nature / PNAS | ~4000-5000 words |
| 4 | **DASH Resolution** (F5→F1→DASH stability API) | In progress (dash-shap PR #255) | With companion paper | Code + API |

---

## Venue strategy

### Paper 1: Universal Impossibility — JMLR (primary)

**Why JMLR:**
- No page limit — the full 30+ page treatment with 9 instances, Lean
  formalization details, tightness proofs, and design space analysis fits
  naturally.
- Strong theory track. JMLR publishes foundational results (e.g., Vapnik's
  SVM papers, PAC-Bayes bounds).
- Open access — maximizes citation potential.
- No conference deadline pressure — rolling submissions.

**Backup venues (in order):**
1. **JAIR** (Journal of AI Research) — open access, broad AI scope, accepts
   long papers. Slightly less prestige than JMLR for theory.
2. **Mathematics of Operations Research** — if framed as a decision-theoretic
   impossibility. Unusual choice but would reach optimization/OR community.
3. **Foundations and Trends in ML** — survey/monograph format. Good if
   reviewers want even more context. Very high impact per paper.

**NOT recommended:**
- NeurIPS/ICML (too long for 10-page format; core result gets compressed)
- AAAI (lower theory prestige)
- Nature Machine Intelligence (possible but editorial risk — they may want
  more empirical content)

**Timeline:** Submit when JMLR formatted version is finalized. Rolling review,
expect 3-6 months for first decision.

### Paper 2: Attribution Impossibility — NeurIPS 2026 + JMLR

**NeurIPS 2026 timeline:**
- Abstract: May 4, 2026
- Paper: May 6, 2026
- 10-page main paper + supplement (79 pages)

**JMLR version:** 54-page extended version, submitted after/alongside NeurIPS.
The NeurIPS paper is the "advertisement"; the JMLR paper is the "complete
treatment." This dual-submission strategy is standard and acceptable as long
as the JMLR version is substantially expanded.

### Paper 3: Cross-Science Impossibility — Science / Nature / PNAS

This is the high-visibility paper that reaches beyond ML. It shows the
theorem unifies 12+ fields' independently discovered phenomena.

**Venue ranking:**

1. **Science** (Research Article, ~4500 words)
   - **Why:** Highest visibility. The "12 fields discovered the same theorem"
     narrative is exactly the kind of cross-disciplinary unification Science
     values. Comparable precedents: Arrow's original work was eventually
     recognized with a Nobel; Noether's theorem is celebrated precisely
     because it unifies across physics.
   - **Risk:** Science may want experimental validation in each domain, not
     just formal instantiation. The ML experiments (from Paper 1) help but
     physics/chemistry/biology instances are purely theoretical.
   - **Format:** Research Article (~4500 words, 5-6 figures) or Report
     (~2500 words, 3-4 figures).

2. **Nature** (Article, ~3000-5000 words)
   - **Why:** Similar visibility. Nature publishes cross-disciplinary theory
     when the unification is striking enough.
   - **Risk:** Nature tends to prefer empirical results. A purely theoretical
     cross-science paper is a harder sell than at Science.
   - **Format:** Article.

3. **PNAS** (Research Article, ~6 pages)
   - **Why:** Strong for mathematically rigorous cross-disciplinary work.
     PNAS has published Arrow-type impossibility results and has a
     "Mathematics" + "Applied Mathematics" classification that fits.
     The contributed/communicated track gives more editorial control.
   - **Risk:** Slightly lower visibility than Science/Nature.
   - **Format:** Research Article. Already have a PNAS-formatted version
     of Paper 1 that can be adapted.
   - **BEST BET for first submission.** PNAS is more receptive to formal
     results without heavy empirical validation, and the dual classification
     (Computer Sciences + Applied Mathematics) fits perfectly.

4. **Proceedings of the Royal Society A** (Mathematical, Physical and
   Engineering Sciences)
   - **Why:** Publishes mathematical results with cross-disciplinary impact.
     The gauge invariance connection makes it natural.
   - **Format:** Research Article, no strict page limit.

5. **Philosophical Transactions of the Royal Society A** (theme issues)
   - If there's a relevant theme issue on foundations of science, explanation,
     or interdisciplinary methodology.

**Recommended strategy:** Submit to **PNAS first** (highest acceptance
probability for a formal cross-disciplinary result). If rejected, revise
for **Science**. If rejected, submit to **Proc. Roy. Soc. A**.

**Timeline:** Draft cross-science paper Q2-Q3 2026. Submit Q3-Q4 2026,
after Paper 1 is accepted or posted to arXiv (so you can cite it).

### Additional venue opportunities

**Philosophy journals** (for the Quine-Duhem formalization specifically):
- **Philosophy of Science** — flagship journal. A short paper formalizing
  Quine-Duhem as an impossibility theorem, pitched to philosophers.
- **British Journal for the Philosophy of Science** — strong for formal
  philosophy of science.
- **Synthese** — broad, publishes formal epistemology.

**Statistics journals** (for the partial identification connection):
- **Statistical Science** — survey/review articles connecting statistics
  to other fields.
- **Annals of Statistics** — if you develop the estimation-theoretic
  consequences (Cramer-Rao type bounds for explanation).

**Physics journals** (for the gauge invariance connection):
- **Reviews of Modern Physics** — if you write a pedagogical treatment
  connecting gauge theory to explanation theory.
- **Foundations of Physics** — publishes on interpretational questions.

**Psychology journals** (for factor indeterminacy):
- **Psychometrika** — the factor analysis community's flagship.
- **Psychological Methods** — methodological implications.

**These are optional satellite papers**, not required. The main three
papers (Universal, Attribution, Cross-Science) are the core program.

---

## Sequencing and dependencies

```
Q2 2026:  Paper 2 (Attribution) → NeurIPS abstract May 4, paper May 6
          Paper 1 (Universal) → arXiv preprint + JMLR submission
          Paper 3 (Cross-Science) → Draft and circulate

Q3 2026:  Paper 1 → JMLR review
          Paper 2 → NeurIPS review
          Paper 3 → Revise based on feedback, submit to PNAS

Q4 2026:  Paper 2 → NeurIPS decision (September)
          Paper 1 → JMLR revision (if needed)
          Paper 3 → PNAS review

Q1 2027:  Paper 1 → JMLR acceptance (target)
          Paper 3 → PNAS decision / revision
          Optional: philosophy, statistics, or physics satellite papers

Q2 2027+: Paper 3 revision cycle (Science/Nature if PNAS declines)
           Conference talks, invited lectures
           Lean formalization tutorial / tool paper (if demand exists)
```

### Key dependency: arXiv timing

Post Paper 1 to arXiv **before** submitting Paper 3. The cross-science
paper cites Paper 1 as the source of the core theorem. Having it on arXiv
(with the Lean formalization publicly available) gives reviewers something
to verify and builds credibility.

---

## Research extensions (future work)

### Near-term (can be done with current framework)

1. **Derive Rashomon from loss landscape geometry.** For neural networks,
   prove that the Rashomon property follows from overparameterization +
   non-convex loss landscape. This would convert the neural net instance
   from "axiomatized" to "derived." Uses results from Choromanska et al.
   (2015) on loss surface structure.

2. **Quantitative bounds for non-attribution instances.** Paper 1 has
   sharp quantitative bounds for attribution (1/(1-ρ²) divergence, DASH
   O(1/√T) convergence). Develop analogous bounds for:
   - Attention (flip rate as function of weight perturbation magnitude)
   - Counterfactuals (decision boundary sensitivity)
   - Causal discovery (number of Markov-equivalent DAGs as function of
     graph size — known results from Gillispie & Perlman 2002)

3. **Approximate impossibility.** The current theorem is exact (all-or-nothing).
   Develop an ε-approximate version: if the Rashomon property holds with
   "strength" δ, then any faithful+stable explanation must sacrifice at
   least f(δ) decisiveness. This would give a quantitative tradeoff curve
   rather than a binary impossibility.

4. **Computational complexity of the resolution.** The G-invariant resolution
   requires averaging over the equivalence class. How hard is this
   computationally? For attributions (DASH), it's polynomial (ensemble
   averaging). For causal discovery (CPDAG), it's polynomial (Chickering
   2002). For general ExplanationSystems, is there a complexity-theoretic
   characterization?

### Medium-term (requires new mathematical development)

5. **Category-theoretic formulation.** The ExplanationSystem is a functor
   from configurations to explanations, with a natural transformation
   (the observation map) creating the fiber structure. A category-theoretic
   formulation could connect to Galois theory (the symmetry group acting
   on fibers is reminiscent of Galois groups acting on roots).

6. **Information-theoretic formulation.** Quantify the "explanation gap" as
   mutual information: I(Θ; H | Y). The impossibility says this mutual
   information is positive when Rashomon holds. The G-invariant resolution
   minimizes it. This could connect to rate-distortion theory.

7. **Dynamic impossibility.** The current theorem is static (one system, one
   explanation). In practice, systems evolve (models are retrained, theories
   are updated). Develop a dynamic version: how does the impossibility
   interact with online learning, sequential model updating, or scientific
   theory revision?

### Long-term (aspirational)

8. **A mathematical theory of explanation.** Analogous to how Shannon's
   theorem founded information theory, develop a general mathematical
   framework for explanation under uncertainty. The ExplanationSystem
   structure is the seed; the full theory would include capacity theorems,
   coding theorems (optimal resolution strategies), and converse theorems
   (impossibility bounds).

9. **Formal Lean library.** Package the Lean formalization as a reusable
   Mathlib-compatible library for reasoning about explanation systems.
   Other researchers could prove new instances by instantiating the
   ExplanationSystem structure.

---

## File index

| File | Purpose |
|------|---------|
| `paper/universal_impossibility.tex` | Paper 1: Universal impossibility (primary) |
| `paper/universal_impossibility_jmlr.tex` | Paper 1: JMLR format |
| `paper/universal_impossibility_neurips.tex` | Paper 1: NeurIPS format |
| `paper/universal_impossibility_pnas.tex` | Paper 1: PNAS format |
| `paper/universal_impossibility_monograph.tex` | Paper 1: Full monograph |
| `paper/main.tex` | Paper 2: Attribution (NeurIPS 10pp) |
| `paper/main_jmlr.tex` | Paper 2: Attribution (JMLR 54pp) |
| `paper/supplement.tex` | Paper 2: Supplement (79pp) |
| `paper/cross_science_impossibility_draft.tex` | Paper 3: Cross-science draft |
| `paper/BEYOND_ML_INSTANCES.md` | Reference catalog of all instances |
| `RESEARCH_ROADMAP.md` | This file |
| `UniversalImpossibility/` | Lean 4 formalization (75 files) |
