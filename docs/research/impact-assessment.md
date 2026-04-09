# Impact Assessment and Submission Strategy

> Honest evaluation from March 30, 2026 session. Vetted across 3 rounds.
> All ratings are calibrated — not inflated or deflated.

## Assessment Summary

**The work:** First formally verified impossibility theorem for feature
attribution. Core theorem has zero axiom dependencies. Quantitative bounds
for 4 model classes. Constructive resolution (DASH). 13 Lean files, 12
axioms, 28 theorems, 0 sorry.

**Honest placement:** Strong contribution, potentially foundational if
the research program continues. The NeurIPS paper alone is a strong
contribution. The full program (inevitability + design space + continued
formalization) could become the reference framework for attribution theory.

**NeurIPS acceptance estimate:** 25-40%. Above base rate (~20%) due to
genuine novelty, but facing headwinds from independent researcher status,
axiomatization gap risk, and "short core proof" perception.

---

## 1. Career Impact

### What it opens
- Collaboration invitations from established XAI researchers
- Workshop/seminar speaking invitations
- Strong PhD application profile (if desired)
- Industry research positions valuing publications
- Credibility for future submissions

### What it doesn't
- One paper ≠ a career (strong START, not an arrival)
- No institutional halo (must prove yourself repeatedly)
- Doesn't substitute for a PhD for tenure-track positions
- Doesn't generate income directly

### The "independent researcher" factor
- MORE impressive (no lab/advisor/resources)
- LESS credibility buffer (reviewers may be unconsciously skeptical)
- Double-blind helps, but GitHub URL reveals identity (MUST anonymize)

## 2. Reputation in the Field

### XAI community
- Will be noticed: impossibility results generate discussion, SHAP
  connection means every data scientist has a stake
- The constructive resolution (DASH) increases practitioner engagement
- The fairness impossibility parallel is a strong positioning move

### Formal methods community
- Curiosity, not deep engagement (unless novel Lean techniques used)
- Value is in the ML community, not the Lean community

### "First machine-verified XAI impossibility"
- Memorable NOW, ages fast (someone will be "second")
- Lasting value depends on the RESULT, not the verification method

## 3. Contribution to the Field

### Honest decomposition

| Component | Novelty | Depth | Impact |
|-----------|---------|-------|--------|
| Core impossibility (Rashomon → ⊥) | Framework new | Technically shallow (4-line proof) | HIGH (conceptually important) |
| Quantitative bounds (1/(1-ρ²)) | New formulation | Moderate (algebra from axioms) | MEDIUM |
| Multi-model coverage | New unified framework | Moderate (IterativeOptimizer) | HIGH |
| Lean formalization | First in XAI | High effort, moderate depth | MEDIUM-HIGH |
| DASH resolution | Method not novel (ensemble avg) | Formal equity proof is new | HIGH (practitioners need this) |
| Design space characterization | Novel (no prior work) | Moderate structural insight | HIGH (potentially foundational) |

### Realistic citation trajectory (if NeurIPS accepted)
- 2 years: 20-60 citations (wide range; inherently unpredictable)
- The DASH method paper may get more practitioner citations than the theory
- Comparison: Bilodeau et al. (PNAS 2024) ~130 in 2 years (stronger authors, broader venue)

### Does it change thinking?
- Practitioners: validates those using ensemble SHAP; warns single-model SHAP users
- Researchers: establishes attribution stability as a THEORETICAL limit, not engineering challenge
- Policy: proves a "known limitation" that EU AI Act requires disclosing

## 4. Risks and Weaknesses

### Reviewer attack vectors (ranked by likelihood)

**#1: Axiomatization gap (HIGH risk)**
Theory predicts 5.3× ratio at ρ=0.9; empirical shows 1.9×. A reviewer
could write: "The axioms overestimate by 2-5×. If wrong, the impossibility
is about a model that doesn't exist."

Defense: qualitative prediction holds; axioms capture leading-order behavior;
the impossibility depends on structure (Rashomon + desiderata), not constants.

**#2: Core theorem is technically shallow (MEDIUM risk)**
4-line proof. Some reviewers conflate "short" with "limited contribution."

Defense: many landmark theorems have short proofs (Arrow, Chouldechova).
The contribution is the framework, not the proof length. Paper includes
bounds, formalization, resolution — substantial total contribution.

**#3: Limited novelty vs. Bilodeau (MEDIUM risk)**
"Another impossibility for feature attribution."

Defense: different desiderata (stability vs. linearity), architecture
discrimination, constructive resolution, formal verification.

**#4: Lean formalization is a translation, not a contribution (LOW-MEDIUM)**
"Same proofs in a different language."

Defense: caught 2 genuine inconsistencies; first use of formal methods in XAI.

**#5: GitHub URL reveals authorship (procedural risk)**
MUST anonymize for double-blind review.

### AI assistance
- Check NeurIPS 2026 AI disclosure policy before submission
- AI helped with execution (Lean tactics, LaTeX), not mathematical ideas
- Perception may not match reality — some will assume "AI did everything"

## 5. Comparison to Landmark Results

### vs. Chouldechova/Kleinberg (fairness impossibility)
- Structurally identical (trilemma of incompatible desiderata)
- Their timing was perfect (fairness was exploding); ours is good (XAI maturing, regulation rising)
- Their surprise factor was higher (people believed "fair" was achievable)
- Realistic impact: 15-30% of their citation trajectory
- Not a criticism of the work — commentary on how academic impact works

### vs. Bilodeau et al. (PNAS 2024)
- Our theoretical result is at least as strong (more desiderata, more models, resolution, formalization)
- Their institutional backing (Been Kim/Google Brain, Pang Wei Koh/Stanford) provides credibility buffer we lack
- Our advantages: formalization, constructive resolution, quantitative architecture discrimination

### Spectrum placement
Incremental → Solid → **Strong** → Foundational
The paper is at "strong," with potential for "foundational" if the program continues.

## 6. What Would Make This Foundational with Immediate Recognition

### The gap between "strong" and "foundational"

Foundational results share three properties:
1. **Inevitability** — the conditions are nearly inescapable
2. **Characterization** — the full landscape of alternatives is mapped
3. **Practical transformation** — practitioners change behavior

Current status:
- Inevitability: PARTIAL (R3 in supplement; not yet a standalone theorem)
- Characterization: PARTIAL (F1-F3 in supplement; not quantitative)
- Practical transformation: UNCLEAR (DASH exists but adoption unmeasured)

### What would close the gap (ordered by impact per effort)

**A. Strengthen the empirical validation (HIGH impact, 1-2 days)**

The 2-5× axiom-empirical discrepancy is the paper's biggest vulnerability.
Running experiments that show the QUALITATIVE predictions hold across
many configurations (depths, learning rates, datasets) would blunt the
"axioms are wrong" attack. Specifically:

1. Show the ratio > 1 across ALL configurations (not just stumps)
2. Show the ratio INCREASES with ρ across all configurations
3. Show RF has ratio → 1 (contrast case confirmed)
4. Show DASH consensus converges across all configurations

Frame as: "The axioms capture the mechanism; the constants are leading-order.
The impossibility is structural, not quantitative."

**B. Close the GitHub anonymity gap (CRITICAL, 30 minutes)**

Create an anonymous mirror of the repo for submission. The current URL
reveals authorship.

Options:
- Anonymous GitHub account with the repo forked
- Remove the URL from the paper entirely (add at camera-ready)
- Use a placeholder: "Code available at [anonymized]"

**C. Add a "robustness" experiment showing the impossibility is real (MEDIUM, 1 day)**

The most CONVINCING experiment would be:
- Train 50 XGBoost models on the same data with different seeds
- Show that SHAP rankings of within-group features flip across seeds
- Show this DOESN'T happen for between-group features
- Show DASH(M=25) eliminates the flipping

This is the "this is real, not just theory" experiment. The current
experiments show it but the framing could be sharper.

**D. Directly address the axiom gap in the paper (MEDIUM, 2 hours)**

Add a paragraph to Section 4 or the Discussion:

"The axiomatized split counts represent leading-order behavior under
idealized greedy splitting. Real gradient boosting with finite depth
and learning rate damping exhibits lower ratios (Figure 1), consistent
with the axioms being an upper bound. Crucially, the qualitative
impossibility — that the first-mover accumulates disproportionate
attribution — holds across all tested configurations. The impossibility
is a property of the sequential fitting architecture, not of the
specific constants."

**E. Sharpen the contribution framing (LOW effort, HIGH impact)**

The paper buries the lead slightly. The STRONGEST claims are:
1. The core theorem requires ZERO axioms (just Rashomon)
2. Rashomon is INEVITABLE for symmetric algorithms (R3 in supplement)
3. The design space is ONE-DIMENSIONAL (M-axis, in supplement)

These are in the supplement, not the main text. If any could be promoted
to main text (even as a one-line reference), it would strengthen the paper.

The current "Generality" paragraph does this but could be sharper:
"The impossibility is inescapable: in the supplement we prove that any
stochastic, symmetric training algorithm on collinear features necessarily
satisfies the Rashomon property (Theorem S5)."

### What would make it foundational with IMMEDIATE recognition

To go from "strong contribution that COULD become foundational" to
"immediately recognized as foundational," you would need ALL of:

1. **The inevitability result IN the main paper** (not supplement)
   — R3 as a main-text theorem, not a supplement remark
   — This transforms "holds for GBDT/Lasso/NN" into "holds for everything"
   — Currently blocked by page limit (9 pages exactly)

2. **The design space characterization IN the main paper** (not supplement)
   — The trilemma → M-axis reduction as a main-text theorem
   — Currently blocked by page limit

3. **Empirical validation that MATCHES theory**
   — The 2-5× discrepancy undermines the quantitative claims
   — Would need experiments designed to match the axiom conditions
     (stumps, η=1, no regularization) to show close agreement
   — OR reframe the axioms as bounds rather than equalities

4. **Institutional endorsement**
   — A co-author or endorser from an established lab
   — Not achievable in 5 weeks, but worth pursuing post-submission

5. **A clear "this is the Chouldechova of XAI" narrative**
   — The paper draws the parallel but doesn't lean into it
   — The introduction could be restructured to lead with the analogy

The honest truth: items 1-2 are blocked by the page limit. Item 3
requires either new experiments or reframing. Item 4 is not in your
control. Item 5 is achievable but risky (reviewers might think you're
overclaiming by comparing to a landmark).

**The most realistic path to "foundational":** Submit the strong NeurIPS
paper as-is. If accepted, the visibility + the supplement's R3 + F1-F3
establish the foundations. Then submit Paper B (full characterization) to
JMLR as the comprehensive reference. The PAIR of papers (NeurIPS + JMLR)
together constitute the foundational contribution. No single 9-page paper
can be foundational — but a research program can.

---

## Immediate Action Items (Before May 4)

| Item | Priority | Effort | Impact |
|------|----------|--------|--------|
| Anonymize GitHub URL in paper | CRITICAL | 30 min | Prevents disqualification |
| Check NeurIPS 2026 AI disclosure policy | CRITICAL | 15 min | Compliance |
| Have co-authors review and approve | CRITICAL | Days | Required |
| Sharpen "Generality" paragraph | HIGH | 30 min | Stronger universality claim |
| Add axiom-gap defense paragraph | MEDIUM | 1 hour | Preempts #1 reviewer attack |
| Run robustness experiments (ratio > 1 across configs) | MEDIUM | 1 day | Blunts empirical discrepancy |
| Post arXiv preprint | HIGH | May 4 | Priority + regulatory window |

---

## Long-Term Research Identity

### "Formal methods + XAI" intersection
- Virtually unoccupied — you could own it
- Lean + XAI is distinctive and hard to replicate
- The intersection will grow as AI regulation demands provable properties

### Building the program
1. NeurIPS 2026: the impossibility (this paper)
2. TMLR: the method (Paper 1 / DASH)
3. JMLR: the characterization (Paper B)
4. Future: applications (clinical ML, finance — domain-specific impossibility)
5. Future: extended formalization (R1 in Lean, full characterization in Lean)

### Financial sustainability
Independent research is not financially self-sustaining. Options:
- AI safety organizations (value formal verification + XAI expertise)
- XAI consulting (growing market given EU AI Act)
- Industry research positions (FAANG, AI labs)
- A NeurIPS publication + Lean expertise is a strong profile for all of these
