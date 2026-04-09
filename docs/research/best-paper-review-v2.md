# Best Paper Review v2: The Attribution Impossibility (Post-Phases A-F)

> Five-reviewer re-assessment of the CURRENT paper (10pp main + 25pp supplement).
> Compared against v1 review (scored 7.4/10 on the pre-Phase-A paper).
> Cross-checks: Lean verified (29 theorems, 15 axioms, 0 sorry). Supplement builds (25pp).

---

## Reviewer 1: Fields Medalist

### What changed since v1
The supplement now contains: (1) a rigorous FIM impossibility theorem with explicit regularity conditions and epsilon bounds (§F3), (2) DASH Pareto optimality with Cramér-Rao efficiency proof and median/trimmed-mean comparison (§F2), (3) Berry-Esseen bounds for the testable condition (§F1), (4) exchangeability lemmas for the split-frequency diagnostic (§F5), and (5) **the Attribution Design Space Theorem** — the single unified theorem I asked for.

### Strengths (new)
1. **The Design Space Theorem (supplement §14) is exactly what I asked for.** Two families (A: faithful-complete, B: ensemble), Pareto frontier, 5 corollaries subsuming all previous results. The design space collapses to a single M-axis. This is clean structural mathematics.
2. **F3 is now rigorous.** Explicit regularity conditions (R1-R4), the epsilon bound (eq. 6) ensuring the cubic remainder is controlled, and the Gaussian specialization showing K₃=0 → the result holds for ALL ε. This is how analysis should be done.
3. **F2 proves DASH is MVUE**, not just "good." The Rao-Blackwell argument and the median comparison (ARE = 2/π ≈ 0.637) give practitioners a precise cost for choosing alternatives.

### Weaknesses (remaining)

**W1 (MEDIUM): The Design Space Theorem is in the supplement, not the main text.**
A reviewer reading only the 10-page main text sees the impossibility + bounds + DASH + a one-paragraph F1 summary. The most important result (the Design Space Theorem) requires reading page 15+ of the supplement. For a conference paper this is acceptable — supplements are reviewed — but it reduces impact for casual readers.

*Fix for JMLR:* Make the Design Space Theorem the centerpiece of the main text.

**W2 (MINOR): The Design Space Theorem Step 3 has a gap.**
The claim "no method lies outside A ∪ B" considers three cases but doesn't cover methods that compute attributions from the combined data of all models (not aggregating per-model attributions). For example, a method that trains one model on the concatenated datasets of M training runs would not fit either case.

*Fix:* Add the qualifier "among methods that operate on per-model attributions" or prove the result extends.

**W3 (MINOR): The core theorem is still 4 lines.**
I said this was not a reason for rejection in v1, and it still isn't. The reframing ("The contribution is the complete framework") is honest and effective. The "simplicity is a feature" sentence (main text §3.3) is the right response.

### THE ONE THING for the final +1 point
Formalize the Design Space Theorem in Lean. Currently it's proved on paper in the supplement but not machine-verified. A Lean proof of the Design Space Theorem — with Theorem 1, F1, F2, path convergence as Lean corollaries — would make this paper unique in the formal methods world.

### Scores
Soundness: 4/4 | Contribution: 4/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 8.5/10** | Confidence: 5/5 | **Strong Accept**

*Change from v1: 7 → 8.5 (+1.5). The Design Space Theorem is exactly what I asked for (+2), minus 0.5 because it's in the supplement not the main text.*

---

## Reviewer 2: Chief Scientist, AI Safety Organization

### What changed since v1
The paper now acknowledges the information-stability tradeoff explicitly (Discussion §8), recommending dual reporting (single-model SHAP + DASH consensus). The broader implications paragraph covers fairness audits, model debugging, and scientific discovery. The supplement has 11-dataset validation including credit scoring data.

### Strengths (new)
1. **The dual-reporting recommendation** (main text §8) directly answers my question: "wrong but specific, or correct but vague?" The answer: report both, with appropriate caveats. This is the responsible answer.
2. **The feature selection stability note** (main text §5) addresses the top-K selection problem: between-group selection is stable, within-group features are interchangeable.
3. **11-dataset validation** (supplement) shows the diagnostic works across domains including credit scoring (Credit-g, r=-0.90).

### Weaknesses (remaining)

**W1 (MEDIUM): The information loss is still not quantified.**
My "+2 thing" was a formal analysis of bits lost by averaging. The paper has a qualitative paragraph but no mutual information calculation. How many bits of attribution information are lost when going from M=1 to M=25? This matters for safety-critical systems where every bit of model explanation counts.

**W2 (MINOR): The regulatory response for ties (supplement) is excellent but not in the main text.**
The template "These features are statistically indistinguishable..." is exactly what a compliance officer needs. It should be more prominent.

### THE ONE THING for the final +1 point
Same as v1: quantify the information loss. Compute I(single-model ranking; true DGP) vs I(DASH ranking; true DGP) as a function of M and ρ.

### Scores
Soundness: 4/4 | Contribution: 4/4 | Clarity: 4/4 | Experiments: 4/4
**Overall: 8.5/10** | Confidence: 4/5 | **Strong Accept**

*Change from v1: 8 → 8.5 (+0.5). Experiments improved from 3→4 (11 datasets). Information loss still qualitative, preventing the full +2.*

---

## Reviewer 3: Editor-in-Chief, JMLR

### What changed since v1
The supplement has grown from ~13 pages to 25 pages and now constitutes a near-complete journal paper. F1-F5 are fully proved. The Design Space Theorem unifies everything. 11 datasets validated. The F1 diagnostic is promoted to the main text.

### Strengths (new)
1. **The supplement is now at JMLR quality.** Five rigorous theorems (F3, F2, F1, F5, Design Space), each with explicit assumptions, complete proofs, and error bounds. The Berry-Esseen bound (F1), the exchangeability lemmas (F5), and the Gaussian FIM specialization (F3) are all publication-ready.
2. **F1 promoted to main text** — exactly what I asked for. The one-paragraph summary (main §6) gives a reader the headline (r=-0.89, robust at r=-0.78 for Z<5) without needing the supplement.
3. **11-dataset table** (supplement) replaces the 2-dataset table. Shows F1 works across medical, financial, housing, and image domains. The honest reporting of weak correlations for Digits (r=-0.42) and Adult (r=-0.29) with the floor-effect explanation is exactly the transparency I expect.
4. **The comprehensive F1 figure** (supplement) showing 4 datasets side-by-side is compelling.

### Weaknesses (remaining)

**W1 (MINOR): The supplement axiom table (Table 1) is stale.**
It still says "12 axioms total" and lists only 6 property axioms. The Lean code now has 15 axioms (including the variance axioms). The proof architecture verbatim listing (§2) also doesn't reflect the variance additions to Corollary.lean.

**W2 (MINOR): For a JMLR submission, I'd want the Design Space Theorem in the main body, not the supplement.**
As a conference supplement, 25 pages is fine. As a journal paper, the structure should be reorganized with the Design Space Theorem as the main result.

### THE ONE THING for +0.5 points
Fix the stale axiom table and proof architecture to reflect the current 15-axiom, 29-theorem state. Consistency between the code and the paper is essential.

### Scores
Soundness: 4/4 | Contribution: 4/4 | Clarity: 4/4 | Experiments: 4/4
**Overall: 9/10** | Confidence: 5/5 | **Strong Accept (Champion)**

*Change from v1: 8 → 9 (+1). F1-F5 publication-ready (+0.5), 11 datasets (+0.5), F1 promoted to main text (his +2 thing → +1 since it's a paragraph not a full section). Would champion for acceptance.*

---

## Reviewer 4: VP of ML, Major Bank

### What changed since v1
Feature selection stability is now addressed. Retrospective DASH is mentioned. Credit-g dataset included. The testable condition is in the main text with the F5 practitioner algorithm in the supplement.

### Strengths (new)
1. **Feature selection stability** (main text §5): "within-group features reported as tied should be documented as interchangeable." This is the answer I need for our OCC examiners.
2. **Retrospective DASH** (main text §5): "can be applied retrospectively to existing model inventories." Critical for my team — we don't want to retrain 200 models from scratch.
3. **Credit-g validation** (supplement, r=-0.90): This is credit scoring data. The diagnostic works. I can cite this.

### Weaknesses (remaining)

**W1 (MEDIUM): Still no FICO or Lending Club case study.**
Credit-g is credit scoring but it's a European dataset from the 1990s with 20 features. My models have 200+ features, income/DTI/credit-score collinearity patterns, and SR 11-7 compliance requirements. A modern financial case study would make this directly citable in regulatory submissions.

**W2 (MINOR): The F5 practitioner algorithm assumes the analyst knows which features are "in correlated groups."**
In practice, identifying groups requires computing the correlation matrix and thresholding. The paper should mention this preprocessing step, or suggest a threshold (e.g., |ρ| > 0.5).

### THE ONE THING for +1 point
A Lending Club or FICO case study showing the full F5 → F1 → DASH workflow on a realistic credit model. This makes the paper directly citable in model validation reports.

### Scores
Soundness: 3/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7.5/10** | Confidence: 4/5 | **Accept**

*Change from v1: 7 → 7.5 (+0.5). Feature selection addressed (+0.25), Credit-g data (+0.25). Still wants FICO case study. Scores unchanged for soundness/contribution because his concerns are about domain relevance, not mathematical correctness.*

---

## Reviewer 5: Adversarial Red-Teamer

### What changed since v1
The "28+1" ambiguity is resolved (now "29 substantive theorems, 0 sorry"). The F1 restricted-range analysis confirms r=-0.78 for Z<5. The gain-based importance check (r=-0.83) rules out SHAP artifacts.

### Strengths (new)
1. **The restricted-range analysis is exactly what I asked for** (+2 thing). r=-0.78 for Z<5, r=-0.61 for Z<3. The headline r=-0.89 IS partially inflated by easy pairs, but the diagnostic still works in the interesting range. Honest reporting.
2. **Gain-based importance** (r=-0.83) rules out my "SHAP computation artifact" attack. The result holds for non-SHAP attributions.
3. **29 theorems, 0 True placeholders.** The variance bound is now axiomatized with real derived theorems. Clean.

### Weaknesses (remaining)

**W1 (MEDIUM): The variance bound is axiomatized, not proved.**
Three of the 29 "substantive theorems" derive from `consensus_variance_bound`, which is an axiom encoding Var(X̄) = Var(X)/M. This is a standard probability result, but it's still an AXIOM in the Lean code. A hostile reader could write: "The authors added an axiom for a textbook identity and count the consequences as theorems."

*Defense (which I expect they'd give):* The axiom captures a standard probability result. Connecting our abstract Model type to Mathlib's measure theory requires infrastructure that doesn't yet exist. The derived theorems (halving, nonnegativity) are genuine Lean proofs, not tautologies.

*My assessment:* This is a legitimate criticism but not a serious one. The total axiom count (15) is transparent and each axiom is justified.

**W2 (MEDIUM): The 11-dataset validation may include incorrectly loaded datasets.**
The comprehensive_validation.py uses OpenML IDs that may resolve to different datasets (e.g., "Adult Income" at id=2 loads steel plates, not the actual Adult/Census dataset). If the loaded data isn't what's claimed, the results are misleading.

*Fix:* Verify every dataset loaded correctly by checking feature names and sample counts. Fix incorrect OpenML IDs.

**W3 (MINOR): α=2/π is still a proof sketch.**
Proposition S1 proves variance capture for the optimal binary quantizer, but the connection to "each boosting stump captures 2/π" still relies on stated but unproved assumptions. Unchanged from v1.

### THE ONE THING for +0.5 points
Verify the comprehensive_validation.py dataset IDs. If "Adult Income" is actually steel plates, fix it and re-run. The 11-dataset claim must be honest.

### Scores
Soundness: 3.5/4 | Contribution: 3.5/4 | Clarity: 4/4 | Experiments: 3.5/4
**Overall: 8/10** | Confidence: 4/5 | **Strong Accept**

*Change from v1: 7 → 8 (+1). F1 restricted-range analysis (+0.5, his +2 thing). Gain-based robustness (+0.25). 29 theorems with no placeholder (+0.25). Soundness up 0.5 (restricted-range confirms claim). Contribution up 0.5 (Design Space Theorem exists, even in supplement).*

---

## Area Chair Meta-Review

### Score Table

| | R1 (Fields) | R2 (Safety) | R3 (JMLR) | R4 (Bank) | R5 (Red) | Mean |
|---|---|---|---|---|---|---|
| **v1** | 7 | 8 | 8 | 7 | 7 | **7.4** |
| **v2** | **8.5** | **8.5** | **9** | **7.5** | **8** | **8.3** |
| **Δ** | +1.5 | +0.5 | +1 | +0.5 | +1 | **+0.9** |

### What drove the improvement

1. **Design Space Theorem** (R1: +1.5, R5: +0.25): The single biggest driver. R1 explicitly asked for this and it was delivered.
2. **Rigorous F1-F5** (R3: +0.5, R1: included above): Publication-ready proofs with error bounds.
3. **11-dataset validation** (R3: +0.5, R2: +0.5 via experiments): Moves from "toy datasets" to "cross-domain validation."
4. **F1 restricted-range** (R5: +0.5): Confirms the headline claim is robust.
5. **Phase 1 fixes** (R4: +0.5, R5: +0.25): Selection stability, "28+1" clarity, broader implications.

### Consensus Weaknesses (remaining, 2+ reviewers)

1. **Design Space Theorem is in supplement** (R1, R3) — for conference: acceptable. For journal: must be main text.
2. **DASH information loss not quantified** (R1, R2) — qualitative paragraph but no mutual information analysis.
3. **Financial case study missing** (R4) — Credit-g is credit scoring but not modern/realistic enough.
4. **Dataset ID verification needed** (R5) — comprehensive_validation.py may have loaded wrong datasets.
5. **Stale axiom table** (R3) — supplement §1 says 12 axioms, code has 15.

### Best Paper Assessment (v2)

**"If every remaining weakness were fixed, would this win Best Paper at NeurIPS 2026?"**

**Closer than before — but still no.** The paper is now a **strong accept** (8.3 mean, up from 7.4). The Design Space Theorem addresses the structural gap R1 identified. The F1-F5 proofs are rigorous. The empirical validation is comprehensive.

**What's STILL missing for Best Paper:**

1. **The Design Space Theorem needs to be in the main text**, not the supplement. A Best Paper must be complete in 9 pages — the most important result can't require reading page 15 of the supplement.

2. **No new proof technique.** The paper still uses standard tools (symmetry arguments, Cramér-Rao, CLT, Berry-Esseen). A Best Paper typically introduces a method that other papers then use. The closest candidate is the Rashomon-to-impossibility reduction, which IS novel as an abstraction but not as a technique.

3. **The practical impact is still theoretical.** No pip-installable library, no case study showing DASH prevented a real decision error, no evidence of practitioner adoption. Best Papers in applied ML typically demonstrate real-world change.

**Realistic assessment:** This paper is now in the **top 5-10%** of NeurIPS submissions (strong accept from 4/5 reviewers, accept from 1). Best Paper requires top 0.1%. The gap is presentation (Design Space Theorem placement) and impact (adoption evidence), not theory.

### The Path Forward

**For NeurIPS (submit as-is):**
- Fix the stale axiom table (30 min)
- Verify dataset IDs (30 min)
- The paper is strong enough to accept

**For JMLR (the landmark paper):**
- Design Space Theorem as main-text centerpiece
- DASH information loss quantified (mutual information)
- FICO/Lending Club case study
- Lean formalization of Design Space Theorem
- 15+ datasets (add the corrected Adult, true FICO)
- This would score 9+ and be the definitive reference

---

### The Elevator Pitch (updated)

Every explanation method used in industry today — SHAP, gain importance, integrated gradients — is mathematically guaranteed to produce unstable feature rankings when features are correlated. This isn't a bug in the algorithm; it's a theorem. We prove it, quantify it for four model architectures, and show exactly what practitioners can do about it: average explanations across 25 independently trained models. The proof is machine-verified in Lean 4 — the first formally verified impossibility in explainable AI.

The supplement proves something deeper: the *complete characterization* of what's achievable. There are exactly two options — accept instability (single model) or accept ties (DASH ensemble) — and DASH is provably optimal. Every previous result in this paper is a corollary of this one theorem.

This is to feature attribution what Arrow's theorem is to voting: it defines the boundaries of what's possible, proves no one can do better, and shows the unique optimal path forward.

**Is this pitch more convincing than v1?**

Yes. The "one fundamental theorem with consequences" that R1 requested now exists. The pitch can now say "every previous result is a corollary" — which is a qualitatively different claim from "here are several related results." The remaining gap is presentation (it's in page 15 of the supplement) and tooling (no library).
