# Best Paper Review v3: The Attribution Impossibility (Cold Re-Read)

> Five-reviewer assessment scoring the CURRENT paper as if seeing it for the first time.
> Calibration anchor: Bilodeau et al. (PNAS 2024, 132 citations in 2.2 years).
> Ground rules enforced: tautological baseline, axiomatized DASH, 24/42 substantive, 4-line core proof.

---

## Reviewer 1: Fields Medalist (Cold)

### First Impression
A formally verified impossibility for feature attribution rankings. The abstract is dense — 42 declarations, 15 axioms, α ≈ 2/π — suggests thorough but possibly over-engineered work. Let me read the proof.

### After Reading the Proof (Theorem 1)
The proof is: "Rashomon gives two models with opposite orderings; any fixed faithful ranking contradicts one of them." This is... correct but immediate. The Rashomon property IS the impossibility, restated. The theorem adds no structure beyond the definition. Calling this an "Attribution Impossibility" in the tradition of Arrow or Chouldechova is aspirational — those theorems had non-obvious structural content.

### What Saves the Paper
The **framework** built around this simple core. The IterativeOptimizer abstraction (Definition 5) is a genuine contribution — unifying GBDT, Lasso, and NN under dominance + surjectivity is clean. The quantitative bounds (§4) show real architectural discrimination. The Design Space Theorem (§7) is the most interesting result: two families, Pareto frontier, M-axis collapse. The path convergence (supplement) having no Arrow analogue is a genuine structural insight.

### Weaknesses

**W1 (HIGH): The paper oversells the core theorem and undersells the framework.**
Theorem 1 gets a full subsection (§3.2) and a proof block for what is a 4-line argument. The Design Space Theorem — the actual structural contribution — gets half a page. The paper should lead with the Design Space Theorem and present Theorem 1 as "the easy base case."

**W2 (MEDIUM): The DASH resolution is axiomatized, not derived.**
Line 362: "In the Lean 4 formalization, this equality is axiomatized directly for balanced ensembles." So the positive result (DASH works) is assumed, not proved. The Lean verification certifies the impossibility chain but NOT the resolution. This is a significant asymmetry: "we prove you can't do X, and we assume Y works."

**W3 (LOW): The Arrow parallel is overstated.**
Line 446 now correctly notes "Arrow aggregates heterogeneous preferences... we aggregate noisy measurements." Good. But the paper still frames the result as "structurally identical" (line 466) to fairness impossibility. It's structurally analogous — not identical. The mechanisms are fundamentally different.

### Score
Soundness: 4/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7.5/10** | Confidence: 5/5 | **Accept**

*Bilodeau calibration: Bilodeau's core insight ("SHAP can be worse than random") was more surprising than ours ("collinear features have unstable rankings"). Our framework is deeper. Net: comparable, slight edge to us on depth, slight edge to Bilodeau on novelty.*

---

## Reviewer 2: Chief Scientist, AI Safety Organization (Cold)

### First Impression
An impossibility for feature attribution with EU AI Act implications. This is directly relevant to my work.

### Strengths
1. **The dual-reporting recommendation** (line 486) is the right answer. Report both single-model SHAP and DASH consensus.
2. **The information loss is quantified** in the supplement (log₂(m!) bits within-group, I_between(M) formula). This answers my question from v1.
3. **The F1 diagnostic** (r = -0.89, baseline r ≈ -0.56, net 60% genuine signal) gives practitioners a testable condition. The honest baseline disclosure is exactly the transparency I expect.
4. **11-dataset validation** including credit scoring (Credit-g).

### Weaknesses

**W1 (MEDIUM): The information loss section is in the supplement.**
For safety-critical applications, the tradeoff between stability and information matters. The main text has two sentences (lines 485-486). A safety reviewer needs the full analysis.

**W2 (LOW): The financial case study uses German Credit (1990s, 20 features).**
Real credit models have 200+ features. But the workflow demonstration is sound.

### Score
Soundness: 4/4 | Contribution: 3.5/4 | Clarity: 4/4 | Experiments: 3.5/4
**Overall: 8/10** | Confidence: 4/5 | **Strong Accept**

*Bilodeau calibration: Their paper had no resolution. Ours does (DASH). Their paper had no diagnostic. Ours does (F1, F5). For a safety organization, ours is more actionable. Edge: us.*

---

## Reviewer 3: Editor-in-Chief, JMLR (Cold)

### First Impression
A 11-page conference paper with a 29-page supplement. The supplement alone is a journal paper. Let me evaluate the combined package.

### Strengths
1. **The supplement is JMLR-quality.** F3 (FIM impossibility with regularity conditions), F2 (DASH MVUE optimality with median/trimmed comparison), F1 (Berry-Esseen bounds), F5 (exchangeability lemmas), Design Space Theorem with 5 corollaries, information loss analysis, 11-dataset table, financial case study. This is a comprehensive treatment.
2. **The honest Lean accounting** (lines 85-86): "42 type-checked declarations... of which 24 are substantive proofs... Corollary 1 depends on the attribution_sum_symmetric axiom." This is transparent. Most papers with formal verification don't distinguish proved from axiomatized.
3. **The structural baseline disclosure** for F1 (supplement): "random data gives r ≈ -0.56... 60% of explained variance reflects genuine patterns." Exemplary scientific transparency.

### Weaknesses

**W1 (HIGH): The paper tries to be both a conference paper and a journal paper.**
The main text compresses the Design Space Theorem into half a page. The supplement has the full proof, the information loss analysis, the financial case study, the rigorous F1-F5. A reviewer reading only the main text gets an incomplete picture. This is a conference-paper problem: the work is too big for 9 pages.

**W2 (MEDIUM): 11 pages may exceed the NeurIPS limit.**
NeurIPS allows 9 pages of content. References and checklist are exempt. Is this paper 9 pages of content? It looks close to 10. This should be verified.

### Score
Soundness: 4/4 | Contribution: 4/4 | Clarity: 3.5/4 | Experiments: 4/4
**Overall: 8.5/10** | Confidence: 5/5 | **Strong Accept**

*Bilodeau calibration: Bilodeau had no supplement of this depth. Our technical treatment is significantly stronger. Edge: us, clearly.*

---

## Reviewer 4: VP of ML, Major Bank (Cold)

### First Impression
A theory paper about SHAP instability. My team sees this weekly. Is there something actionable?

### Strengths
1. **The F5 practitioner algorithm** (supplement) with Step 0 (identify groups at |ρ| > 0.5) is directly implementable.
2. **"Within-group features reported as tied should be documented as interchangeable"** (line 372). This is the answer for our model validation reports.
3. **The M-sizing formula** is useful: M ≥ (z · σ / Δ)² per pair.

### Weaknesses

**W1 (HIGH): The paper doesn't validate on a dataset with realistic financial collinearity.**
German Credit has 20 features with modest correlations (only 1 pair above |ρ| = 0.5, threshold lowered to 0.3). My credit models have income ↔ DTI ↔ credit score with ρ = 0.7-0.9. The impossibility should BITE HARD on financial data, but the paper doesn't show it.

**W2 (MEDIUM): The proportionality axiom (Axiom 1) is strong.**
It assumes φ_j = c · n_j for a SINGLE constant c across all features. In real XGBoost, per-split contributions vary significantly (deep splits contribute less than root splits). This axiom is the weakest link in the axiom chain.

### Score
Soundness: 3/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7/10** | Confidence: 4/5 | **Weak Accept**

*Bilodeau calibration: Their result ("SHAP can be worse than random") is scarier for a practitioner. Ours ("rankings are unstable") was already suspected. But we provide DASH as a fix. Net: comparable for practitioner impact.*

---

## Reviewer 5: Adversarial Red-Teamer (Cold)

### First Impression
Time to break things. What's the weakest claim?

### Attack 1: The axiom-to-theorem ratio
15 axioms, 24 substantive theorems. That's 1.6 theorems per axiom. Some "theorems" are 1-line axiom restatements (the paper admits this: "9 convenience lemmas"). The CORE result (Theorem 1) needs ZERO axioms — it's genuinely clean. But the quantitative results and DASH resolution lean heavily on axioms. The paper is honest about this (lines 85-86), which makes it hard to attack.

### Attack 2: F1 diagnostic methodology
The F1 correlation (r = -0.89) has a structural baseline of r ≈ -0.56. The paper discloses this. The residual (Δr = -0.33, or ~47% excess R²) is the genuine signal. This is honest but means the "r = -0.89" headline in the main text (line 422) is misleading for readers who don't read the supplement. The main text should cite the baseline.

### Attack 3: The proportionality axiom
φ_j = c(f) · n_j for ALL features j with a SINGLE model-specific constant c(f). This means if feature 1 has 100 splits contributing 0.01 each and feature 2 has 50 splits contributing 0.05 each, the axiom says they should have attributions in ratio 2:1. But the actual attributions (by contribution) would be 1:2.5. The axiom holds for gain-based importance (by construction) but only approximately for SHAP.

### Strengths I can't attack
1. **Theorem 1 with zero axiom deps** — unimpeachable
2. **The Lean build passes with 0 sorry** — verified
3. **The Design Space Theorem** — the two-family characterization is genuinely structural
4. **The disclosure of tautological baseline** — can't attack what's already admitted

### Score
Soundness: 3.5/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7.5/10** | Confidence: 4/5 | **Accept**

*Bilodeau calibration: Their axioms (completeness + linearity) are cleaner — universally accepted properties of SHAP/IG. Our axioms (proportionality, split counts) are stronger assumptions. Edge to Bilodeau on axiomatic elegance. Edge to us on breadth of results.*

---

## Area Chair Meta-Review

### Score Table

| | R1 (Fields) | R2 (Safety) | R3 (JMLR) | R4 (Bank) | R5 (Red) | Mean |
|---|---|---|---|---|---|---|
| **v1** | 7 | 8 | 8 | 7 | 7 | **7.4** |
| **v2** | 8.5 | 8.5 | 9 | 7.5 | 8 | **8.3** |
| **v3 (cold)** | **7.5** | **8** | **8.5** | **7** | **7.5** | **7.7** |

### Why v3 < v2

The v2 scores were inflated by familiarity bias — the reviewer watched the fixes being applied and scored the improvement, not the absolute quality. A cold reader sees:
- A simple core theorem dressed up as a major contribution
- An axiomatized resolution (DASH equity is assumed, not derived)
- A 29-page supplement that should be a journal paper, compressed into a conference format
- Strong honest disclosures that paradoxically highlight the limitations

### Consensus Strengths (cold)
1. **The framework is genuinely comprehensive**: impossibility → bounds → resolution → design space → diagnostics. No prior paper covers this spectrum.
2. **The Lean formalization** with honest axiom accounting is a first for XAI.
3. **The transparency** (tautological baseline, axiomatized DASH, proportionality limitations) is exemplary.

### Consensus Weaknesses (cold)
1. **The core theorem is trivial** and the paper doesn't own this sufficiently. The Design Space Theorem should be the centerpiece, not Theorem 1.
2. **The conference format doesn't serve this work.** The best results are in the supplement.
3. **The proportionality axiom is a real limitation** that no reviewer can fully assess without domain expertise in XGBoost internals.

### Best Paper Assessment (v3, cold)

**No.** This paper is a solid accept (7.7 mean) but not a best paper (requires ~9+). The gap:
- No mathematical surprise (the core result is expected by practitioners)
- No new technique (standard symmetry/contradiction arguments)
- The deepest results (Design Space, F1-F5) are in the supplement

**What it IS:** The most thorough treatment of attribution stability theory to date. If submitted to JMLR as a 40-page paper, it would be a strong reference paper. At NeurIPS, it's compressed into a format that undersells its depth.

---

## Part 2: The "10/10 Paper" Gap Analysis

### R1 (Fields Medalist): What would make this 10/10?

1. **A non-trivial proof technique** — something reusable beyond this specific problem. The IterativeOptimizer abstraction is conceptual but the actual proofs are all straightforward. If the Design Space Theorem's exhaustiveness proof (Step 3) were strengthened to a true characterization theorem with a non-obvious proof, that would add mathematical substance.
   - *Achievable?* Partially. The Bayes-optimality argument underlying Step 3 could be developed into a cleaner theorem. Effort: significant (1-2 weeks of mathematical work).
   - *Realistic ceiling:* 8.5. Without a genuinely new technique, this reviewer caps at 8.5.

2. **Restructure: Design Space Theorem as Theorem 1, current Theorem 1 as "a trivial corollary."**
   - *Achievable?* Yes, purely editorial. Effort: 2-4 hours of restructuring.
   - *Impact:* +0.5 from R1.

### R2 (AI Safety): What would make this 10/10?

1. **A real-world case study where DASH prevented a decision error.**
   - *Achievable?* Not without access to a deployed system. The German Credit case study shows the workflow but not a prevented error.
   - *Realistic ceiling:* 9. The information loss quantification + dual reporting + regulatory framing are strong enough.

2. **Information loss section promoted to main text.**
   - *Achievable?* Yes. Effort: 1 hour (cut something else, add the key formula and interpretation).
   - *Impact:* +0.5 from R2.

### R3 (JMLR Editor): What would make this 10/10?

1. **Submit to JMLR instead.** The 29-page supplement IS the JMLR paper. The conference format actively hurts this work.
   - *Achievable?* Yes. Effort: restructure for journal format (1-2 weeks).
   - *Impact:* +1 from R3 (full marks for the comprehensive treatment in proper format).

2. **Fix the page limit issue.** Verify the main text is ≤ 9 pages of content.
   - *Achievable?* Yes. Effort: 1-2 hours of tightening.
   - *Impact:* +0.5 (removes a procedural rejection risk).

### R4 (Bank VP): What would make this 10/10?

1. **FICO or Lending Club case study with ρ = 0.7-0.9 collinearity.**
   - *Achievable?* FICO data requires a license. Lending Club data is public but requires preprocessing.
   - *Effort:* 1-2 days for Lending Club.
   - *Impact:* +1 from R4.
   - *Realistic ceiling with current data:* 7.5.

### R5 (Red-Teamer): What would make this 10/10?

1. **Prove the proportionality axiom for TreeSHAP** (not just assume it).
   - *Achievable?* No. This would require analyzing the TreeSHAP algorithm's per-path contributions, which is a research problem in itself.
   - *Realistic ceiling:* 8. The transparency about axioms caps the reviewer's objections.

2. **Add the F1 tautological baseline to the main text**, not just the supplement.
   - *Achievable?* Yes. Effort: add one sentence to line 422.
   - *Impact:* +0.25 from R5.

---

## Part 3: Ranked Implementation Plan

### Rank by impact-per-hour (NeurIPS submission, before May 6):

| Rank | Item | File | Effort | Score Impact | Impact/Hour |
|------|------|------|--------|-------------|-------------|
| 1 | **Add F1 baseline to main text** | main.tex:422 | 10 min | +0.25 (R5) | Very high |
| 2 | **Verify page count ≤ 9** | main.tex | 30 min | Avoids desk reject | Critical |
| 3 | **Restructure: lead with Design Space, Thm 1 as corollary** | main.tex | 3 hrs | +0.5 (R1) | Medium |
| 4 | **Promote information loss key formula to main text** | main.tex, supplement.tex | 1 hr | +0.5 (R2) | Medium |
| 5 | **Lending Club case study** | scripts/, supplement.tex | 1-2 days | +1 (R4) | Low-Medium |
| 6 | **Strengthen DST Step 3 proof** | supplement.tex | 1-2 weeks | +0.5 (R1) | Low |

### Items NOT achievable before May 6:
- New proof technique (R1's ceiling)
- Prove proportionality axiom (R5)
- Real-world DASH deployment case study (R2)
- JMLR restructure (R3)

### Expected v3 scores after executing items 1-4:

| | R1 | R2 | R3 | R4 | R5 | Mean |
|---|---|---|---|---|---|---|
| v3 (cold) | 7.5 | 8 | 8.5 | 7 | 7.5 | **7.7** |
| After items 1-4 | 8 | 8.5 | 8.5 | 7 | 7.75 | **7.95** |

---

## /vet

### Round 1: Factual
- v3 scores are lower than v2. This is intentional — v2 was inflated by familiarity.
- Bilodeau citation count (132) verified via Semantic Scholar API.
- "24 substantive" matches the paper's current claim.
- Page count concern is real — need to verify.

### Round 2: Am I being too generous or too harsh?
- R1 at 7.5 (down from 8.5): is this realistic? A cold Fields Medalist seeing a 4-line proof called "The Attribution Impossibility" in the tradition of Arrow... yes, 7.5 feels right. The framework saves it from 7.
- R3 at 8.5 (down from 9): the supplement IS journal-quality. But the conference format hurts. 8.5 is fair.
- R4 at 7 (down from 7.5): German Credit with |ρ| > 0.3 is weak for a banking reviewer. 7 is generous — a real bank VP might score 6.5.

⚠️ **Correction:** R4 may be overcredited. Consider 6.5-7 range. I'll keep 7 but note the uncertainty.

### Round 3: Omissions
- I didn't address: what if a NeurIPS AC sees the 29-page supplement and thinks "this should be a journal paper, not a conference paper"? Some ACs actively penalize papers that use supplements as a crutch.
- I didn't address: the paper's contribution list (lines 91-95) doesn't mention the Design Space Theorem explicitly. It should.

### Confidence Ratings
| Finding | Confidence |
|---------|-----------|
| v3 mean 7.7 (down from 8.3) | **HIGH** — cold reads are systematically harsher |
| Items 1-2 are highest priority | **HIGH** — one prevents a desk reject, one costs 10 minutes |
| Restructuring (item 3) would help most | **MEDIUM** — 3 hours is significant for May 6 deadline |
| R1's ceiling is 8.5 without a new technique | **HIGH** — impossibility theorems without novel proofs cap out |
| Overall: strong accept but not best paper | **HIGH** — the work is solid, the format is wrong |
