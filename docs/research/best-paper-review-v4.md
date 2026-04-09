# Best Paper Review v4: Surprise & Impact Audit

> The rigor is handled. Four rounds of auditing fixed 19 items. The score
> is ~8/10. The gap to best paper is NOT rigor — it's SURPRISE and IMPACT.
> This review focuses on closing that gap.
>
> Calibration: Bilodeau et al. (PNAS 2024, 132 citations in 2.2 years).

---

## Part 1: The Surprise Audit

### What in this paper WOULD make a reader say "I didn't expect that"?

| # | Candidate surprise | Surprise level | Who's surprised? |
|---|-------------------|---------------|-----------------|
| 1 | **The path convergence has no Arrow analogue.** Dropping faithfulness and dropping completeness converge to the same solution. In Arrow's setting, the three relaxation paths give different solutions. | MODERATE | Theorists who know Arrow |
| 2 | **Random forests are the opposite of GBDT.** The ratio CONVERGES for RF and DIVERGES for GBDT. Sequential residuals create the problem; independent trees solve it. | MODERATE | ML engineers |
| 3 | **The design space has exactly two families.** Not a continuum — two discrete operating modes. | MODERATE-HIGH | Theorists |
| 4 | **The Lean formalization caught 2 genuine bugs.** Informal proofs had logical inconsistencies that only the type checker found. | MODERATE | Formal methods skeptics |
| 5 | **α = 2/π from binary quantization theory.** The per-stump signal capture fraction has a closed-form derivation from information theory. | LOW-MODERATE | Information theorists |

**Honest assessment:** None of these is a "jaw-dropper." The paper's surprise profile is INCREMENTAL — each finding is interesting but expected by at least some segment of the audience. Bilodeau's "SHAP can be worse than random" was a genuine jaw-dropper because people BELIEVED SHAP worked.

### What result NOT in the paper would create genuine surprise?

| # | Candidate | Feasibility | Surprise factor | ML researchers who'd change behavior |
|---|-----------|-------------|----------------|--------------------------------------|
| 1 | **"Conditional SHAP is also impossible."** Show the impossibility extends to causal/conditional SHAP (Janzing et al. 2020), which many believe resolves collinearity issues. If symmetric features have equal causal effects, the Rashomon property still holds for causal attributions. | HIGH (1-2 days: define causal attributions, prove Rashomon still holds under equal causal effects) | **VERY HIGH** — researchers who switched to conditional SHAP thinking they escaped would be surprised | ~5,000 (everyone using conditional SHAP) |
| 2 | **"The impossibility is TIGHT: there exists a model class where exactly 50% of within-group rankings flip."** Show the 1/2 unfaithfulness bound is achievable, not just an upper bound. Construct the adversarial example. | HIGH (the symmetric DGP already achieves this — just formalize) | MODERATE | ~1,000 |
| 3 | **"DASH needs Ω(1/ε²) models to achieve ε-stability — and this is optimal."** A computational lower bound on ensemble size. Show M=25 isn't just a heuristic recommendation but information-theoretically necessary. | MEDIUM (requires connecting to minimax estimation theory) | HIGH — gives practitioners a lower bound, not just an upper bound | ~3,000 |
| 4 | **"The impossibility extends to I(X_j;X_k) > 0, not just ρ > 0."** Generalize from linear correlation to mutual information. Would cover nonlinear dependence (X_k = X_j²). | LOW-MEDIUM (the Rashomon argument works but the quantitative bounds need information-geometric tools) | HIGH | ~10,000 (all practitioners, since MI > 0 is nearly universal) |
| 5 | **"Feature importance explanations violate EU AI Act Art. 13(3)(b)(ii) for >60% of deployed models."** An empirical estimate of how many production models are affected, based on the collinearity prevalence in standard benchmarks. | HIGH (survey 50 UCI/Kaggle datasets, compute % with ρ > 0.5 feature pairs) | **VERY HIGH for policymakers** — a specific number for regulators | ~50,000 (anyone deploying ML in EU) |

**Recommendation: Candidate 1 (conditional SHAP) and Candidate 5 (regulatory prevalence) are the highest surprise-per-effort items.** Candidate 1 would genuinely change the narrative from "switch to conditional SHAP" to "there IS no escape." Candidate 5 would make this paper cited in regulatory documents.

---

## Part 2: The One-Sentence Test

| # | Sentence | Non-expert understands? | Expert finds novel? | Retweetable? |
|---|----------|----------------------|-------------------|-------------|
| 1 | "Feature explanations are unreliable when features are correlated." | Yes | **No** (known) | No |
| 2 | "No single model can have faithful, stable, and complete feature rankings under collinearity." | Mostly | Somewhat | No (jargon) |
| 3 | "We prove that SHAP rankings can flip when you retrain the model — and formally verify there's no fix." | Yes | Somewhat | **Yes** |
| 4 | "The space of possible feature explanations has exactly two modes: unstable-but-specific, or stable-but-tied." | Yes | **Yes** | **Yes** |
| 5 | "Every attribution method is either unstable or honest about uncertainty — there is no third option." | **Yes** | **Yes** | **Yes** |
| 6 | "Arrow's theorem for feature importance: you can't rank features faithfully, stably, and completely." | Mostly | Yes (the parallel) | Yes |
| 7 | "We found the Chouldechova of explainability — three things practitioners want that math forbids." | No (requires context) | Yes | No |
| 8 | "Train 25 models instead of 1, and your SHAP explanations become provably stable." | **Yes** | Partially | **Yes** |
| 9 | "The first formally verified theorem about the limits of AI explanations." | Yes | Yes (the "first") | Yes |
| 10 | "Feature importance rankings are mathematically guaranteed to contradict themselves under collinearity — here's the proof, the quantification, and the fix." | Yes | Yes | **Yes** |

**Winners:**
- **For theorists:** #5 ("Every attribution method is either unstable or honest about uncertainty — there is no third option.")
- **For practitioners:** #8 ("Train 25 models instead of 1, and your SHAP explanations become provably stable.")
- **For Twitter/social:** #10 (complete package: proof + quantification + fix)
- **For the abstract:** #4 ("exactly two modes") — this captures the Design Space Theorem.

**Current abstract uses none of these.** The abstract leads with "SHAP values are widely used" (boring) and buries the two-family insight. The paper's identity should be: **"Two modes, no third option."**

---

## Part 3: Bilodeau Comparison

| # | Dimension | Bilodeau (PNAS 2024) | Us | Winner |
|---|-----------|---------------------|-----|--------|
| 1 | **Core surprise** | "SHAP can be worse than random" (unexpected) | "Collinear features → unstable rankings" (expected) | **Bilodeau** |
| 2 | **Proof depth** | Moderate (construction of adversarial functions) | Simple (4-line contradiction from Rashomon) | **Bilodeau** |
| 3 | **Scope of impossibility** | Complete + linear methods on rich classes | Faithful + stable + complete under collinearity | **Tie** (different axes) |
| 4 | **Quantitative bounds** | None (pure impossibility) | 4 model classes with divergence rates | **Us** |
| 5 | **Resolution** | None offered | DASH, proved Pareto-optimal | **Us** |
| 6 | **Design space** | Not characterized | Two-family theorem, M-axis | **Us** |
| 7 | **Formal verification** | None | Lean 4, 42 declarations, 0 sorry | **Us** |
| 8 | **Practitioner tool** | None | F1/F5 diagnostics, M-sizing formula | **Us** |
| 9 | **Empirical validation** | Basic (synthetic) | 11 datasets, 2 financial case studies | **Us** |
| 10 | **Institutional backing** | Google DeepMind, Stanford, Toronto | Independent researchers | **Bilodeau** |

**Score: Bilodeau wins 3, we win 6, tie 1.**

**But: Bilodeau's 3 wins (#1 surprise, #2 proof depth, #10 institutions) are the dimensions that matter most for initial reception.** Our 6 wins (#4-9) are the dimensions that matter for long-term impact and citations.

**Flippable dimensions:**
- **#1 (surprise)**: If we prove the conditional SHAP extension (Candidate 1 from Part 1), the surprise factor increases significantly. "Even causal SHAP can't escape" is unexpected.
- **#2 (proof depth)**: Unflippable without a new proof technique. The core proof IS simple.
- **#10 (institutions)**: Unflippable. But the Lean formalization partially compensates (machine credibility substitutes for institutional credibility).

---

## Part 4: Best Paper Criteria

| # | Criterion | Score (1-10) | Evidence | What would raise it? |
|---|-----------|-------------|---------|---------------------|
| 1 | **Surprising result** | 5 | Practitioners already know SHAP is unstable. The Design Space two-family structure is somewhat surprising. | Prove conditional SHAP is also impossible (+2). Show >60% of deployed models affected (+1). |
| 2 | **New technique/abstraction** | 6 | IterativeOptimizer (dominance + surjectivity) is a genuine abstraction. But the proofs use standard tools (symmetry, contradiction, Cramér-Rao). | Develop the Bayes-optimality argument in DST Step 3 into a reusable technique (+1). Ceiling: 7. |
| 3 | **Paradigm shift potential** | 7 | "Train M models, not 1" is a genuine recommendation that would change practice. "Two modes, no third option" is a new way to think about XAI. | Demonstrate the paradigm shift with a real deployment (+2). |
| 4 | **Exceptional clarity** | 7 | Well-written, honest about limitations. But the abstract doesn't lead with the surprising insight, and the conference format compresses the best results into the supplement. | Rewrite abstract around "two modes" framing (+1). Submit to JMLR for full clarity (+1). |
| 5 | **Broad impact** | 8 | Affects every SHAP user (millions). Regulatory implications (EU AI Act). 11-dataset validation. | Quantify regulatory prevalence (+1). Build pip-installable library (+1). |

**Total: 33/50. NeurIPS best papers typically score 40+.**

**Achievable ceiling: ~38/50** (conditional SHAP extension, regulatory prevalence survey, abstract rewrite, JMLR format). The remaining gap (40-38=2) requires a new proof technique, which is not achievable as an engineering task.

---

## Part 5: The "10/10 Version"

**Abstract of the 10/10 paper:**

> We prove that no feature attribution method — including SHAP, conditional SHAP, integrated gradients, and any future method satisfying basic axioms — can simultaneously produce feature rankings that are faithful (reflecting the model), stable (robust to retraining), and complete (ranking all features) whenever features carry mutual information. The impossibility is not a consequence of any specific algorithm's design; it is a theorem about the geometry of the loss landscape under feature dependence, derived from the Fisher information matrix's rank deficiency.
>
> We characterize the complete design space: practitioners must choose between unstable-but-specific explanations (single model) or stable-but-tied explanations (ensemble). There is no third option. DASH ensemble averaging is the provably optimal method in the stable branch, and we derive the minimum ensemble size M ≥ Ω(1/ε²) needed for ε-stability. We show that >60% of production ML models in standard benchmarks have feature pairs where the impossibility applies.
>
> The entire proof — including the impossibility, the design space characterization, and the optimality of DASH — is mechanically verified in Lean 4.

**What's different from our current abstract:**
1. Extends to I(X_j;X_k) > 0 (not just ρ > 0)
2. Covers conditional SHAP explicitly
3. Derives from FIM geometry (not just Rashomon — deeper mechanism)
4. States the Ω(1/ε²) lower bound on M
5. Quantifies regulatory prevalence (>60%)
6. Leads with the design space ("no third option"), not the impossibility

**Theorem 1 of the 10/10 paper:** The Attribution Design Space Theorem (our current §7, elevated to the opening theorem).

**Signature figure:** A 2D plot with stability on the y-axis and unfaithfulness on the x-axis. Family A is a point cloud in the upper-right (high U, variable S). Family B is a curve along the left edge (U=0, S increasing with M). The infeasible region (lower-right: S=1, U=0) is shaded. DASH points lie on the Pareto frontier. Caption: "The two modes of feature explanation."

**Experiment that clinches it:** A table showing, for 50 popular Kaggle/UCI datasets, the fraction with at least one feature pair where the impossibility applies (flip rate > 10%). Prediction: >60%. This makes the result a FACT about deployed ML, not a theoretical possibility.

### Distance from current paper

| Component | Current state | 10/10 state | Gap | Closable? |
|-----------|--------------|-------------|-----|-----------|
| MI generalization | ρ > 0 only | I(X_j;X_k) > 0 | Large | Partially (1-2 weeks) |
| Conditional SHAP | Not addressed | Proved impossible | Medium | YES (1-2 days) |
| FIM as mechanism | Supplement only | Main theorem | Small | YES (editorial) |
| Ω(1/ε²) lower bound | Not stated | Proved | Medium | YES (minimax argument, 2-3 days) |
| Regulatory prevalence | 11 datasets, no % | 50 datasets, >60% | Medium | YES (1-2 days, script) |
| Pareto frontier figure | Not created | Signature figure | Small | YES (1 hour) |
| "No third option" framing | Not in abstract | Leads the abstract | Small | YES (30 min) |

**Fraction of gap closable: ~60%.** The MI generalization and new proof technique are the unclosable parts.

---

## Part 6: Remaining Rigor Items

All 19 items from the v3 panel are RESOLVED across 4 Phase commits (`db31a00`, `7fd5273`, `43ccf91`, `bd0cf51`). No CRITICAL or HIGH findings remain from any prior audit.

One new concern from Phase 3: **the Credit Card Default case study shows 4% DASH flip rate at M=25** (vs 0% for German Credit). This is honest — very high ρ (0.95) requires larger M — but it slightly undermines the "DASH resolves instability" claim. The paper should note: "For ρ > 0.9, M > 25 may be needed."

---

## Part 7: The Honest Verdict

**Is "best paper at NeurIPS" achievable?** Not in the current form — the core proof is too simple and the result is not surprising enough to practitioners. The ceiling at NeurIPS is ~8/10 (strong accept, possible oral, not best paper). The paper's natural home is JMLR, where the 40-page treatment would be a definitive reference scoring 9+.

**Is "foundational" earned?** Partially. The Design Space Theorem IS a structural contribution that others will build on. The "two modes, no third option" insight IS how people will think about attribution stability going forward. But "foundational" in the Arrow/Chouldechova sense requires a result that people didn't expect and can't work around — and practitioners CAN work around our result by switching to conditional SHAP (unless we prove they can't).

**The single most impactful thing:** Prove that the impossibility extends to conditional SHAP. This would close the escape hatch, increase the surprise factor from 5/10 to 7/10, and make the paper genuinely inescapable. Estimated effort: 1-2 days of mathematical work + Lean formalization. This one addition would shift the paper from "important formalization of a known phenomenon" to "you cannot escape this no matter what you do."

---

## /vet

### Round 1: Factual
- Bilodeau 132 citations: verified via Semantic Scholar API. ✓
- v3 scores verified against the v3 document. ✓
- "60% of deployed models" is a GUESS, not a measurement. The 10/10 abstract states it as fact. ⚠️ Mark as "predicted, to be validated."
- Ω(1/ε²) lower bound is standard minimax theory for estimating means, not a new result. The claim "proved" is accurate but not novel. ⚠️

### Round 2: Am I being generous about the gap?
- Conditional SHAP extension: am I SURE this works? If features have equal causal effects AND equal observational effects, then causal SHAP gives equal attributions → no instability. The Rashomon property doesn't apply because causal SHAP uses the causal graph, which is fixed. **Wait — the instability comes from OBSERVATIONAL models, not causal graphs. If you use causal SHAP, you're conditioning on the causal structure, which breaks the symmetry.** So conditional SHAP MAY escape the impossibility if the causal graph is known. The extension to conditional SHAP is NOT guaranteed to work.

⚠️ **Correction:** Candidate 1 (conditional SHAP) needs careful analysis. If the causal graph identifies which feature is "truly" more important, conditional SHAP breaks the symmetry and the Rashomon property fails. The impossibility extends to conditional SHAP ONLY if the causal graph also has symmetric features (equal causal effects). This is a weaker extension than initially claimed.

**Revised surprise assessment for Candidate 1:** MEDIUM (works only when causal effects are also symmetric, which is a subset of the collinearity case).

### Round 3: Omissions
- I didn't address: what if the paper's timing is wrong? The EU AI Act obligations took effect August 2, 2026 — if we submit to NeurIPS May 6 and the paper is published December 2026, the regulatory window is still open. But if we submit to JMLR (6-18 month review), the window may close before publication.
- I didn't address: the Pareto frontier figure doesn't exist yet. It should.
- I didn't address: the "50 datasets, >60%" prevalence survey. This could be done in 2-3 hours and would be the single most cited result in the paper (regulators would use it).

### Corrections Applied
1. ⚠️ Conditional SHAP extension works only for symmetric causal effects. Reduced surprise from VERY HIGH to MEDIUM.
2. ⚠️ "60% of deployed models" is a prediction, not a fact. The 10/10 abstract should say "we estimate" not "we show."
3. ⚠️ Ω(1/ε²) bound is standard, not a novel contribution.

### Confidence Ratings

| Finding | Confidence |
|---------|-----------|
| The gap to best paper is surprise, not rigor | **HIGH** |
| "Two modes, no third option" is the right framing | **HIGH** |
| We beat Bilodeau on 6/10 dimensions | **HIGH** |
| Conditional SHAP extension is the highest-impact addition | **MEDIUM** (works only under equal causal effects) |
| Regulatory prevalence survey (>60%) is high-impact | **HIGH** |
| Realistic ceiling: 8/10 NeurIPS, 9+/10 JMLR | **HIGH** |
| The Pareto frontier figure would be iconic | **MEDIUM** (needs to be created and tested) |
