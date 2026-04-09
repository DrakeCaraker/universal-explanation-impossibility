# Adversarial Full-Stack Audit: The Attribution Impossibility

> Audit date: 2026-03-31
> Goal: Find every error, overclaim, logical gap, and weakness that would
> prevent best paper at NeurIPS 2026 or acceptance at Nature Machine Intelligence.
> All 7 audits complete. Agent results integrated.

---

## Audit 1: Lean-Paper Alignment (via agent)

### Finding 1.1: Design Space Theorem Lean ≠ paper [CRITICAL]
The paper claims "the achievable set is the union of exactly two families." The Lean `design_space_theorem` is a 3-part conjunction of previously-proved facts — it does NOT prove exhaustiveness or Pareto optimality. The Lean docstring admits this.

### Finding 1.2: consensus_equity is axiomatized, not proved [CRITICAL]
Paper Corollary 1 ("DASH achieves equity") is a 1-line unfolding of the `attribution_sum_symmetric` axiom. The actual mathematical content is entirely axiomatized. The paper acknowledges this in the proof text but counts it as a "substantive theorem."

### Finding 1.3: Theorem count inflated [HIGH]
Of 33 theorems, ~9 are trivial (1-line axiom restatements, basic arithmetic, or direct compositions). Genuinely substantive: ~24. The paper variously claims 29 or 33 "substantive" — both overcount.

### Finding 1.4: Spearman m³/P³ bound is axiomatized [HIGH]
The stability bound uses `spearman_classical_bound` (an axiom). The genuinely derived bound is weaker: 3/(P³-P). The paper cites the tighter axiomatized constant.

### Finding 1.5: consensus_variance_rate = axiom verbatim [HIGH]
`exact consensus_variance_bound fs M hM j` — zero proof content.

### Finding 1.6: Axiom system is consistent [CLEAN]
No pair of axioms derives False. The system is sound within its assumptions.

---

## Audit 2: Mathematical Correctness

### Finding 2.1: F3 ε₀ algebra has intermediate error [MEDIUM]

**Location:** supplement.tex:672-675

**The claim:** "K₃/6 · (ε/λ₋)^{3/2} ≤ K₃/6 · λ₋^{9/2}/(K₃³ · λ₋^{3/2}) = λ₋³/(6K₃²)"

**The problem:** Substituting ε₀ = 2λ₋³/K₃² into (ε/λ₋)^{3/2} gives (2λ₋²/K₃²)^{3/2} = 2^{3/2} · λ₋³/K₃³. The paper drops the factor of 2^{3/2} ≈ 2.83 in the intermediate step. The final step claims λ₋³/(6K₃²) ≤ ε/2 = λ₋³/K₃², which requires 1/6 ≤ 1 (true). But the ACTUAL intermediate value is 2^{3/2}·λ₋³/(6K₃²), and the ACTUAL check is 2^{3/2}/6 ≈ 0.47 ≤ 1 (also true).

**Impact:** The conclusion is correct — ε₀ = 2λ₋³/K₃² does control the cubic remainder. But the algebra as written is wrong. A careful reviewer would flag this.

**Fix:** Replace the intermediate step with the correct calculation, or just note "a direct computation shows K₃/6 · (ε₀/λ₋)^{3/2} = 2^{3/2}λ₋³/(6K₃²) < λ₋³/K₃² = ε₀/2."

### Finding 2.2: Berry-Esseen applied to n=1 is non-standard [MEDIUM]

**Location:** supplement.tex:~940-975 (F1 proof Part I)

**The claim:** Berry-Esseen bound gives |ε_BE| ≤ C₀γ_{jk}/σ_{jk}³ for the flip rate approximation.

**The issue:** The Berry-Esseen theorem is about the CLT for sums of n i.i.d. variables. For n=1, there is no averaging — the bound is just the distance between the CDF of D = φ_j - φ_k and the Gaussian CDF. This bound can be vacuous (>1) if D is far from Gaussian.

**The paper acknowledges this** (line ~970: "for n=1, the bound is C₀γ/σ³ ... only informative when D is approximately Gaussian") and provides Shapiro-Wilk evidence. So it's not wrong, just unusual.

**Impact:** A pedantic reviewer could object to calling this "Berry-Esseen" when it's really just the trivial bound on CDF distance. The substance is correct.

**Fix:** Add a footnote: "For n=1, the Berry-Esseen bound reduces to a bound on the CDF distance between D and a Gaussian, which is informative only when D is well-approximated by a normal distribution."

### Finding 2.3: Design Space Theorem — S definition mismatch [CRITICAL]

**Location:** supplement.tex:1586-1587 (Definition) vs 1609 (Family A bound)

**The problem:** S is DEFINED as "between-group stability — the probability that the consensus preserves the true between-group ordering." But Family A's bound "S ≤ 1 - Ω(m³/P³)" comes from the **Spearman correlation** of the full ranking (main text Theorem 2), which includes within-group pair reshuffling. These are DIFFERENT metrics.

**Why this matters:** A majority-vote-of-M-models method is COMPLETE (ranks all pairs), has U = 1/2 for within-group pairs (by symmetry), and has high BETWEEN-GROUP stability (majority vote stabilizes with M). Its BETWEEN-GROUP S approaches 1, but its FULL-RANKING Spearman remains bounded by 1 - m³/P³ because within-group pairs are still randomly ordered.

If S means between-group stability (as defined), the Family A upper bound is WRONG — complete methods can achieve high between-group S with M>1 models while keeping U=1/2. If S means full-ranking Spearman (as needed for the bound), the definition is wrong.

**Impact:** This is the biggest mathematical issue in the paper. The Design Space Theorem's characterization of Family A is either wrong (if S = between-group) or misleadingly defined (if S = Spearman).

**Fix:** Either:
(a) Redefine S as full-ranking stability (expected Spearman) — then Family A's bound holds, but the theorem is about Spearman, not about between-group ordering preservation.
(b) Derive the Family A between-group S bound separately — show that even between-group stability is limited for complete methods (this is actually true: the within-group instability DOES propagate to between-group through rank correlations with nearby-ranked between-group features).
(c) State the theorem with S = full-ranking Spearman and note that between-group S is separately characterized.

### Finding 2.4: Design Space Theorem Step 3 — majority vote not covered [HIGH]

**Location:** supplement.tex:1675-1700 (Step 3 proof)

**The problem:** Step 3 considers 3 cases but doesn't cover the majority-vote method:
- Case 1: faithful to individual models → U = 1/2 ✓
- Case 2: not faithful to some model → optimal unfaithful assigns ties ✓
- Case 3: aggregates M>1 models, complete → U > 0 ✓

The majority vote IS covered by Case 3 (aggregates M>1, complete, U > 0). The claim "any tie-breaking rule disagrees with a fraction of models" is correct for the majority vote (U = 1/2 for symmetric within-group pairs). So the majority vote IS in Family A by its (S, U, C) values.

**BUT:** The Family A definition says S ≤ 1 - m³/P³. If S means between-group stability, the majority vote has S > 1 - m³/P³ for large M (see Finding 2.3). If S means Spearman, the majority vote HAS S ≤ 1 - m³/P³ because within-group randomness dominates.

**Impact:** This is a consequence of Finding 2.3. If Finding 2.3 is fixed (S = Spearman), this finding is resolved.

### Finding 2.5: F1 Proposition S3 first-stump bound — ε₀ arithmetic [LOW]

**Location:** supplement.tex:~160-170 (first-stump proposition)

**The claim:** α₁(n) = (2/π)(1 - (π-2)/n + O(n⁻²))

**Check:** The expansion is: α(δ) = 2/π · (1 - δ²(π-2)/σ² + O(δ⁴)). With E[δ²] = πσ²/(2n): E[α₁] = 2/π · (1 - (π-2)·π/(2n) + ...) = 2/π · (1 - π(π-2)/(2n)).

The paper writes "(1 - (π-2)/n)" but the correct expression is "(1 - π(π-2)/(2n))". Since π(π-2)/(2) ≈ 1.80, versus (π-2) ≈ 1.14, the paper understates the correction by a factor of ~1.6.

**Impact:** The conclusion ("negligible for n=2000") is unchanged — both give corrections of O(0.001). But the formula is wrong.

**Fix:** Replace (π-2)/n with π(π-2)/(2n).

---

## Audit 4: Overclaims

### Finding 4.1: "first formally verified impossibility in XAI" [MEDIUM]

**Location:** main.tex:57, 86

**The claim:** "constituting the first formally verified impossibility result in explainable AI"

**Evidence:** We have no evidence anyone else has done this. Line 86 says "to our knowledge" — the abstract line 57 does NOT have this qualifier.

**Fix:** Add "to our knowledge" to line 57, consistent with line 86.

### Finding 4.2: "33 substantive theorems" — classification needed [MEDIUM]

**Location:** CLAUDE.md, main text

**The claim:** "33 substantive theorems"

**The question:** Are all 33 genuinely substantive? Candidates for "trivial":
- `consensus_variance_rate`: directly applies an axiom (1 line: `exact consensus_variance_bound`)
- `consensus_variance_nonneg`: `div_nonneg` of nonneg axiom
- `infeasible_faithful_stable_complete`: direct call to `attribution_impossibility`
- `dash_variance_decreases`: direct call to `consensus_variance_rate`

At least 4 of the 33 are essentially wrapper theorems that call another theorem in 1 line. A reasonable count: ~29 substantive + 4 trivial wrappers.

**Fix:** Either say "29 substantive theorems" (excluding wrappers) or "33 theorems including convenience lemmas."

### Finding 4.3: "DASH is provably optimal" [MEDIUM]

**Location:** main.tex §7, supplement F2

**The claim:** "DASH(M) achieves this and is Pareto-optimal"

**What the evidence supports:** DASH is optimal among unbiased estimators of E[φ_j] for i.i.d. models with finite variance satisfying Cramér-Rao regularity. Not optimal among ALL methods (biased methods, nonlinear methods, methods that use side information).

**Impact:** The claim is correct within the stated scope but the main text §7 doesn't repeat the scope restriction.

**Fix:** Add "among unbiased aggregations" to the main text.

### Finding 4.4: "design space collapses to a single axis" [HIGH]

**Location:** main.tex:438, supplement Design Space Theorem

**The claim:** The design space collapses to ensemble size M.

**The problem:** This is true IF the only choice is between Family A and Family B. But there are intermediate methods (e.g., partial completion: rank some within-group pairs where the evidence is strong, tie the rest). The design space is actually 2D: (M, completion threshold). The "single axis" claim holds only when the practitioner has already chosen either "full completion" or "no within-group completion."

**Fix:** Acknowledge the completion threshold as a second dimension, or explicitly state the collapse is among methods that are either fully complete or produce full ties.

### Finding 4.5: The Arrow parallel [LOW]

**Location:** main.tex:200-205, 444-446

**The claim:** Structural parallel to Arrow's impossibility

**Assessment:** The parallel IS genuine: both are impossibility theorems for aggregating conflicting orderings, both have the same structure (3 desiderata, prove mutual incompatibility), both are resolved by relaxing completeness. The path convergence having no Arrow analogue is an honest distinction.

**However:** Arrow's theorem is about aggregating PREFERENCES of different agents. Our theorem is about the INSTABILITY of a single quantity (feature importance) across model instances. The "voters" in our case are not agents with preferences but random draws from a distribution. This is a weaker parallel than the paper suggests — Arrow's impossibility is about preference aggregation (a design problem), ours is about statistical estimation (a measurement problem).

**Fix:** Note the distinction: "The structural parallel holds at the formal level; the substantive interpretation differs: Arrow's is about aggregating heterogeneous preferences, ours about aggregating noisy measurements of the same underlying quantity."

---

## Audit 3: Empirical Integrity (via agent)

### Finding 3.1: F1 r=-0.89 is ~40% tautological [HIGH]
Random standard-normal attribution arrays produce r ≈ -0.56 between Z and flip rate. This is structural: Z and flip rate are both functions of the same difference vector. Of the reported R² = 0.79, approximately 0.32 (40%) is tautological. The incremental signal from real data: ~0.47 R² points.

**Fix:** Report the random baseline r ≈ -0.56 in the supplement. The honest claim: "real data increases the correlation from the structural baseline of -0.56 to -0.89."

### Finding 3.2: Financial case study threshold shopping [MEDIUM]
German Credit has only 1 pair at |ρ| > 0.5. The script uses |ρ| > 0.3 to find 4 groups. This is acknowledged in a comment but not in the paper text.

**Fix:** State explicitly that the threshold was lowered for demonstration purposes, or use a dataset with genuine collinearity.

### Finding 3.3: SHAP computed on varying test sets [MEDIUM]
Each of the 50 models uses a different train/test split. SHAP is computed on seed-specific test sets. This conflates model instability with evaluation-set variation, inflating flip rates.

**Fix:** Acknowledge this in the experimental details, or add a fixed-evaluation-set robustness check.

### Finding 3.4: Fixed hyperparameters across 11 datasets [MEDIUM]
Same XGBoost config (100 trees, depth 6, lr 0.1) for all datasets from Iris (4 features) to Communities (126 features). Cross-dataset flip rate comparisons are confounded.

**Fix:** Add caveat that absolute flip rates are not comparable across datasets.

### Finding 3.5: Supplement table numbers match JSON [CLEAN]
All 11 datasets verified internally consistent.

---

## Audit 5: Missing Related Work (via agent)

### Finding 5.1: No scoop found [CLEAN]
No paper proves the specific faithfulness-stability-completeness trilemma under collinearity with quantitative bounds and formal verification. The core result is novel.

### Finding 5.2: Three missing citations [MEDIUM]
Should add:
- **Decker et al. (ICML 2024)**: "Provably Better Explanations with Optimized Aggregation" — aggregates across methods (not models). Distinguish explicitly.
- **Jin et al. (NeurIPS 2025)**: "Probabilistic Stability Guarantees" — constructive stability certification. Complement to our impossibility.
- **Noguer I Alonso (SSRN 2025)**: "Mathematical Foundations of Explainability" — fidelity-stability-comprehensibility trade-offs. Different mechanism (complexity vs collinearity).

### Finding 5.3: Positioning risk with Laberge et al. [MEDIUM]
Laberge et al. (JMLR 2023) already observed that Rashomon sets produce partial orders (i.e., completeness should be abandoned). Our contribution over theirs: (a) prove this is necessary, (b) quantify the divergence, (c) show DASH is optimal. Currently cited but distinction could be sharper.

---

## Audit 6: Nature Machine Intelligence Lens

### Finding 6.1: Accessibility [HIGH for NMI]

**The paper is written for ML researchers**, not for the broad NMI audience (biologists, physicists, policymakers). Specific issues:
- The abstract mentions "SHAP values," "Rashomon property," "collinearity" — jargon unfamiliar outside ML
- The Setup section (§2) assumes knowledge of gradient boosting, split counts, first-mover effects
- No motivating example in the introduction (e.g., "imagine a doctor using an AI that blames blood pressure for a prediction on Monday and cholesterol on Tuesday")

**Fix for NMI:** Rewrite the introduction with a concrete motivating scenario. Move technical setup to methods. Lead with the impossibility result and its implications, not the machinery.

### Finding 6.2: EU AI Act framing [MEDIUM]

**Location:** main.tex:461-464, supplement:~1360

**Assessment:** The EU AI Act reference (Art. 13(3)(b)(ii)) is CORRECT and SUBSTANTIVE. The paper correctly identifies attribution instability as a "known and foreseeable circumstance." The regulatory response template for ties (supplement) is excellent.

**For NMI:** The regulatory angle is the strongest hook for NMI. It should be expanded, not just a paragraph. A standalone "Implications for AI Regulation" section would strengthen an NMI submission.

### Finding 6.3: What a Nature editor would cut [MEDIUM]

The current 10+29 pages would need to become ~4000 words + methods + supplement for NMI. The editor would:
- Cut the entire Setup section (move to methods)
- Cut the formal axiom system
- Cut the Lean formalization details (mention in one sentence)
- Keep: impossibility theorem, the "two families" result, the F1 diagnostic, the financial case study, the regulatory implications
- Expand: motivating example, broader implications, practitioner guidance

---

## Audit 7: The "Foundational" Test

### Finding 7.1: What is our unique reusable technique? [CRITICAL for "foundational"]

Arrow introduced the ultrafilter lemma. Chouldechova introduced the calibration-balance quantitative tradeoff curve. What do WE introduce?

**Candidates:**
1. **The Rashomon-to-impossibility reduction**: If a model class has the Rashomon property, then faithfulness + stability + completeness is impossible. But the Rashomon property was articulated by Rudin (2024) and Fisher et al. (2019). Our contribution is connecting it to impossibility, but the connection is a 4-line proof.

2. **The IterativeOptimizer abstraction**: Dominance + surjectivity → Rashomon. This IS original and could be reused for other impossibility results. But it's a definition, not a technique.

3. **The Design Space Theorem**: Two families, Pareto frontier, M-axis collapse. This IS structural and could be reused for other aggregation problems. But it has the S-definition issue (Finding 2.3).

4. **The F1 diagnostic**: Z_jk test statistic for attribution instability. This IS practical and will be adopted. But it's a standard z-test, not a new technique.

**Honest assessment:** Our unique contribution is the FRAMEWORK — connecting Rashomon to impossibility to design space to optimal resolution. The individual pieces use standard techniques. The novelty is in the ASSEMBLY, not in any single component.

**Is this "foundational"?** It depends on the definition. If foundational means "introduces a technique others reuse" (Arrow, Nash), then NO — we don't introduce a new technique. If foundational means "establishes a structural understanding that changes how people think about a problem" (Chouldechova, our closest comparison), then POTENTIALLY YES — the Design Space Theorem, if correctly stated, does characterize what's achievable.

### Finding 7.2: Comparison to Arrow [HIGH]

| Dimension | Arrow | Us |
|-----------|-------|----|
| Proof technique | Ultrafilter (new) | Contradiction from Rashomon (standard) |
| Quantitative tradeoff | None (pure impossibility) | Yes (ratio, variance) |
| Resolution | Partial orders, Borda | DASH (ensemble averaging) |
| Formal verification | Nipkow (2009, Isabelle) | This paper (Lean) |
| Design space | 3 distinct relaxation paths | 2 converging paths |
| Practical impact | Changed voting theory | TBD |

**Arrow's advantage:** Novel proof technique, 75+ years of impact.
**Our advantage:** Quantitative bounds, constructive resolution, formal verification, practical diagnostic.

### Finding 7.3: Comparison to Chouldechova [MEDIUM]

| Dimension | Chouldechova | Us |
|-----------|-------------|----|
| Core result | Calibration + balance impossible | Faithfulness + stability + completeness impossible |
| Quantitative | Clean tradeoff curve (equation) | Ratio 1/(1-αρ²), variance O(1/M) |
| Resolution | Choose which to sacrifice | DASH (sacrifice completeness) |
| Audience | Fairness community (large) | XAI community (large) |
| Timing | 2017 (fairness debate peak) | 2026 (XAI regulation rising) |

**Chouldechova's advantage:** Single clean equation (calibration = f(base rate, FPR, FNR)). Immediate policy impact. Published in a medical informatics journal with broad audience.
**Our advantage:** Constructive resolution (DASH), formal verification, design space characterization, practical diagnostic.

**Is our result on the same level?** In mathematical depth: comparable (both are applications of pigeonhole/symmetry arguments). In immediate impact: Chouldechova had better timing and a cleaner one-equation result. In long-term potential: our design space characterization is structurally richer.

---

## Vet Round 1: Factual

- Finding 2.1 (ε₀ algebra): Verified computationally. The factor of 2^{3/2} IS dropped. Confirmed.
- Finding 2.3 (S definition): Verified by reading the supplement. S IS defined as "between-group stability" and the bound IS from the Spearman correlation. These ARE different metrics.
- Finding 2.5 (α₁ formula): Need to verify. The expansion step needs careful checking.
- Finding 4.2 (33 vs 29): Need agent results to confirm which theorems are trivial wrappers.

## Vet Round 2: Am I overclaiming about problems?

- Finding 2.3 (S definition mismatch): Am I making this bigger than it is? Let me reconsider. For the MAIN TEXT §7, S is described as "stability S" without specifying between-group or full-ranking. In the supplement, S is defined as between-group. The Spearman bound (main text Theorem 2) IS about full-ranking. So there IS a mismatch, but it could be fixed by clarifying the definition. **Not overclaiming — this is a real issue.**

- Finding 4.4 (single axis overclaim): Am I inventing intermediate methods that don't exist in practice? The "partial completion" method (rank some within-group pairs, tie others) IS a real possibility. Practitioners DO report partial rankings (e.g., "top 5 features" without fully ordering features 6-20). **Not overclaiming — this is a real gap.**

- Finding 7.1 (no unique technique): Am I being too harsh? Arrow's ultrafilter lemma is elegant but Arrow himself didn't use it — it was discovered later by others analyzing his proof. Arrow's ORIGINAL proof used a step-by-step construction. Maybe the right comparison is to Arrow's original proof style, which was also "straightforward." **Possibly underclaiming — our IterativeOptimizer abstraction IS a reusable conceptual tool, even if it's not a "technique" in the narrow sense.**

## Vet Round 3: Omissions

- I didn't check whether the information loss formula is correct (Audit 2 should cover the math in the info loss section).
- I didn't verify the first-stump proposition's expansion formula against a CAS.
- Missing: whether the supplement's Proposition numbers are consistent (do theorem environments reset across sections?).

---

## Ranked Action List for Best Paper

### CRITICAL (must fix — blocks acceptance)

1. **Fix Design Space Theorem S definition** (2.3): Redefine S as full-ranking expected Spearman, or derive the Family A bound for between-group S separately.
2. **Disclose F1 tautological baseline** (3.1): Report that random data gives r ≈ -0.56. The incremental signal is r: -0.56 → -0.89.
3. **Distinguish "proved" from "axiomatized" in Lean claims** (1.2, 1.5): The DASH resolution (Corollary 1) and variance bound are axiomatized. Say "verified by Lean's type checker" not "proved in Lean" for axiom-dependent results.

### HIGH (should fix — weakens the paper significantly)

4. **Fix F3 ε₀ algebra** (2.1): Correct the dropped 2^{3/2} factor.
5. **Honest theorem count** (1.3, 4.2): ~24 substantive + ~9 convenience/trivial. Don't claim all 33 are substantive.
6. **Note Spearman bound uses axiom** (1.4): The m³/P³ constant is axiomatized; the derived bound is 3/(P³-P).
7. **Acknowledge design space is 2D** (4.4): M + completion threshold.
8. **Add DASH optimality scope** (4.3): "among unbiased aggregations" in main text.
9. **Fix first-stump formula** (2.5): π(π-2)/(2n) not (π-2)/n.
10. **Add 3 missing citations** (5.2): Decker et al., Jin et al., Noguer I Alonso.

### MEDIUM (nice to fix)

11. **Financial case study threshold** (3.2): Acknowledge |ρ|>0.3 is non-standard.
12. **SHAP varying test sets** (3.3): Acknowledge or add fixed-eval robustness check.
13. **Abstract qualifier** (4.1): Add "to our knowledge" to line 57.
14. **Berry-Esseen footnote** (2.2).
15. **Arrow parallel nuance** (4.5).
16. **Sharpen Laberge distinction** (5.3).
17. **Fixed hyperparameters caveat** (3.4).

---

## Phased Implementation Plan

### Phase I: Critical Fixes (2-3 hours)

**Task 1: Fix Design Space Theorem S definition.**
- Redefine S as expected Spearman correlation between independent evaluations (full-ranking stability). This makes Family A's bound S ≤ 1 - m³/P³ correct by construction.
- Update: supplement.tex Definition (~line 1586), main.tex §7 (~line 433).
- Verify the proof still holds. (It does: the Spearman bound IS about full-ranking stability.)

**Task 2: Disclose F1 tautological baseline.**
- Add to supplement F1 section: "A null model (random attributions) produces r ≈ -0.56 between Z and flip rate, reflecting the structural dependence between the two metrics. The observed r = -0.89 exceeds this baseline, indicating genuine attribution instability beyond what noise alone would produce."
- Consider reporting partial correlation or excess R².

**Task 3: Distinguish "proved" from "axiomatized" in the paper.**
- main.tex: Change "29 substantive theorems verified by the Lean 4 type checker" to "24 substantive theorems proved and 9 verified lemmas (depending on 15 axioms), with 0 sorry"
- OR: Keep 33 total but add: "of which 24 are non-trivial proofs and 9 are convenience lemmas or axiom restatements"
- Supplement: Add a table classifying each theorem as "proved from axioms" vs "essentially axiomatized"
- Key honesty: Corollary 1 (DASH equity) depends directly on the attribution_sum_symmetric axiom. The variance bound depends on consensus_variance_bound axiom. State this clearly.

### Phase II: High Priority Fixes (1-2 hours)

**Task 4: Fix F3 ε₀ algebra.**
- supplement.tex:672-675: Replace intermediate step with correct computation showing the 2^{3/2} factor.

**Task 5: Fix first-stump formula.**
- supplement.tex: Replace (π-2)/n with π(π-2)/(2n).

**Task 6: Note Spearman bound axiom status.**
- main.tex or supplement: Note that the m³/P³ constant is axiomatized; the fully derived bound is 3/(P³-P).

**Task 7: Add DASH optimality scope to main text.**
- main.tex §7: Add "among unbiased aggregations of per-model attributions."

**Task 8: Acknowledge design space 2D.**
- main.tex §7 or supplement: "The design space has a second dimension—the completion threshold—but at the extremes (full completion or full ties), the M-axis is the unique parameter."

**Task 9: Add 3 missing citations.**
- references.bib: Decker et al. (ICML 2024), Jin et al. (NeurIPS 2025), Noguer I Alonso (SSRN 2025)
- Related Work §8: Brief positioning sentences for each.

### Phase III: Medium Priority (30 min)

**Task 10:** Financial threshold: add "threshold lowered to 0.3 for demonstration; only 1 pair exceeds 0.5."
**Task 11:** SHAP varying test sets: add acknowledgment to experimental details.
**Task 12:** Abstract: add "to our knowledge" to line 57.
**Task 13:** Berry-Esseen: add footnote about n=1 non-standard usage.
**Task 14:** Arrow parallel: add preference-aggregation vs measurement-estimation note.
**Task 15:** Sharpen Laberge distinction in Related Work.
**Task 16:** Fixed hyperparameters: add caveat about cross-dataset comparisons.

---

## /vet of this plan

### Round 1: Factual
- The r≈-0.56 random baseline is from the agent's experiment. Should be verified independently.
- The 24 vs 33 theorem classification should be verified against the agent's list.
- The S-definition fix (Spearman instead of between-group) needs to be checked against EVERY reference to S in both papers.

### Round 2: Am I overclaiming about problems?
- Finding 1.2 (consensus_equity is axiomatized): The paper DOES say "axiomatized directly" in the proof text. The issue is whether calling it "Corollary 1" implies it's derived. This is a PRESENTATION issue, not a mathematical error. **Downgrade from CRITICAL to HIGH** — the paper is honest about what's axiomatized, just not in the numbering.
- Finding 3.1 (tautological r=-0.89): r=-0.56 from random data IS expected — Z and flip rate are both monotone functions of |μ/σ|, so any distribution will show negative correlation. The QUESTION is whether r=-0.89 exceeds what's structurally expected. The agent says yes (0.79 R² vs 0.32 baseline). This is a real finding but the paper's claim is not wrong — it's just incomplete without the baseline. **Keep as HIGH.**

### Round 3: Omissions in the plan
- I don't address the split-count axiom gap (features outside first-mover's group unconstrained). This is a real issue but doesn't affect any theorem — those features aren't used in the proofs.
- I don't address the NMI rewrite (Audit 6). This is deferred — NeurIPS first.
- The plan doesn't include re-running `lake build` after changes. Add this as a verification step.

### Corrections
- ⚠️ Downgraded Finding 1.2 from CRITICAL to HIGH (paper is honest about axiomatization in proof text).
- ⚠️ Added `lake build` verification to Phase I.

**Confidence: HIGH that the S-definition mismatch and F1 tautological baseline are the two most important findings. Everything else is presentation/honesty tuning.**
