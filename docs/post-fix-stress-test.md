# Post-Fix Axiom Stress Test: Phases 2-4

Analysis of `/paper/main_jmlr.tex` (42pp JMLR version), Lean source (36 files), and `trilemma.tex` figure.

Date: 2026-04-03

---

## Phase 2: New Attack Surface

### 2.1 Trilemma Figure Accuracy

**Source:** `paper/figures/trilemma.tex`

The figure has two panels:
- **Panel A** (triangle): Vertices are Faithful, Stable, Complete. Center has red X with "Impossible". Left edge (Faithful--Stable) is labeled "Family B (DASH) / stable + ties". Right edge (Faithful--Complete) is labeled "Family A / unstable (50% flips)". Bottom edge (Stable--Complete) is labeled "Drop faithful / trivial".
- **Panel B** (bar charts): Before DASH shows Model 1 (A tall, B short) and Model 2 (reversed) with "ranking flipped!" annotation. After DASH shows equal bars labeled "tied (honest)".

**Analysis:**
- Family A label "unstable (50% flips)" matches Theorem `unfaithfulness_bound` (U=1/2 for symmetric pairs). CORRECT.
- Family B label "stable + ties" matches `consensus_equity` (DASH produces ties). CORRECT.
- Family B is on the Faithful--Stable edge, meaning it drops Complete. This is accurate: DASH is faithful (to the population) and stable, but produces within-group ties (incomplete). CORRECT.
- "Drop faithful -> trivial" on the bottom edge (Stable+Complete). The paper proves any stable+complete ranking has U=1/2 (`stable_complete_unfaithful`), making it unfaithful to half the models. Calling this "trivial" is a characterization choice -- the label could be read as "trivially bad" or "trivially achievable." The actual content is that a random fixed ranking is stable and complete but unfaithful. DEFENSIBLE but slightly imprecise -- a reviewer might ask "trivial in what sense?"
- The figure does NOT show the U values on the edges. Family A has U=1/2 (shown as "50% flips" implying instability, not unfaithfulness). Strictly, the "50% flips" refers to ranking instability across seeds, which is a different metric from unfaithfulness U. The 50% comes from the flip rate (Proposition `exact_flip`), not from the U metric per se, though they are related.

**Attack:** "The figure conflates instability (rankings flip across models) with unfaithfulness (ranking disagrees with a specific model). Family A is described as 'unstable (50% flips)' but on the Faithful--Complete edge. If it's faithful and complete, the instability is the price. The label should say 'S=1-m^3/P^3' not '50% flips' since the flip rate depends on group size."

**Defense:** The figure is a conceptual summary, not a formal statement. The caption says "rankings flip up to 50% of the time" and the 50% is the worst case (m=2). The formal details are in the theorem statements. The figure's purpose is pedagogical.

**Verdict: DEFENSIBLE (4/5).** Minor imprecision in conflating instability and flip rate, but the figure is pedagogically accurate and the caption is careful.

---

### 2.2 Progressive DASH

**Source:** Line 587-588 of `main_jmlr.tex`

The paper describes Progressive DASH in a single paragraph: "start with M=5 models; for pairs with Z > 3.0, declare stable; for Z < 1.96, flag as unstable; for borderline pairs, train 5 more and repeat. Average-case cost: ~8x (not 25x)."

**Analysis:** This is presented as a heuristic ("we propose"), not as a theorem. There is:
- No formal definition of the adaptive stopping rule
- No proof of any optimality property
- No worst-case cost analysis
- No Lean formalization
- The "~8x" average-case claim is stated without derivation or simulation

**Attack:** "Progressive DASH is described as a resolution to the 25x cost problem, but there is no analysis of Type I/II error for the adaptive procedure, no proof that the average-case cost is 8x, and no guarantee that early stopping does not inflate the false discovery rate. Sequential testing with data-dependent stopping is known to inflate Type I error (Pocock bounds, O'Brien-Fleming). Your Z-thresholds (1.96 and 3.0) are the fixed-sample values; they are NOT valid for group-sequential designs."

**Defense:** The paper does not claim Progressive DASH is optimal or formally analyzed. It is explicitly "proposed" as a practical heuristic. The core result (DASH with fixed M) has full formal backing. Progressive DASH is a cost-reduction suggestion, not a contribution.

**Verdict: DEFENSIBLE (3/5).** The sequential testing concern is valid. A reviewer could ask for Pocock-corrected thresholds or a simulation study. The paper should either add a sentence acknowledging sequential testing inflation or remove the specific thresholds.

---

### 2.3 Adverse Action Example

**Source:** Lines 107-113 (introduction), 1000-1001 (worked example), 1528 (German Credit)

The paper uses ECOA adverse action as a motivating example:
- "Under ECOA, the lender's adverse action notice changes based solely on the training randomness" (line 112-113)
- "Under ECOA, the lender must disclose the primary reason. The disclosure depends on the random seed -- a regulatory violation that is mathematically inevitable under collinearity" (line 1001)

**Analysis:**

**Attack:** "ECOA requires disclosure of 'principal reasons' for adverse action (12 CFR 1002.9(b)(2)). But a production lender uses ONE model with a fixed seed, not a retrained model. The adverse action notice is deterministic for a given deployed model. Your impossibility applies to the model class, not to the deployed model. A defense attorney would argue: 'We have one model, it ranks income above DTI, and the adverse action notice correctly reflects that model. The impossibility is about hypothetical alternative models we never deployed.' The paper conflates 'what could have happened with a different seed' with 'what the regulation requires.' ECOA does not require stability across hypothetical retrainings."

**Defense:** The paper explicitly addresses this in line 1838: "A fixed pipeline with a fixed seed produces a deterministic, reproducible ranking. The instability arises upon retraining, auditing, or model comparison -- all routine in regulated settings." ECOA requires the reasons to be "specific" and the OCC/Federal Reserve require model risk management (SR 11-7) that includes challenger model analysis. When a challenger model with a different seed produces a different adverse action reason, the bank faces a model risk management question. The paper does not claim a single model violates ECOA; it claims the model selection process makes the disclosure arbitrary.

**Verdict: DEFENSIBLE (3/5).** The legal framing is aggressive but not wrong. The paper should be more precise: the violation is not ECOA per se but the model risk management obligation under SR 11-7 to demonstrate that model outputs are robust to reasonable perturbations. A legal reviewer would distinguish "the model violates ECOA" from "the model governance process fails to detect instability."

---

### 2.4 Regulatory Annex (EU AI Act)

**Source:** Lines 129, 992, 1299, 1833

The paper references:
- "EU AI Act (Art. 13(3)(b)(ii)), which requires disclosing 'known and foreseeable circumstances' affecting accuracy" (line 129)
- "Under the EU AI Act Art. 13(3)(b)(ii), providers must disclose 'known and foreseeable circumstances' affecting accuracy" (line 992)

**Analysis:** Art. 13 GDPR is about the right to explanation; Art. 13 of the EU AI Act is about transparency. The actual text of the EU AI Act (Regulation (EU) 2024/1689) Art. 13(3)(b)(ii) requires high-risk AI systems to include instructions for use that contain "where appropriate, the known and foreseeable circumstances related to the use of the high-risk AI system... which may lead to risks to health, safety or fundamental rights." The paper paraphrases as "affecting accuracy" -- the actual text says "risks to health, safety or fundamental rights."

**Attack:** "The actual Art. 13(3)(b)(ii) text is about risks to health/safety/fundamental rights, not about 'accuracy.' Attribution instability may or may not qualify as such a risk depending on context. The paper's paraphrase inflates the regulatory relevance."

**Defense:** In the context of a credit model used for adverse action, attribution instability that causes different explanations for the same decision plausibly creates risks to fundamental rights (non-discrimination, right to explanation). The paraphrase is a simplification, not a fabrication. No Art. 14 or Annex III references were found, so there is no overreach beyond Art. 13.

**Verdict: DEFENSIBLE (3/5).** The paraphrase simplifies but does not fabricate. A legal expert reviewer could quibble. The paper should say "circumstances that may lead to risks" rather than "affecting accuracy."

---

### 2.5 Count Consistency

**Actual counts from Lean source code:**
- **Lean files:** 36
- **Axioms:** 17 (14 in Defs.lean, 1 in SpearmanDef.lean, 2 in QueryComplexity.lean)
- **Theorems + lemmas:** 190

**Paper claims (all occurrences):**

| Location | Theorems | Axioms | Notes |
|----------|----------|--------|-------|
| Abstract (line 58) | 188 | 15+3=18 | |
| Contributions (line 135) | 188 | 15+3=18 | |
| Contributions (line 147) | 188 | 18 | |
| Axiom inventory (line 205) | -- | 18 (6+2+7+3) | Breakdown adds to 18 |
| Table 1 (line 229) | -- | -- | Lists `le_cam_lower_bound` as axiom |
| Lean section (line 1658) | 188 | 18 | |
| Table caption (line 1743) | 188 | 18 | |
| Proof status (line 1850) | 188 | -- | |

**Discrepancies:**
1. **Theorem count: Paper says 188, actual is 190.** The paper is STALE by 2 theorems. Likely added during recent fixes without updating counts.
2. **Axiom count: Paper says 18, actual is 17.** `le_cam_lower_bound` was converted from axiom to theorem (see Phase 2.6), but the paper still counts it as an axiom. Table 1 (line 229) explicitly lists it in the axiom table.
3. **No occurrences of "190" referring to theorem counts.** (The "190" hit at line 1632 is "Credit-g | 20 | 190 | 47" -- dataset size.)
4. **No remaining "188" that is incorrect** -- all 188 references are consistently wrong.
5. **The axiom breakdown "6 type/constant + 2 measure + 7 domain + 3 query = 18" is wrong.** It should be: 6 type/constant (Model, numTrees, numTrees_pos, attribution, splitCount, firstMover) + 2 measure (modelMeasurableSpace, modelMeasure) + 7 domain property (firstMover_surjective, splitCount_firstMover, splitCount_nonFirstMover, proportionality_global, splitCount_crossGroup_symmetric, splitCount_crossGroup_stable, spearman_classical_bound) + 2 query (testing_constant, testing_constant_pos) = **17 total**.

**Verdict: VULNERABLE (2/5).** The paper has stale counts. Both theorem count (188 vs 190) and axiom count (18 vs 17) are wrong. A reviewer who clones the repo and counts will find inconsistencies. The `le_cam_lower_bound` is listed as an axiom in Table 1 but is demonstrably a `theorem` in the code. This undermines the "transparency" narrative. **Requires immediate fix before submission.**

---

### 2.6 le_cam_lower_bound Reclassification

**Source:** `DASHImpossibility/QueryComplexity.lean` lines 51-54

```lean
theorem le_cam_lower_bound (σ Δ : ℝ) (_hσ : 0 < σ) (_hΔ : 0 < Δ) (n : ℕ)
    (h_reliable : (n : ℝ) < testing_constant * σ ^ 2 / Δ ^ 2 → False) :
    testing_constant * σ ^ 2 / Δ ^ 2 ≤ n :=
  not_lt.mp h_reliable
```

`le_cam_lower_bound` is now a **theorem**, not an axiom. The proof is `not_lt.mp h_reliable` -- a one-line proof from the contrapositive. The file header explains: "the contrapositive formulation `not(n < bound) -> bound <= n` is provable from `not_lt` in any linear order."

**Impact on axiom defense:**
- Previous count: 18 axioms (15 domain + 3 query-complexity)
- New count: **17 axioms** (15 domain + 2 query-complexity: `testing_constant` and `testing_constant_pos`)
- This is an IMPROVEMENT. The Le Cam content is now encoded in the hypothesis `h_reliable` (which the caller must discharge), not in the axiom system.
- The remaining 2 query-complexity axioms (`testing_constant : R` and `testing_constant_pos : 0 < testing_constant`) assert only the existence and positivity of a universal constant, which is trivially true (C = 1/8 works).

**Verdict: IMPROVED.** The axiom-laundering defense is strictly stronger with 17 axioms than with 18. The query-complexity axioms are now just "there exists a positive real number" -- essentially zero content. All the Le Cam substance is in the hypothesis structure, not in axioms.

---

## Phase 3: Cold Reviewer Simulation

### Reviewer Profile
JMLR reviewer. Associate professor in ML theory or statistics. 3 weeks to review. Reads the full 42 pages.

### Pages 1-5: Introduction + Setup

**First impression:** The hook is strong -- "Train a GBDT, compute SHAP, retrain, top feature changes" is immediately relatable. The "60% of datasets" claim is backed later. The "coin flip" framing is vivid. The reviewer notes the Arrow's theorem parallel and thinks "ambitious but potentially overblown."

The "deliberately simple" defense (line 150-152) is noticed and appreciated -- the authors are preempting the "4-line proof is trivial" criticism. The "Pick Two" line (line 270) is memorable. The 4-persona framing is not explicit in the text but the paper addresses classical statistician, causal, DL, and industry perspectives implicitly through the extensions.

**Concern:** The abstract mentions "188 theorems from 18 axioms" -- the reviewer mentally flags this for later verification. The introduction is dense (2+ pages) but well-structured. The notation table is helpful.

**Rating at this point:** Cautiously positive. The scope is enormous for a single paper.

### Pages 6-15: Impossibility + Bounds + DASH

**The 4-line proof (lines 309-317):** The reviewer reads this and thinks: "This is clean. But is it too clean? The Rashomon property does all the work." They check Definition 5 (Rashomon property) and recognize it as the key assumption. They note the "zero axiom dependencies" claim and mentally verify: yes, if you accept the Rashomon property as hypothesis, the impossibility is trivially true. This bothers them slightly but they recall Arrow's theorem has the same structure.

**The ratio 1/(1-rho^2) (lines 421-437):** Clean derivation from the split-count axioms. The reviewer checks the proportionality axiom -- CV=0.35 for stumps, 0.66 for depth-6. They think "order-of-magnitude" is a generous description for CV=0.66. The alpha-correction (alpha=2/pi, R^2=0.89) is reassuring. The depth-3 anomaly (alpha=0.30) makes them question the generality.

**DASH Pareto optimality (lines 602-636):** This is substantial. Cramer-Rao lower bound + DASH achieves it. The reviewer appreciates the Rao-Blackwell argument. They note Part III (Pareto optimality) has a slightly informal proof -- "no method A' simultaneously achieves U<0 and S > S_DASH" -- and think "U < 0 is impossible by definition, so this is vacuously true." Wait -- U is unfaithfulness, defined as fraction disagreeing, so U >= 0 always. The Pareto claim is really: "no method achieves lower variance than DASH among unbiased methods." This IS just Cramer-Rao. Correct but standard.

**Rating at this point:** Solid theory section. The impossibility is real but simple; the quantitative bounds add genuine content.

### Pages 16-25: Design Space + SBD

**Design Space Theorem (lines 706-736):** This is the paper's main claim. The exhaustiveness proof (Step 3) is the key non-trivial part. The reviewer checks it carefully: Case 1 (faithful + complete -> Family A) is immediate from the impossibility. Case 2 (not faithful -> optimal unfaithful is ties for symmetric features) uses the DGP symmetry. Case 3 (aggregate M > 1 + complete -> U > 0) uses the same argument. The logic is sound but the "no method outside A union B" claim relies on the assumption that the method operates on per-model attributions. A method that uses gradient information, loss curvature, or neural network internals (not just phi_j(f)) could potentially escape. The paper's scope is methods based on per-model attributions -- this should be more prominent.

**Symmetric Bayes Dichotomy (lines 803-901):** Three instances (attribution, model selection, causal discovery). The reviewer recognizes invariant decision theory (Lehmann-Casella). The novel claim is the "reduction template." Instance 3 (causal discovery / CPDAG) with variable-size symmetry group is the most interesting, showing the technique is not just binary. The reviewer thinks "this is a genuine contribution -- connecting attribution impossibility to Markov equivalence classes via a common group-theoretic framework."

**Lean claims:** The reviewer notes the file tree (lines 1660-1687) and the cross-reference table (lines 1746-1779). They consider cloning the repo but decide to trust the "0 sorry" claim for now. If they clone it, `lake build` would take ~45 minutes, which is feasible.

**Rating at this point:** Growing respect for the scope. The SBD with 3 instances is genuine technique.

### Pages 26-35: Extensions + Diagnostics + Experiments

**Conditional SHAP (lines 912-968):** The threshold table is very useful. The "no intermediate regime" finding (both snap at delta_beta = 0.20 for rho=0.9) is surprising and important. The reviewer wishes for a formal theorem about the threshold, not just a table.

**Diagnostics (F1 and F5):** Well-designed. F1 is theory-driven (Z-test), F5 is practical (split-frequency). The Berry-Esseen completeness guarantee (line 1174) is a nice touch. The restricted-range analysis (line 1202) showing r = -0.53 for Z < 2 is honest -- the correlation weakens for the hardest cases.

**Experiments:** 11 datasets, 3 implementations (XGBoost, LightGBM, CatBoost), NN (87% unstable), permutation importance (91% unstable). The cross-method correlation (r=0.46) confirming the instability is feature-driven, not method-driven. This is thorough.

**What they want more of:** (1) A dataset where DASH clearly helps a downstream decision (not just lower flip rates). (2) The NN experiment is preliminary (KernelSHAP noise control is good but the model is just DistilBERT/SST-2). (3) A comparison with existing methods for handling multicollinearity in SHAP (e.g., SHAP with feature grouping, conditional SHAP in the Owen sense).

### Pages 36-42: Formalization + Related + Discussion

**190 theorems. 17 axioms.** (The paper says 188/18 -- if the reviewer counts, they find the discrepancy.) The proof depth distribution (75 multi-step, 21 with 10+ lines, 3 with 20+) is informative. The three bugs caught during formalization (lines 1715-1722) are a strong selling point.

The "proof status transparency" paragraph (line 1850) is excellent -- clearly delineating what is proved, derived, argued, and empirical. This is rare and the reviewer appreciates it.

**Do they clone the repo?** Maybe. If they do, they find 190 theorems (not 188) and 17 axioms (not 18). This inconsistency is minor but unprofessional.

### Recommendation

**Accept with Minor Revision.**

**Confidence: 4/5.** (High confidence in the core theory; moderate in the quantitative bounds and regulatory implications.)

**One biggest concern:** The theorem/axiom counts in the paper do not match the code. This is a symptom of rapid development without final reconciliation. If the counts are wrong, what else might be stale? The reviewer asks for a final pass reconciling all quantitative claims between paper and code.

**What would upgrade to "Accept":** Fix the counts. Add one sentence to Progressive DASH acknowledging sequential testing issues. Tone down the ECOA language from "regulatory violation" to "model governance concern." Verify all paper-code cross-references are accurate.

---

## Phase 4: Updated Vulnerability Map

| # | Item | Attack | Def. (1-5) | Verdict | Status | Action? |
|---|------|--------|------------|---------|--------|---------|
| 1 | **Count consistency** (188/18 vs 190/17) | Reviewer clones repo, counts disagree. "If counts are wrong, what else is stale?" | 2 | VULNERABLE | **NEW** | **FIX IMMEDIATELY.** Update abstract, contributions, Lean section, Table caption to say 190 theorems, 17 axioms. Remove `le_cam_lower_bound` from axiom table; move to "formerly axiomatized" section. |
| 2 | `proportionality_global` (CV=0.66) | "Global c with 66% variation is not a constant" | 3 | VULNERABLE | UNCHANGED | Emphasize core impossibility independent; validate for stumps only; state as order-of-magnitude |
| 3 | `spearman_classical_bound` (axiom-laundering) | Axiomatizes the real quantitative content; derived bound near-vacuous for small m/P | 3 | VULNERABLE | UNCHANGED | Close combinatorial gap or emphasize derived bound captures m-scaling |
| 4 | Axiom laundering narrative (Reviewer D) | Core proof is trivial; axioms encode the answer | 3 | VULNERABLE | UNCHANGED | Highlight non-trivial quantitative theorems, bug-catching, Consistency.lean |
| 5 | **Progressive DASH thresholds** | Sequential testing inflation: Z=1.96 and Z=3.0 are fixed-sample thresholds, not valid for group-sequential designs | 3 | DEFENSIBLE | **NEW** | Add caveat about sequential testing or remove specific thresholds; cite Pocock/O'Brien-Fleming |
| 6 | **ECOA/adverse action framing** | Single deployed model is deterministic; ECOA does not require stability across seeds | 3 | DEFENSIBLE | **NEW** | Soften from "regulatory violation" to "model risk management concern under SR 11-7" |
| 7 | **EU AI Act paraphrase** | Art. 13(3)(b)(ii) says "risks to health/safety/rights" not "affecting accuracy" | 3 | DEFENSIBLE | **NEW** | Fix paraphrase to match actual text |
| 8 | `firstMover_surjective` | Deterministic tie-breaking may violate | 4 | DEFENSIBLE | UNCHANGED | State colsample_bytree / bootstrap assumption |
| 9 | `splitCount_firstMover` / `_nonFirstMover` | Approximate for finite depth | 4 | DEFENSIBLE | UNCHANGED | Alpha-correction addresses |
| 10 | **Trilemma figure** | Minor: "50% flips" conflates instability and unfaithfulness; "trivial" is ambiguous | 4 | DEFENSIBLE | **NEW** | No action needed; caption is careful |
| 11 | Equicorrelation assumption | Unrealistic for heterogeneous correlations | 4 | DEFENSIBLE | UNCHANGED | Pairwise version (m=2) suffices |
| 12 | Balanced ensemble idealization | Exact balance rare in practice | 4 | DEFENSIBLE | UNCHANGED | O(1/M) convergence for approximate balance |
| 13 | Faithful definition (biconditional) | Straw man; alpha-faithfulness dissolves it | 4 | DEFENSIBLE | UNCHANGED | Weak version + quantitative bounds pre-empt |
| 14 | 25x compute cost | Impractical for expensive models | 4 | DEFENSIBLE | UNCHANGED | Information-theoretically necessary; audit not deploy |
| 15 | **le_cam_lower_bound reclassified** | Was axiom, now theorem | 5 | ROCK SOLID | **IMPROVED** | Strictly improves axiom defense (17 < 18). Update paper to reflect. |
| 16 | `Model : Type` | Could be empty | 5 | ROCK SOLID | UNCHANGED | Consistency witness (Fin 4) |
| 17 | `numTrees` / `numTrees_pos` | Trivial / fixed-T | 5 | ROCK SOLID | UNCHANGED | Core impossibility independent |
| 18 | Stable definition (model-independent) | Too strong | 5 | ROCK SOLID | UNCHANGED | Paper proves both absolute and epsilon-stability |
| 19 | Measure infrastructure axioms | Minimal content | 5 | ROCK SOLID | UNCHANGED | Standard Mathlib infrastructure |

### Summary

**Items IMPROVED since original audit:**
- `le_cam_lower_bound`: axiom -> theorem (17 axioms, not 18). Strictly strengthens the defense.

**NEW items (post-fix paper changes):**
- Count consistency (188/18 -> should be 190/17): **VULNERABLE, must fix**
- Progressive DASH sequential testing: DEFENSIBLE but needs caveat
- ECOA/adverse action framing: DEFENSIBLE but aggressive
- EU AI Act paraphrase: DEFENSIBLE but imprecise
- Trilemma figure: DEFENSIBLE, no action needed

**Critical action before submission:**
1. Update ALL theorem/axiom counts: 188 -> 190, 18 -> 17 (or 15+2)
2. Move `le_cam_lower_bound` from axiom table to "formerly axiomatized" section
3. Update axiom breakdown: "6 type/constant + 2 measure + 7 domain + 2 query = 17"
4. Add one sentence to Progressive DASH about sequential testing boundaries
5. Soften ECOA language from "regulatory violation" to "model governance concern"

**Overall assessment:** The paper is in strong shape. The core theory is sound. The vulnerabilities are all in the periphery (counts, legal framing, heuristic proposals). No reviewer is likely to reject on the basis of the theory. The count inconsistency is the most dangerous issue because it undermines the credibility narrative ("we machine-verified everything") -- if the paper cannot correctly report how many theorems exist, the trust premium of formal verification is eroded.
