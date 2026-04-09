# Best Paper Review: The Attribution Impossibility

> Five-reviewer assessment targeting NeurIPS 2026 Best Paper.
> Cross-checks: 10/10 pass. No claim-code mismatches.

---

## Reviewer 1: Fields Medalist

### Summary
A clean impossibility theorem for feature attribution rankings under
collinearity, with a zero-axiom core (Theorem 1) and quantitative
instantiations. The mathematical content is correct but THIN: the core
proof is 4 lines, and the quantitative bounds rely on axioms that
overestimate empirical behavior. The supplement's F1-F5 additions
significantly deepen the contribution, but they're proof sketches, not
rigorous theorems.

### Strengths
1. **Theorem 1's zero-axiom property** is the right design: the
   impossibility depends ONLY on the Rashomon property, making it
   maximally general. This is how impossibility theorems should be stated.
2. **The IterativeOptimizer abstraction (Definition 4)** cleanly unifies
   GBDT, Lasso, and NN. The factoring into dominance + surjectivity is
   elegant and mirrors the IIA + Pareto decomposition in Arrow's theorem.
3. **The path convergence (Theorem S7)** is the most mathematically
   interesting result. It has no Arrow analogue and suggests a structural
   difference between attribution impossibility and social choice
   impossibility. This deserves more development.
4. **F3 (FIM impossibility)** provides the "right" proof for a
   statistician. The connection FIM rank deficiency → Rashomon is the
   key insight that classical statisticians will recognize.

### Weaknesses

**W1 (MAJOR): The core impossibility (Theorem 1) is mathematically trivial.**
The proof: get two models from Rashomon with opposite orderings; assume
a faithful stable complete ranking; derive contradiction. This is the
textbook "preferences can't be aggregated" argument, translated to
attributions. Arrow's theorem is also short, but its PROOF TECHNIQUE
(ultrafilter argument, or the Geanakoplos elementary proof) introduced
new methods. Theorem 1 introduces no new technique — it's a direct
application of a known argument pattern.

*Evidence:* The Lean proof is 4 tactic lines. No mathematical ingenuity
is required.
*What would fix it:* The contribution is NOT Theorem 1 alone — it's the
FRAMEWORK (Rashomon → impossibility → architecture discrimination → DASH).
But the paper should acknowledge Theorem 1's simplicity rather than
presenting it as the main contribution.
*Justifies rejection?* No — the framework transcends any single theorem.

**W2 (MAJOR): The supplement's F1-F5 are proof sketches, not theorems.**
F1 (testable condition): the "only if" direction is approximate ("the
Rashomon property is PRACTICALLY significant iff..."). F2 (DASH
optimality): restricted to "unbiased linear aggregations." F3 (FIM):
relies on the quadratic approximation (small ε). F5 (diagnostic):
requires exchangeability that only holds under sub-sampling.

Each caveat is acknowledged, but NONE of F1-F5 is a clean, unconditional
theorem. They're directionally correct arguments with approximation gaps.
*Justifies rejection?* No for a conference paper; yes for JMLR.

**W3 (MINOR): There is no SINGLE theorem that subsumes everything.**
The ideal result would be: "The attribution design space under
collinearity is the one-parameter family {DASH(M) : M ∈ [1,∞)}, where
DASH(M) achieves (stability = 1-O(1/M), unfaithfulness = 0 within
groups, faithfulness = 1 between groups). This family is Pareto-optimal
and no point outside it is achievable." This would be one theorem with
F1, F2, F3, F5, and the path convergence as corollaries.

### Questions for Authors
1. Is there a deeper reason why the path convergence has no Arrow
   analogue? (In Arrow's setting, the three relaxation paths give
   different solutions because there's no underlying symmetry group.
   In your setting, the DGP symmetry forces convergence. Is this the
   complete explanation?)

### THE ONE THING for +2 points
Prove the SINGLE UNIFIED THEOREM described in W3, with F1-F5 as
corollaries. This would transform the paper from "a collection of
related results" into "one fundamental theorem with consequences."

### Scores
Soundness: 4/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7/10** | Confidence: 5/5 | **Accept**

---

## Reviewer 2: Chief Scientist, AI Safety Organization

### Summary
The paper proves that regulators asking for "stable feature explanations"
are asking for something mathematically impossible under collinearity.
This has direct implications for the EU AI Act and any framework that
requires model explanations as compliance artifacts. DASH provides a
principled resolution, but it trades completeness for stability — meaning
some feature pairs will be reported as "tied" rather than ranked.

### Strengths
1. **The regulatory implication is precise and actionable** (§8): EU AI
   Act Art. 13(3)(b)(ii) requires disclosing "known and foreseeable
   circumstances" affecting accuracy. This impossibility IS such a
   circumstance, and the paper says so explicitly.
2. **The Lean formalization found 2 genuine axiom inconsistencies**
   (supplement §3). This is the strongest argument for formal verification
   in ML: informal proofs had bugs. A careful human reviewer MIGHT have
   caught them — but didn't, until the type checker forced it.
3. **DASH is a constructive resolution, not just a negative result.**
   The paper doesn't just say "you can't have it all" — it says "here's
   what you CAN have, and it's provably optimal (F2)."
4. **The F1 diagnostic (r=-0.89) gives practitioners a way to CHECK**
   whether the impossibility applies to their data. This is rare for an
   impossibility result — usually they apply universally.

### Weaknesses

**W1 (MAJOR): DASH hides information rather than revealing it.**
When DASH reports a tie for features j and k, it's saying "we can't tell
which is more important." But a practitioner deploying a SPECIFIC model
f knows that IN THAT MODEL, j is more important than k (or vice versa).
DASH discards this model-specific information in favor of population-level
consensus. For safety-critical applications, the model-specific explanation
may be what matters — not the average across hypothetical models.

*Fix:* Discuss this tradeoff explicitly. Suggest reporting BOTH
single-model SHAP (with instability caveat) and DASH consensus (stable
but less specific). Currently the paper presents DASH as a pure
improvement, but it's a tradeoff.
*Justifies rejection?* No — but the paper should acknowledge this.

**W2 (MINOR): The AI safety implications are understated.**
If feature explanations are fundamentally unreliable for correlated
features, this affects: (a) model debugging (wrong feature blamed for
errors), (b) fairness audits (protected feature effects misattributed),
(c) scientific discovery (spurious feature relationships). The paper
focuses on (a) but barely mentions (b) and (c).
*Justifies rejection?* No.

### Questions for Authors
1. For a safety-critical model (e.g., clinical decision support): should
   I trust single-model SHAP with an instability warning, or DASH
   consensus with ties? Which is more dangerous: wrong but specific, or
   correct but vague?

### THE ONE THING for +2 points
A formal analysis of the INFORMATION LOSS from DASH: how many bits of
attribution information are lost by averaging, as a function of M and ρ?
This would quantify the completeness-stability tradeoff precisely.

### Scores
Soundness: 4/4 | Contribution: 4/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 8/10** | Confidence: 4/5 | **Strong Accept**

---

## Reviewer 3: Editor-in-Chief, JMLR

### Summary
A thorough submission with a 9-page main text and a 13-page supplement
that together constitute a near-complete treatment of attribution
stability theory. The main text is well-crafted for NeurIPS. The
supplement extends to foundational territory (F1-F5) but these results
need maturation for a definitive journal publication.

### Strengths
1. **The supplement is extraordinary for a conference paper.** Five
   foundational theorems (F3 FIM, F2 optimality, F1 testable condition,
   F5 diagnostic, plus the path convergence), three datasets, a
   reference implementation, and full experimental details. This is
   approaching JMLR quality in the supplement alone.
2. **The F1 diagnostic figure (r=-0.89)** bridges theory and practice
   in a way that's rare for impossibility papers.
3. **The multi-dataset validation** (Breast Cancer + California Housing)
   shows F1 works in BOTH directions: detecting instability AND
   confirming stability.
4. **The depth×ρ table** reveals non-trivial structure (depth=3 anomaly)
   that wasn't predicted by the theory.

### Weaknesses

**W1 (MAJOR): The supplement's F1-F5 are not publication-ready.**
For JMLR, each theorem needs: (a) a precise, self-contained statement
with ALL assumptions explicit, (b) a complete proof (not a sketch),
(c) a discussion of the approximation gaps (which are currently buried
in parenthetical remarks). The NeurIPS supplement is a PREVIEW; the JMLR
paper must be rigorous.

*What would fix it:* Develop each of F1-F5 into a full theorem with
complete proofs and explicit error bounds.
*Justifies rejection?* No for NeurIPS; the supplement is supplementary.

**W2 (MINOR): The empirical validation needs more datasets for JMLR.**
Two datasets (Breast Cancer, California Housing) is sufficient for a
conference. A definitive JMLR reference should validate on 10+ datasets
across domains.

### Questions for Authors
1. Are you planning a JMLR submission that develops F1-F5 fully?
2. Would you be willing to have the supplement's results reviewed as
   part of a JMLR extended version?

### THE ONE THING for +2 points
Promote F1 (testable condition) and its r=-0.89 figure to the main text.
This is the paper's most impressive result and it's buried in the
supplement. A reviewer who only reads 9 pages misses it entirely.

### Scores
Soundness: 4/4 | Contribution: 4/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 8/10** | Confidence: 4/5 | **Strong Accept**

---

## Reviewer 4: VP of ML, Major Bank

### Summary
This paper describes a problem my model validation team encounters
weekly (SHAP instability under collinearity) and provides a formal
explanation, a diagnostic test, and a fix. The F5 algorithm is
implementable in our existing pipeline. The regulatory framing aligns
with our compliance obligations under Fed SR 11-7.

### Strengths
1. **The F5 practitioner algorithm** (screen → validate → decide) maps
   directly to our model validation workflow. Step 1 (split frequency
   check) can be added to our automated model card generation.
2. **F5 precision of 94%** means I can trust flagged pairs are real
   issues. The 27% recall is acceptable — it's a conservative screen.
3. **The M-sizing formula** (M ≥ (z·σ/Δ)²) gives me a per-model
   recommendation. For our credit models (ρ≈0.5-0.7), M=10 suffices.
4. **The regulatory response for ties** ("statistically indistinguishable")
   is the answer I need for our OCC examiners.

### Weaknesses

**W1 (MAJOR): The paper doesn't address feature SELECTION stability.**
In banking, we don't just rank features — we SELECT the top K for
regulatory reporting (SR 11-7 requires "key risk drivers"). If the top-K
set changes across retraining, that's a model risk finding. The paper
discusses ranking instability but not selection instability.

*Fix:* Note that DASH resolves selection instability for between-group
features (stable rankings → stable selection). For within-group features,
DASH reports ties — the selection among tied features should be
documented as "interchangeable."
*Justifies rejection?* No — selection follows from ranking.

**W2 (MINOR): No validation on financial data.**
Breast Cancer and California Housing are scikit-learn toy datasets.
Financial data has different collinearity patterns (e.g., income ↔
credit score ↔ debt-to-income). The diagnostic may behave differently.

### Questions for Authors
1. Can DASH be applied retrospectively to existing model inventories,
   or does it require retraining from scratch?
2. If two features are tied by DASH, and one is a protected attribute
   (e.g., age), how should we handle the tie in a fair lending report?

### THE ONE THING for +2 points
A case study on a public financial dataset (FICO, Lending Club, or
similar) showing the F5 diagnostic in a realistic model validation
scenario. This would make the paper directly citable in regulatory
submissions.

### Scores
Soundness: 3/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7/10** | Confidence: 4/5 | **Accept**

---

## Reviewer 5: Adversarial Red-Teamer

### Summary
My job is to find the weakest link. The paper is well-defended against
most attacks — the Lean formalization, the honest empirical reporting,
the caveats in the Discussion. But there ARE exploitable gaps.

### Strengths
1. **Transparency is excellent.** Every limitation is disclosed: the
   variance placeholder, the Spearman axiom status, the α vs 2/π gap,
   the Corollary 1 uniform-c assumption. Hard to attack what's already
   acknowledged.
2. **The Lean formalization is a genuine defense.** I can't argue "the
   proof has a bug" when a type checker verified it.

### Weaknesses

**W1 (MAJOR): The paper claims "28 substantive theorems, 0 sorry" but
one theorem proves True.**
`consensus_variance_decreases : True := by trivial` is counted in the
"28 substantive" tally (if you count 36 total - 7 helpers - 1 True = 28,
then the True IS excluded). But the paper says "28 substantive theorems
with no unresolved goals, plus one stated variance bound" — the "plus
one" language is ambiguous. A hostile reader could interpret "28 + 1 =
29 theorems, all verified" when in fact only 28 are substantive and 1
proves True.

*Fix:* Say "28 substantive theorems verified by the Lean type checker.
One additional bound (consensus variance O(1/M)) is stated but not yet
formally proved."
*Justifies rejection?* No — it's a framing issue, not a correctness issue.

**W2 (MAJOR): The F1 r=-0.89 correlation may be inflated by the
distribution of Z values.**
The 435 Breast Cancer feature pairs include many pairs with very high Z
(features with very different importance, e.g., worst concave points ↔
mean smoothness). These pairs have Z >> 10 and flip rate = 0,
contributing many points to the upper-right of the scatter plot. The
r=-0.89 may be driven by these "easy" pairs rather than by F1's ability
to discriminate among ambiguous pairs.

*Evidence:* If you restrict to pairs with Z < 10, the r might be lower.
*Fix:* Report r for the INTERESTING range (Z < 5) separately.
*Justifies rejection?* No — but the r=-0.89 headline is potentially
misleading.

**W3 (MINOR): The α=2/π "derivation" is a proof sketch, not a proof.**
Proposition S1 proves that the optimal binary quantizer of N(0,σ²)
captures 2/π of the variance. The connection to "therefore each boosting
stump captures 2/π" requires additional assumptions (Gaussian residuals,
population-optimal split threshold) that are stated but not proved.

### THE ONE THING for +2 points
Report the F1 correlation for the restricted range Z < 5 (the
diagnostically interesting range). If r > -0.7 even in this range,
the claim is robust. If r drops to -0.3, the headline is misleading.

### Scores
Soundness: 3/4 | Contribution: 3/4 | Clarity: 4/4 | Experiments: 3/4
**Overall: 7/10** | Confidence: 4/5 | **Accept**

---

## Area Chair Meta-Review

### Score Table

| | R1 (Fields) | R2 (Safety) | R3 (JMLR) | R4 (Bank) | R5 (Red) | Mean |
|---|---|---|---|---|---|---|
| Sound | 4 | 4 | 4 | 3 | 3 | 3.6 |
| Contrib | 3 | 4 | 4 | 3 | 3 | 3.4 |
| Clarity | 4 | 4 | 4 | 4 | 4 | 4.0 |
| Exper | 3 | 3 | 3 | 3 | 3 | 3.0 |
| **Overall** | **7** | **8** | **8** | **7** | **7** | **7.4** |

### Consensus Weaknesses (3+ reviewers)

1. **F1-F5 are proof sketches, not rigorous theorems** (R1, R3, R5)
2. **Empirical validation needs more datasets** (R3, R4, R5)
3. **DASH trades information for stability — this tradeoff is underexplored** (R1, R2, R4)
4. **The core theorem is mathematically shallow** (R1, R5)

### Best Paper Assessment

**"If every weakness were fixed, would this win Best Paper at NeurIPS 2026?"**

**No — but it's close.** Best Paper requires one of: (a) a surprising
result that changes how people think, (b) a technical breakthrough, or
(c) extraordinary impact. This paper has elements of (a) — the
impossibility IS surprising to practitioners — and (c) — the regulatory
implications are real. It lacks (b) — there's no deep new proof technique.

**What's STILL missing for Best Paper:**

The SINGLE UNIFIED THEOREM that R1 described:

> **The Attribution Design Space Theorem.** For any model class satisfying
> the Rashomon property under within-group correlation ρ > 0, the set of
> achievable (stability, faithfulness, completeness) triples is exactly
> {(1-O(1/M), 1, partial) : M ∈ [1,∞)} ∪ {(S, 1, complete) : S ≤ 1 - m³/P³}.
> DASH(M) achieves the first family and is Pareto-optimal within it.
> The two families are connected by the path convergence: the optimal
> "drop faithfulness" method equals DASH(∞).

This single theorem would subsume: Theorem 1 (impossibility), F2
(optimality), F1 (characterization), and the path convergence (S7) as
corollaries. A paper centered on THIS theorem, with the current results
as supporting evidence, would be a Best Paper candidate.

---

### Transformation Roadmap

| Consensus Weakness | Fix | Before May 6? | For JMLR? | Score Impact |
|---|---|---|---|---|
| F1-F5 are sketches | Full proofs with error bounds | No | Yes | +1 across R1,R3,R5 |
| More datasets | 10+ dataset study | No | Yes | +1 across R3,R4,R5 |
| DASH information loss | Mutual information analysis | Partial (1 paragraph) | Yes (full section) | +1 across R1,R2,R4 |
| Core theorem shallow | Unified Design Space Theorem | No | Yes | +2 for R1 |
| F1 r=-0.89 inflated? | Report r for Z<5 range | Yes (10 min) | Yes | +1 for R5 |

**For the NeurIPS submission (before May 6):**
- Add 1 sentence on DASH information tradeoff to Discussion
- Compute and report F1 r for restricted Z<5 range
- Clarify the "28 substantive + 1 stated" language

**For the JMLR extension:**
- Full proofs of F1-F5 with error bounds
- 10+ dataset empirical study
- The Unified Design Space Theorem
- DASH information loss analysis

---

### The One-Page Elevator Pitch

**Why this research program matters:**

Machine learning models increasingly make decisions that affect people's
lives — who gets a loan, what medical treatment is recommended, which
job applicants are screened. Regulators worldwide (EU AI Act, FDA, OCC)
require that these decisions be EXPLAINED: which features drove the
model's prediction?

The most widely used explanation method, SHAP, has a dirty secret:
under feature collinearity — a condition present in virtually every
real-world dataset — SHAP rankings are UNSTABLE. Retrain the model and
the explanation changes. This isn't a fixable bug. It's a MATHEMATICAL
IMPOSSIBILITY.

This research program proves the impossibility, quantifies its severity
across model architectures, provides a testable diagnostic (r=-0.89 on
real clinical data), and offers the provably optimal resolution (DASH
ensemble averaging). The entire argument is machine-verified in Lean 4 —
the first formally verified impossibility theorem in explainable AI.

The contribution is not one theorem but a FRAMEWORK: the Attribution
Design Space, parameterized by ensemble size M, where practitioners can
navigate the fundamental tradeoff between explaining individual models
(faithful but unstable) and explaining the model class (stable but
averaged). This framework is to feature attribution what Arrow's theorem
is to voting: it defines the boundaries of what's achievable and shows
the way forward.

The work has immediate regulatory significance: the EU AI Act requires
disclosing "known limitations" of AI systems. This research proves a
specific, quantifiable limitation exists for the most widely used
explanation method under the most common data condition. Standards bodies
drafting AI transparency requirements should build on this foundation.

**Is this pitch convincing?**

Partially. The impossibility IS real and important. The framework IS
clean. The regulatory angle IS timely. But: the mathematical depth is
moderate (the core proof is 4 lines), the empirical validation is on
2 datasets, and the "foundational" claim requires the Unified Design
Space Theorem that hasn't been proved yet. The pitch is strongest as a
RESEARCH PROGRAM description, not a single-paper claim. The NeurIPS
paper is the beginning; the JMLR paper would be the landmark.
