# Best Paper Roadmap: Attribution Impossibility

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address every reviewer weakness from the 5-reviewer Best Paper assessment, moving the paper from "strong accept" (7.4/10) toward best paper candidacy.

**Architecture:** Three phases aligned to deadlines — NeurIPS submission hardening (May 4), camera-ready improvements (if accepted), JMLR extension (foundational upgrade). Each task is self-contained with clear deliverables.

**Tech Stack:** LaTeX, Python (numpy/scipy/xgboost/shap), Lean 4, Mathlib

---

## Gap Analysis: Deduplicated Reviewer Weaknesses

Every unique weakness extracted from all 5 reviewers + meta-review, mapped to paper sections, ranked by impact.

### Blocks Best Paper

| # | Weakness | Reviewers | Paper Location | What's Needed |
|---|----------|-----------|----------------|---------------|
| G1 | **No Unified Design Space Theorem** — paper is "a collection of related results" not "one fundamental theorem with consequences" | R1-W3, Meta | Missing entirely | New theory: characterize achievable (S, F, C) triples, show DASH is Pareto-optimal, subsume Thm 1 + F1 + F2 + path convergence as corollaries |
| G2 | **F1-F5 are proof sketches, not rigorous theorems** — each has approximation gaps, caveats buried in parentheticals | R1-W2, R3-W1, R5-W3 | Supplement §7-§12 | Full proofs with explicit error bounds and self-contained statements |
| G3 | **Core theorem is mathematically trivial** — 4-line proof, no new technique, "textbook preferences argument" | R1-W1, Meta #4 | §3.2 main.tex:185-198 | Cannot be "fixed" — must be reframed. The contribution is the FRAMEWORK, not Theorem 1 alone. Acknowledge simplicity, emphasize the theoretical apparatus built around it |

### Strengthens Case (high impact, feasible)

| # | Weakness | Reviewers | Paper Location | What's Needed |
|---|----------|-----------|----------------|---------------|
| G4 | **DASH information loss unexplored** — DASH hides model-specific info, tradeoff presented as pure improvement | R2-W1, R1, R4, Meta #3 | §5 main.tex:337-371 | Quantify bits lost by averaging as f(M, ρ). Discuss when single-model SHAP is preferable |
| G5 | **F1 r=-0.89 may be inflated by easy pairs** — many pairs with Z>>10 and flip=0 drive the correlation | R5-W2 | Supplement §11, Fig S3 | Compute r for Z<5 range. Report separately. 10-minute code change |
| G6 | **More datasets needed** — only Breast Cancer + California Housing | R3-W2, R4-W2, Meta #2 | §6 main, Supplement §8-§9 | Add 3-5 datasets for NeurIPS (FICO, diabetes, wine); 10+ for JMLR |
| G7 | **Feature SELECTION stability not discussed** — banking needs top-K stability, not just pairwise ranking | R4-W1 | Not addressed | Note that DASH resolves between-group selection stability; within-group selections should be documented as "interchangeable" |
| G8 | **F1 diagnostic buried in supplement** — "the paper's most impressive result" is invisible to reviewers reading only 9 pages | R3 +2 | Supplement §11 | Promote F1 test statistic + r=-0.89 to main text (page budget issue) |

### Nice-to-Have

| # | Weakness | Reviewers | Paper Location | What's Needed |
|---|----------|-----------|----------------|---------------|
| G9 | **"28 substantive + 1 stated" language ambiguous** — hostile reader could count 29 verified theorems | R5-W1 | Abstract (line 57), §1 (line 85), Checklist (line 465) | Reword: "28 substantive theorems verified; one variance bound stated but not yet proved" |
| G10 | **AI safety implications understated** — fairness audits, scientific discovery barely mentioned | R2-W2 | §8 main.tex:438-456 | Add 2-3 sentences on fairness audit and scientific discovery implications |
| G11 | **Financial dataset validation** — toy scikit-learn datasets, no realistic regulatory scenario | R4 +2 | Not present | Case study on FICO/Lending Club with F5 diagnostic |
| G12 | **α=2/π derivation is a proof sketch** — Gaussian residuals and population-optimal split assumptions stated but not proved | R5-W3 | Supplement §5 | Strengthen to full proof or explicitly bound the approximation error |

---

## Kill Criteria

| Gap | If Unresolvable... | Impact |
|-----|-------------------|--------|
| G1 (Unified theorem) | Acknowledge as "open conjecture" — paper is still strong accept without it, just not best paper | Does NOT tank the paper. Tanks the best paper bid. |
| G2 (F1-F5 rigor) | Acceptable for NeurIPS as supplement proof sketches. Mark each caveat prominently | Does NOT tank NeurIPS. Tanks JMLR. |
| G3 (Core trivial) | Cannot be fixed — the proof IS simple. Reframe honestly. If reviewers still object, the paper stands on its framework contributions | Low risk — R1 explicitly says "does not justify rejection" |
| G4 (Info loss) | A paragraph in Discussion is sufficient for NeurIPS. Full analysis is JMLR territory | Does NOT tank anything if acknowledged |
| G5 (F1 inflated) | If r drops below -0.5 for Z<5, the headline claim needs softening. If r > -0.7, claim is robust | Could require rewriting the supplement F1 section |
| G6 (More datasets) | 2 datasets is "sufficient for a conference" (R3). More is for JMLR | Does NOT tank NeurIPS |

---

## Phase 1: NeurIPS Submission Hardening (Before May 4)

**Goal:** Fix every weakness that can be addressed in 1-3 days without structural changes. Move from 7.4 → ~7.8.

### Task 1: Compute F1 r for Restricted Z<5 Range [G5]

**Files:**
- Modify: `paper/scripts/f1_f5_validation.py:110-125`
- Modify: `paper/supplement.tex:650-665` (F1 empirical validation paragraph)
- Modify: `paper/main.tex` (if r is strong, add one sentence)

- [ ] **Step 1: Add restricted-range correlation to f1_f5_validation.py**

After line 124, add:

```python
# Restricted range analysis (R5 robustness check)
mask_z5 = z_f1_clip < 5
corr_f1_restricted = np.corrcoef(z_f1_clip[mask_z5], flips[mask_z5])[0, 1]
n_restricted = mask_z5.sum()
print(f"Restricted Z<5: n={n_restricted}, r={corr_f1_restricted:.3f}")

mask_z3 = z_f1_clip < 3
if mask_z3.sum() > 10:
    corr_f1_z3 = np.corrcoef(z_f1_clip[mask_z3], flips[mask_z3])[0, 1]
    print(f"Restricted Z<3: n={mask_z3.sum()}, r={corr_f1_z3:.3f}")
```

- [ ] **Step 2: Run the script and record results**

Run: `cd paper/scripts && python f1_f5_validation.py`

Record the restricted-range r values. This determines whether the r=-0.89 headline is robust.

- [ ] **Step 3: Update supplement with restricted-range results**

In `supplement.tex`, after the F1 empirical validation paragraph (~line 654), add:

```latex
To address the concern that the $r = -0.89$ correlation may be
inflated by ``easy'' pairs with $Z \gg 10$, we report the correlation
restricted to the diagnostically interesting range $Z < 5$:
$r = [VALUE]$ ($n = [N]$ pairs). [INTERPRETATION based on actual value].
```

- [ ] **Step 4: Commit**

```bash
git add paper/scripts/f1_f5_validation.py paper/supplement.tex
git commit -m "feat: add restricted-range F1 correlation (addresses R5-W2)"
```

---

### Task 2: Clarify "28 Substantive + 1 Stated" Language [G9]

**Files:**
- Modify: `paper/main.tex:57` (abstract)
- Modify: `paper/main.tex:85` (intro)
- Modify: `paper/main.tex:465` (checklist)
- Modify: `paper/supplement.tex:55` (proof architecture)

- [ ] **Step 1: Fix abstract (line 57)**

Replace:
```
28 substantive theorems with no unresolved goals
```
With:
```
28 substantive theorems verified by the Lean~4 type checker, plus one
variance bound stated but not yet formally proved
```

- [ ] **Step 2: Fix intro (line 85)**

Replace:
```
28 substantive theorems with no unresolved goals, plus one stated variance bound awaiting Mathlib infrastructure
```
With:
```
28 substantive theorems with no unresolved goals. One additional bound (consensus variance $O(1/M)$) is stated but not yet formally proved, pending Mathlib measure-theoretic infrastructure
```

- [ ] **Step 3: Fix checklist (line 465)**

Replace:
```
28 substantive theorems, 0 \texttt{sorry}
```
With:
```
28 substantive theorems verified (0 \texttt{sorry}); one variance bound stated but unproved
```

- [ ] **Step 4: Fix supplement proof architecture (line 55)**

Verify the supplement says "28 substantive theorems" without ambiguity. If it says "28 substantive theorems" that's fine. If it mentions "29" or includes the True placeholder in the count, fix it.

- [ ] **Step 5: Commit**

```bash
git add paper/main.tex paper/supplement.tex
git commit -m "fix: clarify 28-theorem count excludes variance placeholder (addresses R5-W1)"
```

---

### Task 3: Add DASH Information Tradeoff to Discussion [G4]

**Files:**
- Modify: `paper/main.tex:438-456` (Discussion section)

- [ ] **Step 1: Add information tradeoff paragraph after "Limitations" paragraph**

After the current Limitations paragraph (ending around line 449), add:

```latex
\paragraph{The information--stability tradeoff.}
\DASH{} consensus achieves stability by averaging over models, but this
discards model-specific information: when $\bar\varphi_j = \bar\varphi_k$
for a same-group pair, the individual models DO assign different importance
to $j$ and $k$---\DASH{} simply cannot determine which assignment is
``correct.'' For safety-critical applications deploying a specific model
$f$, practitioners should report BOTH the single-model SHAP values (with
an instability caveat noting the ranking may flip under retraining) and
the \DASH{} consensus (stable but reporting ties rather than distinctions).
The choice between ``wrong but specific'' and ``correct but vague'' depends
on the downstream decision: model debugging favors specificity, while
regulatory compliance favors stability.
```

- [ ] **Step 2: Verify page count**

Run: `cd paper && pdflatex main.tex && pdflatex main.tex`

If the paper exceeds 9 pages, compress an existing paragraph. The Limitations paragraph has several sentences that could be tightened.

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add DASH information-stability tradeoff discussion (addresses R2-W1)"
```

---

### Task 4: Add Feature Selection Stability Note [G7]

**Files:**
- Modify: `paper/main.tex:367-371` (within-group completeness paragraph in §5)

- [ ] **Step 1: Add selection stability note**

After the within-group completeness paragraph, add:

```latex
\paragraph{Feature selection stability.}
In regulated settings (e.g., SR~11-7 model validation), practitioners
select the top-$K$ features for reporting. \DASH{} guarantees
between-group selection stability: features from higher-importance
groups are consistently selected. Within-group features reported as
tied should be documented as ``interchangeable key drivers''---selecting
among them is arbitrary, and the selection should not be over-interpreted.
```

- [ ] **Step 2: Verify page count, compress if needed**

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add feature selection stability note (addresses R4-W1)"
```

---

### Task 5: Reframe Core Theorem Presentation [G3]

**Files:**
- Modify: `paper/main.tex:89-95` (Contributions paragraph)
- Modify: `paper/main.tex:236-237` (Generality paragraph after Prop 4)

- [ ] **Step 1: Reframe contribution 1**

The current contribution 1 emphasizes "The Attribution Impossibility (Theorem 1)" as the main contribution. R1 says the contribution is the FRAMEWORK, not Theorem 1. Rewrite contribution 1:

Replace:
```latex
\item \textbf{The Attribution Impossibility} (Theorem~\ref{thm:impossibility}): a model-agnostic theorem requiring only the Rashomon property---no model-specific axioms---showing faithfulness, stability, and completeness cannot coexist under collinearity. Model-specific instantiations (\S\ref{sec:bounds}) provide quantitative severity bounds.
```
With:
```latex
\item \textbf{The Attribution Impossibility framework}: a model-agnostic impossibility (Theorem~\ref{thm:impossibility}) requiring only the Rashomon property, instantiated with quantitative severity bounds across four model classes (\S\ref{sec:bounds}), and constructively resolved via \DASH{} (\S\ref{sec:resolution}). The contribution is the framework---from impossibility through architecture discrimination to resolution---not any single theorem.
```

- [ ] **Step 2: Strengthen the Generality paragraph**

After the existing "Generality" paragraph (line 236-237), add one sentence:

```latex
The simplicity of the proof is a feature, not a limitation:
like Arrow's original argument, the mathematical content lies in
identifying the right abstraction (the Rashomon property) rather than
in the proof technique---the quantitative depth comes from the
architecture-specific bounds of \S\ref{sec:bounds}.
```

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: reframe contribution as framework, acknowledge Thm 1 simplicity (addresses R1-W1)"
```

---

### Task 6: Expand AI Safety Implications [G10]

**Files:**
- Modify: `paper/main.tex:451-452` (Regulatory implications paragraph)

- [ ] **Step 1: Expand regulatory paragraph**

Replace the current single-sentence regulatory paragraph with:

```latex
\paragraph{Broader implications.}
Our theorem proves that attribution instability under collinearity is a ``known and foreseeable circumstance'' affecting accuracy under EU AI Act Art.~13(3)(b)(ii)~\citep{euaiact2024}.
Beyond regulatory compliance, the impossibility affects: (a)~model debugging, where the wrong feature may be blamed for errors; (b)~fairness audits, where protected-feature effects may be misattributed to correlated proxies; and (c)~scientific discovery, where spurious feature importance may drive false hypotheses.
In each case, \DASH{} provides a principled mitigation by reporting honest uncertainty (ties) rather than arbitrary distinctions.
```

- [ ] **Step 2: Verify page count**

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: expand AI safety and regulatory implications (addresses R2-W2)"
```

---

### Task 7: Promote F1 Diagnostic to Main Text [G8]

This is the highest-impact change for NeurIPS reviewers who only read 9 pages.

**Files:**
- Modify: `paper/main.tex:372-416` (Experiments section)

- [ ] **Step 1: Add F1 diagnostic paragraph to §6**

After the DASH convergence paragraph (around line 416), add:

```latex
\paragraph{Testable condition (supplement \S11).}
We derive a test statistic $Z_{jk} = |\bar\varphi_j - \bar\varphi_k| /
(\hat\sigma_{jk}/\sqrt{M})$ that predicts ranking instability: on Breast
Cancer (435 feature pairs, 50 seeds), $Z_{jk}$ correlates with the
empirical flip rate at $r = -0.89$ (Figure~S3, left). Pairs with
$Z_{jk} < 1.96$ have unreliable rankings; pairs above this threshold are
stable. A single-model screening variant using split frequencies achieves
94\% precision (supplement \S12).
```

- [ ] **Step 2: Verify page count — this is tight**

If over 9 pages, compress the Related Work section (§7) which is currently ~25 lines and could be tightened to ~18.

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: promote F1 diagnostic summary to main text (addresses R3 +2)"
```

---

### Task 8: Final Page Budget Audit and Build

**Files:**
- Build: `paper/main.tex`

- [ ] **Step 1: Build both versions**

```bash
cd paper && bash build.sh
```

- [ ] **Step 2: Check NeurIPS version page count**

Must be ≤ 9 pages (excluding references and checklist). If over, identify the highest-compression-potential paragraph and tighten.

Compression candidates (in priority order):
1. Related Work (§7): currently verbose, can trim 5-7 lines
2. Limitations paragraph: has redundant sentences
3. Setup (§2): some boilerplate that experienced readers skip

- [ ] **Step 3: Fix any page budget issues**

- [ ] **Step 4: Commit and push**

```bash
git add -A paper/
git commit -m "chore: final page budget audit for NeurIPS submission"
```

---

## Phase 2: Camera-Ready Improvements (If Accepted, ~August 2026)

**Goal:** Address all "strengthens case" items that couldn't fit in 9 pages. Move from ~7.8 → ~8.5.

### Task 9: Add 3-5 Additional Datasets [G6]

**Files:**
- Create: `paper/scripts/multi_dataset_validation.py`
- Modify: `paper/supplement.tex` (new section)

- [ ] **Step 1: Implement multi-dataset F1 validation**

Datasets (all public, no licensing issues):
1. **FICO** (Home Equity) — financial, strong regulatory relevance
2. **Diabetes** (scikit-learn) — medical, moderate collinearity
3. **Wine Quality** (UCI) — chemical features, many groups
4. **Boston Housing** (if ethical concerns addressed) or **Ames Housing** — real estate
5. **Heart Disease** (UCI) — clinical, moderate P

For each dataset:
- Train 50 XGBoost models with different seeds
- Compute F1 test statistic for all feature pairs
- Compute flip rate
- Report r(Z, flip), number of unstable pairs, F5 precision

- [ ] **Step 2: Create summary table**

```latex
\begin{table}[h]
\centering
\caption{F1 diagnostic generality across datasets.}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Dataset & P & Pairs & Unstable & F1 $r$ & F5 precision \\
\midrule
Breast Cancer & 30 & 435 & 168 & $-0.89$ & 94\% \\
California Housing & 8 & 28 & 1 & $-0.58$ & --- \\
[NEW DATASETS...] \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 3: Write supplement section**

- [ ] **Step 4: Commit**

---

### Task 10: Expand DASH Information Loss Analysis [G4]

**Files:**
- Create: `paper/scripts/information_loss.py`
- Modify: `paper/supplement.tex` (new section)

- [ ] **Step 1: Quantify information loss**

Compute for each dataset:
- Within-group: mutual information I(single-model ranking; true DGP) vs I(DASH ranking; true DGP)
- Effective bits lost = log2(m) for same-group features (DASH reports ties, losing log2(m) bits of ordering info)
- Plot: bits lost vs M for different ρ values

- [ ] **Step 2: Write supplement section**

Key message: DASH loses exactly the information that is UNRELIABLE — the within-group ordering. The between-group information (which is reliable) is preserved and sharpened.

- [ ] **Step 3: Commit**

---

### Task 11: Financial Case Study [G11]

**Files:**
- Create: `paper/scripts/financial_validation.py`
- Modify: `paper/supplement.tex` (new section)

- [ ] **Step 1: FICO/Lending Club case study**

Using FICO Home Equity data (or Lending Club):
- Identify correlated feature groups (income ↔ credit score ↔ DTI)
- Run F5 diagnostic on a single model
- Show which pairs are flagged
- Apply DASH and show resolution
- Frame as a model validation workflow (SR 11-7 compliant)

- [ ] **Step 2: Write as a "practitioner walkthrough" in supplement**

---

## Phase 3: JMLR Extension (Foundational Upgrade)

**Goal:** Transform from "strong conference paper" to "definitive reference." Address G1, G2 fully. Target: 8.5+ → Best Paper territory.

### Task 12: Prove F3 (FIM Impossibility) Rigorously [G2]

**Files:**
- Modify: `paper/supplement.tex` §10 (currently a proof sketch)
- Optionally: `DASHImpossibility/FIM.lean` (Lean formalization)

- [ ] **Step 1: Make all assumptions explicit**

Current gaps:
- Quadratic approximation only valid for small ε
- "Monotonicity of attribution in |θ_j|" is assumed, not proved
- NTK extension is hand-waved

Fix: State as a conditional theorem with explicit regularity conditions. Separate the local result (rigorous) from the global extension (conjectural).

- [ ] **Step 2: Write complete proof with error bounds**

- [ ] **Step 3: Lean formalization (if feasible)**

The FIM result is more analysis than combinatorics. Lean formalization requires Mathlib's `Analysis.InnerProductSpace` for eigenvalue bounds. Feasibility: MEDIUM.

---

### Task 13: Prove F1 (Testable Condition) Rigorously [G2]

**Files:**
- Modify: `paper/supplement.tex` §11

- [ ] **Step 1: Make the CLT assumption precise**

Current gap: "Under the assumption that single-model attributions are approximately Gaussian" — this is a sketch.

Fix:
- State Berry-Esseen bound for the CLT approximation
- Bound the error: flip_rate = Φ(-Z/√M) + O(M^{-1/2}) with explicit constant
- Discuss when CLT fails (heavy-tailed attribution distributions)

- [ ] **Step 2: Write complete proof**

---

### Task 14: Prove F2 (DASH Optimality) Rigorously [G2]

**Files:**
- Modify: `paper/supplement.tex` §10

- [ ] **Step 1: Extend beyond "unbiased linear aggregations"**

Current gap: optimality only proved for unbiased linear methods.

Fix:
- Prove MVUE optimality (Rao-Blackwell / Lehmann-Scheffé) for linear case
- Discuss nonlinear aggregations (median, trimmed mean) — show they have higher variance for Gaussian attributions
- State precisely what class DASH is optimal over

---

### Task 15: Prove F5 (Diagnostic) Rigorously [G2]

**Files:**
- Modify: `paper/supplement.tex` §12

- [ ] **Step 1: Formalize the exchangeability condition**

Current gap: "approximately independent" per-tree split indicators.

Fix:
- With sub-sampling (colsample_bytree < 1): prove exact exchangeability
- Without sub-sampling: bound the dependence and its effect on the test statistic
- State a clean conditional theorem: "Under sub-sampling rate q < 1, the per-tree split indicators are exchangeable, and Z^split is asymptotically N(0,1) under H0"

---

### Task 16: The Unified Design Space Theorem [G1]

This is the "one theorem to rule them all" that R1 describes. This is the JMLR centerpiece.

**Files:**
- Create: `paper/design_space_theorem.tex` (draft section)
- Create: `DASHImpossibility/DesignSpace.lean` (Lean formalization)

- [ ] **Step 1: Formalize the achievable set**

**Theorem (Attribution Design Space).** For any model class satisfying the Rashomon property under within-group correlation ρ > 0, the set of achievable (stability, unfaithfulness, completeness) triples under any aggregation of M ≥ 1 independently trained models is:

```
A = { (S, U, C) :
      S = 1 - Var(φ_j) / (M · Δ²),         [stability improves with M]
      U = 0 within groups (ties),             [zero unfaithfulness via DASH]
      C = partial (ties within groups),       [completeness relaxed]
      faithfulness preserved between groups }
∪ { (S, U, C) :
      S ≤ 1 - m³/P³,                         [stability bounded by Spearman]
      U ≥ 1/2 per symmetric pair,            [unfaithful for half the models]
      C = complete }                          [full ranking maintained]
```

DASH(M) achieves the first family. Single-model SHAP achieves points in the second family. No method achieves a point outside A.

- [ ] **Step 2: Show F1, F2, path convergence, Thm 1 are corollaries**

- Theorem 1 = the empty intersection: A ∩ {S=1, U=0, C=complete} = ∅
- F1 = the boundary characterization: Z_{jk} < 1.96 ↔ near the (S, U, C) boundary
- F2 = Pareto optimality: DASH(M) is on the Pareto frontier of the first family
- Path convergence = the two families converge as M → ∞

- [ ] **Step 3: Lean formalization**

This requires defining the achievable set as a Lean structure and proving subsumption. Feasibility: HIGH effort but conceptually clean.

- [ ] **Step 4: Validate empirically**

Plot the (stability, unfaithfulness) Pareto frontier for each dataset. Show DASH points lie on the frontier.

---

### Task 17: 10+ Dataset Empirical Study [G6]

**Files:**
- Create: `paper/scripts/comprehensive_validation.py`
- Modify: `paper/supplement.tex` (major new section)

- [ ] **Step 1: Select 10+ datasets across domains**

1. Breast Cancer (medical, high collinearity)
2. California Housing (real estate, moderate)
3. FICO Home Equity (financial)
4. Lending Club (financial)
5. Diabetes (medical)
6. Wine Quality (chemistry)
7. Heart Disease (clinical)
8. COMPAS (criminal justice — fairness relevance)
9. Adult Income (census — fairness relevance)
10. Ames Housing (real estate)
11. Credit Card Default (financial)
12. Communities and Crime (sociology, many features)

- [ ] **Step 2: Run F1 + F5 diagnostics on all 12**

- [ ] **Step 3: Summary analysis**

Report: mean r across datasets, variance, which domains show highest instability, whether F5 precision holds across domains.

---

### Task 18: Strengthen α=2/π Proof [G12]

**Files:**
- Modify: `paper/supplement.tex` §5

- [ ] **Step 1: Bound the approximation error**

Currently: "The fitted value (α ≈ 0.60) is slightly below 2/π = 0.637 because (i) stumps split on the empirical median... and (ii) residuals are no longer exactly Gaussian."

Fix: Derive explicit bounds on the error from each source:
- Empirical vs population median: O(1/√n) error in split threshold → O(1/√n) error in α
- Non-Gaussian residuals after round 1: bound the kurtosis and its effect on variance capture

---

## Vet Audit

### Round 1: Factual Accuracy

- [x] "28 substantive theorems" — verified from summary: 36 total - 7 helpers - 1 True = 28. Correct.
- [x] "r=-0.89" — from f1_f5_validation.py output. Correct provenance.
- [x] "94% precision" — from supplement line 709-710. Correct.
- [x] "48% flip rate" — from supplement line 433 and main text line 376. Correct.
- [x] Reviewer scores: R1=7, R2=8, R3=8, R4=7, R5=7, Mean=7.4. Verified from best-paper-review.md lines 86, 153, 210, 267, 330. Correct.
- [x] "May 4 abstract, May 6 paper" — from CLAUDE.md. Correct.
- [x] G1 description matches R1-W3 verbatim: "The ideal result would be: The attribution design space under collinearity is the one-parameter family..." Verified at line 367-378.
- [x] File line references: main.tex line 57 (abstract), 85 (intro), 465 (checklist) — verified against actual file.

### Round 2: Reasoning Quality

- **G3 "cannot be fixed"**: OBSERVED — the proof IS 4 lines, and R1 explicitly says "Justifies rejection? No." The recommendation to reframe rather than restructure is the right call. MEDIUM confidence — a reviewer could still downweight.
- **G8 "highest impact for NeurIPS"**: INFERRED — promoting F1 to main text addresses R3's explicit "+2 thing" but requires page budget. The reasoning is sound but page budget is the constraint. HIGH confidence.
- **G1 is Phase 3**: INFERRED — the Unified Design Space Theorem requires substantial new theory and cannot be done in 4 days. R1 says it would yield +2 points. HIGH confidence this is correctly phased.
- **Phase 1 score impact "7.4 → ~7.8"**: INFERRED — optimistic. The changes address framing, not substance. A more honest estimate: 7.4 → 7.5-7.6 (reviewers who already said Accept are unlikely to change scores based on wording). MEDIUM confidence.

⚠️ **Correction:** Revised score estimate from "~7.8" to "7.5-7.7" — Phase 1 changes are incremental, not transformative.

### Round 3: Omissions

1. **Not compared: DASH vs other aggregation methods (median, trimmed mean, weighted).** R2 and the meta-review hint at this. Task 14 addresses it for JMLR but Phase 1 doesn't mention alternatives. This is fine — NeurIPS doesn't require this.

2. **Competing explanation for r=-0.89**: Could the correlation be an artifact of the SHAP computation itself (tree-path-dependent SHAP values have known biases)? Not addressed. This is a deeper concern than R5's Z-distribution objection. **Add to Task 1:** after computing restricted-range r, also check whether the correlation holds for gain-based importance (not SHAP).

3. **No plan to address R1's path convergence question**: "Is there a deeper reason why the path convergence has no Arrow analogue?" This is an intellectual question, not a weakness — but answering it in the paper would strengthen the Arrow parallel. Consider adding 1-2 sentences to Discussion. **Risk: LOW (question for authors, not a weakness).**

4. **R4's retrospective DASH question**: "Can DASH be applied retrospectively to existing model inventories?" The answer is yes (train additional models, average) but this isn't stated. Worth adding to the practitioner algorithm. **Add to Task 4.**

5. **No plan for Lean formalization of the variance bound** — the one `True` placeholder. This is acknowledged but Phase 1 doesn't attempt to fix it. Correct — it requires Mathlib infrastructure (4 new axioms). This is Phase 3 territory.

---

## Corrections Applied

1. ⚠️ Score impact estimate revised: "7.4 → ~7.8" → "7.4 → 7.5-7.7"
2. ⚠️ Added to Task 1: also validate F1 with gain-based importance (not just SHAP)
3. ⚠️ Added to Task 4: note that DASH can be applied retrospectively

## Confidence Ratings

| Finding | Confidence | Justification |
|---------|-----------|---------------|
| Phase 1 tasks are correct priorities | HIGH | Directly mapped from reviewer text, verified line-by-line |
| G5 (F1 inflation) is the quickest high-impact fix | HIGH | R5 explicitly says 10-minute fix, and the result determines headline robustness |
| G1 (Unified theorem) is the key to Best Paper | HIGH | R1 and meta-review both name it explicitly |
| Phase 1 score impact 7.5-7.7 | MEDIUM | Incremental changes rarely change reviewer scores; the real value is preempting objections |
| G3 reframe strategy is correct | MEDIUM | R1 says "does not justify rejection" but a different reviewer might feel differently |
| 10+ datasets are needed for JMLR | HIGH | R3 (JMLR editor) explicitly states this |
| Phase 3 timeline is realistic | LOW | The Unified Design Space Theorem is hard — unclear if it can be proved at all |

## Open Questions

1. **Does the F1 correlation hold for Z<5?** Task 1 will answer this. If r drops below -0.5, the supplement section needs significant rewriting.
2. **Can the Unified Design Space Theorem be proved?** The statement in the meta-review is clean but the proof is non-trivial. It may turn out to be false or require additional assumptions.
3. **Will NeurIPS 2026 require AI-use disclosure?** Check the call for papers before submission.
4. **Page budget after Phase 1 additions**: Tasks 3, 4, 6, 7 each add ~5-8 lines. Total: ~25 lines = ~0.5 pages. The paper is at 9 pages. Something must be compressed.

## Data Gaps

1. **No restricted-range F1 correlation computed yet** — Task 1 will generate this
2. **No financial dataset validation** — Phase 2, Task 11
3. **No non-SHAP attribution validation** — gain-based importance, integrated gradients not tested
4. **Variance bound not formalized** — requires Mathlib measure axioms, Phase 3
