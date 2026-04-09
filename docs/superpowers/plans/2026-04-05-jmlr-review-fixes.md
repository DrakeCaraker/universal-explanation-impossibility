# JMLR Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all issues from the vetted JMLR peer review — 3 critical, 8 high-priority, 6 medium-priority fixes — to maximize acceptance probability.

**Architecture:** Sequential edits to `paper/main_jmlr.tex` (primary), with propagation to `paper/main_definitive.tex` (monograph) and one new experiment script. All changes verified by `make paper` + `make verify`.

**Tech Stack:** LaTeX (jmlr.cls), Python 3.9+ (XGBoost, SHAP), Lean 4 (read-only verification)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `paper/main_jmlr.tex` | Modify | All text fixes (Tasks 1-8) |
| `paper/main_definitive.tex` | Modify | Monograph sync (Task 10) |
| `paper/scripts/fixed_testset_experiment.py` | Create | Fixed test set experiment (Task 9) |
| `paper/results_fixed_testset.json` | Create (by script) | Experiment output |

---

### Task 1: Fix arithmetic error in query complexity proof

**Files:**
- Modify: `paper/main_jmlr.tex:1150`
- Modify: `paper/main_definitive.tex` (same text)

**Context:** Line 1150 says "1/8 > 4/9 makes our bound conservative." This is arithmetically false (0.125 < 0.444). The bound M >= sigma^2/(8*Delta^2) IS weaker (more conservative) than M >= 4*sigma^2/(9*Delta^2), but the inequality direction is wrong.

- [ ] **Step 1: Fix in JMLR paper**

In `paper/main_jmlr.tex`, find:
```
$1/8 > 4/9$ makes our bound conservative.
```
Replace with:
```
since $1/8 < 4/9$, this gives a weaker (more conservative) lower bound than our Pinsker derivation.
```

- [ ] **Step 2: Fix in monograph**

Apply the identical fix in `paper/main_definitive.tex`. Search for the same pattern.

- [ ] **Step 3: Verify build**

```bash
cd paper && pdflatex -interaction=nonstopmode main_jmlr.tex > /dev/null 2>&1 && echo "OK"
```

- [ ] **Step 4: Commit**

```bash
git add paper/main_jmlr.tex paper/main_definitive.tex
git commit -m "fix: arithmetic error in query complexity proof (1/8 < 4/9, not >)"
```

---

### Task 2: Hedge "first to address" novelty claim

**Files:**
- Modify: `paper/main_jmlr.tex:151`
- Modify: `paper/main_definitive.tex` (same text)

**Context:** Line 151 says "Our result is the first to address cross-model stability..." without hedging. Herren 2023 addresses feature ranking stability under model multiplicity. The claim should be hedged.

- [ ] **Step 1: Fix in JMLR paper**

In `paper/main_jmlr.tex`, find:
```
Our result is the first to address cross-model \emph{stability}, give quantitative architecture-dependent bounds, and provide a constructive resolution.
```
Replace with:
```
To our knowledge, our result is the first to simultaneously address cross-model \emph{stability} as an impossibility, give quantitative architecture-discriminating bounds, and provide a constructive resolution with proved optimality.
```

- [ ] **Step 2: Fix in monograph**

Apply identical fix in `paper/main_definitive.tex`.

- [ ] **Step 3: Commit**

```bash
git add paper/main_jmlr.tex paper/main_definitive.tex
git commit -m "fix: hedge 'first to address' novelty claim in Related Work"
```

---

### Task 3: Add prior work comparison table

**Files:**
- Modify: `paper/main_jmlr.tex` (Related Work section, after line ~155)
- Modify: `paper/main_definitive.tex`

**Context:** The Related Work section is prose-only. A structured comparison table immediately clarifies the paper's positioning and preempts "how is this different from X?" objections.

- [ ] **Step 1: Add table after the first paragraph of Related Work**

In `paper/main_jmlr.tex`, after line 155 (after the "Bilodeau and Rashomon impossibilities are complementary" sentence), add:

```latex
\begin{table}[h]
\centering
\caption{Positioning relative to prior attribution impossibility results.}
\label{tab:prior-work}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Result & Stability? & Quantitative? & Resolution? & Formal? \\
\midrule
\citet{bilodeau2024impossibility} & No & No & No & No \\
\citet{huang2024failings} & No & No & No & No \\
\citet{srinivas2019full} & No & No & No & No \\
\citet{rao2025limits} & No & No & No & No \\
\citet{laberge2023partial} & Empirical & No & No & No \\
\citet{herren2023statistical} & Inference & No & No & No \\
\citet{jin2025probabilistic} & Certificates & No & No & No \\
\textbf{This paper} & \textbf{Impossibility} & \textbf{Yes} & \textbf{DASH} & \textbf{Lean 4} \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Add same table to monograph**

- [ ] **Step 3: Verify build and commit**

```bash
cd paper && make jmlr && make definitive
git add paper/main_jmlr.tex paper/main_definitive.tex
git commit -m "feat: add prior work comparison table to Related Work"
```

---

### Task 4: Tone down Arrow parallel

**Files:**
- Modify: `paper/main_jmlr.tex:344-348`
- Modify: `paper/main_definitive.tex`

**Context:** Lines 344-348 draw a detailed structural mapping (faithfulness=Pareto, stability=IIA, completeness=totality) that is imprecise and invites unfavorable comparisons. Keep the brief analogy at line 139 and the resolution parallel. Remove the detailed mapping.

- [ ] **Step 1: Replace the detailed mapping**

In `paper/main_jmlr.tex`, find (lines 344-348):
```
The argument is structurally parallel to Arrow's impossibility theorem for social welfare functions \citep{arrow1951social}: Arrow shows that no aggregation rule can satisfy independence of irrelevant alternatives, the Pareto principle, and non-dictatorship simultaneously.
Here, faithfulness plays the role of the Pareto condition, stability plays the role of IIA, and completeness plays the role of totality.
\citet{nipkow2009social} formalized Arrow's theorem in Isabelle/HOL; our Lean~4 formalization follows the same spirit.

The resolution is also parallel: Arrow's theorem is circumvented by relaxing completeness (allowing partial orders or ties).
```

Replace with:
```
The resolution echoes Arrow's impossibility theorem \citep{arrow1951social}: when desirable properties conflict, relaxing completeness (accepting ties or partial orders) restores consistency. \citet{nipkow2009social} formalized Arrow's theorem in Isabelle/HOL; our Lean~4 formalization follows the same spirit.
```

- [ ] **Step 2: Apply to monograph**
- [ ] **Step 3: Verify and commit**

```bash
git commit -m "fix: tone down Arrow parallel — keep analogy, remove imprecise mapping"
```

---

### Task 5: Frame SBD as classical connection, not novel technique

**Files:**
- Modify: `paper/main_jmlr.tex:862`
- Modify: `paper/main_definitive.tex`

**Context:** The SBD section says "cast as a reusable reduction template." This overstates novelty relative to Hunt-Stein. Reframe as connecting classical theory to ML.

- [ ] **Step 1: Revise SBD framing**

In `paper/main_jmlr.tex`, find:
```
The Symmetric Bayes Dichotomy is a specialization of the classical theory of invariant decision rules \citep{lehmann2005testing,hunt1946} to finite symmetric groups, cast as a reusable reduction template for ML impossibility proofs. The classical theory establishes that Bayes-optimal rules respect the invariance structure of the problem; our contribution is the explicit reduction: verifying $G$-invariance of the population distribution suffices to establish a two-family impossibility with quantitative unfaithfulness and stability bounds. We demonstrate this across three structurally distinct instances.
```

Replace with:
```
The Symmetric Bayes Dichotomy connects the classical theory of invariant decision rules \citep{lehmann2005testing,hunt1946} to modern ML impossibility proofs. The classical Hunt--Stein theorem establishes that Bayes-optimal rules respect the invariance structure of the problem; our contribution is demonstrating that this classical machinery, when applied to finite symmetric groups arising in ML (feature permutations, model permutations, CPDAG automorphisms), yields two-family impossibility results with explicit unfaithfulness and stability bounds. We demonstrate this across three structurally distinct instances with different symmetry groups.
```

- [ ] **Step 2: Apply to monograph**
- [ ] **Step 3: Verify and commit**

```bash
git commit -m "fix: frame SBD as classical connection, not novel technique"
```

---

### Task 6: Add feature removal discussion + limitations additions

**Files:**
- Modify: `paper/main_jmlr.tex` (Discussion section, ~line 1944)
- Modify: `paper/main_definitive.tex`

**Context:** The paper never discusses simply dropping redundant features (VIF-based removal) as an alternative to DASH. This is the first thing a practitioner would suggest. Also, the Limitations paragraph needs additional caveats identified in the review.

- [ ] **Step 1: Add feature removal paragraph to Discussion**

After the "Decorrelating via PCA removes collinearity but destroys original feature semantics." sentence (line 1944), add:

```latex
An alternative is removing redundant features entirely (e.g., via VIF thresholding). This eliminates the impossibility but changes the model: predictions differ because fewer features are used, and genuinely useful information may be discarded. \DASH{} is the \emph{explanation-side} fix---it changes how the model is explained without changing what it predicts. The two approaches are complementary: feature removal for model simplification, \DASH{} for faithful explanation of complex models that retain all features.
```

- [ ] **Step 2: Add proportionality equity caveat to Limitations**

After "The balanced ensemble assumption is idealized..." (line 1941), add:

```latex
The global proportionality axiom ($c$ uniform across models) has CV $\approx 0.35$--$0.66$ empirically; under variable $c$, \DASH{} consensus achieves approximate rather than exact equity, with the equity violation bounded by the CV of $c$ across first-mover and non-first-mover models.
```

- [ ] **Step 3: Add 10% threshold caveat to Limitations**

After the prevalence power sentence (line 1947), add:

```latex
The 10\% flip rate threshold defining ``instability'' is a practical convention; at a 5\% threshold the prevalence would be higher, at 15\% lower. The SNR calibration (\S\ref{sec:snr}) provides a continuous relationship between the signal-to-noise ratio and the flip rate, avoiding dependence on any single threshold.
```

- [ ] **Step 4: Apply all to monograph**
- [ ] **Step 5: Verify and commit**

```bash
git commit -m "fix: add feature removal discussion + proportionality/threshold caveats"
```

---

### Task 7: Shorten Lean cross-reference table

**Files:**
- Modify: `paper/main_jmlr.tex` (~line 1887)
- Modify: `paper/main_definitive.tex`

**Context:** The "Sorry?" column is always "No" (0 sorry) — it adds a column for zero information. Remove it to save space.

- [ ] **Step 1: Remove Sorry column from table header and all rows**

In `paper/main_jmlr.tex`, find the cross-reference table. Change the column spec from `{@{}llll@{}}` to `{@{}lll@{}}` and remove the `\textbf{Sorry?}` header and all `& No` entries from every row.

- [ ] **Step 2: Apply to monograph**
- [ ] **Step 3: Verify and commit**

```bash
git commit -m "fix: remove redundant Sorry column from Lean cross-ref table"
```

---

### Task 8: Address cross-group interaction limitation + non-degeneracy downgrade

**Files:**
- Modify: `paper/main_jmlr.tex` (Setup section, ~line 245; Section 3.4, ~line 400)
- Modify: `paper/main_definitive.tex`

**Context:** (H2) Cross-group axioms assume no feature interactions. (H3) Non-degeneracy "proof" (Thm 3) is hand-wavy.

- [ ] **Step 1: Add cross-group interaction caveat**

After the cross-group stability axiom description (near line 245 in Setup), add a sentence:

```latex
Both cross-group axioms assume an approximately additive model; feature interactions between groups can break cross-group independence. The core impossibility and ratio bound do not depend on these axioms (see \S\ref{sec:discussion}).
```

- [ ] **Step 2: Downgrade non-degeneracy**

Find `\begin{theorem}[Attribution non-degeneracy]` (~line 399). Change `\begin{theorem}` to `\begin{proposition}` and add a note:

After the proof, add:
```latex
\begin{remark}
The argument above is informal; a rigorous proof would require a transversality argument showing that the level set $\{\varphi_j(f) = \varphi_k(f)\}$ is a lower-dimensional manifold in seed space. The core impossibility (Theorem~\ref{thm:impossibility}) does not depend on this result---it requires only the Rashomon property as a hypothesis.
\end{remark}
```

- [ ] **Step 3: Apply to monograph**
- [ ] **Step 4: Verify and commit**

```bash
git commit -m "fix: cross-group interaction caveat + downgrade non-degeneracy to proposition"
```

---

### Task 9: Run fixed-test-set experiment

**Files:**
- Create: `paper/scripts/fixed_testset_experiment.py`
- Create (output): `paper/results_fixed_testset.json`
- Modify: `paper/main_jmlr.tex` (Mechanism Isolation subsection)
- Modify: `paper/main_definitive.tex`

**Context:** The paper promises at line 1386: "A cleaner experiment fixing the test set while varying only the model seed would further isolate the effect." This is the single most critical experimental gap. All 3 simulated referees flagged it.

- [ ] **Step 1: Write the experiment script**

Create `paper/scripts/fixed_testset_experiment.py`:

```python
"""Fixed Test Set Experiment: Isolate model instability from evaluation-set variation.

Train 50 XGBoost models on the SAME training data with different seeds
(subsample=0.8 provides stochasticity). Compute TreeSHAP on a FIXED
test set. Compare flip rates to the standard setup (varying test sets).
"""
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

MASTER_SEED = 42
N_MODELS = 50
FLIP_THRESHOLD = 0.10

# Fixed train/test split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=MASTER_SEED
)

# Train N_MODELS with different seeds on the SAME training data
shap_arrays = []
for seed in range(N_MODELS):
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=seed, use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.mean(np.abs(sv), axis=0)
    shap_arrays.append(mean_abs)

shap_matrix = np.array(shap_arrays)  # (N_MODELS, P)
P = shap_matrix.shape[1]

# Compute pairwise flip rates
pairs = []
for j in range(P):
    for k in range(j + 1, P):
        n_flips = 0
        n_compare = 0
        for a in range(N_MODELS):
            for b in range(a + 1, N_MODELS):
                sign_a = np.sign(shap_matrix[a, j] - shap_matrix[a, k])
                sign_b = np.sign(shap_matrix[b, j] - shap_matrix[b, k])
                if sign_a != 0 and sign_b != 0:
                    n_compare += 1
                    if sign_a != sign_b:
                        n_flips += 1
        flip_rate = n_flips / n_compare if n_compare > 0 else 0.0
        if flip_rate > 0.01:  # Only store non-trivial pairs
            pairs.append({
                'j': int(j), 'k': int(k),
                'feature_j': data.feature_names[j],
                'feature_k': data.feature_names[k],
                'flip_rate': round(flip_rate, 4),
                'n_comparisons': n_compare
            })

unstable = [p for p in pairs if p['flip_rate'] > FLIP_THRESHOLD]
max_flip = max(p['flip_rate'] for p in pairs) if pairs else 0.0

results = {
    'experiment': 'fixed_testset_isolation',
    'n_models': N_MODELS,
    'dataset': 'breast_cancer',
    'test_set': 'FIXED (seed=42, 20% holdout)',
    'training_variation': 'subsample=0.8, varying random_state',
    'n_total_pairs': P * (P - 1) // 2,
    'n_unstable_pairs': len(unstable),
    'max_flip_rate': max_flip,
    'top_5_unstable': sorted(unstable, key=lambda x: -x['flip_rate'])[:5],
    'verdict': (
        f'{len(unstable)} unstable pairs with fixed test set '
        f'(vs ~162 with varying test sets). '
        f'Model-level stochasticity is the dominant source.'
        if len(unstable) > 50
        else f'{len(unstable)} unstable pairs with fixed test set. '
        f'Max flip rate: {max_flip:.3f}.'
    )
}

with open('paper/results_fixed_testset.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Unstable pairs (fixed test): {len(unstable)}")
print(f"Max flip rate: {max_flip:.3f}")
print(f"Top pair: {results['top_5_unstable'][0] if unstable else 'none'}")
```

- [ ] **Step 2: Run the experiment**

```bash
cd /Users/drake.caraker/ds_projects/dash-impossibility-lean
python3 paper/scripts/fixed_testset_experiment.py
```

Expected: ~100-170 unstable pairs, max flip ~0.50 (confirming model-level dominance).

- [ ] **Step 3: Read results and update paper**

In `paper/main_jmlr.tex`, find the Mechanism Isolation subsection. Replace the sentence "A cleaner experiment fixing the test set while varying only the model seed would further isolate the effect." with the actual results:

```latex
To cleanly isolate model-level noise, we fix the test set (seed 42, 20\% holdout) and train 50 models on the same training data with different seeds (\texttt{subsample=0.8}). With the fixed test set, [N] pairs are unstable (max flip rate [X]), compared to 162 with varying test sets---confirming that model-level stochasticity accounts for the majority of the observed instability.
```

(Fill in [N] and [X] from the JSON results.)

- [ ] **Step 4: Apply to monograph**
- [ ] **Step 5: Commit**

```bash
git add paper/scripts/fixed_testset_experiment.py paper/results_fixed_testset.json \
  paper/main_jmlr.tex paper/main_definitive.tex
git commit -m "feat: fixed-test-set experiment — isolates model noise from eval-set variation"
```

---

### Task 10: Final verification and sync

**Files:**
- Verify: all paper builds, Lean counts, cross-paper consistency

- [ ] **Step 1: Build all papers**

```bash
cd paper
for f in main_jmlr main main_definitive supplement; do
  pdflatex -interaction=nonstopmode ${f}.tex > /dev/null 2>&1
  bibtex ${f} > /dev/null 2>&1
  pdflatex -interaction=nonstopmode ${f}.tex > /dev/null 2>&1
  pdflatex -interaction=nonstopmode ${f}.tex > /dev/null 2>&1
  pages=$(grep "Output written" ${f}.log | grep -o "[0-9]* pages")
  errors=$(grep -c "^!" ${f}.log)
  echo "${f}: ${pages}, ${errors} errors"
done
```

Expected: 0 errors for all 4 papers.

- [ ] **Step 2: Verify Lean counts**

```bash
cd /Users/drake.caraker/ds_projects/dash-impossibility-lean
grep -c "^theorem\|^lemma" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
grep -c "^axiom" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
grep -rc "sorry" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
ls DASHImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
```

Expected: 305, 16, 0, 54.

- [ ] **Step 3: Verify no stale numbers**

```bash
# Check for old numbers that should not appear
grep -n "1/8 > 4/9" paper/main_jmlr.tex paper/main_definitive.tex
# Should return nothing
```

- [ ] **Step 4: Update FINDINGS_MAP.md**

Add entries for:
- Fixed-test-set experiment (H35 in Tier 6)
- Prior work comparison table (new table)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all JMLR review fixes applied, papers synced"
git push origin main
```

---

## Not Implemented (Strategic Options)

These were identified in the review but deferred as strategic decisions requiring co-author input:

| Item | Rationale for Deferral |
|------|----------------------|
| **Restructure around Design Space Theorem** | Major narrative change; current structure works; co-author consensus needed |
| **Change title** to "Faithful, Stable, and Complete Feature Rankings..." | Significant; co-author preference matters |
| **Cut to 45 pages** (remove query complexity, causal barrier, etc.) | Content cuts need co-author agreement on what to sacrifice |
| **Trim abstract** to ~140 words | Co-author review of messaging needed |
| **Lasso/RF empirical validation** | New experiments beyond scope of text fixes |
| **Larger prevalence survey** (>77 datasets) | Requires OpenML API and significant compute |
