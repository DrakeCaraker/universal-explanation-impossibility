# Post-Review Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all 35 corrections and 8 limitations identified by 24 simulated peer reviewers, transforming the paper from a Nature rejection into a strong JMLR submission.

**Architecture:** Five phases executed in dependency order. Phase 1 (P0 blockers) must complete first. Phases 2-4 can run in parallel. Phase 5 (infrastructure) runs last. Each task produces a self-contained commit.

**Tech Stack:** Lean 4 + Mathlib, Python 3 (numpy/scipy/sklearn/xgboost/matplotlib), LaTeX, Git

**Key files map:**
- `paper/nature_article.tex` — Nature submission (will become JMLR submission)
- `paper/universal_impossibility_monograph.tex` — Full monograph (3023 lines)
- `UniversalImpossibility/ExplanationSystem.lean` — Core theorem + definitions
- `knockout-experiments/` — New experiment scripts and results
- `paper/scripts/` — Existing experiment scripts
- `CLAUDE.md` — Project documentation

---

## Phase 1: P0 Blockers (Sequential — must complete first)

### Task 1: Fix decisiveness definition mismatch [C1]

**Files:**
- Modify: `paper/nature_article.tex:860-895`

The Methods section (line 863-865) defines decisive as pairwise: `exp(θ₁) ⊥ exp(θ₂) → E(θ₁) ⊥ E(θ₂)`. The Lean code (ExplanationSystem.lean:54-55) defines it pointwise: `∀ θ h, incomp(explain(θ), h) → incomp(E(θ), h)`. Under the pairwise definition, irreflexivity yields the contradiction directly and faithfulness is superfluous — reducing the trilemma to a dilemma. This is the most critical correctness bug.

- [ ] **Step 1: Fix the formal definition in Methods**

In `paper/nature_article.tex`, replace lines 863-865:

```latex
% BEFORE (pairwise — WRONG):
\emph{decisive} if
$\mathsf{exp}(\theta_1) \incomp \mathsf{exp}(\theta_2)$ implies
$E(\theta_1) \incomp E(\theta_2)$.

% AFTER (pointwise — matches Lean):
\emph{decisive} if for all $\theta$ and $h \in \sH$,
$\mathsf{exp}(\theta) \incomp h$ implies $E(\theta) \incomp h$.
```

- [ ] **Step 2: Fix the proof sketch in Methods**

Replace lines 882-895 with a proof sketch matching the Lean chain:

```latex
\paragraph{Proof structure (detailed).}
Let $\mathcal{S}$ be an explanation system with the Rashomon property,
witnessed by $\theta_1, \theta_2$.  Suppose for contradiction that $E$ is
faithful, stable, and decisive.  (1)~Decisiveness at $\theta_1$ with
$h = \mathsf{exp}(\theta_2)$: since $\mathsf{exp}(\theta_1) \incomp
\mathsf{exp}(\theta_2)$, we have $E(\theta_1) \incomp
\mathsf{exp}(\theta_2)$.  (2)~Stability: since
$\mathsf{obs}(\theta_1) = \mathsf{obs}(\theta_2)$, we have
$E(\theta_1) = E(\theta_2)$, so $E(\theta_2) \incomp
\mathsf{exp}(\theta_2)$.  (3)~Faithfulness at $\theta_2$:
$\lnot(E(\theta_2) \incomp \mathsf{exp}(\theta_2))$.
Contradiction between (2) and (3).
```

- [ ] **Step 3: Add a remark about the pairwise variant**

After the proof, add:

```latex
\paragraph{Remark.}  A weaker \emph{pairwise} variant of decisiveness ---
$\mathsf{exp}(\theta_1) \incomp \mathsf{exp}(\theta_2)$ implies
$E(\theta_1) \incomp E(\theta_2)$ --- suffices for a two-property
impossibility (stability and pairwise decisiveness contradict
irreflexivity of $\incomp$), but faithfulness becomes superfluous.
The pointwise definition used here (and in the Lean formalization)
is the minimal condition that makes all three properties load-bearing.
```

- [ ] **Step 4: Verify the informal definition in Results also matches**

Check lines 195-197 of nature_article.tex — the informal prose "commits to every distinction that the system's internal structure makes — whatever the internal structure rules out, the explanation also rules out" — this is consistent with the pointwise definition. No change needed, but verify.

- [ ] **Step 5: Compile and verify**

```bash
cd paper && pdflatex nature_article.tex 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add paper/nature_article.tex
git commit -m "fix: align decisive definition in Methods with Lean (pointwise, not pairwise)

The pairwise variant makes faithfulness superfluous, reducing the trilemma
to a dilemma. Fixed to match ExplanationSystem.lean:54-55."
```

### Task 2: Fix stale counts in monograph [C2]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex` (Section: Lean Formalization)

- [ ] **Step 1: Find all stale count references**

```bash
grep -n "82 files\|378 theorem\|82.*file\|378.*lemma" paper/universal_impossibility_monograph.tex
```

- [ ] **Step 2: Replace all instances with current counts**

Replace every occurrence of "82" files with "95" and "378" theorems with "417". The actual counts (verified by `lake build` on 2026-04-14) are: 95 files, 417 theorems+lemmas, 72 axioms, 0 sorry.

- [ ] **Step 3: Add automated verification guard**

Create `paper/scripts/verify_counts.sh`:

```bash
#!/bin/bash
set -e
FILES=$(ls UniversalImpossibility/*.lean | wc -l | tr -d ' ')
THEOREMS=$(grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}')
AXIOMS=$(grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}')
SORRY=$(grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}')

echo "Files: $FILES | Theorems: $THEOREMS | Axioms: $AXIOMS | Sorry: $SORRY"

# Check monograph
for count_pair in "$FILES:files" "$THEOREMS:theorems" "$AXIOMS:axioms"; do
  NUM="${count_pair%%:*}"
  LABEL="${count_pair##*:}"
  FOUND=$(grep -c "$NUM.*$LABEL\|$NUM~$LABEL" paper/universal_impossibility_monograph.tex 2>/dev/null || echo 0)
  if [ "$FOUND" -eq 0 ]; then
    echo "WARNING: $NUM $LABEL not found in monograph"
  fi
done
```

- [ ] **Step 4: Commit**

```bash
git add paper/universal_impossibility_monograph.tex paper/scripts/verify_counts.sh
git commit -m "fix: update stale Lean counts in monograph (82/378 → 95/417)"
```

### Task 3: Investigate gauge lattice 200× discrepancy [C3]

**Files:**
- Modify: `paper/scripts/gauge_lattice_experiment.py`
- Modify: `paper/results_gauge_lattice.json`

- [ ] **Step 1: Read the gauge experiment code to understand the Wilson loop measurement**

Read `paper/scripts/gauge_lattice_experiment.py` and identify how Wilson loop means are computed at each β value.

- [ ] **Step 2: Compare code output to analytic prediction**

The analytic prediction for a 1×1 Wilson loop in 2D Z₂ gauge theory is `⟨W⟩ = tanh(β)^A` where A is the area. At β=0.1, tanh(0.1)≈0.0997, so for a 1×1 loop: ≈0.0997. Check whether the JSON value 0.022 is for a different loop size or a different observable.

- [ ] **Step 3: Fix the code bug**

CRITICAL: β=0.2 and β=0.3 show NEGATIVE Wilson loop means, which is physically impossible for Z₂ gauge theory (Wilson loops are products of ±1 variables, so means must be in [-1,1] but for physical configurations at finite β should be non-negative for 1×1 loops). This indicates a code bug, not a finite-size effect. Debug the sampling or measurement code, fix, and re-run all β values.

- [ ] **Step 4: Add error bars to all gauge results**

Compute bootstrap 95% CIs for all β values and add to the JSON.

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/gauge_lattice_experiment.py paper/results_gauge_lattice.json
git commit -m "fix: resolve 200x Wilson loop discrepancy at beta=0.1, add error bars"
```

### Task 4: Fix codon entropy p-value mismatch [C4]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`
- Modify: `paper/nature_article.tex`

- [ ] **Step 1: Identify which p-value is cited where**

```bash
grep -n "2.0.*10.*-3\|2.3.*10.*-18\|Kruskal" paper/nature_article.tex paper/universal_impossibility_monograph.tex
```

- [ ] **Step 2: Clarify primary data source**

The real NCBI data (120 species) gives p=2.0e-3. The simulated data gives p=2.3e-18. Both are valid; the paper must specify which. For the Nature/JMLR article, lead with the real data (p=2.0e-3) and note the simulated data (p=2.3e-18) as supplementary.

- [ ] **Step 3: Update all paper references to specify data source**

- [ ] **Step 4: Commit**

```bash
git add paper/nature_article.tex paper/universal_impossibility_monograph.tex
git commit -m "fix: clarify codon entropy p-values (real NCBI: 2.0e-3, simulated: 2.3e-18)"
```

---

## Phase 2: Experimental Rework (Can parallelize tasks 5-9)

### Task 5: Noether counting sensitivity analysis [C7, C9, C10, C32]

**Files:**
- Create: `knockout-experiments/noether_sensitivity.py`
- Modify: `knockout-experiments/results_noether_counting.json`

This is the single most important experiment to strengthen the paper.

- [ ] **Step 1: Write the sensitivity experiment**

Create `knockout-experiments/noether_sensitivity.py` that runs the Noether counting design at ρ ∈ {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99}. For each ρ:
- P=12 features, g=3 groups of 4, β=[5,5,5,5, 2,2,2,2, 0.5,0.5,0.5,0.5]
- N_train=500, noise=1.0, 200 Ridge models
- Compute all 66 pairwise flip rates
- Classify within-group vs between-group
- Compute mean flip rate for each category
- Compute separation gap

Use the existing `noether_counting_v2.py` as a template. Key additions:
- **Permutation test** (not Mann-Whitney): permute group labels of features 10,000 times, compute the test statistic (difference of mean flip rates) under each permutation, report the permutation p-value.
- **Cluster bootstrap**: resample at the model level (n=200 models), not pair level.
- **Benjamini-Hochberg correction** on the 66 pairwise classifications.
- **Power analysis**: for each ρ, compute the minimum detectable difference at 80% power given n=200 models.

- [ ] **Step 2: Run the experiment**

```bash
python3 knockout-experiments/noether_sensitivity.py
```

Expected runtime: ~5-10 minutes (200 Ridge models × 8 ρ values).

- [ ] **Step 3: Analyze results — find the critical ρ threshold**

Report: at what ρ does the bimodal gap emerge? What is the gap at ρ=0.7? Does the permutation test remain significant?

- [ ] **Step 4: Generate multi-panel figure**

Create `knockout-experiments/figures/noether_sensitivity.pdf`: 8-panel figure showing flip rate histograms at each ρ, plus a summary panel showing gap vs ρ.

- [ ] **Step 5: Update the paper with sensitivity results**

Add the sensitivity analysis to the monograph Noether counting subsection.

- [ ] **Step 6: Commit**

```bash
git add knockout-experiments/noether_sensitivity.py knockout-experiments/figures/noether_sensitivity.pdf
git commit -m "feat: Noether counting sensitivity analysis across rho={0.5..0.99}"
```

### Task 6: Redo attention experiment with proper retraining [C6]

**Files:**
- Create: `knockout-experiments/attention_retraining_proper.py`

- [ ] **Step 1: Write the proper retraining experiment**

Train 10 DistilBERT-base models from scratch (or fine-tune from the pretrained checkpoint with different random seeds, full model, not just last 2 layers). Each model should:
- Use a different `seed` for data shuffling and weight initialization
- Train to convergence on SST-2 or a sentiment dataset
- Report final loss and accuracy
- Only include models within ε of the best loss in the "Rashomon set"

Then measure attention argmax flip rate across the genuine Rashomon set.

If full retraining is too expensive (>1 hour per model), use the existing `attention_full_retraining_experiment.py` results (which exist in `paper/results_attention_full_retraining.json`) and document clearly that these are full-retraining results, not perturbation.

- [ ] **Step 2: Report model losses alongside flip rates**

For every model in the comparison, report: loss, accuracy, and whether it's within Rashomon ε of the best.

- [ ] **Step 3: Update the paper to lead with retraining results, not perturbation**

In the monograph and Nature article, present the retraining results (19.9% flip rate from existing data) as primary, and the perturbation results (60%) as a secondary sensitivity analysis with explicit caveat.

- [ ] **Step 4: Commit**

```bash
git add knockout-experiments/attention_retraining_proper.py paper/nature_article.tex
git commit -m "fix: lead with full-retraining attention results (19.9%), demote perturbation (60%)"
```

### Task 7: Fix universal η plot methodology [C8]

**Files:**
- Modify: `knockout-experiments/universal_eta_synthesis.py`

- [ ] **Step 1: Define the "well-characterized" criterion a priori**

A group is "well-characterized" if:
- The group is **exact by construction** (the symmetry is part of the domain's definition, not an approximation)
- The instability metric is **directly comparable** to 1-η (same scale, same type of measurement)

Document this criterion at the top of the script and in PRE_REGISTRATION.md.

- [ ] **Step 2: Report ALL-16 R² as the primary result**

The R² for all 16 instances becomes the headline number. The 7-instance R² becomes a secondary analysis labeled "exact-group subset."

- [ ] **Step 3: Add a residual analysis**

For each of the 9 "approximate" points, explain the deviation: what's the actual group vs the assigned group? How much does the approximation error account for?

- [ ] **Step 4: Commit**

```bash
git add knockout-experiments/universal_eta_synthesis.py knockout-experiments/PRE_REGISTRATION.md
git commit -m "fix: report all-16 R² as primary, pre-specify well-characterized criterion"
```

### Task 8: Add power analyses to all experiments [C10, C25]

**Files:**
- Create: `knockout-experiments/power_analysis.py`

- [ ] **Step 1: For each experiment, compute minimum detectable effect**

Using scipy.stats.power or manual computation:
- Attention: n=10 models, effect = flip rate difference from 0
- Noether: n=200 models, effect = gap between within and between
- Model selection: n=50 models × 20 splits
- Concept probes: n=15 models
- Counterfactuals: n=20 models

- [ ] **Step 2: Report effect sizes (Cohen's d) alongside p-values**

For every experiment, compute and report Cohen's d or equivalent standardized effect size.

- [ ] **Step 3: Add a summary table to the monograph**

- [ ] **Step 4: Commit**

```bash
git add knockout-experiments/power_analysis.py paper/universal_impossibility_monograph.tex
git commit -m "feat: add power analyses and effect sizes for all experiments"
```

### Task 9: Fix statistical methodology [C9, C31, C32]

**Files:**
- Modify: `knockout-experiments/noether_counting_v2.py`
- Modify: `knockout-experiments/noether_sensitivity.py` (if Task 5 not yet done, merge with it)

- [ ] **Step 1: Implement permutation test for Noether counting**

Replace Mann-Whitney with: permute group labels of the 12 features 10,000 times. For each permutation, compute the test statistic (mean between-group flip rate minus mean within-group flip rate). Report the fraction of permutations where the statistic exceeds the observed value.

- [ ] **Step 2: Implement cluster bootstrap**

Resample at the model level (draw 200 models with replacement from the 200), recompute all 66 flip rates, recompute means. Repeat 2,000 times for CIs.

- [ ] **Step 3: Apply Benjamini-Hochberg correction**

For the 66 pairwise classifications (stable vs unstable), apply BH correction at FDR=0.05.

- [ ] **Step 4: Re-run and report corrected results**

- [ ] **Step 5: Commit**

```bash
git add knockout-experiments/noether_counting_v2.py
git commit -m "fix: replace Mann-Whitney with permutation test, add cluster bootstrap and BH correction"
```

---

## Phase 3: Paper Restructure (Can parallelize tasks 10-16)

### Task 10: Restructure paper around quantitative predictions [C11]

**Files:**
- Modify: `paper/nature_article.tex` (major restructure)
- Modify: `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: Plan the new structure**

New outline:
1. Introduction — open with concrete example ("two equally accurate cancer models rank the same gene as most important and least important")
2. The Impossibility (1 paragraph + theorem statement + proof sketch) — scaffolding, not centerpiece
3. Quantitative Predictions — Noether counting (g(g-1)/2 stable queries), η law (dim(V^G)/dim(V)), interpretability ceiling (1/n)
4. Empirical Validation — results for Noether, η, ceiling, plus honestly reported falsified predictions
5. Cross-Domain Context — 2-3 strongest instances as motivation (ML attribution, causal/CPDAG, biology/codon)
6. Discussion — limitations, path forward, regulatory implications

- [ ] **Step 2: Rewrite the abstract**

Open with a concrete example. Delay all formalism. End with the quantitative predictions, not the impossibility.

```latex
Two machine learning models, equally accurate on held-out data, can rank
the same biomarker as the most important and least important predictor
of disease.  We prove this instability is unavoidable: ...
```

- [ ] **Step 3: Move Noether counting and η law from Discussion to Results**

These are currently buried in Discussion paragraphs added during the knockout experiments. They should be in the Results section with figures.

- [ ] **Step 4: Demote 5 of 8 non-ML instances to a brief "Related Work" table**

Keep: ML attribution, causal discovery (CPDAG), biology (codon). Move to one-row-each table: physics (gauge + stat mech), crystallography, linguistics, database, linear algebra.

- [ ] **Step 5: Cut to target word count**

For Nature MI: ~5,000 words. For JMLR: no limit (keep monograph as-is, restructure the submission version).

- [ ] **Step 6: Commit**

```bash
git add paper/nature_article.tex paper/universal_impossibility_monograph.tex
git commit -m "refactor: restructure paper around quantitative predictions as centerpiece"
```

### Task 11: Strengthen the "isn't this obvious?" defense [C12]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex` (Section: Why This Result Is Non-Trivial)

- [ ] **Step 1: Rewrite the defense with three concrete arguments**

```latex
\paragraph{Why the axiom set is not trivially unsatisfiable.}
Three pieces of evidence:
(1)~\emph{Tightness}: each pair of properties is achievable with
Lean-verified witnesses. If the definitions were rigged, no pair
would be satisfiable.
(2)~\emph{Biconditional}: the impossibility holds \emph{if and only if}
the Rashomon property is present (Proposition~2). This means the
axiom set precisely characterizes the phenomenon --- it is neither
too strong (it holds exactly when expected) nor too weak (it fails
exactly when expected).
(3)~\emph{Quantitative corollaries}: the Noether counting theorem
(exactly $\binom{g}{2}$ stable queries) and the $\eta$ law
($R^2 = 0.957$) are non-obvious predictions that follow from the
framework but not from any simpler observation about non-injectivity.
```

- [ ] **Step 2: Add an explicit comparison to Arrow's theorem**

```latex
\paragraph{Comparison with Arrow's theorem.}
Arrow's impossibility is more surprising because IIA, Pareto, and
non-dictatorship appear independently mild. Our axioms are more
transparently in tension --- stability and decisiveness already pull
in opposite directions when fibers are non-trivial. The contribution
is not surprise but \emph{precision}: the axiom set is tight
(each pair achievable), necessary and sufficient (biconditional),
and quantitatively productive (Noether counting, $\eta$ law).
We do not claim depth comparable to Arrow; we claim a precise
characterization of a ubiquitous phenomenon.
```

- [ ] **Step 3: Commit**

```bash
git add paper/universal_impossibility_monograph.tex
git commit -m "docs: strengthen 'non-trivial' defense with tightness, biconditional, quantitative corollaries"
```

### Task 12: Fix terminology — rename bridge theorems [C16]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: Rename "uncertainty principle" to "explanation tradeoff bound"**

```bash
sed -i '' 's/Uncertainty Principle for Explanations/Explanation Tradeoff Bound/g' paper/universal_impossibility_monograph.tex
sed -i '' 's/uncertainty principle/explanation tradeoff bound/g' paper/universal_impossibility_monograph.tex
```

- [ ] **Step 2: Rename "Noether's Principle" to "Invariance Counting Principle"**

```bash
sed -i '' 's/Discrete Analogue of Noether.s Principle/Invariance Counting Principle/g' paper/universal_impossibility_monograph.tex
```

Keep a parenthetical "(cf.\ Noether's theorem for continuous symmetries)" for context but don't claim the name.

- [ ] **Step 3: Remove "Planck's constant" analogy**

Delete the sentence "The Rashomon ratio $r$ plays the role of Planck's constant $\hbar$".

- [ ] **Step 4: Commit**

```bash
git add paper/universal_impossibility_monograph.tex
git commit -m "fix: rename 'uncertainty principle' and 'Noether' terminology to avoid overclaiming"
```

### Task 13: Remove Flyspeck/4CT comparison [C5]

**Files:**
- Modify: `paper/nature_article.tex`
- Modify: `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: Find and remove all Flyspeck/4CT references**

```bash
grep -n "Flyspeck\|four.colour\|four.color\|4CT" paper/nature_article.tex paper/universal_impossibility_monograph.tex
```

- [ ] **Step 2: Replace with honest characterization**

```latex
The core impossibility proof is four tactic steps; the formalization's
value lies in mechanically verifying the eight constructive domain
instances (each requiring correct encoding of domain-specific
concepts) and the GBDT quantitative bounds (which involve real
arithmetic and Mathlib's analysis library).
```

- [ ] **Step 3: Commit**

```bash
git add paper/nature_article.tex paper/universal_impossibility_monograph.tex
git commit -m "fix: remove misleading Flyspeck/4CT comparison, add honest characterization"
```

### Task 14: Reframe resolutions as classification, not motivation [C14]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`
- Modify: `paper/nature_article.tex`

- [ ] **Step 1: Find all "resolution" framing that implies the framework motivated existing practice**

```bash
grep -n "resolution.*strategy\|independently.*discovered\|arrived at the same\|mathematical inevitability" paper/nature_article.tex paper/universal_impossibility_monograph.tex
```

- [ ] **Step 2: Replace with classification framing**

Before: "eight independent communities arrived at the same strategy"
After: "eight domains independently developed what we formally classify as G-invariant resolutions — the framework identifies the common optimality structure, not the historical motivation"

Before: "Wilson loops are G-invariant resolutions"
After: "Wilson loops are the gauge-invariant observables of the theory; within our framework, they correspond to the G-invariant projection, though historically they were introduced as construction principles, not as responses to an impossibility"

- [ ] **Step 3: Commit**

```bash
git add paper/nature_article.tex paper/universal_impossibility_monograph.tex
git commit -m "fix: reframe resolutions as classification of existing practice, not motivation"
```

### Task 15: Engage with Rashomon set literature [C17, C18]

**Files:**
- Modify: `paper/references.bib`
- Modify: `paper/universal_impossibility_monograph.tex` (Related Work section)

- [ ] **Step 1: Add missing citations**

Add to references.bib:
- Laberge et al. (2023) — partial orders from Rashomon sets
- Marx et al. (2024, NeurIPS) — uncertainty-aware explainability
- Selbst & Barocas (2018) — legal impossibility of explanation
- Han et al. (2022) — "Which Explanation Should I Choose?"

- [ ] **Step 2: Write a comparison paragraph in Related Work**

```latex
\paragraph{Rashomon set program.}
Fisher et al.~\citep{fisher2019all} introduced variable importance
clouds visualizing explanation instability across the Rashomon set.
Laberge et al.~\citep{laberge2023partial} independently discovered
the attribution-specific instance of our resolution: extracting
partial orders (consensus rankings) from Rashomon sets.
Marx et al.~\citep{marx2024uncertainty} develop uncertainty-aware
explainability accounting for model multiplicity. Our contribution
relative to this body of work is the abstract framework (subsuming
non-ML domains), the Lean formalization, and the quantitative
$\eta$ law; the core insight --- Rashomon implies explanation
instability, aggregation resolves it --- is shared.
```

- [ ] **Step 3: Add Selbst & Barocas to regulatory discussion**

- [ ] **Step 4: Commit**

```bash
git add paper/references.bib paper/universal_impossibility_monograph.tex
git commit -m "docs: engage with Rashomon set literature (Fisher, Laberge, Marx, Selbst)"
```

### Task 16: Fix regulatory citations [C19] and acknowledge limitations [L1-L8]

**Files:**
- Modify: `paper/nature_article.tex`
- Modify: `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: Cite specific EU AI Act provisions**

Replace "EU AI Act requires meaningful explanations" with:
"The EU AI Act (Regulation 2024/1689), Article 13, requires 'sufficient transparency to enable deployers to interpret the system's output,' and Article 86 grants individuals 'a right to explanation of individual decision-making.'"

- [ ] **Step 2: Cite specific FDA guidance**

"FDA, 'Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device Action Plan' (January 2021)."

- [ ] **Step 3: Add limitation acknowledgments for L1-L8**

Add to the Limitations section of the monograph:

```latex
\item \textbf{Elementary core theorem.} The qualitative impossibility
  follows from elementary logic (a non-injective map cannot be inverted).
  The contribution is the axiomatic framework (tightness, biconditional,
  resolution optimality) and the quantitative corollaries.
\item \textbf{Broken symmetry in trained networks.} The $1/n$
  interpretability ceiling assumes unbroken $S_n$ permutation symmetry.
  Training breaks this symmetry; the bound is an upper limit on the
  fraction of neuron-level interpretations that are \emph{necessarily}
  stable, not a characterization of trained-network interpretability.
\item \textbf{Synonymous codons are not biologically equivalent.}
  Different codons for the same amino acid differ in translational
  efficiency, mRNA stability, and epigenetic marking. The gauge analogy
  applies to the protein-coding function only.
```

- [ ] **Step 4: Commit**

```bash
git add paper/nature_article.tex paper/universal_impossibility_monograph.tex
git commit -m "fix: cite specific EU AI Act articles, FDA guidance; add limitation acknowledgments"
```

---

## Phase 4: Lean Extensions (Can run in parallel with Phase 2-3)

### Task 17: Justify group assignments for ML instances [C13]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`

- [ ] **Step 1: For each ML instance, add a "Group justification" paragraph**

For attention (S₆): "We model 6 attention heads as exchangeable under S₆. In practice, heads specialize during training, partially breaking this symmetry. The η prediction (1-1/6 = 83%) is therefore an upper bound on instability; the observed 60% (perturbation) / 20% (retraining) reflects this symmetry-breaking."

For concept probes (O(64)): "The concept direction lies in ℝ⁶⁴; the full orthogonal group O(64) acts on this space. In practice, the data covariance breaks rotational symmetry to a subgroup. The η prediction (≈1.0) is an upper bound."

- [ ] **Step 2: Add a table summarizing justifications**

- [ ] **Step 3: Commit**

```bash
git add paper/universal_impossibility_monograph.tex
git commit -m "docs: justify group assignments for all ML instances with symmetry-breaking caveats"
```

### Task 18: Fix incompatibility relation in MarkovEquivalence.lean [C28]

**Files:**
- Modify: `UniversalImpossibility/MarkovEquivalence.lean`

- [ ] **Step 1: Check current incompatibility definition**

```bash
grep -A3 "incompatible\|incomp" UniversalImpossibility/MarkovEquivalence.lean | head -20
```

- [ ] **Step 2: If it uses `!=`, add a comment explaining the choice**

The paper's instance_causal.tex uses edge-reversal incompatibility. If the Lean file uses `!=` (any distinct DAGs are incompatible), add a comment:

```lean
/-- Incompatibility is defined as inequality (any distinct DAGs are incompatible).
    This is the weakest choice, which makes the impossibility stronger:
    if the impossibility holds with the weakest incompatibility, it holds
    a fortiori with any stronger relation (e.g., edge reversal). -/
```

- [ ] **Step 3: Rebuild and verify**

```bash
lake build UniversalImpossibility.MarkovEquivalence
```

- [ ] **Step 4: Commit**

```bash
git add UniversalImpossibility/MarkovEquivalence.lean
git commit -m "docs: clarify incompatibility relation choice in MarkovEquivalence.lean"
```

### Task 19: Verify existing Pareto-optimality proof [C15]

**Files:**
- Verify: `UniversalImpossibility/ParetoOptimality.lean` (301 lines, already exists, 0 sorry)

**NOTE**: This file already exists and is fully proved. The plan originally said "Create" — corrected after vet.

- [ ] **Step 1: Read ParetoOptimality.lean and catalog what's proved**

```bash
grep "^theorem\|^lemma" UniversalImpossibility/ParetoOptimality.lean
```

Verify: `dash_pareto_dominance_within_group`, `dash_pareto_gap_exact`, `pareto_frontier_dichotomy` are all proved (no sorry).

- [ ] **Step 2: Verify it compiles cleanly**

```bash
lake build UniversalImpossibility.ParetoOptimality
```

- [ ] **Step 3: Check whether the paper claims match what's proved**

The monograph says Pareto-optimality is "argued (supplement only)." If ParetoOptimality.lean proves it, update the monograph's proof status table to "Proved (ParetoOptimality.lean)."

- [ ] **Step 4: Update proof status in monograph**

Change "Argued (supplement proof only)" to "Lean-verified (ParetoOptimality.lean)" in the proof transparency table.

- [ ] **Step 5: Commit**

```bash
git add paper/universal_impossibility_monograph.tex
git commit -m "docs: update proof status — Pareto-optimality is Lean-verified (ParetoOptimality.lean)"
```

---

## Phase 5: Infrastructure (After Phases 1-4)

### Task 20: Pin all dependencies and create reproducibility pipeline [C33, C34, C35]

**Files:**
- Modify: `paper/scripts/requirements.txt`
- Create: `Dockerfile`
- Create: `.github/workflows/reproduce.yml`
- Create: `paper/data/cytochrome_c_sequences.fasta` (committed static data)

- [ ] **Step 1: Pin PyTorch and Transformers**

Add to requirements.txt:
```
torch==2.4.1
transformers==4.44.0
```

- [ ] **Step 2: Commit cytochrome c sequences**

Save the 120 NCBI sequences as a static file so results don't depend on a live API.

- [ ] **Step 3: Create Dockerfile**

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir -r requirements.txt
COPY . /workspace
WORKDIR /workspace
CMD ["make", "validate"]
```

- [ ] **Step 4: Create Makefile with validate target**

```makefile
validate:
	lake build
	python3 paper/scripts/run_all_universal_experiments.py
	bash paper/scripts/verify_counts.sh
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .github/workflows/reproduce.yml paper/scripts/requirements.txt paper/data/ Makefile
git commit -m "infra: pin deps, add Docker, commit static data, create reproducibility pipeline"
```

---

## Execution Order Summary

```
Phase 1 (sequential, blocking):
  Task 1 → Task 2 → Task 3 → Task 4

Phase 2 (parallel after Phase 1):
  Task 5 (Noether sensitivity) ←→ Task 6 (attention retraining)
  Task 7 (η methodology) ←→ Task 8 (power analyses)
  Task 9 (statistical fixes)

Phase 3 (parallel after Phase 1):
  Task 10 (restructure) — do first, then:
  Task 11 (obvious defense) ←→ Task 12 (terminology)
  Task 13 (Flyspeck) ←→ Task 14 (reframe resolutions)
  Task 15 (Rashomon lit) ←→ Task 16 (regulatory + limitations)

Phase 4 (parallel after Phase 1):
  Task 17 (group justifications) ←→ Task 18 (Lean incompatibility)
  Task 19 (Pareto-optimality) — longest Lean task

Phase 5 (after all):
  Task 20 (infrastructure)
```

**Estimated total effort:** 8-12 hours across 2-4 sessions.
Phase 1: ~1 hour. Phase 2: ~2 hours (using existing retraining data) to ~6 hours (if re-running attention from scratch). Phase 3: ~3 hours. Phase 4: ~1 hour (Pareto already proved, just verify and document). Phase 5: ~1 hour.

**Contingency — Noether sensitivity failure:** If the bimodal gap collapses at ρ=0.7, reframe the result: "The Noether counting prediction holds exactly at high correlation and degrades gracefully, with the gap scaling as f(ρ). The prediction is quantitatively precise for the regime where SHAP instability is most severe (ρ > 0.85) and provides a qualitative bound elsewhere." This is still publishable — a sensitivity curve is more informative than a single extreme point.

**Build verification gate:** After completing Phases 2-4, run `lake build` to confirm all Lean files still compile. If any task modified .lean files, this is mandatory before committing.

**P2 items (C20-C35):** These are deferred to a follow-up session. Create a tracking issue or TODO file listing all P2 items with their correction numbers.
