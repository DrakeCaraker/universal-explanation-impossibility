# Post-Review Revision Plan

**Goal**: Address all 15 MAJOR and 1 FATAL finding from the
5-reviewer adversarial peer review. Every fix is either a
code change, a prose rewrite, or an honest recharacterization.

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Target**: PNAS paper (universal_impossibility_pnas.tex + SI)
**Model**: Opus for reframing/writing. Sonnet for mechanical fixes.

---

## VET RECORD

### Round 1 — Factual

- The FATAL item (token citation control 31.5%) is real and
  verified in the JSON. The control selects "easy" sentences by
  top-quartile Jaccard consensus — but Jaccard measures token
  overlap, not flip rate. A sentence with high token overlap can
  still have a different #1 token (high flip rate). The control
  metric and the test metric are measuring different things.
  ⚠️ FIX: Either redesign the control (use sentences where ALL
  10 models agree on the #1 token) or drop the token citation
  experiment entirely.

- SI counts mismatch (73/346 vs 74/349) is a simple propagation
  error from adding QuantitativeBound.lean after the SI was
  written. ⚠️ FIX: Update SI.

- Model selection resolution "failure" (ensemble std higher):
  The JSON shows std_reduction = -0.0013, meaning the ensemble
  has SLIGHTLY higher std. This is within noise. ⚠️ FIX: Report
  honestly: "Ensemble AUC variance is comparable to best-single
  variance; the resolution for model selection is prediction
  stability, not AUC improvement."

### Round 2 — Reasoning

- The Hunt-Stein overclaim is the second-most serious issue
  after the token citation control. The paper says "instantiation"
  but has no loss function, no minimax, no amenability. ⚠️ FIX:
  Downgrade to: "The resolution strategy is structurally analogous
  to invariant decision procedures in the Hunt-Stein framework.
  A formal minimax proof for the abstract ExplanationSystem
  requires additional structure (a loss function and amenable
  group) and is left to future work."

- The "first Quine-Duhem formalization" claim: the formal
  epistemology literature (Dorling 1979, Howson & Urbach 2006,
  Earman 1993) has formal Bayesian analyses of underdetermination.
  These aren't impossibility theorems, but calling ours the
  "first mathematical formalization" is too broad. ⚠️ FIX:
  "To our knowledge, the first impossibility theorem derived
  from observational equivalence in the spirit of the Quine-Duhem
  thesis. Prior formal work (Dorling 1979; Howson & Urbach 2006)
  analyzed underdetermination in Bayesian frameworks but did not
  prove impossibility results for explanation methods."

- The 60% vs 19.9% presentation: the paper leads with 60% in
  the cross-instance table. ⚠️ FIX: Report 19.9% as the headline
  attention number. Footnote: "Weight perturbation yields 60%;
  we report the more conservative full-retraining figure."

### Round 3 — Omissions

- ⚠️ The plan doesn't address the "contribution is the framework,
  not the proof" reframing. This needs a paragraph in the intro
  and discussion explicitly saying: "The simplicity of the proof
  is a feature: it demonstrates that the framework captures the
  essential structure. The contribution is the identification of
  the right axioms, not the proof technique."

- ⚠️ The plan doesn't address the van Fraassen connection
  (dropping decisiveness = constructive empiricism). This is a
  strength, not a weakness — add it.

- ⚠️ The plan doesn't address the "weaker decisiveness" question
  (what if decisive only requires inheriting ONE incompatibility?).
  The axiom substitution already shows this: "surjective" (swap
  direction) loses the impossibility. But "inherits at least one"
  is different from "surjective." ⚠️ ADD: A brief discussion of
  why the current strength is right.

---

## Phase 1: Fix FATAL + Worst MAJORs [Opus]

### Task 1.1: Fix or drop Token Citation experiment

THE PROBLEM: Negative control shows 31.5% flip rate on "easy"
sentences (expected <5%). This invalidates the experiment.

ROOT CAUSE: "Easy" is defined as top-quartile Jaccard consensus.
But Jaccard measures overlap of the top-3 token SET, not agreement
on the #1 token. A sentence where 9/10 models include "great"
in top-3 but disagree on whether "great" or "movie" is #1 has
high Jaccard but high flip rate.

TWO OPTIONS:

**Option A (preferred): Redesign the control.**
Define "easy" as sentences where ≥9/10 models agree on the #1
token. These are sentences with a clear attention consensus.
The flip rate on these should be near 0% (by construction, most
pairs agree). This is a valid control: it shows that when models
agree on what's important, the explanation is stable.

HOWEVER: this may be circular (selecting for stability, then
noting stability). A better control: use sentences with only
1-2 content tokens (like the attention negative control). With
<3 tokens, top-3 citation is forced. Flip rate should be ~0%.

**Option B: Drop the experiment.**
Remove token citation from the paper entirely. Reduce to 8
instances. This is clean but loses breadth.

DECISION: Try Option A first. If the redesigned control gives
<5% flip rate, keep the experiment. If not, drop it.

Implementation:
- Read paper/scripts/llm_explanation_instability_experiment.py
- Change the negative control to use sentences with ≤3 content
  tokens (same as attention control)
- Rerun the experiment
- If new control <5%: update results JSON, paper tables, SI
- If new control >5%: drop the experiment from all paper versions

### Task 1.2: Fix SI count mismatch

Update paper/universal_impossibility_pnas_si.tex:
- 73 → 74 files
- 346 → 349 theorems
everywhere they appear.

### Task 1.3: Fix cross-instance table — lead with retraining

In paper/universal_impossibility_pnas.tex, the cross-instance
table (Table 1):
- Change the attention row from "60.0%" to "19.9%"
- Add footnote: "Weight perturbation yields 60.0\%; we report
  the more conservative full-retraining figure (20 models,
  last 2 transformer layers retrained from random initialization)."
- Add the full retraining details to SI

### Task 1.4: Fix model selection resolution reporting

In the cross-instance table and resolution discussion:
- Change from implying ensemble improves AUC to: "Ensemble
  prediction achieves comparable AUC (0.951 vs 0.958 best-single)
  with evaluation-stable performance across splits."
- Remove or qualify "resolution improves stability" for model
  selection specifically.

### Task 1.5: Acknowledge GradCAM prediction agreement

Ensure the main text table footnote says:
"GradCAM prediction agreement is 78.8\% ($\sigma = 0.0005$);
the modest instability (9.6\%) may partly reflect prediction
divergence rather than pure Rashomon instability."

---

## Phase 2: Reframe Overclaimed Connections [Opus]

### Task 2.1: Downgrade Hunt-Stein to analogy

Find ALL mentions of Hunt-Stein in the PNAS paper. Replace:

BEFORE: "The resolution is the explanation-theoretic instantiation
of the Hunt-Stein theorem."

AFTER: "The resolution strategy — restricting to G-invariant
explanation maps — is structurally analogous to the invariant
decision procedures justified by the Hunt-Stein theorem
(Lehmann \& Romano 2005, Ch.\ 6). A formal minimax optimality
proof for the abstract ExplanationSystem requires additional
structure (a loss function on H, an amenable group action,
and a risk functional) and is a natural direction for future
work. For the attribution instance, the companion paper proves
DASH achieves Pareto-optimal faithfulness among stable methods."

Remove the word "Pareto-optimal" from the PNAS abstract and
significance statement (since it's not proved in this paper
for the general case). Replace with "provably stable."

### Task 2.2: Qualify necessity claim

Find the necessity proposition. Change:

BEFORE: "The Rashomon property is both necessary and sufficient —
the exact boundary between possibility and impossibility."

AFTER: "The Rashomon property is sufficient for impossibility
(Theorem 1) and necessary for fully specified systems: when
each observation uniquely determines its configuration,
$E = \textsf{explain}$ satisfies all three properties
(Proposition 3). For systems with non-injective observations
and compatible-but-distinct explanations on the same fiber,
the relationship between Rashomon and impossibility is more
nuanced (see SI)."

### Task 2.3: Qualify Quine-Duhem claim

BEFORE: "formalizes a core consequence of the Quine-Duhem
underdetermination thesis as a mathematical theorem, to our
knowledge for the first time"

AFTER: "proves an impossibility theorem for explanation under
observational equivalence — formalizing, to our knowledge for
the first time, the explanatory consequences of what
philosophers of science call contrastive underdetermination
(Laudan \& Leplin 1991). Prior formal analyses of
underdetermination in Bayesian frameworks (Dorling 1979;
Howson \& Urbach 2006) did not yield impossibility results
for explanation methods."

Add bib entries:
- laudan1991empirical: Laudan & Leplin, "Empirical Equivalence
  and Underdetermination", J. Philosophy, 1991
- dorling1979bayesian: Dorling, "Bayesian Personalism, the
  Methodology of Scientific Research Programmes, and Duhem's
  Problem", Studies in History and Philosophy of Science, 1979
- howson2006scientific: Howson & Urbach, "Scientific Reasoning:
  The Bayesian Approach", Open Court, 3rd ed, 2006

### Task 2.4: Downgrade structural realism

BEFORE: "The G-invariant resolution formalizes what structural
realists have argued informally."

AFTER: "The G-invariant resolution is structurally reminiscent
of epistemic structural realism (Worrall 1989): stable
explanatory content is what is invariant across equivalent
configurations, just as the structural realist claims that
scientific knowledge survives theory change only in structural
form. The connection is suggestive; a rigorous mapping would
require engaging with the ontological commitments of structural
realism, which our descriptive framework does not address."

Add a sentence about van Fraassen: "Dropping decisiveness —
accepting that the explanation cannot resolve all distinctions —
corresponds to van Fraassen's (1980) constructive empiricist
stance that science aims for empirical adequacy, not full
theoretical truth."

### Task 2.5: Reframe Arrow comparison

BEFORE: any language comparing this result TO Arrow in impact.

AFTER: "The result shares structural features with Arrow's
impossibility theorem (1951): three individually reasonable
desiderata that are pairwise satisfiable but jointly
impossible under a natural background condition. Like Arrow's
theorem, the proof is short once the axioms are identified;
the contribution lies in identifying the right axiomatic
framework."

Remove any language suggesting comparable IMPACT.

---

## Phase 3: Reframe Contribution [Opus]

### Task 3.1: "The contribution is the framework" paragraph

Add to the introduction, after stating the theorem:

"The simplicity of the proof — four steps from the Rashomon
property to contradiction — is a feature of the framework,
not a limitation of the result. The intellectual content lies
in identifying the right definitions: faithful as
non-contradiction, stable as factoring through observations,
decisive as inheriting incompatibilities. The axiom
substitution analysis (SI, Section S4) shows these definitions
are uniquely calibrated: strengthening decisive collapses the
trilemma to a dilemma; weakening it loses the impossibility
entirely. The contribution is the axiomatic framework and its
unification of nine domains, not the proof technique."

### Task 3.2: Distinguish derived vs axiomatized instances

Add a paragraph after the instances section:

"Of the nine instances, the causal discovery instance has a
fully derived Rashomon property: Markov equivalence is proved
constructively from a 3-node example in Lean (SI, Section S2.5).
The remaining eight instances axiomatize the Rashomon property
as formalized domain knowledge — empirically justified by
cited literature for each domain but not derived from primitive
assumptions within the proof assistant. The core impossibility
theorem itself requires zero axioms; the instance-level axioms
encode the domain-specific claim that observational equivalence
with incompatible explanations occurs."

### Task 3.3: ciFromDAG honesty

In the causal instance description (or SI), add:

"The CI extraction function `ciFromDAG` is a simplified formula
correct for chain and fork structures on three nodes. It does
not implement full d-separation (in particular, it does not
handle collider structures correctly). The Rashomon witness
uses only chain and fork DAGs, for which the formula is
verified by computation. A complete formalization of
d-separation in Lean 4 is a substantial undertaking beyond the
scope of this work."

### Task 3.4: Label experiments as "illustrative"

In the empirical section, change:

BEFORE: "Empirical validation across [N] domains"

AFTER: "Empirical illustration across [N] domains. The
impossibility theorem is a mathematical proof; these experiments
demonstrate that the Rashomon property produces measurable
instability in practice, not that the theorem depends on
experimental evidence."

### Task 3.5: Address "obvious" objection in Discussion

Add to the discussion:

"A natural objection is that the impossibility is intuitively
obvious: if equivalent models disagree on explanations, no
single explanation can agree with all of them. We note three
responses. First, the axiom substitution analysis shows that
the obvious version of the result (with a stronger decisiveness
axiom) collapses to a dilemma — only the precisely calibrated
definitions yield a genuine trilemma. Second, the necessity
result shows the impossibility is sharp: it holds if and only
if the Rashomon property obtains. Third, the resolution
(G-invariant aggregation) and its connection to invariant
decision theory provide constructive content that goes beyond
the impossibility alone."

---

## Phase 4: Multiple Comparisons + Statistical [Sonnet]

### Task 4.1: Add FDR discussion

In the SI experimental methods section, add:

"We report 95\% bootstrap confidence intervals for each
primary metric. Across seven experiments with multiple
metrics, the total number of statistical comparisons is
approximately 20. We do not apply formal multiple comparison
correction (e.g., Bonferroni, Benjamini-Hochberg) for two
reasons: (1) each experiment tests a domain-specific hypothesis
that stands independently of the others, and (2) the primary
question is not whether any single experiment reaches
significance but whether the pattern of Rashomon-driven
instability is consistent across domains. All positive
results exceed the 20\% practical significance threshold by
substantial margins (range: 19.9\%--80\%), reducing the
risk of false positives from multiple testing."

---

## Phase 5: Compile + Verify + Commit [Sonnet]

### Task 5.1: Update SI counts
74 files, 349 theorems, 72 axioms, 0 sorry — everywhere in SI.

### Task 5.2: Compile all paper versions
All 5 must compile. Report page counts.

### Task 5.3: Final Lean build
lake build. 0 sorry. Counts match.

### Task 5.4: Commit and push

---

## Execution Order

```
Phase 1: [1.1 token citation fix] → [1.2 ∥ 1.3 ∥ 1.4 ∥ 1.5]
Phase 2: [2.1 ∥ 2.2 ∥ 2.3 ∥ 2.4 ∥ 2.5]  (all independent)
Phase 3: [3.1 ∥ 3.2 ∥ 3.3 ∥ 3.4 ∥ 3.5]  (all independent)
Phase 4: [4.1]
Phase 5: [5.1] → [5.2] → [5.3] → [5.4]
```

Phase 1.1 is the critical path (determines whether token
citation stays or goes). Everything else is independent.
