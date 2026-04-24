# Universal Explanation Impossibility — Landmark Paper Plan

**Goal**: Transform the Universal Explanation Impossibility from a strong
framing contribution into a foundational, unassailable result that
redefines how the field thinks about explainability.

**Current state**: 70 Lean files, 338 theorems, 68 axioms, 0 sorry.
8 instances, 6 experiments with controls and resolution tests.
26-page paper in 4 formats.

**What separates "good paper" from "landmark"**:

A landmark paper does THREE things:
1. Proves something people knew informally but couldn't state precisely
2. Shows it's UNAVOIDABLE (not a quirk of bad methods)
3. Provides a CONSTRUCTIVE resolution that changes practice

The current paper does all three. What it needs to become unassailable:

---

## TIER 1: MAKE THE THEOREM BULLETPROOF [Critical]

### Phase 1: Strengthen the Core Proof

**Task 1.1: Derive the Rashomon property from first principles [Opus]**

THE GAP: The Rashomon property is currently AXIOMATIZED per instance.
A reviewer says: "You axiomatize the interesting part and prove the
easy part." This is the #1 vulnerability.

THE FIX: For at least 2 instances, DERIVE the Rashomon property from
standard mathematical assumptions — not axiomatize it.

Instance 1 — Attribution (GBDT):
The Rashomon property for GBDT follows from:
- Collinearity: corr(X_j, X_k) = ρ > 0
- The GBDT split selection mechanism
- Already partly done in Trilemma.lean (RashimonProperty from axioms)
- STRENGTHEN: derive RashimonProperty from a WEAKER set of axioms.
  Ideally from just "the model class is rich enough to fit the data
  with different feature utilization patterns."

Instance 2 — Causal discovery:
The Rashomon property for DAGs IS a theorem — Markov equivalence is
a proved mathematical fact (Verma & Pearl 1990). We should DERIVE
it in Lean, not axiomatize it. The statement: if two DAGs have the
same skeleton and v-structures, they have the same conditional
independence relations. The converse gives the Rashomon property.

This converts the causal instance from "axiomatized" to "derived" —
a massive credibility upgrade.

**Deliverable**: At least 1 instance where the Rashomon property is
a THEOREM, not an axiom. Ideally 2.

**Task 1.2: Prove the theorem is SHARP [Opus]**

THE GAP: The impossibility says "you can't have all three." But how
CLOSE can you get? The tightness proof shows each pair is achievable
but says nothing about QUANTITATIVE tradeoffs.

THE FIX: Prove a quantitative version. For the abstract framework:

Theorem (Quantitative Impossibility):
For any stable explanation E of a system with Rashomon group G,
the minimum unfaithfulness is:
  inf_E { d(E, explain) : stable(E) } = f(|G|)
where f is a decreasing function of the Rashomon set structure.

For attributions, this should recover the 1/(1-ρ²) divergence rate.
For the abstract framework, it should be a function of the symmetry
group order or the number of incompatible equivalence classes.

**Deliverable**: Lean theorem quantifying the price of stability.

**Task 1.3: Prove UNIQUENESS of the resolution [Opus]**

THE GAP: The G-invariant projection is shown to be stable, but is it
the UNIQUE optimal resolution? Or just one of many?

THE FIX: Prove that among all stable methods, the G-invariant
projection is Pareto-optimal in a precise sense: no other stable
method is more faithful on every input.

This is the Hunt-Stein theorem. Formalize it in Lean for the
abstract ExplanationSystem framework. The statement:

Theorem (Resolution Uniqueness):
Let E be any stable explanation for a system with Rashomon symmetry G.
If E is faithful (¬incompatible(E(θ), explain(θ)) for all θ), then
E is a coarsening of the G-invariant projection.

**Deliverable**: Lean-verified optimality of the resolution.

---

## TIER 2: MAKE THE EXPERIMENTS UNASSAILABLE [High priority]

### Phase 2: Gold-Standard Experiments

**Task 2.1: True multi-seed retraining for attention [Sonnet]**

THE GAP: The attention experiment uses weight perturbation, not true
retraining. A reviewer says: "perturbation is artificial."

THE FIX: Fine-tune DistilBERT on SST-2 from 10 different random
initializations of the classification head (freeze the transformer,
vary only the head). This is true retraining — different optimization
trajectories — but fast enough to run on CPU.

Compare: perturbation flip rate vs retraining flip rate. If they're
similar, this validates the perturbation methodology. If they differ,
report both and discuss.

**Task 2.2: Scale up concept probes [Sonnet]**

THE GAP: load_digits with MLPs is a toy setup. A reviewer says:
"does this hold on real vision models?"

THE FIX: Use a pretrained ResNet-18 or ResNet-50 as the feature
extractor (freeze weights). Train 15 different LINEAR PROBES on
the penultimate features for a concept (e.g., "outdoor" vs "indoor"
on a subset of CIFAR-10, or use ImageNet features).

The frozen encoder means ALL 15 probes see the SAME features. The
linear probe coefficients should be stable (like logistic regression
on the same features). This is the POSITIVE control for the concept
experiment: when the representation is fixed, concept probes ARE
stable.

Then: unfreeze the encoder and fine-tune 15 models from different
seeds. NOW the concept probes should diverge. This demonstrates
that it's the REPRESENTATION LEARNING, not the probing, that
causes instability.

**Task 2.3: Multi-dataset counterfactual validation [Sonnet]**

THE GAP: Only German Credit. A reviewer says: "one dataset."

THE FIX: Run the counterfactual experiment on 3 datasets:
- German Credit (already done)
- Adult Income (UCI, >30K samples, standard fairness benchmark)
- COMPAS (recidivism, controversial, high-stakes)

Report results for all 3. If the instability pattern holds across
all 3, it's robust. If not, investigate why and report honestly.

**Task 2.4: Pre-registration of experimental hypotheses**

THE FIX: Write a brief experimental protocol document that states:
- The hypothesis (Rashomon → instability)
- The metric (flip rate > 20%)
- The control hypothesis (no Rashomon → flip rate < 5%)
- The analysis plan (bootstrap CIs, Wilcoxon tests)

Timestamp it (git commit). This prevents accusations of p-hacking
or post-hoc hypothesis selection.

---

## TIER 3: MAKE THE FRAMING FOUNDATIONAL [High impact]

### Phase 3: Connect to Deep Theory

**Task 3.1: Information-theoretic formulation [Opus]**

THE FIX: Prove an information-theoretic version of the impossibility.
The data processing inequality gives:

I(E; Θ) ≤ I(E; Y) ≤ H(Y)

Faithfulness requires I(E; Θ) to be high (E captures information
about the configuration). Stability requires E to factor through Y
(E depends only on observables). These compete when H(Θ|Y) > 0
(when the system is underspecified).

Theorem (Information-Theoretic Impossibility):
For any stable E: I(E; Θ|Y) = 0. Therefore I(E; Θ) ≤ I(Y; Θ).
Faithfulness requires I(E; Θ) ≥ I(explain; Θ). If explain carries
more information about Θ than Y does, no stable E can be faithful.

This gives a QUANTITATIVE version: the information gap
I(explain; Θ) - I(Y; Θ) is the fundamental limit.

Formalize in Lean if possible; otherwise present as a rigorous
paper proof.

**Task 3.2: Connect to computational complexity [Opus]**

THE FIX: Prove that even APPROXIMATELY satisfying all three
properties is hard.

Theorem (Computational Hardness):
For the causal discovery instance, determining whether a given
explanation is ε-faithful is coNP-hard (because testing Markov
equivalence is hard in general).

This shows the impossibility is not just information-theoretic
but also computational — you can't even VERIFY faithfulness
efficiently.

**Task 3.3: The metatheorem about metatheorems [Opus]**

THE FIX: Prove that any triple of properties satisfying certain
structural conditions (one is "local to the configuration", one
is "invariant across equivalence classes", one is "commits to
distinctions") will be impossible under the Rashomon property.

This shows the impossibility is not specific to the PARTICULAR
definitions of faithful/stable/decisive but to the STRUCTURE of
the trilemma. Any "faithfulness-stability-decisiveness" triple
with the right structure hits the same wall.

This is the analog of Arrow's theorem being robust to specific
axiom formulations — it's not about IIA specifically, but about
the STRUCTURE of the aggregation problem.

---

## TIER 4: MAKE THE PAPER UNDENIABLE [Polish]

### Phase 4: Presentation and Positioning

**Task 4.1: The "one-page version" [Opus]**

Write a 1-page summary that any ML researcher can read in 5 minutes
and understand the complete result. This goes in the introduction
and/or as a standalone document.

The structure:
- Setup: 3 sentences defining the framework
- Theorem: 1 sentence
- Tightness: 3 sentences (each pair achievable)
- Instances: 1 table (8 rows)
- Experiments: 1 table (cross-instance)
- Resolution: 2 sentences
- Implications: 2 sentences

If a reader can grasp the entire contribution in 1 page, they'll
cite it. If they can't, they won't.

**Task 4.2: The visual abstract [Sonnet]**

Create a single figure (the "trilemma diagram") that captures the
entire result visually:
- Triangle with Faithful/Stable/Decisive at the vertices
- Each EDGE labeled with the achievable pair (E = explain,
  E = neutral constant, E = committal constant)
- The INTERIOR labeled "Impossible (Rashomon)"
- Around the triangle: the 8 instances, each with an icon and
  their flip rate

This figure should be the FIRST thing in the paper. It's the
equivalent of Arrow's impossibility diagram or the CAP theorem
triangle.

**Task 4.3: The "escape routes" analysis [Opus]**

For each property, analyze EXACTLY what you get when you drop it:
- Drop faithful: you get LIME/SHAP-like methods that are stable
  and decisive but may not reflect the model
- Drop stable: you get per-model explanations that are accurate
  but change on retraining
- Drop decisive: you get DASH/CPDAG/ensemble explanations that
  are accurate and stable but have ties

For each escape route, give a CONCRETE example from practice and
cite existing tools that take that escape. This shows the trilemma
is not abstract — it describes the actual landscape of XAI tools.

**Task 4.4: The regulatory analysis [Opus]**

Write a dedicated subsection analyzing:
- EU AI Act Article 13 (transparency): requires "meaningful explanations"
- NIST AI RMF: requires "explainability" as a trustworthy AI property
- OCC SR 11-7: requires "developmental evidence" for model decisions

For each: which legs of the trilemma does the regulation demand?
Can any explanation method satisfy all demands simultaneously?
What does the resolution (G-invariant aggregation) mean in
regulatory terms?

This connects the theorem to REAL-WORLD CONSEQUENCES and makes
it immediately relevant to policy.

---

## TIER 5: MAKE THE ARTIFACT PERMANENT [Longevity]

### Phase 5: Reproducibility and Legacy

**Task 5.1: Docker container [Sonnet]**

Create a Dockerfile that:
1. Installs Lean 4 + Mathlib
2. Installs Python + all experiment dependencies
3. Runs `lake build` to verify the Lean formalization
4. Runs all experiments to verify reproducibility
5. Compiles the paper

Any researcher can pull the container and verify EVERYTHING in
one command: `docker run universal-impossibility verify`

**Task 5.2: Interactive proof explorer [Sonnet]**

Create a simple web page (single HTML file with embedded JS) that:
- Shows the theorem statement
- Lets the user click on each definition to see the Lean code
- Lets the user click on each instance to see the instantiation
- Lets the user toggle between the 3 escape routes
- Shows the experiment results interactively

Host on GitHub Pages. This makes the result ACCESSIBLE to
non-Lean-literate researchers.

**Task 5.3: Textbook-ready exposition [Opus]**

Write a self-contained 10-page "tutorial" version of the result
suitable for inclusion in a textbook or graduate course:
- Minimal prerequisites (basic set theory, functions)
- Full proofs (no "see appendix")
- Worked examples for 3 instances
- Exercises for students
- Historical context (Arrow, bias-variance, CAP)

This is the ultimate longevity play: if it gets into textbooks,
it's foundational by definition.

---

## EXECUTION PRIORITY

The tiers are ordered by impact per effort:

**Tier 1 (Critical — do first):**
- Task 1.1: Derive Rashomon for ≥1 instance (biggest credibility upgrade)
- Task 1.2: Quantitative bound (converts qualitative → quantitative)
- Task 1.3: Resolution uniqueness (completes the constructive story)

**Tier 2 (High — do second):**
- Task 2.1: True retraining for attention
- Task 2.2: Scale up concept probes
- Task 2.3: Multi-dataset counterfactuals

**Tier 3 (High impact — do if time allows):**
- Task 3.1: Information-theoretic formulation
- Task 3.3: Metatheorem about metatheorems

**Tier 4 (Polish — do before submission):**
- Task 4.1: One-page version
- Task 4.2: Visual abstract (trilemma diagram)
- Task 4.3: Escape routes analysis
- Task 4.4: Regulatory analysis

**Tier 5 (Longevity — do after acceptance):**
- Task 5.1: Docker container
- Task 5.2: Interactive explorer
- Task 5.3: Textbook exposition

---

## TIMELINE (for NeurIPS May 4/6)

Given 25 days:

Week 1 (Days 1-7):
- Tier 1: derive Rashomon for causal (most tractable), quantitative bound
- Tier 2: true retraining attention, multi-dataset counterfactual

Week 2 (Days 8-14):
- Tier 1: resolution uniqueness
- Tier 2: scale concept probes
- Tier 3: information-theoretic formulation (if tractable)

Week 3 (Days 15-21):
- Tier 4: all presentation tasks
- Paper integration and polish

Days 22-25:
- Final vet, compilation, submission

---

## WHAT MAKES IT A LANDMARK

If you execute Tiers 1-4, the paper has:

1. **A theorem** that is sharp (quantitative), tight (3 witnesses),
   robust (axiom substitution), and verified (Lean, 0 sorry)

2. **A resolution** that is unique (Pareto-optimal), constructive
   (DASH/CPDAG/ensemble), and connected to classical theory
   (Hunt-Stein)

3. **8 instances** spanning the entire landscape of XAI methods,
   each with empirical validation including negative controls

4. **An information-theoretic formulation** connecting to
   fundamental limits

5. **A regulatory analysis** connecting to real-world policy

6. **A visual identity** (the trilemma triangle) that becomes
   the standard diagram for the field

No existing paper in XAI has all of these. The combination is
what makes it foundational.

---

## CONFIDENCE ASSESSMENT

| Component | Achievable by May 6? | Impact |
|-----------|:---:|:---:|
| Derive Rashomon (causal) | YES (Markov equiv. is pure math) | Very high |
| Quantitative bound | MEDIUM (may need axioms) | Very high |
| Resolution uniqueness | MEDIUM (Hunt-Stein in Lean is hard) | High |
| True retraining attention | YES (freeze + retrain head) | High |
| Multi-dataset CF | YES (mechanical) | Medium |
| Scale concept probes | YES (frozen encoder + probe) | Medium |
| Info-theoretic formulation | MEDIUM (paper proof OK, Lean hard) | High |
| One-page version | YES | High |
| Trilemma diagram | YES | Very high |
| Escape routes | YES | High |
| Regulatory analysis | YES | High |

The trilemma diagram and one-page version are the highest
impact-per-hour items. The Rashomon derivation for causal
discovery is the highest credibility upgrade.
