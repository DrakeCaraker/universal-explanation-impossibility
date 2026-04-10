# The Fiber Impossibility Program — Explainer

## What a Fiber Is

Take any function that maps inputs to outputs. The **fiber** of an
output is the set of all inputs that produce it.

If you train a neural network and it achieves 94% accuracy, the
fiber of "94% accuracy" is every possible weight configuration
that also achieves 94% accuracy. There are many — potentially
infinitely many. That collection is the fiber.

Formally: given a map `observe : Θ → Y`, the fiber of `y ∈ Y`
is `observe⁻¹(y) = {θ ∈ Θ : observe(θ) = y}`.

When the map is one-to-one, every fiber has exactly one element.
When it's many-to-one, fibers have multiple elements. **Many-to-one
is the norm in ML** — many weight configurations produce the same
predictions.

## Why Fibers Are the Heart of the Impossibility

The entire explanation impossibility reduces to one sentence:
**the fiber contains elements with incompatible explanations,
and no map that's constant on the fiber can agree with all of them.**

Unpack it:

1. **Stability** says E must be constant on each fiber (same
   output → same explanation)
2. **Faithfulness** says E must not contradict the native
   explanation at each point in the fiber
3. **Decisiveness** says E must be as specific as the native
   explanation at each point

If the fiber has two configurations θ₁ and θ₂ with incompatible
explanations, then:
- E(θ₁) must agree with explain(θ₁) [faithfulness at θ₁]
- E(θ₂) must agree with explain(θ₂) [faithfulness at θ₂]
- E(θ₁) = E(θ₂) [stability, since they're in the same fiber]
- But explain(θ₁) and explain(θ₂) are incompatible [Rashomon]

One value can't agree with two incompatible things. That's the
entire proof.

## The Fiber Impossibility Pattern

This structure — a map with non-trivial fibers, a feature you
want to recover from the compressed output, and properties that
conflict on the fiber — appears across many domains:

**In ML explainability**: The fiber of "same predictions" contains
models with different SHAP values, different attention patterns,
different concept directions. You can't pick one explanation that's
faithful to all of them.

**In causal discovery**: The fiber of "same conditional independence
relations" contains multiple DAGs (Markov equivalence class). You
can't pick one DAG that's faithful, stable, and decisive.

**In database theory**: The fiber of "same view" contains multiple
base states. You can't translate a view update back to a unique
base update (Bancilhon & Spyratos 1981).

**In econometrics**: The fiber of "same likelihood" contains
multiple structural parameters. You can't identify a unique causal
effect (Manski's partial identification).

**In medicine**: The fiber of "same symptoms" contains multiple
conditions. You can't give a single definitive diagnosis.

In every case, the mathematical structure is identical: a many-to-one
map creates fibers, the fibers contain elements with different
"explanations," and no single explanation can be chosen that satisfies
all three desiderata.

The fiber is the atom of the theory. Everything else — the trilemma,
the resolution, the quantitative bounds, the query-relative version —
is about what happens on and across fibers.

---

## The Four Levels

### Level 0: The Impossibility Exists

*"Faithful, stable, decisive — pick two."*

**Status**: COMPLETE. Theorem proved, Lean-verified, 9 instances,
empirically illustrated.

**What it proves**: When multiple configurations produce the same
output but have incompatible explanations, no explanation method can
satisfy all three properties. The Rashomon property is both sufficient
(Theorem 1) and necessary for fully specified systems (Proposition 3).
The definitions are uniquely calibrated (axiom substitution analysis).

**What it gives the world**: A vocabulary. Before this, each domain
had its own ad hoc complaint ("SHAP is unstable," "attention is
unreliable," "Markov equivalence limits causal discovery"). After
this, they're all the same theorem. The practitioner learns to ask:
"which property am I sacrificing?" The regulator learns: "demanding
all three is demanding the impossible." The researcher learns:
"switching methods doesn't help."

**What it doesn't answer**: How MUCH of each property can you have?
Which specific questions are answerable? It's a binary verdict —
impossible or not.

**Risk**: None. This level is done.

---

### Level 1: The Tradeoff Is Quantitative

*"Here's exactly how much of each you can have."*

**Status**: PARTIALLY DONE. The per-fiber bound is proved in Lean
(QuantitativeBound.lean: on every Rashomon pair, faithfulness +
stability forces a specific decisiveness failure). The full
measure-theoretic bound is stated as a remark. The domain-specific
bound ρ²/(1-ρ²) is in the companion paper.

**What it would prove**: For any stable explanation E with
unfaithfulness rate ε and instability rate δ, decisiveness d is
bounded: ε + δ + (1-d) ≥ μ(Rashomon). The bigger the Rashomon set,
the worse the tradeoff. Domain-specific versions give concrete
numbers for attributions (function of correlation), concept probes
(function of overparameterization), and attention (function of
perturbation magnitude).

**What it would give the world**: A design specification. Instead of
"you can't have all three," an engineer hears: "with DASH over 50
models on this dataset, you get 95% faithfulness, 100% stability,
73% decisiveness." A regulator can set requirements: "δ ≤ 0.05,
ε ≤ 0.10." The cross-method comparison maps SHAP, LIME, attention,
GradCAM, and DASH onto the tradeoff surface, showing exactly what
each tool sacrifices and how much.

**What it doesn't answer**: Which specific questions about the model
are stably answerable? The tradeoff is global — averaged over all
queries. A method might be perfectly stable on some questions and
wildly unstable on others, averaging to a moderate δ.

**Risk**: The union bound ε + δ + (1-d) ≥ μ(Rashomon) may be
loose — there may be a tighter bound that requires deeper analysis.
The domain-specific bounds may need additional axioms. The Lean
formalization requires extending ExplanationSystem with Mathlib's
measure theory, which is feasible but non-trivial.

---

### Level 2: Some Questions Are Answerable

*"Here's exactly which questions you can and can't answer stably."*

**Status**: PARTIALLY DONE. The query-relative impossibility theorem
is proved in Lean (QueryRelative.lean: for each query q with a
Rashomon witness, no E is q-faithful + q-stable + q-decisive). The
Arrow structural comparison is written (Appendix A of the monograph).
The query lattice characterization is conjectured but not proved.

**What it would prove**: The impossibility is per-query. For each
question q you might ask about a model ("is feature x important?",
"is x ranked above y?", "what's the top feature?"), the impossibility
holds if and only if the Rashomon set produces incompatible answers
to q. Questions where equivalent models agree are stably answerable.
Questions where they disagree are not.

**What is conjectured but not yet proved**: The answerable queries
form a downward-closed set in a natural lattice — coarser questions
(top-k membership) are more likely answerable than finer ones
(pairwise rankings). The boundary between answerable and unanswerable
corresponds to an information threshold:
I(explain_q; observe) ≥ H(explain_q). This lattice structure, if it
holds, would give practitioners a map of exactly where the
impossibility bites for their specific model.

**What it would give the world**: Precision. Instead of "you can't
explain this model," you get "you can stably answer these 7 questions
but not these 3." This is where the theory becomes genuinely useful
for individual decisions. This level also completes the Arrow
comparison: Arrow's IIA is query-relative stability for pairwise
queries, and the two theorems operate in complementary regimes (IIA
prevents pairwise Rashomon, so Arrow's impossibility arises from a
different mechanism than ours).

**Risk**: The lattice structure might not be as clean as
conjectured — the answerable/unanswerable boundary might depend on
the specific model and dataset in ways that resist general
characterization. The information-theoretic equivalence might hold
only for deterministic explanations.

---

### Level 3: All Aggregation Impossibilities Share a Pattern

*"Here's why compression under diversity always forces tradeoffs."*

**Status**: CONCEPTUAL. No theorems proved. The Arrow comparison
(complementary regimes, not subsumption) is the main finding so far.

**What it would characterize** (not prove — this is a framework,
not a theorem): Every impossibility result about aggregation involves
compressing a structured input into a simpler summary, and the
compression forces tradeoffs between natural properties. The fiber
pattern (Levels 0-2) covers cases where the impossibility arises from
information loss in the fibers of a many-to-one map. Arrow,
Chouldechova, and CAP have different mathematical structures but
share the philosophical shape.

The precise finding: **these impossibilities are NOT instances of a
single theorem.** They are independent results with a shared abstract
pattern. Arrow's impossibility arises from preference diversity under
aggregation; ours from observational equivalence under explanation;
Chouldechova's from base rate arithmetic under fairness. The diversity
conditions are different, the properties are different, the proofs
are different. What's shared is the SHAPE: three reasonable
desiderata, pairwise satisfiable, jointly impossible under a
diversity condition.

**What it would give the world**: A recipe for generating
impossibility results in new domains. Given a new domain: (1) identify
source space, compression, and feature map; (2) define faithfulness,
stability, decisiveness adapted to the domain; (3) identify the
diversity condition; (4) check if the fiber pattern applies; (5) if
yes, the impossibility follows from the Level 0 theorem; (6) if no,
you need a domain-specific argument. This recipe is a checklist, not
a magic wand — but it systematizes what is currently ad hoc.

**What it is NOT**: A single meta-theorem subsuming Arrow,
Chouldechova, and the explanation impossibility. The investigation
showed these are logically independent (IIA prevents the Rashomon
property; Arrow's impossibility mechanism is complementary to ours,
not nested within it). A category-theoretic formalization might
achieve formal unification, but at the cost of accessibility and
without adding explanatory power.

**Risk**: This level might never crystallize into a theorem. It might
remain a philosophical framework — useful for organizing thought and
generating research questions, but not a mathematical result. That's
still a contribution (Kuhn's *Structure of Scientific Revolutions*
is not a theorem), but it's a different KIND of contribution than
Levels 0-2.

---

## The Key Point

**Each level stands alone.** Level 0 is complete and valuable
regardless of whether Levels 1-3 ever happen. Level 1 adds
quantitative power but doesn't require Level 2. Level 2 adds
precision but doesn't require Level 3. Level 3 adds philosophical
coherence but doesn't require being a theorem.

The program is designed so that stopping at any point leaves a
complete contribution. The worst outcome is Level 0 alone — and
Level 0 alone is a PNAS-viable paper with 351 Lean-verified theorems
and 9 instances. That's not a worst outcome by any reasonable
standard.
