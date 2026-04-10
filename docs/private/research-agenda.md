# The Fiber Impossibility Program — Research Agenda

**Vision**: Develop a complete theory of when and how much
structured information survives compression. The explanation
impossibility is Level 0. The full theory characterizes
answerable vs unanswerable queries across science.

---

## The Hierarchy

```
Level 0: Explanation Impossibility          ← THIS PAPER (done)
         "Faithful, stable, decisive — pick two"
         9 ML instances, Lean-verified

Level 1: Fiber Impossibility Theory         ← Paper 4 (next)
         Quantitative bounds, approximate versions
         View updates + partial identification as instances

Level 2: Query-Relative Fiber Theory        ← Paper 5
         Which questions are answerable? Full characterization.
         Arrow structural comparison. Information geometry.

Level 3: Aggregation Framework              ← Paper 6-7
         Connecting fiber impossibilities to Arrow, Chouldechova, CAP.
         Shared vocabulary, not shared theorem.
         Philosophy of science venue.
```

---

## Level 0: The Explanation Impossibility (DONE)

**Status**: 75 Lean files, 351 theorems, 0 sorry. 9 instances.
Query-relative generalization. Quantitative remark. Two rounds
of adversarial review passed. PNAS (6pp), NeurIPS (10pp),
monograph (45pp) ready.

**Remaining actions**:
1. Post monograph to arXiv
2. Submit NeurIPS (May 4 abstract, May 6 paper)
3. Submit PNAS after NeurIPS decision

---

## Level 1: Fiber Impossibility Theory

**Goal**: Quantitative tradeoff surfaces. Approximate versions.
Non-ML instances formalized.

### Paper 4: Quantitative Tradeoffs
**Target**: JMLR or NeurIPS 2027
**Timeline**: Begin Q3 2026, submit Q1 2027

#### Core results to prove

**4.1 The tradeoff surface theorem**
For any ExplanationSystem with measure μ on Θ:
ε + δ + (1-d) ≥ μ(Rashomon)
where ε = unfaithfulness rate, δ = instability rate,
d = decisiveness rate.

Lean: extend ExplanationSystem with MeasureSpace Θ.
Use Mathlib's ProbabilityTheory.

**4.2 Approximate impossibility**
Define ε-faithful: P[incompatible(E(θ), explain(θ))] ≤ ε.
Define δ-stable: P[E(θ₁) ≠ E(θ₂) | observe(θ₁)=observe(θ₂)] ≤ δ.
Prove: ε + δ ≥ μ(Rashomon) · (1-d).
The binary impossibility is the special case ε=δ=0, d=1.

**4.3 Domain-specific tradeoff curves**
- GBDT attributions: ε ≥ ρ²/(1-ρ²) · (1-δ) (from companion paper)
- MLP concept probes: ε scales with overparameterization ratio
- Attention maps: ε scales with perturbation magnitude

**4.4 Cross-method comparison**
For the SAME model and input, compute (ε, δ, d) for:
SHAP, LIME, attention, GradCAM, DASH, integrated gradients.
Map each onto the tradeoff surface. Show:
- SHAP: high d, low δ, variable ε (drops stability when Rashomon is large)
- LIME: low d, low δ, high ε (approximation introduces unfaithfulness)
- DASH: d ≈ 1-μ(Rashomon), δ=0, ε≈0 (on the Pareto frontier)

**4.5 Regulatory compliance numbers**
For DASH on German Credit with 50 models:
- Compute actual (ε, δ, d) triple
- Compare to SR 11-7 requirements
- "Your explanation is X% faithful, Y% stable, Z% decisive"

#### Experiments
- Tradeoff curves: sweep ε, measure achievable (δ, d)
- Cross-method: SHAP vs LIME vs attention vs GradCAM vs DASH
  on 5 datasets, 20 seeds each
- Regulatory: German Credit + Adult Income + COMPAS

#### Lean work
- Measure-theoretic ExplanationSystem (extend with μ)
- Prove ε + δ + (1-d) ≥ μ(Rashomon)
- Prove tightness of the bound
- ~50-100 new theorems estimated

---

### Paper 4.5: Non-ML Fiber Instances (optional, can merge with Paper 4)

Formalize in Lean:
- View update problem (Bancilhon & Spyratos 1981) as ExplanationSystem
- Partial identification (Manski) as ExplanationSystem
- Prove both inherit the impossibility

These are mechanical (same pattern as ML instances) but
demonstrate the framework's reach beyond ML. Would strengthen
a PNAS submission if combined with Paper 4.

---

## Level 2: Query-Relative Fiber Theory

**Goal**: Full characterization of which queries are stably
answerable. The Arrow comparison. Information geometry.

### Paper 5: The Answerable Boundary
**Target**: Annals of Statistics or JMLR
**Timeline**: Begin Q1 2027, submit Q3 2027

#### Core results

**5.1 Query lattice structure**
Queries form a partial order: q₁ ≤ q₂ if answering q₂
answers q₁. The answerable queries (where Rashomon doesn't
hold) form a downward-closed set in this lattice. The
unanswerable queries form an upward-closed set. The boundary
is a cut in the lattice.

Example for feature attribution:
- "Is feature x relevant?" (top-k membership) — low in lattice
- "Is x ranked above y?" (pairwise comparison) — higher
- "What is the exact ranking of all features?" — top of lattice
The boundary cuts between "top-k membership" (answerable when
feature x is clearly important) and "pairwise comparison"
(unanswerable when x and y are correlated).

**5.2 Information-theoretic characterization**
The answerable boundary corresponds to the information
threshold I(explain_q; observe) ≥ H(explain_q).
Queries below the threshold are answerable; above are not.
This connects to rate-distortion theory.

**5.3 The Arrow comparison (full treatment)**
Expand the appendix from the monograph into a full analysis:
- IIA = query-relative stability (exact)
- Complementary regimes (IIA prevents Rashomon)
- What a unified framework would need
- Connection to Gibbard-Satterthwaite

**5.4 Probabilistic explanations**
Can randomized E break the impossibility?
Prove: no, faithfulness-in-expectation + stability +
decisiveness-in-expectation is still impossible.

**5.5 Side information theorem**
Additional information I helps iff I disambiguates the
Rashomon set: H(explain | observe, I) < H(explain | observe).
The minimum additional information needed to make query q
answerable is H(explain_q | observe).

#### Lean work
- Query lattice formalization
- Probabilistic impossibility
- Side information theorem
- ~100-150 new theorems estimated

---

## Level 3: The Aggregation Framework

**Goal**: Connect fiber impossibilities to non-fiber
impossibilities (Arrow, Chouldechova, CAP). Shared vocabulary.

### Paper 6: Impossibilities of Aggregation Under Diversity
**Target**: Philosophy of Science, or Synthese, or PNAS
**Timeline**: 2028

This is a FRAMEWORK PAPER, not a theorem paper. The contribution
is the shared vocabulary and the precise characterization of
where the structural parallel holds and where it breaks.

#### Structure

**6.1 The fiber pattern**
State the common structure: source → compression → feature,
diversity in fibers → impossibility of faithful+stable+decisive.
Show it covers: explanation, view updates, partial identification,
causal discovery, differential diagnosis.

**6.2 The non-fiber pattern**
Arrow, Chouldechova, CAP have a DIFFERENT structure:
- Arrow: the impossibility is about the aggregation rule itself,
  not about explaining an external system
- Chouldechova: the impossibility is arithmetic (base rate
  equations), not fiber-structural
- CAP: the impossibility is about fault tolerance, not
  information loss

**6.3 The shared abstraction (informal)**
"Any compression of a sufficiently diverse structured input
into a simpler summary must sacrifice some natural property
of the compression."

The diversity takes different forms:
- Rashomon (fiber) → explanation impossibility
- Preference diversity (profile) → Arrow
- Base rate inequality (statistical) → Chouldechova
- Network partitions (physical) → CAP

**6.4 What a formal unification would require**
A category-theoretic framework where:
- "Compression" = a functor between categories
- "Properties" = natural transformations
- "Diversity" = non-triviality of certain hom-sets
- "Impossibility" = non-existence of a natural transformation
  satisfying constraints

This would be mathematically precise but accessible only to
category theorists. The tradeoff between generality and
communicability is real.

**6.5 The recipe for new impossibilities**
Given a new domain:
1. Identify the source, compression, and feature spaces
2. Define the natural properties (faithfulness, stability,
   decisiveness — adapted to the domain)
3. Identify the diversity condition
4. Check if the fiber pattern applies
5. If yes: the impossibility follows from the meta-theorem
6. If no: check if a domain-specific argument works

This recipe has practical value: it tells researchers in
NEW domains how to check whether their explanation methods
face a fundamental impossibility.

---

## Domain Papers (Level 0 applied)

Each is a collaboration with a domain expert. The theory
is already done; the contribution is the domain-specific
formalization and empirical validation.

### Paper 7: Climate Model Explanation Impossibility
**Target**: Nature Climate Change or PNAS
**Collaborator needed**: Climate scientist
**Content**: Multiple GCMs fit historical temperature record.
Attribution of extreme weather events to climate change
is underspecified. The impossibility explains model spread.
Resolution: multi-model ensemble attribution (already
standard practice — the theorem justifies it).

### Paper 8: Protein Structure Explanation Impossibility
**Target**: PNAS or Nature Methods
**Collaborator needed**: Structural biologist / AlphaFold user
**Content**: Multiple conformations consistent with same
sequence. Explaining "why this fold?" faces the trilemma.
Resolution: conformational ensemble.

### Paper 9: Neural Circuit Explanation Impossibility
**Target**: PLOS Computational Biology or Neuron
**Collaborator needed**: Computational neuroscientist (Marder lab?)
**Content**: Degeneracy principle as Instance N. Multiple
circuit parameterizations produce identical behavior.
Resolution: report parameter-space-averaged circuit properties.

### Paper 10: Econometric Explanation Impossibility
**Target**: Journal of Econometrics or Econometrica
**Collaborator needed**: Econometrician (identification expert)
**Content**: Partial identification (Manski) as formal instance.
Structural model explanations face the trilemma. Resolution:
identified-set reporting (already Manski's recommendation —
the theorem proves it's optimal).

---

## Key People to Contact

### Would immediately engage
- **Cynthia Rudin** (Duke) — Rashomon sets, interpretable ML.
  Her research program is the empirical foundation for this
  theory. She'd cite it, assign it, invite talks.
- **Aaron Fisher** (Columbia) — Model reliance, variable
  importance clouds. Co-author of Fisher et al. (2019).
- **Lesia Semenova** (Duke/Harvard) — Existence of simpler
  models in Rashomon sets.
- **Charles Marx** (MIT) — Uncertainty in explanations.
  His DASH-adjacent work on explanation uncertainty.

### Would engage with a bridge paper
- **Judea Pearl** (UCLA) — Causal discovery, Markov equivalence.
  The derived causal instance is directly in his territory.
- **Charles Manski** (Northwestern) — Partial identification.
  The theorem justifies his program.
- **Eve Marder** (Brandeis) — Neural degeneracy. The
  neuroscience application is in her area.
- **Samir Okasha** (Bristol) — Theory choice, Arrow connection.
  Already connected Arrow to theory choice.
- **Neel Nanda** — MI researcher. The circuit impossibility
  is directly relevant to his work.

### For domain papers (need collaborators)
- Climate: Gavin Schmidt (NASA GISS), Kate Marvel
- Protein: AlphaFold team, Mohammed AlQuraishi
- Neuroscience: Eve Marder, Shaul Druckmann (Stanford)
- Econometrics: Isaiah Andrews (Harvard), Francesca Molinari

---

## Timeline

```
2026 Q2:  Paper 3 on arXiv + NeurIPS
          Contact Rudin, Fisher, Semenova
2026 Q3:  NeurIPS decision. Begin Paper 4.
          Contact Pearl, Manski if Paper 3 is well-received.
2026 Q4:  Paper 4 draft (quantitative tradeoffs).
          PNAS submission for Paper 3.
2027 Q1:  Paper 4 submission.
2027 Q2:  Begin Paper 5 (query-relative theory).
          Begin domain paper conversations.
2027 Q3:  Paper 5 draft.
2027-28:  Domain papers with collaborators.
2028:     Paper 6 (aggregation framework) if the
          community has engaged with Levels 0-2.
```

---

## Success Metrics

### Year 1 (by Q2 2027)
- Paper 3 accepted at NeurIPS or PNAS
- Paper 4 submitted
- ≥3 citations of Paper 3 by other groups
- ≥1 invited talk

### Year 2 (by Q2 2028)
- Paper 4 accepted
- Paper 5 submitted
- ≥20 citations of Paper 3
- "Faithful, stable, decisive — pick two" appears in
  ≥2 papers by other authors
- ≥1 domain collaboration initiated

### Year 3 (by Q2 2029)
- ≥1 domain paper published
- ≥50 citations of Paper 3
- The trilemma is referenced in a textbook or survey
- A regulatory document cites the impossibility

### The "it worked" signal
Five different communities cite the result in their own
terminology within 5 years. If a neuroscientist cites it
when discussing degeneracy, an econometrician when
discussing identification, and a fairness researcher when
discussing metric tradeoffs — without needing a bridge
paper — the result has achieved foundational status.

---

## The Principle

More generality makes the theorem more powerful but harder
to explain. The sweet spot for each level:

- Level 0: "Pick two." Catchy, usable, proved.
- Level 1: "Here's exactly how much of each you can have."
  Quantitative, engineering-useful.
- Level 2: "Here's exactly which questions you can answer."
  Precise, domain-adaptable.
- Level 3: "Here's why all aggregation impossibilities share
  a common structure." Philosophical, unifying.

Each level is a complete contribution. No level requires
the next. The program can stop at any point and the work
already done stands on its own.
