# Universal Explanation Impossibility — Final Implementation Plan

**Goal**: Make the result bulletproof, unassailable, and foundational.
Add the mechanistic interpretability instance, prove necessity,
derive Rashomon from first principles, and polish for NeurIPS 2026.

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Current state**: 70 Lean files, 338 theorems, 68 axioms, 0 sorry.
8 instances, 6 experiments with controls. Paper: 26pp base.
**Timeline**: NeurIPS abstract May 4, paper May 6. 25 days.

---

## VET AUDIT RECORD

### Round 1 — Factual Accuracy

1. **Meloux et al. (ICLR 2025)**: User-provided citation. Claims:
   85 valid circuits with zero circuit error, 535.8 valid
   interpretations per circuit, <2% of networks with unique
   mapping. **CANNOT INDEPENDENTLY VERIFY** — plan treats as
   user-provided evidence. Paper should cite precisely.

2. **"Sparse Autoencoders Trained on the Same Data Learn Different
   Features" (Jan 2025)**: User-provided. Claims ~30% feature
   overlap across seeds. Same caveat.

3. **"Nobody has proved the MI impossibility consequence"**:
   Inferred from research. Plan should say "to our knowledge."

4. **Lean state**: 70/338/68/0 — verified from last build.

5. **NeurIPS timeline**: 25 days. Tight but feasible if scoped.

6. ⚠️ **Count references in user's text mention "319 → 322"**:
   Stale. We're at 338 now. Corrected throughout this plan.

### Round 2 — Reasoning Quality

1. **"MI instance crosses the landmark threshold"**: INFERRED.
   Well-reasoned but depends on reception. The MI community
   could engage constructively OR react defensively. **Framing
   is critical**: "MI faces a structural tradeoff" NOT "MI is
   broken." The resolution (circuit equivalence classes) must
   be front-and-center.

2. **"Rashomon property for MI is empirically established"**:
   INFERRED from Meloux et al. The bridge from "multiple valid
   circuits" to our formal Rashomon definition needs explicit
   construction. Must define: what is "observe" for circuits
   (the I/O function), what is "explain" (the circuit), what
   is "incompatible" (circuits attribute computation to different
   components).

3. **Necessity proof nuance**: ¬Rashomon does NOT trivially
   imply existence of faithful+stable+decisive E. The correct
   statement is: "If the system is fully specified (each
   observation uniquely determines the configuration), then
   E = explain satisfies all three." For partially specified
   systems without Rashomon, the situation is more complex.
   ⚠️ Plan revised to state necessity precisely.

4. **Overclaiming risk on MI**: The impossibility doesn't say
   MI is useless. It says no SINGLE circuit explanation can
   satisfy all three properties simultaneously. The resolution
   (report equivalence classes) is constructive. Plan revised
   to emphasize this framing.

### Round 3 — Omissions

1. **MI experiment**: The plan should specify whether to RUN
   an experiment or CITE Meloux et al. Running a replication
   is stronger but requires implementing circuit discovery.
   **Decision**: Cite Meloux et al. as the primary Rashomon
   evidence. Optionally run a small-scale SAE seed comparison
   if time allows.

2. **Neel Nanda quote**: Cannot verify provenance. Plan should
   cite PUBLISHED work only, not informal quotes.

3. **Fairness instance experiment**: Deprioritized. Cite
   Chouldechova (2017) and Kleinberg et al. (2017).

4. **Docker/interactive explorer**: Post-acceptance. Not in
   this plan.

5. **The "decisive" definition for MI**: What does decisiveness
   mean for circuits? It means: if the true circuit explanation
   (explain(θ)) rules out some interpretation h (is incompatible
   with h), then E(θ) also rules out h. I.e., E commits to the
   same level of specificity as the true decomposition.

6. ⚠️ **Missing from earlier plans**: The paper's abstract and
   intro need to be updated for 9 instances (was 8). All four
   paper versions need updating.

### Corrections Applied

- ⚠️ Necessity proof scoped to "fully specified systems" to
  avoid overclaiming
- ⚠️ MI framing explicitly constructive ("tradeoff, not failure")
- ⚠️ Meloux et al. treated as user-provided evidence with caveat
- ⚠️ Neel Nanda quote removed; cite published work only
- ⚠️ MI experiment: cite-first, replicate-if-time-allows
- ⚠️ Instance count: 9 (not 8 or 6)

### Confidence Ratings

| Component | Confidence | Justification |
|-----------|-----------|---------------|
| MI Lean instance | HIGH | Same mechanical pattern as 8 others |
| MI paper section | HIGH | Meloux et al. provides the Rashomon evidence |
| Necessity proof | MEDIUM | Fully-specified case is clean; general case is subtle |
| Causal Rashomon derivation | MEDIUM | Markov equivalence is pure math but Lean formalization of graph theory is non-trivial |
| Trilemma diagram | HIGH | Pure presentation, no technical risk |
| True retraining for attention | HIGH | Freeze transformer, retrain head from 10 seeds |
| Quantitative bound | MEDIUM | May need new axioms; closed form unclear for abstract framework |
| Resolution uniqueness | MEDIUM | Hunt-Stein in Lean is feasible but requires Mathlib measure theory |
| NeurIPS timeline (all items) | MEDIUM | 25 days, tight but scoped to essentials |

---

## Phase 1: Mechanistic Interpretability Instance [Days 1-3]

The single highest-impact addition. Challenges the most active
research program in AI safety.

### Task 1.1: MI Lean Instance [Sonnet]

**Files**:
- Create: `UniversalImpossibility/MechInterpInstance.lean`
- Modify: `UniversalImpossibility/Basic.lean` (add import)

```
import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Mechanistic Interpretability Instance

Neural network circuit explanations as an instance of ExplanationSystem.
The Rashomon property holds because equivalent networks (same I/O behavior)
admit multiple valid circuit decompositions.

Empirical evidence: Meloux et al. (ICLR 2025) found 85 distinct valid
circuits with zero circuit error for a simple XOR task, each admitting
an average of 535.8 valid interpretations. Less than 2% of trained
networks had a unique minimal mapping.
-/

/-- Circuit configuration: neural network weights. -/
axiom CircuitConfig : Type
/-- Circuit explanation: a computational subgraph decomposition. -/
axiom CircuitExplanation : Type
/-- Observable: the input-output function. -/
axiom CircuitObservable : Type

/-- Circuit explanations form an ExplanationSystem.
    observe = the I/O function, explain = circuit decomposition,
    incompatible = circuits attribute computation to different
    components (e.g., different subnetworks responsible for
    the same feature). -/
axiom circuitSystem :
  ExplanationSystem CircuitConfig CircuitExplanation CircuitObservable

/-- Mechanistic interpretability impossibility:
    No circuit explanation can simultaneously be faithful (reflect
    this network's actual computation), stable (same circuit for
    equivalent networks), and decisive (commit to a specific
    decomposition). -/
theorem mech_interp_impossibility
    (E : CircuitConfig → CircuitExplanation)
    (hf : faithful circuitSystem E)
    (hs : stable circuitSystem E)
    (hd : decisive circuitSystem E) : False :=
  explanation_impossibility circuitSystem E hf hs hd
```

Run `lake build`. Commit.

### Task 1.2: MI Proof Sketch [Opus]

**File**: `docs/proof-sketches/mech-interp-impossibility.md`

Setup:
- Θ = neural network weight configurations
- Y = input-output functions (the behavior the network computes)
- H = circuit decompositions (computational subgraphs with
  attributed roles)
- observe(θ) = the function f_θ : inputs → outputs
- explain(θ) = the circuit decomposition of f_θ (which subnetwork
  does what)
- incompatible(h₁, h₂) = the circuits attribute a given
  computation to different components

Rashomon Property:
Meloux et al. (ICLR 2025) establishes this empirically:
- For XOR: 85 valid circuits with zero circuit error
- Each circuit admits avg 535.8 valid interpretations
- <2% of networks have a unique minimal mapping
- Multiplicity scales with hidden layer size

Additional evidence:
- "Sparse Autoencoders Trained on the Same Data Learn Different
  Features" (2025): SAEs share only ~30% of features across seeds
- Anthropic's circuit tracing (March 2025): acknowledges
  non-uniqueness of circuit explanations

Trilemma:
- Faithful: report the actual circuit for THIS network
- Stable: same circuit for functionally equivalent networks
- Decisive: commit to specific computational attributions

Proof: Same 4-step structure. Decisive at θ₁ gives incompatibility
propagation. Stability gives equality. Faithfulness gives
non-incompatibility. Contradiction.

Resolution: Report circuit EQUIVALENCE CLASSES — the structural
invariants shared by all valid decompositions for equivalent
networks. Analogous to CPDAGs for causal discovery: instead of
a single DAG, report the equivalence class.

### Task 1.3: MI Paper Section [Opus]

**File**: `paper/sections/instance_mech_interp.tex`

~1 page subsection. Structure:
1. Motivation: MI is central to AI safety arguments.
   Circuit discovery is the dominant paradigm. But the
   non-uniqueness of circuits is a structural problem,
   not a practical one.

2. 6-tuple definition with Lean references.

3. Rashomon property: cite Meloux et al. with specific numbers.
   Cite SAE seed paper. Frame as: "the empirical evidence for
   circuit non-uniqueness is overwhelming."

4. Instance theorem: direct application of Theorem 1.

5. Resolution: circuit equivalence classes. "Instead of asking
   'what circuit does this network implement?', ask 'what
   structural invariants are shared by ALL valid circuit
   decompositions?'"

6. Implications: "This result does not undermine MI — it
   clarifies what MI can and cannot achieve. Single-circuit
   explanations face a fundamental tradeoff. The constructive
   path forward is to characterize the space of valid circuits,
   not to search for THE circuit."

**CRITICAL FRAMING**: The paper must be generous to the MI
community. Cite Elhage et al. (2022), Conmy et al. (2023),
Anthropic's work. Frame the impossibility as a CONTRIBUTION
to MI (clarifying the achievable space), not a criticism.

### Task 1.4: Add MI Bibliography Entries [Sonnet]

Add to `paper/references.bib`:
- `meloux2025everything` — Meloux et al., "Everything,
  Everywhere, All at Once: Is Mechanistic Interpretability
  Identifiable?", ICLR 2025
- `sae_seeds_2025` — "Sparse Autoencoders Trained on the
  Same Data Learn Different Features", 2025
- `elhage2022toy` — Elhage et al., "Toy Models of
  Superposition", 2022
- `conmy2023automated` — Conmy et al., "Towards Automated
  Circuit Discovery for Mechanistic Interpretability",
  NeurIPS 2023
- `anthropic2025circuit` — Anthropic, "Circuit Tracing:
  Revealing Computational Mechanisms in Language Models", 2025

### GATE 1: MI instance compiles in Lean, paper section is
written, bibliography entries added. Review before proceeding.

---

## Phase 2: Necessity Proof [Days 2-4]

Prove the Rashomon property is the EXACT boundary between
possibility and impossibility.

### Task 2.1: Necessity Theorem in Lean [Opus]

**File**: Modify `UniversalImpossibility/ExplanationSystem.lean`

Prove TWO theorems:

```
/-- Sufficiency (already proved): Rashomon → impossibility. -/
-- explanation_impossibility (already exists)

/-- Necessity: If the system is fully specified (each observation
    uniquely determines the configuration), then E = explain
    satisfies all three properties. -/
theorem fully_specified_escape
    (S : ExplanationSystem Θ H Y)
    (h_specified : ∀ θ₁ θ₂ : Θ,
      S.observe θ₁ = S.observe θ₂ → θ₁ = θ₂)
    : faithful S S.explain ∧ stable S S.explain
      ∧ decisive S S.explain := by
  refine ⟨?_, ?_, ?_⟩
  · -- faithful: by irreflexivity
    intro θ
    exact S.incompatible_irrefl (S.explain θ)
  · -- stable: fully specified means same observe → same θ → same explain
    intro θ₁ θ₂ hobs
    have : θ₁ = θ₂ := h_specified θ₁ θ₂ hobs
    subst this
    rfl
  · -- decisive: E = explain is trivially decisive
    intro θ h hinc
    exact hinc
```

Also prove the CONTRAPOSITIVE of the impossibility:

```
/-- Contrapositive: If a faithful, stable, decisive E exists,
    then the Rashomon property fails. -/
theorem no_rashomon_if_all_three
    (S : ExplanationSystem Θ H Y)
    (E : Θ → H)
    (hf : faithful S E) (hs : stable S E) (hd : decisive S E) :
    ¬(∃ θ₁ θ₂ : Θ,
      S.observe θ₁ = S.observe θ₂ ∧
      S.incompatible (S.explain θ₁) (S.explain θ₂)) := by
  intro ⟨θ₁, θ₂, hobs, hinc⟩
  exact explanation_impossibility S E hf hs hd
```

Wait — this is trivially the same as the impossibility theorem
(it just says False → anything). The more interesting version
uses the ACTUAL Rashomon field of the structure, but since it's
bundled, the proof is circular.

REVISED APPROACH: State necessity for a PARAMETERIZED system
(without bundled Rashomon):

```
/-- A system WITHOUT bundled Rashomon — just the maps. -/
structure ExplanationSetup (Θ : Type) (H : Type) (Y : Type) where
  observe : Θ → Y
  explain : Θ → H
  incompatible : H → H → Prop
  incompatible_irrefl : ∀ (h : H), ¬incompatible h h

/-- The Rashomon property as a proposition, not bundled. -/
def has_rashomon (S : ExplanationSetup Θ H Y) : Prop :=
  ∃ θ₁ θ₂ : Θ, S.observe θ₁ = S.observe θ₂ ∧
    S.incompatible (S.explain θ₁) (S.explain θ₂)

/-- Sufficiency: Rashomon → impossibility. -/
theorem impossibility_from_rashomon (S : ExplanationSetup Θ H Y)
    (hr : has_rashomon S) (E : Θ → H)
    (hf : ...) (hs : ...) (hd : ...) : False := ...

/-- Necessity: ¬Rashomon → all three achievable (by E = explain
    for decisive+faithful, projecting through observe for stable,
    when the explain map is compatible on fibers). -/
theorem possibility_without_rashomon (S : ExplanationSetup Θ H Y)
    (hr : ¬has_rashomon S)
    : ∃ E : Θ → H, faithful_setup S E ∧ stable_setup S E
        ∧ decisive_setup S E := by
  use S.explain
  refine ⟨?_, ?_, ?_⟩
  · intro θ; exact S.incompatible_irrefl _
  · intro θ₁ θ₂ hobs
    -- ¬Rashomon means: for all θ₁ θ₂ with same observe,
    -- ¬incompatible(explain θ₁, explain θ₂)
    -- But this does NOT give explain θ₁ = explain θ₂!
    sorry -- THIS MAY NOT WORK
  · intro θ h hinc; exact hinc
```

The stable part fails! ¬Rashomon does not imply E = explain
is stable. We need a different construction.

REVISED: The correct necessity statement is WEAKER:
"If ¬Rashomon, there EXISTS some E that is faithful+stable+decisive."
But the E might not be explain — it might be a stable projection.

Actually, with ¬Rashomon: for all θ₁ θ₂ in the same fiber,
¬incompatible(explain θ₁, explain θ₂). We need a stable E that
is faithful and decisive. Define E to be constant on each fiber:
pick a representative θ_y for each y, set E(θ) = explain(θ_y)
where y = observe(θ).

Then:
- Stable: E factors through observe by construction ✓
- Faithful at θ: ¬incompatible(E(θ), explain(θ))
  = ¬incompatible(explain(θ_y), explain(θ))
  This follows from ¬Rashomon (no incompatible pair in the fiber) ✓
- Decisive at θ: incompatible(explain(θ), h) → incompatible(E(θ), h)
  = incompatible(explain(θ), h) → incompatible(explain(θ_y), h)
  This does NOT follow from ¬Rashomon! ¬Rashomon says the two
  explanations are compatible, not that they have the same
  incompatibilities.

So necessity in full generality FAILS. The correct statement:
Rashomon is sufficient but not necessary for impossibility.

HOWEVER: for the FULLY SPECIFIED case (injective observe),
necessity holds trivially because E = explain is stable.

Plan revision: Prove fully_specified_escape (the clean case)
and state the general case as a remark noting the gap.

Run `lake build`. Commit.

### Task 2.2: Add Necessity Discussion to Paper [Opus]

Add a remark after the theorem in the paper:

"The Rashomon property is sufficient for the impossibility
(Theorem 1). For fully specified systems — where each observation
uniquely determines the configuration — all three properties are
simultaneously achievable (Proposition X). The Rashomon property
is thus the precise dividing line for fully specified systems.
For partially specified systems without incompatible explanations,
the question of whether all three can be achieved simultaneously
depends on the structure of the incompatibility relation and is
an open problem."

---

## Phase 3: Derive Rashomon for Causal Discovery [Days 3-6]

Convert the causal instance from axiomatized to derived.

### Task 3.1: Formalize Markov Equivalence in Lean [Opus]

**File**: `UniversalImpossibility/MarkovEquivalence.lean`

This is the most technically challenging Lean task. The goal:
prove that two DAGs can have the same conditional independence
relations but different edge orientations.

Minimum viable formalization:
1. Define DAG as a structure with vertices and edges
2. Define d-separation (or a simplified version)
3. Prove: ∃ G₁ G₂, same_CI_relations G₁ G₂ ∧ different_edges G₁ G₂

The SIMPLEST proof: construct a 3-node example.
- G₁: A → B → C (chain)
- G₂: A ← B → C (fork)
Both have the same conditional independencies: A ⊥ C | B.
But they have different edge orientations (A → B vs A ← B).

This is a CONSTRUCTIVE proof of the Rashomon property for
causal discovery — derived from the definition of d-separation,
not axiomatized.

IMPORTANT: This is graph theory in Lean, which requires careful
type definitions. Use Fin n for vertices, matrices or functions
for edges.

If this is too complex for the timeline, FALLBACK: axiomatize
the 3-node example as a concrete structure and prove the
Rashomon property from its explicit construction (still better
than a fully abstract axiom).

Run `lake build`. Commit.

### Task 3.2: Update Causal Instance to Reference Derivation [Sonnet]

Modify `CausalInstance.lean` or add a comment referencing
`MarkovEquivalence.lean` as the derivation of the Rashomon
property (rather than pure axiomatization).

---

## Phase 4: Presentation [Days 4-6, parallel with Phase 3]

### Task 4.1: Trilemma Diagram [Sonnet]

**File**: `paper/figures/trilemma_universal.tex` (TikZ)

Create a publication-quality triangle diagram:

```
         Faithful
          /    \
         /      \
        / IMPOSSIBLE \
       /  (Rashomon)  \
      /________________\
  Stable            Decisive
```

On each EDGE, label the achievable pair:
- Faithful–Stable edge: "E = neutral (ties/abstains)"
  "DASH, CPDAG, ensemble probes"
- Faithful–Decisive edge: "E = explain (unstable)"
  "Single-model SHAP, GradCAM, circuit"
- Stable–Decisive edge: "E = arbitrary (unfaithful)"
  "Fixed ranking, random circuit"

Around the triangle, place the 9 instances as labeled
points/icons.

Compile to PDF. This becomes Figure 1 in the paper.

### Task 4.2: One-Page Summary [Opus]

Write a self-contained 1-page version of the entire result.
Place at the start of the introduction (or as a standalone
`paper/one_page_summary.tex`).

Structure:
- 3 sentences: setup
- 1 sentence: theorem
- 3 sentences: tightness
- 1 table: 9 instances with experiment results
- 2 sentences: resolution
- 2 sentences: implications

### Task 4.3: Escape Routes Analysis [Opus]

For each dropped property, provide:
1. What you sacrifice
2. What you gain
3. Concrete tools that take this escape
4. When this escape is appropriate

Table format in the paper:

| Drop | Sacrifice | Gain | Tools | When |
|------|-----------|------|-------|------|
| Faithful | May contradict model | Stable + decisive | LIME, fixed rules | Regulatory compliance |
| Stable | Changes on retrain | Faithful + decisive | Single-model SHAP, GradCAM | Debugging one model |
| Decisive | Ties, uncertainty | Faithful + stable | DASH, CPDAG, ensemble | Production deployment |

---

## Phase 5: True Retraining Experiment [Days 5-8]

### Task 5.1: Retrain DistilBERT Classification Head [Sonnet]

**File**: Modify `paper/scripts/attention_instability_experiment.py`
(add a second experiment section) OR create a new script.

Design:
1. Load pretrained DistilBERT (frozen transformer weights)
2. Add a fresh classification head (2 classes: positive/negative)
3. Train on 100 sentiment sentences (simple positive/negative)
   for 3 epochs with 10 different random seeds for the head
4. All 10 models should achieve similar accuracy on held-out set
5. Compute attention rollout and measure flip rate

This is TRUE multi-seed retraining (different optimization
trajectories for the classification head) but fast (only the
head is trained, transformer is frozen).

Compare: perturbation flip rate vs retraining flip rate.
Report both. If similar, validates the perturbation methodology.
If different, report honestly and discuss.

---

## Phase 6: Quantitative Bound [Days 6-10]

### Task 6.1: Abstract Quantitative Impossibility [Opus]

**File**: `UniversalImpossibility/QuantitativeBound.lean`

Prove: for any stable E, the minimum "distance" from E to
explain is bounded below by a function of the Rashomon set.

The cleanest version: count the number of observe-fibers
where the Rashomon property holds (explain produces incompatible
outputs). Call this k. Then any stable E must "abstain" on at
least k fibers (fail to be decisive on those fibers).

```
/-- For any stable E, the number of fibers where E is not
    decisive is at least the number of Rashomon fibers. -/
-- (This requires counting, which may need Fintype assumptions)
```

If the full Lean formalization is too complex, state the
theorem precisely in the paper with a rigorous proof and
mark as "argued" (not Lean-verified).

### Task 6.2: Resolution Uniqueness [Opus]

Prove (or argue rigorously) that the G-invariant projection
is Pareto-optimal among stable methods.

The Hunt-Stein connection: in invariant statistical decision
theory, the best invariant estimator is minimax. For our
framework, the G-invariant projection minimizes the worst-case
unfaithfulness among stable methods.

State in the paper. Formalize in Lean if tractable.

---

## Phase 7: Regulatory Analysis [Days 8-10]

### Task 7.1: Regulatory Subsection [Opus]

Add to the Discussion section (or create a new subsection):

Analyze three regulatory frameworks:
1. **EU AI Act Article 13**: "appropriate transparency"
   including "meaningful information" about the logic involved.
   → Requires faithful + decisive. Our theorem: this is
   unstable under model updates.

2. **NIST AI RMF MAP 1.5**: "Interpretability and
   explainability ... appropriate to the AI risk."
   → Requires all three. Our theorem: impossible.

3. **OCC SR 11-7**: Model risk management requires
   "conceptual soundness" (faithful) + "outcomes analysis"
   (stable across versions) + "effective challenge" (decisive).
   → Our theorem: cannot simultaneously satisfy all three.

For each: what the regulation demands, which legs of the
trilemma it requires, and how the G-invariant resolution
provides the best achievable compliance.

---

## Phase 8: Final Integration [Days 10-14]

### Task 8.1: Update All Paper Versions [Sonnet]

For all 4 tex files:
- Update abstract: "nine explanation types" (now includes MI)
- Add MI instance section
- Add necessity remark
- Add trilemma diagram as Figure 1
- Add escape routes table
- Add regulatory analysis subsection
- Update Lean table (new file/theorem counts)
- Update cross-instance summary table (add MI row)
- Compile all versions

### Task 8.2: Add All Missing Bibliography [Sonnet]

Verify EVERY citation in the paper resolves. Add:
- Meloux et al. 2025
- SAE seeds paper 2025
- Elhage et al. 2022
- Conmy et al. 2023
- Anthropic circuit tracing 2025
- Hunt & Stein 1948
- Any others referenced in new sections

### Task 8.3: Lean Consistency Check [Sonnet]

```bash
lake build  # zero errors
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

All counts match paper text. Sorry = 0.

### Task 8.4: Update CLAUDE.md and README [Sonnet]

Update all counts and instance inventories.

### Task 8.5: Final Vet [Opus]

Run full /vet protocol on the completed paper:
- Round 1: every number cross-checked
- Round 2: every claim tagged observed/tested/inferred
- Round 3: what did we miss?

Any LOW confidence finding blocks submission.

---

## Model Assignments

| Task | Model | Rationale |
|------|-------|-----------|
| 1.1 MI Lean instance | Sonnet | Mechanical pattern |
| 1.2 MI proof sketch | Opus | Theoretical reasoning |
| 1.3 MI paper section | Opus | Mathematical exposition, sensitive framing |
| 1.4 MI bibliography | Sonnet | Mechanical |
| 2.1 Necessity Lean | Opus | Novel proof |
| 2.2 Necessity paper text | Opus | Precise statement required |
| 3.1 Markov equivalence Lean | Opus | Graph theory formalization |
| 3.2 Update causal instance | Sonnet | Mechanical |
| 4.1 Trilemma diagram | Sonnet | TikZ |
| 4.2 One-page summary | Opus | Distillation |
| 4.3 Escape routes | Opus | Analysis |
| 5.1 True retraining | Sonnet | Python experiment |
| 6.1 Quantitative bound | Opus | Hard theory |
| 6.2 Resolution uniqueness | Opus | Hard theory |
| 7.1 Regulatory analysis | Opus | Policy writing |
| 8.x Integration | Sonnet | Mechanical |

---

## Execution Order

```
Day 1-3:  [1.1 ∥ 1.2 ∥ 1.4] → [1.3]     (MI instance)
Day 2-4:  [2.1] → [2.2]                    (necessity)
Day 3-6:  [3.1] → [3.2]                    (causal derivation)
Day 4-6:  [4.1 ∥ 4.2 ∥ 4.3]               (presentation, parallel)
Day 5-8:  [5.1]                             (retraining experiment)
Day 6-10: [6.1] → [6.2]                    (quantitative, if time)
Day 8-10: [7.1]                             (regulatory)
Day 10-14:[8.1] → [8.2] → [8.3] → [8.4] → [8.5]  (integration)
```

Phases 1-4 are the MUST-HAVES for NeurIPS.
Phases 5-7 are STRONGLY RECOMMENDED.
Phase 8 is non-negotiable (integration + vet).

If time is short, cut Phase 6 (quantitative bound) first —
it's the hardest and can be added in the JMLR version.

---

## Validation Checklist

### Submission-blocking (MUST PASS):
- [ ] MI instance compiles in Lean (0 sorry)
- [ ] MI paper section written with careful framing
- [ ] Necessity theorem (fully-specified case) in Lean
- [ ] Trilemma diagram created
- [ ] All paper versions compile
- [ ] All Lean counts match paper
- [ ] All citations resolve
- [ ] Mock review for MI has convincing response

### Strongly recommended:
- [ ] Markov equivalence derived (not axiomatized) in Lean
- [ ] True retraining experiment for attention
- [ ] Escape routes table in paper
- [ ] Regulatory analysis subsection
- [ ] One-page summary in introduction
- [ ] Quantitative bound (even if paper-proof only)

---

## What This Achieves

If Phases 1-4 + 8 are completed:

1. **9 instances** including mechanistic interpretability —
   the first formal impossibility for MI circuits

2. **Necessity proof** — Rashomon is the exact boundary
   (for fully specified systems)

3. **Derived Rashomon** for at least 1 instance (causal) —
   addresses "you axiomatize the interesting part"

4. **Visual identity** — the trilemma triangle becomes
   the CAP theorem of XAI

5. **Escape routes** — every practitioner can find their
   tool in the taxonomy

Together with the existing axiom substitution test (optimally
calibrated definitions), tightness proofs (each pair achievable),
and 6 experiments with controls: this is a complete, rigorous,
practically relevant result that changes how the field thinks
about explainability.
