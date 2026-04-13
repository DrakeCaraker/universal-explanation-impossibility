# Research Roadmap: Group-Theoretic Classification of Explainability

**Status**: Theoretical framework identified. First theorem (universal
impossibility) published. Classification program outlined. No formal
proofs of the group-parameterized results yet.

**Core thesis**: The Explanation Impossibility is not a standalone theorem
but the first theorem in a classification theory. The symmetry group G
of the Rashomon set parameterizes *everything*: the severity of the
impossibility, the space of available resolutions, the uniqueness of
the resolution, and the residual ambiguity. The classification connects
explainability to invariant theory, representation theory, and harmonic
analysis.

---

## The Classification Theorem (Conjectured)

**Statement** (informal): For an explanation system (Θ, H, Y, obs, exp, ⊥)
with symmetry group G acting on the Rashomon fibers:

1. The space of stable explanation maps is isomorphic to the space of
   G-invariant functions on Θ.
2. The G-invariant projection (Reynolds operator / orbit average) is
   Pareto-optimal among all stable explanations.
3. The *character* of the impossibility is determined by G:

| G type | Faithfulness bound | Resolution uniqueness | Resolution regularity |
|--------|-------------------|----------------------|----------------------|
| Finite abelian | 1/|G| | Unique (canonical) | Exact |
| Finite non-abelian | 1/|G| | Non-unique (rep choice) | Exact |
| Compact Lie | exp(-dim G) | Unique (Haar) | Exact (Peter-Weyl) |
| Non-compact Lie | — (diverges) | Non-unique (regularizer) | Requires regularization |
| Groupoid | Fiber-dependent | Fiber-dependent | Varies |

**What's proved**: Part 1 is proved (gInvariant_stable in Lean).
Part 2 is argued but not formalized. Part 3 is conjectured with
evidence from 8 instances.

---

## Phase 1: Formalize the Abelian Case (3-6 months)

### Goal
Prove in Lean 4 that for finite abelian G, the orbit average is the
unique G-invariant resolution that maximizes pointwise faithfulness.

### Approach
- Define GroupAction on ExplanationSystem in Lean
- Define the Reynolds operator R(E)(θ) = (1/|G|) Σ_{g∈G} E(g·θ)
- Prove: R(E) is stable (already done for the abstract case)
- Prove: R(E) is maximally faithful among stable maps (Pareto optimality)
- Prove: for abelian G, R is the unique such map

### Lean infrastructure needed
- Mathlib's Group.Action and GroupAction instances
- Finset.sum / Fintype for the averaging
- Decidable equality for concrete group elements

### Deliverable
New Lean files: `GroupClassification.lean`, `ReynoldsOperator.lean`,
`AbelianUniqueness.lean`. ~500 lines, ~20 theorems.

---

## Phase 2: The Non-Abelian Resolution Multiplicity (6-12 months)

### Goal
Prove that for non-abelian G, multiple inequivalent "reasonable"
resolutions exist, and characterize them via the irreducible
representations of G.

### Key insight
For non-abelian G, the group algebra C[G] decomposes into matrix blocks
(one per irrep). The Reynolds operator projects onto the trivial irrep.
But one could also project onto the trivial + some non-trivial irreps,
giving a "partially decisive" resolution that preserves more information
at the cost of some stability.

### Concrete prediction to verify
In causal inference with ≥3 reversible edges, different scoring functions
(BIC, AIC, MDL) should produce different CPDAGs on the same data.
This disagreement corresponds to different choices among equivalent
irreducible representations of the Markov equivalence class symmetry.

### New instances to formalize
- **Molecular symmetry (T_d)**: Tetrahedral point group, order 24.
  Same IR spectrum, different spatial configurations.
  5 irreps (A₁, A₂, E, T₁, T₂). The A₁ irrep gives the totally
  symmetric resolution; others give partially symmetric resolutions.
- **Rubik's cube group**: Order ~4.3×10¹⁹. The "observable" is the
  face coloring; the "configuration" is the internal piece arrangement.
  Resolution: report the orbit (position up to symmetry).

### Deliverable
Paper: "Resolution Multiplicity in Non-Abelian Explanation Systems"
Target: Journal of Machine Learning Research or Annals of Statistics.

---

## Phase 3: The Non-Compact Divergence (12-18 months)

### Goal
Characterize the regularization landscape for non-compact G.
Show that different regularizers correspond to different choices of
"approximate Haar measure" and that the choice of regularizer is
itself a form of model selection.

### Key insight
For G = ℝ^d (null space translations), the orbit average
∫_{ℝ^d} f(x + v) dv diverges. Tikhonov regularization replaces
Lebesgue measure with a Gaussian: ∫ f(x+v) exp(-λ||v||²) dv.
This is a "regularized Reynolds operator." Different regularizers
(L1 → Laplace measure, L2 → Gaussian, elastic net → mixture)
give different resolutions.

### The meta-impossibility
**Conjecture**: For non-compact G, no resolution can be simultaneously
stable, faithful-in-expectation, and *regularizer-independent*.
This is a second-order impossibility: even the resolution is
underspecified when the symmetry group is non-compact.

### Connection to deep learning
The weight space of a neural network has non-compact symmetry:
permutation of hidden units (finite, non-abelian) × rescaling
(non-compact, continuous). This mixed structure predicts:
- Permutation symmetry: orbit average over permutations is
  well-defined (finite group) but non-unique (non-abelian)
- Rescaling symmetry: orbit average diverges, needs regularization
  (weight decay = Tikhonov on the rescaling group)

This explains why weight decay is necessary for stable explanations
of neural networks — it's regularizing the non-compact part of the
symmetry group.

### Deliverable
Paper: "Regularization as Approximate Orbit Averaging: The Non-Compact
Case of the Explanation Impossibility"
Target: Annals of Statistics or JMLR.

---

## Phase 4: The Groupoid Generalization (18-24 months)

### Goal
Extend the framework to handle fiber-varying symmetry, where different
regions of configuration space have different Rashomon set structures.

### Motivation
In ML, the Rashomon set varies across the feature space:
- In some regions, all near-optimal models agree → no Rashomon →
  faithful+stable+decisive is possible
- In other regions, models disagree → Rashomon → impossibility
- The *transition* between these regions is the decision boundary
  of explainability

This is naturally modeled by a groupoid: a category where the
morphisms (symmetries) vary across objects (configurations).

### Mathematical framework
- A **Rashomon fibration** π: E → B where E is the configuration space,
  B is the observable space, and each fiber π⁻¹(y) has its own
  symmetry group G_y.
- The resolution is a **section** of the associated bundle: a choice
  of G_y-invariant explanation for each fiber.
- Global sections may not exist (obstruction theory) → topological
  impossibility beyond the algebraic one.

### Connections
- **Stacks in algebraic geometry**: The moduli stack of explanations
- **Fiber bundles in physics**: Gauge theory IS a fiber bundle theory;
  our framework is the discrete/finite analogue
- **Sheaf theory**: Local-to-global obstruction for explanations

### The topological impossibility (speculative)
**Conjecture**: When the Rashomon fibration has non-trivial monodromy
(going around a loop in observable space, the symmetry group changes),
no globally consistent resolution exists — even locally optimal
resolutions cannot be patched together.

This would be an impossibility result *beyond* the algebraic one:
not just "you can't have all three properties" but "you can't even
have a resolution that varies smoothly across the feature space."

### Deliverable
Paper: "The Topology of Explainability: Obstructions Beyond the
Rashomon Property"
Target: Annals of Mathematics or Journal of the AMS (if the topology
result is strong enough). More realistically: Foundations of
Computational Mathematics.

---

## Phase 5: New Instances (ongoing, parallel with Phases 1-4)

### Lorentz Group Instance (Physics)

**System**: Relativistic field theory.
**Configurations**: Field configurations in different reference frames.
**Observable**: Frame-invariant quantities (proper time, rest mass,
scalar field values).
**Symmetry**: Lorentz group SO(3,1) — non-compact, non-abelian.
**Resolution**: Lorentz-invariant observables (4-scalars).
**Prediction**: The impossibility should be severe (non-compact +
non-abelian = worst case). Resolution requires both regularization
AND a choice among inequivalent representations.

**Why it matters**: Directly connects to the Standard Model. A
theoretical physicist would immediately understand the framework
through this instance.

### Molecular Symmetry Instance (Chemistry)

**System**: Molecular spectroscopy.
**Configurations**: Spatial orientations of a molecule.
**Observable**: IR/Raman spectrum (invariant under point group).
**Symmetry**: Point group (e.g., T_d for methane, order 24).
**Resolution**: Report symmetry species (irreducible representations).
**Prediction**: Multiple inequivalent resolutions exist (one per
choice of retained irreps).

**Why it matters**: Chemistry Nobel connection (group theory in
spectroscopy). Non-abelian finite group.

### Quantum Mechanics Instance (Physics)

**System**: Quantum measurement.
**Configurations**: Quantum states |ψ⟩.
**Observable**: Measurement outcome probabilities |⟨ψ|φ⟩|².
**Symmetry**: U(1) global phase (|ψ⟩ and e^{iθ}|ψ⟩ give same
probabilities).
**Resolution**: Density matrix ρ = |ψ⟩⟨ψ| (phase-invariant).
**Prediction**: Compact abelian symmetry → unique canonical resolution.
The density matrix IS the orbit average over U(1) phase rotations.

**Why it matters**: Foundational quantum mechanics. The density matrix
as a G-invariant resolution is a known fact but not usually stated
in these terms.

### String Theory Compactification (Speculative)

**System**: String compactifications.
**Configurations**: Different Calabi-Yau manifolds.
**Observable**: Low-energy effective field theory.
**Symmetry**: Modular group SL(2,ℤ) and dualities.
**Resolution**: Duality-invariant quantities.

**Why it matters**: The string landscape IS a Rashomon problem at
cosmic scale. This would be a spectacular but speculative instance.

---

## Timeline

| Phase | Duration | Prerequisites | Deliverable |
|-------|----------|---------------|-------------|
| 0 (current paper) | Done | — | Impossibility + 8 instances + resolution |
| 1 (abelian formalization) | 3-6 months | Lean + Mathlib group theory | Lean proofs + short paper |
| 2 (non-abelian multiplicity) | 6-12 months | Phase 1 | Full paper (JMLR/AoS) |
| 3 (non-compact divergence) | 12-18 months | Phase 2 | Full paper (AoS/JMLR) |
| 4 (groupoid/topology) | 18-24 months | Phase 3 | Full paper (FoCM/JAMS) |
| 5 (new instances) | Ongoing | Any | Per-instance short papers |

### Parallel tracks
- **Lean formalization** (Phases 1-3): Continues building the library
- **New instances** (Phase 5): Each is 1-2 months of work, independent
- **Applications papers**: Regulatory (with legal co-author),
  scientific methodology (with domain co-authors)

---

## Collaborator Needs

| Phase | Expertise Needed |
|-------|-----------------|
| 1 | Lean 4 + group theory (could be solo) |
| 2 | Representation theory + statistics (co-author needed) |
| 3 | Functional analysis + regularization (co-author needed) |
| 4 | Algebraic topology / algebraic geometry (co-author essential) |
| 5 (Lorentz) | Theoretical physics (co-author essential) |
| 5 (molecular) | Physical chemistry / spectroscopy (co-author) |
| 5 (quantum) | Quantum information theory (co-author) |

---

## The Vision

The Explanation Impossibility is a *boundary marker*. It says: here is
where explanation breaks down. The classification theory says: here is
the *complete map* of what's possible, parameterized by symmetry.

If the program succeeds, the endgame is a monograph:

**"The Representation Theory of Explainability"**

connecting:
- Invariant theory (what information survives symmetry)
- Harmonic analysis (decomposing explanations into irreducible components)
- Obstruction theory (when global resolutions don't exist)
- Regularization theory (how to handle non-compact symmetry)

...to produce a complete classification of what can and cannot be explained,
in any scientific domain, for any symmetry structure.

That's the field this work creates.
