import UniversalImpossibility.UniversalResolution

/-!
# CPDAG as G-Invariant Projection for Causal Discovery

The CPDAG (Completed Partially Directed Acyclic Graph) is the canonical
G-invariant resolution of the causal discovery impossibility.

**Key insight:** The Markov equivalence class defines a symmetry group G
that acts on edge orientations by permuting among equivalent DAGs. The
CPDAG retains exactly those edges oriented identically across all DAGs
in the equivalence class — it is the G-invariant projection.

Concretely:
- Configurations Θ = DAGs in the Markov equivalence class
- Observables Y = conditional independence relations (identical for all DAGs in the class)
- Explanations H = edge orientation assignments
- Symmetry group G = permutations of equivalent DAGs (Markov equivalence class symmetry)
- The CPDAG maps every DAG to its shared-orientation skeleton, which is
  constant on G-orbits, hence G-invariant
- By `gInvariant_stable`, this G-invariance implies stability automatically

This connects the concrete `cpdag_is_stable` result from `CausalDiscovery.lean`
to the abstract `gInvariant_stable` framework from `UniversalResolution.lean`.
-/

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### CPDAG configuration types -/

/-- Configurations: DAGs within a Markov equivalence class. -/
axiom CPDAGConfig : Type

/-- Explanations: edge orientation assignments (which edges are directed, and how). -/
axiom CPDAGExplanation : Type

/-- Observables: conditional independence structure (shared by all DAGs in the class). -/
axiom CPDAGObservable : Type

/-- The causal discovery explanation system:
    - `observe` maps a DAG to its conditional independence structure
    - `explain` maps a DAG to its full edge orientation
    - `incompatible` holds when two orientations disagree on some edge
    - `rashomon` witnesses two DAGs with same CI structure but different orientations -/
axiom cpdagSystem : ExplanationSystem CPDAGConfig CPDAGExplanation CPDAGObservable

/-! ### Markov equivalence symmetry group -/

/-- The symmetry group of the Markov equivalence class: permutations
    among DAGs that preserve the conditional independence structure. -/
axiom MECGroup : Type

/-- MECGroup forms a group. -/
axiom instMECGroupGroup : Group MECGroup

attribute [instance] instMECGroupGroup

/-- MECGroup acts on DAG configurations by permuting within the equivalence class. -/
axiom instMECGroupAction : MulAction MECGroup CPDAGConfig

attribute [instance] instMECGroupAction

/-- The Markov equivalence class symmetry: G preserves observables (all DAGs
    share the same CI structure) and acts transitively on observe-fibers
    (any two DAGs with the same CI relations are in the same equivalence class). -/
axiom mecSymmetry : HasSymmetry cpdagSystem MECGroup

/-! ### CPDAG as G-invariant resolution -/

/-- The CPDAG resolution: maps each DAG to its shared-orientation skeleton.
    Edges that are oriented identically in all equivalent DAGs remain directed;
    edges that differ across equivalent DAGs become undirected. -/
axiom cpdagResolution : CPDAGConfig → CPDAGExplanation

/-- The CPDAG resolution is G-invariant: permuting among equivalent DAGs
    does not change the CPDAG output, because the CPDAG retains only
    orientations that are shared across the entire equivalence class. -/
axiom cpdagResolution_gInvariant : gInvariant cpdagResolution MECGroup

/-! ### Main result: G-invariance implies stability -/

/-- The CPDAG resolution is stable: DAGs with the same conditional independence
    structure (same observables) get the same CPDAG output.

    This is a direct application of the abstract `gInvariant_stable` theorem:
    G-invariance + fiber transitivity → stability. -/
theorem cpdag_gInvariant_implies_stable :
    stable cpdagSystem cpdagResolution :=
  gInvariant_stable cpdagSystem cpdagResolution mecSymmetry cpdagResolution_gInvariant

end UniversalImpossibility
