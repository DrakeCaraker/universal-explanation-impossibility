import UniversalImpossibility.UniversalResolution

/-!
# DASH Consensus as a G-Invariant Resolution for Attributions

DASH (ensemble averaging of sequential SHAP attributions) resolves the universal
explanation impossibility by averaging over all model orderings. This file
formalizes DASH as a G-invariant resolution in the abstract framework of
`UniversalResolution.lean`.

## Conceptual setup

- **Configuration space Θ = ModelOrdering**: a specific ordering (permutation) of
  models used to compute sequential SHAP attributions. Different orderings produce
  different attributions — this is the Rashomon property. DASH averages over all
  orderings.
- **Group G = OrderingPerm**: the permutation group acting on `ModelOrdering`.
  An element `g : OrderingPerm` maps one ordering to another by rearranging the
  model sequence. DASH averages uniformly across this orbit.
- **Resolution R = dashResolution**: the DASH consensus attribution, defined as an
  average over the G-orbit of each configuration. Because it averages uniformly,
  it is constant on G-orbits (G-invariant).
- **Symmetry**: the observable (the ensemble's aggregate predictive behavior) is
  invariant under reordering; G acts transitively on each observe-fiber (any two
  orderings that produce the same observable are related by a permutation).

## Main result

`dash_gInvariant_implies_stable`: DASH attribution is stable — two model orderings
that produce the same observable output receive identical attributions. This follows
immediately from `gInvariant_stable` once G-invariance and the symmetry structure
are in place.
-/

set_option autoImplicit false

namespace DASHResolution

/-! ### Types for the DASH instance -/

/-- A model ordering: a specific permutation of the ensemble models used to compute
    sequential SHAP attributions. Different orderings yield different attributions,
    instantiating the Rashomon property in the SHAP setting. -/
axiom ModelOrdering : Type

/-- The attribution type: feature attribution vectors produced by SHAP. -/
axiom DASHAttribution : Type

/-- The observable type: the ensemble's aggregate predictions, which are invariant
    under reordering of the models used to compute individual attributions. -/
axiom DASHObservable : Type

/-! ### The DASH explanation system -/

/-- The DASH explanation system:
    - `observe` maps a model ordering to the ensemble's predictive behavior
      (independent of the ordering).
    - `explain` maps a model ordering to the sequential SHAP attribution for
      that specific ordering (dependent on the ordering, exhibiting the Rashomon
      property).
    - `incompatible` holds when two attributions disagree on a feature ranking.
    - `rashomon` witnesses two orderings with identical observables but
      incompatible attributions. -/
axiom dashSystem : ExplanationSystem ModelOrdering DASHAttribution DASHObservable

/-! ### The permutation group for DASH -/

/-- The permutation group acting on `ModelOrdering`.
    Elements represent rearrangements of the model sequence. DASH averages
    attributions over the entire orbit of this action, achieving G-invariance. -/
axiom OrderingPerm : Type

/-- `OrderingPerm` is a group. -/
axiom instOrderingPermGroup : Group OrderingPerm

attribute [instance] instOrderingPermGroup

/-- `OrderingPerm` acts on `ModelOrdering` by permuting the model sequence. -/
axiom instOrderingPermAction : MulAction OrderingPerm ModelOrdering

attribute [instance] instOrderingPermAction

/-! ### Symmetry structure for DASH -/

/-- The symmetry structure for the DASH system: the permutation group preserves
    observable outputs (reordering models does not change ensemble predictions)
    and acts transitively on each observe-fiber (any two orderings that produce
    the same observable are related by some permutation).

    These two properties reflect the core design of DASH: it operates on a space
    where the group connects all orderings of any fixed ensemble. -/
axiom dashSymmetry : HasSymmetry dashSystem OrderingPerm

/-! ### DASH as a G-invariant resolution -/

/-- The DASH resolution: the consensus attribution obtained by averaging sequential
    SHAP attributions across all permutations of the model ordering. -/
axiom dashResolution : ModelOrdering → DASHAttribution

/-- DASH is G-invariant: permuting the model ordering does not change the consensus
    attribution. Averaging uniformly over all orderings in the orbit means the
    result is identical for every representative of that orbit.

    Formally: `∀ (g : OrderingPerm) (θ : ModelOrdering), dashResolution (g • θ) = dashResolution θ`. -/
axiom dashResolution_gInvariant : gInvariant dashResolution OrderingPerm

/-! ### Main result: G-invariance implies stability -/

/-- **DASH consensus is stable.**

    Model orderings that produce the same observable output (same ensemble
    predictions) receive identical DASH attributions. DASH resolves the
    explanation impossibility by sacrificing faithfulness to individual orderings
    in exchange for stability across all orderings.

    **Proof**: direct application of `gInvariant_stable`:
    1. `dashSymmetry` provides that G preserves observables and acts transitively
       on observe-fibers.
    2. `dashResolution_gInvariant` provides G-invariance of DASH.
    3. `gInvariant_stable` combines these to yield stability. -/
theorem dash_gInvariant_implies_stable :
    stable dashSystem dashResolution :=
  gInvariant_stable dashSystem dashResolution dashSymmetry dashResolution_gInvariant

end DASHResolution
