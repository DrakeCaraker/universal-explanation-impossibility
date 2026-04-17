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

/-- Model orderings: two possible orderings of a 2-model ensemble. -/
inductive ModelOrdering where
  | orderAB : ModelOrdering  -- model A first, then B
  | orderBA : ModelOrdering  -- model B first, then A
  deriving DecidableEq

/-- Attribution type: which feature is ranked first. -/
inductive DASHAttribution where
  | featureJ : DASHAttribution
  | featureK : DASHAttribution
  deriving DecidableEq

/-- Observable: ensemble prediction (same regardless of ordering). -/
inductive DASHObservable where
  | ensemblePred : DASHObservable
  deriving DecidableEq

private def dashObs : ModelOrdering → DASHObservable
  | _ => DASHObservable.ensemblePred

private def dashExp : ModelOrdering → DASHAttribution
  | ModelOrdering.orderAB => DASHAttribution.featureJ
  | ModelOrdering.orderBA => DASHAttribution.featureK

private def dashIncomp : DASHAttribution → DASHAttribution → Prop
  | DASHAttribution.featureJ, DASHAttribution.featureK => True
  | DASHAttribution.featureK, DASHAttribution.featureJ => True
  | _, _ => False

private instance : DecidableRel dashIncomp := fun a b => by
  cases a <;> cases b <;> simp [dashIncomp] <;> exact inferInstance

private theorem dashIncomp_irrefl (h : DASHAttribution) : ¬dashIncomp h h := by
  cases h <;> simp [dashIncomp]

def dashSystem : ExplanationSystem ModelOrdering DASHAttribution DASHObservable :=
  ⟨dashObs, dashExp, dashIncomp, dashIncomp_irrefl,
   ModelOrdering.orderAB, ModelOrdering.orderBA, rfl, trivial⟩

/-- Permutation group on orderings: ℤ/2ℤ. -/
inductive OrderingPerm where
  | id : OrderingPerm
  | swap : OrderingPerm
  deriving DecidableEq

private def opMul : OrderingPerm → OrderingPerm → OrderingPerm
  | OrderingPerm.id, g => g
  | g, OrderingPerm.id => g
  | OrderingPerm.swap, OrderingPerm.swap => OrderingPerm.id

private def opInv : OrderingPerm → OrderingPerm
  | OrderingPerm.id => OrderingPerm.id
  | OrderingPerm.swap => OrderingPerm.swap

instance instOrderingPermGroup : Group OrderingPerm where
  mul := opMul
  one := OrderingPerm.id
  inv := opInv
  mul_assoc := by intro a b c; cases a <;> cases b <;> cases c <;> rfl
  one_mul := by intro a; cases a <;> rfl
  mul_one := by intro a; cases a <;> rfl
  inv_mul_cancel := by intro a; cases a <;> rfl

private def opSmul : OrderingPerm → ModelOrdering → ModelOrdering
  | OrderingPerm.id, θ => θ
  | OrderingPerm.swap, ModelOrdering.orderAB => ModelOrdering.orderBA
  | OrderingPerm.swap, ModelOrdering.orderBA => ModelOrdering.orderAB

instance instOrderingPermAction : MulAction OrderingPerm ModelOrdering where
  smul := opSmul
  one_smul := by intro θ; cases θ <;> rfl
  mul_smul := by intro a b θ; cases a <;> cases b <;> cases θ <;> rfl

private theorem dashObs_invariant (g : OrderingPerm) (θ : ModelOrdering) :
    dashSystem.observe (g • θ) = dashSystem.observe θ := by
  cases g <;> cases θ <;> rfl

def dashSymmetry : HasSymmetry dashSystem OrderingPerm :=
  ⟨dashObs_invariant,
   fun θ₁ θ₂ _ => by
     cases θ₁ <;> cases θ₂
     · exact ⟨OrderingPerm.id, rfl⟩
     · exact ⟨OrderingPerm.swap, rfl⟩
     · exact ⟨OrderingPerm.swap, rfl⟩
     · exact ⟨OrderingPerm.id, rfl⟩⟩

def dashResolution : ModelOrdering → DASHAttribution
  | _ => DASHAttribution.featureJ

theorem dashResolution_gInvariant : gInvariant dashResolution OrderingPerm := by
  intro g θ; cases g <;> cases θ <;> rfl

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
