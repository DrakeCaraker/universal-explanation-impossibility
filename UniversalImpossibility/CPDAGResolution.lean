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

/-- Configurations: two DAGs in a Markov equivalence class. -/
inductive CPDAGConfig where
  | chain : CPDAGConfig  -- A → B → C
  | fork  : CPDAGConfig  -- A ← B → C
  deriving DecidableEq

/-- Explanations: edge orientations. -/
inductive CPDAGExplanation where
  | aToB : CPDAGExplanation  -- A → B
  | bToA : CPDAGExplanation  -- B → A
  deriving DecidableEq

/-- Observables: conditional independence structure (same for both). -/
inductive CPDAGObservable where
  | ciABC : CPDAGObservable  -- A ⊥ C | B
  deriving DecidableEq

private def cpdagObs : CPDAGConfig → CPDAGObservable
  | _ => CPDAGObservable.ciABC

private def cpdagExp : CPDAGConfig → CPDAGExplanation
  | CPDAGConfig.chain => CPDAGExplanation.aToB
  | CPDAGConfig.fork  => CPDAGExplanation.bToA

private def cpdagIncomp : CPDAGExplanation → CPDAGExplanation → Prop
  | CPDAGExplanation.aToB, CPDAGExplanation.bToA => True
  | CPDAGExplanation.bToA, CPDAGExplanation.aToB => True
  | _, _ => False

private instance : DecidableRel cpdagIncomp := fun a b => by
  cases a <;> cases b <;> simp [cpdagIncomp] <;> exact inferInstance

private theorem cpdagIncomp_irrefl (h : CPDAGExplanation) : ¬cpdagIncomp h h := by
  cases h <;> simp [cpdagIncomp]

def cpdagSystem : ExplanationSystem CPDAGConfig CPDAGExplanation CPDAGObservable :=
  ⟨cpdagObs, cpdagExp, cpdagIncomp, cpdagIncomp_irrefl,
   CPDAGConfig.chain, CPDAGConfig.fork, rfl, trivial⟩

/-- The symmetry group: ℤ/2ℤ swapping chain ↔ fork. -/
inductive MECGroup where
  | id : MECGroup
  | swap : MECGroup
  deriving DecidableEq

private def mecMul : MECGroup → MECGroup → MECGroup
  | MECGroup.id, g => g
  | g, MECGroup.id => g
  | MECGroup.swap, MECGroup.swap => MECGroup.id

private def mecInv : MECGroup → MECGroup
  | MECGroup.id => MECGroup.id
  | MECGroup.swap => MECGroup.swap

instance instMECGroupGroup : Group MECGroup where
  mul := mecMul
  one := MECGroup.id
  inv := mecInv
  mul_assoc := by intro a b c; cases a <;> cases b <;> cases c <;> rfl
  one_mul := by intro a; cases a <;> rfl
  mul_one := by intro a; cases a <;> rfl
  inv_mul_cancel := by intro a; cases a <;> rfl

private def mecSmul : MECGroup → CPDAGConfig → CPDAGConfig
  | MECGroup.id, θ => θ
  | MECGroup.swap, CPDAGConfig.chain => CPDAGConfig.fork
  | MECGroup.swap, CPDAGConfig.fork => CPDAGConfig.chain

instance instMECGroupAction : MulAction MECGroup CPDAGConfig where
  smul := mecSmul
  one_smul := by intro θ; cases θ <;> rfl
  mul_smul := by intro a b θ; cases a <;> cases b <;> cases θ <;> rfl

/-- Swapping preserves observables (both map to ciABC). -/
private theorem mecObs_invariant (g : MECGroup) (θ : CPDAGConfig) :
    cpdagSystem.observe (g • θ) = cpdagSystem.observe θ := by
  cases g <;> cases θ <;> rfl

def mecSymmetry : HasSymmetry cpdagSystem MECGroup :=
  ⟨mecObs_invariant,
   fun θ₁ θ₂ _ => by
     cases θ₁ <;> cases θ₂
     · exact ⟨MECGroup.id, rfl⟩
     · exact ⟨MECGroup.swap, rfl⟩
     · exact ⟨MECGroup.swap, rfl⟩
     · exact ⟨MECGroup.id, rfl⟩⟩

/-- CPDAG resolution: report undirected (= aToB as canonical, constant on orbits). -/
def cpdagResolution : CPDAGConfig → CPDAGExplanation
  | _ => CPDAGExplanation.aToB

theorem cpdagResolution_gInvariant : gInvariant cpdagResolution MECGroup := by
  intro g θ; cases g <;> cases θ <;> rfl

/-! ### Main result: G-invariance implies stability -/

/-- The CPDAG resolution is stable: DAGs with the same conditional independence
    structure (same observables) get the same CPDAG output.

    This is a direct application of the abstract `gInvariant_stable` theorem:
    G-invariance + fiber transitivity → stability. -/
theorem cpdag_gInvariant_implies_stable :
    stable cpdagSystem cpdagResolution :=
  gInvariant_stable cpdagSystem cpdagResolution mecSymmetry cpdagResolution_gInvariant

end UniversalImpossibility
