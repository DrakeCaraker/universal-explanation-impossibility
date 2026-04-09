/-
  GBDTSetup: bundled infrastructure showing all axioms are structure fields.

  This file demonstrates that the global axioms (Model, attribution, etc.)
  are equivalent to a parametric structure. New developments can use
  GBDTSetup directly instead of relying on global axioms.

  The existing theorems use global axioms for backward compatibility.
  Key theorems are restated here with the structure parameter.
-/
import UniversalImpossibility.Trilemma
import UniversalImpossibility.General
import UniversalImpossibility.Corollary

set_option autoImplicit false

namespace UniversalImpossibility

/-- Bundled GBDT configuration: all infrastructure and behavioral axioms
    as structure fields. This is equivalent to the global axiom system
    but parametric (no global state). -/
structure GBDTSetup (fs : FeatureSpace) where
  /-- The model type -/
  ModelType : Type
  /-- Number of boosting rounds -/
  numTrees : ℕ
  numTrees_pos : 0 < numTrees
  /-- Attribution function -/
  attribution : Fin fs.P → ModelType → ℝ
  /-- Split count function -/
  splitCount : Fin fs.P → ModelType → ℝ
  /-- First-mover function -/
  firstMover : ModelType → Fin fs.P
  /-- Axiom 1: surjectivity -/
  firstMover_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
    j ∈ fs.group ℓ → ∃ f : ModelType, firstMover f = j
  /-- Axiom 2: first-mover split count -/
  splitCount_firstMover : ∀ (f : ModelType) (j : Fin fs.P),
    firstMover f = j → splitCount j f = numTrees / (2 - fs.ρ ^ 2)
  /-- Axiom 3: non-first-mover split count -/
  splitCount_nonFirstMover : ∀ (f : ModelType) (j : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → firstMover f ≠ j → firstMover f ∈ fs.group ℓ →
    splitCount j f = (1 - fs.ρ ^ 2) * numTrees / (2 - fs.ρ ^ 2)
  /-- Axiom 4: global proportionality -/
  proportionality_global : ∃ c : ℝ, 0 < c ∧ ∀ (f : ModelType) (j : Fin fs.P),
    attribution j f = c * splitCount j f
  /-- Axiom 5: cross-group symmetry -/
  splitCount_crossGroup_symmetric : ∀ (f : ModelType) (j k : Fin fs.P) (ℓ : Fin fs.L),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → firstMover f ∉ fs.group ℓ →
    splitCount j f = splitCount k f
  /-- Axiom 6: cross-group stability -/
  splitCount_crossGroup_stable : ∀ (f f' : ModelType) (j : Fin fs.P) (ℓ : Fin fs.L),
    j ∉ fs.group ℓ → firstMover f ∈ fs.group ℓ → firstMover f' ∈ fs.group ℓ →
    splitCount j f = splitCount j f'

/-- The global axioms form a GBDTSetup instance. -/
noncomputable def axiomSetup (fs : FeatureSpace) : GBDTSetup fs where
  ModelType := Model
  numTrees := numTrees
  numTrees_pos := numTrees_pos
  attribution := attribution fs
  splitCount := splitCount fs
  firstMover := firstMover fs
  firstMover_surjective := firstMover_surjective fs
  splitCount_firstMover := splitCount_firstMover fs
  splitCount_nonFirstMover := splitCount_nonFirstMover fs
  proportionality_global := proportionality_global fs
  splitCount_crossGroup_symmetric := splitCount_crossGroup_symmetric fs
  splitCount_crossGroup_stable := splitCount_crossGroup_stable fs

/-- The Rashomon property for a bundled setup. -/
def RashimonProperty_bundled (fs : FeatureSpace) (G : GBDTSetup fs) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
    ∃ f f' : G.ModelType,
      G.attribution j f > G.attribution k f ∧
      G.attribution k f' > G.attribution j f'

/-- Core impossibility restated for bundled setup: no ranking can be
    simultaneously faithful and stable under the Rashomon property.
    This uses ZERO behavioral axioms from the structure — only the
    abstract Rashomon property as hypothesis. -/
theorem attribution_impossibility_bundled (fs : FeatureSpace) (G : GBDTSetup fs)
    (hrash : RashimonProperty_bundled fs G)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : G.ModelType,
      ranking j k ↔ G.attribution j f > G.attribution k f) :
    False := by
  obtain ⟨f, f', h1, h2⟩ := hrash ℓ j k hj hk hjk
  have hrank : ranking j k := (h_faithful f).mpr h1
  have hcontra : G.attribution j f' > G.attribution k f' := (h_faithful f').mp hrank
  linarith

end UniversalImpossibility
