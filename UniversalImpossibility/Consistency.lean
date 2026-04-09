/-
  Axiom Consistency: Concrete Model Construction.

  Constructs an explicit model (P=4, L=2, m=2, ρ=1/2, T=100) satisfying
  all domain-specific property axioms, proving the axiom system is consistent.

  This does NOT instantiate the abstract `Model` type (which is axiomatized).
  Instead, it constructs a parallel concrete system and verifies all properties,
  demonstrating that a model satisfying all property axioms exists and hence
  no contradiction can be derived from the axiom system.

  Parameters:
    P = 4 features, L = 2 groups of 2
    ρ = 1/2, T = 100, c = 1
    First-mover gets T/(2-ρ²) = 400/7 splits
    Non-first-mover (same group) gets (1-ρ²)T/(2-ρ²) = 300/7 splits
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Card

set_option autoImplicit false

namespace UniversalImpossibility.Consistency

/-! ## Concrete parameters -/

def P : ℕ := 4
def L : ℕ := 2
def T : ℕ := 100
noncomputable def cρ : ℝ := 1 / 2

/-! ## Concrete model type -/

/-- Concrete model: one model per feature (model i has feature i as first-mover). -/
abbrev CModel := Fin 4

/-! ## Group assignment -/

/-- Group assignment: features 0,1 → group 0; features 2,3 → group 1. -/
def groupOf (j : Fin 4) : Fin 2 :=
  if j.val < 2 then ⟨0, by norm_num⟩ else ⟨1, by norm_num⟩

/-- The group of feature j. -/
def group (ℓ : Fin 2) : Finset (Fin 4) :=
  Finset.univ.filter (fun j => groupOf j = ℓ)

/-! ## First-mover and split counts -/

/-- First-mover: model i has feature i as first-mover. -/
def cFirstMover : CModel → Fin 4 := id

/-- Split count for feature j in model f.
    First-mover gets T/(2-ρ²), all others get (1-ρ²)T/(2-ρ²). -/
noncomputable def cSplitCount (j : Fin 4) (f : CModel) : ℝ :=
  if j = cFirstMover f then
    (T : ℝ) / (2 - cρ ^ 2)
  else
    (1 - cρ ^ 2) * (T : ℝ) / (2 - cρ ^ 2)

/-- Attribution = 1 · splitCount. -/
noncomputable def cAttribution (j : Fin 4) (f : CModel) : ℝ :=
  1 * cSplitCount j f

/-! ## Axiom verification -/

/-- Axiom 1 (firstMover_surjective): Every feature in a group is a first-mover
    for some model. Since cFirstMover = id, the witness is the feature itself. -/
theorem consistent_firstMover_surjective (ℓ : Fin 2) (j : Fin 4)
    (_hj : j ∈ group ℓ) :
    ∃ f : CModel, cFirstMover f = j :=
  ⟨j, rfl⟩

/-- Axiom 2 (splitCount_firstMover): When cFirstMover f = j,
    splitCount j f = T / (2 - ρ²). -/
theorem consistent_splitCount_firstMover (f : CModel) (j : Fin 4)
    (hfm : cFirstMover f = j) :
    cSplitCount j f = (T : ℝ) / (2 - cρ ^ 2) := by
  unfold cSplitCount
  rw [if_pos hfm.symm]

/-- Axiom 3 (splitCount_nonFirstMover): When cFirstMover f ≠ j and
    firstMover is in the same group as j,
    splitCount j f = (1 - ρ²) · T / (2 - ρ²). -/
theorem consistent_splitCount_nonFirstMover (f : CModel) (j : Fin 4)
    (ℓ : Fin 2) (_hj : j ∈ group ℓ)
    (hfm : cFirstMover f ≠ j)
    (_hfm_group : cFirstMover f ∈ group ℓ) :
    cSplitCount j f = (1 - cρ ^ 2) * (T : ℝ) / (2 - cρ ^ 2) := by
  unfold cSplitCount
  rw [if_neg (Ne.symm hfm)]

/-- Axiom 4 (proportionality_global): There exists a global c > 0 such that
    attribution j f = c · splitCount j f for all f, j. -/
theorem consistent_proportionality_global :
    ∃ c : ℝ, 0 < c ∧ ∀ (f : CModel) (j : Fin 4),
      cAttribution j f = c * cSplitCount j f :=
  ⟨1, one_pos, fun _ _ => rfl⟩

/-- Axiom 5 (splitCount_crossGroup_symmetric): Features in the same group
    have equal split counts when the first-mover is in a different group. -/
theorem consistent_splitCount_crossGroup_symmetric (f : CModel)
    (j k : Fin 4) (ℓ : Fin 2)
    (hj : j ∈ group ℓ) (hk : k ∈ group ℓ)
    (hfm_not_group : cFirstMover f ∉ group ℓ) :
    cSplitCount j f = cSplitCount k f := by
  have hfmj : j ≠ cFirstMover f := by
    intro h; exact hfm_not_group (h ▸ hj)
  have hfmk : k ≠ cFirstMover f := by
    intro h; exact hfm_not_group (h ▸ hk)
  unfold cSplitCount
  rw [if_neg hfmj, if_neg hfmk]

/-- Axiom 6 (splitCount_crossGroup_stable): Changing the first-mover within
    a group does not affect split counts for features outside that group.
    In the concrete model, if j ∉ group(ℓ) and both f, f' have first-movers
    in group(ℓ), then j ≠ f and j ≠ f', so both evaluate to the else-branch
    of cSplitCount. -/
theorem consistent_splitCount_crossGroup_stable (f f' : CModel)
    (j : Fin 4) (ℓ : Fin 2)
    (hj : j ∉ group ℓ) (hfm : cFirstMover f ∈ group ℓ)
    (hfm' : cFirstMover f' ∈ group ℓ) :
    cSplitCount j f = cSplitCount j f' := by
  have hfmj : j ≠ cFirstMover f := by
    intro h; exact hj (h ▸ hfm)
  have hfmj' : j ≠ cFirstMover f' := by
    intro h; exact hj (h ▸ hfm')
  unfold cSplitCount
  rw [if_neg hfmj, if_neg hfmj']

/-! ## Consistency statement -/

/-- The axiom system is consistent: there exists a concrete model satisfying
    all six domain-specific property axioms simultaneously. -/
theorem axiom_system_consistent :
    -- Axiom 1: first-mover surjectivity
    (∀ (ℓ : Fin 2) (j : Fin 4), j ∈ group ℓ → ∃ f : CModel, cFirstMover f = j) ∧
    -- Axiom 2: split count for first-mover
    (∀ (f : CModel) (j : Fin 4), cFirstMover f = j →
      cSplitCount j f = (T : ℝ) / (2 - cρ ^ 2)) ∧
    -- Axiom 3: split count for non-first-mover (same group)
    (∀ (f : CModel) (j : Fin 4) (ℓ : Fin 2),
      j ∈ group ℓ → cFirstMover f ≠ j → cFirstMover f ∈ group ℓ →
      cSplitCount j f = (1 - cρ ^ 2) * (T : ℝ) / (2 - cρ ^ 2)) ∧
    -- Axiom 4: global proportionality
    (∃ c : ℝ, 0 < c ∧ ∀ (f : CModel) (j : Fin 4),
      cAttribution j f = c * cSplitCount j f) ∧
    -- Axiom 5: cross-group symmetry
    (∀ (f : CModel) (j k : Fin 4) (ℓ : Fin 2),
      j ∈ group ℓ → k ∈ group ℓ → cFirstMover f ∉ group ℓ →
      cSplitCount j f = cSplitCount k f) ∧
    -- Axiom 6: cross-group stability
    (∀ (f f' : CModel) (j : Fin 4) (ℓ : Fin 2),
      j ∉ group ℓ → cFirstMover f ∈ group ℓ → cFirstMover f' ∈ group ℓ →
      cSplitCount j f = cSplitCount j f') :=
  ⟨consistent_firstMover_surjective,
   consistent_splitCount_firstMover,
   fun f j ℓ hj hfm hfmg => consistent_splitCount_nonFirstMover f j ℓ hj hfm hfmg,
   consistent_proportionality_global,
   consistent_splitCount_crossGroup_symmetric,
   consistent_splitCount_crossGroup_stable⟩

end UniversalImpossibility.Consistency
