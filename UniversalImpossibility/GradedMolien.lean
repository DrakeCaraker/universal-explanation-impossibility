/-
  GradedMolien.lean — closing the graded-classification asterisk.

  With the finite-dimensionality blocker removed (`homogeneousSubmodule_finiteDimensional`),
  the graded Molien classification is no longer missing mathematics: it is the degree-1
  character-average engine applied to each symmetric power. This file builds the graded
  representation and proves it.

  The symmetric group `Equiv.Perm (Fin n)` acts on degree-`d` homogeneous polynomials by
  renaming variables; `rename` is a genuine monoid homomorphism on permutations
  (`rename_rename` + `Equiv.Perm.coe_mul`) and preserves homogeneity, so this is a bona fide
  finite-dimensional representation `renameRep`. Applying the degree-1 formula
  (`capacity_reynolds_eq_char_average` / `capacity_reynolds_eq_finrank_invariants`) gives:

    dim of the degree-`d` symmetric (invariant) homogeneous polynomials
        = the degree-`d` graded Molien coefficient
        = (1/|G|) Σ_g (character of the degree-`d` rename action).

  The left side is the graded stable-fact count `C_d` (degree 1 = main effects, degree 2 =
  interactions, …); the theorem holds for every degree `d`. This is the machine-checked
  graded classification, closing the asterisk.

  Axiom-clean (Lean core + Mathlib).
-/
import UniversalImpossibility.HomogeneousFinite
import UniversalImpossibility.MolienCapacity

open MvPolynomial LinearMap Module
open UniversalImpossibility.Capacity UniversalImpossibility.Molien

namespace UniversalImpossibility.Graded

variable (𝕜 : Type*) [Field 𝕜] [CharZero 𝕜] (n d : ℕ)

/-- Renaming by a permutation preserves the degree-`d` homogeneous submodule. -/
theorem rename_mapsTo (σ : Equiv.Perm (Fin n)) (p : MvPolynomial (Fin n) 𝕜)
    (hp : p ∈ homogeneousSubmodule (Fin n) 𝕜 d) :
    (rename ⇑σ).toLinearMap p ∈ homogeneousSubmodule (Fin n) 𝕜 d := by
  rw [AlgHom.toLinearMap_apply, mem_homogeneousSubmodule]
  rw [mem_homogeneousSubmodule] at hp
  exact hp.rename_isHomogeneous

/-- The endomorphism of the degree-`d` homogeneous submodule given by renaming variables
along a permutation. -/
noncomputable def renameEnd (σ : Equiv.Perm (Fin n)) :
    Module.End 𝕜 (homogeneousSubmodule (Fin n) 𝕜 d) :=
  LinearMap.restrict (rename ⇑σ).toLinearMap (rename_mapsTo 𝕜 n d σ)

/-- **The graded rename representation.** `Equiv.Perm (Fin n)` acts on degree-`d`
homogeneous polynomials by renaming variables — a finite-dimensional representation. -/
noncomputable def renameRep :
    Equiv.Perm (Fin n) →* Module.End 𝕜 (homogeneousSubmodule (Fin n) 𝕜 d) where
  toFun := renameEnd 𝕜 n d
  map_one' := by
    apply LinearMap.ext; intro x; apply Subtype.ext
    simp only [renameEnd, LinearMap.restrict_coe_apply, AlgHom.toLinearMap_apply,
      Equiv.Perm.coe_one, rename_id, AlgHom.id_apply, Module.End.one_apply]
  map_mul' σ₁ σ₂ := by
    apply LinearMap.ext; intro x; apply Subtype.ext
    simp only [renameEnd, Module.End.mul_apply, LinearMap.restrict_coe_apply,
      AlgHom.toLinearMap_apply, Equiv.Perm.coe_mul, ← rename_rename]

/-- **Graded Molien classification (closed).** The dimension of the degree-`d`
symmetric-group-invariant homogeneous polynomials — the graded Molien coefficient `C_d`,
i.e. the number of stable degree-`d` explanation facts — equals the average of the degree-`d`
rename character. This is the degree-1 character-average formula applied to the `d`-th
symmetric power, for every degree `d`. -/
theorem graded_molien_char_average :
    (finrank 𝕜 (Representation.invariants (renameRep 𝕜 n d)) : 𝕜)
      = (Fintype.card (Equiv.Perm (Fin n)) : 𝕜)⁻¹
        * ∑ σ : Equiv.Perm (Fin n),
            LinearMap.trace 𝕜 (homogeneousSubmodule (Fin n) 𝕜 d) (renameRep 𝕜 n d σ) := by
  haveI : FiniteDimensional 𝕜 (homogeneousSubmodule (Fin n) 𝕜 d) :=
    homogeneousSubmodule_finiteDimensional 𝕜 n d
  rw [← UniversalImpossibility.Molien.capacity_reynolds_eq_finrank_invariants (renameRep 𝕜 n d)]
  exact UniversalImpossibility.Molien.capacity_reynolds_eq_char_average (renameRep 𝕜 n d)

end UniversalImpossibility.Graded
