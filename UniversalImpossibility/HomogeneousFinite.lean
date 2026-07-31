/-
  HomogeneousFinite.lean — the finite-dimensionality crux for the graded classification.

  Completing the graded Molien classification in Lean was blocked on one thing: the
  degree-`d` piece of the invariant polynomial ring must be finite-dimensional for the
  capacity/trace machinery (and the degree-1 character-average engine) to apply to it.
  Mathlib provides `Module.Finite` for the total-degree-≤-`N` submodule but NOT for the
  homogeneous piece. This file supplies the missing lemma: the degree-`d` homogeneous
  submodule of `MvPolynomial (Fin n) 𝕜` is finite-dimensional, because it sits inside the
  finite-dimensional total-degree-≤-`d` submodule.

  This downgrades the graded-classification asterisk from "research-scale blocked" to "a
  bounded, unblocked construction": the remaining work is to package the `renameEquiv`
  action on this (now finite-dimensional) space as a representation and apply the
  already-proven degree-1 formula `capacity_reynolds_eq_finrank_invariants` degree by
  degree. The mathematics is done; this removes the last infrastructure blocker.
  (Candidate for upstreaming to Mathlib.)

  Axiom-clean (Lean core + Mathlib).
-/
import Mathlib.RingTheory.MvPolynomial.Homogeneous
import Mathlib.RingTheory.MvPolynomial.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Defs

open MvPolynomial

namespace UniversalImpossibility.Graded

variable (𝕜 : Type*) [Field 𝕜] (n d : ℕ)

/-- The degree-`d` homogeneous submodule is contained in the total-degree-≤-`d` submodule
(a homogeneous polynomial of degree `d` has total degree `d`, or `0` if it is zero). -/
theorem homogeneousSubmodule_le_restrictTotalDegree :
    homogeneousSubmodule (Fin n) 𝕜 d ≤ restrictTotalDegree (Fin n) 𝕜 d := by
  intro p hp
  rw [mem_restrictTotalDegree]
  by_cases hp0 : p = 0
  · simp [hp0]
  · have hhom : p.IsHomogeneous d := (mem_homogeneousSubmodule d p).mp hp
    have := hhom.totalDegree hp0
    omega

/-- **The degree-`d` homogeneous piece is finite-dimensional.** The last infrastructure
blocker for the graded Molien classification in Lean: with this, the degree-1
character-average formula applies to each symmetric power / homogeneous degree, giving the
graded stable-fact counts. -/
theorem homogeneousSubmodule_finiteDimensional :
    FiniteDimensional 𝕜 (homogeneousSubmodule (Fin n) 𝕜 d) := by
  have hle := homogeneousSubmodule_le_restrictTotalDegree 𝕜 n d
  haveI : FiniteDimensional 𝕜 (restrictTotalDegree (Fin n) 𝕜 d) := inferInstance
  exact FiniteDimensional.of_injective (Submodule.inclusion hle) (Submodule.inclusion_injective hle)

end UniversalImpossibility.Graded
