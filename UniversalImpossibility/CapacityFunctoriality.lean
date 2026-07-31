/-
  CapacityFunctoriality.lean — explanation capacity as a well-behaved invariant,
  and the bridge from the structural/statistical dichotomy to the capacity loss rate.

  `CapacityTrace.lean` proves capacity = trace of the Reynolds projection = dim(V^G);
  `MolienCapacity.lean` ties it to the average character. This file makes capacity a
  proper *theory of an invariant*:

  * **Structural-floor dimension = η · dim** (`structural_floor_dim`, `lossRate_eq_ker_ratio`).
    The incurable/structural instability directions are exactly the kernel of the
    Reynolds operator (the non-invariant "orbit" directions the average annihilates),
    and their dimension is `dim V − capacity = η · dim V`. This is the quantitative
    link between the structural/statistical dichotomy (whose incurable component is the
    orbit) and the capacity loss rate η — and the mathematical justification for
    predicting the large-n instability *floor* from η.
  * **Monotonicity** (`capacity_mono`, `capacity_antitone_subgroup`): more symmetry ⇒
    fewer invariants ⇒ smaller capacity. A subgroup has at least as much capacity as
    the whole group.
  * **Trivial symmetry ⇒ full capacity** (`capacity_id`): with no symmetry to quotient
    by, every direction is stable.

  Axiom-clean (Lean core + Mathlib only).
-/
import UniversalImpossibility.CapacityTrace
import UniversalImpossibility.MolienCapacity
import Mathlib.RepresentationTheory.Invariants

open LinearMap Module

namespace UniversalImpossibility.Capacity

variable {𝕜 V : Type*} [Field 𝕜] [AddCommGroup V] [Module 𝕜 V] [FiniteDimensional 𝕜 V]

/-- **Structural-floor dimension.** The incurable (structural) instability subspace is
the kernel of the projection `R` — the directions it annihilates — and its dimension is
`dim V − capacity R`. -/
theorem structural_floor_dim (R : V →ₗ[𝕜] V) :
    finrank 𝕜 (LinearMap.ker R) = finrank 𝕜 V - capacity R := by
  have h := capacity_add_ker R
  omega

/-- **η is the structural-floor fraction.** The loss rate `η = 1 − capacity/dim` equals
the fraction of directions in the incurable kernel: `η = dim(ker R)/dim V`. This is the
identity that justifies predicting the large-n instability floor from the capacity. -/
theorem lossRate_eq_ker_ratio (R : V →ₗ[𝕜] V) (hV : finrank 𝕜 V ≠ 0) :
    lossRate R = (finrank 𝕜 (LinearMap.ker R) : ℚ) / (finrank 𝕜 V : ℚ) := by
  have hle := capacity_le_dim R
  have hVQ : (finrank 𝕜 V : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hV
  rw [lossRate, structural_floor_dim, Nat.cast_sub hle]
  field_simp

/-- **Capacity is monotone in the fixed subspace.** If the range of `R₁` sits inside the
range of `R₂`, its capacity is no larger. -/
theorem capacity_mono {R₁ R₂ : V →ₗ[𝕜] V}
    (h : LinearMap.range R₁ ≤ LinearMap.range R₂) : capacity R₁ ≤ capacity R₂ :=
  Submodule.finrank_mono h

/-- **No symmetry ⇒ full capacity.** The identity projection (nothing to average away)
has capacity equal to the ambient dimension: every direction is stable. -/
theorem capacity_id : capacity (LinearMap.id : V →ₗ[𝕜] V) = finrank 𝕜 V := by
  rw [capacity, LinearMap.range_id, finrank_top]

section Subgroup

variable {G : Type*} [Group G]

/-- **More symmetry ⇒ smaller capacity.** Restricting a representation to a subgroup can
only enlarge the invariant subspace, so a subgroup has at least the capacity of the whole
group: `dim(V^G) ≤ dim(V^H)` for `H ≤ G`. Capacity is antitone in the symmetry group. -/
theorem capacity_antitone_subgroup (ρ : Representation 𝕜 G V) (H : Subgroup G) :
    finrank 𝕜 (Representation.invariants ρ)
      ≤ finrank 𝕜 (Representation.invariants (ρ.comp H.subtype)) := by
  apply Submodule.finrank_mono
  intro v hv
  rw [Representation.mem_invariants] at hv ⊢
  intro h
  exact hv (H.subtype h)

end Subgroup

end UniversalImpossibility.Capacity
