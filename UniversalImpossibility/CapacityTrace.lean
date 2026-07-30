/-
  Deriving the explanation-capacity law from first principles.

  The monograph's central quantitative object is the explanation capacity
  `C = dim(V^G)` and the loss rate `η = 1 − dim(V^G)/dim(V)`, where V^G is the
  fixed subspace of the orbit-average / Reynolds operator R. Prior to this file
  that identity lived only in a doc-comment (`UncertaintyFromSymmetry.lean`
  states η but proves only the Pythagorean algebra). Here we PROVE the load-
  bearing identity: for the Reynolds operator (an idempotent projection),

      trace R = dim(V^G),   with V^G = range R = { v | R v = v },

  so the capacity is literally the trace of the projection and η is a computed
  quantity, not a fitted one. This is the honest core of the capacity law;
  the separate sphere-expectation form E‖Rv‖² = dim(V^G)/dim(V) additionally
  needs Haar/sphere integration and is not claimed here.
-/
import Mathlib.LinearAlgebra.Trace
import Mathlib.LinearAlgebra.Projection
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

open LinearMap Module

namespace UniversalImpossibility.Capacity

variable {𝕜 V : Type*} [Field 𝕜] [AddCommGroup V] [Module 𝕜 V] [FiniteDimensional 𝕜 V]

/-- The explanation capacity of a Reynolds/orbit-average operator `R`: the
    dimension of its fixed subspace `V^G = range R`. -/
noncomputable def capacity (R : V →ₗ[𝕜] V) : ℕ := finrank 𝕜 (LinearMap.range R)

/-- The range of an idempotent operator is exactly its fixed subspace
    `V^G = { v | R v = v }`. (So `capacity R = dim V^G`.) -/
theorem mem_range_iff_fixed (R : V →ₗ[𝕜] V) (hR : IsIdempotentElem R) (v : V) :
    v ∈ LinearMap.range R ↔ R v = v := by
  constructor
  · rintro ⟨w, rfl⟩
    exact congrFun (congrArg (DFunLike.coe) hR) w
  · intro h
    exact ⟨v, h⟩

/-- **Capacity = trace of the Reynolds projection.**
    For an idempotent `R` (the orbit-average operator, `R ∘ R = R`), the trace
    equals the dimension of the fixed subspace `V^G = range R`. This turns the
    monograph's `C = dim(V^G)` into a computed trace identity. -/
theorem capacity_eq_trace (R : V →ₗ[𝕜] V) (hR : IsIdempotentElem R) :
    LinearMap.trace 𝕜 V R = (capacity R : 𝕜) := by
  have hproj : LinearMap.IsProj (LinearMap.range R) R :=
    (LinearMap.isProj_range_iff_isIdempotentElem R).mpr hR
  simpa [capacity] using hproj.trace

/-- The explanation loss rate `η = 1 − C/dim(V)`, with `C = capacity = tr R`. -/
noncomputable def lossRate (R : V →ₗ[𝕜] V) : ℚ :=
  1 - (capacity R : ℚ) / (finrank 𝕜 V : ℚ)

/-- Capacity is bounded by the ambient dimension, so `η ∈ [0,1]` is well-posed. -/
theorem capacity_le_dim (R : V →ₗ[𝕜] V) : capacity R ≤ finrank 𝕜 V :=
  Submodule.finrank_le _

/-! ### The theorem applies to the ACTUAL Reynolds operator, not just any idempotent.

The capacity theorem above assumes an idempotent `R`. To close the objection "that is
an abstract projection, not the orbit-average the framework actually uses", we build the
genuine Reynolds operator of a finite-group linear representation and prove it idempotent,
so `capacity_eq_trace` gives: capacity of the orbit-average = dim(V^G). -/

section Reynolds

variable {G : Type*} [Group G] [Fintype G]
variable {𝕜 V : Type*} [Field 𝕜] [CharZero 𝕜] [AddCommGroup V] [Module 𝕜 V]

/-- The Reynolds operator (orbit average) of a finite-group linear representation
    `ρ : G →* (V →ₗ V)`: `R = |G|⁻¹ • Σ_g ρ g`. Its fixed subspace is `V^G`. -/
noncomputable def reynolds (ρ : G →* (V →ₗ[𝕜] V)) : V →ₗ[𝕜] V :=
  (Fintype.card G : 𝕜)⁻¹ • ∑ g : G, ρ g

/-- Right-invariance of the raw sum: `(Σ_g ρ g) ∘ ρ h = Σ_g ρ g`. -/
theorem sum_rep_mul (ρ : G →* (V →ₗ[𝕜] V)) (h : G) :
    (∑ g : G, ρ g) * ρ h = ∑ g : G, ρ g := by
  rw [Finset.sum_mul]
  simp_rw [← ρ.map_mul]
  exact Fintype.sum_bijective (· * h) (Equiv.mulRight h).bijective (fun g => ρ (g * h)) ρ
    (fun _ => rfl)

/-- **The Reynolds operator is idempotent** (`R ∘ R = R`), so it is a genuine projection
    onto `V^G` and `capacity_eq_trace` applies to it. -/
theorem reynolds_isIdempotent (ρ : G →* (V →ₗ[𝕜] V)) :
    IsIdempotentElem (reynolds ρ) := by
  have hcard : (Fintype.card G : 𝕜) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hSS : (∑ g : G, ρ g) * (∑ g : G, ρ g) = (Fintype.card G : 𝕜) • ∑ g : G, ρ g := by
    rw [Finset.mul_sum]
    simp_rw [sum_rep_mul ρ]
    rw [Finset.sum_const, Finset.card_univ, Nat.cast_smul_eq_nsmul]
  unfold IsIdempotentElem reynolds
  rw [smul_mul_assoc, mul_smul_comm, hSS, smul_smul, smul_smul]
  congr 1
  field_simp

/-- **Capacity of the orbit-average = dim(V^G).** Combining `reynolds_isIdempotent` with
    `capacity_eq_trace`: for the actual Reynolds operator of a finite-group representation
    on a finite-dimensional space, `trace R = capacity R = dim(V^G)`. -/
theorem reynolds_capacity_eq_trace [FiniteDimensional 𝕜 V] (ρ : G →* (V →ₗ[𝕜] V)) :
    LinearMap.trace 𝕜 V (reynolds ρ) = (capacity (reynolds ρ) : 𝕜) :=
  capacity_eq_trace _ (reynolds_isIdempotent ρ)

end Reynolds

end UniversalImpossibility.Capacity
