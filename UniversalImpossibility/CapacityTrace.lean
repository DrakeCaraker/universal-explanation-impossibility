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

end UniversalImpossibility.Capacity
