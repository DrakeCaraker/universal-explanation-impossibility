/-
  MolienCapacity.lean — the explanation capacity IS a Molien / character-average.

  The capacity theory (`CapacityTrace.lean`) proves `capacity = trace of the Reynolds
  projection = dim(V^G)`. This file connects that to classical invariant theory: the
  capacity equals the group-average of the character (the trace of `ρ g`), which is the
  **degree-1 Molien coefficient** — the dimension of the linear invariants — and, via
  Mathlib's character theory, literally `finrank 𝕜 (Representation.invariants ρ)`.

      capacity(R) = (1/|G|) · Σ_g tr(ρ g)  =  dim of the G-invariant subspace.

  This closes the "Molien in Lean" gap at the level the capacity theory actually uses:
  the a-priori dimension of the stable (invariant) explanation subspace is a computable
  group-average, machine-checked and tied to Mathlib's `Representation.invariants`. It is
  exactly the `C_1` the empirical Molien computation reports (for the S_k permutation rep,
  the average number of fixed points = the number of orbits = 1).

  Honest scope: this is the ungraded / degree-1 Molien coefficient. The full graded Molien
  SERIES (the generating function Σ_d dim(Sym^d V)^G t^d = (1/|G|) Σ_g 1/det(1 - t·ρ g),
  whose degree-2 coefficient is the interaction-capacity C_2) needs the symmetric-power
  representations and the determinant identity, which Mathlib does not yet provide; those
  coefficients remain the province of the (rigorous, but non-Lean) `molien_coeffs`
  computation. What is machine-checked here is the load-bearing identity capacity = average
  character = dim of invariants.

  Axiom-clean: depends only on Lean core + Mathlib (no `gbdtWorld`, no `native_decide`, no `sorry`).
-/
import UniversalImpossibility.CapacityTrace
import Mathlib.RepresentationTheory.Character

open LinearMap Module
open UniversalImpossibility.Capacity

namespace UniversalImpossibility.Molien

variable {G : Type*} [Group G] [Fintype G]
variable {𝕜 V : Type*} [Field 𝕜] [CharZero 𝕜] [AddCommGroup V] [Module 𝕜 V]
  [FiniteDimensional 𝕜 V]

/-- **The capacity is the character-average (degree-1 Molien coefficient).** For the
Reynolds operator of a finite-group representation, the explanation capacity equals the
group-average of the trace of `ρ g` — i.e. `(1/|G|) Σ_g χ(g)`, the classical formula for
the dimension of the invariant subspace. -/
theorem capacity_reynolds_eq_char_average (ρ : G →* (V →ₗ[𝕜] V)) :
    (capacity (reynolds ρ) : 𝕜)
      = (Fintype.card G : 𝕜)⁻¹ * ∑ g : G, LinearMap.trace 𝕜 V (ρ g) := by
  rw [← reynolds_capacity_eq_trace ρ]
  simp only [reynolds, map_smul, map_sum, smul_eq_mul]

/-- **Capacity = dimension of the G-invariant subspace**, via Mathlib's character theory.
Chains the character-average form with `Representation.card_inv_mul_sum_char_eq_finrank`:
the explanation capacity of the orbit-average is exactly `finrank 𝕜 (invariants ρ)`. This
is the machine-checked "capacity = dim(V^G)" tied to classical invariant theory. -/
theorem capacity_reynolds_eq_finrank_invariants (ρ : G →* (V →ₗ[𝕜] V)) :
    capacity (reynolds ρ) = finrank 𝕜 (Representation.invariants ρ) := by
  haveI : Invertible (Nat.card G : 𝕜) :=
    invertibleOfNonzero (by rw [Nat.card_eq_fintype_card]; exact_mod_cast Fintype.card_ne_zero)
  have hchar : ∀ g : G, Representation.character ρ g = LinearMap.trace 𝕜 V (ρ g) := fun _ => rfl
  have h2 := Representation.card_inv_mul_sum_char_eq_finrank (k := 𝕜) (G := G) (V := V) ρ
  have hcast : ((capacity (reynolds ρ) : ℕ) : 𝕜)
      = ((finrank 𝕜 (Representation.invariants ρ) : ℕ) : 𝕜) := by
    rw [capacity_reynolds_eq_char_average ρ, ← h2, Nat.card_eq_fintype_card]
    simp only [hchar]
  exact_mod_cast hcast

end UniversalImpossibility.Molien
