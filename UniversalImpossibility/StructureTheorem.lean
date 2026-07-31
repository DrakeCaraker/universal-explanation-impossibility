/-
  StructureTheorem.lean — the keystone that closes the theory: stable ⟺ G-invariant,
  and the canonical decomposition of an explanation system.

  The capacity theory proves capacity = dim(V^G) is an UPPER bound on the stable-fact
  count, and coverage-validates it. The keystone converse -- previously a Tier-B
  conjecture -- makes it EXACT:

    * `stable_iff_gInvariant`: for a system with a symmetry group `G` (acting so as to
      preserve observables and transitively on observe-fibers), the stable resolutions
      are EXACTLY the G-invariant ones. Achievability (invariant ⇒ stable) is
      `gInvariant_stable`; the converse (stable ⇒ invariant) needs only that `G`
      preserves observables. So nothing outside the invariants is stable, and the
      capacity is the exact stable-fact count, not merely a bound.

    * `explanation_structure_theorem`: in the linear setting, the orbit-average
      (Reynolds) operator `R` gives the canonical decomposition `V = V^G ⊕ ker R`:
      `R` is a projection, its range is exactly the stable subspace `V^G`, its trace /
      dimension is the capacity = the average character, and the complementary unstable
      subspace `ker R` (the incurable structural instability) has dimension
      `dim V − capacity`. Impossibility, resolution, capacity, and structural floor are
      thus four facets of ONE decomposition.

  Axiom-clean (Lean core + Mathlib).
-/
import UniversalImpossibility.UniversalResolution
import UniversalImpossibility.MolienCapacity

set_option autoImplicit false

/-! ## Part 1 — the abstract characterization: stable ⟺ G-invariant -/

section Abstract

variable {Θ H Y G : Type}

/-- **Converse of `gInvariant_stable` (the keystone).** A stable resolution is
G-invariant: if `R` factors through the observable map and `G` preserves observables,
then `R` is constant on `G`-orbits. Needs only observable-preservation. -/
theorem stable_gInvariant (S : ExplanationSystem Θ H Y) (R : Θ → H)
    [Group G] [MulAction G Θ] (sym : HasSymmetry S G) (hstab : stable S R) :
    gInvariant R G := by
  intro g θ
  exact hstab (g • θ) θ (sym.observe_invariant g θ)

/-- **The stable resolutions are exactly the G-invariant ones.** Under a symmetry
structure, `stable S R ↔ gInvariant R G`. This upgrades the capacity from an upper
bound on the stable-fact count to the exact count: the stable maps are precisely the
invariant maps, so `dim(V^G)` is achieved and not exceeded. -/
theorem stable_iff_gInvariant (S : ExplanationSystem Θ H Y) (R : Θ → H)
    [Group G] [MulAction G Θ] (sym : HasSymmetry S G) :
    stable S R ↔ gInvariant R G :=
  ⟨stable_gInvariant S R sym, gInvariant_stable S R sym⟩

end Abstract

/-! ## Part 2 — the linear structure theorem: one canonical decomposition -/

namespace UniversalImpossibility.Capacity

open LinearMap Module

variable {G : Type*} [Group G] [Fintype G]
variable {𝕜 V : Type*} [Field 𝕜] [CharZero 𝕜] [AddCommGroup V] [Module 𝕜 V]
  [FiniteDimensional 𝕜 V]

/-- **The structure theorem for explanation systems (linear form).** The orbit-average
(Reynolds) operator `R = |G|⁻¹ Σ_g ρ(g)` of a finite-group representation gives the
canonical decomposition of the explanation space into a stable and an unstable part.
The four facets of the theory are facets of this one operator:

* `R` is a projection (idempotent);
* its range is exactly the stable/fixed subspace `V^G = {v | R v = v}`;
* the capacity `dim(V^G)` equals the average character `|G|⁻¹ Σ_g tr(ρ g)`;
* the ambient space splits as `capacity + dim(ker R) = dim V`, so the incurable
  structural-instability subspace `ker R` has dimension `dim V − capacity`. -/
theorem explanation_structure_theorem (ρ : G →* (V →ₗ[𝕜] V)) :
    IsIdempotentElem (reynolds ρ)
    ∧ (∀ v : V, v ∈ LinearMap.range (reynolds ρ) ↔ reynolds ρ v = v)
    ∧ (capacity (reynolds ρ) : 𝕜)
        = (Fintype.card G : 𝕜)⁻¹ * ∑ g : G, LinearMap.trace 𝕜 V (ρ g)
    ∧ capacity (reynolds ρ) + finrank 𝕜 (LinearMap.ker (reynolds ρ)) = finrank 𝕜 V := by
  refine ⟨reynolds_isIdempotent ρ, ?_, UniversalImpossibility.Molien.capacity_reynolds_eq_char_average ρ,
    capacity_add_ker (reynolds ρ)⟩
  exact fun v => mem_range_iff_fixed (reynolds ρ) (reynolds_isIdempotent ρ) v

end UniversalImpossibility.Capacity
