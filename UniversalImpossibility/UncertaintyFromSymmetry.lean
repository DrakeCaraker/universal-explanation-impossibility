/-
  UncertaintyFromSymmetry.lean — The Uncertainty from Symmetry theorem.

  Proves the algebraic core: for a self-adjoint idempotent linear operator R
  (the Reynolds operator) on an inner product space, the Pythagorean
  decomposition ‖v - Rv‖² + ‖Rv‖² = ‖v‖² holds. This is the mathematical
  content connecting the η law (explanation instability) to quantum information
  loss under symmetry — both arise from the same projection onto the trivial
  representation.

  Zero new axioms. All proofs complete.

  Supplement: Theorem 5.3 (Uncertainty from Symmetry) in the monograph.
-/

import Mathlib.Analysis.InnerProductSpace.Basic

set_option autoImplicit false

namespace UniversalImpossibility

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-! ## Self-adjoint idempotent operators (orthogonal projections) -/

/-- For a self-adjoint idempotent operator R, ⟨Rv, v⟩ = ⟨Rv, Rv⟩.
    Proof: ⟨Rv, v⟩ = ⟨R(Rv), v⟩ (idempotent) = ⟨Rv, Rv⟩ (self-adjoint
    applied backwards: ⟨R(Rv), v⟩ = ⟨Rv, Rv⟩ since R is self-adjoint). -/
theorem inner_Rv_v_eq_inner_Rv_Rv
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    @inner ℝ V _ (R v) v = @inner ℝ V _ (R v) (R v) := by
  conv_lhs => rw [← hIdem v]
  exact hSA (R v) v

/-- For a self-adjoint idempotent operator R, ⟨v, Rv⟩ = ⟨Rv, Rv⟩.
    Follows from the previous lemma and symmetry of real inner product. -/
theorem inner_v_Rv_eq_inner_Rv_Rv
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    @inner ℝ V _ v (R v) = @inner ℝ V _ (R v) (R v) := by
  rw [real_inner_comm]
  exact inner_Rv_v_eq_inner_Rv_Rv R hIdem hSA v

/-- For a self-adjoint idempotent operator R, Rv ⊥ (v - Rv).
    This is the key orthogonality lemma. -/
theorem reynolds_orthogonal
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    @inner ℝ V _ (R v) (v - R v) = (0 : ℝ) := by
  rw [inner_sub_right]
  rw [inner_Rv_v_eq_inner_Rv_Rv R hIdem hSA]
  simp

/-- **Uncertainty from Symmetry (Pythagorean decomposition).**
    For a self-adjoint idempotent R, ‖v - Rv‖² + ‖Rv‖² = ‖v‖².

    This is the algebraic core of the uncertainty theorem. -/
theorem uncertainty_from_symmetry
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    ‖v - R v‖ ^ 2 + ‖R v‖ ^ 2 = ‖v‖ ^ 2 := by
  -- Use ‖a - b‖² = ‖a‖² - 2⟨a,b⟩ + ‖b‖² (norm_sub_sq_real)
  rw [norm_sub_sq_real]
  -- Goal: ‖v‖² - 2 * ⟨v, Rv⟩ + ‖Rv‖² + ‖Rv‖² = ‖v‖²
  -- Use ⟨v, Rv⟩ = ⟨Rv, Rv⟩ = ‖Rv‖²
  have hinner : @inner ℝ V _ v (R v) = ‖R v‖ ^ 2 := by
    rw [inner_v_Rv_eq_inner_Rv_Rv R hIdem hSA]
    rw [real_inner_self_eq_norm_sq]
  linarith

/-- The information loss: ‖v - Rv‖² = ‖v‖² - ‖Rv‖². -/
theorem information_loss
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    ‖v - R v‖ ^ 2 = ‖v‖ ^ 2 - ‖R v‖ ^ 2 := by
  linarith [uncertainty_from_symmetry R hIdem hSA v]

/-- The stable content is bounded: ‖Rv‖ ≤ ‖v‖. -/
theorem stable_bounded
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    ‖R v‖ ≤ ‖v‖ := by
  have hsq : ‖R v‖ ^ 2 ≤ ‖v‖ ^ 2 := by
    nlinarith [uncertainty_from_symmetry R hIdem hSA v, sq_nonneg ‖v - R v‖]
  nlinarith [sq_nonneg (‖v‖ - ‖R v‖), norm_nonneg (R v), norm_nonneg v]

/-- The unstable content is bounded: ‖v - Rv‖ ≤ ‖v‖. -/
theorem unstable_bounded
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v : V) :
    ‖v - R v‖ ≤ ‖v‖ := by
  have hsq : ‖v - R v‖ ^ 2 ≤ ‖v‖ ^ 2 := by
    nlinarith [uncertainty_from_symmetry R hIdem hSA v, sq_nonneg ‖R v‖]
  nlinarith [sq_nonneg (‖v‖ - ‖v - R v‖), norm_nonneg (v - R v), norm_nonneg v]

/-- The residual R(v - Rv) = 0: projecting the residual gives zero. -/
theorem reynolds_residual_zero
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (v : V) :
    R (v - R v) = 0 := by
  rw [map_sub, hIdem]
  simp

/-- The residual is orthogonal to any fixed vector. -/
theorem residual_orthogonal_to_fixed
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v w : V) (hw : R w = w) :
    @inner ℝ V _ (v - R v) w = (0 : ℝ) := by
  rw [← hw]
  conv_lhs => rw [← hSA]
  rw [reynolds_residual_zero R hIdem]
  exact inner_zero_left w

/-- **Best approximation.** Rv minimises ‖v - w‖ among all fixed points w.
    Proof: ‖v - w‖² = ‖v - Rv‖² + ‖Rv - w‖² ≥ ‖v - Rv‖² since the
    residual v - Rv is orthogonal to Rv - w (both lie in complementary
    subspaces). -/
theorem best_approximation
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v w : V) (hw : R w = w) :
    ‖v - R v‖ ^ 2 ≤ ‖v - w‖ ^ 2 := by
  -- Write v - w = (v - Rv) + (Rv - w)
  have hdecomp : v - w = (v - R v) + (R v - w) := by abel
  -- The cross term vanishes: ⟨v - Rv, Rv - w⟩ = 0
  have horth : @inner ℝ V _ (v - R v) (R v - w) = (0 : ℝ) := by
    rw [inner_sub_right]
    -- ⟨v - Rv, Rv⟩ = 0 (residual orthogonal to Rv, which is fixed)
    have h1 : @inner ℝ V _ (v - R v) (R v) = (0 : ℝ) :=
      residual_orthogonal_to_fixed R hIdem hSA v (R v) (hIdem v)
    -- ⟨v - Rv, w⟩ = 0 (residual orthogonal to w, which is fixed)
    have h2 : @inner ℝ V _ (v - R v) w = (0 : ℝ) :=
      residual_orthogonal_to_fixed R hIdem hSA v w hw
    linarith
  -- ‖v - w‖² = ‖(v-Rv) + (Rv-w)‖² = ‖v-Rv‖² + 2⟨v-Rv, Rv-w⟩ + ‖Rv-w‖²
  rw [hdecomp, norm_add_sq_real]
  linarith [sq_nonneg ‖R v - w‖]

/-! ## Connection to the η law and quantum information

The Pythagorean decomposition ‖v - Rv‖² + ‖Rv‖² = ‖v‖² means that
for any explanation vector v, the orbit average Rv captures exactly
the G-invariant content, and the remainder ‖v - Rv‖² is the information
cost of stability.

For v uniformly distributed on the unit sphere S^{n-1} in ℝ^n:
  E[‖Rv‖²] = dim(V^G) / dim(V) = 1 - η
  E[‖v - Rv‖²] = η = 1 - dim(V^G) / dim(V)

This is the η law: the fraction of explanation content lost by orbit
averaging is exactly η, determined by the symmetry group G.

The SAME formula governs quantum information loss: for a quantum system
with symmetry G acting on Hilbert space H, the fraction of state
information accessible to G-covariant measurements is dim(H^G)/dim(H).

Both arise from the Pythagorean decomposition under the Reynolds/twirl
operator — the same algebraic structure applied to different spaces.

The expected-value statement (part iii of the theorem in the monograph)
uses integration over the unit sphere. The algebraic core — the Pythagorean
decomposition, information loss formula, best approximation property, and
norm bounds — is fully proved above. -/

end UniversalImpossibility
