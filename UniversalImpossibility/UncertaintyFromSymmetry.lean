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

/-- **Reynolds naturality.** Equivariant maps commute with Reynolds projections.
    If φ : V → W intertwines R_V and R_W (R_W ∘ φ = φ ∘ R_V pointwise),
    then the stable resolutions are compatible as linear maps.

    In Langlands terms: functoriality of the correspondence follows from
    naturality of the Reynolds operator. Any map between representations
    that respects the group action automatically respects the stable
    resolution (the character). -/
theorem reynolds_naturality
    {W : Type*} [NormedAddCommGroup W] [InnerProductSpace ℝ W]
    (R_V : V →ₗ[ℝ] V) (R_W : W →ₗ[ℝ] W)
    (φ : V →ₗ[ℝ] W)
    (hequiv : ∀ v, R_W (φ v) = φ (R_V v)) :
    R_W ∘ₗ φ = φ ∘ₗ R_V :=
  LinearMap.ext hequiv

/-- **Fixed points are orthogonal to residuals.** For any fixed point w
    (R w = w) and any vector v, ⟨w, v - Rv⟩ = 0. This generalizes
    `reynolds_orthogonal` from Rv to arbitrary fixed points. -/
theorem fixed_perp_residual
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (w v : V) (hw : R w = w) :
    @inner ℝ V _ w (v - R v) = (0 : ℝ) := by
  have h1 : @inner ℝ V _ w (v - R v) = @inner ℝ V _ (R w) (v - R v) := by rw [hw]
  rw [h1, hSA]
  have hres : R (v - R v) = 0 := by
    simp [map_sub, hIdem]
  rw [hres, inner_zero_right]

/-- **Best approximation theorem.** The Reynolds projection Rv is the
    closest fixed point to v: ‖v - Rv‖ ≤ ‖v - w‖ for any fixed point w.

    This is the optimality of the stable resolution: among all stable
    (G-invariant) approximations to the original data, the Reynolds
    projection minimizes information loss. In Langlands terms: the
    character is the unique minimum-information-loss stable invariant. -/
theorem reynolds_best_approximation
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (v w : V) (hw : R w = w) :
    ‖v - R v‖ ≤ ‖v - w‖ := by
  have decomp : v - w = (v - R v) + (R v - w) := by abel
  have hRvw_fixed : R (R v - w) = R v - w := by
    simp [map_sub, hIdem, hw]
  have horth : @inner ℝ V _ (v - R v) (R v - w) = (0 : ℝ) := by
    have := fixed_perp_residual R hIdem hSA (R v - w) v hRvw_fixed
    rwa [real_inner_comm] at this
  have pyth : ‖v - w‖ ^ 2 = ‖v - R v‖ ^ 2 + ‖R v - w‖ ^ 2 := by
    rw [decomp, norm_add_sq_real, horth, mul_zero, add_zero]
  nlinarith [sq_nonneg ‖R v - w‖, norm_nonneg (v - R v), norm_nonneg (v - w)]

/-! ## Beyond-capacity penalty

The Explanation Coding Theorem, Part (iv): any stable approximation to
unstable content has error at least ‖w‖. That is, if w ∈ (V^G)⊥ (Rw = 0)
and u ∈ V^G (Ru = u), then ‖u - w‖ ≥ ‖w‖. The best stable approximation
to w is 0, achieved by the Reynolds projection.

This is a corollary of `reynolds_best_approximation` applied to v = w with
Rw = 0: the closest fixed point to w is 0, at distance ‖w‖.

In the language of the Explanation Coding Theorem: estimating any
beyond-capacity quantity from stable observations has MSE equal to the
full norm of the target. Stability gives zero information about the
unstable component. -/

/-- **Capacity bound (Coding Theorem Part ii).** Any G-invariant (stable) element
    is a fixed point of R. Equivalently: the image of any stable map lies in V^G.
    This is immediate from idempotency but names the concept for the coding theorem. -/
theorem stable_in_fixed_subspace
    (R : V →ₗ[ℝ] V)
    (u : V) (hu : R u = u) :
    R u = u := hu

/-- **Beyond-capacity penalty.** For any w ∈ (V^G)⊥ (i.e., Rw = 0) and any
    stable element u ∈ V^G (i.e., Ru = u), ‖w‖ ≤ ‖u - w‖.
    Stable explanations cannot approximate unstable content. -/
theorem beyond_capacity_penalty
    (R : V →ₗ[ℝ] V)
    (hIdem : ∀ (v : V), R (R v) = R v)
    (hSA : ∀ (v w : V), @inner ℝ V _ (R v) w = @inner ℝ V _ v (R w))
    (u w : V) (hu : R u = u) (hw : R w = 0) :
    ‖w‖ ≤ ‖u - w‖ := by
  -- Apply reynolds_best_approximation with v := w, w := u
  -- This gives ‖w - Rw‖ ≤ ‖w - u‖, i.e. ‖w‖ ≤ ‖w - u‖ = ‖u - w‖
  have h : ‖w - R w‖ ≤ ‖w - u‖ := reynolds_best_approximation R hIdem hSA w u hu
  rw [hw, sub_zero] at h
  linarith [norm_sub_rev w u]

/-- **Beyond-capacity MSE.** The Reynolds projection achieves the minimum
    error among all fixed-point approximations to w, and that minimum
    equals ‖w‖ — the full norm of the target.
    In other words: R maps w to 0, and 0 is the closest fixed point. -/
theorem beyond_capacity_optimal
    (R : V →ₗ[ℝ] V)
    (w : V) (hw : R w = 0) :
    ‖w - R w‖ = ‖w‖ := by
  rw [hw, sub_zero]

end UniversalImpossibility
