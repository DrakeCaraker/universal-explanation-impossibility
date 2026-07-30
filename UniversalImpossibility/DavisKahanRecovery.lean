/-
  DavisKahanRecovery.lean — A rigorous spectral-perturbation fragment toward
  Davis–Kahan / the paper's "feature-geometry capacity bound".

  ─────────────────────────────────────────────────────────────────────────
  WHAT IS PROVED (fully, `sorry`-free, Lean core axioms only)
  ─────────────────────────────────────────────────────────────────────────
  Weyl's perturbation inequality for the EXTREME eigenvalues of a self-adjoint
  operator on a finite-dimensional real/complex inner-product space.

  For self-adjoint continuous linear operators `A`, `Δ` on a nontrivial
  finite-dimensional inner-product space `H` over `𝕜 ∈ {ℝ, ℂ}`, writing

      topRayleigh T = ⨆ x ≠ 0, re⟪T x, x⟫ / ‖x‖²   (the largest eigenvalue)
      botRayleigh T = ⨅ x ≠ 0, re⟪T x, x⟫ / ‖x‖²   (the smallest eigenvalue)

  we prove:
    * `hasEigenvalue_topRayleigh` / `hasEigenvalue_botRayleigh`
        `topRayleigh A` and `botRayleigh A` are genuine eigenvalues of `A`
        (largest and smallest), via Mathlib's Rayleigh-quotient spectral theory.
    * `abs_topRayleigh_sub_le` / `abs_botRayleigh_sub_le`
        |λ_max(A+Δ) − λ_max(A)| ≤ ‖Δ‖   and   |λ_min(A+Δ) − λ_min(A)| ≤ ‖Δ‖.
    * `weyl_extreme_eigenvalues`
        the combined statement (both extreme eigenvalues are eigenvalues and
        move by at most the operator norm of the perturbation).
    * `abs_rayleighQuotient_perturb`
        the pointwise ingredient |R_{A+Δ}(x) − R_A(x)| ≤ ‖Δ‖ used throughout.

  This is exactly the `k = 1` (extreme-eigenvalue) case of Weyl's inequality —
  a genuine, standard ingredient of any Davis–Kahan argument (Weyl locates the
  perturbed spectrum before the sin-θ bound controls the eigenspace rotation).

  ─────────────────────────────────────────────────────────────────────────
  WHAT IS **NOT** PROVED (honest scope — deliberately not claimed)
  ─────────────────────────────────────────────────────────────────────────
    * The sin-θ subspace bound itself (angle between top-k eigenspaces of `A`
      and `A+Δ` ≤ ‖Δ‖ / gap). This is the actual Davis–Kahan theorem and is
      NOT formalized here.
    * Weyl's inequality for INTERIOR eigenvalues λ_i (all i). That needs the
      Courant–Fischer min-max characterisation, which is absent from Mathlib
      at this revision; only the two extreme eigenvalues are handled.
    * Any statement about movement of the gapped spectral projector.

  The bound `topRayleigh (S+F) ≤ topRayleigh S + ‖F‖` (and its `bot`/`inf`
  analogue) is proved for ARBITRARY continuous linear maps `S`, `F`
  (self-adjointness is only used to identify the extreme Rayleigh values with
  actual eigenvalues).

  Zero new axioms; no `sorry`. Builds against Mathlib's
  `Analysis.InnerProductSpace.Rayleigh` and `.Adjoint`.
-/

import Mathlib.Analysis.InnerProductSpace.Rayleigh
import Mathlib.Analysis.InnerProductSpace.Adjoint

set_option autoImplicit false

namespace UniversalImpossibility.DavisKahan

variable {𝕜 : Type*} [RCLike 𝕜]
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace 𝕜 H]

/-- The largest Rayleigh value `⨆ x ≠ 0, re⟪T x, x⟫ / ‖x‖²`.
For a self-adjoint `T` on a nontrivial finite-dimensional space this is the
largest eigenvalue (see `hasEigenvalue_topRayleigh`). -/
noncomputable def topRayleigh (T : H →L[𝕜] H) : ℝ :=
  ⨆ x : {x : H // x ≠ 0}, T.rayleighQuotient (x : H)

/-- The smallest Rayleigh value `⨅ x ≠ 0, re⟪T x, x⟫ / ‖x‖²`.
For a self-adjoint `T` on a nontrivial finite-dimensional space this is the
smallest eigenvalue (see `hasEigenvalue_botRayleigh`). -/
noncomputable def botRayleigh (T : H →L[𝕜] H) : ℝ :=
  ⨅ x : {x : H // x ≠ 0}, T.rayleighQuotient (x : H)

/-! ### Boundedness of the Rayleigh range

The Rayleigh quotient of any continuous linear map is bounded in absolute value
by the operator norm (`ContinuousLinearMap.rayleighQuotient_le_norm`), hence its
range over nonzero vectors is bounded above and below. -/

theorem bddAbove_range (T : H →L[𝕜] H) :
    BddAbove (Set.range fun x : {x : H // x ≠ 0} => T.rayleighQuotient (x : H)) := by
  refine ⟨‖T‖, ?_⟩
  intro a ha
  obtain ⟨x, rfl⟩ := ha
  exact (le_abs_self _).trans (T.rayleighQuotient_le_norm _)

theorem bddBelow_range (T : H →L[𝕜] H) :
    BddBelow (Set.range fun x : {x : H // x ≠ 0} => T.rayleighQuotient (x : H)) := by
  refine ⟨-‖T‖, ?_⟩
  intro a ha
  obtain ⟨x, rfl⟩ := ha
  exact (abs_le.mp (T.rayleighQuotient_le_norm _)).1

/-- Each Rayleigh quotient is `≤` the top Rayleigh value. -/
theorem rayleighQuotient_le_topRayleigh (T : H →L[𝕜] H) (x : {x : H // x ≠ 0}) :
    T.rayleighQuotient (x : H) ≤ topRayleigh T :=
  le_ciSup (bddAbove_range T) x

/-- The bottom Rayleigh value is `≤` each Rayleigh quotient. -/
theorem botRayleigh_le_rayleighQuotient (T : H →L[𝕜] H) (x : {x : H // x ≠ 0}) :
    botRayleigh T ≤ T.rayleighQuotient (x : H) :=
  ciInf_le (bddBelow_range T) x

/-! ### Pointwise perturbation of the Rayleigh quotient

The key elementary estimate: adding `Δ` moves the Rayleigh quotient at any
fixed vector by at most `‖Δ‖`. Holds for arbitrary continuous linear maps. -/

theorem abs_rayleighQuotient_perturb (A Δ : H →L[𝕜] H) (x : H) :
    |(A + Δ).rayleighQuotient x - A.rayleighQuotient x| ≤ ‖Δ‖ := by
  rw [ContinuousLinearMap.rayleighQuotient_add, add_sub_cancel_left]
  exact Δ.rayleighQuotient_le_norm x

/-! ### Weyl bound for the largest eigenvalue -/

/-- One-sided perturbation bound for the top Rayleigh value. Holds for arbitrary
continuous linear maps `S`, `F` (no self-adjointness needed). -/
theorem topRayleigh_add_le [Nontrivial H] (S F : H →L[𝕜] H) :
    topRayleigh (S + F) ≤ topRayleigh S + ‖F‖ := by
  obtain ⟨x0, hx0⟩ := exists_ne (0 : H)
  haveI : Nonempty {x : H // x ≠ 0} := ⟨⟨x0, hx0⟩⟩
  apply ciSup_le
  intro x
  rw [ContinuousLinearMap.rayleighQuotient_add]
  have h2 := rayleighQuotient_le_topRayleigh S x
  have h3 : F.rayleighQuotient (x : H) ≤ ‖F‖ := (le_abs_self _).trans (F.rayleighQuotient_le_norm _)
  linarith

/-- **Weyl's inequality, largest eigenvalue.** The top Rayleigh value (largest
eigenvalue, for self-adjoint operators) moves by at most `‖Δ‖`. Stated for
arbitrary continuous linear maps. -/
theorem abs_topRayleigh_sub_le [Nontrivial H] (A Δ : H →L[𝕜] H) :
    |topRayleigh (A + Δ) - topRayleigh A| ≤ ‖Δ‖ := by
  rw [abs_sub_le_iff]
  refine ⟨?_, ?_⟩
  · have h := topRayleigh_add_le A Δ
    linarith
  · have h := topRayleigh_add_le (A + Δ) (-Δ)
    rw [show A + Δ + -Δ = A by abel, norm_neg] at h
    linarith

/-! ### Weyl bound for the smallest eigenvalue -/

/-- One-sided perturbation bound for the bottom Rayleigh value. Holds for
arbitrary continuous linear maps `S`, `F`. -/
theorem le_botRayleigh_add [Nontrivial H] (S F : H →L[𝕜] H) :
    botRayleigh S - ‖F‖ ≤ botRayleigh (S + F) := by
  obtain ⟨x0, hx0⟩ := exists_ne (0 : H)
  haveI : Nonempty {x : H // x ≠ 0} := ⟨⟨x0, hx0⟩⟩
  apply le_ciInf
  intro x
  rw [ContinuousLinearMap.rayleighQuotient_add]
  have h2 := botRayleigh_le_rayleighQuotient S x
  have h3 : -‖F‖ ≤ F.rayleighQuotient (x : H) := (abs_le.mp (F.rayleighQuotient_le_norm _)).1
  linarith

/-- **Weyl's inequality, smallest eigenvalue.** The bottom Rayleigh value
(smallest eigenvalue, for self-adjoint operators) moves by at most `‖Δ‖`. -/
theorem abs_botRayleigh_sub_le [Nontrivial H] (A Δ : H →L[𝕜] H) :
    |botRayleigh (A + Δ) - botRayleigh A| ≤ ‖Δ‖ := by
  rw [abs_sub_le_iff]
  refine ⟨?_, ?_⟩
  · have h := le_botRayleigh_add (A + Δ) (-Δ)
    rw [show A + Δ + -Δ = A by abel, norm_neg] at h
    linarith
  · have h := le_botRayleigh_add A Δ
    linarith

/-! ### Identifying the extreme Rayleigh values with actual eigenvalues

Via Mathlib's `LinearMap.IsSymmetric.hasEigenvalue_iSup/iInf_of_finiteDimensional`,
in finite dimension the extreme Rayleigh values are genuine eigenvalues of a
self-adjoint operator. This is what turns the Rayleigh bounds above into
statements about the actual spectrum. -/

/-- The top Rayleigh value of a self-adjoint operator on a nontrivial
finite-dimensional space is its largest eigenvalue. -/
theorem hasEigenvalue_topRayleigh [FiniteDimensional 𝕜 H] [CompleteSpace H] [Nontrivial H]
    {A : H →L[𝕜] H} (hA : IsSelfAdjoint A) :
    Module.End.HasEigenvalue (A : H →ₗ[𝕜] H) (topRayleigh A) :=
  hA.isSymmetric.hasEigenvalue_iSup_of_finiteDimensional

/-- The bottom Rayleigh value of a self-adjoint operator on a nontrivial
finite-dimensional space is its smallest eigenvalue. -/
theorem hasEigenvalue_botRayleigh [FiniteDimensional 𝕜 H] [CompleteSpace H] [Nontrivial H]
    {A : H →L[𝕜] H} (hA : IsSelfAdjoint A) :
    Module.End.HasEigenvalue (A : H →ₗ[𝕜] H) (botRayleigh A) :=
  hA.isSymmetric.hasEigenvalue_iInf_of_finiteDimensional

/-! ### Combined statement -/

/-- **Weyl perturbation bound for the extreme eigenvalues.**

For self-adjoint operators `A`, `Δ` on a nontrivial finite-dimensional
inner-product space over `ℝ` or `ℂ`:
the largest and smallest eigenvalues of `A` and of `A + Δ` are genuine
eigenvalues, and each moves by at most `‖Δ‖`.

This is the extreme-eigenvalue (`k = 1`) case of Weyl's inequality, the standard
spectrum-localisation ingredient of a Davis–Kahan sin-θ argument. The sin-θ
subspace bound itself and Weyl for interior eigenvalues are NOT proved here
(see the file header). -/
theorem weyl_extreme_eigenvalues [FiniteDimensional 𝕜 H] [CompleteSpace H] [Nontrivial H]
    {A Δ : H →L[𝕜] H} (hA : IsSelfAdjoint A) (hΔ : IsSelfAdjoint Δ) :
    Module.End.HasEigenvalue (A : H →ₗ[𝕜] H) (topRayleigh A) ∧
      Module.End.HasEigenvalue ((A + Δ : H →L[𝕜] H) : H →ₗ[𝕜] H) (topRayleigh (A + Δ)) ∧
      |topRayleigh (A + Δ) - topRayleigh A| ≤ ‖Δ‖ ∧
      Module.End.HasEigenvalue (A : H →ₗ[𝕜] H) (botRayleigh A) ∧
      Module.End.HasEigenvalue ((A + Δ : H →L[𝕜] H) : H →ₗ[𝕜] H) (botRayleigh (A + Δ)) ∧
      |botRayleigh (A + Δ) - botRayleigh A| ≤ ‖Δ‖ :=
  ⟨hasEigenvalue_topRayleigh hA, hasEigenvalue_topRayleigh (hA.add hΔ),
    abs_topRayleigh_sub_le A Δ, hasEigenvalue_botRayleigh hA,
    hasEigenvalue_botRayleigh (hA.add hΔ), abs_botRayleigh_sub_le A Δ⟩

end UniversalImpossibility.DavisKahan
