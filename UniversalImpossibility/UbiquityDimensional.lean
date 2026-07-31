/-
  UbiquityDimensional.lean — the linear/infinitesimal core of the ubiquity argument.

  The ubiquity claim (Part I of the monograph) is that the Rashomon property — and
  hence the Explanation Impossibility — holds generically, because a configuration
  space of higher dimension than the observable space is underspecified. The prior
  Lean content (`generic_underspecification`) captured only the arithmetic proxy
  `n > m → n - m > 0`; the mathematically substantive step — that underspecification
  genuinely *forces* distinct-but-observationally-equivalent configurations to exist
  — was argued informally via the preimage theorem of differential topology.

  This file discharges the core of that step by machine, via rank–nullity:

    * `exists_kernel_direction` / `exists_observational_collision`: if the (linear)
      observation map has strictly more configuration dimensions than observable
      dimensions, then from every configuration there is a genuinely distinct one
      that observes identically. The Rashomon precondition is now derived, not assumed.
    * `finrank_ker_ge`: the naive count `n - m` is a genuine lower bound on the true
      fibre dimension (the kernel), not merely an arithmetic proxy.
    * `clm_exists_kernel_direction` / `fderiv_exists_kernel_direction`: the
      infinitesimal core of the preimage theorem — at any point, a map from a
      higher-dimensional configuration space has a nonzero tangent direction along
      which the observation is stationary to first order (its derivative kills it).
    * `underspecified_impossibility`: assembles the dimensional fact with the existing
      non-degeneracy bridge and the core `explanation_impossibility` — the colliding
      pair is produced by dimension; the residual hypothesis is exactly non-degeneracy
      of the explanation.

  What remains genuinely informal (and is stated as such in the monograph): the
  globalisation from "the derivative has a flat direction at every point" to "the
  fibre is a positive-measure submanifold of genuinely distinct models" (constant-rank
  / implicit-function theorem + a non-degeneracy condition), and the "generic value"
  refinement (Sard's theorem). Mathlib has the implicit-function machinery but not the
  assembled positive-measure-submanifold statement; that is a separate project.

  Axiom-clean (Lean core only: propext, Classical.choice, Quot.sound).
-/
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Analysis.Calculus.FDeriv.Basic
import UniversalImpossibility.Ubiquity

open Module

namespace UniversalImpossibility.Ubiquity

-- ============================================================================
-- §1  Linear core: underspecification forces a genuine observational collision
-- ============================================================================

section Linear
variable {𝕜 V W : Type*} [DivisionRing 𝕜]
  [AddCommGroup V] [Module 𝕜 V] [FiniteDimensional 𝕜 V]
  [AddCommGroup W] [Module 𝕜 W] [FiniteDimensional 𝕜 W]

/-- Underspecification produces a genuine flat direction: a nonzero configuration
    change the observation cannot see. -/
theorem exists_kernel_direction (f : V →ₗ[𝕜] W)
    (h : finrank 𝕜 W < finrank 𝕜 V) :
    ∃ u : V, u ≠ 0 ∧ f u = 0 := by
  obtain ⟨u, hu, hune⟩ := Submodule.exists_mem_ne_zero_of_ne_bot (f.ker_ne_bot_of_finrank_lt h)
  exact ⟨u, hune, LinearMap.mem_ker.mp hu⟩

/-- **The substantive form of underspecification.** If the observation map has
    strictly more configuration dimensions than observable dimensions, then from
    every configuration there is a genuinely *distinct* one that observes
    identically. This is the Rashomon precondition, derived rather than assumed —
    the content the arithmetic `n - m > 0` was previously standing in for. -/
theorem exists_observational_collision (f : V →ₗ[𝕜] W)
    (h : finrank 𝕜 W < finrank 𝕜 V) (θ : V) :
    ∃ θ' : V, θ' ≠ θ ∧ f θ' = f θ := by
  obtain ⟨u, hune, hu0⟩ := exists_kernel_direction f h
  refine ⟨θ + u, ?_, ?_⟩
  · intro he
    exact hune (add_left_cancel (a := θ) (by rw [add_zero]; exact he))
  · rw [map_add, hu0, add_zero]

/-- The naive count `n - m` is a genuine lower bound on the true fibre dimension
    (the kernel of the observation map), not merely an arithmetic proxy. -/
theorem finrank_ker_ge (f : V →ₗ[𝕜] W) :
    finrank 𝕜 V - finrank 𝕜 W ≤ finrank 𝕜 (LinearMap.ker f) := by
  have hrn := f.finrank_range_add_finrank_ker
  have hle : finrank 𝕜 (LinearMap.range f) ≤ finrank 𝕜 W := (LinearMap.range f).finrank_le
  omega

end Linear

-- ============================================================================
-- §2  Infinitesimal core: the honest heart of the preimage-theorem step
-- ============================================================================

section Differential
variable {𝕜 E F : Type*} [NontriviallyNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F] [FiniteDimensional 𝕜 F]

/-- The kernel-direction fact for continuous linear maps (e.g. derivatives). -/
theorem clm_exists_kernel_direction (f' : E →L[𝕜] F)
    (h : finrank 𝕜 F < finrank 𝕜 E) :
    ∃ u : E, u ≠ 0 ∧ f' u = 0 := by
  obtain ⟨u, hune, hu0⟩ := exists_kernel_direction f'.toLinearMap h
  exact ⟨u, hune, by simpa using hu0⟩

/-- **The infinitesimal core of the preimage theorem.** At any point, a map from a
    higher-dimensional configuration space has a nonzero tangent direction along
    which the observation is stationary to first order (its Fréchet derivative kills
    it). This is the machine-checked heart of the differential-topology step in the
    ubiquity argument; the globalisation to a positive-measure fibre of genuinely
    distinct models is the residual informal step (see the file header). -/
theorem fderiv_exists_kernel_direction (f : E → F) (x : E)
    (h : finrank 𝕜 F < finrank 𝕜 E) :
    ∃ u : E, u ≠ 0 ∧ (fderiv 𝕜 f x) u = 0 :=
  clm_exists_kernel_direction (fderiv 𝕜 f x) h

end Differential

-- ============================================================================
-- §3  Assembled: dimension → collision → impossibility
-- ============================================================================

section Bridge
variable {V W H : Type} {𝕜 : Type*} [DivisionRing 𝕜]
  [AddCommGroup V] [Module 𝕜 V] [FiniteDimensional 𝕜 V]
  [AddCommGroup W] [Module 𝕜 W] [FiniteDimensional 𝕜 W]

/-- **Dimensional ubiquity, assembled and wired to the impossibility.** If the
    observation map is linear with strictly more configuration dimensions than
    observable dimensions, then a genuinely distinct configuration observing
    identically *exists* (produced by §1, no longer assumed); given a non-degenerate
    explanation — one that reports incompatible explanations on that fibre, a property
    of `explain`, not of the dimensions — no explanation can be faithful, stable, and
    decisive. This closes the arithmetic-proxy gap in the ubiquity argument for the
    linear case: the colliding pair is derived; only explanation non-degeneracy
    remains a hypothesis. -/
theorem underspecified_impossibility
    (obs : V →ₗ[𝕜] W) (explain : V → H) (incomp : H → H → Prop)
    (hdim : finrank 𝕜 W < finrank 𝕜 V) (θ : V)
    (hnd : ∀ θ', θ' ≠ θ → obs θ' = obs θ → incomp (explain θ') (explain θ))
    (E : V → H)
    (hf : ∀ t, ¬ incomp (E t) (explain t))
    (hs : ∀ a b, obs a = obs b → E a = E b)
    (hd : ∀ t h, incomp (explain t) h → incomp (E t) h) :
    False := by
  obtain ⟨θ', hne, hobs⟩ := exists_observational_collision obs hdim θ
  exact fiber_nondegeneracy_implies_impossibility
    (fun t => obs t) explain incomp θ' θ hobs (hnd θ' hne hobs) E hf hs hd

end Bridge

end UniversalImpossibility.Ubiquity
