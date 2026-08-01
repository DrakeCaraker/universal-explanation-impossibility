/-
  UbiquityInfiniteDimensional.lean — the ubiquity argument on infinite-dimensional
  configuration spaces.

  `UbiquityDimensional.lean` (§§1–5) establishes the dimensional ubiquity ladder under
  `[FiniteDimensional ℝ E]` with `finrank ℝ F < finrank ℝ E`. The physically meaningful
  regime it leaves open is the one where the configuration space is a *function space* —
  an infinite-dimensional Banach space (fields, potentials, policies, weight functions)
  — observed through *finitely many measurements* (F stays finite-dimensional). This
  file extends every rung of the ladder that genuinely generalizes to that regime, and
  documents honestly (§6) the one rung that does not — because it is mathematically
  false there, not merely unformalised.

  Dimension hypothesis. `finrank` degenerates to `0` in infinite dimension, so the
  finite-dimensional comparison `finrank F < finrank E` is replaced by the cardinal-rank
  comparison

      (finrank 𝕜 F : Cardinal) < Module.rank 𝕜 E                                (gap)

  with `F` finite-dimensional. Since `(finrank 𝕜 F : Cardinal) = Module.rank 𝕜 F` for
  finite-dimensional `F`, this *is* the comparison `Module.rank F < Module.rank E`,
  written so as to be universe-polymorphic (a natural number casts into `Cardinal` at
  any universe; no `Cardinal.lift` noise). §0 proves the bookkeeping equivalences:
  (gap) ↔ `Module.rank 𝕜 F < Module.rank 𝕜 E` (same universe), (gap) ← the old
  `finrank F < finrank E` when `E` is finite-dimensional — so every result here
  *strictly generalizes* its finite-dimensional counterpart — and (gap) ← `¬
  FiniteDimensional 𝕜 E`, the clean sufficient condition in the function-space regime.

  The rungs, extended:

    * §1 Linear: `exists_kernel_direction_infinite` / `exists_observational_collision_infinite`
      — a linear observation map under (gap) has nontrivial kernel; from every
      configuration there is a genuinely distinct one observing identically. The
      cardinal rank–nullity strengthening `rank_ker_eq_rank_of_infiniteDimensional`:
      over an infinite-dimensional configuration space with finitely many observables,
      the kernel is as large as the whole space — underspecification is not merely
      present but total.
    * §2 Infinitesimal: `fderiv_exists_kernel_direction_infinite` — at every point the
      derivative has a nonzero flat direction.
    * §3 Bridge: `underspecified_impossibility_infinite` and the headline
      `function_space_impossibility` — infinite-dimensional configurations, finitely
      many measurements, non-degenerate explanation ⟹ no faithful+stable+decisive
      explanation.
    * §4 Local smooth (the substantive rung): `regular_fiber_not_isolated_infinite` —
      at a regular point of a map from a Banach space to a finite-dimensional space,
      the exact fibre is not isolated. The implicit-function machinery goes through
      because the kernel of a surjective continuous linear map onto finite-dimensional
      `F` is closed with finite codimension, hence closed-complemented: Mathlib's
      `ContinuousLinearMap.ker_closedComplemented_of_finiteDimensional_range`, already
      wired into `HasStrictFDerivAt.implicitToOpenPartialHomeomorph`, whose only
      hypotheses on the domain are `[CompleteSpace E]`.
    * §5 Genericity: `regular_value_fiber_not_isolated_infinite`,
      `submersion_fiber_not_isolated_infinite`, the conditional
      `generic_ubiquity_of_sard_infinite`, and two *unconditional* capstones —
      `generic_ubiquity_of_continuousLinearMap_infinite` and
      `generic_ubiquity_of_submersion_infinite` — since `sardProperty_of_continuousLinearMap`
      and `sardProperty_of_submersion` in `UbiquityDimensional.lean` never used the
      finite-dimensionality of `E` in the first place.

  What must NOT be claimed (§6): the full Morse–Sard theorem
  (`sardProperty_of_contDiff`, `MorseSard.lean`) does not extend, because it is false
  in this regime — see the scoping note in §6.

  Axiom-clean (Lean core only: propext, Classical.choice, Quot.sound).
-/
import Mathlib.LinearAlgebra.Dimension.DivisionRing
import Mathlib.LinearAlgebra.Dimension.Free
import Mathlib.Analysis.Calculus.ContDiff.Basic
import UniversalImpossibility.UbiquityDimensional

open Module Filter Topology MeasureTheory

namespace UniversalImpossibility.Ubiquity

-- ============================================================================
-- §0  The dimension-gap hypothesis and its equivalences
-- ============================================================================

section DimensionGap
variable {𝕜 : Type*} [DivisionRing 𝕜]

/-- In a common universe, the gap hypothesis `(finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V`
    is *literally* the cardinal-rank comparison `Module.rank 𝕜 W < Module.rank 𝕜 V`
    (for finite-dimensional observable space `W`). The `finrank`-cast form used
    throughout this file is just its universe-polymorphic spelling. -/
theorem dim_gap_iff_rank_lt {V W : Type u}
    [AddCommGroup V] [Module 𝕜 V] [AddCommGroup W] [Module 𝕜 W] [FiniteDimensional 𝕜 W] :
    ((finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V) ↔ Module.rank 𝕜 W < Module.rank 𝕜 V := by
  rw [Module.finrank_eq_rank 𝕜 W]

/-- Backward compatibility: when the configuration space is finite-dimensional, the
    old hypothesis `finrank W < finrank V` implies the gap hypothesis — every theorem
    in this file strictly generalizes its finite-dimensional counterpart. -/
theorem dim_gap_of_finrank_lt {V W : Type*}
    [AddCommGroup V] [Module 𝕜 V] [FiniteDimensional 𝕜 V] [AddCommGroup W] [Module 𝕜 W]
    (h : finrank 𝕜 W < finrank 𝕜 V) :
    (finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V := by
  rw [← Module.finrank_eq_rank 𝕜 V]
  exact_mod_cast h

/-- The function-space regime: an infinite-dimensional configuration space out-ranks
    any finite observable count. This is the clean sufficient condition for the gap
    hypothesis when `E` is a genuine function space. -/
theorem dim_gap_of_infiniteDimensional {V : Type*} [AddCommGroup V] [Module 𝕜 V]
    (hV : ¬ FiniteDimensional 𝕜 V) (n : ℕ) :
    (n : Cardinal) < Module.rank 𝕜 V := by
  have haleph : Cardinal.aleph0 ≤ Module.rank 𝕜 V := by
    by_contra hlt
    exact hV (Module.rank_lt_aleph0_iff.mp (not_le.mp hlt))
  exact lt_of_lt_of_le Cardinal.natCast_lt_aleph0 haleph

end DimensionGap

-- ============================================================================
-- §1  Linear core: infinite-dimensional underspecification forces collisions
-- ============================================================================

section Linear
variable {𝕜 V W : Type*} [DivisionRing 𝕜]
  [AddCommGroup V] [Module 𝕜 V]
  [AddCommGroup W] [Module 𝕜 W] [FiniteDimensional 𝕜 W]

/-- Under the gap hypothesis, the observation map has nontrivial kernel — no
    finite-dimensionality of the configuration space required. (If the kernel were
    trivial the map would embed `V` into the finite-dimensional `W`, forcing
    `V` finite-dimensional with `finrank V ≤ finrank W`, contradicting the gap.) -/
theorem ker_ne_bot_of_rank_lt (f : V →ₗ[𝕜] W)
    (h : (finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V) :
    LinearMap.ker f ≠ ⊥ := by
  intro hbot
  have hinj : Function.Injective f := LinearMap.ker_eq_bot.mp hbot
  haveI : FiniteDimensional 𝕜 V := FiniteDimensional.of_injective f hinj
  have hle : finrank 𝕜 V ≤ finrank 𝕜 W := LinearMap.finrank_le_finrank_of_injective hinj
  rw [← Module.finrank_eq_rank 𝕜 V] at h
  have h' : finrank 𝕜 W < finrank 𝕜 V := by exact_mod_cast h
  omega

/-- Underspecification produces a genuine flat direction, in arbitrary (possibly
    infinite) configuration dimension: a nonzero configuration change the observation
    cannot see. Infinite-dimensional generalization of `exists_kernel_direction`. -/
theorem exists_kernel_direction_infinite (f : V →ₗ[𝕜] W)
    (h : (finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V) :
    ∃ u : V, u ≠ 0 ∧ f u = 0 := by
  obtain ⟨u, hu, hune⟩ := Submodule.exists_mem_ne_zero_of_ne_bot (ker_ne_bot_of_rank_lt f h)
  exact ⟨u, hune, LinearMap.mem_ker.mp hu⟩

/-- **The substantive form of underspecification, infinite-dimensional case.** If the
    (possibly infinite-dimensional) configuration space out-ranks the finite-dimensional
    observable space, then from every configuration there is a genuinely *distinct* one
    that observes identically. The Rashomon precondition, derived rather than assumed,
    now valid for function-space configuration spaces. -/
theorem exists_observational_collision_infinite (f : V →ₗ[𝕜] W)
    (h : (finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V) (θ : V) :
    ∃ θ' : V, θ' ≠ θ ∧ f θ' = f θ := by
  obtain ⟨u, hune, hu0⟩ := exists_kernel_direction_infinite f h
  refine ⟨θ + u, ?_, ?_⟩
  · intro he
    exact hune (add_left_cancel (a := θ) (by rw [add_zero]; exact he))
  · rw [map_add, hu0, add_zero]

/-- **Cardinal rank–nullity: underspecification is total.** Over an
    infinite-dimensional configuration space with finitely many observables, the kernel
    of the observation map is as large as the entire configuration space — the
    infinite-dimensional strengthening of `finrank_ker_ge` (which degenerates to
    `0 ≤ …` here). Finitely many measurements pin down *nothing*, cardinally speaking:
    the unobservable directions have full rank. -/
theorem rank_ker_eq_rank_of_infiniteDimensional (f : V →ₗ[𝕜] W)
    (hV : ¬ FiniteDimensional 𝕜 V) :
    Module.rank 𝕜 (LinearMap.ker f) = Module.rank 𝕜 V := by
  -- the quotient V ⧸ ker f embeds in W via the first isomorphism theorem, so it is
  -- finite-dimensional …
  haveI hq : Module.Finite 𝕜 (V ⧸ LinearMap.ker f) :=
    Module.Finite.equiv f.quotKerEquivRange.symm
  have hrn := rank_quotient_add_rank_of_divisionRing (LinearMap.ker f)
  have hqlt : Module.rank 𝕜 (V ⧸ LinearMap.ker f) < Cardinal.aleph0 :=
    Module.rank_lt_aleph0 𝕜 _
  -- … while V itself is infinite-dimensional …
  have hVinf : Cardinal.aleph0 ≤ Module.rank 𝕜 V := by
    by_contra hlt
    exact hV (Module.rank_lt_aleph0_iff.mp (not_le.mp hlt))
  -- … so the kernel must be infinite-dimensional …
  have hkinf : Cardinal.aleph0 ≤ Module.rank 𝕜 (LinearMap.ker f) := by
    by_contra hlt
    rw [← hrn] at hVinf
    exact absurd hVinf (not_le.mpr (Cardinal.add_lt_aleph0 hqlt (not_le.mp hlt)))
  -- … and infinite-cardinal arithmetic absorbs the finite quotient.
  calc Module.rank 𝕜 (LinearMap.ker f)
      = Module.rank 𝕜 (V ⧸ LinearMap.ker f) + Module.rank 𝕜 (LinearMap.ker f) :=
        (Cardinal.add_eq_right hkinf (hqlt.le.trans hkinf)).symm
    _ = Module.rank 𝕜 V := hrn

end Linear

-- ============================================================================
-- §2  Infinitesimal core: flat derivative directions in infinite dimension
-- ============================================================================

section Differential
variable {𝕜 E F : Type*} [NontriviallyNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F] [FiniteDimensional 𝕜 F]

/-- The kernel-direction fact for continuous linear maps (e.g. derivatives), with the
    configuration space possibly infinite-dimensional. -/
theorem clm_exists_kernel_direction_infinite (f' : E →L[𝕜] F)
    (h : (finrank 𝕜 F : Cardinal) < Module.rank 𝕜 E) :
    ∃ u : E, u ≠ 0 ∧ f' u = 0 := by
  obtain ⟨u, hune, hu0⟩ := exists_kernel_direction_infinite f'.toLinearMap h
  exact ⟨u, hune, by simpa using hu0⟩

/-- **The infinitesimal core of the preimage theorem, infinite-dimensional case.** At
    any point, a map from a configuration space that out-ranks its finite-dimensional
    observable space has a nonzero tangent direction along which the observation is
    stationary to first order. No completeness or finite-dimensionality of `E` is
    needed at the infinitesimal level. -/
theorem fderiv_exists_kernel_direction_infinite (f : E → F) (x : E)
    (h : (finrank 𝕜 F : Cardinal) < Module.rank 𝕜 E) :
    ∃ u : E, u ≠ 0 ∧ (fderiv 𝕜 f x) u = 0 :=
  clm_exists_kernel_direction_infinite (fderiv 𝕜 f x) h

end Differential

-- ============================================================================
-- §3  Assembled: infinite dimension → collision → impossibility
-- ============================================================================

section Bridge
variable {V W H : Type} {𝕜 : Type*} [DivisionRing 𝕜]
  [AddCommGroup V] [Module 𝕜 V]
  [AddCommGroup W] [Module 𝕜 W] [FiniteDimensional 𝕜 W]

/-- **Dimensional ubiquity, assembled, for possibly infinite-dimensional configuration
    spaces.** If the linear observation map's configuration space out-ranks its
    finite-dimensional observable space, a genuinely distinct configuration observing
    identically exists (produced by §1); given a non-degenerate explanation, no
    explanation can be faithful, stable, and decisive. Infinite-dimensional
    generalization of `underspecified_impossibility`. -/
theorem underspecified_impossibility_infinite
    (obs : V →ₗ[𝕜] W) (explain : V → H) (incomp : H → H → Prop)
    (hdim : (finrank 𝕜 W : Cardinal) < Module.rank 𝕜 V) (θ : V)
    (hnd : ∀ θ', θ' ≠ θ → obs θ' = obs θ → incomp (explain θ') (explain θ))
    (E : V → H)
    (hf : ∀ t, ¬ incomp (E t) (explain t))
    (hs : ∀ a b, obs a = obs b → E a = E b)
    (hd : ∀ t h, incomp (explain t) h → incomp (E t) h) :
    False := by
  obtain ⟨θ', hne, hobs⟩ := exists_observational_collision_infinite obs hdim θ
  exact fiber_nondegeneracy_implies_impossibility
    (fun t => obs t) explain incomp θ' θ hobs (hnd θ' hne hobs) E hf hs hd

/-- **The function-space impossibility.** The headline form for the physically
    meaningful regime: the configuration space is an infinite-dimensional function
    space, the observation consists of finitely many (linear) measurements, and the
    explanation is non-degenerate on the observation fibre — then no explanation can
    be faithful, stable, and decisive. The dimension gap here is automatic: infinitely
    many configuration degrees of freedom always out-rank finitely many measurements. -/
theorem function_space_impossibility
    (obs : V →ₗ[𝕜] W) (explain : V → H) (incomp : H → H → Prop)
    (hV : ¬ FiniteDimensional 𝕜 V) (θ : V)
    (hnd : ∀ θ', θ' ≠ θ → obs θ' = obs θ → incomp (explain θ') (explain θ))
    (E : V → H)
    (hf : ∀ t, ¬ incomp (E t) (explain t))
    (hs : ∀ a b, obs a = obs b → E a = E b)
    (hd : ∀ t h, incomp (explain t) h → incomp (E t) h) :
    False :=
  underspecified_impossibility_infinite obs explain incomp
    (dim_gap_of_infiniteDimensional hV _) θ hnd E hf hs hd

end Bridge

-- ============================================================================
-- §4  Local smooth globalisation on Banach configuration spaces
-- ============================================================================

/-
  The substantive rung. Mathlib's implicit function theorem for a map into a
  finite-dimensional codomain (`HasStrictFDerivAt.implicitToOpenPartialHomeomorph`)
  requires of the domain only completeness: the kernel of the surjective derivative is
  closed of finite codimension, hence closed-complemented
  (`ContinuousLinearMap.ker_closedComplemented_of_finiteDimensional_range` — Hahn–Banach
  supplies the continuous projection onto a finite-dimensional complement). So the §4
  argument of `UbiquityDimensional.lean` goes through verbatim on any Banach
  configuration space once the kernel direction is produced by rank (§1) instead of
  `finrank`.
-/

section SmoothLocal
variable {𝕜 E F : Type*} [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E] [CompleteSpace E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F] [FiniteDimensional 𝕜 F]

/-- **Local smooth globalisation on a Banach configuration space.** At a regular point
    (`f'.range = ⊤`) of a map from a complete normed space that out-ranks its
    finite-dimensional observable space, the *exact* fibre is not isolated: genuinely
    distinct configurations that observe identically exist within any `ε`. This is the
    infinite-dimensional generalization of `regular_fiber_not_isolated`, via the
    implicit function theorem; the closed-complementedness of the derivative's kernel
    (automatic for finite-dimensional codomain) replaces finite-dimensionality of the
    domain. -/
theorem regular_fiber_not_isolated_infinite {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f f' a) (hf' : f'.range = ⊤)
    (hdim : (finrank 𝕜 F : Cardinal) < Module.rank 𝕜 E) (ε : ℝ) (hε : 0 < ε) :
    ∃ x : E, x ≠ a ∧ f x = f a ∧ dist x a < ε := by
  have hker : f'.ker ≠ ⊥ := ker_ne_bot_of_rank_lt f'.toLinearMap hdim
  obtain ⟨b, hb_mem, hb_ne⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  set u : (f'.ker) := ⟨b, hb_mem⟩ with hu_def
  have hu_ne : u ≠ 0 := fun h => hb_ne (congrArg Subtype.val h)
  set g := hf.implicitToOpenPartialHomeomorph f f' hf' with hg
  have hself : g a = (f a, 0) := hf.implicitToOpenPartialHomeomorph_self hf'
  have hsrc : a ∈ g.source := hf.mem_implicitToOpenPartialHomeomorph_source hf'
  have htgt : (f a, (0 : f'.ker)) ∈ g.target := hf.mem_implicitToOpenPartialHomeomorph_target hf'
  have hfst : ∀ x, (g x).fst = f x := fun x => hf.implicitToOpenPartialHomeomorph_fst hf' x
  set k : 𝕜 → F × (f'.ker) := fun t => (f a, t • u) with hk
  have hk0 : k 0 = (f a, 0) := by simp [hk]
  have hk_cont : Continuous k := by rw [hk]; fun_prop
  set γ : 𝕜 → E := fun t => g.symm (k t) with hγ
  have hγ0 : γ 0 = a := by
    have h := g.left_inv hsrc
    simpa [hγ, hk0, hself] using h
  have hsymm_ca : ContinuousAt g.symm (f a, 0) :=
    g.continuousOn_symm.continuousAt (g.open_target.mem_nhds htgt)
  have hγ_ca : ContinuousAt γ 0 := by
    have hca : ContinuousAt g.symm (k 0) := by rw [hk0]; exact hsymm_ca
    simpa [hγ, Function.comp] using hca.comp hk_cont.continuousAt
  have hTgt : ∀ᶠ t in 𝓝 (0 : 𝕜), k t ∈ g.target :=
    hk_cont.continuousAt.eventually_mem (by rw [hk0]; exact g.open_target.mem_nhds htgt)
  have hDist : ∀ᶠ t in 𝓝 (0 : 𝕜), dist (γ t) a < ε := by
    have ht := hγ_ca.tendsto
    rw [hγ0] at ht
    exact (Metric.tendsto_nhds.mp ht) ε hε
  have h1 : ∀ᶠ t in 𝓝[≠] (0 : 𝕜), t ≠ 0 := by
    filter_upwards [eventually_mem_nhdsWithin] with t ht using by simpa using ht
  have hcomb : ∀ᶠ t in 𝓝[≠] (0 : 𝕜), t ≠ 0 ∧ k t ∈ g.target ∧ dist (γ t) a < ε := by
    filter_upwards [h1, hTgt.filter_mono nhdsWithin_le_nhds,
      hDist.filter_mono nhdsWithin_le_nhds] with t ht1 ht2 ht3 using ⟨ht1, ht2, ht3⟩
  obtain ⟨t, ht_ne, ht_tgt, ht_dist⟩ := hcomb.exists
  have hgt : g (γ t) = k t := g.right_inv ht_tgt
  refine ⟨γ t, ?_, ?_, ht_dist⟩
  · intro he
    have hk_eq : k t = (f a, 0) := hgt.symm.trans (by rw [he]; exact hself)
    rw [hk] at hk_eq
    exact smul_ne_zero ht_ne hu_ne ((Prod.mk.injEq _ _ _ _).mp hk_eq).2
  · have hf1 := hfst (γ t)
    rw [hgt, hk] at hf1
    exact hf1.symm

/-- **The smooth ubiquity impossibility on a Banach configuration space, locally.** At
    a regular point of a map from a complete normed configuration space out-ranking its
    finite-dimensional observable space, a genuinely distinct configuration observing
    identically exists on the true curved fibre; given a non-degenerate explanation, no
    explanation can be faithful, stable, and decisive. Infinite-dimensional analogue of
    `smooth_regular_underspecified_impossibility`. -/
theorem smooth_regular_underspecified_impossibility_infinite
    {H : Type*} {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f f' a) (hf' : f'.range = ⊤)
    (hdim : (finrank 𝕜 F : Cardinal) < Module.rank 𝕜 E)
    (explain : E → H) (incomp : H → H → Prop)
    (hnd : ∀ x, x ≠ a → f x = f a → incomp (explain x) (explain a))
    (Efn : E → H)
    (hfaith : ∀ t, ¬ incomp (Efn t) (explain t))
    (hstab : ∀ p q, f p = f q → Efn p = Efn q)
    (hdec : ∀ t h, incomp (explain t) h → incomp (Efn t) h) :
    False := by
  obtain ⟨x, hx_ne, hx_obs, _⟩ := regular_fiber_not_isolated_infinite hf hf' hdim 1 one_pos
  have hinc : incomp (Efn x) (explain a) := hdec x (explain a) (hnd x hx_ne hx_obs)
  have heq : Efn x = Efn a := hstab x a hx_obs
  rw [heq] at hinc
  exact hfaith a hinc

end SmoothLocal

section Functional
variable {𝕜 : Type*} [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜]
  {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]

omit [CompleteSpace 𝕜] in
/-- A nonzero continuous linear functional is a regular (surjective) observation map:
    its range is a nonzero subspace of the scalar field, hence everything. -/
theorem clm_range_eq_top_of_ne_zero (L : E →L[𝕜] 𝕜) (hL : L ≠ 0) :
    L.range = ⊤ := by
  obtain ⟨x, hx⟩ : ∃ x, L x ≠ 0 := by
    by_contra hcon
    exact hL (ContinuousLinearMap.ext fun x => by
      simpa using not_not.mp (not_exists.mp hcon x))
  refine Submodule.eq_top_iff'.mpr fun y => ?_
  exact LinearMap.mem_range.mpr ⟨(y * (L x)⁻¹) • x, by
    simp [smul_eq_mul, mul_assoc, inv_mul_cancel₀ hx]⟩

/-- **Non-vacuity of the Banach-space hypotheses.** Any nonzero continuous linear
    measurement on any infinite-dimensional Banach configuration space satisfies every
    hypothesis of `regular_fiber_not_isolated_infinite`, and its fibres accumulate on
    themselves: from every configuration, genuinely distinct configurations with the
    identical measured value exist within any `ε`. (Hahn–Banach guarantees such
    functionals exist on every nontrivial space, so the regime is populated.) -/
theorem functional_fiber_not_isolated [CompleteSpace E]
    (hE : ¬ FiniteDimensional 𝕜 E) (L : E →L[𝕜] 𝕜) (hL : L ≠ 0)
    (a : E) (ε : ℝ) (hε : 0 < ε) :
    ∃ x : E, x ≠ a ∧ L x = L a ∧ dist x a < ε :=
  regular_fiber_not_isolated_infinite L.hasStrictFDerivAt
    (clm_range_eq_top_of_ne_zero L hL) (dim_gap_of_infiniteDimensional hE _) ε hε

end Functional

-- ============================================================================
-- §5  Genericity of regular values on Banach configuration spaces
-- ============================================================================

/-
  The `IsRegularValue` and `SardProperty` predicates of `UbiquityDimensional.lean` §5
  never used the finite-dimensionality of `E`, so they apply verbatim here; likewise
  `sardProperty_of_continuousLinearMap` and `sardProperty_of_submersion` were already
  proved without it. What genuinely needs re-proving with the weakened hypotheses is
  the fibre analysis, which routes through §4.
-/

section RegularValue
variable {E F : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F]

/-- **A regular value's fibre is a Rashomon locus, infinite-dimensional case.** If `y`
    is a regular value of a C¹ map from a Banach configuration space that out-ranks its
    finite-dimensional observable space, every configuration observing `y` has genuinely
    distinct configurations observing `y` arbitrarily close. -/
theorem regular_value_fiber_not_isolated_infinite {f : E → F} (hf : ContDiff ℝ 1 f)
    (hdim : (finrank ℝ F : Cardinal) < Module.rank ℝ E) {y : F} (hy : IsRegularValue f y)
    {x : E} (hx : f x = y) (ε : ℝ) (hε : 0 < ε) :
    ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε :=
  regular_fiber_not_isolated_infinite (hf.hasStrictFDerivAt one_ne_zero) (hy x hx) hdim ε hε

/-- **The submersion case on Banach configuration spaces, closed unconditionally.** A
    C¹ submersion from a Banach space out-ranking its finite-dimensional observable
    space has *every* fibre a Rashomon locus. Submersions have no critical points, so
    no Sard input is needed — this class survives the passage to infinite dimension
    intact (see §6: it is one of only two that do). -/
theorem submersion_fiber_not_isolated_infinite {f : E → F} (hf : ContDiff ℝ 1 f)
    (hsub : ∀ x, (fderiv ℝ f x).range = ⊤)
    (hdim : (finrank ℝ F : Cardinal) < Module.rank ℝ E) (x : E) (ε : ℝ) (hε : 0 < ε) :
    ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε :=
  regular_value_fiber_not_isolated_infinite hf hdim (fun x' _ => hsub x') rfl ε hε

section Measure
variable [MeasurableSpace F]

/-- **Generic ubiquity on Banach configuration spaces, conditional on Sard's
    property.** The observable space is finite-dimensional, so it still carries Haar
    measures and the genericity statement types unchanged; given `SardProperty f μ`,
    for `μ`-almost every observable value the fibre is a Rashomon locus. In infinite
    dimension the Sard hypothesis is *not* discharged by smoothness for general maps —
    it is genuinely false there (§6) — so unlike the finite-dimensional case this
    conditional form is the honest endpoint for general nonlinear observation maps. -/
theorem generic_ubiquity_of_sard_infinite {f : E → F} (hf : ContDiff ℝ 1 f)
    (hdim : (finrank ℝ F : Cardinal) < Module.rank ℝ E)
    (μ : Measure F) (hsard : SardProperty f μ) :
    ∀ᵐ y ∂μ, ∀ x, f x = y → ∀ ε : ℝ, 0 < ε →
      ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε := by
  have hreg : ∀ᵐ y ∂μ, IsRegularValue f y :=
    ae_iff.mpr (hsard : μ {y | ¬ IsRegularValue f y} = 0)
  filter_upwards [hreg] with y hy x hx ε hε
  exact regular_value_fiber_not_isolated_infinite hf hdim hy hx ε hε

variable [BorelSpace F]

/-- **Generic ubiquity for linear observation maps on Banach configuration spaces,
    unconditional.** `sardProperty_of_continuousLinearMap` never needed the
    configuration space finite-dimensional, so the linear class closes outright in
    infinite dimension: for any continuous linear observation map under the gap
    hypothesis, almost every observable value (w.r.t. any additive Haar measure) has a
    Rashomon-locus fibre. -/
theorem generic_ubiquity_of_continuousLinearMap_infinite (L : E →L[ℝ] F)
    (hdim : (finrank ℝ F : Cardinal) < Module.rank ℝ E)
    (μ : Measure F) [μ.IsAddHaarMeasure] :
    ∀ᵐ y ∂μ, ∀ x, L x = y → ∀ ε : ℝ, 0 < ε →
      ∃ x' : E, x' ≠ x ∧ L x' = L x ∧ dist x' x < ε :=
  generic_ubiquity_of_sard_infinite L.contDiff hdim μ
    (sardProperty_of_continuousLinearMap L μ)

/-- **Generic ubiquity for submersions on Banach configuration spaces,
    unconditional.** The almost-everywhere form of
    `submersion_fiber_not_isolated_infinite` (which in fact holds at *every* value);
    stated for symmetry with the linear capstone. With the linear case, these are the
    two observation-map classes that close unconditionally in infinite dimension. -/
theorem generic_ubiquity_of_submersion_infinite {f : E → F} (hf : ContDiff ℝ 1 f)
    (hsub : ∀ x, (fderiv ℝ f x).range = ⊤)
    (hdim : (finrank ℝ F : Cardinal) < Module.rank ℝ E) (μ : Measure F) :
    ∀ᵐ y ∂μ, ∀ x, f x = y → ∀ ε : ℝ, 0 < ε →
      ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε :=
  generic_ubiquity_of_sard_infinite hf hdim μ (sardProperty_of_submersion hsub μ)

end Measure

end RegularValue

-- ============================================================================
-- §6  Scoping: what does NOT extend, and why it cannot
-- ============================================================================

/-
  **The Morse–Sard rung does not extend to infinite-dimensional configuration spaces —
  and this is mathematics, not a formalization gap.**

  In finite dimension, `MorseSard.lean` discharges `SardProperty` outright for
  `C^{n−m+1}` maps (`sardProperty_of_contDiff`), making the generic-ubiquity statement
  unconditional for all sufficiently smooth observation maps. That theorem is
  intrinsically finite-dimensional, for two independent reasons:

  1. **Morse–Sard is false in this regime.** Kupka's counterexample (I. Kupka,
     *Counterexample to the Morse–Sard theorem in the case of infinite-dimensional
     manifolds*, Proc. Amer. Math. Soc. 16 (1965), 954–957) exhibits a C^∞ function
     `f : ℓ² → ℝ` whose critical values contain an interval — the set of critical
     values is not null, so `SardProperty f μ` fails for Lebesgue measure. No
     smoothness hypothesis rescues it: the statement itself is false for E = ℓ²,
     F = ℝ.

  2. **The Fredholm escape route is closed.** The infinite-dimensional replacement for
     Morse–Sard is the Sard–Smale theorem, which applies to *Fredholm* maps — maps
     whose derivative has finite-dimensional kernel and finite-codimensional range.
     But any continuous linear map from an infinite-dimensional `E` onto a
     finite-dimensional `F` has a kernel of finite codimension, hence
     infinite-dimensional kernel (§1, `rank_ker_eq_rank_of_infiniteDimensional`: the
     kernel has *full* rank). So no map in our regime (dim E = ∞, dim F < ∞) is ever
     Fredholm, and Sard–Smale is inapplicable by hypothesis — not merely unformalised.

  The honest classification in infinite dimension is therefore:

    * **Unconditionally closed classes**: linear observation maps
      (`generic_ubiquity_of_continuousLinearMap_infinite`) and submersions
      (`submersion_fiber_not_isolated_infinite`,
      `generic_ubiquity_of_submersion_infinite`). Both are machine-checked above with
      no Sard input, because neither has critical values to control.
    * **Conditionally closed**: general C¹ maps whose critical-value set happens to be
      null (`generic_ubiquity_of_sard_infinite`) — the `SardProperty` hypothesis must
      now be established per-map, since no general theorem supplies it.
    * **Genuinely open**: nonlinear observation maps with critical points. Positive
      results exist only under structural hypotheses that restore finite-dimensional
      behaviour transverse to the fibre (e.g. Sard–Smale after restricting to a
      finite-dimensional or Fredholm reduction; Moreira-type theorems are likewise
      finite-dimensional). This is a frontier of mathematics, not of this
      formalization.

  Consequently `sardProperty_of_contDiff` and `generic_ubiquity_of_contDiff`
  (`MorseSard.lean`) intentionally retain their `[FiniteDimensional ℝ E]` hypothesis;
  any "extension" of them to Banach `E` would be provably false.
-/

end UniversalImpossibility.Ubiquity
