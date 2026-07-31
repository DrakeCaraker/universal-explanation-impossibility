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
import Mathlib.Analysis.Calculus.Implicit
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import UniversalImpossibility.Ubiquity

open Module Filter Topology MeasureTheory

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

-- ============================================================================
-- §4  Local smooth globalisation: distinct models on the true curved fibre
-- ============================================================================

/-
  The infinitesimal result of §2 gives a flat tangent direction of the derivative.
  Here we globalise it locally, via Mathlib's implicit function theorem: at a
  *regular* point of a (nonlinear) map whose configuration space out-dimensions its
  observable space, the true curved fibre contains genuinely distinct configurations
  arbitrarily close — not merely an infinitesimal direction. This is the local part
  of the smooth globalisation; what remains is the genericity of regular points
  (Sard's theorem, not in Mathlib) and the global positive-measure statement.
-/

section SmoothLocal
variable {𝕜 E F : Type*} [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F] [FiniteDimensional 𝕜 F]

/-- **Local smooth globalisation.** At a regular point (`f'.range = ⊤`) of a map whose
    configuration space has strictly larger dimension than its observable space, the
    *exact* fibre is not isolated: genuinely distinct configurations that observe
    identically exist arbitrarily close (within any `ε`). This upgrades the
    infinitesimal `fderiv_exists_kernel_direction` (a flat tangent direction) to
    genuinely distinct points on the true curved fibre, via the implicit function
    theorem — no Sard genericity required. -/
theorem regular_fiber_not_isolated {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f f' a) (hf' : f'.range = ⊤)
    (hdim : finrank 𝕜 F < finrank 𝕜 E) (ε : ℝ) (hε : 0 < ε) :
    ∃ x : E, x ≠ a ∧ f x = f a ∧ dist x a < ε := by
  haveI : CompleteSpace E := FiniteDimensional.complete 𝕜 E
  have hker : f'.ker ≠ ⊥ := f'.toLinearMap.ker_ne_bot_of_finrank_lt hdim
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

/-- **The smooth (nonlinear) ubiquity impossibility, locally.** At a regular point of a
    map whose configuration space out-dimensions its observable space, a genuinely
    distinct configuration observing identically exists (produced on the true curved
    fibre by the implicit function theorem); given a non-degenerate explanation, no
    explanation can be faithful, stable, and decisive. The nonlinear analogue of
    `underspecified_impossibility`, with the colliding pair now derived rather than
    assumed. -/
theorem smooth_regular_underspecified_impossibility
    {H : Type*} {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f f' a) (hf' : f'.range = ⊤)
    (hdim : finrank 𝕜 F < finrank 𝕜 E)
    (explain : E → H) (incomp : H → H → Prop)
    (hnd : ∀ x, x ≠ a → f x = f a → incomp (explain x) (explain a))
    (Efn : E → H)
    (hfaith : ∀ t, ¬ incomp (Efn t) (explain t))
    (hstab : ∀ p q, f p = f q → Efn p = Efn q)
    (hdec : ∀ t h, incomp (explain t) h → incomp (Efn t) h) :
    False := by
  obtain ⟨x, hx_ne, hx_obs, _⟩ := regular_fiber_not_isolated hf hf' hdim 1 one_pos
  have hinc : incomp (Efn x) (explain a) := hdec x (explain a) (hnd x hx_ne hx_obs)
  have heq : Efn x = Efn a := hstab x a hx_obs
  rw [heq] at hinc
  exact hfaith a hinc

/-- Non-vacuity: the projection `ℝ² → ℝ` is a regular map with configuration dimension
    `2` strictly above observable dimension `1`, so the hypotheses above are satisfiable
    (and its fibres are genuinely positive-dimensional lines of distinct models). -/
example :
    HasStrictFDerivAt (Prod.fst : ℝ × ℝ → ℝ) (ContinuousLinearMap.fst ℝ ℝ ℝ) (0, 0)
    ∧ (ContinuousLinearMap.fst ℝ ℝ ℝ).range = ⊤
    ∧ finrank ℝ ℝ < finrank ℝ (ℝ × ℝ) :=
  ⟨(ContinuousLinearMap.fst ℝ ℝ ℝ).hasStrictFDerivAt,
   Submodule.eq_top_iff'.mpr (fun y => ⟨(y, 0), rfl⟩),
   by rw [Module.finrank_prod, Module.finrank_self]; norm_num⟩

end SmoothLocal

-- ============================================================================
-- §5  Genericity of regular values: the linear case in full, the rest reduced to Sard
-- ============================================================================

/-
  The local smooth globalisation of §4 holds at a *regular* point. To upgrade "at a
  regular point" to "at almost every observable value" — the honest remaining residual —
  one needs the genericity of regular values, i.e. **Sard's theorem**: the set of
  critical values is null. Full Morse–Sard for `dim E > dim F` is not in Mathlib (it is
  work-in-progress in Y. Kudryashov's external `SardMoreira` development and, for the
  manifold form, M. Rothgang's `fpvandoorn/sard`); Mathlib has only the equidimensional
  Sard *lemma* (`MeasureTheory.…addHaar_image_eq_zero_of_det_fderivWithin_eq_zero`).

  So we do two honest things here. (1) We isolate Sard's conclusion as a named predicate
  `SardProperty` and prove that, *conditional on it*, the full generic-ubiquity statement
  follows (`generic_ubiquity_of_sard`) — every step except Sard is machine-checked. (2) We
  prove `SardProperty` *outright for the linear observation maps* that are the primary
  ubiquity setting (`sardProperty_of_continuousLinearMap`): their critical values sit in a
  proper subspace, which is Haar-null. The linear case therefore has **no residual**; the
  nonlinear case is reduced to exactly the one classical theorem that remains unformalised
  in Mathlib.
-/

section RegularValue
variable {E F : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F]

/-- `y` is a *regular value* of `f`: every configuration mapping to `y` is a regular
    point (surjective derivative). -/
def IsRegularValue (f : E → F) (y : F) : Prop :=
  ∀ x, f x = y → (fderiv ℝ f x).range = ⊤

/-- **A regular value's fibre is a positive-dimensional Rashomon locus.** If `y` is a
    regular value of a C¹ map with dim(config) > dim(observable), then every
    configuration observing `y` has genuinely distinct configurations observing `y`
    arbitrarily close. The per-value form of the local smooth globalisation; no Sard. -/
theorem regular_value_fiber_not_isolated {f : E → F} (hf : ContDiff ℝ 1 f)
    (hdim : finrank ℝ F < finrank ℝ E) {y : F} (hy : IsRegularValue f y)
    {x : E} (hx : f x = y) (ε : ℝ) (hε : 0 < ε) :
    ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε :=
  regular_fiber_not_isolated (hf.hasStrictFDerivAt one_ne_zero) (hy x hx) hdim ε hε

end RegularValue

section Sard
variable {E F : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F]
  [MeasurableSpace F] [BorelSpace F]

/-- **Sard's property**: the set of non-regular (critical) values of `f` is `μ`-null.
    This is exactly the conclusion of Sard's theorem — NOT proved here in general (full
    Morse–Sard for `dim E > dim F` is not in Mathlib; WIP in Kudryashov's `SardMoreira`),
    but isolated as a named hypothesis, and proved outright for linear maps below. -/
def SardProperty (f : E → F) (μ : Measure F) : Prop :=
  μ {y | ¬ IsRegularValue f y} = 0

omit [BorelSpace F] in
/-- **Generic ubiquity, conditional on Sard.** Given Sard's property, for almost every
    observable value the fibre is a positive-dimensional Rashomon locus: every
    configuration observing it has genuinely distinct configurations observing the same
    value arbitrarily close. Everything except `SardProperty` is machine-checked. -/
theorem generic_ubiquity_of_sard {f : E → F} (hf : ContDiff ℝ 1 f)
    (hdim : finrank ℝ F < finrank ℝ E) (μ : Measure F) (hsard : SardProperty f μ) :
    ∀ᵐ y ∂μ, ∀ x, f x = y → ∀ ε : ℝ, 0 < ε →
      ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε := by
  have hreg : ∀ᵐ y ∂μ, IsRegularValue f y :=
    ae_iff.mpr (hsard : μ {y | ¬ IsRegularValue f y} = 0)
  filter_upwards [hreg] with y hy x hx ε hε
  exact regular_value_fiber_not_isolated hf hdim hy hx ε hε

omit [FiniteDimensional ℝ E] in
/-- **Sard's property holds outright for linear observation maps.** The critical values
    lie in `range L`, a proper subspace when `L` is not onto, hence Haar-null; so the
    linear case of the generic ubiquity statement needs no external Sard input. -/
theorem sardProperty_of_continuousLinearMap
    (L : E →L[ℝ] F) (μ : Measure F) [μ.IsAddHaarMeasure] :
    SardProperty (⇑L) μ := by
  have hfd : ∀ x, fderiv ℝ (⇑L) x = L := fun x => (L.hasFDerivAt).fderiv
  by_cases hL : LinearMap.range (L : E →ₗ[ℝ] F) = ⊤
  · have hempty : {y | ¬ IsRegularValue (⇑L) y} = ∅ := by
      ext y
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_not]
      intro x _
      rw [hfd x]; exact hL
    rw [SardProperty, hempty]; exact measure_empty
  · refine measure_mono_null ?_ (Measure.addHaar_submodule μ (LinearMap.range (L : E →ₗ[ℝ] F)) hL)
    intro y hy
    have hy' : ¬ ∀ x, L x = y → (fderiv ℝ (⇑L) x).range = ⊤ := hy
    rw [not_forall] at hy'
    obtain ⟨x, hx⟩ := hy'
    exact LinearMap.mem_range.mpr ⟨x, (Classical.not_imp.mp hx).1⟩

end Sard

end UniversalImpossibility.Ubiquity
