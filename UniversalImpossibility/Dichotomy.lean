import Mathlib.GroupTheory.GroupAction.Defs
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.Linarith

/-!
# The Structural / Statistical Dichotomy of Explanation Instability

The most common dismissal of Rashomon-driven explanation instability is
*"more data will fix it."* This file makes the answer a theorem. Explanation
instability splits into two components with opposite fates as data grows:

* **Structural (incurable).** If a symmetry `g` of the model space preserves the
  loss on *every* dataset — the defining property of a *function-class* symmetry
  such as neuron permutation or a sign-flip pair — then the entire group orbit of
  any minimizer stays inside the Rashomon set at *every* sample size. Ambiguity
  across the orbit is never resolved by more data. (`orbit_subset_rashomon`,
  `minimizer_orbit_closed`.)

* **Statistical (curable).** If, transverse to the orbit, the loss is strongly
  convex (grows at least quadratically in the distance to the orbit), then the
  Rashomon set at tolerance `ε` is contained in a tube of radius `√(ε/c)` around
  the orbit. As the statistically-indistinguishable tolerance `εₙ → 0` with more
  data, the non-orbit ambiguity shrinks to zero. (`rashomon_within_tube`,
  `tube_radius_tendsto_zero`.)

Together: total instability = a structural floor (the orbit, constant in `n`) ⊕ a
statistical part (the transverse tube, `→ 0`). The certificate measures the sum;
this theory decomposes it. The structural floor is exactly the incurable
component the empirical `flip-vs-n` experiment exhibits (a positive floor for
exactly-symmetric features, decay to zero for a genuinely dominant one).

All results are axiom-clean (no domain axioms); the symmetry and convexity are
hypotheses, not assumptions about any particular model.
-/

set_option autoImplicit false

namespace Dichotomy

variable {Param Data G : Type} [Group G] [MulAction G Param]

/-- A learning problem with a group of *structural* symmetries: a loss
`loss d θ` on each dataset `d`, a per-dataset optimal value `opt d` lower-bounding
the loss, and a group `G` acting on the model space `Param` that preserves the
loss on every dataset. The last field is the mathematical content of
"function-class symmetry": `g` does not change the computed function, hence the
loss is identical for *all* data. -/
structure SymLearning (Param Data G : Type) [Group G] [MulAction G Param] where
  /-- Empirical loss/risk of model `θ` on dataset `d`. -/
  loss : Data → Param → ℝ
  /-- A per-dataset optimal (lower-bounding) value. -/
  opt : Data → ℝ
  /-- `opt d` lower-bounds the loss on `d`. -/
  opt_le : ∀ (d : Data) (θ : Param), opt d ≤ loss d θ
  /-- **Structural symmetry**: every `g ∈ G` preserves the loss on every dataset. -/
  sym : ∀ (g : G) (d : Data) (θ : Param), loss d (g • θ) = loss d θ

/-- The Rashomon set at tolerance `ε` on dataset `d`: all models within `ε` of the
optimum. -/
def Rashomon (S : SymLearning Param Data G) (d : Data) (ε : ℝ) : Set Param :=
  {θ | S.loss d θ ≤ S.opt d + ε}

/-! ## Structural half — the incurable floor -/

/-- **Structural theorem.** The Rashomon set is closed under the symmetry group:
if `θ` is within `ε` of optimal, so is every `g • θ`, on *every* dataset and at
*every* tolerance. Hence the whole orbit of any near-optimal model is Rashomon —
an ambiguity no amount of data removes. -/
theorem orbit_subset_rashomon (S : SymLearning Param Data G) (d : Data) (ε : ℝ)
    (θ : Param) (hθ : θ ∈ Rashomon S d ε) (g : G) : g • θ ∈ Rashomon S d ε := by
  simp only [Rashomon, Set.mem_setOf_eq] at hθ ⊢
  rw [S.sym g d θ]; exact hθ

/-- The set of exact minimizers is orbit-closed on every dataset: if `θ` attains
`opt d`, so does every `g • θ`. The choice of orbit representative is therefore
undetermined at every sample size. -/
theorem minimizer_orbit_closed (S : SymLearning Param Data G) (d : Data) (θ : Param)
    (g : G) (h : S.loss d θ = S.opt d) : S.loss d (g • θ) = S.opt d := by
  rw [S.sym g d θ]; exact h

/-- **Incurable by data.** Fix any minimizer `θ` and any `g`. On *every* dataset
`d` (hence at every sample size) both `θ` and `g • θ` lie in the Rashomon set, for
every `ε ≥ 0`. If the explanation map distinguishes them, the ambiguity is present
at all `n`. -/
theorem structural_floor (S : SymLearning Param Data G) (θ : Param) (g : G)
    (hmin : ∀ d, S.loss d θ = S.opt d) (ε : ℝ) (hε : 0 ≤ ε) :
    ∀ d, θ ∈ Rashomon S d ε ∧ g • θ ∈ Rashomon S d ε := by
  intro d
  have h1 : θ ∈ Rashomon S d ε := by
    simp only [Rashomon, Set.mem_setOf_eq, hmin d]; linarith
  exact ⟨h1, orbit_subset_rashomon S d ε θ h1 g⟩

/-! ## Statistical half — the curable tube -/

/-- A `SymLearning` whose loss is, transverse to the orbit, strongly convex: it
grows at least quadratically in a nonnegative "distance to the orbit" `distOrbit`
(which vanishes on the orbit). This is the only extra hypothesis needed to make
the non-orbit Rashomon ambiguity shrink with data. -/
structure TransverseConvex (Param Data G : Type) [Group G] [MulAction G Param]
    extends SymLearning Param Data G where
  /-- Distance from a model to the symmetry orbit (0 on the orbit). -/
  distOrbit : Param → ℝ
  /-- The distance is nonnegative. -/
  distOrbit_nonneg : ∀ θ, 0 ≤ distOrbit θ
  /-- Transverse strong-convexity constant. -/
  c : ℝ
  /-- The constant is positive. -/
  c_pos : 0 < c
  /-- Quadratic growth away from the orbit: `opt d + c · dist² ≤ loss d θ`. -/
  transverse : ∀ (d : Data) (θ : Param),
    opt d + c * (distOrbit θ) ^ 2 ≤ loss d θ

/-- **Statistical theorem.** Under transverse strong convexity, every model in the
Rashomon set at tolerance `ε` lies within distance `√(ε/c)` of the orbit. The
non-structural ambiguity is confined to a tube whose radius is set entirely by the
tolerance and the curvature. -/
theorem rashomon_within_tube (T : TransverseConvex Param Data G) (d : Data) (ε : ℝ)
    (θ : Param) (hθ : θ ∈ Rashomon T.toSymLearning d ε) :
    T.distOrbit θ ≤ Real.sqrt (ε / T.c) := by
  simp only [Rashomon, Set.mem_setOf_eq] at hθ
  -- from transverse growth and membership: c · dist² ≤ ε
  have hgrow := T.transverse d θ
  have hcd : T.c * (T.distOrbit θ) ^ 2 ≤ ε := by linarith
  -- hence dist² ≤ ε / c
  have hsq : (T.distOrbit θ) ^ 2 ≤ ε / T.c := by
    rw [le_div_iff₀ T.c_pos, mul_comm]; exact hcd
  -- take square roots; √(dist²) = dist since dist ≥ 0
  have h := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (T.distOrbit_nonneg θ)] at h

/-- The tube radius `√(εₙ/c)` tends to `0` as the statistically-indistinguishable
tolerance `εₙ → 0`. So with more data (smaller `εₙ`) the transverse ambiguity
vanishes — the curable half of the dichotomy. -/
theorem tube_radius_tendsto_zero (c : ℝ) (_hc : 0 < c) (ε : ℕ → ℝ)
    (hε : Filter.Tendsto ε Filter.atTop (nhds 0)) :
    Filter.Tendsto (fun n => Real.sqrt (ε n / c)) Filter.atTop (nhds 0) := by
  have hdiv : Filter.Tendsto (fun n => ε n / c) Filter.atTop (nhds 0) := by
    simpa using hε.div_const c
  have := (Real.continuous_sqrt.tendsto 0).comp hdiv
  simpa [Real.sqrt_zero] using this

/-! ## The dichotomy, assembled -/

/-- **Structural / statistical dichotomy.** For a transverse-strongly-convex
learning problem with a structural symmetry:

* (structural) the entire orbit of any exact minimizer stays in the Rashomon set
  at every dataset and tolerance — a floor independent of the sample size; and
* (statistical) the Rashomon set at tolerance `ε` is contained in the `√(ε/c)`
  tube around the orbit — a component that shrinks to the orbit as `ε → 0`.

Total instability is the orbit (incurable) plus a transverse tube (curable). -/
theorem dichotomy (T : TransverseConvex Param Data G) (θ : Param) (g : G)
    (hmin : ∀ d, T.loss d θ = T.opt d) (d : Data) (ε : ℝ) (hε : 0 ≤ ε) :
    -- structural floor: both orbit representatives are Rashomon here
    (θ ∈ Rashomon T.toSymLearning d ε ∧ g • θ ∈ Rashomon T.toSymLearning d ε) ∧
    -- statistical tube: everything Rashomon is within √(ε/c) of the orbit
    (∀ ψ, ψ ∈ Rashomon T.toSymLearning d ε → T.distOrbit ψ ≤ Real.sqrt (ε / T.c)) := by
  refine ⟨structural_floor T.toSymLearning θ g hmin ε hε d, ?_⟩
  intro ψ hψ
  exact rashomon_within_tube T d ε ψ hψ

end Dichotomy
