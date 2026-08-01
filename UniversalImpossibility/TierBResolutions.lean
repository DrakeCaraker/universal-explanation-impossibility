/-
  TierBResolutions.lean — Machine-checked resolutions of the program's
  Tier-B conjectures from "The Limits of Explanation".

  ## Scope

  A Tier-B claim in the monograph is one "supported by experiments/audits" but
  NOT yet a Lean-verified theorem. Most Tier-B items in the monograph are
  empirical validation sections (flip-rate audits, η-law fits, etc.) or
  formalisation-heavy open problems (the irreducible-representation count of
  non-dominated profiles; the categorical-enrichment adjunction). This file
  resolves the ONE Tier-B claim that is a crisp, purely mathematical assertion:

    > "The stronger statement that *every* stable map has image in V^G is
    >  stated here as a conjecture [Tier B], not a Lean-verified theorem."
    >  — universal_impossibility_monograph.tex, Part (ii) of the Explanation
    >    Stability Theorem (also flagged at "the general form of Part (ii)
    >    is a conjecture").

  The repo already documents, honestly, that the existing lemma
  `fixed_point_in_invariant_subspace` (aka the deprecated
  `stable_in_fixed_subspace`) proves only the definitional
  `(hu : R u = u) : R u = u`, NOT the conjecture. This file settles the
  conjecture proper.

  ## Outcome: REFUTED as literally stated, and CORRECTED to a sharp theorem.

  (R1) `gstable_not_image_in_fixed` — the literal conjecture is FALSE. A
       constant map is G-stable (trivially constant on orbits) yet its image is
       a single point that need not lie in V^G. Formalised in the exact
       `MulAction` / fixed-point language of the positive theorem, using the
       left-regular action of any nontrivial group. This is the counterexample
       the monograph itself floats ("a constant map v ↦ w₀ with w₀ ∉ V^G").

  (R2) `linear_sigmaStable_image_not_in_diag` — SHARPER refutation: even a
       genuinely ℝ-LINEAR stable map can have image outside V^G. Over ℝ² with
       the ℤ/2 coordinate-swap σ (so V^G = the diagonal), the linear map
       L(x,y) = (x+y, 0) is σ-stable and its image is the x-axis ⊄ diagonal.
       So linearity does NOT rescue the conjecture.

  (C) `gstable_equivariant_image_subset_fixed` — the CORRECTED theorem, sharp:
       a map that is BOTH stable (T(g•x) = T x) AND equivariant
       (T(g•x) = g•T x) has image in V^G. Three-line proof, any action.
       `Lsum_not_equivariant` pins equivariance as exactly the hypothesis the
       linear counterexample violates, so (C) is the sharp boundary that (R2)
       borders.

  Together: a refutation PLUS the sharp corrected theorem converts a conjecture
  into a boundary.

  Target axioms: [propext, Classical.choice, Quot.sound].
-/

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.GroupTheory.GroupAction.Basic

set_option autoImplicit false

namespace UniversalImpossibility.TierB

/-! ## Definitions: stable, equivariant, and the fixed subspace V^G

We work with a general scalar action `SMul G ·`. In the monograph G is a finite
group acting orthogonally on an inner product space V, but the corrected
theorem below needs no group axioms whatsoever, so it is stated for an arbitrary
`SMul` and specialises to the monograph's finite-group setting. -/

variable {G X V : Type*}

/-- A map `T : X → V` is **G-stable** if it is constant on G-orbits:
    `T (g • x) = T x`. This is the linear-space analogue of the framework's
    `stable` (constant on observe-fibres), specialised to a transitive
    symmetry action on each fibre (cf. `gInvariant` in `UniversalResolution`). -/
def GStable [SMul G X] (T : X → V) : Prop :=
  ∀ (g : G) (x : X), T (g • x) = T x

/-- A map `T : X → V` is **G-equivariant** if it intertwines the actions:
    `T (g • x) = g • T x`. -/
def GEquivariant [SMul G X] [SMul G V] (T : X → V) : Prop :=
  ∀ (g : G) (x : X), T (g • x) = g • T x

/-- The **fixed subspace** `V^G = {v | ∀ g, g • v = v}`. Definitionally equal to
    Mathlib's `MulAction.fixedPoints G V` when G is a monoid; kept self-contained
    so it also names V^G for the bare `SMul` used by the corrected theorem. -/
def fixedSubspace (G V : Type*) [SMul G V] : Set V :=
  {v | ∀ g : G, g • v = v}

/-! ## (C) The CORRECTED theorem — stable + equivariant ⟹ image in V^G -/

/-- **Corrected Tier-B theorem (RESOLVES the stable-image conjecture).**

    The monograph's Part-(ii) conjecture — *every stable map has image in V^G* —
    is false as stated (see `gstable_not_image_in_fixed` and
    `linear_sigmaStable_image_not_in_diag` below). The sharp true statement adds
    equivariance: if `T` is **both** G-stable and G-equivariant, then every
    output `T x` lies in the fixed subspace `V^G`.

    Proof (three lines): for any g, `g • T x = T (g • x)` by equivariance
    (reversed) `= T x` by stability; that is exactly `T x ∈ V^G`. No group
    axioms are used — this holds for any `SMul`. -/
theorem gstable_equivariant_image_subset_fixed
    [SMul G X] [SMul G V] (T : X → V)
    (hS : GStable (G := G) T) (hE : GEquivariant (G := G) T) :
    ∀ x, T x ∈ fixedSubspace G V := by
  intro x
  show ∀ g : G, g • T x = T x
  intro g
  rw [← hE g x, hS g x]

/-! ## (R1) The literal conjecture is FALSE — the constant-map counterexample

The monograph itself floats the candidate: *"a constant map v ↦ w₀ with
w₀ ∉ V^G is G-invariant/stable but its image is not in V^G."* We formalise it in
the exact `MulAction` / fixed-point language, using the left-regular action of a
nontrivial group `G` on itself (`g • w = g * w`), for which V^G is empty. -/

/-- **Refutation of the literal Tier-B conjecture.**

    Stability ALONE does not imply image in V^G. For any nontrivial group `G`
    acting on itself by left multiplication, the constant map `T ≡ 1` is
    G-stable, yet its image `{1}` is not contained in the fixed subspace: taking
    any `g ≠ 1` gives `g • 1 = g ≠ 1`. Hence "every stable map has image in V^G"
    is false. -/
theorem gstable_not_image_in_fixed {G : Type*} [Group G] [Nontrivial G] :
    ∃ T : G → G, GStable (G := G) T ∧ ¬ (∀ x, T x ∈ fixedSubspace G G) := by
  refine ⟨fun _ => 1, ?_, ?_⟩
  · intro _ _; rfl
  · intro h
    obtain ⟨g, hg⟩ := exists_ne (1 : G)
    have hfix : (1 : G) ∈ fixedSubspace G G := h 1
    have hg1 : g • (1 : G) = 1 := hfix g
    rw [smul_eq_mul, mul_one] at hg1
    exact hg (hg1)

/-! ## (R2) The SHARPER refutation — even LINEAR stable maps escape V^G

To show the failure is not an artefact of the pathological (empty-V^G) regular
action, we exhibit it over ℝ² with the ℤ/2 coordinate swap σ, whose fixed
subspace is the diagonal — a genuine 1-dimensional V^G (the monograph's running
S₂-on-ℝ² example, η = 1/2). We prove that a genuinely ℝ-linear stable map has
image *off* the diagonal. So linearity does NOT rescue the conjecture. -/

/-- The coordinate swap σ on ℝ², i.e. `(x, y) ↦ (y, x)`. -/
def swapR2 : ℝ × ℝ → ℝ × ℝ := fun p => (p.2, p.1)

/-- σ is an involution, so ⟨σ⟩ ≅ ℤ/2 and its fixed set is genuinely V^G for the
    two-element group generated by σ. -/
theorem swapR2_involutive (p : ℝ × ℝ) : swapR2 (swapR2 p) = p := rfl

/-- σ-stable: constant on the ℤ/2-orbit `{p, σ p}`. For G = ⟨σ⟩ = ℤ/2, being
    constant on all orbits is exactly this single condition. -/
def SigmaStable (T : ℝ × ℝ → ℝ × ℝ) : Prop := ∀ p, T (swapR2 p) = T p

/-- σ-equivariant: `T (σ p) = σ (T p)`. -/
def SigmaEquivariant (T : ℝ × ℝ → ℝ × ℝ) : Prop := ∀ p, T (swapR2 p) = swapR2 (T p)

/-- `V^G = V^{⟨σ⟩}` = the diagonal `{p | σ p = p}`. -/
def diagFixed : Set (ℝ × ℝ) := {p | swapR2 p = p}

/-- The linear map `L(x, y) = (x + y, 0)`, bundled as an honest `ℝ`-linear map,
    witnessing that the counterexample is linear (not merely a constant). -/
def Lsum : (ℝ × ℝ) →ₗ[ℝ] (ℝ × ℝ) where
  toFun p := (p.1 + p.2, 0)
  map_add' p q := by
    simp only [Prod.fst_add, Prod.snd_add, Prod.mk_add_mk, add_zero]
    ring_nf
  map_smul' c p := by
    simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Prod.smul_mk,
      RingHom.id_apply, mul_zero]
    rw [mul_add]

/-- **Sharper refutation: linear + stable still escapes V^G.**

    `L(x, y) = (x + y, 0)` is σ-stable (its output is symmetric in the two
    coordinates) and ℝ-linear, yet its image is the x-axis, which is not
    contained in the diagonal `V^G`: `L(1, 0) = (1, 0)` and `σ(1, 0) = (0, 1)
    ≠ (1, 0)`. So the Tier-B conjecture fails even for linear stable maps. -/
theorem linear_sigmaStable_image_not_in_diag :
    SigmaStable (fun p => Lsum p) ∧ ¬ (∀ p, Lsum p ∈ diagFixed) := by
  constructor
  · intro p
    show (Lsum (swapR2 p)) = Lsum p
    simp only [Lsum, swapR2, LinearMap.coe_mk, AddHom.coe_mk]
    rw [add_comm]
  · intro h
    have h2 : swapR2 (Lsum (1, 0)) = Lsum (1, 0) := h (1, 0)
    simp only [Lsum, swapR2, LinearMap.coe_mk, AddHom.coe_mk, add_zero,
      Prod.mk.injEq] at h2
    exact absurd h2.1 (by norm_num)

/-- The constant-map counterexample in the genuine-V^G setting: `T ≡ (1, 0)` is
    σ-stable but `(1, 0) ∉ diagonal`. This is the monograph's literal candidate
    `v ↦ w₀`, now with V^G an honest nontrivial subspace. -/
theorem const_sigmaStable_image_not_in_diag :
    SigmaStable (fun _ : ℝ × ℝ => ((1 : ℝ), (0 : ℝ))) ∧
      ¬ (∀ p : ℝ × ℝ, (fun _ : ℝ × ℝ => ((1 : ℝ), (0 : ℝ))) p ∈ diagFixed) := by
  constructor
  · intro _; rfl
  · intro h
    have h2 : swapR2 ((1 : ℝ), (0 : ℝ)) = ((1 : ℝ), (0 : ℝ)) := h (0, 0)
    simp only [swapR2, Prod.mk.injEq] at h2
    exact absurd h2.1 (by norm_num)

/-- **The boundary is sharp.** `Lsum` violates exactly the hypothesis that the
    corrected theorem `gstable_equivariant_image_subset_fixed` adds: it is
    stable and linear but NOT equivariant. Thus equivariance is precisely what
    the conjecture was missing. -/
theorem Lsum_not_equivariant : ¬ SigmaEquivariant (fun p => Lsum p) := by
  intro h
  have h2 : Lsum (swapR2 (1, 0)) = swapR2 (Lsum (1, 0)) := h (1, 0)
  simp only [Lsum, swapR2, LinearMap.coe_mk, AddHom.coe_mk, add_zero, zero_add,
    Prod.mk.injEq] at h2
  exact absurd h2.1 (by norm_num)

/-- **Concrete instance of the corrected theorem** on ℝ² with the ℤ/2 swap: a
    σ-stable AND σ-equivariant map has image in the diagonal `V^G`. This is
    `gstable_equivariant_image_subset_fixed` specialised, confirming the sharp
    theorem holds in the very setting where the literal conjecture fails. -/
theorem sigmaStable_equivariant_image_in_diag (T : ℝ × ℝ → ℝ × ℝ)
    (hS : SigmaStable T) (hE : SigmaEquivariant T) :
    ∀ p, T p ∈ diagFixed := by
  intro p
  show swapR2 (T p) = T p
  rw [← hE p, hS p]

end UniversalImpossibility.TierB
