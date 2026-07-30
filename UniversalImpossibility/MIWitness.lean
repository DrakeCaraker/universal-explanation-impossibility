/-
  A concrete, fully-discharged witness for the MI-generalized impossibility.

  `impossibility_from_mi` (MutualInformation.lean) proves the attribution
  impossibility for MI-dependent feature pairs, but takes the *bridge*
  hypothesis `hdep_implies_diff` (statistical dependence ⇒ some model
  distinguishes the features) as an assumption. That assumption is genuinely
  necessary in the abstract framework — a degenerate (constant) attribution
  makes it false — so it cannot be discharged there.

  This file exhibits ONE concrete model in which the bridge is a *theorem*, not
  an assumption, and the impossibility therefore holds unconditionally:

  * the mutual information is COMPUTED from an explicit joint distribution
    (two perfectly-correlated fair bits) via the Shannon formula, and shown to
    equal `log 2 > 0` — it is not stipulated;
  * the attribution difference is DERIVED from a real mechanism — the two
    Rashomon models are the feature-swap pair that perfect correlation forces
    (each attributes all mass to a different one of the two features), so the
    same correlation that makes MI positive is what makes the two models
    prediction-equivalent-but-attribution-different.

  Nothing is co-defined to align by fiat: MI comes from the distribution,
  the attribution split from model non-uniqueness under the correlation.
-/
import UniversalImpossibility.MutualInformation
import Mathlib.Analysis.SpecialFunctions.Log.Basic

set_option autoImplicit false

namespace UniversalImpossibility
namespace MIWitness

/-! ### An explicit joint distribution: two perfectly-correlated fair bits -/

/-- Joint pmf on `Bool × Bool`: `p(x,y) = 1/2` if `x = y`, else `0`.
    This is the maximally-dependent fair-bit distribution (`X₂ = X₁`). -/
noncomputable def p (x y : Bool) : ℝ := if x = y then 1/2 else 0

/-- Marginal of the first coordinate (`= 1/2` for each value). -/
noncomputable def pX (x : Bool) : ℝ := p x true + p x false

/-- Marginal of the second coordinate (`= 1/2` for each value). -/
noncomputable def pY (y : Bool) : ℝ := p true y + p false y

theorem pX_eq (x : Bool) : pX x = 1/2 := by cases x <;> norm_num [pX, p]
theorem pY_eq (y : Bool) : pY y = 1/2 := by cases y <;> norm_num [pY, p]

/-- Shannon mutual information of `p`, from the definition
    `I = Σ p(x,y) · log( p(x,y) / (pₓ(x)·p_Y(y)) )`.
    Zero-probability cells contribute `0` automatically because Mathlib sets
    `Real.log 0 = 0`, so no `0 log 0` side condition is needed. -/
noncomputable def MI : ℝ :=
  ∑ x : Bool, ∑ y : Bool, p x y * Real.log (p x y / (pX x * pY y))

/-- The mutual information of two perfectly-correlated fair bits is `log 2`.
    (The two diagonal cells each contribute `½·log 2`; the off-diagonal cells
    contribute `0`.) -/
theorem MI_eq_log_two : MI = Real.log 2 := by
  simp only [MI, Fintype.sum_bool, p, pX, pY]
  norm_num
  ring

/-- Hence the dependence is genuine: `I(X₁;X₂) = log 2 > 0`. -/
theorem MI_pos : 0 < MI := by
  rw [MI_eq_log_two]; exact Real.log_pos (by norm_num)

/-! ### The feature space, the swap mechanism, and the attribution -/

/-- Two features in a single collinear group. `abbrev` so `fs.P` reduces to the
    literal `2`, giving `Fin fs.P` its numeral instances. -/
abbrev fs : FeatureSpaceMI where
  P := 2
  L := 1
  hP := by norm_num
  groupOf := fun _ => 0
  group_size_ge_two := by intro ℓ; fin_cases ℓ; decide

/-- The two Rashomon models (indexed by `Bool`): under perfect correlation
    `X₂ = X₁`, a model may attribute all importance to feature 0 (`false`) or
    all to feature 1 (`true`); both make identical predictions. -/
def attribution : Fin fs.P → Bool → ℝ :=
  fun j f => if j = 0 then (if f then 0 else 1) else (if f then 1 else 0)

/-- The feature swap: exchanging the two features flips which Rashomon model
    is selected (identity on a pair with itself). This is the concrete
    instance of the symmetric-DGP swap used by `rashomon_from_mi_dependence`. -/
def swap : Fin fs.P → Fin fs.P → Bool → Bool :=
  fun j k f => if j = k then f else !f

theorem hsym_j : ∀ (j k : Fin fs.P) (f : Bool),
    attribution j (swap j k f) = attribution k f := by
  intro j k f
  fin_cases j <;> fin_cases k <;> cases f <;> simp [attribution, swap]

theorem hsym_k : ∀ (j k : Fin fs.P) (f : Bool),
    attribution k (swap j k f) = attribution j f := by
  intro j k f
  fin_cases j <;> fin_cases k <;> cases f <;> simp [attribution, swap]

/-- The MI assignment, with the off-diagonal value COMPUTED as `MI` (the
    distribution's mutual information above), not stipulated. -/
noncomputable def mia : MutualInfoAssignment fs where
  mi := fun j k => if j = k then 0 else MI
  mi_nonneg := by
    intro j k; dsimp only; split
    · norm_num
    · exact le_of_lt MI_pos
  mi_symm := by
    intro j k; dsimp only
    rcases eq_or_ne j k with h | h
    · simp [h]
    · rw [if_neg h, if_neg (Ne.symm h)]
  mi_self_nonneg := by intro j; dsimp only; simp

/-! ### The bridge is a theorem here, and the impossibility is unconditional -/

/-- **The bridge, discharged.** For this concrete model, MI-dependence of a
    feature pair implies some model distinguishes them — proved, not assumed.
    Dependence forces `j ≠ k` (the diagonal MI is `0`), and for the genuinely
    dependent pair the feature-selection model `false` attributes `1` to
    feature 0 and `0` to feature 1. -/
theorem hdep_implies_diff : ∀ (j k : Fin fs.P),
    MIDependent fs mia j k → ∃ f : Bool, attribution j f ≠ attribution k f := by
  intro j k hdep
  have hne : j ≠ k := by
    intro h
    rw [MIDependent] at hdep
    simp [mia, h] at hdep
  refine ⟨false, ?_⟩
  fin_cases j <;> fin_cases k <;> simp_all [attribution]

/-- A general (model-type-parametric) restatement of `impossibility_from_mi`;
    the library version is the `M := Model` specialization. The proof is the
    same four-step chain and never uses any structure on `M`. -/
theorem impossibility_from_mi_general {M : Type}
    (attribution : Fin fs.P → M → ℝ)
    (swap : Fin fs.P → Fin fs.P → M → M)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k → ∃ f : M, attribution j f ≠ attribution k f)
    (j k : Fin fs.P) (hdep : MIDependent fs mia j k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : M, ranking j k ↔ attribution j f > attribution k f) :
    False := by
  obtain ⟨f, hdiff⟩ := hdep_implies_diff j k hdep
  rcases lt_or_gt_of_ne hdiff with h | h
  · have h1 : attribution j (swap j k f) > attribution k (swap j k f) := by
      rw [hsym_j, hsym_k]; exact h
    have hrank : ranking j k := (h_faithful (swap j k f)).mpr h1
    have hcon : attribution j f > attribution k f := (h_faithful f).mp hrank
    linarith
  · have hrank : ranking j k := (h_faithful f).mpr h
    have h2 : attribution j (swap j k f) > attribution k (swap j k f) :=
      (h_faithful (swap j k f)).mp hrank
    rw [hsym_j, hsym_k] at h2
    linarith

/-- **Unconditional concrete impossibility.** No faithful stable ranking of the
    two dependent features exists in this model — with the bridge discharged,
    there are no remaining hypotheses about mutual information or attributions.
    `#print axioms` on this theorem shows it depends on no custom axioms. -/
theorem concrete_mi_impossibility
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Bool,
      ranking 0 1 ↔ attribution 0 f > attribution 1 f) :
    False := by
  have hdep : MIDependent fs mia 0 1 := by
    rw [MIDependent]; show (0 : ℝ) < mia.mi 0 1
    simp only [mia]; rw [if_neg (by decide)]; exact MI_pos
  exact impossibility_from_mi_general attribution swap hsym_j hsym_k mia
    hdep_implies_diff 0 1 hdep ranking h_faithful

end MIWitness
end UniversalImpossibility
