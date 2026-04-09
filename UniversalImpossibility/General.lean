/-
  General impossibility: model-faithful attributions produce opposite
  rankings for symmetric features across training runs.

  This is the Arrow-type layer — model-agnostic in principle, instantiated
  here via our gradient boosting axioms.
-/
import UniversalImpossibility.Trilemma
import UniversalImpossibility.SplitGap
import UniversalImpossibility.Iterative

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### First-mover dominates within group -/

/-- The first-mover's split count strictly exceeds that of any other
    feature in the same group. -/
theorem splitCount_firstMover_gt (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm : firstMover fs f = j) (hjk : j ≠ k) :
    splitCount fs k f < splitCount fs j f := by
  have hne : firstMover fs f ≠ k := by rw [hfm]; exact hjk
  have hgap := split_gap_exact fs f j k ℓ hj hk hfm hne
  have hpos : 0 < fs.ρ ^ 2 * (numTrees : ℝ) / (2 - fs.ρ ^ 2) :=
    div_pos (mul_pos (pow_pos fs.hρ_pos 2) (Nat.cast_pos.mpr numTrees_pos)) (denom_pos fs)
  linarith

/-- The first-mover's attribution strictly exceeds that of any other
    feature in the same group. -/
theorem attribution_firstMover_gt (f : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm : firstMover fs f = j) (hjk : j ≠ k) :
    attribution fs k f < attribution fs j f := by
  obtain ⟨c, hc_pos, hc_eq⟩ := attribution_proportional fs f
  rw [hc_eq j, hc_eq k]
  exact mul_lt_mul_of_pos_left (splitCount_firstMover_gt fs f j k ℓ hj hk hfm hjk) hc_pos

/-- Two models with different first-movers produce opposite orderings:
    if j is first-mover in f and k is first-mover in f', then
    attribution j f > attribution k f AND attribution k f' > attribution j f'. -/
theorem attribution_reversal (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k)
    (hjk : j ≠ k) :
    attribution fs k f < attribution fs j f ∧
    attribution fs j f' < attribution fs k f' :=
  ⟨attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk,
   attribution_firstMover_gt fs f' k j ℓ hk hj hfm' (Ne.symm hjk)⟩

/-- No ranking can be simultaneously faithful to two models with different
    first-movers. A ranking faithful to f says j > k, but a ranking
    faithful to f' says k > j — contradiction. -/
theorem no_stable_ranking (f f' : Model) (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hfm : firstMover fs f = j) (hfm' : firstMover fs f' = k)
    (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful_f : ranking j k ↔ attribution fs k f < attribution fs j f)
    (h_faithful_f' : ranking j k ↔ attribution fs k f' < attribution fs j f') :
    False := by
  have ⟨h1, h2⟩ := attribution_reversal fs f f' j k ℓ hj hk hfm hfm' hjk
  have hrank : ranking j k := h_faithful_f.mpr h1
  have hnotrank : ¬ (attribution fs k f' < attribution fs j f') := by linarith
  exact hnotrank (h_faithful_f'.mp hrank)

/-! ### GBDT satisfies the Rashomon Property -/

/-- Sequential gradient boosting satisfies the Rashomon property:
    for any two features in the same group, Axiom 1 (firstMover_surjective)
    provides models ranking them in opposite orders, and Axiom 4
    (attribution_proportional) translates split-count dominance to
    attribution dominance. -/
theorem gbdt_rashomon : RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := firstMover_surjective fs ℓ j hj
  obtain ⟨f', hfm'⟩ := firstMover_surjective fs ℓ k hk
  exact ⟨f, f',
    attribution_firstMover_gt fs f j k ℓ hj hk hfm hjk,
    attribution_firstMover_gt fs f' k j ℓ hk hj hfm' (Ne.symm hjk)⟩

/-- The Attribution Impossibility instantiated for gradient boosting:
    no stable faithful ranking exists for sequential GBDT under collinearity. -/
theorem gbdt_impossibility (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs (gbdt_rashomon fs) ℓ j k hj hk hjk ranking h_faithful

/-! ### GBDT as an Iterative Optimizer -/

/-- Sequential gradient boosting is an iterative optimizer where the
    dominant feature is the first-mover (root split of tree 1). -/
noncomputable def gbdtOptimizer : IterativeOptimizer fs where
  dominant := firstMover fs
  dominant_gt := by
    intro f ℓ k h_dom_in hk hne
    have hjk : firstMover fs f ≠ k := Ne.symm hne
    exact attribution_firstMover_gt fs f (firstMover fs f) k ℓ h_dom_in hk rfl hjk
  dominant_surjective := firstMover_surjective fs

end UniversalImpossibility
