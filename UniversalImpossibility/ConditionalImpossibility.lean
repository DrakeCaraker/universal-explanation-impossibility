/-
  Conditional Attribution Impossibility.

  The impossibility extends to conditional/causal SHAP whenever features
  have equal causal effects and symmetric causal position.

  Supplement: §Extension to Conditional and Causal Attributions (S44)
-/
import UniversalImpossibility.RashomonUniversality

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Conditional attribution function (e.g., conditional SHAP, causal SHAP).
    Separate from the marginal `attribution` in Defs.lean. -/
def ConditionalAttribution := Fin fs.P → Model → ℝ

/-- The Rashomon property for conditional attributions. -/
def ConditionalRashimonProperty (condAttr : ConditionalAttribution fs) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
    ∃ f f' : Model,
      condAttr j f > condAttr k f ∧
      condAttr k f' > condAttr j f'

/-- Symmetric swap for conditional attributions. -/
structure IsConditionalSymmetricSwap (condAttr : ConditionalAttribution fs)
    (swap : FeatureSwap fs) : Prop where
  cond_swap : ∀ (j k : Fin fs.P) (f : Model),
    condAttr j (swap j k f) = condAttr k f
  cond_swap_sym : ∀ (j k : Fin fs.P) (f : Model),
    condAttr k (swap j k f) = condAttr j f

/-- Conditional Rashomon from symmetry: if a symmetric swap exists for
    conditional attributions and some model distinguishes each pair,
    the Rashomon property holds for conditional attributions. -/
theorem conditional_rashomon_from_symmetry
    (condAttr : ConditionalAttribution fs)
    (swap : FeatureSwap fs)
    (hcsym : IsConditionalSymmetricSwap fs condAttr swap)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, condAttr j f ≠ condAttr k f) :
    ConditionalRashimonProperty fs condAttr := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hdiff⟩ := hnd ℓ j k hj hk hjk
  rcases lt_or_gt_of_ne hdiff with h | h
  · exact ⟨swap j k f, f,
      by rw [hcsym.cond_swap, hcsym.cond_swap_sym]; exact h,
      h⟩
  · exact ⟨f, swap j k f,
      h,
      by rw [hcsym.cond_swap_sym, hcsym.cond_swap]; exact h⟩

/-- **Conditional Attribution Impossibility**: if the causal structure preserves
    symmetry (symmetric swap exists for conditional attributions) and
    non-degeneracy holds, the impossibility extends to conditional SHAP.

    When features have equal causal effects (β_j = β_k) and symmetric causal
    position, the causal graph preserves the swap symmetry, so conditional
    attributions inherit the Rashomon property and the impossibility follows. -/
theorem conditional_attribution_impossibility
    (condAttr : ConditionalAttribution fs)
    (swap : FeatureSwap fs)
    (hcsym : IsConditionalSymmetricSwap fs condAttr swap)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, condAttr j f > condAttr k f)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ condAttr j f > condAttr k f) :
    False := by
  -- Get a model where condAttr j f > condAttr k f
  obtain ⟨f, hf⟩ := hnd ℓ j k hj hk hjk
  -- The swapped model reverses the ordering
  let f' := swap j k f
  have hf'_k : condAttr k f' > condAttr j f' := by
    show condAttr k (swap j k f) > condAttr j (swap j k f)
    rw [hcsym.cond_swap_sym, hcsym.cond_swap]
    exact hf
  -- Faithfulness at f gives ranking j k
  have hrank : ranking j k := (h_faithful f).mpr hf
  -- Faithfulness at f' requires condAttr j f' > condAttr k f'
  have hcontra : condAttr j f' > condAttr k f' := (h_faithful f').mp hrank
  -- But we showed condAttr k f' > condAttr j f', contradiction
  linarith

/-- **Escape condition**: if conditional attributions break symmetry
    (no symmetric swap exists), the impossibility does not apply.
    Specifically: if there is a pair where all models agree on the
    conditional ranking, stability is achievable. -/
theorem conditional_escape
    (condAttr : ConditionalAttribution fs)
    (j k : Fin fs.P)
    -- All models agree on the conditional ranking of j vs k
    (h_agree : ∀ f : Model, condAttr j f > condAttr k f) :
    -- Then a faithful, stable, complete ranking exists for this pair
    ∃ ranking : Fin fs.P → Fin fs.P → Prop,
      (∀ f, ranking j k ↔ condAttr j f > condAttr k f) ∧ ranking j k := by
  -- Define the ranking as the conditional attribution ordering at any model
  refine ⟨fun a b => ∀ f : Model, condAttr a f > condAttr b f, ?_, ?_⟩
  · intro f
    constructor
    · intro h
      exact h f
    · intro _
      exact h_agree
  · exact h_agree

end UniversalImpossibility
