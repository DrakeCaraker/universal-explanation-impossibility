/-
  Local vs. Global Attribution Instability.

  The impossibility applies to ANY attribution function satisfying the
  Rashomon property — whether global (mean |SHAP|) or local (per-instance SHAP).

  Supplement: §Local vs. Global Attribution Instability
-/
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-- Local attribution: attribution for feature j in model f at data point x.
    This is a generalization of the global `attribution` from Defs.lean. -/
def LocalAttribution (DataPoint : Type) := Fin fs.P → Model → DataPoint → ℝ

/-- The Rashomon property for local attributions at a fixed data point x. -/
def LocalRashimonProperty (DataPoint : Type) (localAttr : LocalAttribution fs DataPoint)
    (x : DataPoint) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
    ∃ f f' : Model,
      localAttr j f x > localAttr k f x ∧
      localAttr k f' x > localAttr j f' x

/-- The local attribution impossibility: if the Rashomon property holds
    at a specific data point x, no ranking can be faithful, stable,
    and complete at that point. -/
theorem local_attribution_impossibility
    (DataPoint : Type) (localAttr : LocalAttribution fs DataPoint)
    (x : DataPoint)
    (hrash_local : LocalRashimonProperty fs DataPoint localAttr x)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ localAttr j f x > localAttr k f x) :
    False := by
  -- The proof is identical to attribution_impossibility
  -- just with localAttr j f x instead of attribution fs j f
  obtain ⟨f, f', h1, h2⟩ := hrash_local ℓ j k hj hk hjk
  have hrank : ranking j k := (h_faithful f).mpr h1
  have hcontra : localAttr j f' x > localAttr k f' x := (h_faithful f').mp hrank
  linarith

/-- Corollary: global Rashomon implies local impossibility.
    If the global attribution is defined as the mean of local attributions,
    and the Rashomon property holds globally, we can't conclude local Rashomon
    directly (Jensen's inequality goes the wrong way for the existential).

    However, we CAN state: the impossibility structure is the same at both
    levels — it applies to ANY attribution function satisfying Rashomon. -/
theorem impossibility_is_attribution_agnostic
    (attr₁ attr₂ : Fin fs.P → Model → ℝ)
    -- Both satisfy Rashomon
    (hrash₁ : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f f' : Model, attr₁ j f > attr₁ k f ∧ attr₁ k f' > attr₁ j f')
    (hrash₂ : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f f' : Model, attr₂ j f > attr₂ k f ∧ attr₂ k f' > attr₂ j f')
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    -- Neither admits a faithful stable complete ranking
    (∀ ranking : Fin fs.P → Fin fs.P → Prop,
      (∀ f, ranking j k ↔ attr₁ j f > attr₁ k f) → False) ∧
    (∀ ranking : Fin fs.P → Fin fs.P → Prop,
      (∀ f, ranking j k ↔ attr₂ j f > attr₂ k f) → False) := by
  constructor
  · intro ranking h_faith
    obtain ⟨f, f', h1, h2⟩ := hrash₁ ℓ j k hj hk hjk
    have := (h_faith f).mpr h1
    linarith [(h_faith f').mp this]
  · intro ranking h_faith
    obtain ⟨f, f', h1, h2⟩ := hrash₂ ℓ j k hj hk hjk
    have := (h_faith f).mpr h1
    linarith [(h_faith f').mp this]

end UniversalImpossibility
