import UniversalImpossibility.ExplanationSystem

/-!
# Attribution Instance of the Abstract Explanation System

This file ports the attribution impossibility into the abstract `ExplanationSystem`
framework from `ExplanationSystem.lean`.

The three types are axiomatized as opaque types so the proof is model-agnostic.
`attrSystem` bundles the four fields (observe, explain, incompatible, rashomon)
required by `ExplanationSystem`.  `attribution_impossibility_abstract` then follows
immediately from `explanation_impossibility`.
-/

set_option autoImplicit false

/-! ### Opaque carrier types -/

/-- The configuration space for attribution: a model together with a feature-space
    configuration (e.g. a trained GBDT and the input it sees). -/
axiom AttrConfig : Type

/-- The explanation space: a feature-attribution vector (e.g. SHAP values). -/
axiom AttrExplanation : Type

/-- The observable space: what the model produces (e.g. a prediction). -/
axiom AttrObservable : Type

/-! ### The attribution explanation system -/

/-- An `ExplanationSystem` instance for feature attribution.

    The four fields are axiomatized because their concrete constructions depend on
    model-class specifics (GBDT, Lasso, etc.) that are handled in the model-specific
    files.  Axiomatizing the *bundled* structure here means the impossibility below
    depends on zero model-specific behavioral axioms — only on the Rashomon property
    encoded in `attrSystem.rashomon`. -/
axiom attrSystem : ExplanationSystem AttrConfig AttrExplanation AttrObservable

/-! ### Impossibility -/

/-- **The Attribution Impossibility (abstract framework version).**

    No explanation map `E : AttrConfig → AttrExplanation` can simultaneously be
    faithful, stable, and decisive with respect to `attrSystem`.

    This is a direct application of `explanation_impossibility` to the attribution
    instance; the proof is entirely in the abstract framework and carries zero
    model-specific axiom dependencies beyond `attrSystem` itself. -/
theorem attribution_impossibility_abstract
    (E : AttrConfig → AttrExplanation)
    (hf : faithful attrSystem E)
    (hs : stable attrSystem E)
    (hd : decisive attrSystem E) :
    False :=
  explanation_impossibility attrSystem E hf hs hd
