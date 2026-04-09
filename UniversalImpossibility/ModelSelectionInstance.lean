/-
  Model Selection Instance for the Abstract ExplanationSystem Framework.

  Instantiates the abstract ExplanationSystem with model selection types:
  - MSConfig      : model configurations (the configuration space Θ)
  - MSExplanation : selected / recommended model (the explanation space H)
  - MSObservable  : observed performance metrics (the observable space Y)

  The Rashomon property holds when two configurations produce the same
  performance profile yet lead to incompatible model recommendations —
  a standard consequence of model multiplicity in near-optimal model sets.

  Supplement: §Model Selection under Model Multiplicity (S45–S47)
-/
import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-! ### Axiomatized types for the model selection domain -/

/-- MSConfig: the space of model configurations (hyperparameters, architectures, etc.).
    Two distinct configurations can yield the same observable performance
    while recommending incompatible models — the Model Rashomon set. -/
axiom MSConfig : Type

/-- MSExplanation: the space of model selections / recommendations produced
    by an explanation method.  An explanation method maps each configuration
    to a recommended model. -/
axiom MSExplanation : Type

/-- MSObservable: the space of observable performance metrics (e.g. validation
    loss, accuracy on a held-out split). -/
axiom MSObservable : Type

/-! ### The concrete ExplanationSystem instance -/

/-- msSystem: a concrete ExplanationSystem over (MSConfig, MSExplanation, MSObservable).

    The system is fully axiomatized: `observe` maps a configuration to its
    performance metrics, `explain` maps it to the recommended model selection,
    and `incompatible` captures the conflict relation between recommendations.
    The bundled `rashomon` field witnesses two configurations with identical
    performance metrics but incompatible recommendations. -/
axiom msSystem : ExplanationSystem MSConfig MSExplanation MSObservable

/-! ### Model selection instance impossibility -/

/-- **Model Selection Instance Impossibility.**

    No explanation function E : MSConfig → MSExplanation can be simultaneously
    faithful (= msSystem.explain), stable (factors through msSystem.observe),
    and decisive (never maps two configurations to incompatible outputs)
    when the Rashomon property holds.

    Proof: direct application of the universal explanation_impossibility theorem
    to the axiomatized msSystem. -/
theorem model_selection_instance_impossibility
    (E : MSConfig → MSExplanation)
    (hf : faithful msSystem E)
    (hs : stable msSystem E)
    (hd : decisive msSystem E) :
    False :=
  explanation_impossibility msSystem E hf hs hd
