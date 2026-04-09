import UniversalImpossibility.ExplanationSystem

/-!
# LLM Self-Explanation Impossibility Instance

Instantiates the universal explanation impossibility framework for LLM self-explanations.
LLM self-explanations (chain-of-thought reasoning) as natural language explanation space
cannot simultaneously be faithful to the model's internal computation, stable across
observationally equivalent models, and decisive about which reasoning chain is correct.
-/

set_option autoImplicit false

/-- LLM weight configurations (parameters): the full set of model weights. -/
axiom LLMConfig : Type

/-- LLM explanations: natural language explanation strings (chain-of-thought traces). -/
axiom LLMExplanation : Type

/-- Observable behavior: input→output mappings (the function the LLM computes). -/
axiom LLMObservable : Type

/-- The LLM self-explanation system.
    - observe(θ) = f_θ (the function the LLM computes on inputs)
    - explain(θ) = c_θ (the chain-of-thought explanation for a given prompt)
    - incompatible(c₁, c₂) = the explanations assert contradictory reasoning steps
    - Rashomon: ∃ θ₁ θ₂ with same input→output behavior but contradictory CoT explanations
      (Lanham et al., 2023; Turpin et al., 2023) -/
axiom llmSystem : ExplanationSystem LLMConfig LLMExplanation LLMObservable

/-- LLM explanation impossibility: no explanation of an LLM's reasoning can be simultaneously
    faithful, stable, and decisive. Direct application of the universal impossibility
    theorem to the LLM self-explanation system. -/
theorem llm_explanation_impossibility
    (E : LLMConfig → LLMExplanation)
    (hf : faithful llmSystem E)
    (hs : stable llmSystem E)
    (hd : decisive llmSystem E) :
    False :=
  explanation_impossibility llmSystem E hf hs hd
