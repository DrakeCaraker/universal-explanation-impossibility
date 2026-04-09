import UniversalImpossibility.ExplanationSystem

/-!
# Attention Map Impossibility Instance

Instantiates the universal explanation impossibility framework for attention maps.
Attention distributions over tokens cannot simultaneously be faithful to the model's
actual attention, stable across equivalent models, and decisive about which token
is most attended.
-/

set_option autoImplicit false

/-- Neural network weight configurations (parameters). -/
axiom AttentionConfig : Type

/-- Attention distributions: probability simplex over T tokens. -/
axiom AttentionMap : Type

/-- Observable behavior: input→output mappings (the function the network computes). -/
axiom AttentionObservable : Type

/-- The attention explanation system.
    - observe(θ) = f_θ (the function the network computes)
    - explain(θ) = α_θ (the attention distribution for a given input)
    - incompatible(α₁, α₂) = argmax differs (different most-attended tokens)
    - Rashomon: ∃ θ₁ θ₂ with same predictions but different max-attended tokens
      (D'Amour et al., 2020) -/
axiom attentionSystem : ExplanationSystem AttentionConfig AttentionMap AttentionObservable

/-- Attention map impossibility: no explanation of attention can be simultaneously
    faithful, stable, and decisive. Direct application of the universal impossibility
    theorem to the attention system. -/
theorem attention_impossibility
    (E : AttentionConfig → AttentionMap)
    (hf : faithful attentionSystem E)
    (hs : stable attentionSystem E)
    (hd : decisive attentionSystem E) :
    False :=
  explanation_impossibility attentionSystem E hf hs hd
