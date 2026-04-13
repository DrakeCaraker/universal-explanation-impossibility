import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Attention Map Impossibility — Constructive Instance

Two transformer configurations that compute the same output token but attend to
different positions.  All types are finite inductives; the Rashomon property is
**derived** via `decide` (no axioms).

Pattern follows GeneticCode.lean.
-/

/-- Minimal attention configurations: two heads that route information
    differently but produce the same output. -/
inductive AttnConfig where
  | headA : AttnConfig   -- attends to position 0
  | headB : AttnConfig   -- attends to position 1
  deriving DecidableEq, Repr

/-- The output token produced by the transformer. -/
inductive OutputToken where
  | tok : OutputToken
  deriving DecidableEq, Repr

/-- Which position receives the highest attention weight (argmax). -/
inductive AttnArgmax where
  | pos0 : AttnArgmax
  | pos1 : AttnArgmax
  deriving DecidableEq, Repr

/-- Both heads produce the same output token. -/
def attnObserve : AttnConfig → OutputToken
  | AttnConfig.headA => OutputToken.tok
  | AttnConfig.headB => OutputToken.tok

/-- Each head attends to a different position. -/
def attnExplain : AttnConfig → AttnArgmax
  | AttnConfig.headA => AttnArgmax.pos0
  | AttnConfig.headB => AttnArgmax.pos1

/-- Same output for both configs. -/
theorem attn_same_output : attnObserve AttnConfig.headA = attnObserve AttnConfig.headB := by
  decide

/-- Different attention argmax. -/
theorem attn_different_argmax : AttnArgmax.pos0 ≠ AttnArgmax.pos1 := by
  decide

/-- Constructive attention explanation system with derived Rashomon. -/
def attentionSystemConstructive : ExplanationSystem AttnConfig AttnArgmax OutputToken where
  observe := attnObserve
  explain := attnExplain
  incompatible := fun a₁ a₂ => a₁ ≠ a₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨AttnConfig.headA, AttnConfig.headB, attn_same_output, attn_different_argmax⟩

/-- **Attention Map Impossibility (Constructive).**
    No explanation of attention can be simultaneously faithful, stable, and
    decisive.  Rashomon is derived, not axiomatized. -/
theorem attention_impossibility_constructive
    (E : AttnConfig → AttnArgmax)
    (hf : faithful attentionSystemConstructive E)
    (hs : stable attentionSystemConstructive E)
    (hd : decisive attentionSystemConstructive E) : False :=
  explanation_impossibility attentionSystemConstructive E hf hs hd
