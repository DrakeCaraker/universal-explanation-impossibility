import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# LLM Self-Explanation Impossibility — Constructive Instance

Two LLM configurations (prompt-completion pairs) that produce the same output
token but cite different source tokens in their chain-of-thought explanations.
All types are finite inductives; the Rashomon property is **derived** via
`decide` (no axioms).
-/

/-- Minimal LLM configurations: two models that produce the same completion
    but with different internal reasoning traces. -/
inductive LLMCfg where
  | citesTokenA : LLMCfg   -- chain-of-thought cites source token A
  | citesTokenB : LLMCfg   -- chain-of-thought cites source token B
  deriving DecidableEq, Repr

/-- The output token produced by the LLM. -/
inductive LLMOutput where
  | sameToken : LLMOutput
  deriving DecidableEq, Repr

/-- Which source token the chain-of-thought cites as the reason. -/
inductive LLMCitation where
  | sourceA : LLMCitation
  | sourceB : LLMCitation
  deriving DecidableEq, Repr

/-- Both configurations produce the same output token. -/
def llmObserve : LLMCfg → LLMOutput
  | _ => LLMOutput.sameToken

/-- Each configuration cites a different source token. -/
def llmExplain : LLMCfg → LLMCitation
  | LLMCfg.citesTokenA => LLMCitation.sourceA
  | LLMCfg.citesTokenB => LLMCitation.sourceB

/-- Same output for both configs. -/
theorem llm_same_output : llmObserve LLMCfg.citesTokenA = llmObserve LLMCfg.citesTokenB := by
  decide

/-- Different citations. -/
theorem llm_different_citations : LLMCitation.sourceA ≠ LLMCitation.sourceB := by
  decide

/-- Constructive LLM self-explanation system with derived Rashomon. -/
def llmSystemConstructive : ExplanationSystem LLMCfg LLMCitation LLMOutput where
  observe := llmObserve
  explain := llmExplain
  incompatible := fun c₁ c₂ => c₁ ≠ c₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨LLMCfg.citesTokenA, LLMCfg.citesTokenB, llm_same_output, by decide⟩

/-- **LLM Self-Explanation Impossibility (Constructive).**
    No explanation of LLM reasoning can be simultaneously faithful, stable,
    and decisive.  Rashomon is derived, not axiomatized. -/
theorem llm_explanation_impossibility_constructive
    (E : LLMCfg → LLMCitation)
    (hf : faithful llmSystemConstructive E)
    (hs : stable llmSystemConstructive E)
    (hd : decisive llmSystemConstructive E) : False :=
  explanation_impossibility llmSystemConstructive E hf hs hd
