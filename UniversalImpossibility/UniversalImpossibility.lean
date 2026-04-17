import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.AttributionInstanceConstructive  -- was AttributionInstance
import UniversalImpossibility.AttentionInstanceConstructive     -- was AttentionInstance
import UniversalImpossibility.CounterfactualInstanceConstructive -- was CounterfactualInstance
import UniversalImpossibility.ConceptInstanceConstructive       -- was ConceptInstance
import UniversalImpossibility.CausalInstanceConstructive        -- was CausalInstance
import UniversalImpossibility.ModelSelectionInstanceConstructive -- was ModelSelectionInstance
import UniversalImpossibility.MechInterpInstanceConstructive    -- was MechInterpInstance
import UniversalImpossibility.SaliencyInstanceConstructive      -- was SaliencyInstance
import UniversalImpossibility.LLMExplanationInstanceConstructive -- was LLMExplanationInstance
import UniversalImpossibility.CausalExplanationSystem

/-!
# Universal Impossibility

All nine explanation types are instances of the abstract ExplanationSystem.
The impossibility theorem applies uniformly to all of them via a single
proof: `explanation_impossibility`.

This file serves as the import hub and documents the instance inventory.
-/

-- The abstract theorem is proved ONCE in ExplanationSystem.lean.
-- Each instance file axiomatizes only the Rashomon property for that
-- explanation type. The impossibility is inherited, not re-proved.

-- Instance inventory:
-- 1. Additive attributions (SHAP, IG, LIME) — AttributionInstance.lean
-- 2. Attention maps — AttentionInstance.lean
-- 3. Counterfactual explanations — CounterfactualInstance.lean
-- 4. Concept probes (TCAV) — ConceptInstance.lean
-- 5. Causal discovery — CausalInstance.lean
-- 6. Model selection — ModelSelectionInstance.lean
