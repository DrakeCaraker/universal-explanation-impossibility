import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Concept Probe (TCAV) Impossibility — Constructive Instance

Two model configurations that activate different internal concepts (concept A vs
concept B) but produce the same prediction.  All types are finite inductives;
the Rashomon property is **derived** via `decide` (no axioms).
-/

/-- Minimal concept configurations: two models whose internal representations
    encode different concept directions but compute the same function. -/
inductive ConceptCfg where
  | conceptA : ConceptCfg   -- activates concept direction A
  | conceptB : ConceptCfg   -- activates concept direction B
  deriving DecidableEq, Repr

/-- The prediction produced by the model. -/
inductive ConceptPrediction where
  | samePrediction : ConceptPrediction
  deriving DecidableEq, Repr

/-- Which concept activation direction the probe identifies. -/
inductive ConceptDirection where
  | dirA : ConceptDirection
  | dirB : ConceptDirection
  deriving DecidableEq, Repr

/-- Both configurations produce the same prediction. -/
def conceptObserve : ConceptCfg → ConceptPrediction
  | _ => ConceptPrediction.samePrediction

/-- Each configuration activates a different concept direction. -/
def conceptExplain : ConceptCfg → ConceptDirection
  | ConceptCfg.conceptA => ConceptDirection.dirA
  | ConceptCfg.conceptB => ConceptDirection.dirB

/-- Same prediction for both configs. -/
theorem concept_same_output : conceptObserve ConceptCfg.conceptA = conceptObserve ConceptCfg.conceptB := by
  decide

/-- Different concept directions. -/
theorem concept_different_directions : ConceptDirection.dirA ≠ ConceptDirection.dirB := by
  decide

/-- Constructive concept probe explanation system with derived Rashomon. -/
def conceptSystemConstructive : ExplanationSystem ConceptCfg ConceptDirection ConceptPrediction where
  observe := conceptObserve
  explain := conceptExplain
  incompatible := fun d₁ d₂ => d₁ ≠ d₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨ConceptCfg.conceptA, ConceptCfg.conceptB, concept_same_output, by decide⟩

/-- **Concept Probe Impossibility (Constructive).**
    No explanation of concept directions can be simultaneously faithful, stable,
    and decisive.  Rashomon is derived, not axiomatized. -/
theorem concept_impossibility_constructive
    (E : ConceptCfg → ConceptDirection)
    (hf : faithful conceptSystemConstructive E)
    (hs : stable conceptSystemConstructive E)
    (hd : decisive conceptSystemConstructive E) : False :=
  explanation_impossibility conceptSystemConstructive E hf hs hd
