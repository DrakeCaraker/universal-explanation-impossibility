import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Counterfactual Explanation Impossibility — Constructive Instance

Two model configurations with the same predictions on a test set but whose
nearest counterfactual explanations point in different directions (e.g.,
"increase income" vs "decrease debt").  All types are finite inductives;
the Rashomon property is **derived** via `decide`.

Pattern follows GeneticCode.lean.
-/

/-- Two models with different decision boundaries near the query point. -/
inductive CFConfig where
  | boundaryA : CFConfig  -- boundary tilted one way
  | boundaryB : CFConfig  -- boundary tilted another way
  deriving DecidableEq, Repr

/-- Predictions on the test set (both models agree). -/
inductive CFPrediction where
  | approved : CFPrediction
  deriving DecidableEq, Repr

/-- Direction of the nearest counterfactual. -/
inductive CFDirection where
  | increaseIncome : CFDirection  -- "you would be approved if income were higher"
  | decreaseDebt   : CFDirection  -- "you would be approved if debt were lower"
  deriving DecidableEq, Repr

/-- Both models produce the same prediction. -/
def cfObserve : CFConfig → CFPrediction
  | CFConfig.boundaryA => CFPrediction.approved
  | CFConfig.boundaryB => CFPrediction.approved

/-- Each model's nearest counterfactual points in a different direction. -/
def cfExplain : CFConfig → CFDirection
  | CFConfig.boundaryA => CFDirection.increaseIncome
  | CFConfig.boundaryB => CFDirection.decreaseDebt

/-- Same predictions. -/
theorem cf_same_prediction :
    cfObserve CFConfig.boundaryA = cfObserve CFConfig.boundaryB := by
  decide

/-- Different counterfactual directions. -/
theorem cf_different_directions :
    CFDirection.increaseIncome ≠ CFDirection.decreaseDebt := by
  decide

/-- Constructive counterfactual system with derived Rashomon. -/
def cfSystemConstructive : ExplanationSystem CFConfig CFDirection CFPrediction where
  observe := cfObserve
  explain := cfExplain
  incompatible := fun d₁ d₂ => d₁ ≠ d₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨CFConfig.boundaryA, CFConfig.boundaryB,
              cf_same_prediction, cf_different_directions⟩

/-- **Counterfactual Explanation Impossibility (Constructive).**
    No counterfactual explanation method can be simultaneously faithful,
    stable, and decisive.  Rashomon is derived, not axiomatized. -/
theorem counterfactual_impossibility_constructive
    (E : CFConfig → CFDirection)
    (hf : faithful cfSystemConstructive E)
    (hs : stable cfSystemConstructive E)
    (hd : decisive cfSystemConstructive E) : False :=
  explanation_impossibility cfSystemConstructive E hf hs hd
