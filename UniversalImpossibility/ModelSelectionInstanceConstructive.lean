import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Model Selection Impossibility — Constructive Instance

Two model configurations (e.g., a tree ensemble and a neural network) that
achieve the same validation accuracy but are different models.  All types are
finite inductives; the Rashomon property is **derived** via `decide`.

Pattern follows GeneticCode.lean.
-/

/-- Two competing model configurations in the Rashomon set. -/
inductive MSCfg where
  | treeEnsemble : MSCfg
  | neuralNet    : MSCfg
  deriving DecidableEq, Repr

/-- Observable performance metric (e.g., validation accuracy bucket). -/
inductive MSPerf where
  | optimal : MSPerf
  deriving DecidableEq, Repr

/-- Model identity: which model is selected / recommended. -/
inductive MSModelId where
  | tree   : MSModelId
  | neural : MSModelId
  deriving DecidableEq, Repr

/-- Both configurations achieve the same performance. -/
def msObserve : MSCfg → MSPerf
  | MSCfg.treeEnsemble => MSPerf.optimal
  | MSCfg.neuralNet    => MSPerf.optimal

/-- Each configuration recommends its own model type. -/
def msExplain : MSCfg → MSModelId
  | MSCfg.treeEnsemble => MSModelId.tree
  | MSCfg.neuralNet    => MSModelId.neural

/-- Same observable performance. -/
theorem ms_same_performance :
    msObserve MSCfg.treeEnsemble = msObserve MSCfg.neuralNet := by
  decide

/-- Different model recommendations. -/
theorem ms_different_models : MSModelId.tree ≠ MSModelId.neural := by
  decide

/-- Constructive model selection system with derived Rashomon. -/
def msSystemConstructive : ExplanationSystem MSCfg MSModelId MSPerf where
  observe := msObserve
  explain := msExplain
  incompatible := fun m₁ m₂ => m₁ ≠ m₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨MSCfg.treeEnsemble, MSCfg.neuralNet,
              ms_same_performance, ms_different_models⟩

/-- **Model Selection Impossibility (Constructive).**
    No model selection explanation can be simultaneously faithful, stable,
    and decisive.  Rashomon is derived, not axiomatized. -/
theorem model_selection_impossibility_constructive
    (E : MSCfg → MSModelId)
    (hf : faithful msSystemConstructive E)
    (hs : stable msSystemConstructive E)
    (hd : decisive msSystemConstructive E) : False :=
  explanation_impossibility msSystemConstructive E hf hs hd
