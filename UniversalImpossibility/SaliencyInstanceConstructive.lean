import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Saliency Map (GradCAM) Impossibility — Constructive Instance

Two vision model configurations whose gradient-based saliency maps highlight
different spatial regions but produce the same classification.  All types are
finite inductives; the Rashomon property is **derived** via `decide` (no axioms).
-/

/-- Minimal saliency configurations: two models with different gradient flows
    but identical input-output behavior. -/
inductive SaliencyCfg where
  | regionOne : SaliencyCfg   -- gradient focuses on spatial region 1
  | regionTwo : SaliencyCfg   -- gradient focuses on spatial region 2
  deriving DecidableEq, Repr

/-- The classification produced by the vision model. -/
inductive SaliencyClassification where
  | sameClass : SaliencyClassification
  deriving DecidableEq, Repr

/-- Which spatial region the saliency map highlights (argmax of heatmap). -/
inductive SaliencyRegion where
  | region1 : SaliencyRegion
  | region2 : SaliencyRegion
  deriving DecidableEq, Repr

/-- Both configurations produce the same classification. -/
def saliencyObserve : SaliencyCfg → SaliencyClassification
  | _ => SaliencyClassification.sameClass

/-- Each configuration highlights a different spatial region. -/
def saliencyExplain : SaliencyCfg → SaliencyRegion
  | SaliencyCfg.regionOne => SaliencyRegion.region1
  | SaliencyCfg.regionTwo => SaliencyRegion.region2

/-- Same classification for both configs. -/
theorem saliency_same_output : saliencyObserve SaliencyCfg.regionOne = saliencyObserve SaliencyCfg.regionTwo := by
  decide

/-- Different saliency regions. -/
theorem saliency_different_regions : SaliencyRegion.region1 ≠ SaliencyRegion.region2 := by
  decide

/-- Constructive saliency map explanation system with derived Rashomon. -/
def saliencySystemConstructive : ExplanationSystem SaliencyCfg SaliencyRegion SaliencyClassification where
  observe := saliencyObserve
  explain := saliencyExplain
  incompatible := fun r₁ r₂ => r₁ ≠ r₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨SaliencyCfg.regionOne, SaliencyCfg.regionTwo, saliency_same_output, by decide⟩

/-- **Saliency Map Impossibility (Constructive).**
    No explanation of saliency maps can be simultaneously faithful, stable,
    and decisive.  Rashomon is derived, not axiomatized. -/
theorem saliency_impossibility_constructive
    (E : SaliencyCfg → SaliencyRegion)
    (hf : faithful saliencySystemConstructive E)
    (hs : stable saliencySystemConstructive E)
    (hd : decisive saliencySystemConstructive E) : False :=
  explanation_impossibility saliencySystemConstructive E hf hs hd
