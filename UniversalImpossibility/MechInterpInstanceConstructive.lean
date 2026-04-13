import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Mechanistic Interpretability Impossibility — Constructive Instance

Two neural network configurations that compute the same function but admit
different circuit decompositions.  All types are finite inductives; the
Rashomon property is **derived** via `decide` (no axioms).

Empirical motivation: Meloux et al. (ICLR 2025) found 85 distinct valid
circuits with zero circuit error for a simple XOR task.
-/

/-- Minimal circuit configurations: two networks with different internal
    wiring but identical input-output behavior. -/
inductive MechInterpCfg where
  | circuitAlpha : MechInterpCfg   -- implements function via circuit alpha
  | circuitBeta  : MechInterpCfg   -- implements function via circuit beta
  deriving DecidableEq, Repr

/-- The function computed by the network. -/
inductive MechInterpOutput where
  | sameFunction : MechInterpOutput
  deriving DecidableEq, Repr

/-- Which circuit decomposition the interpretability method identifies. -/
inductive CircuitDecomp where
  | decompAlpha : CircuitDecomp
  | decompBeta  : CircuitDecomp
  deriving DecidableEq, Repr

/-- Both configurations compute the same function. -/
def mechInterpObserve : MechInterpCfg → MechInterpOutput
  | _ => MechInterpOutput.sameFunction

/-- Each configuration has a different circuit decomposition. -/
def mechInterpExplain : MechInterpCfg → CircuitDecomp
  | MechInterpCfg.circuitAlpha => CircuitDecomp.decompAlpha
  | MechInterpCfg.circuitBeta  => CircuitDecomp.decompBeta

/-- Same function for both configs. -/
theorem mechInterp_same_output : mechInterpObserve MechInterpCfg.circuitAlpha = mechInterpObserve MechInterpCfg.circuitBeta := by
  decide

/-- Different circuit decompositions. -/
theorem mechInterp_different_circuits : CircuitDecomp.decompAlpha ≠ CircuitDecomp.decompBeta := by
  decide

/-- Constructive mechanistic interpretability explanation system with derived Rashomon. -/
def mechInterpSystemConstructive : ExplanationSystem MechInterpCfg CircuitDecomp MechInterpOutput where
  observe := mechInterpObserve
  explain := mechInterpExplain
  incompatible := fun d₁ d₂ => d₁ ≠ d₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨MechInterpCfg.circuitAlpha, MechInterpCfg.circuitBeta, mechInterp_same_output, by decide⟩

/-- **Mechanistic Interpretability Impossibility (Constructive).**
    No circuit explanation method can be simultaneously faithful, stable,
    and decisive.  Rashomon is derived, not axiomatized. -/
theorem mech_interp_impossibility_constructive
    (E : MechInterpCfg → CircuitDecomp)
    (hf : faithful mechInterpSystemConstructive E)
    (hs : stable mechInterpSystemConstructive E)
    (hd : decisive mechInterpSystemConstructive E) : False :=
  explanation_impossibility mechInterpSystemConstructive E hf hs hd
