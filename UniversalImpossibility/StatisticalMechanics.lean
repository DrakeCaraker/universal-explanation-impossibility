import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Statistical Mechanics — Derived Rashomon Property

The foundational example of underspecification in physics:
multiple microstates correspond to the same macrostate.

Two coins: configurations (H,T) and (T,H) both have macrostate
"1 head." The macrostate (observable) does not determine the
microstate (explanation).

The permutation group S₂ acts by swapping the coins:
(H,T) ↔ (T,H). The macrostate is invariant under this action.
The microcanonical ensemble (uniform average over microstates
at fixed macrostate) is the G-invariant resolution.
-/

namespace UniversalImpossibility

/-! ### Microstate: two coins -/

/-- A microstate: two coins, each Heads (true) or Tails (false). -/
structure CoinPair where
  coin1 : Bool  -- true = Heads
  coin2 : Bool
  deriving DecidableEq, Repr

/-! ### Macrostate and permutation -/

/-- Macrostate: number of heads. -/
def numHeads (c : CoinPair) : Nat :=
  (if c.coin1 then 1 else 0) + (if c.coin2 then 1 else 0)

/-- Permutation: swap the two coins. -/
def swapCoins (c : CoinPair) : CoinPair :=
  ⟨c.coin2, c.coin1⟩

/-! ### Concrete witnesses -/

/-- Microstate 1: (Heads, Tails). -/
def microstate1 : CoinPair := ⟨true, false⟩

/-- Microstate 2: (Tails, Heads). -/
def microstate2 : CoinPair := ⟨false, true⟩

/-! ### Properties -/

/-- Both microstates have 1 head. -/
theorem same_macrostate :
    numHeads microstate1 = numHeads microstate2 := by decide

/-- The two microstates are distinct. -/
theorem different_microstates :
    microstate1 ≠ microstate2 := by decide

/-- Microstate 2 is the swap of microstate 1. -/
theorem swap_related :
    swapCoins microstate1 = microstate2 := by decide

/-- Swapping coins preserves the macrostate for all configurations. -/
theorem swap_preserves_macrostate (c : CoinPair) :
    numHeads (swapCoins c) = numHeads c := by
  cases c with | mk c1 c2 =>
  cases c1 <;> cases c2 <;> decide

/-! ### ExplanationSystem construction -/

/-- The statistical mechanics explanation system.
    - `observe` = numHeads (macrostate, the coarse-grained observable)
    - `explain` = id (the full microstate is the "explanation")
    - `incompatible` = (≠)
    - `rashomon` = microstate1 and microstate2 (same macrostate, different microstates) -/
def statMechSystem : ExplanationSystem CoinPair CoinPair Nat where
  observe := numHeads
  explain := id
  incompatible := fun c₁ c₂ => c₁ ≠ c₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨microstate1, microstate2, same_macrostate, different_microstates⟩

/-! ### Impossibility -/

/-- **Statistical Mechanics Impossibility.**
    No description of a thermodynamic system can simultaneously be
    faithful (specify a particular microstate), stable (assign the same
    description to macrostate-equivalent configurations), and decisive
    (distinguish every pair of distinct microstates).

    Zero sorry, zero axioms — Rashomon is constructively witnessed. -/
theorem stat_mech_impossibility
    (E : CoinPair → CoinPair)
    (hf : faithful statMechSystem E)
    (hs : stable statMechSystem E)
    (hd : decisive statMechSystem E) : False :=
  explanation_impossibility statMechSystem E hf hs hd

end UniversalImpossibility
