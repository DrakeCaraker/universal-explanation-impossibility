import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Linear System Underspecification — Derived Rashomon Property

The linear system Ax = b with A = [1, 1] and b = 2 has multiple solutions:
x₁ = (1, 1) and x₂ = (0, 2). Both produce the same output (dot product = 2)
but are different solution vectors. This is a concrete instance of the
Rashomon property in numerical linear algebra.
-/

/-- A 2D integer vector. -/
structure Vec2 where
  x : Int
  y : Int
  deriving DecidableEq, Repr

/-- Dot product with A = [1, 1]: computes v.x + v.y. -/
def dotA (v : Vec2) : Int := v.x + v.y

/-- First solution: x₁ = (1, 1). -/
def sol1 : Vec2 := ⟨1, 1⟩

/-- Second solution: x₂ = (0, 2). -/
def sol2 : Vec2 := ⟨0, 2⟩

/-- Both solutions produce the same output: 1+1 = 0+2 = 2. -/
theorem same_output : dotA sol1 = dotA sol2 := by
  native_decide

/-- The two solutions are different vectors. -/
theorem different_solutions : sol1 ≠ sol2 := by
  decide

/-- The linear system as an ExplanationSystem.
    - Θ = Vec2 (solution vectors)
    - H = Vec2 (explanations: the solution itself)
    - Y = Int (observables: the dot product)
    - observe = dotA
    - explain = id
    - incompatible = (≠) -/
def linearSystem : ExplanationSystem Vec2 Vec2 Int where
  observe := dotA
  explain := id
  incompatible := fun v₁ v₂ => v₁ ≠ v₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨sol1, sol2, same_output, different_solutions⟩

/-- **Linear System Impossibility.**
    No explanation of an underspecified linear system can be simultaneously
    faithful, stable, and decisive. -/
theorem linear_system_impossibility
    (E : Vec2 → Vec2)
    (hf : faithful linearSystem E)
    (hs : stable linearSystem E)
    (hd : decisive linearSystem E) : False :=
  explanation_impossibility linearSystem E hf hs hd
