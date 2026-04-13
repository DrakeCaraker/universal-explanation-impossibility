import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# View Update Problem — Derived Rashomon Property

The view update problem in databases: projecting to a single column loses
information. Two rows (true, true) and (true, false) both project to the
same view (colA = true), but they are different rows. This is a concrete
instance of the Rashomon property arising from information loss in database
views.
-/

/-- A database row with two Boolean columns. -/
structure Row where
  colA : Bool
  colB : Bool
  deriving DecidableEq, Repr

/-- The view function: projects a row to column A. -/
def view (r : Row) : Bool := r.colA

/-- First row: (true, true). -/
def row1 : Row := ⟨true, true⟩

/-- Second row: (true, false). -/
def row2 : Row := ⟨true, false⟩

/-- Both rows have the same view (colA = true). -/
theorem same_view : view row1 = view row2 := by
  decide

/-- The two rows are different. -/
theorem different_rows : row1 ≠ row2 := by
  decide

/-- The view update system as an ExplanationSystem.
    - Θ = Row (database rows)
    - H = Row (explanations: the full row)
    - Y = Bool (observables: the projected view)
    - observe = view
    - explain = id
    - incompatible = (≠) -/
def viewUpdateSystem : ExplanationSystem Row Row Bool where
  observe := view
  explain := id
  incompatible := fun r₁ r₂ => r₁ ≠ r₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨row1, row2, same_view, different_rows⟩

/-- **View Update Impossibility.**
    No explanation of a lossy database view can be simultaneously faithful,
    stable, and decisive. -/
theorem view_update_impossibility
    (E : Row → Row)
    (hf : faithful viewUpdateSystem E)
    (hs : stable viewUpdateSystem E)
    (hd : decisive viewUpdateSystem E) : False :=
  explanation_impossibility viewUpdateSystem E hf hs hd
