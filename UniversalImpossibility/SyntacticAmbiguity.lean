import UniversalImpossibility.ExplanationSystem

/-!
# Syntactic Ambiguity — Derived Rashomon Property

The classic NLP ambiguity: "V NP PP" has two parse trees (bracketings).
Left-attach: [V [NP PP]] — the PP modifies the NP.
Right-attach: [[V NP] PP] — the PP modifies the verb.

Both bracketings yield the same surface string [V, NP, PP], but they
assign different syntactic structures. This is a Rashomon witness:
the observable (token sequence) is the same, but the explanations
(parse trees) are incompatible.
-/

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Token and bracketing types -/

/-- Tokens in a simple "V NP PP" sentence. -/
inductive Token where
  | V  : Token
  | NP : Token
  | PP : Token
  deriving DecidableEq, Repr

/-- Two bracketings of "V NP PP".
    - leftAttach:  [V [NP PP]] — PP attaches to NP
    - rightAttach: [[V NP] PP] — PP attaches to V -/
inductive Bracketing where
  | leftAttach  : Bracketing
  | rightAttach : Bracketing
  deriving DecidableEq, Repr

/-! ### Yield function -/

/-- The yield (surface token sequence) of a bracketing.
    Both bracketings produce the same token sequence [V, NP, PP]. -/
def yield : Bracketing → List Token
  | Bracketing.leftAttach  => [Token.V, Token.NP, Token.PP]
  | Bracketing.rightAttach => [Token.V, Token.NP, Token.PP]

/-! ### Properties -/

/-- Both bracketings have the same yield. -/
theorem same_yield : yield Bracketing.leftAttach = yield Bracketing.rightAttach := by
  rfl

/-- The two bracketings are different. -/
theorem different_bracketings : Bracketing.leftAttach ≠ Bracketing.rightAttach := by
  decide

/-! ### ExplanationSystem construction -/

/-- The syntactic ambiguity explanation system.
    - `observe` = yield (surface token sequence)
    - `explain` = id (the bracketing itself is the "explanation")
    - `incompatible` = (≠)
    - `rashomon` = leftAttach and rightAttach (same yield, different trees) -/
def syntaxSystem : ExplanationSystem Bracketing Bracketing (List Token) where
  observe := yield
  explain := id
  incompatible := fun b₁ b₂ => b₁ ≠ b₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨Bracketing.leftAttach, Bracketing.rightAttach,
               same_yield, different_bracketings⟩

/-! ### Impossibility -/

/-- **Syntactic Ambiguity Impossibility.**
    No syntactic explanation can simultaneously be faithful (report the
    actual parse tree), stable (assign the same tree to sentences with
    the same token sequence), and decisive (distinguish different trees).

    Zero sorry, zero axioms — Rashomon is constructively witnessed
    by the two bracketings of "V NP PP". -/
theorem syntactic_ambiguity_impossibility
    (E : Bracketing → Bracketing)
    (hf : faithful syntaxSystem E)
    (hs : stable syntaxSystem E)
    (hd : decisive syntaxSystem E) : False :=
  explanation_impossibility syntaxSystem E hf hs hd

end UniversalImpossibility
