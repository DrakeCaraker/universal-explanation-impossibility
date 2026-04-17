import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-! # Attribution Impossibility — Constructive Instance

Two models that produce the same predictions but rank features oppositely.
All types are finite inductives; the Rashomon property is derived via `decide`. -/

inductive AttrConfigC where
  | modelA : AttrConfigC  -- ranks feature j > k
  | modelB : AttrConfigC  -- ranks feature k > j
  deriving DecidableEq, Repr

inductive AttrPrediction where
  | pred : AttrPrediction
  deriving DecidableEq, Repr

inductive AttrRanking where
  | jAboveK : AttrRanking
  | kAboveJ : AttrRanking
  deriving DecidableEq, Repr

def attrObserveC : AttrConfigC → AttrPrediction
  | AttrConfigC.modelA => AttrPrediction.pred
  | AttrConfigC.modelB => AttrPrediction.pred

def attrExplainC : AttrConfigC → AttrRanking
  | AttrConfigC.modelA => AttrRanking.jAboveK
  | AttrConfigC.modelB => AttrRanking.kAboveJ

def attrIncompC : AttrRanking → AttrRanking → Prop
  | AttrRanking.jAboveK, AttrRanking.kAboveJ => True
  | AttrRanking.kAboveJ, AttrRanking.jAboveK => True
  | _, _ => False

instance : DecidableRel attrIncompC := fun a b => by
  cases a <;> cases b <;> simp [attrIncompC] <;> exact inferInstance

theorem attrIncompC_irrefl (h : AttrRanking) : ¬attrIncompC h h := by
  cases h <;> simp [attrIncompC]

def attrSystemC : ExplanationSystem AttrConfigC AttrRanking AttrPrediction :=
  ⟨attrObserveC, attrExplainC, attrIncompC, attrIncompC_irrefl,
   AttrConfigC.modelA, AttrConfigC.modelB, rfl, trivial⟩

theorem attribution_impossibility_constructive :
    ∀ (E : AttrConfigC → AttrRanking),
      faithful attrSystemC E → stable attrSystemC E → decisive attrSystemC E → False :=
  explanation_impossibility attrSystemC
