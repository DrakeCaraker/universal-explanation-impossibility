import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-! # Causal Discovery Impossibility — Constructive Instance

Chain A→B→C and Fork A←B→C share conditional independence structure
but have incompatible edge orientations. -/

inductive CausalConfigC where
  | chain : CausalConfigC  -- A → B → C
  | fork  : CausalConfigC  -- A ← B → C
  deriving DecidableEq, Repr

inductive CIStructure where
  | ci : CIStructure  -- A ⊥ C | B
  deriving DecidableEq, Repr

inductive EdgeOrientation where
  | aToB : EdgeOrientation  -- A → B (chain)
  | bToA : EdgeOrientation  -- B → A (fork)
  deriving DecidableEq, Repr

def causalObserveC : CausalConfigC → CIStructure
  | CausalConfigC.chain => CIStructure.ci
  | CausalConfigC.fork => CIStructure.ci

def causalExplainC : CausalConfigC → EdgeOrientation
  | CausalConfigC.chain => EdgeOrientation.aToB
  | CausalConfigC.fork => EdgeOrientation.bToA

def causalIncompC : EdgeOrientation → EdgeOrientation → Prop
  | EdgeOrientation.aToB, EdgeOrientation.bToA => True
  | EdgeOrientation.bToA, EdgeOrientation.aToB => True
  | _, _ => False

instance : DecidableRel causalIncompC := fun a b => by
  cases a <;> cases b <;> simp [causalIncompC] <;> exact inferInstance

theorem causalIncompC_irrefl (h : EdgeOrientation) : ¬causalIncompC h h := by
  cases h <;> simp [causalIncompC]

def causalSystemC : ExplanationSystem CausalConfigC EdgeOrientation CIStructure :=
  ⟨causalObserveC, causalExplainC, causalIncompC, causalIncompC_irrefl,
   CausalConfigC.chain, CausalConfigC.fork, rfl, trivial⟩

theorem causal_impossibility_constructive :
    ∀ (E : CausalConfigC → EdgeOrientation),
      faithful causalSystemC E → stable causalSystemC E → decisive causalSystemC E → False :=
  explanation_impossibility causalSystemC
