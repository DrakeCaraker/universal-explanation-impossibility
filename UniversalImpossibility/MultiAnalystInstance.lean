/-
  MultiAnalystInstance.lean — Multi-analyst scientific aggregation as an
  ExplanationSystem instance.

  Formalizes the NARPS/Silberzahn/Breznau multi-analyst setting where
  multiple valid analysis pipelines produce incompatible scientific
  conclusions from the same data.

  This makes the multi-analyst connection FORMAL (Lean-verified), not
  merely structural. The Rashomon property is constructively witnessed.

  Zero axioms, zero sorry.
-/

import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility

set_option autoImplicit false

namespace UniversalImpossibility

/-! ## Multi-analyst explanation system

Configurations = analysis pipelines (preprocessing, model, thresholds)
Observables = statistical test outputs (p-values, effect sizes)
Explanations = scientific conclusions (which hypothesis is supported)
Incompatibility = contradictory conclusions (support vs reject) -/

/-- Scientific conclusions: binary (support or reject the hypothesis). -/
inductive ScientificConclusion where
  | support : ScientificConclusion
  | reject : ScientificConclusion
deriving DecidableEq

/-- Incompatibility: support and reject are incompatible. -/
def ScientificConclusion.incompatible : ScientificConclusion → ScientificConclusion → Prop
  | .support, .reject => True
  | .reject, .support => True
  | _, _ => False

instance : DecidableRel ScientificConclusion.incompatible := by
  intro a b; cases a <;> cases b <;> simp [ScientificConclusion.incompatible] <;> infer_instance

/-- Incompatibility is irreflexive. -/
theorem scientificConclusion_irrefl (h : ScientificConclusion) :
    ¬ScientificConclusion.incompatible h h := by
  cases h <;> simp [ScientificConclusion.incompatible]

/-- Two analysis pipelines. -/
inductive AnalysisPipeline where
  | pipelineA : AnalysisPipeline  -- e.g., FSL with standard preprocessing
  | pipelineB : AnalysisPipeline  -- e.g., SPM with different preprocessing
deriving DecidableEq

/-- Both pipelines produce the same statistical output (same data, valid methods). -/
def multiAnalyst_observe : AnalysisPipeline → Unit
  | _ => ()

/-- But they reach different scientific conclusions.
    This is the NARPS finding: different valid pipelines → different conclusions. -/
def multiAnalyst_explain : AnalysisPipeline → ScientificConclusion
  | .pipelineA => .support
  | .pipelineB => .reject

/-- The Rashomon property: two pipelines agree on observables but
    disagree on conclusions. Constructively witnessed. -/
theorem multiAnalyst_rashomon :
    ∃ θ₁ θ₂ : AnalysisPipeline,
      multiAnalyst_observe θ₁ = multiAnalyst_observe θ₂ ∧
      ScientificConclusion.incompatible
        (multiAnalyst_explain θ₁) (multiAnalyst_explain θ₂) := by
  exact ⟨.pipelineA, .pipelineB, rfl, trivial⟩

/-- The multi-analyst explanation system. -/
noncomputable def multiAnalystSystem :
    ExplanationSystem AnalysisPipeline ScientificConclusion Unit :=
  { observe := multiAnalyst_observe
    explain := multiAnalyst_explain
    incompatible := ScientificConclusion.incompatible
    incompatible_irrefl := scientificConclusion_irrefl
    rashomon := multiAnalyst_rashomon }

/-- **Multi-analyst impossibility.**
    No method for aggregating scientific conclusions can be simultaneously
    faithful, stable, and decisive when valid analysis pipelines disagree.

    This is the formal version of what NARPS (70 teams),
    Silberzahn (29 teams), and Breznau (73 teams) demonstrate
    empirically. The orbit average (consensus across pipelines) is
    the Pareto-optimal resolution. -/
theorem multiAnalyst_impossibility
    (E : AnalysisPipeline → ScientificConclusion)
    (hf : faithful multiAnalystSystem E)
    (hs : stable multiAnalystSystem E)
    (hd : decisive multiAnalystSystem E) :
    False :=
  explanation_impossibility multiAnalystSystem E hf hs hd

/-- The multi-analyst system is maximally incompatible (binary conclusions).
    Therefore the bilemma applies: even faithful + stable is impossible. -/
theorem multiAnalyst_maxIncompat (c₁ c₂ : ScientificConclusion) :
    ¬ScientificConclusion.incompatible c₁ c₂ → c₁ = c₂ := by
  cases c₁ <;> cases c₂ <;> simp [ScientificConclusion.incompatible]

/-- **Multi-analyst bilemma.**
    For binary scientific conclusions (support/reject), even faithful + stable
    alone is impossible. No aggregation can avoid both unfaithfulness to some
    valid pipeline and instability across equivalent pipelines. -/
theorem multiAnalyst_bilemma
    (E : AnalysisPipeline → ScientificConclusion)
    (hf : faithful multiAnalystSystem E)
    (hs : stable multiAnalystSystem E) :
    False :=
  bilemma multiAnalystSystem multiAnalyst_maxIncompat E hf hs

end UniversalImpossibility
