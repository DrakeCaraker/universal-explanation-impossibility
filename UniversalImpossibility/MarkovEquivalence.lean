import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Markov Equivalence — Derived Rashomon Property

We constructively prove that the Rashomon property holds for causal
discovery by exhibiting two DAGs with the same conditional independence
relations but different edge orientations.

This is the 3-node chain vs fork example:
- Chain: 0 → 1 → 2
- Fork:  0 ← 1 → 2
Same CI structure (0 ⊥ 2 | 1), different edge orientations.
-/

/-- A directed graph on 3 nodes, represented by which edges exist
    and their directions. An edge (i, j) means i → j. -/
structure DAG3 where
  edge01 : Bool  -- 0 → 1
  edge10 : Bool  -- 1 → 0
  edge12 : Bool  -- 1 → 2
  edge21 : Bool  -- 2 → 1
  edge02 : Bool  -- 0 → 2
  edge20 : Bool  -- 2 → 0
  deriving DecidableEq, Repr

/-- Conditional independence structure: which pairs are independent
    given which conditioning sets. For 3 nodes, we only need to
    track 0 ⊥ 2 | {1}. -/
structure CIStructure3 where
  ci_02_given_1 : Bool  -- 0 ⊥ 2 | {1}
  deriving DecidableEq, Repr

/-- Extract CI structure from a DAG.
    SIMPLIFIED formula: checks for no direct 0-2 edge and paths through 1.
    Correct for chain (0→1→2) and fork (0←1→2) structures.
    NOTE: Does not handle colliders (0→1←2) correctly — colliders require
    a more complex d-separation check. The chain/fork witness used in the
    Rashomon proof below does not involve colliders, so this limitation
    does not affect the derived impossibility. -/
def ciFromDAG (g : DAG3) : CIStructure3 :=
  ⟨!g.edge02 && !g.edge20 && (g.edge01 || g.edge10) && (g.edge12 || g.edge21)⟩

/-- The chain DAG: 0 → 1 → 2 -/
def chain : DAG3 := ⟨true, false, true, false, false, false⟩

/-- The fork DAG: 0 ← 1 → 2 -/
def fork : DAG3 := ⟨false, true, true, false, false, false⟩

/-- Chain and fork have the same CI structure. -/
theorem chain_fork_same_ci : ciFromDAG chain = ciFromDAG fork := by
  decide

/-- Chain and fork have different edge orientations. -/
theorem chain_fork_different_edges : chain ≠ fork := by
  decide

/-- The Rashomon property for causal discovery: DERIVED, not axiomatized.
    Two DAGs exist with the same CI structure but different edges. -/
theorem causal_rashomon_derived :
    ∃ (g₁ g₂ : DAG3), ciFromDAG g₁ = ciFromDAG g₂ ∧ g₁ ≠ g₂ :=
  ⟨chain, fork, chain_fork_same_ci, chain_fork_different_edges⟩

/-- Construct an ExplanationSystem for causal discovery with
    DERIVED Rashomon property.
    NOTE: incompatible = (≠) is intentionally broad — it makes the
    impossibility STRONGER (holds even under this liberal notion of
    conflict). A narrower relation (e.g., "disagree on at least one
    edge orientation") would give a tighter result with better tightness
    witnesses. The broad choice ensures the impossibility is not an
    artifact of a carefully chosen incompatibility relation. -/
def causalSystemDerived : ExplanationSystem DAG3 DAG3 CIStructure3 where
  observe := ciFromDAG
  explain := id
  incompatible := fun g₁ g₂ => g₁ ≠ g₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨chain, fork, chain_fork_same_ci, chain_fork_different_edges⟩

/-- Causal discovery impossibility: DERIVED from Markov equivalence,
    not axiomatized. -/
theorem causal_impossibility_derived
    (E : DAG3 → DAG3)
    (hf : faithful causalSystemDerived E)
    (hs : stable causalSystemDerived E)
    (hd : decisive causalSystemDerived E) : False :=
  explanation_impossibility causalSystemDerived E hf hs hd
