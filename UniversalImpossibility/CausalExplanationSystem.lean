import UniversalImpossibility.MarkovEquivalence

set_option autoImplicit false

/-!
# Causal Discovery — Instance of Universal Explanation Impossibility

Causal discovery exhibits the Rashomon property via Markov equivalence:
the chain DAG (0 → 1 → 2) and the fork DAG (0 ← 1 → 2) induce the
same conditional independence structure (0 ⊥ 2 | {1}) but have
incompatible edge orientations.

This file constructs an `ExplanationSystem` instance using the concrete
types from `MarkovEquivalence.lean` and derives the impossibility as a
direct corollary of the universal meta-theorem.
-/

/-- The causal discovery explanation system, instantiated from
    Markov equivalence first principles.
    - Θ = DAG3 (3-node directed acyclic graphs)
    - H = DAG3 (the DAG itself is the explanation)
    - Y = CIStructure3 (conditional independence structure)
    - observe = ciFromDAG (extract CI relations from DAG)
    - explain = id (the DAG is its own explanation)
    - incompatible = (≠) (any structural disagreement counts)
    - rashomon = chain vs fork (same CI, different edges) -/
def causalExplanationSystem : ExplanationSystem DAG3 DAG3 CIStructure3 where
  observe := ciFromDAG
  explain := id
  incompatible := fun g₁ g₂ => g₁ ≠ g₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨chain, fork, chain_fork_same_ci, by decide⟩

/-- **Causal Discovery Explanation Impossibility.**
    No causal explanation method can simultaneously be faithful,
    stable, and decisive when Markov equivalence holds.

    Derived as a direct corollary of `explanation_impossibility`. -/
theorem causal_explanation_impossibility
    (E : DAG3 → DAG3)
    (hf : faithful causalExplanationSystem E)
    (hs : stable causalExplanationSystem E)
    (hd : decisive causalExplanationSystem E) : False :=
  explanation_impossibility causalExplanationSystem E hf hs hd
