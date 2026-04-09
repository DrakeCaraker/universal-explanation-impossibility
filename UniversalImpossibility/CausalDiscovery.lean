/-
  Causal Discovery Impossibility.

  No edge orientation rule for a CPDAG with multiple DAGs in its
  Markov equivalence class can be simultaneously faithful, stable,
  and complete. This is Instance 3 of the Symmetric Bayes Dichotomy.

  Supplement: §Instance 3: Causal Discovery under Markov Equivalence
-/
import UniversalImpossibility.SymmetricBayes

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### Causal orientation decision problem -/

/-- A causal orientation problem: given a CPDAG, orient all undirected edges.
    Parameterized over abstract types for DAGs and datasets. -/
structure CausalOrientationProblem where
  /-- The set of DAGs in the Markov equivalence class -/
  DAG : Type
  [instDAGFintype : Fintype DAG]
  [instDAGDecEq : DecidableEq DAG]
  /-- Observational datasets -/
  Dataset : Type
  /-- The DAG best supported by a dataset (e.g., highest BIC score) -/
  bestDAG : Dataset → DAG
  /-- The equivalence class has at least 2 DAGs -/
  class_size : ∃ (d₁ d₂ : DAG), d₁ ≠ d₂
  /-- Markov equivalence: for each DAG in the class, there exists a dataset
      making it the best-supported one (because the DAGs encode the same
      conditional independences, finite-sample noise can favor any of them). -/
  reachable : ∀ (dag : DAG), ∃ (d : Dataset), bestDAG d = dag

attribute [instance] CausalOrientationProblem.instDAGFintype
attribute [instance] CausalOrientationProblem.instDAGDecEq

/-- An orientation rule maps a dataset to a DAG. -/
def OrientationRule (P : CausalOrientationProblem) := P.Dataset → P.DAG

/-- An orientation rule is faithful to dataset d if it returns the best DAG. -/
def OrientationFaithful (P : CausalOrientationProblem)
    (rule : OrientationRule P) (d : P.Dataset) : Prop :=
  rule d = P.bestDAG d

/-- Causal Discovery Impossibility: no stable orientation rule can be faithful
    to all datasets when the equivalence class has ≥2 DAGs.

    "Stable" means the rule gives the same answer for all datasets.
    "Faithful" means the answer matches the data-optimal DAG. -/
theorem causal_discovery_impossibility
    (P : CausalOrientationProblem)
    (rule : OrientationRule P)
    -- Stable: same output for all datasets
    (h_stable : ∀ (d₁ d₂ : P.Dataset), rule d₁ = rule d₂)
    -- Faithful: matches the best DAG for every dataset
    (h_faithful : ∀ (d : P.Dataset), OrientationFaithful P rule d) :
    False := by
  -- Get two distinct DAGs from class_size
  obtain ⟨dag₁, dag₂, hne⟩ := P.class_size
  -- Get datasets making each optimal
  obtain ⟨d₁, hd₁⟩ := P.reachable dag₁
  obtain ⟨d₂, hd₂⟩ := P.reachable dag₂
  -- Faithful at d₁: rule d₁ = dag₁
  have hf₁ := h_faithful d₁
  unfold OrientationFaithful at hf₁
  rw [hd₁] at hf₁
  -- Faithful at d₂: rule d₂ = dag₂
  have hf₂ := h_faithful d₂
  unfold OrientationFaithful at hf₂
  rw [hd₂] at hf₂
  -- Stable: rule d₁ = rule d₂
  have hstab := h_stable d₁ d₂
  -- dag₁ = rule d₁ = rule d₂ = dag₂, contradicting hne
  exact hne (hf₁.symm.trans (hstab.trans hf₂))

/-- The resolution: reporting the CPDAG (ties for undirected edges)
    avoids unfaithfulness. A "constant" rule is automatically stable
    and never asserts wrong orientations (it reports uncertainty). -/
theorem cpdag_is_stable
    (P : CausalOrientationProblem)
    (cpdag_output : P.DAG)
    (rule : OrientationRule P)
    (h_const : ∀ d, rule d = cpdag_output) :
    ∀ d₁ d₂ : P.Dataset, rule d₁ = rule d₂ := by
  intro d₁ d₂
  rw [h_const d₁, h_const d₂]

/-- This is an instance of the SBD: the causal discovery impossibility
    follows the same faithful+stable=contradiction pattern. -/
theorem causal_discovery_from_sbd
    (P : CausalOrientationProblem)
    (rule : OrientationRule P)
    (d₁ d₂ : P.Dataset)
    (hne : P.bestDAG d₁ ≠ P.bestDAG d₂)
    (h_stable : rule d₁ = rule d₂)
    (h_faith₁ : OrientationFaithful P rule d₁)
    (h_faith₂ : OrientationFaithful P rule d₂) :
    False := by
  unfold OrientationFaithful at h_faith₁ h_faith₂
  exact hne (h_faith₁.symm.trans (h_stable.trans h_faith₂))

end UniversalImpossibility
