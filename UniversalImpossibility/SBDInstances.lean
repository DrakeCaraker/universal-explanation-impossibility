/-
  SBD Instance Constructions and Abstract Aggregation.

  S51: Feature attribution as an SBD instance
  S52: Model selection as an SBD instance
  S58: Abstract aggregation problem definition

  Supplement: §Previous results as corollaries + §Abstract perspective
-/
import UniversalImpossibility.SymmetricBayes
import UniversalImpossibility.Trilemma
import UniversalImpossibility.ModelSelection

set_option autoImplicit false

namespace UniversalImpossibility

/-! ### S51: Feature attribution as SBD instance

  The feature attribution decision problem is an instance of the Symmetric
  Bayes Dichotomy:
  - Θ = {j > k, k > j} (two orderings, represented as Bool)
  - G = Z/2Z (swapping the ordering)
  - D = Model
  - optimal(f) = true iff φ_j(f) > φ_k(f)
  - orbit_reachable follows from the Rashomon property

  Rather than constructing a literal `SymmetricDecisionProblem` (which
  requires Group + MulAction instances for Bool that are fiddly in Lean 4),
  we prove that the SBD premises hold directly and that the impossibility
  follows from `sbd_infeasible`.
-/

/-- S51: The attribution impossibility is an instance of the SBD pattern.
    The SBD says: if two elements in the same orbit can both be optimal,
    a stable estimator is unfaithful to one. For attribution:
    - "j > k" and "k > j" are the two orbit elements
    - Models f (with φ_j > φ_k) and f' (with φ_k > φ_j) make each optimal
    - A stable ranking picks one ordering, so is unfaithful to the other model.

    We show both SBD premises hold: (1) the Rashomon diversity condition,
    and (2) the impossibility of a faithful stable complete ranking. -/
theorem attribution_is_sbd_instance
    (fs : FeatureSpace)
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k) :
    -- (1) Two distinct "optimal decisions" exist (Rashomon diversity)
    (∃ f f' : Model,
      attribution fs j f > attribution fs k f ∧
      attribution fs k f' > attribution fs j f') ∧
    -- (2) Any faithful stable complete ranking derives False
    (∀ ranking : Fin fs.P → Fin fs.P → Prop,
      (∀ f : Model, ranking j k ↔ attribution fs j f > attribution fs k f) →
      False) := by
  exact ⟨hrash ℓ j k hj hk hjk,
         fun ranking h => attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h⟩

/-- S51 corollary: the SBD unfaithfulness witness applies to attribution.
    Given two models that rank features j, k differently, any stable ranking
    is unfaithful to at least one of them. -/
theorem attribution_sbd_unfaithful
    (fs : FeatureSpace)
    (hrash : RashimonProperty fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    -- There exist two models witnessing unfaithfulness
    ∃ f f' : Model,
      attribution fs j f > attribution fs k f ∧
      attribution fs k f' > attribution fs j f' ∧
      -- The ranking is unfaithful to one of them
      (¬ (ranking j k ↔ attribution fs j f' > attribution fs k f') ∨
       ¬ (ranking j k ↔ attribution fs j f > attribution fs k f)) := by
  -- The ranking is faithful to all models by hypothesis, so derive False
  exfalso
  exact attribution_impossibility fs hrash ℓ j k hj hk hjk ranking h_faithful

/-! ### S52: Model selection as SBD instance

  The model selection decision problem is structurally identical:
  - Θ = {m₁ > m₂, m₂ > m₁}
  - G = Z/2Z
  - D = EvalInstance
  - optimal(d) = (m₁ > m₂ iff quality(m₁,d) > quality(m₂,d))
  - orbit_reachable follows from the Model Rashomon property
-/

/-- S52: The model selection impossibility is an instance of the SBD pattern. -/
theorem model_selection_is_sbd_instance
    (CandidateModel EvalInstance : Type)
    (quality : CandidateModel → EvalInstance → ℝ)
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂) :
    -- (1) Two distinct "optimal decisions" exist (Model Rashomon diversity)
    (∃ d d' : EvalInstance,
      quality m₁ d > quality m₂ d ∧
      quality m₂ d' > quality m₁ d') ∧
    -- (2) Any faithful stable complete ranking derives False
    (∀ ranking : CandidateModel → CandidateModel → Prop,
      (∀ d : EvalInstance, ranking m₁ m₂ ↔ quality m₁ d > quality m₂ d) →
      False) := by
  exact ⟨hrash m₁ m₂ hne,
         fun ranking h =>
           model_selection_impossibility CandidateModel EvalInstance quality
             hrash m₁ m₂ hne ranking h⟩

/-! ### S58: Abstract Aggregation Problem

  All three impossibilities (attribution, model selection, causal discovery)
  share a common structure:
  - A set of alternatives being ranked
  - A population of "instances" (models, datasets, etc.) that may rank them differently
  - The question: can we aggregate instance-level rankings into a stable ranking?

  The abstract aggregation problem captures this shared structure. The
  impossibility is a direct consequence of the diversity condition: if
  instances disagree on the ranking, no single aggregation can be faithful
  to all instances.
-/

/-- An abstract aggregation problem: a set of alternatives, a population of
    instances, a score function, and a diversity condition guaranteeing
    that instances disagree on the ranking of some pair. -/
structure AbstractAggregationProblem where
  /-- The set of alternatives being ranked -/
  Alternative : Type
  /-- The set of instances (models, datasets, etc.) -/
  Instance : Type
  /-- Instance-level quality/score for each alternative -/
  score : Alternative → Instance → ℝ
  /-- Diversity: for some pair of alternatives, instances disagree on the ranking -/
  diversity : ∃ (a₁ a₂ : Alternative) (d₁ d₂ : Instance),
    a₁ ≠ a₂ ∧ score a₁ d₁ > score a₂ d₁ ∧ score a₂ d₂ > score a₁ d₂

/-- Any abstract aggregation problem with diversity admits no faithful
    stable complete ranking of the diverse pair. -/
theorem abstract_aggregation_impossibility (P : AbstractAggregationProblem) :
    ∃ (a₁ a₂ : P.Alternative),
      a₁ ≠ a₂ ∧ ∀ (ranking : P.Alternative → P.Alternative → Prop),
        (∀ d : P.Instance, ranking a₁ a₂ ↔ P.score a₁ d > P.score a₂ d) →
        False := by
  obtain ⟨a₁, a₂, d₁, d₂, hne, h1, h2⟩ := P.diversity
  refine ⟨a₁, a₂, hne, fun ranking h_faith => ?_⟩
  have hr : ranking a₁ a₂ := (h_faith d₁).mpr h1
  have hcontra : P.score a₁ d₂ > P.score a₂ d₂ := (h_faith d₂).mp hr
  linarith

/-- The abstract aggregation impossibility also holds in the weak
    (implication-only) form: given faithfulness (→) and antisymmetry,
    completeness is impossible. -/
theorem abstract_aggregation_impossibility_weak (P : AbstractAggregationProblem) :
    ∃ (a₁ a₂ : P.Alternative),
      a₁ ≠ a₂ ∧ ∀ (ranking : P.Alternative → P.Alternative → Prop),
        (∀ d : P.Instance, P.score a₁ d > P.score a₂ d → ranking a₁ a₂) →
        (∀ d : P.Instance, P.score a₂ d > P.score a₁ d → ranking a₂ a₁) →
        ¬ (ranking a₁ a₂ ∧ ranking a₂ a₁) →
        ¬ (ranking a₁ a₂ ∨ ranking a₂ a₁) := by
  obtain ⟨a₁, a₂, d₁, d₂, hne, h1, h2⟩ := P.diversity
  refine ⟨a₁, a₂, hne, fun ranking hf₁₂ hf₂₁ hantisym hcomp => ?_⟩
  cases hcomp with
  | inl h12 => exact hantisym ⟨h12, hf₂₁ d₂ h2⟩
  | inr h21 => exact hantisym ⟨hf₁₂ d₁ h1, h21⟩

/-! ### Embedding: Attribution → Abstract Aggregation -/

/-- Feature attribution embeds into the abstract aggregation framework.
    Given a FeatureSpace and two features j, k in the same group with the
    Rashomon property, we construct an AbstractAggregationProblem. -/
noncomputable def attributionAsAggregation
    (fs : FeatureSpace)
    (j k : Fin fs.P) (ℓ : Fin fs.L)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (hrash : RashimonProperty fs) : AbstractAggregationProblem where
  Alternative := Fin fs.P
  Instance := Model
  score := fun i f => attribution fs i f
  diversity := by
    obtain ⟨f, f', h1, h2⟩ := hrash ℓ j k hj hk hjk
    exact ⟨j, k, f, f', hjk, h1, h2⟩

/-- Model selection embeds into the abstract aggregation framework. -/
noncomputable def modelSelectionAsAggregation
    (CandidateModel EvalInstance : Type)
    (quality : CandidateModel → EvalInstance → ℝ)
    (m₁ m₂ : CandidateModel) (hne : m₁ ≠ m₂)
    (hrash : ModelRashimonProperty CandidateModel EvalInstance quality) :
    AbstractAggregationProblem where
  Alternative := CandidateModel
  Instance := EvalInstance
  score := quality
  diversity := by
    obtain ⟨d, d', h1, h2⟩ := hrash m₁ m₂ hne
    exact ⟨m₁, m₂, d, d', hne, h1, h2⟩

end UniversalImpossibility
