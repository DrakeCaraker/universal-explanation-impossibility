import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility

/-!
# Quantum Measurement as a Scientific Revolution

The measurement problem in quantum mechanics exhibits the bilemma pattern:

Pre-revolution: H = {definite_value, superposition} -- maximally incompatible
Rashomon: same measurement statistics, incompatible ontological claims
Bilemma: no interpretation is simultaneously faithful AND stable
Enrichment: Copenhagen's "no fact of the matter" = neutral element
Decisiveness sacrificed: can't say what happens between measurements

This formalizes the PARADIGM SHIFT structure, not the physics content.
The physics (Hilbert spaces, Born rule) is in the Rashomon witness grounding.

**Caveat:** The singleton observable space (both preparations map to `bornRuleStats`)
makes the Rashomon property vacuously true. This formalizes the structure of the
interpretive disagreement, not the physics of quantum measurement. Resolutions that
change the configuration space (Bohmian mechanics) or the observable space
(Many-Worlds) are system changes outside the bilemma's scope.

## The Mapping

- Theta (configurations) = preparation procedures
- Y (observables) = measurement statistics (Born rule probabilities)
- H (explanations) = ontological claims about the system's state
- observe = measurement statistics
- explain = ontological claim the preparation suggests
- incompatible = (ne)
- Rashomon: eigenstate prep and superposition prep yield the same Born rule
  statistics but incompatible ontological claims

## Minimal Witness -- Stern-Gerlach on spin-1/2

- Eigenstate preparation: prepare in |up>, measure spin-z. Realism: the
  particle had definite spin-up before measurement.
- Superposition preparation: prepare in |+>, measure spin-z. QM: the
  particle was in superposition; measurement outcome is probabilistic.
- Both predict: spin-up with P = |<up|state>|^2 (Born rule).
- But: "definite value" != "superposition" (incompatible explanations).
-/

set_option autoImplicit false

/-- Two ontological claims about a quantum system. -/
inductive QuantumOntology where
  | definiteValue   -- the system has a definite value (realism)
  | superposition   -- the system is in superposition (no definite value)
  deriving DecidableEq, Repr

/-- Measurement statistics (the observable). Both ontologies predict
    the same statistics for appropriately prepared states. -/
inductive MeasurementStats where
  | bornRuleStats   -- P(outcome) = |<outcome|state>|^2
  deriving DecidableEq, Repr

/-- Two preparations: one where realism seems natural (eigenstate),
    one where superposition seems natural (superposition state).
    Both produce the same measurement statistics on a specific observable. -/
inductive Preparation where
  | eigenstatePrep    -- prepare in eigenstate of measurement basis
  | superpositionPrep -- prepare in superposition, then measure
  deriving DecidableEq, Repr

def qmRevObserve : Preparation → MeasurementStats
  | _ => .bornRuleStats  -- both give Born rule statistics

def qmRevExplain : Preparation → QuantumOntology
  | .eigenstatePrep => .definiteValue      -- eigenstate -> definite value
  | .superpositionPrep => .superposition   -- superposition -> no definite value

/-- The quantum measurement problem as an ExplanationSystem.
    - Theta = Preparation (eigenstate, superposition)
    - H = QuantumOntology (definiteValue, superposition)
    - Y = MeasurementStats (bornRuleStats)
    - observe = qmRevObserve (constant -- same stats)
    - explain = qmRevExplain (ontological claim)
    - incompatible = (ne) -/
def quantumMeasurementSystem : ExplanationSystem Preparation QuantumOntology MeasurementStats where
  observe := qmRevObserve
  explain := qmRevExplain
  incompatible := (· ≠ ·)
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨.eigenstatePrep, .superpositionPrep, rfl, by decide⟩

/-- QuantumOntology with incompatible = (ne) is maximally incompatible.
    Binary H with DecidableEq and incompatible = ne is always maximally
    incompatible: not(a ne b) is not-not(a = b), which gives a = b
    by double negation elimination. -/
theorem quantum_ontology_maxIncompat : maximallyIncompatible quantumMeasurementSystem :=
  fun _ _ hc => Classical.byContradiction (fun hne => hc hne)

/-- **The Quantum Measurement Bilemma.**

No interpretation of the quantum measurement problem is simultaneously
faithful (matches the ontological claim of each preparation) and stable
(gives the same ontology for preparations with the same statistics).

This is the formal content of the measurement problem: realism (definite
values) is faithful to eigenstate preparations but unstable across
equivalent preparations. Copenhagen resolves this via enrichment: adding
"no fact of the matter" as a neutral element, sacrificing decisiveness
(can't say what happens between measurements). -/
theorem quantum_measurement_revolution
    (E : Preparation → QuantumOntology)
    (hf : faithful quantumMeasurementSystem E)
    (hs : stable quantumMeasurementSystem E) : False :=
  bilemma quantumMeasurementSystem quantum_ontology_maxIncompat E hf hs
