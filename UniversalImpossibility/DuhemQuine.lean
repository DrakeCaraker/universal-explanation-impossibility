import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Duhem-Quine Thesis — Derived Rashomon Property

The Duhem-Quine thesis (Duhem 1906, Quine 1951): no scientific hypothesis can
be tested in isolation. Any experimental test of hypothesis H depends on
auxiliary hypotheses A₁, A₂, ... When the experiment fails, we cannot determine
whether H is wrong or one of the auxiliaries is wrong.

This is a natural instance of the Rashomon property: two different
hypothesis-sets (configurations) produce the same observable output
(experimental failure) but have incompatible explanations (different blame
assignments — which hypothesis is responsible for the failure).

## The Mapping

- Θ (configurations) = hypothesis-sets (main hypothesis H + auxiliary A)
- Y (observables) = experimental outcomes (success or failure)
- H (explanations) = blame assignments (which hypothesis is at fault)
- obs(θ) = the experimental outcome produced by hypothesis-set θ
- explain(θ) = the blame assignment for θ
- incompatible(b₁, b₂) = b₁ ≠ b₂ (different blame assignments are incompatible)

## Minimal Witness

- Config 1: H true, A false → failure (A is to blame)
- Config 2: H false, A true → failure (H is to blame)
- Same outcome (failure), incompatible explanations (blame A vs blame H)
-/

/-- A hypothesis can be correct or incorrect. -/
inductive HypStatus where
  | correct : HypStatus
  | incorrect : HypStatus
  deriving DecidableEq, Repr

/-- A hypothesis-set: main hypothesis H and auxiliary hypothesis A. -/
structure HypothesisSet where
  mainH : HypStatus
  auxA : HypStatus
  deriving DecidableEq, Repr

/-- Experimental outcome: success or failure. -/
inductive ExpOutcome where
  | success : ExpOutcome
  | failure : ExpOutcome
  deriving DecidableEq, Repr

/-- Which hypothesis is to blame for the failure. -/
inductive Blame where
  | blameMain : Blame   -- the main hypothesis H is at fault
  | blameAux : Blame    -- the auxiliary hypothesis A is at fault
  | noBlame : Blame     -- no one is at fault (experiment succeeded)
  deriving DecidableEq, Repr

/-- The observation map: experimental outcome of a hypothesis-set.
    Success requires both hypotheses to be correct; failure otherwise. -/
def experimentOutcome : HypothesisSet → ExpOutcome
  | ⟨.correct, .correct⟩ => .success
  | ⟨.correct, .incorrect⟩ => .failure
  | ⟨.incorrect, .correct⟩ => .failure
  | ⟨.incorrect, .incorrect⟩ => .failure

/-- The explanation map: blame assignment.
    When the experiment fails, blame is assigned to the incorrect hypothesis.
    When both are incorrect, blame the main hypothesis (arbitrary convention). -/
def blameAssignment : HypothesisSet → Blame
  | ⟨.correct, .correct⟩ => .noBlame
  | ⟨.correct, .incorrect⟩ => .blameAux
  | ⟨.incorrect, .correct⟩ => .blameMain
  | ⟨.incorrect, .incorrect⟩ => .blameMain

/-- The Duhem-Quine witness, configuration 1: H correct, A incorrect.
    Experiment fails because A is wrong. -/
def dqConfig1 : HypothesisSet := ⟨.correct, .incorrect⟩

/-- The Duhem-Quine witness, configuration 2: H incorrect, A correct.
    Experiment fails because H is wrong. -/
def dqConfig2 : HypothesisSet := ⟨.incorrect, .correct⟩

/-- Both configurations produce experimental failure. -/
theorem dq_same_outcome :
    experimentOutcome dqConfig1 = experimentOutcome dqConfig2 := by
  decide

/-- The blame assignments are different: blameAux ≠ blameMain. -/
theorem dq_blame_different :
    blameAssignment dqConfig1 ≠ blameAssignment dqConfig2 := by
  decide

/-- The Duhem-Quine thesis as an ExplanationSystem.
    - Θ = HypothesisSet (hypothesis-sets)
    - H = Blame (blame assignments)
    - Y = ExpOutcome (experimental outcomes)
    - observe = experimentOutcome
    - explain = blameAssignment
    - incompatible = (≠) -/
def duhemQuineSystem : ExplanationSystem HypothesisSet Blame ExpOutcome where
  observe := experimentOutcome
  explain := blameAssignment
  incompatible := fun b₁ b₂ => b₁ ≠ b₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨dqConfig1, dqConfig2, dq_same_outcome, dq_blame_different⟩

/-- **Duhem-Quine Impossibility.**

No method of interpreting experimental failure can simultaneously be:
- **Faithful**: correctly identifies which hypothesis is wrong
- **Stable**: gives the same blame assignment for all hypothesis-sets that
  produce the same experimental outcome
- **Decisive**: commits to a specific blame assignment

This IS the Duhem-Quine thesis: from the experimental outcome alone, you
cannot determine which hypothesis is responsible for the failure. -/
theorem duhem_quine_impossibility
    (E : HypothesisSet → Blame)
    (hf : faithful duhemQuineSystem E)
    (hs : stable duhemQuineSystem E)
    (hd : decisive duhemQuineSystem E) : False :=
  explanation_impossibility duhemQuineSystem E hf hs hd
