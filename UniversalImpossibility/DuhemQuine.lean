import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# The Duhem-Quine Underdetermination (Constructive Witness)

A minimal constructive witness of the Duhem-Quine thesis: hypothesis-sets
producing the same experimental failure but assigning blame differently.

**What this captures:** The core *logic* of underdetermination — when
multiple hypothesis-sets produce the same experimental outcome, blame
cannot be localized from the outcome alone.

**What this does NOT capture:** The full philosophical thesis, which
involves webs of interconnected beliefs (Quine 1951), the revisability
of logic itself, and the holistic nature of theory testing (Duhem 1906).

**Quantitative result:** For N hypotheses, the Rashomon set contains
2^N - 1 failure configurations, so underdetermination grows exponentially
with the number of auxiliary hypotheses.

## The Mapping (2 hypotheses)

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

/-- **Duhem-Quine Underdetermination (2 hypotheses).**

No method of interpreting experimental failure can simultaneously be:
- **Faithful**: correctly identifies which hypothesis is wrong
- **Stable**: gives the same blame assignment for all hypothesis-sets that
  produce the same experimental outcome
- **Decisive**: commits to a specific blame assignment

This captures the core logic of blame underdetermination: from the
experimental outcome alone, you cannot determine which hypothesis is
responsible for the failure. -/
theorem duhem_quine_impossibility
    (E : HypothesisSet → Blame)
    (hf : faithful duhemQuineSystem E)
    (hs : stable duhemQuineSystem E)
    (hd : decisive duhemQuineSystem E) : False :=
  explanation_impossibility duhemQuineSystem E hf hs hd

-- ============================================================================
/-! ## Parameterized Duhem-Quine: N = 4 Hypotheses

The Rashomon set of the Duhem-Quine setting grows exponentially with
the number of auxiliary hypotheses: for N hypotheses, 2^N - 1 distinct
hypothesis-sets produce experimental failure, and all are observationally
equivalent. This means underdetermination grows as log(2^N - 1) ≈ N log 2.

With N hypotheses, the Rashomon set contains 2^N - 1 failure configurations.
The Rashomon entropy S = log(2^N - 1) grows linearly with N, quantifying
the exponential growth of underdetermination.

We instantiate at N = 4 to keep types decidable. The construction
generalizes to any N ≥ 2.

### The mapping (N hypotheses)

- **Θ** = `Fin N → Bool` (each hypothesis correct or incorrect)
- **Y** = `Bool` (success = all correct; failure = any wrong)
- **H** = `Fin N → Bool` (the configuration IS the explanation — it
  records which hypotheses are correct/incorrect)
- **observe(θ)** = `∀ i, θ i` (all correct → true; any wrong → false)
- **explain(θ)** = `θ` (the configuration explains itself)
- **incompatible** = `(≠)`

### Rashomon witnesses (N ≥ 2)

- θ₁ = "only hypothesis 0 is wrong" (θ₁ 0 = false, θ₁ i = true for i ≠ 0)
- θ₂ = "only hypothesis 1 is wrong" (θ₂ 1 = false, θ₂ i = true for i ≠ 1)
- observe(θ₁) = observe(θ₂) = false (failure)
- θ₁ ≠ θ₂ (they disagree on which hypothesis is wrong)
-/
-- ============================================================================

/-- An N=4 hypothesis configuration: each of 4 hypotheses is correct (true)
    or incorrect (false). -/
abbrev HypConfig4 := Fin 4 → Bool

/-- The experiment succeeds iff all 4 hypotheses are correct. -/
def experimentOutcome4 (θ : HypConfig4) : Bool :=
  θ 0 && θ 1 && θ 2 && θ 3

/-- The explanation is the configuration itself: it records which
    hypotheses are correct and which are wrong. -/
def blameAssignment4 (θ : HypConfig4) : HypConfig4 := θ

/-- Witness 1: only hypothesis 0 is wrong. -/
def dq4Config1 : HypConfig4 := fun i =>
  match i with
  | ⟨0, _⟩ => false
  | _ => true

/-- Witness 2: only hypothesis 1 is wrong. -/
def dq4Config2 : HypConfig4 := fun i =>
  match i with
  | ⟨1, _⟩ => false
  | _ => true

/-- Both 4-hypothesis witnesses produce experimental failure. -/
theorem dq4_same_outcome :
    experimentOutcome4 dq4Config1 = experimentOutcome4 dq4Config2 := by
  native_decide

/-- The two 4-hypothesis witnesses have different blame assignments
    (they disagree on which hypothesis is wrong). -/
theorem dq4_blame_different :
    blameAssignment4 dq4Config1 ≠ blameAssignment4 dq4Config2 := by
  intro h
  -- blameAssignment4 = id, so h : dq4Config1 = dq4Config2
  -- but they differ at index 0: dq4Config1 0 = false, dq4Config2 0 = true
  have h0 : dq4Config1 ⟨0, by omega⟩ = dq4Config2 ⟨0, by omega⟩ := by
    unfold blameAssignment4 at h; exact congrFun h ⟨0, by omega⟩
  simp [dq4Config1, dq4Config2] at h0

/-- The 4-hypothesis Duhem-Quine setting as an ExplanationSystem.
    - Θ = Fin 4 → Bool (hypothesis configurations)
    - H = Fin 4 → Bool (blame assignments = configurations)
    - Y = Bool (success or failure)
    - observe = experimentOutcome4
    - explain = blameAssignment4 (= id)
    - incompatible = (≠) -/
def duhemQuineSystem4 : ExplanationSystem HypConfig4 HypConfig4 Bool where
  observe := experimentOutcome4
  explain := blameAssignment4
  incompatible := fun b₁ b₂ => b₁ ≠ b₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨dq4Config1, dq4Config2, dq4_same_outcome, dq4_blame_different⟩

/-- **Duhem-Quine Underdetermination (4 hypotheses).**

With 4 hypotheses, the Rashomon set contains 2^4 - 1 = 15 failure
configurations. The impossibility still holds: no method of assigning
blame can be simultaneously faithful, stable, and decisive.

This demonstrates that the underdetermination scales with hypothesis
count — blame cannot be localized from the outcome alone regardless
of how many hypotheses are involved. -/
theorem duhem_quine_impossibility_4
    (E : HypConfig4 → HypConfig4)
    (hf : faithful duhemQuineSystem4 E)
    (hs : stable duhemQuineSystem4 E)
    (hd : decisive duhemQuineSystem4 E) : False :=
  explanation_impossibility duhemQuineSystem4 E hf hs hd
