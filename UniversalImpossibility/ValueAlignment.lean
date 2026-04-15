import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility

/-!
# Value Alignment Impossibility

When multiple RLHF training runs produce models with identical behavioral
performance but different implicit value rankings, the explanation
impossibility applies to value alignment:

- Faithful: the value explanation matches the model's actual priorities
- Stable: the explanation is consistent across equivalent training runs
- Decisive: the explanation commits to a specific value ranking

For binary value spaces (safety-first vs helpfulness-first), the bilemma
applies: not even faithful + stable is achievable without enrichment.

The enrichment (adding "balanced/uncertain" as a neutral element) restores
F+S at the cost of decisiveness — the model can't commit to which value
it prioritizes. This is the abstract mechanism behind "harmless and helpful"
frameworks that explicitly balance competing values rather than ranking them.

**Caveat:** This formalizes the STRUCTURE of value alignment under
multiplicity, not the full complexity of AI alignment. Real alignment
involves strategic agents, distributional shift, and ontological uncertainty
that the framework does not capture.
-/

set_option autoImplicit false

/-- Two value priorities. -/
inductive ValuePriority where
  | safetyFirst      -- prioritize safety over helpfulness
  | helpfulnessFirst -- prioritize helpfulness over safety
  deriving DecidableEq, Repr

/-- Behavioral performance (the observable). -/
inductive BehavioralPerformance where
  | passesBenchmark  -- meets the behavioral evaluation threshold
  deriving DecidableEq, Repr

/-- Two RLHF training configurations. -/
inductive RLHFConfig where
  | configA  -- training run emphasizing safety examples
  | configB  -- training run emphasizing helpfulness examples
  deriving DecidableEq, Repr

def valueObserve : RLHFConfig → BehavioralPerformance
  | _ => .passesBenchmark  -- both configs pass the benchmark

def valueExplain : RLHFConfig → ValuePriority
  | .configA => .safetyFirst
  | .configB => .helpfulnessFirst

def valueAlignmentSystem : ExplanationSystem RLHFConfig ValuePriority BehavioralPerformance where
  observe := valueObserve
  explain := valueExplain
  incompatible := (· ≠ ·)
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨.configA, .configB, rfl, by decide⟩

theorem value_maxIncompat : maximallyIncompatible valueAlignmentSystem :=
  fun _ _ hc => Classical.byContradiction (fun hne => hc hne)

/-- The value alignment impossibility: no value explanation is
    simultaneously faithful to the model's priorities AND stable
    across equivalent RLHF runs. -/
theorem value_alignment_impossibility
    (E : RLHFConfig → ValuePriority)
    (hf : faithful valueAlignmentSystem E)
    (hs : stable valueAlignmentSystem E) : False :=
  bilemma valueAlignmentSystem value_maxIncompat E hf hs
