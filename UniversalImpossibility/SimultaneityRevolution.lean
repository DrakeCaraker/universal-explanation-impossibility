import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility

/-!
# Relativity of Simultaneity as a Scientific Revolution

The relativity of simultaneity exhibits the bilemma pattern:

Pre-revolution: H = {simultaneous, not_simultaneous} -- maximally incompatible
Rashomon: two frames agree on the laws of physics but disagree on simultaneity
Bilemma: no framework is faithful to one frame's simultaneity AND stable across frames
Enrichment: spacetime (events have no absolute temporal ordering) = neutral element
Decisiveness sacrificed: can't say whether spacelike-separated events are simultaneous

**Caveat:** The singleton observable space (both frames agree on Lorentz-invariant
physics) is a deliberate simplification. The full revolution involves Lorentz
transformations, mass-energy equivalence, and geometric reformulation — the
enrichment pattern captures only the simultaneity component.

## The Mapping

- Theta (configurations) = inertial reference frames
- Y (observables) = physical laws (Lorentz invariant)
- H (explanations) = simultaneity judgments for a pair of spacelike-separated events
- observe = physical laws (same in all frames -- first postulate of SR)
- explain = simultaneity judgment from each frame's perspective
- incompatible = (ne)
- Rashomon: two frames agree on the laws but disagree on simultaneity

## Minimal Witness

- Frame A: a frame in which two spacelike-separated events are simultaneous
  (e.g., the frame where both events occur at t = 0).
- Frame B: a Lorentz-boosted frame in which the same events are not simultaneous
  (e.g., in the boosted frame, event 1 occurs at t' < 0 and event 2 at t' > 0).
- Both frames observe the same physical laws (Lorentz invariance).
- But: "simultaneous" != "not simultaneous" (incompatible explanations).

Einstein's resolution: spacetime. Events don't have absolute temporal ordering;
simultaneity is frame-dependent. This is the neutral element -- compatible with
both judgments because it abstains from absolute simultaneity claims.
-/

set_option autoImplicit false

/-- Two claims about the temporal ordering of spacelike-separated events. -/
inductive SimultaneityJudgment where
  | simultaneous      -- the events occur at the same time
  | notSimultaneous   -- the events occur at different times
  deriving DecidableEq, Repr

/-- The physical laws are the same in all inertial frames. -/
inductive PhysicalLaws where
  | lorentzInvariant
  deriving DecidableEq, Repr

/-- Two reference frames observing the same pair of spacelike-separated events. -/
inductive ReferenceFrame where
  | frameA  -- frame where the events appear simultaneous
  | frameB  -- frame where they appear non-simultaneous
  deriving DecidableEq, Repr

def relObserve : ReferenceFrame → PhysicalLaws
  | _ => .lorentzInvariant

def relExplain : ReferenceFrame → SimultaneityJudgment
  | .frameA => .simultaneous
  | .frameB => .notSimultaneous

/-- The simultaneity system as an ExplanationSystem.
    - Theta = ReferenceFrame (frameA, frameB)
    - H = SimultaneityJudgment (simultaneous, notSimultaneous)
    - Y = PhysicalLaws (lorentzInvariant)
    - observe = relObserve (constant -- same laws in all frames)
    - explain = relExplain (simultaneity judgment)
    - incompatible = (ne) -/
def simultaneitySystem : ExplanationSystem ReferenceFrame SimultaneityJudgment PhysicalLaws where
  observe := relObserve
  explain := relExplain
  incompatible := (· ≠ ·)
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨.frameA, .frameB, rfl, by decide⟩

/-- SimultaneityJudgment with incompatible = (ne) is maximally incompatible. -/
theorem simultaneity_maxIncompat : maximallyIncompatible simultaneitySystem :=
  fun _ _ hc => Classical.byContradiction (fun hne => hc hne)

/-- **The Simultaneity Bilemma.**

No framework for judging simultaneity is simultaneously faithful (matches
each frame's own judgment) and stable (gives the same judgment for all
frames that observe the same laws).

This is the formal content of the relativity of simultaneity: Newtonian
absolute time is faithful to each frame's judgment but unstable across
frames. Einstein's spacetime resolves this via enrichment: events have
no absolute temporal ordering (the neutral element), sacrificing
decisiveness (can't say whether spacelike-separated events are
simultaneous). -/
theorem simultaneity_revolution
    (E : ReferenceFrame → SimultaneityJudgment)
    (hf : faithful simultaneitySystem E)
    (hs : stable simultaneitySystem E) : False :=
  bilemma simultaneitySystem simultaneity_maxIncompat E hf hs
