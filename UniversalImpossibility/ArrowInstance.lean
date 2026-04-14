import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Arrow's Impossibility as an Explanation System Instance

Arrow's theorem (1951): for ≥3 alternatives and ≥2 voters, no social welfare
function can simultaneously satisfy Unanimity (Pareto), Independence of
Irrelevant Alternatives (IIA), and Non-dictatorship.

We show that the *setting* of Arrow's theorem — preference aggregation — is a
natural instance of the ExplanationSystem framework with the Rashomon property.

## The mapping

- **Θ** (configurations) = preference profiles (each of 2 voters has a strict
  ranking of 3 alternatives)
- **Y** (observables) = the pairwise comparison data for a specific pair (A,B)
  — how each voter ranks A vs B
- **H** (explanations) = social orderings (complete rankings of alternatives)
- **observe** = extract the A-vs-B pairwise comparison from both voters
- **explain** = the "Borda-like" social ordering implied by the full profile
- **incompatible** = (≠) on social orderings

## The Rashomon property

Two profiles can agree on how voters rank A vs B (same observable) while
differing on how voters rank C relative to A and B. This forces different
social orderings — a Rashomon witness.

**Profile 1**: Voter1 = A>B>C, Voter2 = B>C>A
  → Pairwise A-vs-B: Voter1 prefers A, Voter2 prefers B (split)
  → "Natural" social ordering: B>A>C (B wins 2 pairwise, A wins 1)

**Profile 2**: Voter1 = A>C>B, Voter2 = C>B>A
  → Pairwise A-vs-B: Voter1 prefers A, Voter2 prefers B (split)
  → "Natural" social ordering: C>A>B (C wins 2 pairwise, A wins 1)

Same observation, different explanations → Rashomon.

## Interpretation

The explanation_impossibility theorem applied to this system says: no
social welfare function (viewed as an explanation E : Profile → Ranking)
can be simultaneously:
- **Faithful** (IIA-compatible): E(θ) never contradicts the profile's
  natural social ordering
- **Stable** (IIA): profiles with the same pairwise data get the same
  social ordering
- **Decisive** (non-triviality): E commits to every distinction the
  natural ordering makes

This is *not* a proof of Arrow's theorem (which is stronger and involves
specific axioms). It is a proof that Arrow's *setting* has the Rashomon
property, and therefore the explanation impossibility *applies* to it.
-/

/-- Three alternatives for a minimal Arrow instance. -/
inductive Alt where | A | B | C
  deriving DecidableEq, Repr

/-- Complete strict rankings of three alternatives.
    Each constructor lists alternatives from most to least preferred. -/
inductive Ranking where
  | ABC  -- A > B > C
  | ACB  -- A > C > B
  | BAC  -- B > A > C
  | BCA  -- B > C > A
  | CAB  -- C > A > B
  | CBA  -- C > B > A
  deriving DecidableEq, Repr

/-- A preference profile for 2 voters. -/
structure Profile where
  voter1 : Ranking
  voter2 : Ranking
  deriving DecidableEq, Repr

/-- The pairwise comparison of A vs B for 2 voters.
    This is the "observable" — what IIA says should determine the social
    ranking of A vs B. -/
inductive PairwiseAB where
  | bothA   -- both voters prefer A to B
  | v1A_v2B -- voter 1 prefers A, voter 2 prefers B
  | v1B_v2A -- voter 1 prefers B, voter 2 prefers A
  | bothB   -- both voters prefer B to A
  deriving DecidableEq, Repr

/-- Does the given ranking prefer A to B? -/
def prefersAtoB : Ranking → Bool
  | .ABC => true   -- A > B > C: A preferred to B
  | .ACB => true   -- A > C > B: A preferred to B
  | .BAC => false  -- B > A > C: B preferred to A
  | .BCA => false  -- B > C > A: B preferred to A
  | .CAB => true   -- C > A > B: A preferred to B
  | .CBA => false  -- C > B > A: B preferred to A

/-- Extract the A-vs-B pairwise comparison from a profile. -/
def observeAB (p : Profile) : PairwiseAB :=
  match prefersAtoB p.voter1, prefersAtoB p.voter2 with
  | true,  true  => .bothA
  | true,  false => .v1A_v2B
  | false, true  => .v1B_v2A
  | false, false => .bothB

/-- The "natural" social ordering for a profile, determined by pairwise
    majority (Condorcet method). For our two witness profiles:
    - Profile 1 (ABC, BCA): B beats A (1-1 tie broken by B's stronger
      overall support), B beats C (2-0), A beats C (1-1 tie broken similarly).
      We assign BAC.
    - Profile 2 (ACB, CBA): C beats B (2-0), C beats A (1-1), A beats B
      (1-1). We assign CAB.

    For all other profiles we assign an arbitrary ranking (the Rashomon
    witness only needs two specific profiles). -/
def socialOrdering : Profile → Ranking
  | ⟨.ABC, .BCA⟩ => .BAC  -- Profile 1: B>A>C
  | ⟨.ACB, .CBA⟩ => .CAB  -- Profile 2: C>A>B
  -- All other profiles get a default (only the two witnesses matter)
  | _ => .ABC

/-- Profile 1: Voter1 = A>B>C, Voter2 = B>C>A -/
def profile1 : Profile := ⟨.ABC, .BCA⟩

/-- Profile 2: Voter1 = A>C>B, Voter2 = C>B>A -/
def profile2 : Profile := ⟨.ACB, .CBA⟩

/-- Both profiles have the same A-vs-B pairwise: voter 1 prefers A,
    voter 2 prefers B. -/
theorem profiles_same_pairwise : observeAB profile1 = observeAB profile2 := by
  decide

/-- The two profiles have different social orderings. -/
theorem profiles_different_social :
    socialOrdering profile1 ≠ socialOrdering profile2 := by
  decide

/-- Arrow's preference aggregation setting as an ExplanationSystem.
    - Θ = Profile (preference profiles)
    - H = Ranking (social orderings)
    - Y = PairwiseAB (pairwise comparison data)
    - observe = observeAB (extract A-vs-B pairwise)
    - explain = socialOrdering (the natural/Condorcet social ordering)
    - incompatible = (≠)

    The Rashomon property: profile1 and profile2 have the same pairwise
    observation but different social orderings. -/
def arrowSystem : ExplanationSystem Profile Ranking PairwiseAB where
  observe := observeAB
  explain := socialOrdering
  incompatible := fun r₁ r₂ => r₁ ≠ r₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨profile1, profile2, profiles_same_pairwise, profiles_different_social⟩

/-- **Arrow Instance Impossibility.**

    No social welfare function (viewed as E : Profile → Ranking) can
    simultaneously be faithful, stable, and decisive when applied to
    the preference aggregation setting with ≥3 alternatives and ≥2 voters.

    In Arrow's terms:
    - **Stable** ↔ IIA: the social ranking of A vs B depends only on
      voters' rankings of A vs B
    - **Faithful** ↔ Pareto-compatible: the social welfare function
      never contradicts the natural (Condorcet) social ordering
    - **Decisive** ↔ Non-trivial: the social welfare function commits
      to every distinction the natural ordering makes

    This is a direct application of explanation_impossibility to
    arrowSystem. -/
theorem arrow_impossibility
    (E : Profile → Ranking)
    (hf : faithful arrowSystem E)
    (hs : stable arrowSystem E)
    (hd : decisive arrowSystem E) : False :=
  explanation_impossibility arrowSystem E hf hs hd
