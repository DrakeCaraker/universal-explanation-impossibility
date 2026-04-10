# Arrow Embedding Investigation Plan

**Question**: Is Arrow's impossibility theorem a special case of
the universal explanation impossibility?

**If YES**: A 1951 Nobel Prize-winning result is a corollary of
a 2026 theorem about AI explainability. This would be the single
most attention-grabbing finding in the research program.

**If NO**: A precise structural comparison showing which axioms
correspond and which don't is still a publishable paper.

**Either way**: We learn something valuable. The investigation
cannot fail — it either proves the embedding or characterizes
exactly where it breaks.

---

## VET RECORD

### Round 1 — Factual

- Arrow's theorem (1951): No social welfare function over ≥3
  alternatives satisfying Unrestricted Domain, Pareto (Weak),
  IIA, and Non-Dictatorship exists.
  Formally: ∀ F : (N → L(A)) → L(A), if |A| ≥ 3 and F satisfies
  UD + WP + IIA, then F is dictatorial.

- Our theorem: For any ExplanationSystem with the Rashomon
  property, no E can be faithful + stable + decisive.

- Key structural difference: Arrow has FOUR axioms (UD, WP, IIA,
  ND). We have THREE (faithful, stable, decisive) plus one
  structural assumption (Rashomon). Arrow's conclusion is
  "F is dictatorial." Ours is "False" (impossibility).

- ⚠️ Arrow's theorem is about a SINGLE function F (the social
  welfare function). Our theorem is about a function E that is
  DISTINCT from the native explanation (explain). In Arrow, there
  is no "native explanation" — F IS the thing being constrained.
  This is a fundamental structural difference that may prevent
  a clean embedding.

- ⚠️ Nipkow (2009) formalized Arrow in Isabelle/HOL. Wiedijk
  (2007) in Mizar. The exact formal statement is well-established.
  We should reference these for the precise axiom statements.

### Round 2 — Reasoning

- The most promising mapping uses the QUERY-RELATIVE framework:
  - Query q = (a,b): "does society rank a above b?"
  - IIA ≈ query-relative stability (the answer to q depends
    only on individual answers to q)
  - Pareto ≈ query-relative faithfulness (if all individuals
    answer q the same way, F agrees)
  - Non-dictatorship is the HARD part — it constrains the
    structure of F, not a property like faithful/stable/decisive

- ⚠️ The mapping might work for a RESTRICTED version of Arrow
  (e.g., Arrow without non-dictatorship, which gives a weaker
  result) but fail for the full theorem. This would still be
  interesting — showing which Arrow axioms correspond to which
  explanation properties.

- ⚠️ Arrow's theorem applies to functions from MULTIPLE inputs
  (N voters) to one output. Our framework has a SINGLE
  configuration θ mapping to one observation and one explanation.
  The "multiple voters" structure might not map to
  ExplanationSystem at all without extending the framework.

### Round 3 — Omissions

- ⚠️ Should investigate the GIBBARD-SATTERTHWAITE theorem too.
  GS says: any non-dictatorial, surjective voting rule for ≥3
  alternatives is manipulable. This has a different structure
  (strategy-proofness rather than IIA) and might map differently.

- ⚠️ Should check Okasha (2011) carefully — he already
  connected Arrow to theory choice. Our mapping should be
  DIFFERENT from Okasha's or clearly extend it.

- ⚠️ The plan should specify WHEN TO STOP. If after Phase 2
  the mapping doesn't work, don't force it. A negative result
  ("Arrow is NOT an instance, here's why") is also publishable.

---

## Phase 1: Precise Formal Setup [Opus]

### Task 1.1: State Arrow's theorem precisely

Write down the EXACT formal statement of Arrow's theorem
using our notation. Use Nipkow (2009) as reference.

**Arrow's ingredients**:
- A: a finite set of alternatives, |A| ≥ 3
- N: a finite set of voters (agents), |N| ≥ 2
- L(A): the set of strict linear orders on A (complete,
  transitive, antisymmetric, irreflexive)
- A preference profile: π = (π₁, ..., πₙ) ∈ L(A)^N
- A social welfare function: F : L(A)^N → L(A)

**Arrow's axioms**:
- Unrestricted Domain (UD): F is defined on all of L(A)^N
- Weak Pareto (WP): if ∀i, a >_πi b, then a >_F(π) b
- Independence of Irrelevant Alternatives (IIA): for all a,b ∈ A,
  the social ranking of a vs b depends only on individual
  rankings of a vs b. I.e., if π and π' agree on all
  pairwise (a,b) preferences, then F(π) and F(π') agree
  on (a,b).
- Non-Dictatorship (ND): there is no voter i such that
  F(π) = πᵢ for all π.

**Arrow's conclusion**: No F satisfying UD + WP + IIA + ND exists
(for |A| ≥ 3, |N| ≥ 2).

### Task 1.2: Attempt the ExplanationSystem mapping

Define the candidate mapping:

```
Θ = L(A)^N          -- preference profiles (configurations)
H = L(A)             -- social orderings (explanations)
Y = ???               -- observables (THIS IS THE HARD PART)
observe : Θ → Y      -- ???
explain : Θ → H      -- explain(π) = F(π) = the social ordering
incompatible : H → H → Prop  -- ???
```

**The Y problem**: What is the "observable" in social choice?
Several candidates:

(a) Y = A (the winning alternative). observe(π) = top element
    of F(π). Then Rashomon = two profiles with the same winner
    but different full rankings. This is common for |A| ≥ 3.

(b) Y = 2^(A×A) (pairwise comparison results). observe(π) =
    the set of all individual pairwise preferences. Then
    stability = IIA (same individual pairwise prefs → same
    social ranking). BUT: this makes stability = IIA, which
    is ONE of Arrow's four axioms, not the full theorem.

(c) Y = Θ (observe = identity). Then stability is trivial
    (always satisfied). This defeats the purpose.

(d) QUERY-RELATIVE: For each query q = (a,b), define
    observe_q(π) = the restriction of π to the (a,b) pairwise
    comparison across all voters. Then q-stability = IIA
    restricted to query q. This is the most promising approach.

### Task 1.3: Map Arrow's axioms to explanation properties

For the query-relative mapping with q = (a,b):

**IIA → q-stability**: IIA says the social ranking of (a,b)
depends only on individual rankings of (a,b). In our framework:
observe_q(π) = individual pairwise preferences on (a,b).
Stability says: if observe_q(π₁) = observe_q(π₂), then
E(π₁) agrees with E(π₂) on query q. This IS IIA.
**MAPS CLEANLY** ✓

**Weak Pareto → q-faithfulness**: WP says if all voters rank
a > b, then F ranks a > b. In our framework: if explain(π)
(= F(π)) ranks a > b, then a faithful E must not contradict
this. BUT: Pareto is STRONGER than faithfulness — Pareto says
E must AGREE with the unanimous individual preference, not
just not contradict F. Faithfulness says E doesn't contradict
explain(π), which is F(π). If F satisfies Pareto, then
E being faithful to F automatically respects Pareto
(transitively). **MAPS, but indirectly** — Pareto constrains
F, faithfulness constrains E relative to F.

**Non-Dictatorship → ???**: ND says F ≠ πᵢ for any fixed i.
In our framework, this would be a constraint on the EXPLAIN
map, not on E. Our theorem constrains E (the explanation method),
not explain (the native explanation). ND constrains the native
F itself. **DOES NOT MAP** to faithful/stable/decisive.

This is the critical mismatch: Arrow constrains the social
welfare function F directly. Our theorem constrains an
EXTERNAL explanation E of a given system. Arrow doesn't have
the E/F distinction — F is both the "system" and the
"explanation."

### Task 1.4: Identify the precise structural gap

The gap: Arrow's theorem says "no F exists satisfying all four
axioms." Our theorem says "given a system with Rashomon, no E
exists satisfying all three properties." The structures are:

Arrow:  ∀F, ¬(UD(F) ∧ WP(F) ∧ IIA(F) ∧ ND(F))
Ours:   ∀S with Rashomon, ∀E, ¬(faithful(S,E) ∧ stable(S,E) ∧ decisive(S,E))

The existential/universal structure is different:
- Arrow: ∀F ¬P(F) — no good F exists
- Ours: ∀E ¬P(E) given S — no good E exists for a given S

To embed Arrow, we'd need: construct an ExplanationSystem S
from the social choice setup such that the impossibility of E
implies the impossibility of F.

One approach: set explain(π) = πᵢ (voter i's preference) and
let E = F (the social welfare function). Then:
- F being faithful to πᵢ ≈ F respecting voter i's preferences
- F being stable ≈ F satisfying IIA
- F being decisive ≈ F producing a complete ranking

But this makes F "faithful to voter i" for a SPECIFIC i —
which is dictatorship! So the impossibility of faithful +
stable + decisive E would imply... that any F satisfying
IIA and producing complete rankings must be unfaithful to
every individual voter (or faithful to exactly one = dictator).

This is CLOSE to Arrow but not exactly Arrow. Arrow says:
the only F satisfying IIA + Pareto is a dictatorship. Our
framework would say: no F can be IIA-stable + faithful-to-all
+ decisive. The difference is Pareto vs faithful-to-all.

### GATE: After Task 1.4, assess whether a PRECISE embedding
exists. If YES → Phase 2. If NO → characterize the gap
precisely and write it up as a structural comparison.

---

## Phase 2: Lean Formalization [Opus]

Only proceed if Phase 1 identifies a precise embedding.

### Task 2.1: Define social choice ExplanationSystem in Lean

Create `UniversalImpossibility/ArrowEmbedding.lean`:

```lean
-- Social choice types
-- A : alternatives (Fin m, m ≥ 3)
-- N : voters (Fin n, n ≥ 2)
-- Preference profile: Fin n → Fin m → Fin m → Prop (pairwise orderings)
-- Social ordering: Fin m → Fin m → Prop

-- Define the ExplanationSystem for social choice
-- (exact definitions depend on Phase 1 findings)
```

### Task 2.2: Prove Arrow as a corollary

If the embedding works, the proof would be:
1. Construct the social choice ExplanationSystem
2. Verify it has the Rashomon property
3. Apply `explanation_impossibility` (or `query_impossibility`)
4. Show this implies Arrow's conclusion

### Task 2.3: Verify with #print axioms

The Arrow corollary should depend only on the social choice
axioms (finiteness of A, |A| ≥ 3, etc.), not on any
explanation-specific axioms.

---

## Phase 3: Paper [Opus]

### Task 3.1: Write the result

If embedding works: "Arrow's impossibility theorem is Instance 10
of the universal explanation impossibility."

If embedding fails: "Arrow's theorem and the explanation
impossibility share a precise structural parallel. The IIA axiom
corresponds to query-relative stability. Pareto corresponds to
a form of faithfulness. Non-dictatorship has no analog in the
explanation framework, which is the fundamental reason the
theorems are independent despite their structural similarity."

Either outcome is a paper.

---

## Decision Tree

```
Phase 1.1-1.3: Map axioms
  ↓
Phase 1.4: Does the mapping work?
  ├─ YES (exact embedding) → Phase 2 (Lean) → Phase 3 (paper)
  │   Title: "Arrow's Theorem as an Instance of
  │           the Universal Explanation Impossibility"
  │   Venue: PNAS or Econometrica
  │
  ├─ PARTIAL (some axioms map, ND doesn't) → Phase 3 (comparison)
  │   Title: "The Structural Parallel Between Arrow's
  │           Impossibility and the Explanation Trilemma"
  │   Venue: Social Choice and Welfare, or AAAI
  │
  └─ NO (fundamental structural mismatch) → Write up negative result
      Title: "Why Arrow's Theorem Is Not an Instance
              of the Explanation Impossibility (And What Is)"
      Venue: Working paper / blog post
```

---

## Confidence Assessment

| Component | Confidence |
|-----------|-----------|
| IIA maps to q-stability | HIGH |
| Pareto maps to some form of faithfulness | MEDIUM |
| Non-dictatorship maps to anything | LOW |
| Full Arrow embedding exists | LOW-MEDIUM |
| Partial structural comparison is publishable | HIGH |
| The investigation produces useful knowledge | HIGH |

The most likely outcome is PARTIAL: IIA = stability, Pareto ≈
faithfulness, but non-dictatorship breaks the embedding. This
is still a good paper — it precisely characterizes the
relationship between two fundamental impossibility results.

---

## Timeline

- Week 1: Phase 1 (formal mapping, axiom correspondence)
- Week 2: Assessment. If promising → Phase 2. If not → Phase 3.
- Week 3-4: Lean formalization (if proceeding) or comparison paper
- Total: 1 month for a definitive answer

This is SEPARATE from the NeurIPS submission. The Arrow
investigation does not block Paper 3. It is Paper 5.

---

## The Key Insight That Might Make It Work

The query-relative framework is the bridge. Arrow's IIA is
inherently per-query (per-pairwise-comparison). The standard
ExplanationSystem has a single global incompatibility, which
doesn't capture IIA's pairwise structure. But `query_impossibility`
does — it parameterizes by query, and each query q = (a,b)
gets its own Rashomon property.

The embedding attempt:
- For EACH pair (a,b), define a query ExplanationSystem
- IIA for (a,b) = stability for query (a,b)
- Pareto for (a,b) = faithfulness for query (a,b)
  (when all voters agree, F must agree = F is faithful
  to the unanimous preference)
- Apply `query_impossibility` to each (a,b) where Rashomon holds
- The question: does the Rashomon property hold for every
  pair (a,b) with |A| ≥ 3?

For |A| ≥ 3, there exist preference profiles where:
- All voters agree on a > b (Pareto direction)
- But different profiles with the same (a,b) individual
  preferences produce different social rankings of (a,b)
  when a third alternative c is permuted

WAIT — this is exactly what IIA prevents! If IIA holds,
then the social ranking of (a,b) DOES factor through the
individual (a,b) preferences. So if F satisfies IIA, there
is NO Rashomon for any query — stability is satisfied.

This means: for social welfare functions satisfying IIA,
the Rashomon property does NOT hold query-by-query. The
impossibility doesn't apply. Arrow's theorem says something
different: F satisfying IIA + Pareto must be dictatorial.
Our theorem says: if Rashomon holds, F can't be
faithful+stable+decisive. These are different claims about
different settings.

**This is the NEGATIVE RESULT.** Arrow's theorem is NOT
an instance because Arrow's IIA PREVENTS the Rashomon
property from holding. The two theorems operate in
complementary regimes:
- Our theorem: Rashomon holds → impossibility
- Arrow's theorem: IIA holds (= no pairwise Rashomon) →
  only dictatorships satisfy the remaining axioms

They're about DIFFERENT failure modes of aggregation.

This negative result is itself a clean finding worth
documenting. It precisely characterizes why the structural
analogy breaks down and identifies the boundary between
the two impossibility regimes.
