# Axiom Reduction & Enrichment Stack Tightening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the axiom count from 83 to ~69 and close the two genuine structural gaps in the enrichment stack formalization, strengthening the Lean codebase without changing any theorems.

**Date:** 2026-04-16
**Priority:** Medium (improves reviewer response to "how many axioms?" questions; does not block submission)

---

## Context: What This Plan Addresses

A comprehensive audit of the 83 axioms and non-proved components identified:

### Current Axiom Budget (83 total)

| Category | Count | Files |
|----------|-------|-------|
| Core model (Model, numTrees, attribution, splitCount, firstMover) | 6 | Defs.lean |
| GBDT behavioral (surjective, proportionality, split-count formulas) | 6 | Defs.lean |
| Measure infrastructure | 2 | Defs.lean |
| Query complexity (Le Cam constant) | 2 | QueryComplexity.lean |
| DASH symmetry group (OrderingPerm, Group, MulAction, HasSymmetry, resolution, G-invariance) | ~10 | DASHResolution.lean |
| CPDAG symmetry group (MECGroup, Group, MulAction, HasSymmetry, resolution, G-invariance) | ~10 | CPDAGResolution.lean |
| Axiomatized instances (Attention, Counterfactual, Concept, ModelSelection, MechInterp, Saliency, LLM, Causal) | ~32 | *Instance.lean files |
| Physics frameworks (BHFramework, QGFramework) | 11 | EnrichmentStack.lean |

### What's Already Strong

- Core impossibility (`explanation_impossibility`): **ZERO axioms**
- Core attribution trilemma (`attribution_impossibility`): **ZERO behavioral axioms**
- 7/8 instance types have constructive versions with **ZERO axioms**
- MarkovEquivalence.lean derives causal Rashomon from first principles
- ParetoOptimality.lean fully formalizes DASH Pareto optimality
- AxiomSubstitution.lean demonstrates the `decisive` axiom is at the Goldilocks point
- ApproximateRashomon.lean extends to epsilon-stability with zero axioms

### Genuine Gaps Identified

1. **EnrichmentStack is a hollow envelope** -- `decisiveness_sacrificed_at` is `True` at every level; the general structure doesn't require explicit systems or bilemmas
2. **Multi-level genericity is a comment, not a theorem** -- the cardinality argument (|Theta| > |H|^k implies k independent binary questions) is informal
3. **CausalInstanceConstructive.lean doesn't exist** -- the only instance type without a constructive version
4. **OrderingPerm axiomatizes Equiv.Perm** -- 6 axioms that are literally Mathlib's `Equiv.Perm (Fin n)`
5. **MECGroup axiomatizes a quotient** -- 4 axioms that could be a quotient type

### Inherently Informal (Do NOT attempt)

- Physics frameworks (BH, QG) -- empirical hypotheses about the universe
- The Goedel parallel mechanism -- structural pattern, not a unified theorem
- The `next_level_holds` causal narrative -- interpretation, not formalizable without item 1

---

## Phase 1: Low-Hanging Fruit (Axiom Elimination)

**Estimated time: 2-3 hours | Axioms eliminated: ~14**

### Task 1.1: Add CausalInstanceConstructive.lean

- [ ] **Create `UniversalImpossibility/CausalInstanceConstructive.lean`**

Follow the pattern of the 7 existing constructive instance files. Use the chain/fork DAG construction already proved in MarkovEquivalence.lean:
- Define `CausalConfig` as an inductive type (e.g., `chain | fork`)
- Define `CausalObservation` (CI structure, same for chain and fork)
- Define `CausalExplanation` (edge orientation, differs)
- Construct the `ExplanationSystem` with Rashomon proved by `decide`
- Derive impossibility via `explanation_impossibility`

Reference files:
- `MarkovEquivalence.lean` lines 1-120 (the chain/fork construction)
- `AttentionInstanceConstructive.lean` (template for constructive instances)
- `CounterfactualInstanceConstructive.lean` (template)

Verify: `#print axioms causal_constructive_impossibility` should show only Lean builtins.

### Task 1.2: Replace OrderingPerm with Equiv.Perm

- [ ] **Modify `UniversalImpossibility/DASHResolution.lean`**

Replace:
```lean
axiom OrderingPerm : Type
axiom instOrderingPermGroup : Group OrderingPerm
axiom instOrderingPermAction : MulAction OrderingPerm ModelOrdering
```

With:
```lean
def OrderingPerm := Equiv.Perm (Fin n)  -- or appropriate concrete type
instance : Group OrderingPerm := inferInstance
instance : MulAction OrderingPerm ModelOrdering := ...
```

The `HasSymmetry` and `gInvariant` axioms may still need to remain (they encode domain-specific properties), but the pure group theory axioms are eliminated.

Expected axiom reduction: 3-6 axioms.

Verify: `lake build` succeeds; `#print axioms dash_gInvariant_implies_stable` shows fewer axioms.

### Task 1.3: Replace MECGroup with Quotient Type

- [ ] **Modify `UniversalImpossibility/CPDAGResolution.lean`**

Replace:
```lean
axiom MECGroup : Type
axiom instMECGroupGroup : Group MECGroup
axiom instMECGroupAction : MulAction MECGroup CPDAGConfig
```

With a quotient construction using the Markov equivalence relation from MarkovEquivalence.lean.

Expected axiom reduction: 3-4 axioms.

This is harder than Task 1.2 because the group structure on a quotient type requires more work. If blocked, defer to Phase 3.

---

## Phase 2: Structural Tightening (No Axiom Change)

**Estimated time: 3-4 hours | Axioms eliminated: 0 | Structural gaps closed: 2**

### Task 2.1: Typed EnrichmentStack

- [ ] **Create or modify `UniversalImpossibility/EnrichmentStack.lean`**

Add a new structure alongside the existing one (don't break existing code):

```lean
structure TypedEnrichmentStack where
  depth : Nat
  /-- At each level, an ExplanationSystem with Bool explanations -/
  system_at : Fin depth → Σ (Theta Y : Type), ExplanationSystem Theta Bool Y
  /-- The bilemma holds at each level (F+S impossible) -/
  bilemma_at : ∀ (i : Fin depth) (E : (system_at i).2.1 → Bool),
    faithful (system_at i).2.2 E → stable (system_at i).2.2 E → False
  /-- Enrichment at level i does not resolve level i+1 -/
  independence : ∀ (i : Fin depth) (hi : i.val + 1 < depth),
    -- The level-(i+1) bilemma persists after level-i enrichment
    True  -- refine this to use hasMultiLevelStructure
```

Then prove the infinite bit-stack instantiates `TypedEnrichmentStack`:

```lean
def infiniteTypedStack (k : Nat) : TypedEnrichmentStack := ...
```

This makes the enrichment recursion non-vacuous for the concrete case.

### Task 2.2: Generic Multi-Level Theorem

- [ ] **Add to `UniversalImpossibility/EnrichmentStack.lean` or `Ubiquity.lean`**

Prove the cardinality-based genericity claim:

```lean
theorem generic_multilevel_structure
    {Theta : Type} [Fintype Theta] [DecidableEq Theta]
    {Y : Type} (observe : Theta -> Y)
    (k : Nat)
    (hcard : k < Fintype.card Theta) :
    -- There exist k independent binary questions on Theta
    -- that are constant on observe-fibers of size >= 2
    ...
```

This requires Mathlib's `Fintype.card` and pigeonhole (`Fintype.exists_ne_map_eq_of_card_lt`). The exact statement needs care -- the binary questions need to be independent AND respect the fiber structure. A simpler version:

```lean
theorem multilevel_from_fiber_size
    {Theta : Type} [Fintype Theta]
    (observe : Theta -> Y) (y : Y)
    (fiber : Finset Theta)
    (hfiber : ∀ t ∈ fiber, observe t = y)
    (hsize : 2 ^ k ≤ fiber.card) :
    -- There exist k independent Bool-valued functions on fiber
    ∃ (fs : Fin k -> Theta -> Bool),
      ∀ i j, i ≠ j -> ∃ t₁ t₂ ∈ fiber,
        fs i t₁ = fs i t₂ ∧ fs j t₁ ≠ fs j t₂
```

This uses the fact that a set of size >= 2^k can be partitioned by k independent bits.

---

## Phase 3: Medium-Difficulty Strengthening (Optional)

**Estimated time: 8-12 hours | Axioms eliminated: ~6**

### Task 3.1: Concrete Model Type

- [ ] **Define `GBDTModel` as a concrete structure in a new file or in `Setup.lean`**

```lean
structure GBDTModel (P : Nat) (T : Nat) where
  trees : Fin T -> (Fin P -> Bool) -> Real
  -- Each tree maps feature-split patterns to leaf values
```

Then derive `attribution`, `splitCount`, `firstMover` as definitions on this structure, and prove the behavioral axioms (surjectivity, proportionality, split-count formulas) as theorems under explicit Gaussian DGP assumptions.

This is substantial work. The split-count formulas require formalizing the spectral solution to the Gaussian conditioning cascade. Consider deferring unless a reviewer specifically asks.

### Task 3.2: Formalize d-separation for MECGroup (if Task 1.3 was deferred)

- [ ] **Add d-separation to `MarkovEquivalence.lean`**

Define d-separation on finite DAGs, prove that chain and fork DAGs have the same CI structure (d-separation equivalence), and use this to construct the MECGroup as a proper quotient.

---

## Phase 4: Documentation Updates

**Estimated time: 1 hour**

### Task 4.1: Update CLAUDE.md Axiom Inventory

- [ ] **After all changes, update the axiom count in CLAUDE.md**

Run the verification block:
```bash
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
```

Update the axiom table, file count, and theorem count in CLAUDE.md.

### Task 4.2: Update Paper Axiom Claims

- [ ] **Update `paper/nature_article.tex` and `paper/universal_impossibility_monograph.tex`**

Any references to "83 axioms" should be updated to the new count. Run the paper-code consistency check.

---

## Verification Checklist

After all changes:

- [ ] `lake build` succeeds with zero errors
- [ ] `grep -c "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'` returns 0
- [ ] `grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'` returns expected count
- [ ] `#print axioms explanation_impossibility` still shows zero domain axioms
- [ ] `#print axioms causal_constructive_impossibility` shows only Lean builtins
- [ ] Paper axiom counts match code

---

## Risk Assessment

| Task | Risk | Mitigation |
|------|------|------------|
| 1.1 CausalInstanceConstructive | Low -- pattern is established | Copy from AttentionInstanceConstructive |
| 1.2 OrderingPerm replacement | Low-Medium -- may need MulAction instance work | Keep axiomatized fallback |
| 1.3 MECGroup replacement | Medium -- quotient group is nontrivial | Defer to Phase 3 if blocked |
| 2.1 TypedEnrichmentStack | Low -- additive (doesn't break existing) | Add alongside, don't replace |
| 2.2 Generic multi-level | Medium -- exact statement needs care | Start with the simpler fiber-size version |
| 3.1 Concrete Model | High -- substantial formalization | Only if reviewer asks |

---

## Expected Outcome

| Metric | Before | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|--------|---------------|---------------|---------------|
| Axiom count | 83 | ~69 | ~69 | ~63 |
| Constructive instances | 7/8 | 8/8 | 8/8 | 8/8 |
| Structural gaps | 2 | 2 | 0 | 0 |
| Files | 107 | 108 | 108 | 108-109 |
| Theorems+lemmas | 493 | ~497 | ~502 | ~510 |
