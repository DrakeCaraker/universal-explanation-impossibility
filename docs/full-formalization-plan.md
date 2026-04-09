# Complete Formalization, Symmetric Bayes Dichotomy Generalization, and Deferred Work Resolution

**Generated**: 2026-04-01
**Context**: Post-NeurIPS improvement plan. Targets JMLR Q4 2026.
**Current state**: 49 declarations (40 theorems + 9 lemmas), 15 axioms, 0 sorry. 48 supplement results unformalised.

---

## PLANNING CONSTRAINTS

- **Timeline**: JMLR target Q4 2026. No NeurIPS deadline pressure — this is the deep work.
- **Resources**: One author + Claude. Lean 4 + Mathlib. Apple Silicon. No GPU.
- **Quality over speed**: Every new Lean theorem must compile. Every new proof must use no sorry. If a result cannot be formalized without sorry, state precisely what Mathlib infrastructure is missing and defer.
- **Model allocation**: Use **opus** for novel mathematical reasoning (new proofs, the dichotomy generalization, hard Lean tactics). Use **sonnet** for routine Lean scaffolding (definitions, easy proofs, boilerplate).
- **Lean compilation**: `lake build` takes ~5 min. Batch related changes and compile once per batch.
- **File organization**: Create new Lean files for new modules (don't bloat existing files). Follow the existing architecture (one file per conceptual unit).

---

## COMPLETE INVENTORY OF UNFORMALISED RESULTS

### Already Formalized (49 declarations)

| Lean Name | File | Supplement Reference |
|-----------|------|---------------------|
| `RashimonProperty` | Trilemma.lean | Def 4 in main text |
| `attribution_impossibility`, `attribution_impossibility_weak` | Trilemma.lean | Thm 1 |
| `IterativeOptimizer`, `iterative_rashomon`, `iterative_impossibility` | Iterative.lean | §3 iterative |
| `gbdt_rashomon`, `gbdt_impossibility`, `gbdtOptimizer` | General.lean | §3 GBDT |
| `split_gap_exact`, `split_gap_ge_half`, `split_gap_lower_bound` | SplitGap.lean | Lemma 2 |
| `splitCount_ratio`, `attribution_ratio` | Ratio.lean | Thm 3 |
| `ratio_tendsto_atTop` | Ratio.lean | Ratio divergence |
| `spearmanCorr`, `spearmanCorr_lt_one_of_reversal`, `spearmanCorr_bound` | SpearmanDef.lean | Spearman bounds |
| `attribution_ratio_ge`, `not_equitable` | Impossibility.lean | Equity violation |
| `not_stable` | Impossibility.lean | Stability bound |
| `impossibility` | Impossibility.lean | Combined |
| `consensus_equity`, `consensus_difference_zero` | Corollary.lean | Cor 1a |
| `consensus_variance_rate`, `consensus_variance_halves`, `consensus_variance_nonneg` | Corollary.lean | Cor 1b |
| `attribution_sum_symmetric` | SymmetryDerive.lean | Derived |
| `lasso_impossibility`, `lasso_ratio_infinite` | Lasso.lean | §3.2 |
| `nn_impossibility` | NeuralNet.lean | §3.3 |
| `design_space_theorem` | DesignSpace.lean | Thm 5 (partial) |
| `strongly_faithful_impossible` | DesignSpace.lean | Step 3 Case 1 |
| `balanced_flip_symmetry` | DesignSpace.lean | Flip symmetry |
| `infeasible_faithful_stable_complete` | DesignSpace.lean | Infeasible point |
| `dash_produces_ties`, `dashMethod` | DesignSpace.lean | DASH ties |
| `dash_variance_decreases` | DesignSpace.lean | Variance |

### Not Yet Formalized (48 results)

#### Tier 1 — EASY (< 4 hrs each, 13 results)

These use existing axiom infrastructure and require only straightforward Lean tactics:

| ID | Result | Supplement Lines | Strategy |
|----|--------|-----------------|----------|
| S11 | Relaxation path convergence | 621-643 | Follows from S10 + existing consensus_equity; conditional on S10 |
| S13 | Group-size dependence (m/(m-1)) | 1006-1012 | Pure arithmetic corollary of S12 |
| S20 | Attribution aggregation method (def) | 1341-1356 | Extend existing `AggregationMethod` in DesignSpace.lean |
| S36 | Attribution design space (def) | 2510-2529 | Extend existing definitions in DesignSpace.lean |
| S38 | Theorem 1 as corollary | 2654-2658 | Trivial from existing `attribution_impossibility` |
| S40 | Path convergence as corollary | 2667-2675 | Trivial from S11 |
| S45 | Model selection desiderata (def) | 2988-2997 | New simple Prop definitions |
| S46 | Model Rashomon property (def) | 2999-3005 | Simple Prop definition |
| S47 | Model Selection Impossibility | 3014-3028 | Mirrors `attribution_impossibility` exactly |
| S51 | Feature attribution as SBD instance | 3151-3158 | Instance verification |
| S52 | Model selection as SBD instance | 3160-3166 | Instance verification |
| S58 | Abstract aggregation problem (def) | 3787-3799 | Simple structure definition |
| S66 | α-faithfulness (def) | 4134-4143 | Simple Prop definition |
| S67 | Approximate faithfulness-stability tradeoff | 4145-4164 | Easy from DGP symmetry |

#### Tier 2 — MODERATE (4-15 hrs each, 17 results)

These require new Mathlib imports or non-trivial proof engineering:

| ID | Result | Strategy | Key Mathlib Needed |
|----|--------|----------|-------------------|
| S3 | Permutation closure (def) | `Equiv.Perm`, group action on Model | `Mathlib.GroupTheory.Perm.Basic` |
| S4 | Rashomon from symmetry | Permutation invariance of population loss | S3 + loss function defs |
| S8 | Exact GBDT flip rate (5 parts) | Extends `balanced_flip_symmetry`; counting over Finset | `Finset.filter`, `Finset.card` |
| S9 | Unfaithfulness lower bound (U=1/2) | DGP symmetry ⇒ Pr[φ_j > φ_k] = 1/2 | Symmetry hypothesis |
| S10 | Optimal unfaithful ranking | Bayes-optimal = ties for symmetric features | S9 + conditional on measure |
| S12 | Efficiency-induced negative correlation | Variance decomposition under sum constraint | `MeasureTheory.variance` |
| S14 | Efficiency doesn't constrain across-model | Model-level vs data-level distinction | `MeasureTheory.variance` |
| S21 | Pairwise stability/unfaithfulness (def) | Probability definitions | `MeasureTheory.Measure` |
| S35 | Local instability dominates global | Jensen's inequality for variance | `ConvexOn`, `MeasureTheory.variance` |
| S37 | Design Space (Step 3 exhaustiveness) | Close gap in existing `design_space_theorem`; needs S10 | S9, S10 |
| S39 | F2 as corollary | Needs S22 (DASH Pareto optimality) | S22 |
| S44 | Conditional Attribution Impossibility | Extends core impossibility to conditional SHAP | S4 + conditional defs |
| S48 | Model Selection Design Space | Mirrors `design_space_theorem` structure | S47, S22 |
| S49 | Symmetric Decision Problem (def) | `MulAction`, group actions, decision theory | `Mathlib.GroupTheory.GroupAction.Basic` |
| S55 | Causal discovery as SBD instance | Instance verification for CPDAG groups | S50, S53 |
| S57 | Expected Kendall tau distance | Counting + normal CDF | `Finset.sum`, `Real.Phi` |
| S69 | Spearman bound for stable rankings | Uniform random permutations | `Finset.sum`, expected value |

#### Tier 3 — HARD (15-40 hrs each, 11 results)

Require substantial Mathlib machinery or extended proof engineering:

| ID | Result | Missing Infrastructure |
|----|--------|----------------------|
| S1 | Gaussian binary quantizer (α=2/π) | Half-normal distribution, `Real.pi`, Gaussian expectations |
| S2 | First-stump variance capture | Taylor expansion, asymptotic notation |
| S5 | Attribution non-degeneracy | Continuous measure, irrational arguments |
| S6 | Rashomon inevitability | Probabilistic symmetry, almost-sure events |
| S17 | Gaussian FIM specialization | `Matrix`, eigenvalues, Cramér-Rao |
| S18 | Loss landscape geometry | Ellipsoid projection, `Matrix.PosDef` |
| S23 | Median is less efficient (ARE=2/π) | Order statistics, asymptotic variance |
| S24 | Trimmed mean interpolates | Truncated distribution variance |
| S26 | Ensemble size lower bound | Normal CDF inverse, Cramér-Rao |
| S29 | Z-test near-optimality | Power calculation, `Real.Phi` |
| S32 | Exact exchangeability under subsampling | Conditional independence, exchangeability |
| S43 | Conditional attribution (def) | Do-calculus, conditional expectations |
| S60 | Interventional sample complexity | Neyman-Pearson, Gaussian hypothesis testing |

**Note on `Real.Phi`**: The normal CDF may not exist as a standalone Lean function in Mathlib. Several Tier 3 results (S26, S29, S57) need it. May require defining it from `MeasureTheory.Measure.gaussian`.

#### Tier 4 — RESEARCH (40+ hrs each, 7 results)

Require new mathematics or Mathlib development:

| ID | Result | Why Research-Level |
|----|--------|-------------------|
| S16 | FIM Impossibility (rigorous) | Full Taylor remainder in Lean, Hessian eigenvalue analysis |
| S22 | DASH Pareto Optimality | Cramér-Rao in Lean, Rao-Blackwell, Hoeffding. Cramér-Rao is NOT a packaged Mathlib theorem. |
| S28 | Query complexity lower bound | Le Cam's method, total variation, chi-squared divergence. Not in Mathlib. |
| S31 | Rashomon Characterization (F1) | Berry-Esseen in Lean, CLT |
| S33 | Approximate exchangeability | Markov chains, mixing times |
| S34 | Split-frequency diagnostic (F5) | CLT for dependent sequences |
| S50 | Symmetric Bayes Dichotomy | Group actions + decision theory + Bayes optimality |
| S53 | Causal orientation problem (def) | DAG/CPDAG types, graph automorphisms |
| S54 | Causal Discovery Impossibility | Graph theory + S50 |

---

## PHASE STRUCTURE

### Phase 1: Lean Foundations — Easy Wins (Tier 1)

Formalize all 13 EASY results. These extend the existing axiom system with minimal risk.

**Deliverables**:
- New file `DASHImpossibility/ModelSelection.lean` — S45, S46, S47 (model selection impossibility)
- New file `DASHImpossibility/AlphaFaithful.lean` — S66, S67 (approximate tradeoff)
- Extend `DASHImpossibility/DesignSpace.lean` — S20, S36, S38
- New file `DASHImpossibility/PathConvergence.lean` — S11, S40 (conditional on S10)
- New file `DASHImpossibility/SBDInstances.lean` — S51, S52, S58

**Agent allocation**: Sonnet for definitions and trivial corollaries. Opus for S47 (needs to mirror the impossibility proof structure).

**Expected output**: ~15 new declarations, all compiling, 0 sorry.
**Expected total after Phase 1**: ~64 declarations.

**Exit criteria**: `lake build` passes. `#print axioms` for each new theorem shows expected dependencies. No sorry in any file.

### Phase 2: Lean Core — Moderate Formalizations (Tier 2, high-impact subset)

Prioritize by impact on the paper's claims.

#### Batch 2a: Design Space Completeness (highest priority)

Closes the "half-formalized" gap in the Design Space Theorem.

- **S9**: Unfaithfulness lower bound (U=1/2)
- **S10**: Optimal unfaithful ranking
- **S37**: Design Space Step 3 exhaustiveness

The main new idea is axiomatizing DGP symmetry (Pr[φ_j > φ_k] = 1/2) or deriving it from the existing axiom system.

**Agent**: Opus for all three (core mathematical reasoning).

#### Batch 2b: Rashomon Universality

- **S3**: Permutation closure definition
- **S4**: Rashomon from symmetry theorem
- **S44**: Conditional Attribution Impossibility

**Agent**: Opus for S4, S44 (novel symmetry arguments). Sonnet for S3 (definition).

#### Batch 2c: Quantitative Extensions

- **S8**: Exact GBDT flip rate
- **S12**: Efficiency-induced negative correlation
- **S35**: Local instability dominates global

**Agent**: Sonnet for S8, S12 (moderate proofs). Opus for S35 (Jensen's inequality argument).

#### Batch 2d: Symmetric Bayes Dichotomy Prerequisites

Must be completed before Phase 3.

- **S49**: Symmetric Decision Problem definition
- **S21**: Pairwise stability/unfaithfulness definitions

**Agent**: Sonnet (definitions with Mathlib imports).

**Expected output**: ~20 new declarations.
**Expected total after Phase 2**: ~84 declarations.

**Exit criteria**: `design_space_theorem` now covers all 4 steps (not just 3). `#print axioms` for S37 shows it depends on S9, S10. Conditional Attribution Impossibility compiles.

### Phase 3: Symmetric Bayes Dichotomy Generalization

The crown jewel — proving the general theorem that G-invariance implies the two-family decomposition.

**What exists in the supplement** (Theorem S50, lines 3101-3149):
- Definition: Symmetric Decision Problem (Θ, D, π, G) with G acting on Θ
- Part (i): Any faithful complete estimator has U ≥ 1/|G·θ|
- Part (ii): Bayes estimator reports orbit centers (ties), U = 0, stability O(1/M)
- Part (iii): U=0 AND completeness is infeasible
- Exhaustiveness: every estimator is in Family A or Family B

**Lean approach**:

```
-- New file: DASHImpossibility/SymmetricBayes.lean

structure SymmetricDecisionProblem where
  Θ : Type                           -- Decision set
  [instΘ : Fintype Θ]
  [instΘDec : DecidableEq Θ]
  G : Type                           -- Symmetry group
  [instG : Group G]
  [instGA : MulAction G Θ]
  -- G-invariance: optimal decision is equidistributed over orbits
  orbit_uniform : ∀ (θ₁ θ₂ : Θ),
    θ₂ ∈ MulAction.orbit G θ₁ →
    prob_optimal θ₁ = prob_optimal θ₂
  -- Non-degeneracy: orbits have size ≥ 2
  orbit_nontrivial : ∃ θ : Θ, 2 ≤ (MulAction.orbit G θ).toFinset.card
```

Part (i) is essentially: under uniform distribution over an orbit of size k, any deterministic complete rule picks one element and is wrong (k-1)/k of the time. This is a counting argument over `Finset`.

Part (ii) requires Bayes optimality. **Fallback**: axiomatize "the Bayes estimator under symmetric prior reports orbit centers" and derive the consequences.

Part (iii) follows directly from Part (i).

**Key Mathlib dependencies**:
- `Mathlib.GroupTheory.GroupAction.Basic` — `MulAction`, `orbit`
- `Mathlib.GroupTheory.GroupAction.Defs` — orbit membership
- `Mathlib.Data.Fintype.Card` — `Fintype.card` for orbit sizes
- `MeasureTheory.Measure` — for probability
- `Mathlib.Probability.ProbabilityMassFunction` — for discrete distributions

**Agent**: Opus for everything in Phase 3.

**Expected output**: ~10-15 new declarations including the general SBD theorem.
**Expected total after Phase 3**: ~95-99 declarations.

**Risk**: Part (ii) may need an axiom for Bayes optimality. If so, clearly document this as the one remaining axiom in the SBD formalization.

**Exit criteria**: `symmetric_bayes_dichotomy` compiles with `#print axioms` showing dependencies. The three instance corollaries (S51, S52, S55) verify as special cases.

### Phase 4: Hard Formalizations (Tier 3, selective)

Prioritize by which results are cited most in the paper:

**S6 (Rashomon inevitability)**: The most important Tier 3 result — makes impossibility inescapable for standard ML. Requires probability measure and almost-sure arguments.

**S26 (Ensemble size lower bound)**: Practically important — gives M_min formula. Requires normal CDF but structure is algebraic.

**S60 (Interventional sample complexity)**: Causal identification barrier. Requires Neyman-Pearson.

**Remaining Tier 3**: Defer. The Gaussian quantization (S1-S2), median efficiency (S23-S24), and FIM specialization (S17-S18) are nice-to-have but not cited in main text.

**Agent**: Opus for S6 (deep probability). Sonnet for S26 (algebraic after defining Phi).

**Expected output**: ~5-8 new declarations.
**Expected total after Phase 4**: ~103-107 declarations.

### Phase 5: Research-Level Formalizations (Tier 4, long-term)

**S22 (DASH Pareto Optimality)**: Second most impactful Tier 4 result. Requires Cramér-Rao in Lean. Approach: axiomatize Cramér-Rao and derive DASH optimality from it.

**S28 (Query complexity)**: Le Cam's method. Very deep. Defer unless someone contributes Le Cam to Mathlib.

**S31, S33, S34**: Statistical results requiring CLT/Berry-Esseen. Deep Mathlib contributions in their own right. Defer.

**S53, S54**: Causal discovery formalization. Requires DAG/CPDAG types not in Mathlib. Separate project.

**Expected output**: ~5 declarations (S22 and helpers).
**Expected total after Phase 5**: ~108-112 declarations.

### Phase 6: Remaining Deferred Work

1. **Consensus variance bound from IndepFun.variance_sum**: Derive `consensus_variance_bound` from Mathlib, eliminating the last "convenience" axiom. Requires product measures and independence.

2. **Spearman classical bound**: Derive the m³/P³ constant, eliminating `spearman_classical_bound` axiom. Requires combinatorial counting.

3. **Axiom consistency in Lean**: Construct the Fin 4 model in Lean (currently only Python in `paper/scripts/axiom_consistency_model.py`).

4. **MI generalization** (ρ > 0 → I(X_j;X_k) > 0): Theoretical discussion only — no Lean. Add to Open Problems.

5. **Proportionality axiom derivation for TreeSHAP**: Open research problem. Not formalizable without algorithmic analysis of the TreeSHAP computation.

6. **JMLR restructure**: 35-40 page main text integrating supplement. Pure writing task.

---

## DEPENDENCY GRAPH

```
Phase 1 (EASY, no dependencies)
  ├── S45, S46, S47 (ModelSelection.lean)
  ├── S66, S67 (AlphaFaithful.lean)
  ├── S20, S36, S38 (DesignSpace.lean extensions)
  └── S58 (AbstractAggregation.lean)

Phase 2a (Design Space completeness)
  S9 (Unfaithfulness) ← DGP symmetry
  S10 (Optimal unfaithful) ← S9
  S37 (Exhaustiveness) ← S9, S10

Phase 2b (Rashomon universality)
  S3 (Permutation closure) ← none
  S4 (Rashomon from symmetry) ← S3
  S44 (Conditional impossibility) ← S4

Phase 2c (Quantitative)
  S8 (Exact flip rate) ← existing axioms
  S12 (Efficiency correlation) ← none
  S35 (Local ≥ global) ← none

Phase 2d → Phase 3
  S49 (SDP definition) ← MulAction
  S21 (Pairwise defs) ← MeasureTheory
  S50 (SBD theorem) ← S49, S21, S9
  S51, S52 ← S50

Phase 1 S11 ← S10 (so S11 waits for Phase 2a)
Phase 1 S40 ← S11

Phase 4
  S6 (Inevitability) ← S4, S5
  S26 (Ensemble bound) ← Real.Phi
  S60 (Interventional) ← none

Phase 5
  S22 (Pareto) ← Cramér-Rao axiom
  S39, S48 ← S22
```

---

## AXIOM ELIMINATION ROADMAP

Current: 15 axioms (6 type + 2 infrastructure + 7 domain)

| Axiom | Elimination Strategy | Phase | Difficulty |
|-------|---------------------|-------|-----------|
| `consensus_variance_bound` | Derive from `IndepFun.variance_sum` | Phase 6 | HARD (product measures) |
| `spearman_classical_bound` | Derive m³/P³ from combinatorial counting | Phase 6 | MODERATE |
| `proportionality_global` | Open research (TreeSHAP analysis) | Beyond scope | RESEARCH |
| `firstMover_surjective` | Could derive from DGP symmetry + training algorithm | Beyond scope | RESEARCH |
| Others (splitCount_*) | Derive from Gaussian conditioning formalization | Beyond scope | RESEARCH |

**Realistic target**: Eliminate 2 axioms (consensus_variance_bound, spearman_classical_bound), reducing from 15 to 13.

---

## DECLARATION COUNT PROJECTION

| Phase | New Declarations | Running Total | Axioms |
|-------|-----------------|---------------|--------|
| Current | — | 49 | 15 |
| Phase 1 | ~15 | ~64 | 15 |
| Phase 2 | ~20 | ~84 | 15 (possibly +1 DGP symmetry) |
| Phase 3 | ~12 | ~96 | 15 (+1 Bayes optimality if needed) |
| Phase 4 | ~6 | ~102 | same |
| Phase 5 | ~5 | ~107 | +1 Cramér-Rao if axiomatized |
| Phase 6 | ~5 | ~112 | 13 (2 eliminated) |

**Target for JMLR**: 100+ declarations, 13-14 axioms, 0 sorry.

---

## THE SINGLE MOST IMPACTFUL THING TO FORMALIZE

**S50: The Symmetric Bayes Dichotomy** (Phase 3)

Why: It elevates the paper from "an impossibility theorem about SHAP" to "a general proof technique for symmetric decision problems." Every future application (feature selection, hyperparameter tuning, causal discovery, model selection) inherits the impossibility automatically by checking G-invariance. A Lean formalization of the SBD would be — to our knowledge — the first formally verified result in invariant decision theory, extending the Nipkow lineage (Arrow → SBD) and establishing a reusable Lean library for symmetric impossibility proofs.

If the SBD is formalized and the three instances verified, the declaration count rises to ~96, the paper can claim "the first formally verified general impossibility technique in decision theory," and the JMLR contribution is unambiguously at the "notable paper" level.

---

## ITEMS CLASSIFIED AS (X) — NOT WORTH DOING IN LEAN

| ID | Result | Why Not |
|----|--------|---------|
| S1-S2 | Gaussian quantizer proofs | Deep real analysis with no structural insight; the α=2/π value is validated empirically |
| S23-S24 | Median/trimmed mean ARE | Asymptotic statistics; standard textbook results not worth the Lean effort |
| S33 | Approximate exchangeability | Markov chain mixing — a separate Mathlib contribution |
| S34 | F5 diagnostic rigor | CLT for dependent sequences — deep probability |
| S59, S62-S65 | Remarks | Not theorems; nothing to formalize |

---

## VERIFICATION REQUIREMENTS

For each new Lean file:
1. `lake build` passes with 0 errors
2. `#print axioms <theorem_name>` shows expected dependencies (no unexpected axioms)
3. `#check <theorem_name>` confirms the type signature matches the supplement statement
4. No `sorry` anywhere in the file
5. Each file has a module docstring explaining what it formalizes and which supplement results it covers
