# Universal Explanation Impossibility — Lean 4 Formalization

Lean 4 formalization of the universal explanation impossibility theorem. Target venue: JMLR (primary). Companion attribution paper housed in [dash-shap](https://github.com/DrakeCaraker/dash-shap). The F5→F1→DASH stability API is in [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255).

## What This Proves

No explanation of an underspecified system can be simultaneously faithful (reflect the system's actual structure), stable (consistent across observationally equivalent configurations), and decisive (commit to a single answer) when the Rashomon property holds. The core theorem (`explanation_impossibility` in `ExplanationSystem.lean`) requires **zero model-specific axioms** — only the Rashomon property as a hypothesis.

This is a **meta-theorem**: it applies uniformly to all nine explanation types — additive attributions (SHAP, IG, LIME), attention maps, counterfactual explanations, concept probes (TCAV), causal discovery (DAGs), model selection, saliency maps (GradCAM), LLM self-explanations, and mechanistic interpretability (circuit discovery) — because each is an instance of the abstract `ExplanationSystem` structure.

Model-specific instantiations from the companion attribution paper: GBDT has ratio 1/(1-ρ²) → ∞, Lasso has ratio ∞, neural nets have conditional violations, and random forests have bounded O(1/√T) violations. DASH (ensemble averaging) resolves the impossibility for balanced ensembles.

## Architecture

```
Level 0 (pure logic):     Trilemma.lean — attribution_impossibility + _weak (zero axiom deps)
Level 1 (framework):      Iterative.lean — IterativeOptimizer → Rashomon → impossibility
Level 2 (instantiation):  General.lean (GBDT), Lasso.lean, NeuralNet.lean
Level 3 (quantitative):   SplitGap.lean, Ratio.lean (1/(1-ρ²) divergence)
Level 4 (Spearman):       SpearmanDef.lean (defined from scratch, qualitative bound derived)
Level 5 (resolution):     Corollary.lean (DASH equity), Impossibility.lean (combined)
Level 6 (design space):   DesignSpace.lean + DesignSpaceFull.lean (all 4 steps complete)
Level 7 (derivation):     SymmetryDerive.lean (attribution_sum_symmetric, DERIVED)
Level 8 (generalization):  SymmetricBayes.lean (general SBD theorem)
Level 9 (instances):      ModelSelection.lean, CausalDiscovery.lean, SBDInstances.lean
Level 10 (extensions):    ConditionalImpossibility.lean, FairnessAudit.lean, FlipRate.lean
Level 11 (bounds):        EnsembleBound.lean, Efficiency.lean, AlphaFaithful.lean
Level 12 (universality):  RashomonUniversality.lean, RashomonInevitability.lean, LocalGlobal.lean
Strengthening:            ProportionalityLocal.lean (impossibility from per-model c only)
                          Qualitative.lean (impossibility from 2 axioms: dominance + surjectivity)
                          ApproximateEquity.lean (Rashomon from bounded proportionality)
                          Setup.lean (GBDTSetup structure bundling all axioms)
Contrast:                 RandomForest.lean (bounded violations, no formal proofs)

Universal framework (new):
  ExplanationSystem.lean        — Abstract ExplanationSystem structure; explanation_impossibility
  AttributionInstance.lean      — Additive attribution instance (SHAP, IG, LIME)
  AttentionInstance.lean        — Attention map instance (DistilBERT)
  CounterfactualInstance.lean   — Counterfactual explanation instance
  ConceptInstance.lean          — Concept probe instance (TCAV)
  CausalInstance.lean           — Causal discovery instance (DAG Markov equivalence)
  ModelSelectionInstance.lean   — Model selection instance (Rashomon multiplicity)
  UniversalImpossibility.lean   — Import hub; documents 9-instance inventory
  UniversalResolution.lean      — G-invariant resolution framework; gInvariant_stable
  UniversalDesignSpace.lean     — Universal design space dichotomy (Family A / Family B)
  DASHResolution.lean           — DASH as G-invariant resolution for attributions
  CPDAGResolution.lean          — CPDAG as G-invariant resolution for causal discovery
  Ubiquity.lean                 — Structural ubiquity: dimensional argument + impossibility bridge
```

## File Structure

```
UniversalImpossibility/
  ── Core attribution (Levels 0–12 + strengthening) ──
  Defs.lean              — FeatureSpace, 13 axioms, stability/equity defs, consensus, variance from Mathlib
  Trilemma.lean          — RashomonProperty, attribution_impossibility, attribution_impossibility_weak
  Iterative.lean         — IterativeOptimizer abstraction
  General.lean           — GBDT instance, gbdt_impossibility, gbdtOptimizer
  SplitGap.lean          — split_gap_exact, split_gap_ge_half (pure algebra)
  Ratio.lean             — attribution_ratio = 1/(1-ρ²), ratio_tendsto_atTop
  SpearmanDef.lean       — Spearman defined from midranks, qualitative + quantitative bounds
  Lasso.lean             — lasso_impossibility (ratio = ∞)
  NeuralNet.lean         — nn_impossibility (conditional on captured feature)
  RandomForest.lean      — Contrast case (documentation, no formal proofs)
  Impossibility.lean     — Combined: equity violation + stability bound
  Corollary.lean         — DASH consensus equity, variance convergence
  DesignSpace.lean       — Design Space Theorem (composite), DASH ties
  DesignSpaceFull.lean   — Design Space exhaustiveness (Step 3: Family A or B)
  SymmetryDerive.lean    — attribution_sum_symmetric (DERIVED from axioms)
  ModelSelection.lean    — Model selection impossibility (S45-S47)
  ModelSelectionDesignSpace.lean — Model selection design space (S48)
  AlphaFaithful.lean     — α-faithfulness bound (S66-S67)
  UnfaithfulBound.lean   — Unfaithfulness ≥ 1/2, ties optimal (S9-S11)
  PathConvergence.lean   — Relaxation path convergence (S38, S40)
  RashomonUniversality.lean  — Rashomon from symmetry via feature swap (S3-S4)
  RashomonInevitability.lean — Impossibility is inescapable (S5-S6)
  ConditionalImpossibility.lean — Conditional SHAP impossibility + escape (S44)
  FlipRate.lean          — Exact GBDT flip rate, binary group = coin flip (S8)
  Efficiency.lean        — SHAP efficiency amplification m/(m-1) (S12-S14)
  LocalGlobal.lean       — Local ≥ global instability (S35)
  SymmetricBayes.lean    — General SBD: orbit bounds, trichotomy, exhaustiveness (S49-S50)
  GaussianFlipRate.lean  — Standard normal CDF Φ, flip rate formula (S31 Gaussian)
  FIMImpossibility.lean  — Gaussian FIM impossibility, Rashomon ellipsoid (S16-S17)
  QueryComplexity.lean   — Query complexity Ω(σ²/Δ²), Le Cam structural (S28)
  CausalDiscovery.lean   — Causal discovery impossibility (S53-S55)
  SBDInstances.lean      — SBD instances + abstract aggregation (S51-S52, S58)
  FairnessAudit.lean     — Fairness audit impossibility (S56)
  EnsembleBound.lean     — DASH variance optimality + ensemble size (S22, S26)
  ProportionalityLocal.lean — Impossibility from per-model c only
  Qualitative.lean       — Impossibility from 2 axioms: dominance + surjectivity
  ApproximateEquity.lean — Rashomon from bounded proportionality
  Setup.lean             — GBDTSetup structure bundling all axioms
  Basic.lean             — Import hub
  ── Universal framework (new, ExplanationSystem + instances) ──
  ExplanationSystem.lean        — Abstract ExplanationSystem; explanation_impossibility (0 axioms)
  AttributionInstance.lean      — Additive attribution instance (SHAP, IG, LIME)
  AttentionInstance.lean        — Attention map instance (DistilBERT; 60% flip rate)
  CounterfactualInstance.lean   — Counterfactual explanation instance
  ConceptInstance.lean          — Concept probe instance (TCAV)
  CausalInstance.lean           — Causal discovery instance (Markov equivalence; Rashomon Derived)
  ModelSelectionInstance.lean   — Model selection instance (Rashomon multiplicity)
  MechInterpInstance.lean       — Mechanistic interpretability instance (circuit non-uniqueness; mech_interp_impossibility)
  MarkovEquivalence.lean        — Derives Rashomon property from Markov equivalence first principles
  Necessity.lean                — Necessity of Rashomon property (possibility iff no Rashomon)
  UniversalImpossibility.lean   — Import hub; 9-instance inventory
  UniversalResolution.lean      — G-invariant resolution; gInvariant_stable
  UniversalDesignSpace.lean     — universal_design_space_dichotomy (Family A / B)
  DASHResolution.lean           — DASH as G-invariant resolution for attributions
  CPDAGResolution.lean          — CPDAG as G-invariant resolution for causal discovery
  Ubiquity.lean                 — generic_underspecification, ubiquity_impossibility
paper/
  universal_impossibility.tex — JMLR universal paper (primary, this paper)
  main.tex                    — NeurIPS 2026 companion attribution paper (10 pages)
  main_jmlr.tex               — JMLR attribution paper (54 pages)
  supplement.tex              — Supplementary for attribution paper (79 pages)
  references.bib              — References
  scripts/                    — Experiment scripts (figure generation, validation, diagnostics)
    run_all_universal_experiments.py — Runner for the 4 universal experiments
    README.md                  — How to reproduce all experiments
  figures/                    — PDF figures (ratio, instability, DASH, design space, SNR calibration, etc.)
  sections/                   — LaTeX section fragments for universal paper instances
```

## Lean State: 75 files, 72 axioms, 351 theorems+lemmas, 0 sorry

## Axiom Inventory (68 total)

| Category | Axioms | Used by |
|----------|--------|---------|
| Type declarations | Model, numTrees, numTrees_pos, attribution, splitCount, firstMover | Infrastructure (bundled in Setup.lean) |
| Core properties | firstMover_surjective, splitCount_firstMover, splitCount_nonFirstMover, proportionality_global, splitCount_crossGroup_symmetric, splitCount_crossGroup_stable | GBDT bounds |
| Measure infrastructure | modelMeasurableSpace, modelMeasure | Variance (Mathlib connection) |
| Query complexity | testing_constant, testing_constant_pos | Query complexity scaling |
| Universal instances | AttentionConfig, AttentionMap, attention_rashomon, etc. | Per-instance Rashomon witnesses |

**Axiom stratification (verified by `#print axioms`):**
- **Core universal impossibility** (`explanation_impossibility`): ZERO axioms (Rashomon is a hypothesis)
- **Core attribution impossibility** (`attribution_impossibility`): ZERO behavioral axioms (only Model + attribution types)
- **Qualitative impossibility** (`impossibility_qualitative`): ZERO behavioral axioms (dominance + surjectivity as hypotheses)
- **GBDT impossibility** (`gbdt_impossibility_local`): 4 axioms (surj, fm, nfm — NO proportionality_global)
- **Quantitative impossibility** (`impossibility`): 5 axioms (+ proportionality_global for ratio)
- **DASH resolution** (`consensus_equity`): 6 axioms (+ cross-group symmetric)
- **Bundled impossibility** (`attribution_impossibility_bundled`): ZERO axioms (fully parametric via GBDTSetup)

**Formerly axiomatized, now derived:**
- `spearman_classical_bound` → `spearman_instability_bound` in SpearmanDef.lean (derived from split-count structure; bound 3(m-1)²/(P³-P) is weaker than classical m³/P³ but fully proved)
- `le_cam_lower_bound` — theorem in QueryComplexity.lean (provable by `not_lt.mp`; the contrapositive formulation ¬(n < bound) → bound ≤ n is a tautology in any linear order)
- `consensus_variance_bound` — theorem in Defs.lean (from attribution_variance_nonneg + Nat.cast_nonneg)
- `attribution_sum_symmetric` — theorem in SymmetryDerive.lean (from proportionality + split-count + cross-group + balance)
- `attribution_variance` — noncomputable def from ProbabilityTheory.variance (Mathlib)
- `attribution_variance_nonneg` — theorem from Mathlib's variance_nonneg
- `attribution_proportional` — backward-compatible theorem wrapper from proportionality_global

The core universal impossibility theorem (`explanation_impossibility`) uses **none** of these — only the Rashomon property as hypothesis.

## Building

```bash
make help          # show all targets
make lean          # compile Lean (~5 min)
make paper         # compile all paper versions
make verify        # build + count consistency check
make validate      # run 3 key experiments (~5 min)
make setup         # full setup for new contributors
```

## Two-Paper Structure

- **Universal paper** (this repo, primary): `paper/universal_impossibility.tex` — The meta-theorem that unifies nine explanation types via the abstract `ExplanationSystem` framework. Target: **JMLR**.
- **Companion attribution paper**: `paper/main.tex` (NeurIPS 10pp) and `paper/main_jmlr.tex` (JMLR 54pp) — The attribution-specific impossibility with GBDT/Lasso/NeuralNet instantiations, quantitative bounds, DASH resolution, and empirical experiments.

## Submission

- **JMLR** (primary for universal paper): `paper/universal_impossibility.tex`. JMLR class: https://jmlr.org/format/
- **NeurIPS 2026** (companion attribution paper): `paper/main.tex` (10 pages) + `paper/supplement.tex` (79 pages). Abstract May 4, Paper May 6. Official `neurips_2026.sty` (verified identical to neurips.cc download).
- **arXiv**: Run `paper/scripts/prepare_arxiv.sh` to uncomment authors and fill URLs.
- Universal paper title: "The Universal Explanation Impossibility: No Explanation Is Faithful, Stable, and Decisive Under Underspecification"
- Attribution paper title: "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity"
- Authors: Drake Caraker, Bryan Arnold, David Rhoads
- Companion code: [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255)

## Do NOT

- Commit paper changes without verifying paper-code consistency. Run this verification block and confirm all numbers match the paper text before committing:
  ```bash
  grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
  grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
  grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
  ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
  ```
- Use `sorry` without a `-- TODO:` comment explaining what's needed
- Change axioms without re-running the SymPy verification (in companion repo: `dash-shap/paper/proofs/verify_lemma6_algebra.py`)
- Add `autoImplicit true` — all variables must be explicit
- Claim "N theorems" without verifying — count with `grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'` (currently 351)
- Run parallel subagents that both modify the same file (causes build cache corruption)
- Axiomatize quantities that can be defined — prefer definitions with axiomatized bounds (see SpearmanDef.lean pattern)
- Claim empirical results as "proved" or "Lean-verified" — distinguish: **proved** (zero axiom deps), **derived** (from axioms), **argued** (supplement proof only), **empirical** (experiments). The paper's "Proof status transparency" paragraph is the reference.
