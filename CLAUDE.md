# The Attribution Impossibility — Lean 4 Formalization

Lean 4 formalization of the impossibility theorem for feature attribution under collinearity. Target venue: NeurIPS 2026 (abstract May 4, paper May 6). Paper 3 in the 5-paper research program housed in [dash-shap](https://github.com/DrakeCaraker/dash-shap). The F5→F1→DASH stability API is in [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255).

## What This Proves

No single-model feature ranking can simultaneously be faithful (reflect the model's attributions), stable (consistent across equivalent models), and complete (rank all feature pairs) when features are collinear. The core theorem requires **zero model-specific axioms** — only the Rashomon property.

Model-specific instantiations show GBDT has ratio 1/(1-ρ²) → ∞, Lasso has ratio ∞, neural nets have conditional violations, and random forests have bounded O(1/√T) violations. DASH (ensemble averaging) resolves the impossibility for balanced ensembles.

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
```

## File Structure

```
UniversalImpossibility/
  Defs.lean              — FeatureSpace, 13 axioms, stability/equity defs, consensus, variance from Mathlib
  Trilemma.lean          — RashimonProperty, attribution_impossibility, attribution_impossibility_weak
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
  RashomonUniversality.lean — Rashomon from symmetry via feature swap (S3-S4)
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
  Basic.lean             — Import hub
paper/
  main.tex           — NeurIPS 2026 paper (10 pages)
  supplement.tex     — Supplementary (79 pages)
  references.bib     — 49 references
  scripts/           — 51 scripts (figure generation, validation, diagnostics)
  figures/           — PDF figures (ratio, instability, DASH, design space, SNR calibration, conditional threshold, etc.)
```

## Lean State: 54 files, 16 axioms, 305 theorems+lemmas, 0 sorry

## Axiom Inventory (16 total)

| Category | Axioms | Used by |
|----------|--------|---------|
| Type declarations | Model, numTrees, numTrees_pos, attribution, splitCount, firstMover | Infrastructure (bundled in Setup.lean) |
| Core properties | firstMover_surjective, splitCount_firstMover, splitCount_nonFirstMover, proportionality_global, splitCount_crossGroup_symmetric, splitCount_crossGroup_stable | GBDT bounds |
| Measure infrastructure | modelMeasurableSpace, modelMeasure | Variance (Mathlib connection) |
| Query complexity | testing_constant, testing_constant_pos | Query complexity scaling |

**Axiom stratification (verified by `#print axioms`):**
- **Core impossibility** (`attribution_impossibility`): ZERO behavioral axioms (only Model + attribution types)
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

The core impossibility theorem (Levels 0-1) uses **none** of these — only the Rashomon property as hypothesis.

## Building

```bash
make help          # show all targets
make lean          # compile Lean (~5 min)
make paper         # compile all paper versions
make verify        # build + count consistency check
make validate      # run 3 key experiments (~5 min)
make setup         # full setup for new contributors
```

## Submission

- **JMLR** (primary target): `paper/main_jmlr.tex` (54 pages, `jmlr.cls` from TeX Live). JMLR class: https://jmlr.org/format/
- **NeurIPS 2026** (backup): `paper/main.tex` (10 pages) + `paper/supplement.tex` (79 pages). Abstract May 4, Paper May 6. Official `neurips_2026.sty` (verified identical to neurips.cc download).
- **arXiv**: Run `paper/scripts/prepare_arxiv.sh` to uncomment authors and fill URLs.
- Title: "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity"
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
- Claim "N theorems" without verifying — count with `grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'` (currently 305)
- Run parallel subagents that both modify the same file (causes build cache corruption)
- Axiomatize quantities that can be defined — prefer definitions with axiomatized bounds (see SpearmanDef.lean pattern)
- Claim empirical results as "proved" or "Lean-verified" — distinguish: **proved** (zero axiom deps), **derived** (from axioms), **argued** (supplement proof only), **empirical** (experiments). The paper's "Proof status transparency" paragraph is the reference.
