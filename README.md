# The Attribution Impossibility

**No feature ranking can be simultaneously faithful, stable, and complete when features are correlated — and we prove it in Lean 4.**

![Theorems](https://img.shields.io/badge/theorems-305-blue)
![Axioms](https://img.shields.io/badge/axioms-16-orange)
![Sorry](https://img.shields.io/badge/sorry-0-brightgreen)
![Lean 4](https://img.shields.io/badge/Lean-4-purple)
![Files](https://img.shields.io/badge/Lean_files-54-informational)

<!-- Verify badges with:
  grep -c '^theorem\|^lemma' UniversalImpossibility/*.lean | awk -F: '{s+=$2}END{print s}'  # 305
  grep -c '^axiom' UniversalImpossibility/*.lean | awk -F: '{s+=$2}END{print s}'              # 16
  grep -rn 'sorry' UniversalImpossibility/*.lean                                               # (empty)
  ls UniversalImpossibility/*.lean | wc -l                                                     # 54
-->

If you have ever retrained an XGBoost model and noticed the "most important feature" changed, this paper proves that is not a bug — it is a mathematical inevitability. When features are correlated, no attribution method can simultaneously tell you what the model computed (faithfulness), give you the same answer across retrains (stability), and decide every pair of features (completeness). You must pick two of three. We prove it, quantify it, characterize the entire design space of solutions, and give you a toolkit to handle it. The formalization comprises 305 theorems across 54 Lean 4 files with 16 axioms and 0 sorry. The core impossibility theorem is machine-checked with zero domain-specific axiom dependencies. The resolution — DASH (Diversified Aggregation of SHAP) — is proved to be the minimum-variance unbiased estimator via Cramer-Rao.

---

## Quick Start

```bash
# Full setup (Lean + Python)
make setup

# Build Lean proofs (~5 min cached)
make lean

# Compile all 4 paper versions
make paper

# Verify counts match papers
make verify

# Run key validation experiments (~5 min)
make validate
```

## What This Proves

### 1. The Problem

SHAP feature rankings flip when you retrain a model. For collinear features with similar importance, the ranking is literally a coin flip — 50% of the time, the "most important" feature changes. In a survey of 77 public datasets, 68% exhibit this instability. This is not a software bug, not a tuning problem, and not specific to any one SHAP implementation. It is a mathematical inevitability that applies to XGBoost, LightGBM, CatBoost, Lasso, neural networks, and permutation importance alike.

The root cause is the **Rashomon property**: when features are correlated, multiple near-optimal models exist that rank them in opposite orders. Every model class with collinear features admits these equally-good-but-differently-explained solutions. The question is not whether to fix it, but what tradeoff to accept.

### 2. The Impossibility (Theorem 1)

No feature ranking can be simultaneously faithful (reflect what the model computed), stable (robust to retraining), and complete (decide every pair of features). **Faithful, Stable, Complete: Pick Two.**

```
                       The Attribution Trilemma

       Faithful ────────── IMPOSSIBLE ────────── Stable
           │                    X                    │
           │              (Theorem 1)                │
           │                                         │
       Family A                                 Family B
       (single model)                           (DASH ensemble)
       + Faithful, Complete                     + Faithful, Stable
       - Unstable (50% flips)                   - Within-group ties
```

- **Lean:** `attribution_impossibility` in [`Trilemma.lean`](UniversalImpossibility/Trilemma.lean)
- **Axiom dependencies:** Zero. The proof uses only the Rashomon property as a hypothesis.
- **Proof:** A 4-line contradiction. If a ranking is faithful, it must agree with each model. But collinear features admit models ranking them in opposite orders (Rashomon). So the ranking cannot agree with both, violating stability. If stable, it cannot be faithful to all models. Completeness forces the choice.

This is the paper's center of gravity. Everything else quantifies, extends, or resolves this result.

### 3. How Bad Is It? (Quantitative Bounds)

The impossibility is not a mild nuisance. The attribution ratio — how much the dominant feature is overweighted relative to the true contribution — diverges or explodes depending on the model class:

| Model Class | Ratio | Behavior | Lean File |
|-------------|-------|----------|-----------|
| **GBDT** | 1/(1-rho^2) | Diverges to infinity as rho approaches 1 | `Ratio.lean`, `SplitGap.lean` |
| **Lasso** | Infinity | One feature gets everything, the other gets zero | `Lasso.lean` |
| **Neural networks** | Conditional | Depends on which feature the network "captures" | `NeuralNet.lean` |
| **Random forests** | O(1/sqrt(T)) | Converges (bounded violations) — the contrast case | `RandomForest.lean` |

Key theorems:
- `ratio_tendsto_atTop` in [`Ratio.lean`](UniversalImpossibility/Ratio.lean) — the attribution ratio tends to infinity as correlation approaches 1
- `split_gap_exact` in [`SplitGap.lean`](UniversalImpossibility/SplitGap.lean) — the exact split gap formula (pure algebra)
- `lasso_impossibility` in [`Lasso.lean`](UniversalImpossibility/Lasso.lean) — Lasso attribution ratio is infinite
- `nn_impossibility` in [`NeuralNet.lean`](UniversalImpossibility/NeuralNet.lean) — neural net impossibility conditional on captured feature

The GBDT ratio 1/(1-rho^2) means that at rho=0.9, the dominant feature gets approximately 5.3x its fair share. At rho=0.99, it gets approximately 50x. The flip rate for binary collinear groups is exactly 1/2 — a literal coin flip (`binary_group_flip_rate` in [`FlipRate.lean`](UniversalImpossibility/FlipRate.lean)).

### 4. The Fix: DASH

**DASH** (**D**iversified **A**ggregation of **SH**AP): average |SHAP| values across M independently trained models. This is not just "try averaging" — it is provably the minimum-variance unbiased estimator via the Cramer-Rao bound.

- `consensus_equity` in [`Corollary.lean`](UniversalImpossibility/Corollary.lean) — DASH achieves equity (zero unfaithfulness) for collinear feature pairs in balanced ensembles
- `sum_squares_ge_inv_M` in [`EnsembleBound.lean`](UniversalImpossibility/EnsembleBound.lean) — Cramer-Rao optimality via Titu's lemma (the sum of squares of weights is minimized by equal weights)
- Variance decreases as 1/M, with tight ensemble size formula: **M_min = ceil(2.71 * sigma^2 / Delta^2)**

DASH is Pareto-optimal: no attribution method achieves better stability without sacrificing more faithfulness. The proof is in [`DesignSpace.lean`](UniversalImpossibility/DesignSpace.lean). The tradeoff: DASH produces ties for within-group features (it cannot distinguish features that are genuinely interchangeable), but ranks between-group features faithfully and stably.

### 5. The Complete Map (Design Space Theorem)

The achievable set of attribution methods under collinearity has exactly two families — and nothing else:

```
                    The Attribution Design Space

    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   Family A (single-model methods)                    │
    │   + Faithful + Complete                              │
    │   - Unstable: 50% flip rate for within-group pairs   │
    │   Examples: TreeSHAP, KernelSHAP, permutation imp.  │
    │                                                      │
    ├──────────────────────────────────────────────────────┤
    │                                                      │
    │   Family B (DASH ensemble methods)                   │
    │   + Faithful + Stable (variance -> 0 as M -> inf)    │
    │   - Incomplete: within-group ties                    │
    │   Example: DASH with M >= M_min models               │
    │                                                      │
    └──────────────────────────────────────────────────────┘

    No third family exists. Every attribution method falls into A or B.
```

- `design_space_theorem` in [`DesignSpace.lean`](UniversalImpossibility/DesignSpace.lean) — the design space characterization
- `family_a_or_family_b` in [`DesignSpaceFull.lean`](UniversalImpossibility/DesignSpaceFull.lean) — exhaustiveness proof

Both relaxation paths (drop completeness, drop stability) converge to the same solution: DASH. This convergence is itself proved (`relaxation_paths_converge` in [`PathConvergence.lean`](UniversalImpossibility/PathConvergence.lean)) and contrasts with Arrow's theorem, where relaxation paths diverge.

### 6. It Is Not Just SHAP (Symmetric Bayes Dichotomy)

The impossibility is an instance of a general pattern: any symmetric decision problem admits exactly two strategy families — the faithful-but-unstable individual strategy and the stable-but-tied aggregate strategy. This is the **Symmetric Bayes Dichotomy** (SBD), a reusable proof technique from invariant decision theory.

- `symmetric_bayes_dichotomy` in [`SymmetricBayes.lean`](UniversalImpossibility/SymmetricBayes.lean) — the general theorem with orbit bounds via `MulAction`

Three verified instances, each with a different symmetry group:

| Instance | Symmetry Group | Impossibility | Lean File |
|----------|---------------|---------------|-----------|
| Feature attribution | S_2 transpositions | Cannot rank collinear features | `Trilemma.lean` |
| Model selection | S_2 permutations | Cannot select among Rashomon-equivalent models | `ModelSelection.lean` |
| Causal discovery | CPDAG automorphisms | Cannot orient edges in Markov equivalence class | `CausalDiscovery.lean` |

### 7. Extensions

Each extension is a self-contained theorem in its own Lean file:

| Extension | Key Result | Lean File |
|-----------|-----------|-----------|
| **Conditional SHAP** | Cannot escape when beta_j = beta_k; escapes when Delta-beta > ~0.2 | [`ConditionalImpossibility.lean`](UniversalImpossibility/ConditionalImpossibility.lean) |
| **Fairness audits** | SHAP-based proxy audits are coin flips; (1/2)^K intersectional compounding | [`FairnessAudit.lean`](UniversalImpossibility/FairnessAudit.lean) |
| **Intersectional fairness** | Multi-attribute fairness impossibility | [`IntersectionalFairness.lean`](UniversalImpossibility/IntersectionalFairness.lean) |
| **Fisher information** | Independent proof path via FIM; Rashomon ellipsoid | [`FIMImpossibility.lean`](UniversalImpossibility/FIMImpossibility.lean) |
| **Query complexity** | Omega(sigma^2/Delta^2) queries needed; Le Cam lower bound | [`QueryComplexity.lean`](UniversalImpossibility/QueryComplexity.lean) |
| **Query complexity (parametric)** | Parametric query complexity bounds | [`QueryComplexityParametric.lean`](UniversalImpossibility/QueryComplexityParametric.lean) |
| **Rashomon inevitability** | Impossibility is inescapable for standard ML under symmetry | [`RashomonInevitability.lean`](UniversalImpossibility/RashomonInevitability.lean) |
| **Local vs global** | Local instability >= global instability | [`LocalGlobal.lean`](UniversalImpossibility/LocalGlobal.lean) |
| **Flip rate** | Exact GBDT flip rate; binary group = coin flip | [`FlipRate.lean`](UniversalImpossibility/FlipRate.lean) |
| **Efficiency** | SHAP efficiency amplification factor m/(m-1) | [`Efficiency.lean`](UniversalImpossibility/Efficiency.lean) |
| **Alpha-faithfulness** | Bound on alpha-faithful attribution | [`AlphaFaithful.lean`](UniversalImpossibility/AlphaFaithful.lean) |
| **Unfaithfulness bound** | Unfaithfulness >= 1/2 for complete rankings; ties are optimal | [`UnfaithfulBound.lean`](UniversalImpossibility/UnfaithfulBound.lean) |
| **Rashomon universality** | Rashomon property from symmetry via feature swap | [`RashomonUniversality.lean`](UniversalImpossibility/RashomonUniversality.lean) |
| **Gaussian flip rate** | Standard normal CDF, phi(0)=1/2, flip rate from Gaussian noise | [`GaussianFlipRate.lean`](UniversalImpossibility/GaussianFlipRate.lean) |
| **Mutual information** | Information-theoretic impossibility bounds | [`MutualInformation.lean`](UniversalImpossibility/MutualInformation.lean) |
| **Robustness (Lipschitz)** | Lipschitz robustness bounds for attribution | [`RobustnessLipschitz.lean`](UniversalImpossibility/RobustnessLipschitz.lean) |
| **Local sufficiency** | Local sufficiency conditions for attribution | [`LocalSufficiency.lean`](UniversalImpossibility/LocalSufficiency.lean) |
| **Stump proportionality** | Decision stump proportionality derivation | [`StumpProportionality.lean`](UniversalImpossibility/StumpProportionality.lean) |
| **Model selection design space** | Design space for model selection (SBD instance) | [`ModelSelectionDesignSpace.lean`](UniversalImpossibility/ModelSelectionDesignSpace.lean) |

## Repository Structure

```
dash-impossibility-lean/
│
├── UniversalImpossibility/                    # 54 Lean 4 files, 305 theorems, 16 axioms, 0 sorry
│   │
│   │  ── Level 0: Pure Logic ──
│   ├── Trilemma.lean                     # attribution_impossibility (zero axiom deps, 4-line proof)
│   │
│   │  ── Level 1: Framework ──
│   ├── Iterative.lean                    # IterativeOptimizer connects to Rashomon property
│   │
│   │  ── Level 2: Model Instantiation ──
│   ├── General.lean                      # GBDT instance, gbdt_impossibility, gbdtOptimizer
│   ├── Lasso.lean                        # lasso_impossibility (ratio = infinity)
│   ├── NeuralNet.lean                    # nn_impossibility (conditional on captured feature)
│   │
│   │  ── Level 3: Quantitative Bounds ──
│   ├── SplitGap.lean                     # split_gap_exact, split_gap_ge_half (pure algebra)
│   ├── Ratio.lean                        # attribution_ratio = 1/(1-rho^2), ratio_tendsto_atTop
│   │
│   │  ── Level 4: Spearman ──
│   ├── SpearmanDef.lean                  # Spearman from midranks, spearman_instability_bound (derived)
│   │
│   │  ── Level 5: Resolution ──
│   ├── Corollary.lean                    # DASH consensus equity, variance convergence
│   ├── Impossibility.lean                # Combined: equity violation + stability bound
│   │
│   │  ── Level 6: Design Space ──
│   ├── DesignSpace.lean                  # design_space_theorem, DASH ties, Pareto structure
│   ├── DesignSpaceFull.lean              # family_a_or_family_b (exhaustiveness)
│   │
│   │  ── Level 7: Derivation ──
│   ├── SymmetryDerive.lean               # attribution_sum_symmetric (DERIVED from axioms)
│   │
│   │  ── Level 8: Generalization ──
│   ├── SymmetricBayes.lean               # General SBD: orbit bounds, trichotomy, exhaustiveness
│   │
│   │  ── Level 9: SBD Instances ──
│   ├── ModelSelection.lean               # Model selection impossibility
│   ├── ModelSelectionDesignSpace.lean     # Model selection design space
│   ├── CausalDiscovery.lean              # Causal discovery impossibility (Markov equivalence)
│   ├── SBDInstances.lean                 # SBD instances + abstract aggregation
│   │
│   │  ── Level 10: Extensions ──
│   ├── ConditionalImpossibility.lean     # Conditional SHAP impossibility + escape condition
│   ├── FairnessAudit.lean                # Fairness audit impossibility (coin-flip audits)
│   ├── IntersectionalFairness.lean       # Multi-attribute fairness impossibility
│   ├── FlipRate.lean                     # Exact GBDT flip rate, binary group = coin flip
│   │
│   │  ── Level 11: Bounds ──
│   ├── EnsembleBound.lean                # DASH variance optimality + ensemble size (Titu's lemma)
│   ├── Efficiency.lean                   # SHAP efficiency amplification m/(m-1)
│   ├── AlphaFaithful.lean                # alpha-faithfulness bound
│   ├── UnfaithfulBound.lean              # Unfaithfulness >= 1/2, ties optimal
│   ├── PathConvergence.lean              # Relaxation path convergence
│   ├── QueryComplexity.lean              # Query complexity Omega(sigma^2/Delta^2), Le Cam
│   ├── QueryComplexityParametric.lean    # Parametric query complexity bounds
│   ├── MutualInformation.lean            # Information-theoretic impossibility bounds
│   ├── RobustnessLipschitz.lean          # Lipschitz robustness bounds
│   ├── LocalSufficiency.lean             # Local sufficiency conditions
│   │
│   │  ── Level 12: Universality ──
│   ├── RashomonUniversality.lean         # Rashomon from symmetry via feature swap
│   ├── RashomonInevitability.lean        # Impossibility is inescapable for standard ML
│   ├── LocalGlobal.lean                  # Local instability >= global instability
│   │
│   │  ── Strengthening ──
│   ├── ProportionalityLocal.lean         # Impossibility from per-model c only
│   ├── Qualitative.lean                  # Impossibility from 2 axioms: dominance + surjectivity
│   ├── ApproximateEquity.lean            # Rashomon from bounded proportionality
│   ├── StumpProportionality.lean         # Decision stump proportionality derivation
│   ├── Setup.lean                        # GBDTSetup structure bundling all axioms
│   │
│   │  ── Infrastructure ──
│   ├── Defs.lean                         # FeatureSpace, 16 axioms, stability/equity defs, Mathlib
│   ├── Consistency.lean                  # Axiom system consistency (Fin 4 construction)
│   ├── GaussianFlipRate.lean             # Standard normal CDF, flip rate formula
│   ├── FIMImpossibility.lean             # Gaussian FIM impossibility, Rashomon ellipsoid
│   ├── RandomForest.lean                 # Contrast case (documentation, no formal proofs)
│   └── Basic.lean                        # Import hub (all 54 files)
│
├── paper/
│   ├── main_definitive.tex               # 66-page monograph (source of truth)
│   ├── main_jmlr.tex                     # 54-page JMLR submission
│   ├── main.tex                          # 10-page NeurIPS version
│   ├── supplement.tex                    # 79-page NeurIPS supplement
│   ├── references.bib                    # 49 references
│   ├── FINDINGS_MAP.md                   # Complete 109-finding reference with tier classification
│   ├── scripts/                          # 51 experiment scripts + utilities
│   │   ├── requirements.txt              # Pinned Python dependencies
│   │   ├── axiom_consistency_model.py    # Constructs Fin 4 model, 16/16 axioms PASS
│   │   ├── f1_f5_validation.py           # multi-model Z-test + single-model screen diagnostics
│   │   ├── monte_carlo_flip_rate.py      # 1K-trial Monte Carlo flip rate validation
│   │   ├── prevalence_survey.py          # 77-dataset prevalence (68% unstable)
│   │   ├── prevalence_survey_openml.py   # OpenML prevalence survey
│   │   ├── snr_calibration.py            # SNR predicts flip rate (R^2=0.94 for SNR>=0.5)
│   │   ├── cross_implementation_validation.py  # XGBoost/LightGBM/CatBoost comparison
│   │   ├── dash_vs_alternatives.py       # DASH vs SAGE, PDP, permutation importance
│   │   ├── nn_shap_validation.py         # Neural network SHAP instability (87%)
│   │   ├── llm_attention_instability.py  # LLM attention instability under fine-tuning
│   │   ├── conditional_shap_threshold.py # Conditional escape threshold
│   │   ├── generate_figures.py           # All figure generation
│   │   └── ...                           # 37 more scripts (see paper/scripts/)
│   ├── figures/                          # 12 PDF figures
│   └── results_*.json                    # 33 JSON result files
│
├── docs/
│   ├── co-author-guide.md                # Plain English onboarding for collaborators
│   ├── verification-audit.md             # 32-item ranked human verification checklist
│   ├── self-verification-report.md       # Machine-verified #print axioms results
│   └── ...                               # Assessment, audit, and planning documents
│
├── lakefile.toml                         # Lean build configuration (Mathlib dependency)
├── lean-toolchain                        # leanprover/lean4:v4.29.0-rc8
├── CLAUDE.md                             # AI development instructions and conventions
└── README.md                             # This file
```

## Paper Versions

| Version | File | Pages | Target | Role |
|---------|------|-------|--------|------|
| **Monograph** | `paper/main_definitive.tex` | 66 | arXiv / Internal | Source of truth — every result, proof, experiment, documented failure |
| **JMLR** | `paper/main_jmlr.tex` | 54 | JMLR (primary) | Archival submission — full technical narrative |
| **NeurIPS** | `paper/main.tex` | 10 | NeurIPS 2026 | Impact version — core story in 10 pages |
| **Supplement** | `paper/supplement.tex` | 79 | NeurIPS | Everything that doesn't fit in 10 pages |

```
main_definitive.tex  (66pp, monograph, source of truth)
    ├── main_jmlr.tex    (54pp, JMLR submission)
    └── main.tex          (10pp, NeurIPS submission)
        └── supplement.tex   (79pp, NeurIPS supplement)
```

**Edit flow:** monograph → JMLR → NeurIPS. Always update the monograph first.

## Proof Architecture

**305 theorems. 16 axioms. 0 sorry. 54 files. 13 abstraction levels. 80 multi-step proofs (>=5 tactic lines).**

The Lean formalization caught 2 logical inconsistencies and 1 type mismatch that survived informal review. The axiom consistency proof (a `Fin 4` construction in [`Consistency.lean`](UniversalImpossibility/Consistency.lean)) demonstrates the axiom system is non-vacuous — there exists a concrete model satisfying all 16 axioms.

| Level | What It Proves | Files |
|-------|---------------|-------|
| 0 (pure logic) | `attribution_impossibility` — zero axiom deps | `Trilemma.lean` |
| 1 (framework) | IterativeOptimizer connects to Rashomon | `Iterative.lean` |
| 2 (instantiation) | Model-specific impossibilities | `General.lean`, `Lasso.lean`, `NeuralNet.lean` |
| 3 (quantitative) | 1/(1-rho^2) divergence (pure algebra) | `SplitGap.lean`, `Ratio.lean` |
| 4 (Spearman) | Spearman from midranks, stability bounds (derived) | `SpearmanDef.lean` |
| 5 (resolution) | DASH equity, combined results | `Corollary.lean`, `Impossibility.lean` |
| 6 (design space) | Complete design space characterization | `DesignSpace.lean`, `DesignSpaceFull.lean` |
| 7 (derivation) | `attribution_sum_symmetric` derived from axioms | `SymmetryDerive.lean` |
| 8 (generalization) | General SBD theorem | `SymmetricBayes.lean` |
| 9 (instances) | SBD applied to three domains | `ModelSelection.lean`, `ModelSelectionDesignSpace.lean`, `CausalDiscovery.lean`, `SBDInstances.lean` |
| 10 (extensions) | Conditional SHAP, fairness, intersectional fairness, flip rates | `ConditionalImpossibility.lean`, `FairnessAudit.lean`, `IntersectionalFairness.lean`, `FlipRate.lean` |
| 11 (bounds) | DASH optimality, efficiency, alpha-faithfulness, query complexity, mutual info, robustness, local sufficiency | `EnsembleBound.lean`, `Efficiency.lean`, `AlphaFaithful.lean`, `UnfaithfulBound.lean`, `PathConvergence.lean`, `QueryComplexity.lean`, `QueryComplexityParametric.lean`, `MutualInformation.lean`, `RobustnessLipschitz.lean`, `LocalSufficiency.lean` |
| 12 (universality) | Rashomon is inescapable, local >= global | `RashomonUniversality.lean`, `RashomonInevitability.lean`, `LocalGlobal.lean` |
| Strengthening | Per-model impossibility, qualitative impossibility, approximate equity, stump proportionality, bundled setup | `ProportionalityLocal.lean`, `Qualitative.lean`, `ApproximateEquity.lean`, `StumpProportionality.lean`, `Setup.lean` |
| Contrast | Bounded violations (documentation only) | `RandomForest.lean` |

### Proof Status Transparency

The paper distinguishes four levels of verification:
- **Proved:** Lean theorem with zero domain-axiom dependencies (e.g., `attribution_impossibility`)
- **Derived:** Lean theorem conditional on the axiom system (e.g., `gbdt_impossibility`)
- **Argued:** Supplement proof, not formalized in Lean (e.g., random forest convergence rate)
- **Empirical:** Experiment with reproducible script (e.g., 48% flip rate on Breast Cancer)

## Axiom System

16 axioms total (14 domain-specific + 2 query-complexity). The core impossibility (Level 0) uses **none** of these — only the Rashomon property as a hypothesis.

| # | Lean Name | Category | Plain English |
|---|-----------|----------|---------------|
| 1 | `Model` | Type declaration | "There exist trained models" |
| 2 | `numTrees` | Type declaration | "Models have T boosting rounds" |
| 3 | `numTrees_pos` | Type declaration | "T > 0" |
| 4 | `attribution` | Type declaration | "Each feature has an importance score in each model" |
| 5 | `splitCount` | Type declaration | "Each feature has a utilization count" |
| 6 | `firstMover` | Type declaration | "Each model has a first-mover feature" |
| 7 | `firstMover_surjective` | Domain | "Every feature can be the first-mover in some model" |
| 8 | `splitCount_firstMover` | Domain | "The first-mover gets T/(2-rho^2) splits" |
| 9 | `splitCount_nonFirstMover` | Domain | "Other features get (1-rho^2)T/(2-rho^2) splits" |
| 10 | `proportionality_global` | Domain | "Attribution = constant * split count" |
| 11 | `splitCount_crossGroup_symmetric` | Domain | "Features in different groups get equal splits" |
| 12 | `splitCount_crossGroup_stable` | Domain | "Cross-group splits are stable across models" |
| 13 | `modelMeasurableSpace` | Measure | "Models form a measurable space" |
| 14 | `modelMeasure` | Measure | "Models have a probability distribution" |
| 15 | `testing_constant` | Query complexity | "Le Cam testing constant exists" |
| 16 | `testing_constant_pos` | Query complexity | "Testing constant > 0" |

**Axiom stratification (verified by `#print axioms`):**

| Theorem | Axioms Used |
|---------|-------------|
| `attribution_impossibility` (core) | **0** (only Rashomon as hypothesis) |
| `impossibility_qualitative` | **0** (dominance + surjectivity as hypotheses) |
| `attribution_impossibility_bundled` | **0** (fully parametric via GBDTSetup) |
| `gbdt_impossibility_local` | **4** (surj, fm, nfm — no proportionality_global) |
| `impossibility` (quantitative) | **5** (+ proportionality_global for ratio) |
| `consensus_equity` (DASH) | **6** (+ cross-group symmetric) |
| Full system (DASH convergence + query complexity) | **16** |

**Formerly axiomatized, now derived:**
- `spearman_classical_bound` → `spearman_instability_bound` in SpearmanDef.lean (derived from split-count structure)
- `consensus_variance_bound` — theorem in Defs.lean (from attribution_variance_nonneg + Nat.cast_nonneg)
- `attribution_sum_symmetric` — theorem in SymmetryDerive.lean (from proportionality + split-count + cross-group + balance)
- `le_cam_lower_bound` — theorem in QueryComplexity.lean (contrapositive tautology)

## Experiments & Results

51 experiment scripts produce 33 JSON result files and 12 figures. 109 distinct findings are catalogued in [`paper/FINDINGS_MAP.md`](paper/FINDINGS_MAP.md) with tier classification.

| Experiment | Key Finding | Script |
|-----------|-------------|--------|
| Synthetic Gaussian | 50% flip rate for rho=0.9, matches theory | `generate_figures.py` |
| Monte Carlo flip rate | 1K trials, all 8 rho values validated | `monte_carlo_flip_rate.py` |
| Breast Cancer (Wisconsin) | 48% flip rate for correlated features | `f1_f5_validation.py` |
| 11 real datasets | Z-test \|r\| > 0.89 across all datasets | `real_world_validation.py` |
| XGBoost/LightGBM/CatBoost | Instability is implementation-independent | `cross_implementation_validation.py` |
| Permutation importance | 91% of correlated pairs unstable | `permutation_importance_validation.py` |
| Neural networks | 87% unstable; KernelSHAP noise 11% vs model instability 87% | `nn_shap_validation.py` |
| P=500 scalability | Runs in 2.5 minutes, \|r\|=0.876 | `high_dimensional_validation.py` |
| DASH vs alternatives | DASH dominates SAGE, PDP, permutation importance | `dash_vs_alternatives.py` |
| Prevalence survey | 77 datasets, 68% exhibit instability | `prevalence_survey.py` |
| German Credit, Taiwan CC, Lending Club | Positive control: correlated financial features flip | `financial_case_study.py` |
| LLM attention | Instability under fine-tuning | `llm_attention_instability.py` |
| Conditional SHAP sweep | Delta-beta threshold for escape | `conditional_shap_threshold.py` |
| SNR calibration | R^2=0.94 for SNR>=0.5 | `snr_calibration.py` |
| Causal DAG | Causal discovery instability | `causal_dag_experiment.py` |
| Longitudinal retraining | Time-series attribution stability | `longitudinal_retraining.py` |

All scripts use fixed random seeds and run on a standard laptop. Quick validation: `make validate` (~5 min). Full suite: `make experiments`.

## Building

### Lean 4
- Toolchain: `leanprover/lean4:v4.29.0-rc8`
- Build: `lake build` or `make lean` (~5 min cached, ~20 min first build)
- Expected: 2886 jobs, 0 errors, some unused-variable linter warnings

### LaTeX
- Requires TeX Live (jmlr.cls bundled in paper/)
- Build all: `make paper`
- Individual: `make jmlr`, `make neurips`, `make definitive`, `make supplement`

### Python
- Python 3.9+
- `pip install -r paper/scripts/requirements.txt`
- Optional: `pip install lightgbm torch openml` (for cross-impl, LLM, OpenML experiments)

## CI Pipeline

4 GitHub Actions jobs on every push:
1. **Lean 4 Build** — `lake build` (2886 jobs)
2. **Verify Theorem/Axiom Counts** — grep-based count check
3. **Paper Compilation Check** — pdflatex+bibtex for all 3 submission papers
4. **Axiom Consistency Model** — Python numerical verification of all 16 axioms

## Who Is This For?

**Data scientist using SHAP.** Your feature rankings for correlated features are unreliable. The instability is not noise or a software bug — it is a provable consequence of how gradient boosting interacts with collinearity. The fix is DASH: average SHAP values from multiple independently trained models. See the [dash-shap](https://github.com/DrakeCaraker/dash-shap) companion package and the [stability API in PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255) for the single-model screen to Z-test (multi-model validation) to DASH (ensemble consensus) workflow.

**Researcher in XAI or ML theory.** This is a formally verified impossibility theorem with 305 Lean proofs and 0 sorry. The Symmetric Bayes Dichotomy (Section 6 of the definitive paper) is a general proof technique from invariant decision theory that applies to any symmetric decision problem — we demonstrate it on feature attribution, model selection, and causal discovery under Markov equivalence. The Design Space Theorem characterizes the full achievable set.

**Regulator or model risk officer.** Single-model SHAP explanations are provably unreliable under collinearity. In a survey of 77 public datasets, 68% exhibit attribution instability. This affects EU AI Act Art. 13(3)(b)(ii) requirements for disclosing "known and foreseeable circumstances" affecting accuracy, and SR 11-7 model risk management compliance. The paper provides disclosure templates and a diagnostic workflow.

**Lean or Mathlib community.** 305 theorems across 54 files, 13 abstraction levels, using `MulAction` for orbit bounds, `ProbabilityTheory.cdf` for the Gaussian flip rate, and `Analysis.Calculus` for the FIM impossibility. The Gaussian CDF symmetry proofs (phi(0)=1/2, phi(-x)=1-phi(x)) via `NoAtoms` + `prob_compl_eq_one_sub` may be of independent interest. The axiom consistency proof constructs a `Fin 4` model satisfying all 16 axioms.

## Contributing

### Conventions (full list in CLAUDE.md)
- No `sorry` without a `-- TODO:` comment
- No `autoImplicit true` — all variables explicit
- Verify counts before committing: `make verify`
- Keep all 4 papers synchronized (monograph first)
- Do NOT commit paper changes without running the verification block

### Adding a New Theorem
1. Write the theorem in the appropriate .lean file
2. `lake build` to verify
3. Add to the monograph (main_definitive.tex) — cite by `\texttt{name}`
4. Add to the cross-reference table in the Lean section
5. Update paper/FINDINGS_MAP.md
6. Propagate to JMLR and NeurIPS papers as appropriate

### Adding a New Experiment
1. Write script in paper/scripts/
2. Run and save results to paper/results_*.json
3. Add subsection to monograph with key numbers
4. Add to experiment summary table
5. Update paper/FINDINGS_MAP.md
6. Propagate to JMLR paper

## Current State (verified 2026-04-08)

```
Theorems+lemmas: 305
Axioms:          16
Sorry:           0
Files:           54
```

## Citation

```bibtex
@article{caraker2026attribution,
  title={The Attribution Impossibility: No Feature Ranking Is Faithful, Stable,
         and Complete Under Collinearity},
  author={Caraker, Drake and Arnold, Bryan and Rhoads, David},
  year={2026},
  note={Lean 4 formalization: \url{https://github.com/DrakeCaraker/dash-impossibility-lean}}
}
```

## FAQ

**What is SHAP?**
SHAP (SHapley Additive exPlanations) assigns each feature an importance score for a model's prediction, based on cooperative game theory. It is the most widely used feature attribution method. See [Lundberg & Lee, 2017](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).

**What is Lean 4?**
A programming language and interactive theorem prover. You write mathematical proofs as code, and the computer checks every step. Think of it as a compiler for math. See [leanprover.github.io](https://leanprover.github.io/).

**What does "sorry" mean in Lean?**
It is a placeholder that says "trust me, this is true" without providing a proof — the Lean equivalent of a TODO. This project has 0 sorry: every claim is machine-verified.

**Is DASH just averaging?**
Yes — but with provable optimality guarantees. DASH computes the mean |SHAP| across M independently trained models. The paper proves this is the minimum-variance unbiased estimator (via Cramer-Rao / Titu's lemma) and gives the tight ensemble size formula M_min = ceil(2.71 * sigma^2 / Delta^2). "Just averaging" is like saying OLS is "just matrix inversion." The value is in understanding exactly when and why it is optimal.

**Does this apply to my dataset?**
Check two things: (1) Do you have features with |correlation| > 0.5? (2) Do those features have similar importance? If both yes, your SHAP rankings for those features are unreliable. Adapt `paper/scripts/f1_f5_validation.py` to your data to check.

**What about conditional SHAP?**
It does not help when features have equal causal effects (beta_j = beta_k). The impossibility extends to conditional attributions in that case. When causal effects differ (Delta-beta > ~0.2), conditional SHAP can resolve the instability. See [`ConditionalImpossibility.lean`](UniversalImpossibility/ConditionalImpossibility.lean).

**How does this relate to Arrow's theorem?**
Same structure: Arrow proves no voting system can satisfy IIA + Pareto + non-dictatorship simultaneously. We prove no ranking can satisfy faithfulness + stability + completeness simultaneously. Both are resolved by allowing ties (partial orders). Our relaxation paths converge (unlike Arrow's, which diverge) — reflecting the DGP symmetry that Arrow's heterogeneous voters lack.

**What is the relationship to dash-shap?**
This repository contains the theory (Lean proofs + paper). [dash-shap](https://github.com/DrakeCaraker/dash-shap) is the Python implementation: DASH pipeline, experiments, diagnostics. The stability API ([PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255)) provides `screen()`, `validate()`, `consensus()`, `report()`.

## References

- **[CLAUDE.md](CLAUDE.md)** — AI agent conventions, project rules, axiom inventory, "Do NOT" list
- **[paper/FINDINGS_MAP.md](paper/FINDINGS_MAP.md)** — Complete 109-finding reference with tier classification
- **[docs/co-author-guide.md](docs/co-author-guide.md)** — Co-author onboarding, verification checklist, timeline
- **[Companion code](https://github.com/DrakeCaraker/dash-shap)** — dash-shap Python package (stability API in PR #255)

## Related Work

- **Bilodeau et al. (2024):** Proved completeness + linearity cannot coexist; we address stability instead and provide a constructive resolution.
- **Chouldechova (2017):** Proved calibration + balance + equal FPR cannot coexist when base rates differ; our trilemma is the explainability analogue.
- **Arrow (1951):** Proved IIA + Pareto + non-dictatorship cannot coexist; same structural template, different domain.
- **Nipkow (2009):** Formalized Arrow's theorem in Isabelle/HOL; we extend this tradition to explainable AI in Lean 4.
- **Zhang et al. (2026):** Formalized statistical learning theory in Lean 4; complementary Lean formalization work.

## Authors

**Drake Caraker** — conceptualization, Lean formalization, experiments, software
**Bryan Arnold** — [role]
**David Rhoads** — [role]

Independent researchers. Contact: drakecaraker@gmail.com

---

**Paper:** "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity"
**Primary target:** JMLR | **Backup:** NeurIPS 2026 (abstract May 4, paper May 6)
**arXiv:** Preprint forthcoming. Run `paper/scripts/prepare_arxiv.sh` to prepare submission.
