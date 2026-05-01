# Universal Explanation Impossibility — Lean 4 Formalization

Lean 4 formalization of the universal explanation impossibility theorem. Target venues: Nature (flagship), JMLR (full technical), NeurIPS 2026 (companion). Companion attribution paper housed in [dash-shap](https://github.com/DrakeCaraker/dash-shap). The F5→F1→DASH stability API is in [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255). Physics companion: [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility).

## What This Proves

No explanation of an underspecified system can be simultaneously faithful (reflect the system's actual structure), stable (consistent across observationally equivalent configurations), and decisive (commit to a single answer) when the Rashomon property holds. The core theorem (`explanation_impossibility` in `ExplanationSystem.lean`) requires **zero model-specific axioms** — only the Rashomon property as a hypothesis.

A strengthened form, the **explanation bilemma** (`MaximalIncompatibility.lean`), shows that for maximally incompatible systems, even faithful + stable alone is impossible. The **tightness classification meta-theorem** (`BilemmaCharacterization.lean`) characterizes exactly which property pairs are achievable based on neutral/committal element existence. Enrichment (adding a neutral element) restores F+S at the cost of decisiveness; the enrichment is unique on the Rashomon fiber.

This is a **meta-theorem**: it applies uniformly to 9 ML explanation types — additive attributions (SHAP, IG, LIME), attention maps, counterfactual explanations, concept probes (TCAV), causal discovery (DAGs), model selection, saliency maps (GradCAM), LLM self-explanations, and mechanistic interpretability — and 14 cross-domain instances (Arrow's theorem, quantum contextuality, Duhem-Quine, gauge theory, statistical mechanics, genetic code, phase problem, QM interpretation, syntactic ambiguity, value alignment, view-update, linear systems, quantum measurement revolution, simultaneity revolution).

Model-specific instantiations from the companion attribution paper: GBDT has ratio 1/(1-ρ²) → ∞, Lasso has ratio ∞, neural nets have conditional violations, and random forests have bounded O(1/√T) violations. DASH (Diversified Aggregation for Stable Hypotheses; ensemble averaging) resolves the impossibility for balanced ensembles.

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
  ValueAlignment.lean           — AI value alignment impossibility; bilemma instance
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
  ── Universal framework (ExplanationSystem + instances) ──
  ExplanationSystem.lean        — Abstract ExplanationSystem; explanation_impossibility (0 axioms)
  AttributionInstance.lean      — Additive attribution instance (SHAP, IG, LIME)
  AttentionInstance.lean        — Attention map instance (DistilBERT)
  AttentionInstanceConstructive.lean — Constructive attention instance
  CounterfactualInstance.lean   — Counterfactual explanation instance
  CounterfactualInstanceConstructive.lean — Constructive counterfactual instance
  ConceptInstance.lean          — Concept probe instance (TCAV)
  ConceptInstanceConstructive.lean — Constructive concept instance
  CausalInstance.lean           — Causal discovery instance (Markov equivalence)
  CausalExplanationSystem.lean  — Causal explanation system abstraction
  ModelSelectionInstance.lean   — Model selection instance (Rashomon multiplicity)
  ModelSelectionInstanceConstructive.lean — Constructive model selection
  MechInterpInstance.lean       — Mechanistic interpretability instance
  MechInterpInstanceConstructive.lean — Constructive mech interp
  SaliencyInstance.lean         — Saliency map instance (GradCAM)
  SaliencyInstanceConstructive.lean — Constructive saliency
  LLMExplanationInstance.lean   — LLM self-explanation instance
  LLMExplanationInstanceConstructive.lean — Constructive LLM instance
  MarkovEquivalence.lean        — Derives Rashomon from Markov equivalence first principles
  Necessity.lean                — Necessity of Rashomon (possibility iff no Rashomon)
  NecessityBiconditional.lean   — Biconditional necessity
  UniversalImpossibility.lean   — Import hub; 9-instance inventory
  UniversalResolution.lean      — G-invariant resolution; gInvariant_stable
  UniversalDesignSpace.lean     — universal_design_space_dichotomy (Family A / B)
  DASHResolution.lean           — DASH as G-invariant resolution for attributions
  CPDAGResolution.lean          — CPDAG as G-invariant resolution for causal discovery
  Ubiquity.lean                 — generic_underspecification, ubiquity_impossibility
  ── Bilemma + strengthening ──
  MaximalIncompatibility.lean   — Bilemma, S+D impossibility, tightness, recovery (8 theorems)
  BilemmaCharacterization.lean  — Neutral element characterization (3 theorems)
  PredictiveConsequences.lean   — All-or-nothing, Rashomon unfaithfulness, faithful uniqueness (5 theorems)
  ApproximateRashomon.lean      — ε-stability extension (4 theorems)
  ── Cross-domain instances (14) ──
  ArrowInstance.lean            — Arrow's theorem (social choice)
  PeresMermin.lean              — Quantum contextuality
  DuhemQuine.lean               — Theory underdetermination
  GaugeTheory.lean              — Gauge theory (physics)
  StatisticalMechanics.lean     — Statistical mechanics
  GeneticCode.lean              — Codon degeneracy
  PhaseProblem.lean             — Crystallographic phase problem
  QMInterpretation.lean         — Quantum measurement interpretation
  SyntacticAmbiguity.lean       — Linguistic ambiguity
  ValueAlignment.lean           — AI value alignment impossibility
  ViewUpdate.lean               — Database view-update problem
  LinearSystem.lean             — Linear systems
  QuantumMeasurementRevolution.lean — Quantum measurement as paradigm shift
  SimultaneityRevolution.lean   — Relativity of simultaneity as paradigm shift
  ── Additional theorems ──
  AxiomSubstitution.lean        — Axiom substitution framework
  BayesOptimalTie.lean          — Bayes-optimal tie resolution
  BinaryQuantizer.lean          — Binary quantization
  Consistency.lean              — Internal consistency checks
  IntersectionalFairness.lean   — Intersectional fairness impossibility
  LocalSufficiency.lean         — Local sufficiency
  LossPreservation.lean         — Loss preservation under aggregation
  MeasureHypotheses.lean        — Measure-theoretic hypotheses
  MutualInformation.lean        — Mutual information non-uniqueness
  ParetoOptimality.lean         — Pareto optimality of resolution
  QuantitativeBound.lean        — Quantitative instability bounds
  QueryComplexityDerived.lean   — Derived query complexity
  QueryComplexityParametric.lean — Parametric query complexity
  QueryRelative.lean            — Relative query complexity
  RobustnessLipschitz.lean      — Lipschitz robustness
  StumpProportionality.lean     — Decision stump proportionality
  UnfaithfulQuantitative.lean   — Quantitative unfaithfulness
  VarianceDerivation.lean       — Variance derivation from axioms
paper/
  nature_article.tex            — Nature submission (~2650 words)
  universal_impossibility_monograph.tex — Definitive arXiv version (~4400 lines)
  universal_impossibility_jmlr.tex — JMLR submission
  universal_impossibility_neurips.tex — NeurIPS 2026 version
  supplementary_information.tex — Supplementary Information
  references.bib                — References
  scripts/                      — Experiment scripts (figure generation, validation, diagnostics)
  figures/                      — PDF figures
  sections/                     — LaTeX section fragments for universal paper instances
knockout-experiments/           — Empirical validation (90+ scripts, 80+ result JSONs)
  CORRECTIONS.md                — Issue tracker (P0-P3 priority)
  RESULTS_SYNTHESIS.md          — 3 confirmed, 2 falsified, 1 negative
  PRE_REGISTRATION.md           — Pre-registered predictions
```

## Lean State: 102 files, 2 axioms, 519 theorems+lemmas, 0 sorry

## Axiom Inventory (2 total)

All GBDT infrastructure is bundled into two structure axioms in Defs.lean:

| Axiom | Type | Contains |
|-------|------|----------|
| `gbdtWorld` | `GBDTWorld` | Model type, numTrees, numTrees_pos, modelMeasurableSpace, modelMeasure |
| `gbdtAxioms` | `GBDTAxiomsBundle gbdtWorld fs` | attribution, splitCount, firstMover + 6 behavioral properties |

All former axioms (Model, attribution, firstMover_surjective, etc.) are now `noncomputable def` or `theorem` extracting from these bundles. Physics axioms (BH, QG frameworks) converted to section variables.

**Axiom stratification (verified by `#print axioms`):**
- **Core universal impossibility** (`explanation_impossibility`): ZERO axioms (Rashomon is a hypothesis)
- **Core attribution impossibility** (`attribution_impossibility`): ZERO behavioral axioms (only gbdtWorld)
- **Bundled impossibility** (`attribution_impossibility_bundled`): ZERO axioms (fully parametric via GBDTSetup)
- **GBDT quantitative bounds**: 2 axioms (gbdtWorld + gbdtAxioms)

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

## Paper Strategy

Four papers forming a layered publication strategy:

| Paper | File | Venue | Status |
|-------|------|-------|--------|
| **Universal (flagship)** | `paper/nature_article.tex` | Nature | Ready for submission |
| **Universal (full technical)** | `paper/universal_impossibility_monograph.tex` | arXiv (definitive) | Ready |
| **Universal (JMLR)** | `paper/universal_impossibility_jmlr.tex` | JMLR | Ready |
| **Universal (NeurIPS)** | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 | Ready |
| **Attribution (companion)** | In dash-shap repo | NeurIPS 2026 / JMLR | ~70% ready |
| **Physics (companion)** | `ostrowski-impossibility/paper/notes.tex` | arXiv (hep-th) | Draft complete |

**Strategy:** Nature for flagship (broadest audience, cross-domain framing). arXiv monograph for priority + definitive reference. Attribution paper to NeurIPS (different paper — SHAP-specific). Physics companion to arXiv for priority.

**Do not dual-submit** the universal paper to both Nature and NeurIPS — they overlap substantially.

## Naming Conventions

**Canonical reference:** `docs/naming-conventions.md`. All papers, docs, and new code must use these names. Key terms:

- **The explanation trilemma** (F+S+D impossible), **the explanation bilemma** (F+S impossible for binary H)
- **Explanation capacity** (C = dim V^G), **explanation loss rate** (η = 1 − C/dim V)
- **Explanation Capacity Theorem** (capacity predicts instability; NOT "law")
- **Explanation uncertainty bound** (unfaith₁+unfaith₂ ≥ Δ−δ; NOT "tradeoff bound" or "uncertainty relation")
- **The stable projection** (orbit average / Reynolds projection / DASH; NOT "the explanation code")
- **Stable fact count** (g(g−1)/2 stable pairwise facts; NOT "Noether counting law")
- **Explanatory information loss** (‖v−Rv‖²+‖Rv‖²=‖v‖²; NOT "Pythagorean decomposition" as prose name)
- **Over-explanation penalty** (‖w‖ ≤ ‖u−w‖; NOT "beyond-capacity penalty")
- **Explanation Stability Theorem** (the 4-part theorem; NOT "Explanation Coding Theorem")
- **Explanation stability convergence rate** (MSE = tr(RΣR)/M)
- **Stability threshold** (M*(ε) = tr(RΣR)/ε)

## Submission

- **Nature**: `paper/nature_article.tex` (~2500 words) + SI + Extended Data. Rolling submission.
- **arXiv**: Post monograph (`paper/universal_impossibility_monograph.tex`) for priority. Run `paper/scripts/prepare_arxiv.sh` to uncomment authors and fill URLs.
- **NeurIPS 2026** (attribution paper): Abstract May 4, Paper May 6. Official `neurips_2026.sty`.
- Universal paper title: "The Limits of Explanation"
- Attribution paper title: "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity"
- Authors: Drake Caraker, Bryan Arnold, David Rhoads
- Companion code: [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255)

## Do NOT

- Commit paper changes without verifying paper-code consistency. Run this verification block and confirm all numbers match the paper text before committing:
  ```bash
  grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
  grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
  grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'  # Note: grep counts "sorry" in doc-comments too; actual sorry tactics = 0
  ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
  ```
- Use `sorry` without a `-- TODO:` comment explaining what's needed
- Change axioms without re-running the SymPy verification (in companion repo: `dash-shap/paper/proofs/verify_lemma6_algebra.py`)
- Add `autoImplicit true` — all variables must be explicit
- Claim "N theorems" without verifying — count with `grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'` (currently 519)
- Run parallel subagents that both modify the same file (causes build cache corruption)
- Axiomatize quantities that can be defined — prefer definitions with axiomatized bounds (see SpearmanDef.lean pattern)
- Claim empirical results as "proved" or "Lean-verified" — distinguish: **proved** (zero axiom deps), **derived** (from axioms), **argued** (supplement proof only), **empirical** (experiments). The paper's "Proof status transparency" paragraph is the reference.

## Comprehensive Reference: Proofs, Experiments, Results

**Full reference with explanations, methodology, and provenance:** [`docs/comprehensive-reference.md`](docs/comprehensive-reference.md)

**Archived publication strategy (superseded):** [`docs/archive/publication-strategy-complete-2026-04-30.md`](docs/archive/publication-strategy-complete-2026-04-30.md)

### Core Lean Theorems (519 total, 2 axioms, 0 sorry)

**Level 0 — Universal impossibility (zero axioms):**
- `explanation_impossibility` (ExplanationSystem.lean): F+S+D impossible under Rashomon. 4-line proof.
- `bilemma` (MaximalIncompatibility.lean): F+S impossible for maximally incompatible H.
- `universal_design_space_dichotomy` (UniversalDesignSpace.lean): Every method is Family A or B.
- `mi_is_exact_boundary` (MutualInformation.lean): MI > 0 is necessary and sufficient for impossibility.
- `mi_quantitative_unfaithfulness` (MIQuantitativeBridge.lean): MI > 0 → unfaithfulness ≥ Δ/2.

**Level 1 — Resolution and optimality:**
- `gInvariant_stable` (UniversalResolution.lean): G-invariant maps are stable.
- `uncertainty_from_symmetry` (UncertaintyFromSymmetry.lean): Pythagorean decomposition ‖v-Rv‖²+‖Rv‖²=‖v‖².
- `best_approximation` / `reynolds_best_approximation`: Rv is closest fixed point to v.
- `beyond_capacity_penalty`: ‖w‖ ≤ ‖u-w‖ for w ∈ (V^G)⊥.
- `stable_in_fixed_subspace`: Stable maps have image in V^G.
- `dash_unique_pareto_optimal` (ParetoOptimality.lean): DASH is unique Pareto-optimal for within-group.
- `pareto_frontier_dichotomy`: Disagreement is exactly 0 (tie) or 1/2 (commitment).

**Level 2 — Tightness and enrichment:**
- `tightness_full` / `tightness_collapsed` (BilemmaCharacterization.lean): Classification by neutral/committal elements.
- `enrichment_unique_on_fiber` (MaximalIncompatibility.lean): Enrichment is unique on Rashomon fiber.
- `enrichment_creates_new_impossibility` / `levels_independent` (RecursiveImpossibility.lean): Enrichment stack.
- `bh_bilemma` / `qg_bilemma` (EnrichmentStack.lean): Physics enrichment levels (now parametric, 0 axioms).

**Level 3 — Quantitative bounds (2 axioms: gbdtWorld, gbdtAxioms):**
- `attribution_impossibility` (Trilemma.lean): GBDT-specific impossibility.
- `ratio_tendsto_atTop` (Ratio.lean): Attribution ratio 1/(1-ρ²) → ∞.
- `consensus_equity` (Corollary.lean): DASH achieves equity for balanced ensembles.
- `spearman_instability_bound` (SpearmanDef.lean): Derived Spearman bound.
- `flip_rate_binary_group` (FlipRate.lean): Exact GBDT flip rate.

**Level 4 — 9 ML instances + 14 cross-domain (all constructive, zero axioms):**
- AttributionInstanceConstructive, AttentionInstanceConstructive, CounterfactualInstanceConstructive, ConceptInstanceConstructive, CausalInstanceConstructive, ModelSelectionInstanceConstructive, MechInterpInstanceConstructive, SaliencyInstanceConstructive, LLMExplanationInstanceConstructive
- ArrowInstance, PeresMermin, DuhemQuine, GaugeTheory, StatisticalMechanics, GeneticCode, PhaseProblem, QMInterpretation, SyntacticAmbiguity, ValueAlignment, ViewUpdate, LinearSystem, QuantumMeasurementRevolution, SimultaneityRevolution

### Key Experiment Results (116 JSON files)

**Capacity Audit (149 datasets):**
- `results_audit_150_final.json`: 149 datasets, 53 domains, 50 seeds each. 75% exceed capacity (ρ*=0.70). Wilcoxon p=5.09e-11 (cross-dataset). 27:1 at p<0.005.
- `results_audit_strengthening.json`: Null model test (randomization p=0.001 synthetic), bootstrap group stability (ARI>0.94), family-level Wilcoxon (p=2.5e-8), block bootstrap CI [0.036,0.069], η law on real data (ρ=0.40, p=0.011).
- `results_final_gaps.json`: Pooled per-pair OLS (6 datasets), real-world stratified (5/5 quintiles on 5 datasets), SAGE vs MI vs correlation comparison.
- `results_open_questions_final.json`: SAGE wins 6/7 (gap 0.25 vs 0.05), 14-dataset OLS (coef=0.037, CI [0.014,0.045]), cluster-bootstrap confirms.
- `results_mi_reaudit.json`: MI vs correlation across 131 PMLB datasets. ARI=0.84 mean, identical 77%. MI ≈ correlation for continuous features.

**Gene Expression:**
- `results_gene_expression_replication.json`: TSPAN8 92%/CEACAM5 6% on AP_Colon_Kidney. Replicates on 3 additional datasets. ρ=0.858 feature correlation. max_depth=4.

**Mechanistic Interpretability:**
- `results_mi_v2_final_validation.json`: 10 transformers, ρ=0.518 raw → 0.929 G-invariant. Fourier Jaccard=2.2%. Within-layer flip=0.500 exactly.
- `results_mech_interp_definitive_v2.json`: Full circuit stability analysis, Noether counting, control hierarchy.
- `results_mi_audit.json`: Adversarial audit — alignment lift 1.92x, deeper than permutation symmetry.

**Neuroimaging:**
- `results_brain_imaging_bulletproof.json`: M₉₅=16 [10,22]. Network structure predicts disagreement (d=0.32 after activation control).

**Universal η Law:**
- `results_universal_eta.json`: 16 instances, R²=0.60 (all), R²=0.957 (7 well-characterised). Slope=0.91.
- `results_eta_law_oos_gof.json`: LOO-CV R²=0.79, holdout R²=0.24 (9 approximate groups).

**Drug Discovery:**
- `results_drug_discovery_mi_clustering.json`: BBBP binary fingerprints. Pearson: 0% predicted, MI: 19%, actual: 23%.

**Capstone (OQ-29 through OQ-50):**
- `results_open_questions_capstone.json`: Gaussian flip formula per-pair R²=0.946–0.980 (6 datasets). Regularization creates Rashomon (falsified prediction). Conformal SHAP intervals (84–88% coverage). Rashomon topology (1/6 bimodal). DASH tie rate (4/6 directional).

### Paper Files

| Paper | File | Venue | Key numbers |
|-------|------|-------|-------------|
| Nature (flagship) | `paper/nature_article.tex` | Nature | 519 theorems, 4 instances, R²=0.96 |
| Monograph (definitive) | `paper/universal_impossibility_monograph.tex` | arXiv | ~4500 lines, all proofs |
| Supplement | `paper/supplementary_information.tex` | Nature SI | SAGE algorithm, falsifications |
| JMLR | `paper/universal_impossibility_jmlr.tex` | JMLR | Full technical |
| NeurIPS | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 | Companion |

### Verified State (as of 2026-04-30)
```
Theorems: 519 | Axioms: 2 | Files: 102 | Sorry: 0
Build: lake build → 2954 jobs, 0 errors
Papers: 0 stale numbers (last verified this session)
```
