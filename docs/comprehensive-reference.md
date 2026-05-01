# Comprehensive Reference: The Limits of Explanation

Every proof, experiment, result, and paper in this project. Designed for sharing with other sessions.

**Verified state (2026-04-30): 519 theorems, 2 axioms, 102 Lean files, 0 sorry. Build: 2954 jobs, 0 errors.**

---

## 1. The Core Theorem

No explanation of an underspecified system can simultaneously be **faithful** (reflect actual structure), **stable** (consistent across equivalent configurations), and **decisive** (commit to a single answer) — whenever the **Rashomon property** holds.

- **Lean**: `ExplanationSystem.lean` → `explanation_impossibility` (4 lines, zero axioms)
- **Paper**: Nature Theorem 1; Monograph §2

The proof requires ONLY the Rashomon property as a hypothesis — no domain-specific axioms. A strengthened form (the **bilemma**, `MaximalIncompatibility.lean`) shows F+S alone is impossible for binary explanation spaces.

---

## 2. Complete Lean File Inventory (102 files, 519 theorems)

### Universal Framework (zero axioms)

| File | Theorems | Content |
|------|----------|---------|
| `ExplanationSystem.lean` | 4 | Abstract ExplanationSystem type; `explanation_impossibility` |
| `MaximalIncompatibility.lean` | 7 | Bilemma, S+D impossibility, tightness, recovery |
| `BilemmaCharacterization.lean` | 3 | Neutral element characterization |
| `PredictiveConsequences.lean` | 5 | All-or-nothing, Rashomon unfaithfulness |
| `ApproximateRashomon.lean` | 4 | ε-stability extension |
| `Necessity.lean` | 3 | Necessity of Rashomon (possibility iff no Rashomon) |
| `NecessityBiconditional.lean` | 3 | Biconditional necessity |
| `UniversalDesignSpace.lean` | 1 | `universal_design_space_dichotomy` (Family A/B) |
| `UniversalResolution.lean` | 1 | `gInvariant_stable` |
| `UniversalImpossibility.lean` | 0 | Import hub; 9-instance inventory documentation |
| `Ubiquity.lean` | 3 | Structural ubiquity: dimensional argument + impossibility bridge |
| `GeneralizedBilemma.lean` | 13 | Generalized bilemma for arbitrary incompatibility |
| `AxiomSubstitution.lean` | 14 | Axiom substitution framework |
| `RecursiveImpossibility.lean` | 7 | Enrichment stack: levels independent, depth unbounded |
| `EnrichmentStack.lean` | 18 | Physical enrichment stack (BH, QG — now parametric) |
| `GoedelIncompleteness.lean` | 5 | Gödel-pattern parallel |
| `ExplanationLandscape.lean` | 8 | Coverage conflict ↔ no neutral element bridge |

### MI Generalization (zero axioms)

| File | Theorems | Content |
|------|----------|---------|
| `MutualInformation.lean` | 11 | `mi_is_exact_boundary`: MI>0 is necessary and sufficient |
| `MIQuantitativeBridge.lean` | 5 | MI>0 → unfaithfulness ≥ Δ/2 |

### Resolution and Optimality

| File | Theorems | Content |
|------|----------|---------|
| `UncertaintyFromSymmetry.lean` | 16 | Pythagorean decomposition, best approximation, over-explanation penalty |
| `ParetoOptimality.lean` | 7 | DASH unique Pareto-optimal; frontier dichotomy (0 or 1/2) |
| `BayesOptimalTie.lean` | 4 | Bayes-optimality of ties for symmetric features |
| `DASHResolution.lean` | 2 | DASH as G-invariant resolution for attributions |
| `CPDAGResolution.lean` | 2 | CPDAG as G-invariant resolution for causal discovery |

### 9 ML Instances (all constructive, zero axioms)

| File | Theorems | Domain |
|------|----------|--------|
| `AttributionInstanceConstructive.lean` | 2 | SHAP, IG, LIME |
| `AttentionInstanceConstructive.lean` | 3 | Attention maps (DistilBERT) |
| `CounterfactualInstanceConstructive.lean` | 3 | Counterfactual explanations |
| `ConceptInstanceConstructive.lean` | 3 | Concept probes (TCAV) |
| `CausalInstanceConstructive.lean` | 2 | Causal discovery (DAG Markov equivalence) |
| `ModelSelectionInstanceConstructive.lean` | 3 | Model selection (Rashomon multiplicity) |
| `MechInterpInstanceConstructive.lean` | 3 | Mechanistic interpretability |
| `SaliencyInstanceConstructive.lean` | 3 | Saliency maps (GradCAM) |
| `LLMExplanationInstanceConstructive.lean` | 3 | LLM self-explanations |
| `MarkovEquivalence.lean` | 4 | Derives Rashomon from Markov equivalence first principles |
| `CausalExplanationSystem.lean` | 1 | Causal explanation system abstraction |

### 14 Cross-Domain Instances (all constructive, zero axioms)

| File | Theorems | Domain |
|------|----------|--------|
| `ArrowInstance.lean` | 3 | Arrow's theorem (social choice) |
| `PeresMermin.lean` | 3 | Quantum contextuality |
| `DuhemQuine.lean` | 6 | Theory underdetermination |
| `GaugeTheory.lean` | 5 | Gauge theory (physics) |
| `StatisticalMechanics.lean` | 5 | Statistical mechanics |
| `GeneticCode.lean` | 3 | Codon degeneracy |
| `PhaseProblem.lean` | 4 | Crystallographic phase problem |
| `QMInterpretation.lean` | 3 | Quantum measurement interpretation |
| `SyntacticAmbiguity.lean` | 3 | Linguistic ambiguity |
| `ValueAlignment.lean` | 2 | AI value alignment impossibility |
| `ViewUpdate.lean` | 3 | Database view-update problem |
| `LinearSystem.lean` | 3 | Linear systems |
| `QuantumMeasurementRevolution.lean` | 2 | Quantum measurement as paradigm shift |
| `SimultaneityRevolution.lean` | 2 | Relativity of simultaneity as paradigm shift |

### GBDT-Specific Quantitative Bounds (2 bundled axioms)

| File | Theorems | Content |
|------|----------|---------|
| `Defs.lean` | 11 | FeatureSpace, GBDTWorld/GBDTAxiomsBundle structures, definitions |
| `Trilemma.lean` | 2 | RashomonProperty, `attribution_impossibility` |
| `General.lean` | 6 | GBDT instance, gbdt_impossibility |
| `Iterative.lean` | 2 | IterativeOptimizer abstraction |
| `SplitGap.lean` | 7 | split_gap_exact, split_gap_ge_half |
| `Ratio.lean` | 3 | Attribution ratio 1/(1-ρ²), `ratio_tendsto_atTop` |
| `SpearmanDef.lean` | 20 | Spearman from midranks, qualitative + quantitative bounds |
| `Lasso.lean` | 2 | Lasso impossibility (ratio = ∞) |
| `NeuralNet.lean` | 1 | Neural net impossibility (conditional) |
| `Impossibility.lean` | 4 | Combined: equity violation + stability bound |
| `Corollary.lean` | 5 | DASH consensus equity, variance convergence |
| `DesignSpace.lean` | 6 | Design Space Theorem (composite), DASH ties |
| `DesignSpaceFull.lean` | 3 | Design Space exhaustiveness (Family A or B) |
| `SymmetryDerive.lean` | 1 | `attribution_sum_symmetric` (DERIVED from axioms) |
| `FlipRate.lean` | 13 | Exact GBDT flip rate, binary group = coin flip |
| `GaussianFlipRate.lean` | 13 | Gaussian CDF Φ, flip rate formula |
| `FIMImpossibility.lean` | 6 | Gaussian FIM impossibility, Rashomon ellipsoid |
| `QueryComplexity.lean` | 16 | Query complexity Ω(σ²/Δ²), Le Cam structural |
| `QueryComplexityDerived.lean` | 12 | Chebyshev-derived query complexity |
| `QueryComplexityParametric.lean` | 10 | Parametric query complexity (axiom-free) |
| `QueryRelative.lean` | 2 | Relative query complexity |
| `EnsembleBound.lean` | 9 | DASH variance optimality + ensemble size |
| `Efficiency.lean` | 10 | SHAP efficiency amplification m/(m-1) |
| `ModelSelection.lean` | 2 | Model selection impossibility |
| `ModelSelectionDesignSpace.lean` | 3 | Model selection design space |
| `CausalDiscovery.lean` | 3 | Causal discovery impossibility |
| `SymmetricBayes.lean` | 16 | General SBD: orbit bounds, trichotomy |
| `SBDInstances.lean` | 5 | SBD instances + abstract aggregation |
| `ConditionalImpossibility.lean` | 3 | Conditional SHAP impossibility + escape |
| `FairnessAudit.lean` | 4 | Fairness audit impossibility |
| `IntersectionalFairness.lean` | 13 | Intersectional fairness compounding |
| `LocalGlobal.lean` | 2 | Local ≥ global instability |
| `RashomonUniversality.lean` | 4 | Rashomon from symmetry via feature swap |
| `RashomonInevitability.lean` | 5 | Impossibility is inescapable for standard ML |
| `AlphaFaithful.lean` | 3 | α-faithfulness bound |
| `UnfaithfulBound.lean` | 3 | Unfaithfulness ≥ 1/2, ties optimal |
| `UnfaithfulQuantitative.lean` | 2 | Pr(unfaithfulness) = 1/2 |
| `PathConvergence.lean` | 3 | Relaxation path convergence |
| `ProportionalityLocal.lean` | 4 | Impossibility from per-model c only |
| `Qualitative.lean` | 4 | Impossibility from 2 axioms: dominance + surjectivity |
| `ApproximateEquity.lean` | 3 | Rashomon from bounded proportionality |
| `VarianceDerivation.lean` | 6 | Variance derivation from independence |
| `StumpProportionality.lean` | 6 | Decision stump proportionality |
| `BinaryQuantizer.lean` | 7 | Binary quantization |
| `Consistency.lean` | 7 | Internal consistency checks |
| `LossPreservation.lean` | 3 | Loss preservation under aggregation |
| `MeasureHypotheses.lean` | 0 | Measure-theoretic hypothesis documentation |
| `LocalSufficiency.lean` | 6 | Local proportionality suffices for all bounds |
| `RobustnessLipschitz.lean` | 11 | Lipschitz robustness |
| `QuantitativeBound.lean` | 3 | Quantitative instability bounds |
| `Setup.lean` | 1 | GBDTSetup structure (backward-compatible bundling) |
| `RandomForest.lean` | 0 | Documentation only (bounded violations, no formal proofs) |
| `Basic.lean` | 0 | Import hub |

---

## 3. Complete Experiment Results Inventory (116 JSON files)

### Capacity Audit (the 149-dataset validation)

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_audit_150_final.json` | 149 datasets, 53 domains, 50 seeds each | 75% exceedance, Wilcoxon p=5.1e-11, 27:1 |
| `results_audit_150_clean.json` | Cleaned version of above | Same |
| `results_audit_150_checkpoint.json` | Intermediate checkpoint | — |
| `results_capacity_audit_150.json` | Raw capacity audit (153 datasets) | Predecessor |
| `results_capacity_audit_expanded.json` | 63-dataset expanded audit | Earlier version |
| `results_capacity_audit_pilot.json` | 9-dataset pilot | Earlier version |
| `results_audit_expanded_partial.json` | Partial expanded results | — |

### Audit Strengthening (session experiments)

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_audit_strengthening.json` | Null model, bootstrap, family-level, weighted η | Family p=2.5e-8, CI [0.036,0.069], η ρ=0.40 |
| `results_final_gaps.json` | Pooled OLS, stratified, alt clustering | OLS coef=0.028, 5/5 quintiles (9 datasets) |
| `results_final_comprehensive.json` | 6-dataset pooled (incl Dermatology) | Dermatology preprocessing discrepancy |
| `results_final_9dataset.json` | 9-dataset summary | coef=0.036, CI [0.007,0.045] |
| `results_open_questions_final.json` | SAGE comparison, 14-dataset pooled | SAGE gap=0.25, 14-dataset CI [0.014,0.045] |
| `results_open_questions_capstone.json` | Gaussian flip R², conformal, reg, topology | R²=0.946-0.980, coverage 84-88% |
| `results_mi_reaudit.json` | MI vs correlation across 131 PMLB datasets | ARI=0.84, identical 77% |
| `results_coding_theorem_validation.json` | Convergence rate, beyond-capacity MSE | β=-1.28, MSE/‖w‖²=1.000 |

### Gene Expression

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_gene_expression_replication.json` | 4 datasets, TSPAN8/CEACAM5 alternation | 92%/6%, ρ=0.858 |
| `results_gene_expression.json` | Earlier version | — |
| `results_gene_expression_validation.json` | Validation run | — |
| `results_gene_expression_shap.json` | SHAP-specific analysis | — |
| `results_go_enrichment.json` | Gene Ontology enrichment | Zero shared BP terms |

### Mechanistic Interpretability

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_mi_v2_final_validation.json` | 10 transformers, G-invariant projection | ρ=0.518→0.929, Jaccard=2.2% |
| `results_mi_v2_comprehensive.json` | Comprehensive MI v2 analysis | — |
| `results_mech_interp_definitive_v2.json` | Full circuit stability | Noether counting confirmed |
| `results_mech_interp_definitive.json` | Earlier version | — |
| `results_mech_interp_stability.json` | Basic stability analysis | — |
| `results_mech_interp_rashomon.json` | Rashomon set analysis | — |
| `results_mech_interp_rashomon_gpu.json` | GPU-accelerated version | — |
| `results_mi_nonuniqueness.json` | MLP feature probing non-uniqueness | Jaccard=0.041≈chance |
| `results_mi_audit.json` | Adversarial audit of non-uniqueness | Alignment lift 1.92x |
| `results_mi_perturbation_dose_response.json` | Dose-response analysis | — |
| `results_comprehensive_circuit_stability.json` | Multi-config circuit stability | Configs A/B/C |
| `results_circuit_stability_configA/B/C.json` | Per-config results | — |
| `results_tinystories_circuit_stability_pilot.json` | TinyStories pilot | ρ=0.565→0.972 |
| `results_mean_ablation_comparison.json` | Mean vs zero ablation | Both give ρ≈0.97 |
| `results_continuous_symmetry.json` | CCA spectrum analysis | — |
| `results_neuron_sage.json` | SAGE on neurons | — |

### Brain Imaging / Neuroimaging

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_brain_imaging_bulletproof.json` | Definitive NARPS analysis | M₉₅=16 [10,22], d=0.32 |
| `results_brain_imaging_definitive.json` | Definitive version | — |
| `results_brain_imaging_rashomon_v3.json` | Rashomon analysis v3 | — |
| `results_brain_imaging_rashomon_v2.json` | v2 | — |
| `results_brain_imaging_rashomon.json` | v1 | — |
| `results_brain_imaging_noether_direct.json` | Noether direct test | — |
| `results_brain_imaging_knockout.json` | Knockout analysis | — |
| `results_brain_imaging_eta_approx.json` | Approximate η | — |
| `results_brain_imaging_resolution.json` | Resolution analysis | — |
| `results_multi_analyst_bulletproof.json` | Multi-analyst convergence | — |
| `results_multi_analyst_resolution.json` | Multi-analyst resolution | — |

### Universal η Law / Capacity Theorem

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_universal_eta.json` | 16 instances | R²=0.957 (7), R²=0.60 (16) |
| `results_eta_law_oos_gof.json` | Out-of-sample goodness of fit | LOO R²=0.79, holdout R²=0.24 |
| `results_universal_law.json` | Earlier version | — |
| `results_gaussian_eta_rescue.json` | Gaussian correction | R²: 0.60→0.79 |

### Gaussian Flip Formula

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_gaussian_flip_validated.json` | Validated formula | — |
| `results_gaussian_flip_cv.json` | Cross-validated | — |
| `results_gaussian_flip_knockout.json` | Knockout test | — |
| `results_gaussian_diagnostics.json` | Diagnostic tests | — |
| `results_gaussian_multimodel.json` | Multi-model comparison | — |

### Stable Fact Count (Noether)

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_noether_counting.json` | Core counting experiment | 47.1pp bimodal gap |
| `results_noether_permutation_test.json` | Permutation test | — |
| `results_noether_sensitivity.json` | Sensitivity across ρ | Gap invariant 0.50-0.99 |
| `results_noether_real.json` | Real datasets | — |
| `results_noether_treeshap.json` | TreeSHAP version | — |
| `results_noether_cross_domain.json` | Cross-domain | — |

### SAGE Algorithm

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_sage_discovery.json` | Group discovery | — |
| `results_sage_audit.json` | Adversarial audit | R²=0.809, LOO=0.689 |
| `results_sage_baseline_comparison.json` | vs 5 baselines | SAGE 28× better |
| `results_sage_real.json` | Real-world datasets | — |
| `results_sage_treeshap.json` | TreeSHAP version | — |

### Drug Discovery

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_drug_discovery_mi_clustering.json` | MI on binary fingerprints | Pearson 0%, MI 19%, actual 23% |
| `results_drug_discovery_prediction.json` | Prediction validation | — |
| `results_drug_discovery_prospective.json` | Prospective test | — |

### Clinical / Regulatory

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_clinical_decision_reversal.json` | Decision reversal analysis | — |
| `results_clinical_decision_reversal_v2.json` | v2 | 48% of credit applicants |
| `results_clinical_shap_audit.json` | Clinical TreeSHAP audit | 28% unreliable top-5 |
| `results_regulatory_compliance.json` | Regulatory compliance metrics | 83% gap |
| `results_regulatory_topk.json` | Top-k regulatory audit | — |
| `results_reliability_summary.json` | Reliability summary | — |

### Multiverse Analysis

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_multiverse.json` | Multiverse analysis | — |
| `results_multiverse_corrected.json` | Corrected version | — |
| `results_multiverse_expanded.json` | Expanded version | — |
| `results_multiverse_layer_decomposition.json` | Layer decomposition | — |

### Enrichment / Abstraction

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_enrichment_demo.json` | Enrichment demonstration | — |
| `results_enrichment_statistical_tests.json` | Statistical tests | — |
| `results_abstraction_balanced_control.json` | Balanced control | — |
| `results_abstraction_enrichment_expanded.json` | Expanded enrichment | — |
| `results_abstraction_random_control.json` | Random control | — |
| `results_alignment_abstraction.json` | Alignment abstraction | — |

### Other Experiments

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_approximate_symmetry.json` | Approximate symmetry | — |
| `results_approximate_symmetry_v2.json` | v2 | — |
| `results_attribution_pca_eigenspectrum.json` | PCA eigenspectrum | Gap at PC10 not PC3 |
| `results_bootstrap_calibration.json` | Bootstrap calibration | — |
| `results_causal_sachs.json` | Sachs causal dataset | — |
| `results_dimVG_vs_classes.json` | dim(V^G) vs classes | — |
| `results_explanation_landscape_bridge.json` | Landscape bridge | — |
| `results_explanation_landscape_bridge_expanded.json` | Expanded | ρ=0.961 across 15 datasets |
| `results_final_experiments.json` | Final experiments | — |
| `results_flip_generalization.json` | Flip generalization | — |
| `results_flip_gen_expanded.json` | Expanded | — |
| `results_flip_rate_robustness.json` | Flip rate robustness | — |
| `results_knockout_battery.json` | Knockout battery | — |
| `results_lean_complexity.json` | Lean complexity analysis | — |
| `results_llm_self_explanation.json` | LLM self-explanation | Citation flip rate 34.5% |
| `results_molecular_evolution.json` | Molecular evolution | R²=0.0 (falsified) |
| `results_phase_transition_r.json` | Phase transition | Falsified (r*≠1) |
| `results_power_analysis.json` | Power analysis | — |
| `results_predictor_comparison.json` | 7-predictor comparison | Max ρ=0.26 |
| `results_proportionality_sensitivity.json` | Proportionality sensitivity | — |
| `results_published_findings_audit.json` | Published findings audit | — |
| `results_quantitative_bilemma_test.json` | Bilemma test | — |
| `results_quantum_verification.json` | Quantum verification | — |
| `results_rashomon_topology.json` | Rashomon topology | — |
| `results_spectral_gap_ensemble.json` | Spectral gap | — |
| `results_topk_expanded_and_null.json` | Top-k analysis | — |
| `results_uncertainty_principle.json` | Uncertainty principle | — |
| `results_value_alignment_compas.json` | Value alignment COMPAS | — |

---

## 4. Tightness Classification (23 theorems, 14+ domains)

| System | Domain | Tightness | Evidence |
|--------|--------|-----------|----------|
| **Bilemma** | **ML explanation** | **Collapsed** | **structural (Lean)** |
| **Quantum linearity** | **Quantum info** | **Collapsed** | **model** |
| **GL(n) representation** | **Langlands** | **Collapsed** | **structural (Lean)** |
| Arrow | Social choice | Full | structural (Lean) |
| Gibbard-Satterthwaite | Social choice | Full | structural (Lean) |
| Bell (CHSH) | Quantum physics | Full | structural (Lean) |
| Kochen-Specker | Quantum physics | Full | structural (Lean) |
| Gödel | Mathematical logic | Full | structural (Lean) |
| ML fairness | ML prediction | Full | model |
| Bias-variance | Statistics | Full | model |
| Sen's paradox | Social choice | Full | model |
| CAP theorem | Distributed systems | Full | model |
| FLP | Distributed systems | Full | model |
| Mundell-Fleming | Economics | Full | model |
| Perpetual motion | Thermodynamics | Full | model |
| Second law | Thermodynamics | Full | model |
| Münchhausen | Epistemology | Full | model |
| Competitive exclusion | Ecology | Full | model |
| Penrose-Hawking | General relativity | Full | model |
| Eastin-Knill | Quantum computing | p12-blocked | model |
| Shannon secrecy | Cryptography | p23-blocked | model |
| Navier-Stokes (3D) | Math physics | Smooth-blocked | numerical (54 runs) |
| Navier-Stokes (2D) | Math physics | Full (control) | numerical |

---

## 5. Paper Files

| Paper | File | Venue | Status |
|-------|------|-------|--------|
| Nature (flagship) | `paper/nature_article.tex` | Nature | Ready |
| Monograph | `paper/universal_impossibility_monograph.tex` | arXiv | Ready |
| Supplement | `paper/supplementary_information.tex` | Nature SI | Ready |
| JMLR | `paper/universal_impossibility_jmlr.tex` | JMLR | Ready |
| NeurIPS | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 | Ready |
| Capacity audit | `paper/explanation_capacity_audit.tex` | — | Companion |

---

## 6. Honest Negatives (8 falsified predictions)

1. Phase transition location (predicted r*=1, observed 0.01-0.12)
2. Uncertainty bound (predicted α+σ+δ≤2, observed max 2.86)
3. Molecular evolution from character theory (partial R²=0.0)
4. Spectral gap convergence rate (14-100× too fast)
5. Flip correlations from irreducible decomposition (within ≈ between)
6. Regularization increases instability (opposite of SAM prediction)
7. Rashomon topology does not predict bimodality (1/6 datasets)
8. DASH tie rate vs Rashomon fraction (directional 4/6, weak quantitatively)

---

## 7. Axiom Architecture

2 bundled axioms in `Defs.lean`:
- `gbdtWorld : GBDTWorld` — Model type, numTrees, measure infrastructure
- `gbdtAxioms (fs) : GBDTAxiomsBundle gbdtWorld fs` — attribution, splitCount, firstMover + 6 behavioral properties

Core impossibility uses ZERO of these. Reduction history: 25 → 14 → 2.

---

## 8. Build and Verification

```bash
lake build              # Full Lean build (~5 min, 2954 jobs)
make verify             # Build + count consistency check

# Manual verification:
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'  # → 519
grep -c "^axiom " UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'           # → 2
ls UniversalImpossibility/*.lean | wc -l                                                     # → 102
```
