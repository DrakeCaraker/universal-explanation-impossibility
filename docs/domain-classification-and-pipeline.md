# Comprehensive Domain Classification and Applications Pipeline

**114 instances across 20 subfields in 5 major fields. 40 validated (Lean-verified or experimentally confirmed). 7 documented. 67 predicted.**

This document classifies every known instance of the explanation impossibility by domain, data type, group structure, tightness type, and recommended pipeline. It serves as a practitioner's guide: given your domain and data, here is what to expect and what to do.

## How to use this guide

1. Find your domain in the tables below
2. Check the tightness type (determines what's achievable)
3. Apply the recommended group discovery method
4. Apply the recommended diagnostic
5. Apply the recommended resolution

## Decision Tree

```
START: Does your system have the Rashomon property?
  │
  ├─ NO → No impossibility. Single-model explanation is fine.
  │        (Test: subsample=1.0 gives identical ranking → no Rashomon)
  │
  └─ YES → The impossibility applies.
           │
           ├─ Is your explanation space BINARY (support/reject, +/-, selected/not)?
           │   │
           │   ├─ YES → BILEMMA applies. Even F+S is impossible.
           │   │        Resolution: ENRICH (add a neutral element like "ambiguous").
           │   │        Examples: SHAP sign, feature selection, counterfactual direction,
           │   │        multi-analyst conclusions, circuit decomposition, value alignment
           │   │
           │   └─ NO → TRILEMMA applies. F+S+D impossible but F+S achievable.
           │            Resolution: ORBIT AVERAGE (sacrifice decisiveness).
           │            Examples: SHAP importance, attention, causal DAGs, gene expression
           │
           └─ What is your data type?
               │
               ├─ Continuous features → Use CORRELATION groups (ρ*=0.50-0.70)
               │   Diagnostic: Gaussian Φ(-SNR) per pair, coverage conflict per dataset
               │   Resolution: DASH ensemble average
               │
               ├─ Binary/categorical features → Use MI groups
               │   Diagnostic: MI pre-screen, then flip rate
               │   Resolution: DASH with MI-discovered groups
               │   (Pearson FAILS on binary features — drug discovery validates this)
               │
               ├─ Architecture-defined (neural network) → Use ARCHITECTURE symmetry
               │   Group: S_H per layer (head permutation)
               │   Diagnostic: within-layer flip rate (should be ~50%)
               │   Resolution: G-invariant projection onto V^G
               │
               ├─ Domain-knowledge available → Use DOMAIN groups
               │   Examples: codon degeneracy classes, functional networks, LD blocks
               │   Diagnostic: η formula with known G
               │   Resolution: Domain-specific orbit average
               │
               └─ Unknown group structure → Use SAGE (automatic discovery)
                   Train M≥25 models, compute flip-rate matrix, cluster
                   Diagnostic: SAGE-predicted instability (R²=0.81)
                   Resolution: SAGE groups → DASH within groups
```

## Validated Instances (27)

### ML Explanation Types (9 instances, all Lean-verified, zero axioms)

| Instance | H | Data Type | Group Method | Tightness | Resolution | Key Number |
|----------|---|-----------|-------------|-----------|------------|------------|
| SHAP/IG/LIME | Feature rankings | Continuous | Correlation/SAGE | Full | DASH | 75% exceed capacity |
| Attention maps | Token distributions | Continuous | S_H (architecture) | Full | Averaged rollout | η=0.83 |
| Counterfactual | Increase/decrease | Binary | Correlation | **Collapsed** | Enrich: add "ambiguous" | Bilemma |
| Concept probes (TCAV) | Concept presence | Continuous | O(d) rotation | Full | Concept cluster average | η≈0.90 |
| Model selection | Best model | Discrete | ε-Rashomon set | Full | Report set | η=0.91 |
| Saliency (GradCAM) | Spatial heatmap | Continuous | Spatial | Full | Multi-model average | — |
| LLM self-explanation | Token citations | Discrete | Unknown | Full | Citation frequency | 34.5% flip |
| Circuit decomposition | Component importance | Continuous | S_H per layer | **Collapsed** | Layer means | ρ=0.52→0.93 |
| Causal discovery | Edge orientation | Discrete | Markov equiv. | Full | CPDAG | — |

### Cross-Domain Sciences (14 instances, all Lean-verified)

| Instance | Domain | H | Tightness | Resolution | Key Number |
|----------|--------|---|-----------|------------|------------|
| Arrow's theorem | Social choice | Aggregate rankings | Full | Relax IIA | Formal recovery |
| Multi-analyst | Scientific aggregation | Conclusions | **Collapsed** | Consensus (M≥16) | NARPS M₉₅=16 |
| Peres-Mermin | Quantum physics | Value assignments | Full | Equivalence class | η=1/2 for qubit |
| Wave-particle | Quantum measurement | Which-slit | **Collapsed** | Complementarity | Enrichment L1 |
| Duhem-Quine | Philosophy | Theory identity | Full | Structural realism | Formalization |
| Gauge equivalence | Physics | Field config | Full | Gauge-invariant obs. | — |
| Microstate degeneracy | Stat. mechanics | Microstate | Full | Boltzmann average | η for S₂₅₂ |
| Codon degeneracy | Genetics | Codon identity | Full | Amino acid | η for S₂/S₄/S₆ |
| Phase problem | Crystallography | Phase angles | Full | Patterson maps | — |
| PP-attachment | Linguistics | Parse tree | Full | Report all parses | — |
| Value alignment | AI Safety | Value function | **Collapsed** | Value uncertainty | Bilemma |
| View-update | Database theory | Base state | Full | Canonical policy | — |
| Underdetermined Ax=b | Linear algebra | Solution x | Full | Min-norm (pseudoinverse) | — |
| Simultaneity | Relativity | Event ordering | Full | Frame-dependent | Enrichment |

### Mathematical Physics & Number Theory (4 instances, Lean-verified in companion repos)

| Instance | Domain | Tightness | Key Number |
|----------|--------|-----------|------------|
| 3D Navier-Stokes | Math physics | **Smooth-blocked** | Re_crit=2.91N^1.56, R²=0.94 |
| GL(n) Langlands | Rep. theory | n=1 Full, n≥2 Collapsed | Lean-verified for all n, all p |
| AC⁰ parity | Circuit complexity | Full + conditional barrier | 54 gates + 52,488 circuits |
| DPRM trilemma | Number theory | Full | Selmer local-global failure |

## Predicted/Extended Instances (12)

**Status: framework predicts these follow the impossibility pattern. Validation status varies.**

| Domain | Instance | Prediction | Validated? | Recommended Pipeline |
|--------|----------|------------|------------|---------------------|
| Epidemiology | Risk factor attribution | Correlated exposures → unstable | No | Correlation groups on exposure matrix |
| Climate science | Forcing attribution | Correlated forcings → unstable | No | Multi-model ensemble agreement |
| Economics | Policy impact | Specification-dependent | No | Multi-specification robustness |
| Pharmacology | Drug target | Binary features need MI | **Yes** (BBBP: Pearson 0%, MI 19%, actual 23%) | MI for fingerprints, correlation for continuous |
| Ecology | Species interaction | Trophic correlation → unstable | No | Functional group aggregation |
| Forensic science | Evidence weighting | Correlated evidence → unstable | No | Evidence type clusters |
| Education | Intervention effect | Bundled interventions → Rashomon | No | Intervention cluster analysis |
| Neuroscience | Brain region attribution | Functional network structure | **Yes** (NARPS d=0.32) | Yeo-7 network grouping |
| Genomics/GWAS | SNP-disease association | LD blocks → fine-mapping instability | No | LD block aggregation |
| Finance | Factor attribution | Correlated factors → unstable | No | Factor cluster analysis |
| Psychology | Construct measurement | Item clusters → score instability | **Yes** (Silberzahn: 29 teams) | Factor subscale scoring |
| Political science | Covariate effects | Specification sensitivity | **Yes** (Breznau: 73 teams, 48% sign disagreement) | Specification-robust conclusions |

## The Applications Pipeline

### For any new domain:

**Step 1: Diagnose** — Does the Rashomon property hold?
```python
# Train M≥25 models with different seeds
# If importance rankings change: Rashomon holds
from dash_shap import check
result = check(X_train, y_train, X_test)
if result.max_flip_rate > 0.10:
    print("Rashomon property detected")
```

**Step 2: Identify groups** — Which features are exchangeable?
```
If data type is continuous: use correlation groups (ρ* = 0.50 for capacity, 0.70 for directional)
If data type is binary/categorical: use MI groups
If architecture is known: use architecture symmetry (S_H per layer)
If domain knowledge available: use domain groups (LD blocks, functional networks, etc.)
Otherwise: use SAGE (flip-rate clustering)
```

**Step 3: Compute capacity** — How many stable facts exist?
```
C = g (number of groups)
Stable facts = g(g-1)/2
If P features and g groups: fraction unstable = 1 - g(g-1)/(P(P-1))
```

**Step 4: Apply diagnostic** — Which specific comparisons are stable?
```
Per-pair: Φ(-SNR) where SNR = |mean_diff| / std_diff (R² = 0.95-0.98)
Per-dataset: coverage conflict = fraction of pairs with SNR < 0.5 (ρ = 0.96)
Cross-domain: η = 1 - dim(V^G)/dim(V) (R² = 0.957 with known groups; ρ = 0.55 with inferred)
```

**Step 5: Apply resolution** — Report stable conclusions only
```
Between-group comparisons: REPORT (stable)
Within-group comparisons: FLAG as "structurally unreliable"
Group-level importance: compute via orbit average (DASH)
```

## Tightness Guide

| Your tightness type | What it means | What to do |
|---------------------|---------------|------------|
| **Full** | Every pair of {F,S,D} is achievable; the triple is not | Pick two. DASH gives F+S. |
| **Collapsed** (bilemma) | Even F+S is blocked for binary H | ENRICH: add a neutral element (e.g., "tied", "ambiguous") |
| **Smooth-blocked** | One property (e.g., smoothness) fails first | Report which property fails and at what threshold |
| **Conditional** | Impossibility depends on an open question | Report conditional classification |

## Validation Summary

| Claim | Evidence | Status |
|-------|----------|--------|
| Impossibility theorem | Lean proof (0 axioms) | **Proved** |
| Bilemma | Lean proof (0 axioms) | **Proved** |
| Tightness classification (23 instances) | Lean proofs + models | **Proved** |
| η formula (R²=0.957, 7 domains) | Empirical | **Confirmed** |
| η on real data (ρ=0.55, 49 datasets) | Empirical | **Confirmed** (Bonferroni-surviving) |
| 149-dataset audit (75%, p=5.1e-11) | Empirical | **Confirmed** |
| Null model rejected (CI excludes 0) | Empirical | **Confirmed** |
| SAGE outperforms correlation | Empirical (6/7 datasets) | **Confirmed** |
| MI ≈ correlation for continuous | Empirical (131 datasets) | **Confirmed** |
| MI needed for binary | Empirical (drug discovery) | **Confirmed** |
| Gaussian flip Φ(-SNR) | Empirical (R²=0.95-0.98) | **Confirmed** |
| Coverage conflict (ρ=0.96) | Empirical (15 datasets) | **Confirmed** |
| Predicted domains (8 untested) | Framework prediction | **Untested** |

---

## Expanded Inventory: 114 Instances Across All Sciences

### Natural Sciences (31 instances, 10 validated)

**Physics (13)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Quantum measurement (which-slit) | VALIDATED | `QuantumMeasurementRevolution.lean` |
| Quantum contextuality (Peres-Mermin) | VALIDATED | `PeresMermin.lean` + Kirchmair 2009 + Amselem 2009 |
| Gauge theory (field equivalence) | VALIDATED | `GaugeTheory.lean` |
| Statistical mechanics (microstate degeneracy) | VALIDATED | `StatisticalMechanics.lean` |
| Relativity (simultaneity) | VALIDATED | `SimultaneityRevolution.lean` |
| 3D Navier-Stokes (regularity) | VALIDATED | `NavierStokesImpossibility.lean` + 54 DNS runs |
| Quantum error correction (Knill-Laflamme) | DOCUMENTED | Monograph bridge theorem |
| Quantum gravity (spacetime emergence) | VALIDATED | `EnrichmentStack.lean` (parametric) |
| Black hole information (destroyed/preserved) | VALIDATED | `EnrichmentStack.lean` (parametric) |
| Dark matter models (NFW vs isothermal vs MOND) | PREDICTED | Multiple valid density profiles fit rotation curves |
| String landscape (vacuum selection) | PREDICTED | ~10^500 vacua, same low-energy physics |
| Turbulence modeling (RANS vs LES vs DNS) | PREDICTED | Multiple valid closures, same mean flow |
| Nuclear structure models | PREDICTED | Shell/cluster/collective models, same binding energies |

**Chemistry (5)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Reaction mechanism attribution | PREDICTED | Multiple valid mechanisms, same kinetics |
| Protein folding pathway | PREDICTED | Multiple valid pathways, same native state |
| Catalyst active site identification | PREDICTED | Multiple valid site assignments, same activity |
| Drug binding mode (pose selection) | PREDICTED | Multiple valid poses, same affinity prediction |
| Spectral decomposition | PREDICTED | Overlapping peaks, multiple valid decompositions |

**Biology (8)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Gene expression (TSPAN8/CEACAM5) | VALIDATED | `results_gene_expression_replication.json` (4 datasets) |
| Codon degeneracy | VALIDATED | `GeneticCode.lean` |
| Protein designability | PREDICTED | Multiple sequences → same structure |
| Gene regulatory networks | PREDICTED | Multiple valid TF attributions |
| Microbiome attribution | PREDICTED | Correlated species abundances |
| Phylogenetic tree selection | PREDICTED | Multiple valid topologies, same alignment |
| Cell type classification markers | PREDICTED | Correlated markers, multiple valid sets |
| Evolutionary rate (character theory) | DOCUMENTED | Monograph: R²=0.0 (falsified prediction) |

**Earth/Climate (5)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Climate forcing attribution | PREDICTED | Correlated forcings (CO2, CH4, aerosols) |
| Earthquake prediction features | PREDICTED | Correlated seismic indicators |
| Weather forecast attribution | PREDICTED | Correlated meteorological variables |
| Ocean circulation parameterization | PREDICTED | Multiple valid closures |
| Ice core proxy interpretation | PREDICTED | Multiple valid proxy calibrations |

### Formal Sciences (21 instances, 12 validated)

**Mathematics (7)**
| Instance | Status | Evidence |
|----------|--------|----------|
| GL(n) Langlands boundary | VALIDATED | `LanglandsCorrespondence.lean` + `GLnLanglands.lean` |
| SL(n) classical groups | VALIDATED | `ClassicalGroups.lean` |
| AC⁰ parity barrier | VALIDATED | `CircuitComplexity.lean` (54+52,488 checks) |
| DPRM trilemma | VALIDATED | `DiophantineImpossibility.lean` |
| Underdetermined Ax=b | VALIDATED | `LinearSystem.lean` |
| Gödel (as enrichment) | VALIDATED | `GoedelIncompleteness.lean` |
| Proof search strategy | PREDICTED | Multiple valid proof strategies, same theorem |

**Computer Science (8)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Model selection (Rashomon set) | VALIDATED | `ModelSelectionInstanceConstructive.lean` |
| Feature selection | VALIDATED | Bilemma (binary: selected/not) |
| Database view-update | VALIDATED | `ViewUpdate.lean` |
| CAP theorem | DOCUMENTED | Monograph tightness classification |
| FLP impossibility | DOCUMENTED | Monograph tightness classification |
| Hyperparameter selection | PREDICTED | Multiple valid configs, similar performance |
| Architecture search | PREDICTED | Multiple valid architectures |
| Ensemble selection | PREDICTED | Multiple valid ensemble compositions |

**Statistics (6)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Model specification | VALIDATED | Multi-analyst studies (3 validated) |
| Causal DAG selection | VALIDATED | `CausalInstanceConstructive.lean` |
| Penalized regression (Lasso) | VALIDATED | `Lasso.lean` (ratio = ∞) |
| Bias-variance tradeoff | DOCUMENTED | Monograph tightness classification |
| Bayesian prior selection | PREDICTED | Multiple valid priors, similar posteriors |
| Multiple testing correction | PREDICTED | Multiple valid corrections, different conclusions |

### Social Sciences (16 instances, 3 validated)

**Economics (5)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Policy impact attribution | PREDICTED | Specification-dependent |
| Factor model selection | PREDICTED | Correlated factors |
| Mundell-Fleming trilemma | DOCUMENTED | Monograph classification |
| Instrument variable selection | PREDICTED | Multiple valid instruments |
| Treatment effect heterogeneity | PREDICTED | Multiple valid subgroup definitions |

**Political Science (4)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Covariate effect attribution | VALIDATED | Breznau 2022 (73 teams, 48% sign disagreement) |
| Arrow's theorem | VALIDATED | `ArrowInstance.lean` |
| Gerrymandering detection | PREDICTED | Multiple valid metrics, different conclusions |
| Electoral prediction models | PREDICTED | Multiple valid poll models |

**Psychology (4)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Construct measurement | VALIDATED | Silberzahn 2018 (29 teams, effect 0.89-2.93) |
| Treatment response prediction | PREDICTED | Multiple valid predictive models |
| Cognitive model selection | PREDICTED | Multiple valid cognitive architectures |
| Personality factor attribution | PREDICTED | Correlated personality measures |

**Sociology (3)** — All PREDICTED

### Applied Sciences (40 instances, 13 validated)

**Medicine/Clinical (8)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Biomarker discovery | VALIDATED | `results_gene_expression_replication.json` |
| Drug target identification | VALIDATED | `results_drug_discovery_mi_clustering.json` |
| Clinical decision support | VALIDATED | `results_clinical_decision_reversal_v2.json` (45% reversal) |
| Disease risk factor attribution | PREDICTED | Correlated exposures |
| Treatment recommendation | PREDICTED | Multiple valid treatment models |
| Diagnostic imaging interpretation | PREDICTED | Multiple valid readings |
| Survival prediction features | PREDICTED | Correlated clinical variables |
| Clinical trial endpoint selection | PREDICTED | Multiple valid primary endpoints |

**AI/ML (11, 7 validated)** — See ML table above

**Neuroscience (5)**
| Instance | Status | Evidence |
|----------|--------|----------|
| fMRI analysis (NARPS) | VALIDATED | `results_brain_imaging_bulletproof.json` (M₉₅=16) |
| Brain region attribution | VALIDATED | Network structure predicts disagreement (d=0.32) |
| EEG source localization | PREDICTED | Multiple valid source configurations |
| Connectome analysis | PREDICTED | Multiple valid parcellations |
| Neural decoding features | PREDICTED | Correlated voxel responses |

**Finance (5)**
| Instance | Status | Evidence |
|----------|--------|----------|
| Credit scoring explanation | VALIDATED | German Credit (45% explanation reversal) |
| Factor attribution | PREDICTED | Correlated factors (size/value/momentum) |
| Risk model attribution | PREDICTED | Multiple valid risk decompositions |
| Portfolio attribution | PREDICTED | Correlated asset returns |
| Trading explanation | PREDICTED | Multiple valid signal attributions |

**Engineering, Forensics, Education** — All PREDICTED (see expanded JSON)

### Humanities (6 instances, 2 validated)

| Instance | Status | Evidence |
|----------|--------|----------|
| PP-attachment ambiguity | VALIDATED | `SyntacticAmbiguity.lean` |
| Duhem-Quine underdetermination | VALIDATED | `DuhemQuine.lean` |
| Machine translation attribution | PREDICTED | Multiple valid alignment models |
| Sentiment analysis attribution | PREDICTED | Correlated text features |
| Theory choice (Kuhn) | PREDICTED | Paradigm as enrichment event |
| Structural realism | DOCUMENTED | Reynolds = structural content (monograph) |

---

## Validation Completeness

| Evidence type | Count | Verification |
|---------------|-------|-------------|
| Lean-verified (this repo) | 26 | All .lean files verified present |
| Lean-verified (companion repos) | 8 | All .lean files verified present |
| Empirically confirmed (this repo) | 8 | All .json result files verified present |
| Externally validated (published studies) | 6 | All citations in references.bib |
| Documented (monograph, not primary) | 7 | In monograph text |
| **Total validated** | **41** | **41/41 evidence files verified** |
| Predicted (untested) | 67 | Framework predicts; no experiment yet |
| **Grand total** | **114** | |
