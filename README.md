# The Limits of Explanation

**No explanation of an underspecified system can be simultaneously faithful, stable, and decisive.**

Train a cancer classifier on 10,935 gene-expression features and the algorithm names TSPAN8 — an invasion/metastasis gene — as the most important biomarker. Change only the random seed: now the top gene is CEACAM5, an immune-evasion marker with zero shared biological function. Both models predict patient labels with near-identical accuracy. This instability is not a bug — it is a mathematical certainty.

This repository contains the Lean 4 formalization, empirical experiments, and paper manuscripts proving and applying this impossibility theorem. The theorem applies to any explanation of any underspecified system — not just ML, but any science where correlated variables or equivalent configurations create multiple valid explanations.

## The Papers

### Main Article (`paper/nature_article.tex`)

The flagship paper. Proves the impossibility from first principles, demonstrates it in four empirical domains, identifies the unique optimal resolution, and classifies 20 impossibility theorems from 12 domains by structural type.

**Contents:**
- The impossibility theorem (4-line proof, zero axioms)
- The bilemma (faithful + stable alone impossible for binary explanations)
- Tightness classification (which property pairs survive)
- Four empirical instances: genomics, mechanistic interpretability, causal inference, neuroimaging
- The universal η law (R² = 0.957, zero free parameters)
- Classification of 20 impossibility theorems — the bilemma is one of only two with collapsed tightness
- Recursive resolution (enrichment creates new impossibility; Gödel/Galois parallels)
- Connection to Heisenberg's uncertainty principle (same structural type, proved)

### Supplementary Information (`paper/supplementary_information.tex`)

Full details for all experiments and domain instances referenced in the Nature paper.

**Contents:**
- Eight domain instances with first-principles Rashomon derivations (linear algebra, codon degeneracy, gauge theory, statistical mechanics, linguistics, crystallography, database views, causal discovery)
- Lean formalization details (axiom stratification, proof status)
- Three falsified extensions (phase transition, tradeoff bound, molecular evolution)
- SAGE algorithm (stability-aware grouped explanation)
- CCA spectrum and continuous symmetry analysis
- Noether sensitivity analysis across correlation range ρ = 0.50–0.99
- Multi-model validation (XGBoost, RandomForest, Ridge × 3 datasets)
- Enrichment tradeoff details (Pareto frontier, DASH at the knee)
- Gene expression pathway divergence (4 datasets, two instability modes, GO enrichment)
- MI v2 comprehensive results (per-component, Noether counting, controls, η reconciliation)
- Neuroimaging multi-analyst reanalysis (activation control, convergence, aggregation comparison)
- Classification of impossibility theorems (evidence tiers, caveats)
- Approximate Rashomon extension (ε-stability)

### Monograph (`paper/universal_impossibility_monograph.tex`)

The definitive technical reference (~4,400 lines). Contains everything in the main article and supplement plus:

- Complete abstract framework with all definitions and proof details
- Full tightness analysis with constructive witnesses for all property pairs
- Derived instances across eight sciences with explicit Rashomon constructions
- Nine ML explanation instances with proof sketches and empirical validation
- The universal resolution framework (Reynolds operator, G-invariant projection, Pareto optimality)
- Ubiquity arguments (generic underspecification, neural network dimensional argument)
- Complete Lean formalization documentation (axiom stratification, proof status transparency)
- Group-theoretic classification of explainability (η law, Noether counting, hierarchy of difficulty)
- Uncertainty from Symmetry theorem (representation-theoretic connection to quantum mechanics)
- The Galois analogy (enrichment stack as tower of field extensions)
- Regulatory implications (EU AI Act, SR 11-7) and liability analysis
- Scope and applicability beyond ML (epidemiology, climate science, economics, ecology, pharmacology, forensics, social science, psychology)
- Scientific revolutions as enrichment events (quantum mechanics, relativity, DASH)
- Recursive impossibility and the enrichment stack (Gödel parallel, Tarski parallel, unification consequence)
- Empirical predictions from the mathematical structure (4 testable predictions)
- Cross-domain transfer experiments (ensemble causal discovery, stability transitions, Rashomon boundary)
- Invariance counting, SAGE algorithm, CCA spectrum
- Universal η law with boundary conditions and three falsified extensions
- Gaussian flip rate formula with multi-model validation
- Gene expression pathway divergence, MI circuit-level Rashomon, model-class universality, neuroimaging reanalysis, cross-domain multi-analyst studies
- Bridge theorems (sufficient statistics, tradeoff bound, EM as orbit averaging, Bayesian interpretation, quantum error correction)
- Classification of 20 impossibility theorems by tightness with evidence tiers
- Arrow's theorem structural comparison
- Complete Lean theorem statements for all core results
- Detailed experiment methodology and proof sketches for all nine instances

## The Theorem

Whenever observationally equivalent configurations coexist (the **Rashomon property**), no explanation can simultaneously be:
- **Faithful** — reflect the system's actual internal structure
- **Stable** — consistent across equivalent configurations
- **Decisive** — commit to a specific answer

The core theorem `explanation_impossibility` is machine-checked in Lean 4 with **zero domain-specific axioms**.

A strengthened form — the **bilemma** — shows that for binary explanations (positive/negative, selected/not), even faithful + stable alone is impossible. The **tightness classification** determines exactly which property pairs are achievable based on the structure of the explanation space.

## Four Empirical Instances

### Instance 1: Genomics
50 XGBoost classifiers with TreeSHAP on colon-vs-kidney cancer data produce an alternating top gene: TSPAN8 (#1 in 92% of seeds) vs CEACAM5 (6%), with zero shared Gene Ontology biological process terms. Replicates across 4 datasets with two modes of instability — between pathways and within pathways. The DASH ensemble resolution reports both genes without privileging either.

### Instance 2: Mechanistic Interpretability
10 transformers trained on modular addition all grok to 100% accuracy but their circuit-importance rankings agree at only ρ = 0.518. Fourier frequency overlap: 2.2% Jaccard. Projecting onto the architecturally-derived invariant subspace (S₄ × S₄) lifts agreement from ρ = 0.518 to ρ = 0.929. Safety cases should be built on equivalence classes of circuits, not individual circuit diagrams.

### Instance 3: Causal Inference
A chain A→B→C and fork A←B→C are Markov-equivalent. No algorithm observing conditional independence can orient the A-B edge. The CPDAG (reporting "undirected" where ambiguous) is the neutral-element resolution — proved Pareto-optimal.

### Instance 4: Neuroimaging
Botvinik-Nezer et al. (Nature 2020) gave the same fMRI data to 70 teams who reached different conclusions. Orbit averaging across 48 accessible teams is near-optimal (all aggregation methods within 3%). 16 independent analyses [95% CI: 10-22] suffice for 95% stability.

## Key Theoretical Results

### The Universal η Law
The character-theoretic formula η = 1 − dim(V^G)/dim(V) predicts instability from the symmetry group alone, with zero free parameters. Across 7 well-characterized domains: **R² = 0.957**, slope 0.91. A Noether-type counting law follows: within-group comparisons are coin flips (50.0% vs 0.2%, bimodal gap = 50pp, p = 2.7 × 10⁻¹³).

### Classification of 20 Impossibility Theorems
A classification by **tightness type** — which property pairs survive when the full triple fails:

| Tightness | Count | Examples |
|-----------|-------|----------|
| **Full** (pick two works) | 16 | Arrow, Gödel, Bell, KS, CAP, FLP, Mundell-Fleming, thermodynamics, ... |
| **Collapsed** (pairs blocked) | 2 | Explanation bilemma, quantum linearity trilemma |
| **Intermediate** | 2 | Eastin-Knill (p12), Shannon secrecy (p23) |

The explanation bilemma is one of only two collapsed instances — structurally more severe than Arrow, Gödel, Bell, CAP, or fairness impossibilities.

### Uncertainty from Symmetry
The η law and quantum uncertainty arise from the same representation-theoretic decomposition. The Reynolds operator (explanation) and the quantum twirl (QM) are instances of the same algebraic structure. The approximate bilemma (unfaith(θ₁) + unfaith(θ₂) ≥ Δ − δ) is a formal uncertainty relation analogous to Heisenberg's ΔxΔp ≥ ℏ/2. Proved in Lean (`UncertaintyFromSymmetry.lean`, 10 theorems, 0 sorry).

### Recursive Resolution
Enrichment restores blocked property pairs but creates a new impossibility at the next level. Levels are independent (proved). The enrichment stack mirrors the algebraic structure of a tower of field extensions in Galois theory.

## Eight Communities, One Resolution

Over the past century, eight scientific communities independently converged on the same mathematical resolution — projecting onto the symmetry-invariant subspace:

| Community | Problem | Resolution |
|-----------|---------|------------|
| ML/XAI | Feature ranking instability | DASH ensemble averaging |
| Crystallography | Phase retrieval ambiguity | Patterson maps |
| Causal inference | Markov-equivalent DAGs | CPDAGs |
| Gauge theory | Gauge equivalence | Gauge-invariant observables |
| Neuroimaging | Multi-analyst disagreement | Overlap maps |
| Statistical mechanics | Microstate degeneracy | Boltzmann averaging |
| Genetics | Codon degeneracy | Amino acid abstraction |
| Database theory | View-update ambiguity | Canonical views |

The theorem proves this convergence is not coincidence: the orbit-averaged projection is the **unique Pareto-optimal strategy** for any system with the Rashomon property.

## Lean Formalization

Three repositories, 1,292 theorems total, **0 sorry**:

| Repository | Files | Theorems | Axioms | Content |
|------------|-------|----------|--------|---------|
| [universal-explanation-impossibility](.) | 101 | 501 | 25 | Core theorem, 9 ML instances, 14 cross-domain instances, η law, resolution, uncertainty from symmetry |
| [dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean) | 58 | 357 | 6 | SHAP-specific: GBDT ratios, Lasso, neural nets, DASH equity |
| [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) | 31 | 434 | 10 | Bilemma, tightness classification, enrichment stack, physics application |

The core impossibility theorem uses **zero axioms** — only the Rashomon property as a hypothesis.

```bash
lake build          # build Lean (requires v4.30.0-rc1 + Mathlib, ~5 min)
make verify         # full build + consistency check
make counts         # verify theorem/axiom/sorry counts
```

## Reproducing Experiments

```bash
pip install -r paper/scripts/requirements.txt
pip install torch transformers datasets
make validate       # run key experiments (~5 min)
```

Knockout experiments (90+ scripts, 80+ result JSONs):
```bash
cd knockout-experiments
python3 gene_expression_replication.py              # Instance 1: genomics
python3 mech_interp_definitive_v2.py                # Instance 2: MI (modular addition)
python3 comprehensive_circuit_stability.py          # Instance 2+: MI (TinyStories, multi-scale)
python3 brain_imaging_definitive.py                 # Instance 4: neuroimaging
python3 noether_sensitivity.py                      # η law validation
python3 uncertainty_from_symmetry.py                # Uncertainty theorem numerical proof
```

## Repository Structure

```
UniversalImpossibility/           # Lean 4 source (101 files)
  ExplanationSystem.lean          # Core theorem (zero axioms)
  MaximalIncompatibility.lean     # Bilemma + tightness
  UniversalResolution.lean        # G-invariant resolution
  UncertaintyFromSymmetry.lean    # Reynolds = twirl, Pythagorean decomposition
  AttributionInstance.lean        # SHAP/IG/LIME instance
  CausalInstance.lean             # Markov equivalence instance
  MechInterpInstance.lean         # Mechanistic interpretability
  SyntacticAmbiguity.lean         # Linguistic parse tree ambiguity
  ...                             # 9 ML + 14 cross-domain instances

paper/
  nature_article.tex              # Main article (~2700 words)
  supplementary_information.tex   # Supplement (13 sections)
  universal_impossibility_monograph.tex  # arXiv monograph (~4400 lines)
  references.bib                  # References
  figures/                        # Publication-quality figures
  scripts/                        # Figure generation, validation, numerical proofs

knockout-experiments/             # Empirical validation
  comprehensive_circuit_stability.py    # Multi-scale TinyStories experiment
  90+ experiment scripts          # Reproducible validation
  80+ result JSONs                # Machine-readable results

docs/                             # Documentation + handoffs
```

## Key Results Summary

| Result | Value | Source |
|--------|-------|--------|
| η law R² (7 domains) | 0.957 | `results_universal_eta.json` |
| MI G-invariant resolution | ρ: 0.518 → 0.929 | `results_mi_v2_final_validation.json` |
| TinyStories pilot (language) | ρ: 0.567 → 0.952 | `results_tinystories_circuit_stability_pilot.json` |
| Noether bimodal gap | 50 pp (p = 2.7 × 10⁻¹³) | `results_noether_counting.json` |
| NARPS convergence (M₉₅) | 16 [10, 22] | `results_brain_imaging_bulletproof.json` |
| Gene alternation (TSPAN8) | 92% of seeds | `results_gene_expression_replication.json` |
| Tightness classification | 20 impossibilities, 12 domains | Lean-verified |
| Lean theorems (total) | 1,292 across 3 repos | 0 sorry |
| Falsified predictions | 5 of 8 pre-registered | Honest reporting |

## Related Papers

| Paper | Location |
|-------|----------|
| **The Limits of Explanation** (main article + SI) | `paper/nature_article.tex`, `paper/supplementary_information.tex` |
| Universal Impossibility Monograph | `paper/universal_impossibility_monograph.tex` |
| The Attribution Impossibility | [dash-shap](https://github.com/DrakeCaraker/dash-shap) repo |
| Ostrowski Impossibility | [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) repo |

**Authors:** Drake Caraker, Bryan Arnold, David Rhoads

## Citation

```bibtex
@article{caraker2026limits,
  author  = {Caraker, Drake and Arnold, Bryan and Rhoads, David},
  title   = {The Limits of Explanation},
  year    = {2026},
  note    = {Lean 4 formalization: 1,292 theorems, 0 sorry}
}
```

## License

Apache-2.0
