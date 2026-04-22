# The Limits of Explanation

**No explanation of an underspecified system can be simultaneously faithful, stable, and decisive.**

Train a cancer classifier on 10,935 gene-expression features and the algorithm names TSPAN8 — an invasion/metastasis gene — as the most important biomarker. Change only the random seed: now the top gene is CEACAM5, an immune-evasion marker with zero shared biological function. Both models predict patient labels with indistinguishable accuracy. This instability is not a bug — it is a mathematical certainty.

This repository contains the Lean 4 formalization, empirical experiments, and paper manuscripts proving and applying this impossibility theorem across four high-stakes domains.

## The Theorem

Whenever observationally equivalent configurations coexist (the **Rashomon property**), no explanation can simultaneously be:
- **Faithful** — reflect the system's actual internal structure
- **Stable** — consistent across equivalent configurations
- **Decisive** — commit to a specific answer

The core theorem `explanation_impossibility` is machine-checked in Lean 4 with **zero domain-specific axioms**. The proof is four lines (see Methods in the Nature paper).

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

## The Universal η Law

The character-theoretic formula η = 1 − dim(V^G)/dim(V) predicts instability from the symmetry group alone, with zero free parameters. Across 7 well-characterized domains: **R² = 0.957**, slope 0.91.

A Noether-type counting law: for P features in g correlation groups, exactly g(g−1)/2 ranking facts are stable. Within-group comparisons are coin flips (50.0% vs 0.2%, bimodal gap = 50pp, p = 2.7 × 10⁻¹³).

## Classification of 20 Impossibility Theorems

A classification of 20 impossibility theorems from 12 domains by **tightness type** — which property pairs survive when the full triple fails:

| Tightness | Count | Examples |
|-----------|-------|----------|
| **Full** (pick two works) | 16 | Arrow, Gödel, Bell, KS, CAP, FLP, Mundell-Fleming, thermodynamics, ... |
| **Collapsed** (pairs blocked) | 2 | Explanation bilemma, quantum linearity trilemma |
| **Intermediate** | 2 | Eastin-Knill (p12), Shannon secrecy (p23) |

The explanation bilemma is one of only two collapsed instances — structurally more severe than Arrow, Gödel, Bell, CAP, or fairness impossibilities. The other is the quantum linearity trilemma (no-cloning + measurement disturbance), which shares a structural parallel: linearity plays the role of stability.

## Recursive Resolution

Enrichment restores blocked property pairs but creates a new impossibility at the next level. Levels are independent (proved). The same recursive pattern appears in Gödel's incompleteness and Galois theory. A complete explanatory theory classifies which questions have formulation-independent answers, rather than answering all questions.

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

Three repositories, 1,115 theorems total, **0 sorry**:

| Repository | Files | Theorems | Axioms | Content |
|------------|-------|----------|--------|---------|
| [universal-explanation-impossibility](.) | 101 | 501 | 25 | Core theorem, 9 ML instances, 14 cross-domain instances, η law, resolution |
| [dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean) | 58 | 357 | 6 | SHAP-specific: GBDT ratios, Lasso, neural nets, DASH equity |
| [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) | 20 | 257 | 10 | Bilemma, tightness classification, enrichment stack, physics application |

The core impossibility theorem uses **zero axioms** — only the Rashomon property as a hypothesis.

```bash
lake build          # build Lean (requires v4.30.0-rc1 + Mathlib, ~5 min)
make verify         # full build + consistency check
make counts         # verify theorem/axiom/sorry counts
```

## Reproducing Experiments

```bash
pip install -r paper/scripts/requirements.txt
pip install torch transformers
make validate       # run key experiments (~5 min)
```

Knockout experiments (90+ scripts, 80+ result JSONs):
```bash
cd knockout-experiments
python3 gene_expression_replication.py    # Instance 1
python3 mi_v2_comprehensive.py            # Instance 2
python3 brain_imaging_definitive.py       # Instance 4
python3 noether_sensitivity.py            # η law validation
```

## Paper Versions

| Paper | File | Venue | Status |
|-------|------|-------|--------|
| **The Limits of Explanation** | `paper/nature_article.tex` | Nature | Ready for submission |
| Universal Impossibility Monograph | `paper/universal_impossibility_monograph.tex` | arXiv | Ready (definitive reference) |
| The Attribution Impossibility | In [dash-shap](https://github.com/DrakeCaraker/dash-shap) repo | NeurIPS 2026 | Abstract May 4 |
| Ostrowski Impossibility (FoP) | In [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) repo | Foundations of Physics | Submission-ready (21/21 Accept) |

**Title:** "The Limits of Explanation"
**Authors:** Drake Caraker, Bryan Arnold, David Rhoads

## Repository Structure

```
UniversalImpossibility/           # Lean 4 source (100 files)
  ExplanationSystem.lean          # Core theorem (zero axioms)
  MaximalIncompatibility.lean     # Bilemma + tightness
  UniversalResolution.lean        # G-invariant resolution
  AttributionInstance.lean        # SHAP/IG/LIME instance
  CausalInstance.lean             # Markov equivalence instance
  MechInterpInstance.lean         # Mechanistic interpretability
  ...                             # 9 ML + 14 cross-domain instances

paper/
  nature_article.tex              # Nature submission (~2650 words)
  universal_impossibility_monograph.tex  # arXiv monograph (~4400 lines)
  supplementary_information.tex   # Supplementary Information
  references.bib                  # References
  figures/                        # Publication-quality figures
  scripts/                        # Figure generation + validation

knockout-experiments/             # Empirical validation
  RESULTS_SYNTHESIS.md            # 3 confirmed, 2 falsified, 1 negative
  PRE_REGISTRATION.md             # Pre-registered predictions
  90+ experiment scripts          # Reproducible validation
  80+ result JSONs                # Machine-readable results

docs/                             # Documentation + handoffs
```

## Key Results Summary

| Result | Value | Source |
|--------|-------|--------|
| η law R² (7 domains) | 0.957 | `results_universal_eta.json` |
| MI G-invariant resolution | ρ: 0.518 → 0.929 | `results_mi_v2_final_validation.json` |
| Noether bimodal gap | 50 pp (p = 2.7 × 10⁻¹³) | `results_noether_counting.json` |
| NARPS convergence (M₉₅) | 16 [10, 22] | `results_brain_imaging_bulletproof.json` |
| Gene alternation (TSPAN8) | 92% of seeds | `results_gene_expression_replication.json` |
| Tightness classification | 20 impossibilities, 12 domains | Lean-verified |
| Lean theorems (total) | 1,115 across 3 repos | 0 sorry |
| Falsified predictions | 5 of 8 pre-registered | Honest reporting |

## Citation

```bibtex
@article{caraker2026limits,
  author  = {Caraker, Drake and Arnold, Bryan and Rhoads, David},
  title   = {The Limits of Explanation},
  journal = {Nature},
  year    = {2026},
  note    = {Under review. Lean 4 formalization: 1,115 theorems, 0 sorry}
}
```

## License

Apache-2.0
