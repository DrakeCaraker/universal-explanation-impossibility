# Universal Explanation Impossibility

**No explanation of an underspecified system can be simultaneously faithful, stable, and decisive.**

A Lean 4 formalization proving this universal impossibility theorem, instantiated across 9 ML explanation types and 14 cross-domain instances, with empirical validation and a constructive resolution via G-invariant aggregation.

## The Theorem

The Universal Explanation Impossibility is a meta-theorem: for any explanation system satisfying the Rashomon property (observationally equivalent configurations with incompatible explanations), no explanation can be simultaneously faithful (reflect the system's actual structure), stable (consistent across equivalent configurations), and decisive (commit to a single answer).

The core theorem `explanation_impossibility` is machine-checked in Lean 4 with **zero axiom dependencies** — only the Rashomon property is required as a hypothesis.

A strengthened form, the **bilemma**, shows that for maximally incompatible systems (where `incompatible = ≠`), even faithful + stable alone is impossible. The tightness classification meta-theorem characterizes exactly which property pairs are achievable based on the existence of neutral and committal elements.

## Instances

### ML Explanation Types (9)

| Instance | Explanation type | Instability metric | Empirical result |
|----------|-----------------|-------------------|-----------------|
| 1. Attribution (SHAP/IG/LIME) | Feature rankings | SHAP flip rate | 33% |
| 2. Attention maps | Token distributions | Argmax flip rate | 19.9% (retraining) |
| 3. Counterfactual explanations | Nearest contrastive | Direction flip rate | 23.5% |
| 4. Concept probes (TCAV) | Concept directions | 1−\|cos\| | 0.90 |
| 5. Causal discovery | DAG orientations | Markov equivalence | Derived |
| 6. Model selection | Best model | Best-model flip rate | 80% |
| 7. Saliency maps (GradCAM) | Pixel attributions | GradCAM flip rate | — |
| 8. LLM self-explanations | Token citations | Citation flip rate | — |
| 9. Mechanistic interpretability | Circuit structure | Circuit non-uniqueness | — |

### Cross-Domain Instances (14)

Arrow's theorem (social choice), Peres-Mermin (quantum contextuality), Duhem-Quine (theory underdetermination), gauge theory, statistical mechanics, genetic code (codon degeneracy), phase problem (crystallography), QM interpretation, syntactic ambiguity (linguistics), value alignment (AI safety), view-update (databases), linear systems, quantum measurement revolution, simultaneity revolution.

## Key Empirical Results

- **Gaussian flip formula**: Predicts pairwise ranking instability with OOS R² = 0.848 across 5 datasets, validated on XGBoost, Random Forest, and Ridge
- **Noether counting**: Symmetry-group structure predicts within-group flip rate ≈ 0.50 with 47pp bimodal gap, invariant across ρ = 0.50–0.99
- **Universal η law**: dim(V^G)/dim(V) predicts instability with R² = 0.957 for well-characterized groups
- **Clinical audit**: 57–62% of SHAP feature-pair comparisons are unreliable across clinical/financial datasets; SAGE algorithm reduces false discovery from 62% to < 5%
- **Enrichment validation**: Class merging reduces instability by 6–19pp across 4/6 datasets; semantic merging outperforms balance-matched random controls (+9.6pp on Fashion-MNIST)

## Lean Formalization

104 files, 463 theorems+lemmas, 72 axioms, 0 sorry.

The core theorem `explanation_impossibility` has **zero axiom dependencies**.

```bash
lake build          # requires Lean 4 v4.30.0-rc1 + Mathlib
make counts         # verify theorem/axiom/sorry counts
make verify         # full build + consistency check
```

## Reproducing Experiments
```bash
pip install -r paper/scripts/requirements.txt
pip install torch transformers
python paper/scripts/run_all_universal_experiments.py
```

Knockout experiments (Gaussian flip, Noether counting, enrichment, etc.):
```bash
cd knockout-experiments
python3 gaussian_flip_cv.py
python3 noether_sensitivity.py
python3 abstraction_enrichment_expanded.py
```

## Paper Versions

| Version | File | Target |
|---------|------|--------|
| Monograph (definitive) | `paper/universal_impossibility_monograph.tex` | arXiv |
| Nature article | `paper/nature_article.tex` | Nature |
| JMLR | `paper/universal_impossibility_jmlr.tex` | JMLR |
| NeurIPS | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 |

Companion: [Ostrowski Impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) — bilemma applied to spacetime geometry via Ostrowski's classification.

## Citation

```bibtex
@software{caraker2026universal,
  author       = {Caraker, Drake and Arnold, Bryan and Rhoads, David},
  title        = {The Universal Explanation Impossibility},
  year         = {2026},
  url          = {https://github.com/DrakeCaraker/universal-explanation-impossibility},
  note         = {Lean 4 formalization; 104 files, 463 theorems, 72 axioms, 0 sorry}
}
```

## License

Apache-2.0
