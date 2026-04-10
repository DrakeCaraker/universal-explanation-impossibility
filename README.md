# Universal Explanation Impossibility

**No explanation of an underspecified system can be simultaneously faithful, stable, and decisive.**

A Lean 4 formalization proving this universal impossibility theorem, instantiated across six explanation types, with empirical validation and a constructive resolution via G-invariant aggregation.

## The Theorem

The Universal Explanation Impossibility is a meta-theorem: for any explanation system satisfying the Rashomon property (multiple observationally equivalent configurations exist), no explanation can be simultaneously faithful (reflect the system's actual structure), stable (consistent across equivalent configurations), and decisive (commit to a single answer). The core theorem `explanation_impossibility` is machine-checked in Lean 4 with zero axiom dependencies — only the Rashomon property is required as a hypothesis. Because the abstract `ExplanationSystem` framework subsumes all six explanation types below, the impossibility holds universally across the entire landscape of ML explanation methods.

## Six Instances

| Instance | Explanation type | Instability metric | Empirical result |
|----------|-----------------|-------------------|-----------------|
| 1. Attribution (SHAP/IG/LIME) | Feature rankings | SHAP flip rate | 33% |
| 2. Attention maps | Token distributions | Argmax flip rate | 60% |
| 3. Counterfactual explanations | Nearest contrastive | Direction flip rate | 23.5% |
| 4. Concept probes (TCAV) | Concept directions | 1-\|cos\| | 0.90 |
| 5. Causal discovery | DAG orientations | Classical (Verma & Pearl) | — |
| 6. Model selection | Best model | Best-model flip rate | 80% |

## Lean Formalization

75 files, 351 theorems+lemmas, 72 axioms, 0 sorry.

The core theorem `explanation_impossibility` has **zero axiom dependencies**.

### Building
```bash
lake build  # requires Lean 4 v4.30.0-rc1 + Mathlib
```

## Reproducing Experiments
```bash
pip install -r paper/scripts/requirements.txt
pip install torch transformers
python paper/scripts/run_all_universal_experiments.py
```

## Paper Versions

| Version | File | Target |
|---------|------|--------|
| Monograph (definitive) | `paper/universal_impossibility_monograph.tex` | arXiv |
| JMLR | `paper/universal_impossibility_jmlr.tex` | JMLR submission |
| NeurIPS | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 |

## Citation

```bibtex
@software{caraker2026universal,
  author       = {Caraker, Drake and Arnold, Bryan and Rhoads, David},
  title        = {The Universal Explanation Impossibility},
  year         = {2026},
  url          = {https://github.com/DrakeCaraker/universal-explanation-impossibility},
  note         = {Lean 4 formalization; 75 files, 351 theorems, 72 axioms, 0 sorry}
}
```

## License

Apache-2.0
