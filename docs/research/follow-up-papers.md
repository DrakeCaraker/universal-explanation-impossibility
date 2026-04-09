# Follow-Up Papers: From Strong Contribution to Foundational Result

> Analysis from the March 30, 2026 research session. Vetted across 6 rounds
> (axiom audit, foundational implications, plan, literature integration, final vet).
> All claims have confidence ratings from the vet process.

## Current State (NeurIPS 2026 Submission)

**Paper:** "The Attribution Impossibility: Faithful, Stable, and Complete Feature Rankings Cannot Coexist Under Collinearity"

**What it proves:**
- Core impossibility (zero axiom deps, pure logic from Rashomon property)
- Instantiated for GBDT (ratio 1/(1-ρ²) → ∞), Lasso (∞), NN (conditional), RF (bounded)
- DASH resolution (consensus equity for balanced ensembles)
- Spearman defined from scratch (qualitative bound fully derived)
- 13 Lean files, 12 axioms, 28 substantive theorems, 0 sorry

**What it doesn't (yet) prove:**
- Rashomon is inevitable for general model classes
- Full relaxation characterization
- The "trilemma → single-axis" reduction (partially in supplement)

---

## Paper 2: "Rashomon Inevitability Under Collinearity"

### Headline Result

**Theorem R3 (Rashomon Inevitability):** For any stochastic, symmetric
training algorithm on a permutation-closed model class with ρ > 0, the
Rashomon property holds. Therefore the Attribution Impossibility is
inescapable for any standard ML pipeline under collinearity.

### Theorem Decomposition

**R1 (Rashomon from Symmetry)** [HIGH confidence — provable]

*Statement:* DGP symmetry + model class permutation closure → for any
model f with φ_j(f) ≠ φ_k(f), the permuted model f' = f ∘ π has
L(f') = L(f) and opposite feature utilization.

*Proof:* Population loss is invariant under within-group permutation
(DGP symmetry). Attributions permute by construction.

*Note:* This is for POPULATION loss. The permuted model exists in the
model class but may not be reachable by a specific training run.

*Scope:* Already in NeurIPS supplement. Can be formalized in Lean.

**R2 (Attribution Non-Degeneracy)** [MEDIUM confidence — needs regularity]

*Statement:* For any stochastic training algorithm with continuous
dependence on randomness, trained on n samples with ρ > 0,
Pr[φ_j(f) ≠ φ_k(f)] = 1.

*Proof sketch:* Finite samples break exact DGP symmetry (empirical
correlation ≠ ρ). Under continuous dependence, the training algorithm
generically distinguishes features. The event φ_j = φ_k has measure
zero.

*Machinery:* Measure theory, transversality/genericity arguments.

*Assumptions:* Continuous dependence on data/randomness. Holds for SGD,
bootstrap sampling, random feature selection. May fail for exact
discrete algorithms.

**R3 (Full Inevitability)** [HIGH confidence — composition of R1+R2]

*Statement:* R1 + R2 → Rashomon property holds for the training
distribution. By algorithmic symmetry, Pr[φ_j > φ_k] = Pr[φ_k > φ_j] = 1/2.

*Significance:* Makes the impossibility inescapable. The only escapes:
ρ = 0 (no collinearity), deterministic model class (one model), or
asymmetric algorithm (built-in feature preference).

### Mathematical Machinery

- Group actions on feature spaces (Mathlib: `Equiv.Perm`, `MulAction`)
- Measure theory for R2 (Mathlib: `MeasureTheory`)
- Connection to Fisher information: near-singular FIM under collinearity
  → non-identifiability → multiple near-optimal parameter configurations

### Related Work to Cite

- Fisher, Rudin, Dominici (2019): MCR framework — our R1 implies MCR > 0
- Donoho-Elad (2003): mutual coherence → non-unique sparse recovery (compressed sensing analogue)
- Zhao-Yu (2006): irrepresentable condition failure → Lasso inconsistency
- D'Amour et al. (2022): underspecification (empirical; we provide the theory)
- Semenova-Rudin-Parr (2022): Rashomon ratio → simpler models exist

### Lean Formalization Plan

- R1: define `FeaturePermutation`, `PermutationClosed`, prove Rashomon.
  Feasible with Mathlib's `Equiv.Perm`.
- R2: requires `MeasureTheory`. Harder but achievable.
- R3: composition. Depends on R1+R2.

### Venue

ICML 2027 or AISTATS 2027. Can also be a NeurIPS 2026 workshop paper
if compressed to 4 pages.

---

## Paper 3: "The Attribution Design Space"

### Headline Result

**The Relaxation Characterization:** The Attribution Impossibility has
three relaxation paths, but two converge — the design space is a
single axis parameterized by ensemble size M.

### Theorem Inventory

**F1 (Unfaithfulness Lower Bound)** [HIGH confidence — trivial symmetry]

*Statement:* Any stable, complete ranking has expected unfaithfulness
≥ 1/2 per symmetric pair.

*Proof:* By DGP symmetry, Pr[φ_j > φ_k] = 1/2. Any deterministic
ranking is wrong for half the models.

*Already in NeurIPS supplement.*

**F2 (Optimal Unfaithful Ranking)** [HIGH confidence — Bayes-optimal]

*Statement:* The population-level ranking
R*(j,k) = 𝟙[E[φ_j] > E[φ_k]] minimizes expected unfaithfulness.
For symmetric features, E[φ_j] = E[φ_k], so R* assigns ties.

*Proof:* R* is the Bayes-optimal decision for "which feature is more
important?" under the symmetric model distribution.

*Note:* Holds for symmetric loss functions (L0, L2). Asymmetric losses
may give different optima.

**F3 (Path Convergence)** [HIGH confidence — novel structural insight]

*Statement:* The optimal "drop faithfulness" solution = population-level
attributions = ties for symmetric features = the "drop completeness"
solution. The relaxation paths converge.

*Consequence:* The trilemma collapses to a single tradeoff axis:
ensemble size M.
- M = 1: faithful + complete, unstable
- M → ∞: stable + between-group faithful, within-group ties

*This has NO analogue in Arrow's theorem* (where the three relaxation
paths give different solutions: Borda, majority rule, single-peaked).
The convergence is specific to the DGP symmetry structure.

*Already in NeurIPS supplement (as observation).*

**S1-S2 (Architecture Comparison)** [HIGH — compile existing results]

*Statement:* The equity-stability pair is fixed per architecture:

| Architecture | Equity violation | Stability violation | Mechanism |
|---|---|---|---|
| GBDT | 1/(1-ρ²) | 1 - m³/P³ | Sequential residuals |
| Lasso | ∞ | 1 - m³/P³ | Hard selection |
| Random forest | 1 + O(1/√T) | 1 - O(m³/(T·P³)) | Independent trees |
| DASH(M) | 1 | 1 - O(1/M) | Ensemble averaging |

*These are compiled from existing results + RF analysis.*

**S3 (Single-Point Pareto)** [HIGH within axioms, MEDIUM for real GBDT]

*Statement:* For sequential methods under our axioms, the equity-stability
pair is a fixed point (not a tradeoff curve). The ratio is determined by ρ
and the architecture, with no tuning knob. The only way to improve is to
change architecture class.

*Caveat:* Real GBDT with varying hyperparameters (learning rate, depth,
sub-sampling) creates a neighborhood, not a single point. Our axioms
capture the leading-order (worst-case) behavior.

### The Complete Design Space Map

```
                    Faithful
                       |
                       |  M = 1 (single model)
                       |  faithful + complete, unstable
                       |  equity violation = 1/(1-ρ²)
                       |
         M increases   |
         ─────────────>|
                       |
                       |  M → ∞ (population level)
                       |  stable + between-group faithful
                       |  within-group: ties (equity = 1)
                       |
                  Stable + Equitable
```

The design space is ONE-DIMENSIONAL: ensemble size M.
- Small M: more faithful to individual models, less stable
- Large M: more stable, lose within-group completeness
- Between-group faithfulness preserved at all M

### Key Insight: "Drop Faithfulness" Is Surgically Targeted

When you use population-level attributions:
- Between-group rankings: E[φ_j] ≠ E[φ_k] → FAITHFUL and STABLE
- Within-group rankings: E[φ_j] = E[φ_k] → ties (unfaithful for
  any specific model, but correctly indeterminate)

You don't lose ALL faithfulness — only for the pairs where faithfulness
was indeterminate anyway.

### What This Means for Practitioners

1. **If you need per-model explanations:** Accept instability. Report
   confidence intervals. Use RF over GBDT when collinearity is high.
2. **If you need stable explanations:** Use DASH with M ≥ 25. Accept
   that symmetric features will be tied.
3. **There is no option 3.** The design space is one-dimensional.

### Computational Cost

| Method | Training cost | Stability | Equity |
|---|---|---|---|
| Single model | 1× | Low | 1/(1-ρ²) |
| DASH(M) | M× | High | 1 |
| Population SHAP | M× (same as DASH) | High | 1 |

Paths A (drop completeness) and C (drop faithfulness) have the same
computational cost. Path B (single model) is the only "cheap" option.

### Venue

This is substantial enough for a standalone paper at NeurIPS/ICML.
Could also be combined with Paper 2 (Rashomon inevitability) for a
single comprehensive follow-up.

---

## Paper 4: Lean Formalization of Inevitability

### What to Formalize

- R1 (permutation closure → Rashomon): feasible, uses Mathlib's Equiv.Perm
- F1 (unfaithfulness bound): feasible, pure arithmetic
- F3 (path convergence): the observation is informal; formalizing would
  require defining "optimal ranking" which needs measure theory

### What NOT to Formalize (yet)

- R2 (non-degeneracy): requires measure theory + genericity arguments
- Full Pareto characterization: requires defining architecture families
- The M-parameterized spectrum: requires probability on Model

### Connection to Existing Code

The Lean codebase already has:
- `RashimonProperty` definition (Trilemma.lean)
- `IterativeOptimizer` abstraction (Iterative.lean)
- `spearmanCorr` definition (SpearmanDef.lean)

R1 would add a NEW proof path to `RashimonProperty` — complementary to
`IterativeOptimizer`, not replacing it:
- IterativeOptimizer: constructive (gives quantitative bounds)
- R1: existential (more general, no bounds)

---

## Limitations and Caveats (from Vet Rounds)

### What the impossibility does NOT cover

1. **Causal attributions:** Causal SHAP (Janzing et al., 2020) attributes
   importance to causal effects, not model reliance. Under collinearity,
   if the causal structure is known, causal attributions can break the
   symmetry. Our impossibility applies to MODEL-FAITHFUL attributions.

2. **Nonlinear dependence:** The framework uses linear correlation ρ.
   Features with nonlinear dependence but ρ = 0 (e.g., X_k = X_j²) don't
   trigger the impossibility. Generalizing to I(X_j; X_k) > 0 is open.

3. **Low collinearity:** At ρ = 0.1, the ratio is 1/(1-0.01) ≈ 1.01 —
   the impossibility is technically present but practically negligible.
   The result is most impactful at moderate-to-high ρ (≥ 0.5).

4. **Hyperparameter variation:** Our axioms capture leading-order behavior.
   Real GBDT with learning rate, depth, regularization has a neighborhood
   of possible ratios, not a single point. Sub-sampling reduces the
   first-mover advantage (approaching the RF regime).

### What we claim vs. what we've proved

| Claim | Status | Confidence |
|---|---|---|
| Core impossibility (faithfulness + stability + completeness) | Proved in Lean (zero axioms) | HIGH |
| GBDT ratio = 1/(1-ρ²) | Proved in Lean (from axioms) | HIGH |
| Rashomon inevitable for symmetric algorithms (R3) | Argued informally | HIGH (provable) |
| Paths A and C converge (F3) | Argued informally, in supplement | HIGH |
| M-parameterized spectrum | Stated, not formalized | HIGH (structure is clear) |
| Stability-equity Pareto is single point | Within axioms only | MEDIUM (real GBDT varies) |

---

## Priority Ordering

1. **Immediate (before NeurIPS May 4):** R1 + F1 + F3 in supplement — DONE
2. **Near-term:** Lean formalization of R1 (permutation → Rashomon)
3. **Medium-term:** Full characterization paper (F1-F3, S1-S3, M-spectrum)
4. **Long-term:** R3 (Rashomon inevitability), MI generalization, causal extension

---

## Connection to the 5-Paper Research Program

| Paper | Topic | Status |
|---|---|---|
| Paper 1 | DASH method + empirical validation | Published/submitted |
| Paper 2 | Classical proofs (impossibility.tex) | In dash-shap repo |
| **Paper 3** | **Lean formalization + NeurIPS paper** | **Submitting May 2026** |
| Paper 4 | Neural network attribution theory | Planned |
| Paper 5 | Applications (clinical ML, finance) | Planned |

The follow-up papers (Rashomon inevitability, design space characterization)
would be Papers 3b/3c — extensions of the impossibility program.

---

## Appendix: Reviewer Guidance on Foundational Path

### R1 (Skeptic): What would make you cite this?
1. Derive α from XGBoost splitting criterion (α(1)=2/π proved, not fitted)
2. Prove Rashomon inevitability via Fisher information rank deficiency
3. Demonstrate on regulatory-relevant datasets (FICO, MIMIC, COMPAS)

### R2 (Theorist): What would make this Arrow's theorem of XAI?
1. Prove Rashomon for ε-balls around Bayes optimum (near-optimality → permutation closure)
2. Complete characterization: Pareto frontier of (stability, equity) parameterized by M
3. Unifying meta-theorem subsuming Bilodeau, Huang, and our result

### R3 (Engineer): What would change industry practice?
1. pip-installable DASH library with auto-M selection and diagnostics
2. Case study where DASH prevents a real decision error
3. EU AI Act compliance template using DASH


---

## Library Status (assessed 2026-03-31)

### dash-shap installability
- **pyproject.toml EXISTS** — pip-installable via `pip install -e .`
- **Package name:** `dash-shap` v0.1.0
- **Core API:** `DASHPipeline` in `dash_shap/core/pipeline.py`
- **Existing diagnostics:** `bootstrap_stability_test` in `dash_shap/evaluation/`
- **MISSING:** F5 split-frequency diagnostic (`check_split_stability()`)
- **MISSING:** F1 attribution diagnostic (`check_attribution_stability()`)
- **Action needed:** Add F1/F5 diagnostics to the library, then publish to PyPI

### Lean variance bound feasibility
- **Mathlib HAS:** `IndepFun.variance_sum` (the key theorem)
- **MISSING:** `MeasureSpace` on our axiom `Model` type
- **Needs 4 new axioms:** measurableSpace, measure, attribution_measurable, models_indep
- **Estimated effort:** 1-2 days once axioms added
