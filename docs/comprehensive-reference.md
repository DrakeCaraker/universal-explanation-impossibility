# Comprehensive Reference: The Limits of Explanation

This document provides a thorough explanation of every proof, experiment, result, and paper in this project. It is designed to be shared with other sessions so they can quickly understand what is available, what it means, and where to find it.

**Verified state (2026-04-30): 519 theorems, 2 axioms, 102 Lean files, 0 sorry. Full build: 2954 jobs, 0 errors.**

---

## 1. The Core Theorem

### What it proves

No explanation of an underspecified system can simultaneously be **faithful** (reflect the system's actual structure), **stable** (consistent across observationally equivalent configurations), and **decisive** (commit to a single answer) — whenever the **Rashomon property** holds (multiple valid configurations produce incompatible explanations).

### Why it matters

This is not a limitation of current methods. It is a mathematical theorem. No future algorithm can overcome it. The gap between what a system can *predict* and what can be *explained* about it is permanent.

### Where it is

- **Lean proof**: `ExplanationSystem.lean` → `explanation_impossibility` (4 lines, zero axioms)
- **Paper**: Nature article Theorem 1 (lines 178-182); Monograph §2
- The proof requires ONLY the Rashomon property as a hypothesis — no domain-specific axioms.

### The strengthened form (bilemma)

For binary explanation spaces (significant/not, positive/negative), even **faithful + stable alone** is impossible when every pair of distinct explanations is incompatible. This triggers for SHAP sign, feature selection, edge orientation.

- **Lean**: `MaximalIncompatibility.lean` → `bilemma`
- **Paper**: Nature lines 188-192

---

## 2. The Resolution

### What it is

The orbit-averaged projection onto the symmetry-invariant subspace V^G. For feature attributions: DASH (Diversified Aggregation for Stable Hypotheses) — average importances within each dependence group. For causal graphs: CPDAG. For gauge theory: gauge-invariant observables.

### Why it's optimal

The orbit average is the **unique Pareto-optimal** strategy: it achieves zero disagreement for within-group comparisons (ties), while ANY committed ranking disagrees with exactly half the model distribution.

- **Lean**: `ParetoOptimality.lean` → `dash_unique_pareto_optimal` (7 theorems, zero axioms beyond GBDT infrastructure)
- The proof does NOT use Cramér-Rao — it uses the measure-theoretic fact that DGP symmetry forces μ({φ_k > φ_j}) = 1/2 for within-group pairs.

### The design space dichotomy

Every explanation method falls into exactly two families:
- **Family A**: Faithful + decisive, but unstable (any single model's explanation)
- **Family B**: Stable, with ties where data is ambiguous (orbit average)
No third option exists.

- **Lean**: `UniversalDesignSpace.lean` → `universal_design_space_dichotomy`; `DesignSpaceFull.lean` → `design_space_exhaustiveness`

---

## 3. The Explanation Capacity

### C = dim(V^G)

The **explanation capacity** is the dimension of the fixed subspace under the symmetry group G. It equals the maximum number of independent, stable ranking facts any method can extract. For P features in g dependence groups: C = g, and the number of stable pairwise facts is g(g-1)/2.

### The explanatory information loss identity

For any explanation vector v, the Reynolds projection R decomposes it:

‖v − Rv‖² + ‖Rv‖² = ‖v‖²

The stable component Rv ∈ V^G has dimension C; the unstable component (v − Rv) ∈ (V^G)⊥ is irretrievably lost. These are orthogonal — zero cross-talk.

- **Lean**: `UncertaintyFromSymmetry.lean` → `uncertainty_from_symmetry`, `best_approximation`, `beyond_capacity_penalty` (16 theorems)

### The Explanation Stability Theorem (4 parts)

Mirrors Shannon's coding theorem:
1. **Convergence rate**: MSE = tr(RΣR)/M (source coding rate)
2. **Capacity bound**: Stable maps have image in V^G (channel capacity)
3. **Rate optimality**: Orbit average minimizes MSE among linear unbiased estimators (achievability)
4. **Over-explanation penalty**: ‖w‖ ≤ ‖u − w‖ for w ∈ (V^G)⊥ (strong converse — exact MSE, not asymptotic)

- **Lean**: `stable_in_fixed_subspace`, `beyond_capacity_penalty`, `beyond_capacity_optimal`
- **Paper**: Monograph §5.2 (The Explanation Stability Theorem)

### The capacity theorem (η law)

η = 1 − dim(V^G)/dim(V) predicts instability rate with R² = 0.957 across 7 well-characterised domains and zero free parameters.

- **Empirical**: `results_universal_eta.json` (16 instances; 7 well-characterised: R²=0.957; holdout 9: R²=0.24)
- **Paper**: Nature Extended Data Fig 1; Monograph §6 (The Explanation Capacity Theorem)

---

## 4. The MI Exact Boundary

### MI > 0 is necessary and sufficient

The Lean theorem `mi_is_exact_boundary` proves that mutual information I(X_j; X_k) > 0 — not correlation — is the necessary and sufficient condition for the attribution impossibility. Features with ρ = 0 but I > 0 (binary fingerprints, nonlinear dependencies like X_k = X_j²) are subject to the impossibility.

- **Lean**: `MutualInformation.lean` → `mi_is_exact_boundary`, `gaussian_mi_positive_of_corr_nonzero`, `impossibility_from_mi` (11 theorems)
- **Lean**: `MIQuantitativeBridge.lean` → `mi_quantitative_unfaithfulness`: MI > 0 → unfaithfulness ≥ Δ/2 (5 theorems)

### Empirical validation

- **Drug discovery (BBBP)**: Pearson predicts 0% instability on binary fingerprints. MI predicts 19%. Actual: 23%. (`results_drug_discovery_mi_clustering.json`)
- **131-dataset comparison**: MI-vs-correlation grouping across 131 PMLB datasets: ARI = 0.84 mean, identical groups 77%. MI ≈ correlation for continuous features. (`results_mi_reaudit.json`)

---

## 5. The 149-Dataset Capacity Audit

### Setup

149 datasets from 53 scientific domains. 50 XGBoost models per dataset (subsample=0.8, colsample_bytree=0.5). TreeSHAP importances. Correlation groups via hierarchical clustering at |Spearman ρ| > 0.70.

### Key results

| Metric | Value | Source |
|--------|-------|--------|
| Datasets exceeding capacity | 75% (111/149 at ρ*=0.70) | `results_audit_150_final.json` |
| Cross-dataset Wilcoxon | p = 5.09 × 10⁻¹¹ (110 testable) | Computed from audit data |
| Real-world only | p = 1.7 × 10⁻⁶ (59 datasets) | Computed from audit data |
| Family-level (cluster-robust) | p = 2.5 × 10⁻⁸ (72 families) | `results_audit_strengthening.json` |
| Block bootstrap CI | [0.036, 0.069], excludes zero | `results_audit_strengthening.json` |
| Confirmation-to-reversal | 27:1 at p<0.005 | Computed from audit data |
| Bonferroni | 22:0 at p<0.05/149 | Computed from audit data |
| Exceedance sweep | 83%/81%/75%/69%/50% at ρ*=0.50/0.60/0.70/0.80/0.90 | `results_audit_150_final.json` |

### Null model rejection

The alternative hypothesis — "within-group features flip more simply because they have smaller importance differences" — is rejected:

- **Synthetic**: Randomization test rejects at p=0.001 (4/4 settings with ρ≥0.80)
- **Real-world (14 datasets, 13,511 pairs)**: Quadratic OLS coef_within = 0.037, cluster-bootstrap CI [0.014, 0.045] excludes zero
- **Stratified**: Within > between in lowest |diff| quintile (p<10⁻⁴); attenuates at higher |diff|
- **SAGE vs correlation**: SAGE (flip-rate clustering) produces 5× larger within-between gaps (0.25 vs 0.05) — see `results_open_questions_final.json`

### The Gaussian flip formula

Φ(-SNR) predicts per-pair flip rate at R² = 0.946–0.980 across 6 datasets. The strongest per-pair predictor in the framework. See `results_open_questions_capstone.json`.

---

## 6. Four Domain Instances (Nature paper)

### Instance 1: Genomics

TSPAN8 (#1 gene in 92% of seeds) vs CEACAM5 (6%) on AP_Colon_Kidney. Zero shared GO Biological Process terms. Replicates on 3 additional datasets. Resolution: DASH reports both.

- **Data**: `results_gene_expression_replication.json`
- **Key numbers**: ρ=0.858, max_depth=4, 50 seeds

### Instance 2: Mechanistic Interpretability

10 modular-addition transformers: raw agreement ρ=0.518, Fourier Jaccard=2.2%. G-invariant projection (S₄×S₄) lifts to ρ=0.929. Replicates on TinyStories (ρ=0.565→0.972) and 6L/8H scale (ρ=0.540→0.982).

- **Data**: `results_mi_v2_final_validation.json`, `results_mech_interp_definitive_v2.json`
- **Adversarial audit**: `results_mi_audit.json` — non-uniqueness goes deeper than permutation symmetry (alignment Jaccard=0.079)

### Instance 3: Causal Inference

Chain A→B→C and fork A←B→C are Markov-equivalent. CPDAG is the neutral-element resolution. PC/GES already implement this — the framework proves their output is Pareto-optimal.

### Instance 4: Neuroimaging

48 teams, NARPS data. M₉₅=16 [10,22] for 95% stability. Network membership predicts disagreement (d=0.32 after activation control).

- **Data**: `results_brain_imaging_bulletproof.json`

---

## 7. Tightness Classification

23 impossibility theorems from 14+ domains classified by tightness type:

- **Collapsed** (3): Explanation bilemma, quantum linearity, GL(n) representation — "pick two" fails; enrichment required
- **Full** (17): Arrow, Gödel, Bell, fairness, CAP, etc. — every pair achievable
- **Intermediate** (2): Eastin-Knill (p12-blocked), Shannon secrecy (p23-blocked)
- **Numerical** (2): Navier-Stokes 3D (smooth-blocked), 2D (full, control)

The explanation bilemma is one of only 3 collapsed instances — structurally more severe than Arrow, Gödel, or Bell.

- **Lean**: All 23 instances have Lean formalization (structural or model evidence)
- **Paper**: Nature Extended Data Table 2

---

## 8. The Enrichment Stack

Enrichment (adding a neutral element) restores F+S at the cost of decisiveness, but creates new impossibility at the next level. The levels are proved independent.

Three physics levels formalized (now as parametric theorems, not axioms):
- Level 1: Quantum measurement (which-slit) → complementarity
- Level 2: Black hole information (destroyed/preserved) → complementarity
- Level 3: Spacetime emergence (fundamental/emergent) → description-dependence

- **Lean**: `EnrichmentStack.lean` (18 theorems), `RecursiveImpossibility.lean`
- Mirrors Gödel's incompleteness pattern (proved as common RecursiveImpossibility interface)

---

## 9. The SAGE Algorithm

### What it does

Automatically discovers which feature comparisons are stable vs unreliable, without requiring knowledge of the symmetry group G.

### Algorithm (6 steps)

1. Train M ≥ 25 models (bootstrap/seed variation)
2. Compute pairwise flip-rate matrix F
3. Cluster features by flip-rate similarity
4. Count groups g → capacity C = g
5. Report g(g-1)/2 between-group comparisons as stable
6. Flag within-group comparisons as unreliable

### Performance

- R² = 0.809 (8 datasets, LOO-CV R² = 0.689)
- Outperforms correlation baseline by 28× (R² = 0.915 vs 0.033)
- SAGE gap = 0.25 vs correlation gap = 0.05 (6/7 datasets)
- Reduces false discovery rate from 57% to <5%
- <1% computational overhead

- **Data**: `results_sage_audit.json`, `results_sage_baseline_comparison.json`
- **Paper**: Monograph §5.5; SI §SAGE Algorithm

---

## 10. Axiom Architecture

### Current state: 2 axioms

All GBDT infrastructure is bundled into two structures:

| Axiom | Structure | Contains |
|-------|-----------|----------|
| `gbdtWorld` | `GBDTWorld` | Model type, numTrees, measure infrastructure |
| `gbdtAxioms` | `GBDTAxiomsBundle` | attribution, splitCount, firstMover + 6 behavioral properties |

### Axiom stratification

- **Core impossibility** (`explanation_impossibility`): ZERO axioms
- **MI boundary** (`mi_is_exact_boundary`): ZERO axioms
- **Pareto optimality** (`dash_unique_pareto_optimal`): ZERO axioms (beyond GBDT infra)
- **Quantitative bounds** (ratio, flip rate, etc.): 2 axioms (gbdtWorld + gbdtAxioms)

### Reduction history

25 → 14 (physics axioms → variables) → 2 (GBDT axioms → bundled structures)

---

## 11. Paper Strategy

| Paper | File | Venue | Status |
|-------|------|-------|--------|
| Nature (flagship) | `paper/nature_article.tex` | Nature | Ready for submission |
| Monograph (definitive) | `paper/universal_impossibility_monograph.tex` | arXiv | Ready |
| Supplement | `paper/supplementary_information.tex` | Nature SI | Ready |
| JMLR | `paper/universal_impossibility_jmlr.tex` | JMLR | Ready |
| NeurIPS | `paper/universal_impossibility_neurips.tex` | NeurIPS 2026 | Ready |
| Attribution (companion) | In dash-shap repo | NeurIPS 2026 | ~70% ready |
| Physics (companion) | ostrowski-impossibility repo | arXiv (hep-th) | Draft complete |

---

## 12. Key Experiment Files

| File | Content | Key Numbers |
|------|---------|-------------|
| `results_audit_150_final.json` | 149-dataset capacity audit | 75% exceedance, Wilcoxon p=5.1e-11 |
| `results_audit_strengthening.json` | Null model, bootstrap, cluster-robust | Family p=2.5e-8, CI [0.036,0.069] |
| `results_final_gaps.json` | Pooled OLS, stratified, alt clustering | 14 datasets, coef=0.037 |
| `results_open_questions_final.json` | SAGE comparison, extended sample | SAGE gap 0.25, CI [0.014,0.045] |
| `results_open_questions_capstone.json` | Gaussian flip R², conformal, regularization | R²=0.946-0.980, coverage 84-88% |
| `results_mi_reaudit.json` | MI vs correlation at scale | 131 datasets, ARI=0.84 |
| `results_gene_expression_replication.json` | Gene alternation | TSPAN8 92%, ρ=0.858 |
| `results_mi_v2_final_validation.json` | Mech interp | ρ=0.518→0.929 |
| `results_brain_imaging_bulletproof.json` | NARPS | M₉₅=16 [10,22] |
| `results_universal_eta.json` | η law across domains | R²=0.957 (7), R²=0.60 (16) |
| `results_drug_discovery_mi_clustering.json` | MI on binary features | Pearson 0%, MI 19%, actual 23% |

---

## 13. Honest Negatives

Five pre-registered predictions were falsified:
1. Phase transition location (predicted r*=1, observed 0.01-0.12)
2. Uncertainty bound (predicted α+σ+δ≤2, observed max 2.86)
3. Molecular evolution from character theory (partial R²=0.0)
4. Spectral gap convergence rate (14-100× too fast)
5. Flip correlations from irreducible decomposition (within ≈ between)

Additional negatives from this session:
6. Regularization increases instability (opposite of SAM prediction — but consistent with theory: regularization creates the Rashomon property)
7. Rashomon topology does not predict bimodality (1/6 datasets)
8. DASH tie rate vs Rashomon fraction (directional 4/6, weak quantitatively)

These define the framework's boundary: the impossibility and resolution work universally; the quantitative predictions work only with known groups.

---

## 14. Building and Verification

```bash
make lean          # Compile Lean (~5 min)
make paper         # Compile all paper versions
make verify        # Build + count consistency check
lake build         # Direct Lean build (2954 jobs)

# Verification block (run before committing paper changes):
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems:", s}'
grep -c "^axiom " UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
```
