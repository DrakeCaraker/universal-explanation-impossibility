# Complete Findings Map: The Attribution Impossibility

> **Synced with:** `main_definitive.tex` (monograph, 64 pages)
> **Last audit:** 2026-04-04
> **Census:** 54 Lean files, 305 theorems, 16 axioms, 0 sorry, 89 definitions, 42 experiment scripts, 32 JSON result files, 12 figures
> **Monograph:** 64 pages, 23 sections, 78 subsections, 19 theorems, 14 propositions, 3 lemmas, 16 definitions, 11 corollaries, 11 tables, 30 citations
> **Total distinct findings:** 109

When updating the monograph, update this map. When adding experiments or Lean theorems, add entries here.

---

## TIER 1: Core Results

| # | Finding | Type | Theory | Empirical | Lean | Novelty | Strength |
|---|---------|------|--------|-----------|------|---------|----------|
| A1 | **Attribution Impossibility** — no ranking can be faithful + stable + complete under Rashomon | THEOREM | 4-line proof from Rashomon property | Validated 11 datasets, 3 implementations, NNs | `attribution_impossibility` (zero axioms) | **HIGH** — first stability impossibility in XAI | **VERY STRONG** — zero axioms, machine-verified |
| D1 | **Design Space Theorem** — exactly two families, DASH Pareto-optimal | THEOREM | 4-step proof (forced, achievable, exhaustive, infeasible) | DASH benchmark confirms two families | `design_space_theorem`, `family_a_or_family_b` | **HIGH** — complete characterization | **STRONG** — Steps 1,2,4 in Lean; Step 3 on paper |
| E1 | **Symmetric Bayes Dichotomy** — general two-family theorem for symmetric decision problems | THEOREM | Proof via G-invariance | 3 instances (attribution, model selection, causal discovery) | `symmetric_bayes_dichotomy` | **HIGH** — reusable proof technique | **STRONG** — 3 structurally distinct instances |
| C1 | **DASH Consensus Equity** — balanced ensembles produce equal attributions for symmetric features | COROLLARY | Derived from proportionality + split-count + balance | DASH convergence on synthetic + Breast Cancer | `consensus_equity` | **MEDIUM** — constructive resolution | **STRONG** — addresses the impossibility |

---

## TIER 2: Architecture-Discriminating Bounds

| # | Finding | Type | Theory↔Empirical Link | Lean | Novelty | Strength |
|---|---------|------|----------------------|------|---------|----------|
| B2 | **GBDT ratio 1/(1-ρ²)** diverges as ρ→1 | THEOREM | Corrected to 1/(1-αρ²) with α≈2/π; R²=0.89 fit | `ratio_tendsto_atTop` | **HIGH** | **STRONG** |
| B6 | **Lasso ratio = ∞** at any ρ>0 | THEOREM | No empirical validation (limitation noted) | `lasso_impossibility` | **MEDIUM** | **MODERATE** |
| B7 | **NN impossibility** conditional on dominance | THEOREM | 87% of NN pairs unstable (H8) | `nn_impossibility` | **MEDIUM** | **MODERATE** |
| B8 | **RF ratio 1+O(1/√T) → 1** (convergent) | DERIVATION | No empirical validation | None (informal) | **MEDIUM** | **WEAK** |
| B10 | **Exact flip rate = 1/2** for m=2, 2/m² for general m | PROPOSITION | 48% empirical on Breast Cancer matches theory | `binary_group_flip_rate` | **HIGH** | **VERY STRONG** |
| B3 | **α = 2/π** from binary quantization theory | PROPOSITION | Fitted α=0.60 vs theory 0.637; gap explained | — | **MEDIUM** | **MODERATE** |
| B1 | **Split Gap Lemma** ρ²T/(2-ρ²) ≥ ρ²T/2 | LEMMA | SymPy-verified | `split_gap_exact` | **LOW** | **STRONG** |
| B4 | **Depth dependence** — depth 3 is most stable | EMPIRICAL | Table of split ratios across depths and ρ | — | **MEDIUM** | **MODERATE** |
| B5 | **Alpha error analysis** — two sources explain 0.037 gap | DERIVATION | First-stump variance capture | — | **LOW** | **MODERATE** |
| B9 | **Spearman stability bound** ρ_S ≤ 1 - m³/P³ | DERIVATION | — | `spearmanCorr_bound` (weaker derived) | **LOW** | **MODERATE** |
| B11 | **Proportionality validation** CV≈0.35 stumps, ≈0.66 depth 6 | EMPIRICAL | 50 models on Breast Cancer | — | **LOW** | **MODERATE** |

---

## TIER 3: Resolution & Optimality

| # | Finding | Theory↔Empirical Link | Strength |
|---|---------|----------------------|----------|
| C2 | **DASH Pareto Optimality** — achieves Cramér-Rao bound | Lean components + benchmark: 3-4% vs 16-19% | **STRONG** |
| C6 | **DASH benchmark** — 4-5× reduction vs alternatives | Bootstrap/Subsampled = straw men; CI SHAP trades completeness | **STRONG** |
| C7 | **Ensemble size formula** M_min = ⌈2.71σ²/Δ²⌉ | Breast Cancer worst pair needs M≈120 | **STRONG** |
| C3 | **DASH = MVUE** by Rao-Blackwell | Standard theory | **STRONG** |
| C4 | **Median ARE = 2/π** — less efficient | Standard asymptotics | **MODERATE** |
| C5 | **Trimmed mean interpolates** mean↔median | Standard asymptotics | **MODERATE** |
| C8 | **Progressive DASH** — expected 11.75 models (2.1× savings) | 37-dataset survey | **MODERATE** |
| C9 | **Information loss** — discards log₂(m!) unreliable bits | Theoretical + JSON validation | **MODERATE** |
| C10 | **Robustness** — mean optimal for clean; trimmed for contaminated | Contamination sweep: <5% at 80% | **MODERATE** |

---

## TIER 4: Extensions

| # | Finding | Links To | Novelty | Strength |
|---|---------|----------|---------|----------|
| F1 | **Conditional SHAP impossibility** when β_j=β_k | Both flip ~50% at Δβ=0 (H32) | **HIGH** | **STRONG** |
| F3 | **Fairness audit = coin flip** | 43.2% unstable adverse actions (H20) | **HIGH** | **STRONG** |
| F5 | **FIM impossibility** — Rashomon from Fisher information | ε₀ = 9λ₋³/K₃²; Gaussian closed-form | **MEDIUM** | **MODERATE** |
| F8 | **Query complexity Ω(σ²/Δ²)** | Z-test near-optimal; M=120 for worst BC pair | **MEDIUM** | **MODERATE** |
| E3 | **Model selection impossibility** | SBD instance; CV = analogue of DASH | **MEDIUM** | **MODERATE** |
| E5 | **Causal discovery impossibility** | Variable-size symmetry group | **MEDIUM** | **MODERATE** |
| F10 | **Causal ID barrier** Ω(1/(1-ρ)²) | Same singularity; hard interventions resolve | **LOW** | **MODERATE** |
| F11 | **Local ≥ Global instability** | Law of total variance | **LOW** | **MODERATE** |
| F12-13 | **SHAP efficiency amplification** m/(m-1) | Explains PI (91%) > SHAP (41%) | **LOW** | **MODERATE** |
| F2 | **Conditional escape** at Δβ≈0.20 | Binary transition; no intermediate regime | **MEDIUM** | **STRONG** |
| F4 | **Intersectional compounding** (1/2)^K | K=3: 12.5% correct | **MEDIUM** | **MODERATE** |
| F6 | **Gaussian FIM specialization** | Closed-form ellipsoid | **LOW** | **MODERATE** |
| F7 | **NTK extension** (conjectural) | Research direction only | **LOW** | **WEAK** |
| F14 | **α-faithfulness bound** α ≤ 1/2 | Coin flip is optimal | **LOW** | **MODERATE** |
| F15 | **Negative Spearman** E[ρ_S] = -1/(m-1) for stable rankings | Counterintuitive | **LOW** | **MODERATE** |
| A3 | **Iterative optimizer → Rashomon** | GBDT, Lasso, NN all qualify | **MEDIUM** | **STRONG** |
| A4 | **Rashomon from symmetry** (permutation closure) | Covers all standard model classes | **MEDIUM** | **STRONG** |
| A5 | **Non-degeneracy** Pr[φ_j ≠ φ_k] = 1 | Measure-zero argument | **LOW** | **MODERATE** |
| A6 | **Rashomon inevitability** | Impossibility is inescapable for standard ML | **HIGH** | **STRONG** |

---

## TIER 5: Diagnostics

| # | Finding | Empirical Validation | Strength |
|---|---------|---------------------|----------|
| G1 | **F1 diagnostic** Z-test, flip = Φ(-Z/√M) | r = -0.89 on BC; |r| > 0.8 on 9/10 datasets | **VERY STRONG** |
| G3-4 | **single-model screen** single-model, 94% precision | 94-100% clean; 48-67% high-dim | **STRONG** |
| G7 | **SNR calibration** Φ(-SNR) on 1,325 pairs | All SNR > 1.96 have flip < 5% | **STRONG** |
| G8 | **Expected Kendall τ formula** | Predicted 26.8 vs empirical 36.6 (-27%) | **MODERATE** |
| G2 | **F1 restricted-range robustness** | r=-0.78 for Z<5; structural baseline r=-0.56 | **MODERATE** |
| G5-6 | **Exchangeability conditions** for F5 | T_eff ≈ 74 for typical settings | **MODERATE** |

---

## TIER 6: Empirical Breadth (34 experiments)

### Core Validations (directly test theory)
| # | Experiment | Key Number | Script |
|---|-----------|------------|--------|
| H1 | Synthetic — ratio | R²=0.89 for 1/(1-αρ²) | `validate_ratio.py` |
| H2 | Synthetic — instability | 48% at ρ=0.9 (theory: 50%) | `generate_figures.py` |
| H3 | Synthetic — DASH convergence | <1% at M=25 | `generate_figures.py` |
| H4 | Breast Cancer — instability | 48% flip (worst perim/area, |ρ|=0.98) | `real_world_validation.py` |
| H5 | Breast Cancer — DASH convergence | M=50: 0% flips | `real_world_validation.py` |
| H33 | Mechanism isolation | Deterministic=0%, stochastic=22.8% | `subsample_check.py` |
| H31 | Monte Carlo validation | Φ(-SNR) validated to <1.3% error | `monte_carlo_flip_rate.py` |

### Breadth Validations (demonstrate generality)
| # | Experiment | Key Number | Script |
|---|-----------|------------|--------|
| H6 | Cross-implementation | 160-183 unstable pairs; max=0.500 in all 3 | `cross_implementation_validation.py` |
| H7 | Permutation importance | 91% unstable (vs SHAP 41%); cross-method r=0.46 | `permutation_importance_validation.py` |
| H8 | Neural networks | 87% unstable; max=0.500; F1 r=-0.871 | `nn_shap_validation.py` |
| H9 | KernelSHAP noise | Model instability dominates 8:1 | `kernelshap_noise_control.py` |
| H10 | High-dimensional P=500 | F1 |r|>0.78; 148s wall time | `high_dimensional_validation.py` |
| H11 | Prevalence (77 datasets) | 68% unstable [56%,77%]; P≥20: 93% [81%,98%] | `prevalence_survey_openml.py` |
| H14 | F1 generality (11 datasets) | |r|>0.8 on 9/10; F5 94% precision | `comprehensive_validation.py` |
| H25 | SAGE/Boruta | SAGE 92%, Boruta 94% unstable | `sage_comparison.py` |

### Domain-Specific Validations
| # | Experiment | Key Number | Script |
|---|-----------|------------|--------|
| H12 | Healthcare prevalence | 6/9 (67%) exhibit instability | `healthcare_prevalence.py` |
| H15 | LLM attention (DistilBERT) | Fine-tuning: 14.5% unstable; perturbation: 88% | `llm_attention_instability.py` |
| H16 | Replication study | 4 published rankings are seed artifacts | — |
| H17 | German Credit | F5 flags job↔telephone; DASH: 0% | `financial_case_study.py` |
| H18 | Taiwan Credit Card | 29 pairs; worst retains 16% at M=25 | `financial_case_study.py` |
| H19 | Synthetic Credit | income↔DTI flips 8%; income↔score 0% | `financial_case_study.py` |
| H20 | Regulatory adverse action | 43.2% unstable top-1; DASH: 100% stable | `regulatory_case_study.py` |
| H21 | Lending Club (specificity control) | Zero unstable pairs (Z>36) | `lending_club_case_study.py` |
| H26 | NLP bag-of-words | 60% top-1 unstable; 91% top-3 | `nlp_token_instability.py` |
| H27 | Time-series features | 27% within-series unstable | `timeseries_instability.py` |

### Robustness Validations (close loopholes)
| # | Experiment | Key Number | Script |
|---|-----------|------------|--------|
| H13 | Hyperparameter robustness | 50-75% across configs; 4/8 always unstable | `prevalence_robustness.py` |
| H22 | Class imbalance | 1:5+ → 100% pairs unstable | `class_imbalance_instability.py` |
| H23 | Missing data | MCAR 10% → 20/20 unstable | `missing_data_instability.py` |
| H24 | Longitudinal drift | Spearman 1.0→0.18 over 31 rounds | `longitudinal_retraining.py` |
| H28 | Adversarial grid | All 108 configs hit 0.500 at ρ=0.9 | `adversarial_max_instability.py` |
| H29 | Hyperparameter sensitivity | Min 38.7%; never 0% | `hyperparameter_sensitivity.py` |
| H30 | DASH contamination | <5% degradation at 80% contamination | `dash_breakdown_point.py` |
| H32 | Conditional SHAP sweep | Escape at Δβ≈0.20; no intermediate regime | `conditional_shap_threshold.py` |
| H34 | Causal structure | Symmetric=50%, asymmetric=0% | `conditional_shap_causal.py` |

---

## TIER 7: Connections & Negative Results

| # | Finding | What It Shows |
|---|---------|---------------|
| J1 | Arrow's theorem parallel | Structural analogy but no common generalization (documented failure) |
| J3 | Fairness impossibility parallel | Same structure for different domain |
| J4 | Compressed sensing connection | Coherence = collinearity; both resolve by relaxing uniqueness |
| J5 | Topological analysis (**negative**) | No topological content; purely combinatorial |
| J6 | Classical multicollinearity distinction | Rankings, not estimation variance |
| J7 | Underspecification connection | Design Space = attribution stress testing |
| J8 | Loss landscape geometry | FIM ridges = Rashomon set |
| J9 | Invariant decision theory | SBD = reduction template from Lehmann & Romano |
| J2 | Sen's liberalism paradox | Closer structural analogue than Arrow |

---

## TIER 8: Lean Formalization

| # | Finding | What It Shows |
|---|---------|---------------|
| I1 | 305 theorems, 16 axioms, 0 sorry, 54 files | Scale of verification |
| I2 | Axiom stratification: 0→4→9→16 levels | Core result robust to axiom weakening |
| I3 | Explicit consistency model (Fin 4) | Axiom system not vacuously true |
| I4 | attribution_sum_symmetric derived (35 lines) | Reduced axiom count |
| I5 | 3 inconsistencies caught by Lean | Formalization as bug-finding |
| I6 | Top-5 deepest proofs (35, 29, 27, 26, 23 lines) | Verification depth |
| I7 | SymPy cross-verification | Three-layer: SymPy + Lean + empirical |

---

## TIER 9: Practical/Regulatory

| # | Finding | What It Shows |
|---|---------|---------------|
| K1 | EU AI Act Art. 13(3)(b)(ii) compliance | Attribution instability = "known circumstance" |
| K2 | ECOA adverse action instability (43.2%) | Model governance concern |
| K3 | Practitioner workflow (Screen→Z-test→DASH) | Actionable 4-step protocol |
| K4 | Progressive DASH (11.75 models expected) | Cost-effective deployment |
| K5 | Instability disclosure template | Ready-to-use regulatory language |
| K6 | Group-level reporting format | Report groups, not individual rankings |
| K7 | Depth 3 guidance | Lowest instability configuration |
| K8 | Three named reusable techniques | Rashomon reduction, FIM bridge, SBD |

---

## Maintenance Protocol

When updating the monograph (`main_definitive.tex`):
1. Add new findings to this map with the next available ID (e.g., H35, F16)
2. Update the census numbers at the top
3. Run the verification block: `grep -c "^theorem\|^lemma"`, `grep -c "^axiom"`, `grep -rc "sorry"`, `ls | wc -l`
4. Rebuild all 4 papers: `make paper` or manual pdflatex+bibtex cycle
5. Verify the total findings count matches

When adding Lean theorems:
1. Classify as CITED, IMPLICIT, UNCITED, or INFRA
2. If UNCITED and substantive (>3 tactic lines), add to the cross-reference table in the monograph
3. Update I1 counts

When adding experiments:
1. Ensure JSON results exist and are non-empty
2. Add a subsection to the monograph's experimental section
3. Add to the experiment summary table
4. Add an entry to the appropriate tier in this map
