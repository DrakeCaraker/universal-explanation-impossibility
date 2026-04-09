# JMLR Pre-Submission Audit — 2026-04-08

Comprehensive scientific audit of the monograph (paper/main_definitive.tex) across five levels: axiom justification, Lean-paper correspondence, theorem soundness, empirical methodology, and framing/actionability.

---

## Phase A: Axiom & Assumption Audit

### Executive Summary

The axiom system is well-stratified. The core impossibility (Theorem 1) depends on ZERO behavioral axioms — only the Rashomon property as a hypothesis. This means axiom failures cannot undermine the main result; they only affect downstream quantitative bounds and the DASH resolution. The weakest link is `proportionality_global` (global constant c across all models), which is stronger than needed for impossibility but required for DASH equity. The consistency construction in `Consistency.lean` is sound but covers only the 6 property axioms, not the 2 measure-theoretic or 2 query-complexity axioms (by design — those are infrastructure).

### A1. Property Axiom Audit

#### Axiom 1: `firstMover_surjective`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Strong** |
| **Empirical grounding** | DGP symmetry: when features in a group have identical marginal distributions, sub-sampling and tie-breaking randomness ensure each can serve as first-mover. Well-established in GBDT literature. |
| **Failure mode** | Fails when (a) features have different marginal distributions within a group (non-exchangeable), (b) the algorithm has a deterministic tie-breaking rule that always favors a fixed feature index, (c) one feature has systematically higher gain due to slight distributional differences. |
| **Weakness check** | This IS the weakest form. The qualitative impossibility (`impossibility_qualitative` in `Qualitative.lean`) needs exactly this + dominance. Cannot be weakened further while keeping the impossibility. |
| **Acceptability** | A GBDT researcher would accept this for exchangeable features under random subsampling. A statistician would note it requires exact exchangeability within groups, which is an idealization. In practice, approximate exchangeability suffices because the impossibility only needs the Rashomon property (existence of two models with reversed ordering), not exact surjectivity. |
| **Downstream impact** | Used by: `gbdt_rashomon`, `gbdt_impossibility`, `rashomon_from_bounded_proportionality`, all quantitative bounds. NOT used by core `attribution_impossibility`. |
| **Results affected if false** | GBDT-specific instantiation fails, but core impossibility still holds if the Rashomon property is established by other means (e.g., `rashomon_from_symmetry` or `rashomon_inevitability`). |

#### Axiom 2: `splitCount_firstMover`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Moderate** |
| **Empirical grounding** | Derived from the Gaussian conditioning argument. The formula n_{j1} = T/(2-rho^2) is the leading-order split count under three assumptions: (i) root splits capture dominant signal (valid for stumps/low-depth), (ii) sequential residual fitting, (iii) approximately independent feature competition across trees (valid for small learning rate). SymPy-verified algebra. |
| **Failure mode** | Fails when (a) tree depth > 1 (deeper trees distribute splits more evenly), (b) learning rate is large (violates independence assumption), (c) regularization constrains split allocation, (d) non-Gaussian features. These are leading-order approximations, not exact values. |
| **Weakness check** | The qualitative impossibility (`Qualitative.lean`) does NOT need this axiom at all. Only needs "first-mover dominates." The core impossibility uses none of these. |
| **Acceptability** | A GBDT researcher would accept as useful first-order approximation for stumps with small learning rate. Would push back for deep trees or large learning rates. |
| **Downstream impact** | Used by: `split_gap_exact`, `splitCount_ratio`, `attribution_ratio` (the 1/(1-rho^2) result). |
| **Results affected if false** | The exact ratio 1/(1-rho^2) and the exact split gap formula. The qualitative impossibility and the divergence direction (ratio grows with rho) are robust to perturbations. |

#### Axiom 3: `splitCount_nonFirstMover`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Moderate** (same as Axiom 2 — matched pair) |
| **Empirical grounding** | Same Gaussian conditioning argument. Residual variance after conditioning is (1-rho^2), giving split count (1-rho^2)T/(2-rho^2). |
| **Failure mode** | Same as Axiom 2. Additionally, assumes ALL non-first-mover features in the group get the SAME split count — requires equal footing after first-mover absorbs its share. In groups of size > 2, this is a further approximation. |
| **Weakness check** | For the impossibility, only "first-mover split count > non-first-mover" is needed. The exact formula is stronger than necessary. |

#### Axiom 4: `proportionality_global`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Weak to Moderate** — the MOST debatable axiom |
| **Empirical grounding** | Justified by the "uniform-contribution model" of Lundberg & Lee (2017): each split contributes equally to attribution. The GLOBAL constant (same c for ALL models) requires identical hyperparameters and homogeneous per-split contributions across training runs. |
| **Failure mode** | Fails when (a) different training runs produce different hyperparameters, (b) per-split contributions vary across models, (c) SHAP values computed with different backgrounds, (d) regularization differs across models. The per-model version (c varies by model) is strictly weaker and suffices for impossibility. |
| **Weakness check** | **CRITICAL FINDING**: `ProportionalityLocal.lean` proves `gbdt_impossibility_local` using only per-model proportionality. `Qualitative.lean` proves `impossibility_qualitative` needing NO proportionality at all. The global version is ONLY needed for DASH equity (`consensus_equity`). |
| **Acceptability** | A GBDT researcher would push back: different random seeds can yield models with different split depths and leaf values. A statistician would want this relaxed to bounded proportionality, which `ApproximateEquity.lean` addresses. |
| **Downstream impact** | Used by: `attribution_ratio`, `consensus_equity`, `attribution_sum_symmetric`. |
| **Results affected if false** | If global c fails but per-model c holds: impossibility is unaffected, but DASH equity becomes approximate. If proportionality fails entirely: qualitative impossibility still holds. |

**Paper transparency**: The paper states (line 309): "This axiom holds exactly under the uniform-contribution model...and approximately whenever per-split contributions are roughly homogeneous." However, it does NOT explicitly state that proportionality_global is stronger than needed for the core impossibility. **Recommendation**: Add a remark that per-model proportionality suffices for impossibility, and the global version is only needed for DASH equity.

#### Axiom 5: `splitCount_crossGroup_symmetric`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Moderate** |
| **Failure mode** | Feature interactions between groups (acknowledged at line 372). NOT needed for core impossibility, GBDT impossibility, or ratio bound. Only needed for DASH equity (full sum-symmetry argument). |

#### Axiom 6: `splitCount_crossGroup_stable`

| Aspect | Assessment |
|--------|-----------|
| **Justification** | **Moderate** |
| **Failure mode** | Same as Axiom 5. NOT needed for core impossibility. |

### A2. Idealization Quality

- **`Model` type**: Correctly abstract. The bundled `GBDTSetup` in Setup.lean resolves the global-vs-parametric issue.
- **`attribution : Fin P → Model → ℝ`**: Paper says "nonnegative" but Lean allows negative values. The formalization is MORE general than the paper claims — a virtue worth noting.
- **`firstMover : Model → Fin P`**: Assumes unique first-mover per model. Reasonable — exact ties have measure zero under continuous distributions.
- **`splitCount : Fin P → Model → ℝ`**: Real split counts are ℕ, but axiomatized values T/(2-ρ²) are irrational. Using ℝ is the correct design choice. Paper says "utilization count" (integer connotation) — minor mismatch.

### A3. Rashomon Property

- **Definition match**: Paper and Lean (`Trilemma.lean`) match exactly.
- **Standard vs novel**: The paper's definition captures attribution orderings, not loss proximity (Rudin 2024). This is a feature — captures exactly what's needed without epsilon-ball formalism.
- **Strength**: Neither too strong nor too weak. Three independent derivation paths provide robustness.
- **Failure modes**: ρ=0, deterministic model class, asymmetric algorithm. All correctly identified in the paper.

### A4. Cross-Model Validity

- **GBDT**: Correct axiom subset (Axioms 1-4 for impossibility; 5-6 only for DASH equity).
- **Lasso**: ZERO global axioms. All properties as theorem hypotheses. Excellent design.
- **Neural nets**: ZERO global axioms. Same clean design.
- **Random forests**: O(1/√T) claim is informal and unproved. Labeled as "contrast case, documentation only" — transparent but should cite supporting literature.

### A5. Axiom Consistency

- **Fin 4 construction** in Consistency.lean satisfies all 6 property axioms simultaneously. Sound.
- **No redundancy or tension detected.**
- The axiom system is "overkill" for the main result but enables quantitative and resolution results.

### Summary Table

| Axiom | Justification | Failure Mode | Downstream Impact | Results Affected |
|-------|:---:|---|---|---|
| firstMover_surjective | **Strong** | Non-exchangeable features | GBDT Rashomon | NOT core impossibility |
| splitCount_firstMover | **Moderate** | Deep trees, large LR | Exact ratio 1/(1-ρ²) | `attribution_ratio` |
| splitCount_nonFirstMover | **Moderate** | Same as above | Exact ratio | Same |
| proportionality_global | **Weak-Moderate** | Varying hyperparameters | DASH equity (exact) | `consensus_equity` |
| splitCount_crossGroup_symmetric | **Moderate** | Feature interactions | DASH equity (cross-group) | `consensus_equity` |
| splitCount_crossGroup_stable | **Moderate** | Feature interactions | DASH equity (cross-group) | `consensus_equity` |
| modelMeasurableSpace | **Strong** (infra) | None | Variance defs | `attribution_variance` |
| modelMeasure | **Strong** (infra) | None | Variance defs | `attribution_variance` |
| testing_constant | **Strong** | None (existential) | Query complexity | `query_complexity_lower_bound` |
| testing_constant_pos | **Strong** | None | Query complexity | Same |

---

## Phase B: Lean-Paper Correspondence

### B4. "Four-Line Proof" Claim

The Lean proof of `attribution_impossibility` (Trilemma.lean, lines 70-74) is exactly four tactic lines:
```lean
obtain ⟨f, f', h1, h2⟩ := hrash ℓ j k hj hk hjk
have hrank : ranking j k := (h_faithful f).mpr h1
have hcontra : attribution fs j f' > attribution fs k f' := (h_faithful f').mp hrank
linarith
```
**Verdict: Accurate.**

### B3. Axiom Dependency Verification

| Result | Paper axiom count | Actual | Match? |
|--------|:-:|:-:|:-:|
| `attribution_impossibility` | 0 behavioral | 0 behavioral | YES |
| `consensus_equity` | 6 | 6 property axioms + IsBalanced | YES |
| `split_gap_exact` | 2 | 2 (splitCount axioms) | YES |
| `attribution_ratio` | 3 | 3 | YES |
| `fairness_audit_impossibility` | 0 behavioral | 0 | YES |

### B1-B2. Named Results (17)

| # | Paper Result | Lean Name | Match | Hidden Hyps | Classification |
|---|---|---|:-:|---|:-:|
| 1 | Thm 1 (Attribution Impossibility) | `attribution_impossibility` | **Y** | None | ✓ |
| 2 | Thm 2 (Rashomon from Symmetry) | `rashomon_from_symmetry` | **Y** | Non-degeneracy is universal over pairs | ✓ |
| 3 | Thm 3 (Rashomon Inevitability) | `rashomon_inevitability` | **Partial** | "Informal" remark honestly disclosed | ✓ |
| 4 | Lem 4 (Split Gap) | `split_gap_exact` | **Y** | None | ✓ |
| 5 | Thm 5 (Attribution Ratio) | `attribution_ratio` | **Y** | None | ✓ |
| 6 | Prop 6 (Exact Flip Rate) | `binary_group_flip_rate` | **Partial** | Part (v) for general m not in Lean | ✓ |
| 7 | Cor 7 (DASH Equity) | `consensus_equity` | **Y** | IsBalanced as hypothesis — stated | ✓ |
| 8 | Thm 8 (DASH Pareto) | `dash_unique_pareto_optimal` | **Partial** | Cramér-Rao, Hoeffding not formalized | ✓ |
| 9 | Prop 9 (Ensemble Size) | `ensemble_bound_formula` | **Partial** | Lean is algebraic scaffolding only | Should be "argued" |
| 10 | Thm 10 (Design Space) | `design_space_theorem` | **Partial** | Step 3 exhaustiveness partial; Bayes argued | ✓ |
| 11 | Thm 11 (Unfaithfulness Bound) | `stable_complete_unfaithful` | **Y** | Exact 1/2 in separate file (quantitative) | Minor gap |
| 12 | Thm 12 (Path Convergence) | `relaxation_paths_converge` | **Y** | None | ✓ |
| 13 | Thm 13 (SBD) | `sbd_infeasible` + orbit bounds | **Partial** | General orbit counting partial | ✓ |
| 14 | Thm 14 (Conditional Impossibility) | `conditional_attribution_impossibility` | **Y** | None | ✓ |
| 15 | Thm 15 (Fairness Audit) | `fairness_audit_impossibility` | **Y** | "Coin flip" is qualitative in Lean | Minor gap |
| 16 | Thm 16 (FIM Impossibility) | `gaussian_rashomon_witnesses` | **Y** | Gaussian specialization only | ✓ |
| 17 | Thm 17 (Query Complexity) | `le_cam_lower_bound` | **Partial** | Le Cam content axiomatized | ✓ |

### Key Findings

- **No major misrepresentations.** When results are partially formalized, this is disclosed.
- **Three minor gaps**: Thm 11 "=1/2" named theorem is existential (quantitative in separate file); Thm 15 "coin flip" qualitative in Lean; Prop 9 is algebraic scaffolding, not the full Cramér-Rao result.

---

## Phase C: Theorem Soundness & Overclaiming

### C1. Novelty Assessment

| Claim | Novelty |
|-------|---------|
| Attribution Impossibility (Thm 1) | **New** — first to formalize cross-model stability as impossibility |
| Attribution Ratio 1/(1-ρ²) | **New** — architecture-discriminating quantitative bound |
| Design Space Theorem | **New** — complete characterization of achievable methods |
| DASH Pareto Optimality | **Incremental** — sample mean MVUE is classical; application is new |
| Symmetric Bayes Dichotomy | **Incremental** — application of Hunt-Stein to ML impossibilities |
| Lean formalization | **New** — no prior XAI impossibility formalization in any prover |

### C2. Overclaiming Probes

| Claim | Risk | Finding |
|-------|:---:|---------|
| "No feature ranking..." | **Low** | Rashomon scope stated clearly in theorem statements and abstract. Introduction opening could add "for correlated features" to first sentence. |
| "Provably Pareto-optimal" | **Medium** | Class ("among unbiased aggregations") specified deep in paper but NOT in abstract or executive summary. |
| "First formally verified impossibility in XAI" | **Low** | Hedged with "to our knowledge." No counterexample found. |
| "68% of datasets" | **Low** | Properly presented as estimate with power analysis and Wilson CI. |
| "Coin flip" | **Low** | Used specifically for binary symmetric groups. No loose usage. |
| "Four-line proof" | **Low** | Accurate (4 Lean tactic lines). Always accompanied by "from the Rashomon property." |

### C3. Design Space Exhaustiveness

The proof is `by_cases hcomp : ranking j k ∨ ranking k j` — excluded middle on a binary proposition. Logically airtight within the paper's definitions, but only covers methods producing deterministic binary rankings. Probabilistic, set-valued, or confidence-interval methods are not covered.

**Recommendation**: Add sentence noting exhaustiveness is relative to deterministic binary rankings.

### C4. Spearman Bound Utility

At P=30, m=2: bound gives ≈0.9996 — essentially useless quantitatively. The bound's role is existential (proving S < 1 in Lean), not practical.

**Recommendation**: Add note that the bound is qualitative, not tight.

### C5. SBD Novelty

Correctly credited to Hunt-Stein. The theorem is a specialization to finite groups with explicit unfaithfulness bounds. A decision theorist would view the general theorem as a corollary but might find the three ML instances useful. The paper's positioning as "proof technique" (line 270) is honest.

### C6. Scope Limitations

All five escapes (ρ=0, deterministic model, asymmetric algorithm, between-group features, conditional with unequal effects) are clearly documented. The Discussion has a thorough Limitations paragraph.

Most likely practitioner misapplication: assuming any ρ>0 makes rankings useless. The SNR analysis and Z-test mitigate this, but opening claims are strong.

---

## Phase D: Empirical Methodology Audit

### Critical Issues

1. **Class imbalance experiment: paper-JSON mismatch.** Paper claims ρ=0.9 and 20 seeds (line 3183); JSON shows ρ=0.8 and 30 models. The numbers in the paper tables match the JSON — the text description is stale.

2. **Flip rates exceeding 0.500.** Multiple experiments report max flip rates of 0.517 (class imbalance, missing data, time series). The theoretical ceiling for symmetric features is 0.500. The min(j>k, k>j) formulation should never exceed 0.500. Suggests either computation error, asymmetric features, or statistical noise with small N.

3. **SNR calibration R² misleading.** Overall R² across 6 datasets is **-1.07** (worse than mean). It only achieves R²=0.94 when restricted to Breast Cancer with SNR ≥ 0.5. The paper presents the result as "validated" and "conservative at low SNR" — generous framing for a poor fit (theory overestimates flip rates by 2.2x for SNR < 0.5).

### Moderate Issues

4. **CatBoost Z-test correlation.** r=-0.892, which does not satisfy "|r| > 0.89" (rounds to 0.89, not >0.89).

5. **M_min formula not empirically tested.** The claimed optimal ensemble size formula has no direct validation experiment.

6. **DASH 1/M scaling not formally tested.** Convergence is demonstrated directionally but no experiment plots variance vs 1/M to verify linearity.

7. **LightGBM not pinned in requirements.txt** despite being used in cross-implementation validation.

### Experiment Summary

| Experiment | Script | Grade | Key Issue | Theory Match |
|---|---|:-:|---|:-:|
| Synthetic Gaussian (ratio) | validate_ratio.py | B+ | α correction needed; uncorrected overpredicts 2-3x | Partial |
| Synthetic Gaussian (flips) | validate_ratio.py | A | Train/test confound acknowledged | Y |
| DASH convergence | validate_ratio.py | B | 1/M scaling not formally tested | Partial |
| Monte Carlo flip rate | monte_carlo_flip_rate.py | A | 10K trials, very rigorous | Y |
| Breast Cancer instability | real_world_validation.py | A- | Well-controlled, N=50 | Y |
| Cross-implementation | cross_implementation_validation.py | A- | CatBoost r=-0.892 rounds to 0.89 | Y |
| Non-SHAP attribution | permutation_importance_validation.py | B+ | Cross-method r=0.46 modest | Y |
| Neural network | nn_shap_validation.py | B | N=20 MLPs modest | Y |
| Fixed test set | fixed_testset_experiment.py | B | "63%" is pair-count, not variance decomposition | N/A |
| SNR calibration | snr_calibration.py | B- | R²=-1.07 overall; 0.94 only restricted | Partial |
| Prevalence survey | prevalence_survey_openml.py | B+ | 32% power acknowledged; CC-18 bias | N/A |
| Class imbalance | class_imbalance_instability.py | B- | **Paper-JSON mismatch on ρ and N** | Y |
| Hyperparameter sweep | hyperparameter_sensitivity.py | A | 1,620 fits, comprehensive | Y |
| Adversarial configs | adversarial_max_instability.py | A | 2,160 fits, exhaustive | Y |
| Conditional threshold | conditional_shap_threshold.py | A- | Sharp transition confirmed | Y |
| Financial (German Credit) | financial_case_study.py | B+ | Synthetic, not real lending data | Y |
| Financial (Lending Club) | lending_club_validation.py | A | Good specificity control | Y |
| LLM attention | llm_attention_instability.py | C+ | Attention ≠ SHAP; heavily caveated | N/A |

### Theory-Experiment Alignment

| Prediction | Validated? | Notes |
|---|:-:|---|
| Ratio 1/(1-ρ²) | Partial | Needs α correction; corrected model R²=0.89 |
| Flip rate 1/2 (binary) | **Yes** | Monte Carlo within 1.3% |
| DASH variance 1/M | Partial | Convergence shown but scaling not tested |
| M_min formula | **No** | No direct validation |
| Conditional escape Δβ~0.2 | **Yes** | Sharp transition confirmed |
| Spearman bound | **No** | No experiment |
| Efficiency amplification m/(m-1) | **No** | Theoretical only |
| Query complexity Ω(σ²/Δ²) | **No** | Theoretical only |

---

## Phase E: Actionability, Framing & Fairness

### E1. Practitioner Value

| Aspect | Assessment | Risk |
|--------|-----------|:---:|
| Single-model screen | Works from 1 model, 94% precision. GBDT-specific — not noted in exec summary. | Low |
| M_min circularity | Addressed via "pilot run of 5 models" (line 1942) but not where formula first appears | Medium |
| Z-test implementability | Clearly described, could be implemented from paper alone | None |
| DASH naming | Borderline — "you named the arithmetic mean." Defensible via Pareto-optimality proof and diagnostic framework. | Low |
| "5 lines of Python" | Pseudocode, not runnable. Missing imports, wrong syntax. | Medium |

### E2. Regulatory Claims

| Aspect | Assessment | Risk |
|--------|-----------|:---:|
| EU AI Act Art. 13(3)(b)(ii) | Quote omits "related to the use of" qualifier. Paper uses "proves" for legal conclusions — should use "establishes" or "argues." | Medium |
| ECOA adverse action | "Compliance failure" too strong — should be "compliance risk." Reg B doesn't mandate SHAP. | Medium |
| SR 11-7 | Correctly characterized as supervisory guidance. Modest claims. | None |
| Regulatory appendix | Thorough but assertive. Needs disclaimer that it's interpretation, not legal advice. | Low |

### E3. Related Work Fairness

| Aspect | Assessment | Risk |
|--------|-----------|:---:|
| Lundberg (SHAP) | Cited fairly but never explicitly exonerated. Paper should clarify SHAP itself is not the target. | Low |
| Bilodeau (2024) | Clear distinction. Table gives all "No" across comparison columns — technically accurate but stark. | Medium |
| Rudin (2024) | Properly credited. Paper strengthens her argument. | None |
| Chouldechova/Kleinberg | Accurate analogy. | None |
| Prior work table overall | One-sided comparison designed to highlight this paper's advantages. | Medium |

### E4. Naming & Branding

| Aspect | Assessment | Risk |
|--------|-----------|:---:|
| "The Attribution Impossibility" | "The" implies uniqueness. Formal verification gives credibility. | Low |
| "Faithful, Stable, Complete: Pick Two" | Accurate tagline, apt CAP analogy. | None |
| "DASH" | Borderline naming of averaging. Defended by full framework. | Low |
| "Symmetric Bayes Dichotomy" | Honest about Hunt-Stein roots. Novelty is in ML application. | Low |

---

## Phase F: Confidence Synthesis

### F1. Per-Result Confidence Table

| Result | Axiom Quality | Lean Match | Soundness | Empirical | Overall | Key Risk |
|--------|:---:|:---:|:---:|:---:|:---:|------|
| **Thm 1** (Core Impossibility) | BULLETPROOF | BULLETPROOF | BULLETPROOF | HIGH | **BULLETPROOF** | None |
| **Thm 2** (Rashomon from Symmetry) | BULLETPROOF | HIGH | HIGH | N/A | **HIGH** | Non-degeneracy is a hypothesis |
| **Thm 5** (Ratio 1/(1-ρ²)) | MEDIUM | HIGH | HIGH | MEDIUM | **MEDIUM** | Uncorrected ratio overpredicts 2-3x |
| **Cor 7** (DASH Equity) | MEDIUM | HIGH | HIGH | HIGH | **HIGH** | Global c is strongest axiom |
| **Thm 8** (DASH Pareto) | MEDIUM | MEDIUM | MEDIUM | MEDIUM | **MEDIUM** | Class of methods unspecified in abstract |
| **Thm 10** (Design Space) | HIGH | MEDIUM | MEDIUM | N/A | **MEDIUM** | "Exactly two families" is definitional |
| **Thm 13** (SBD) | HIGH | HIGH | MEDIUM | N/A | **HIGH** | Novelty is in application |
| **Thm 14** (Conditional) | BULLETPROOF | HIGH | HIGH | HIGH | **HIGH** | None |
| **Thm 15** (Fairness Audit) | BULLETPROOF | MEDIUM | HIGH | N/A | **HIGH** | Legal framing too strong |
| **Prevalence (68%)** | N/A | N/A | HIGH | MEDIUM | **MEDIUM** | Power is low |
| **M_min formula** | HIGH | MEDIUM | HIGH | **LOW** | **LOW** | Never empirically validated |
| **SNR calibration** | N/A | N/A | MEDIUM | **LOW** | **LOW** | R²=-1.07 overall |

### F2. Top 10 Vulnerabilities

| # | Attack | Current Defense | Severity |
|---|--------|----------------|:---:|
| **1** | Class imbalance: paper says ρ=0.9/20 seeds but JSON shows ρ=0.8/30 models | None — data-text mismatch | **Submission-blocking** |
| **2** | Flip rates >0.500 in multiple experiments (theoretical max is 0.500) | None — not addressed | **Submission-blocking** |
| **3** | SNR calibration R²=-1.07 overall but only R²=0.94 reported (cherry-picked subset) | "Conservative at low SNR" | **Weakening** |
| **4** | M_min formula has zero empirical validation | None | **Weakening** |
| **5** | "Provably Pareto-optimal" — class not specified in abstract/exec summary | Specified deep in paper | **Weakening** |
| **6** | Design Space "exactly two families" is trivially true (excluded middle) | None | **Weakening** |
| **7** | "5 lines of Python" is pseudocode that doesn't run | None | **Cosmetic** |
| **8** | CatBoost |r|=0.892 doesn't satisfy "|r|>0.89" | None | **Cosmetic** |
| **9** | "Proves" used for legal conclusions (EU AI Act, ECOA) | None | **Weakening** |
| **10** | Prior-work table gives every competitor "No" across all columns | Accurate but one-sided | **Weakening** |

### F3. Overall Assessment

**Ready for JMLR submission?** Almost. Items #1 and #2 are submission-blocking.

**Acceptance probability at current quality:** ~55-65%

**After fixing all 10 items:** ~75-80%

**Top 3 changes for maximum impact:**

1. Fix data integrity issues (#1, #2)
2. Add M_min validation experiment (#4)
3. Qualify "Pareto-optimal" and Design Space exhaustiveness (#5, #6)
