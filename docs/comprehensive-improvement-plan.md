# Comprehensive Improvement Plan — Attribution Impossibility

> 72 ideas generated → 17 cut → 2 deferred → **53 items in 6 phases**
>
> Vetted 3 rounds. 10 corrections applied. Confidence: HIGH (Phases 0,2,3,4), MEDIUM (Phase 1), LOW (Phase 5).

---

## Phase 0: Pre-Submission Critical Path (Now → May 1)

**Goal:** Maximize NeurIPS abstract + arXiv preprint quality. 8 working days.

### 0A. Large-Scale Prevalence Survey [3 days]
- Download 100+ datasets from OpenML (prefer CC-18 benchmark suite for reproducibility)
- Run F5 screen + F1 validate on each with XGBoost defaults (subsample=0.8)
- Report: N/100 datasets with instability, stratified by P, ρ_max, domain
- Include P=500-1000 datasets for enterprise-scale claim
- Update paper: "60% of 37 datasets" → "X% of N datasets"
- **Caveat:** OpenML skews toward tabular ML benchmarks — acknowledge selection bias in paper
- **Script:** Extend `prevalence_survey.py` with `openml.datasets.list_datasets()` bulk loader
- **Output:** Table + figure in supplement; one-line summary in abstract

### 0B. Cross-Framework Validation [1 day]
- Extend `cross_implementation_validation.py` with LightGBM, CatBoost, sklearn GBM
- Same synthetic data setup (P=20, L=4, m=5, ρ sweep)
- Table: framework × ρ × flip rate × ratio
- Verdict: "Instability is framework-independent" or "Framework X shows anomaly because..."
- **Output:** Table in supplement, one sentence in main text

### 0C. DASH vs Existing Methods Benchmark [2 days]
- Compare DASH (M=25 ensemble average) against:
  - Bootstrap aggregated SHAP (resample data, single model)
  - SHAP with confidence intervals (single model, nsamples variance)
  - Subsampled SHAP averaging (subsample features, single model)
  - BayesSHAP (Slack et al. 2021, if available)
  - SAGE (Covert et al. 2020)
  - Boruta (feature selection stability comparison)
- Metrics: flip rate, ranking variance, computational cost (wall time)
- **Output:** Comparison table in supplement; winner summary in Discussion

### 0D. Permutation Importance Comparison [0.5 days]
- Check existing `permutation_importance_validation.py` output
- Run permutation importance with 50 seeds, same synthetic setup
- Show permutation importance is ALSO unstable (same Rashomon cause)
- Preempts "just use permutation importance" objection
- **Output:** One paragraph + figure in supplement

### 0E. Monte Carlo Flip Rate Validation [0.5 days]
- 10,000 simulations per ρ value
- Compare empirical flip rate to Φ(-SNR) prediction
- Report: max |empirical - predicted| across ρ ∈ {0.3, 0.5, 0.7, 0.9, 0.95, 0.99}
- **Output:** Validation table in supplement

### 0F. arXiv Preprint [0.5 days]
- Run `make arxiv`
- Final proofread pass on JMLR version
- Submit to cs.LG + cs.AI + stat.ML + cs.LO
- Time: April 30 (before May 4 NeurIPS abstract deadline)

### 0G. D'Amour et al. Engagement [0.5 days]
- Add paragraph to Discussion explicitly connecting to D'Amour et al. (2022) underspecification
- Frame: "Our result formalizes the attribution-specific consequence of underspecification"
- Cite and differentiate: they show predictions vary; we prove rankings must vary
- **Output:** 1 paragraph in Discussion

---

## Phase 1: Theory Hardening (May 7 → June 30)

**Goal:** Eliminate axioms, close gaps, strengthen for JMLR. Research-uncertain items flagged.

**Contingency:** If 1A, 1B, or 1C fail after 1 week of effort, document the attempt in the paper as "open problem with progress" and move on. The paper is strong without these — they are upside, not requirements.

### 1A. Derive Proportionality from TreeSHAP [1 week, MEDIUM confidence]
- Formalize TreeSHAP's split-attribution mechanism in Lean
- Show proportionality_global emerges as leading-order approximation
- Success: eliminate 1 axiom (17 → 16)
- Failure mode: derivation requires assumptions about tree structure that are themselves axioms
- **Fallback:** Document derivation path in paper, keep axiom, note "derivable under [conditions]"

### 1B. Spearman Midrank Transposition Lemma [1 week, MEDIUM confidence]
- Prove the combinatorial counting argument for midrank transpositions
- Close gap between derived O(m²/P³) and axiomatized m³/P³
- Success: eliminate spearman_classical_bound axiom (16 → 15)
- **Fallback:** Keep both bounds, note derived bound captures qualitative scaling

### 1C. Query Complexity from First Principles [1-2 weeks, LOW confidence]
- Formalize testing algorithms and error probability guarantees
- Eliminate testing_constant + testing_constant_pos axioms
- This requires formalizing Le Cam's method of two fuzzy hypotheses
- **Fallback:** Keep 2 axioms — they are clearly stated and well-justified. Attempting this is upside-only.

### 1D. Robustness Conjecture → Theorem [3 days, MEDIUM-HIGH confidence]
- Prove: if ‖Σ̂ - Σ‖_F < ε, flip rates change by O(ε/σ)
- Uses: Lipschitz property of Φ (standard normal CDF), matrix perturbation bounds
- Lean formalization possible (Φ already defined in GaussianFlipRate.lean)
- Convert "conjecture" → "theorem" in Discussion

### 1E. Intersectional Fairness Compounding [2 days, HIGH confidence]
- Prove: K protected attributes with symmetric correlation → flip probability ≥ 1 - (1/2)^K
- Lean formalization in FairnessAudit.lean (extends existing 4 theorems)
- Quantitative: K=3 → 87.5% probability of at least one wrong audit conclusion

### 1F. Generalize to Mutual Information [1 week, MEDIUM confidence]
- Weaken Rashomon property: ρ > 0 → I(X_j; X_k) > 0
- Show: I = 0 ⟺ features independent ⟺ no Rashomon property ⟺ stable ranking exists
- Requires: defining mutual information in Lean (not in Mathlib yet)
- **Fallback:** State as theorem with proof in supplement, defer Lean formalization

### 1G. Paper Sync [2 days]
- After all Phase 1 work: update theorem/axiom counts in ALL paper versions
- Run `make verify` to confirm consistency
- Update CLAUDE.md counts
- Run `make paper` to recompile all versions

---

## Phase 2: Experimental Expansion (May 7 → July 31)

**Goal:** Address every empirical gap. All items HIGH confidence (standard ML experiments).

### 2A. True Causal DAG Experiment [2 days]
- Generate data from known DAG: X₁ → Y, X₂ → Y, X₁ ↔ X₂ (confounded)
- Vary causal effect difference: Δβ ∈ {0, 0.1, 0.2, ..., 1.0}
- Show: conditional SHAP resolves when Δβ > threshold, fails when Δβ ≈ 0
- Validates Theorem conditional-impossibility + escape condition
- **Output:** Figure + 1 paragraph in conditional SHAP section

### 2B. Longitudinal Retraining Study [2 days]
- Train XGBoost on Breast Cancer 50 times with simulated data drift (5% noise injection per round)
- Track: feature ranking at each retrain, cumulative flip count
- Show: instability persists and may grow with distribution shift
- **Output:** Time-series figure in supplement

### 2C. Adversarial Dataset Construction [1 day]
- Grid search: ρ ∈ [0.9, 0.999], m ∈ {2,3,...,10}, P ∈ {10,...,100}, T ∈ {10,...,500}
- Find configuration maximizing flip rate
- Report: worst-case scenario for SHAP users
- **Output:** Table + guidance paragraph

### 2D. DASH Breakdown Point [2 days]
- Contaminate: replace K of M=25 models with adversarial (trained on permuted labels)
- Measure: at what K/M does DASH ranking degrade?
- Compare: standard mean vs trimmed mean (5% trim)
- Report breakdown point as fraction
- **Output:** Figure + 1 paragraph in Discussion

### 2E. NLP Token Attribution [1-2 days]
- Extend existing `llm_attention_instability.py`
- Fine-tune DistilBERT on SST-2 (5 seeds, 3 epochs each)
- Compute token-level SHAP via partition explainer
- Measure token ranking instability across seeds
- **Output:** Table in supplement; one sentence in Discussion

### 2F. Hyperparameter Sensitivity [1 day]
- Vary: learning_rate ∈ {0.01, 0.1, 0.3}, max_depth ∈ {3,6,9}, n_estimators ∈ {50,100,500}
- Full factorial (27 configurations) × 20 seeds × ρ sweep
- Show: instability varies with hyperparameters but never vanishes (for ρ > 0)
- **Output:** Heatmap figure in supplement

### 2G. Class Imbalance Interaction [1 day]
- Subsample majority class: ratios 1:1, 1:5, 1:10, 1:50
- Show: instability amplified by class imbalance (fewer effective samples → wider Rashomon set)
- **Output:** Figure in supplement

### 2H. Missing Data Interaction [1 day]
- Inject: MCAR (random), MAR (depends on observed), MNAR (depends on missing value)
- Missing rates: 5%, 10%, 20%
- Show: missing data + collinearity = compounding instability
- **Output:** Figure in supplement

### 2I. Time-Series Attribution [2 days]
- Synthetic: 10 autocorrelated features (AR(1) with cross-correlations)
- Real: stock return features (lagged returns, rolling averages, volatility — naturally correlated)
- Show: impossibility applies to temporal feature construction
- **Output:** Section in supplement

### 2J. PDP Stability Comparison [1 day]
- Compute partial dependence plots for top 5 features, 50 seeds
- Compare PDP ranking stability to SHAP ranking stability
- Result: PDP likely more stable (marginal, not conditional) but less faithful
- **Output:** Comparison paragraph in supplement

### 2K. Regulatory Case Study [3 days]
- Simulate ECOA adverse action pipeline: 10,000 synthetic loan applications
- 5 income features (salary, bonus, investment, rental, freelance) with ρ=0.7
- Train M=25 XGBoost models, compute SHAP
- Count: how many applicants receive different "top reason for denial" across retrains
- Report: "X% of denied applicants would receive a different adverse action reason"
- Include legal analysis: this meets SR 11-7 definition of "material model risk"
- **Output:** Case study section in JMLR; one paragraph in NeurIPS

### 2L. Additional Method Comparisons [1 day]
- SAGE (Covert et al. 2020): Shapley-based, handles correlations differently
- Boruta: wrapper-based feature selection stability
- Report stability comparison with SHAP and DASH
- **Output:** Comparison table in supplement

### 2M. Paper Sync [3 days]
- Write up ALL Phase 2 experiments into JMLR version
- Add summary table: "Experiment × Dataset × Key Finding"
- Update abstract prevalence number if Phase 0A changed it
- Run `make paper` and `make verify`

---

## Phase 3: Software & Reproducibility (June → August)

**Goal:** Make the work citable AND usable. All items HIGH confidence.

### 3A. PyPI Package [3 days]
- Package dash-shap with proper pyproject.toml
- Public API: `screen()`, `validate()`, `consensus()`, `report()`, `recommend_M()`
- `pip install dash-shap` works
- Add badges to README

### 3B. Docker Container [1 day]
- Dockerfile: Lean 4 (elan) + Python 3.11 + TeX Live
- `docker build -t dash-impossibility . && docker run dash-impossibility make all`
- Publish to Docker Hub or ghcr.io
- Add to README

### 3C. Interactive Demo [2 days]
- Streamlit app: sliders for ρ, m, P, T, M
- Real-time: flip rate, ratio, DASH convergence curve, SNR diagnostic
- Deploy on Streamlit Cloud (free hosting)
- Memorable for conference attendees

### 3D. Upstream SHAP PR [1 day]
- Draft already exists at `shap_pr/`
- Clean up, add tests, submit to slundberg/shap
- Feature: `shap.stability_check(explainer, X)` → warning if unstable pairs detected

### 3E. GitHub Action [2 days]
- `dash-shap/stability-check@v1`
- Inputs: model path, data path, threshold
- Output: stability report (pass/warn/fail)
- Users add to their CI: "fail if SHAP instability detected"

### 3F. Lean Package on Reservoir [1 day]
- Register DASHImpossibility on Lean 4 Reservoir
- Other formalization projects can `require DASHImpossibility` in their lakefile
- First formally verified XAI impossibility as reusable library

---

## Phase 4: Communication & Impact (July → October)

**Goal:** Maximize citations and adoption.

### 4A. Blog Post [3 days]
- Title: "Your Feature Rankings Are a Coin Flip (and We Proved It)"
- Interactive d3.js trilemma triangle (click vertices to see tradeoffs)
- Embed ratio divergence animation
- Publish: personal site + Medium + Towards Data Science
- Time with arXiv posting

### 4B. Workshop Submissions [1 week total]
- **NeurIPS Formal Verification workshop** (4pp): formalization methodology angle
  - "190 Theorems, 0 Sorry: Lessons from Formally Verifying an Impossibility in XAI"
- **ICML Responsible AI workshop** (4pp): fairness angle
  - "Fairness Audits Under Collinearity: A Formal Impossibility"
- **FAccT** (8pp): regulatory compliance angle
  - "The Attribution Impossibility and Its Consequences for Algorithmic Accountability"
- Each largely extractable from main paper with venue-specific framing

### 4C. Regulatory White Paper [3 days]
- 10-page non-technical document for compliance officers
- Maps impossibility to: SR 11-7 (model risk), ECOA (adverse action), EU AI Act Art. 13
- Decision flowchart: "Should you trust this SHAP ranking?"
- Distribute via banking/insurance compliance channels

### 4D. Twitter/X Thread [0.5 days]
- 15-tweet thread with key figures
- Hook: "I trained the same model twice. The 'most important feature' changed. Here's why that's not a bug — it's a theorem."
- Time with arXiv release

### 4E. Conference Tutorial Proposal [2 days]
- Submit to PyData or SciPy 2027
- "Stable Feature Attribution in Practice" (90 min hands-on)
- Participants run F5→F1→DASH on their own datasets
- Materials: notebook + dash-shap package

### 4F. Guest Lecture Materials [2 days]
- 30-slide deck covering: motivation, impossibility, DST, DASH, formalization
- 3 homework exercises (prove simplified impossibility, run DASH, interpret F1 output)
- Target: ML courses (grad), XAI courses, formal methods courses

### 4G. Press Outreach [0.5 days]
- Pitch to Quanta Magazine, Ars Technica, MIT Technology Review
- Angle: "Mathematicians prove AI explanations are fundamentally unreliable under common conditions"
- Include: accessible summary, trilemma figure, real-world implications
- Time with arXiv release

---

## Phase 5: Extensions (Post-Publication, Q1 2027+)

**Goal:** New papers building on the framework. LOW confidence — research directions.

### 5A. Counterfactual Explanation Impossibility
- Analogous impossibility for algorithmic recourse (Wachter et al. 2017)
- Collinear features → non-unique counterfactuals → unstable recourse recommendations
- SBD framework applies directly
- **Scope:** Full paper (Paper 4 in research program)

### 5B. Attention Attribution Impossibility
- Impossibility for attention-based explanations in transformers
- Connect to Jain & Wallace (2019): attention weights are unstable across equivalent models
- **Scope:** Full paper

### 5C. Social Choice Formal Connection
- Arrow's impossibility and attribution impossibility share structure
- Lean formalization of the shared categorical framework
- **Scope:** Short paper at LICS or AAAI

### 5D. Calibration Impossibility Connection
- Connect to Kleinberg et al. (2016) calibration impossibility
- Both arise from symmetry-breaking in symmetric decision problems
- SBD as unifying framework
- **Scope:** Workshop paper or note

### 5E. Progressive DASH Formalization
- Formalize sequential testing guarantee for adaptive M selection
- Prove: progressive DASH is asymptotically optimal
- **Scope:** Lean extension + supplement update

### 5F. Game-Theoretic Formalization
- Mechanism design perspective: DASH as information aggregation mechanism
- Explore connection to VCG mechanism (speculative — may not hold)
- **Scope:** Exploratory, may lead to paper

### 5G. Information-Theoretic Lower Bound
- Prove: Ω(log m) bits of information necessarily lost per symmetric group
- Tighter than current log₂(m!) characterization
- **Scope:** Technical note or supplement addition

---

## Timeline Summary

```
Apr 3-May 1:   Phase 0 (8 days) — Pre-submission experiments + arXiv
May 4:         NeurIPS abstract deadline
May 6:         NeurIPS paper deadline
May 7-Jun 30:  Phase 1 (8 weeks) — Theory hardening for JMLR
May 7-Jul 31:  Phase 2 (12 weeks) — Experimental expansion, overlapping with Phase 1
Jun-Aug:       Phase 3 (scattered) — Software ecosystem
Jul-Oct:       Phase 4 (scattered) — Communication & outreach
Q4 2026:       JMLR submission
Q1 2027+:      Phase 5 — Extensions (new papers)
```

## Vet Summary

| Finding | Confidence | Justification |
|---------|-----------|---------------|
| Phase 0 is achievable by May 1 | HIGH | All scripts exist or extend existing ones; 8 working days is sufficient |
| Phase 1 axiom elimination will succeed | MEDIUM | 1A and 1B are stated open problems; 1C is research-hard. Contingency: keep axioms, document progress |
| Phase 2 experiments will produce useful results | HIGH | Standard ML experiments with clear methodology |
| 100-dataset survey will change the prevalence number | HIGH | Current 37 datasets likely biased low; more data = more power |
| Cross-framework validation will confirm universality | HIGH | Rashomon property is framework-independent by construction |
| Workshop papers will be accepted | MEDIUM | Depends on workshop existence and reviewer alignment |
| Phase 5 extensions will produce new papers | LOW | Research directions with uncertain outcomes |

## Open Questions
1. Use OpenML CC-18 (72 curated) or arbitrary 100+? CC-18 is more defensible.
2. Should DASH vs BayesSHAP include Slack et al. (2021)? Yes if package is pip-installable.
3. Is it better to keep Le Cam axioms (clearly stated) than attempt risky derivation? Probably yes — attempt is upside-only.
4. NeurIPS 2026 workshop CFPs not yet announced — monitor neurips.cc in June.
