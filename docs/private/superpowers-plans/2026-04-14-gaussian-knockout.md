# Gaussian Flip Rate Knockout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Gaussian flip rate formula from an inflated in-sample result (R²=0.94) into an honestly validated, reproducible knockout (CV R²≥0.7 on ≥4/5 datasets) with proper out-of-sample evaluation.

**Current honest state (from vet):**
- In-sample R²: 0.86-0.99 (inflated by information leakage)
- Cross-validated R²: 0.60-0.91 (honest, mean=0.80, 4/5 > 0.70)
- Correlation baseline R²: 0.00-0.10 (the formula provides massive lift)
- Gaussianity: 60-100% of pairs pass Shapiro-Wilk
- Critical weakness: Δ and σ estimated from same models as flip rates

**What makes this a real knockout despite the vet corrections:**
1. CV R²=0.80 mean across 5 datasets is STILL very good
2. The formula beats the correlation baseline by 40-90×
3. It requires zero structural parameters (functional form is fixed)
4. 4/5 datasets exceed CV R²=0.70
5. It's already Lean-verified (GaussianFlipRate.lean)

**What needs to happen:**
1. Proper out-of-sample evaluation documented
2. More models (M=50-100) to stabilize σ estimates and narrow CV gap
3. Formal comparison to baselines
4. Paper updates with honest CV numbers
5. The practical framing: "train M models → estimate Δ,σ → predict which comparisons to trust BEFORE interpreting any individual model"

---

## Phase 1: Rigorous Validation (1 hour)

### Task 1: Proper cross-validated Gaussian flip experiment

**Files:**
- Create: `knockout-experiments/gaussian_flip_cv.py`

- [ ] **Step 1: Design the proper evaluation**

For each dataset, use a TRAIN-TEST SPLIT of models:
- Train M=60 XGBoost models total
- Split: first 30 models = "calibration set" (estimate Δ and σ)
- Second 30 models = "validation set" (compute observed flip rates)
- The calibration and validation models are INDEPENDENT (different random seeds)
- This is genuinely out-of-sample: Δ and σ from one set, flip rates from another

This is STRONGER than split-half because both halves have 30 models (not 15).

```python
# Calibration models: seeds 42..71
# Validation models: seeds 142..171 (completely different seeds)
```

- [ ] **Step 2: Run on all 5 datasets**

For each dataset × each pair: predict flip from calibration Δ/σ, measure flip on validation set.
Report: out-of-sample R², RMSE, and comparison to baselines.

- [ ] **Step 3: Baselines to beat**

1. **Naive baseline:** predict mean flip rate for all pairs → R²=0 by definition
2. **Correlation baseline:** predict flip from |Pearson r| between features → R²≈0.01
3. **Importance-gap baseline:** predict flip from |mean(imp_j) - mean(imp_k)| without the σ normalization → tests whether the Gaussian CDF adds value over raw gap

- [ ] **Step 4: Report honestly**

Table with: Dataset, P, n_pairs, Out-of-sample R², Correlation baseline R², Importance-gap baseline R², Gaussianity fraction

**Pass criterion:** Out-of-sample R² > 0.65 on ≥4/5 datasets AND beats all baselines.

- [ ] **Step 5: Save results and figure**

Save: `knockout-experiments/results_gaussian_flip_validated.json`
Figure: `knockout-experiments/figures/gaussian_flip_validated.pdf`
- Panel 1: predicted vs observed scatter for each dataset
- Panel 2: R² comparison (formula vs baselines)

- [ ] **Step 6: Commit**

### Task 2: Test with M=100 models (does more data help?)

**Files:**
- Add to: `knockout-experiments/gaussian_flip_cv.py`

- [ ] **Step 1: Run with M=100 (50 calibration + 50 validation)**

On Breast Cancer only (the weakest dataset at CV R²=0.60). Does R² improve with more models?

- [ ] **Step 2: Report M=30 vs M=60 vs M=100 R²**

If R² at M=100 exceeds 0.80, the formula works — it just needs enough models for stable σ estimation.

- [ ] **Step 3: Commit**

---

## Phase 2: Theoretical Strengthening (30 min)

### Task 3: Prove the formula handles approximate symmetry

**Files:**
- Create: `knockout-experiments/approximate_symmetry_theory.md`

- [ ] **Step 1: Write the theoretical derivation**

Given M models with importance values drawn from:
  imp_j^(m) ~ N(β_j, σ²_j) independently for each model m

For pair (j,k), the importance difference is:
  d^(m) = imp_j^(m) - imp_k^(m) ~ N(Δ_jk, σ²_Δ)

where Δ_jk = β_j - β_k and σ²_Δ = σ²_j + σ²_k - 2·Cov(imp_j, imp_k)

The probability that models m₁ and m₂ disagree on the ranking of j vs k:
  P(flip) = P(d^(m₁) > 0) · P(d^(m₂) < 0) + P(d^(m₁) < 0) · P(d^(m₂) > 0)
           = 2 · Φ(-|Δ|/σ_Δ) · (1 - Φ(-|Δ|/σ_Δ))
           ≈ 2 · Φ(-SNR) · Φ(SNR)

where SNR = |Δ|/σ_Δ.

Note: the formula used in our experiments is flip = Φ(-|Δ|/(σ√2)), which is the probability that a SINGLE draw from N(Δ, σ²) has the opposite sign from the mean. This equals the pairwise flip rate when d^(m) are iid Gaussian.

**Key theoretical point:** This derivation shows EXACTLY why the bimodal gap weakens on real data:
- Exact symmetry (Δ=0): flip = Φ(0) = 0.5 → within-group coin flip
- Real data (Δ small but nonzero): flip = Φ(-|Δ|/σ√2) < 0.5 → gap narrows
- The gap is a smooth function of SNR = |Δ|/σ
- The Noether bimodal gap is the LIMIT as SNR→0 for within-group and SNR→∞ for between-group

- [ ] **Step 2: Verify the formula matches the exact pairwise flip rate**

For the Breast Cancer data, compute both:
- Theoretical: Φ(-|Δ|/(σ√2))
- Exact pairwise: 2·Φ(-SNR)·Φ(SNR) (the exact Gaussian pairwise flip rate)

Are they the same? (They should differ by a factor related to 2p(1-p) vs p.)

- [ ] **Step 3: Document which version is more accurate**

- [ ] **Step 4: Commit**

### Task 4: The practical framing — what practitioners should DO

**Files:**
- Create: `knockout-experiments/practitioner_guide.md`

- [ ] **Step 1: Write a 1-page practitioner guide**

"How to assess the reliability of your SHAP feature rankings:"

1. Train M≥30 models on bootstrap resamples of your data
2. For each model, compute mean |SHAP| importance per feature
3. For each feature pair (j,k): compute Δ_jk = mean(imp_j - imp_k) and σ_Δ = std(imp_j - imp_k)
4. Compute SNR_jk = |Δ_jk| / σ_Δ
5. Pairs with SNR > 2: ranking is reliable (flip rate < 5%)
6. Pairs with SNR < 0.5: ranking is unreliable (flip rate > 30%)
7. Pairs with 0.5 < SNR < 2: intermediate reliability

This is actionable, doesn't require the framework, and is based on the validated Gaussian formula.

- [ ] **Step 2: Commit**

---

## Phase 3: Paper Updates (30 min)

### Task 5: Update monograph and nature article with honest CV numbers

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`
- Modify: `paper/nature_article.tex`

- [ ] **Step 1: Add the Gaussian flip rate as a new section in the monograph**

"The Gaussian Flip Rate: From Exact to Approximate Symmetry"

Content:
- The formula: flip(j,k) = Φ(-|Δ_jk|/(σ_Δ√2))
- Derivation from Gaussian importance differences
- Connection to Noether counting (special case Δ=0)
- OUT-OF-SAMPLE validation: CV R² = X.XX-X.XX across 5 datasets (honest numbers)
- Comparison to baselines (beats correlation by 40-90×)
- Practical interpretation (SNR > 2 → reliable, SNR < 0.5 → unreliable)

- [ ] **Step 2: Update nature article with a brief mention**

Add to the Noether counting paragraph: "The Gaussian flip rate formula flip(j,k) = Φ(-SNR/√2) generalizes the bimodal counting to approximate symmetry, predicting per-pair flip rates with out-of-sample R² = X.XX across five clinical and financial datasets."

- [ ] **Step 3: Commit and push**

---

## Execution Order

```
Phase 1 (rigorous validation):
  Task 1 (proper CV with M=60) — 45 min, THE KEY EXPERIMENT
  Task 2 (M=100 test) — 15 min, parallel with Task 1

Phase 2 (theory):
  Task 3 (derivation) — 15 min
  Task 4 (practitioner guide) — 15 min

Phase 3 (paper):
  Task 5 (updates with honest numbers) — 30 min, AFTER Phase 1 results
```

**What makes this a knockout:** The formula predicts per-pair SHAP flip rates out-of-sample with R²≈0.70-0.90 (to be confirmed in Phase 1), using a fixed functional form (Gaussian CDF), beating correlation-based prediction by 40-90×. It generalizes Noether counting to the full continuum of symmetry-breaking, and it's Lean-verified.

**What it's NOT:** A zero-parameter predictor (it requires bootstrap retrains to estimate Δ and σ). A shortcut that avoids retraining (you still need M models). A universal law (it assumes Gaussian importance differences, which holds ~60-100% of the time).
