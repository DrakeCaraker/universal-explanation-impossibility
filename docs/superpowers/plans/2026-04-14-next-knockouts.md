# Next Knockout Experiments: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the framework from "a language" into "a tool" by demonstrating SAGE and Noether counting on real-world high-stakes data with actual SHAP values, and laying theoretical groundwork for continuous symmetry extension.

**Architecture:** Three phases. Phase 1 (validation with TreeSHAP) closes the gap between theory (SHAP) and experiment (gain-based importance). Phase 2 (real-world datasets) produces the knockout — SAGE predicting instability on clinical/financial data. Phase 3 (continuous symmetry theory) is theoretical groundwork for a follow-up paper.

**Tech Stack:** Python 3, shap, xgboost, sklearn, scipy, numpy

---

## Phase 1: Validate with Actual SHAP Values (Critical Gap)

### Task 1: Noether counting with TreeSHAP

**Files:**
- Create: `knockout-experiments/noether_treeshap.py`

The existing Noether counting used `feature_importances_` (gain-based). The theory is about SHAP values. These can give different rankings. Must verify the bimodal gap holds with actual TreeSHAP.

- [ ] **Step 1: Replicate the Noether counting design with TreeSHAP**

Same design as noether_counting_v2.py (P=12, g=3, β=[5,5,5,5,2,2,2,2,0.5,0.5,0.5,0.5], ρ=0.70, N=500) but:
- Train 50 XGBoost models (fewer because SHAP is slow)
- For each model, compute TreeSHAP values on 200 test points: `shap.TreeExplainer(model).shap_values(X_test)`
- For each model, compute mean absolute SHAP per feature: `np.mean(np.abs(shap_values), axis=0)`
- Compute pairwise flip rates on the SHAP-based rankings (not gain-based)

- [ ] **Step 2: Compare to gain-based result**

Report: within-group flip rate (SHAP) vs within-group flip rate (gain). Between-group flip rate (SHAP) vs between-group (gain). Does the bimodal gap persist?

- [ ] **Step 3: Test at ρ = 0.50 and ρ = 0.99 with SHAP**

Confirm ρ-invariance holds for SHAP, not just gain.

**Pass/fail:** Bimodal gap > 30pp with SHAP at ρ=0.70.

- [ ] **Step 4: Commit**

### Task 2: SAGE with TreeSHAP

**Files:**
- Create: `knockout-experiments/sage_treeshap.py`

- [ ] **Step 1: Run SAGE using SHAP-based flip rates instead of gain-based**

Same algorithm but using mean |SHAP| per feature instead of feature_importances_. Test on Breast Cancer and Wine (where correlation groups are known).

- [ ] **Step 2: Compare SAGE R² (SHAP-based) vs SAGE R² (gain-based)**

If SHAP-based SAGE gives higher R², the framework is better validated (since the theory IS about SHAP). If lower, investigate why.

- [ ] **Step 3: Commit**

---

## Phase 2: Real-World High-Stakes Datasets (The Knockout)

### Task 3: SAGE on clinical data (MIMIC or similar)

**Files:**
- Create: `knockout-experiments/sage_clinical.py`

A result on clinical data would transform the paper. The pitch: "SAGE predicts which feature importance comparisons a clinician can trust, and which are coin flips, BEFORE computing SHAP values."

- [ ] **Step 1: Find an accessible clinical dataset**

Options (in order of accessibility):
1. **UCI Heart Disease** (303 patients, 13 features) — sklearn-accessible
2. **Pima Indians Diabetes** (768, 8 features) — via OpenML
3. **Wisconsin Breast Cancer** — already tested, but it IS a real clinical dataset
4. **MIMIC-III** — requires PhysioNet access, likely not available now

Use UCI Heart Disease + Pima Diabetes + Breast Cancer (all accessible via sklearn/OpenML).

- [ ] **Step 2: For each dataset, run SAGE and report:**
1. Discovered feature groups (which clinical features are in the same orbit?)
2. Predicted instability
3. Observed instability (from 100 XGBoost retrains)
4. The specific feature pairs that are stable vs unstable
5. Clinical interpretation: "a clinician can trust that age is more important than cholesterol, but cannot trust the ranking between systolic and diastolic blood pressure"

- [ ] **Step 3: The knockout figure**

Create a figure showing: for 3 clinical datasets, the SAGE-predicted group structure overlaid on a heatmap of the flip-rate matrix. The visual should make it immediately clear which comparisons are trustworthy.

- [ ] **Step 4: Commit**

### Task 4: SAGE on financial data

**Files:**
- Create: `knockout-experiments/sage_financial.py`

- [ ] **Step 1: Use German Credit or similar accessible financial dataset**

German Credit (1000 samples, 20 features) is available via OpenML and was already used in the counterfactual experiment.

- [ ] **Step 2: Run SAGE and report which credit features are interchangeable**

"A loan officer can trust that income matters more than age, but the ranking between employment_duration and housing_status is a coin flip."

- [ ] **Step 3: Commit**

### Task 5: Noether counting on a real tabular benchmark

**Files:**
- Create: `knockout-experiments/noether_tabular_benchmark.py`

- [ ] **Step 1: Run Noether counting on 5 real datasets with known correlation structure**

Use actual TreeSHAP values. For each dataset:
- Train 50 XGBoost models
- Compute TreeSHAP-based flip rates for all feature pairs
- Apply SAGE to discover groups
- Compute observed bimodal gap
- Report: does the bimodal gap persist on real (not synthetic) data?

This is THE experiment that would make or break the paper for practitioners. If the bimodal gap exists on real data with real SHAP values, the Noether counting result is immediately actionable. If it collapses, the result is limited to synthetic data.

**Pass/fail:** Bimodal gap > 20pp on at least 3 of 5 real datasets.

- [ ] **Step 2: Commit**

---

## Phase 3: Continuous Symmetry Theory (Groundwork)

### Task 6: Theoretical derivation of the spectral prediction

**Files:**
- Create: `knockout-experiments/continuous_symmetry_theory.md`

- [ ] **Step 1: Derive the CCA spectrum prediction for O(n) symmetry**

For a neural network with hidden dimension n and symmetry group O(n):
- The representation of O(n) on ℝ^n decomposes as the trivial representation (1-dim, the mean) plus the standard representation (n-1 dim)
- But the ACTUAL representation (on the space of neuron importances) is more complex — it depends on how O(n) acts on the probe weights
- For a linear probe with weight matrix W ∈ ℝ^{10×n}, the O(n) action is W → W·O for O ∈ O(n)
- The invariant subspace: W such that W·O = W for all O ∈ O(n) → only W=0 satisfies this (since O(n) acts irreducibly on ℝ^n)
- Wait — this predicts dim(V^G) = 0, not 30

The resolution: O(n) is NOT the full symmetry group. Training breaks it. The effective symmetry group G_eff is the stabilizer of the training objective in O(n). This stabilizer is NOT O(n) — it's the subgroup of O(n) that preserves the learned decision boundary. The size of this subgroup determines dim(V^G).

Prediction: dim(V^G) ≈ number of distinct decision-relevant directions learned by the network. For MNIST (10 classes), this is ~10. For a problem with k classes and d features, dim(V^G) ≈ min(k, d).

- [ ] **Step 2: Test this prediction against the CCA data**

We found 9 dims with CCA > 0.99 and 30 with CCA > 0.90. MNIST has 10 classes. Is dim(V^G) ≈ 10? This would be a REAL prediction.

- [ ] **Step 3: Document the theory and test**

### Task 7: Derive the calibration constant theoretically

**Files:**
- Create: `knockout-experiments/calibration_theory.md`

- [ ] **Step 1: Model the calibration slope as a function of within-group coefficient variance**

The SAGE calibration slope is 0.345. Hypothesis: the slope equals the probability that a within-group flip actually changes the sign of the importance difference. If within-group features have true coefficients β_j + ε_j where ε_j ~ N(0, σ_ε²), then the flip rate is Φ(-|Δβ|/σ_ε) where Δβ is the between-group gap. The fraction of theoretical instability that manifests = average of Φ(-|Δβ|/σ_ε) across all pairs, where for within-group pairs Δβ=0 (gives 0.5) and for between-group pairs Δβ is large (gives ~0).

The calibration slope = E[flip_observed] / E[flip_predicted] = (n_within × 0.5 + n_between × 0) / (P(P-1)/2 × predicted_instability).

This is just the proportion of within-group pairs... which is the formula already. So the calibration should be ~1.0, not 0.345.

The discrepancy: in real data, within-group features do NOT have exactly equal coefficients. The flip rate within a "group" is < 0.5 because the features aren't truly exchangeable. The calibration slope measures the average within-group flip rate, which is < 0.5 when symmetry is approximate.

Derive: calibration slope ≈ average within-group flip rate × (n_within_pairs / n_total_pairs).

- [ ] **Step 2: Verify this formula against the 8 SAGE datasets**

- [ ] **Step 3: Commit**

---

## Execution Order

```
Phase 1 (validation): Tasks 1-2 — ~1 hour
  Task 1 (Noether with TreeSHAP) — 30 min
  Task 2 (SAGE with TreeSHAP) — 30 min

Phase 2 (knockout): Tasks 3-5 — ~2 hours
  Task 3 (clinical) ←→ Task 4 (financial) — parallel, 30 min each
  Task 5 (tabular benchmark) — 1 hour (the KEY experiment)

Phase 3 (theory): Tasks 6-7 — ~2 hours
  Task 6 (continuous symmetry) ←→ Task 7 (calibration) — parallel
```

**What would make this session's work a clear knockout for Nature MI:**
- Phase 1 confirms Noether bimodality with actual SHAP values (not just gain)
- Phase 2 Task 5 shows bimodal gap on ≥3 real datasets with SHAP → "practitioners can use this TODAY"
- Phase 2 Tasks 3-4 provide compelling clinical/financial narratives
- Phase 3 Task 6 predicts dim(V^G) ≈ 10 for MNIST and confirms against CCA → first genuine prediction of continuous symmetry theory
