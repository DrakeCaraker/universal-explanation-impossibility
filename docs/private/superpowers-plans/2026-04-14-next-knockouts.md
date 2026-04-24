# Next Knockout Experiments: Implementation Plan (v2 — Vetted)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the framework from "a language" into "a tool" with two verified knockout results (Noether + SAGE validated on real data with SHAP), and explore continuous symmetry as a theoretical extension.

**Architecture:** Four phases. Phase 1 (TreeSHAP validation) is blocking — if it fails, we stop. Phase 2 (real-world data) is the knockout attempt. Phases 3-4 (continuous symmetry) are exploratory, not confirmatory.

**Honest status:** 1 clear knockout (Noether counting on synthetic data with gain-based importance). 1 near-knockout (SAGE). Critical gap: neither has been validated with actual SHAP values on real-world data. If TreeSHAP breaks the bimodal gap, the knockout is gone.

---

## Phase 1: TreeSHAP Validation (BLOCKING — must pass before Phase 2)

### Task 1: Noether counting with actual TreeSHAP values

**Files:**
- Create: `knockout-experiments/noether_treeshap.py`

**The critical question:** Does the 50pp bimodal gap between within-group and between-group pairs persist when using actual TreeSHAP values instead of gain-based `feature_importances_`?

- [ ] **Step 1: Design**

Same Noether design: P=12 features, g=3 groups of 4, β=[5,5,5,5, 2,2,2,2, 0.5,0.5,0.5,0.5]. Test at ρ ∈ {0.50, 0.70, 0.99}. N_train=500, noise=1.0.

Reduce to 50 models (SHAP is 10x slower than gain). For each model:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:200])
importance = np.mean(np.abs(shap_values), axis=0)  # mean |SHAP|
```

Compute rankings from mean |SHAP| importance. Compute pairwise flip rates.

- [ ] **Step 2: Run and compare to gain-based**

Report: within-group flip rate (SHAP), between-group flip rate (SHAP), bimodal gap (SHAP).

- [ ] **Step 3: Go/no-go**

**GO (proceed to Phase 2):** Bimodal gap > 30pp at ρ=0.70 with SHAP
**NO-GO (stop and investigate):** Bimodal gap < 15pp with SHAP

- [ ] **Step 4: Commit**

### Task 2: SAGE with TreeSHAP on Breast Cancer

**Files:**
- Create: `knockout-experiments/sage_treeshap.py`

- [ ] **Step 1: Run SAGE using SHAP-based flip rates on Breast Cancer**

50 XGBoost models, TreeSHAP importance. Discover groups. Compare to gain-based SAGE groups.

- [ ] **Step 2: Report whether the SAME groups are discovered**

If SHAP and gain give different groups → the two importance measures disagree, complicating the story.
If SHAP and gain give the same groups → the result is robust to the importance metric.

- [ ] **Step 3: Commit**

---

## Phase 2: Real-World Knockout (only if Phase 1 passes)

### Task 3: Noether counting on 5 real tabular datasets with TreeSHAP

**Files:**
- Create: `knockout-experiments/noether_real_datasets.py`

**THE knockout experiment.** If the bimodal gap persists on real data with real SHAP values, practitioners can use this TODAY.

- [ ] **Step 1: Select 5 datasets with ≥10 features**

1. Breast Cancer Wisconsin (30 features, classification)
2. Diabetes Pima (8 features, classification)
3. California Housing (8 features, regression)
4. Wine (13 features, classification)
5. Heart Disease UCI (13 features, classification)

- [ ] **Step 2: For each dataset**

1. Train 50 XGBoost models (bootstrap resampling, different seeds)
2. Compute TreeSHAP mean |SHAP| importance for each model
3. Compute all P(P-1)/2 pairwise flip rates
4. Apply SAGE clustering to discover groups
5. Classify pairs as within-group vs between-group
6. Report: flip rate histogram, gap, permutation p-value

- [ ] **Step 3: The knockout figure**

Multi-panel figure: one panel per dataset showing the flip-rate histogram colored by SAGE-discovered within-group (red) vs between-group (blue). If bimodal on ≥3 datasets → knockout confirmed.

- [ ] **Step 4: Clinical/financial interpretation**

For Breast Cancer: "Oncologists can trust that [feature X] is more important than [feature Y], but the comparison between [features A and B] is unreliable."

For Heart Disease: "Cardiologists can trust that [age/cholesterol] ranking is stable, but [blood pressure metrics] comparisons are coin flips."

- [ ] **Step 5: Go/no-go for the paper**

**KNOCKOUT:** Bimodal gap > 20pp on ≥3 of 5 datasets → write up for Nature MI
**PARTIAL:** Bimodal on 1-2 datasets → publishable in JMLR but not knockout
**FAIL:** No bimodality on real data → the result is limited to synthetic settings

- [ ] **Step 6: Commit**

### Task 4: SAGE prediction accuracy on real datasets

**Files:**
- Create: `knockout-experiments/sage_real_prediction.py`

- [ ] **Step 1: For each of the 5 datasets**

1. Run SAGE to discover groups and predict instability
2. Measure actual instability
3. Report calibrated R² across all 5 datasets (using LOO-CV calibration)

- [ ] **Step 2: Compare to correlation baseline**

Does SAGE still beat simple correlation-based clustering on real data?

- [ ] **Step 3: Commit**

---

## Phase 3: Continuous Symmetry Empirical Investigation (Exploratory)

### Task 5: dim(V^G) vs task complexity across datasets

**Files:**
- Create: `knockout-experiments/continuous_symmetry_empirical.py`

**Preliminary finding:** For MNIST, dim(V^G) at CCA>0.99 is:
- k=2 classes: 6 stable dims
- k=5 classes: 8 stable dims
- k=10 classes: 10 stable dims

The k=10 match (10 dims for 10 classes) is suggestive but k=2 shows a floor of ~6 that we cannot explain. The prediction dim(V^G) ≈ k is NOT confirmed for all k.

- [ ] **Step 1: Extend to more task configurations**

Test with MNIST:
- k ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10} classes
- hidden ∈ {64, 128, 256} neurons
- 5 models per configuration, CCA between all pairs

Report: dim(V^G) vs k for each hidden size.

- [ ] **Step 2: Test on a DIFFERENT dataset (CIFAR-10 or Fashion-MNIST)**

Does the relationship hold on different data distributions?

- [ ] **Step 3: Characterize the "floor"**

The floor (~6 stable dims for binary MNIST) may be the number of dominant features in the data (edges, curves, loops). Compute: for binary MNIST, what features do the 6 stable CCA dimensions correspond to? Visualize the weight vectors of the stable dimensions.

- [ ] **Step 4: Honest assessment**

Is dim(V^G) ≈ k a real prediction, or was the k=10 match coincidental?

**KNOCKOUT:** dim(V^G) scales linearly with k (slope ≈ 1) across ≥3 hidden sizes AND ≥2 datasets
**INTERESTING BUT NOT KNOCKOUT:** Relationship exists but with a data-dependent floor
**FAIL:** No systematic relationship between k and dim(V^G)

- [ ] **Step 5: Commit**

### Task 6: Spectral decay prediction

**Files:**
- Create: `knockout-experiments/spectral_theory.md`

- [ ] **Step 1: Theoretical derivation**

For a network with hidden dimension n learning k classes:
- The representation decomposes as V = V_task ⊕ V_free
- V_task (dimension ≈ k) is constrained by the classification objective
- V_free (dimension ≈ n-k) is unconstrained — different training runs explore different directions

Prediction for the CCA spectrum:
- Dimensions 1 to k: CCA ≈ 1.0 (task-constrained, stable)
- Dimensions k+1 to n: CCA decays as a function of the "stiffness" of the objective landscape

The decay rate depends on the Hessian eigenspectrum of the loss at the trained solution. Eigenvalues > threshold → constrained direction → high CCA. Eigenvalues < threshold → flat direction → low CCA.

This connects the CCA spectrum to the loss landscape geometry — a genuinely new theoretical prediction.

- [ ] **Step 2: Test: does the Hessian eigenspectrum predict the CCA spectrum?**

For one model, compute the top-k Hessian eigenvalues (via Lanczos iteration or power method). Compare the Hessian spectrum to the CCA spectrum. If they correlate, the theoretical prediction is validated.

Note: full Hessian for a 784×128 network is 100K×100K — intractable. But the top eigenvalues can be estimated via Hessian-vector products.

This is a computationally expensive test. Mark as EXPLORATORY.

- [ ] **Step 3: Commit**

---

## Phase 4: Lean Formalization of Continuous Symmetry (If Phase 3 succeeds)

### Task 7: Continuous symmetry impossibility in Lean

**Files:**
- Create: `UniversalImpossibility/ContinuousSymmetry.lean`

- [ ] **Step 1: State the theorem for compact groups**

The impossibility theorem already works for any group (no finiteness assumed). The NEW content is the resolution theorem for compact groups:

```lean
/-- For a compact topological group G with Haar measure μ,
    the orbit-average resolution R(θ) = ∫_G ρ(g) · E(g·θ) dμ(g)
    is stable and maximally faithful among stable maps. -/
theorem compact_orbit_average_stable
    [TopologicalGroup G] [CompactSpace G]
    [MeasurableSpace G] [MulAction G Θ]
    (μ : MeasureTheory.Measure G) [μ.IsHaarMeasure]
    ...
```

- [ ] **Step 2: Assess Mathlib readiness**

Check if Mathlib has: `TopologicalGroup`, `CompactSpace`, `IsHaarMeasure`, `MeasureTheory.integral`. If yes, the formalization may be feasible. If not, state the theorem as a `sorry` with a clear comment about what's missing.

- [ ] **Step 3: At minimum, formalize the spectral prediction statement**

Even if the full orbit-average theorem needs `sorry`, formalize:
```lean
/-- For a compact group G acting on V, the number of stable CCA
    dimensions equals dim(V^G), which can be computed from the
    character of the representation. -/
```

- [ ] **Step 4: Commit**

### Task 8: Add the CCA/continuous-symmetry findings to the monograph

- [ ] **Step 1: Write a new section "Extension to Continuous Symmetry"**

Content:
- The CCA spectrum finding (continuous vs discrete symmetry)
- The dim(V^G) vs k data (whatever Phase 3 found — positive or negative)
- The theoretical connection to loss landscape geometry
- Honest assessment of what works and what doesn't

- [ ] **Step 2: Update the nature article if results are positive**

- [ ] **Step 3: Commit and push**

---

## Execution Order

```
Phase 1 (BLOCKING):
  Task 1 (Noether TreeSHAP) → GO/NO-GO decision
  Task 2 (SAGE TreeSHAP) — parallel with Task 1

Phase 2 (only if Phase 1 GO):
  Task 3 (Noether on 5 real datasets) — THE knockout
  Task 4 (SAGE prediction accuracy) — parallel with Task 3

Phase 3 (exploratory, parallel with Phase 2):
  Task 5 (dim(V^G) vs k) — empirical
  Task 6 (spectral theory) — theoretical

Phase 4 (if Phase 3 positive):
  Task 7 (Lean formalization)
  Task 8 (paper updates)
```

**Estimated time:**
Phase 1: ~1 hour (SHAP is slow)
Phase 2: ~2 hours (50 models × 5 datasets × SHAP)
Phase 3: ~2 hours (multiple configurations)
Phase 4: ~2-4 hours (Lean work)

**What would make this a Nature MI knockout:**
Phase 1 passes + Phase 2 Task 3 shows bimodal gap on ≥3 real datasets + clinical narrative is compelling + Phase 3 finds a clean dim(V^G) relationship

**What would make this Nature main:**
All of the above + Phase 3 Task 6 connects CCA spectrum to Hessian eigenspectrum (new theory linking representation stability to loss landscape geometry, testable, confirmed)
