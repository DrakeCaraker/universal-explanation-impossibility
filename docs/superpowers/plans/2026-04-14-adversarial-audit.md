# Adversarial Audit of Revolutionary Directions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rigorously stress-test all three "revolutionary direction" results. For each result, identify the strongest possible attack, execute it, and determine whether the result survives. No invented criticisms — every test targets a specific plausible failure mode.

**Architecture:** Three parallel audit tracks (one per direction) + one cross-cutting track. Each track has explicit pass/fail criteria that would cause us to retract the claim.

**Tech Stack:** Python 3, numpy, scipy, sklearn

---

## Track A: SAGE G-Discovery Audit

### Task 1: Threshold sensitivity — is R² robust to the clustering threshold?

**Files:**
- Create: `knockout-experiments/sage_audit.py`

**The attack:** The threshold of 0.30 for clustering may have been implicitly tuned. If R² collapses at other thresholds, the result is an artifact.

- [ ] **Step 1: Sweep thresholds from 0.05 to 0.50 in steps of 0.05**

For each threshold, re-run the SAGE clustering on all 5 datasets and compute the calibrated R². Plot R² vs threshold.

- [ ] **Step 2: Pass/fail criterion**

**PASS:** R² > 0.7 for at least 60% of thresholds in [0.10, 0.45]
**FAIL:** R² > 0.7 only at the chosen threshold of 0.30

### Task 2: Leave-one-out cross-validation — does calibration generalize?

**The attack:** With only 5 points and 2 free parameters, ANY monotonic relationship gives high R². The calibration may not generalize.

- [ ] **Step 1: For each of the 5 datasets, hold it out**

Fit the calibration (slope, intercept) on the other 4, predict the held-out dataset's instability, compute absolute error.

- [ ] **Step 2: Compute LOO-CV R² and mean absolute error**

**PASS:** LOO-CV R² > 0.5 AND mean absolute error < 0.10
**FAIL:** LOO-CV R² < 0.3 (calibration doesn't generalize)

### Task 3: Trivial baseline comparison — does correlation matrix give the same answer?

**The attack:** Maybe you don't need the framework at all. Just cluster features by pairwise correlation, count groups, and predict instability.

- [ ] **Step 1: For each dataset, compute the feature correlation matrix**

Cluster features using hierarchical clustering on |correlation| with threshold 0.70 (features with |r| > 0.70 are in the same group).

- [ ] **Step 2: Compute predicted instability from correlation-based groups**

Use the same formula as SAGE (1 - g/P) but with correlation-based groups instead of flip-rate-based groups.

- [ ] **Step 3: Compare R² of correlation-baseline vs SAGE**

**PASS:** SAGE R² > correlation-baseline R² (SAGE adds value beyond simple correlation)
**FAIL:** Correlation-baseline R² ≥ SAGE R² (the framework adds nothing)

### Task 4: Expand to 10+ datasets

**The attack:** 5 datasets is too few. Results may not hold on more diverse data.

- [ ] **Step 1: Add 5-10 more datasets from sklearn/OpenML**

Candidates: boston housing (if available), digits, covtype (subsample), adult (subsample), or any sklearn toy dataset with 5+ features.

- [ ] **Step 2: Re-compute SAGE on all 10+ datasets**

Report calibrated R² on the expanded set.

**PASS:** R² > 0.7 on 10+ datasets
**FAIL:** R² < 0.5 on expanded set

---

## Track B: MI Non-Uniqueness Audit

### Task 5: Convergence check — are models properly trained?

**The attack:** max_iter=50 is too few. Models may not have converged, meaning they're in DIFFERENT local minima — and the Jaccard=chance result reflects loss landscape diversity, not permutation symmetry.

- [ ] **Step 1: Re-run with max_iter=500 (10x more training)**

Train 10 MLPs with max_iter=500 and measure: (a) accuracy improvement, (b) Jaccard change, (c) subspace cosine change.

- [ ] **Step 2: Check if accuracy is now >99%**

If accuracy improves to >99%, models are more likely to be in the SAME basin. If Jaccard is still chance-level, that's stronger evidence for permutation symmetry.

**PASS:** Accuracy > 99% AND Jaccard still < 0.10
**FAIL:** Accuracy stays at 97% (models still haven't converged — the experiment is flawed)

### Task 6: Permutation alignment — does Hungarian matching help?

**The attack:** The Jaccard is at chance because we're comparing RAW neuron indices. If we ALIGN neurons using the Hungarian algorithm (finding the optimal permutation), agreement should improve. If aligned Jaccard is high (>0.5), the non-uniqueness is JUST permutation symmetry (which is boring/known). If aligned Jaccard is STILL low, something deeper is going on.

- [ ] **Step 1: For each model pair, compute the optimal neuron alignment**

Use scipy.optimize.linear_sum_assignment to find the permutation π that maximizes the agreement between model A's neuron importance vector and model B's (permuted).

- [ ] **Step 2: Compute ALIGNED Jaccard after applying the optimal permutation**

- [ ] **Step 3: Interpret**

**If aligned Jaccard > 0.5:** The non-uniqueness IS just S_n permutation symmetry (the models learn the same thing with different neuron labels). The 1/n ceiling is the right explanation. This is the EXPECTED result under the framework.

**If aligned Jaccard < 0.2:** The non-uniqueness is DEEPER than permutation symmetry. The models learn genuinely different representations. This is more interesting but is NOT what the framework predicts (the framework only predicts S_n symmetry, not representation non-uniqueness).

### Task 7: Subspace analysis — why is subspace cosine so low?

**The attack:** Subspace cosine = 0.18 means the 10-dimensional probe subspaces are nearly orthogonal across models. This could mean: (a) the probe is capturing noise, not structure, or (b) the representations genuinely differ. Need to distinguish.

- [ ] **Step 1: Compute probe accuracy for each model**

If probes achieve >90% accuracy, they're capturing real structure. If probes are at <70% accuracy, they're mostly noise.

- [ ] **Step 2: Compute within-model subspace stability**

Retrain the probe 10 times on the SAME model with different random seeds. Compute subspace cosine across probe retrains. If this is also low (< 0.5), the probe itself is unstable — the low cross-model cosine is a probe artifact, not a model property.

- [ ] **Step 3: Pass/fail criterion**

**PASS (for the framework):** Probe accuracy > 90% AND within-model cosine > 0.8 AND cross-model cosine < 0.3 → the models genuinely have different representations
**FAIL (for the experiment):** Probe accuracy < 80% OR within-model cosine < 0.5 → the probe is too noisy to draw conclusions

---

## Track C: Quantum Measurement Audit

### Task 8: Is the quantum η result circular?

**The attack:** We DEFINED dim(V^G) to be the dimension of the diagonal subspace (populations), which is BY DEFINITION the space of properties accessible from a computational basis measurement. Then we "predicted" that η = dim(V^G)/dim(V) equals the fraction of accessible information. This is a tautology: we defined η to be the thing we're measuring.

- [ ] **Step 1: Write out the logical chain explicitly**

Document: (a) the definition of V^G in the quantum setting, (b) the definition of "accessible information fraction," (c) whether these are the same thing by definition or whether there is a non-trivial step connecting them.

- [ ] **Step 2: Determine if there's a TESTABLE prediction**

A non-circular test: use the framework's η formula to predict a quantity that is NOT part of the definition of V^G. For example:
- Predict the sample complexity of quantum state tomography from η
- Predict the classical capacity of the measurement channel from η
- Predict the error rate of a specific tomography protocol from η

If η predicts any of these quantities (which are NOT how V^G was defined), the result is non-circular. If it only predicts dim(V^G)/dim(V) itself, it's tautological.

**PASS:** η predicts a quantity that was not used to define V^G
**FAIL:** η only predicts itself

### Task 9: Is the "unification" vacuous?

**The attack:** Any projection onto a subspace preserves dim(subspace)/dim(total) of the information. Saying "η = dim(V^G)/dim(V) works for SHAP AND quantum measurement" is like saying "ratios work for fractions AND percentages." The formula is so general it applies to everything, which means it says nothing.

- [ ] **Step 1: State the strongest version of the unification claim**

What specifically does the framework ADD beyond "projection loses information proportional to the projected-away dimensions"?

- [ ] **Step 2: Identify what would be non-trivial**

A non-trivial unification would be: the framework predicts that two domains have the SAME η (because they have isomorphic symmetry groups) even though the domains appear unrelated. Does this happen? Are there domains with the same G where the framework predicts the same instability and this is confirmed?

- [ ] **Step 3: Pass/fail criterion**

**PASS:** There exists at least one non-obvious cross-domain prediction (same G → same η → same instability, confirmed)
**FAIL:** Every application of η = dim(V^G)/dim(V) is just "projection math" applied independently to each domain

---

## Track D: Cross-Cutting Audit

### Task 10: The "so what" test — does anything fail WITHOUT the framework?

**The attack:** Strip away all the framework language. Can you make every prediction using only: (a) "correlation implies instability," (b) "more parameters → more ambiguity," (c) "projection loses information"? If so, the framework is a vocabulary, not a tool.

- [ ] **Step 1: For each of the 3 results, state the prediction in plain language without the framework**

- SAGE: "Features with high pairwise correlation have unstable relative importance rankings. More correlated features → more instability."
- MI: "Neural networks trained from different random seeds learn different internal representations."
- Quantum: "Measuring a quantum state in one basis gives you 1/3 of the state information (for a qubit)."

- [ ] **Step 2: Is ANY of the above non-obvious without the framework?**

- SAGE plain language: obvious (correlation → instability is textbook)
- MI plain language: known since the 1990s (Hecht-Nielsen, weight space symmetry)
- Quantum plain language: known since the 1970s (quantum state tomography)

- [ ] **Step 3: Identify what the framework adds that plain language doesn't**

The framework's value must be in one of:
(a) Making a QUANTITATIVE prediction that plain language can't (e.g., the exact calibrated slope, the exact Jaccard level)
(b) Connecting domains that plain language doesn't connect (e.g., predicting that SHAP instability and quantum measurement share the same formula)
(c) Providing a TOOL that plain language doesn't provide (e.g., the SAGE algorithm)

**PASS:** At least one of (a), (b), or (c) is real and non-trivial
**FAIL:** All three are either trivial or achievable without the framework

---

## Execution Order

All tracks run in parallel. Each track is independent.

```
Track A (SAGE): Tasks 1-4 — ~1 hour
Track B (MI): Tasks 5-7 — ~30 min
Track C (Quantum): Tasks 8-9 — ~30 min (theoretical, no code)
Track D (Cross-cutting): Task 10 — ~15 min (analytical)
```
