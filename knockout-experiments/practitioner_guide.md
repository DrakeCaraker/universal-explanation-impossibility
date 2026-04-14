# How to Assess SHAP Feature Ranking Reliability

A practitioner's guide from the Universal Explanation Impossibility framework.

---

## The Problem

SHAP feature rankings can change when you retrain the model. Some comparisons
are reliable; others are coin flips. How do you know which is which?

---

## The Method

### Step 1: Train an ensemble of models

Train M >= 30 models on bootstrap resamples of your training data.

- Same hyperparameters, different random seeds and bootstrap samples.
- All models should achieve comparable performance (within 2% of best).

### Step 2: Compute importance estimates

For each model, compute mean |SHAP| per feature on a **held-out** test set
(the same test set for all models).

### Step 3: Compute pairwise SNR

For each feature pair (j, k), compute across the M models:

    Delta = mean(importance_j - importance_k)
    sigma = std(importance_j - importance_k)
    SNR   = |Delta| / sigma

### Step 4: Interpret the SNR

| SNR       | Flip Rate | Interpretation                              |
|-----------|-----------|---------------------------------------------|
| > 3.0     | < 1%      | Very reliable. Trust this ranking.          |
| 2.0 - 3.0 | 1% - 5%  | Reliable.                                   |
| 1.0 - 2.0 | 5% - 30% | Moderate. Report with uncertainty.          |
| 0.5 - 1.0 | 30% - 50%| Unreliable. Do not report as a finding.     |
| < 0.5     | ~ 50%     | Coin flip. Features are interchangeable.    |

---

## Example

For a clinical risk model with features {age, BMI, cholesterol, systolic_BP,
diastolic_BP}:

| Comparison                   | SNR  | Verdict                                     |
|------------------------------|------|---------------------------------------------|
| age vs BMI                   | 4.2  | Reliable (age consistently ranks higher)    |
| age vs cholesterol           | 2.8  | Reliable                                    |
| systolic_BP vs diastolic_BP  | 0.3  | Coin flip (interchangeable in importance)   |

**What to report:** "Age is the most important predictor, followed by BMI and
cholesterol. Systolic and diastolic blood pressure contribute similarly, but
their relative ranking is not stable across retrains."

---

## Reference

**Exact formula:**

    flip_rate = 2 * Phi(Delta/sigma) * Phi(-Delta/sigma)

where Phi is the standard normal CDF.

This reduces to:
- flip = 0.5 when Delta = 0 (exact symmetry between features)
- flip ~ 0 when |Delta| >> sigma (strong signal)

Derived from the Universal Explanation Impossibility framework.
Lean-verified in GaussianFlipRate.lean.
