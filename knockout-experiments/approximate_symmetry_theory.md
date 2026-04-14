# Approximate Symmetry Theory

## From Exact Noether Counting to the Gaussian Flip Rate Formula

This document derives the continuous Gaussian flip rate formula and shows how
the discrete Noether counting theorem is its special case under exact symmetry.

---

## 1. Setup

Let **M** independently trained models each produce feature importance estimates
for features j = 1, ..., p. For model m:

    imp_j^(m) ~ N(beta_j, sigma_j^2)

For any feature pair (j, k), define the importance difference:

    d^(m) = imp_j^(m) - imp_k^(m) ~ N(Delta_jk, sigma_Delta^2)

where:
- Delta_jk = beta_j - beta_k  (the true importance gap)
- sigma_Delta^2 = Var(d^(m))   (variance of the difference across retrains)

The **signal-to-noise ratio** is:

    SNR = |Delta_jk| / sigma_Delta

---

## 2. The Exact Flip Rate Formula

Two models m1, m2 **disagree** on the ranking of j vs k when d^(m1) and d^(m2)
have opposite signs. Since the d^(m) are i.i.d.:

    P(flip) = P(d^(m1) > 0) * P(d^(m2) < 0) + P(d^(m1) < 0) * P(d^(m2) > 0)

Let p = P(d > 0) = Phi(Delta / sigma_Delta), where Phi is the standard normal CDF.

Then:

    P(flip) = 2 * p * (1 - p)

Equivalently:

    **P(flip) = 2 * Phi(Delta/sigma) * Phi(-Delta/sigma)**

This is the **exact** pairwise flip rate under the Gaussian model.

---

## 3. The Phi(-|Delta|/(sigma*sqrt(2))) Approximation

An alternative formula sometimes seen in the literature is:

    P_approx(flip) = Phi(-|Delta| / (sigma * sqrt(2)))

This is an **approximation** of 2p(1-p). The two formulas agree at the
exact-symmetry point and diverge in the moderate-SNR regime:

| Regime           | p    | 2p(1-p) (exact) | Phi(-\|Delta\|/(sigma*sqrt(2))) (approx) |
|------------------|------|------------------|------------------------------------------|
| Exact symmetry   | 0.50 | 0.500            | 0.500                                    |
| Mild asymmetry   | 0.40 | 0.480            | 0.429                                    |
| Moderate signal  | 0.30 | 0.420            | 0.355                                    |
| Strong signal    | 0.10 | 0.180            | 0.130                                    |
| Very strong      | 0.01 | 0.020            | 0.010                                    |

**Key differences:**
- They agree exactly when Delta = 0 (both give 0.500).
- They agree asymptotically when |Delta| >> sigma (both approach 0).
- The approximation **underestimates** flip rate in the moderate-SNR regime.
- Maximum relative error occurs near SNR ~ 0.5-1.5.

**Which our experiments use:** Our codebase uses the exact formula 2p(1-p) in
the theoretical predictions and the Lean verification (GaussianFlipRate.lean).
The Phi(-|Delta|/(sigma*sqrt(2))) form appears only in some early derivation
notes and should be understood as an approximation.

**Recommendation:** Always use the exact formula 2*Phi(Delta/sigma)*Phi(-Delta/sigma).
The approximation is adequate only when SNR > 3 (where both formulas give < 1%).

---

## 4. Connection to Noether Counting

The discrete Noether counting theorem states that for g equivalence classes of
features under a symmetry group, there are exactly g(g-1)/2 stable pairwise
queries (between-group comparisons) and the remaining within-group comparisons
are unstable.

This is the **special case** of the Gaussian flip formula under three conditions:

### Exact symmetry (Delta = 0)

For features within the same equivalence class, beta_j = beta_k exactly, so
Delta_jk = 0. This gives:

    p = Phi(0) = 0.5
    P(flip) = 2 * 0.5 * 0.5 = 0.500

This is the **50% within-group flip rate** from Noether counting. Any pairwise
comparison within a group is a coin flip.

### Broken symmetry (|Delta| >> sigma)

For features in different equivalence classes, the importance gap is large
relative to noise:

    p -> 0 or p -> 1
    P(flip) -> 0

This gives the **stable between-group comparisons**. There are exactly g(g-1)/2
such pairs.

### The bimodal gap

The gap between "stable" (flip ~ 0) and "unstable" (flip ~ 0.5) is sharp in
the Noether case because the SNR distribution is **bimodal**:
- Within-group pairs: SNR = 0 (exact symmetry)
- Between-group pairs: SNR >> 1 (strong signal)

There is nothing in between, so a clear threshold separates stable from unstable.

---

## 5. What Happens on Real Data

On real data, the three Noether conditions break down gracefully:

| Noether (ideal)                        | Real data                                      |
|----------------------------------------|------------------------------------------------|
| Within-group: Delta = 0 exactly        | Small but nonzero Delta -> flip < 0.5          |
| Between-group: \|Delta\| >> sigma      | Moderate Delta -> flip > 0 but small            |
| Sharp threshold (bimodal SNR)          | Blurred threshold (continuous SNR distribution) |

Consequences:
1. **Within-group features** have small but nonzero Delta, so flip rates are
   high (30-50%) but not exactly 50%.
2. **Between-group features** have moderate Delta, so flip rates are low (1-10%)
   but not exactly 0%.
3. **The threshold is blurred** because SNR is continuous. There is no single
   cutoff that cleanly separates "stable" from "unstable."

---

## 6. The Gaussian Formula as Continuous Generalization

The Gaussian flip formula:

    P(flip) = 2 * Phi(Delta/sigma) * Phi(-Delta/sigma)

handles **both** regimes with one equation:

- **Exact symmetry limit:** Set Delta = 0, recover flip = 0.5 (Noether within-group).
- **Strong signal limit:** Let |Delta|/sigma -> infinity, recover flip -> 0 (Noether between-group).
- **Intermediate regime:** Continuously interpolates between these extremes.

This means the Noether counting theorem is not an alternative to the Gaussian
formula -- it is the **boundary case** recovered when the SNR distribution
degenerates to two point masses at 0 and infinity.

The practical value of the Gaussian formula is that it provides **quantitative**
flip rate predictions for any SNR, rather than the binary stable/unstable
classification of Noether counting. This is essential for real-data applications
where SNR values are continuous.

---

## Summary

| Concept                  | Formula                                           | Domain          |
|--------------------------|---------------------------------------------------|-----------------|
| Exact flip rate          | 2 * Phi(Delta/sigma) * Phi(-Delta/sigma)          | All SNR         |
| Approximate flip rate    | Phi(-\|Delta\| / (sigma * sqrt(2)))               | SNR > 3 only    |
| Noether within-group     | flip = 0.5 (special case: Delta = 0)              | Exact symmetry  |
| Noether between-group    | flip = 0 (special case: \|Delta\| >> sigma)       | Strong signal   |
| Noether stable count     | g(g-1)/2 (special case: bimodal SNR)              | Discrete groups |
