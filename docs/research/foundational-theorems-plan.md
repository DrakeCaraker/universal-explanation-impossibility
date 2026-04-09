# Foundational Theorems Plan: F1-F5

> Extending the NeurIPS 2026 paper to a definitive JMLR reference.
> Each theorem has: precise statement, proof strategy, failure points,
> Lean plan, experiments, and paper integration.

---

## Dependency Graph

```
F3 (FIM impossibility) ──→ F1 (testable condition)
                              ↓
F2 (DASH optimality) ←── F1 (provides the boundary)
                              ↓
F5 (efficient diagnostic) ←── F1 (specifies what to test)

F4 (rate-distortion) is independent of F1-F3-F5.
F2 depends on formalizing the variance bound (Lean placeholder).
```

**Optimal proving order:**
1. **F3** first (shortest, unlocks F1)
2. **F2** in parallel (independent of F3, uses existing supplement results)
3. **F1** after F3 (builds on FIM)
4. **F5** after F1 (operationalizes F1)
5. **F4** whenever (independent, conjectural)

---

## F3: Direct FIM Impossibility

### Precise Statement

**Theorem F3 (FIM Impossibility).** Let $\mathcal{F}$ be a parametric
model class with parameter $\theta \in \Theta \subseteq \mathbb{R}^p$,
population risk $L(\theta) = \mathbb{E}[\ell(f_\theta(X), Y)]$, and
minimizer $\theta^* = \arg\min L(\theta)$. Let $I(\theta^*)$ denote the
Fisher information matrix. Define the attribution function
$\varphi_j(\theta) = |[\nabla_\theta L]_j|$ (or any attribution
monotone in $|\theta_j|$).

If features $j, k$ belong to the same collinear group (i.e., the FIM
has an eigenvalue $\lambda_- \leq \varepsilon / \sigma^2$ along the
direction $e_j - e_k$), then for any $\varepsilon > 0$, the Rashomon
set $R_\varepsilon = \{\theta : L(\theta) \leq L(\theta^*) + \varepsilon\}$
contains models $\theta_1, \theta_2$ with
$\varphi_j(\theta_1) > \varphi_k(\theta_1)$ and
$\varphi_k(\theta_2) > \varphi_j(\theta_2)$.

### Proof Strategy

**Step 1:** Characterize the Rashomon set geometry.

For a twice-differentiable loss, the Rashomon set is approximately
an ellipsoid:
$$R_\varepsilon \approx \{\theta : (\theta - \theta^*)^T I(\theta^*)
(\theta - \theta^*) \leq 2\varepsilon / \sigma^2\}$$

This follows from the second-order Taylor expansion of $L(\theta)$
around $\theta^*$.

*Existing result used:* Standard asymptotic statistics (van der Vaart 1998, Ch. 5).

*Failure point:* The approximation is only valid locally. For
$\varepsilon$ large, the ellipsoid is a poor approximation. **Fallback:**
State the theorem for $\varepsilon$ sufficiently small (local Rashomon set).

**Step 2:** Show the ellipsoid extends along $e_j - e_k$.

The FIM eigenvalue $\lambda_-$ along $e_j - e_k$ controls the
ellipsoid's extent in this direction. The semi-axis length is
$\sqrt{2\varepsilon / (\sigma^2 \lambda_-)}$.

When $\lambda_- \leq \varepsilon / \sigma^2$, the semi-axis length
is $\geq \sqrt{2}$, meaning the ellipsoid contains points with
$\theta_j - \theta_k$ varying by $\pm\sqrt{2}$ (or more) around
$\theta^*_j - \theta^*_k$.

*New lemma needed:* "Ellipsoid semi-axis bound" — the projection of
$R_\varepsilon$ onto $e_j - e_k$ has length $\geq \sqrt{2\varepsilon /
(\sigma^2 \lambda_-)}$. Proof: direct from the eigendecomposition.

**Step 3:** Construct the two models.

At $\theta^*$, by DGP symmetry, $\theta^*_j = \theta^*_k$ (or
$\theta^*_j \approx \theta^*_k$). The ellipsoid contains:
- $\theta_1 = \theta^* + \delta(e_j - e_k)$ with $\delta > 0$
  → $\varphi_j(\theta_1) > \varphi_k(\theta_1)$
- $\theta_2 = \theta^* - \delta(e_j - e_k)$ with $\delta > 0$
  → $\varphi_k(\theta_2) > \varphi_j(\theta_2)$

Both are in $R_\varepsilon$ by Step 2.

*Failure point:* Requires $\theta^*_j = \theta^*_k$ (DGP symmetry).
If features have different true effects, $\theta^*_j \neq \theta^*_k$,
and the construction needs $\delta$ large enough to overcome the gap.
**Fallback:** Add hypothesis: "features j, k have equal population
coefficients" (already the DGP symmetry assumption).

**Step 4:** Apply Theorem 1 (attribution impossibility).

The existence of $\theta_1, \theta_2$ with opposite orderings is exactly
the Rashomon property. By Theorem 1, no faithful + stable + complete
ranking exists. QED.

### New Definitions/Lemmas Needed

1. `FisherInfo (θ : Params) : Matrix P P ℝ` — the FIM at θ
2. `rashomon_set (ε : ℝ) : Set Params` — {θ : L(θ) ≤ L(θ*) + ε}
3. `ellipsoid_semi_axis` lemma — projection bound from eigenvalue
4. `fim_rashomon` theorem — low eigenvalue → Rashomon property

### Lean Formalization Plan

- **Formalizable now:** Steps 3-4 (constructing the two models, applying
  Theorem 1). These are algebraic/logical.
- **Needs Mathlib:** Step 1 (Taylor expansion of loss functions),
  Step 2 (eigendecomposition of real symmetric matrices).
  Mathlib HAS: `Matrix.IsHermitian.eigenvalues`, `InnerProductSpace`.
  Mathlib MISSING: connection between FIM and confidence ellipsoids.
- **Stay classical:** The Taylor approximation argument (Step 1).
  Formalizing the $O(\varepsilon^2)$ remainder would be excessive.

**New file:** `DASHImpossibility/FIMImpossibility.lean`

### Empirical Validation

No new experiments needed. The Fisher information section in the
supplement already demonstrates the FIM eigenvalue structure.
Could add: a plot of FIM eigenvalues vs ρ for the Breast Cancer
dataset features, showing which pairs have near-singular FIM.

### Paper Integration

**Main text (journal version):** New Section 3.4 "The Classical Statistics
Perspective" — 1 page. States F3, proof sketch, connects to non-identifiability.

**Narrative change:** The impossibility now has TWO independent proofs:
(1) Rashomon-based (model-agnostic), (2) FIM-based (classical statistics).
This makes the result more credible ("two roads to the same destination").

---

## F2: Optimal Attribution Theorem

### Precise Statement

**Theorem F2 (DASH Pareto Optimality).** Define the attribution
design space as pairs $(S, U)$ where:
- $S = \mathbb{E}[\rho_S(f, f')]$ is the expected Spearman correlation
  between attribution rankings from independent models (stability)
- $U = \mathbb{E}[\text{flip}(j,k)]$ is the expected within-group
  unfaithfulness (fraction of models where the ranking disagrees with
  the consensus)

For any attribution method $A$ mapping models to rankings:
1. If $A$ is faithful to each model: $U \geq 1/2$ per symmetric pair
   and $S \leq 1 - m^3/P^3$ (single-model bound).
2. If $A$ is the DASH consensus with ensemble size $M$:
   $S = 1 - O(1/M)$ for between-group pairs, and $U = 0$ for
   within-group pairs (ties).
3. **Pareto optimality:** There exists no method $A'$ with $S(A') > S(A)$
   and $U(A') \leq U(A)$ simultaneously for the DASH consensus.

### Proof Strategy

**Step 1:** Establish the lower bounds.

For faithful methods:
- $U \geq 1/2$: Already proved (Theorem S6). By DGP symmetry,
  $\Pr[\varphi_j > \varphi_k] = 1/2$, so any fixed ranking is wrong
  for half the models.
- $S \leq 1 - m^3/P^3$: Already proved (spearman_classical_bound axiom
  + empirical validation).

For DASH:
- $U = 0$ for within-group: Already proved (Corollary 1, consensus equity).
- $S = 1 - O(1/M)$ for between-group: Needs the variance bound.

*Key gap:* The variance bound. Currently `consensus_variance_decreases`
proves True (placeholder). Need:

**Lemma (DASH Variance Bound).** For i.i.d. models $f_1, ..., f_M$:
$$\text{Var}(\bar\varphi_j) = \text{Var}(\varphi_j) / M$$

*Proof:* Independence of models → variance of the mean is variance
divided by M. This is the standard i.i.d. mean variance formula.

*Lean formalization:* Requires `MeasureTheory.Probability.Variance`
(exists in Mathlib) + `ProbabilityTheory.IndepFun` (exists).
The gap is connecting the abstract Lean `Model` type to a measurable
space. Need: `axiom Model.measurable : MeasurableSpace Model`.

**Step 2:** Show the Pareto frontier.

The achievable set is:
- Single model (M=1): $(S, U) = (1-m^3/P^3, 1/2)$
- DASH(M): $(S, U) = (1-O(1/M), 0)$ for between-group
- M→∞: $(S, U) = (1, 0)$

These trace a curve parameterized by M. No point outside this curve
is achievable because:

- Any method with $U < 1/2$ for within-group pairs must aggregate
  across models (by Theorem S6, single-model faithfulness implies $U=1/2$).
  Aggregation over M models reduces variance by 1/M, so $S = 1-O(1/M)$.
- No aggregation method can improve beyond the i.i.d. rate (by
  the Cramér-Rao bound on the variance of any unbiased estimator
  of $E[\varphi_j]$).

*New lemma:* "Variance lower bound" — for any unbiased estimator of
$E[\varphi_j]$ based on M models, $\text{Var}(\hat\mu) \geq \text{Var}(\varphi_j)/M$.
This is the classical efficiency bound for the sample mean.

**Step 3:** Conclude optimality.

DASH(M) achieves the boundary: $S = 1-\text{Var}(\varphi_j)/(M \cdot
\text{gap}^2)$ where gap is the between-group attribution difference.
No method achieves higher $S$ for the same $M$. QED.

*Failure point:* The Pareto optimality claim assumes the only knob is
M (ensemble size). A method that uses a DIFFERENT aggregation (e.g.,
weighted average, median) could potentially achieve better $(S, U)$.
**Fallback:** Restrict to "unbiased linear aggregation methods" (which
includes DASH but excludes e.g., trimmed means). The optimality holds
within this class.

### Lean Formalization Plan

- **Formalizable:** The Pareto frontier structure (algebraic).
  The $U \geq 1/2$ bound (already essentially proved).
- **Needs:** `MeasurableSpace Model`, `IndepFun.variance_sum`.
  The variance bound is the key missing piece.
- **New file:** `DASHImpossibility/Optimality.lean`

### Empirical Validation

Existing data suffices. The synthetic DASH convergence (Figure 1c) and
Breast Cancer DASH convergence (supplement) already show the $(S, U)$
curve. Could add: a Pareto frontier plot with M on one axis and
$(S, U)$ on the other two.

### Paper Integration

**Main text (journal):** Expand §5 to include F2 as the main result.
Change framing from "DASH resolves the impossibility" to "DASH is the
provably optimal resolution."

---

## F1: Statistical Indistinguishability Characterization

### Precise Statement

**Theorem F1 (Rashomon Characterization).** For features $j, k$ in the
same collinear group under a Gaussian DGP with correlation $\rho$ and $n$
training samples, the Rashomon property holds if and only if:
$$\frac{|\mu_j - \mu_k|}{\sigma_{jk}/\sqrt{n}} < z_{\alpha/2}$$
where $\mu_j = E[\varphi_j]$, $\mu_k = E[\varphi_k]$,
$\sigma_{jk}^2 = \text{Var}(\varphi_j - \varphi_k)$, and $z_{\alpha/2}$
is the normal quantile at level $\alpha/2$.

Equivalently: **the impossibility applies precisely when the data cannot
distinguish the two features' expected attributions at significance
level $\alpha$.**

### Proof Strategy

**Step 1:** The "if" direction (low power → Rashomon).

If the test has low power, then $|\mu_j - \mu_k|$ is small relative to
the sampling noise $\sigma_{jk}/\sqrt{n}$. This means:
- With probability $\geq \alpha/2$, a random model has
  $\hat\varphi_j > \hat\varphi_k$ even when $\mu_j \leq \mu_k$.
- With probability $\geq \alpha/2$, the reverse occurs.

Both events have positive probability → models on both sides exist →
Rashomon property.

*Uses:* Theorem S5 (Rashomon inevitability) + the Gaussian tail bound.

**Step 2:** The "only if" direction (Rashomon → low power).

If the test has HIGH power ($|\mu_j - \mu_k| \gg \sigma_{jk}/\sqrt{n}$),
then with high probability all models agree on the ranking.
The Rashomon property requires models on BOTH sides — but when the
signal is strong, the fraction on the minority side is exponentially
small (by the Gaussian tail bound).

More precisely: $\Pr[\hat\varphi_j < \hat\varphi_k \mid \mu_j > \mu_k]
= \Phi(-|\mu_j-\mu_k|\sqrt{n}/\sigma_{jk}) \to 0$ as
$|\mu_j-\mu_k|\sqrt{n}/\sigma_{jk} \to \infty$.

*Failure point:* The "only if" is APPROXIMATE, not exact. For any finite
n, there's always a nonzero (possibly tiny) probability of reversal.
So the Rashomon property holds FORMALLY for any $\mu_j \neq \mu_k$
with finite n — it's just that the practical SEVERITY is negligible
when power is high.

**Fallback:** Restate as: "The Rashomon property is PRACTICALLY
significant (flip rate $\geq \delta$) iff the test power is $\leq 1-2\delta$."
This is an equivalence between practical significance and statistical
power.

**Step 3:** Connect to FIM (from F3).

The test statistic $|\mu_j-\mu_k|/(\sigma_{jk}/\sqrt{n})$ is related
to the FIM eigenvalue:
$$\text{power} = 1 - \Phi\left(z_{\alpha/2} -
\frac{|\mu_j-\mu_k|\sqrt{n}}{\sqrt{2/\lambda_-}}\right)$$

where $\lambda_- = (1-\rho)/\sigma^2$. As $\rho \to 1$, $\lambda_- \to 0$,
and the power drops even for large $|\mu_j - \mu_k|$.

### Lean Formalization Plan

- **Stay classical:** The hypothesis testing characterization involves
  Gaussian CDFs and power calculations. Not worth formalizing in Lean.
- **Formalizable:** The connection between FIM eigenvalue and Rashomon
  severity (Step 3). This is algebraic.

### Empirical Validation

**New experiment:** For the Breast Cancer dataset, compute the test
statistic $|\hat\mu_j - \hat\mu_k|/(\hat\sigma_{jk}/\sqrt{50})$ for
each feature pair. Plot test statistic vs. empirical flip rate. The
prediction: pairs with low test statistic have high flip rate (near 0.5),
pairs with high test statistic have low flip rate (near 0).

This would be the MOST COMPELLING figure in the journal paper: a single
plot showing the theory (F1) predicts exactly which feature pairs are
unstable.

### Paper Integration

**Main text (journal):** New Section 4.5 "When Does the Impossibility
Apply? A Statistical Test." States F1, gives the diagnostic formula,
shows the Breast Cancer validation figure.

**Narrative change:** The paper goes from "the impossibility holds when
ρ > 0" (theoretical condition) to "the impossibility holds when THIS
TEST FAILS" (empirical diagnostic).

---

## F5: Efficient Rashomon Diagnostic

### Precise Statement

**Theorem F5 (Split-Frequency Diagnostic).** For a trained gradient-boosted
ensemble with $T$ trees, define:
- $n_j(t) = 1$ if feature $j$ is used in tree $t$, 0 otherwise
- $\bar{n}_j = T^{-1}\sum_t n_j(t)$ (split frequency)

The test statistic:
$$Z_{jk} = \frac{|\bar{n}_j - \bar{n}_k|}
{\sqrt{(\hat{p}_j(1-\hat{p}_j) + \hat{p}_k(1-\hat{p}_k))/T}}$$

tests $H_0$: "features $j, k$ have equal expected split frequency"
(the null under which the Rashomon property holds).

When $Z_{jk} < z_{\alpha/2}$: the Rashomon property is supported →
rankings are unreliable → use DASH.

When $Z_{jk} > z_{\alpha/2}$: the ranking is reliable for this pair →
single-model SHAP is sufficient.

**Computational cost:** $O(T)$ per feature pair, computed from a single
trained model. No retraining needed.

### Proof Strategy

**Step 1:** Establish approximate exchangeability.

For XGBoost with `subsample < 1` or `colsample_bytree < 1`, the trees
are trained on different data subsets, making the indicators $n_j(t)$
approximately independent across $t$.

For full XGBoost (no sub-sampling), the trees are NOT independent (each
fits the residual from the previous). But the split indicators are still
approximately exchangeable under the boosting steady state (after a
burn-in of $O(1/\eta)$ trees).

*New lemma:* "Steady-state exchangeability" — for $t > T_0(\eta)$, the
split indicators $\{n_j(t)\}_{t > T_0}$ are approximately exchangeable
with marginal $p_j = E[n_j(t)]$.

*Failure point:* The exchangeability is approximate, not exact. The
steady-state assumption requires the boosting process to mix, which
depends on the learning rate η. For η=1 (full fitting), the process
doesn't mix and the indicators are correlated.

**Fallback:** Restrict the theorem to `subsample < 1` or
`colsample_bytree < 1` settings where tree-level independence holds.

**Step 2:** Standard two-sample proportion test.

Under exchangeability + the null $p_j = p_k$, the test statistic
$Z_{jk}$ is asymptotically $N(0,1)$. The power calculation follows
from the standard formula.

**Step 3:** Connect to F1.

The split-frequency test (F5) is a CHEAP APPROXIMATION to the
attribution test (F1). F1 tests $E[\varphi_j] = E[\varphi_k]$ using M
models. F5 tests $p_j = p_k$ using split counts from ONE model.

Under proportionality ($\varphi_j = c \cdot n_j$), these are equivalent.

### Lean Formalization Plan

- **Stay classical:** The exchangeability argument and the CLT-based
  test are standard probability theory, not worth formalizing.
- **Formalizable:** The connection between the split-frequency test
  and the Rashomon property (if $Z_{jk}$ is small → the Rashomon
  property is supported → Theorem 1 applies). This is logical.

### Empirical Validation

**New experiment:** For each Breast Cancer model (50 seeds):
1. Compute $Z_{jk}$ for all feature pairs from a SINGLE model's split
   frequencies
2. Compare to the empirical flip rate (from 50 models)
3. Plot: $Z_{jk}$ vs. flip rate

**Prediction:** $Z_{jk}$ is inversely correlated with flip rate. Pairs
with $Z_{jk} < 1.96$ have high instability; pairs with $Z_{jk} > 1.96$
are stable.

This is the "VIF for SHAP instability" — a single-model diagnostic.

### Paper Integration

**Main text (journal):** New Section 6.3 "A Single-Model Diagnostic."
Practical algorithm: "Compute $Z_{jk}$ from split frequencies. If
$Z_{jk} < 1.96$ for any within-group pair, use DASH."

---

## F4: Rate-Distortion Bound for α(d)

### Precise Statement

**Conjecture F4.** For a depth-$d$ gradient-boosted tree on Gaussian
features with correlation $\rho$, the effective signal capture fraction
satisfies:
$$\alpha(d) = 1 - 2^{-2R(d)}$$
where $R(d) = d \cdot C$ bits is the effective rate of a depth-$d$ tree
and $C$ depends on the signal-to-noise ratio.

For $d=1$: $\alpha(1) = 1 - 2^{-2C}$. If $C = \frac{1}{2}\log_2(\pi/2)$,
then $\alpha(1) = 2/\pi$.

### Proof Strategy (SPECULATIVE)

**Step 1:** Model each tree level as a quantization step.

A depth-$d$ tree partitions the input space into $2^d$ regions.
For a Gaussian source, the optimal $2^d$-level quantizer achieves
distortion $D(d) = \sigma^2 \cdot 2^{-2R}$ where $R = d \cdot C$
is the rate (Lloyd 1982).

The variance captured is $\sigma^2 - D(d) = \sigma^2(1 - 2^{-2R}) = \alpha(d) \cdot \sigma^2$.

**Step 2:** Connect to the boosting dynamics.

In each boosting round, the tree captures $\alpha(d)$ of the selected
feature's residual signal. The steady-state split allocation follows
the same Markov chain as the depth-1 case, with $\alpha$ replaced by
$\alpha(d)$.

*Major failure point:* This step assumes each tree acts as an independent
quantizer of the current residual. In reality, trees are deterministic
functions of the data — they don't optimize the rate-distortion tradeoff
independently. The connection between tree depth and quantizer rate is
heuristic, not proven.

**Step 3:** Verify against the depth table.

| Depth | Predicted α(d) | Empirical α |
|-------|----------------|-------------|
| 1 | 2/π ≈ 0.637 | 0.60 |
| 3 | 1-2^{-6C} | 0.30 (!) |
| 6 | 1-2^{-12C} | 0.60 |

The depth=3 empirical α=0.30 CONTRADICTS the prediction that α increases
with depth. The rate-distortion model predicts α(3) > α(1), but the
data shows α(3) < α(1).

*Root cause:* At depth=3, multi-feature trees distribute splits, which
reduces the FIRST-MOVER's signal capture even though each tree captures
more TOTAL signal. The rate-distortion model doesn't account for the
multi-feature tree structure.

**Assessment: F4 in its current form is LIKELY FALSE for d > 1.** The
rate-distortion connection works for stumps (d=1) but breaks for deeper
trees because the quantization model doesn't capture the multi-feature
split distribution effect.

**Fallback:** Restrict F4 to stumps only: "For stumps (d=1) on Gaussian
features, α(1) = 2/π." This is already Proposition S1 in the supplement.
A deeper analysis of α(d) for d > 1 requires a tree-structure-aware
model, which is a different (harder) problem.

### Recommendation: DEPRIORITIZE F4.

The rate-distortion connection is elegant for stumps but doesn't
generalize. The depth×ρ table in the supplement already provides the
empirical data. A proper theory of α(d) would require modeling the
within-tree feature allocation (how splits are distributed across
features at each depth level), which is a substantial research problem
beyond the scope of a single theorem.

---

## Scope Assessment

### Minimum Viable Foundational Paper (JMLR)

**F2 + F3:** DASH optimality + FIM impossibility.

- F3 gives the classical statistics proof ("one theorem, no jargon")
- F2 gives the "DASH is THE method" claim
- Together they transform the paper from "here's an impossibility and
  a fix" to "here's the impossibility from first principles and the
  provably optimal resolution"

Estimated addition: ~8 pages to the current 9+9 = 18 pages.
Total JMLR paper: ~26 pages.

### Full Foundational Paper

**F1 + F2 + F3 + F5:** Testable condition + optimal method + FIM proof
+ efficient diagnostic.

- F1 gives practitioners the diagnostic condition
- F5 gives the O(T) single-model test
- Together with F2 + F3, this is a COMPLETE treatment: theory (F3),
  characterization (F1), optimality (F2), practice (F5)

Estimated addition: ~15 pages.
Total JMLR paper: ~33 pages. This is large but within JMLR norms for
comprehensive treatments.

### Splitting into Two Papers

**Paper A (theory):** F3 + F2 + parts of F1. "The Attribution
Impossibility: Optimality of Ensemble Explanations." JMLR.

**Paper B (practice):** F1 + F5 + experiments. "Testing for Attribution
Instability: A Statistical Diagnostic." TMLR or JMLR.

This split is clean: Paper A is for theorists, Paper B is for
practitioners. Both cite the NeurIPS paper for the core impossibility.

### Recommendation

**One paper (JMLR) with F2 + F3 + F1 + F5.** The four theorems form
a coherent package. Splitting dilutes impact. JMLR has no page limit.
Drop F4 (the rate-distortion conjecture is broken for d > 1).
