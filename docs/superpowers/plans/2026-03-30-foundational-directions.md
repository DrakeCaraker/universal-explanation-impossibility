# Foundational Directions: Rashomon Inevitability + Relaxation Characterization

> Research plan for Directions 4 and 5 from the Attribution Impossibility program.
> These are the paths from "strong contribution" to "foundational result."

**Goal:** (4) Prove the Rashomon property is inevitable under collinearity for any symmetric training algorithm, making the impossibility inescapable. (5) Characterize all three relaxation paths, mapping the full landscape of what's achievable.

**Status:** Vetted. Research integrated (Rashomon set literature + relaxation precedents).

**Literature confirmation:** No existing theorem proves Rashomon inevitability under collinearity for general model classes (R1 is novel). No published "optimal unfaithfulness" or stability-equity Pareto frontier exists in XAI (F1-F3 and S1-S3 fill genuine gaps). Pattanaik & Fishburn (1973) showed Arrow weakens with partial orders — direct precedent for our "drop completeness" path.

**Key references to add:** Fisher et al. (2019) MCR framework, Donoho-Elad (2003) mutual coherence, Zhao-Yu (2006) irrepresentable condition, Pattanaik (1973) / Fishburn (1973) partial order relaxation of Arrow, Semenova-Rudin-Parr (2022) Rashomon ratio.

---

## Direction 4: Rashomon Inevitability

### The Gap

The Attribution Impossibility (Theorem 1) holds for ANY model class satisfying the Rashomon property. We currently VERIFY Rashomon for GBDT, Lasso, and NNs individually. A universality result would show Rashomon is inevitable for essentially any ML pipeline under collinearity — closing the escape hatch.

### The Key Insight

The Rashomon property follows from two assumptions:
1. **DGP symmetry:** Permuting features within a collinear group leaves the population loss invariant
2. **Algorithmic symmetry:** The training algorithm doesn't have a deterministic preference for one feature over another

Under these two symmetries, different random seeds produce models with different feature dominance patterns → the Rashomon property.

### Precise Theorems to Prove

**Theorem R1 (Rashomon from DGP + Model Class Symmetry):**

*Statement:* Let (X, Y) be a DGP where X_j, X_k have equal marginal relationships with Y (DGP symmetry within group). Let F be a model class closed under within-group feature permutation (i.e., if f ∈ F, then the model f ∘ π obtained by permuting features j ↔ k is also in F). Then for any model f ∈ F with φ_j(f) ≠ φ_k(f), the permuted model f' = f ∘ π satisfies L(f') = L(f) and φ_k(f') = φ_j(f) > φ_k(f) = φ_j(f').

*Proof sketch:* By DGP symmetry, the permuted model has the same loss. By construction, its attributions are the permuted attributions. Since φ_j(f) ≠ φ_k(f), we get opposite orderings.

*Machinery:* Group actions on feature spaces, permutation equivariance.

*Assessment:* **Provable.** This is essentially a symmetry argument. The key assumption (model class closure under permutation) is mild — it's satisfied by any universal approximator or any class without built-in feature ordering.

**Theorem R2 (Attribution Non-Degeneracy):**

*Statement:* For any stochastic training algorithm A that is symmetric (A applied to permuted data produces the permuted model in distribution), trained on n samples from a DGP with ρ > 0, we have Pr[φ_j(f) ≠ φ_k(f)] = 1 for j, k in the same group.

*Proof sketch:* With finite samples, the empirical correlation is not exactly ρ (it's ρ + O(1/√n) noise). This breaks the exact symmetry, so the training algorithm generically distinguishes the features. The event φ_j = φ_k has measure zero under any continuous perturbation of the data.

*Machinery:* Measure theory, transversality / genericity arguments.

*Assessment:* **Provable for most model classes.** May require mild regularity conditions on the training algorithm (e.g., continuous dependence on data). The "measure zero" argument needs the attribution function to be continuous in the data, which holds for differentiable models.

**Theorem R3 (Rashomon Inevitability — the main result):**

*Statement:* Let A be a stochastic, symmetric training algorithm for a model class F closed under within-group permutation. For any DGP with ρ > 0, the Rashomon property holds: for any j, k in the same collinear group, there exist models f, f' in the support of A such that φ_j(f) > φ_k(f) and φ_k(f') > φ_j(f').

*Proof:* Combine R1 (symmetry gives the permuted model) and R2 (non-degeneracy ensures φ_j ≠ φ_k). By algorithmic symmetry, Pr[φ_j > φ_k] = Pr[φ_k > φ_j] = 1/2. Both events have positive probability → models on both sides exist.

*Machinery:* R1 + R2 + symmetry of the training distribution.

*Assessment:* **Provable.** This is a clean composition of R1 and R2. The key contribution is the formalization and the explicit statement that the impossibility is inescapable for symmetric algorithms.

### What This Means

With R3, the Attribution Impossibility becomes:

> For any stochastic, symmetric training algorithm on collinear features (ρ > 0), no faithful, stable, complete feature ranking exists.

The only escape is to use an algorithm that is NOT symmetric — i.e., one that deterministically prefers certain features. But such algorithms are explicitly biased, which is exactly what attribution methods are supposed to detect.

### Minimum Viable Theorem

**R1 alone** is already publishable and valuable. It says: model class symmetry + DGP symmetry → Rashomon. This is a 1-page argument that can be a theorem in the NeurIPS paper (Section 3, after the iterative optimizer results).

**R3** is the full result and merits a standalone paper or a major extension.

### Feasibility for Lean Formalization

- **R1:** Requires defining group actions on feature spaces and permutation equivariance. Mathlib has `Equiv.Perm` and group action infrastructure. **Feasible** (2-4 weeks).
- **R2:** Requires measure theory and genericity arguments. Harder in Lean — the "measure zero" argument needs `MeasureTheory`. **Feasible but hard** (4-8 weeks).
- **R3:** Composition of R1 + R2. Feasible if R1 and R2 are done.

---

## Direction 5: Relaxation Characterization

### The Gap

We prove three things can't coexist: faithfulness, stability, completeness. We show dropping completeness works (DASH). But what about the other two relaxations? A complete characterization maps the entire design space for attribution methods.

### The Three Relaxation Paths

#### Path A: Drop Completeness (DONE)

**What you get:** Partial orders where symmetric features are tied/incomparable.

**Formal result:** DASH consensus attributions are exactly equal for symmetric features in balanced ensembles (Corollary, proved in Lean).

**Practical meaning:** Use ensemble averaging. Report confidence intervals. Don't claim one symmetric feature is more important than another.

**Connection to literature:** Laberge et al. (2023) propose partial orders from Rashomon sets — our formalization gives the theoretical justification.

#### Path B: Drop Stability — The Equity-Stability Tradeoff

**What you get:** Single-model explanations that are faithful and complete, but different for each model.

**Precise theorem to prove:**

**Theorem S1 (Equity-Stability Tradeoff):**

*Statement:* For any single model from a GBDT with correlation ρ:
- The equity violation (attribution ratio) is exactly 1/(1-ρ²)
- The stability violation (expected Spearman over random seeds) is bounded by 1 - m³/P³
- These are NOT tunable — they're fixed by ρ and the architecture

*Proof:* Direct from existing results (attribution_ratio + spearman_bound).

*This is essentially already proved.* The contribution is the FRAMING: the equity-stability pair (1/(1-ρ²), 1-m³/P³) is a FIXED POINT, not a tradeoff curve. You can't improve one by sacrificing the other within a single model class.

**Theorem S2 (Architecture Comparison):**

*Statement:* The equity-stability pair differs by architecture:
- GBDT: (1/(1-ρ²), 1-m³/P³) — severe and fixed
- Lasso: (∞, 1-m³/P³) — maximally inequitable
- Random Forest: (1+O(1/√T), 1-O(m³/(T·P³))) — both improve with T
- DASH(M): (1, 1-O(1/M)) — converges to perfect

*Proof:* Compile existing results + RF analysis + DASH corollary.

*Assessment:* **Mostly proved.** The RF bound needs formalization. The DASH stability bound (O(1/M)) is stated but not proved (variance bound placeholder).

**Theorem S3 (Pareto Frontier):**

*Statement:* There is no model class that achieves equity ratio < 1/(1-ρ²) and Spearman > 1-m³/P³ simultaneously (for sequential methods). The Pareto frontier for sequential methods is a single point. The only way to move along the frontier is to change architecture class (sequential → parallel → ensemble).

*Proof:* For sequential methods, the ratio is determined by the first-mover advantage, which is fixed by the Gaussian conditioning argument. Within a class, there's no tuning knob.

*Assessment:* **Provable** for sequential methods under our axioms. The "no tuning knob" claim follows from the axioms being equalities (not inequalities).

#### Path C: Drop Faithfulness — The Price of Stability

**What you get:** A stable, complete ranking that doesn't reflect any specific model's attributions.

**Precise theorems to prove:**

**Theorem F1 (Unfaithfulness Lower Bound):**

*Statement:* Any stable, complete ranking R has unfaithfulness ≥ 1/2 per symmetric pair: for any j, k in the same group, Pr_f[R ranks j > k but φ_k(f) > φ_j(f)] ≥ 1/2 where the probability is over the symmetric distribution of models.

*Proof:* By DGP symmetry, the distribution of (φ_j(f), φ_k(f)) is invariant under j ↔ k swap. Any stable ranking must fix j > k or k > j. By symmetry, whichever it picks is wrong for exactly half the models.

*Machinery:* Symmetry of the model distribution, basic probability.

*Assessment:* **Provable.** This is a simple symmetry argument. The key insight: any deterministic ranking of symmetric features is unfaithful for half the model population.

**Theorem F2 (Optimal Unfaithful Ranking):**

*Statement:* The population-level ranking R*(j,k) = 𝟙[E[φ_j] > E[φ_k]] minimizes expected unfaithfulness over the model distribution. For symmetric features, E[φ_j] = E[φ_k], so R* assigns a tie — reducing to Path A (drop completeness). Any completion of R* (breaking ties) has unfaithfulness exactly 1/2 per symmetric pair.

*Proof:* The optimal stable ranking is the Bayes-optimal decision: rank according to E[φ_j - φ_k]. For symmetric features, this expectation is 0, so no direction is preferred.

*Assessment:* **Provable.** Shows that dropping faithfulness optimally REDUCES to dropping completeness — the paths converge.

**Theorem F3 (Convergence of Relaxation Paths):**

*Statement:* The optimal way to drop faithfulness is to use population-level attributions, which are incomplete for symmetric features. The optimal way to drop completeness is DASH, which produces population-level attributions. Therefore: the "drop faithfulness" and "drop completeness" relaxation paths converge to the same solution (population-level attributions / ensemble averaging).

*Proof:* Combine F2 with the DASH equity corollary.

*Assessment:* **Provable and significant.** This means there are really only TWO distinct relaxation paths, not three:
1. Accept instability (single-model, faithful, complete) — Path B
2. Use population-level attributions (ensemble, stable, faithful for the ensemble) — Paths A and C converge

### The Complete Characterization

| Relaxation | What you give up | What you get | Optimal method | Achievable bound |
|------------|-----------------|-------------|----------------|-----------------|
| Drop completeness | Total order | Faithful + stable partial order | DASH | Equity = 1, ties for symmetric features |
| Drop stability | Reproducibility | Faithful + complete order per model | Any single model | Ratio = 1/(1-ρ²) for GBDT, → 1 for RF |
| Drop faithfulness | Model-specificity | Stable + complete (but half-wrong) | Population-level → reduces to DASH | Unfaithfulness = 1/2 per symmetric pair |

**Key insight: Paths A and C converge.** This simplifies the landscape from a trilemma to a DILEMMA: single-model (faithful but unstable) vs. population-level (stable but ties/unfaithful for symmetric features).

### Minimum Viable Theorems

For a **section in the NeurIPS paper** (if space allows):
- F1 (unfaithfulness lower bound) — 1 paragraph
- The convergence observation (F3) — 1 paragraph

For a **follow-up paper:**
- S1-S3 (full equity-stability tradeoff characterization)
- F1-F3 (full unfaithfulness characterization)
- The dilemma reduction (trilemma → dilemma)

---

## Phased Research Plan

### Phase 1: Quick wins for the NeurIPS paper (1-2 weeks)

**Goal:** Add R1 and F1 to the current paper if space allows.

- **R1 (Rashomon from symmetry):** 1 theorem + short proof. Can be added to Section 3 as a remark or short subsection. Needs: formal statement of "model class closed under permutation" and the symmetry argument.

- **F1 (unfaithfulness lower bound):** 1 theorem. Can be added to Section 5 (Discussion) or as a remark. Pure symmetry argument, no new machinery.

- **The dilemma observation (F3):** Can be added as 1-2 sentences in Discussion.

*These three additions strengthen the paper significantly with minimal page cost.*

### Phase 2: Lean formalization of R1 (2-4 weeks)

**Goal:** Formalize R1 in Lean, reducing axiom dependencies further.

- Define `FeaturePermutation` (within-group permutation of features)
- Define `PermutationClosed` (model class closed under permutation)
- Prove: DGP symmetry + permutation closure → Rashomon property
- This replaces `firstMover_surjective` with a MORE GENERAL condition

*Deliverable: updated Lean formalization with stronger generality claim.*

### Phase 3: Full Rashomon inevitability (4-8 weeks)

**Goal:** Prove R2 and R3 (classically, possibly in Lean).

- R2 requires measure-theoretic arguments about generic non-degeneracy
- R3 composes R1 + R2
- May be better as a classical proof with Lean formalization deferred

*Deliverable: standalone theorem for the follow-up paper.*

### Phase 4: Relaxation characterization (6-12 weeks)

**Goal:** Prove S1-S3 and F1-F3, establishing the full design space.

- S1-S2 are mostly compiled from existing results (needs framing)
- S3 (Pareto frontier) needs the "no tuning knob" argument
- F1-F2 are new but straightforward symmetry arguments
- F3 (convergence) is the key insight connecting the paths

*Deliverable: follow-up paper on "The Attribution Design Space" or similar.*

### Phase 5: Lean formalization of the characterization (8-16 weeks)

**Goal:** Machine-verify the key characterization theorems.

- F1 and F3 are formalizable with existing infrastructure
- S3 may require new axioms about architecture families
- Full characterization in Lean would be a landmark

*Deliverable: extended formalization, possibly a second Lean artifact paper.*

---

## Impact Assessment

| Result | Scope | Impact | Feasibility |
|--------|-------|--------|-------------|
| R1 (Rashomon from symmetry) | Section in current paper | HIGH — makes impossibility inescapable | Easy (1 week) |
| F1 (unfaithfulness bound) | Remark in current paper | MEDIUM — completes the picture | Easy (1 week) |
| F3 (path convergence) | Observation in current paper | HIGH — simplifies trilemma to dilemma | Easy (1 day) |
| R3 (full Rashomon inevitability) | Standalone result | VERY HIGH — universality | Medium (2 months) |
| Full characterization (S1-S3, F1-F3) | Follow-up paper | VERY HIGH — the design space map | Medium-hard (3-4 months) |
| Lean formalization of all | Extended artifact | HIGH — credibility | Hard (4-6 months) |

**Recommendation:** Do Phase 1 immediately (before NeurIPS submission). Phases 2-3 for a workshop paper or ICML follow-up. Phases 4-5 for the characterization paper.
