# From SHAP Instability to Representation Theory

*How a practical ML problem leads to orbit averaging, the Reynolds operator, and a universal information-loss formula*

---

## Starting Point: A Bug That Isn't a Bug

You train an XGBoost model to predict breast cancer diagnosis. You run SHAP. The top three features are:

```
Model A (seed=42):  worst_concave_points > worst_perimeter > worst_radius
Model B (seed=43):  worst_perimeter > worst_concave_points > worst_radius
```

Both models have identical accuracy (94.7%). The predictions agree on every patient. But the explanations disagree on which feature matters most.

Your instinct: something is wrong with SHAP. But it's not SHAP. The three features have pairwise correlations above 0.95. They carry nearly identical information about the outcome. Asking "which one matters most" is like asking "which leg of a tripod is most important" — the question has no stable answer because the legs are interchangeable.

The mathematical structure hiding here: the three correlated features form a **symmetry group**. Permuting them doesn't change the model's predictions (because the information is redundant), but it does change the SHAP rankings. The instability IS the symmetry.

---

## The Symmetry Group of an Explanation Problem

### Where the group comes from

Given a trained model f and a feature pair (j, k) with correlation ρ ≈ 1, retraining produces a new model f' that may swap the roles of j and k. The set of "equivalent retrainings" forms a group acting on the model space.

For two perfectly correlated features: **G = S₂ = ℤ/2ℤ** (swap or don't swap). For a cluster of k correlated features: **G = Sₖ** (all permutations). The group acts on the SHAP attribution vector by permuting coordinates within the cluster.

Concretely: if features {3, 7, 12} form a correlation group with |ρ| > 0.8, then G = S₃ acts on the SHAP vector φ = (φ₁, ..., φ_P) by permuting the entries (φ₃, φ₇, φ₁₂). Models related by this permutation have the same accuracy but different attributions.

### What the group action means for explanations

An explanation map E assigns each model a SHAP vector. E is:
- **Stable** (G-invariant) if E(g·f) = E(f) for all g ∈ G. The explanation doesn't change when you permute equivalent features.
- **Faithful** if E(f) agrees with model f's actual SHAP values.
- **Decisive** if E(f) commits to a ranking of all features, including within the group.

The impossibility: G-invariance (stability) forces E to be constant on G-orbits. But the actual SHAP values vary across the orbit (that's what the permutation does). So E can't simultaneously be constant (stable) and match each model's values (faithful) while ranking everything (decisive).

---

## The Resolution: Orbit Averaging

### The Reynolds operator

The canonical way to make a function G-invariant is the **Reynolds operator** (or orbit average):

$$\bar{E}(f) = \frac{1}{|G|} \sum_{g \in G} E(g \cdot f)$$

For SHAP attributions with G = S₃ acting on features {3, 7, 12}:

$$\bar{\varphi}_3 = \bar{\varphi}_7 = \bar{\varphi}_{12} = \frac{\varphi_3 + \varphi_7 + \varphi_{12}}{3}$$

The three features get the same averaged attribution. This is what we call **DASH** (Diversified Aggregation of SHAP): train M models with different seeds, average their SHAP values. Within each correlation group, the averaged values converge to a common value — a "tie."

### Why it's optimal

The orbit average is not just one possible resolution. It's the best one.

**Theorem (Pareto optimality).** Among all stable (G-invariant) explanation maps, the orbit average maximises pointwise faithfulness. No stable map can be more faithful to any individual model without being less faithful to another.

The proof uses Cauchy-Schwarz (Titu's lemma): for any weights w_i with Σw_i = 1, the variance Σw_i²σ² ≥ σ²/M with equality iff w_i = 1/M. Equal weighting (the orbit average) minimises variance — and in this context, variance IS unfaithfulness.

This is also the Rao-Blackwell theorem in disguise: conditioning on a sufficient statistic (the orbit-averaged SHAP) improves any unbiased estimator. The orbit average IS the Rao-Blackwell estimator.

---

## The Information-Loss Formula

### How much do you lose?

The orbit average kills the within-group variation and preserves the between-group variation. How much information survives?

The SHAP vector lives in V = ℝᴾ. The group G acts on V by permuting coordinates within each cluster. The G-invariant subspace is:

$$V^G = \{v \in V : g \cdot v = v \text{ for all } g \in G\}$$

For G = Sₖ acting on k coordinates, V^G is the 1-dimensional subspace where all k coordinates are equal (the average). So dim(V^G) = 1 out of dim(V) = k.

**The η law:**

$$\eta = \frac{\dim V^G}{\dim V}$$

This is the fraction of explanation content that survives orbit averaging. For a single group of size k: η = 1/k. For multiple groups of sizes k₁, ..., k_g:

$$\eta = \frac{\text{number of groups}}{\text{number of features}} = \frac{g}{P}$$

because each group contributes 1 dimension to V^G (its average) out of k_i dimensions in V.

### The Noether counting theorem

The η law has a sharp consequence for **which queries are stably answerable.**

With P features in g groups, the number of pairwise ranking facts is P(P-1)/2. How many are stable? Only the **between-group** rankings. There are g(g-1)/2 between-group pairs (one per pair of groups), and each is stably answerable (the groups have different average importances).

The within-group rankings (k_i(k_i-1)/2 per group) are ALL unstable — they flip with probability ≈ 50%.

This gives a **bimodal distribution** of flip rates: a spike near 0% (between-group, stable) and a spike near 50% (within-group, coin flip). The gap between these spikes is the **Noether gap** — analogous to how Noether's theorem in physics says symmetries produce conservation laws, here symmetries produce stable ranking facts.

Empirically: the Noether gap is 47 percentage points, invariant across ρ = 0.50 to 0.99, replicated across both Ridge and XGBoost. Pre-registered and confirmed.

---

## The Same Structure Across Domains

### Causal inference: Markov equivalence

In causal discovery, the configuration space Θ is the set of DAGs (directed acyclic graphs). The observable is the conditional independence (CI) structure. Multiple DAGs can encode the same CI structure — they form a **Markov equivalence class** (MEC).

The symmetry group: **edge orientation reversals** within the MEC. For the classic 3-node example:

```
Chain:  A → B → C     (CI: A ⊥ C | B)
Fork:   A ← B → C     (CI: A ⊥ C | B)
```

Same CI structure, different causal explanations. G = ℤ/2ℤ permuting chain ↔ fork.

The resolution: **CPDAGs** (completed partially directed acyclic graphs). Instead of committing to A → B → C or A ← B → C, report the edge A — B as undirected. This is exactly the orbit average: replace the two directed versions with their invariant content (the undirected skeleton + compelled edges).

The η law: for a MEC of size k (k DAGs sharing the same CI), η = 1/k. Only 1/k of the edge orientations are compelled (stably answerable). The rest are arbitrary.

DASH applied to causal discovery: train multiple models on bootstrap samples of the data, run the PC algorithm on each, report the edges that are consistently oriented. This reduces the orientation flip rate by 4× (from 1.4% to 0.3%, p < 10⁻¹⁰) — at the cost of reporting more undirected edges.

### Physics: gauge invariance

In electromagnetism, the configuration space Θ is the space of vector potentials A_μ(x). The observable is the electromagnetic field tensor F_μν = ∂_μA_ν - ∂_νA_μ. Multiple potentials produce the same field — they differ by a gauge transformation A_μ → A_μ + ∂_μχ.

The symmetry group: **gauge transformations** (adding gradients of arbitrary functions). This is an infinite-dimensional group — much larger than the finite permutation groups in ML.

The resolution: **gauge-invariant observables** (Wilson loops, field strengths). Instead of reporting A_μ (gauge-dependent, "unfaithful to the physics" in the sense that it contains gauge artifacts), report F_μν or ∮A·dl (gauge-invariant, stable across equivalent descriptions).

The η law: for a discrete lattice gauge theory with G = ℤ₂^(|V|-1), the fraction of gauge-invariant information is η = 1/|G|. In the continuum, G is infinite-dimensional and η → 0: almost no information about the potential survives gauge-fixing. This is why physicists work with field strengths, not potentials.

The structural correspondence is exact at the level of the group action:

| | ML (SHAP) | Physics (gauge) | Causal (DAGs) |
|---|---|---|---|
| Configuration | Trained model | Vector potential | DAG |
| Observable | Predictions | Field tensor | CI structure |
| Explanation | SHAP values | Potential values | Edge orientations |
| Symmetry group | S_k (feature perm) | Gauge group | MEC permutations |
| Resolution | DASH (orbit avg) | Gauge-invariant obs | CPDAG |
| What's lost | Within-group ranking | Potential information | Undirected edges |

The mathematical move is identical in all three: project onto V^G via the Reynolds operator. The objects are completely different. The fact that three fields independently discovered the same mathematical resolution — without knowing the connection — is what the theorem explains.

### Statistics: sufficient statistics

Fisher's sufficiency principle says: a statistic T(X) is sufficient for a parameter θ if the conditional distribution of X given T(X) doesn't depend on θ. In our language: T is G-invariant, where G is the group of data permutations that preserve T.

The Rao-Blackwell theorem says: conditioning any estimator on a sufficient statistic improves it (reduces variance without introducing bias). This is exactly the orbit average: project the estimator onto the G-invariant subspace.

The η law becomes: the fraction of information in X that T preserves is dim(V^G)/dim(V). For a minimal sufficient statistic, this is the optimal compression ratio.

EM algorithm: the E-step computes expected sufficient statistics over latent variables — averaging over the "missing data" symmetry group. The M-step then optimises using these averaged statistics. EM is DASH for mixture models.

---

## The Architecture of the Theory

Stepping back, the theory has a layered structure:

```
Layer 1: The Impossibility (pure logic, 0 axioms)
    |
    |  "faithful + stable + decisive is impossible under Rashomon"
    |
Layer 2: The Algebraic Classification (2×2 table)
    |
    |  neutral/committal elements determine which pairs are achievable
    |  bilemma for maximally incompatible H
    |
Layer 3: The Resolution (representation theory)
    |
    |  G-invariant projection = Reynolds operator
    |  Pareto-optimal among stable maps
    |  η = dim(V^G)/dim(V) quantifies information loss
    |
Layer 4: The Quantitative Predictions (testable)
    |
    |  Noether counting: g(g-1)/2 stable facts, bimodal gap
    |  Gaussian flip rate: Φ(-SNR) per pair
    |  η law: dim(V^G)/dim(V) predicts instability cross-domain
    |
Layer 5: The Recursive Structure (enrichment tower)
    |
    |  bilemma at each level → forced enrichment → new level
    |  parallels Gödel/Tarski (pattern, not reduction)
    |
Layer 6: Domain Instantiation (8 constructive + 9 axiomatized)
    |
    |  each domain identifies Θ, H, Y, G and verifies Rashomon
    |  all share the same 4-line proof
```

Each layer is independently valuable. Layer 1 alone is a classification result. Layers 1-3 give a complete theory with optimal resolution. Layer 4 makes it empirically testable. Layer 5 is speculative but mathematically precise. Layer 6 is the unification claim.

The Lean formalization verifies Layers 1-3 and most of Layer 6 mechanically (493 theorems, 0 sorry). Layer 4 is verified empirically (15 experiments). Layer 5 is formalised but with physics axioms at Levels 2-3.

---

## Natural Questions

**Q: Isn't the orbit average just "averaging"? What's deep about that?**

Averaging is the implementation. The depth is in three things: (1) proving it's Pareto-optimal (not just "a good idea" but "the uniquely best stable method"), (2) quantifying exactly how much information it kills (η = dim(V^G)/dim(V), not "some"), and (3) showing that eight independent fields converged on it for the same structural reason (not coincidence but mathematical necessity).

**Q: What if G is wrong? What if the features aren't really symmetric?**

Then the instability prediction is wrong — which is precisely what the η law's R² = 0.25 on unknown-group instances shows. The framework is conditional on correct group identification. This is the main open problem. The drug discovery experiment demonstrated it concretely: Pearson correlation identifies zero groups in binary fingerprints (predicting 0% instability vs. observed 23%), while mutual information recovers the prediction (within 16%).

**Q: What about nonlinear symmetries?**

The theory works for any finite group G acting linearly on V. For continuous groups (U(1) for crystallographic phases, SO(d) for concept probe directions), you replace the sum with a Haar integral. For non-compact groups, no finite Haar measure exists and you need regularisation. The CCA spectrum analysis shows the boundary: discrete groups give bimodal flip rates; continuous groups give a continuous spectrum of instabilities.

**Q: How does this relate to the Langlands program?**

At a high level: the Langlands program studies representations of adelic groups GL_n(A_ℚ) and their connection to Galois representations. Our framework studies representations of symmetry groups acting on explanation spaces and their connection to achievable explanation properties. Both involve: (1) identifying the relevant symmetry group, (2) decomposing a function space into irreducible representations, (3) extracting the invariant content (automorphic forms / stable explanations). The adelic resolution in the Ostrowski application — treating all completions simultaneously rather than one at a time — is the direct analogue of working over the adeles rather than at a single place. The connection is structural, not formal: we don't claim to use Langlands machinery, but the mathematical pattern (invariant theory over symmetry groups) is the same.

**Q: What's the relationship between the η law and character theory?**

The dimension dim(V^G) can be computed via Burnside's lemma: dim(V^G) = (1/|G|) Σ_{g∈G} χ_V(g) where χ_V is the character of the representation V. For G = S_k acting on ℝ^k by permutation, χ_V(g) = number of fixed points of g. Burnside gives dim(V^G) = 1 (the trivial representation has multiplicity 1 in the permutation representation). So η = 1/k for a single group of size k. For the general case with multiple groups, the representation decomposes as a direct sum and dim(V^G) = number of groups. The character-theoretic formula is what makes the η law parameter-free: it depends only on the group structure, not on the data or the model.
