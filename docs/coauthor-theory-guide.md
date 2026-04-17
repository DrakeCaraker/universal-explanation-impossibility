# The Theory of Explanation Impossibility

*A mathematical companion for co-authors — April 2026*

*Assumes: representation theory, group actions, basic category theory, familiarity with Arrow's theorem. One reader is a physicist.*

---

## Part I: The Theorem and Why It's Not Trivial

### The Setup

An **explanation system** is a tuple (Θ, H, Y, obs, exp, ⊥) where:
- Θ is a configuration space (trained models, gauge configurations, codons, DAGs, ...)
- Y is an observable space (predictions, experimental outcomes, amino acids, CI structures, ...)
- H is an explanation space (feature rankings, local field values, codons, edge orientations, ...)
- obs: Θ → Y collapses configurations to observables
- exp: Θ → H assigns each configuration its "native explanation"
- ⊥ ⊂ H × H is an irreflexive incompatibility relation

The **Rashomon property** says: ∃ θ₁, θ₂ ∈ Θ such that obs(θ₁) = obs(θ₂) and exp(θ₁) ⊥ exp(θ₂). Two configurations look the same from outside but explain themselves incompatibly.

An **explanation map** E: Θ → H is:
- **Faithful** if ¬(E(θ) ⊥ exp(θ)) for all θ — E never contradicts the system's own account
- **Stable** if obs(θ₁) = obs(θ₂) ⟹ E(θ₁) = E(θ₂) — equivalent configurations get the same explanation
- **Decisive** if exp(θ) ⊥ h ⟹ E(θ) ⊥ h for all θ, h — E inherits every incompatibility of the native explanation

**Theorem (explanation impossibility).** Under the Rashomon property, no E is simultaneously faithful, stable, and decisive.

*Proof.* Take the Rashomon witnesses θ₁, θ₂. Decisiveness at θ₁ gives E(θ₁) ⊥ exp(θ₂). Stability gives E(θ₁) = E(θ₂), so E(θ₂) ⊥ exp(θ₂). But faithfulness at θ₂ gives ¬(E(θ₂) ⊥ exp(θ₂)). Contradiction. □

### Why you should care despite the short proof

Your first reaction is probably: "This is immediate from the definitions." You're right. The proof is four lines. But consider:

1. **Arrow's theorem** also has a short proof once you have the right definitions (IIA + Pareto + non-dictatorship). Arrow's contribution was identifying that these three natural desiderata are jointly inconsistent. Similarly, our contribution is identifying that faithful + stable + decisive — three things every practitioner wants — are jointly inconsistent under underspecification.

2. **The theorem is tight.** Each pair of properties is independently achievable:
   - Faithful + Decisive: set E = exp (the native explanation). Faithful by irreflexivity of ⊥; decisive by identity. But unstable (different Rashomon witnesses give different answers).
   - Faithful + Stable: set E(θ) = h₀ for some neutral element h₀ compatible with everything. Stable trivially; faithful because h₀ ⊥ exp(θ) never holds. But indecisive (h₀ commits to nothing).
   - Stable + Decisive: set E(θ) = h* for some maximally committal h*. Stable trivially; decisive if h* inherits all incompatibilities. But unfaithful (h* contradicts most exp(θ)).

   The impossibility is specifically the triple. This is non-trivial: many impossibility results are not tight.

3. **The theorem is necessary.** If the Rashomon property fails (obs is injective on fibers relevant to exp), then E = exp is faithful, stable, and decisive simultaneously. So: impossibility ⟺ Rashomon. This biconditional means the Rashomon property is the *exact* boundary.

4. **The theorem is constructive.** The resolution (G-invariant projection) is not just "you can't have all three" but "here is the Pareto-optimal achievable set, and here is the unique optimal point on it."

---

## Part II: The Algebraic Structure

### The Tightness Classification

The severity of the impossibility depends on the algebraic structure of H. Two properties matter:

- A **neutral element** n ∈ H satisfies ¬(n ⊥ h) for all h. It's compatible with every explanation. (Think: "I don't know" or "tied" or "indeterminate.")
- A **committal element** c ∈ H satisfies: if h ≠ c, then c ⊥ h. It's incompatible with everything else. (Think: "definitely Feature A" or "definitely smooth geometry.")

These two properties completely determine which pairs of desiderata are achievable:

|  | Committal ∃ | No committal |
|--|-------------|-------------|
| **Neutral ∃** | F+D, F+S, S+D (full tightness, like Arrow) | F+D, F+S |
| **No neutral** | F+D, S+D | **F+D only** (the bilemma) |

This is a 2×2 classification of all explanation systems. The bottom-right cell — maximally incompatible H, no neutral element — gives the **bilemma**: even faithful + stable alone is impossible. You can't even drop decisiveness and keep the other two.

*For the physicist:* The bilemma applies whenever the explanation space is binary (two incompatible states with no neutral ground). Spacetime geometry under Ostrowski's classification is binary: smooth (archimedean) vs. ultrametric (non-archimedean), with no intermediate. Quantum measurement before decoherence is binary: definite vs. superposition. The bilemma says: in these systems, there is no faithful, stable description — period. You must enrich the space.

### The Enrichment Mechanism

When H is maximally incompatible (bilemma regime), the only escape is to **enrich** H by adding a neutral element. Define H' = H ∪ {n} where n is compatible with everything. Now the enriched system has a neutral element, and faithful + stable becomes achievable (set E(θ) = n on Rashomon fibers).

The enrichment is:
- **Forced**: no other operation on H restores F+S
- **Unique**: the neutral resolution (E = n on Rashomon fibers) is the *only* faithful + stable map on H' (proved in Lean as `constructive_unique_faithful_stable`)
- **Costly**: decisiveness is lost (n commits to nothing)

*For the physicist:* This is exactly the structure of paradigm shifts for binary explanation spaces:

| Pre-revolution H | Enrichment | What's lost |
|-------------------|-----------|-------------|
| {definite, superposition} | "no fact of the matter" | Pre-measurement state |
| {simultaneous, not} | "frame-dependent" | Absolute simultaneity |
| {smooth, ultrametric} | "indeterminate" | Spacetime geometry |
| {positive SHAP, negative SHAP} | "tied" (DASH) | Feature ranking |

Each adds a neutral element that is compatible with both pre-existing explanations, at the cost of decisiveness. The bilemma proves this enrichment is forced — not a choice, but a mathematical necessity.

---

## Part III: The Resolution as Representation Theory

### The G-Invariant Projection

Where does the symmetry group come from? The Rashomon property says obs identifies certain configurations. The set of "equivalent" configurations {θ : obs(θ) = y} is the **fiber** over y. If a group G acts transitively on fibers (permuting equivalent configurations), then:

**Theorem.** Any G-invariant map E: Θ → H is stable.

*Proof.* If obs(θ₁) = obs(θ₂), transitivity gives g ∈ G with g·θ₁ = θ₂. G-invariance gives E(θ₁) = E(g·θ₁) = E(θ₂). □

So the question becomes: among G-invariant maps, which is the "best"? The answer is the **orbit average** (Reynolds operator):

$$\bar{E}(\theta) = \frac{1}{|G|} \sum_{g \in G} E(g \cdot \theta)$$

This is:
- The projection onto the trivial representation of G in the function space V = {E : Θ → H}
- Faithful in expectation (the averaged explanation doesn't contradict the average native explanation)
- Pareto-optimal among stable maps (no stable map achieves higher pointwise faithfulness)

*For the representation theorist:* The explanation space V decomposes as V = V^G ⊕ V^⊥ where V^G is the space of G-invariant functions (stable explanations) and V^⊥ is the complement (the "noise" killed by averaging). The fraction of information that survives is:

$$\eta = \frac{\dim V^G}{\dim V}$$

This is the **η law**: the fraction of explanation content preserved by the resolution. For G = S_k (permutation of k equivalent features), η = 1/k. For G = ℤ/2ℤ (binary swap), η = 1/2.

Empirically, η predicts the observed instability rate across 7 well-characterized domains with R² = 0.957. The character-theoretic formula works as a parameter-free quantitative predictor spanning ML, biology, and statistical mechanics.

### The Design Space Dichotomy

For the attribution case (SHAP values under collinearity), the design space is completely characterized:

- **Family A** (single-model methods): E = exp. Faithful + decisive, but unstable. Flip rate = 50% for maximally collinear pairs.
- **Family B** (DASH ensemble): E = orbit average. Faithful + stable, but indecisive. Reports ties for equivalent features.

These are the only two non-dominated profiles. Any other method is Pareto-dominated by one of these. The proof uses the Cauchy-Schwarz / Titu's lemma bound: the orbit average achieves the minimum variance σ²/M among all unbiased estimators, matching the Cramér-Rao lower bound.

*Natural question: What about biased methods (James-Stein shrinkage)?* Shrinkage trades faithfulness for stability (biasing toward a prior reduces variance). The qualitative impossibility still holds — you're just choosing a different point on the Pareto frontier. Quantifying the full bias-variance-decisiveness Pareto surface for shrinkage estimators is open.

---

## Part IV: The Recursive Structure and Its Parallels

### The Enrichment Stack

The enrichment mechanism is recursive. Enriching H at level 0 creates H' at level 1. But H' may itself be maximally incompatible at a higher level of description, requiring a second enrichment to H'' at level 2. This gives a tower:

$$H_0 \hookrightarrow H_1 \hookrightarrow H_2 \hookrightarrow \cdots$$

Each level has its own binary question, its own bilemma, and its own forced enrichment. The physical enrichment stack has depth ≥ 3:

| Level | Binary question | Enrichment |
|-------|----------------|-----------|
| 0 | Are values definite or superposed? | Quantum: "no fact of the matter" |
| 1 | Is geometry smooth or ultrametric? | Adelic: "indeterminate" |
| 2 | Is information preserved or destroyed? | Black hole complementarity |
| 3 | Is spacetime fundamental or emergent? | (Open — quantum gravity) |

We formalise this as `RecursiveImpossibility`: a structure where every level has an impossibility, and resolving level n creates level n+1.

### The Gödel and Tarski Parallels

This recursive structure has a pattern shared with two classical results:

**Gödel's incompleteness:** Any consistent formal system containing arithmetic has true statements it cannot prove. Adding these statements as axioms gives a new system — which has its own unprovable truths. The tower: T₀ ⊂ T₁ ⊂ T₂ ⊂ ... where each Tₙ₊₁ extends Tₙ with statements independent of Tₙ.

**Tarski's undefinability:** Truth for a formal language L₀ cannot be defined within L₀. You need a metalanguage L₁. But truth for L₁ cannot be defined within L₁ — you need L₂. The tower: L₀ ⊂ L₁ ⊂ L₂ ⊂ ...

**The trilemma enrichment:** Explanation for a maximally incompatible H₀ requires enriching to H₁. But H₁ may be maximally incompatible at a higher level, requiring H₂. The tower: H₀ ⊂ H₁ ⊂ H₂ ⊂ ...

All three share the **consequence**: impossibility at every level, with unbounded depth. The **mechanisms** differ: Gödel uses diagonal self-reference, Tarski uses definability ascent, the trilemma uses enrichment. We capture the shared consequence as `RecursiveImpossibility` in Lean and prove both Gödel's theorem (from `hasGoedelProperty`, a consequence of the diagonal lemma weaker than the full lemma) and the enrichment tower instantiate this structure.

*Important caveat:* This is a **pattern-level** parallel, not a formal reduction. We do not claim the trilemma "implies" Gödel or vice versa. The shared structure is the unbounded recursive tower, not the mechanism. The Tarski parallel is actually tighter than the Gödel one: Tarski's hierarchy of metalanguages is structurally closer to the enrichment stack (both involve ascending to a new descriptive level) than Gödel's diagonal construction (which is specifically about self-reference).

### The Unification Consequence

If the enrichment stack continues to arbitrary depth — which the genericity argument suggests whenever dim(Θ) > dim(Y) at each level — then any unified description (one that is faithful and stable at every level simultaneously) must sacrifice decisiveness at every level. This is an infinite hierarchy of indeterminacies: the deeper you look, the more you must abstain from definite claims.

For physics: a "theory of everything" that is faithful to observations at every level of description (quantum, geometric, information-theoretic, emergent) must report "indeterminate" at every level for Rashomon fibers. The cost of unification is permanent partial agnosticism.

---

## Part V: The Physics Application

### Ostrowski's Theorem and the Spacetime Bilemma

Ostrowski's theorem (1916) classifies all nontrivial absolute values on ℚ: they are equivalent to either the standard (archimedean) absolute value or a p-adic (ultrametric) absolute value for some prime p.

For physics, this means: if we model spacetime geometry as arising from a valued field completion of ℚ, there are exactly two geometric paradigms — smooth (archimedean completion → ℝ) and ultrametric (p-adic completion → ℚ_p). These are maximally incompatible: no intermediate geometry exists.

The Rashomon property comes from the Freund-Witten product formula for tree-level bosonic string amplitudes:

$$S_\infty(s) \cdot \prod_p S_p(s) = 1$$

This says: the archimedean and non-archimedean amplitudes are not independent but related by a product constraint. Both contribute to the same observable (the scattering amplitude), but they "explain" the amplitude through incompatible geometric frameworks.

The bilemma then gives: no description of spacetime can be simultaneously faithful (respect each completion's geometry), stable (same description regardless of which completion you start from), and decisive (commit to smooth or ultrametric). The resolution: the adelic framework, which treats all completions simultaneously through the adele ring A_ℚ = ℝ × ∏'_p ℚ_p — enriching the explanation space with an "indeterminate" geometry that is compatible with both.

*For the Langlands-aware reader:* The adelic approach to number theory (Tate's thesis, automorphic forms) does exactly this: it treats all places simultaneously rather than committing to one. The G-invariant projection in our framework is the analogue of taking automorphic forms (functions on GL_n(A)) rather than working at a single place. The fact that the same mathematical move — orbit averaging over a symmetry group — appears in both contexts is not coincidental. Both are instances of the general principle: when equivalent descriptions exist, the invariant content is the only stable description.

### The Gauge Correspondence

The G-invariant resolution is mathematically identical to gauge invariance. In both:
- A group G acts on configurations (gauge transforms / Rashomon permutations)
- Observables are G-invariant functions (gauge-invariant quantities / stable explanations)
- The orbit average is the projection onto the trivial representation (Reynolds operator)
- Pareto-optimality follows from the same argument

The structural differences are real:
- Gauge symmetry is local and continuous; attribution symmetry is global and discrete
- Gauge theory has dynamics (equations of motion); attribution is static
- Gauge fixing selects a representative per orbit; orbit averaging sums over the orbit

These are different operations: gauge fixing is a section of the principal bundle; orbit averaging is the Reynolds operator. The mathematical identity is at the level of the group action formalism, not a claim that SHAP instability is "the same as" gauge redundancy in any physical sense.

---

## Part VI: Connections to Classical Mathematics

### Bridge Theorems

The framework connects to five established mathematical results:

1. **Fisher sufficiency + Rao-Blackwell.** A sufficient statistic T(X) is a G-invariant function where G permutes X-values that share the same T. Rao-Blackwell says: conditioning on a sufficient statistic improves any estimator. Our orbit average IS the Rao-Blackwell estimator when G is the symmetry group of T.

2. **Noether's theorem.** In physics, continuous symmetries correspond to conservation laws. In our framework, the symmetry group G determines which explanation queries are stably answerable: exactly dim(V^G) of them. This is a discrete analogue: symmetry → stable information, with the amount quantified by η = dim(V^G)/dim(V).

3. **EM algorithm.** The E-step of EM computes expected sufficient statistics over the latent variables — which is orbit averaging over the missing-data symmetry group. The M-step then optimises using these stable statistics. EM is the framework's resolution applied to mixture models.

4. **Bayesian posterior predictive.** Under a uniform prior on G-orbits, the posterior predictive distribution is exactly the orbit average. DASH with equal weights is the frequentist version of Bayesian model averaging.

5. **Quantum error correction.** A quantum error-correcting code protects k logical qubits in n physical qubits against errors from a group G. The code rate k/n equals our η = dim(V^G)/dim(V). The interpretability ceiling (≤ 1/n stably interpretable neurons for S_n permutation symmetry) is the error correction rate.

### What's Open

- **The full representation-theoretic classification.** We classify non-dominated explanation profiles for G = ℤ/2ℤ (two profiles: trivial and sign representations). For general finite G, we conjecture the number of non-dominated profiles equals the number of irreducible representations. This is proved for abelian G (all irreps are 1-dimensional) but open for non-abelian G, where the character theory is richer.

- **Non-compact groups.** For continuous symmetry groups (e.g., the rotation group for concept probe directions, or U(1)^N for crystallographic phases), orbit averaging requires a Haar measure. For non-compact groups, no finite Haar measure exists and regularisation is needed. The CCA spectrum analysis shows this boundary: discrete groups give clean bimodal flip rates; continuous groups give a spectrum.

- **The categorical enrichment.** We conjecture that enrichment corresponds to the free adjunction of an initial object to the incompatibility poset viewed as a thin category. The universal property would explain uniqueness. The precise 2-categorical formulation is open.

- **Group identification.** The biggest practical bottleneck. Given data, how do you identify the correct symmetry group G? Pearson correlation works for continuous linear features, mutual information for binary features, but there's no universal method. SAGE is a heuristic; a principled approach (perhaps related to invariant theory or representation learning) is needed.

---

## Part VII: The Evidence

### What's Confirmed

| Prediction | Evidence | Pre-registered? |
|-----------|---------|----------------|
| Noether bimodality (47pp gap) | Confirmed across ρ = 0.50–0.99, both Ridge and XGBoost | Yes |
| η law (R² = 0.957, 7 domains) | LOO R² = 0.79, permutation p = 0.010, F(1,5) = 38.78 | Yes (subset criterion) |
| Interpretability ceiling (0/n stable neurons) | Confirmed: frac_stable = 0, invariant variance < 10⁻³² | Yes |
| Gaussian flip rate (R² = 0.814, 647 pairs) | Bootstrap CI [0.791, 0.835], calibration slope 1.17 (conservative) | No (exploratory) |
| SAGE beats all baselines | LOO R² = 0.686, Spearman ρ = 0.952, permutation p = 0.002 | No (exploratory) |
| Proportionality robustness | 0/11,000 Pareto violations (two families intact at all CV levels) | No |
| Clinical reversal (45% on German Credit) | Cluster-level, robust across XGBoost and LightGBM | No |
| Drug discovery: MI recovers prediction | MI: 16% error, Jaccard: 4% error (vs Pearson: 100% failure) | No (diagnostic) |

### What's Falsified

| Prediction | What happened | What it means |
|-----------|--------------|--------------|
| Phase transition at r* ≈ 1 | r* ∈ [0.01, 0.12] — ten times earlier | Overparameterisation threshold is wrong; instability onset is earlier |
| Tradeoff bound α+σ+δ ≤ 2 | Max sum = 2.86 | Binary impossibility holds around (1,1,1) but linear budget is wrong |
| Molecular evolution from character theory | Partial R² = 0.0 after controlling for neighbor count | Group theory adds nothing for evolutionary rate; biochemistry suffices |
| Drug discovery: Noether bimodality | Bimodal gap = 0.005 (uniform instability, not clustered) | Discrete group structure required; diffuse overlap doesn't bimodalise |

### What's Inconclusive

| Question | Status |
|---------|--------|
| Are transformer circuits non-unique? (MI v2) | Running on SageMaker — modular addition, 30 models from random init |
| Does the framework extend to non-abelian groups? | Theoretical conjecture, no empirical test |
| Does DASH outperform shrinkage on the bias-variance-decisiveness frontier? | Open; James-Stein analysis deferred to future work |

---

## The Punchline

If you remember one thing: **eight fields, 150 years apart, independently discovered that the only stable description of an underspecified system is the orbit average over equivalent configurations — and we can now prove this is the unique Pareto-optimal resolution, quantify the information loss as dim(V^G)/dim(V), and verify the entire framework in Lean 4.**

The theorem is elementary. The framework is the contribution.
