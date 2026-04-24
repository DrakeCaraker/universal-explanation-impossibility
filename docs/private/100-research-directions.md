# 100 Research Directions: Filtered to Feasible

*Generated from the framework's mathematical machinery. Each direction uses existing tools (representation theory, group actions, topology, information theory) applied to the explanation impossibility. Directions marked IMPOSSIBLE or BLOCKED are removed. Remaining directions are ranked by feasibility × impact.*

---

## Category A: Representation Theory (18 directions)

1. **Irreducible decomposition predicts flip correlations.** Features in the same irreducible of the G-action flip together; different irreducibles flip independently. Testable on existing data.

2. **Spectral gap predicts DASH convergence rate.** For S_k, gap = (k-1)/k. DASH flip rate should decay as (1/k)^M. Gives group-theoretic ensemble size formula.

3. **Character values at specific group elements predict per-model SHAP deviation.** χ(g) for the specific permutation g relating two models predicts how different their SHAP vectors are.

4. **Schur orthogonality gives independence of between-group and within-group instability.** The orthogonality relations should predict that within-group and between-group flip rates are statistically independent — testable.

5. **Plancherel measure predicts the distribution of flip rates.** The distribution of per-pair flip rates should match the Plancherel measure of G (probability of each irreducible appearing in a random representation).

6. **Branching rules predict intermediate-ρ behaviour.** When S_k breaks to S_{k-1} × S_1 at ρ < 1, the branching rules predict which irreducibles survive and which split.

7. **Induced representations predict transfer learning instability.** Fine-tuning changes the symmetry group G → H ⊂ G. The induced representation Ind_H^G predicts explanation instability of the fine-tuned model.

8. **Tensor product decomposition predicts second-order statistics.** V ⊗ V decomposes into irreducibles; the decomposition predicts the variance of flip rates (not just the mean).

9. **Frobenius reciprocity connects instance-level and group-level predictions.** The number of times the trivial rep of H appears in the restriction of a G-rep equals the number of G-invariants in the induced H-rep.

10. **Hecke algebra structure for non-transitive actions.** When G doesn't act transitively on fibers (partial Rashomon), the Hecke algebra H(G,H) governs the partial averaging.

11. **Representation stability (in the FI-module sense) as features grow.** As P → ∞ with fixed group structure, does the η law stabilize? Church-Ellenberg-Farb representation stability might apply.

12. **Molien's theorem gives the Hilbert series of stable explanations.** The generating function for the dimension of G-invariant polynomials at each degree — predicts how many "explanation queries" of each complexity are stably answerable.

13. **Weyl character formula for continuous groups.** For O(d) acting on concept probe directions, the WCF gives the exact decomposition and η ratio.

14. **Schur-Weyl duality connects feature permutations to model permutations.** The commutant of S_k acting on V^{⊗n} gives the GL(V)-structure — connecting feature symmetry to model diversity.

15. **Maschke's theorem guarantees the decomposition exists.** Over ℝ with finite G, the representation is completely reducible. This is why orbit averaging works — the complement V^⊥ exists and is killed cleanly.

16. **Peter-Weyl for compact G gives L² decomposition.** For continuous symmetry groups (O(d), U(n)), the L² decomposition into matrix coefficients generalizes the finite-group η law.

17. **The regular representation contains every irreducible.** If we view the full Rashomon set as the regular representation, every instability mode appears. The η ratio counts how many survive averaging.

18. **Automorphism groups of explanation spaces.** The group Aut(H, ⊥) of structure-preserving maps on the explanation space constrains which resolutions are possible.

## Category B: Topology & Geometry (16 directions)

19. **Rashomon topology: connected components predict bimodality.** R_ε with two connected components in SHAP space → bimodal flip distribution. Testable with clustering on existing 50-model data.

20. **Persistent homology of the Rashomon filtration.** As ε increases, track the births/deaths of topological features. The "Rashomon barcode" encodes the full stability landscape.

21. **Critical ε where R_ε disconnects = onset of bimodality.** The critical tolerance where the Rashomon set splits into basins should predict the correlation threshold where bimodality appears.

22. **Compatibility complex Euler characteristic.** χ(K) for the compatibility complex on H. Relates to the achievability count via the Euler-Poincaré formula.

23. **Nerve theorem connects the compatibility complex to achievability.** If the compatible subsets cover H, the nerve of the cover has the homotopy type of H — connecting achievability to topology.

24. **Morse theory on the loss landscape.** Critical points of the loss function correspond to models with distinct explanations. The Morse index predicts the type of instability (flip vs. rotation vs. magnitude change).

25. **Geodesics on the Rashomon manifold.** The shortest path between two Rashomon models in parameter space passes through models with intermediate explanations — predicting the landscape of "partially stable" methods.

26. **Curvature of the Rashomon boundary.** Where R_ε has high curvature, small changes in ε produce large changes in the explanation set. High curvature = fragile stability boundary.

27. **Fiber bundle structure of obs⁻¹(y).** The explanation map is a section of the bundle Θ →^{obs} Y. The impossibility says no global section is faithful+stable+decisive. The obstruction class lives in the bundle's characteristic class.

28. **Stiefel-Whitney classes detect binary obstructions.** For binary H, the first Stiefel-Whitney class w₁ detects whether a global section exists — directly related to the bilemma.

29. **Homotopy groups of the explanation space.** π₁(H, ⊥) measures loops in the compatibility structure. Non-trivial π₁ = persistent explanation ambiguity.

30. **Discrete Morse theory on the compatibility complex.** Reduces K to a simpler complex with the same homotopy type — efficiently computing the tightness classification.

31. **Topological data analysis on SHAP vectors across models.** Apply TDA (persistent homology via Rips complex) to the cloud of SHAP vectors from 50 models. The persistence diagram should distinguish stable from unstable features.

32. **Mapper algorithm on the Rashomon set.** Construct a topological summary of R_ε using the Mapper algorithm — visualizing the explanation landscape for practitioners.

33. **Alexander duality between faithful and stable.** In the compatibility complex, the complement of the faithful region and the complement of the stable region may be Alexander dual — making the impossibility a topological identity.

34. **Borsuk-Ulam for continuous H.** For H = S^{n-1} (unit sphere of attributions), Borsuk-Ulam says any continuous map from S^n to ℝ^n identifies antipodal points — a continuous version of the bilemma.

## Category C: Information Theory (12 directions)

35. **Rate-distortion theory for explanations.** The faithful explanation has rate R = H(exp|obs) and distortion D = 0. The stable explanation has R = 0 and distortion D = H(exp|obs). The tradeoff curve is the rate-distortion function of the explanation channel.

36. **Mutual information I(E;exp) measures faithfulness.** I(E;obs) measures stability. The impossibility is: I(E;exp) + I(E;obs) cannot both be maximal when H(exp|obs) > 0.

37. **Channel capacity of the explanation channel.** The maximum mutual information between the explanation and the true structure, subject to stability constraints. DASH achieves capacity.

38. **Fisher information on the Rashomon manifold.** The Fisher-Rao metric on model space gives the "information cost" of resolution in natural units (nats/bits).

39. **Cramér-Rao bound for explanation estimators.** The quantitative bilemma IS a Cramér-Rao-type result: faithfulness = unbiasedness, stability = low variance, and the bound relates them.

40. **Data processing inequality for explanations.** obs is a noisy channel: Θ → Y loses information. The DPI says any function of Y carries less information about Θ than Y itself. The explanation is a function of the model, not of obs.

41. **Entropy of the Rashomon set predicts instability magnitude.** H(Rashomon) = log|R_ε| should predict the overall instability level — larger Rashomon sets → more instability.

42. **KL divergence between models' SHAP distributions.** D_KL(SHAP_m₁ ‖ SHAP_m₂) measures how different two models' explanations are — a more refined metric than flip rate.

43. **Minimum description length for explanations.** The shortest description of a stable explanation has length ≥ log(|R_ε|/|G|) — the coding cost of quotienting out the symmetry.

44. **Conditional entropy H(ranking | group) = 0 for exact symmetry.** The Noether counting theorem in information-theoretic language: the ranking is fully determined by the group identity.

45. **Extrinsic information transfer (EXIT) charts for DASH.** As DASH processes more models, the mutual information between the consensus and each model's true ranking increases. EXIT charts (from coding theory) visualize this convergence.

46. **Source coding theorem for explanations.** The minimum number of bits to encode a stable explanation is the entropy of the G-invariant subspace — computable from the representation.

## Category D: Dynamical Systems & Optimisation (10 directions)

47. **Gradient flow on the loss landscape traces explanation trajectories.** As training progresses, the model traces a path in parameter space. The SHAP values trace a corresponding path in explanation space. The endpoint's explanation depends on the path — path-dependence = Rashomon.

48. **Lyapunov stability of explanations under retraining.** An explanation is "Lyapunov stable" if small perturbations to the training (different seed) produce small changes in the explanation. The impossibility says: no explanation is Lyapunov stable for all features when Rashomon holds.

49. **Basin of attraction volumes predict explanation probability.** Each Rashomon model has a basin of attraction under gradient descent. The volume of the basin determines how often that model (and its explanation) is found. The bimodal distribution reflects two basins with different volumes.

50. **Mode connectivity implies explanation stability.** If two models are connected by a low-loss path, their explanations should interpolate smoothly. Linear mode connectivity → SHAP interpolation.

51. **Saddle points between basins have ambiguous explanations.** Models at saddle points between Rashomon basins should have the most unstable explanations — they're at the boundary between explanation regimes.

52. **Stochastic gradient descent as a random walk on the Rashomon set.** SGD with different seeds explores different regions of R_ε. The mixing time of this random walk determines how many retraining runs are needed to see the full explanation diversity.

53. **Neural tangent kernel regime predicts explanation stability.** In the NTK regime (infinite width), the model is approximately linear and explanations should be approximately stable. Finite-width corrections break stability.

54. **Lottery ticket hypothesis ↔ circuit Rashomon.** Different winning tickets (subnetworks) = different circuits = Rashomon for mechanistic interpretability. The lottery ticket distribution predicts the MI flip rate.

55. **Edge of chaos and explanation instability.** Models trained at the "edge of chaos" (critical initialisation) should have maximally diverse explanations — the system is maximally sensitive to perturbations.

56. **Sharpness-aware minimisation (SAM) reduces explanation instability.** SAM explicitly seeks flat minima. If flat minima = mode connectivity = explanation stability, then SAM-trained models should have lower SHAP flip rates than SGD-trained models. Directly testable.

## Category E: Causal Inference & Statistics (11 directions)

57. **Causal SHAP under the framework.** Conditional SHAP (interventional) resolves some Rashomon when causal effects differ. The escape condition (β₁ ≠ β₂) is precisely when the Rashomon property fails under causal conditioning. Extends the conditional impossibility theorem.

58. **Structural causal models as explanation systems.** An SCM defines (Θ = exogenous variables, Y = endogenous, H = causal mechanisms). The Rashomon property = multiple causal models fit the data (causal underdetermination).

59. **Sufficient statistics as G-invariants (formalised).** Fisher sufficiency = G-invariance for the symmetry group of the data. The Rao-Blackwell theorem IS the framework's resolution theorem. This bridge is stated but never experimentally validated.

60. **EM algorithm convergence rate from the spectral gap.** EM = orbit averaging for mixture models. The spectral gap of the missing-data symmetry group predicts EM convergence — a new result in the EM theory literature.

61. **Bayesian model averaging as weighted orbit averaging.** BMA with a uniform prior over the Rashomon set IS DASH. Non-uniform priors give different resolutions with different stability-faithfulness tradeoffs.

62. **Posterior predictive = the unique Bayes-optimal stable explanation.** Under a uniform prior, the posterior predictive is the orbit average. Under any prior, it's the unique Bayes-optimal stable explanation. This connects the framework to Bayesian decision theory.

63. **Conformal prediction intervals for explanations.** Use conformal inference to construct prediction intervals for SHAP values that have guaranteed coverage under retraining. The width of the interval = instability.

64. **Bootstrap aggregation (bagging) as approximate orbit averaging.** Bagging aggregates over bootstrap samples — an approximation to averaging over the Rashomon set. When does bagging approximate DASH? When the bootstrap distribution approximates the Rashomon distribution.

65. **Knockoff features for stable explanations.** The knockoff framework (Barber & Candès 2015) controls false discovery rate in variable selection. The Rashomon instability IS a source of false discoveries. Knockoff-SHAP would be stable by construction.

66. **Multiple testing correction for SHAP rankings.** Each feature pair comparison is a hypothesis test (j > k or k > j). The Rashomon property inflates the family-wise error rate. BH correction on SHAP rankings would give stability guarantees.

67. **Cross-validation instability as a Rashomon indicator.** CV fold variation IS a form of Rashomon (different training sets → different models → different explanations). CV-SHAP stability should predict full Rashomon stability.

## Category F: Applications & Domain-Specific (15 directions)

68. **Genomics: linkage disequilibrium blocks as symmetry groups.** LD blocks define correlation groups in GWAS. The η law should predict which SNP associations are stable. Testable on UK Biobank summary statistics.

69. **Drug discovery: molecular symmetry groups predict attribution stability.** Molecules with rotational symmetry (aromatic rings) should have rotationally-unstable atom-level attributions. The molecular point group IS the symmetry group.

70. **Climate attribution: ensemble model spread as Rashomon.** Climate ensembles (CMIP6) with similar global temperature projections but different regional attributions = Rashomon. The framework predicts which attributions are stable.

71. **Financial risk: model risk from Rashomon.** Different models with the same VaR give different risk factor attributions. The framework quantifies "model risk" — a term used in finance but never formally defined.

72. **Medical imaging: augmentation invariance as symmetry.** Data augmentation (rotation, flipping) defines a symmetry group. GradCAM explanations should be invariant to augmentations — the G-invariant projection is augmentation-averaged GradCAM.

73. **NLP: paraphrase invariance as symmetry.** Paraphrases are the "Rashomon set" of text — different inputs with the same meaning. Explanation instability across paraphrases IS the impossibility for NLP.

74. **Recommender systems: item permutation symmetry.** Interchangeable items (similar products) define a symmetry group. Explanations for "why this recommendation" are unstable among interchangeable items.

75. **Fairness: protected attribute correlation as Rashomon source.** When a protected attribute correlates with non-protected features, the "most important feature" explanation can flip between the protected and non-protected feature — a concrete fairness violation.

76. **Autonomous driving: sensor redundancy as symmetry.** Multiple sensors measuring the same quantity (camera + LiDAR + radar) create Rashomon for perception explanations. The G-invariant explanation fuses sensors.

77. **Protein structure: AlphaFold residue attribution stability.** Different AlphaFold sampling runs give different per-residue importance. The symmetry group comes from protein symmetry (multimers, domains).

78. **Time series: stationarity as symmetry.** Time-translation invariance is a symmetry group. SHAP values for time series should be invariant to time shifts — but aren't for non-stationary data.

79. **Quantum chemistry: orbital symmetry determines attribution stability.** Molecular orbital symmetry groups predict which orbital attributions are stable. The η law should work with the molecular point group.

80. **Supply chain: substitutable components as symmetry.** Interchangeable suppliers/components define a symmetry group. Risk attributions among substitutable components are Rashomon-unstable.

81. **Education: equivalent assessments as Rashomon.** Different test questions measuring the same skill = Rashomon for skill attribution. The framework predicts which skill attributions are reliable.

82. **Ecology: species interaction redundancy as symmetry.** Functionally redundant species create Rashomon for ecosystem attribution. The framework predicts which species roles are stably attributable.

## Category G: Computational & Algorithmic (10 directions)

83. **Fast orbit averaging via FFT.** For abelian groups (ℤ/nℤ, products of cyclic groups), orbit averaging is a convolution computable via FFT in O(n log n) instead of O(n²).

84. **Approximate orbit averaging for large groups.** When |G| is too large for exact averaging, sample uniformly from G and average. The Hoeffding bound gives convergence: O(log(1/δ)/ε²) samples for (ε,δ)-approximation.

85. **DASH with importance sampling.** Instead of uniform orbit averaging, weight models by their inverse loss — models near the optimum get higher weight. This is the importance-weighted orbit average.

86. **Online DASH: incremental orbit averaging.** As new models are trained, update the DASH consensus incrementally without recomputing from scratch. The running average IS the orbit average.

87. **Pruned orbit averaging: skip redundant models.** If two models have SHAP correlation > 0.99, including both in DASH adds no new information. Pruning the ensemble to maximally diverse models improves convergence.

88. **Parallel DASH with MapReduce.** DASH is embarrassingly parallel: train M models independently, average SHAP values. Scales to arbitrary ensemble sizes on distributed compute.

89. **Symbolic DASH: closed-form ensemble average.** For linear models, the ensemble-averaged SHAP value has a closed-form expression in terms of the covariance matrix. No need to train multiple models.

90. **Streaming DASH for non-stationary data.** As the data distribution shifts, the Rashomon set changes. Streaming DASH maintains a sliding window of recent models and recomputes the orbit average online.

91. **Differentiable orbit averaging.** If the orbit average is differentiable with respect to model parameters, you can optimise for stability directly during training — a "stability-aware" loss.

92. **DASH for black-box models via perturbation.** When TreeSHAP isn't available (black-box API), approximate SHAP via perturbation sampling. DASH still applies: average perturbation-SHAP across multiple model copies.

## Category H: Foundational & Meta-Theoretical (8 directions)

93. **The impossibility as a no-go theorem in the Abramsky hierarchy.** Position the impossibility relative to other no-go theorems: Bell (nonlocality), KS (contextuality), Arrow (aggregation). The compatibility complex analysis places it at a specific level.

94. **Category-theoretic formulation via presheaves.** The explanation system is a presheaf on the observation category. The impossibility is the non-existence of a global section satisfying the three constraints. Functorial.

95. **Galois connection between explanation properties.** Faithful ⊣ Decisive forms a Galois connection on the lattice of explanation maps. The closure operator is the orbit average.

96. **Operadic structure of multi-level explanations.** The enrichment stack is an operad: each level's enrichment is a "multi-ary operation" producing the next level's explanation space.

97. **Topos-theoretic interpretation.** The explanation system defines a topos (the category of sheaves on the observation space). The impossibility is the non-existence of a certain morphism in this topos.

98. **Monoidal category of explanation systems.** The tensor product of two explanation systems combines them. The impossibility for the tensor product relates to the impossibilities of the factors.

99. **2-categorical naturality of the resolution.** The G-invariant projection is a natural transformation between two functors: the identity and the orbit quotient. Naturality encodes the universality of the resolution.

100. **Impossibility as a conservation law (Noether for information).** Noether's theorem: symmetry → conservation. Here: Rashomon symmetry → conserved stable queries (exactly g(g-1)/2). The impossibility IS the information conservation law for underspecified systems.

---

## Filtering: What's Actually Feasible

**Removed as literally impossible (need unavailable data/tools):**
- None explicitly impossible, but items 77 (AlphaFold), 68 (UK Biobank), 70 (CMIP6) require specific dataset access
- Items 93-99 (category theory) are mathematically sound but not empirically testable

**Feasible with existing data (testable TODAY):**

| # | Direction | Data needed | Time | Impact |
|---|-----------|-------------|------|--------|
| 1 | Flip correlation from irreducibles | Existing 50-model SHAP data | 2 hours | Medium |
| 2 | Spectral gap → ensemble size | Existing DASH convergence data | 1 hour | High (practical) |
| 19 | Rashomon topology → bimodality | Existing 50-model SHAP data + clustering | 3 hours | Very high |
| 31 | TDA on SHAP vectors | Existing data + ripser/gudhi | 2 hours | High |
| 49 | Basin volumes → explanation probability | Existing models + retraining counts | 2 hours | Medium |
| 50 | Mode connectivity → SHAP interpolation | Need to train interpolated models | 4 hours | Very high |
| 56 | SAM → lower flip rates | Need to train SAM models | 4 hours | High (prescriptive) |
| 67 | CV instability as Rashomon indicator | Existing CV data | 1 hour | Medium |
| 100 | Noether conservation for information | Existing Noether data | 1 hour | High (framing) |

**Feasible with modest new experiments (testable THIS WEEK):**

| # | Direction | New experiment needed | Time | Impact |
|---|-----------|---------------------|------|--------|
| 7 | Induced rep → transfer learning | Fine-tune pretrained model, compare SHAP | 1 day | High |
| 35 | Rate-distortion for explanations | Compute R-D curve from SHAP data | 1 day | Very high |
| 56 | SAM vs SGD SHAP stability | Train SAM models, compare flip rates | 1 day | Very high |
| 63 | Conformal prediction intervals for SHAP | Apply conformal inference to SHAP | 1 day | High (practical) |
| 72 | Augmentation-averaged GradCAM | Apply to ImageNet model | 1 day | Medium |

---

## Top 10 Ranked by Feasibility × Impact

| Rank | # | Direction | Why |
|------|---|-----------|-----|
| 1 | 19 | **Rashomon topology predicts bimodality** | Connects two hot fields (loss landscape + XAI). Testable on existing data. If R_ε has 2 connected components ↔ bimodal flips, that's Nature-level. |
| 2 | 56 | **SAM reduces SHAP instability** | Turns impossibility into design principle. "Train with SAM → stable explanations." Immediately actionable. |
| 3 | 2 | **Spectral gap gives exact DASH ensemble size** | Replaces heuristic M=25 with group-theoretic M=f(k). Instantly useful for practitioners. |
| 4 | 35 | **Rate-distortion curve for explanations** | Unifies faithfulness-stability tradeoff with classical information theory. Beautiful and deep. |
| 5 | 50 | **Mode connectivity → explanation interpolation** | If low-loss paths exist, explanations interpolate smoothly. Connects to Frankle/Carlin mode connectivity. |
| 6 | 1 | **Irreducible decomposition predicts flip correlations** | New structural prediction. Which features flip TOGETHER? Testable now. |
| 7 | 100 | **Noether conservation law for information** | Reframes the η law as a conservation law. g(g-1)/2 stable facts ARE the "conserved quantity." Conceptually powerful. |
| 8 | 63 | **Conformal prediction intervals for SHAP** | Practical tool with theoretical guarantee. Distribution-free coverage. |
| 9 | 20 | **Rashomon barcode (persistent homology)** | New mathematical object. Future paper in J. Applied & Computational Topology. |
| 10 | 7 | **Induced representations for transfer learning** | Predicts fine-tuning instability from the pretrained model's symmetry. Relevant to LLM era. |
