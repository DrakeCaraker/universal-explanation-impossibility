# All 100 Directions Ranked by Knockout Potential

*A "knockout" = a result that makes a Nature editor say "we must publish this." Criteria: surprises non-specialists, connects unrelated fields, has immediate consequences, is not trivially predictable.*

---

## Tier S: Genuine Knockout Potential (5 directions)

These could each be Nature-level results if they work.

### Rank 1: #19 — Rashomon topology predicts bimodality
**Why knockout:** It would prove that SHAP instability is a TOPOLOGICAL property of the loss landscape. The bimodal flip distribution = the Rashomon set having two connected components in parameter space. This connects the impossibility theorem to loss landscape geometry — the hottest topic in ML theory (Draxler, Garipov, Fort, all 2018-2023). No one has connected explanation stability to loss landscape topology. If confirmed, every loss landscape paper becomes relevant to explainability and vice versa. The headline: "The reason your SHAP values flip is the same reason your loss landscape has multiple basins."

### Rank 2: #56 — SAM training produces stable explanations
**Why knockout:** It turns the impossibility from a diagnostic ("you can't have all three") into a PRESCRIPTION ("train this way and you automatically get stability"). Sharpness-aware minimisation seeks flat minima. Flat minima = mode connectivity = stable explanations. If SAM-trained XGBoost/neural nets have measurably lower SHAP flip rates than SGD-trained ones, practitioners have an immediate fix. The headline: "Don't fix your explainability method — fix your training procedure." This is actionable on every ML model in production today.

### Rank 3: #50 — Mode connectivity implies explanation interpolation
**Why knockout:** Frankle, Draxler, and Garipov showed that loss-connected models have similar performance. Nobody has shown they have similar EXPLANATIONS. If low-loss paths between models produce smoothly interpolating SHAP values, it means explanation stability is a CONSEQUENCE of loss landscape connectivity. This would unify two active research programs (mode connectivity + XAI) and give a geometric criterion for when explanations are reliable: "if the loss landscape is connected at your tolerance, your explanations are stable."

### Rank 4: #35 — Rate-distortion curve for explanations
**Why knockout:** It would show that the faithfulness-stability tradeoff is EXACTLY the rate-distortion tradeoff from information theory. Faithful explanation = high rate, zero distortion. Stable explanation = zero rate, high distortion. DASH = the optimal code on the R(D) curve. This connects the impossibility to Shannon's foundational theory (1948). Every information theorist would immediately understand the result. The headline: "The limit of explanation is the channel capacity of the Rashomon channel."

### Rank 5: #54 — Lottery ticket hypothesis ↔ circuit Rashomon
**Why knockout (conditional on MI v2):** If the MI v2 experiment shows circuits are non-unique AND different winning tickets correspond to different circuits, then the lottery ticket hypothesis IS circuit Rashomon. The impossibility theorem would predict: no circuit explanation can be simultaneously faithful to all tickets, stable across tickets, and decisive about circuit identity. The headline: "The lottery ticket hypothesis is an impossibility theorem in disguise."

---

## Tier A: High Impact, Not Quite Knockout (12 directions)

These would be strong publications in top venues but probably not Nature standalone.

### Rank 6: #2 — Spectral gap → exact DASH ensemble size
**Why high impact:** Replaces the heuristic M=25 with M = k·ln(1/α)/(1-λ₁) where λ₁ is the spectral gap. For S₄: M = 4·ln(20) ≈ 12. For S₂: M = 2·ln(20) ≈ 6. Adaptive to group structure. Immediately useful for every DASH user.
**Why not knockout:** It's an optimisation of an existing tool, not a new insight.

### Rank 7: #100 — Noether conservation law for information
**Why high impact:** Reframes the η law as: "Rashomon symmetry CONSERVES exactly g(g-1)/2 stable ranking facts, just as physical symmetry conserves energy/momentum." The Noether counting theorem IS a conservation law. This framing makes the result accessible to every physicist.
**Why not knockout:** It's a reframing, not a new result. The η law is already in the paper.

### Rank 8: #63 — Conformal prediction intervals for SHAP
**Why high impact:** Distribution-free guaranteed coverage intervals for SHAP values under retraining. "This feature's SHAP value is 0.35 ± 0.12 with 95% coverage across all possible retraining seeds." Immediately deployable.
**Why not knockout:** It's a tool, not a discovery. Conformal inference is well-established.

### Rank 9: #7 — Induced representations predict transfer learning instability
**Why high impact:** Fine-tuning changes G → H ⊂ G. The induced representation predicts how much instability transfers from the pretrained model. Relevant to every LLM fine-tuning pipeline.
**Why not knockout:** Requires substantial mathematical development and the prediction might not be sharp enough.

### Rank 10: #1 — Irreducible decomposition predicts flip correlations
**Why high impact:** Predicts WHICH features flip TOGETHER (correlated instability), not just which ones flip (marginal instability). New structural prediction testable on existing data.
**Why not knockout:** It's a refinement of the η law, not a new type of result.

### Rank 11: #20 — Rashomon barcode (persistent homology)
**Why high impact:** A new mathematical object encoding the full explanation stability landscape at every tolerance level. Beautiful mathematics. Would launch a research subfield.
**Why not knockout:** Too abstract for Nature. Strong candidate for Journal of Applied & Computational Topology.

### Rank 12: #65 — Knockoff-SHAP for stable feature selection
**Why high impact:** Combines the knockoff framework (FDR control) with DASH (stability). Knockoff-SHAP would select features that are both statistically significant AND stable across retraining. Solves two problems at once.
**Why not knockout:** It's a method paper, not a discovery.

### Rank 13: #47 — Gradient flow traces explanation trajectories
**Why high impact:** Visualising how SHAP values evolve during training reveals the dynamics of explanation formation. Path-dependence of the trajectory IS the Rashomon property.
**Why not knockout:** Visualisation, not prediction.

### Rank 14: #60 — EM convergence rate from spectral gap
**Why high impact:** New result in EM theory: the spectral gap of the missing-data symmetry group predicts convergence. Connects the framework to a classic statistical algorithm.
**Why not knockout:** Narrow audience (EM theorists).

### Rank 15: #68 — GWAS with LD blocks as symmetry groups
**Why high impact:** Predicts which SNP associations are stable using the LD block structure as the symmetry group. Would be immediately relevant to hundreds of GWAS studies.
**Why not knockout:** Requires UK Biobank access and domain collaboration.

### Rank 16: #75 — Fairness: protected attribute correlation as Rashomon
**Why high impact:** When a protected attribute correlates with non-protected features, SHAP can flip between attributing a decision to the protected vs non-protected feature. This IS a fairness violation caused by Rashomon. Connects the impossibility to algorithmic fairness — a huge policy area.
**Why not knockout:** The connection is conceptual, not a new theorem.

### Rank 17: #91 — Differentiable orbit averaging (stability-aware loss)
**Why high impact:** Train neural networks with a loss that penalises explanation instability. The orbit average becomes a differentiable regulariser. Models trained this way are stable BY CONSTRUCTION.
**Why not knockout:** Engineering contribution, not scientific discovery.

---

## Tier B: Solid Contributions (25 directions)

Would be good papers in ICML/NeurIPS/AISTATS but not field-changing.

| Rank | # | Direction | Best venue |
|------|---|-----------|-----------|
| 18 | 8 | Tensor product → variance of flips | ICML |
| 19 | 21 | Critical ε for bimodality onset | NeurIPS |
| 20 | 37 | Channel capacity of explanation channel | IT Workshop |
| 21 | 49 | Basin volumes → explanation probability | ICML |
| 22 | 57 | Causal SHAP under framework | CLeaR |
| 23 | 61 | BMA as weighted orbit averaging | Bayesian Analysis |
| 24 | 64 | Bagging ≈ orbit averaging (when?) | JMLR |
| 25 | 67 | CV instability as Rashomon indicator | AISTATS |
| 26 | 41 | Entropy of Rashomon set → instability | COLT |
| 27 | 53 | NTK regime → stable explanations | ICLR |
| 28 | 62 | Posterior predictive = Bayes-optimal stable explanation | Bayesian Analysis |
| 29 | 72 | Augmentation-averaged GradCAM | CVPR |
| 30 | 73 | Paraphrase invariance as symmetry | EMNLP |
| 31 | 6 | Branching rules for intermediate ρ | COLT |
| 32 | 13 | Weyl character formula for O(d) | J. Rep Theory |
| 33 | 31 | TDA on SHAP vectors | ICML |
| 34 | 38 | Fisher-Rao metric on Rashomon manifold | Info Geometry |
| 35 | 42 | KL divergence between models' SHAP | AISTATS |
| 36 | 52 | SGD mixing time on Rashomon set | COLT |
| 37 | 58 | SCMs as explanation systems | UAI |
| 38 | 66 | Multiple testing correction for SHAP | Biometrika |
| 39 | 71 | Financial model risk from Rashomon | Quantitative Finance |
| 40 | 85 | Importance-weighted DASH | TMLR |
| 41 | 89 | Symbolic DASH (closed-form for linear) | JMLR |
| 42 | 92 | DASH for black-box models | AAAI |

---

## Tier C: Incremental / Niche (30 directions)

Publishable but narrow audience.

| Rank | # | Direction | Notes |
|------|---|-----------|-------|
| 43 | 3 | Character values → per-model deviation | Niche representation theory |
| 44 | 4 | Schur orthogonality → independence | Clean but small |
| 45 | 5 | Plancherel measure → flip distribution | Beautiful math, hard to test |
| 46 | 9 | Frobenius reciprocity | Very technical |
| 47 | 10 | Hecke algebra for partial Rashomon | Hard to compute |
| 48 | 12 | Molien's theorem → stable queries | Nice but follows from η law |
| 49 | 14 | Schur-Weyl duality | Deep but no clear prediction |
| 50 | 16 | Peter-Weyl for compact G | Extension of η law |
| 51 | 22 | Compatibility complex Euler char | Topological invariant |
| 52 | 24 | Morse theory on loss landscape | Hard to formalize |
| 53 | 25 | Geodesics on Rashomon manifold | Interesting geometry |
| 54 | 26 | Curvature of Rashomon boundary | Hard to compute |
| 55 | 30 | Discrete Morse theory | Computational tool |
| 56 | 32 | Mapper algorithm visualization | Practical tool |
| 57 | 36 | MI measures faithfulness/stability | Reframing |
| 58 | 40 | Data processing inequality | Follows from framework |
| 59 | 43 | MDL for explanations | Coding theory connection |
| 60 | 44 | H(ranking|group) = 0 | Reformulation of Noether |
| 61 | 45 | EXIT charts for DASH | Coding theory visualization |
| 62 | 46 | Source coding for explanations | Information theory |
| 63 | 48 | Lyapunov stability | Reframing |
| 64 | 51 | Saddle points → ambiguous explanations | Hard to test |
| 65 | 55 | Edge of chaos | Speculative |
| 66 | 59 | Sufficient statistics experimental validation | Validation only |
| 67 | 69 | Molecular symmetry → stability | Domain-specific |
| 68 | 74 | Recommender item symmetry | Domain-specific |
| 69 | 76 | Autonomous driving sensors | Domain-specific |
| 70 | 78 | Time series stationarity | Domain-specific |
| 71 | 80 | Supply chain substitutability | Domain-specific |
| 72 | 82 | Ecology species redundancy | Domain-specific |

---

## Tier D: Primarily Theoretical / Not Empirically Testable (20 directions)

Mathematically interesting but won't produce empirical results.

| Rank | # | Direction | Notes |
|------|---|-----------|-------|
| 73 | 11 | Representation stability (FI-modules) | Asymptotic theory |
| 74 | 15 | Maschke's theorem | Already implicit |
| 75 | 17 | Regular representation | Conceptual |
| 76 | 18 | Automorphism groups | Abstract |
| 77 | 23 | Nerve theorem | Topological |
| 78 | 27 | Fiber bundle obstruction | Differential topology |
| 79 | 28 | Stiefel-Whitney classes | Characteristic classes |
| 80 | 29 | Homotopy groups | Algebraic topology |
| 81 | 33 | Alexander duality | Beautiful but abstract |
| 82 | 34 | Borsuk-Ulam | Continuous bilemma |
| 83 | 39 | Cramér-Rao = quantitative bilemma | Already proved |
| 84 | 93 | Abramsky no-go hierarchy | Foundational |
| 85 | 94 | Presheaf formulation | Category theory |
| 86 | 95 | Galois connection | Category theory |
| 87 | 96 | Operadic enrichment | Category theory |
| 88 | 97 | Topos-theoretic | Category theory |
| 89 | 98 | Monoidal category | Category theory |
| 90 | 99 | 2-categorical naturality | Category theory |

---

## Tier E: Engineering / Implementation (10 directions)

Useful for practitioners, not for Nature.

| Rank | # | Direction | Notes |
|------|---|-----------|-------|
| 91 | 83 | FFT orbit averaging | O(n log n) speedup |
| 92 | 84 | Approximate orbit averaging | Hoeffding bound |
| 93 | 86 | Online DASH | Incremental |
| 94 | 87 | Pruned DASH | Efficiency |
| 95 | 88 | Parallel DASH | MapReduce |
| 96 | 90 | Streaming DASH | Non-stationary |
| 97 | 70 | Climate CMIP6 | Needs data access |
| 98 | 77 | AlphaFold | Needs data access |
| 99 | 79 | Quantum chemistry | Needs domain expertise |
| 100 | 81 | Education assessment | Niche domain |

---

## Summary: Where Are the Knockouts?

| Tier | Count | Knockout? |
|------|-------|-----------|
| S (genuine knockout) | 5 | Yes — any of these could be Nature |
| A (high impact) | 12 | No — strong but not transformative |
| B (solid contribution) | 25 | No — good papers, not field-changing |
| C (incremental) | 30 | No |
| D (theoretical) | 20 | No (for Nature; yes for math journals) |
| E (engineering) | 10 | No |

**The five Tier S directions share a common property: they CONNECT the impossibility theorem to an existing major research program.** Loss landscape geometry (#19, #50), training algorithms (#56), information theory (#35), and lottery tickets (#54) are all active areas with large communities. The knockout comes from showing the impossibility theorem is relevant to THEIR work, not just to XAI.

The other 95 directions improve, extend, or apply the framework. They're valuable for building a research program but won't individually change how the field thinks.
