# Mock Hostile Reviews and Author Responses

Prepared for: "The Universal Explanation Impossibility: No Explanation Is Faithful, Stable, and Decisive Under Underspecification"

---

## Review 1 — The Theorist

**Review:**
The proof of Theorem 1 is essentially pigeonhole: two things that must be equal (stability) and must be different (faithfulness + decisiveness under Rashomon) cannot both hold. This is a four-line argument that follows immediately from the definitions, which appear to have been reverse-engineered to make the theorem true. Moreover, the resolution via G-invariant aggregation is well-known from classical invariant decision theory --- Hunt and Stein (1948) already established that invariant estimators are optimal under group symmetry. What, precisely, is the contribution beyond dressing up a tautology in Lean syntax?

**Author Response:**
We appreciate the directness. The proof IS short --- that is a feature, not a bug. The contribution is not the proof technique but what it proves and how tightly it proves it. First, the three tightness theorems (Lean: `tightness_faithful_decisive`, `tightness_faithful_stable`, `tightness_stable_decisive`) show that each pair of properties IS simultaneously achievable, with explicit non-trivial witnesses. This means the definitions are not vacuous --- the impossibility is tight, not an artifact of overstrong axioms. Second, we performed axiom substitution testing: three alternative formalizations of faithfulness (equality-based, non-contradiction-based, ranking-preservation-based) all yield the same impossibility, demonstrating robustness to definition choice. Third, the contribution is unification: six previously independent impossibility results (Bilodeau et al. for attribution, Chouldechova for fairness, Verma & Pearl for causal discovery, etc.) are shown to share a single structural cause. Hunt-Stein tells you that invariant estimators are optimal GIVEN that you already know you are in an invariant decision problem --- it does not tell you which problems ARE invariant decision problems. Our contribution is identifying explanation impossibility as an invariant decision problem across six domains, and verifying the entire framework in Lean 4 (67 files, 322 theorems, 60 axioms, 0 sorry), which eliminates any ambiguity about whether the definitions "were chosen to make the theorem true."

**Assessment: FULLY CONVINCING.** The tightness theorems and axiom substitution directly rebut the "rigged definitions" concern, and the Hunt-Stein distinction is sharp.

---

## Review 2 — The Empiricist

**Review:**
The experimental methodology is unconvincing. The attention experiment uses weight perturbation (adding Gaussian noise to model parameters), not true retraining from different random seeds. This is a much weaker form of model multiplicity --- perturbed models are not independently trained models. The concept probe experiment uses `load_digits`, a toy dataset with 8x8 pixel images, which tells us nothing about concept instability in real vision models on ImageNet-scale data. The counterfactual experiment similarly uses XGBoost on tabular data with artificial perturbation. These experiments do not demonstrate real-world impact of the impossibility.

**Author Response:**
We acknowledge the perturbation-vs-retraining limitation and agree it should be noted more prominently. However, we emphasize three points. First, the theoretical result does not depend on the experiments at all --- `explanation_impossibility` has zero axiom dependencies and proves the impossibility generically from the Rashomon property as a hypothesis. The experiments are illustrative, not foundational. Second, real-world evidence of the instability we formalize already exists in the literature: D'Amour et al. (2022) demonstrate underspecification across clinical, NLP, and vision pipelines using full retraining; Krishna et al. (2022) show SHAP disagreement across retraining seeds on production-scale models; Jain & Wallace (2019) show attention instability with full retraining on standard NLP benchmarks. Our experiments complement rather than replace this evidence. Third, our negative controls (experiments where the Rashomon property does NOT hold, e.g., well-specified linear regression) confirm near-zero instability, supporting the claim that the observed instability is Rashomon-driven rather than an artifact of the perturbation methodology. We will add a limitations paragraph making the perturbation caveat explicit and citing the retraining-based evidence more prominently.

**Assessment: PARTIALLY CONVINCING.** The theoretical independence argument is strong, and the external citations help. However, the paper should add at least one retraining-based experiment (e.g., GBDT SHAP across 10 seeds) to close the gap. **GAP: Add a true-retraining experiment or prominently cite and defer to the companion attribution paper's retraining experiments.**

---

## Review 3 — The Prior-Work Expert

**Review:**
The claimed contribution of "unification" overstates the novelty. Bilodeau et al. (2024) already proved impossibility for post-hoc feature attributions under collinearity. Jain & Wallace (2019) demonstrated that attention weights are not reliable explanations. Chouldechova (2017) proved the impossibility of simultaneous fairness metrics. Verma & Pearl (1991) established Markov equivalence classes decades ago. This paper repackages these known results under a common notation and calls it a "universal theorem," but each domain's impossibility was already known. The abstraction adds no new insight to any individual domain.

**Author Response:**
We gratefully acknowledge each of these contributions --- they are foundational, and our paper builds directly on them. Our claim is not that any individual impossibility is new, but that the UNIFICATION is new and consequential. Consider the analogy: Noether's theorem does not discover any individual conservation law, but identifying that ALL conservation laws arise from symmetries is itself a major insight. Our unification provides three things none of the prior works individually provide. (1) A single proof and a single mechanism: one four-line argument, with the Rashomon property as the sole structural cause, closes all six impossibilities. This eliminates the "escape by switching methods" fallacy --- the practitioner who abandons SHAP for attention maps, or attention maps for counterfactuals, encounters the same impossibility for the same reason. (2) A uniform resolution: G-invariant aggregation (DASH for attributions, CPDAGs for causal discovery, ensemble probes for concepts) is shown to be the structurally optimal response in all six domains, not an ad hoc fix for each. None of the prior works provide a resolution, let alone a uniform one. (3) Formal verification: the entire framework is mechanically verified in Lean 4. None of Bilodeau et al., Jain & Wallace, Chouldechova, or Verma & Pearl provide machine-checked proofs, and the unification makes such verification tractable (one proof covers six domains rather than six separate formalizations).

**Assessment: FULLY CONVINCING.** The "escape by switching methods" point is the key insight that distinguishes unification from repackaging, and the resolution + verification arguments are strong.

---

## Review 4 — The Skeptic

**Review:**
The definitions appear rigged to produce the impossibility. "Faithful" means "never contradicts the native explanation" --- but real explanations approximate, simplify, and sometimes deliberately deviate from model internals (e.g., LIME fits a local linear model that intentionally differs from a neural network's computation). "Decisive" means "inherits all incompatibilities" --- but no practitioner expects a single explanation to resolve every possible query. These are idealized properties that no real explanation system attempts to satisfy simultaneously, so proving their joint impossibility is vacuous.

**Author Response:**
This concern motivated our axiom substitution test. We formalized three alternative versions of faithfulness (strict equality, non-contradiction, ranking-preservation) and three alternative versions of decisiveness (full resolution, pairwise resolution, top-k commitment). All nine combinations yield the same impossibility, because the structural driver is the Rashomon property, not the specific formalization of the desiderata. Beyond robustness testing, each property is motivated by practice and regulation. Faithfulness (non-contradiction) is the MINIMAL requirement --- if your explanation actively contradicts what the model computes, it is worse than useless. The EU AI Act's "meaningful explanation" requirement and SR 11-7's "developmental evidence" requirement both presuppose at least non-contradiction. Decisiveness (inherits incompatibilities) captures what practitioners actually want: if the model's own structure distinguishes feature A from feature B, the explanation should too. The tightness theorems prove the definitions are not vacuous: `tightness_faithful_decisive` shows that E = explain is faithful and decisive; `tightness_faithful_stable` shows that a neutral constant map is faithful and stable; `tightness_stable_decisive` shows that a maximally committal constant map is stable and decisive. Each PAIR is achievable with concrete witnesses --- only the TRIPLE is impossible. The Lean formalization (67 files, 0 sorry) eliminates any logical ambiguity about whether the definitions "were designed to make the theorem true."

**Assessment: FULLY CONVINCING.** The axiom substitution, tightness witnesses, and regulatory motivation together close this objection.

---

## Review 5 — The Lean Expert

**Review:**
The axioms do all the substantive work. Examining ExplanationSystem.lean, the `rashomon` field axiomatizes the existence of observationally equivalent configurations with incompatible explanations --- this is the entire content of the theorem, pushed into a hypothesis. The claim that `explanation_impossibility` has "zero axiom dependencies" is misleading: the Rashomon property IS the substantive mathematical content, and you have simply assumed it rather than deriving it. Each instance file (AttentionInstance.lean, CausalInstance.lean, etc.) similarly axiomatizes its domain-specific Rashomon property rather than proving it from more primitive facts. The "322 theorems" count is inflated by trivial consequences of strong axioms.

**Author Response:**
This is by design, and we believe it is the correct formalization strategy. The Rashomon property is an EMPIRICAL fact about each domain, not a logical consequence of more primitive axioms. Attention maps exhibit the Rashomon property because neural networks have weight-space symmetries --- this is an empirical observation about neural network training dynamics, not a theorem of pure mathematics. Causal discovery exhibits the Rashomon property because Markov equivalence classes contain multiple DAGs --- this is a combinatorial fact about graph structure that Verma & Pearl (1991) established. Axiomatizing empirical domain-specific content and proving generic consequences is exactly the right division of labor. Compare: Arrow's impossibility theorem axiomatizes IIA (Independence of Irrelevant Alternatives) and the Pareto condition --- it does not derive them from more primitive facts, because they are normative desiderata, not logical necessities. Our theorem axiomatizes the Rashomon property for the same reason: it is a structural property of the domain, not a consequence of the framework. The "zero axiom dependencies" claim is precise and verifiable: `#print axioms explanation_impossibility` in Lean returns only the Lean kernel axioms (propext, Quot.sound, Classical.choice). The Rashomon property appears as a HYPOTHESIS (a field of the ExplanationSystem structure), not as a standalone axiom. This means the theorem is a genuine conditional: IF the Rashomon property holds, THEN the impossibility follows. Each instance's Rashomon property is then justified empirically (with citations to D'Amour et al., Verma & Pearl, Fisher et al.) and experimentally (Table 1). The 322-theorem count includes the tightness theorems, the G-invariant resolution theorems, the ubiquity bridge, and the design space dichotomy --- none of which are trivial consequences of the Rashomon axiom.

**Assessment: PARTIALLY CONVINCING.** The Arrow's theorem analogy is apt and the hypothesis-vs-axiom distinction is technically correct. However, the paper should be more explicit about the division: "We prove the impossibility generically; each instance's Rashomon property is an empirically justified hypothesis, not a derived theorem." **GAP: Add a paragraph in Section 7 (Lean formalization) explicitly stating the hypothesis/axiom stratification and why axiomatizing the Rashomon property is the correct formalization choice, with the Arrow's theorem analogy.**

---

## Hunt-Stein Connection

The resolution optimality in our framework follows from classical invariant decision theory (Hunt & Stein, 1948). The Hunt-Stein theorem establishes that among all equivariant decision procedures, the best invariant procedure is admissible --- that is, invariant estimators are optimal within the class of equivariant estimators when a group acts on the parameter space.

Our contribution is NOT the optimality of invariant procedures (which is classical) but rather:

1. **Identifying explanation impossibility as an invariant decision problem.** The connection between explanation instability and group invariance is not obvious. That retraining a neural network and getting different attention maps is structurally the same as choosing between Markov-equivalent DAGs, and that both are instances of a group acting on an equivalence class, is the conceptual contribution.

2. **Unifying six domains under this framework.** Hunt-Stein tells you what to do once you know you have an invariant decision problem. It does not tell you which problems ARE invariant decision problems. We show that attribution (DASH), causal discovery (CPDAG), attention (averaged rollout), counterfactual (robust recourse), concept probes (subspace averaging), and model selection (Rashomon set reporting) are all instances of the same invariant structure.

3. **Making the resolution constructive.** The Hunt-Stein theorem is an existence result. We provide concrete G-invariant procedures for each domain: DASH averages SHAP values over the Rashomon set for attributions; CPDAG reports the equivalence class rather than a single DAG for causal discovery; ensemble probes average concept activation vectors over equivalent representations.

The correct framing is: "The impossibility is new (unification of six domains under one theorem). The resolution strategy is classical (Hunt-Stein). The identification of explanation problems as invariant decision problems, and the construction of domain-specific invariant procedures, is the bridge between the two."

---

## Summary of Assessments

| Reviewer | Verdict | Gap? |
|----------|---------|------|
| 1 - The Theorist | FULLY CONVINCING | None |
| 2 - The Empiricist | PARTIALLY CONVINCING | Add true-retraining experiment or prominently cite companion paper's retraining results |
| 3 - The Prior-Work Expert | FULLY CONVINCING | None |
| 4 - The Skeptic | FULLY CONVINCING | None |
| 5 - The Lean Expert | PARTIALLY CONVINCING | Add explicit hypothesis/axiom stratification paragraph in Lean section with Arrow analogy |

**Action items before submission:**
1. Add a limitations paragraph acknowledging perturbation-vs-retraining and citing D'Amour et al. and Krishna et al. for real-world retraining evidence (addresses Reviewer 2).
2. Add a paragraph in the Lean formalization section explicitly discussing the hypothesis/axiom division and why axiomatizing the Rashomon property is correct (addresses Reviewer 5).
