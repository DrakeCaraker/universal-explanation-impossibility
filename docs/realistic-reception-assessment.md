# Realistic Reception Assessment

Generated: 2026-04-02
Methodology: Non-adversarial, full-spectrum audience analysis. No scores, no "weakest claim," no adversarial red-teaming. Designed to predict actual reception, not identify flaws.

---

## Phase 1: Historical Precedent

### Arrow (1951) — Social Choice and Individual Values
- **What it proved**: No voting system satisfies IIA + Pareto + non-dictatorship simultaneously.
- **Initial reception**: Mixed within economics — elegant but abstract. Some questioned practical import ("so what? we already knew voting was messy").
- **Published**: Monograph (Wiley), later recognized as foundational.
- **What drove impact**: The theorem's clean structure made it teachable; its generality meant it applied everywhere; the welfare economics community needed formal language for disagreements they'd been having informally. Slow burn — textbook-canonical by the 1970s.

### Chouldechova (2017) — Fair prediction with disparate impact
- **What it proved**: Calibration + balance + equal FPR impossible when base rates differ.
- **Initial reception**: Strong at FAccT, immediately recognized. ProPublica's COMPAS analysis (2016) had primed the audience.
- **Published**: Big Data (not a top ML venue).
- **What drove impact**: Timeliness (algorithmic fairness was politically salient), clean enough for non-mathematicians, gave formal backing to practitioner intuition. ~1200 citations in 7 years.

### Bilodeau et al. (2024) — Impossibility of feature attribution
- **What it proved**: Completeness + linearity impossible for attribution methods.
- **Initial reception**: Respectful but contained within XAI theory community.
- **Published**: PNAS.
- **What drove impact**: Engaged the SHAP community directly. Limitation: somewhat expected (practitioners knew SHAP had tradeoffs), no constructive fix offered. Likely 50-100 citations in 5 years.

### Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions
- **What it proved**: Unified multiple explanation methods via Shapley values.
- **Initial reception**: Enthusiastic at NeurIPS.
- **Published**: NeurIPS.
- **What drove impact**: THE SOFTWARE (`shap` library). The paper gave theoretical grounding; the library made it the default tool. Practitioners adopted it without reading the paper. ~25,000 citations — almost entirely library-driven.

### Nipkow (2009) — Social Choice Theory in HOL
- **What it proved**: Formalized Arrow's theorem in Isabelle/HOL.
- **Initial reception**: Appreciated within formal verification community; essentially no uptake in economics or political science.
- **Published**: Journal of Automated Reasoning.
- **What drove impact**: Demonstrated proof assistants could handle social science theorems. Methodology precedent. ~80 citations, almost all from formal methods.

### Where "The Attribution Impossibility" Sits

It occupies a position most comparable to **Chouldechova** — a clean impossibility in an applied domain that is politically and practically salient — with the added distinction of formal verification (like Nipkow) and a companion library (like Lundberg).

The core impossibility is simpler than Arrow but less general. The formalization is more ambitious than Nipkow (188 theorems vs. a single theorem). The practical tools (DASH, diagnostics) give it a Lundberg-like software path to adoption.

**Honest comparison**: The core impossibility is not as deep as Arrow or as surprising as Chouldechova. Its strength is in the COMBINATION — impossibility + quantification + resolution + verification + software — which no single predecessor achieves. Whether this combination drives impact depends on whether the DASH library gets adopted, not on whether the theorem gets cited.

---

## Phase 2: Ten Realistic Readers

### Reader 1: The SHAP Practitioner
A senior data scientist at a fintech company. Uses SHAP daily. Has noticed that retraining shifts feature rankings.

- **What they read**: Abstract + intro + experiments. Skips proofs entirely.
- **First reaction**: "Finally, someone proved this is real." The 48% flip rate on Breast Cancer resonates — they've seen similar numbers on their own data.
- **What they do**: Downloads dash-shap, tries the F5 screen on their credit model. If it flags the pairs they already suspected, they trust it. If it works, tells their team lead.
- **Do they cite it**: In an internal model risk document — yes. In a published paper — only if they write one.
- **This is the most important reader for actual impact.**

### Reader 2: The JMLR Reviewer
An associate professor in statistical ML. Assigned this paper. Has 3 weeks. Reads everything carefully.

- **What they focus on**: The axiom system first. Are the 18 axioms reasonable? Then the Design Space exhaustiveness proof. Then the SBD — is it genuinely new vs. classical invariant decision theory?
- **Their honest assessment**: "The core theorem is trivial — it's literally a four-line contradiction from the Rashomon property. But the Design Space Theorem is nontrivial, the quantitative bounds are useful, and the formalization is genuine. The proportionality axiom (CV=0.66) is the weak link."
- **Their recommendation**: Major revision. Asks for cleaner separation between what is proved (zero axiom dependencies) vs. derived (conditional on axioms). Likely accepts on second round.

### Reader 3: The NeurIPS Reviewer
A postdoc with 200 papers. Spends 20 minutes on this one.

- **What they notice**: The title. The "coin flip" line in the abstract. The trilemma figure (if visible). The "188 theorems, 0 sorry" claim. The three experimental figures.
- **Their gut reaction**: "Interesting but the core result is basically 'you can't stably rank things that are essentially the same.'" Gives credit for completeness and the Lean verification.
- **Their score**: 5-6 out of 10. High marks for completeness and clarity, moderate for novelty. The paper's fate depends on the other two reviewers and whether one is a champion.

### Reader 4: The Model Risk Officer
VP at a top-10 US bank. Responsible for SR 11-7 compliance. Doesn't read papers — hears about this from a vendor or consultant.

- **What they care about**: "Can this help me document model risk for SHAP-based explanations?"
- **What they extract**: The instability disclosure template. The F5 screen workflow. The "known and foreseeable circumstance" argument for EU AI Act.
- **Does this change their process**: Yes — if the diagnostics work on their internal data, this becomes part of the model risk framework. They don't need the math; they need the workflow and the regulatory language.
- **This is where the regulatory impact comes from, and it is real.**

### Reader 5: The Formal Methods Researcher
Active in Lean/Mathlib. Sees the repo on GitHub or the Lean Zulip.

- **What impresses them**: 188 theorems, 0 sorry, clean axiom system. The Gaussian CDF symmetry proofs (Φ(0)=1/2 via NoAtoms + prob_compl_eq_one_sub). The MulAction usage for the SBD.
- **What doesn't**: The proofs are mostly short tactic scripts — not deep Lean engineering. The axiom system does heavy lifting; the proofs are consequences. This is a well-executed application of Lean, not a Mathlib contribution.
- **Do they build on it**: Possibly — may contribute a PR deriving the spearman_classical_bound. More likely, they note it as an example of "Lean for ML theory" and move on.

### Reader 6: The ML PhD Student
Starting a thesis on XAI. Looking for open problems and positioning.

- **How they use this paper**: As a roadmap. The open problems section lists 5+ thesis-worthy directions (adaptive DASH, SBD for new domains, deriving axioms algorithmically, generalizing from ρ>0 to mutual information).
- **What thesis chapters it enables**: Chapter 2 (background — cite the impossibility), Chapter 3 (extension — SBD for feature selection), Chapter 4 (experiments — extend to LLM attention with proper methodology).
- **Opportunity or closed topic**: Opportunity. The Design Space Theorem opens more questions than it closes. The SBD is explicitly designed to be reused.
- **This reader is the most likely source of follow-on citations.**

### Reader 7: The Skeptic
A senior ML researcher who thinks SHAP instability is well-known. Believes confidence intervals already handle the problem.

- **Their strongest objection**: "This is just saying that correlated features have unstable rankings, which everyone knows. Confidence intervals on SHAP values already communicate uncertainty. The impossibility is formalizing the obvious."
- **Is their objection valid**: Partially. The qualitative intuition IS known. But two things are sharper than intuition: (a) the flip rate is literally 1/2 (not just "noisy"), and (b) confidence intervals don't fix the ranking instability — they tell you the interval is wide, but the ranking still flips. The Design Space Theorem goes beyond CI-based reasoning by proving exactly two families exist.
- **What would change their mind**: The conditional SHAP result (switching methods doesn't help when causal effects are equal) and the Pareto optimality of DASH (it's provably the best you can do, not just "another approach").

### Reader 8: The Fairness/Accountability Researcher
FAccT community. Works on proxy discrimination in lending.

- **What catches their attention**: The fairness audit impossibility — "SHAP-based proxy audits are coin flips." The (1/2)^K intersectional compounding. The adverse action notice example (income vs. DTI).
- **How it connects**: Directly. Proxy discrimination detection via SHAP is common practice. This paper proves it's unreliable under collinearity — which is exactly the case that matters (protected attribute is correlated with non-protected features).
- **Do they cite it**: Yes, in their next paper on robust fairness auditing. The coin-flip result for single-model audits is a citable, quotable finding. This is a genuine contribution to the fairness literature.

### Reader 9: The Science Journalist
Writing "Can We Trust AI Explanations?" for a mainstream outlet. Needs a concrete example.

- **Can they understand the result**: From the README — yes. The "if you've ever retrained a model" framing is accessible. From the abstract — partially (too technical for general audience).
- **The headline they'd write**: "Researchers Prove That Widely Used AI Explanation Tool Gives Random Answers for Correlated Features"
- **Would they feature it**: Yes, if they're writing the article during the arXiv posting window. The credit model example and adverse action story are exactly what they need. This coverage, if it happens, reaches more people than the paper itself.

### Reader 10: The Person Who Skims and Moves On
90% of the audience. Sees the title on arXiv or social media.

- **What impression they form**: "SHAP rankings are unreliable for correlated features. Good to know." Files it in mental category of "SHAP has limitations."
- **Do they bookmark it**: If the title or a tweet made them stop — maybe. "The Attribution Impossibility" is memorable enough to stick. "Faithful, Stable, Complete: Pick Two" would be more memorable.
- **What would make them stop and read more**: A tweet from a prominent ML figure. A blog post with the before/after bar chart. A Hacker News discussion. The paper itself doesn't reach this person — the surrounding media does.

---

## Phase 3: The Importance Question

### 1. Is this result important?

Yes, but to different audiences for different reasons:
- **To practitioners**: The diagnostic workflow is immediately useful. They can test for instability today.
- **To regulators**: The impossibility is directly relevant to EU AI Act and SR 11-7 compliance. It gives formal language to an informal concern.
- **To the XAI theory community**: The Design Space Theorem is a genuine structural contribution that subsumes the impossibility as a special case.
- **To the fairness community**: The proxy audit impossibility is a concrete, citable result.

The core impossibility alone is not "important" in the sense of being surprising — it formalizes something practitioners already suspected. The importance lies in the package: making it precise, quantifying it, proving it cannot be fixed (except by DASH), and providing tools.

### 2. Is this result surprising?

The core impossibility: **no, not really**. Anyone who has retrained a model and seen feature rankings change has already internalized this.

What IS surprising:
- The quantitative precision: literally a coin flip (50%), not just "noisy." The ratio diverges as 1/(1-ρ²).
- The Design Space Theorem: exactly two families and nothing else. This is sharper than intuition would suggest.
- The conditional SHAP result: switching methods doesn't help when causal effects are equal.
- The relaxation path convergence: both "drop faithfulness" and "drop completeness" lead to the same solution (DASH). This doesn't happen in Arrow's setting.

The formalization adds certainty, not surprise. Its value is that it closes the door on "maybe there's a clever trick we're missing."

### 3. Does the formalization matter?

**For the core impossibility**: No. The proof is four lines. Anyone can check it by hand.

**For the full package (188 theorems, Design Space exhaustiveness, SBD instances)**: Yes, meaningfully:
- The formalization caught two genuine logical inconsistencies in the original axiom system.
- The zero-sorry guarantee means a reviewer can skip proof-checking and focus on whether the axioms are appropriate.
- The axiom consistency proof (Fin 4 model) demonstrates the system is non-vacuous.

**For the paper's credibility with skeptical reviewers**: The formalization is a strong differentiator. No other XAI impossibility has this level of verification.

**For citation impact**: Negligible. Almost nobody will run `lake build`. The citations will come from the result, not the verification.

### 4. Is DASH "just averaging"?

Both. The operation is simple ("average SHAP across models"). The contribution is:
- Proving it's the minimum-variance unbiased estimator (Cramér-Rao / Titu's lemma)
- The tight ensemble size formula M_min = ⌈2.71·σ²/Δ²⌉
- The Pareto optimality proof (no method beats DASH on stability without sacrificing more)
- The diagnostic workflow (F5 screen → F1 validate → DASH resolve)

"Just averaging" is like saying OLS is "just matrix inversion." The value is in understanding exactly when and why it is optimal. The practical contribution is the WORKFLOW, not the averaging itself.

### 5. Is the SBD a real technique?

Yes. It applies to a well-defined class of problems (symmetric decision problems with finite-sample noise). The three instances are genuinely distinct — different symmetry groups (S₂ transpositions, S₂ permutations, variable-size CPDAG automorphisms). The connection to classical invariant decision theory (Hunt-Stein) is legitimate.

**40% odds** of being cited AS a technique (not just as a component of this paper) within 5 years. Whether it gets adopted depends on whether other researchers find new instances. Likely candidates: feature selection instability, hyperparameter sensitivity, cross-validation ranking instability.

### 6. What's the shelf life?

The core impossibility is **permanent** — it will be true as long as people use correlated features. The quantitative bounds are specific to current model classes and may need updating as architectures evolve. The DASH workflow is useful now and will remain useful as long as SHAP (or similar attributions) are standard practice.

**If SHAP is supplanted** by something fundamentally different (e.g., mechanistic interpretability for all models), the paper's practical relevance fades but the theoretical result remains.

**Best estimate**: Still cited in 2031, but probably as background rather than as a current contribution. The same trajectory as Chouldechova — still important, still cited, but the active research has moved to consequences and extensions.

### 7. Realistic citation trajectory

| Scenario | Citations (5yr) | Probability | Determining Factors |
|----------|----------------|-------------|---------------------|
| **Pessimistic** | ~20 | 20% | Published in venue that doesn't reach practitioners; dash-shap not adopted; XAI moves entirely to LLM explanations; SHAP becomes less central |
| **Expected** | 80-150 | 60% | Published in JMLR; dash-shap gains moderate adoption; cited by fairness community, model risk literature, Rashomon-set literature; a few follow-on papers extend SBD; 37-dataset survey becomes reference point |
| **Optimistic** | 300+ | 20% | Prominent ML figure amplifies; journalist article reaches wide audience; regulatory body references in guidance; dash-shap becomes standard part of SHAP ecosystem |

**What determines which trajectory**: Library adoption is the single most important factor. Second: timing with AI regulation (EU AI Act implementation 2025-2027 overlaps perfectly). Third: whether the Rashomon set / model multiplicity literature coalesces into a recognized subfield.

---

## Phase 4: The Combined Package

Most papers are just papers. This is:
- A paper (3 venue-optimized versions: 42pp JMLR, 10pp NeurIPS, 59pp definitive)
- 188 Lean 4 theorems (0 sorry, 36 files)
- A Python package (dash-shap with stability API: screen, validate, consensus, report)
- 33 experiment scripts (all reproducible on a laptop)
- A diagnostic workflow (F5→F1→DASH, validated on 11 datasets)
- A README that onboards 4 different audiences
- Financial case studies with regulatory framing (German Credit, Taiwan CC, Lending Club)
- A trilemma figure designed for textbook reproduction

**Does the combination matter?** Yes — significantly more than any piece alone.

- The theorem alone risks "so what?"
- The Lean proofs alone interest only the formal methods community
- The Python package alone is a tool without theoretical backing
- The diagnostics alone are an engineering contribution
- The financial case studies alone are applied work

Together, they form something unusual: a complete pipeline from mathematical proof to deployable tool.

**The Lundberg comparison**: That paper had a clean theoretical contribution (unifying explanation methods via Shapley values) paired with an immediately useful library. The theoretical contribution was modest; the library was transformative. Here, the theoretical contribution is arguably deeper (an impossibility theorem is stronger than a unifying framework), but the library faces a harder adoption problem: it requires practitioners to change their workflow (train 25 models instead of 1), while `shap` required only adding a function call after training.

**The README**: Targeting four audiences simultaneously is unusually well-crafted. Most researchers write for one audience. This increases the surface area for discovery. The README may be the single most impactful artifact in the repository for initial adoption.

**The 33-script reproducibility package**: Thorough but will be used by fewer than 10 people. Its value is as a credibility signal, not as a practical resource.

---

## Phase 5: The Honest Summary

### What this work is

A formally verified impossibility theorem establishing that no feature ranking can be simultaneously faithful to a model, stable across retrains, and complete when features are correlated. The core result is simple — a four-line contradiction from the Rashomon property — but it is embedded in substantial infrastructure: quantitative bounds showing the attribution ratio diverges as 1/(1-ρ²) for gradient boosting and is infinite for Lasso; a Design Space Theorem characterizing the complete set of achievable attribution methods as exactly two families; a constructive resolution (DASH ensemble averaging) proved to be the minimum-variance unbiased estimator; a general proof technique (the Symmetric Bayes Dichotomy) demonstrated across three domains with different symmetry groups; practical diagnostics validated on 11 datasets and 3 GBDT implementations; and 188 machine-checked Lean 4 theorems with zero sorry. It is accompanied by a Python package (dash-shap) with a diagnostic workflow (F5→F1→DASH) designed for production deployment.

### Who will care and why

Three groups will care most. **First**, data scientists and model validators at financial institutions, healthcare organizations, and tech companies who use SHAP and have encountered feature ranking instability — they get a diagnosis (the impossibility), a test (the F1/F5 diagnostics), and a fix (DASH). **Second**, the regulatory and governance community — the paper provides formal language for what was previously an informal concern, directly relevant to EU AI Act Art. 13(3)(b)(ii) and SR 11-7 model risk management. The fairness audit impossibility (proxy discrimination audits are coin flips) connects directly to ECOA compliance. **Third**, the XAI theory community — the Design Space Theorem, the SBD generalization, and the conditional SHAP impossibility are substantive contributions that will generate follow-on work. The formal methods community will notice the Lean formalization but is unlikely to engage deeply with the applied result. The broader ML community will absorb the headline — "SHAP rankings for correlated features are unreliable" — without reading the paper.

### What determines whether this matters in 10 years

Three factors will determine lasting impact. **First**, whether the dash-shap package gets integrated into the SHAP ecosystem or adopted by a major ML platform — tooling drives practice more than theorems do. If `dash_shap.screen()` becomes a standard post-training check, the work's practical impact exceeds its academic impact by an order of magnitude. **Second**, whether AI regulation actually mandates explanation stability or robustness testing — if the EU AI Act's implementing acts require demonstrating explanation reliability, this work becomes a compliance reference, potentially the Chouldechova of explainability. **Third**, whether the field's attention stays on feature attribution at all — if interpretability research moves entirely to mechanistic interpretability of large language models, the work's practical relevance narrows to tabular ML in regulated industries (banking, insurance, healthcare), which is a large but less prestigious audience. The theorem itself is permanent mathematics and will not be invalidated, but whether anyone needs it depends on whether the problems it addresses remain central to the field's concerns.
