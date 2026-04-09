# Impact Assessment

**Project:** The Attribution Impossibility — Lean 4 Formalization
**Audience:** Co-authors, potential collaborators, grant reviewers
**Date:** April 1, 2026

---

## 1. What We Proved and Why It Matters

### The core result, in plain English

When two or more features in a dataset are correlated, machine learning models trained on the same data will sometimes rely more heavily on one feature, sometimes on another. This is not a bug — it is a mathematical consequence of the data having multiple "equivalent" explanations. From this observation, we prove a fundamental impossibility: any method that tries to rank features by importance cannot simultaneously be faithful to what individual models actually computed, consistent across equivalent models, and decisive about every pair of features. You must give up at least one of these three properties. This is unavoidable — it follows from the structure of the problem, not from any limitation of current methods.

### Why this matters: three concrete scenarios

**Scenario 1: The hospital readmission model.** A hospital trains an XGBoost model to predict 30-day readmissions. They use TreeSHAP to identify the most important features for a particular patient. Two data scientists run the analysis on independently trained models (same data, same hyperparameters, different random seeds). One model says "previous admissions" is the most important feature. The other says "chronic conditions" is most important. Both models have identical test performance. The data scientists disagree on what to tell the clinical team. Our theorem says: this is not a failure of the data scientists or the method. It is impossible to resolve the disagreement using any faithful, stable, complete ranking method. The only resolution is to use ensemble averaging (DASH), which correctly identifies the two features as tied — reflecting the genuine ambiguity in the data.

**Scenario 2: The debugging scenario.** A regulatory auditor asks a bank to explain why their credit model denied a loan. The bank provides a feature importance ranking. The auditor asks: if you retrained the model with the same data, would the ranking change? For correlated features (income and employment status, say), the answer is yes — potentially dramatically. The bank cannot provide a stable explanation. Our theorem says this is not fixable by using a different attribution method. Any method that correctly reflects the model's attributions will produce unstable rankings for correlated features. The instability is not a limitation of TreeSHAP; it is a property of the underlying problem.

**Scenario 3: The reproducibility crisis.** A machine learning paper reports that feature X is the most important predictor in their dataset, justified by attribution analysis. A reviewer tries to reproduce the result and gets a different ranking. Is this a bug, or normal variation? Our theorem gives a precise answer: for correlated features, variation in feature rankings across independent training runs is expected and irreducible. A paper that reports a single-run feature ranking as a definitive finding, without accounting for this variation, is reporting an artifact of its random seed, not a property of the data.

### The resolution: DASH

The impossibility theorem characterizes the design space but does not leave practitioners without options. DASH (averaging attributions over independently trained models) achieves zero unfaithfulness for collinear features: in a balanced ensemble, all correlated features receive equal average attribution, correctly reflecting that they are interchangeable in the data. The variance of DASH attributions decreases as 1/M, approaching a stable ranking as the ensemble grows. This is proved — not argued, not demonstrated empirically, but proved — in Lean 4 from the ensemble definition and the split-count structure of gradient boosting.

---

## 2. How This Compares to Existing Work

| Paper | Claim | Scope | Formalization | Key difference |
|-------|-------|-------|---------------|----------------|
| Bilodeau et al. (2024) | Post-hoc explanation methods are limited | General | None | Shows fundamental limits; does not characterize the design space or provide a resolution |
| Laberge et al. (2023) | SHAP rankings are unstable under perturbation | SHAP | None | Empirical instability under input perturbation; our instability is across training runs |
| Rudin (2024) | Use interpretable models, not post-hoc explanations | General | None | Argues against post-hoc explanations; we characterize when they fail and how to fix them |
| Huang et al. (2024) | SHAP values change under collinearity | SHAP | None | Describes the phenomenon; does not prove an impossibility or provide a resolution |
| **This paper** | No faithful/stable/complete ranking exists; DASH is optimal | Any attribution method | Lean 4, zero domain-axiom core | Impossibility theorem + design space + resolution + machine-checked proof |

**Key differentiators:**
1. This is, to our knowledge, the first formal impossibility theorem for feature attribution analogous to Arrow's theorem for social choice.
2. The Lean formalization provides a machine-checked certificate for the core result that no existing work provides.
3. We provide a constructive resolution (DASH) alongside the impossibility, with the resolution also proved.
4. The design space characterization is new: we characterize not just that faithful/stable/complete is impossible, but the full set of achievable triples.

---

## 3. The Lean Formalization: What It Means

### The zero-axiom claim

The headline claim is that `attribution_impossibility` — the core theorem — depends on zero domain-specific axioms. This is verified by `#print axioms` in Lean 4, which prints the complete list of axioms a theorem depends on. The output for `attribution_impossibility` is:

```
propext, Classical.choice, Quot.sound, Model, attribution
```

`propext` (propositional extensionality), `Classical.choice` (the axiom of choice), and `Quot.sound` (quotient soundness) are the standard logical foundations of Lean 4. Every theorem in Lean depends on them. `Model` and `attribution` are type declarations — they introduce an abstract type and function without asserting any properties about them.

What this means: the impossibility theorem holds for *any* model type and *any* attribution function. The proof is:

1. Suppose a faithful, stable ranking exists.
2. By the Rashomon property (a *hypothesis* of the theorem, not an axiom), there exist models f and f' that rank features j and k in opposite orders.
3. Faithfulness to f forces the ranking to say j > k.
4. Faithfulness to f' forces the ranking to say k > j.
5. Contradiction.

The entire proof is five lines of Lean. The Rashomon property is the only domain-specific content, and it is a *hypothesis* — the theorem says "if the Rashomon property holds, then..." The theorem works for any domain that satisfies this hypothesis.

### What the axiom system captures

The 7 domain axioms are used by the GBDT instantiation and the quantitative theorems, not by the core impossibility. They encode:

- **Proportionality:** TreeSHAP attributions are proportional to split counts (from Lundberg & Lee 2017, justified by the uniform-contribution model).
- **Split-count structure:** The first-mover feature in sequential boosting gets T/(2-ρ²) splits; other features in the same group get (1-ρ²)T/(2-ρ²) splits. This follows from the Gaussian conditioning argument (Lemma 1 in the supplement).
- **Variance:** Ensemble variance decreases as 1/M (standard result for i.i.d. means).
- **Spearman bound:** The Spearman stability drop is at least (m/P)³ when first-movers differ (classical rank correlation bound).

These axioms are mathematical idealizations, not algorithmic descriptions. They capture the leading-order behavior of XGBoost under Gaussian data and are verified empirically.

### What `#print axioms` does not tell you

`#print axioms` verifies the deductive structure. It does not verify that the axioms accurately describe reality. The Lean proof certifies: *if the stated axioms hold, then the stated conclusions follow*. Whether the axioms are true is a question of domain knowledge and empirical verification, not formal logic.

This distinction is explicit in the paper's "Proof status transparency" paragraph, which distinguishes: proved (zero domain-axiom dependencies), derived (from domain axioms), argued (supplement proof), and empirical (experiments).

---

## 4. How the Field Would React

### XAI community

The XAI community will likely receive this work positively but with some pushback on the axioms. The positive: the impossibility theorem formalizes something many practitioners have observed informally — that SHAP rankings change across runs for correlated features. Having a theorem is better than having an observation. The pushback: "your GBDT axioms are too strong" or "your proportionality assumption fails for non-Gaussian data." These are legitimate concerns and should be addressed in the paper with appropriate caveats and empirical validation.

The DASH resolution will be less controversial. Ensemble averaging is a standard technique; proving it resolves the impossibility is a concrete positive contribution.

### Regulatory and governance community

The regulatory community (GDPR explainability requirements, EU AI Act, NIST AI Risk Management Framework) is actively looking for guidance on when explanations can be trusted. Our impossibility theorem provides a precise answer: single-model explanations of correlated features cannot be trusted to be stable. This is exactly the kind of formal result that regulatory guidance can reference. We expect this paper to be cited in regulatory discussions, though the timeline for that is long (5–10 years for a formal result to reach regulatory language).

### Formal methods community

The Lean formalization will be of independent interest to the formal methods community as an example of machine-checked impossibility proofs for ML systems. The zero-axiom core proof is an unusually clean formal result. We expect interest from the ITP (Interactive Theorem Proving) and FMCAD communities, though this is a secondary audience.

### Statistics community

Statisticians will recognize the connection to identifiability: when features are collinear, model coefficients are not identifiable, and attributions inherit this non-identifiability. The impossibility theorem is a precise statement of this intuition in the context of feature rankings. The connection to Arrow's theorem will be of interest to social choice theorists. We expect some engagement from the JRSS-B / Annals of Statistics community if we pursue the JMLR submission.

---

## 5. Impact for Authors

### What this paper establishes

For all three authors, this paper establishes expertise at the intersection of formal methods and machine learning — a relatively uncrowded area. The Lean formalization is a genuine methodological contribution: it demonstrates that ML impossibility results can be machine-checked, which opens a path to a research program in formally verified ML theory.

The specific results (impossibility theorem + design space + DASH resolution) are a complete treatment of a well-defined problem. "Complete treatment" papers have a different citation trajectory than "first result" papers: they accumulate citations over a longer period because they become the standard reference.

### Career opportunities

- **Teaching:** The Arrow's-theorem-for-ML framing is pedagogically valuable. This paper will be cited in ML courses discussing the limits of explainability.
- **Grants:** The intersection of formal methods and ML is an active NSF/DARPA/IARPA priority. This paper is relevant to proposals on "trustworthy AI" and "formal verification of ML systems."
- **Collaborations:** The Lean formalization creates natural collaborations with the Lean/Mathlib community and with formal-methods researchers interested in ML applications.
- **Industry:** As regulatory scrutiny of ML explanations increases, companies working on explainable AI will find this result relevant. Consulting and advisory opportunities are plausible on a 2–5 year horizon.

### Realistic citation trajectory

Comparable papers in the general area of ML impossibility results and XAI limitations:

- Chouldechova (2017), "Fair prediction with disparate impact" — the foundational fairness impossibility paper. ~2,000 citations by 2026 (9 years). It took 2–3 years to gain significant traction.
- Bilodeau et al. (2024) — too recent for a citation count comparison.
- Arrow (1951), applied to ML — citations in this form are indirect and hard to track.

**Realistic trajectory for this paper:**
- Year 1 (2026–2027): 10–25 citations. NeurIPS papers get initial citations quickly, but formal methods papers take time to penetrate the ML community.
- Years 2–3 (2027–2029): 50–100 citations. If the paper is adopted as a reference for SHAP instability discussions, citations accumulate from survey papers, regulatory documents, and follow-on research.
- Year 5+ (2031+): 200–500 citations. This is the range for papers that become standard references in their area, assuming the XAI field remains active.

The comparison to Chouldechova is instructive: independent researchers without institutional prestige should expect a slower initial citation trajectory (fewer endorsements from prominent names) but comparable long-run impact if the result is correct and the paper is well-written. The JMLR submission (expanded version) will help the long-run trajectory, as JMLR papers are more likely to become canonical references than NeurIPS papers.

---

## 6. Publication Strategy

### Recommended sequence

| Date | Action |
|------|--------|
| April 30, 2026 | Post arXiv preprint (establishes priority, generates feedback) |
| May 4, 2026 | NeurIPS 2026 abstract deadline |
| May 6, 2026 | NeurIPS 2026 paper deadline |
| May 2026 | Begin JMLR submission preparation |
| August 2026 (est.) | NeurIPS decisions |
| September 2026 | If accepted: camera-ready; if rejected, revise for ICML 2027 |
| October 2026 | JMLR submission (expanded, with full supplement integrated) |

### Why arXiv first

Posting to arXiv before NeurIPS establishes priority clearly. It also generates community feedback before the NeurIPS review, which can identify issues that reviewers would otherwise flag. The arXiv posting should be complete and polished — it will be read by reviewers.

### Why NeurIPS + JMLR in parallel

NeurIPS for visibility and speed; JMLR for archival quality and long-term citation impact. NeurIPS papers are frequently superseded or forgotten; JMLR papers become canonical references. The two are complementary. NeurIPS acceptance requires an 8-page version; the JMLR version can include the full supplement and expanded discussion.

### Alternative venues

If NeurIPS rejects: ICML 2027 (January deadline), ICLR 2027 (October 2026 deadline). Both are appropriate venues. ICLR has historically been more receptive to theoretical work with empirical validation.

---

## 7. The Honest Assessment

### What this paper IS

**A comprehensive impossibility theorem with a constructive resolution, formally verified, and empirically validated.** The paper does three things that together constitute a genuine contribution:

1. Proves (in Lean, machine-checked) that faithful/stable/complete attribution is impossible under the Rashomon property — a property that holds for any model class with multiple near-optimal solutions under collinearity.
2. Characterizes the full design space of attribution methods under this constraint, showing exactly what tradeoffs are available.
3. Proves (in Lean, from domain axioms) that DASH resolves the impossibility optimally — with zero unfaithfulness and stability approaching 1.

The combination of impossibility + design space + resolution in a single paper, with formal verification, is unusual. Most papers do one of these three things.

### What this paper IS NOT

**Technically deep proofs.** The core impossibility proof is five lines. The algebra (split-count gap, ratio divergence, attribution symmetry) is elementary. The Lean code is clean, but it is not proving hard mathematical theorems — it is verifying that a clear logical structure holds. Reviewers with a formal methods background will recognize this immediately. The value is not in the technical depth of the proofs; it is in (a) identifying the right structure of the problem, (b) formalizing it cleanly, and (c) connecting the formal theory to empirical evidence.

**A solution to the general XAI problem.** We prove DASH resolves the impossibility for correlated features under the Rashomon property. This does not mean all XAI problems are solved. Many important problems — explaining individual predictions, handling distribution shift, addressing model misspecification — are outside the scope of this theorem.

**Empirically definitive.** The experimental CV of 0.35–0.66 is measured on 11 datasets with specific feature engineering. Other datasets, features, or hyperparameters may show different values. The experiments confirm the theorem's predictions but do not prove them.

### Why it matters anyway

The completeness of treatment is the paper's strongest selling point. The XAI field has a large number of "this method has a problem" papers and a small number of "here is the fundamental structure of the problem" papers. This is the latter. The Lean formalization adds credibility — unlike a paper that proves an impossibility theorem informally, this paper's core result can be inspected by a machine. In a field where proofs in papers are frequently wrong or imprecise, this is a meaningful methodological contribution.

The practical relevance is immediate. The hospital scenario, the regulatory scenario, and the reproducibility scenario are all real problems that practitioners encounter today. A theorem that explains why these problems occur and what to do about them has direct practical value.
