# Publication Strategy — The Attribution Impossibility Program

> Vetted across 4 rounds. All recommendations have confidence ratings.

## Decision: 2 Papers + 1 Companion + Optionals

### Paper A: "The Attribution Impossibility" → NeurIPS 2026

**Content:** Core impossibility + quantitative bounds (GBDT/Lasso/NN/RF) +
DASH resolution + Lean 4 formalization + empirical validation.

**Supplement (5 pages):** Axiom system, proof architecture, inconsistencies
found, Gaussian conditioning, SymPy verification, R1 (Rashomon from
symmetry), R2 (non-degeneracy), R3 (Rashomon inevitability), F1
(unfaithfulness ≥ 1/2), F2 (optimal unfaithful = population-level),
F3 (path convergence, trilemma → M-axis).

**Status:** Ready. 9 pages main + 5-page supplement.

**Timeline:**
- May 4: Submit abstract + post arXiv preprint (priority + regulatory window)
- May 6: Submit full paper
- September: NeurIPS decisions
- December: Conference presentation

**Confidence:** HIGH

### Paper 1: "First-Mover Bias" → TMLR (or arXiv companion)

**Content:** DASH method paper. 19 experiments, 3 real-world datasets,
systematic ablations (M, K, ε, δ, colsample), nonlinear DGP, variance
decomposition, Wilcoxon tests, computational cost analysis.

**Status:** Draft v7 ready. ArXiv v6 already published.

**Assessment:** Paper 1 has substantial standalone value (19 experiments
vs 5 in NeurIPS paper). NOT subsumed by Paper A.

**Options:**
- TMLR: rolling submission, thorough review, good for methods papers
- Domain venue (ML4H, CHIL, KDD): if experiments have domain-specific value
- arXiv companion: if just supplementary to Paper A

**Confidence:** HIGH that standalone submission is justified

### Paper B: "The Attribution Design Space" → ICML 2027 or JMLR

**Content:** Full relaxation characterization.
- R3 (Rashomon inevitability, extended with Fisher information connections)
- F1-F3 (unfaithfulness characterization)
- S1-S3 (equity-stability Pareto, architecture comparison)
- The M-parameterized spectrum (quantitative tradeoff curves)
- Comparison with Arrow's relaxation landscape
- Practical guidelines ("The Practitioner's Decision Tree")
- Lean formalization of R1 (if complete)

**Venue choice:**
| Venue | Best for | Page limit | Review speed |
|-------|----------|-----------|-------------|
| ICML 2027 | Visibility, conference talk | 9 + supplement | ~4 months |
| JMLR | Depth, comprehensive reference | Unlimited | 6-18 months |

ICML if the paper is tight (9 pages + supplement).
JMLR if it grows beyond 15 pages with full characterization.

**Timeline:**
- October 2026: Workshop paper preview (NeurIPS 2026 workshop)
- January 2027: Submit to ICML (if chosen)
- November 2026: Submit to JMLR (if chosen, rolling)

**Confidence:** HIGH

### Optionals

**NeurIPS 2026 Workshop (4 pages):** Preview of design space results.
Establishes priority, gets reviewer feedback. Target: "Trustworthy ML"
or "XAI" workshop. Deadline: ~October 2026.

**JAR/ITP paper:** Lean formalization as standalone contribution for
formal methods community. "Formalizing the Attribution Impossibility in
Lean 4." Only if bandwidth allows.

**Nature Machine Intelligence perspective (2-3 pages):** Non-technical
piece on regulatory implications. "Formal Proof That Feature Explanations
Are Unreliable: Implications for the EU AI Act." Directly targets
policymakers.

**PNAS (if NeurIPS rejects):** Bilodeau et al. (PNAS 2024) sets the
precedent. Our result + regulatory angle fits PNAS's "broad significance"
criterion. Would need rewriting for PNAS style. Requires NAS member
communication for contributed article (ask co-authors about connections).

## Rejection Contingency

If NeurIPS rejects Paper A:
1. Read reviewer feedback carefully
2. Incorporate R3 + design space results from Paper B
3. Submit comprehensive version to ICML 2027 (January deadline)
4. Alternatively: submit to PNAS (Bilodeau precedent) or JMLR

The rejection scenario actually STRENGTHENS the follow-up: NeurIPS
feedback + R3 + design space = more comprehensive paper for ICML.

## Scooping Risk Assessment

| Result | Risk | Mitigation |
|--------|------|-----------|
| Core impossibility | LOW | Lean code on GitHub, arXiv on May 4 |
| R1 (symmetry → Rashomon) | MEDIUM | Simple argument, could be independently discovered. In supplement of arXiv preprint. |
| R3 (inevitability) | MEDIUM | Composition of R1+R2, also in supplement. |
| F1-F3 (unfaithfulness, convergence) | LOW-MEDIUM | In supplement. Novel structural insight (no Arrow analogue). |
| Design space characterization | LOW | Requires our impossibility framework. |

**Key mitigation: The arXiv preprint on May 4 date-stamps ALL results
(main text + supplement). This covers R1-R3 and F1-F3.**

## Timeline Summary

| Date | Action |
|------|--------|
| **May 4, 2026** | NeurIPS abstract + arXiv preprint |
| **May 6, 2026** | NeurIPS full paper |
| **July 2026** | Paper 1 to TMLR (or arXiv) |
| **Aug 2, 2026** | EU AI Act high-risk obligations take effect |
| **Sep 2026** | NeurIPS decisions |
| **Oct 2026** | NeurIPS workshop paper (design space preview) |
| **Nov-Jan 2027** | Paper B to JMLR or ICML |
| **Dec 2026** | NeurIPS conference (present Paper A + workshop) |
| **Q4 2026** | CEN-CENELEC harmonised standards expected |

## The Citation Chain

```
Paper 1 (TMLR/arXiv)     Paper A (NeurIPS 2026)     Paper B (ICML/JMLR 2027)
    "DASH method"      →    "The impossibility"    →    "The design space"
    Empirical               Formal proof                Characterization
    19 experiments          28 theorems, 0 sorry        Full relaxation map
    Classical proofs        Machine-verified             Universal inevitability
```

Paper A cites Paper 1 for experiments. Paper B cites Paper A for the
impossibility. Together they form a self-reinforcing research program.
