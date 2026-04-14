# Peer Review Synthesis — 22 Simulated Reviewers

**Date**: 2026-04-14
**Scope**: 10 review agents, 22 reviewer personas (12 domain + 10 editorial)
**Verdict**: REJECT (19/22 reviewers)

## Consensus Fatal Flaws (raised by 3+ reviewers)

1. **Core theorem is trivially true** (10 reviewers) — "non-injective maps lose information"
2. **Every domain instance relabels known results** (7 reviewers) — no domain expert found new insight
3. **Empirical validation compromised** (4 reviewers) — 3/6 predictions falsified, weight perturbation ≠ retraining
4. **Definitional mismatch paper ↔ Lean** (3 reviewers) — strongest differentiator may not verify paper's claims
5. **Massive overclaiming** (5 reviewers) — rhetoric of "universal law" not matched by content

## Consensus Strengths

1. **Lean formalization is genuine and high-quality** (4 reviewers)
2. **Breadth of ambition** (3 reviewers)
3. **Topic is timely** (3 reviewers)

## Venue Probability

| Venue | P(accept) |
|-------|-----------|
| Nature | < 1% |
| Nature MI | 5-10% |
| PNAS | 5-10% |
| JMLR | 15-25% |
| Science | < 1% |

## Top 10 Issues

1. Core theorem trivially true (10 reviewers)
2. Domain instances are relabeling (7 reviewers)
3. 3/6 predictions falsified (4 reviewers)
4. Weight perturbation ≠ retraining (2 reviewers)
5. Paper-Lean definitional mismatch (3 reviewers)
6. Statistical errors: non-independence, no correction (3 reviewers)
7. Post-hoc group selection for η plot (2 reviewers)
8. Overclaiming (5 reviewers)
9. 2× over Nature word limit (2 reviewers)
10. Inconsistent counts, no CI pipeline (2 reviewers)

## Path Forward

1. **Retarget JMLR** — abandon Nature/Science for this version
2. **Remove 4/6 domain instances** — keep ML + causal, demote rest
3. **Lead with Lean formalization** — fix definitional mismatch first
4. **Redo empirical work** — proper retraining, pre-registered groups, corrected statistics
5. **Derive non-trivial corollary** — tight bound or constructive algorithm
6. **Cut 50% of word count** — remove "universal" from title
7. **Add Noether counting as centerpiece** — the genuinely novel quantitative result
