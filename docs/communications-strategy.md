# Communications Strategy: "Faithful, Stable, Complete: Pick Two"

> Generated 2026-04-06. Review before executing.

## Brand Architecture

**Paper title:** "Faithful, Stable, Complete: Pick Two"
**Result name:** "The Attribution Impossibility" (let community add author names)
**Primary one-liner:** "Faithful, stable, complete: pick two."
**Secondary one-liner:** "No feature ranking survives retraining."
**Empirical hook:** "68% of public datasets have unreliable SHAP rankings."

**Analogy portfolio:**
- PRIMARY: Arrow's impossibility (academic), CAP theorem (CS/engineering)
- SECONDARY: Coin flip (qualified: "for collinear pairs"), fairness impossibility (governance)
- CUT: Heisenberg uncertainty, Goodhart's law

## Narrative Strategy

**Lead with the PROBLEM in academic channels, the GIFT in practitioner channels:**
- Academic: "We prove it's impossible..." → theorem → bounds → resolution
- Practitioner: "Here's how to check (single-model screen)..." → "Here's how to fix it (DASH)..." → "Here's why it's fundamental..."

**The "enemy":** The assumption that single-model SHAP rankings are reliable (NOT SHAP itself).
**The "gift":** single-model screen (1 model) + Z-test (5 models) + DASH (25 models) + M_min formula.

## Channel Strategy

### Twitter/X (5 tweets)

**1/5:** New paper: "Faithful, Stable, Complete: Pick Two"

We prove that no feature ranking can be simultaneously faithful to the model, stable across retraining, and complete for all feature pairs — when features are correlated.

In 68% of public datasets, SHAP rankings are unreliable.

**2/5:** The core proof is four lines. The Rashomon property (collinear features admit models ranking them in opposite orders) immediately implies faithfulness + stability + completeness are mutually exclusive.

For pairs of correlated features, ranking is literally a coin flip.

**3/5:** But we don't just prove it's impossible — we fix it.

DASH averages attributions across 25 models. It is provably Pareto-optimal: minimum-variance, zero within-group unfaithfulness, Cramér-Rao bound is tight.

Implementation: 5 lines of Python.

**4/5:** The entire framework is machine-verified in Lean 4:
- 305 theorems, 16 axioms, 0 sorry
- Core impossibility: ZERO axiom dependencies
- First formally verified impossibility in explainable AI

**5/5:** For practitioners: Run the screen to check your model. One line of code tells you which feature pairs are unreliable.

For regulators: SHAP-based proxy discrimination audits are provably unreliable under collinearity.

Paper: [arXiv] | Code: [GitHub]

### Reddit r/MachineLearning

**Title:** [R] "Faithful, Stable, Complete: Pick Two" — We prove SHAP feature rankings are mathematically unreliable under collinearity, verify it in Lean 4, and provide a fix (DASH)

### Hacker News

**Title:** "Faithful, Stable, Complete: Pick Two — An impossibility theorem for feature rankings, verified in Lean 4"

### Blog Post (TDS)

**Title:** "Your SHAP Rankings Are Unreliable — And That's a Theorem, Not a Bug"

1. The experiment that should worry you (credit model example)
2. Why this is not a bug (the four-line proof)
3. How bad is it, really? (1/(1-ρ²) divergence, coin flip)
4. The fix: DASH in 5 lines of Python
5. Why this matters beyond ML (fairness, EU AI Act, Lean verification)

### arXiv

**Categories:** stat.ML (primary), cs.LG, cs.AI, cs.LO
**Timing:** Tuesday or Wednesday for maximum visibility

## Pre-Circulation List

**Tier 1 (1-2 weeks before arXiv):**
- Scott Lundberg (SHAP creator) — courtesy + preempt adversarial reaction
- Cynthia Rudin (Rashomon sets) — builds on her framework
- Alexandra Chouldechova / Jon Kleinberg (fairness impossibilities)
- Tobias Nipkow (formalized Arrow in Isabelle)

**Tier 2 (3-5 days before):**
- Tianqi Chen (XGBoost), Guolin Ke (LightGBM)
- FAccT researchers on SHAP auditing
- Lean Mathlib maintainers

## Launch Calendar

| When | What | Channel |
|------|------|---------|
| Day 0 (Tue) | arXiv + JMLR submission | arXiv |
| Day 0-1 | Twitter thread | Twitter/X |
| Day 1 (Wed) | Reddit post | r/MachineLearning |
| Day 2 (Thu) | HN submission | Hacker News |
| Day 3 (Fri) | Lean announcement | Lean Zulip |
| Week 2 | Blog post | TDS / personal |
| Week 3 | Follow-up thread | Twitter/X |
| Week 4 | Meetup/seminar talk | Local ML group |
| Month 2 | YouTube explainer | YouTube |
| Month 2 | Workshop submission | ICML/NeurIPS workshop |
| Month 3 | Implementation release | dash-shap PR #255 |

## Guardrails

Every communication must pass:
1. **Hostile reviewer:** Would a senior JMLR reviewer accept this?
2. **Lundberg test:** Does SHAP's creator feel the paper is fair?
3. **Regret test:** Embarrassing in 5 years?
4. **Textbook test:** Would a 2035 textbook use this framing?

Never say "SHAP values are a coin flip" — say "SHAP **rankings** for collinear pairs."
