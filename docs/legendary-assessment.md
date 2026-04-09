# Legendary Submission Assessment

Generated: 2026-04-02

## Legendary Scores (calibrated: Arrow=10, Chouldechova=8, Bilodeau=6)

| Version | Current | Achievable | Ceiling |
|---------|---------|-----------|---------|
| JMLR (38pp) | 6.5 | 7.5 | 7.5 |
| NeurIPS (10pp) | 6.5 | 7.5 | 7.5 |
| Definitive (54pp) | 6.0 | 7.0 | 7.5 |

## The 3 Changes Per Version That Close the Gap

### JMLR (6.5 → 7.5)
1. Cut 8 pages of dead-end content (topology, categorical sections)
2. Add deployed-system case study (real bank/hospital SHAP dashboard)
3. Add iconic trilemma figure (triangle with X)

### NeurIPS (6.5 → 7.5)
1. Restructure outside-in (impossibility first, DST as consequence)
2. Rewrite opening paragraph with practitioner pain ("Train. Retrain. Rankings flip.")
3. Add trilemma triangle as Figure 1

### Definitive (6.0 → 7.0)
1. Reorganize into textbook-chapter structure
2. Cut 10 pages of dead-end sections to appendix footnotes
3. Add regulatory annex mapping to specific AI Act articles

## The Core Narrative Finding

**The paper tells its story inside-out. Legendary papers are outside-in.**

Current: "Here's the design space → the impossibility is the foundation → DASH is optimal"
Legendary: "Your rankings flip → we prove it's impossible → here's the complete map → here's the fix"

## The Iconic Figure (doesn't exist yet)

Panel A: Equilateral triangle — Faithful / Stable / Complete vertices, red X center.
- Drop Stable → Family A (single-model, flips)
- Drop Complete → Family B (DASH, ties)
- Drop Faithful → trivial (constant ranking)

Panel B: Before/After — two bar charts showing reversed rankings → one DASH chart with ties.

## The Opening Sentence Problem

Current (all 3 versions): "We characterize the complete design space..."
This is the ANSWER. It should be the QUESTION.

Strongest opening: "No feature ranking can be simultaneously faithful, stable, and complete when features are correlated."

Or with practitioner energy: "For ranking collinear pairs, SHAP is literally a coin flip."

## The Name

- "The Attribution Impossibility" — 3.5/5 stickiness
- "The Explainability Trilemma" — 4/5 (but slightly oversells scope)
- "Faithful, Stable, Complete: Pick Two" — 4.5/5 (most memorable, least formal)

Recommendation: Keep formal title, use "Pick Two" as the tweet/talk hook.

## The 5 Follow-Up Papers This Enables

1. DASH-Causal: conditional SHAP + causal graph → complete stable rankings
2. Rashomon Coefficient: dataset-level reliability metric, 100+ dataset validation
3. Adaptive Ensemble Sizing: progressive DASH with minimax sample complexity
4. LLM Attribution Instability: formal analysis for transformer attention
5. Regulatory Implications: AI Act compliance templates and standards

## The Ceiling (7.5, not 8+)

The path from 7.5 to 8+ requires POST-PUBLICATION events:
- Adoption by a regulatory body or major company
- The SBD being used by another research group
- A real-world deployment story demonstrating harm prevented by DASH

These cannot be achieved before submission. The pre-submission ceiling is 7.5.

## What Will Still Stand in 10 Years

- The core impossibility (zero axioms, just Rashomon)
- The DST two-family structure
- The fairness audit impossibility
- The Lean formalization as methodology

## What Will Be Superseded

- The specific 1/(1-ρ²) formula (refined with better axioms)
- The M=25 recommendation (will be adaptive)
- DASH itself (replaced by something handling unbalanced ensembles)
- The equicorrelation assumption (heterogeneous correlations)
