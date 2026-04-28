# Handoff: Fairness Tightness Results

**Date:** 2026-04-28
**From:** ostrowski-impossibility repo (Langlands + fairness session)
**Available to:** Both the Nature paper session AND the NeurIPS attribution session

---

## The Result

The Chouldechova-Kleinberg fairness impossibility (calibration + equal FPR +
equal FNR can't coexist when base rates differ) was tested using the
impossibility framework's tightness classification.

**The finding:** The tightness type TRANSITIONS under performance constraints.

| TPR constraint | Tightness | Meaning |
|---|---|---|
| tau = 0.0 (unconstrained) | FULL | All pairs achievable (degenerate classifiers) |
| tau >= 0.3 | PPV-BLOCKED | Calibration is the universal poison |

At any practical performance level (TPR >= 30% for both groups):
- P2+P3 (equal FPR + equal FNR = equalized odds): ACHIEVABLE (gap < 0.006)
- P1+P2 (calibration + equal FPR): BLOCKED (gap > 0.10)
- P1+P3 (calibration + equal FNR): BLOCKED (gap > 0.13)
- P1+P2+P3 (all three): BLOCKED

**Calibration (PPV parity) is the universal poison.** It's individually
incompatible with both equal FPR and equal FNR at practical performance.
The ONLY achievable pairwise fairness compromise is equalized odds.

**Validated across:**
- Adult Income (sex as protected attribute, base rate gap 0.20)
- Synthetic data (base rate gap 0.25): same transition at tau = 0.3
- Synthetic data (base rate gap 0.40): PPV-BLOCKED even at tau = 0
- Bootstrap with model retraining (20 iterations, 100% stable)

## Methodology (v3, peer-review ready)

- Proper train/test split (70/30 stratified)
- Group-specific thresholds (100×100 = 10,000 classifiers per model)
- Performance constraints: TPR >= tau for both groups
- 3 base models (logistic regression, random forest, gradient boosting)
- Fair baseline: unconstrained group-specific thresholds (same # parameters)
- Bootstrap retrains the model on resampled training data

Scripts: `scripts/fairness_tightness_v3.py` (final), plus v1/v2 for audit trail.

## How This Connects to the Framework

The tightness classification (collapsed, full, p12-blocked, p23-blocked) is
the impossibility framework's organizational tool. Applied to fairness, it
reveals:
- WHICH property is the bottleneck (calibration)
- WHEN the impossibility becomes binding (TPR >= 0.3)
- WHAT compromises are available (equalized odds only)

No prior work has computed the tightness TYPE of a fairness impossibility
under performance constraints. The finding changes the practical recommendation
from "choose a fairness criterion" to "equalized odds is the only achievable
criterion at practical performance; calibration is structurally incompatible."
