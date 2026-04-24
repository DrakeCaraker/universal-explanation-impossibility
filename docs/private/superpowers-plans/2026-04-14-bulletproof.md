# Bulletproof Plan: Address Every Open Issue

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every rejection-level and major-revision issue so the paper can survive adversarial peer review at Nature.

**Architecture:** Three phases by urgency. Phase 1 (rejection-level, 1 hour) fixes issues that would cause outright rejection. Phase 2 (major-revision, 30 min) addresses reviewer concerns. Phase 3 (paper polish, 30 min) ensures format compliance and framing.

---

## Phase 1: Rejection-Level Fixes

### Task 1: Gaussian flip on Random Forest + Ridge [G4/W4]

**The attack:** "You only tested on XGBoost. Does the formula generalize?"

**Files:**
- Create: `knockout-experiments/gaussian_flip_multimodel.py`

- [ ] **Step 1: Run Gaussian flip on 3 model classes × 3 datasets**

For Breast Cancer, Wine, CalHousing:
- **XGBoost** (already done: OOS R²=0.79-0.97)
- **Random Forest**: RandomForestClassifier/Regressor(n_estimators=100, random_state=seed)
- **Ridge Regression**: Ridge(alpha=1.0)

For each: 30 calibration models (seeds 42-71) + 30 validation models (seeds 142-171). Compute per-pair Gaussian flip prediction on calibration, measure flip rate on validation.

- [ ] **Step 2: Report OOS R² per model class per dataset**

Table: 3 model classes × 3 datasets = 9 cells.

**Pass criterion:** OOS R² > 0.60 for ≥7/9 cells. The formula must work across model classes, not just XGBoost.

- [ ] **Step 3: Commit**

### Task 2: Empirical enrichment demonstration [V3/W6]

**The attack:** "You claim enrichment restores F+S but never demonstrate it."

**Files:**
- Create: `knockout-experiments/enrichment_demo.py`

- [ ] **Step 1: Demonstrate that adding "tied" restores stability**

Design: P=8 features, 2 groups of 4, ρ_within=0.90, β=[3,3,3,3, 1,1,1,1].
Train 100 XGBoost models.

**Without enrichment** (binary ranking: j>k or k>j):
- Within-group flip rate should be ~50% (bilemma applies)
- F+S impossible

**With enrichment** (ternary: j>k, k>j, or "tied" when |importance_j - importance_k| < threshold):
- Within-group pairs classified as "tied" → flip rate drops to ~0%
- F+S achieved at the cost of decisiveness (the "tied" pairs)

- [ ] **Step 2: Report: fraction of pairs tied, within-group flip rate before/after, stability improvement**

The key number: what fraction of pairs must be "tied" (sacrificing decisiveness) to achieve <5% flip rate?

- [ ] **Step 3: This directly validates the bilemma's enrichment prediction on real data**

- [ ] **Step 4: Commit**

### Task 3: Nature word count check + trim [V5]

**Files:**
- Modify: `paper/nature_article.tex`

- [ ] **Step 1: Count words**

```bash
detex paper/nature_article.tex | wc -w
```

Nature Articles: ~3,000 words main text (excluding Methods, refs, legends).
Nature Letters: ~1,500 words.

- [ ] **Step 2: If over limit, trim**

Priority cuts:
1. Remove the weakest cross-domain instance paragraphs
2. Compress the 8-instance table to essentials
3. Move proofs to Methods
4. Consolidate the Noether/Gaussian/SAGE paragraphs

- [ ] **Step 3: Commit**

### Task 4: Reframe real-data Noether result honestly [W5]

**The attack:** "Your main result (50pp bimodal gap) only works on synthetic data."

**Files:**
- Modify: `paper/nature_article.tex` and monograph

- [ ] **Step 1: Reframe the narrative**

The story is NOT "Noether counting gives 50pp on real data" (it doesn't — it gives 4-22pp).

The story IS:
1. **Exact symmetry** (synthetic, β_j = β_k): bimodal gap = 50pp, invariant across all ρ (the theorem)
2. **Approximate symmetry** (real data, β_j ≈ β_k): gap weakens proportionally to within-group coefficient variance (the generalization)
3. **The Gaussian flip formula** bridges both regimes: OOS R²=0.85 on real data (the practical tool)

The bimodal gap is the THEOREM; the Gaussian flip is the TOOL. Both are needed; neither alone is sufficient.

- [ ] **Step 2: Ensure the paper doesn't overclaim bimodal gap on real data**

Check all claims about "bimodal" and ensure they specify synthetic context or qualify with "weakens on real data."

- [ ] **Step 3: Commit**

---

## Phase 2: Major-Revision Fixes

### Task 5: "Maximal incompatibility is natural" argument [V2]

**Files:**
- Modify: `paper/nature_article.tex` (bilemma section)

- [ ] **Step 1: Add a paragraph listing natural instances**

"Maximal incompatibility is not an artificial condition. It holds for any explanation space where distinct explanations are inherently incompatible:
- **Binary feature rankings**: 'j > k' and 'k > j' cannot both be correct
- **DAG edge orientations**: 'X → Y' and 'Y → X' are mutually exclusive
- **Quantum measurement outcomes**: spin-up and spin-down are orthogonal
- **Social choice rankings**: two different aggregate orderings are incompatible
- **All instances where incompatible = ≠**: which is the most common choice in the framework (used by 14 of 16 domain instances)

In fact, 14 of the 16 Lean-verified instances in this paper use incompatible = ≠, making them maximally incompatible. The bilemma thus applies to the MAJORITY of instances, not an artificial subset."

- [ ] **Step 2: Verify the 14/16 claim**

```bash
grep "incompatible.*≠\|incompatible.*fun.*=>.*≠\|incompatible := fun.*=>.*≠\|incomp.*Ne\|fun.*h₁ h₂ => h₁ ≠ h₂" UniversalImpossibility/*.lean | wc -l
```

- [ ] **Step 3: Commit**

### Task 6: Comparison to Fisher VIC and Laberge [W1/W2]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex` (Related Work)

- [ ] **Step 1: Add explicit comparison paragraph**

Already partially done (citations added). Ensure the comparison states:
- Fisher et al. (2019) VIC visualizes explanation instability across the Rashomon set — our framework formalizes WHY this instability exists (the impossibility) and WHEN it's maximal (the bilemma)
- Laberge et al. (2023) extract partial orders from Rashomon sets — this IS our G-invariant resolution for the attribution instance; our contribution is the abstract framework and generalization to non-ML domains
- Marx et al. (2024) develop uncertainty-aware explainability — complementary to our framework

- [ ] **Step 2: Commit**

### Task 7: Gaussianity verification [G3]

**Files:**
- Add to: `knockout-experiments/gaussian_flip_cv.py` or create new

- [ ] **Step 1: For each dataset, compute fraction of pairs passing Shapiro-Wilk**

Already computed (60-100%). Ensure this is REPORTED in the paper with the actual numbers.

- [ ] **Step 2: Add Q-Q plots as Extended Data**

For one representative dataset (Breast Cancer), generate Q-Q plots of importance differences for 5 representative feature pairs showing the Gaussian fit.

- [ ] **Step 3: Commit**

---

## Phase 3: Paper Polish

### Task 8: Verify bilemma LaTeX in monograph [V6]

- [ ] **Step 1: Read the monograph bilemma section manually**

Verify: correct theorem statement, correct proof sketch, correct tightness table, correct enrichment paragraph.

- [ ] **Step 2: Fix any issues**

### Task 9: Final consistency check

- [ ] **Step 1: Verify all counts match**

```bash
# Lean
ls UniversalImpossibility/*.lean | wc -l  # should be 96
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'  # 424

# Papers
grep -o "96.file\|424.theorem" paper/nature_article.tex | sort -u
grep -o "96.file\|424.theorem" paper/universal_impossibility_monograph.tex | sort -u

# CLAUDE.md
grep "96 files\|424 theorem" CLAUDE.md
```

- [ ] **Step 2: Build everything**

```bash
lake build  # Lean
```

- [ ] **Step 3: Final commit and push**

---

## Execution Order

```
Phase 1 (parallel):
  Task 1 (multi-model Gaussian) — 30 min, agent
  Task 2 (enrichment demo) — 30 min, agent
  Task 3 (word count) — 5 min, inline
  Task 4 (reframe narrative) — 15 min, inline

Phase 2 (parallel, after Phase 1):
  Task 5 (natural argument) — 10 min, inline
  Task 6 (VIC comparison) — 10 min, inline
  Task 7 (Gaussianity) — 10 min, inline

Phase 3 (sequential, last):
  Task 8 (verify bilemma LaTeX) — 10 min
  Task 9 (consistency) — 5 min
```
