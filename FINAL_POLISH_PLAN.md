# Final Polish — Submission Readiness

**Goal**: Fix every remaining validation gap. After this, the only
items requiring human action are citation verification (Meloux,
Rao, Noguer i Alonso) and the actual submission.

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**State**: 74 files, 349 theorems, 72 axioms, 0 sorry

---

## Phase 1: Mechanical Fixes [Sonnet, 15 min]

### Task 1.1: Update all stale counts

Files to update:
- CLAUDE.md: 73→74 files, 346→349 theorems
- README.md: update counts to 74/349/72/0
- Makefile verify target: update expected counts

### Task 1.2: Trim PNAS to 6 pages

Read paper/universal_impossibility_pnas.tex. It's currently 7pp.
Cut ~500 words by:
- Compress the 4 instance descriptions (currently ~1200 words → ~900)
- Move the practitioner decision table to SI
- Compress the info-theoretic proposition to 3 sentences (from full subsection)
- Tighten the regulatory paragraph

Compile. Must be ≤6 pages main text (references can overflow to page 7).

### Task 1.3: Fix experimental honesty flags

In paper/universal_impossibility_pnas.tex AND paper/universal_impossibility.tex:

a) GradCAM: Add to the GradCAM result description:
"Prediction agreement is 78.8\%, reflecting the sensitivity of
ImageNet-pretrained ResNet-18 to parameter perturbation on
CIFAR-10 images."

b) Full retraining: Add to the attention result description:
"Four of twenty models achieved 72--78\% held-out accuracy
(below the 80\% target); all twenty are included in the analysis."

c) Token Citation: Ensure the methodology description says:
"We measure token citation instability via attention rollout
as a proxy for explanation instability" — NOT "LLM self-explanation."

### Task 1.4: Update experiment runner

Edit paper/scripts/run_all_universal_experiments.py to include:
- gradcam_instability_experiment.py
- llm_explanation_instability_experiment.py (now Token Citation)
- attention_full_retraining_experiment.py

---

## Phase 2: Propagate to All Paper Versions [Sonnet, 30 min]

### Task 2.1: Update monograph, JMLR, NeurIPS

For each of the 3 stale paper versions, apply these changes:
- Cross-domain tiering (remove QM/legal/crypto, add 3-tier structure)
- "LLM Self-Explanation" → "Token Citation" throughout
- Add quantitative per-fiber bound reference
- Add full retraining result (19.9%) to cross-instance table
- Add info-theoretic proposition (brief version)
- Add practitioner decision table
- Update Lean counts to 74/349/72/0
- Fix "uniquely optimal" → "Pareto-optimal" if not already done
- Fix "eliminating logical gaps" → "...in the formal proofs"

Compile each. Report page counts.

---

## Phase 3: arXiv Preparation [Sonnet, 15 min]

### Task 3.1: Prepare monograph for arXiv

The monograph (universal_impossibility_monograph.tex) is the arXiv version.

- Verify author names are unmasked
- Add GitHub repo URL: github.com/DrakeCaraker/universal-explanation-impossibility
- Add arXiv category suggestions as a comment:
  Primary: cs.LG. Cross-list: cs.AI, stat.ML, cs.LO
- Verify it compiles standalone (no missing \input dependencies)
- Compile. Report page count.

---

## Phase 4: Final Verification [Sonnet, 10 min]

### Task 4.1: Lean clean build

```bash
lake build
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

### Task 4.2: Compile all 5 paper versions

All must compile cleanly. Report page counts.

### Task 4.3: Code-paper count verification

Verify the Lean counts in EVERY paper version match the grep output.

### Task 4.4: Commit and push
