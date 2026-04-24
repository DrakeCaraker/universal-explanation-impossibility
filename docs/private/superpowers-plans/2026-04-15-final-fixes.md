# Final Fixes: All Remaining Review Items

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every remaining issue from peer review before submission.

---

## Phase 1: Top-K Expansion + Null Model (1 hour)

### Task 1: Expand top-K to 15 bridge datasets (gain-based)

**Source:** Reviewer A3 — K=3 on 5 datasets is too small.

- [ ] **Step 1:** Use the 15-dataset bridge results (already computed). For each dataset, the SNR per pair is available. Rank features by mean importance. Compute top-K reliability for K=3, 5, 10, all.
- [ ] **Step 2:** Report whether the inversion (gap worse at smaller K) persists across 15 datasets.
- [ ] **Step 3:** Fisher's exact test: is fraction reliable at K=3 significantly lower than at K=all?
- [ ] **Step 4:** Save to `results_topk_15datasets.json`

### Task 2: Null model comparison (B2)

**Source:** Reviewer B2 — Is the inversion just order statistics?

- [ ] **Step 1:** Generate 15 synthetic datasets matching the real ones (same P, same n_models=30).
  - Draw importance vectors from N(μ, σ) where μ is drawn from Uniform(0,1) (no Rashomon — just sampling noise).
  - Compute SNR for each pair. Compute top-K reliability.
- [ ] **Step 2:** Compare: does the inversion appear WITHOUT Rashomon?
  - If yes: the inversion is partly statistical (order statistics effect).
  - If no: the inversion is a Rashomon-specific phenomenon.
- [ ] **Step 3:** Either way, report honestly. The impossibility theorem's contribution is that the instability is STRUCTURAL, not just statistical. The null model distinguishes the two.
- [ ] **Step 4:** Save to `results_topk_null_model.json`

## Phase 2: Mech Interp on SageMaker (run separately)

### Task 3: Run GPU mech interp experiment

- [ ] **Step 1:** Launch ml.g4dn.xlarge SageMaker notebook instance
- [ ] **Step 2:** Upload `mech_interp_rashomon_gpu.py`
- [ ] **Step 3:** `pip install transformers datasets && python mech_interp_rashomon_gpu.py`
- [ ] **Step 4:** Download `results_mech_interp_rashomon_gpu.json`
- [ ] **Step 5:** Stop instance
- [ ] **Step 6:** Commit results

**Not blocking:** This runs independently. Other tasks don't depend on it.

## Phase 3: Text Fixes (30 min)

### Task 4: Connect top-K to coverage conflict (C1)

- [ ] Add to monograph: "Among top-K features, the incompatibility relation is dense: most pairs have SNR < 0.5 because the top features are closely matched. This is precisely the regime where the bilemma applies most strongly — the coverage conflict degree is highest at the top of the importance distribution."

### Task 5: Label top-K as exploratory (B3)

- [ ] Add to monograph limitations: "The top-K regulatory gap analysis is exploratory (not pre-registered)."

### Task 6: Mech interp → one sentence in limitations (A3, C3)

- [ ] Replace any mech interp discussion with: "A pilot test of circuit stability on GPT-2 small (10 independently fine-tuned models on 500 SST-2 examples) did not achieve the Rashomon condition (accuracy range 21%), preventing a test of the circuit stability prediction. A properly powered experiment (20 models, 2000 examples, GPU fine-tuning) is planned."

### Task 7: Report absolute counts alongside fractions for top-K (A1)

- [ ] Add to regulatory analysis: "At K=3, 2 of 15 comparisons are reliable (13%); at K=all, ~250 of ~800 are reliable (32%). The absolute number of reliable comparisons increases with K, but the fraction AMONG the top-K comparisons decreases."

## Phase 4: Commit and Verify (15 min)

### Task 8: Update all counts if needed
### Task 9: Run `make counts` to verify
### Task 10: Commit and push

---

## Execution Order

```
Phase 1 (parallel, 1 hour):
  Task 1 (top-K 15 datasets) — 30 min, script
  Task 2 (null model) — 30 min, script

Phase 2 (async, run on SageMaker):
  Task 3 (mech interp GPU) — 30-45 min on ml.g4dn.xlarge

Phase 3 (sequential, 30 min):
  Tasks 4-7 — text edits

Phase 4 (sequential, 15 min):
  Tasks 8-10 — verify and commit
```
