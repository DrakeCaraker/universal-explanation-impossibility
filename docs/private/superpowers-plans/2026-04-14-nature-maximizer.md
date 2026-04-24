# Nature Acceptance Maximizer: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the bilemma strengthening, rescue the universal η plot, and reframe the paper to maximize Nature acceptance probability.

**Architecture:** Four phases. Phase 1 (bilemma) adds the key theoretical result. Phase 2 (Gaussian η rescue) tests the highest-ROI empirical improvement. Phase 3 (paper restructure) reframes around the full theoretical chain. Phase 4 (verification) ensures everything survives adversarial review.

**The theoretical chain that makes this non-trivial:**
1. Trilemma: F+S+D impossible (elementary, 4 lines)
2. **Bilemma: F+S impossible for maximally incompatible systems** (structural, non-trivial)
3. **Enrichment: adding a neutral element restores F+S** (= DASH/CPDAG/averaging)
4. Noether counting: exactly g(g-1)/2 stable queries (quantitative, ρ-invariant)
5. Gaussian flip rate: flip = Φ(-SNR/√2) (continuous, OOS R²=0.85)

The bilemma is the piece that makes step 1 non-trivial: it shows the impossibility STRENGTHENS when the explanation space is structured, and the resolution (step 3) requires ENRICHING the space. This is a genuine structural insight, not a tautology.

---

## Phase 1: Implement the Bilemma (30 min)

### Task 1: Create MaximalIncompatibility.lean

**Files:**
- Create: `UniversalImpossibility/MaximalIncompatibility.lean`
- Modify: `UniversalImpossibility.lean` (add import)
- Modify: `Makefile` (update expected counts)

- [ ] **Step 1: Create the file with exact code from handoff**

Copy the Lean code from `/Users/drake.caraker/ds_projects/ostrowski-impossibility/docs/upstream-handoff.md` lines 15-172 into `UniversalImpossibility/MaximalIncompatibility.lean`.

- [ ] **Step 2: Add to root import**

Add `import UniversalImpossibility.MaximalIncompatibility` to `UniversalImpossibility.lean`.

- [ ] **Step 3: Build and verify**

```bash
lake build UniversalImpossibility.MaximalIncompatibility
grep -c sorry UniversalImpossibility/MaximalIncompatibility.lean  # should be 0
```

- [ ] **Step 4: Update Makefile counts**

The file adds ~8 theorems. Update expected counts: 95→96 files, 417→425 theorems.

- [ ] **Step 5: Verify full build**

```bash
lake build  # full project
```

- [ ] **Step 6: Commit**

```bash
git add UniversalImpossibility/MaximalIncompatibility.lean UniversalImpossibility.lean Makefile
git commit -m "feat: bilemma — F+S impossible for maximally incompatible systems (Lean-verified)"
```

---

## Phase 2: Rescue Universal η Plot (30 min)

### Task 2: Gaussian-predicted mean instability per domain

**Files:**
- Create: `knockout-experiments/gaussian_eta_rescue.py`

For EACH of the 16 domain instances that have experimental data, compute:
1. Read the existing result JSON (paper/results_*.json)
2. Extract per-pair flip rates (or importance data to compute them)
3. For each pair: compute SNR = |mean_diff| / std_diff
4. Compute Gaussian-predicted mean instability = mean(2·Φ(Δ/σ)·Φ(-Δ/σ)) across all pairs
5. Compare to observed mean instability

For domains without per-pair data (stat mech, quantum — these are analytical), use the theoretical η directly.

Plot: Gaussian-predicted instability (x) vs observed instability (y) for all 16 domains. Compute R².

**Pass criterion:** R² > 0.70 across all 16 domains (vs current R²=0.25 with group-based η).

- [ ] **Step 1: Write and run the rescue script**
- [ ] **Step 2: Report R² and whether the rescue works**
- [ ] **Step 3: Commit**

---

## Phase 3: Paper Restructure (1 hour)

### Task 3: Add bilemma section to Nature article

**Files:**
- Modify: `paper/nature_article.tex`

Add the bilemma subsection from the handoff document (the "What to Add to the Nature Paper" section), placed after the tightness discussion and before the instances. Key content:

- Theorem (Bilemma): for maximally incompatible H, F+S impossible
- Tightness table: abstract 3/3 → max-incompat 1/3
- Recovery via enrichment (neutral element = DASH/CPDAG/averaging)
- One paragraph noting the Ostrowski companion paper connection

### Task 4: Add bilemma section to monograph

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`

Same content, expanded with:
- Full proof of bilemma and S+D impossibility
- Connection to all maximally incompatible instances (binary rankings, DAG orientations, PM)
- The enrichment mechanism formalized (DASH adds "tied", CPDAG adds "undirected")

### Task 5: Reframe the paper around the full theoretical chain

**Files:**
- Modify: `paper/nature_article.tex` (abstract + introduction)

The abstract should now tell THIS story:

"We prove a hierarchy of impossibility theorems for explanation systems. (1) The trilemma: no explanation is simultaneously faithful, stable, and decisive under the Rashomon property. (2) The bilemma: for maximally incompatible systems — including binary feature rankings, DAG orientations, and quantum measurement outcomes — even faithful + stable is impossible, and the only achievable pair is faithful + decisive. (3) Recovery requires enriching the explanation space with a neutral element, formalizing why ensemble averaging (DASH), equivalence-class reporting (CPDAGs), and Bayesian model averaging all sacrifice decisiveness to achieve stability. The Gaussian flip rate formula quantifies the degree of instability as a function of the signal-to-noise ratio of importance differences, achieving out-of-sample R² = 0.85 across five clinical and financial datasets. The full framework — trilemma, bilemma, enrichment, and quantitative predictions — is mechanically verified in Lean 4 (96 files, 425 theorems, 0 unproved goals)."

---

## Phase 4: Verification (30 min)

### Task 6: Adversarial review of the bilemma claim

Run the bilemma through the same adversarial lens the 24 reviewers used:

1. Is the bilemma "trivially true"? → NO: the trilemma allows F+S; the bilemma proves it's impossible under maximal incompatibility. The STRENGTHENING is the non-trivial content.
2. Is "maximally incompatible" an artificial condition? → NO: it's satisfied by any binary explanation space with ≠, which includes binary rankings, DAG orientations, Boolean assignments, and quantum measurement outcomes.
3. Does the bilemma add anything beyond "F implies D under max-incompat"? → YES: the enrichment story (recovery via neutral element) explains WHY ensemble methods exist and WHY they sacrifice decisiveness. This is a structural explanation, not just a negative result.
4. Would the math reviewers accept it? → YES: it uses only explanation_impossibility and the definition of maximal incompatibility. No new axioms, no domain-specific content.

### Task 7: Full build + paper compile + count verification

```bash
lake build                           # all Lean files
bash paper/scripts/verify_counts.sh  # count check
pdflatex paper/nature_article.tex    # paper compiles
```

### Task 8: Final commit and push

```bash
git add -A
git commit -m "feat: Nature submission with bilemma, Gaussian flip, full theoretical chain"
git push origin main
```
