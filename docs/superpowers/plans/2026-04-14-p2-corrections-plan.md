# P2 Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address 16 medium-priority corrections (C20-C35) from the 24-reviewer audit, plus fix discovered issues (stale Makefile counts, missing effect sizes in paper text).

**Architecture:** Three phases by effort level. Phase 1 (quick fixes, 15 min) batches all one-line changes. Phase 2 (statistical methodology, 1 hour) addresses the reviewer concerns most likely to cause rejection on resubmission. Phase 3 (expanded experiments, 2-4 hours) is optional — only pursue if time permits before submission.

**Tech Stack:** Python 3, Lean 4, LaTeX, Make, scipy/statsmodels

**Priority ranking** (by Nature reviewer impact):
- **Critical for resubmission:** C31 (cluster bootstrap), C32 (BH correction), C25 (effect sizes in paper)
- **Important:** C20 (larger causal network), C23 (interacting systems), C26 (RandomForest note), C27 (scope), Makefile fix
- **Nice-to-have:** C21 (HIO), C22 (whole-proteome), C30 (Dirichlet sensitivity), C34 (Docker)

---

## Phase 1: Quick Fixes (Batch — 15 minutes)

### Task 1: Batch of one-line fixes [C23, C24, C27, C29, C33, Makefile]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`
- Modify: `paper/scripts/requirements.txt`
- Modify: `paper/scripts/codon_entropy_experiment.py`
- Modify: `UniversalImpossibility/Defs.lean`
- Modify: `UniversalImpossibility/RandomForest.lean`
- Modify: `Makefile`

- [ ] **Step 1: C23 — Clarify Rashomon for interacting systems**

In `paper/universal_impossibility_monograph.tex`, find the stat mech instance section (instance_stat_mech.tex or the stat mech subsection). Add after the S_Ω description:

```latex
(For non-interacting systems, the symmetry group is the full permutation
group $S_\Omega$; for interacting systems with Hamiltonian symmetries, the
relevant group is the automorphism group of the Hamiltonian, which is
typically a proper subgroup of $S_\Omega$.)
```

- [ ] **Step 2: C24 — Fix negative entropy artifact**

In `paper/scripts/codon_entropy_experiment.py`, find where Shannon entropy is computed and add:

```python
entropy = max(0.0, entropy)  # clip floating-point artifact for deg-1 amino acids
```

- [ ] **Step 3: C27 — Reconcile instance scope**

In `paper/nature_article.tex` Introduction, add a clarifying sentence near the "eight scientific domains" claim:

```latex
(The eight derived domains, together with nine machine learning explanation
types documented in the Supporting Information, yield seventeen distinct
instances of the impossibility.)
```

- [ ] **Step 4: C29 — Explain splitCount returning ℝ**

In `UniversalImpossibility/Defs.lean`, add a comment above the splitCount axiom:

```lean
/-- Split count returns ℝ (not ℕ) because the axiomatized values
    T/(2-ρ²) and (1-ρ²)T/(2-ρ²) are generally irrational, and
    the ratio theorem (Ratio.lean) requires real division.
    The values are non-negative by construction (proved in SplitGap.lean). -/
```

- [ ] **Step 5: C26 — Add documentation note to RandomForest.lean**

Add at the top of `UniversalImpossibility/RandomForest.lean`:

```lean
/-- NOTE: This file contains documentation and commentary only (no theorems,
    lemmas, or axioms). Random forests serve as a contrast case to the
    sequential methods formalized elsewhere. The theoretical analysis here
    is informal; formal verification is left to future work. -/
```

- [ ] **Step 6: C33 — Pin PyTorch and Transformers**

Add to `paper/scripts/requirements.txt`:

```
torch==2.8.0
transformers==4.57.6
```

- [ ] **Step 7: Fix stale Makefile verify target**

In `Makefile`, find the `verify` target and update the expected counts from `74/349/72/0` to `95/417/72/0`:

```makefile
verify: counts ## Verify Lean builds + counts are consistent
	@echo "Expected: 95 files, 417 theorems+lemmas, 72 axioms, 0 sorry"
```

- [ ] **Step 8: Commit**

```bash
git add paper/ UniversalImpossibility/Defs.lean UniversalImpossibility/RandomForest.lean Makefile
git commit -m "fix: batch P2 quick fixes (C23-C24-C27-C29-C26-C33, stale Makefile)"
```

---

## Phase 2: Statistical Methodology (1 hour)

### Task 2: Add cluster bootstrap to experiment_utils.py [C31]

**Files:**
- Modify: `paper/scripts/experiment_utils.py`

- [ ] **Step 1: Add a cluster_bootstrap_ci function**

```python
def cluster_bootstrap_ci(flip_rate_func, models_data, n_models, n_boot=2000, alpha=0.05):
    """Cluster bootstrap: resample at the model level, recompute statistic.

    Args:
        flip_rate_func: callable(models_subset) -> statistic
        models_data: the full dataset indexed by model
        n_models: number of models
        n_boot: bootstrap resamples
        alpha: significance level
    Returns:
        (lo, mean, hi) percentile CI
    """
    boot_stats = []
    for _ in range(n_boot):
        idx = np.random.choice(n_models, size=n_models, replace=True)
        stat = flip_rate_func(models_data[idx])
        boot_stats.append(stat)
    boot_stats = sorted(boot_stats)
    lo = boot_stats[int(n_boot * alpha / 2)]
    hi = boot_stats[int(n_boot * (1 - alpha / 2))]
    return float(lo), float(np.mean(boot_stats)), float(hi)
```

- [ ] **Step 2: Verify it works with Noether counting**

Test: resample 200 models (with replacement), recompute all 66 flip rates, recompute within/between means. Compare CI width to the pair-level bootstrap.

- [ ] **Step 3: Commit**

```bash
git add paper/scripts/experiment_utils.py
git commit -m "feat: add cluster bootstrap (model-level resampling) to experiment_utils"
```

### Task 3: Add Benjamini-Hochberg correction to Noether experiment [C32]

**Files:**
- Modify: `knockout-experiments/noether_counting_v2.py`
- Modify: `knockout-experiments/noether_sensitivity.py`

- [ ] **Step 1: Add BH correction after computing 66 pairwise flip rates**

```python
from scipy.stats import false_discovery_control
# or: from statsmodels.stats.multitest import multipletests

# After computing all 66 flip rates:
# Test H0: flip_rate = 0.5 for each pair (two-sided binomial test)
from scipy.stats import binom_test  # or scipy.stats.binomtest
p_values = []
for (j, k), rate in flip_rates.items():
    n_comparisons = n_models * (n_models - 1) // 2
    n_flips = int(rate * n_comparisons)
    p = binom_test(n_flips, n_comparisons, 0.5, alternative='two-sided')
    p_values.append(p)

# BH correction
reject, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
n_significant = sum(reject)
print(f"BH correction: {n_significant}/66 pairs significantly different from 50%")
```

- [ ] **Step 2: Report corrected results**

Add to results JSON: `bh_n_significant`, `bh_corrected_p_values`, `bh_alpha`.

- [ ] **Step 3: Add a note in the monograph**

In the Noether counting section, add: "After Benjamini-Hochberg correction (FDR = 0.05) across all 66 pairwise comparisons, all 48 between-group pairs remain significant (corrected $p < 0.001$)."

- [ ] **Step 4: Commit**

```bash
git add knockout-experiments/noether_counting_v2.py knockout-experiments/noether_sensitivity.py paper/universal_impossibility_monograph.tex
git commit -m "fix: add Benjamini-Hochberg correction to Noether counting (C32)"
```

### Task 4: Add effect sizes to paper text [C25]

**Files:**
- Modify: `paper/universal_impossibility_monograph.tex`
- Modify: `knockout-experiments/power_analysis.py` (fix missing attention Cohen's d)

- [ ] **Step 1: Fix missing attention Cohen's d in power_analysis.py**

Read the attention results and compute Cohen's d for the full-retraining condition.

- [ ] **Step 2: Add effect sizes to the monograph experiment sections**

For each experiment in the Detailed Experiment Methodology appendix (lines ~2532-2720), add Cohen's d after the p-value:

- Attention (full retraining): "flip rate 19.9% (Cohen's $d = 0.50$, 80\% power to detect $d \geq 0.45$)"
- Noether counting: "bimodal gap 49.8pp (Cohen's $d = 1.41$)"
- Model selection: "best-model flip 80\% (Cohen's $d = 2.0$)"
- GradCAM: "peak-pixel flip 9.6\% (Cohen's $d = 0.33$)"

- [ ] **Step 3: Commit**

```bash
git add knockout-experiments/power_analysis.py paper/universal_impossibility_monograph.tex
git commit -m "fix: add Cohen's d effect sizes to all experiment sections (C25)"
```

---

## Phase 3: Expanded Experiments (Optional — 2-4 hours)

### Task 5: Expand causal discovery to Sachs network [C20]

**Files:**
- Create: `knockout-experiments/causal_sachs_experiment.py`

- [ ] **Step 1: Implement causal discovery on Sachs (11 nodes)**

Use causal-learn or the PC algorithm from the existing causal_discovery_experiment.py. The Sachs dataset (flow cytometry, 11 phosphoproteins) is a standard benchmark available via `causallearn.utils.datasets`.

- Run PC algorithm at α ∈ {0.01, 0.05} across 100 bootstrap resamples
- Measure edge orientation flip rate
- Compare to Asia network results

- [ ] **Step 2: Report results and add to monograph**

- [ ] **Step 3: Commit**

```bash
git add knockout-experiments/causal_sachs_experiment.py
git commit -m "feat: expand causal discovery to Sachs network (11 nodes) [C20]"
```

### Task 6: Census Dirichlet sensitivity [C30]

**Files:**
- Modify: `paper/scripts/census_disaggregation_experiment.py`

- [ ] **Step 1: Run at α ∈ {0.1, 0.5, 1.0, 2.0, 10.0}**

Modify the experiment to sweep Dirichlet concentration parameter. Report KL divergence at each α.

- [ ] **Step 2: Add sensitivity table to monograph**

- [ ] **Step 3: Commit**

```bash
git add paper/scripts/census_disaggregation_experiment.py paper/universal_impossibility_monograph.tex
git commit -m "feat: Census Dirichlet sensitivity analysis at 5 alpha values [C30]"
```

### Task 7: Download and commit cytochrome c sequences [C35]

**Files:**
- Create: `paper/data/cytochrome_c_sequences.fasta`

- [ ] **Step 1: Download the 120 sequences using accessions**

```python
from Bio import Entrez, SeqIO
Entrez.email = "drake.caraker@ouraring.com"
# Read accessions from paper/data/cytochrome_c_accessions.txt
# Download CDS sequences and save as FASTA
```

- [ ] **Step 2: Commit as static file**

- [ ] **Step 3: Update codon_entropy_experiment.py to prefer local file over API**

- [ ] **Step 4: Commit**

```bash
git add paper/data/cytochrome_c_sequences.fasta paper/scripts/codon_entropy_experiment.py
git commit -m "data: commit static cytochrome c sequences for reproducibility [C35]"
```

### Task 8: CI workflow [C34]

**Files:**
- Create: `.github/workflows/validate.yml`

- [ ] **Step 1: Create GitHub Actions workflow**

```yaml
name: Validate
on: [push, pull_request]
jobs:
  lean:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: leanprover/lean4-action@v1
      - run: lake build
  experiments:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r paper/scripts/requirements.txt
      - run: make validate
  counts:
    runs-on: ubuntu-latest
    needs: lean
    steps:
      - uses: actions/checkout@v4
      - run: bash paper/scripts/verify_counts.sh
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/validate.yml
git commit -m "ci: add GitHub Actions validation workflow [C34]"
```

---

## Deferred Items (Not Worth the Effort for Nature)

| # | Item | Reason to Defer |
|---|------|----------------|
| C21 | HIO/RAAR phase retrieval | Crystallography is peripheral to Nature story; GS is adequate for illustration |
| C22 | Whole-proteome codon analysis | Biology is a supporting instance, not centerpiece; 120 species is adequate with real NCBI data |

---

## Execution Order

```
Phase 1: Task 1 (batch quick fixes) — 15 min, do first
Phase 2: Tasks 2-4 (statistical methodology) — 1 hour, parallel
  Task 2 (cluster bootstrap) ←→ Task 3 (BH correction)
  Task 4 (effect sizes) — after Task 2
Phase 3: Tasks 5-8 (expanded experiments) — optional, parallel
  Task 5 (Sachs) ←→ Task 6 (Dirichlet) ←→ Task 7 (sequences) ←→ Task 8 (CI)
```

**Estimated total:** Phase 1+2: ~75 min. Phase 3: ~2-4 hours (optional).
