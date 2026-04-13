# Execute Cross-Domain Experiments — Implementation Plan

**Goal**: Run 8 experiments across 8 sciences. Each produces
a results JSON, a figure, and a LaTeX table fragment.
Integrate all into the monograph and Nature Brief Communication.

**Model**: Sonnet for all experiment scripts (Python).
Opus for paper integration.

---

## VET RECORD

### Round 1 — Factual

- ⚠️ The biology experiment requires codon-aligned sequences
  for 50+ species. Ensembl Compara has these for some genes
  but access requires the Ensembl REST API or BioMart.
  FALLBACK: Use the Codon Usage Database (kazusa.or.jp)
  which has pre-computed codon usage tables per species.
  This avoids alignment but only gives species-level stats,
  not per-position codon entropy. For per-position entropy,
  NCBI Orthologs + BioPython codon alignment is needed.
  SIMPLER FALLBACK: Use a pre-aligned dataset from the
  literature. The Selectome database has codon-aligned
  orthologs. Or use the UCSC Genome Browser's multiz46way
  alignment which includes codon-level conservation.

- ⚠️ The linguistics experiment requires all 5 parsers
  installed. spaCy and Stanza are pip-installable. CoreNLP
  requires Java. Berkeley Neural Parser requires specific
  PyTorch version. FALLBACK: Use only pip-installable
  parsers (spaCy, Stanza, plus spacy-transformers for a
  third model). 3 parsers is sufficient if all are trained
  on Penn Treebank.

- ⚠️ The crystallography experiment uses Gerchberg-Saxton
  which requires iterative Fourier transforms. NumPy FFT
  handles this. No special libraries needed.

- ⚠️ The causal discovery experiment requires causal-learn
  or pcalg. causal-learn is pip-installable. Check if the
  Sachs dataset is bundled or needs download.

- ⚠️ The census experiment needs US Census API access.
  The census package for Python handles this, or use
  pre-downloaded CSV from data.census.gov.

### Round 2 — Reasoning

- ⚠️ The experiments are ORDERED by dependency: the math
  experiment (solvers) is the fastest and tests the
  infrastructure. Biology is the most important but hardest.
  Linguistics is the most accessible. Run easy ones first
  to catch pipeline issues.

- ⚠️ Each experiment should follow the SAME template as
  the ML experiments: use experiment_utils.py for seeds,
  CIs, figure saving. This ensures consistency.

- ⚠️ The stat mech "experiment" is a computation, not an
  experiment. It should produce a figure (entropy vs
  macrostate) but NOT be called an experiment. Frame as
  "Mathematical Observation" in the paper.

- ⚠️ For Nature, the FIGURE is key. Each experiment
  should produce one clean, publication-quality figure
  that tells the story at a glance. The most impactful
  figures: codon entropy dose-response, solver RMSD vs
  null space dimension.

### Round 3 — Omissions

- ⚠️ The plan should specify how FAILED experiments are
  handled. If the codon entropy doesn't show a clear
  dose-response (because GC bias dominates), the
  experiment is reported honestly and the domain is
  moved to "consistent with but not conclusively
  demonstrated."

- ⚠️ The plan should specify a DATA AVAILABILITY
  strategy. All datasets must be public or reproducible.
  Census data is public. Sachs data is published.
  Codon sequences are in NCBI. Parse trees are
  reproducible from parsers.

- ⚠️ The plan doesn't address FIGURE STYLE. All figures
  should match the existing ML experiment figures
  (publication_style.mplstyle, Type 1 fonts, PDF output).

---

## Phase 1: Easy Experiments [Day 1]

Run the 4 simplest experiments in parallel.

### Task 1.1: Mathematics — Solver Disagreement [Sonnet]

**File**: `paper/scripts/linear_solver_experiment.py`

Generate 100 random underdetermined systems (m×n, m<n).
Vary null space dim d = n-m from 1 to 50 (2 systems per d).
Solve with 5 methods: pseudoinverse, LSQR, L1 (scipy linprog
or sklearn Lasso), random null-space projection, Tikhonov.
Compute pairwise RMSD. Plot RMSD vs d.
Control: d=0 (square systems).

Output: results_linear_solver.json, figures/linear_solver.pdf,
sections/table_linear_solver.tex

### Task 1.2: Physics (Gauge) — Lattice Simulation [Sonnet]

**File**: `paper/scripts/gauge_lattice_experiment.py`

ℤ₂ lattice gauge theory on N×N grids (N=4,6,8,10).
Generate 1000 random configs per size.
Group by plaquette values (gauge-invariant).
Within each group: compute link variance (gauge-variant).
Plot: variant variance vs lattice size.
Control: 1×1 lattice.

Output: results_gauge_lattice.json, figures/gauge_lattice.pdf,
sections/table_gauge_lattice.tex

### Task 1.3: Physics (Stat Mech) — Entropy Computation [Sonnet]

**File**: `paper/scripts/stat_mech_entropy.py`

For N = 10, 20, 50 binary spins:
Compute Ω(k) = C(N,k) for each macrostate k.
Compute S_R(k) = ln Ω(k).
Compute max faithfulness = 1/Ω(k).
Plot: S_R vs k (bell curve peaking at N/2).
Plot: faithfulness vs entropy (monotonic decrease).

Output: results_stat_mech_entropy.json,
figures/stat_mech_entropy.pdf
(No table — this goes in theory section, not experiments.)

### Task 1.4: Computer Science — Census Disaggregation [Sonnet]

**File**: `paper/scripts/census_disaggregation_experiment.py`

Download or use pre-saved US county population data.
For each state: record county populations, compute state total.
Generate 100 Dirichlet samples consistent with state total.
Compute KL-divergence from true distribution.
Plot: KL vs number of counties per state.
Control: DC (1 county-equivalent).

Output: results_census_disagg.json,
figures/census_disaggregation.pdf,
sections/table_census.tex

FALLBACK if Census API fails: use synthetic data
(generate 50 "states" with 1-100 "counties" each,
known true distributions).

---

## Phase 2: Medium Experiments [Day 2-3]

### Task 2.1: Biology — Codon Entropy [Sonnet]

**File**: `paper/scripts/codon_entropy_experiment.py`

APPROACH 1 (preferred): Use Ensembl Compara or UCSC
multiz alignment to get codon-aligned orthologs for
cytochrome c (or ribosomal protein L7) across 50+ species.

APPROACH 2 (fallback): Use pre-computed codon usage
tables from Kazusa database. For each amino acid,
compute the entropy of codon usage across species.
This gives species-level entropy (not per-position)
but still tests the dose-response prediction.

APPROACH 3 (simplest fallback): Download 50 cytochrome c
CDS from NCBI using Entrez. Align proteins with Clustal.
Back-translate to codon alignment. Compute per-position
codon entropy. This is the most manual but most flexible.

For each approach: group positions by degeneracy level.
Plot: mean entropy per degeneracy level (box plot).
Test: Kruskal-Wallis for monotonic trend.
GC control: compute null-model entropy from species
GC content, show observed exceeds null.

Output: results_codon_entropy.json,
figures/codon_entropy.pdf,
sections/table_codon_entropy.tex

### Task 2.2: Linguistics — Parser Disagreement [Sonnet]

**File**: `paper/scripts/parser_disagreement_experiment.py`

Install: pip install spacy stanza
Download models: python -m spacy download en_core_web_trf
                 stanza.download('en')

Sentence sets:
- 50 ambiguous: manually curated PP-attachment and
  coordination ambiguity sentences. Source: Jurafsky &
  Martin textbook examples + Ratnaparkhi (1994) examples.
  Store in paper/data/ambiguous_sentences.txt
- 50 unambiguous: simple SVO sentences matched for length.
  Store in paper/data/unambiguous_sentences.txt

Parse with: spaCy (en_core_web_sm), spaCy (en_core_web_trf),
Stanza (default English model).
(3 parsers, all available via pip, all trained on UD/PTB)

Metric: For each sentence, extract dependency tree.
Compute pairwise Unlabeled Attachment Score (UAS).
Report mean inter-parser UAS for ambiguous vs unambiguous.

If CoreNLP is available (Java): add as 4th parser.

Output: results_parser_disagreement.json,
figures/parser_disagreement.pdf,
sections/table_parser.tex

### Task 2.3: Crystallography — Phase Retrieval [Sonnet]

**File**: `paper/scripts/phase_retrieval_experiment.py`

Test cases:
(a) 1D signal, length 64 (random positive + negative values)
(b) 2D image, 16×16 (random)
(c) 1D signal, length 64, positive (minimum-phase control)

For each: compute |FFT|², discard phase, reconstruct 20
times from random initial phases using Gerchberg-Saxton
(alternating projections: magnitude constraint in Fourier
domain, support constraint in real domain).

GS algorithm:
1. Start with random phases
2. Apply Fourier magnitude constraint: replace |F| with
   measured |F|, keep current phase
3. Inverse FFT
4. Apply real-space constraint (support, positivity if
   applicable)
5. FFT
6. Repeat 500 iterations

Measure: pairwise RMSD, feature agreement.
Plot: RMSD vs signal length (16, 32, 64, 128).
Control: positive signal (minimum-phase).

Output: results_phase_retrieval.json,
figures/phase_retrieval.pdf,
sections/table_phase_retrieval.tex

---

## Phase 3: Causal Discovery [Day 3]

### Task 3.1: Statistics — Algorithm Disagreement [Sonnet]

**File**: `paper/scripts/causal_discovery_experiment.py`

Install: pip install causal-learn

Datasets: Sachs (bundled in causal-learn or downloadable).
If Sachs unavailable: generate synthetic data from a known
DAG (Asia network) using causal-learn's data generators.

Run 3 algorithms: PC, GES, and one continuous method
(NOTEARS or DirectLiNGAM if data allows).

For each edge in the estimated graph:
- Classify as compelled or reversible (compute CPDAG)
- Measure inter-algorithm agreement

Plot: agreement rate for compelled vs reversible edges.
Control: very large sample (N=100,000) from known DAG.

Output: results_causal_discovery_exp.json,
figures/causal_discovery_exp.pdf,
sections/table_causal_discovery_exp.tex

---

## Phase 4: Paper Integration [Day 4]

### Task 4.1: Add All Experiments to Monograph [Opus]

For each experiment that produces clean results:
- Add \input{sections/table_X} to the appropriate
  derived instance section
- Add \includegraphics{figures/X} where appropriate
- Add an "Empirical illustration" paragraph to each
  instance section describing the result
- Update the cross-domain experiment summary table

### Task 4.2: Update Nature Brief Communication [Opus]

Add experiment results to the Nature BC:
- Main text: one sentence per experiment result
- The codon dose-response and parser disagreement
  get the most space (2-3 sentences each)
- Others: one sentence each
- Move detailed methodology to SI

### Task 4.3: Update Abstract [Opus]

Add to monograph abstract: "empirically verified in
[N] non-ML sciences" with the key numbers.

### Task 4.4: Compile Everything [Sonnet]

All paper versions + arXiv package.

---

## Phase 5: Verify + Commit [Sonnet]

### 5.1: Lean build (unchanged — no Lean changes)
### 5.2: Compile all paper versions
### 5.3: Rebuild arXiv package
### 5.4: Commit and push

---

## Execution Order

```
Day 1: Phase 1 [1.1 ∥ 1.2 ∥ 1.3 ∥ 1.4]  (4 easy experiments)
Day 2: Phase 2 [2.1 ∥ 2.2 ∥ 2.3]         (3 medium experiments)
Day 3: Phase 3 [3.1]                       (causal discovery)
Day 4: Phase 4 [4.1 → 4.2 → 4.3 → 4.4]   (paper integration)
       Phase 5 [5.1 → 5.2 → 5.3 → 5.4]   (verify + commit)
```

All Phase 1 experiments are independent. Phase 2 experiments
are independent. Phase 4 depends on experiment results.

## Fallback Strategy

If ANY experiment fails to produce clean results:
1. Report honestly in the paper
2. Move to "consistent with but not conclusively demonstrated"
3. The Lean derivation still stands — the experiment is
   illustrative, not constitutive
4. The theorem does not depend on any experiment

## Confidence

| Experiment | Will it work? | Risk |
|-----------|:---:|---------|
| Math (solvers) | VERY HIGH | Standard numerical methods |
| Gauge (lattice) | VERY HIGH | Pure computation |
| Stat mech (entropy) | VERY HIGH | Combinatorics |
| Census (disagg) | HIGH | Public data, standard stats |
| Biology (codons) | MEDIUM | Bioinformatics pipeline complexity |
| Linguistics (parsers) | HIGH | Standard NLP tools |
| Crystallography (phase) | HIGH | Standard signal processing |
| Causal (algorithms) | HIGH | Standard causal-learn |
