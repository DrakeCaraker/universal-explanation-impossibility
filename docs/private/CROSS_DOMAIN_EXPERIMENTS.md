# Cross-Domain Experiments — One Per Derived Instance

**Goal**: One unimpeachable experiment per derived domain.
Each must have: a testable prediction from the theorem,
real domain-specific data or computation, a negative control,
and a dose-response or scaling analysis.

**The standard**: A domain expert reviewing their own
experiment should say "this is correct methodology, the
prediction holds, and the control is valid."

---

## 1. MATHEMATICS — Solver Disagreement on Underdetermined Systems

**Prediction**: Different solvers for the same underdetermined
system produce different solutions. The disagreement scales
with the dimension of the null space (= the Rashomon set).

**Data**: Generate 100 random underdetermined systems Ax=b
where A is m×n with m < n. Vary the null space dimension
d = n - m from 1 to 50.

**Method**: Solve each system with 5 standard methods:
(a) NumPy pseudoinverse (minimum L2-norm)
(b) SciPy LSQR (iterative, random start)
(c) L1-minimization (basis pursuit via cvxpy)
(d) Random projection into null space + particular solution
(e) Tikhonov regularization (λ=0.01)

**Metric**: For each system, compute pairwise RMSD between
the 5 solutions. Report mean RMSD as a function of null
space dimension d.

**Prediction (quantitative)**: RMSD increases monotonically
with d. When d=0 (fully determined), RMSD=0. The theorem
predicts this: larger null space = larger Rashomon set =
more explanation instability.

**Negative control**: Square, full-rank systems (d=0).
All 5 solvers must agree (RMSD ≈ 0, up to numerical
precision ~1e-12).

**Resolution test**: The pseudoinverse (minimum-norm solution)
is the G-invariant projection under Euclidean symmetry of
the null space. Measure: is the pseudoinverse solution
the centroid of the other 4 solutions? (It should be closest
to the mean.)

**Effort**: Python + numpy + scipy + cvxpy. Half a day.

**Why it's unimpeachable**: Uses standard numerical methods
on randomly generated systems. No subjective choices. The
dose-response with null space dimension is a quantitative
prediction that any mathematician can verify.

---

## 2. BIOLOGY — Codon Entropy Across the Tree of Life

**Prediction**: For conserved proteins, codon usage entropy
scales with the degeneracy level of each amino acid position.
More synonymous codons → more "explanation variance" across
species.

**Data**: Cytochrome c coding sequences from 50+ eukaryotes.
Source: Ensembl Compara pre-computed codon alignments, or
NCBI Orthologs database.

**Method**: For each position in the multiple codon alignment:
(a) Identify the amino acid (conserved across species)
(b) Count which codons each species uses at this position
(c) Compute Shannon entropy: H = -Σ p_i log₂ p_i
(d) Record the degeneracy level of that amino acid
    (1-fold: Met, Trp; 2-fold: Phe, Tyr, His, etc.;
     3-fold: Ile; 4-fold: Val, Ala, Pro, etc.; 6-fold: Leu, Ser, Arg)
(e) Filter to positions conserved across ≥90% of species
    (to ensure we're measuring codon choice variation,
    not amino acid substitution)

**Metric**: Mean codon entropy per degeneracy level.
Box plot: entropy distribution for 1-fold, 2-fold, 3-fold,
4-fold, 6-fold positions.

**Prediction (quantitative)**: Monotonic increase.
1-fold: H = 0 (only one codon possible).
2-fold: H > 0 but < 1 bit.
4-fold: H > 2-fold.
6-fold: H > 4-fold.
Kruskal-Wallis test across degeneracy levels: p < 0.001.

**Negative control (intrinsic)**: Met and Trp positions
(degeneracy = 1). Entropy must be exactly 0 by
construction — there is only one codon for each.

**GC content control**: For each position, compute the
expected entropy under a null model where codon choice
is determined solely by the species' genomic GC content.
Show that observed entropy EXCEEDS the GC-null at
degenerate positions (the Rashomon property contributes
variance beyond what mutation bias explains).

**Resolution**: Codon optimization tables (as used in
Moderna's mRNA vaccine design) average over synonymous
codons weighted by expression data — the G-invariant
projection for molecular biology. Show: optimized codons
have LOWER cross-species variance than native codons
(the resolution reduces instability).

**Effort**: Python + BioPython + Ensembl REST API. One day.
May need 1-2 additional days for pipeline debugging if
codon alignment quality is uneven.

**Why it's unimpeachable**: Real genomic data from 50+
species. The negative control is biochemically intrinsic
(Met/Trp have no choice). The dose-response tests a
quantitative prediction. The GC correction addresses the
main confounder. A molecular biologist would find this
methodology standard.

---

## 3. PHYSICS (GAUGE) — Gauge-Variant vs Gauge-Invariant
      Measurements in Lattice Simulation

**Prediction**: On a discrete lattice, gauge-variant
quantities (individual link variables) show high variance
across gauge-equivalent configurations, while gauge-invariant
quantities (Wilson loops) show zero variance.

**Data**: ℤ₂ lattice gauge theory on a 10×10 periodic grid.

**Method**:
(a) Generate 1000 random link configurations
(b) Compute the plaquette (smallest Wilson loop) for each
(c) Group configurations by plaquette value (this is the
    "macrostate" / gauge-equivalence class)
(d) Within each group: measure variance of individual link
    values (gauge-variant) and variance of Wilson loop
    values (gauge-invariant)

**Metric**:
- Gauge-variant quantity variance within equivalence class
- Gauge-invariant quantity variance within equivalence class
- Ratio of the two

**Prediction**: Gauge-invariant variance = 0 (by construction).
Gauge-variant variance > 0 and scales with lattice size.

**Negative control**: Trivial lattice (1×1, single plaquette).
Only one configuration per plaquette value. Variant variance
= 0.

**Scaling**: Repeat for lattice sizes 4×4, 6×6, 8×8, 10×10.
Plot gauge-variant variance vs lattice size. The theorem
predicts monotonic increase (larger lattice = larger gauge
orbit = larger Rashomon set).

**Resolution**: Gauge-invariant observables (Wilson loops)
are the G-invariant projection. They're stable by construction.

**Effort**: Python + numpy. Half a day. Pure computation.

**Why it's unimpeachable**: Standard lattice gauge theory
methodology. Any physicist who has taken a QFT course
recognizes the setup. The gauge-invariant variance being
exactly zero is a mathematical fact, not a statistical claim.

---

## 4. PHYSICS (STAT MECH) — Microstate Entropy as
      Rashomon Entropy

**This is a theoretical observation, NOT an experiment.**

**The result**: For a discrete system of N particles with
macrostate M, the Rashomon set is the set of microstates
consistent with M. Its log-size is ln Ω = S/k_B, the
Boltzmann entropy. The impossibility theorem's quantitative
bound becomes: any stable explanation of M has faithfulness
bounded by exp(-S/k_B). Higher entropy → worse faithfulness.

**The computation** (illustrative, not experimental):
For N = 20 binary spins, plot for each macrostate k:
- The Rashomon set size Ω(k) = C(20,k)
- The Rashomon entropy S_R(k) = ln Ω(k)
- The maximum achievable faithfulness under stability:
  1/Ω(k) (probability of guessing the correct microstate)

Show: S_R(k) is maximized at k = N/2, where faithfulness
is minimized. The impossibility is MOST severe at the
macrostate with the highest entropy.

**Frame as**: A mathematical observation connecting the
impossibility bound to the second law, not an experiment.
Goes in the theory section alongside the information-
theoretic DPI proposition.

**Effort**: Python. Two hours.

---

## 5. LINGUISTICS — Parser Disagreement on Ambiguous vs
      Unambiguous Sentences

**Prediction**: Syntactic parsers disagree on structurally
ambiguous sentences but agree on unambiguous sentences.

**Data**: Two sentence sets:
(a) 50 PP-attachment ambiguous sentences from the
    Ratnaparkhi et al. (1994) PP-attachment dataset
    (or manually curated from Jurafsky & Martin examples)
(b) 50 unambiguous sentences matched for length and
    vocabulary complexity (simple SVO sentences, no
    modifiers, no embeddings)

**Method**: Parse all 100 sentences with 5 parsers, ALL
trained on the Penn Treebank (to control for training data):
(a) Stanford CoreNLP (PCFG)
(b) Stanford CoreNLP (neural)
(c) Berkeley Neural Parser
(d) spaCy (transformer-based)
(e) Stanza

**Metric**: For each sentence, compute pairwise Labeled
Attachment Score (LAS) between all 5 parsers' outputs.
Report mean inter-parser LAS for ambiguous vs unambiguous.

**Prediction (quantitative)**:
- Unambiguous: mean LAS > 95% (parsers agree)
- Ambiguous: mean LAS < 80% (parsers disagree)
- The difference is statistically significant (Wilcoxon
  p < 0.001)

**Negative control**: Unambiguous sentences. High LAS
confirms parsers can agree when there's no ambiguity.

**PP-attachment specific analysis**: For the subset of
ambiguous sentences where the ambiguity is specifically
PP-attachment ("V NP PP"), measure: what fraction of
parser pairs disagree on the PP attachment site? The
theorem predicts > 20%.

**Resolution**: k-best parse output (reporting top 5 parses
with probabilities) = the G-invariant projection for
parsing. Measure: does the true parse appear in the top-5
more often than in the top-1?

**Effort**: Python + spaCy + stanza + CoreNLP. One day.

**Why it's unimpeachable**: All parsers trained on the
same treebank (controls for training data). Pre-labeled
ambiguity set from the linguistics literature (not
subjective). Matched unambiguous control set. Standard
evaluation metric (LAS).

---

## 6. CRYSTALLOGRAPHY — Phase Retrieval Reconstruction
      Variance

**Prediction**: Reconstructions from the same magnitude
data but different initial phases produce different
electron density maps. Variance scales with the degree
of phase ambiguity.

**Data**: Three test cases:
(a) 1D signal, length 64 — moderate ambiguity
(b) 2D image, 32×32 — high ambiguity
(c) 2D image with positivity constraint — reduced ambiguity
    (positivity constrains the phase)

**Method**: For each test case:
(a) Compute the Fourier transform of the true signal
(b) Keep only the magnitudes (drop phases)
(c) Run 20 reconstructions with Gerchberg-Saxton (GS)
    algorithm from different random initial phases
(d) Measure pairwise RMSD between reconstructions
(e) Measure agreement on "structural features" (local
    maxima positions)

**Metric**: Mean pairwise RMSD across 20 reconstructions.
Feature agreement rate (fraction of local maxima that
appear in >50% of reconstructions).

**Prediction**:
- Test (a): moderate RMSD, moderate feature agreement
- Test (b): high RMSD, low feature agreement
- Test (c): low RMSD, high feature agreement (positivity
  constrains phase, reducing Rashomon)

**Negative control**: A minimum-phase signal (real,
positive, supported on half-line). Phase is uniquely
determined. All 20 reconstructions should converge.
RMSD ≈ 0.

**Scaling**: Plot RMSD vs signal length (16, 32, 64, 128).
Longer signals have more phase ambiguity. Predict
monotonic increase.

**Resolution**: The Patterson function (autocorrelation)
is the gauge-invariant observable — it's what you can
stably determine. Measure: do all 20 reconstructions have
the same autocorrelation? (Yes, by construction.)

**Effort**: Python + numpy FFT. One day.

**Why it's unimpeachable**: Standard phase retrieval
methodology. Any crystallographer recognizes Gerchberg-
Saxton. The minimum-phase control is a known special case.
The scaling analysis provides a quantitative prediction.

---

## 7. COMPUTER SCIENCE — View Disaggregation Ambiguity
      on Real Census Data

**Prediction**: Aggregated statistics can be disaggregated
in multiple ways, all consistent with the aggregate. The
ambiguity scales with the aggregation ratio.

**Data**: US Census Bureau county-level population data
(public, American Community Survey 5-year estimates).

**Method**:
(a) For each state, record the county-level populations
    (the "base table")
(b) Compute state-level totals (the "view" = aggregation)
(c) Given ONLY the state total, generate 100 synthetic
    county-level distributions consistent with that total
    (Dirichlet sampling with uniform prior)
(d) Measure: how different are the synthetic distributions
    from the true distribution?

**Metric**: Mean KL-divergence between synthetic and true
county distributions, as a function of the number of
counties per state.

**Prediction**: KL-divergence increases with number of
counties (more counties = more ways to disaggregate =
larger Rashomon set). States with 1 county equivalent
(DC) have KL = 0.

**Negative control**: DC (single "county"). The
disaggregation is unique. KL = 0.

**Scaling**: Plot KL vs number of counties. Texas (254
counties) should show much higher ambiguity than
Wyoming (23 counties). Predict monotonic scaling.

**Resolution**: The aggregate statistic (state total)
is the G-invariant projection — it's what's stably
determined. Any finer-grained "explanation" is ambiguous.

**Effort**: Python + Census API. Half a day.

**Why it's unimpeachable**: Real public data. Standard
statistical methodology (Dirichlet sampling). The
scaling with number of counties is a clean quantitative
prediction. No subjective choices.

---

## 8. STATISTICS — Causal Discovery Algorithm Disagreement

**Prediction**: Causal discovery algorithms disagree on
edge orientations within Markov equivalence classes but
agree on v-structures (compelled edges).

**Data**: Three benchmark datasets:
(a) Sachs et al. (2005) — protein signaling, 11 nodes
(b) Asia (Lauritzen & Spiegelhalter 1988) — 8 nodes
(c) ALARM (Beinlich et al. 1989) — 37 nodes

**Method**: Run 5 algorithms on each dataset:
(a) PC algorithm (constraint-based)
(b) GES (score-based, Chickering 2002)
(c) FCI (allows latent confounders)
(d) NOTEARS (continuous optimization, Zheng et al. 2018)
(e) DirectLiNGAM (if data is non-Gaussian)

For each edge in the true graph:
(a) Classify as: compelled (same orientation in all
    Markov-equivalent DAGs) or reversible (differs across
    Markov-equivalent DAGs)
(b) Measure inter-algorithm agreement rate

**Metric**: Agreement rate on compelled edges vs
reversible edges.

**Prediction**:
- Compelled edges: agreement > 80%
- Reversible edges: agreement < 50%
- The gap is statistically significant (Fisher exact test)

**Negative control**: Run on data generated from the
KNOWN ground-truth DAG with infinite samples (N=100,000).
With enough data, all constraint-based methods should
converge to the CPDAG. Agreement should be high on all
edges.

**Resolution**: The CPDAG is the G-invariant projection.
Measure: does the CPDAG (undirected for reversible edges)
agree across all 5 algorithms? (It should, approximately.)

**Effort**: Python + causal-learn library. One day.

**Why it's unimpeachable**: Standard causal discovery
benchmarks. Standard algorithms. The compelled/reversible
edge classification comes directly from Markov equivalence
theory (Verma & Pearl 1991). Any causal inference
researcher recognizes this setup.

---

## Priority and Effort

| # | Domain | Effort | Impact | Do? |
|---|--------|--------|--------|-----|
| 1 | Mathematics (solvers) | 0.5 day | Medium | YES — easy baseline |
| 2 | Biology (codons) | 1-2 days | **Very high** | **YES — Nature's biology reviewer** |
| 3 | Physics (gauge lattice) | 0.5 day | Medium | YES — validates gauge instance |
| 4 | Physics (stat mech) | 2 hours | High (as theory) | YES — but frame as math, not experiment |
| 5 | Linguistics (parsers) | 1 day | High | **YES — accessible to all readers** |
| 6 | Crystallography (phase) | 1 day | High | YES — Nobel-adjacent |
| 7 | CS (census disagg.) | 0.5 day | Medium | YES — uses real public data |
| 8 | Statistics (causal) | 1 day | Medium | YES — standard benchmarks |

**Total: ~7 days for all 8.**

## The Nature-optimal subset

If doing all 8 is too much, the minimum viable set:

**Must do**: Biology (codons) + Linguistics (parsers)
These are the two experiments on REAL DATA from NON-ML
sciences with clean controls and quantitative predictions.

**Should do**: Crystallography (phase) + Mathematics (solvers)
These add physics/math with clean scaling analyses.

**Nice to have**: Gauge lattice + Census + Causal
These complete the picture but have lower marginal value.

**Frame as theory**: Stat mech entropy connection
This is a mathematical observation, not an experiment.
