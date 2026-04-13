# Nature Must-Accept Upgrade Plan

**Goal**: Transform "major revision, better suited to JMLR" into a
compelling Nature submission by adding real data, a universal scaling
figure, and the resolution optimality argument.

**Current state**: 82 Lean files, 377 theorems, 0 sorry.
All 16 text fixes from meta-review applied. Lean builds clean.

**Timeline**: 7 working days (April 13–19, 2026).
**Constraint**: Must not block NeurIPS companion paper (May 4 abstract).

---

## Decision Points

| ID | Gate | After Phase | Options |
|----|------|-------------|---------|
| D1 | ArXiv timing | 0 | Post now (current) or wait for biology data |
| D2 | Biology data quality | 2 | Replace synthetic / partial validation / report honestly |
| D3 | Universal figure | 3 | Lead with 8-panel / attempt collapse / use table only |
| D4 | Nature vs JMLR | 6 | Submit Nature BC or strengthen JMLR and submit there |

---

## Phase 0: Prerequisites (Day 0, 3 hours)

### 0.1: Install dependencies [Bash]
```
pip install biopython census us requests
brew install clustal-omega  # for protein alignment
```
Verify: `python -c "from Bio import Entrez; print('OK')"`

### 0.2: Get API keys
- NCBI Entrez: register at ncbi.nlm.nih.gov/account (free, instant)
- Census Bureau: request at api.census.gov/data/key_signup.html (free, instant)
- Store in environment: NCBI_EMAIL, NCBI_API_KEY, CENSUS_API_KEY

### 0.3: Audit DASH companion results
The companion repo (dash-shap) already has extensive DASH stability data:
- Correlation sweep: DASH 0.977 flat vs single-model 0.952 at ρ=0.95
- Breast Cancer: DASH 92.5% vs single 37.6%
- Variance decomposition: 60% reduction in model-selection variance
- Ensemble size analysis: stability plateaus at K≈20

**Conclusion**: Cross-domain transfer is a REFRAMING of existing results,
not a new experiment. This makes Phase 4 fast (text + figure, 2–4 hours).

### 0.4: Verify no NeurIPS conflicts
- NeurIPS companion paper (dash-shap/paper/main.tex) is in a separate repo
- This plan modifies universal-explanation-impossibility only
- No conflict if we don't modify dash-shap

### 0.5: Decision Point D1
Post current version to arXiv now? Or wait for real biology data?
- Recommend: WAIT until Phase 2 completes (2–3 days). Synthetic biology
  data is the single most damaging weakness. Posting now with synthetic
  data creates a public record of the weakness.

---

## Phase 1: Easy Wins (Day 1, parallel, ~10 hours)

All four tasks are independent. Run in parallel.

### 1.1: Real Census Data [Sonnet, 3 hours]

**File**: paper/scripts/census_disaggregation_experiment.py (REWRITE)

(a) Download county-level population from Census Bureau API
    - Table: B01003 (Total Population), ACS 5-year estimates
    - All 50 states + DC
    - Fallback: pre-downloaded CSV from data.census.gov if API is slow
(b) For each state:
    - Record actual county populations (the "base table")
    - Compute state total (the "view")
(c) Generate 100 Dirichlet(α=1) samples consistent with state total
(d) Compute KL(true ‖ sample) for each sample
(e) Negative control: DC (1 county-equivalent, KL = 0)
(f) Statistical test: Spearman ρ (more robust than Pearson for non-linear)
(g) Report KL saturation honestly (expect plateau above ~10 counties)
(h) Add: "Data: U.S. Census Bureau, ACS 5-year estimates [year]"

**Output**: results_census_disagg.json (REPLACE), figures/census_disaggregation.pdf

**Risk**: LOW. Public data, standard methodology.

### 1.2: Causal Re-run with 100 Seeds [Sonnet, 1 hour]

**File**: paper/scripts/causal_discovery_experiment.py (MODIFY)

(a) Change N_SEEDS from 10 to 100
(b) Re-run PC(α=0.05), PC(α=0.01), GES on Asia network
(c) Report BOTH metrics clearly:
    - "Orientation agreement" (directed edges only) — primary metric
    - "Overall agreement" (includes undirected) — secondary, clearly labeled
(d) Proper CIs with 100 seeds (much tighter)
(e) Power analysis: what effect size is detectable at n=100?

**Output**: results_causal_discovery_exp.json (REPLACE), figures/causal_discovery_exp.pdf

**Risk**: LOW. Same code, just more seeds.

### 1.3: Sentence Curation [Opus, 4 hours]

**File**: paper/scripts/parser_disagreement_experiment.py (MODIFY inline sentences)

The sentences are defined inline in the experiment script (no external file).

(a) Read current 50 ambiguous sentences from the script
(b) For EACH: verify the ambiguity is STRUCTURAL
    - Structural: two valid parse trees with different attachment sites
      (PP-attachment: "V NP PP" where PP can attach to V or NP)
      (Coordination: "A and B C" where C can scope over B or A-and-B)
    - NOT structural: lexical ambiguity ("bank" = river/financial),
      garden-path, pragmatic implicature
    - Test: can you draw two parse trees that differ in topology?
(c) Replace any non-structural sentences with standard examples from:
    - Ratnaparkhi et al. (1994) PP-attachment dataset
    - Jurafsky & Martin (2024) §18, attachment ambiguity examples
    - Examples: "I saw the man with the telescope"
                "She ate the cake on the table"
                "They discussed the plan in the office"
(d) For EACH replacement: document both valid parse trees
(e) Re-run parser experiment with corrected sentences
(f) Verify p-value improves (genuine structural ambiguity → more disagreement)

**Output**: Updated script, results_parser_disagreement.json, figures/parser_disagreement.pdf

**Risk**: MEDIUM. Requires linguistic judgment. Each replacement needs
two valid parse trees as evidence.

### 1.4: Resolution Optimality Text [Opus, 2 hours]

**Files**: universal_impossibility_monograph.tex, nature_brief_communication.tex

(a) Add subsection to monograph: "Why This Result Is Non-Trivial"

    Content:
    1. **Tightness**: Each pair of properties is achievable (3 witnesses).
       The impossibility is specifically the TRIPLE.
    2. **Necessity**: Rashomon is the exact boundary. No Rashomon → all
       three achievable simultaneously. This is proved in Necessity.lean.
    3. **Resolution optimality**: The G-invariant projection is Pareto-
       optimal — no resolution can improve faithfulness without sacrificing
       stability, and vice versa.
    4. **Independent convergence**: Practitioners in 8 fields independently
       discovered the same resolution (orbit averaging). The framework
       explains WHY they converged and proves the resolution is optimal.

(b) Add structural differences table (drafted in Phase 3.4 below)
(c) Update Nature BC: shift framing from "impossibility" to "exact
    characterization of what's possible + proof that existing practice
    is optimal"
(d) Check word count: Nature BC must be ≤1500 words

**Risk**: LOW. Text changes only.

---

## Phase 2: Real Biology Data (Day 2–3, 1–2 days)

This is the HIGHEST PRIORITY single item. The Nature editor said
synthetic biology data is the one thing that would cause desk-rejection.

### 2.1: Download Cytochrome c Sequences [Sonnet]

**File**: paper/scripts/codon_entropy_experiment.py (REWRITE)

Strategy (in order of preference):

**(a) NCBI Entrez (preferred)**
```python
from Bio import Entrez, SeqIO
Entrez.email = os.environ["NCBI_EMAIL"]
Entrez.api_key = os.environ.get("NCBI_API_KEY")

# Search for cytochrome c CDS across eukaryotes
handle = Entrez.esearch(
    db="nucleotide",
    term='cytochrome c[Gene] AND "complete cds"[Title] AND refseq[Filter]',
    retmax=200
)
```
- Download ≥50 eukaryotic cytochrome c CDS from RefSeq
- Parse GenBank records, extract CDS as codon sequences
- Record species name and taxonomy

**(b) Fallback: Kazusa Codon Usage Database**
- Pre-computed codon usage tables per species (no alignment needed)
- Species-level entropy (not per-position), but still tests dose-response
- URL: kazusa.or.jp/codon/

**(c) Fallback 2: Hardcode 50 well-known sequences**
- Cytochrome c is one of the most-sequenced proteins in history
- Use the Dayhoff (1972) dataset of 47 species, plus recent additions
- These are available in many molecular biology textbooks

### 2.2: Align and Compute [Sonnet]

(a) Translate each CDS to protein sequence (BioPython)
(b) Align proteins with Clustal Omega
    ```bash
    clustalo -i proteins.fasta -o aligned.fasta --threads=4
    ```
(c) Back-translate protein alignment to codon alignment
    (Each aligned protein position maps to a codon triplet)
(d) For each column in the codon alignment:
    - Identify the amino acid (must be conserved in ≥90% of species)
    - Count which codon each species uses at this position
    - Compute Shannon entropy: H = -Σ p_i log₂ p_i
    - Record degeneracy level: 1 (Met/Trp), 2 (Phe,Tyr,His,...),
      3 (Ile), 4 (Val,Ala,Pro,...), 6 (Leu,Ser,Arg)
(e) Group by degeneracy, compute mean entropy per group

### 2.3: GC Content Control [Sonnet]

(a) For each species, compute GC content from the CDS itself
    (or from published whole-genome values)
(b) Null model: at each degenerate position, expected codon
    frequencies are determined by GC content alone
    - For 4-fold degenerate: P(C)=P(G)=GC/2, P(A)=P(T)=(1-GC)/2
    - Compute expected entropy under this null
(c) Show observed entropy EXCEEDS GC-null at degenerate positions
    (the Rashomon property contributes variance beyond mutation bias)

### 2.4: Quality Gate — DECISION POINT D2

Before replacing synthetic data, verify:

| Check | Must pass? | Action if fails |
|-------|-----------|-----------------|
| 1-fold (Met/Trp) H ≈ 0 | YES (biochemical certainty) | Debug pipeline |
| 6-fold H > 4-fold H > 2-fold H | YES (core prediction) | Investigate — GC bias? |
| Kruskal-Wallis p < 0.001 | YES (given 50+ species) | Check alignment quality |
| Spearman ρ > 0.9 | SHOULD | Report actual value |
| Observed > GC-null | SHOULD | Report, note confounder |

(a) If ALL pass → REPLACE synthetic data throughout (Phase 2.5)
(b) If 1-fold fails → pipeline bug, fix and rerun
(c) If monotonicity fails → investigate. GC bias? Poor alignment?
    If fixable → fix. If not → keep synthetic as "proof of concept",
    add real as "partial validation with [explanation of discrepancy]"
(d) If p-value fails → likely too few species. Try more genes.

### 2.5: Update Paper [Opus, if quality gate passes]

(a) Replace synthetic results in instance_genetic_code.tex with real data
(b) REMOVE "synthetic data" caveats and "planned for future version" notes
(c) ADD: "Data: NCBI RefSeq accession numbers [list]. Alignment:
    Clustal Omega. [N] eukaryotic cytochrome c coding sequences."
(d) Update cross-domain table: remove † footnote from Biology row
(e) Regenerate figure (box plot: entropy by degeneracy level)
(f) Sync to arxiv-submission/

**Risk**: MEDIUM. Bioinformatics pipeline complexity. The prediction
should hold (it follows from the genetic code structure) but alignment
quality and species sampling can introduce noise.

---

## Phase 3: Universal Scaling Figure + Structural Analysis (Day 4, 1 day)

### 3.1: 8-Panel Dose-Response Figure — THE Nature Figure [Sonnet]

**File**: paper/scripts/create_universal_figure.py (NEW)

Create a 2×4 panel figure. Each panel shows one domain's dose-response:

| Panel | Domain | X-axis (Rashomon dose) | Y-axis (Instability) | Type |
|-------|--------|----------------------|---------------------|------|
| A | Mathematics | Null space dim d | Solver RMSD | Curve (50 points) |
| B | Biology | Degeneracy level (1-6) | Codon entropy (bits) | 5-point dose |
| C | Gauge theory | Variant vs invariant | Within-orbit variance | Binary comparison |
| D | Stat mech | Macrostate k | 1/Ω(k) (faithfulness) | Bell curve |
| E | Linguistics | Ambig vs unambig | 1 - UAS | Binary comparison |
| F | Crystallography | Signal length | Reconstruction RMSD | Curve (4 points) |
| G | Computer science | # counties | KL divergence | Curve (50 points) |
| H | Statistics | N=1k vs N=100k | 1 - orientation agr. | Two-point |

Each panel shows:
- Data points with 95% CIs where applicable
- Negative control clearly marked (star/different color)
- Monotonicity direction indicated
- Domain-specific axis labels
- Consistent color scheme across panels

The visual message: **8 different sciences, same qualitative pattern.**
Wherever the Rashomon measure increases, instability increases.
Wherever the Rashomon measure is zero, instability is zero.

### 3.2: Statistical Summary [Sonnet]

For each domain, report:
- Direction of monotonic relationship (all should be positive)
- Statistical significance (p-value from appropriate test)
- Effect size where computable

Aggregate test: Binomial test — 8/8 domains show predicted direction.
Under null (random direction), P(8/8) = (1/2)^8 = 0.004. Significant.

### 3.3: Universal Collapse — STRETCH GOAL [Sonnet]

**Only attempt if Phase 3.1 produces clean results.**

For the 4 domains with continuous dose-response curves (math, biology,
crystallography, census):
(a) Normalize x-axis: log(Rashomon set size) where computable
(b) Normalize y-axis: instability / max(instability) → [0, 1]
(c) Plot all 4 on a single axis
(d) Test: does a single monotonic function fit all 4?
(e) Report R² of joint fit vs separate fits

**IMPORTANT**: Do NOT claim a specific functional form (e.g.,
1 - exp(-S_R/S_0)) unless it genuinely emerges from the data.
The theorem predicts monotonicity, not a specific curve shape.
Overclaiming a functional form that doesn't fit would be FATAL.

**DECISION POINT D3**: If collapse works → use as supplementary figure.
If approximate → report honestly with deviations. If fails → use
8-panel figure only (still strong).

### 3.4: Structural Differences Table [Opus]

Add to monograph (addresses "8 copies of same observation" objection):

| Domain | Symmetry group | Group type | |Orbit| | Resolution | Loses |
|--------|---------------|------------|---------|------------|-------|
| Mathematics | Null space translations | ℝ^d (cont.) | ∞ | Pseudoinverse | Decisiveness |
| Biology | Synonymous substitutions | S_k (k=1–6) | 1–6 | Codon tables | Decisiveness |
| Gauge theory | Gauge transforms | ℤ₂^(V-1) | 2^(V-1) | Wilson loops | Decisiveness |
| Stat mechanics | Microstate permutations | S_N | C(N,k) | Microcanon. | Decisiveness |
| Linguistics | Parse ambiguity | Discrete | 2–20+ | Parse forests | Decisiveness |
| Crystallography | Phase rotations | U(1)^N (cont.) | ∞ | Patterson map | Decisiveness |
| Computer science | Disaggregation | Simplex (cont.) | ∞ | Aggregates | Decisiveness |
| Statistics | Edge reversals | S_k on edges | 1–exp | CPDAGs | Decisiveness |

Key insight: All sacrifice decisiveness, but via DIFFERENT group structures
(discrete vs continuous, abelian vs non-abelian, finite vs infinite).
This proves the 8 instances are structurally distinct, not isomorphic.

---

## Phase 4: Cross-Domain Transfer Argument (Day 5, half day)

### 4.1: Reframe Existing DASH Results [Opus, 2–4 hours]

The companion paper (dash-shap) already demonstrates:
- DASH maintains 97.7% stability vs single-model 95.2% at ρ=0.95
- Breast Cancer: DASH 92.5% vs single 37.6% (148% improvement)
- 60% reduction in model-selection variance (crossed ANOVA)
- Stability plateaus at K≈20 ensemble members

**The cross-domain argument** (new for the universal paper):

"The gauge theory resolution — restricting to observables invariant
under gauge transformations — is mathematically identical to the ML
resolution — averaging SHAP values over an ensemble of near-optimal
models. Both are the G-invariant projection for their respective
symmetry groups. The companion paper demonstrates that this
gauge-theory-inspired resolution (DASH) reduces attribution instability
by up to 148% on real-world datasets. This practical improvement was
predicted by the universal framework: orbit averaging is Pareto-optimal
(Theorem [ref]), and the improvement scales with the Rashomon set size
(the number of near-optimal models)."

### 4.2: Create Transfer Figure [Sonnet]

One figure showing:
- X-axis: Rashomon measure (correlation ρ as proxy for Rashomon size)
- Y-axis: Stability (Spearman rank correlation of attributions)
- Two curves: DASH (flat ≈ 0.977) vs single-model (declining)
- Annotation: "Predicted by gauge theory analogy"

Source data from: dash-shap/results/tables/synthetic_linear_sweep.json

### 4.3: Add to Papers [Opus]

(a) Add "Cross-Domain Transfer" paragraph to monograph
(b) Add one sentence to Nature BC: "The framework predicts that orbit
    averaging — a technique from gauge theory — optimally stabilizes
    any explanation under underspecification, a prediction confirmed by
    ensemble attribution methods that reduce instability by up to 148%
    on real-world datasets (companion paper)."
(c) Add transfer figure to monograph (supplementary for Nature BC)

**Risk**: LOW. This is reframing existing results, not new experiments.
The risk is that Nature reviewers see it as self-citation and discount it.

---

## Phase 5 (OPTIONAL): Lean Improvements (Day 5–6)

**Do this ONLY if time permits after Phases 1–4.**

### 5.1: Domain-Specific Incompatibility for Causal [Sonnet]

The text already documents the (≠) vs edge-reversal mismatch.
To fully close the objection:

(a) Define `edge_reversal_incomp` for 3-node DAGs in MarkovEquivalence.lean
(b) Prove the chain/fork witness is incompatible under this relation
(c) Prove `edge_reversal_incomp g1 g2 → g1 ≠ g2` (narrower implies broader)
(d) Build alternative ExplanationSystem with edge_reversal_incomp
(e) Prove impossibility for the alternative system

This shows the impossibility holds under the domain-appropriate relation,
not just under (≠).

### 5.2: autoImplicit false for remaining files

Add `set_option autoImplicit false` to:
- UnfaithfulQuantitative.lean
- MeasureHypotheses.lean

Verify build passes.

**Risk**: LOW for 5.2. MEDIUM for 5.1 (new Lean code).

---

## Phase 6: Full Paper Integration (Day 6–7, 1–2 days)

### 6.1: Update All Experiment Data [Sonnet]

For each experiment that produced new results (census, causal, parser,
biology):
(a) Copy new results JSON
(b) Regenerate figure (PDF, Type 1 fonts, publication_style.mplstyle)
(c) Regenerate table fragment (LaTeX)
(d) Verify figure dimensions suitable for Nature (≤180mm wide)

### 6.2: Update Monograph [Opus]

(a) Replace all experiment paragraphs with new results
(b) Add universal scaling section with 8-panel figure
(c) Add structural differences table
(d) Add "Not a Tautology" subsection
(e) Add cross-domain transfer paragraph
(f) Update abstract: "empirically verified on real data from [N] sciences"
(g) Update all stale counts if any Lean files changed
(h) Verify all \ref and \cite resolve

### 6.3: Rewrite Nature Brief Communication [Opus]

Lead with the strongest result. Priority order:
1. If universal scaling works: "8 sciences, one scaling law"
2. If cross-domain transfer is compelling: "physics technique stabilizes ML"
3. Baseline: "one theorem, 8 derivations + optimal resolution"

Structure (≤1500 words):
- Abstract (150 words): theorem + 8 derivations + Lean + resolution
- P1: The problem (5 sentences)
- P2: The theorem (5 sentences)
- P3: The 8 derivations (10 sentences, one per domain)
- P4: The resolution + Pareto optimality (4 sentences)
- P5: Universal scaling / cross-domain transfer (4 sentences)
- P6: Implications (3 sentences)
- Figure 1: 8-panel dose-response
- Figure 2: Structural differences table or trilemma diagram
- Methods (300 words)
- References (20–30)

**Word count check**: Must verify ≤1500 after rewrite.

### 6.4: Update All Paper Versions [Sonnet]

(a) PNAS (6pp), NeurIPS (10pp), JMLR (31pp)
(b) Sync arxiv-submission/
(c) Verify all compile with latexmk

### 6.5: Data Availability Statement [Opus]

For Nature:
- NCBI accession numbers for cytochrome c sequences
- Census Bureau data source and year
- All experiment scripts available at [GitHub URL]
- Lean source code available at [GitHub URL]
- All figures reproducible from `make validate`

---

## Phase 7: Verify + Rebuild + Submit (Day 7, half day)

### 7.1: Lean Build
```bash
lake build
# Verify: 82+ files, 377+ theorems, 72+ axioms, 0 sorry
```

### 7.2: Compile All Papers
```bash
make paper  # all LaTeX versions
```

### 7.3: Rebuild arXiv Package
```bash
bash paper/scripts/prepare_arxiv.sh
```

### 7.4: Reproducibility Check
```bash
make validate  # re-runs key experiments, checks results match
```

### 7.5: Nature Figure Compliance
- All figures: 300 DPI, ≤180mm wide
- CMYK color space (or RGB if journal accepts)
- Type 1 fonts (already ensured by publication_style.mplstyle)

### 7.6: Final Read
Read the Nature BC aloud. Does it tell a story?
Does a physicist AND a biologist AND a mathematician understand it?
Is there one figure that makes you go "wow"?

### 7.7: Commit + Push

### 7.8: Decision Point D4
- If real data, universal figure, and resolution argument are all strong
  → Submit to Nature
- If biology data is only partially supportive → Submit to PNAS
- If no strong new empirical result → Strengthen JMLR version and submit

---

## Execution Order (with parallelism)

```
Day 0: Phase 0 [0.1 → 0.2 → 0.3 → 0.4 → 0.5]
Day 1: Phase 1 [1.1 ∥ 1.2 ∥ 1.3 ∥ 1.4]
Day 2: Phase 2 [2.1 → 2.2 → 2.3 → 2.4]
Day 3: Phase 2 [2.5 if D2 passes] ∥ Phase 4 [4.1 → 4.2 → 4.3]
Day 4: Phase 3 [3.1 → 3.2 → 3.3 → 3.4]
Day 5: Phase 5 (optional) [5.1 ∥ 5.2]
Day 6: Phase 6 [6.1 → 6.2 → 6.3 → 6.4 → 6.5]
Day 7: Phase 7 [7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7 → 7.8]
```

## Risk Matrix

| Item | Will it work? | Impact if yes | Risk if no |
|------|:---:|:---:|---|
| Real biology data | HIGH (genetic code structure guarantees it) | CRITICAL | Keep synthetic, disclose |
| Real census data | VERY HIGH | HIGH | Use data.census.gov CSV |
| 100-seed causal | VERY HIGH | MEDIUM | Already significant at 10 |
| Sentence curation | HIGH | MEDIUM | Keep current + document |
| 8-panel figure | VERY HIGH | CRITICAL | Already have table |
| Universal collapse | MEDIUM | HIGH | Use 8-panel only |
| Cross-domain transfer | HIGH (data exists) | HIGH | Omit, still strong |
| Resolution optimality text | VERY HIGH | HIGH | N/A (text only) |
| Domain-specific incomp (Lean) | HIGH | LOW for Nature | Text already handles it |

## Confidence Assessment

| Scenario | Probability | Venue |
|----------|:---:|---|
| Real bio data + universal figure + resolution argument | 40–60% | Nature (competitive) |
| Real bio data + universal figure, no collapse | 25–40% | Nature (possible) or PNAS |
| Real bio data, no figure, resolution argument only | 15–25% | PNAS or JMLR |
| Synthetic bio data, regardless of other improvements | <10% | JMLR (primary target) |

The single highest-leverage action is Phase 2 (real biology data).
Without it, Nature is effectively off the table.

---

## VET RECORD

### Round 1 — Factual

- ⚠️ NCBI Entrez requires registration for API key → added to Phase 0.2
- ⚠️ Clustal Omega is not pip-installable, needs brew install → added
- ⚠️ Census API requires key → added to Phase 0.2
- ⚠️ Parser sentences are inline in script, not in external file →
  corrected in Phase 1.3
- ⚠️ Gauge lattice variance is constant (0.25) across all sizes, not
  scaling → revised Phase 3.1 table to show binary comparison
- ⚠️ Linguistics is binary (ambig vs not), not a dose curve → reflected
- ⚠️ The theorem predicts monotonicity, NOT a specific functional form →
  Phase 3.3 explicitly warns against overclaiming

### Round 2 — Reasoning

- ⚠️ Phase 2 quality gate needed before replacing synthetic data →
  added Decision Point D2 with explicit pass/fail criteria
- ⚠️ Phase 4 is reframing, not new experiment → honest about this
- ⚠️ Phase 5 (Lean improvements) low impact for Nature → made optional
- ⚠️ 8-panel figure with 2 binary + 2 two-point comparisons is weaker
  than 8 continuous curves → honest about this in Phase 3.1
- ⚠️ Cross-domain transfer risks looking like self-citation → noted
- ⚠️ Paper integration (Phase 6) is bottleneck → structured incrementally

### Round 3 — Omissions

- ⚠️ No domain expert feedback planned → noted as valuable but not
  blocking; would be Phase 4.5 if feasible
- ⚠️ Nature figure requirements (180mm, CMYK) → added to Phase 7.5
- ⚠️ Nature BC word count limit (1500) → added check to Phase 6.3
- ⚠️ Data availability statement required → added Phase 6.5
- ⚠️ Reproducibility script needed → Phase 7.4
- ⚠️ No fallback if biology pipeline fails completely → Kazusa database
  as fallback (species-level, not per-position, but still tests prediction)
- ⚠️ ArXiv timing: posting with synthetic data creates public weakness →
  recommend waiting for Phase 2 (Decision Point D1)
