# Nature Must-Accept Comprehensive Vet Plan

> **For agentic workers:** Execute phases sequentially. Within each phase, launch reviewer agents in parallel. Each reviewer returns a structured report with FATAL/MAJOR/MINOR/STYLISTIC findings. Phase 4 synthesizes all findings before Phase 5 fixes them.

**Goal:** Vet the Nature Article (`paper/nature_article.tex`), the monograph (`paper/universal_impossibility_monograph.tex`), the Lean formalization (82 files), and all experiments to the standard of unanimous acceptance at Nature.

**Standard:** Every reviewer finds no FATAL issues. The paper survives 25 hostile reviewers — domain experts, methodologists, a Nature editor, a philosopher, and a devil's advocate — all seeing this for the first time.

**Root directory:** `/Users/drake.caraker/ds_projects/universal-explanation-impossibility`

---

## Severity Classification

| Level | Definition | Impact |
|-------|-----------|--------|
| **FATAL** | Factual error, incorrect theorem, wrong data, claim contradicted by evidence | Blocks submission. Must fix. |
| **MAJOR** | Misleading framing, overclaiming, stale number, weak methodology, missing control | Likely causes reviewer rejection. Must fix. |
| **MINOR** | Imprecise language, suboptimal figure, unclear passage, missing citation | Would appear in reviewer report but not block acceptance. Should fix. |
| **STYLISTIC** | Formatting, typo, word choice, figure aesthetics | Fix if time permits. |

---

## Phase 0: Foundation Verification

**Executor:** 1 agent, sequential. Must pass before any review begins.

- [ ] **0.1: Lean build**
```bash
cd /Users/drake.caraker/ds_projects/universal-explanation-impossibility
lake build 2>&1 | tail -5
```
Expected: Clean build, no errors.

- [ ] **0.2: Lean counts — verify paper claims**
```bash
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
```
Expected: 82 files, 377 theorems+lemmas, 72 axioms, 0 sorry. If ANY number differs from paper, that is FATAL.

- [ ] **0.3: Core theorem axiom check**
Run in Lean: `#print axioms explanation_impossibility`
Expected: Only `propext`, `Quot.sound`, `Classical.choice` (Lean kernel axioms). Zero domain-specific axioms.

- [ ] **0.4: All 8 derived instances axiom check**
For each of: `GeneticCode.lean`, `GaugeTheory.lean`, `LinearSystem.lean`, `StatisticalMechanics.lean`, `SyntacticAmbiguity.lean`, `PhaseProblem.lean`, `ViewUpdate.lean`, `CausalDiscovery.lean` (in `UniversalImpossibility/`):
Run `#print axioms <instance_impossibility_theorem>`.
Expected: Zero domain-specific axioms for each.

- [ ] **0.5: Compile Nature Article**
```bash
cd paper && latexmk -pdf -interaction=nonstopmode nature_article.tex
```
Verify: 0 errors, 0 undefined references, 0 undefined citations.

- [ ] **0.6: Compile monograph**
```bash
cd paper && latexmk -pdf -interaction=nonstopmode universal_impossibility_monograph.tex
```
Verify: 0 errors.

- [ ] **0.7: Run all experiments and verify results match JSONs**
```bash
cd paper && python3 scripts/run_all_universal_experiments.py
```
Compare outputs against existing `results_*.json` files. Flag any discrepancy.

- [ ] **0.8: Reference check**
Verify every `\citep` and `\citet` in `nature_article.tex` resolves to a valid entry in `references.bib`. Spot-check 5 references for correct author names and years.

- [ ] **0.9: Figure compliance**
For each PDF in `paper/figures/`: verify ≤180mm wide, Type 1 fonts (check with `pdffonts`), legible at 50% zoom.

**Gate:** If Phase 0 has ANY failure, fix before proceeding.

---

## Phase 1: Domain Expert Review (16 reviewers, 2 per domain)

Launch 8 parallel groups (2 reviewers each). Each reviewer reads ONLY their assigned files and answers their specific questions. Report format: finding, severity, evidence, suggested fix.

---

### Domain 1: Mathematics (Linear Algebra)

**R1: Numerical Linear Algebraist**

Read:
- `UniversalImpossibility/LinearSystem.lean`
- `paper/sections/instance_linear_system.tex`
- `paper/scripts/linear_solver_experiment.py`
- `paper/results_linear_solver.json`
- Nature Article §"Eight sciences" (mathematics paragraph)

Questions:
1. Is the underdetermined system $x_1 + x_2 = 2$ with solutions $(1,1)$ and $(0,2)$ correctly formalized in Lean?
2. Are the 4 solvers (least-squares, ridge, minimum-norm, randomized projection) standard numerical methods? Are they correctly implemented?
3. Does pairwise RMSD increase monotonically with null-space dimension as claimed?
4. Is Mann-Whitney $U$ appropriate for comparing RMSD distributions across conditions?
5. Is $p = 6.3 \times 10^{-9}$ consistent with the results JSON? Is it plausible?
6. Is the negative control ($m = d$, full rank) genuine — do all solvers agree exactly?
7. Does the paper accurately describe what the experiment shows?

**R2: Algebraist / Group Theorist**

Read:
- `UniversalImpossibility/LinearSystem.lean`
- `UniversalImpossibility/ExplanationSystem.lean`
- `UniversalImpossibility/UniversalResolution.lean`
- Nature Article structural differences table (Figure 2)

Questions:
1. Is the null space correctly identified as the symmetry group acting on solutions?
2. Is the pseudoinverse correctly characterized as the $G$-invariant projection (orbit average) over null-space translations?
3. Is $\mathbb{R}^{n-r}$ the correct characterization of this symmetry group?
4. Is "continuous, abelian" a correct classification?
5. Is the orbit-averaging framework mathematically sound for this group (integral over continuous group)?

---

### Domain 2: Biology (Genetics)

**R3: Molecular Biologist / Geneticist**

Read:
- `UniversalImpossibility/GeneticCode.lean`
- `paper/sections/instance_genetic_code.tex`
- `paper/results_codon_entropy.json` (real data section)
- Nature Article §"Eight sciences" (biology paragraph)

Questions:
1. Is the standard genetic code correctly represented? Are there any errors in codon-to-amino-acid mappings?
2. Is "UCU and UCC both encode Serine" correct?
3. Is the degeneracy classification (1-fold: Met/Trp; 2-fold: Phe, Tyr, etc.; 3-fold: Ile; 4-fold: Val, Ala, etc.; 6-fold: Leu, Ser, Arg) correct? Are there edge cases (e.g., start codons, selenocysteine)?
4. Is cytochrome c an appropriate gene for cross-species codon usage analysis? Would a reviewer suggest a different gene?
5. Does the 120-species dataset from NCBI RefSeq adequately represent eukaryotic diversity, or is there sampling bias (e.g., overrepresentation of mammals)?
6. Is the negative control (Met/Trp, entropy = 0) genuinely guaranteed by biochemistry?
7. Are codon usage tables correctly identified as the domain-specific resolution? Is this standard practice in the field?
8. Is the dose-response prediction (entropy scales with degeneracy) biologically meaningful or trivially expected?

**R4: Bioinformatician**

Read:
- `paper/scripts/codon_entropy_experiment.py`
- `paper/results_codon_entropy.json`

Questions:
1. Is the NCBI query (`CYCS[Gene] AND mRNA[Filter] AND refseq[Filter]`) appropriate and reproducible?
2. Is deduplication by organism sufficient, or should it be by taxonomic family to avoid overrepresentation?
3. Is per-amino-acid codon entropy (aggregated across all species and positions) the right metric? Would per-position entropy across aligned sequences be more standard?
4. Is the GC-null comparison well-designed? Does the null model correctly capture GC bias?
5. Is Kruskal-Wallis appropriate for 5 degeneracy groups with very different sample sizes (2 vs 9 vs 1 vs 5 vs 3 amino acids)?
6. With only 20 amino acids total, is the effective sample size adequate for the claimed statistics?
7. Are there confounders (CpG depletion, codon usage bias, translational selection) not accounted for?

---

### Domain 3: Physics (Gauge Theory)

**R5: Theoretical Physicist**

Read:
- `UniversalImpossibility/GaugeTheory.lean`
- `paper/sections/instance_gauge_theory.tex`
- `paper/scripts/gauge_lattice_experiment.py`
- `paper/results_gauge_lattice.json`
- Nature Article §"Eight sciences" (physics paragraph)

Questions:
1. Is the $\mathbb{Z}_2$ gauge theory on a triangle graph correctly formalized?
2. Is holonomy (XOR around the triangle) the correct discrete analogue of a Wilson loop?
3. Does `gauge_preserves_holonomy` hold in general, not just for the two witnesses?
4. Is the claim that gauge-invariant observables constitute "orbit averaging" accurate in the physics sense? A physicist would say gauge fixing, not orbit averaging — is the paper's framing a stretch?
5. Is the connection to continuous gauge theory (E&M, Yang-Mills) appropriately qualified, or overstated?
6. Would a Nature physics reviewer accept a 3-vertex, $\mathbb{Z}_2$ toy model as representing "gauge theory"?

**R6: Lattice Gauge Theory / Mathematical Physics Specialist**

Read:
- `UniversalImpossibility/GaugeTheory.lean`
- `UniversalImpossibility/UniversalResolution.lean`

Questions:
1. Is $\mathbb{Z}_2^{|V|-1}$ the correct size of the gauge group for a triangle graph? (Should be $\mathbb{Z}_2^{|V|-1} = \mathbb{Z}_2^2$ for 3 vertices.)
2. Is the within-orbit variance experiment meaningful, or is it trivially 0.25 for all configurations?
3. Does the formalization capture the essential mathematical structure of lattice gauge theory, or is it missing critical features (e.g., plaquette actions, coupling constants)?
4. Would a lattice gauge theorist cite this as correctly representing their field?

---

### Domain 4: Statistical Mechanics

**R7: Statistical Physicist**

Read:
- `UniversalImpossibility/StatisticalMechanics.lean`
- `paper/sections/instance_stat_mech.tex`
- `paper/results_stat_mech_entropy.json`
- Nature Article §"Eight sciences" (stat mech paragraph)

Questions:
1. Is the microstate/macrostate framing standard and correct?
2. Are $(H,T)$ and $(T,H)$ correctly identified as distinct microstates of the same macrostate?
3. Is the claim "Rashomon entropy = Boltzmann entropy" accurately stated? Is this an exact equality or an analogy?
4. Is a 2-coin system a legitimate "statistical mechanics" example, or is it too trivial? Would a reviewer say this is just combinatorics, not statistical mechanics?
5. Is the microcanonical ensemble correctly identified as the orbit-averaging resolution?
6. Is the connection to the canonical ensemble appropriate or confusing?

**R8: Information Theorist / Thermodynamicist**

Read:
- `UniversalImpossibility/StatisticalMechanics.lean`
- `UniversalImpossibility/Ubiquity.lean`

Questions:
1. Is the entropy calculation $\log_2 \binom{N}{k}$ correct for the binomial macrostates?
2. Is the "ubiquity" argument (generic underspecification in high dimensions) sound from a statistical mechanics perspective?
3. Is the claim that the Rashomon property is "generic" appropriately qualified?
4. Does the dimensional argument in Ubiquity.lean hold up to scrutiny?

---

### Domain 5: Linguistics

**R9: Computational Linguist**

Read:
- `UniversalImpossibility/SyntacticAmbiguity.lean`
- `paper/sections/instance_syntax.tex`
- `paper/scripts/parser_disagreement_experiment.py`
- `paper/results_parser_disagreement.json`
- Nature Article §"Eight sciences" (linguistics paragraph)

Questions:
1. Is "V NP PP" attachment ambiguity correctly described?
2. Are the 50 ambiguous sentences genuinely structurally ambiguous (two valid parse trees differing in topology), or do some exhibit only lexical ambiguity?
3. Are the 50 unambiguous sentences genuinely unambiguous for all 4 parsers?
4. Is UAS (unlabeled attachment score) the right metric for measuring inter-parser agreement?
5. Is Wilcoxon rank-sum the appropriate test for this comparison?
6. Is $p = 1.6 \times 10^{-3}$ consistent with the results JSON? Is it convincing with 4 parsers and 50+50 sentences?
7. Is "packed parse forest" correctly described as the linguistic resolution?
8. Are spaCy sm/md/lg and Stanza standard parsers for this kind of study?

**R10: Formal Syntactician**

Read:
- `UniversalImpossibility/SyntacticAmbiguity.lean`
- `paper/sections/instance_syntax.tex`

Questions:
1. Is the Lean formalization of parse trees accurate? Does it capture genuine syntactic structure?
2. Is the left-attach vs right-attach characterization of PP-attachment ambiguity standard?
3. Is Chomsky (1957) the appropriate citation for structural ambiguity? Would a linguist prefer Jurafsky & Martin, or Church & Patil (1982)?
4. Would a syntactician consider the Lean formalization (two-element parse tree type) a fair representation?

---

### Domain 6: Crystallography

**R11: Structural Biologist / Crystallographer**

Read:
- `UniversalImpossibility/PhaseProblem.lean`
- `paper/sections/instance_phase_problem.tex`
- `paper/scripts/phase_retrieval_experiment.py`
- `paper/results_phase_retrieval.json`
- Nature Article §"Eight sciences" (crystallography paragraph)

Questions:
1. Is the phase problem correctly described in the paper?
2. Is the energy-based formalization (sum of squares of signal components) a valid simplification of the Fourier magnitude problem?
3. Is Gerchberg-Saxton correctly implemented in the experiment?
4. Is the positivity constraint the right control for reducing the Rashomon set?
5. Is Hauptman & Karle (1953/1985) the appropriate citation?
6. Would a structural biologist find the 1.5-1.8$\times$ divergence ratio realistic?
7. Is the Patterson map correctly identified as the domain-specific resolution?

**R12: Phase Retrieval / Applied Mathematics Specialist**

Read:
- `UniversalImpossibility/PhaseProblem.lean`
- `paper/scripts/phase_retrieval_experiment.py`

Questions:
1. Is the signal model (finite-dimensional, real-valued) appropriate as a stand-in for the phase problem?
2. Is $U(1)^n$ the correct symmetry group? (Phase rotations on each component independently?)
3. Is reconstruction RMSD a standard metric in phase retrieval literature?
4. Are signal lengths 4-256 a reasonable range for demonstrating the dose-response?
5. Is $p < 10^{-57}$ plausible, and is Mann-Whitney $U$ appropriate here?

---

### Domain 7: Computer Science (Databases)

**R13: Database Theorist**

Read:
- `UniversalImpossibility/ViewUpdate.lean`
- `paper/sections/instance_view_update.tex`
- Nature Article §"Eight sciences" (CS paragraph)

Questions:
1. Is the view update problem correctly formalized? Is the projection (hiding one column) a fair representation?
2. Is Bancilhon & Spyratos (1981) correctly cited and characterized?
3. Is "complement view" correctly identified as the domain-specific resolution?
4. Is the two-row, two-column example ($(T,T)$ and $(T,F)$ projecting to the same view) correct?
5. Would a database theorist accept this formalization, or would they object that the view update problem is more nuanced (e.g., involves update translation, not just query ambiguity)?

**R14: Census / Disaggregation Specialist**

Read:
- `paper/scripts/census_disaggregation_experiment.py`
- `paper/results_census_disagg.json`

Questions:
1. Are the 2020 Census county counts correctly hardcoded?
2. Is Dirichlet($\alpha = 1$) an appropriate prior for generating county populations consistent with a state total?
3. Is KL divergence the right metric for disaggregation ambiguity?
4. Is the Zipf(s=1.0) model for county population distribution appropriate?
5. Is the DC negative control (1 county-equivalent, KL = 0) genuine?
6. Is Spearman rank correlation (county count vs mean KL) the right test?
7. Would a demographer or privacy researcher find this methodology standard?

---

### Domain 8: Statistics (Causal Inference)

**R15: Causal Inference Researcher**

Read:
- `UniversalImpossibility/CausalDiscovery.lean`
- `UniversalImpossibility/MarkovEquivalence.lean`
- `paper/sections/instance_causal.tex`
- `paper/scripts/causal_discovery_experiment.py`
- `paper/results_causal_discovery_exp.json`
- Nature Article §"Eight sciences" (statistics paragraph)

Questions:
1. Is chain $A \to B \to C$ and fork $A \leftarrow B \to C$ correctly identified as Markov equivalent?
2. Does the Lean formalization of `ciFromDAG` correctly capture conditional independence? Does ignoring colliders introduce errors?
3. Are PC ($\alpha = 0.05$, $\alpha = 0.01$) and GES standard algorithms?
4. Is the Asia network (8 nodes) an appropriate benchmark?
5. Is 100 seeds per condition adequate for statistical power?
6. Is orientation agreement (directed edges only) the right primary metric?
7. Is $p = 4.4 \times 10^{-38}$ consistent with the JSON? Is Mann-Whitney $U$ appropriate?
8. Would Pearl, Spirtes, or Chickering agree with this formalization?

**R16: Graphical Models Expert**

Read:
- `UniversalImpossibility/MarkovEquivalence.lean`
- `UniversalImpossibility/CausalInstance.lean`
- `UniversalImpossibility/CPDAGResolution.lean`

Questions:
1. Is the CPDAG correctly identified as the $G$-invariant resolution?
2. Is the Markov equivalence class correctly characterized in Lean?
3. Is the edge-reversal symmetry group correctly formalized?
4. Does the Necessity theorem (no Rashomon $\to$ all three properties achievable) hold for the causal domain?
5. Is it accurate to say that observational data cannot distinguish Markov-equivalent graphs? (What about faithfulness assumption violations?)

---

## Phase 2: Cross-Cutting Review (6 reviewers, parallel)

Each reviewer reads the full Nature Article and/or monograph from their specific angle.

---

**R17: Lean 4 / Formal Verification Expert**

Read: ALL 82 `.lean` files (scan headers + key theorems). Run `lake build`.

Questions:
1. Does `lake build` succeed with 0 errors?
2. Does `#print axioms explanation_impossibility` show zero domain-specific axioms?
3. Do ALL 8 derived instance impossibility theorems have zero domain-specific axioms?
4. Do the Lean type definitions match the paper's mathematical definitions exactly?
5. Is `autoImplicit false` set in all files?
6. Are there any `sorry` statements (even in comments counted by grep)?
7. Are the 72 axioms correctly classified in the paper's axiom inventory?
8. Is the `ExplanationSystem` structure correctly parametric?
9. Does the tightness proof (each pair achievable) actually construct valid witnesses?
10. Does the Necessity proof (no Rashomon $\to$ all three achievable) use the correct converse?

---

**R18: Experimental Statistician**

Read: ALL 16 `results_*.json` files, `paper/scripts/experiment_utils.py`, Nature Article Methods section.

Questions:
1. For EACH of the 7 cross-domain experiments: is the statistical test appropriate for the data type and hypothesis?
2. For EACH: are bootstrap CIs computed correctly (percentile method, 2000 resamples)?
3. For EACH: is the sample size adequate for the claimed effect?
4. For EACH negative control: is it genuine (nonzero comparison, not vacuous)?
5. Is there a multiple comparison problem across 7 experiments? Is the paper's justification (independent tests, binomial aggregation) sound?
6. Could ANY positive result be explained by a confounder rather than the Rashomon property?
7. Is the binomial test (7/7 domains, $p = 0.008$) correctly computed and validly applied?
8. Do the experiments test the theorem's *prediction* (Rashomon $\to$ instability) or merely demonstrate a *correlation*?
9. For the biology experiment: with only 20 amino acids grouped into 5 degeneracy levels, is the effective $N$ adequate for Kruskal-Wallis?

---

**R19: Nature Article $\leftrightarrow$ Monograph Consistency Checker**

Read: `paper/nature_article.tex`, `paper/universal_impossibility_monograph.tex`, ALL `results_*.json`.

Questions:
1. Does every number in the Nature Article match the corresponding number in the monograph?
2. Does every number in both papers match its source JSON?
3. Do the Lean counts (82/377/72/0) in the paper match the actual counts from Phase 0?
4. Does every p-value in the Nature Article match the monograph and the JSON?
5. Does every figure caption accurately describe the figure content?
6. Does the 8-panel figure data match the 8 corresponding results JSONs?
7. Are there any claims in the Nature Article not supported by the monograph?
8. Are there stale numbers from previous versions (e.g., "50 species" that should be "120", "p = 10^{-17}" that should be "p = 2.0 × 10^{-3}")?

---

**R20: Resolution Convergence Validator**

Read: Nature Article §"The resolution", monograph §"The resolution", ALL 8 `instance_*.tex` sections (resolution paragraphs).

For EACH of the 8 domain resolutions, answer:
1. Is "orbit averaging" an accurate mathematical characterization of this domain's resolution?
2. At what level of abstraction does this become a stretch?
3. Would a domain expert recognize their field's practice in this description?

Specific checks:
- Gauge-invariant observables: is this orbit averaging, or gauge fixing? (These are different operations.)
- Pseudoinverse: is selecting the minimum-norm solution the same as averaging over the null space? (Technically it's projecting, not averaging.)
- CPDAG: is reporting the equivalence class the same as orbit averaging? (It's more like orbit identification.)
- Microcanonical ensemble: is this genuinely averaging over microstates? (Yes for thermodynamic quantities.)
- Codon usage tables: is reporting frequency distributions orbit averaging? (Yes — marginalizing over synonymous codons.)
- Patterson map: is extracting phase-invariant information orbit averaging? (It extracts the autocorrelation, which is phase-invariant.)
- Packed parse forest: is representing all parses simultaneously orbit averaging? (Arguably it's orbit enumeration, not averaging.)
- Complement view: is restricting to the unambiguous projection orbit averaging? (It's projection, not averaging.)

For any characterization that is a stretch, assess: FATAL (fundamentally misleading), MAJOR (inaccurate but fixable with rewording), or MINOR (matter of perspective)?

---

**R21: Non-Isomorphism & Structural Diversity Validator**

Read: Nature Article Figure 2 (structural differences table), ALL 8 derived instance Lean files.

Questions:
1. Are the 8 symmetry groups genuinely mathematically distinct?
2. Could any pair of instances be shown to be isomorphic (same group, same structure, just relabeled)?
3. Is the variation in group type (finite/infinite, abelian/non-abelian, discrete/continuous) genuine and correctly stated?
4. Does the table accurately represent each domain's symmetry group?
5. Is the claim "not notational variants of a single example" well-supported?
6. Could a hostile reviewer argue that the 8 instances are trivially different (different types, but all 2-element witnesses with the same logical structure)?

---

**R22: Science Communicator / Accessibility Reviewer**

Read: `paper/nature_article.tex` (main text only, not Methods).

Questions:
1. Can a biologist understand the physics (gauge theory) paragraph without specialized knowledge?
2. Can a physicist understand the biology (codon degeneracy) paragraph?
3. Can a computer scientist understand the linguistics (parse trees) paragraph?
4. Is the theorem stated clearly enough for a scientist with no formal methods background?
5. Is jargon minimized? Flag any terms that need glossing for a Nature audience.
6. Does the paper tell a compelling story, or does it read as a list of 8 examples?
7. Is the cover letter (`paper/nature_cover_letter.tex`) compelling? Would it survive a 30-second skim by an editor?
8. Are the regulatory claims (EU AI Act, FDA) accurately stated, or overclaiming?
9. Are the historical dates in the Resolution section (Boltzmann 1870s, Weyl 1920s, Patterson 1930s, etc.) correct?

---

## Phase 3: Adversarial Review (3 reviewers, parallel)

---

**R23: Nature Editor Simulation**

Read: `paper/nature_article.tex`, `paper/nature_cover_letter.tex`, `paper/nature_article.pdf`.

Adopt the persona of a Nature editor receiving this submission. You have 15 minutes.

Questions:
1. Would you send this to review or desk-reject? Give your decision and reasoning.
2. Is the title "The Limits of Explanation" appropriate for Nature?
3. Is the abstract accessible to a researcher outside all 8 domains?
4. Does the significance justify Nature over a specialist journal (JMLR, PNAS)?
5. Are the 8 domains genuinely different, or are they padding to inflate the claim?
6. Is "one theorem, eight sciences, one resolution" a sufficient hook?
7. Which 3 reviewers would you assign? (Name the specialties.)
8. What is the ONE thing most likely to cause desk-rejection?
9. What is the ONE thing that most strengthens the submission?
10. If you were writing a decision letter, what would the first sentence be?

---

**R24: Devil's Advocate (Maximally Hostile)**

Read: `paper/nature_article.tex` — abstract, introduction, and theorem statement only. Then skim the rest.

Adopt the persona of a reviewer who wants to reject. Find every weakness.

Questions:
1. In ONE sentence: what does this paper prove?
2. What is the most devastating objection?
3. Is this a theorem or a tautology? (The proof is 4 steps — is it obvious?)
4. Is the 4-step proof genuinely non-obvious, or is it just "if things disagree, you can't satisfy everyone"?
5. Are 8 "derived instances" genuinely different from 8 copies of the same trivial observation?
6. Does the resolution add anything beyond "average when uncertain"?
7. Is formal verification impressive or just showing off? Does it add scientific value?
8. The paper claims to connect phenomena across 8 sciences — but is the connection deep or superficial?
9. Why hasn't anyone proved this before? Is the answer (a) nobody thought of it, (b) it's too obvious to publish, or (c) the framework is genuinely novel?
10. Would you cite this paper in your own work? Why or why not?
11. Draft the most hostile referee report you can write (3-5 sentences).

---

**R25: Philosopher of Science**

Read: `paper/nature_article.tex`, monograph §"The theorem" and §"Why This Result Is Non-Trivial".

Questions:
1. Is "explanation" used consistently with its philosophical meaning (scientific explanation, mechanistic explanation, etc.)?
2. Are "faithful," "stable," and "decisive" natural desiderata for explanation, or are they contrived to produce the impossibility?
3. Could a philosopher object that the formalization of "explanation" is too narrow (e.g., it doesn't capture probabilistic explanations, contrastive explanations, interventionist explanations)?
4. Is the comparison to Arrow's impossibility theorem appropriately drawn? Are the structural parallels genuine?
5. Is the claim "impossibility, not tautology" well-defended? What distinguishes this from the observation "if two things disagree, you can't agree with both"?
6. Does the theorem apply to explanation in the ordinary scientific sense, or only to a highly formalized version?
7. Is the "uniquely calibrated" claim (axiom substitution analysis) philosophically significant?
8. Would a philosopher of science find this paper interesting or dismiss it as trivial?

---

## Phase 4: Meta-Review and Synthesis

**Executor:** 1 agent. Reads all 25 reviewer reports.

- [ ] **4.1: Compile all findings**
Create a table: Finding | Reviewer(s) | Severity | Evidence | Suggested Fix

- [ ] **4.2: Deduplicate**
Merge findings that identify the same issue from different angles.

- [ ] **4.3: Classify and prioritize**
Order: all FATAL first, then MAJOR, then MINOR, then STYLISTIC.

- [ ] **4.4: Identify the 5 hardest questions**
These are the questions that a real Nature reviewer would ask and that are hardest to answer. For each, draft a 2-3 sentence response suitable for a revision letter.

- [ ] **4.5: Overall verdict**
Score: Unanimous Accept / Majority Accept / Split / Majority Reject / Unanimous Reject.
For each non-Accept reviewer, state what would change their mind.

- [ ] **4.6: Draft response letter skeleton**
For each FATAL and MAJOR finding, draft a response paragraph:
"We thank the reviewer for this observation. [Action taken]. [Evidence that the fix is correct]."

---

## Phase 5: Fix, Verify, Re-Vet

**Executor:** As needed based on Phase 4 findings.

- [ ] **5.1: Fix all FATAL findings**
Each fix must include: what changed, which files, verification that the fix is correct.

- [ ] **5.2: Fix all MAJOR findings**
Same structure.

- [ ] **5.3: Re-run affected experiments**
If any experiment methodology was changed, re-run and verify results.

- [ ] **5.4: Re-compile Lean**
If any Lean file was changed: `lake build`, re-verify counts.

- [ ] **5.5: Re-compile papers**
```bash
cd paper && latexmk -pdf nature_article.tex && latexmk -pdf universal_impossibility_monograph.tex
```

- [ ] **5.6: Targeted re-review**
For each FATAL fix: have the original reviewer (or equivalent) confirm the fix addresses their concern.

- [ ] **5.7: Final word count and page check**
```bash
texcount paper/nature_article.tex
pdfinfo paper/nature_article.pdf | grep Pages
```
Verify: ≤5000 words, ≤12 pages.

- [ ] **5.8: Commit and push**
```bash
git add -A && git commit -m "fix: comprehensive vet — all FATAL and MAJOR findings addressed"
git push
```

---

## Execution Strategy

| Phase | Parallelism | Agents | Estimated Time |
|-------|------------|--------|---------------|
| 0: Foundation | Sequential | 1 | 10 min |
| 1: Domain review | 8 parallel groups × 2 reviewers | 16 | 15 min |
| 2: Cross-cutting | 6 parallel agents | 6 | 15 min |
| 3: Adversarial | 3 parallel agents | 3 | 10 min |
| 4: Meta-review | 1 agent | 1 | 10 min |
| 5: Fix + verify | As needed | 1-3 | Variable |
| **Total** | | **25 reviewers** | **~60 min + fixes** |

---

## Success Criteria

The vet passes when:
1. **Zero FATAL findings** remain after Phase 5
2. **Zero MAJOR findings** remain after Phase 5
3. **Lean builds clean** (82/377/72/0)
4. **All experiments reproduce** (results match JSONs)
5. **Nature Article compiles** (0 undefined references, ≤5000 words)
6. **The Nature Editor (R23) would send to review** (not desk-reject)
7. **The Devil's Advocate (R24) cannot find a FATAL objection**
8. **Every domain has at least one reviewer who says "correct for my field"**
