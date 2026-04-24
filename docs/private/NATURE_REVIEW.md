# Nature Peer Review Simulation — Unanimous Accept Standard

**Paper**: "The Limits of Explanation" (62pp monograph + 4pp Nature BC)
**Standard**: Every reviewer must find no FATAL issues. The paper
must survive a molecular biologist, a physicist, a linguist, a
crystallographer, a mathematician, a computer scientist, a
statistician, an ML theorist, a philosopher, a Nature editor,
a Lean expert, and an experimental statistician — all hostile,
all seeing this for the first time.

**Model**: ALL Opus (requires domain judgment).

---

## Panel A: Domain Experts (8 reviewers, one per derived instance)

Each reads ONLY their domain's Lean file, paper section, and
experiment results. Each asks: "Is this correct for MY field?"

### R1: Molecular Biologist
Read: GeneticCode.lean, instance_genetic_code.tex,
results_codon_entropy.json

- Is codon degeneracy correctly stated?
- Is the dose-response prediction meaningful biologically?
- Is "30 species with synthetic GC content" adequate, or
  should this use REAL genomic data?
- Would a Nature biology reviewer accept synthetic data?
- Is the codon optimization / mRNA vaccine connection accurate?
- Are the citations appropriate (Crick 1966)?

### R2: Physicist (Gauge Theory)
Read: GaugeTheory.lean, instance_gauge_theory.tex,
results_gauge_lattice.json

- Is the ℤ₂ gauge theory correctly formalized?
- Is holonomy correctly defined (XOR around triangle)?
- Does gauge_preserves_holonomy hold generally (not just for witnesses)?
- Is the lattice experiment standard methodology?
- Is the connection to continuous gauge theory (E&M, Yang-Mills) overstated?
- Would a Nature physics reviewer find the 2×2 triangle toy convincing?

### R3: Physicist (Statistical Mechanics)
Read: StatisticalMechanics.lean, instance_stat_mech.tex,
results_stat_mech_entropy.json

- Is the microstate/macrostate framing correct?
- Is "Rashomon entropy = Boltzmann entropy" accurately stated?
- Is the 2-coin model a legitimate "statistical mechanics" example?
- Is the microcanonical ensemble correctly identified as the resolution?
- Would a stat mech reviewer object to the canonical ensemble discussion?

### R4: Linguist
Read: SyntacticAmbiguity.lean, instance_syntax.tex,
results_parser_disagreement.json

- Is "V NP PP" ambiguity correctly stated?
- Is the left-attach vs right-attach distinction accurate?
- Are the 50 ambiguous sentences genuinely structurally ambiguous?
- Is UAS the right metric for inter-parser agreement?
- Is p=8.9e-4 convincing with 4 parsers and 50+50 sentences?
- Is the "packed parse forest" resolution correctly described?
- Would a computational linguist find this methodology standard?

### R5: Crystallographer
Read: PhaseProblem.lean, instance_phase_problem.tex,
results_phase_retrieval.json

- Is the phase problem correctly described?
- Is the energy-based formalization (sum of squares) a valid
  simplification of the Fourier magnitude problem?
- Is Gerchberg-Saxton correctly implemented?
- Is the positivity constraint the right control?
- Is the Hauptman & Karle (1985) citation appropriate?
- Would a structural biologist find this convincing?

### R6: Mathematician
Read: LinearSystem.lean, instance_linear_system.tex,
results_linear_solver.json

- Is Ax=b with rank(A) < dim(x) correctly formalized?
- Are the 4 solvers standard numerical methods?
- Is the pseudoinverse correctly identified as the G-invariant resolution?
- Is the negative slope (RMSD decreasing with d) correctly explained?
- Is p=6.3e-9 from the Mann-Whitney test meaningful here?

### R7: Computer Scientist (Databases)
Read: ViewUpdate.lean, instance_view_update.tex,
results_census_disagg.json

- Is the view update problem correctly formalized?
- Is Bancilhon & Spyratos (1981) correctly cited?
- Is the synthetic census disaggregation appropriate?
- Should this use REAL census data instead?
- Is KL-divergence the right metric?

### R8: Statistician (Causal Inference)
Read: MarkovEquivalence.lean, instance_causal.tex,
results_causal_discovery_exp.json

- Is chain/fork Markov equivalence correctly stated?
- Is the ciFromDAG simplification (no colliders) a problem?
- Are PC and GES standard algorithms?
- Is the multi-seed comparison (10 seeds per condition) adequate?
- Would Pearl or Chickering agree with this formalization?

---

## Panel B: Cross-Cutting Reviewers (4 reviewers)

### R9: Lean 4 / Formal Verification Expert
Read: ALL 82 .lean files (headers + key theorems).
Run: lake build. Run: #print axioms for all 8 derived instances.

- Do ALL derived instances have zero behavioral axioms?
- Do the Lean definitions match the paper definitions?
- Does the appendix code listing match the actual files?
- Are there any sorry (even in comments counted by grep)?
- Is autoImplicit false everywhere?
- Are the 72 axioms correctly classified?

### R10: Experimental Statistician
Read: ALL 15 experiment results JSONs (8 cross-domain + 7 ML).
Read: experiment_utils.py.

- Are ALL bootstrap CIs computed correctly?
- Are ALL p-values from appropriate tests?
- Is there multiple comparison correction across 15 experiments?
- For EACH experiment: is the sample size adequate?
- For EACH negative control: is it genuine (nonzero comparisons,
  not vacuous)?
- Could ANY positive result be explained without the
  Rashomon property?
- Is the synthetic biology data adequate, or does it need
  REAL genomic data?

### R11: Nature Editor Simulation
Read: ONLY the Nature Brief Communication (4pp).

- Would you send this to review or desk-reject?
- Is the title appropriate for Nature?
- Is the abstract accessible to a general audience?
- Does the significance statement justify Nature?
- Are the 8 domains genuinely different or padding?
- Is "one theorem, eight sciences" a sufficient hook?
- Which 3 reviewers would you assign?
- What is the ONE thing that would make you desk-reject?

### R12: Devil's Advocate (Maximally Hostile)
Read: ONLY the abstract of the monograph.

- In ONE sentence: what does this paper prove?
- What is the most devastating objection?
- Is this a theorem or a tautology?
- Is 62 pages justified for a 4-line proof?
- Would you cite this paper? Why or why not?
- Are 8 "derived instances" genuinely different from
  8 copies of the same trivial observation?

---

## Execution

Launch 4 parallel groups of 3 reviewers each:
- Group 1: R1 (biologist) + R2 (gauge physicist) + R3 (stat mech)
- Group 2: R4 (linguist) + R5 (crystallographer) + R6 (mathematician)
- Group 3: R7 (CS) + R8 (statistician) + R9 (Lean expert)
- Group 4: R10 (exp. statistician) + R11 (Nature editor) + R12 (devil)

Each reviewer reports: FATAL / MAJOR / MINOR / STYLISTIC findings.

After all 12 return: META-REVIEW with:
1. All findings classified and deduplicated
2. The 3 hardest questions with best responses
3. Overall: Unanimous Accept / Split / Reject
4. What would change Reject to Accept?
