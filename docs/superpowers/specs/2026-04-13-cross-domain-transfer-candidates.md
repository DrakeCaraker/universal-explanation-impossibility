# Cross-Domain Transfer, Prediction, and Novel Impossibility Candidates

> **Goal:** Demonstrate that the universal explanation impossibility framework is generative — it produces novel, surprising, testable predictions across domains, not just post-hoc unification.

---

## Candidate List (107 entries)

### Category A: Quantitative Bound Transfers (Character Theory → Domain Predictions)

| # | Transfer | From → To | Prediction | Testable? | Impact | Surprise | Novelty | Feasibility | Score |
|---|----------|-----------|------------|-----------|--------|----------|---------|-------------|-------|
| A1 | **Codon 1/k → MEC edge ambiguity** | Biology → Statistics | Fraction of orientation-ambiguous edges in a random MEC of size k scales as (k-1)/k, matching the S_k character prediction | Yes (simulate random DAGs, compute MECs) | 9 | 9 | 10 | 8 | **36** |
| A2 | **Codon 1/k → parse tree disagreement rate** | Biology → Linguistics | Parser disagreement rate on k-way ambiguous sentences scales as (k-1)/k | Yes (controlled sentences with known k) | 7 | 7 | 8 | 7 | 29 |
| A3 | **Gauge dim(V^G)/dim(V) → ML attribution stability** | Physics → ML | Fraction of stable attribution components = dim(V^G)/dim(V) where G is the Rashomon symmetry of the model class | Yes (compute on synthetic models) | 9 | 8 | 9 | 6 | 32 |
| A4 | **Stat mech 1/Ω → ensemble disagreement** | Stat Mech → ML | Model selection instability among Ω near-optimal models scales as (Ω-1)/Ω | Yes (sweep Rashomon set size) | 7 | 6 | 7 | 8 | 28 |
| A5 | **Crystallography phase count → reconstruction variance** | Crystal → Signal Processing | Phase retrieval variance scales with the number of trivial ambiguities (2n for real signals of length n) | Yes (vary signal length) | 6 | 5 | 6 | 9 | 26 |
| A6 | **Census entropy → ecological inference bound** | CS → Political Science | Irreducible uncertainty in ecological inference (King's problem) bounded by Dirichlet Rashomon set diameter | Yes (voting data) | 8 | 8 | 9 | 7 | **32** |
| A7 | **Codon 1/k → protein designability** | Biology → Protein Engineering | Number of sequences folding to the same structure (designability) predicts per-position sequence entropy via 1/k law | Yes (PDB data) | 8 | 9 | 10 | 5 | **32** |
| A8 | **Gauge invariant count → identifiable causal effects** | Physics → Causal Inference | Number of identifiable causal effects in a DAG = dim(V^G) of the MEC permutation action | Yes (enumeration on benchmark DAGs) | 8 | 8 | 9 | 7 | **32** |
| A9 | **S_k character → color metamerism dimension** | Biology → Color Science | Dimensionality of color space = dim(V^G) where G is the metamerism group on spectral space. Predicts: animals with n cone types have n-dimensional color space | Yes (comparative zoology data) | 9 | 10 | 10 | 4 | **33** |
| A10 | **Boltzmann entropy → Rashomon entropy in ML** | Stat Mech → ML | The log of the number of near-optimal models (Rashomon set size) equals the "Boltzmann entropy" of the model space, with temperature = tolerance | Yes (count near-optimal models) | 7 | 7 | 7 | 8 | 29 |
| A11 | **Null space dim → attention head redundancy** | Math → ML | Number of redundant attention heads in a transformer = null space dimension of the attention-to-output map | Yes (prune heads, measure output change) | 8 | 8 | 9 | 6 | 31 |
| A12 | **Phase 1/k → quantum tomography bound** | Crystal → Quantum | Information recoverable from k measurements of an n-qubit state bounded by character theory of the measurement symmetry group | Yes (quantum simulation) | 8 | 8 | 9 | 5 | 30 |
| A13 | **Molien series → invariant polynomial count** | Math → All domains | For each domain, the Molien series predicts the number of independent G-invariant explanation functions at each polynomial degree | Yes (compute for concrete groups) | 7 | 7 | 8 | 7 | 29 |
| A14 | **Burnside count → number of distinct explanations** | Math → All domains | Burnside's lemma gives |orbits| = (1/|G|) Σ |Fix(g)|. Predicts the number of genuinely distinct explanations (orbits) in each domain | Yes (compute for finite groups) | 6 | 6 | 7 | 9 | 28 |
| A15 | **Reynolds operator norm → faithfulness bound** | Math → All domains | The norm of the Reynolds operator (orbit average projection) gives an upper bound on faithfulness loss. For S_k on R^k: faithfulness loss ≤ (k-1)/k | Yes (compute) | 7 | 6 | 8 | 9 | 30 |

### Category B: Resolution Algorithm Transfers

| # | Transfer | From → To | Prediction | Impact | Surprise | Novelty | Feasibility | Score |
|---|----------|-----------|------------|--------|----------|---------|-------------|-------|
| B1 | **Wilson loops → attribution invariants** | Physics → ML | Define "attribution Wilson loops": functions of SHAP values that are constant across the Rashomon set. These are more efficient than full ensemble averaging (DASH) because they extract invariants directly rather than averaging. | 10 | 9 | 10 | 6 | **35** |
| B2 | **Patterson maps → causal autocorrelation** | Crystal → Statistics | Define "causal Patterson maps": the autocorrelation of DAG edge weights across the MEC, which is invariant under edge reversal. This gives a phase-invariant representation of causal structure. | 8 | 9 | 10 | 5 | 32 |
| B3 | **Pseudoinverse → optimal ensemble weighting** | Math → ML | The minimum-norm solution (pseudoinverse) for underdetermined systems suggests: among all model ensembles that produce the same prediction, the one with minimum total complexity (L2 norm of weights) is Pareto-optimal. | 7 | 6 | 7 | 8 | 28 |
| B4 | **Packed parse forests → model Rashomon forests** | Linguistics → ML | Represent the entire Rashomon set as a "packed model forest" — a compact data structure encoding all near-optimal models simultaneously, analogous to packed parse forests. | 8 | 8 | 9 | 5 | 30 |
| B5 | **Microcanonical ensemble → uniform Rashomon sampling** | Stat Mech → ML | Sample models uniformly from the Rashomon set (microcanonical ensemble), weighted only by whether they're within ε of optimal. Use this for attribution stability. | 7 | 6 | 6 | 8 | 27 |
| B6 | **Complement views → minimal sufficient explanations** | CS → ML | The complement view theory (Bancilhon) suggests: for each explanation query, identify the minimal set of additional information needed to make the answer unique. This "complement explanation" is the smallest supplementary data that resolves the ambiguity. | 7 | 7 | 8 | 6 | 28 |
| B7 | **Gauge fixing → model selection fixing** | Physics → ML | Gauge fixing selects one representative per orbit. Analogue: select one representative model per equivalence class in the Rashomon set, using a "model gauge" condition (e.g., minimum description length). The framework predicts Gribov-like ambiguities: multiple valid gauge-fixed models. | 9 | 9 | 10 | 5 | **33** |
| B8 | **Codon usage tables → attribution usage tables** | Biology → ML | Report the frequency distribution over Rashomon-set attributions for each feature, rather than a single SHAP value. Direct analogue of codon usage tables. | 6 | 5 | 6 | 9 | 26 |
| B9 | **CPDAG → partial attribution ordering** | Statistics → ML | Report only the attribution ORDERINGS that are stable across the Rashomon set (analogous to directed edges in a CPDAG). Features whose relative importance is ambiguous are left unordered. | 8 | 7 | 8 | 7 | 30 |
| B10 | **Boltzmann weighting → Rashomon-weighted explanations** | Stat Mech → ML | Weight models in the Rashomon set by exp(-loss/T) where T is a temperature parameter. At T→0, the best model dominates (decisive but unstable). At T→∞, all models contribute equally (stable but not decisive). The framework predicts an optimal T that maximizes a tradeoff. | 8 | 7 | 8 | 7 | 30 |
| B11 | **Phase retrieval algorithms → causal structure recovery** | Crystal → Statistics | Gerchberg-Saxton iterates between Fourier and real-space constraints. Analogue for causal discovery: iterate between CI constraints and acyclicity constraints to recover the "causal phase" (edge orientations). | 7 | 8 | 9 | 4 | 28 |
| B12 | **Haar integral → continuous Rashomon averaging** | Math → ML | For continuous model spaces (neural networks), the orbit average requires a Haar integral over the Rashomon set. Use the framework to derive the correct measure for averaging. | 7 | 6 | 7 | 5 | 25 |

### Category C: Novel Impossibility Predictions (New Domains)

| # | Domain | Rashomon Property | Impossibility Prediction | Impact | Surprise | Novelty | Feasibility | Score |
|---|--------|-------------------|-------------------------|--------|----------|---------|-------------|-------|
| C1 | **Quantum state tomography** | Multiple density matrices → same measurement statistics (incomplete measurements) | No tomographic method can be faithful (match true state), stable (same output for equivalent states), and decisive (determine all properties) | 9 | 8 | 9 | 6 | **32** |
| C2 | **Color metamerism** | Different spectral distributions → same color percept | No color descriptor can be faithful to the spectrum, stable across metamers, and decisive about spectral details | 8 | 9 | 10 | 5 | **32** |
| C3 | **Protein folding degeneracy** | Multiple AA sequences → same 3D structure | No sequence-to-structure map can be faithful, stable across designable sequences, and decisive about sequence identity | 9 | 8 | 9 | 5 | 31 |
| C4 | **Molecular chirality** | Enantiomers have identical scalar properties (melting point, solubility) but different optical rotation | No characterization from scalar properties alone can distinguish enantiomers stably | 7 | 7 | 8 | 7 | 29 |
| C5 | **Neural decoding** | Multiple neural activity patterns → same behavioral output | No decoder can be faithful to the neural code, stable across equivalent patterns, and decisive about individual neuron contributions | 9 | 8 | 9 | 5 | 31 |
| C6 | **Gravitational lensing** | Multiple source configurations → same lensed image | No reconstruction from a single lensed image can uniquely determine source structure | 7 | 7 | 8 | 5 | 27 |
| C7 | **Polymorphism in materials** | Multiple crystal structures → same bulk properties at certain conditions | No bulk characterization can determine crystal structure when polymorphs coexist | 7 | 7 | 8 | 6 | 28 |
| C8 | **Ecological inference** | Individual behavior → aggregate statistics (King's problem) | No method can infer individual voting patterns from precinct-level results faithfully and stably | 8 | 7 | 8 | 8 | 31 |
| C9 | **Inverse problems in geophysics** | Multiple subsurface structures → same seismic response | No seismic inversion can uniquely determine subsurface geology | 7 | 6 | 7 | 6 | 26 |
| C10 | **Music theory (voice leading)** | Multiple chord voicings → same harmonic function | No analysis can be faithful to the voicing, stable across equivalent voicings, and decisive about all voice-leading details | 6 | 8 | 9 | 5 | 28 |
| C11 | **Network tomography** | Multiple internal link states → same end-to-end measurements | No monitoring system can uniquely determine internal link performance from end-to-end probes | 8 | 6 | 7 | 7 | 28 |
| C12 | **Phylogenetic inference** | Multiple trees → same distance matrix (when taxa are few) | No tree reconstruction from distances alone can be faithful, stable, and decisive when the number of taxa is small | 7 | 7 | 8 | 7 | 29 |
| C13 | **Compiler optimization** | Multiple instruction schedules → same program output | No decompiler can uniquely recover the original source from optimized binary | 6 | 6 | 7 | 8 | 27 |
| C14 | **fMRI inverse problem** | Multiple neural sources → same BOLD signal | No fMRI analysis can uniquely localize neural activity from BOLD responses | 8 | 6 | 7 | 6 | 27 |
| C15 | **Economic identification** | Multiple structural models → same reduced form | No econometric method can identify structural parameters from reduced-form estimates when the model is underidentified | 7 | 5 | 6 | 8 | 26 |
| C16 | **LLM interpretation** | Multiple circuit pathways → same output token | No mechanistic interpretation of an LLM can uniquely attribute output to specific circuits when redundant pathways exist | 9 | 8 | 8 | 6 | 31 |
| C17 | **Radiomics** | Multiple tumor microstructures → same imaging phenotype | No radiomics feature can faithfully represent microstructure while being stable across equivalent tumors | 8 | 7 | 8 | 5 | 28 |
| C18 | **Ancient DNA interpretation** | Multiple population histories → same allele frequencies | No demographic inference from aDNA can uniquely determine migration history | 7 | 7 | 8 | 5 | 27 |
| C19 | **Climate model interpretation** | Multiple parameter sets → same global mean temperature trajectory | No attribution of warming to specific forcings is unique when the climate model is underspecified | 9 | 8 | 8 | 4 | 29 |
| C20 | **Drug mechanism inference** | Multiple MOAs → same phenotypic response | No drug explanation method can uniquely determine mechanism of action from phenotypic screening | 8 | 7 | 8 | 5 | 28 |

### Category D: Structural Predictions (Group Classification → Domain Properties)

| # | Prediction | Domain | Impact | Surprise | Novelty | Feasibility | Score |
|---|-----------|--------|--------|----------|---------|-------------|-------|
| D1 | **Non-abelian groups predict methodological disagreement.** Domains with non-abelian symmetry (S_k, k≥3) should have more competing resolution methods than domains with abelian symmetry (Z_2). | All | 8 | 8 | 9 | 7 | **32** |
| D2 | **Compact groups predict unique resolution; non-compact predict regularization wars.** Mathematics (R^{n-r}) should have more "which regularizer?" debates than physics (Z_2). | All | 7 | 7 | 8 | 7 | 29 |
| D3 | **Group size predicts information loss rate.** Domains with larger symmetry groups lose more information in the resolution. Stat mech (S_Ω, Ω ~ 10^23) loses nearly everything; gauge theory (Z_2^k) loses about half. | All | 7 | 6 | 7 | 9 | 29 |
| D4 | **The representation determines whether averaging or enumeration is adopted.** Vector-valued explanation spaces → averaging (biology, stat mech). Discrete explanation spaces → enumeration (linguistics, statistics). This is testable for new domains. | New domains | 8 | 7 | 8 | 8 | 31 |
| D5 | **Phase transitions in explainability.** For domains with a coupling parameter (like β in gauge theory), there should be a phase transition: below a critical coupling, explanations are maximally unstable; above it, they stabilize. The critical point is determined by the group structure. | Physics → ML, Biology | 9 | 9 | 10 | 4 | **32** |
| D6 | **Gribov copies predict multiple valid gauge-fixings in ML.** For non-abelian model symmetries (e.g., permutation of hidden units in neural networks), gauge-fixing (selecting a canonical representation) should have multiple solutions, analogous to Gribov copies. | Physics → ML | 9 | 9 | 10 | 5 | **33** |
| D7 | **The number of irreducible representations = the number of independent "explanation modes."** Each irreducible representation of G corresponds to an independent mode of variation that the resolution destroys. Counting these gives the number of independent "questions you can't stably answer." | All | 7 | 7 | 8 | 8 | 30 |
| D8 | **Spontaneous symmetry breaking in explanations.** When the Rashomon set has a degenerate ground state, explanations should exhibit spontaneous symmetry breaking: the system "chooses" one explanation from the equivalence class, breaking the symmetry. The pattern of breaking is predicted by the group structure. | Physics → All | 8 | 9 | 10 | 4 | 31 |
| D9 | **Topological invariants of Rashomon sets.** The Rashomon set has a topology (connected components, holes). The topological invariants (Betti numbers) predict qualitatively different resolution behaviors: connected Rashomon sets allow smooth interpolation; disconnected ones don't. | Math → All | 8 | 9 | 10 | 3 | 30 |
| D10 | **Supersymmetry-inspired explanation pairing.** For every "bosonic" (stable) explanation component, there should be a "fermionic" (unstable) partner. The framework predicts paired structure in the decomposition. | Physics → Math | 6 | 9 | 10 | 2 | 27 |

### Category E: Experimental Design Transfers

| # | Transfer | From → To | Impact | Surprise | Novelty | Feasibility | Score |
|---|----------|-----------|--------|----------|---------|-------------|-------|
| E1 | **Dose-response design from biology → ML** | Use degeneracy-level sweeps (vary Rashomon set size) to produce ML dose-response curves analogous to the codon experiment | 7 | 5 | 6 | 9 | 27 |
| E2 | **Negative control from crystallography → causal inference** | Use the positivity constraint (reduces Rashomon set) as inspiration: find interventions that reduce the MEC size and verify agreement improves | 7 | 7 | 8 | 6 | 28 |
| E3 | **Coupling sweep from gauge → ensemble size** | Vary ensemble size K (analogous to coupling β) and measure attribution variance. Predict: variance = sech²(β(K)) where β increases with K | 8 | 7 | 8 | 8 | 31 |
| E4 | **GC-null from biology → baseline instability in ML** | Compute a "null model" for ML attribution instability (random permutation of models in Rashomon set) and report observed-vs-null instability, analogous to the GC-null comparison | 7 | 6 | 7 | 8 | 28 |
| E5 | **Pairwise KL from census → Rashomon diameter in ML** | Compute pairwise KL between explanations from different models in the Rashomon set, as a direct measure of "explanation diameter" | 7 | 5 | 6 | 9 | 27 |

### Category F: Computational Complexity Transfers

| # | Transfer | From → To | Impact | Surprise | Novelty | Feasibility | Score |
|---|----------|-----------|--------|----------|---------|-------------|-------|
| F1 | **NP-hardness of gauge fixing → NP-hardness of model canonicalization** | If selecting a canonical gauge-fixed configuration is NP-hard for certain groups, then selecting a canonical model from the Rashomon set is also NP-hard | 8 | 8 | 9 | 4 | 29 |
| F2 | **#P-hardness of MEC enumeration → #P-hardness of Rashomon enumeration** | If counting the number of DAGs in an MEC is #P-hard (known), then counting models in a Rashomon set should be similarly hard | 7 | 6 | 7 | 6 | 26 |
| F3 | **Polynomial-time invariant extraction → efficient stable explanations** | For groups where invariant extraction is polynomial (abelian groups), stable explanations are efficiently computable. For non-abelian groups, they may not be. | 8 | 7 | 8 | 5 | 28 |
| F4 | **Phase retrieval is NP-hard → some explanation problems are NP-hard** | Phase retrieval from magnitude measurements is known to be NP-hard in general. The framework predicts: recovering a decisive explanation from stable measurements is NP-hard when the symmetry group is sufficiently complex. | 7 | 7 | 8 | 4 | 26 |

### Category G: Novel Mathematical Connections

| # | Connection | Impact | Surprise | Novelty | Feasibility | Score |
|---|-----------|--------|----------|---------|-------------|-------|
| G1 | **Explanation impossibility as a fiber bundle obstruction.** The impossibility is equivalent to the non-existence of a global section of the fiber bundle Θ →^{obs} Y with fiber = equivalence class. The obstruction is measured by a characteristic class. | 9 | 9 | 10 | 3 | 31 |
| G2 | **Cohomological formulation.** The impossibility can be formulated as H¹(G, H) ≠ 0: the first cohomology group of G with coefficients in H is nontrivial, measuring the obstruction to finding a G-equivariant section. | 9 | 9 | 10 | 3 | 31 |
| G3 | **Connection to Noether's theorem.** Noether: symmetry → conservation law. Here: symmetry (Rashomon) → impossibility of decisive explanation. The "conserved quantity" is the G-invariant projection. The analogy: conserved quantities in physics = stably answerable queries in explanation. | 8 | 9 | 10 | 5 | **32** |
| G4 | **K-theory of Rashomon sets.** The K-group of the Rashomon set (as a topological space) classifies the possible resolution strategies up to stable equivalence. Different K-theory classes correspond to qualitatively different resolution behaviors. | 7 | 9 | 10 | 2 | 28 |
| G5 | **Explanation entropy as a topological invariant.** Define "explanation entropy" = log |orbits| / log |Θ|. This is a topological invariant of the observation map obs: Θ → Y. It measures the "fraction of the configuration space that is observationally indistinguishable." | 7 | 7 | 8 | 6 | 28 |
| G6 | **Galois theory of explanations.** The Rashomon symmetry group G is the "Galois group" of the explanation problem, analogous to the Galois group of a polynomial. Just as the Galois group determines which roots are expressible in radicals, G determines which explanation queries are stably answerable. | 9 | 10 | 10 | 3 | **32** |
| G7 | **Explanation impossibility as a no-go theorem in quantum foundations.** The Bell/Kochen-Specker no-go theorems prove that certain classical explanations of quantum mechanics are impossible. The framework may subsume these as instances where the "Rashomon property" is the contextuality of quantum measurements. | 9 | 10 | 10 | 3 | **32** |
| G8 | **Derived categories of explanations.** The chain of resolution strategies (from decisive to fully stable) forms a derived category, with morphisms = information loss maps. The framework predicts: the derived category structure determines which partial resolutions are compatible. | 6 | 8 | 10 | 2 | 26 |
| G9 | **Explanation as a functor.** The assignment "domain ↦ ExplanationSystem" is a functor from the category of scientific domains to the category of explanation systems. The impossibility is a natural transformation obstruction. Cross-domain transfer = functorial image. | 8 | 8 | 10 | 4 | 30 |
| G10 | **Langlands-type duality.** Just as Langlands duality relates representations of a group to representations of its dual, there may be a duality between "explanation spaces" and "observation spaces" that exchanges the roles of faithfulness and decisiveness. | 8 | 10 | 10 | 2 | 30 |

---

## TOP 15 BY COMPOSITE SCORE

| Rank | # | Candidate | Score | Category |
|------|---|-----------|-------|----------|
| 1 | A1 | **Codon 1/k → MEC edge ambiguity** | 36 | Quantitative bound |
| 2 | B1 | **Wilson loops → attribution invariants** | 35 | Algorithm transfer |
| 3 | A9 | **Color metamerism → dimensionality of color space** | 33 | Quantitative bound |
| 4 | B7 | **Gauge fixing → model selection (Gribov copies)** | 33 | Algorithm transfer |
| 5 | D6 | **Gribov copies in neural network symmetry** | 33 | Structural prediction |
| 6 | A3 | **Gauge dim(V^G)/dim(V) → ML attribution stability** | 32 | Quantitative bound |
| 7 | A6 | **Census entropy → ecological inference bound** | 32 | Quantitative bound |
| 8 | A7 | **Codon 1/k → protein designability** | 32 | Quantitative bound |
| 9 | A8 | **Gauge invariant count → identifiable causal effects** | 32 | Quantitative bound |
| 10 | B2 | **Patterson maps → causal autocorrelation** | 32 | Algorithm transfer |
| 11 | C1 | **Quantum state tomography impossibility** | 32 | Novel impossibility |
| 12 | C2 | **Color metamerism impossibility** | 32 | Novel impossibility |
| 13 | D1 | **Non-abelian → methodological disagreement** | 32 | Structural prediction |
| 14 | D5 | **Phase transitions in explainability** | 32 | Structural prediction |
| 15 | G3 | **Connection to Noether's theorem** | 32 | Mathematical |
| 15 | G6 | **Galois theory of explanations** | 32 | Mathematical |
| 15 | G7 | **Bell/KS as explanation impossibility** | 32 | Mathematical |

---

## PHASED IMPLEMENTATION PROGRAM

### Phase 1: Immediate (existing data, 1-2 days each)

**1.1: Codon 1/k → MEC edge ambiguity (A1)**
- Generate random DAGs on 5-15 nodes at various edge densities
- Compute MEC for each, count orientation-ambiguous edges
- Test prediction: fraction ambiguous = (|MEC|-1)/|MEC|
- Compare to character-theoretic 1/k for the permutation group
- If prediction holds → first genuine cross-domain transfer

**1.2: Gauge dim(V^G)/dim(V) → identifiable causal effects (A8)**
- For benchmark causal graphs (Asia, Alarm, Insurance)
- Count identifiable effects (those shared across all MEC members)
- Count total effects
- Test: ratio = dim(V^G)/dim(V) from character theory of MEC group
- Requires only graph enumeration, no experiments

**1.3: Non-abelian → methodological disagreement (D1)**
- Survey literature: count competing methods per domain
- Correlate with group structure (abelian vs non-abelian)
- Biology: CUB, RSCU, ENC, tAI, CAI (many methods — S_k non-abelian ✓)
- Physics: Wilson loops (essentially one method — Z_2 abelian ✓)
- Stat mech: microcanonical average (one method — but S_Ω non-abelian?)
- This needs careful analysis but is purely literature-based

**1.4: Census entropy → ecological inference bound (A6)**
- Apply the census pairwise-KL framework to voting data
- Use actual precinct-level election results as "ground truth"
- Aggregate to county/district level and attempt disaggregation
- Measure KL of disaggregated estimates vs truth
- Test: Rashomon diameter predicts irreducible inference error

### Phase 2: Moderate effort (new computation, 3-7 days each)

**2.1: Wilson loops → attribution invariants (B1)**
- Define "attribution Wilson loops": nonlinear functions of SHAP values that are constant across the Rashomon set
- Implement for GBDT models (using the dash-shap codebase)
- Compare: (a) single-model SHAP, (b) DASH ensemble average, (c) attribution Wilson loops
- Metric: stability (variance across Rashomon set) and efficiency (computation time)
- Prediction: Wilson loops are as stable as DASH but faster (extract invariants rather than averaging)

**2.2: Gauge fixing → model canonicalization + Gribov copies (B7, D6)**
- For a neural network with hidden-unit permutation symmetry
- Implement "gauge fixing": sort hidden units by some criterion (e.g., L2 norm of weights)
- Count Gribov copies: how many distinct gauge-fixed representations exist?
- Test prediction: for networks with k-fold symmetry, expect O(k!) Gribov copies
- Test: Gribov copies increase with network width (more hidden units → more permutation symmetry)

**2.3: Codon 1/k → protein designability (A7)**
- Download PDB/UniProt data on protein families with known designability
- For each structural position, count the number of amino acids observed across homologs
- Compute sequence entropy per position, grouped by "designability" (number of sequences folding to same structure)
- Test: entropy scales as 1/k where k relates to the structural constraints at each position
- This is the protein-level analogue of the codon experiment

**2.4: Phase transitions in explainability (D5)**
- Implement: vary the tolerance ε (defining the Rashomon set) from tight to loose
- At tight ε: few models qualify, explanations are stable (ordered phase)
- At loose ε: many models qualify, explanations are unstable (disordered phase)
- Test: is there a sharp transition? Plot explanation variance vs ε
- Compare to the gauge theory experiment (variance vs β)
- The prediction: the transition sharpness depends on the model class's symmetry group

**2.5: Color metamerism dimensionality (A9)**
- Compute the metamerism group G for human vision (3 cone types)
- The quotient space Spectra / G should be 3-dimensional
- Compute dim(V^G) from character theory
- Test on comparative data: organisms with 2 cones (dichromats) → 2D color space; 4 cones (tetrachromats, e.g., birds) → 4D
- Prediction: dim(color space) = number of cone types = dim(V^G)

### Phase 3: High effort (new theory + data, 2-4 weeks each)

**3.1: Quantum state tomography as explanation impossibility (C1)**
- Formalize: Θ = density matrices, Y = measurement statistics, H = state descriptions
- Identify the symmetry group (unitary freedom in mixed-state decomposition)
- Derive the impossibility as an instance of the meta-theorem
- Prove in Lean 4 (new derived instance)
- Resolution: quantum state estimation via maximum likelihood (the orbit average)
- Would be 9th derived instance

**3.2: Fiber bundle formulation (G1)**
- Formalize: the observation map obs: Θ → Y is a fiber bundle
- The impossibility = non-existence of global section
- The obstruction is a characteristic class (first Chern class for U(1), etc.)
- This gives a topological invariant measuring "severity of impossibility"
- Could unify all instances in a single geometric framework

**3.3: Noether correspondence (G3)**
- Formalize: symmetry (Rashomon) → conservation law (G-invariant)
- The "conserved quantities" are the stably answerable queries
- The "Noether current" is the resolution map
- This would place the impossibility in the same conceptual category as Noether's theorem
- Very high philosophical impact if the analogy is tight

**3.4: Galois theory of explanations (G6)**
- Formalize the Rashomon group as the "Galois group" of the explanation problem
- Just as solvability of the Galois group ↔ solvability by radicals, properties of the Rashomon group ↔ solvability of the explanation problem
- Predict: "unsolvable" explanation problems (non-solvable Rashomon group) should be qualitatively harder than "solvable" ones
- This is highly speculative but would be extraordinary if it works

### Phase 4: Visionary (multi-year research program)

**4.1: Bell/Kochen-Specker as explanation impossibility (G7)**
- Bell's theorem: no local hidden variable theory reproduces quantum correlations
- Reframe: quantum observables = explanation system, hidden variables = configurations, measurement outcomes = observables
- The impossibility: no hidden variable assignment can be faithful (match quantum predictions), stable (same assignment for compatible measurements), and decisive (assign definite values)
- If this works, the explanation impossibility subsumes Bell's theorem
- This would be a major result in foundations of physics

**4.2: Langlands-type duality for explanations (G10)**
- Seek a duality between "observation space" and "explanation space" that exchanges faithfulness and decisiveness
- In Langlands, the dual group controls the representation theory of the original
- For explanations: the "dual explanation system" would have observations and explanations swapped
- Purely speculative but would create a deep mathematical research program

**4.3: K-theory classification of resolutions (G4)**
- Classify all possible resolution strategies for a given symmetry group using K-theory
- Different K-theory classes = qualitatively different resolution behaviors
- Predict: the K-group of the Rashomon set determines which resolutions are topologically possible

---

## RECOMMENDED EXECUTION ORDER

Start with the candidates that have:
1. Highest composite score
2. Existing data (no new experiments)
3. Clear pass/fail criteria
4. Maximum surprise if confirmed

**Immediate priorities (this week):**
1. **A1: Codon 1/k → MEC edge ambiguity** — Score 36, testable now
2. **A8: Gauge invariant count → identifiable causal effects** — Score 32, testable now
3. **D1: Non-abelian → methodological disagreement** — Score 32, literature survey

**If ANY of these confirms the prediction, the paper transforms from taxonomy to generative theory.**
