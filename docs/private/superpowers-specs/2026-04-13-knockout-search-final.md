# Final Knockout Search — 200 Candidates

> **Criterion:** The result must be (1) NOT derivable from the target domain's existing frameworks, (2) quantitatively testable, (3) surprising to domain experts, and (4) confirmed by data. Lessons from prior failures: avoid circular computation, avoid predictions that reduce to "conservation predicts function," avoid analogies that don't produce specific numbers.

---

## What the framework UNIQUELY provides (that no single domain has)

1. The (d-1)/d formula from character theory of S_k — predicts diversity scaling from symmetry alone
2. The biconditional: Rashomon ↔ impossibility — predicts EXACTLY where stability fails
3. The resolution Pareto-optimality — predicts that orbit averaging is uniquely optimal
4. The group classification — predicts resolution FORM from symmetry type
5. Cross-domain bridges — knowledge from domain A translated to domain B

## Candidate Generation Strategy

For each candidate, specify:
- The SPECIFIC quantitative prediction
- WHY it's not derivable from within the target domain
- HOW to test it with available data/tools
- What would FALSIFY it

---

## 200 Candidates (organized by target domain)

### ML / Deep Learning (1-40)

1. The number of functionally distinct attention heads in a transformer = dim(V^G) where G = head permutation subgroup. Prediction: most heads are redundant (in the non-trivial representation). Test: prune heads, measure performance, compare to dim(V^G) prediction.

2. Dropout rate p that maximizes validation accuracy satisfies p = 1 - dim(V^G)/dim(V) for the dropout mask group. Test: sweep dropout rates, compare optimal p to character-theory prediction.

3. The rank of the feature importance vector (number of features with nonzero SHAP) is bounded by dim(V^G). Test: compute SHAP for many Rashomon models, count features that are consistently ranked.

4. Weight decay optimal λ = 1/|G| where G is the model's Rashomon symmetry group. Test: sweep λ, compare optimal to 1/|Rashomon set|.

5. The loss landscape has exactly |G| equivalent minima (one per orbit) near the global minimum. Test: train from many random seeds, cluster the solutions, count clusters.

6. The effective dimensionality of the Rashomon set (measured by PCA on the parameter vectors of near-optimal models) = dim(V) - dim(V^G). Test: train many near-optimal models, PCA their parameter vectors, count significant components.

7. Batch size that maximizes training efficiency satisfies B* ∝ |G|. Larger Rashomon sets need larger batches to average over the symmetry. Test: sweep batch size at different Rashomon sizes.

8. Learning rate warmup duration should be proportional to log|G|. Test: sweep warmup, compare optimal to Rashomon-predicted value.

9. The number of epochs to convergence scales as log|G| (more equivalent minima → more time to find one). Test: measure convergence time vs Rashomon set size.

10. Model ensembles show diminishing returns at K = |G| members. Beyond |G|, additional models are in the same orbit and add no new information. Test: stability vs K, check for a knee at K ≈ |G|.

11. The gap between train and test SHAP values (attribution generalization gap) scales as √(S_Rashomon/n). Test: measure train vs test SHAP correlation at varying sample sizes and Rashomon sizes.

12. Data augmentation effectiveness is predicted by |G_aug ∩ G_Rashomon| — augmentations that align with Rashomon symmetries help most. Test: compare augmentation strategies by their overlap with the model's symmetry.

13. The fraction of adversarial examples in a ball of radius ε around a clean input scales as 1 - 1/|G| for models with Rashomon set G. Test: compute adversarial attack success rate vs Rashomon set size.

14. Early stopping optimal epoch = argmin_t S_Rashomon(t). The Rashomon set first expands (overfitting) then contracts (regularization). Stop when it's smallest. Test: track Rashomon set size during training, compare optimal stop point.

15. The number of distinct feature interactions (non-zero SHAP interactions) ≤ C(dim(V^G), 2). Only features in the G-invariant subspace have stable interactions. Test: compute SHAP interactions, count significant ones, compare to bound.

16. Fine-tuning a pretrained model: the number of layers that need updating = dim(V) - dim(V^G_target) where G_target is the target task's symmetry. Test: measure per-layer importance for fine-tuning, compare to character-theory prediction.

17. The information bottleneck in deep learning: the mutual information between hidden representation and input decreases as 1/|G_layer| per layer, where G_layer is the per-layer symmetry group. Test: compute MI profile, compare to prediction.

18. Gradient noise during SGD training has variance proportional to S_Rashomon. Test: measure gradient noise magnitude vs Rashomon set size during training.

19. The number of "dead" ReLU neurons in a trained network ≈ n × (1 - 1/|G|) where G is the Rashomon symmetry. Test: count dead neurons, compare to prediction.

20. Knowledge distillation compression ratio = dim(V^G)/dim(V). The teacher's G-invariant content is all the student needs. Test: measure minimal student size that matches teacher performance.

21. The success rate of membership inference attacks ∝ 1/|G|. Larger Rashomon sets make individual training points less identifiable. Test: run MIA attacks at different Rashomon sizes.

22. Differential privacy budget ε_required ∝ log|G|. Larger Rashomon sets provide more natural privacy. Test: measure privacy leakage vs Rashomon size.

23. The Rashomon set shrinks as O(1/n) with sample size n (central limit theorem for models). The framework predicts the constant: shrinkage rate = |G|/n. Test: measure |Rashomon set| vs n.

24. Curriculum learning: the optimal training order presents examples that MAXIMALLY REDUCE |G| at each step. Test: compare curriculum strategies by Rashomon reduction rate.

25. The prediction interval (uncertainty) of a Bayesian neural network = the "Rashomon diameter" along the prediction direction. Test: compare Bayesian posterior width to Rashomon set spread.

26. Neural scaling laws: the exponent α in loss ∝ N^{-α} = 1/log|G(N)| where G(N) is the Rashomon group at model size N. Test: fit α at various N, compare to Rashomon group size.

27. The rank of the Hessian at a minimum = dim(V^G). Flat directions are the non-invariant modes. Test: compute Hessian eigenvalues, count positive ones, compare to dim(V^G).

28. Sparse autoencoder features = irreducible representations of the residual stream's symmetry group. Test: count SAE features, compare to number of irreps.

29. The optimal number of SAE features = Σ (dim of irrep)² by Peter-Weyl. Test: sweep SAE dictionary size, find optimal, compare to prediction.

30. Model merging (weight averaging) works iff the models are in the same G-orbit. Test: predict which model pairs successfully merge based on Rashomon overlap.

31. LoRA rank for effective fine-tuning = dim(V) - dim(V^G_pretrain). Test: sweep LoRA rank, compare optimal to prediction.

32. The number of modes in a mixture-of-experts = |orbits| of the Rashomon set. Each expert captures one orbit. Test: compare MoE expert count to Rashomon orbit count.

33. Chain-of-thought reasoning: the number of steps needed = the "diameter" of the Rashomon set (longest path between equivalent solutions). Test: correlate CoT length with problem Rashomon diameter.

34. In-context learning sample efficiency: k-shot learning succeeds when k ≥ log|G_task| where G_task is the task's Rashomon group. Test: measure k-shot accuracy vs task ambiguity.

35. The temperature T* that maximizes useful diversity in LLM sampling = 1/β_c from the Rashomon phase transition. Test: sweep temperature, compare optimal to predicted.

36. Retrieval-augmented generation: the number of retrieved documents needed = log|G_query| where G_query is the Rashomon set of valid answers. Test: vary k in top-k retrieval, find sufficient k per query type.

37. Reward model disagreement in RLHF = the Rashomon diameter of the reward model class. Test: train multiple reward models, measure disagreement, compare to Rashomon prediction.

38. The "alignment tax" (performance loss from alignment) = (dim(V^G_aligned) - dim(V^G_unaligned)) / dim(V). Test: measure alignment performance drop, compare to character-theory prediction.

39. Calibration error of a model ensemble = sech²(K/K_c) where K = ensemble size and K_c = |G|. Test: measure calibration error vs K.

40. The optimal checkpoint for model selection during training = the point where S_Rashomon is minimized (the training process crosses the "phase transition"). Test: track Rashomon size during training, compare to optimal checkpoint.

### Biology / Medicine (41-80)

41. The mutation rate at synonymous vs non-synonymous sites should differ by exactly the attenuation factor α. Specifically: synonymous mutation rate / non-synonymous rate = 1/α. Test: compute from molecular clock data.

42. Codon usage bias (CUB) across organisms scales with effective population size Ne as CUB = 1 - α(Ne) where α(Ne) = (Ne-1)/Ne from the framework. Test: correlate CUB with Ne across species.

43. The number of alternative splicing isoforms per gene follows the 1/k law. Genes with more exons (more ways to splice) should have higher isoform diversity scaling as (k-1)/k. Test: correlate isoform count with exon count in Ensembl data.

44. Antibiotic resistance evolution rate ∝ log(number of equivalent resistance mechanisms). More mechanisms → faster evolution. Test: compare resistance emergence rates across drug classes with known mechanism counts.

45. Codon optimization for mRNA vaccines: the optimal codon at each position should be predictable from the 1/k law. The most "stable" codon (least sensitive to mutations) is the one closest to the orbit average. Test: compare vaccine codon choices to framework predictions.

46. The error rate of protein structure prediction (RMSD to true structure) ∝ log(number of conformational states). More conformational flexibility → harder prediction. Test: correlate AlphaFold pLDDT with conformational ensemble size from MD simulations.

47. Drug combination synergy is predicted by mechanism overlap: drugs targeting the same pathway but through different Rashomon-equivalent mechanisms should be antagonistic (each covers the same Rashomon orbit). Test: correlate synergy scores with mechanism overlap.

48. The Shannon diversity of the gut microbiome at functionally redundant positions (multiple species performing the same metabolic function) should follow the 1/k law. Test: compute per-function species diversity vs functional redundancy.

49. Cancer driver gene identification: the framework predicts that genes with larger "mutation equivalence classes" (more mutations producing the same phenotypic effect) are harder to identify as drivers. Test: correlate driver identification difficulty with mutation equivalence class size.

50. The number of pharmacological targets for a disease = dim(V^G) where G is the symmetry group of drug-target equivalence. Test: compare to known target counts.

51. Vaccination strategy: the framework predicts that vaccines targeting conserved epitopes (low d, small Rashomon set) are more durable than those targeting variable epitopes (high d, large Rashomon set). Test: correlate vaccine durability with epitope conservation.

52. Gene expression noise ∝ (d-1)/d where d = number of regulatory inputs. More regulatory redundancy → more noise (more ways to achieve the same expression level). Test: correlate expression noise with regulatory complexity in yeast data.

53. Horizontal gene transfer rate between species ∝ the Rashomon entropy of the shared metabolic environment. Species in more functionally redundant environments should transfer genes more. Test: correlate HGT rates with environmental functional redundancy.

54. The number of independent clinical symptoms of a disease = dim(V^G) of the pathophysiology symmetry group. Diseases with more equivalent pathways produce fewer distinguishable symptoms. Test: correlate symptom count with pathway redundancy.

55. Phylogenetic tree uncertainty (bootstrap support) should follow the 1/k law where k = number of equally parsimonious trees. Test: correlate bootstrap values with parsimony tree count across clades.

56. The neutral mutation rate at a position should equal the (d-1)/d ceiling minus the observed diversity. The gap = the rate removed by selection. Test: compare to dN/dS estimates.

57. Organ transplant rejection risk ∝ the Rashomon overlap of HLA types between donor and recipient. More shared near-equivalent HLA types → more immune confusion → higher rejection. Test: correlate rejection rates with HLA Rashomon overlap.

58. The diversity of T-cell receptor (TCR) sequences targeting the same antigen follows the 1/k law where k = structural degeneracy of the epitope-TCR interface. Test: compute TCR diversity per epitope, correlate with structural degeneracy.

59. Disease genetic heterogeneity (number of genes causing the same disease) creates a diagnostic Rashomon property. The framework predicts: diagnostic accuracy is bounded by 1/k where k = number of equivalent genetic causes. Test: correlate diagnostic accuracy with genetic heterogeneity.

60. The effective population size at which drift overwhelms selection for synonymous codons = the critical coupling β_c from the framework's phase transition. Test: compare to known Ne thresholds for codon bias.

61. Microbiome stability (resilience to perturbation) ∝ 1/S_Rashomon where S_Rashomon = functional redundancy entropy. More redundancy → less stable community composition (more equivalent configurations). Test: correlate stability with functional redundancy.

62. Cancer clonal heterogeneity follows the 1/k law where k = number of equivalent mutational paths to the same phenotype. Test: correlate clonal diversity with mutational pathway degeneracy.

63. The number of off-target effects of a CRISPR guide RNA ∝ the genomic Rashomon entropy (number of similar loci). Test: correlate off-target rates with sequence similarity counts.

64. The minimum number of biomarkers needed to distinguish disease subtypes = dim(V^G) of the subtype symmetry group. Test: compare to feature selection results on disease genomic data.

65. Epistasis (gene-gene interaction) strength ∝ the Rashomon overlap between gene functions. Genes with overlapping functions show stronger epistasis. Test: correlate epistasis measures with functional overlap.

66. The response heterogeneity in clinical trials ∝ the Rashomon entropy of the treatment mechanism. Treatments with more equivalent mechanisms produce more variable patient responses. Test: correlate response variance with mechanism redundancy.

67. Protein evolution rate is predicted by the attenuation factor α with r = 0.62 (already measured). The framework predicts a SECOND relationship: proteins with more S_k subgroups (mixed degeneracy) should evolve at intermediate rates. Test: partition proteins by degeneracy distribution, check if mixed-degeneracy proteins have intermediate α.

68. The optimal number of drug candidates in a clinical pipeline ∝ log|G_mechanism| where G_mechanism is the Rashomon set of equivalent mechanisms. Test: correlate pipeline size with mechanism degeneracy.

69. Single-cell RNA-seq clustering instability follows the 1/k law where k = number of equivalent cell state representations. Test: measure clustering stability across random seeds, correlate with representation degeneracy.

70. Metagenomic functional profiling accuracy is bounded by the Rashomon entropy of the gene-function mapping. Multiple genes → same function → ambiguous profiling. Test: correlate accuracy with functional redundancy.

71-80. [RESERVE: domain-specific applications to ecology, epidemiology, agriculture, neuroscience — each applying the same pattern: identify Rashomon property, predict instability/diversity from 1/k or S_Rashomon, test on available data]

### Physics / Engineering (81-110)

81. The number of equivalent circuit implementations (for a given Boolean function) follows the 1/k law. Circuit entropy ∝ (k-1)/k. Test: enumerate equivalent circuits for small functions, measure entropy.

82. Quantum computing error correction threshold = β_c from the framework's phase transition, where β = code distance. Test: compare known thresholds to framework prediction.

83. The number of degenerate ground states in a quantum spin system predicts the instability of ground-state property measurements. Test: correlate measurement variance with degeneracy.

84. Signal processing: the uncertainty in time-frequency analysis follows the 1/k law where k = number of equivalent time-frequency representations (Gabor atoms). Test: compare to known Gabor analysis uncertainty.

85. Communication channel capacity with ambiguous decoding (multiple valid decodings per codeword) follows C = C_0 × (1/k) where k = decoding ambiguity. Test: compute on toy channels.

86. Radar target identification: targets with more equivalent scattering profiles (Rashomon of radar cross-section) are harder to classify. Classification accuracy ∝ 1/k. Test: simulate with known target models.

87. The precision of analog-to-digital conversion is bounded by the Rashomon entropy of the quantization levels. Test: theoretical analysis.

88. Control theory: the number of equivalent controllers for a given plant follows the 1/k law. Controller design instability ∝ (k-1)/k. Test: enumerate equivalent controllers for standard benchmark plants.

89. Image compression: the SSIM (structural similarity) between compression levels follows the 1/k law where k = number of equivalent compressed representations. Test: measure SSIM vs compression level.

90. Power grid stability: the number of equivalent load distributions ∝ the Rashomon entropy. Grid instability ∝ S_Rashomon. Test: simulate on IEEE test networks.

91-110. [RESERVE: applications to materials science, fluid dynamics, optics, acoustics, thermodynamics — each with the same structure]

### Social Science / Economics (111-140)

111. The replication crisis: studies with more "researcher degrees of freedom" (larger Rashomon set of valid analyses) should replicate less often. Replication probability ∝ exp(-S_Rashomon). Test: compute S_Rashomon for studies in Many Labs, correlate with replication success.

112. Survey response instability ∝ 1/k where k = number of equivalent valid interpretations of each question. Ambiguous questions produce more variable responses. Test: correlate response variance with question ambiguity rating.

113. Election forecast uncertainty ∝ the Rashomon entropy of the polling model class. More equivalent models → less stable forecasts. Test: correlate forecast spread with model multiplicity.

114. The number of distinct economic models producing the same macroeconomic prediction = |G|. Policy instability ∝ (|G|-1)/|G|. Test: enumerate equivalent DSGE models, measure policy prediction variance.

115. Legal precedent ambiguity: cases with more valid legal interpretations (larger Rashomon set) should produce more split decisions. Split rate ∝ (k-1)/k. Test: correlate split decisions with interpretation count.

116. Academic peer review: inter-reviewer agreement should follow 1/k where k = number of "valid" evaluation frameworks. Fields with more valid evaluation criteria should have lower agreement. Test: correlate inter-reviewer κ with evaluation framework count across disciplines.

117. The stability of social science measurement scales ∝ 1/S_Rashomon of the construct being measured. More ambiguous constructs produce less reliable scales. Test: correlate scale reliability (Cronbach's α) with construct ambiguity.

118. The optimal size of a deliberative body (committee, jury) = log|G_decision| where G is the symmetry group of equivalent valid decisions. Test: theoretical prediction vs known optimal committee sizes.

119. The reproducibility of qualitative research findings ∝ 1/S_Rashomon where S_Rashomon = analyst degrees of freedom. Test: correlate reproducibility rates with the many-analysts-many-results literature.

120. Market volatility ∝ S_Rashomon of the market model class. Periods with more equivalent valid models produce more volatile prices. Test: compute model multiplicity from options data, correlate with realized volatility.

121-140. [RESERVE: applications to political science, sociology, education, psychology, criminology]

### Mathematics / Computer Science (141-170)

141. The number of valid proofs of a theorem ∝ exp(S_Rashomon) where S_Rashomon = the symmetry entropy of the proof space. Theorems with more equivalent proofs are "more provable." Test: count distinct proofs for standard theorems, correlate with proof-space symmetry.

142. SAT solving difficulty: instances with more equivalent satisfying assignments (larger Rashomon set) should be EASIER to solve (more paths to a solution). Test: correlate SAT runtime with solution count.

143. Database query optimization: the number of equivalent query plans ∝ the Rashomon entropy. Optimizer instability ∝ (k-1)/k. Test: measure plan stability across optimization runs.

144. Version control merge conflicts ∝ the Rashomon entropy of the code change space. More equivalent valid implementations → more merge conflicts. Test: correlate conflict rates with implementation degeneracy.

145. The optimal hash function collision resistance = log|G_hash| where G is the symmetry group of the hash function's equivalence classes. Test: theoretical analysis.

146. Sorting algorithm stability: the number of equivalent valid orderings (for equal elements) follows the 1/k law. Sorting instability ∝ (k-1)/k where k = number of equivalent orderings. Test: measure sort stability vs k.

147. Graph isomorphism testing difficulty: graphs with more automorphisms (larger symmetry group) should be HARDER to test. Test: correlate GI runtime with automorphism count.

148. The minimum description length of a dataset = log|G| + dim(V^G) × precision. The Rashomon entropy contributes a constant overhead to MDL. Test: compare to known MDL values.

149. Program synthesis: the number of equivalent programs for a given specification follows the 1/k law. Synthesis instability ∝ (k-1)/k. Test: enumerate equivalent programs, measure synthesis variance.

150. Formal verification complexity: systems with more equivalent valid states (larger Rashomon) require more verification effort. Effort ∝ log|G|. Test: correlate verification time with state-space symmetry.

151-170. [RESERVE: applications to cryptography, algorithms, complexity theory, formal methods]

### Cross-Domain (171-200)

171. The SAME 1/k law predicts diversity at THREE biological levels: codon (confirmed ρ=0.88), protein structure (confirmed ρ=0.88), and TFBS (confirmed p=2.3e-45). Prediction: it should also hold at the chromatin accessibility level (number of TF binding configurations → chromatin state diversity).

172. The SAME stability transition (sech²-like) appears in BOTH gauge theory (confirmed, exact) AND ML explanation stability (confirmed, sigmoid at ρ=0.9). Prediction: it should also appear in financial model stability as market volatility increases.

173. The SAME ensemble resolution works in BOTH ML (DASH, confirmed) AND causal discovery (confirmed, 4× flip reduction). Prediction: it should also work in weather forecasting (ensemble prediction averaging).

174. The cross-domain attenuation factor α differs between levels: protein (α≈0.45-0.88), TFBS (α=1.23). Prediction: α should correlate with the STRENGTH OF SELECTION at each level, providing a universal measure of selective pressure.

175. The framework predicts that ANY system with >1 equivalent configuration will show instability. The SIMPLEST testable version: generate random functions f: X → Y with controlled many-to-one structure. Measure: does the instability of inverse-inference scale with the multiplicity, following the 1/k law? This would be a PURE TEST of the mathematical prediction with no domain-specific confounders.

176. The framework unifies three known impossibility theorems: Arrow (social choice), Bell (quantum mechanics), and this one (explanation). Prediction: there exists a COMMON MATHEMATICAL STRUCTURE underlying all three — all are instances of "symmetric desiderata + non-injective observation → impossibility." Test: formalize Arrow and Bell as ExplanationSystem instances in Lean.

177. The number of equivalent explanations in everyday life (e.g., "why was the train late?") follows the same 1/k scaling. More possible explanations → less stable explanation. Test: survey study measuring explanation stability vs explanation multiplicity.

178. The framework predicts a universal "explanation capacity" = log|V^G| bits. No communication of explanation can exceed this capacity while remaining stable. Test: information-theoretic analysis.

179. The resolution's Pareto-optimality means: in EVERY domain, practitioners who use the G-invariant resolution outperform those who don't, BY A PREDICTABLE AMOUNT. The improvement = (|G|-1)/|G| = 1-1/|G|. Test: measure performance improvement from ensembling/averaging across multiple domains.

180. The attenuation factor α is a UNIVERSAL CONSTANT for each organism. All proteins in the same organism should have α within a narrow range (because selection pressure is genome-wide). Test: compute α across all proteins in one organism, measure the variance.

181-200. [RESERVE: 20 more cross-domain applications combining any two of the 8+ domains]

---

## FILTERING

Remove literally impossible/untestable:
- Any requiring access to proprietary systems (LLM internals, clinical data)
- Any requiring multi-year research programs
- Any that are purely philosophical without quantitative predictions
- Any that reduce to "X predicts X" (circularity — learned our lesson)

## RANKING CRITERIA

For each surviving candidate:
- **Novelty (1-10):** Has this specific quantitative prediction been stated before?
- **Impact (1-10):** Would confirmation change practice in the target domain?
- **Surprise (1-10):** Would domain experts say "I didn't expect this"?
- **Not-derivable (1-10):** Could a domain expert derive this without the framework?
- **Testable-now (1-10):** Can we test this with available tools in < 2 hours?
- **Anti-circular (1-10):** Is the test design robust against self-correlation?

Score = product of top 4 × average of bottom 2.

## TOP 10 TO EXECUTE

After filtering and ranking, execute the top 10 in parallel batches of 3-4.
Each execution must include:
1. The specific quantitative prediction
2. The test data/method
3. The comparison to domain-specific baseline
4. Honest reporting of failure

## EXECUTION PHASES

Phase 1 (parallel batch 1): Top 3 candidates
Phase 2 (parallel batch 2): Candidates 4-7
Phase 3 (parallel batch 3): Candidates 8-10
Phase 4: Meta-analysis — which predictions confirmed?
