# Complete Corrections List

Compiled from 24 simulated peer reviews (12 domain experts + 12 editorial).
Organized by priority: P0 (must fix before any submission), P1 (fix before JMLR),
P2 (fix if time permits), P3 (acknowledge as limitation).

---

## P0: BLOCKING — Fix Before Any Submission

### C1. Definitional mismatch: decisive (pairwise vs pointwise)
**Source**: Tech Rigor Ref 1, Causal Rev 1
**File**: `paper/nature_article.tex` lines 863-865
**Bug**: The Methods section defines decisive as `exp(θ₁) ⊥ exp(θ₂) → E(θ₁) ⊥ E(θ₂)` (pairwise). The Lean code defines it as `∀ θ h, incomp(explain(θ), h) → incomp(E(θ), h)` (pointwise). These are logically distinct. Under the pairwise definition, the proof contradicts irreflexivity directly — faithfulness is superfluous, reducing the "trilemma" to a dilemma.
**Fix**: Change the Nature article Methods to match the Lean (pointwise) definition. Verify the proof sketch matches. Add a remark that the pairwise version is a strictly weaker consequence.

### C2. Stale theorem/file counts in monograph
**Source**: Lean Rev 2, Repro Ref 7
**File**: `paper/universal_impossibility_monograph.tex` (Table 5.1 area)
**Bug**: Monograph says "82 files, 378 theorems" while CLAUDE.md and Nature article say "95 files, 417 theorems." The actual count (verified by `lake build`) is 95 files.
**Fix**: Update the monograph to match the current Lean state. Add a `make verify` target that auto-checks paper-code consistency before committing.

### C3. Gauge lattice numerical discrepancy (200× error)
**Source**: Gauge Theory Rev 2
**File**: `paper/results_gauge_lattice.json`
**Bug**: Wilson loop means at β=0.1 are 0.022 in the JSON vs analytic prediction of 9.87e-05 — a 200× discrepancy. No error bars reported.
**Fix**: Diagnose the discrepancy (likely a finite-size effect or code bug). Report proper statistical error estimates. If the code is wrong, re-run and correct.

### C4. Kruskal-Wallis p-value mismatch
**Source**: Bio Rev 1
**File**: `paper/results_codon_entropy.json` vs paper text
**Bug**: Paper text reports p = 2.0e-3; JSON shows p = 2.3e-18 (simulated) and p = 2.0e-3 (real NCBI). Paper doesn't specify which dataset.
**Fix**: Clarify in paper text which dataset is primary. Report both values.

---

## P1: HIGH PRIORITY — Fix Before JMLR Submission

### C5. Remove Flyspeck / four-colour theorem comparison
**Source**: ML Rev 1, Physics Rev 1, Math Rev 1, Lean Rev 2, Causal Rev 1 (5 reviewers)
**File**: `paper/nature_article.tex`, `paper/universal_impossibility_monograph.tex`
**Issue**: Comparing 417 theorems (mostly one-liners) to Flyspeck (deep geometric arguments) and the four-colour theorem is misleading.
**Fix**: Remove the comparison. Instead, state honestly: "The core theorem is 4 tactic steps; the formalization's value is in verifying the correctness of 8 constructive domain instances and the GBDT quantitative bounds."

### C6. Redo attention experiment with actual retraining
**Source**: ML Rev 3, Empirical Ref 3 (2 reviewers)
**File**: `paper/scripts/attention_instability_experiment.py`
**Issue**: Weight perturbation (σ=0.01-0.02) creates degraded models, not independently optimal ones. This is not the Rashomon property.
**Fix**: Retrain DistilBERT from 10+ different random seeds to convergence. Report loss of each model. Only compare models with loss within ε of the best. Report attention flip rate for this genuine Rashomon set.

### C7. Test Noether counting at moderate correlations
**Source**: ML Rev 1, ML Rev 3, Empirical Ref 3, Stats Ref 4 (4 reviewers)
**File**: `knockout-experiments/noether_counting_v2.py`
**Issue**: ρ=0.99 is unrealistically extreme. The bimodal gap may collapse at ρ=0.5-0.7.
**Fix**: Run at ρ ∈ {0.5, 0.7, 0.85, 0.95, 0.99}. Report the ρ threshold where bimodal structure emerges. This sensitivity analysis is essential for the prediction to be non-trivial.

### C8. Pre-specify "well-characterized" group criterion for η plot
**Source**: ML Rev 3, Empirical Ref 3, Stats Ref 4 (3 reviewers)
**File**: `knockout-experiments/universal_eta_synthesis.py`
**Issue**: Selecting 7/16 points post-hoc for R²=0.957 is data dredging.
**Fix**: Either (a) report R² for all 16 as the primary result (R²≈0.25-0.60), or (b) pre-specify a formal criterion for "well-characterized" (e.g., "group is exact by construction, not approximate"), document it in PRE_REGISTRATION.md, and test it. Report both R² values with the criterion stated first.

### C9. Fix Mann-Whitney non-independence in Noether counting
**Source**: Stats Ref 4
**File**: `knockout-experiments/noether_counting_v2.py`
**Issue**: 66 pairwise flip rates share model-level dependence. Mann-Whitney treats them as independent. p=2.7e-13 is inflated.
**Fix**: Replace with a permutation test that permutes group labels of features (not individual pair flip rates). Also report a cluster bootstrap resampling at the model level.

### C10. Add power analyses for all experiments
**Source**: ML Rev 3, Stats Ref 4 (2 reviewers)
**File**: All experiment scripts
**Issue**: No experiment has a power analysis. Sample sizes (10 DistilBERT models, 50 XGBoost models, 50 sentences) are not justified.
**Fix**: For each experiment, compute the minimum detectable effect size at 80% power for the given N. Report this alongside results.

### C11. Restructure paper: impossibility as scaffolding, quantitative predictions as centerpiece
**Source**: Red Team Ref 10, Handling Editor, Board Member, Novelty Ref 5 (4 reviewers)
**File**: All paper files
**Issue**: The paper leads with the weakest element (a formalized tautology) and buries the strongest (Noether counting, η law, interpretability ceiling).
**Fix**: Restructure so that Section 1 motivates via one concrete example (e.g., "two equally accurate models rank the same gene as most important and least important"), Section 2 states the impossibility briefly, Section 3 derives the quantitative predictions, Section 4 validates them, Section 5 discusses cross-domain instances as context.

### C12. Explicitly address the "trivially true" critique
**Source**: 10/24 reviewers
**File**: Monograph and Nature article
**Issue**: The paper's existing "Isn't This Obvious?" section is not persuasive.
**Fix**: Strengthen the defense with: (a) the tightness argument — each pair is achievable, so the axioms are not artificially weakened; (b) the biconditional — the impossibility holds IFF Rashomon, meaning the axiom set precisely characterizes the phenomenon; (c) the quantitative corollaries (Noether counting, η law) are non-obvious consequences. Acknowledge that the qualitative impossibility is simple but the quantitative implications are substantive.

### C13. Justify group assignments for ML instances
**Source**: Math Rev 1
**File**: Monograph Table 2 and instance files
**Issue**: S₆ for attention (assumes all 6 heads are exchangeable — empirically false), O(64) for concept probes (assumes full orthogonal group). These are not derived from architecture.
**Fix**: For each ML instance, state explicitly: (a) what group is claimed, (b) what architectural property justifies it, (c) what breaks the symmetry in practice, (d) how the η prediction is affected by symmetry-breaking.

### C14. Address the "resolutions are backwards" critique
**Source**: Physics Rev 1, Physics Rev 2, Crystallography Rev 2, Causal Rev 1, DB Rev 3 (5 reviewers)
**Issue**: The paper frames Wilson loops, CPDAGs, Patterson maps, codon usage tables as "resolutions of the impossibility." Domain experts say these were foundational constructions that historically preceded any impossibility awareness.
**Fix**: Reframe: "These domains independently developed what we formalize as G-invariant resolutions. The framework does not claim to motivate these constructions but to classify them under a common optimality result." Remove language suggesting the impossibility *explains* existing practice.

### C15. Formalize Pareto-optimality in Lean
**Source**: Math Rev 1, Math Foundations Ref 2
**File**: New Lean file needed
**Issue**: The strongest positive result (G-invariant resolution is Pareto-optimal) is "argued (supplement only)," not Lean-verified.
**Fix**: Prove this in Lean. It would be the most non-trivial Lean contribution and directly addresses the "trivially true" concern.

### C16. Remove or qualify "uncertainty principle" and "Noether correspondence" terminology
**Source**: Math Foundations Ref 2
**File**: `paper/universal_impossibility_monograph.tex` bridge theorems section
**Issue**: "Uncertainty principle" (for a pigeonhole bound) and "Noether correspondence" (for V^G = V^G) misappropriate prestigious names for shallow observations.
**Fix**: Either derive genuine quantitative analogues that merit the names, or rename to "explanation tradeoff bound" and "invariance counting principle."

### C17. Engage seriously with the Rashomon set literature
**Source**: Competing Work Ref 6
**Issue**: Fisher et al. (2019) introduced variable importance clouds; Laberge et al. (2023) extracted partial orders from Rashomon sets (= the G-invariant resolution for attributions); Marx et al. (2024) developed uncertainty-aware explainability. The paper doesn't distinguish its contribution from this body of work.
**Fix**: Add a detailed comparison. State: "Laberge et al. (2023) independently discovered the attribution-specific instance of our resolution; our contribution is the abstract framework and its formalization."

### C18. Cite Selbst & Barocas (2018) on legal impossibility of explanation
**Source**: Competing Work Ref 6
**Fix**: Add citation and discussion.

### C19. Cite specific EU AI Act articles and FDA guidance documents
**Source**: Ethics Ref 9
**Issue**: "EU AI Act requires meaningful explanations" is vague. The Act uses different language.
**Fix**: Cite Article 13 (transparency) and Article 86 (right to explanation) specifically. Cite the specific FDA guidance document by name and date.

---

## P2: MEDIUM PRIORITY — Fix If Time Permits

### C20. Expand causal discovery beyond the Asia network
**Source**: ML Rev 3
**Fix**: Add results for Sachs (11 nodes) or ALARM (37 nodes).

### C21. Use proper crystallography algorithm (HIO/RAAR instead of Gerchberg-Saxton)
**Source**: ML Rev 3
**Fix**: Replace error-reduction with a standard phase retrieval algorithm.

### C22. Expand biology to whole-proteome analysis
**Source**: Bio Rev 1, ML Rev 3
**Fix**: Analyze codon usage across >1000 species from all domains of life, not just 120 cytochrome c sequences.

### C23. Clarify Rashomon property for interacting systems
**Source**: Physics Rev 1
**Issue**: S_Ω (full permutation group on microstates) is wrong for interacting systems where not all microstates are physically related.
**Fix**: Qualify: "For non-interacting systems, the symmetry group is S_Ω; for interacting systems, the relevant group is the automorphism group of the Hamiltonian."

### C24. Fix negative entropy artifact
**Source**: Bio Rev 1
**Issue**: Met shows entropy = -0.0 (floating-point artifact).
**Fix**: Clip to max(0, H).

### C25. Report effect sizes (Cohen's d) for all experiments
**Source**: ML Rev 3
**Fix**: Add effect size reporting alongside p-values.

### C26. Separate RandomForest.lean from verified content
**Source**: Lean Rev 2
**Issue**: RandomForest.lean has "no formal proofs" but is included in the file count.
**Fix**: Either prove the results or exclude from the "95 files, 417 theorems" count.

### C27. Reconcile 9 ML instances vs 8 scientific domains scope
**Source**: ML Rev 1
**Fix**: State clearly in Introduction: "9 ML explanation types + 8 non-ML scientific domains = 17 total instances."

### C28. Fix incompatibility relation mismatch in MarkovEquivalence.lean
**Source**: Causal Rev 1
**Issue**: Lean uses `!=` (any distinct DAGs are incompatible); paper uses edge-reversal incompatibility. These are different.
**Fix**: Align the Lean relation with the paper's stated relation, or note the difference explicitly.

### C29. Explain splitCount returning ℝ instead of ℕ
**Source**: Lean Rev 2
**Fix**: Add prominent justification in CLAUDE.md or a comment in Defs.lean.

### C30. Test sensitivity of Census experiment to Dirichlet α
**Source**: DB Rev 3
**Fix**: Run at α ∈ {0.1, 0.5, 1.0, 2.0, 10.0} and report sensitivity.

### C31. Use cluster bootstrap for all pairwise-comparison experiments
**Source**: Stats Ref 4
**Fix**: Replace pair-level bootstrap with model-level (cluster) bootstrap throughout.

### C32. Add multiple-testing correction within Noether experiment
**Source**: Stats Ref 4
**Fix**: Apply Benjamini-Hochberg correction to the 66 pairwise classifications.

### C33. Pin PyTorch and Transformers versions
**Source**: Repro Ref 7
**Fix**: Add `torch==X.Y.Z` and `transformers==X.Y.Z` to requirements.txt.

### C34. Create Docker/CI reproducibility pipeline
**Source**: Repro Ref 7
**Fix**: Add Dockerfile and GitHub Actions workflow for `make validate`.

### C35. Commit cytochrome c sequences as static file
**Source**: Repro Ref 7
**Fix**: Save the 120 NCBI sequences to a committed file (not rely on live API).

---

## P3: ACKNOWLEDGE AS LIMITATIONS

### L1. Core theorem is elementary (pigeonhole-level)
**Acknowledge**: "The qualitative impossibility follows from elementary logic. The contribution is the axiomatic framework (tightness, biconditional, resolution optimality) and the quantitative corollaries."

### L2. Domain instances are minimal witnesses (two-element)
**Acknowledge**: "The constructive witnesses are minimal by design — they suffice for the impossibility but do not capture the full complexity of each domain."

### L3. 1/n interpretability ceiling assumes unbroken S_n symmetry
**Acknowledge**: "The bound applies to untrained networks; training breaks S_n symmetry. The bound is an upper limit, not a tight characterization of trained-network interpretability."

### L4. Cross-domain instances do not produce new domain-specific knowledge
**Acknowledge**: "Each instance was independently known. The contribution is the formal unification and the demonstration that a common resolution strategy applies."

### L5. Rashomon property is axiomatized (not derived) for ML instances
**Acknowledge**: "Deriving the Rashomon property from loss landscape geometry or SGD dynamics would strengthen the ML results."

### L6. Simulated reviewers may share biases due to adversarial prompting
**Acknowledge in internal notes (not paper)**: The 75% reject rate reflects worst-case adversarial framing.

### L7. Synonymous codons are not truly "gauge-equivalent"
**Acknowledge**: "Synonymous codons produce the same amino acid but differ in translational efficiency, mRNA stability, and epigenetic marking. The gauge analogy applies to the protein-coding function only."

### L8. Modern crystallography achieves decisive structure determination
**Acknowledge**: "Experimental phasing methods (SAD, MAD, MIR) effectively break the Rashomon property by introducing additional measurements."
