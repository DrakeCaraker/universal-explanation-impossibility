# Pre-Submission Audit Checklist

## Phase 1: Lean Verification

### Universal repo (100 files, 491 theorems, 25 axioms, 0 sorry)
- [ ] `lake build` compiles with zero errors
- [ ] Counts match: `grep -c "^theorem\|^lemma"` = 491, `grep -c "^axiom"` = 25, `grep -c "sorry"` in tactics = 0
- [ ] The core theorem `explanation_impossibility` prints with `#print axioms` showing ZERO domain axioms
- [ ] Each constructive instance (Attribution, Attention, Causal, Concept, MechInterp, ModelSelection, Saliency, LLMExplanation) compiles and uses zero domain axioms

### Attribution repo (58 files, 357 theorems, 6 axioms, 0 sorry)
- [ ] `lake build` compiles
- [ ] Counts match
- [ ] `attribution_impossibility` uses zero behavioral axioms
- [ ] `consensus_equity` (DASH) uses 6 axioms as documented

### Ostrowski repo (20 files, 240 theorems, 10 axioms, 0 sorry)
- [ ] `lake build` compiles
- [ ] Counts match
- [ ] Classification theorem `complete_classification` compiles
- [ ] All 10 axioms are in EnrichmentStack.lean (physics Levels 2-3)

## Phase 2: Every Number in the Nature Paper

Each number must trace to a specific JSON file and field.

### Abstract
- [ ] "10,935 gene-expression features" → dataset shape in results_gene_expression_replication.json
- [ ] "ρ = 0.52" → results_mi_v2_final_validation.json invariant_decomposition.full_10dim.spearman
- [ ] "ρ = 0.93" → same, g_invariant_4dim.spearman
- [ ] "48 analysis teams" → results_brain_imaging_definitive.json
- [ ] "16 independent analyses" → results_brain_imaging_bulletproof.json convergence.M_95_median
- [ ] "491 theorems" → grep count on universal repo
- [ ] "R² = 0.96" → results_universal_eta.json (should be 0.957, paper rounds to 0.96)

### Instance 1: Genomics
- [ ] "92% of seeds" → results_gene_expression_replication.json ap_colon_kidney_colsample.dominant_fraction
- [ ] "6%" → same, top1_distribution.CEACAM5 / 50
- [ ] "remaining 2%" → same, top1_distribution has IGFBP3: 1
- [ ] "ρ = 0.858" → same, TSPAN8_CEACAM5_feature_correlation
- [ ] "zero overlap in Biological Process terms" → results_go_enrichment.json pathway_comparison.TSPAN8_vs_CEACAM5.bp_overlap
- [ ] "78/22" → same, ap_endometrium_colon.top1_distribution
- [ ] "72/28" → same, ap_breast_colon.top1_distribution
- [ ] "98/2" → same, ap_breast_lung_control.top1_distribution
- [ ] "96.4% mean accuracy, 95.1-97.6%" → computed in session (not in JSON — needs recording)
- [ ] "ρ = 0.858, 0.781, 0.843" correlations per dataset → verify in JSON

### Instance 2: MI
- [ ] "ρ = 0.518" → results_mi_v2_final_validation.json
- [ ] "2.2% Jaccard" → results_mech_interp_definitive_v2.json fourier.mean_frequency_jaccard
- [ ] "ρ = 0.300 / 0.668 / 0.518" controls → results_mi_v2_comprehensive.json controls
- [ ] "p = 2.0 × 10⁻⁴" → same, post_vs_random_p
- [ ] "r = 0.9998" determinism → same, determinism_r
- [ ] "ρ = 0.929" → results_mi_v2_final_validation.json invariant_decomposition.g_invariant_4dim.spearman
- [ ] "ρ = 0.822" → same, invariant_excl_mlp1_3dim.spearman
- [ ] "0.500 exactly" within-layer flip → same, noether_counting.within_mean_flip
- [ ] "0.000" head-vs-MLP → verify from noether_counting (note: between_mean_flip is 0.227, not 0.000 — this is for head-vs-MLP1 specifically)
- [ ] "gap = 0.273, p = 2.4 × 10⁻⁵" → same, noether_counting.gap
- [ ] "CV = 0.027" MLP1 → verify source

### Instance 4: Neuroimaging
- [ ] "d = 0.32" → results_brain_imaging_definitive.json activation_control.controlled_d
- [ ] "permutation p < 0.001" → results_brain_imaging_rashomon_v3.json phase2 permutation_p
- [ ] "16 [10, 22]" → results_brain_imaging_bulletproof.json convergence.M_95_median and M_95_ci
- [ ] "within 3%" → results_brain_imaging_resolution.json pareto (verify all methods within 3%)
- [ ] "p < 10⁻⁴ and p < 10⁻⁵⁴" software → results_brain_imaging_definitive.json software

### η Law
- [ ] "R² = 0.957, slope 0.91" → results_universal_eta.json
- [ ] "50.0% within-group, 0.2% between-group" → results_noether_counting.json
- [ ] "p = 2.7 × 10⁻¹³" → same

### Discussion
- [ ] "0.79-0.94 within-family" → Ostrowski docs/model-class-rigorous-results.md
- [ ] "0.19-0.57 cross-family" → same
- [ ] "mean ρ = 0.82 coverage conflict" → same
- [ ] "0/12 violations Var[SHAP]" → same
- [ ] "20 impossibility theorems from 12 domains" → unified-meta-theorem-results.md

## Phase 3: Experimental Reproducibility

- [ ] Every experiment script in knockout-experiments/ runs without error
- [ ] `make validate` succeeds in the universal repo
- [ ] Gene expression replication script produces results matching the JSON
- [ ] MI v2 results match the JSON (models are cached as .pt files)
- [ ] Brain imaging scripts produce results matching JSONs (NIfTI cache required)

## Phase 4: Reference Verification

- [ ] Every \cite{} in nature_article.tex resolves in references.bib
- [ ] Every \ref{} resolves to a \label{}
- [ ] Botvinik-Nezer 2020 citation is correct (Nature, volume 582)
- [ ] Silberzahn 2018 citation is correct (AMPPS, volume 1)
- [ ] Breznau 2022 citation is correct (PNAS, volume 119)
- [ ] Nanda 2023 citation is correct (arXiv)
- [ ] Bilodeau 2024 citation is correct (PNAS, volume 121)
- [ ] FoP companion paper citation is correct

## Phase 5: Nature Formatting

- [ ] Main text word count ≤ 3,500 (Nature Articles limit)
- [ ] Methods section is separate (doesn't count toward word limit)
- [ ] Extended Data: ≤ 10 items (currently 1 table + 2 figures + 1 classification table = 4)
- [ ] All figures are ≥ 300 dpi
- [ ] No supplementary figures in main text
- [ ] Data availability statement is complete
- [ ] Code availability statement is complete
- [ ] Competing interests declared
- [ ] Acknowledgements include AI assistance disclosure
- [ ] Author contributions should be added

## Phase 6: Adversarial Review Simulation

### Reviewer 1 (Computational Biologist)
- Is the gene expression finding robust to different feature selection methods (not just top-50 by variance)?
- Is the GO enrichment analysis standard or cherry-picked?
- Are the microarray probe mappings correct (Affymetrix HG-U133A)?

### Reviewer 2 (ML Theorist)
- Is the theorem genuinely novel vs Bilodeau et al. 2024?
- Is the proof trivial enough that it doesn't warrant Nature?
- Does the tightness classification add enough beyond "pick two vs enrich"?

### Reviewer 3 (Statistician)
- Are the bootstrap CIs correctly computed?
- Is the NARPS convergence prescription valid for new datasets (not just this pool)?
- Is the η law R² inflated by the stat mech point (which is an outlier at 0.996/1.0)?

### Reviewer 4 (AI Safety)
- Does the MI finding extend beyond 2-layer transformers?
- Is activation patching a reliable enough measurement?
- Is the G-invariant projection trivially MLP-driven?
