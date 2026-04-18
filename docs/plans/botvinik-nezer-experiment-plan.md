# Rigorous Experiment Plan: Brain Imaging (Botvinik-Nezer Reanalysis)

## The Opportunity

Botvinik-Nezer et al. (Nature 2020) gave the same fMRI dataset to 70 independent analysis teams. Each team chose their own preprocessing pipeline, statistical model, and thresholding procedure. They reported which brain regions were "significantly activated" for the task. The teams reached dramatically different conclusions — fewer than 25% of the brain regions were identified by a majority of teams.

This IS the Rashomon property for neuroimaging: 70 equivalent analysis pipelines (same data, same task, different methods) produce incompatible brain-region explanations. The paper documented the problem. Nobody has explained WHY specific regions disagree or predicted WHICH regions would be unstable.

**The prediction:** The framework predicts that brain regions with correlated activation patterns (similar BOLD signals across voxels) will have unstable attribution across analysis teams, while regions with unique, uncorrelated activation will be stable. The η law should quantify the instability rate from the spatial correlation structure of the brain imaging data.

## Data Availability

**The Botvinik-Nezer data is public:**
- Original dataset: OpenNeuro ds001734 (https://openneuro.org/datasets/ds001734)
- Team results: Available in supplementary materials of the Nature paper
- Specifically: Table S1 contains each team's reported significant regions for each hypothesis

**What we need from their data:**
1. The binary activation maps from all 70 teams (which regions each team flagged as significant)
2. The underlying fMRI data OR a brain-region correlation matrix (to define the symmetry structure)

**What we DON'T need:**
- To rerun any fMRI analysis (the 70 teams already did this)
- To access raw imaging data (we only need the RESULTS of each team's analysis + the spatial correlation structure)

## Experimental Design

### Phase 1: Data Acquisition and Preprocessing

**Step 1.1:** Download team results from the Botvinik-Nezer supplementary materials.
- The paper reports results for 9 hypotheses tested by 70 teams
- For each team × hypothesis: a binary map of which regions are "significant"
- Convert to a matrix: teams × regions × hypotheses

**Step 1.2:** Obtain or compute brain-region spatial correlation.
- Option A (preferred): Use Schaefer atlas parcellation (400 regions) correlation matrix from resting-state fMRI (available from HCP or other public sources). This gives the "baseline" spatial correlation of brain regions independent of this specific task.
- Option B: Compute inter-region correlation from the task fMRI data itself, using the average time series per region.
- Option C: Use anatomical adjacency as a proxy for correlation (regions that are anatomically close tend to have correlated BOLD signals). This is the weakest but requires no additional data.

**Step 1.3:** Define the "symmetry group" structure.
- The symmetry group here is NOT a permutation group on features. It's a correlation structure on brain regions.
- Two brain regions are "approximately exchangeable" if they have high spatial correlation (|r| > threshold).
- This defines correlation clusters analogous to our feature groups in the SHAP framework.
- Use multiple thresholds (|r| > 0.3, 0.5, 0.7, 0.8) for sensitivity analysis.

### Phase 2: The Prediction (compute BEFORE looking at team disagreement)

**Step 2.1:** From the spatial correlation matrix, compute correlation clusters (connected components at threshold |r| > τ).

**Step 2.2:** For each brain region, compute:
- Its cluster size k (how many other regions it's correlated with)
- The predicted instability: η = 1 - 1/k (fraction of information lost under orbit averaging)
- Alternative prediction: coverage conflict = whether both "significant" and "not significant" appear across teams for this region

**Step 2.3:** Save the predictions to a JSON file with a hash BEFORE examining team disagreement. This is the pre-registration.

### Phase 3: The Observation (measure team disagreement)

**Step 3.1:** For each brain region and each hypothesis, compute:
- The agreement rate = fraction of teams that agree on the majority verdict (significant vs not)
- The flip rate = 1 - agreement rate
- This is analogous to the SHAP flip rate: how often does the "explanation" (significant or not) change across "models" (analysis teams)

**Step 3.2:** Compute the bimodal structure:
- Are flip rates bimodal (some regions always agree, others always disagree)?
- Hartigan's dip test with permutation control
- Within-cluster vs between-cluster flip rates (Noether counting analogy)

### Phase 4: The Comparison

**Step 4.1:** Correlate predicted instability (from spatial correlation, Step 2.2) with observed disagreement (from team results, Step 3.1).
- Primary metric: Spearman ρ between predicted η and observed flip rate
- The null model comparison: does raw spatial correlation predict disagreement as well as the framework?
- The coverage conflict comparison: does the nonparametric predictor (both verdicts present) outperform the parametric prediction?

**Step 4.2:** Control analyses:
- Permutation control: shuffle region labels and recompute correlation. Does the prediction still hold? (Should not.)
- Hypothesis-specific analysis: does the prediction work equally well for all 9 hypotheses? (Some may have stronger effects than others.)
- Sensitivity to correlation threshold τ: does the prediction degrade gradually or sharply as τ changes?

**Step 4.3:** The headline test:
- Can we predict which brain regions will have >50% team disagreement using ONLY the spatial correlation structure?
- Precision, recall, F1 of the prediction
- Comparison to the trivial baseline: "regions near the activation boundary disagree more"

### Phase 5: Reporting

**What constitutes success:**
- Spearman ρ > 0.3 between predicted and observed instability (meaningful correlation)
- The framework outperforms or matches the trivial baseline (spatial adjacency)
- Bimodal structure detected (some regions always agree, others disagree) with dip p < 0.05
- Results robust to correlation threshold choice (at least 3 of 4 thresholds work)

**What constitutes failure:**
- Spearman ρ < 0.1 (no meaningful prediction)
- The trivial baseline (adjacency) beats the framework
- No bimodal structure (uniform disagreement across regions)

**What to report either way:**
- The prediction, the observation, and the comparison — regardless of outcome
- All sensitivity analyses
- The null model comparison
- Honest discussion of what drives the result (spatial correlation vs task difficulty vs thresholding effects)

## Potential Confounds and How to Address Them

**Confound 1: Team disagreement might be driven by thresholding, not by Rashomon.**
Different teams use different significance thresholds (p < 0.001 vs p < 0.01). Regions near the boundary disagree more because of thresholding, not because of genuine analysis instability.
→ Control: Use the Botvinik-Nezer unthresholded maps (continuous t-statistics) if available, not just binary significant/not-significant.

**Confound 2: The spatial correlation matrix might be dominated by distance, not functional similarity.**
Nearby brain regions are always correlated (spatial autocorrelation). The "symmetry group" might just be "neighboring regions."
→ Control: Compare to a pure distance-based predictor (adjacency matrix). If the framework beats distance, the prediction uses functional correlation beyond mere proximity.

**Confound 3: The 70 teams used different software packages (SPM, FSL, AFNI) which have systematic differences.**
Teams using the same software might agree more — the "symmetry group" might be software-based, not brain-region-based.
→ Control: Stratify by software and check if within-software agreement > between-software agreement. If so, the primary "Rashomon" is software choice, not brain-region correlation.

**Confound 4: Some hypotheses are easier than others.**
Hypothesis 1 (large effect size) might show high agreement across all regions; Hypothesis 9 (small effect size) might show low agreement.
→ Control: Analyse each hypothesis separately. The framework predicts that the PATTERN of instability (which regions disagree) is predicted by spatial correlation regardless of overall agreement level.

## Timeline and Requirements

**Data acquisition:** 1-2 days (download from OpenNeuro + supplementary tables)
**Correlation computation:** 1 day (parcellation + correlation matrix)
**Prediction computation:** 1 hour (apply η law to correlation clusters)
**Observation computation:** 2-3 hours (parse 70 teams × 9 hypotheses × regions)
**Analysis:** 1 day (correlations, controls, sensitivity)
**Writing:** 1-2 days

**Total:** ~1 week with focused effort.

**Requirements:**
- Python + nibabel/nilearn for neuroimaging data handling
- The Botvinik-Nezer supplementary data (public)
- A brain parcellation atlas (Schaefer 400, available in nilearn)
- Optionally: a resting-state correlation matrix (HCP, available publicly)

## Why This Could Be a Knockout

Botvinik-Nezer et al. (2020) is one of the most-cited neuroimaging papers of the decade (>1000 citations). It demonstrated a crisis in neuroimaging analysis. But it didn't explain WHY certain regions disagree or predict WHICH ones would. If the framework predicts this — using only the spatial correlation structure, with no free parameters — it would:

1. Reframe a known crisis through the impossibility theorem
2. Give neuroimagers a tool to predict which findings are reliable BEFORE running the analysis
3. Connect neuroscience to ML, physics, and biology through a single mathematical structure
4. Appear in Nature as a reanalysis of a paper Nature already published

The weakness: it might not work. Brain-region disagreement might be driven by thresholding choices, software differences, or effect sizes rather than spatial correlation structure. The controls (confounds 1-4) are designed to detect this. If the confounds dominate, we report honestly — another well-defined boundary condition.

## Pre-Registration Statement

Before examining team disagreement (Phase 3), we will:
1. Compute predicted instability from spatial correlation (Phase 2)
2. Save predictions to a timestamped, hashed JSON file
3. Define success criteria (Spearman ρ > 0.3, bimodality, robustness to threshold)
4. Only then compare to observed disagreement

This ensures the prediction is genuinely prospective within the analysis pipeline, even though the Botvinik-Nezer data already exists.
