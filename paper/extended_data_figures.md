# Extended Data Figures

Nature allows up to 10 Extended Data figures. Each figure below is listed with its source file and caption.

## Extended Data Figure 1: Universal eta plot
- **Source:** `knockout-experiments/figures/universal_eta_plot.pdf`
- **Caption:** Predicted vs. observed instability rate across 16 domain instances using the character-theoretic formula eta = dim(V^G)/dim(V). The parameter-free predictor achieves R^2 = 0.957 across seven well-characterized domains spanning ML attribution, concept probes, model selection, codon degeneracy, and statistical mechanics.

## Extended Data Figure 2: Invariance counting sensitivity
- **Source:** `knockout-experiments/figures/noether_sensitivity.pdf`
- **Caption:** Sensitivity of the invariance counting gap (between-group vs. within-group flip rate) across correlation strengths rho = 0.50 to 0.99. The 47-percentage-point gap is invariant across the entire range, confirming the structural prediction.

## Extended Data Figure 3: Gaussian flip validated scatter
- **Source:** `knockout-experiments/figures/gaussian_flip_validated.pdf`
- **Caption:** Predicted vs. observed per-pair flip rates across five clinical and financial datasets using independent calibration and validation model sets (30 models each). Out-of-sample R^2 = 0.848.

## Extended Data Figure 4: Multi-model 3x3 grid
- **Source:** `knockout-experiments/figures/gaussian_multimodel.pdf`
- **Caption:** Gaussian flip rate validation across a 3x3 grid of model classes (XGBoost, RandomForest, Ridge) and datasets (California Housing, Adult Income, Diabetes). Eight of nine combinations confirm the formula; Ridge on California Housing fails due to violated Gaussianity assumption.

## Extended Data Figure 5: Enrichment tradeoff curve
- **Source:** `knockout-experiments/figures/enrichment_demo.pdf`
- **Caption:** Faithfulness-informativeness tradeoff under bilemma enrichment on synthetic and real data. The G-invariant resolution sits at the Pareto-optimal knee of the convex frontier.

## Extended Data Figure 6: Gene expression reliability scatter
- **Source:** `knockout-experiments/figures/gene_expression.pdf`
- **Caption:** Feature importance reliability on AP_Breast_Lung cancer classification (P=200 genes). 70% of gene importance comparisons are unreliable (|Delta|/sigma < 0.5); only 4% are reliable (|Delta|/sigma > 2).

## Extended Data Figure 7: M sensitivity curve
- **Source:** `knockout-experiments/figures/m_sensitivity.pdf`
- **Caption:** Sensitivity of flip rate estimates to the number of bootstrap retrains M. Convergence is reached by M=30 for all datasets, justifying the calibration/validation protocol.

## Extended Data Figure 8: Q-Q plots for Gaussianity
- **Source:** `knockout-experiments/figures/gaussianity_qq.pdf`
- **Caption:** Q-Q plots of feature importance differences across model retrains for three representative feature pairs. The Gaussian assumption holds for XGBoost and RandomForest but fails for Ridge on near-deterministic features.

## Extended Data Figure 9: CCA spectrum
- **Source:** `knockout-experiments/figures/continuous_symmetry.pdf`
- **Caption:** CCA singular value spectrum showing the gap between invariant and non-invariant subspace dimensions. The spectral gap corresponds to the G-invariant subspace dimension, providing a data-driven estimator of eta.

## Extended Data Figure 10: Clinical/financial audit summary
- **Source:** `knockout-experiments/figures/sage_audit.pdf`
- **Caption:** SAGE algorithm audit across 8 datasets (clinical, financial, genomics). SAGE reduces false discovery rate from 62% to <5% for feature ranking claims while preserving 100% of between-group ranking accuracy.
