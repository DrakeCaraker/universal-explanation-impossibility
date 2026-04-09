# Rebuttal Cheat Sheet

Quick-lookup rebuttals for every vulnerability identified in the axiom audit.
Sorted by likelihood of being raised (most likely first).

---

## VULNERABLE Items (must address pre-emptively)

### "The core proof is trivial / axiom laundering"
**Attack**: "The 4-line proof is just modus ponens with contradictory premises. The axioms encode the Rashomon property, which trivially implies the impossibility."
**Defense**: The core impossibility is deliberately simple — like Arrow's theorem, the contribution is identifying the right abstraction (faithfulness/stability/completeness as the conflicting trio), not proof complexity. The non-trivial content is: (a) quantitative bounds (36-line proofs for FIM witnesses, 33 lines for Spearman), (b) Design Space exhaustiveness (all methods fall in Family A or B), (c) the SBD generalization to 3 domains, and (d) the formalization catching 2 logical inconsistencies that survived informal review.
**Paper location**: Introduction, after contributions list (all 3 versions).

### "Proportionality CV=0.66 means the axiom doesn't hold"
**Attack**: "The proportionality axiom (φ = c·n for uniform c) has CV=0.66 for depth-6 trees. That's a 66% coefficient of variation — the axiom is grossly approximate."
**Defense**: The proportionality axiom is used ONLY for quantitative bounds, not for the core impossibility (Theorem 1, zero axiom dependencies). CV=0.35 for stumps (the idealized setting) validates the leading-order approximation. The α-correction (fitted α≈0.60, R²=0.89) handles finite depth. All quantitative bounds should be interpreted as order-of-magnitude predictions.
**Paper location**: Quantitative Bounds section + Discussion limitations (all 3 versions).

### "The Spearman bound is axiomatized, not proved"
**Attack**: "The tighter Spearman bound (m³/P³) is an axiom, not a theorem. The derived bound is much weaker. The one result doing real quantitative work is the one you didn't prove."
**Defense**: Two bounds exist: the fully derived 3(m-1)²/(P³-P) and the axiomatized m³/P³. Both capture the key qualitative insight (instability grows with m). The gap is a combinatorial counting argument about midrank transposition — domain-neutral mathematics, not a domain-specific assumption. The derived bound already gives the m-scaling.
**Paper location**: Proof status transparency paragraph (all 3 versions).

### "The ratio formula is only accurate for stumps"
**Attack**: "Your 1/(1-ρ²) formula assumes α=1 (full signal capture). Real XGBoost with depth 3-6 has α≈0.3-0.6. The formula overpredicts by 2-5x."
**Defense**: The paper explicitly presents 1/(1-ρ²) as a theoretical upper bound with α=1, and the α-corrected 1/(1-αρ²) as the practical prediction (R²=0.89). The impossibility holds for ANY α>0: the ratio still diverges as ρ→1. The depth table (supplement) shows the depth 3 anomaly and provides practitioner guidance.
**Paper location**: Quantitative Bounds, after Theorem 2 (all 3 versions).

---

## DEFENSIBLE Items (have ready if asked)

### "Surjectivity requires stochastic training"
**Attack**: "Without subsampling (subsample=1.0, colsample_bytree=1.0), XGBoost is deterministic and surjectivity fails."
**Defense**: The XGBoost default is subsample=0.8. Without subsampling, the model is deterministic and there IS no Rashomon set (one model, one ranking, no instability). The impossibility is precisely about the stochastic case — which is the practical default and the recommended regularization.
**Paper location**: Cross-implementation section (supplement).

### "Split count formulas are approximate"
**Attack**: "T/(2-ρ²) is leading-order only. What about higher-order terms?"
**Defense**: The leading-order analysis is standard for theoretical bounds (cf. Tsybakov's minimax lower bounds). The α-correction captures the dominant finite-depth effect. The impossibility holds qualitatively for ANY positive split-count gap, not just the exact formula.
**Paper location**: Setup, after Axiom 2.

### "Real data doesn't have equicorrelation"
**Attack**: "Your equicorrelation assumption (all within-group pairs share ρ) is unrealistic."
**Defense**: The equicorrelation simplifies the axiom system. The Rashomon property holds pairwise — any pair (j,k) with |ρ_{jk}|>0 and similar importance admits models ranking them in opposite orders. The F1 and F5 diagnostics operate on individual pairs and make no distributional assumptions.
**Paper location**: Discussion limitations (all 3 versions).

### "Confidence intervals already handle this"
**Attack**: "I use SHAP confidence intervals. They tell me when features are indistinguishable."
**Defense**: Confidence intervals tell you the interval is wide. They do NOT fix the ranking: if you report a ranking (as required for adverse action notices, feature selection, or variable importance tables), the ranking still flips 50% of the time. The impossibility is about forcing continuous attributions into discrete rankings. CI reporting is an alternative to DASH ties — the paper acknowledges this (Discussion, "CI-based reporting").
**Paper location**: Discussion (all versions).

### "25x training cost is unacceptable"
**Attack**: "Training 25 models is a 25x cost increase. Impractical for production."
**Defense**: The cost is information-theoretically necessary: Ω(σ²/Δ²) model trainings are required even to DETECT instability (query complexity lower bound). Progressive DASH reduces average cost to ~8x. The F5 single-model screen (zero additional training) catches most unstable pairs. For batch pipelines, 25 models in parallel takes the same wall time as 1.
**Paper location**: Discussion computational cost + progressive DASH (JMLR/definitive).

### "SHAP will be obsolete"
**Attack**: "Mechanistic interpretability is replacing SHAP. This paper is about a dying tool."
**Defense**: SHAP is mandated by regulators (EU AI Act, SR 11-7) and is the dominant tool for tabular ML — the majority of production ML. Mechanistic interpretability applies to transformers, not to the XGBoost/random forest models used in banking, insurance, and healthcare. The SBD generalizes beyond SHAP to any symmetric decision problem.
**Paper location**: Discussion broader implications.

### "This is just formalizing the obvious"
**Attack**: "Non-identifiability under collinearity is textbook statistics (VIF, condition number). What's new?"
**Defense**: Classical multicollinearity concerns estimation variance (Var(β̂_j - β̂_k) → ∞). This paper proves ranking impossibility — a qualitatively different claim. Even with perfect coefficient estimates, the ranking of symmetric features is a coin flip because different near-optimal models rank them in opposite orders. The Design Space Theorem (exactly two families) and DASH optimality go well beyond VIF.
**Paper location**: Discussion, multicollinearity paragraph (all versions).

### "Equal causal effects is rare"
**Attack**: "The impossibility requires equal causal effects (β_j = β_k). In practice, features have different effects."
**Defense**: The impossibility is about the PREDICTIVE setting (which feature does the model use more), not the causal setting. When features have different causal effects, conditional SHAP CAN resolve the ranking — and the paper proves this precisely (conditional impossibility + escape condition). The diagnostic measures whether the gap is large enough to overcome the noise.
**Paper location**: Conditional SHAP section (all versions).

---

## Usage

During rebuttal: Ctrl-F for the reviewer's key phrase, find the matching entry, adapt the defense to their specific wording. Every defense points to the paper location where the argument already appears — cite the section number in the response.
