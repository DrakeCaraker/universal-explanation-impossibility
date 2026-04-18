# Knockout Scorecard: Final Status

*All locally-testable candidates have been run. This is the definitive record.*

---

## Tested and Resolved

| # | Direction | Verdict | Key Number | Lesson |
|---|-----------|---------|------------|--------|
| S1 | Rashomon topology → bimodality | ❌ ELIMINATED | ρ(cluster,flip)≈0 across 4 datasets | Bimodality is from feature structure, not model topology |
| S2 | SAM → stable explanations | ⚠️ PARTIAL | 53% flip reduction on BC, 7% on synthetic | Real but dataset-dependent; not a universal fix |
| A2 | Spectral gap → ensemble size | ❌ ELIMINATED | Theory 14-100× too fast, even at ρ=0.999 | Exact-group theory doesn't predict approximate-group convergence |
| A6 | Flip correlations from irreducibles | ❌ ELIMINATED | Within-group corr (-0.031) ≈ between (-0.013) | Features in same irreducible do NOT flip together |

## Running

| # | Direction | Status | Expected completion |
|---|-----------|--------|-------------------|
| S5 | Lottery tickets / MI v2 (circuit Rashomon) | 🔄 Running locally (10 models) | Hours |

## Deferred (require external data, tools, or collaborators)

| Rank | # | Direction | What's needed | Knockout potential | Why |
|------|---|-----------|--------------|-------------------|-----|
| **1** | 7 | Brain imaging (Botvinik-Nezer) | Their public dataset (70 teams, same fMRI data) | **Very high** | Already in Nature (2020). Showing their disagreement IS the impossibility — with η law predicting which brain regions are stable — would reframe a known result through the framework. Nature readers already know the paper. |
| **2** | 15 | AI safety benchmarks | HELM/Open LLM Leaderboard runs | **Very high** | AI safety is the defining issue. Showing benchmark rankings face the impossibility — and predicting which rankings are stable — would reach every AI policymaker. |
| **3** | 11 | Clinical published scores | Specific papers with code/data | **High** | "Published clinical risk factor X is unstable" with the paper named. Requires finding the right paper. |
| **4** | 8 | Meta-analysis = orbit averaging | None (theoretical proof) | **High** | Proving meta-analysis is Pareto-optimal via the framework. Elegant but requires careful formulation to avoid being dismissed as "obvious." |
| **5** | S3 | Mode connectivity → explanation interpolation | Model interpolation code (e.g., git re-basin) | **High** | If mode-connected models have smoothly interpolating explanations, explanation stability = loss landscape connectivity. Connects two research programs. |
| **6** | S4 | Rate-distortion curve | Theoretical + numerical | **Medium-High** | Beautiful connection to Shannon theory but may be too abstract for Nature. Strong for IEEE IT or COLT. |
| **7** | 10 | Human experts show impossibility | IRB approval, 30+ physicians | **Very high but slow** | If human diagnosticians show the same bimodal flip pattern predicted by η law, the theorem applies to minds, not just machines. 6+ months to execute. |
| **8** | 14 | Diagnostic disagreement predicted by η law | Clinical dataset with multiple readers | **High** | Radiology inter-rater reliability predicted from imaging correlation structure. |
| **9** | 9 | Neutral theory = the resolution | Evolutionary data (dN/dS ratios) | **Medium-High** | Kimura's neutral theory as an instance. Beautiful for biologists but niche. |
| **10** | 12 | Particle physics analysis instability | LHC analysis data | **Medium** | Most rigorous science faces the impossibility too. But needs physics collaborator. |
| **11** | 13 | Protein LM attribution stability | ESM-2 fine-tuning | **Medium** | Hot topic (protein LMs) but requires structural biology expertise. |
| **12** | 6 | Replication crisis prediction | Many Labs / Reproducibility Project data | **Very high but hard** | Predicting replication failure from Rashomon measure. Transformative if confirmed but methodologically challenging. |

---

## The Empirical Pattern

After testing 7 knockout candidates and 8 exploratory experiments:

**What works (exact symmetry + theorem-level predictions):**
- η law: R²=0.957 on 7 domains with known groups
- Noether counting: 47pp bimodal gap, invariant across ρ
- Quantitative bilemma: Δ predicts faithfulness loss (ρ=0.832)
- Coverage conflict: Spearman 0.96 per-feature prediction
- Bimodality: confirmed (dip p<0.002) under collinearity

**What fails (approximate symmetry + extension predictions):**
- Null model beats framework at aggregate flip rate (R²=0.925 vs -6.08)
- Spectral gap theory 14-100× too fast
- Rashomon topology doesn't predict flips
- Irreducible decomposition doesn't predict flip correlations
- Drug discovery: Pearson clustering fails (MI recovers magnitude, not structure)

**The boundary:** The framework's quantitative predictions work when the symmetry group is EXACTLY KNOWN and the system has EXACT exchangeability. At approximate symmetry, simpler methods (correlation-based null model, minority fraction) match or beat the framework's specific predictions. The framework's unique value is in:
1. Proving the instability is unavoidable (no method escapes)
2. Proving the resolution is optimal (DASH is Pareto-best)
3. Cross-domain unification (8 fields, same structure)
4. The quantitative bilemma (Δ → faithfulness loss, not capturable by null model)
5. Exact-symmetry structural predictions (bimodality, Noether counting)

---

## What This Means for the Papers

**For Nature:** Lead with the unification + η law + Noether counting + resolution optimality. These are unchallenged. Acknowledge the approximate-symmetry boundary as a limitation. The SAM result (53%) is worth a sentence in Discussion as a future direction.

**For NeurIPS:** Lead with the coverage conflict diagnostic (0.96) + quantitative bilemma (0.83) + clinical reversal (45%) + DASH resolution. These are the practical contributions. The approximate-symmetry failures don't affect the diagnostic tools.

**For future work:** The deferred knockout candidates (#7 brain imaging, #15 AI safety, #11 clinical scores) are the highest-value next steps. All require external data or collaborators. The SAM direction (#S2) is worth developing with proper SAM implementation (not just regularization proxy).

---

## Lessons Learned

1. **Test predictions BEFORE publishing them.** Every untested prediction we checked empirically either failed or was weaker than expected. The honest negatives (5 failures) are more valuable than the positives — they define the framework's actual boundary.

2. **Simple baselines are essential.** The null model Φ(-c√(1-ρ²)) exposed that much of the "prediction" is trivially achievable without group theory. Always compare to the simplest possible baseline.

3. **The framework's value is in the THEOREM, not in empirical prediction.** The impossibility proof, the Pareto optimality, the cross-domain unification — these are the genuine contributions. The quantitative predictions (η law, Noether) work at exact symmetry; at approximate symmetry, simpler tools suffice.

4. **Honest reporting of failures builds credibility.** 3 pre-registered falsifications + 5 knockout elimination experiments + 1 honest negative (drug discovery) = a paper that's very hard to dismiss as cherry-picked.
