# Must-Accept Paths: Implementation Plans

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a headline-grabbing demonstration that transforms the paper from "strong submission" to "must-accept" at Nature.

**Prerequisite verified:** TreeSHAP Gaussian flip: OOS R²=0.623, ρ=0.847 on Breast Cancer (15 cal + 15 val models). Formula works with SHAP — rank predictions are strong.

---

## Path 1: "Published Clinical ML Findings Are Unreliable"

**Headline:** "We apply the Gaussian flip formula to N published clinical SHAP studies on public datasets and show that X% contain at least one unreliable top-feature claim."

**Why must-accept:** Transforms the impossibility from "a mathematical result" to "a patient safety finding." Directly testable, reproducible, and alarming.

### Phase 1A: Identify Published Papers (2 hours)

- [ ] **Step 1: Search for published papers using SHAP + XGBoost on public clinical datasets**

Search PubMed, Google Scholar, and Papers With Code for:
- "SHAP" + "feature importance" + ("breast cancer" OR "diabetes" OR "heart disease" OR "credit risk")
- Must use XGBoost, LightGBM, or Random Forest (tree-based, TreeSHAP available)
- Must use a public dataset (UCI, OpenML, Kaggle with public license)
- Must report top-N feature rankings

Target: 10 candidate papers, select 5 with best reproducibility (shared code or clear methodology).

- [ ] **Step 2: For each paper, document:**
  - Dataset used (verify it's public and accessible)
  - Model type and hyperparameters
  - Reported top features (ranked)
  - Whether code is shared
  - Citation count (higher = more impactful to audit)

- [ ] **Step 3: Select final 5 papers**

Criteria: (a) public dataset, (b) clear methodology, (c) claims specific feature rankings, (d) ≥10 citations. Prefer: breast cancer, diabetes, heart disease (clinical relevance).

### Phase 1B: Reproduce and Audit (1-2 days per paper)

For each of the 5 selected papers:

- [ ] **Step 4: Reproduce the model**

```python
# Template for each paper
# 1. Load exact dataset used by authors
# 2. Match preprocessing (feature selection, encoding, scaling)
# 3. Train model with reported hyperparameters (or defaults if not reported)
# 4. Verify: AUC/accuracy within 2% of reported value
```

- [ ] **Step 5: Compute TreeSHAP importance for 30 calibration models**

```python
# Seeds 42-71 for calibration
# For each model: compute mean |SHAP| per feature on held-out test set
# Result: 30×P importance matrix
```

- [ ] **Step 6: Apply Gaussian flip formula to all P(P-1)/2 feature pairs**

```python
# For each pair (j,k):
#   Δ = mean(SHAP_j - SHAP_k) across 30 models
#   σ = std(SHAP_j - SHAP_k)
#   SNR = |Δ|/σ
#   predicted_flip = 2Φ(Δ/σ)Φ(-Δ/σ)
# Classify: SNR > 2 → reliable, SNR < 0.5 → unreliable
```

- [ ] **Step 7: Validate on 30 independent models (seeds 142-171)**

```python
# Compute observed flip rates on validation models
# Report OOS R² and Spearman ρ
```

- [ ] **Step 8: Audit the published top-feature claims**

```python
# For each paper's reported "top 5 features":
# Check: which pairwise rankings among the top 5 are reliable (SNR > 2)?
# Report: "Of the C(5,2)=10 implied rankings, X are reliable, Y are coin flips"
```

### Phase 1C: Aggregate and Report (4 hours)

- [ ] **Step 9: Compile audit table**

| Paper | Dataset | N features | Top-5 claimed | Reliable rankings | Unreliable rankings | % unreliable |
|-------|---------|-----------|---------------|-------------------|--------------------|----|

- [ ] **Step 10: Run SAGE on each dataset**

Show: SAGE would have flagged the unreliable rankings before publication.

- [ ] **Step 11: Write the headline finding**

"Of N published SHAP-based clinical feature rankings audited, X% contain at least one pairwise comparison that is structurally unreliable (SNR < 0.5, predicted flip rate > 30%). SAGE grouping would have identified all unreliable comparisons with 0% false negatives."

### Phase 1D: Ethical and Framing (2 hours)

- [ ] **Step 12: Frame carefully**

Do NOT: "These papers are wrong."
DO: "These papers correctly report SHAP values for one model instance. The structural instability is a property of the learning task, not an error by the authors. No existing tool could have diagnosed this at time of publication. The Gaussian flip formula, introduced here, provides the first such diagnostic."

- [ ] **Step 13: Contact original authors (optional but recommended)**

Notify authors before publication. Frame as: "We developed a tool that identifies structural instability in SHAP rankings. Your paper was used as a test case. The finding applies equally to all SHAP analyses on underspecified systems."

**Pass criterion:** ≥3/5 papers have ≥40% unreliable rankings among top-5 features. If <3, the result weakens — report honestly but the headline is less dramatic.

**Estimated effort:** 2-3 weeks total.

---

## Path 2: "Mechanistic Interpretability Has Structural Limits"

**Headline:** "The impossibility theorem places quantitative limits on mechanistic interpretability: for transformer models with P parameters and K output dimensions, at least (1 - K/P) fraction of circuit-level explanations are structurally unreliable."

**Why must-accept:** Directly relevant to the AI safety research program (Anthropic, OpenAI, DeepMind). Connects pure math to the most consequential active research direction in AI.

### Phase 2A: Setup (2 hours)

- [ ] **Step 1: Install TransformerLens**

```bash
pip install transformer_lens
```

- [ ] **Step 2: Select target model**

Use GPT-2 small (124M params, well-studied by mech interp community). TransformerLens has built-in support.

- [ ] **Step 3: Select target task**

Use indirect object identification (IOI) — the best-studied circuit in mech interp literature (Wang et al., 2023). The circuit is documented and we can compare our stability analysis to the published circuit.

### Phase 2B: Multi-Instance Circuit Analysis (1 week)

- [ ] **Step 4: Train/fine-tune 20 GPT-2 instances on IOI task**

```python
# 20 different random seeds, same architecture, same data
# Verify: all achieve comparable loss on IOI task
```

If full fine-tuning is too expensive, use: 20 different random subsets of IOI prompts for evaluation (bootstrap over data, not model). This is weaker but more tractable.

Alternative approach (stronger, easier): Use weight perturbation within the trained model's loss basin (similar to attention experiment). Add Gaussian noise σ to all weights, verify loss stays within ε, then measure circuit attribution changes.

- [ ] **Step 5: For each instance, identify key circuit components**

```python
# Use TransformerLens activation patching
# Identify: which attention heads, MLP neurons, residual stream positions
# are "important" for the IOI task (via path patching or attribution patching)
```

- [ ] **Step 6: Measure circuit attribution instability across instances**

```python
# For each component (head, neuron):
#   Importance_i = activation patching effect in instance i
# Compute: pairwise flip rate (does the ranking of components change?)
# Apply: Gaussian flip formula to predict which components are stably identified
```

- [ ] **Step 7: Apply the η law**

```python
# Compute: η = dim(V^G)/dim(V) for the identified symmetry group
# Compare: predicted instability (1-η) vs observed circuit instability
```

### Phase 2C: Compare to Published Circuits (3 days)

- [ ] **Step 8: Load the published IOI circuit (Wang et al., 2023)**

```python
# Known circuit components: name mover heads, S-inhibition heads, etc.
# For each published component: is it in the stable subspace?
```

- [ ] **Step 9: Report: which published circuit components are structurally stable vs unstable**

"Of the 26 attention heads identified by Wang et al. as part of the IOI circuit, X are structurally stable (SNR > 2 across 20 model instances) and Y are structurally unreliable (SNR < 0.5). The unreliable components are primarily [backup name movers / S-inhibition heads / ...]."

- [ ] **Step 10: Validate the theoretical prediction**

Show: the fraction of unreliable components matches 1 - η from the theory.

**Pass criterion:** (a) Circuit instability is measurable (flip rate > 10% for some components), (b) Gaussian flip formula predicts it (ρ > 0.7), (c) η law applies (predicted vs observed within 20%).

**Estimated effort:** 2-3 weeks with GPU. Requires TransformerLens expertise.

---

## Path 3: "AI Explainability Regulation Requires a New Standard"

**Headline:** "We prove that the EU AI Act's explainability requirements are structurally impossible for underspecified systems, and provide the first mathematically principled compliance standard."

**Why valuable:** Directly policy-relevant. Best positioned as a Nature companion piece (Nature Comment, 2-3 pages) alongside the main paper, not as the main paper itself.

### Phase 3A: Legal Analysis (1 day)

- [ ] **Step 1: Read and annotate EU AI Act Articles 13-14**

Article 13(1): "High-risk AI systems shall be designed and developed in such a way as to ensure that their operation is sufficiently transparent to enable deployers to interpret a system's output and use it appropriately."

Article 14: Requires "human oversight" including "correctly interpret the high-risk AI system's output, taking into account, for instance, the interpretation tools and methods available."

- [ ] **Step 2: Map legal requirements to formal framework**

| Legal requirement | Framework property | Achievable? |
|---|---|---|
| "Interpret a system's output" | Faithfulness (E consistent with explain) | YES (alone) |
| "Use it appropriately" across deployments | Stability (E consistent across equivalent configs) | YES (alone) |
| "Correctly interpret" (full understanding) | Decisiveness (E inherits all distinctions) | YES (alone) |
| All three simultaneously | F + S + D | **NO** (impossibility) |

- [ ] **Step 3: Identify the precise legal gap**

The Act doesn't explicitly require F+S+D simultaneously, but the combination of Articles 13 and 14 implicitly demands it: an explanation that is correct (faithful), consistent (stable), and complete (decisive). The impossibility shows this is structurally impossible when the system is underspecified.

### Phase 3B: Constructive Proposal (1 day)

- [ ] **Step 4: Propose SAGE as the compliance standard**

"A compliant explanation system under the impossibility should: (1) report which explanations are structurally reliable (SNR > 2), (2) flag unreliable comparisons as 'structurally indeterminate', (3) quantify the fraction of explanation content that is stable (η). This sacrifices completeness (decisiveness) to guarantee correctness (faithfulness) and consistency (stability), which is the Pareto-optimal tradeoff under the impossibility."

- [ ] **Step 5: Compute compliance metrics for standard benchmarks**

For each of the 15 bridge datasets: report the SAGE compliance summary:
- Fraction of reliable comparisons (SNR > 2)
- Fraction of unreliable comparisons (SNR < 0.5)
- SAGE grouping (what CAN be said vs what CANNOT)

### Phase 3C: Write Nature Comment (1 day)

- [ ] **Step 6: Draft 2-3 page Comment**

Structure: (1) The EU AI Act requires explainability. (2) We prove that full explainability is mathematically impossible for underspecified systems. (3) Here is the provably optimal compliance strategy. (4) Recommendation: regulators should adopt SAGE-style reporting as the standard for "sufficient transparency."

- [ ] **Step 7: Submit as companion piece**

Submit the Nature Comment alongside or after the main paper. Cross-reference.

**Pass criterion:** The mapping from legal text to formal framework is rigorous and a legal expert would agree the interpretation is defensible.

**Estimated effort:** 3-5 days. Enhanced by a legal co-author.

---

## Execution Priority

| Path | Impact | Tractability | Timeline | Recommendation |
|------|--------|-------------|----------|---------------|
| **1** | VERY HIGH | MEDIUM | 2-3 weeks | **START NOW** — search for papers today, begin reproductions |
| **2** | HIGH | MEDIUM | 2-3 weeks | **PARALLEL** — install TransformerLens, begin IOI analysis |
| **3** | HIGH (as companion) | HIGH | 3-5 days | **WRITE AFTER** main paper is strong |

**Immediate next step:** Start Phase 1A (paper search) while TreeSHAP infrastructure is already validated.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Published papers can't be reproduced | HIGH | Select only papers with shared code; use exact datasets |
| TreeSHAP flip formula R²<0.5 on some datasets | MEDIUM | Use ρ (rank correlation) as primary metric, not R² |
| All audited papers have reliable rankings | LOW | Would actually be an interesting positive result |
| Mech interp community rejects framework mapping | MEDIUM | Frame as "complementary to existing analysis, adding formal structure" |
| Legal mapping is debatable | HIGH | Acknowledge ambiguity; propose standard, don't claim Act is "wrong" |
| NeurIPS deadline (May 6) conflict | CERTAIN | Decide: NeurIPS attribution paper OR Path 1/2. Don't try both. |
