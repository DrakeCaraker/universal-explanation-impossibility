# Definitive MI Rashomon Experiment: Activation Patching on IOI

## The Question

Do independently trained transformers learn the same circuits for the same task,
or do functionally equivalent models decompose computation differently?

## Why the Current Experiment Failed to Answer This

Zero ablation measures `importance = Δaccuracy(head removed)`. This conflates:
- What the head computes (circuit contribution)
- How the classifier reads the head (readout weight)

Control A (same transformer, different classifier) showed flip rate ≈ main experiment,
proving the classifier readout term dominates. We need a measurement that bypasses
the classifier entirely.

## Design: Activation Patching on Indirect Object Identification

### Why Activation Patching

Activation patching (Vig et al. 2020, Wang et al. 2023) measures a head's
*causal contribution* to the output by intervening on the residual stream:

1. Run model on clean input: "When Mary and John went to the store, John gave a drink to"
2. Run model on corrupted input: names scrambled → "When Alice and Bob went to the store, Bob gave a drink to"
3. For each head: replace clean activation with corrupted activation
4. Measure: change in logit(Mary) - logit(John)

This measures what each head COMPUTES — not how a classifier reads it. The
logit difference is computed through the model's own unembedding matrix (W_U),
which is shared across all fine-tunes from the same pretrained base. No
classifier head is involved.

### Why IOI (Indirect Object Identification)

- Best-characterized circuit in MI literature (Wang et al. 2023)
- Known circuit components: duplicate token heads → S-inhibition heads → name mover heads
- Specific heads identified: L9H9, L9H6, L10H0 (name movers in GPT-2 small)
- Provides ground-truth validation: if our method finds different heads than
  Wang et al., our method is wrong, not the circuit

### Models

- Architecture: GPT-2 small (12 layers × 12 heads = 144 components)
- N = 30 models, fine-tuned from pretrained GPT-2 on IOI task
  - Training data: 10,000 synthetic IOI sentences
  - Fine-tuning: LoRA (rank 4) on attention weights only — keeps pretrained
    structure while allowing circuit reorganization
  - Early stopping on validation loss (patience = 3)
  - Different seeds only (architecture, data identical)
- Rashomon filter: accuracy within 2% of best on held-out IOI test set (500 sentences)

### Why LoRA, Not Full Fine-Tuning

Full fine-tuning (as in the current experiment) changes the entire model,
including the unembedding matrix W_U. This means even the logit-based
measurement has a "readout noise" component. LoRA constrains changes to
low-rank attention perturbations, preserving W_U exactly. This means:
- Logit attribution is through the SAME unembedding matrix for all models
- Differences in head importance are PURELY from attention circuit changes
- The "classifier noise" confound is eliminated by construction

### Measurements (Per Model)

For each of 144 heads, compute:

1. **Activation patching importance**: Δ(logit_IO - logit_S) when head is patched
   (clean → corrupted). Positive = head moves information toward correct answer.

2. **Attention pattern**: For each IOI test sentence, extract the attention
   matrix. Compute mean attention from final token position to IO name position.

3. **Circuit role classification**: Based on patching importance sign and
   attention pattern, classify each head as:
   - Name mover (positive importance, attends IO → final)
   - S-inhibition (negative importance at S position, attends duplicate → S)
   - Duplicate token (attends IO → IO duplicate)
   - Irrelevant (|importance| < threshold)

### Controls

**Control A — Measurement validation:**
Run activation patching on the BASE (pretrained, unfine-tuned) GPT-2 small.
Wang et al.'s known circuit should be recovered. If not, our measurement is wrong.

**Control B — Same-seed replication:**
Run the same model twice with identical seed. Patching importance should be
identical (flip rate = 0%). This validates the measurement is deterministic.

**Control C — Maximal perturbation:**
Compare GPT-2 small fine-tuned on IOI vs. fine-tuned on a DIFFERENT task
(e.g., sentiment). These should have maximally different circuits. If flip
rate is low even here, the measurement lacks sensitivity.

### Analysis (Pre-Registered)

**Primary metric:** Spearman ρ between head importance vectors across all
Rashomon model pairs. This is rank-order correlation, robust to scale differences.

**Secondary metrics:**
1. Jaccard similarity on top-K important heads (K = 10, 20, 30)
2. Circuit role agreement: fraction of heads classified the same way across models
3. Name mover identity agreement: are L9H9, L9H6, L10H0 always name movers?

**Quantitative prediction test:**
Coverage conflict (from the framework) should predict flip rate.
Compute Spearman ρ between CC and observed flip rate.

**Three possible outcomes with interpretation:**

| Outcome | Criterion | Interpretation |
|---------|-----------|----------------|
| **Circuits stable** | ρ > 0.8, Jaccard(top-20) > 0.6 | Rashomon holds for predictions but NOT for circuits. The impossibility applies to the model-to-circuit mapping but circuits themselves are conserved. Report as: "MI may be more stable than other explanation types — the symmetry group is smaller." |
| **Circuits non-unique** | ρ < 0.3, Jaccard(top-20) < 0.15 | Genuine Rashomon for circuits. Different training runs find genuinely different computational pathways. Report as: "The impossibility theorem is empirically confirmed for MI." |
| **Partial stability** | 0.3 < ρ < 0.8 | Some circuit components conserved ("skeleton"), others vary ("flesh"). Report as: "A circuit skeleton exists but is not the full circuit. Safety claims should be qualified to skeleton-level only." |

**ALL THREE OUTCOMES ARE PUBLISHABLE.** This is what makes the experiment definitive.

### Power Analysis

- 144 heads per model
- C(17,2) = 136 Rashomon model pairs (assuming 17/30 pass filter)
- 144 × 136 = 19,584 head-pair comparisons
- At α = 0.05, power > 0.99 to detect ρ difference of 0.1
- Minimum detectable Jaccard difference: 0.05

### Computational Requirements

- 30 LoRA fine-tunes: ~2 min each on T4 GPU = 1 hour
- Activation patching: 144 heads × 500 sentences × 17 models = ~3 hours
- Controls: ~1 hour
- Total: ~5 hours on ml.g4dn.xlarge (vs. 6 hours for current zero-ablation experiment)

### What This Tells Us That Zero Ablation Cannot

| Property | Zero Ablation | Activation Patching |
|----------|--------------|-------------------|
| Measures | Accuracy change | Logit change |
| Classifier involved | Yes (dominates) | No (uses W_U directly) |
| Causal | Deletion (crude) | Intervention (precise) |
| Composition | Cannot detect | Detects via path patching |
| Known validation | None | Wang et al. 2023 IOI circuit |

### Connection to the Impossibility Theorem

The experiment maps directly onto the ExplanationSystem:
- Θ = set of LoRA-fine-tuned GPT-2 models (different seeds)
- Y = IOI accuracy (observe)
- H = head importance vectors in R^144 (explain)
- incompatible = different pairwise ranking for a head pair
- Rashomon = multiple models achieve same accuracy with different importance rankings

The trilemma predicts: no explanation method can be simultaneously faithful
(report each model's actual circuit), stable (same circuit across Rashomon models),
and decisive (commit to a single circuit ranking). Activation patching is a
faithful method (it measures causal contribution). The experiment tests whether
it is also stable.
