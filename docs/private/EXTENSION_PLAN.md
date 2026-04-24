# Universal Explanation Impossibility — Extension Plan

**Goal**: Add GradCAM (saliency map) and LLM self-explanation experiments,
integrate as new instances, update all paper versions, and finalize for
NeurIPS 2026 submission.

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Current state**: 68 Lean files, 336 theorems, 60 axioms, 0 sorry.
4 experiments with negative controls + resolution tests.
Paper versions: base (23pp), monograph (39pp), JMLR (30pp), NeurIPS (10pp).
Definitions corrected (genuine trilemma with tightness).

**Timeline**: NeurIPS abstract May 4, paper May 6.
**Model assignment**: Opus for proofs/theory/writing, Sonnet for experiments/scaffolding.

**Rules**:
- Zero sorry without `-- TODO:` comment
- `set_option autoImplicit false` in all Lean files
- Run `lake build` after every Lean change
- Every experiment has: positive test, negative control, resolution test, 95% CIs
- No parallel agents modifying the same file

---

## Phase 1: GradCAM / Saliency Map Experiment [Days 1-2]

### Task 1.1: Saliency Map Instance in Lean [Sonnet]

**Files**:
- Create: `UniversalImpossibility/SaliencyInstance.lean`
- Modify: `UniversalImpossibility/Basic.lean` (add import)

The saliency map instance of the universal impossibility:
- Θ = CNN weight configurations
- H = ℝ^(W×H) (spatial heatmaps — per-pixel importance)
- Y = class predictions
- observe(θ) = the classification function f_θ
- explain(θ) = GradCAM heatmap for a given input
- incompatible(h₁, h₂) = argmax regions don't overlap (peak importance in different spatial locations)
- Rashomon: equivalent CNNs produce different GradCAM heatmaps

Follow the same pattern as `AttentionInstance.lean`:
- Axiomatize `SaliencyConfig`, `SaliencyMap`, `SaliencyObservable`
- Axiomatize `saliencySystem : ExplanationSystem SaliencyConfig SaliencyMap SaliencyObservable`
- Prove `saliency_impossibility` by applying `explanation_impossibility`

Run `lake build`. Commit.

---

### Task 1.2: GradCAM Experiment Script [Sonnet]

**File**: `paper/scripts/gradcam_instability_experiment.py`
**Depends on**: `paper/scripts/experiment_utils.py`

#### Design

**Research question**: Do functionally equivalent CNNs highlight different
image regions as important?

**Dataset**: CIFAR-10 via torchvision (50,000 train, 10,000 test).
Use 10,000 training images for speed (sufficient for ResNet-18 to reach >90%).

**Models**: 10 ResNet-18 models trained from different random initializations:
- `torchvision.models.resnet18(weights=None)` — train from scratch
- SGD, lr=0.01, momentum=0.9, 10 epochs, batch_size=128
- Different seeds (42+i for i in range(10))
- Verify all achieve >88% test accuracy (confirming functional equivalence)
- If training is too slow, FALLBACK: load pretrained ResNet-18 and create
  10 perturbed copies (same approach as attention experiment, σ=0.02).
  Verify >90% prediction agreement.

**GradCAM implementation** (self-contained, no external library):
```python
def gradcam(model, input_tensor, target_class, target_layer):
    """Compute GradCAM heatmap for a given input and target class."""
    # Register hooks on target_layer to capture activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    target_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    target_score.backward()

    # GradCAM: weight activations by mean gradient
    weights = gradients['value'].mean(dim=[2, 3], keepdim=True)  # GAP
    cam = (weights * activations['value']).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    cam = cam / (cam.max() + 1e-8)  # normalize to [0, 1]

    handle_fwd.remove()
    handle_bwd.remove()
    return cam  # shape: (H, W), values in [0, 1]
```

Target layer: `model.layer4[-1]` (last conv block of ResNet-18).

**What to measure** (100 test images, correctly classified by ALL 10 models):

1. **Spatial overlap (IoU)**: For each model pair and image, threshold
   GradCAM at top-20% intensity. Compute IoU of the highlighted regions.
   Report mean pairwise IoU across all (model, image) pairs.
   Low IoU = high instability.

2. **Peak location flip rate**: For each image, find the (row, col) of
   the maximum GradCAM activation. Measure fraction of model pairs where
   the peak location differs by more than 2 pixels.

3. **Rank correlation of pixel importance**: Kendall tau between flattened
   GradCAM heatmaps for model pairs.

4. **Prediction agreement**: Fraction of test images where all 10 models
   agree on the predicted class.

5. **95% bootstrap CIs** on all metrics.

**Negative control**: Same 10 models, but evaluate on images from a
"trivial" class where the salient object fills the entire frame (or use
solid-color images where GradCAM should be uniform). ALTERNATIVE: use
identical models (same weights) — GradCAM should be deterministic,
IoU = 1.0.

**Resolution test**: Average GradCAM heatmaps across all 10 models.
Compute IoU between each individual model's heatmap and the averaged
heatmap. Compare: mean(IoU with average) vs mean(pairwise IoU).
The average should be a better consensus (higher IoU with individuals).

**Figure**: `paper/figures/gradcam_instability.pdf`
- 3-panel figure:
  - Left: Grid of GradCAM heatmaps for 3 images × 5 models. Show the
    original image with heatmap overlay. Same prediction, different highlights.
    This is the "money shot" — visually compelling.
  - Middle: Distribution of pairwise IoU values (histogram). Vertical line
    at mean. Compare to negative control distribution.
  - Right: Bar chart comparing positive/control/resolution metrics.

**Table**: `paper/sections/table_gradcam.tex`

**Expected results**: Mean IoU < 0.5 (substantial spatial disagreement).
Peak flip rate > 30%. Prediction agreement > 90%. Negative control IoU > 0.95.
Resolution IoU > pairwise IoU.

**Output files**:
- `paper/results_gradcam_instability.json`
- `paper/figures/gradcam_instability.pdf`
- `paper/sections/table_gradcam.tex`

---

### Task 1.3: GradCAM Proof Sketch [Opus]

**File**: `docs/proof-sketches/gradcam-impossibility.md`

Write the proof sketch following the same structure as the other instances:
- Setup: Θ, H, Y, observe, explain
- Rashomon property: cite Adebayo et al. (2018) "Sanity Checks for
  Saliency Maps" (saliency maps can be insensitive to model parameters),
  Hooker et al. (2019) "A Benchmark for Interpretability Methods"
- Incompatibility: different peak regions
- Proof: 4-step standard structure
- Verdict: GO

---

## Phase 2: LLM Self-Explanation Experiment [Days 1-2, parallel with Phase 1]

### Task 2.1: LLM Explanation Instance in Lean [Sonnet]

**Files**:
- Create: `UniversalImpossibility/LLMExplanationInstance.lean`
- Modify: `UniversalImpossibility/Basic.lean` (add import)

The LLM self-explanation instance:
- Θ = LLM weight configurations (different training runs, fine-tuning seeds,
  or temperature-sampled generation paths)
- H = natural language explanations (text strings as explanation space)
- Y = output predictions (classification labels or final answers)
- observe(θ) = the model's answer to a question
- explain(θ) = the model's stated reasoning / chain of thought
- incompatible(h₁, h₂) = the explanations cite different evidence or
  give contradictory reasoning for the same answer
- Rashomon: same answer, different reasoning chains

Follow the same Lean pattern:
- Axiomatize `LLMConfig`, `LLMExplanation`, `LLMObservable`
- Axiomatize `llmSystem : ExplanationSystem LLMConfig LLMExplanation LLMObservable`
- Prove `llm_explanation_impossibility` by applying `explanation_impossibility`

Run `lake build`. Commit.

---

### Task 2.2: LLM Self-Explanation Experiment Script [Sonnet]

**File**: `paper/scripts/llm_explanation_instability_experiment.py`

#### Design

**Research question**: Do functionally equivalent LLMs give contradictory
explanations for the same answer?

**Approach**: Use a locally-runnable model via the transformers library.

**Primary approach — Weight perturbation on text classifier + generated explanation**:

Use DistilBERT (already loaded in the attention experiment) for classification.
Use a small text-generation model for explanation.

Actually, cleaner approach:

**Primary approach — Multiple prompted explanations from a single model**:

1. Load a small instruction-tuned model. Check availability in order:
   - `microsoft/phi-2` (2.7B) — if memory allows
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B)
   - `distilgpt2` (82M) — fallback, less coherent but runs anywhere

2. Construct 50 sentiment classification inputs:
   ```
   "Review: {review_text}\nIs this review positive or negative? First explain your reasoning step by step, then give your answer."
   ```

3. For each input, generate 10 responses with temperature=0.7, top_p=0.9.

4. Parse each response to extract:
   - The final answer (positive/negative)
   - The reasoning text (everything before the answer)

5. Filter to inputs where ALL 10 generations agree on the answer.

6. Among those, measure explanation consistency:
   - **Keyword overlap**: Jaccard similarity of content words in reasoning
   - **Evidence citation**: Which words from the original review are mentioned
     in the explanation? Compute overlap of cited evidence across generations.
   - **Reasoning structure**: Do explanations cite the same features as
     important? (e.g., "excellent" vs "recommend" vs "enjoyed")

**FALLBACK if no generative model runs locally**:

Use the perturbation approach from the attention experiment, but measure
explanation disagreement differently:

1. Same 10 perturbed DistilBERT models from the attention experiment
2. For each model + input, rank tokens by attention rollout importance
3. Generate a template explanation: "The model classified this as {label}
   primarily because of the words: {top-3 tokens by attention}"
4. Measure: how often do different models cite different top-3 tokens?
   This is the token-citation flip rate.
5. This is essentially the attention flip rate repackaged as an
   "explanation disagreement" metric — which is exactly the point:
   attention instability IS explanation instability.

**Negative control**: Use inputs where the sentiment is unambiguous AND
the model is highly confident (logit difference > 3.0). On these inputs,
the explanation should be stable because there's only one reasonable
evidence path. Expected keyword overlap: > 0.7.

**Resolution test**: Compute the "consensus explanation" — the set of
words cited by >50% of generations. Measure: what fraction of individual
explanations are consistent with the consensus? Expected: > 80%.

**Figure**: `paper/figures/llm_explanation_instability.pdf`
- 2-panel figure:
  - Left: Example showing 3 different explanations for the same input
    and same answer, with highlighted evidence words (different colors
    per generation). Visually show the divergence.
  - Right: Distribution of keyword Jaccard similarity across all
    (input, generation-pair) combinations. Compare positive vs control.

**Table**: `paper/sections/table_llm_explanation.tex`

**Expected results**: Mean keyword Jaccard similarity < 0.5 for positive
test, > 0.7 for control. Evidence citation overlap < 0.6.

**Output files**:
- `paper/results_llm_explanation_instability.json`
- `paper/figures/llm_explanation_instability.pdf`
- `paper/sections/table_llm_explanation.tex`

---

### Task 2.3: LLM Explanation Proof Sketch [Opus]

**File**: `docs/proof-sketches/llm-explanation-impossibility.md`

Setup:
- Θ = LLM training configurations (initialization seed, fine-tuning data
  order, RLHF reward model seed)
- Y = output answers (classification, QA, etc.)
- H = natural language reasoning chains
- Rashomon property: multiple training runs converge to models that give
  the same answers but different internal representations, producing
  different reasoning chains when asked to explain
- Cite: Turpin et al. (2024) "Language Models Don't Always Say What They
  Think" — chain-of-thought explanations are unfaithful to the actual
  reasoning process. Lanham et al. (2023) "Measuring Faithfulness in
  Chain-of-Thought Reasoning." Ye & Durrett (2022) "The Unreliability
  of Explanations in Few-Shot Prompting."
- Incompatibility: explanations cite different evidence or give
  contradictory causal reasoning for the same answer

---

## Phase 3: Paper Integration [Days 2-3]

### Task 3.1: Write GradCAM Instance Section [Opus]

**File**: `paper/sections/instance_gradcam.tex`

Write a ~1 page subsection following the same structure as the other instances:
- Define the 6-tuple for saliency maps
- State the Rashomon property (cite Adebayo et al. 2018, Hooker et al. 2019)
- Define incompatibility (spatial IoU below threshold)
- State the impossibility as an instance of Theorem 1
- Add "Empirical validation" paragraph with experiment results
- Include figure and table references

### Task 3.2: Write LLM Explanation Instance Section [Opus]

**File**: `paper/sections/instance_llm_explanation.tex`

Write a ~1 page subsection:
- Define the 6-tuple for LLM self-explanations
- State the Rashomon property (cite Turpin et al. 2024, Lanham et al. 2023)
- Define incompatibility (contradictory evidence citation)
- State the impossibility
- Add empirical validation paragraph
- Discuss implications: chain-of-thought "explanations" from LLMs are
  subject to the same impossibility — they cannot simultaneously reflect
  the model's actual reasoning (faithful), be consistent across equivalent
  models (stable), and commit to specific causal claims (decisive)

### Task 3.3: Update Cross-Instance Summary Table [Opus]

Update `tab:cross-instance` in `paper/universal_impossibility.tex` to add:
```
GradCAM (ResNet-18) & Spatial IoU & X.XX & $>$90\% \\
LLM explanation & Keyword Jaccard & X.XX & $>$XX\% \\
```

### Task 3.4: Update All Paper Versions [Sonnet]

For each of the 4 paper files:
1. Add `\input{sections/instance_gradcam}` and
   `\input{sections/instance_llm_explanation}` to the Six Instances section
2. Update the abstract: "eight explanation types" (was six)
3. Update the introduction: add GradCAM and LLM to the contribution list
4. Update the Lean formalization table: new file count and theorem count
5. Update the ubiquity section if needed
6. Compile and verify

### Task 3.5: Add Missing Bibliography Entries [Sonnet]

Add to `paper/references.bib`:
- `adebayo2018sanity` — Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018
- `hooker2019benchmark` — Hooker et al., "A Benchmark for Interpretability Methods in Deep Neural Networks," NeurIPS 2019
- `selvaraju2017gradcam` — Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017
- `turpin2024language` — Turpin et al., "Language Models Don't Always Say What They Think," NeurIPS 2023
- `lanham2023measuring` — Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning," 2023
- `ye2022unreliability` — Ye & Durrett, "The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning," NeurIPS 2022

### Task 3.6: Update CLAUDE.md [Sonnet]

Update Lean counts, file counts, instance inventory, and paper structure
documentation.

---

## Phase 4: Verification [Day 3]

### Task 4.1: Lean Consistency Check [Sonnet]

```bash
lake build
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

All counts must match paper text. Sorry must be 0.

### Task 4.2: Paper Compilation Check [Sonnet]

Compile all 4 versions. All must succeed. Verify page counts:
- Base: ~25pp
- Monograph: ~42pp
- JMLR: ~33pp
- NeurIPS: ~10pp (main) + supplement

### Task 4.3: Experiment Reproducibility Check [Sonnet]

Run `python paper/scripts/run_all_universal_experiments.py` — but first
update it to include the two new experiments. All must complete without error.

### Task 4.4: Update Mock Reviews [Opus]

Add a 6th mock review:

**Reviewer 6 — The LLM Skeptic**:
"The LLM experiment is a stretch. Temperature sampling is artificial
randomness, not genuine model multiplicity. Chain-of-thought
explanations are known to be unfaithful (Turpin et al.). This isn't
a new finding."

Response must cite: the theorem applies regardless of the SOURCE of
multiplicity (temperature, retraining, architecture), Turpin et al.
confirms the Rashomon property (different reasoning for same answer),
and the contribution is showing this is an INSTANCE of the universal
impossibility, not a standalone observation.

---

## Phase 5: Final Polish [Day 3-4]

### Task 5.1: Update the Experiment Section Prose [Opus]

In the NeurIPS version, the empirical section needs to cover 6 experiments
in ~1.5 pages. Structure:

- 1 paragraph intro (6 experiments, 4 model types, all with controls)
- Cross-instance summary table (the money table)
- 1 paragraph per experiment (2-3 sentences: setup, key result, control)
- 1 paragraph on resolution (aggregation improves stability in all cases)

Move detailed methodology to supplement.

### Task 5.2: Create NeurIPS Supplement for New Experiments [Sonnet]

Update `paper/universal_impossibility_neurips_supplement.tex` to include
GradCAM and LLM experiment details.

### Task 5.3: Final Commit and Push [Sonnet]

```bash
cd ~/ds_projects/universal-explanation-impossibility
lake build  # verify
git add -A
git commit -m "feat: 8 instances — add GradCAM + LLM self-explanation

Two new instances of the universal impossibility:
7. GradCAM saliency maps (ResNet-18, CIFAR-10)
8. LLM self-explanations (temperature-sampled reasoning)

Both with positive tests, negative controls, resolution tests.
New Lean instances compile (0 sorry).
All paper versions updated and compile.

8 instances, 1 abstract theorem, 6 experiments with controls."
git push
```

---

## Model Assignment Summary

| Task | Model | Rationale |
|------|-------|-----------|
| 1.1 Saliency Lean instance | Sonnet | Mechanical (same pattern) |
| 1.2 GradCAM experiment | Sonnet | Python scripting |
| 1.3 GradCAM proof sketch | Opus | Theoretical reasoning |
| 2.1 LLM Lean instance | Sonnet | Mechanical |
| 2.2 LLM experiment | Sonnet | Python scripting |
| 2.3 LLM proof sketch | Opus | Theoretical reasoning |
| 3.1 GradCAM paper section | Opus | Mathematical exposition |
| 3.2 LLM paper section | Opus | Mathematical exposition |
| 3.3-3.6 Integration | Sonnet | Mechanical updates |
| 4.x Verification | Sonnet | Scripted checks |
| 4.4 Mock review update | Opus | Argument construction |
| 5.1 Experiment prose | Opus | Writing |
| 5.2-5.3 Final polish | Sonnet | Mechanical |

---

## Execution Order

```
Phase 1: [1.1] → [1.2 + 1.3 parallel]
Phase 2: [2.1] → [2.2 + 2.3 parallel]  (Phase 2 parallel with Phase 1)
Phase 3: [3.1 ∥ 3.2] → [3.3] → [3.4] → [3.5 ∥ 3.6]
Phase 4: [4.1 ∥ 4.2 ∥ 4.3] → [4.4]
Phase 5: [5.1] → [5.2] → [5.3]
```

Phases 1 and 2 are independent and run in parallel.
Phase 3 depends on experiment results from 1+2.
Phase 4 is verification. Phase 5 is final polish.

---

## Confidence Ratings

| Component | Confidence | Risk | Mitigation |
|-----------|-----------|------|------------|
| GradCAM experiment | HIGH | Training ResNet-18 may be slow | Fallback: perturbation approach |
| GradCAM Lean instance | HIGH | Same pattern as 6 others | Mechanical |
| LLM experiment (generative) | MEDIUM | Local model may not generate coherent explanations | Fallback: attention-based token citation |
| LLM experiment (fallback) | HIGH | Reuses existing attention infrastructure | Already validated |
| LLM Lean instance | HIGH | Same pattern | Mechanical |
| Paper integration | HIGH | Straightforward additions | Follow existing structure |
| NeurIPS page budget | MEDIUM | 8 instances in 10 pages is tight | Move 2 new instances to supplement |

---

## Key Decision: NeurIPS Page Budget

Adding 2 more instances to a 10-page paper is tight. Options:

**A)** Keep all 8 in the main paper (0.4 pages each instead of 0.5)
**B)** Keep 6 in main paper, move GradCAM + LLM to supplement
**C)** Keep all 8 but compress to a single "instance table" format
    (no individual subsections, just a comprehensive table)

Recommendation: **C** for NeurIPS (table-only in main, full sections in
supplement). **A** for JMLR/monograph (no page limit).

---

## Open Questions

1. Does Drake have GPU access for ResNet-18 training on CIFAR-10?
   (~30 min on GPU, ~3 hours on CPU). Perturbation fallback if not.

2. Which local LLM is available? Check:
   `python -c "from transformers import AutoModelForCausalLM; print('OK')"`
   Memory requirement: TinyLlama needs ~4GB, phi-2 needs ~8GB.

3. Should LLM experiment use temperature sampling (simpler, less rigorous)
   or weight perturbation (more rigorous, matches other experiments)?
   Recommendation: perturbation for rigor, mention temperature in discussion.

4. The paper now claims "8 explanation types" — is that too many for a
   clean narrative? Or does the breadth strengthen the universality claim?
   Recommendation: breadth IS the point. 8 > 6 strengthens the claim.
