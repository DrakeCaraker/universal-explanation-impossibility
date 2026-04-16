# Definitive MI Experiment v2: Circuits From Scratch

## Why v1 Was Inconclusive

The LoRA experiment (v1) found circuit stability (ρ=0.885), but this was
confounded by three factors:

1. **LoRA rank 4** constrains changes to a 4-dimensional subspace of the
   attention projection (0.7% of parameters). 99.3% of weights are shared
   across all models. Stability is expected, not informative.

2. **All 30 models achieved 100% accuracy** on IOI. The Rashomon filter was
   vacuous — no model was excluded. There was no accuracy-circuit tradeoff
   to exploit.

3. **GPT-2 pretrained already solves IOI.** Fine-tuning polishes an existing
   circuit rather than creating a new one. Circuit stability may reflect
   pretrained structure, not a property of the task.

## Design: Modular Addition on Small Transformers

### Why This Task

Modular addition (a + b mod p) on small transformers is the canonical
benchmark for mechanistic interpretability (Nanda et al., "Progress Measures
for Grokking Generalisation," ICLR 2023). The key properties:

1. **No pretrained solution.** Models train from random initialization.
   Every circuit must be learned from scratch.
2. **Well-characterised circuit.** Post-grokking, models learn a Fourier
   multiplication algorithm. The circuit structure is known: specific heads
   implement cosine/sine computations at specific frequencies.
3. **Grokking creates sharp convergence.** Models transition from memorisation
   to generalisation, reaching ~99% test accuracy. This creates a natural
   Rashomon set: multiple models that generalise correctly but potentially
   via different frequency assignments.
4. **Fast training.** A 1-layer, 4-head transformer trains in ~2 minutes on
   GPU. 30 models = 1 hour.

### Architecture

- **Model:** 1-layer transformer, d_model=128, 4 attention heads (head_dim=32)
- **Why 1 layer:** The grokking circuit for modular addition is known to
  fit in 1 layer. More layers add redundancy without new circuit structure.
- **Components:** 4 heads = 4 components to compare (simple, interpretable)
- **Embedding:** Learned token embeddings for 0..p-1 (p=113)
- **Task:** Given tokens (a, b), predict (a + b) mod 113

### Training

- 30 models from different random seeds (seeds 0-29)
- Optimizer: AdamW, lr=1e-3, weight_decay=1.0 (standard grokking setup)
- Training data: 50% of all (a,b) pairs = 6,328 examples
- Test data: remaining 50% = 6,327 examples
- Train for 50,000 steps (well past grokking, which typically occurs at
  ~10,000 steps for this configuration)
- Log test accuracy every 1,000 steps to verify grokking

### Rashomon Filter

- **Threshold:** Test accuracy within 1% of best
- **Expected:** All or most models should grok (reach >98%), but
  the KEY question is whether they grok via the SAME circuit

### Measurement: Activation Patching

For each head in each model:
1. Run model on clean input (a, b)
2. Run model on corrupted input (a', b) where a' is random
3. Patch: replace head's output from clean run with corrupted run
4. Measure: change in logit for correct answer (a+b mod p)
5. Importance = |Δ logit|

### Additional Measurements

1. **Fourier analysis of embeddings.** Nanda et al. showed that post-grokking
   embeddings contain cosine/sine components at specific frequencies. Extract
   the dominant frequencies per model and compare.

2. **Weight-space cosine similarity.** Direct comparison of head weight
   matrices (W_Q, W_K, W_V) up to permutation alignment (Hungarian method).

3. **Functional equivalence classes.** For each head, compute its response
   to all (a,b) pairs. Cluster heads by functional similarity across models.

### Controls

A. **Pre-grokking snapshot.** Save model at step 5,000 (before grokking).
   Circuits should be random/memorisation-based. Expect LOW stability.
   This calibrates the measurement: if pre-grokking shows ρ=0.9, the
   tool can't distinguish circuits.

B. **Determinism.** Same seed, same data → identical model. Expect ρ=1.0.

C. **Different task.** Train on a DIFFERENT modular operation (a × b mod p).
   These models should have DIFFERENT circuits from addition models.
   Expect LOW cross-task stability.

### Three Possible Outcomes

| Outcome | Criterion | Interpretation |
|---------|-----------|----------------|
| **Same circuit** | ρ > 0.8, same dominant frequencies | The loss landscape has a unique basin for modular addition. Circuits are determined by the task, not the seed. MI is reliable for this task. |
| **Different circuits** | ρ < 0.3, different frequencies | Multiple circuit solutions exist. MI faces genuine Rashomon: different equally-valid decompositions. The impossibility theorem is empirically confirmed. |
| **Same algorithm, different heads** | Frequencies match but head assignments differ | The FUNCTION is stable but the HEAD ASSIGNMENT is unstable (permutation symmetry). This is the S_n prediction: circuits are unique up to permutation. |

ALL THREE OUTCOMES ARE INFORMATIVE. Outcome 3 is particularly interesting:
it would mean the "circuit" is an equivalence class (like DASH reports ties),
validating the orbit-averaging resolution.

### Why This Is Definitive

| Confound | v1 (LoRA) | v2 (from scratch) |
|----------|-----------|-------------------|
| Shared weights | 99.3% shared | 0% shared |
| Task difficulty | 100% accuracy (trivial) | ~99% after grokking (learned) |
| Pretrained circuit | Yes (GPT-2 knows IOI) | No (random init) |
| Measurement tool | Activation patching ✓ | Activation patching ✓ |
| Ground truth | Wang et al. IOI circuit | Nanda et al. Fourier circuit |

### Runtime

- Training: 30 models × ~2 min = 1 hour
- Activation patching: 4 heads × 12,655 examples × 30 models = ~1 hour
- Fourier analysis: ~10 minutes
- Controls: ~1 hour
- **Total: ~3 hours on T4 GPU**

### Connection to the Impossibility Theorem

- Θ = set of trained transformer weight vectors (different seeds)
- Y = test accuracy (observe)
- H = head importance vectors ∈ R^4 (explain)
- incompatible = different pairwise ranking for a head pair
- Rashomon = multiple models achieve same accuracy via different circuits

The trilemma predicts: if circuits genuinely differ (Rashomon holds),
no circuit description can be simultaneously faithful (report each model's
circuit), stable (same across Rashomon set), and decisive (rank all heads).

If Outcome 1 (same circuit): Rashomon doesn't hold for circuits → the
impossibility's precondition is unmet → report as honest negative.

If Outcome 2 (different circuits): Rashomon confirmed → impossibility
empirically demonstrated for MI → orbit averaging (report equivalence
class of circuits) is the resolution.

If Outcome 3 (same function, different heads): Rashomon holds for head
assignment but not for function → the symmetry group is S_n (permutation
of heads) → orbit averaging over S_n gives the stable description.
