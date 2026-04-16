#!/usr/bin/env python3
"""
Definitive MI Rashomon Experiment: Activation Patching on IOI

Eliminates the classifier-noise confound from the zero-ablation experiment
by using activation patching through the model's own unembedding matrix.

Design:
- 30 GPT-2 small models, LoRA fine-tuned on IOI from different seeds
- Activation patching (not zero ablation) measures causal head contribution
- LoRA preserves W_U exactly → no readout noise
- IOI task has known ground-truth circuit (Wang et al. 2023) for validation
- Three controls: measurement validation, determinism, sensitivity

Run: python mech_interp_definitive.py
Expected: ~5 hours on ml.g4dn.xlarge T4 GPU
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, random, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from itertools import combinations

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path(__file__).resolve().parent
N_MODELS = 30
N_LAYERS = 12
N_HEADS = 12
N_COMPONENTS = N_LAYERS * N_HEADS  # 144
RASHOMON_THRESHOLD = 0.02  # 2% accuracy
N_IOI_TRAIN = 10000
N_IOI_VAL = 500
N_IOI_TEST = 500
N_PATCHING_EXAMPLES = 200  # sentences for patching
LORA_RANK = 4
MAX_EPOCHS = 10
PATIENCE = 3
MODEL_DIR = '/tmp/mi_definitive_models'

# =========================================================================
# IOI Data Generation
# =========================================================================

NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah", "James", "Emma", "David",
    "Lisa", "Tom", "Kate", "Mike", "Anna", "Chris", "Laura", "Steve",
    "Diana", "Peter", "Helen", "Mark", "Grace", "Paul", "Susan", "Jack",
    "Amy", "Brian", "Carol", "Dan", "Emily", "Frank"
]

PLACES = [
    "the store", "the park", "the beach", "the office", "the library",
    "the cafe", "the gym", "the museum", "the airport", "the station"
]

OBJECTS = [
    "a drink", "a book", "a gift", "a letter", "a ticket",
    "a key", "a phone", "a bag", "a hat", "a coat"
]

TEMPLATES = [
    "When {IO} and {S} went to {place}, {S} gave {object} to",
    "After {IO} and {S} arrived at {place}, {S} handed {object} to",
    "Once {IO} and {S} were at {place}, {S} passed {object} to",
]


def generate_ioi_dataset(n, seed=0):
    """Generate IOI (Indirect Object Identification) sentences.

    Returns list of dicts with:
    - text: the prompt (answer should be IO name)
    - io_name: the indirect object (correct answer)
    - s_name: the subject (wrong answer)
    - corrupted_text: same sentence with IO and S names swapped
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        io, s = rng.sample(NAMES, 2)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        template = rng.choice(TEMPLATES)

        text = template.format(IO=io, S=s, place=place, object=obj)
        # Corrupted: swap the names
        corrupted = template.format(IO=s, S=io, place=place, object=obj)

        data.append({
            'text': text,
            'io_name': io,
            's_name': s,
            'corrupted_text': corrupted,
        })
    return data


# =========================================================================
# LoRA Implementation (minimal, no dependencies)
# =========================================================================

class LoRALinear(nn.Module):
    """Low-rank adaptation of a linear layer (supports both nn.Linear and Conv1D)."""
    def __init__(self, original_layer, rank=4):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # GPT-2 uses Conv1D (weight shape: [out, in]) instead of nn.Linear
        # Conv1D stores weight as (nf, nx) where nx=in_features, nf=out_features
        if hasattr(original_layer, 'nf'):
            # transformers Conv1D: weight is (in_features, out_features)
            in_features = original_layer.weight.shape[0]
            out_features = original_layer.nf
        else:
            in_features = original_layer.in_features
            out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T)


def apply_lora(model, rank=4):
    """Apply LoRA to all attention Q, K, V projections."""
    lora_params = []
    for layer in model.transformer.h:
        # Replace c_attn (combined QKV projection) with LoRA version
        original = layer.attn.c_attn
        lora_layer = LoRALinear(original, rank=rank)
        layer.attn.c_attn = lora_layer
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    return lora_params


# =========================================================================
# Model Training
# =========================================================================

def train_ioi_model(model, tokenizer, train_data, val_data, seed, model_idx):
    """Fine-tune GPT-2 with LoRA on IOI task."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Apply LoRA — only LoRA params are trainable
    lora_params = apply_lora(model, rank=LORA_RANK)

    # Freeze everything except LoRA
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lora_params:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []

        random.Random(seed + epoch).shuffle(train_data)

        for i in range(0, len(train_data), 16):  # batch size 16
            batch = train_data[i:i+16]

            # Tokenize
            texts = [d['text'] + ' ' + d['io_name'] for d in batch]
            encodings = tokenizer(texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=64)
            input_ids = encodings['input_ids'].to(DEVICE)
            attention_mask = encodings['attention_mask'].to(DEVICE)

            # Forward pass — predict last token
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Loss: cross-entropy on the IO name token (last real token)
            # Find position of last non-pad token
            seq_lens = attention_mask.sum(dim=1) - 1
            target_logits = logits[torch.arange(len(batch)), seq_lens]

            # Target is the IO name token
            io_tokens = tokenizer([' ' + d['io_name'] for d in batch],
                                 add_special_tokens=False)['input_ids']
            io_token_ids = torch.tensor([t[0] for t in io_tokens]).to(DEVICE)

            loss = nn.CrossEntropyLoss()(target_logits, io_token_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        val_acc = evaluate_ioi(model, tokenizer, val_data)
        val_loss = np.mean(train_losses[-len(train_losses)//4:])  # approximate

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                'lora_state': {name: p.data.clone() for name, p in
                              zip([f'lora_{i}' for i in range(len(lora_params))], lora_params)},
                'seed': seed,
                'epoch': epoch,
                'val_acc': val_acc
            }, f'{MODEL_DIR}/model_{model_idx}.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return val_acc


def evaluate_ioi(model, tokenizer, data):
    """Evaluate IOI accuracy: does the model put higher logit on IO than S?"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            texts = [d['text'] for d in batch]
            encodings = tokenizer(texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=64)
            input_ids = encodings['input_ids'].to(DEVICE)
            attention_mask = encodings['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            seq_lens = attention_mask.sum(dim=1) - 1
            final_logits = logits[torch.arange(len(batch)), seq_lens]

            for j, d in enumerate(batch):
                io_tok = tokenizer(' ' + d['io_name'], add_special_tokens=False)['input_ids'][0]
                s_tok = tokenizer(' ' + d['s_name'], add_special_tokens=False)['input_ids'][0]
                if final_logits[j, io_tok] > final_logits[j, s_tok]:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0


# =========================================================================
# Activation Patching
# =========================================================================

def get_head_activations(model, tokenizer, text, layer_idx, head_idx):
    """Extract a specific head's output activation."""
    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output[0].detach()

    handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook_fn)

    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    handle.remove()
    return activations['output']


def activation_patching_importance(model, tokenizer, examples):
    """
    Compute activation patching importance for all 144 heads.

    For each head:
    1. Run clean forward pass, cache all activations
    2. Run corrupted forward pass, cache all activations
    3. Replace clean head output with corrupted head output
    4. Measure change in logit(IO) - logit(S)

    Returns: importance vector of shape (144,) = (12 layers × 12 heads)
    """
    model.eval()
    head_dim = model.config.n_embd // model.config.n_head  # 64 for GPT-2 small
    n_heads = model.config.n_head

    importance = np.zeros(N_COMPONENTS)

    for ex in examples:
        # Get clean logit difference
        clean_enc = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=64)
        clean_ids = clean_enc['input_ids'].to(DEVICE)
        clean_mask = clean_enc['attention_mask'].to(DEVICE)

        corrupt_enc = tokenizer(ex['corrupted_text'], return_tensors='pt',
                               truncation=True, max_length=64)
        corrupt_ids = corrupt_enc['input_ids'].to(DEVICE)
        corrupt_mask = corrupt_enc['attention_mask'].to(DEVICE)

        io_tok = tokenizer(' ' + ex['io_name'], add_special_tokens=False)['input_ids'][0]
        s_tok = tokenizer(' ' + ex['s_name'], add_special_tokens=False)['input_ids'][0]

        # Cache all activations for clean and corrupted runs
        clean_cache = {}
        corrupt_cache = {}

        def make_cache_hook(cache, layer_idx):
            def hook_fn(module, input, output):
                # output[0] is the attention output, shape (batch, seq, n_embd)
                cache[layer_idx] = output[0].detach().clone()
            return hook_fn

        # Register hooks for all layers
        handles = []
        for l in range(N_LAYERS):
            h = model.transformer.h[l].attn.register_forward_hook(
                make_cache_hook(clean_cache, l))
            handles.append(h)

        with torch.no_grad():
            clean_out = model(input_ids=clean_ids, attention_mask=clean_mask)
        for h in handles:
            h.remove()

        clean_logit_diff = (clean_out.logits[0, -1, io_tok] -
                           clean_out.logits[0, -1, s_tok]).item()

        # Corrupted run
        handles = []
        for l in range(N_LAYERS):
            h = model.transformer.h[l].attn.register_forward_hook(
                make_cache_hook(corrupt_cache, l))
            handles.append(h)

        with torch.no_grad():
            model(input_ids=corrupt_ids, attention_mask=corrupt_mask)
        for h in handles:
            h.remove()

        # For each head: patch clean with corrupted and measure effect
        for layer_idx in range(N_LAYERS):
            for head_idx in range(n_heads):
                comp_idx = layer_idx * n_heads + head_idx

                # Create patched activation: replace this head's slice
                # Head h occupies dimensions [h*head_dim : (h+1)*head_dim]
                def make_patch_hook(layer_idx, head_idx, corrupt_act):
                    def hook_fn(module, input, output):
                        patched = output[0].clone()
                        h_start = head_idx * head_dim
                        h_end = (head_idx + 1) * head_dim
                        # Replace this head's contribution with corrupted version
                        min_seq = min(patched.shape[1], corrupt_act.shape[1])
                        patched[0, :min_seq, h_start:h_end] = \
                            corrupt_act[0, :min_seq, h_start:h_end]
                        return (patched,) + output[1:]
                    return hook_fn

                handle = model.transformer.h[layer_idx].attn.register_forward_hook(
                    make_patch_hook(layer_idx, head_idx, corrupt_cache[layer_idx]))

                with torch.no_grad():
                    patched_out = model(input_ids=clean_ids, attention_mask=clean_mask)

                handle.remove()

                patched_logit_diff = (patched_out.logits[0, -1, io_tok] -
                                     patched_out.logits[0, -1, s_tok]).item()

                # Importance = how much patching changes the logit difference
                importance[comp_idx] += (clean_logit_diff - patched_logit_diff)

    importance /= len(examples)
    return importance


# =========================================================================
# Main Experiment
# =========================================================================

def main():
    start = time.time()

    print("=" * 70)
    print("DEFINITIVE MI EXPERIMENT: Activation Patching on IOI")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Models: {N_MODELS}, LoRA rank: {LORA_RANK}")
    print(f"Measurement: Activation patching (NOT zero ablation)")
    print(f"Task: Indirect Object Identification")
    print()

    # Load tokenizer and base model
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Generate IOI data
    print("Generating IOI datasets...")
    train_data = generate_ioi_dataset(N_IOI_TRAIN, seed=0)
    val_data = generate_ioi_dataset(N_IOI_VAL, seed=1)
    test_data = generate_ioi_dataset(N_IOI_TEST, seed=2)
    patching_data = generate_ioi_dataset(N_PATCHING_EXAMPLES, seed=3)

    # ===== Phase 1: Train 30 models =====
    print(f"\n{'='*70}")
    print(f"PHASE 1: Training {N_MODELS} LoRA fine-tuned models")
    print(f"{'='*70}")

    accuracies = []
    for i in range(N_MODELS):
        seed = 42 + i
        print(f"\n  Model {i+1}/{N_MODELS} (seed={seed})...")

        # Fresh model each time
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)

        acc = train_ioi_model(model, tokenizer, train_data, val_data, seed, i)
        accuracies.append(acc)
        print(f"    Accuracy: {acc:.3f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Rashomon filter
    best_acc = max(accuracies)
    rashomon_indices = [i for i, a in enumerate(accuracies)
                       if a >= best_acc - RASHOMON_THRESHOLD]
    rashomon_accs = [accuracies[i] for i in rashomon_indices]

    print(f"\n  All accuracies: {[f'{a:.3f}' for a in sorted(accuracies)]}")
    print(f"  Best: {best_acc:.3f}")
    print(f"  Rashomon set ({RASHOMON_THRESHOLD*100:.0f}% threshold): "
          f"{len(rashomon_indices)}/{N_MODELS} models")
    print(f"  Rashomon accuracy range: {max(rashomon_accs)-min(rashomon_accs):.4f}")

    # ===== Phase 2: Activation patching for Rashomon models =====
    print(f"\n{'='*70}")
    print(f"PHASE 2: Activation patching ({len(rashomon_indices)} models × "
          f"{N_COMPONENTS} heads × {N_PATCHING_EXAMPLES} examples)")
    print(f"{'='*70}")

    importance_vectors = {}
    for idx in rashomon_indices:
        print(f"\n  Patching model {idx} (acc={accuracies[idx]:.3f})...")

        # Load model with LoRA
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
        lora_params = apply_lora(model, rank=LORA_RANK)

        # Load saved LoRA weights
        ckpt = torch.load(f'{MODEL_DIR}/model_{idx}.pt', map_location=DEVICE)
        for (name, p), saved_p in zip(
            zip([f'lora_{i}' for i in range(len(lora_params))], lora_params),
            ckpt['lora_state'].values()
        ):
            p.data.copy_(saved_p)

        importance = activation_patching_importance(model, tokenizer, patching_data)
        importance_vectors[idx] = importance.tolist()

        print(f"    Top 5 heads: {', '.join(f'L{h//12}H{h%12}={importance[h]:.4f}' for h in np.argsort(np.abs(importance))[-5:][::-1])}")

        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # ===== Phase 3: Controls =====
    print(f"\n{'='*70}")
    print(f"PHASE 3: Controls")
    print(f"{'='*70}")

    # Control A: Base model (should recover known IOI circuit)
    print("\n  Control A: Base GPT-2 (should find known IOI circuit)...")
    base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
    base_importance = activation_patching_importance(base_model, tokenizer, patching_data)

    # Wang et al. name movers: L9H9, L9H6, L10H0
    known_movers = [9*12+9, 9*12+6, 10*12+0]
    top_20_base = set(np.argsort(np.abs(base_importance))[-20:])
    known_in_top20 = sum(1 for m in known_movers if m in top_20_base)
    print(f"    Known name movers in top-20: {known_in_top20}/3")
    print(f"    Top 5: {', '.join(f'L{h//12}H{h%12}={base_importance[h]:.4f}' for h in np.argsort(np.abs(base_importance))[-5:][::-1])}")

    del base_model
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Control B: Same-seed determinism
    print("\n  Control B: Same-seed determinism check...")
    model_b1 = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
    lora_params_b1 = apply_lora(model_b1, rank=LORA_RANK)
    ckpt = torch.load(f'{MODEL_DIR}/model_{rashomon_indices[0]}.pt', map_location=DEVICE)
    for p, saved_p in zip(lora_params_b1, ckpt['lora_state'].values()):
        p.data.copy_(saved_p)
    imp_b1 = activation_patching_importance(model_b1, tokenizer, patching_data[:50])
    imp_b2 = activation_patching_importance(model_b1, tokenizer, patching_data[:50])
    det_corr = np.corrcoef(imp_b1, imp_b2)[0, 1]
    print(f"    Same model, same data: r = {det_corr:.6f} (should be 1.0)")
    del model_b1
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # ===== Phase 4: Analysis =====
    print(f"\n{'='*70}")
    print(f"PHASE 4: Analysis")
    print(f"{'='*70}")

    indices = list(importance_vectors.keys())
    vectors = [np.array(importance_vectors[i]) for i in indices]

    # Pairwise Spearman correlations
    spearman_rhos = []
    flip_rates = []
    for i, j in combinations(range(len(indices)), 2):
        rho, p = spearmanr(vectors[i], vectors[j])
        spearman_rhos.append(rho)

        # Flip rate: fraction of head pairs where ranking disagrees
        n_flips = 0
        n_pairs = 0
        for h1, h2 in combinations(range(N_COMPONENTS), 2):
            if abs(vectors[i][h1] - vectors[i][h2]) > 1e-6 and \
               abs(vectors[j][h1] - vectors[j][h2]) > 1e-6:
                sign_i = np.sign(vectors[i][h1] - vectors[i][h2])
                sign_j = np.sign(vectors[j][h1] - vectors[j][h2])
                if sign_i != sign_j:
                    n_flips += 1
                n_pairs += 1
        flip_rates.append(n_flips / n_pairs if n_pairs > 0 else 0)

    mean_rho = np.mean(spearman_rhos)
    std_rho = np.std(spearman_rhos)
    mean_flip = np.mean(flip_rates)

    # Jaccard on top-K heads
    jaccard_results = {}
    for K in [10, 20, 30]:
        jaccards = []
        for i, j in combinations(range(len(indices)), 2):
            top_i = set(np.argsort(np.abs(vectors[i]))[-K:])
            top_j = set(np.argsort(np.abs(vectors[j]))[-K:])
            jacc = len(top_i & top_j) / len(top_i | top_j)
            jaccards.append(jacc)
        jaccard_results[K] = {
            'mean': float(np.mean(jaccards)),
            'std': float(np.std(jaccards)),
            'min': float(np.min(jaccards)),
            'max': float(np.max(jaccards))
        }

    # Coverage conflict and Gaussian prediction
    mean_vec = np.mean(vectors, axis=0)
    std_vec = np.std(vectors, axis=0)
    snr = np.zeros(N_COMPONENTS * (N_COMPONENTS - 1) // 2)
    cc = np.zeros_like(snr)
    observed_flip = np.zeros_like(snr)

    pair_idx = 0
    for h1, h2 in combinations(range(N_COMPONENTS), 2):
        diffs = [v[h1] - v[h2] for v in vectors]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if std_diff > 1e-10:
            snr[pair_idx] = abs(mean_diff) / std_diff
        cc[pair_idx] = 1.0 if std_diff > abs(mean_diff) * 0.5 else 0.0

        flips = sum(1 for k in range(len(vectors)-1)
                   for l in range(k+1, len(vectors))
                   if np.sign(vectors[k][h1] - vectors[k][h2]) !=
                      np.sign(vectors[l][h1] - vectors[l][h2]))
        total_pairs = len(vectors) * (len(vectors) - 1) // 2
        observed_flip[pair_idx] = flips / total_pairs if total_pairs > 0 else 0
        pair_idx += 1

    # Gaussian prediction vs observed
    from scipy.stats import norm
    predicted_flip = norm.cdf(-snr)
    valid = snr > 0
    if valid.sum() > 10:
        gauss_rho, gauss_p = spearmanr(predicted_flip[valid], observed_flip[valid])
    else:
        gauss_rho, gauss_p = 0.0, 1.0

    # ===== Phase 5: Verdict =====
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"\n  Rashomon set: {len(indices)} models")
    print(f"  Accuracy range: {max(rashomon_accs)-min(rashomon_accs):.4f}")
    print(f"\n  PRIMARY METRIC:")
    print(f"    Mean Spearman rho: {mean_rho:.3f} ± {std_rho:.3f}")
    print(f"    Mean flip rate: {mean_flip:.3f}")

    print(f"\n  JACCARD SIMILARITY (top-K heads):")
    for K, res in jaccard_results.items():
        print(f"    K={K}: {res['mean']:.3f} ± {res['std']:.3f} "
              f"[{res['min']:.3f}, {res['max']:.3f}]")

    print(f"\n  GAUSSIAN PREDICTION:")
    print(f"    Spearman rho (predicted vs observed flip): {gauss_rho:.3f} (p={gauss_p:.4f})")
    print(f"    Coverage conflict degree: {np.mean(cc):.3f}")

    # Determine outcome
    if mean_rho > 0.8:
        verdict = "CIRCUITS_STABLE"
        interpretation = ("Circuits are conserved across Rashomon models. "
                        "The impossibility applies to the model-to-circuit "
                        "mapping but circuits themselves are stable.")
    elif mean_rho < 0.3:
        verdict = "CIRCUITS_NONUNIQUE"
        interpretation = ("Genuine circuit Rashomon confirmed. Different training "
                        "runs find different computational pathways.")
    else:
        verdict = "PARTIAL_STABILITY"
        interpretation = ("Some circuit components conserved (skeleton), others "
                        "vary. Safety claims should be qualified to skeleton level.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {interpretation}")

    print(f"\n  CONTROLS:")
    print(f"    A (base model, known circuit): {known_in_top20}/3 name movers in top-20")
    print(f"    B (determinism): r = {det_corr:.6f}")

    elapsed = time.time() - start
    print(f"\n  Elapsed: {elapsed:.0f}s")

    # Save results
    results = {
        'experiment': 'mech_interp_definitive',
        'status': 'SUCCESS',
        'method': 'activation_patching',
        'task': 'IOI (Indirect Object Identification)',
        'model': 'GPT-2 small + LoRA (rank 4)',
        'measurement': 'Activation patching through W_U (no classifier head)',
        'n_total_models': N_MODELS,
        'n_rashomon_models': len(indices),
        'rashomon_threshold': RASHOMON_THRESHOLD,
        'rashomon_accuracy_range': float(max(rashomon_accs) - min(rashomon_accs)),
        'all_accuracies': accuracies,
        'rashomon_accuracies': rashomon_accs,
        'primary_metric': {
            'mean_spearman_rho': float(mean_rho),
            'std_spearman_rho': float(std_rho),
            'all_spearman_rhos': [float(r) for r in spearman_rhos],
        },
        'flip_rate': {
            'mean': float(mean_flip),
            'all': [float(f) for f in flip_rates],
        },
        'jaccard': jaccard_results,
        'gaussian_prediction': {
            'spearman_rho': float(gauss_rho),
            'spearman_p': float(gauss_p),
            'coverage_conflict': float(np.mean(cc)),
        },
        'controls': {
            'A_base_model': {
                'known_movers_in_top20': known_in_top20,
                'top_5': [{'layer': int(h//12), 'head': int(h%12),
                          'importance': float(base_importance[h])}
                         for h in np.argsort(np.abs(base_importance))[-5:][::-1]]
            },
            'B_determinism': {
                'correlation': float(det_corr),
            },
        },
        'verdict': verdict,
        'interpretation': interpretation,
        'importance_vectors': {str(k): v for k, v in importance_vectors.items()},
        'elapsed_seconds': elapsed,
    }

    out_path = OUT_DIR / 'results_mech_interp_definitive.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
