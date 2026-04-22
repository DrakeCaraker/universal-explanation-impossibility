#!/usr/bin/env python3
"""
GPT-2 IOI Circuit Stability Experiment

Trains 10 GPT-2-scale transformers from independent random initialization
on language modeling, then measures circuit importance for the Indirect Object
Identification (IOI) task via activation patching. Tests whether the
impossibility theorem's predictions hold at realistic scale.

Architecture: 6 layers, 8 heads/layer, d_model=512 (31M params)
Training data: WikiText-103 (via HuggingFace)
Evaluation: IOI task (Wang et al. 2022)
Measurement: activation patching (mean ablation per component)

Components: 6×8 = 48 attention heads + 6 MLPs = 54 total
Symmetry group: S_8^6 (within-layer head permutations)
dim(V^G) = 12 (mean head importance per layer + MLP per layer)
Predicted η = 1 - 12/54 = 0.778

Predictions:
1. Full-component rankings disagree across seeds (ρ ~ 0.5 or lower)
2. G-invariant projection (12-dim) lifts agreement dramatically
3. Within-layer head pairs flip at ~50%; head-vs-MLP pairs are stable
4. The resolution reveals which LAYERS matter, not which heads

Run: python gpt2_ioi_circuit_stability.py 2>&1 | tee gpt2_ioi_log.txt
Expected: ~4-8 hours per model on A100, ~1 hour per model on 8×A100
Total: ~2-3 days on a single A100, or ~8 hours with 10 parallel GPUs

Requirements:
  pip install torch transformers datasets einops tqdm scipy
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from itertools import combinations

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# =========================================================================
# Configuration
# =========================================================================

N_MODELS = 10          # independent random seeds
N_LAYERS = 6           # transformer layers
N_HEADS = 8            # heads per layer
D_MODEL = 512          # model dimension
D_MLP = 4 * D_MODEL    # MLP hidden dim
HEAD_DIM = D_MODEL // N_HEADS  # 64
VOCAB_SIZE = 50257     # GPT-2 tokenizer vocabulary
MAX_SEQ_LEN = 256      # context length for training
BATCH_SIZE = 32        # training batch size
LR = 3e-4              # learning rate
WEIGHT_DECAY = 0.1     # weight decay
N_STEPS = 50000        # training steps (~6.5M tokens seen at batch_size=32, seq_len=256... wait, that's 50000 * 32 * 256 = 409M tokens)
WARMUP_STEPS = 2000    # warmup
LOG_EVERY = 1000       # logging interval
SAVE_EVERY = 10000     # checkpoint interval

# IOI evaluation
N_IOI_EXAMPLES = 500   # IOI prompts for circuit evaluation
N_PATCH_EXAMPLES = 200 # examples for activation patching (faster)

N_COMPONENTS = N_LAYERS * N_HEADS + N_LAYERS  # 48 heads + 6 MLPs = 54
N_INVARIANT = N_LAYERS * 2  # 12 (mean_heads_per_layer + mlp_per_layer)
ETA_PREDICTED = 1 - N_INVARIANT / N_COMPONENTS  # 0.778

OUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUT_DIR / 'gpt2_ioi_models'
MODEL_DIR.mkdir(exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# =========================================================================
# Transformer (GPT-2 style, from scratch)
# =========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.W_QKV = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.W_O = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, return_head_outputs=False):
        B, T, C = x.shape
        qkv = self.W_QKV(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, n_heads, T, head_dim)

        if return_head_outputs:
            # Return per-head contributions before combining
            head_outs = out.transpose(1, 2).reshape(B, T, self.n_heads, self.head_dim)
            # Each head's contribution through W_O
            W_O_reshaped = self.W_O.weight.reshape(D_MODEL, self.n_heads, self.head_dim)
            per_head = torch.einsum('bthd,ohd->btho', head_outs, W_O_reshaped)
            # Wait, this doesn't quite work. Let me just return the combined output.
            pass

        out = out.transpose(1, 2).reshape(B, T, D_MODEL)
        out = self.W_O(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.fc2 = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# IOI Task (Indirect Object Identification)
# =========================================================================

# IOI templates following Wang et al. 2022
IOI_TEMPLATES = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} gave a ball to",
    "When {A} and {B} went to the office, {B} handed a file to",
    "When {A} and {B} went to the restaurant, {B} passed the menu to",
    "When {A} and {B} went to the library, {B} gave a book to",
    "When {A} and {B} went to the gym, {B} threw a towel to",
    "When {A} and {B} went to the café, {B} brought a coffee to",
    "When {A} and {B} went to the market, {B} sold a fish to",
    "When {A} and {B} went to the beach, {B} tossed the ball to",
    "When {A} and {B} went to the school, {B} gave a pencil to",
]

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]


def generate_ioi_prompts(n, tokenizer):
    """Generate IOI prompts with ground truth (indirect object = A)."""
    prompts = []
    rng = np.random.RandomState(42)
    for _ in range(n):
        template = IOI_TEMPLATES[rng.randint(len(IOI_TEMPLATES))]
        names = rng.choice(NAMES, size=2, replace=False)
        A, B = names[0], names[1]
        text = template.format(A=A, B=B)
        tokens = tokenizer.encode(text)
        # The correct answer is A (the indirect object)
        answer_token = tokenizer.encode(" " + A)[0]
        # The wrong answer is B (the subject)
        wrong_token = tokenizer.encode(" " + B)[0]
        prompts.append({
            'text': text,
            'tokens': tokens,
            'answer_token': answer_token,
            'wrong_token': wrong_token,
            'A': A,
            'B': B,
        })
    return prompts


def eval_ioi_accuracy(model, prompts, tokenizer):
    """Evaluate IOI accuracy: does the model predict A over B?"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for p in prompts:
            tokens = torch.tensor([p['tokens']], device=DEVICE)
            if tokens.shape[1] > MAX_SEQ_LEN:
                continue
            logits, _ = model(tokens)
            last_logits = logits[0, -1, :]
            if last_logits[p['answer_token']] > last_logits[p['wrong_token']]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# =========================================================================
# Activation Patching (Mean Ablation)
# =========================================================================

def measure_circuit_importance(model, prompts, tokenizer):
    """
    Measure importance of each component via activation patching.

    For each of the 54 components (48 attention heads + 6 MLPs):
    - Replace the component's output with its mean activation
    - Measure the drop in IOI performance (logit difference)

    Returns: importance vector of length 54
    """
    model.eval()

    # First, collect baseline logit differences and mean activations
    head_activations = {(l, h): [] for l in range(N_LAYERS) for h in range(N_HEADS)}
    mlp_activations = {l: [] for l in range(N_LAYERS)}

    baseline_logit_diffs = []

    # Collect activations with hooks
    activation_cache = {}

    def make_head_hook(layer, head):
        def hook_fn(module, input, output):
            # output shape: (B, T, D_MODEL) for the full attention
            # We need per-head activations. Store the full attention output.
            activation_cache[(layer, 'attn')] = output.detach()
        return hook_fn

    def make_mlp_hook(layer):
        def hook_fn(module, input, output):
            activation_cache[(layer, 'mlp')] = output.detach()
        return hook_fn

    # Register hooks for activation collection
    hooks = []
    for l in range(N_LAYERS):
        hooks.append(model.blocks[l].attn.register_forward_hook(make_head_hook(l, None)))
        hooks.append(model.blocks[l].mlp.register_forward_hook(make_mlp_hook(l)))

    # Collect mean activations and baseline performance
    all_attn_acts = {l: [] for l in range(N_LAYERS)}
    all_mlp_acts = {l: [] for l in range(N_LAYERS)}

    with torch.no_grad():
        for p in prompts[:N_PATCH_EXAMPLES]:
            tokens = torch.tensor([p['tokens']], device=DEVICE)
            if tokens.shape[1] > MAX_SEQ_LEN:
                continue
            activation_cache.clear()
            logits, _ = model(tokens)

            last_logits = logits[0, -1, :]
            diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
            baseline_logit_diffs.append(diff)

            for l in range(N_LAYERS):
                if (l, 'attn') in activation_cache:
                    all_attn_acts[l].append(activation_cache[(l, 'attn')])
                if (l, 'mlp') in activation_cache:
                    all_mlp_acts[l].append(activation_cache[(l, 'mlp')])

    for h in hooks:
        h.remove()

    baseline_mean = np.mean(baseline_logit_diffs)

    # Compute mean activations
    mean_attn = {}
    mean_mlp = {}
    for l in range(N_LAYERS):
        if all_attn_acts[l]:
            # Mean across examples (shape varies by seq len, so use per-position mean)
            mean_attn[l] = torch.cat(all_attn_acts[l], dim=0).mean(dim=0, keepdim=True)
        if all_mlp_acts[l]:
            mean_mlp[l] = torch.cat(all_mlp_acts[l], dim=0).mean(dim=0, keepdim=True)

    # Now measure importance: ablate each component and measure performance drop

    # For attention heads: we need to ablate individual heads within the attention output.
    # Since our attention module outputs the combined result, we'll use a different approach:
    # Hook into the attention module and zero out specific heads before the output projection.

    importance = np.zeros(N_COMPONENTS)  # 54 components

    # Measure attention head importance by patching Q/K/V for each head
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            component_idx = l * N_HEADS + h

            patched_diffs = []

            def make_ablation_hook(target_layer, target_head, mean_act):
                def hook_fn(module, input, output):
                    # output is the full attention output: (B, T, D_MODEL)
                    # We replace the output with the mean for this layer
                    # This ablates ALL heads at once per layer - we need finer control

                    # For per-head ablation, we'd need to modify the internal computation
                    # Simpler approach: replace the full attention output with mean
                    # and weight by 1/n_heads for approximate per-head ablation

                    # Actually, for a clean experiment, let's use layer-level ablation
                    # and then decompose using the G-invariant projection
                    return mean_act.expand_as(output)
                return hook_fn

            # For simplicity and correctness, ablate the full attention layer
            # (This gives layer-level importance, which the G-invariant projection recovers)
            # Per-head ablation requires modifying the attention internals, which we do below

            patched_diffs.append(0)  # placeholder

        # Layer-level attention ablation
        attn_ablation_diffs = []
        hook = model.blocks[l].attn.register_forward_hook(
            lambda mod, inp, out, ml=mean_attn[l]: ml.expand_as(out) if ml.shape[-1] == out.shape[-1] else out
        )
        with torch.no_grad():
            for p in prompts[:N_PATCH_EXAMPLES]:
                tokens = torch.tensor([p['tokens']], device=DEVICE)
                if tokens.shape[1] > MAX_SEQ_LEN:
                    continue
                logits, _ = model(tokens)
                last_logits = logits[0, -1, :]
                diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
                attn_ablation_diffs.append(diff)
        hook.remove()

        attn_importance = baseline_mean - np.mean(attn_ablation_diffs)
        # Distribute equally across heads in this layer (pre-projection assumption)
        for h in range(N_HEADS):
            importance[l * N_HEADS + h] = attn_importance / N_HEADS

    # MLP ablation
    for l in range(N_LAYERS):
        mlp_idx = N_LAYERS * N_HEADS + l  # indices 48-53

        hook = model.blocks[l].mlp.register_forward_hook(
            lambda mod, inp, out, ml=mean_mlp[l]: ml.expand_as(out) if ml.shape[-1] == out.shape[-1] else out
        )
        mlp_ablation_diffs = []
        with torch.no_grad():
            for p in prompts[:N_PATCH_EXAMPLES]:
                tokens = torch.tensor([p['tokens']], device=DEVICE)
                if tokens.shape[1] > MAX_SEQ_LEN:
                    continue
                logits, _ = model(tokens)
                last_logits = logits[0, -1, :]
                diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
                mlp_ablation_diffs.append(diff)
        hook.remove()

        importance[mlp_idx] = baseline_mean - np.mean(mlp_ablation_diffs)

    return importance, baseline_mean


# =========================================================================
# Per-Head Ablation (finer-grained)
# =========================================================================

def measure_per_head_importance(model, prompts, tokenizer):
    """
    Measure per-head importance by ablating individual heads.

    For each head: zero out that head's Q/K/V contribution within the
    attention computation, measure logit difference drop.

    This requires hooking into the attention internals.
    """
    model.eval()

    # Baseline
    baseline_diffs = []
    with torch.no_grad():
        for p in prompts[:N_PATCH_EXAMPLES]:
            tokens = torch.tensor([p['tokens']], device=DEVICE)
            if tokens.shape[1] > MAX_SEQ_LEN:
                continue
            logits, _ = model(tokens)
            last_logits = logits[0, -1, :]
            diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
            baseline_diffs.append(diff)
    baseline_mean = np.mean(baseline_diffs)

    importance = np.zeros(N_COMPONENTS)

    # Per-head ablation: zero out each head's output
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            component_idx = l * N_HEADS + h

            def make_zero_head_hook(target_head):
                def hook_fn(module, input, output):
                    # Reconstruct per-head and zero the target
                    # output shape: (B, T, D_MODEL)
                    B, T, _ = output.shape
                    # We need to redo the attention computation with one head zeroed
                    # Simpler: zero out the corresponding slice of the output
                    # Each head contributes HEAD_DIM dimensions before W_O mixes them
                    # But W_O mixes, so we can't simply slice the output.

                    # Alternative: hook into the attention BEFORE W_O
                    # This requires a different hook point. For now, use an approximation:
                    # Zero out 1/N_HEADS of the attention output (rough but fast)
                    out = output.clone()
                    start = target_head * HEAD_DIM
                    end = start + HEAD_DIM
                    # This zeros the target head's slice in the pre-W_O space
                    # It's approximate because W_O mixes, but captures the main effect
                    return out
                return hook_fn

            # More accurate approach: modify W_QKV to zero out the target head
            original_weight = model.blocks[l].attn.W_QKV.weight.data.clone()

            # Zero out Q, K, V weights for the target head
            # W_QKV has shape (3*D_MODEL, D_MODEL)
            # Q for head h: rows [h*HEAD_DIM : (h+1)*HEAD_DIM]
            # K for head h: rows [D_MODEL + h*HEAD_DIM : D_MODEL + (h+1)*HEAD_DIM]
            # V for head h: rows [2*D_MODEL + h*HEAD_DIM : 2*D_MODEL + (h+1)*HEAD_DIM]
            with torch.no_grad():
                for offset in [0, D_MODEL, 2 * D_MODEL]:
                    model.blocks[l].attn.W_QKV.weight.data[
                        offset + h * HEAD_DIM : offset + (h + 1) * HEAD_DIM, :
                    ] = 0.0

            # Measure performance with this head ablated
            head_diffs = []
            with torch.no_grad():
                for p in prompts[:N_PATCH_EXAMPLES]:
                    tokens = torch.tensor([p['tokens']], device=DEVICE)
                    if tokens.shape[1] > MAX_SEQ_LEN:
                        continue
                    logits, _ = model(tokens)
                    last_logits = logits[0, -1, :]
                    diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
                    head_diffs.append(diff)

            # Restore weights
            model.blocks[l].attn.W_QKV.weight.data.copy_(original_weight)

            importance[component_idx] = baseline_mean - np.mean(head_diffs)

    # MLP ablation (same as before)
    for l in range(N_LAYERS):
        mlp_idx = N_LAYERS * N_HEADS + l

        original_fc1 = model.blocks[l].mlp.fc1.weight.data.clone()
        with torch.no_grad():
            model.blocks[l].mlp.fc1.weight.data.zero_()

        mlp_diffs = []
        with torch.no_grad():
            for p in prompts[:N_PATCH_EXAMPLES]:
                tokens = torch.tensor([p['tokens']], device=DEVICE)
                if tokens.shape[1] > MAX_SEQ_LEN:
                    continue
                logits, _ = model(tokens)
                last_logits = logits[0, -1, :]
                diff = (last_logits[p['answer_token']] - last_logits[p['wrong_token']]).item()
                mlp_diffs.append(diff)

        model.blocks[l].mlp.fc1.weight.data.copy_(original_fc1)
        importance[mlp_idx] = baseline_mean - np.mean(mlp_diffs)

    return importance, baseline_mean


# =========================================================================
# G-Invariant Projection
# =========================================================================

def g_invariant_projection(importance_vectors):
    """
    Project 54-dim importance vectors onto the 12-dim G-invariant subspace.

    Symmetry group: S_8^6 (within-layer head permutations).
    Invariant subspace: mean head importance per layer (6 dims) +
                       MLP importance per layer (6 dims) = 12 dims.
    """
    n_models = len(importance_vectors)
    projected = np.zeros((n_models, N_INVARIANT))

    for i, imp in enumerate(importance_vectors):
        for l in range(N_LAYERS):
            # Mean head importance for layer l
            head_start = l * N_HEADS
            head_end = head_start + N_HEADS
            projected[i, l] = np.mean(imp[head_start:head_end])
            # MLP importance for layer l
            mlp_idx = N_LAYERS * N_HEADS + l
            projected[i, N_LAYERS + l] = imp[mlp_idx]

    return projected


# =========================================================================
# Analysis
# =========================================================================

def compute_agreement(vectors):
    """Compute pairwise Spearman correlations."""
    n = len(vectors)
    rhos = []
    for i, j in combinations(range(n), 2):
        rho, _ = spearmanr(vectors[i], vectors[j])
        rhos.append(rho)
    return np.array(rhos)


def compute_flip_rates(importance_vectors):
    """Compute pairwise flip rates for within-layer vs between-group pairs."""
    n_models = len(importance_vectors)
    within_flips = []
    between_flips = []

    for m1, m2 in combinations(range(n_models), 2):
        imp1 = importance_vectors[m1]
        imp2 = importance_vectors[m2]

        for l in range(N_LAYERS):
            # Within-layer head pairs (same symmetry group)
            for h1 in range(N_HEADS):
                for h2 in range(h1 + 1, N_HEADS):
                    idx1 = l * N_HEADS + h1
                    idx2 = l * N_HEADS + h2
                    rank1 = imp1[idx1] > imp1[idx2]
                    rank2 = imp2[idx1] > imp2[idx2]
                    within_flips.append(int(rank1 != rank2))

            # Between-group: head vs MLP in same layer
            mlp_idx = N_LAYERS * N_HEADS + l
            for h in range(N_HEADS):
                head_idx = l * N_HEADS + h
                rank1 = imp1[head_idx] > imp1[mlp_idx]
                rank2 = imp2[head_idx] > imp2[mlp_idx]
                between_flips.append(int(rank1 != rank2))

    return np.mean(within_flips), np.mean(between_flips)


# =========================================================================
# Training
# =========================================================================

def load_training_data():
    """Load WikiText-103 via HuggingFace."""
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    print("Loading WikiText-103 and GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

    # Tokenize and chunk into sequences of MAX_SEQ_LEN
    all_tokens = []
    for text in dataset['text']:
        if len(text.strip()) > 0:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        if len(all_tokens) > 20_000_000:  # ~20M tokens is enough
            break

    # Chunk into sequences
    n_sequences = len(all_tokens) // (MAX_SEQ_LEN + 1)
    all_tokens = all_tokens[:n_sequences * (MAX_SEQ_LEN + 1)]
    data = torch.tensor(all_tokens).reshape(n_sequences, MAX_SEQ_LEN + 1)

    print(f"Loaded {n_sequences} sequences of length {MAX_SEQ_LEN}")
    return data, tokenizer


def train_model(train_data, seed, tokenizer):
    """Train one GPT-2 model from scratch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GPT2Small().to(DEVICE)
    n_params = model.count_params()
    print(f"\nSeed {seed}: Training model ({n_params / 1e6:.1f}M params)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine LR schedule with warmup
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (N_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    n_sequences = train_data.shape[0]
    step = 0
    epoch = 0

    while step < N_STEPS:
        # Shuffle each epoch
        perm = torch.randperm(n_sequences)
        for batch_start in range(0, n_sequences - BATCH_SIZE, BATCH_SIZE):
            if step >= N_STEPS:
                break

            indices = perm[batch_start:batch_start + BATCH_SIZE]
            batch = train_data[indices].to(DEVICE)
            x = batch[:, :-1]
            y = batch[:, 1:]

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % LOG_EVERY == 0:
                print(f"  Step {step}/{N_STEPS}, loss={loss.item():.3f}, lr={scheduler.get_last_lr()[0]:.2e}")

            if step % SAVE_EVERY == 0:
                torch.save(model.state_dict(), MODEL_DIR / f'model_seed{seed}_step{step}.pt')

        epoch += 1

    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / f'model_seed{seed}_final.pt')
    print(f"  Seed {seed}: Training complete. Final loss={loss.item():.3f}")

    return model


# =========================================================================
# Main
# =========================================================================

def main():
    train_data, tokenizer = load_training_data()

    # Generate IOI prompts
    ioi_prompts = generate_ioi_prompts(N_IOI_EXAMPLES, tokenizer)
    print(f"Generated {len(ioi_prompts)} IOI prompts")
    print(f"Example: '{ioi_prompts[0]['text']}' → {ioi_prompts[0]['A']}")

    all_importance = []
    all_ioi_acc = []
    all_baseline = []

    for seed in range(N_MODELS):
        model_path = MODEL_DIR / f'model_seed{seed}_final.pt'

        if model_path.exists():
            print(f"\nSeed {seed}: Loading cached model...")
            model = GPT2Small().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            model = train_model(train_data, seed, tokenizer)

        # Evaluate IOI accuracy
        ioi_acc = eval_ioi_accuracy(model, ioi_prompts, tokenizer)
        print(f"  Seed {seed}: IOI accuracy = {ioi_acc:.3f}")
        all_ioi_acc.append(ioi_acc)

        # Measure circuit importance (per-head)
        importance, baseline = measure_per_head_importance(model, ioi_prompts, tokenizer)
        print(f"  Seed {seed}: Baseline logit diff = {baseline:.3f}")
        print(f"  Seed {seed}: Top component = {np.argmax(np.abs(importance))}, "
              f"importance = {np.max(np.abs(importance)):.3f}")
        all_importance.append(importance)
        all_baseline.append(baseline)

        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    all_importance = np.array(all_importance)

    # =====================================================================
    # Analysis
    # =====================================================================

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # 1. Full-component agreement
    full_rhos = compute_agreement(all_importance)
    print(f"\nFull 54-dim agreement: mean ρ = {np.mean(full_rhos):.3f}, "
          f"min = {np.min(full_rhos):.3f}, max = {np.max(full_rhos):.3f}")

    # 2. G-invariant projection
    projected = g_invariant_projection(all_importance)
    proj_rhos = compute_agreement(projected)
    print(f"G-invariant 12-dim:    mean ρ = {np.mean(proj_rhos):.3f}, "
          f"min = {np.min(proj_rhos):.3f}, max = {np.max(proj_rhos):.3f}")

    # 3. Pearson agreement
    full_pearson = []
    proj_pearson = []
    for i, j in combinations(range(N_MODELS), 2):
        r, _ = pearsonr(all_importance[i], all_importance[j])
        full_pearson.append(r)
        r, _ = pearsonr(projected[i], projected[j])
        proj_pearson.append(r)
    print(f"Full Pearson:          mean r = {np.mean(full_pearson):.3f}")
    print(f"G-invariant Pearson:   mean r = {np.mean(proj_pearson):.3f}")

    # 4. Noether counting (flip rates)
    within_flip, between_flip = compute_flip_rates(all_importance)
    print(f"\nNoether counting:")
    print(f"  Within-layer head pair flip rate: {within_flip:.3f} (predicted: ~0.500)")
    print(f"  Head-vs-MLP flip rate:            {between_flip:.3f} (predicted: ~0.000)")
    print(f"  Gap:                              {within_flip - between_flip:.3f} pp")

    if len(all_importance) >= 2:
        # Mann-Whitney test
        within_all = []
        between_all = []
        for m1, m2 in combinations(range(N_MODELS), 2):
            for l in range(N_LAYERS):
                for h1 in range(N_HEADS):
                    for h2 in range(h1+1, N_HEADS):
                        idx1 = l * N_HEADS + h1
                        idx2 = l * N_HEADS + h2
                        within_all.append(int((all_importance[m1][idx1] > all_importance[m1][idx2]) !=
                                             (all_importance[m2][idx1] > all_importance[m2][idx2])))
                mlp_idx = N_LAYERS * N_HEADS + l
                for h in range(N_HEADS):
                    head_idx = l * N_HEADS + h
                    between_all.append(int((all_importance[m1][head_idx] > all_importance[m1][mlp_idx]) !=
                                          (all_importance[m2][head_idx] > all_importance[m2][mlp_idx])))

        try:
            stat, p = mannwhitneyu(within_all, between_all, alternative='greater')
            print(f"  Mann-Whitney p = {p:.2e}")
        except Exception:
            pass

    # 5. IOI accuracy summary
    print(f"\nIOI accuracy: mean = {np.mean(all_ioi_acc):.3f}, "
          f"std = {np.std(all_ioi_acc):.3f}, "
          f"range = [{np.min(all_ioi_acc):.3f}, {np.max(all_ioi_acc):.3f}]")

    # 6. η law check
    print(f"\nη law:")
    print(f"  Predicted η = {ETA_PREDICTED:.3f}")
    print(f"  Components = {N_COMPONENTS}, Invariant dims = {N_INVARIANT}")

    # =====================================================================
    # Save results
    # =====================================================================

    results = {
        'config': {
            'n_models': N_MODELS,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'd_model': D_MODEL,
            'n_components': N_COMPONENTS,
            'n_invariant': N_INVARIANT,
            'eta_predicted': ETA_PREDICTED,
            'n_steps': N_STEPS,
        },
        'ioi_accuracy': {
            'mean': float(np.mean(all_ioi_acc)),
            'std': float(np.std(all_ioi_acc)),
            'per_model': [float(x) for x in all_ioi_acc],
        },
        'full_agreement': {
            'mean_spearman': float(np.mean(full_rhos)),
            'min_spearman': float(np.min(full_rhos)),
            'max_spearman': float(np.max(full_rhos)),
            'mean_pearson': float(np.mean(full_pearson)),
        },
        'g_invariant_agreement': {
            'mean_spearman': float(np.mean(proj_rhos)),
            'min_spearman': float(np.min(proj_rhos)),
            'max_spearman': float(np.max(proj_rhos)),
            'mean_pearson': float(np.mean(proj_pearson)),
        },
        'noether_counting': {
            'within_layer_flip_rate': float(within_flip),
            'between_group_flip_rate': float(between_flip),
            'gap': float(within_flip - between_flip),
        },
        'importance_vectors': all_importance.tolist(),
        'projected_vectors': projected.tolist(),
    }

    out_path = OUT_DIR / 'results_gpt2_ioi_circuit_stability.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Architecture:      {N_LAYERS}L/{N_HEADS}H, d={D_MODEL} ({GPT2Small().count_params()/1e6:.1f}M params)")
    print(f"IOI accuracy:      {np.mean(all_ioi_acc):.1%} (mean across {N_MODELS} seeds)")
    print(f"Full agreement:    ρ = {np.mean(full_rhos):.3f}")
    print(f"G-inv agreement:   ρ = {np.mean(proj_rhos):.3f}")
    print(f"Lift:              {np.mean(full_rhos):.3f} → {np.mean(proj_rhos):.3f}")
    print(f"Within-layer flip: {within_flip:.3f}")
    print(f"Head-vs-MLP flip:  {between_flip:.3f}")
    print(f"Predicted η:       {ETA_PREDICTED:.3f}")


if __name__ == '__main__':
    main()
