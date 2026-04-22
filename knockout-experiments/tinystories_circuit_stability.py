#!/usr/bin/env python3
"""
TinyStories Circuit Stability Experiment

Trains 10 small transformer language models from independent random
initialization on TinyStories (Eldan & Li 2023), then measures circuit
importance via activation patching. Tests whether the impossibility
theorem's predictions hold on REAL LANGUAGE, not just modular addition.

Architecture: 4 layers, 4 heads/layer, d_model=256 (~8M params)
Training data: TinyStories (~470K stories)
Components: 16 attention heads + 4 MLPs = 20
Symmetry group: S_4^4 (within-layer head permutations)
dim(V^G) = 8, predicted η = 0.6

PRE-REGISTERED PREDICTIONS (stated before any data generated):
  1. All models achieve similar perplexity (CV < 5%)
  2. Full 20-dim agreement: ρ < 0.70
  3. G-invariant 8-dim agreement: ρ > 0.80
  4. Lift is significant (permutation test p < 0.01)
  5. Within-layer head flip rate > 0.40
  6. Between-group (head vs MLP) flip rate < 0.15
  7. Mann-Whitney within > between: p < 0.01
  8. Random projection control: ρ < G-invariant ρ

Run:
  # Quick pilot (2 models, ~2 hours)
  python tinystories_circuit_stability.py --pilot

  # Full experiment (10 models, ~8-10 hours)
  python tinystories_circuit_stability.py

  # Analysis only (if models already trained)
  python tinystories_circuit_stability.py --analysis-only

Requirements: torch, transformers, datasets, scipy, numpy
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, math, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from itertools import combinations

# =========================================================================
# Configuration
# =========================================================================

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

N_LAYERS = 4
N_HEADS = 4
D_MODEL = 256
D_MLP = 4 * D_MODEL
HEAD_DIM = D_MODEL // N_HEADS  # 64
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 500
LOG_EVERY = 500

N_COMPONENTS = N_LAYERS * N_HEADS + N_LAYERS  # 20
N_INVARIANT = N_LAYERS * 2  # 8
ETA_PREDICTED = 1 - N_INVARIANT / N_COMPONENTS  # 0.6

N_PATCH_EXAMPLES = 200  # examples for activation patching
N_EVAL_EXAMPLES = 500   # examples for perplexity evaluation

OUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUT_DIR / 'tinystories_models'
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

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.W_Q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_O = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.W_O(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.fc2 = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Data Loading
# =========================================================================

def load_tinystories(tokenizer, max_tokens=5_000_000):
    """Load and tokenize TinyStories."""
    from datasets import load_dataset
    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    all_tokens = []
    for i, example in enumerate(ds):
        text = example.get('text', '')
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens:
            break

    # Also load validation set
    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    val_tokens = []
    for example in ds_val:
        text = example.get('text', '')
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        val_tokens.extend(tokens)
        if len(val_tokens) >= 500_000:
            break

    # Chunk into sequences
    seq_len = MAX_SEQ_LEN + 1
    n_train = len(all_tokens) // seq_len
    n_val = len(val_tokens) // seq_len

    train_data = torch.tensor(all_tokens[:n_train * seq_len]).reshape(n_train, seq_len)
    val_data = torch.tensor(val_tokens[:n_val * seq_len]).reshape(n_val, seq_len)

    print(f"  Train: {n_train} sequences ({len(all_tokens):,} tokens)")
    print(f"  Val:   {n_val} sequences ({len(val_tokens):,} tokens)")
    return train_data, val_data


# =========================================================================
# Training
# =========================================================================

def train_model(train_data, val_data, seed, vocab_size, n_steps):
    """Train one language model from scratch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE == 'mps':
        torch.mps.manual_seed(seed)

    model = TinyLM(vocab_size).to(DEVICE)
    print(f"\n{'='*60}")
    print(f"Seed {seed}: Training ({model.n_params()/1e6:.1f}M params, {n_steps} steps)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, n_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    n_seqs = train_data.shape[0]
    step = 0
    t0 = time.time()

    model.train()
    while step < n_steps:
        perm = torch.randperm(n_seqs)
        for batch_start in range(0, n_seqs - BATCH_SIZE, BATCH_SIZE):
            if step >= n_steps:
                break
            batch = train_data[perm[batch_start:batch_start + BATCH_SIZE]].to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                eta_min = (n_steps - step) / steps_per_sec / 60
                print(f"  Step {step}/{n_steps} | loss={loss.item():.3f} | "
                      f"{steps_per_sec:.1f} step/s | ETA {eta_min:.0f}min")

    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, min(len(val_data), N_EVAL_EXAMPLES), BATCH_SIZE):
            batch = val_data[i:i+BATCH_SIZE].to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]
            _, loss = model(x, y)
            val_losses.append(loss.item())

    val_ppl = math.exp(np.mean(val_losses))
    print(f"  Seed {seed}: val_loss={np.mean(val_losses):.3f}, val_ppl={val_ppl:.1f}")

    # Save
    torch.save(model.state_dict(), MODEL_DIR / f'model_seed{seed}.pt')
    return model, val_ppl


# =========================================================================
# Circuit Importance via Activation Patching
# =========================================================================

def measure_importance(model, val_data, vocab_size):
    """
    Measure importance of each component via weight zeroing.

    For each of 20 components (16 heads + 4 MLPs):
    - Zero out the component's weights
    - Measure perplexity increase on held-out data
    - Restore weights

    Returns: 20-dim importance vector (perplexity increase per component)
    """
    model.eval()

    # Baseline perplexity
    baseline_losses = []
    with torch.no_grad():
        for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), BATCH_SIZE):
            batch = val_data[i:i+BATCH_SIZE].to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]
            _, loss = model(x, y)
            baseline_losses.append(loss.item())
    baseline_loss = np.mean(baseline_losses)

    importance = np.zeros(N_COMPONENTS)

    # Per-head ablation: zero out Q, K, V weights for target head
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            idx = l * N_HEADS + h

            # Save original weights
            orig_Q = model.blocks[l].attn.W_Q.weight.data.clone()
            orig_K = model.blocks[l].attn.W_K.weight.data.clone()
            orig_V = model.blocks[l].attn.W_V.weight.data.clone()

            # Zero out this head's slice
            with torch.no_grad():
                model.blocks[l].attn.W_Q.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM, :] = 0
                model.blocks[l].attn.W_K.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM, :] = 0
                model.blocks[l].attn.W_V.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM, :] = 0

            # Measure ablated performance
            ablated_losses = []
            with torch.no_grad():
                for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), BATCH_SIZE):
                    batch = val_data[i:i+BATCH_SIZE].to(DEVICE)
                    x, y = batch[:, :-1], batch[:, 1:]
                    _, loss = model(x, y)
                    ablated_losses.append(loss.item())

            importance[idx] = np.mean(ablated_losses) - baseline_loss

            # Restore
            model.blocks[l].attn.W_Q.weight.data.copy_(orig_Q)
            model.blocks[l].attn.W_K.weight.data.copy_(orig_K)
            model.blocks[l].attn.W_V.weight.data.copy_(orig_V)

    # MLP ablation
    for l in range(N_LAYERS):
        mlp_idx = N_LAYERS * N_HEADS + l

        orig_fc1 = model.blocks[l].mlp.fc1.weight.data.clone()
        with torch.no_grad():
            model.blocks[l].mlp.fc1.weight.data.zero_()

        ablated_losses = []
        with torch.no_grad():
            for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), BATCH_SIZE):
                batch = val_data[i:i+BATCH_SIZE].to(DEVICE)
                x, y = batch[:, :-1], batch[:, 1:]
                _, loss = model(x, y)
                ablated_losses.append(loss.item())

        importance[mlp_idx] = np.mean(ablated_losses) - baseline_loss
        model.blocks[l].mlp.fc1.weight.data.copy_(orig_fc1)

    return importance, baseline_loss


# =========================================================================
# Determinism Control
# =========================================================================

def determinism_check(model, val_data, vocab_size):
    """Run importance measurement twice to verify determinism."""
    imp1, _ = measure_importance(model, val_data, vocab_size)
    imp2, _ = measure_importance(model, val_data, vocab_size)
    r, _ = pearsonr(imp1, imp2)
    return r


# =========================================================================
# G-Invariant Projection
# =========================================================================

def g_invariant_projection(importance_vectors):
    """Project onto S_4^4-invariant subspace: mean head per layer + MLP per layer."""
    n = len(importance_vectors)
    projected = np.zeros((n, N_INVARIANT))
    for i, imp in enumerate(importance_vectors):
        for l in range(N_LAYERS):
            projected[i, l] = np.mean(imp[l*N_HEADS:(l+1)*N_HEADS])
            projected[i, N_LAYERS + l] = imp[N_LAYERS*N_HEADS + l]
    return projected


def random_projection(importance_vectors, target_dim=N_INVARIANT, n_trials=100):
    """Random projection control: project onto random 8-dim subspaces."""
    rhos_per_trial = []
    for _ in range(n_trials):
        # Random orthogonal projection matrix
        M = np.random.randn(N_COMPONENTS, target_dim)
        M, _ = np.linalg.qr(M)
        projected = np.array([imp @ M for imp in importance_vectors])
        rhos = []
        for i, j in combinations(range(len(importance_vectors)), 2):
            r, _ = spearmanr(projected[i], projected[j])
            rhos.append(r)
        rhos_per_trial.append(np.mean(rhos))
    return np.array(rhos_per_trial)


def permutation_test(importance_vectors, n_perms=1000):
    """
    Permutation test: randomly reassign heads to layers (breaking symmetry).
    If actual G-invariant ρ > permuted ρ in 95%+ of cases, the symmetry matters.
    """
    actual_proj = g_invariant_projection(importance_vectors)
    actual_rhos = []
    for i, j in combinations(range(len(importance_vectors)), 2):
        r, _ = spearmanr(actual_proj[i], actual_proj[j])
        actual_rhos.append(r)
    actual_mean = np.mean(actual_rhos)

    perm_means = []
    for _ in range(n_perms):
        # Shuffle head assignments (break within-layer structure)
        shuffled = []
        for imp in importance_vectors:
            head_imp = imp[:N_LAYERS * N_HEADS].copy()
            np.random.shuffle(head_imp)  # shuffle all heads across layers
            new_imp = np.concatenate([head_imp, imp[N_LAYERS * N_HEADS:]])
            shuffled.append(new_imp)
        perm_proj = g_invariant_projection(shuffled)
        perm_rhos = []
        for i, j in combinations(range(len(shuffled)), 2):
            r, _ = spearmanr(perm_proj[i], perm_proj[j])
            perm_rhos.append(r)
        perm_means.append(np.mean(perm_rhos))

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= actual_mean)
    return actual_mean, perm_means, p_value


# =========================================================================
# Flip Rate Analysis
# =========================================================================

def compute_flip_rates(importance_vectors):
    """Noether counting: within-layer vs between-group flip rates."""
    n = len(importance_vectors)
    within_flips = []
    between_flips = []

    for m1, m2 in combinations(range(n), 2):
        imp1, imp2 = importance_vectors[m1], importance_vectors[m2]

        for l in range(N_LAYERS):
            # Within-layer head pairs
            for h1 in range(N_HEADS):
                for h2 in range(h1 + 1, N_HEADS):
                    i1, i2 = l*N_HEADS + h1, l*N_HEADS + h2
                    flip = int((imp1[i1] > imp1[i2]) != (imp2[i1] > imp2[i2]))
                    within_flips.append(flip)

            # Between-group: head vs MLP in same layer
            mlp_idx = N_LAYERS * N_HEADS + l
            for h in range(N_HEADS):
                head_idx = l * N_HEADS + h
                flip = int((imp1[head_idx] > imp1[mlp_idx]) != (imp2[head_idx] > imp2[mlp_idx]))
                between_flips.append(flip)

    return np.array(within_flips), np.array(between_flips)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run 2-model pilot')
    parser.add_argument('--analysis-only', action='store_true', help='Skip training')
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=30000)
    args = parser.parse_args()

    n_models = 2 if args.pilot else args.n_models
    n_steps = 15000 if args.pilot else args.n_steps

    print(f"Device: {DEVICE}")
    print(f"Models: {n_models}, Steps: {n_steps}")
    print(f"Components: {N_COMPONENTS} ({N_LAYERS}×{N_HEADS} heads + {N_LAYERS} MLPs)")
    print(f"Invariant dims: {N_INVARIANT}, Predicted η: {ETA_PREDICTED:.3f}")

    # Load tokenizer and data
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    train_data, val_data = load_tinystories(tokenizer)

    # =====================================================================
    # Training Phase
    # =====================================================================

    all_importance = []
    all_ppl = []

    for seed in range(n_models):
        model_path = MODEL_DIR / f'model_seed{seed}.pt'

        if args.analysis_only or model_path.exists():
            print(f"\nSeed {seed}: Loading cached model...")
            model = TinyLM(vocab_size).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            # Compute val perplexity
            model.eval()
            losses = []
            with torch.no_grad():
                for i in range(0, min(len(val_data), N_EVAL_EXAMPLES), BATCH_SIZE):
                    batch = val_data[i:i+BATCH_SIZE].to(DEVICE)
                    x, y = batch[:, :-1], batch[:, 1:]
                    _, loss = model(x, y)
                    losses.append(loss.item())
            ppl = math.exp(np.mean(losses))
            print(f"  Val perplexity: {ppl:.1f}")
        else:
            model, ppl = train_model(train_data, val_data, seed, vocab_size, n_steps)

        all_ppl.append(ppl)

        # Determinism check (first model only)
        if seed == 0:
            det_r = determinism_check(model, val_data, vocab_size)
            print(f"  Determinism check: r = {det_r:.6f}")

        # Measure circuit importance
        print(f"  Measuring circuit importance ({N_COMPONENTS} components)...")
        importance, baseline = measure_importance(model, val_data, vocab_size)
        print(f"  Baseline loss: {baseline:.3f}")

        # Print top components
        sorted_idx = np.argsort(-np.abs(importance))
        for rank, idx in enumerate(sorted_idx[:5]):
            if idx < N_LAYERS * N_HEADS:
                l, h = idx // N_HEADS, idx % N_HEADS
                name = f"L{l}H{h}"
            else:
                l = idx - N_LAYERS * N_HEADS
                name = f"MLP{l}"
            print(f"    #{rank+1}: {name} = {importance[idx]:.4f}")

        all_importance.append(importance)
        del model
        if DEVICE == 'mps':
            torch.mps.empty_cache()

    all_importance = np.array(all_importance)

    # =====================================================================
    # Analysis Phase
    # =====================================================================

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Perplexity consistency
    ppl_cv = np.std(all_ppl) / np.mean(all_ppl) * 100
    print(f"\nPerplexity: mean={np.mean(all_ppl):.1f}, std={np.std(all_ppl):.1f}, CV={ppl_cv:.1f}%")
    print(f"  Pre-registered: CV < 5%  →  {'PASS' if ppl_cv < 5 else 'FAIL'}")

    # Full agreement
    full_rhos = []
    for i, j in combinations(range(n_models), 2):
        r, _ = spearmanr(all_importance[i], all_importance[j])
        full_rhos.append(r)
    full_rhos = np.array(full_rhos)
    print(f"\nFull {N_COMPONENTS}-dim agreement: ρ = {np.mean(full_rhos):.3f} "
          f"[{np.min(full_rhos):.3f}, {np.max(full_rhos):.3f}]")
    print(f"  Pre-registered: ρ < 0.70  →  {'PASS' if np.mean(full_rhos) < 0.70 else 'FAIL'}")

    # G-invariant projection
    projected = g_invariant_projection(all_importance)
    proj_rhos = []
    for i, j in combinations(range(n_models), 2):
        r, _ = spearmanr(projected[i], projected[j])
        proj_rhos.append(r)
    proj_rhos = np.array(proj_rhos)
    print(f"G-invariant {N_INVARIANT}-dim:   ρ = {np.mean(proj_rhos):.3f} "
          f"[{np.min(proj_rhos):.3f}, {np.max(proj_rhos):.3f}]")
    print(f"  Pre-registered: ρ > 0.80  →  {'PASS' if np.mean(proj_rhos) > 0.80 else 'FAIL'}")
    print(f"  LIFT: {np.mean(full_rhos):.3f} → {np.mean(proj_rhos):.3f}")

    # Pearson
    full_pearson = [pearsonr(all_importance[i], all_importance[j])[0]
                    for i, j in combinations(range(n_models), 2)]
    proj_pearson = [pearsonr(projected[i], projected[j])[0]
                    for i, j in combinations(range(n_models), 2)]
    print(f"Full Pearson:  r = {np.mean(full_pearson):.3f}")
    print(f"G-inv Pearson: r = {np.mean(proj_pearson):.3f}")

    # Excl-MLP control: G-invariant projection using ONLY head means (no MLPs)
    # This tests whether the lift is from symmetry-averaging of heads,
    # not just from MLPs being trivially stable.
    heads_only_proj = g_invariant_projection(all_importance)[:, :N_LAYERS]  # first 4 dims = mean heads
    heads_only_rhos = []
    for i, j in combinations(range(n_models), 2):
        r, _ = spearmanr(heads_only_proj[i], heads_only_proj[j])
        heads_only_rhos.append(r)
    heads_only_rhos = np.array(heads_only_rhos)
    print(f"\nExcl-MLP control (heads-only, {N_LAYERS}-dim):")
    print(f"  Mean heads ρ = {np.mean(heads_only_rhos):.3f} "
          f"[{np.min(heads_only_rhos):.3f}, {np.max(heads_only_rhos):.3f}]")
    print(f"  (If this >> full ρ, the lift is from symmetry-averaging, not MLP dominance)")

    # Heads-only full agreement (no projection, just the 16 head components)
    heads_raw_rhos = []
    for i, j in combinations(range(n_models), 2):
        r, _ = spearmanr(all_importance[i][:N_LAYERS*N_HEADS],
                         all_importance[j][:N_LAYERS*N_HEADS])
        heads_raw_rhos.append(r)
    heads_raw_rhos = np.array(heads_raw_rhos)
    print(f"  Raw heads (16-dim) ρ = {np.mean(heads_raw_rhos):.3f}")
    print(f"  Head lift: {np.mean(heads_raw_rhos):.3f} → {np.mean(heads_only_rhos):.3f}")

    # Pearson/Spearman divergence explanation
    print(f"\nPearson vs Spearman divergence:")
    print(f"  Full Pearson r = {np.mean(full_pearson):.3f}, Full Spearman ρ = {np.mean(full_rhos):.3f}")
    print(f"  (High Pearson + low Spearman = models agree on magnitude structure")
    print(f"   but disagree on rankings of smaller components — exactly what the")
    print(f"   theorem predicts: between-group stable, within-group unstable)")

    # Random projection control
    print(f"\nRandom projection control (100 trials, {N_INVARIANT}-dim)...")
    rand_rhos = random_projection(all_importance)
    print(f"  Random:       ρ = {np.mean(rand_rhos):.3f} ± {np.std(rand_rhos):.3f}")
    print(f"  G-invariant:  ρ = {np.mean(proj_rhos):.3f}")
    print(f"  Pre-registered: G-inv > random  →  "
          f"{'PASS' if np.mean(proj_rhos) > np.mean(rand_rhos) else 'FAIL'}")

    # Permutation test
    if n_models >= 3:
        print(f"\nPermutation test (1000 permutations)...")
        actual_mean, perm_dist, perm_p = permutation_test(all_importance)
        print(f"  Actual:    ρ = {actual_mean:.3f}")
        print(f"  Permuted:  ρ = {np.mean(perm_dist):.3f} ± {np.std(perm_dist):.3f}")
        print(f"  p-value:   {perm_p:.4f}")
        print(f"  Pre-registered: p < 0.01  →  {'PASS' if perm_p < 0.01 else 'FAIL'}")

    # Noether counting (flip rates)
    within_flips, between_flips = compute_flip_rates(all_importance)
    within_rate = np.mean(within_flips)
    between_rate = np.mean(between_flips)
    print(f"\nNoether counting:")
    print(f"  Within-layer flip rate:  {within_rate:.3f}  (predicted ~0.500)")
    print(f"  Head-vs-MLP flip rate:   {between_rate:.3f}  (predicted ~0.000)")
    print(f"  Gap:                     {within_rate - between_rate:.3f}")
    print(f"  Pre-registered: within > 0.40  →  {'PASS' if within_rate > 0.40 else 'FAIL'}")
    print(f"  Pre-registered: between < 0.15 →  {'PASS' if between_rate < 0.15 else 'FAIL'}")

    if len(within_flips) > 0 and len(between_flips) > 0:
        try:
            stat, p = mannwhitneyu(within_flips, between_flips, alternative='greater')
            print(f"  Mann-Whitney p = {p:.2e}")
            print(f"  Pre-registered: p < 0.01  →  {'PASS' if p < 0.01 else 'FAIL'}")
        except Exception as e:
            print(f"  Mann-Whitney: {e}")

    # Cross-layer head flip rates (additional breakdown)
    cross_layer_flips = []
    for m1, m2 in combinations(range(n_models), 2):
        imp1, imp2 = all_importance[m1], all_importance[m2]
        for l1 in range(N_LAYERS):
            for l2 in range(l1+1, N_LAYERS):
                for h1 in range(N_HEADS):
                    for h2 in range(N_HEADS):
                        i1 = l1 * N_HEADS + h1
                        i2 = l2 * N_HEADS + h2
                        flip = int((imp1[i1] > imp1[i2]) != (imp2[i1] > imp2[i2]))
                        cross_layer_flips.append(flip)
    cross_layer_rate = np.mean(cross_layer_flips) if cross_layer_flips else 0
    print(f"\nFlip rate breakdown:")
    print(f"  Within-layer head pairs:  {within_rate:.3f}  (predicted ~0.500)")
    print(f"  Cross-layer head pairs:   {cross_layer_rate:.3f}  (predicted ~0.000 if layers differ)")
    print(f"  Head-vs-MLP pairs:        {between_rate:.3f}  (predicted ~0.000)")

    # =====================================================================
    # Save Results
    # =====================================================================

    results = {
        'config': {
            'n_models': n_models,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'd_model': D_MODEL,
            'n_steps': n_steps,
            'n_components': N_COMPONENTS,
            'n_invariant': N_INVARIANT,
            'eta_predicted': ETA_PREDICTED,
            'device': DEVICE,
            'task': 'TinyStories language modeling',
        },
        'pre_registered_predictions': {
            'perplexity_cv_lt_5': ppl_cv < 5,
            'full_rho_lt_0.70': float(np.mean(full_rhos)) < 0.70,
            'ginv_rho_gt_0.80': float(np.mean(proj_rhos)) > 0.80,
            'within_flip_gt_0.40': within_rate > 0.40,
            'between_flip_lt_0.15': between_rate < 0.15,
        },
        'perplexity': {
            'mean': float(np.mean(all_ppl)),
            'std': float(np.std(all_ppl)),
            'cv_pct': float(ppl_cv),
            'per_model': [float(x) for x in all_ppl],
        },
        'full_agreement': {
            'mean_spearman': float(np.mean(full_rhos)),
            'min_spearman': float(np.min(full_rhos)),
            'max_spearman': float(np.max(full_rhos)),
            'mean_pearson': float(np.mean(full_pearson)),
            'all_spearman': [float(x) for x in full_rhos],
        },
        'g_invariant_agreement': {
            'mean_spearman': float(np.mean(proj_rhos)),
            'min_spearman': float(np.min(proj_rhos)),
            'max_spearman': float(np.max(proj_rhos)),
            'mean_pearson': float(np.mean(proj_pearson)),
            'all_spearman': [float(x) for x in proj_rhos],
        },
        'excl_mlp_control': {
            'heads_only_mean_spearman': float(np.mean(heads_only_rhos)),
            'heads_raw_mean_spearman': float(np.mean(heads_raw_rhos)),
            'head_lift': f"{float(np.mean(heads_raw_rhos)):.3f} -> {float(np.mean(heads_only_rhos)):.3f}",
        },
        'flip_rate_breakdown': {
            'within_layer': float(within_rate),
            'cross_layer': float(cross_layer_rate),
            'head_vs_mlp': float(between_rate),
        },
        'random_projection_control': {
            'mean_rho': float(np.mean(rand_rhos)),
            'std_rho': float(np.std(rand_rhos)),
        },
        'noether_counting': {
            'within_layer_flip_rate': float(within_rate),
            'between_group_flip_rate': float(between_rate),
            'gap': float(within_rate - between_rate),
        },
        'importance_vectors': all_importance.tolist(),
        'projected_vectors': projected.tolist(),
    }

    if n_models >= 3:
        results['permutation_test'] = {
            'actual_rho': float(actual_mean),
            'permuted_mean': float(np.mean(perm_dist)),
            'permuted_std': float(np.std(perm_dist)),
            'p_value': float(perm_p),
        }

    suffix = '_pilot' if args.pilot else ''
    out_path = OUT_DIR / f'results_tinystories_circuit_stability{suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")

    # Summary
    n_pass = sum(v for v in results['pre_registered_predictions'].values())
    n_total = len(results['pre_registered_predictions'])
    print(f"\n{'='*60}")
    print(f"PRE-REGISTERED PREDICTIONS: {n_pass}/{n_total} PASS")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
