#!/usr/bin/env python3
"""
Definitive MI Experiment v2: Circuit Stability From Scratch

Trains 30 small transformers from RANDOM INITIALIZATION on modular addition
(a + b mod 113), then measures circuit agreement via activation patching.

Eliminates all v1 confounds:
- No pretrained weights (0% shared, vs 99.3% in LoRA v1)
- Task requires genuine learning (grokking from ~0% to ~99%)
- No classifier head (logit for correct answer token)

Architecture: 2-layer transformer, 8 heads (4 per layer), d_model=128
Task: modular addition mod 113 (Nanda et al. 2023)
Measurement: activation patching + Fourier analysis of embeddings

Run: python mech_interp_definitive_v2.py 2>&1 | tee mi_v2_log.txt
Expected: ~3 hours on T4 GPU
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.stats import spearmanr
from itertools import combinations

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
OUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path('/home/ec2-user/SageMaker/mi_v2_models') if 'ec2-user' in str(Path.home()) else OUT_DIR / 'mi_v2_models'

# Hyperparameters (following Nanda et al. 2023)
P = 113                  # prime modulus
D_MODEL = 128            # model dimension
N_LAYERS = 2             # transformer layers
N_HEADS = 4              # heads per layer
HEAD_DIM = D_MODEL // N_HEADS  # 32
D_MLP = 4 * D_MODEL      # MLP hidden dim (512)
N_MODELS = 30
TRAIN_FRAC = 0.5
LR = 1e-3
WEIGHT_DECAY = 1.0
N_STEPS = 50000
LOG_EVERY = 1000
RASHOMON_THRESHOLD = 0.01  # 1% accuracy
N_COMPONENTS = N_LAYERS * N_HEADS  # 8 attention heads
N_COMPONENTS_TOTAL = N_LAYERS * (N_HEADS + 1)  # 8 heads + 2 MLPs = 10


# =========================================================================
# Data
# =========================================================================

def make_data():
    """Generate all (a, b, (a+b) mod p) triples and split 50/50."""
    pairs = [(a, b) for a in range(P) for b in range(P)]
    labels = [(a + b) % P for a, b in pairs]

    idx = np.random.RandomState(0).permutation(len(pairs))
    split = int(TRAIN_FRAC * len(pairs))

    train_idx, test_idx = idx[:split], idx[split:]

    def to_tensors(indices):
        x = torch.tensor([(pairs[i][0], pairs[i][1]) for i in indices], dtype=torch.long)
        y = torch.tensor([labels[i] for i in indices], dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)

    return to_tensors(train_idx), to_tensors(test_idx)


# =========================================================================
# Transformer (minimal, from scratch)
# =========================================================================

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, n_heads, S, head_dim)
        # Return per-head outputs for patching
        per_head = out.transpose(1, 2).contiguous()  # (B, S, n_heads, head_dim)
        combined = per_head.view(B, S, D)
        projected = self.W_O(combined)
        return projected, per_head


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        return self.W_out(F.gelu(self.W_in(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        # Attention with residual
        normed = self.ln1(x)
        attn_out, per_head = self.attn(normed)
        x = x + attn_out
        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x, per_head, mlp_out


class ModularAdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(P, D_MODEL)
        self.pos_embed = nn.Embedding(3, D_MODEL)  # positions 0, 1, 2 (a, b, =)
        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_MLP) for _ in range(N_LAYERS)
        ])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.unembed = nn.Linear(D_MODEL, P, bias=False)

    def forward(self, x, return_internals=False):
        B = x.shape[0]
        # x is (B, 2) containing (a, b). We append an "=" token (index 0, arbitrary)
        eq_token = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        tokens = torch.cat([x, eq_token], dim=1)  # (B, 3)

        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.tok_embed(tokens) + self.pos_embed(pos)

        all_heads = []
        all_mlps = []
        for block in self.blocks:
            h, per_head, mlp_out = block(h)
            all_heads.append(per_head)
            all_mlps.append(mlp_out)

        h = self.ln_final(h)
        logits = self.unembed(h[:, -1, :])  # predict from last position

        if return_internals:
            return logits, all_heads, all_mlps
        return logits


# =========================================================================
# Training
# =========================================================================

def train_model(seed, train_data, test_data, model_idx):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ModularAdditionTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    x_train, y_train = train_data
    x_test, y_test = test_data

    history = []
    best_test_acc = 0

    for step in range(1, N_STEPS + 1):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(x_train).argmax(-1) == y_train).float().mean().item()
                test_acc = (model(x_test).argmax(-1) == y_test).float().mean().item()

            history.append({
                'step': step,
                'loss': loss.item(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            })

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_DIR / f'model_{model_idx}.pt')

            if step % (LOG_EVERY * 10) == 0:
                print(f'    Step {step}: loss={loss.item():.4f} '
                      f'train={train_acc:.3f} test={test_acc:.3f}')

    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / f'model_{model_idx}.pt',
                                     map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        final_test_acc = (model(x_test).argmax(-1) == y_test).float().mean().item()

    return model, final_test_acc, history


# =========================================================================
# Activation Patching
# =========================================================================

def activation_patching(model, x_test, y_test, n_examples=2000):
    """
    Compute importance of each component (8 heads + 2 MLPs = 10 components).

    For each component:
    1. Run clean forward pass
    2. Run corrupted forward pass (random input)
    3. Patch: replace clean component output with corrupted
    4. Measure: change in logit for correct answer
    """
    model.eval()
    n = min(n_examples, len(x_test))
    idx = np.random.RandomState(42).choice(len(x_test), n, replace=False)
    x_clean = x_test[idx]
    y_clean = y_test[idx]

    # Corrupted inputs: shuffle BOTH a and b independently
    x_corrupt = x_clean.clone()
    perm_a = torch.randperm(n, device=DEVICE)
    perm_b = torch.randperm(n, device=DEVICE)
    x_corrupt[:, 0] = x_clean[perm_a, 0]
    x_corrupt[:, 1] = x_clean[perm_b, 1]

    # Clean forward pass
    with torch.no_grad():
        clean_logits, clean_heads, clean_mlps = model(x_clean, return_internals=True)
        corrupt_logits, corrupt_heads, corrupt_mlps = model(x_corrupt, return_internals=True)

    # Clean logit for correct answer
    clean_correct_logit = clean_logits[torch.arange(n), y_clean]

    importance = np.zeros(N_COMPONENTS_TOTAL)

    # Patch each attention head (hook-based, no deepcopy)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            comp_idx = layer * N_HEADS + head

            def make_hook(head_idx, corrupt_head_out):
                def hook_fn(module, input, output):
                    projected, per_head = output
                    per_head_patched = per_head.clone()
                    per_head_patched[:, :, head_idx, :] = corrupt_head_out[:, :, head_idx, :]
                    B, S, NH, HD = per_head_patched.shape
                    combined = per_head_patched.view(B, S, NH * HD)
                    new_projected = module.W_O(combined)
                    delta = new_projected - projected
                    return (projected + delta, per_head_patched)
                return hook_fn

            handle = model.blocks[layer].attn.register_forward_hook(
                make_hook(head, corrupt_heads[layer])
            )

            with torch.no_grad():
                patched_logits = model(x_clean, return_internals=False)

            handle.remove()

            patched_correct_logit = patched_logits[torch.arange(n), y_clean]
            importance[comp_idx] = (clean_correct_logit - patched_correct_logit).mean().item()

    # Patch each MLP (hook-based, no deepcopy)
    for layer in range(N_LAYERS):
        comp_idx = N_COMPONENTS + layer

        def make_mlp_hook(corrupt_mlp_out):
            def hook_fn(module, input, output):
                h, per_head, mlp_out = output
                delta = corrupt_mlp_out - mlp_out
                return (h + delta, per_head, corrupt_mlp_out)
            return hook_fn

        handle = model.blocks[layer].register_forward_hook(
            make_mlp_hook(corrupt_mlps[layer])
        )

        with torch.no_grad():
            patched_logits = model(x_clean, return_internals=False)

        handle.remove()

        patched_correct_logit = patched_logits[torch.arange(n), y_clean]
        importance[comp_idx] = (clean_correct_logit - patched_correct_logit).mean().item()

    return importance


# =========================================================================
# Fourier Analysis
# =========================================================================

def fourier_analysis(model):
    """Extract dominant Fourier frequencies from token embeddings."""
    embed = model.tok_embed.weight.detach().cpu().numpy()  # (P, D_MODEL)

    # DFT of each embedding dimension
    freqs = np.fft.fft(embed, axis=0)
    power = np.abs(freqs) ** 2

    # Average power per frequency across dimensions
    avg_power = power.mean(axis=1)  # (P,)

    # Top 5 frequencies (excluding DC component at index 0)
    top_freq_idx = np.argsort(avg_power[1:])[-5:][::-1] + 1
    top_freq_power = avg_power[top_freq_idx]

    return {
        'top_frequencies': top_freq_idx.tolist(),
        'top_powers': top_freq_power.tolist(),
        'total_power': float(avg_power.sum()),
        'dc_power': float(avg_power[0]),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    start = time.time()

    print("=" * 70)
    print("DEFINITIVE MI EXPERIMENT v2: Modular Addition From Scratch")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Architecture: {N_LAYERS}-layer transformer, {N_HEADS} heads/layer, d={D_MODEL}")
    print(f"Task: (a + b) mod {P}")
    print(f"Models: {N_MODELS} from random init")
    print(f"Training: {N_STEPS} steps, lr={LR}, wd={WEIGHT_DECAY}")
    print(f"Components: {N_COMPONENTS} heads + {N_LAYERS} MLPs = {N_COMPONENTS_TOTAL}")
    print()

    # Generate data
    (x_train, y_train), (x_test, y_test) = make_data()
    print(f"Data: {len(x_train)} train, {len(x_test)} test")

    # ===== Phase 1: Train 30 models =====
    print(f"\n{'='*70}")
    print(f"PHASE 1: Training {N_MODELS} models from random initialization")
    print(f"{'='*70}")

    checkpoint_path = OUT_DIR / 'mi_v2_phase1_checkpoint.json'
    models = []
    accuracies = []
    histories = []
    fourier_results = []
    grokked = []

    # Resume from checkpoint if available
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = json.load(open(checkpoint_path))
        accuracies = ckpt['accuracies']
        grokked = ckpt['grokked']
        fourier_results = ckpt['fourier_results']
        histories = ckpt.get('histories', [[] for _ in range(len(accuracies))])
        start_idx = len(accuracies)
        print(f"  Resuming from checkpoint: {start_idx}/{N_MODELS} models done")

    for i in range(start_idx, N_MODELS):
        seed = i
        print(f"\n  Model {i+1}/{N_MODELS} (seed={seed})...")

        model, acc, history = train_model(seed, (x_train, y_train), (x_test, y_test), i)
        models.append(None)  # Don't keep in memory — reload for patching
        accuracies.append(acc)
        histories.append(history)
        fourier_results.append(fourier_analysis(model))

        did_grok = acc > 0.95
        grokked.append(did_grok)
        print(f"    Final test accuracy: {acc:.4f} {'(GROKKED)' if did_grok else '(NOT grokked)'}")

        # Free memory
        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

        # Checkpoint after each model
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'accuracies': accuracies,
                'grokked': grokked,
                'fourier_results': fourier_results,
            }, f, cls=NpEncoder)

    print(f"\n  Grokked: {sum(grokked)}/{N_MODELS}")
    print(f"  Accuracies: {[f'{a:.3f}' for a in sorted(accuracies)]}")

    # Rashomon filter (only grokked models)
    best_acc = max(accuracies)
    rashomon_idx = [i for i, a in enumerate(accuracies)
                    if a >= best_acc - RASHOMON_THRESHOLD and grokked[i]]
    rashomon_accs = [accuracies[i] for i in rashomon_idx]

    print(f"\n  Rashomon set ({RASHOMON_THRESHOLD*100:.0f}% threshold, grokked only): "
          f"{len(rashomon_idx)}/{N_MODELS}")
    if rashomon_accs:
        print(f"  Accuracy range: {max(rashomon_accs)-min(rashomon_accs):.4f}")

    if len(rashomon_idx) < 5:
        print("\n  WARNING: Too few grokked models for meaningful analysis.")
        print("  Consider increasing N_STEPS or N_MODELS.")

    # ===== Phase 2: Activation patching =====
    print(f"\n{'='*70}")
    print(f"PHASE 2: Activation patching ({len(rashomon_idx)} models × "
          f"{N_COMPONENTS_TOTAL} components)")
    print(f"{'='*70}")

    importance_vectors = {}
    for idx in rashomon_idx:
        print(f"\n  Patching model {idx} (acc={accuracies[idx]:.4f})...")

        model = ModularAdditionTransformer().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / f'model_{idx}.pt',
                                         map_location=DEVICE))
        model.eval()

        importance = activation_patching(model, x_test, y_test)
        importance_vectors[idx] = importance.tolist()

        head_labels = [f'L{l}H{h}' for l in range(N_LAYERS) for h in range(N_HEADS)]
        mlp_labels = [f'MLP{l}' for l in range(N_LAYERS)]
        all_labels = head_labels + mlp_labels
        top5 = np.argsort(np.abs(importance))[-5:][::-1]
        print(f"    Top 5: {', '.join(f'{all_labels[j]}={importance[j]:.4f}' for j in top5)}")

        del model
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # ===== Phase 3: Controls =====
    print(f"\n{'='*70}")
    print(f"PHASE 3: Controls")
    print(f"{'='*70}")

    # Control A: Pre-grokking snapshot
    print("\n  Control A: Pre-grokking snapshot (step 2000)...")
    pre_grok_importance = {}
    for idx in rashomon_idx[:10]:  # 10 models for C(10,2)=45 pairs
        torch.manual_seed(idx)
        np.random.seed(idx)
        model = ModularAdditionTransformer().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Train only 2000 steps (pre-grokking)
        model.train()
        for step in range(2000):
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pre_acc = (model(x_test).argmax(-1) == y_test).float().mean().item()

        imp = activation_patching(model, x_test, y_test, n_examples=500)
        pre_grok_importance[idx] = imp.tolist()
        print(f"    Model {idx}: test_acc={pre_acc:.3f}")
        del model, optimizer
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Pre-grokking pairwise correlation
    pre_keys = list(pre_grok_importance.keys())
    pre_rhos = []
    for i, j in combinations(range(len(pre_keys)), 2):
        rho, _ = spearmanr(pre_grok_importance[pre_keys[i]],
                           pre_grok_importance[pre_keys[j]])
        pre_rhos.append(rho)
    pre_grok_mean_rho = float(np.mean(pre_rhos)) if pre_rhos else 0.0
    print(f"  Pre-grokking mean Spearman rho: {pre_grok_mean_rho:.3f}")

    # Control B: Determinism
    print("\n  Control B: Determinism check...")
    torch.manual_seed(rashomon_idx[0])
    np.random.seed(rashomon_idx[0])
    model_b = ModularAdditionTransformer().to(DEVICE)
    model_b.load_state_dict(torch.load(MODEL_DIR / f'model_{rashomon_idx[0]}.pt',
                                       map_location=DEVICE))
    model_b.eval()
    imp_b1 = activation_patching(model_b, x_test, y_test, n_examples=500)
    imp_b2 = activation_patching(model_b, x_test, y_test, n_examples=500)
    det_corr = float(np.corrcoef(imp_b1, imp_b2)[0, 1])
    print(f"  Same model, same data: r = {det_corr:.6f}")
    del model_b

    # ===== Phase 4: Analysis =====
    print(f"\n{'='*70}")
    print(f"PHASE 4: Analysis")
    print(f"{'='*70}")

    indices = list(importance_vectors.keys())
    vectors = [np.array(importance_vectors[i]) for i in indices]

    # Pairwise Spearman
    spearman_rhos = []
    for i, j in combinations(range(len(indices)), 2):
        rho, _ = spearmanr(vectors[i], vectors[j])
        spearman_rhos.append(rho)

    mean_rho = float(np.mean(spearman_rhos))
    std_rho = float(np.std(spearman_rhos))
    min_rho = float(np.min(spearman_rhos))
    max_rho = float(np.max(spearman_rhos))

    # Flip rate
    flip_rates = []
    for i, j in combinations(range(len(indices)), 2):
        n_flips = 0
        n_pairs = 0
        for h1, h2 in combinations(range(N_COMPONENTS_TOTAL), 2):
            if abs(vectors[i][h1] - vectors[i][h2]) > 1e-8 and \
               abs(vectors[j][h1] - vectors[j][h2]) > 1e-8:
                sign_i = np.sign(vectors[i][h1] - vectors[i][h2])
                sign_j = np.sign(vectors[j][h1] - vectors[j][h2])
                if sign_i != sign_j:
                    n_flips += 1
                n_pairs += 1
        flip_rates.append(n_flips / n_pairs if n_pairs > 0 else 0)

    mean_flip = float(np.mean(flip_rates))

    # Jaccard on top-K
    jaccard_results = {}
    for K in [3, 5, 8]:
        jaccards = []
        for i, j in combinations(range(len(indices)), 2):
            top_i = set(np.argsort(np.abs(vectors[i]))[-K:])
            top_j = set(np.argsort(np.abs(vectors[j]))[-K:])
            jacc = len(top_i & top_j) / len(top_i | top_j) if len(top_i | top_j) > 0 else 0
            jaccards.append(jacc)
        jaccard_results[K] = {
            'mean': float(np.mean(jaccards)),
            'std': float(np.std(jaccards)),
            'min': float(np.min(jaccards)),
            'max': float(np.max(jaccards)),
        }

    # Fourier frequency agreement
    fourier_freqs_by_model = {}
    for idx in rashomon_idx:
        fourier_freqs_by_model[idx] = fourier_results[idx]['top_frequencies']

    freq_jaccards = []
    for i, j in combinations(rashomon_idx, 2):
        fi = set(fourier_freqs_by_model[i])
        fj = set(fourier_freqs_by_model[j])
        jacc = len(fi & fj) / len(fi | fj) if len(fi | fj) > 0 else 0
        freq_jaccards.append(jacc)
    mean_freq_jaccard = float(np.mean(freq_jaccards)) if freq_jaccards else 0.0

    # Gaussian prediction
    from scipy.stats import norm
    snr_vals = []
    obs_flips = []
    for h1, h2 in combinations(range(N_COMPONENTS_TOTAL), 2):
        diffs = [v[h1] - v[h2] for v in vectors]
        mu = np.mean(diffs)
        sigma = np.std(diffs)
        if sigma > 1e-10:
            snr_vals.append(abs(mu) / sigma)
            flips = sum(1 for k in range(len(vectors)-1)
                       for l in range(k+1, len(vectors))
                       if np.sign(vectors[k][h1] - vectors[k][h2]) !=
                          np.sign(vectors[l][h1] - vectors[l][h2]))
            total = len(vectors) * (len(vectors) - 1) // 2
            obs_flips.append(flips / total if total > 0 else 0)

    pred_flips = [norm.cdf(-s) for s in snr_vals]
    if len(snr_vals) > 5:
        gauss_rho, gauss_p = spearmanr(pred_flips, obs_flips)
    else:
        gauss_rho, gauss_p = 0.0, 1.0

    # ===== Verdict =====
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"\n  Rashomon set: {len(indices)} models")
    if rashomon_accs:
        print(f"  Accuracy range: {max(rashomon_accs)-min(rashomon_accs):.4f}")

    print(f"\n  PRIMARY METRIC (post-grokking):")
    print(f"    Mean Spearman rho: {mean_rho:.3f} ± {std_rho:.3f}")
    print(f"    Range: [{min_rho:.3f}, {max_rho:.3f}]")
    print(f"    Mean flip rate: {mean_flip:.3f}")

    print(f"\n  PRE-GROKKING CONTROL:")
    print(f"    Mean Spearman rho: {pre_grok_mean_rho:.3f}")
    print(f"    (If pre-grok rho ≈ post-grok rho, measurement is insensitive)")

    print(f"\n  JACCARD (top-K):")
    for K, res in jaccard_results.items():
        print(f"    K={K}: {res['mean']:.3f} ± {res['std']:.3f} "
              f"[{res['min']:.3f}, {res['max']:.3f}]")

    print(f"\n  FOURIER FREQUENCY AGREEMENT:")
    print(f"    Jaccard (top-5 frequencies): {mean_freq_jaccard:.3f}")

    print(f"\n  GAUSSIAN PREDICTION:")
    print(f"    Spearman rho: {gauss_rho:.3f} (p={gauss_p:.4f})")

    # Determine verdict
    if mean_rho > 0.8 and pre_grok_mean_rho < 0.5:
        verdict = "CIRCUITS_STABLE"
        interp = ("Post-grokking circuits are conserved across seeds (rho={:.3f}). "
                  "Pre-grokking circuits are NOT conserved (rho={:.3f}). "
                  "The stability is a property of the learned algorithm, not "
                  "the measurement tool.".format(mean_rho, pre_grok_mean_rho))
    elif mean_rho > 0.8 and pre_grok_mean_rho > 0.5:
        verdict = "MEASUREMENT_INSENSITIVE"
        interp = ("Both pre-grokking (rho={:.3f}) and post-grokking (rho={:.3f}) "
                  "show high correlation. The measurement tool may not distinguish "
                  "learned circuits from random structure.".format(
                      pre_grok_mean_rho, mean_rho))
    elif mean_rho < 0.3:
        verdict = "CIRCUITS_NONUNIQUE"
        interp = ("Post-grokking circuits differ across seeds (rho={:.3f}). "
                  "Genuine Rashomon for mechanistic interpretability.".format(mean_rho))
    else:
        verdict = "PARTIAL_STABILITY"
        interp = ("Moderate circuit agreement (rho={:.3f}). Some components "
                  "conserved (skeleton), others vary.".format(mean_rho))

    print(f"\n  VERDICT: {verdict}")
    print(f"  {interp}")

    elapsed = time.time() - start
    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Save results
    results = {
        'experiment': 'mech_interp_definitive_v2',
        'status': 'SUCCESS',
        'architecture': f'{N_LAYERS}-layer transformer, {N_HEADS} heads/layer, d={D_MODEL}',
        'task': f'modular addition mod {P}',
        'training': 'from random initialization (NO pretrained weights)',
        'measurement': 'activation patching (heads + MLPs)',
        'confounds_eliminated': [
            'No pretrained weights (0% shared, vs 99.3% in LoRA v1)',
            'Task requires genuine learning (grokking)',
            'No classifier head (logit for correct token)',
        ],
        'n_total_models': N_MODELS,
        'n_grokked': sum(grokked),
        'n_rashomon_models': len(rashomon_idx),
        'rashomon_threshold': RASHOMON_THRESHOLD,
        'rashomon_accuracy_range': float(max(rashomon_accs) - min(rashomon_accs)) if rashomon_accs else 0,
        'all_accuracies': accuracies,
        'rashomon_accuracies': rashomon_accs,
        'grokked': grokked,
        'primary_metric': {
            'mean_spearman_rho': mean_rho,
            'std_spearman_rho': std_rho,
            'min_spearman_rho': min_rho,
            'max_spearman_rho': max_rho,
            'all_spearman_rhos': [float(r) for r in spearman_rhos],
        },
        'flip_rate': {
            'mean': mean_flip,
            'all': [float(f) for f in flip_rates],
        },
        'jaccard': jaccard_results,
        'fourier': {
            'mean_frequency_jaccard': mean_freq_jaccard,
            'per_model': {str(k): v for k, v in zip(rashomon_idx,
                         [fourier_results[i] for i in rashomon_idx])},
        },
        'gaussian_prediction': {
            'spearman_rho': float(gauss_rho),
            'spearman_p': float(gauss_p),
        },
        'controls': {
            'A_pre_grokking': {
                'mean_spearman_rho': pre_grok_mean_rho,
                'n_models': len(pre_grok_importance),
                'interpretation': 'LOW rho = measurement distinguishes learned from random circuits',
            },
            'B_determinism': {
                'correlation': det_corr,
            },
        },
        'verdict': verdict,
        'interpretation': interp,
        'importance_vectors': {str(k): v for k, v in importance_vectors.items()},
        'elapsed_seconds': elapsed,
    }

    out_path = OUT_DIR / 'results_mech_interp_definitive_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
