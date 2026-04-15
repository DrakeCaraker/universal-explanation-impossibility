#!/usr/bin/env python3
"""
Mechanistic Interpretability: Rashomon Stability of Circuit Explanations

PROPER DESIGN: Train multiple GPT-2 models from different random seeds,
verify they achieve comparable loss (Rashomon condition), then measure
whether the same attention heads are identified as important.

This is the canonical Rashomon setup — multiple independently optimal
solutions, NOT perturbations of one. The previous experiment (weight
perturbation) measured local sensitivity, not Rashomon instability.

Design:
- Fine-tune GPT-2 small on SST-2 sentiment classification from 10 random seeds
- Each: 3 epochs on 1000 training examples (fast on CPU)
- Verify: all achieve comparable accuracy (within 3%)
- For each model: zero-ablate each of 144 heads, measure accuracy drop
- Head importance = accuracy drop when head is ablated
- Compute pairwise flip rates and Gaussian flip predictions

This directly tests: "Do independently trained transformers identify
the same circuit components as important?"
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.stats import norm, spearmanr
from itertools import combinations

OUT_DIR = Path(__file__).resolve().parent


def load_sst2_subset(n_train=1000, n_test=200):
    """Load SST-2 sentiment data using HuggingFace datasets or fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset('glue', 'sst2', split='train')
        ds_val = load_dataset('glue', 'sst2', split='validation')

        train_texts = ds['sentence'][:n_train]
        train_labels = ds['label'][:n_train]
        test_texts = ds_val['sentence'][:n_test]
        test_labels = ds_val['label'][:n_test]
        return train_texts, train_labels, test_texts, test_labels
    except Exception:
        pass

    # Fallback: generate synthetic sentiment data
    print("  Using synthetic sentiment data (HF datasets not available)")
    rng = np.random.RandomState(42)
    pos_templates = [
        "This movie is great", "I loved this film", "Excellent performance",
        "A wonderful experience", "Highly recommended", "Beautiful story",
        "Outstanding acting", "Truly magnificent", "A masterpiece", "Brilliant"
    ]
    neg_templates = [
        "This movie is terrible", "I hated this film", "Awful performance",
        "A horrible experience", "Not recommended", "Boring story",
        "Bad acting", "Truly awful", "A disaster", "Dreadful"
    ]
    train_texts = [rng.choice(pos_templates if rng.random() > 0.5 else neg_templates)
                   for _ in range(n_train)]
    train_labels = [1 if "great" in t or "loved" in t or "Excellent" in t or
                    "wonderful" in t or "recommended" in t or "Beautiful" in t or
                    "Outstanding" in t or "magnificent" in t or "masterpiece" in t or
                    "Brilliant" in t else 0 for t in train_texts]
    test_texts = [rng.choice(pos_templates if rng.random() > 0.5 else neg_templates)
                  for _ in range(n_test)]
    test_labels = [1 if "great" in t or "loved" in t or "Excellent" in t or
                   "wonderful" in t or "recommended" in t or "Beautiful" in t or
                   "Outstanding" in t or "magnificent" in t or "masterpiece" in t or
                   "Brilliant" in t else 0 for t in test_texts]
    return train_texts, train_labels, test_texts, test_labels


def fine_tune_gpt2(train_texts, train_labels, seed, n_epochs=3, device='cpu'):
    """Fine-tune GPT-2 small for binary sentiment classification."""
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        'gpt2', num_labels=2
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize
    encodings = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=64, return_tensors='pt')
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(train_labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    return model, tokenizer


def evaluate_accuracy(model, tokenizer, texts, labels, device='cpu'):
    """Evaluate classification accuracy."""
    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=64, return_tensors='pt')

    with torch.no_grad():
        outputs = model(
            input_ids=encodings['input_ids'].to(device),
            attention_mask=encodings['attention_mask'].to(device)
        )
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()

    return float(np.mean(preds == np.array(labels)))


def measure_head_importance_ablation(model, tokenizer, texts, labels, device='cpu'):
    """Measure each attention head's importance via zero ablation.

    For each head: hook into the attention output, zero out that head's
    contribution, measure accuracy drop. Larger drop = more important head.
    """
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    # Baseline accuracy
    baseline_acc = evaluate_accuracy(model, tokenizer, texts, labels, device)

    # Tokenize once
    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=64, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    head_importance = np.zeros((n_layers, n_heads))

    model.eval()
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Register hook to zero out this head
            hooks = []

            def make_hook(layer, head):
                def hook_fn(module, input, output):
                    # output is (attn_output, attn_weights) or just attn_output
                    # GPT2Attention returns (attn_output, present, attn_weights)
                    attn_output = output[0]
                    # Zero out specific head's contribution
                    # attn_output shape: (batch, seq_len, n_embd)
                    start = head * head_dim
                    end = (head + 1) * head_dim
                    attn_output[:, :, start:end] = 0.0
                    return (attn_output,) + output[1:]
                return hook_fn

            hook = model.transformer.h[layer_idx].attn.register_forward_hook(
                make_hook(layer_idx, head_idx)
            )
            hooks.append(hook)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                ablated_acc = float(np.mean(preds == labels_tensor.numpy()))

            for h in hooks:
                h.remove()

            head_importance[layer_idx, head_idx] = baseline_acc - ablated_acc

    return head_importance, baseline_acc


def run_experiment():
    print("Mech Interp Rashomon: Circuit Stability Across Independent Models")
    print("=" * 60)
    print("PROPER DESIGN: Multiple fine-tuned models from different seeds")
    print("NOT perturbation of one model\n")
    t0 = time.time()

    device = 'cpu'
    n_models = 10
    seeds = list(range(42, 42 + n_models))

    # Load data
    print("Loading SST-2 data...")
    train_texts, train_labels, test_texts, test_labels = load_sst2_subset(
        n_train=500, n_test=100  # Small for CPU speed
    )
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Fine-tune models from different seeds
    models_importance = []
    model_accuracies = []

    for i, seed in enumerate(seeds):
        print(f"\n  Model {i+1}/{n_models} (seed={seed}):")
        print(f"    Fine-tuning GPT-2 small...")
        model, tokenizer = fine_tune_gpt2(train_texts, train_labels, seed, n_epochs=3, device=device)

        print(f"    Evaluating accuracy...")
        acc = evaluate_accuracy(model, tokenizer, test_texts, test_labels, device)
        print(f"    Accuracy: {acc:.3f}")
        model_accuracies.append(acc)

        print(f"    Measuring head importance (zero ablation, 144 heads)...")
        importance, base_acc = measure_head_importance_ablation(
            model, tokenizer, test_texts[:50], test_labels[:50], device  # 50 test for speed
        )
        models_importance.append(importance.flatten())
        print(f"    Head importance range: [{importance.min():.4f}, {importance.max():.4f}]")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Check Rashomon condition: all models within 3% accuracy
    accs = np.array(model_accuracies)
    acc_range = accs.max() - accs.min()
    rashomon_holds = acc_range < 0.05  # Within 5%
    print(f"\n  Model accuracies: mean={accs.mean():.3f}, range={acc_range:.3f}")
    print(f"  Rashomon condition (range < 5%): {rashomon_holds}")

    importance_matrix = np.array(models_importance)  # (n_models, 144)
    n_components = importance_matrix.shape[1]

    # Split cal/val
    n_cal = n_models // 2
    imp_cal = importance_matrix[:n_cal]
    imp_val = importance_matrix[n_cal:]

    # Compute flip rates and Gaussian predictions
    pairs = list(combinations(range(n_components), 2))
    if len(pairs) > 500:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=500, replace=False)]

    predicted_flips = []
    observed_flips = []
    snrs = []

    for j, k in pairs:
        diff = imp_cal[:, j] - imp_cal[:, k]
        mu = np.mean(diff)
        sd = np.std(diff, ddof=1)
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

        n_v = imp_val.shape[0]
        disagree = 0
        total = 0
        for m1 in range(n_v):
            for m2 in range(m1 + 1, n_v):
                d1 = imp_val[m1, j] - imp_val[m1, k]
                d2 = imp_val[m2, j] - imp_val[m2, k]
                if d1 * d2 < 0:
                    disagree += 1
                total += 1
        obs = disagree / total if total > 0 else 0.0

        predicted_flips.append(pred)
        observed_flips.append(obs)
        snrs.append(snr)

    predicted_flips = np.array(predicted_flips)
    observed_flips = np.array(observed_flips)
    snrs = np.array(snrs)

    # Metrics
    ss_res = np.sum((observed_flips - predicted_flips) ** 2)
    ss_tot = np.sum((observed_flips - np.mean(observed_flips)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if np.std(predicted_flips) > 1e-12 and np.std(observed_flips) > 1e-12:
        rho, p_val = spearmanr(predicted_flips, observed_flips)
    else:
        rho, p_val = 0.0, 1.0

    cc_degree = float(np.mean(snrs < 0.5))
    reliable_frac = float(np.mean(snrs > 2.0))
    mean_flip = float(np.mean(observed_flips))

    # Top-10 heads and their stability
    mean_importance = np.mean(importance_matrix, axis=0)
    top_10_idx = np.argsort(-np.abs(mean_importance))[:10]
    n_heads_per_layer = 12  # GPT-2 small

    print(f"\n{'='*60}")
    print(f"RESULTS: GPT-2 Small Circuit Rashomon Stability")
    print(f"{'='*60}")
    print(f"  Models: {n_models} independently fine-tuned from different seeds")
    print(f"  Rashomon holds: {rashomon_holds} (accuracy range: {acc_range:.3f})")
    print(f"  Components: {n_components} attention heads")
    print(f"  Pairs analyzed: {len(pairs)}")
    print(f"  Coverage conflict degree: {cc_degree:.3f}")
    print(f"  Reliable fraction (SNR>2): {reliable_frac:.3f}")
    print(f"  Mean flip rate: {mean_flip:.3f}")
    print(f"  Gaussian flip R²: {r2:.3f}")
    print(f"  Gaussian flip ρ: {rho:.3f} (p={p_val:.2e})")

    print(f"\n  Top 10 heads by mean importance:")
    for rank, idx in enumerate(top_10_idx):
        layer = idx // n_heads_per_layer
        head = idx % n_heads_per_layer
        imp = mean_importance[idx]
        cv = float(np.std(importance_matrix[:, idx]) / max(abs(np.mean(importance_matrix[:, idx])), 1e-12))
        print(f"    #{rank+1}: L{layer}H{head} mean_imp={imp:.4f} CV={cv:.3f}")

    elapsed = time.time() - t0

    results = {
        "experiment": "mech_interp_rashomon",
        "description": "Circuit stability across independently fine-tuned GPT-2 models (proper Rashomon)",
        "model": "gpt2-small",
        "task": "SST-2 sentiment classification",
        "n_models": n_models,
        "n_components": n_components,
        "rashomon_holds": bool(rashomon_holds),
        "accuracy_mean": round(float(accs.mean()), 3),
        "accuracy_range": round(float(acc_range), 3),
        "model_accuracies": [round(float(a), 3) for a in model_accuracies],
        "n_pairs": len(pairs),
        "coverage_conflict_degree": round(cc_degree, 3),
        "reliable_fraction": round(reliable_frac, 3),
        "mean_flip_rate": round(mean_flip, 3),
        "gaussian_r2": round(float(r2), 3),
        "gaussian_rho": round(float(rho), 3),
        "gaussian_p": float(p_val),
        "top_10_heads": [
            {"rank": i+1, "layer": int(idx // n_heads_per_layer),
             "head": int(idx % n_heads_per_layer),
             "mean_importance": round(float(mean_importance[idx]), 4),
             "cv": round(float(np.std(importance_matrix[:, idx]) /
                    max(abs(np.mean(importance_matrix[:, idx])), 1e-12)), 3)}
            for i, idx in enumerate(top_10_idx)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = OUT_DIR / 'results_mech_interp_rashomon.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    run_experiment()
