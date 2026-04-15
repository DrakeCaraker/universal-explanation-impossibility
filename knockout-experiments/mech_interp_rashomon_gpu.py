#!/usr/bin/env python3
"""
Mechanistic Interpretability Rashomon: GPU Version for SageMaker

Properly designed experiment:
- Fine-tune GPT-2 small on SST-2 from 20 different random seeds
- Use 2000 training examples, 5 epochs (enough for convergence)
- Filter to models within 3% of best accuracy (Rashomon condition)
- Zero-ablate each of 144 attention heads per model
- Compute pairwise flip rates and Gaussian flip predictions

Run on SageMaker ml.g4dn.xlarge (~$0.53/hr, ~30-45 min).

Usage:
  # On SageMaker notebook or Processing job:
  pip install transformers datasets shap torch
  python mech_interp_rashomon_gpu.py

  # Results saved to results_mech_interp_rashomon_gpu.json
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_sst2(n_train=2000, n_test=200):
    """Load SST-2 from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset('glue', 'sst2')
    train = ds['train']
    val = ds['validation']

    train_texts = train['sentence'][:n_train]
    train_labels = train['label'][:n_train]
    test_texts = val['sentence'][:n_test]
    test_labels = val['label'][:n_test]
    return train_texts, train_labels, test_texts, test_labels


def fine_tune_gpt2(train_texts, train_labels, seed, n_epochs=5, device=DEVICE):
    """Fine-tune GPT-2 small for binary sentiment classification."""
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        'gpt2', num_labels=2
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    encodings = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=128, return_tensors='pt')
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(train_labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * len(loader))

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    return model, tokenizer


def evaluate(model, tokenizer, texts, labels, device=DEVICE):
    """Evaluate classification accuracy."""
    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(
            input_ids=encodings['input_ids'].to(device),
            attention_mask=encodings['attention_mask'].to(device)
        )
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
    return float(np.mean(preds == np.array(labels)))


def measure_head_importance(model, tokenizer, texts, labels, device=DEVICE):
    """Measure each head's importance via zero ablation."""
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    labels_np = np.array(labels)

    # Baseline accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        baseline_acc = float(np.mean(baseline_preds == labels_np))

    head_importance = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            def make_hook(head):
                def hook_fn(module, input, output):
                    attn_output = output[0]
                    start = head * head_dim
                    end = (head + 1) * head_dim
                    attn_output[:, :, start:end] = 0.0
                    return (attn_output,) + output[1:]
                return hook_fn

            hook = model.transformer.h[layer_idx].attn.register_forward_hook(
                make_hook(head_idx)
            )

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                ablated_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                ablated_acc = float(np.mean(ablated_preds == labels_np))

            hook.remove()
            head_importance[layer_idx, head_idx] = baseline_acc - ablated_acc

    return head_importance, baseline_acc


def main():
    print("Mech Interp Rashomon: GPU Version")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    t0 = time.time()

    n_seeds = 20
    seeds = list(range(42, 42 + n_seeds))
    rashomon_threshold = 0.03  # Models must be within 3% of best

    # Load data
    print("\nLoading SST-2 (2000 train, 200 test)...")
    train_texts, train_labels, test_texts, test_labels = load_sst2(n_train=2000, n_test=200)
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Fine-tune all models
    models_data = []
    print(f"\nFine-tuning {n_seeds} GPT-2 models...")

    for i, seed in enumerate(seeds):
        print(f"  Model {i+1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        model, tokenizer = fine_tune_gpt2(train_texts, train_labels, seed)
        acc = evaluate(model, tokenizer, test_texts, test_labels)
        print(f"accuracy={acc:.3f}")
        models_data.append({
            'seed': seed,
            'accuracy': acc,
            'model': model,
            'tokenizer': tokenizer,
        })

    # Filter to Rashomon set
    best_acc = max(m['accuracy'] for m in models_data)
    rashomon_models = [m for m in models_data if best_acc - m['accuracy'] <= rashomon_threshold]
    print(f"\nRashomon filter: {len(rashomon_models)}/{n_seeds} models within {rashomon_threshold*100}% of best ({best_acc:.3f})")
    print(f"  Accuracies: {[m['accuracy'] for m in rashomon_models]}")

    if len(rashomon_models) < 6:
        print("WARNING: Fewer than 6 models in Rashomon set. Relaxing threshold to 5%.")
        rashomon_threshold = 0.05
        rashomon_models = [m for m in models_data if best_acc - m['accuracy'] <= rashomon_threshold]
        print(f"  Relaxed: {len(rashomon_models)} models")

    if len(rashomon_models) < 4:
        print("ERROR: Too few models converged to comparable accuracy.")
        # Still save results
        results = {
            "experiment": "mech_interp_rashomon_gpu",
            "rashomon_holds": False,
            "n_total_models": n_seeds,
            "n_rashomon_models": len(rashomon_models),
            "best_accuracy": best_acc,
            "all_accuracies": [m['accuracy'] for m in models_data],
            "error": "Too few models in Rashomon set",
        }
        json.dump(results, open(OUT_DIR / 'results_mech_interp_rashomon_gpu.json', 'w'), indent=2)
        return

    # Measure head importance for Rashomon models
    n_test_ablation = min(100, len(test_texts))
    test_subset_texts = test_texts[:n_test_ablation]
    test_subset_labels = test_labels[:n_test_ablation]

    importance_list = []
    for i, m in enumerate(rashomon_models):
        print(f"\n  Ablation {i+1}/{len(rashomon_models)} (seed={m['seed']}, acc={m['accuracy']:.3f})...")
        imp, base_acc = measure_head_importance(
            m['model'], m['tokenizer'], test_subset_texts, test_subset_labels
        )
        importance_list.append(imp.flatten())
        # Free GPU memory
        del m['model']
        torch.cuda.empty_cache()

    importance_matrix = np.array(importance_list)  # (n_models, 144)
    n_models = importance_matrix.shape[0]
    n_components = importance_matrix.shape[1]

    # Split cal/val
    n_cal = n_models // 2
    imp_cal = importance_matrix[:n_cal]
    imp_val = importance_matrix[n_cal:]

    # Compute flip rates
    pairs = list(combinations(range(n_components), 2))
    if len(pairs) > 1000:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=1000, replace=False)]

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

    # Top-10 heads
    mean_importance = np.mean(importance_matrix, axis=0)
    top_10_idx = np.argsort(-np.abs(mean_importance))[:10]
    n_heads = 12

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS: GPT-2 Small Circuit Rashomon Stability (GPU)")
    print(f"{'='*60}")
    print(f"  Models in Rashomon set: {n_models}/{n_seeds}")
    print(f"  Accuracy range (Rashomon): {min(m['accuracy'] for m in rashomon_models):.3f} - {max(m['accuracy'] for m in rashomon_models):.3f}")
    print(f"  Components: {n_components} attention heads")
    print(f"  Pairs analyzed: {len(pairs)}")
    print(f"  Coverage conflict degree: {cc_degree:.3f}")
    print(f"  Reliable fraction (SNR>2): {reliable_frac:.3f}")
    print(f"  Mean flip rate: {mean_flip:.3f}")
    print(f"  Gaussian flip R²: {r2:.3f}")
    print(f"  Gaussian flip ρ: {rho:.3f} (p={p_val:.2e})")

    print(f"\n  Top 10 heads:")
    for rank, idx in enumerate(top_10_idx):
        layer = idx // n_heads
        head = idx % n_heads
        imp = mean_importance[idx]
        cv = float(np.std(importance_matrix[:, idx]) / max(abs(np.mean(importance_matrix[:, idx])), 1e-12))
        print(f"    #{rank+1}: L{layer}H{head} imp={imp:.4f} CV={cv:.3f}")

    results = {
        "experiment": "mech_interp_rashomon_gpu",
        "model": "gpt2-small",
        "task": "SST-2",
        "device": DEVICE,
        "n_total_models": n_seeds,
        "n_rashomon_models": n_models,
        "rashomon_threshold": rashomon_threshold,
        "rashomon_holds": True,
        "best_accuracy": best_acc,
        "rashomon_accuracies": [round(m['accuracy'], 3) for m in rashomon_models],
        "all_accuracies": [round(m['accuracy'], 3) for m in models_data],
        "n_components": n_components,
        "n_pairs": len(pairs),
        "coverage_conflict_degree": round(cc_degree, 3),
        "reliable_fraction": round(reliable_frac, 3),
        "mean_flip_rate": round(mean_flip, 3),
        "gaussian_r2": round(float(r2), 3),
        "gaussian_rho": round(float(rho), 3),
        "gaussian_p": float(p_val),
        "top_10_heads": [
            {"rank": i+1, "layer": int(idx // n_heads), "head": int(idx % n_heads),
             "mean_importance": round(float(mean_importance[idx]), 4),
             "cv": round(float(np.std(importance_matrix[:, idx]) /
                    max(abs(np.mean(importance_matrix[:, idx])), 1e-12)), 3)}
            for i, idx in enumerate(top_10_idx)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = OUT_DIR / 'results_mech_interp_rashomon_gpu.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
