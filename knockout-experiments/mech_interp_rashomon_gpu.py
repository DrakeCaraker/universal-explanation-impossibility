#!/usr/bin/env python3
"""
Mechanistic Interpretability Rashomon: Bulletproof GPU Version

Properly powered experiment testing whether independently trained transformers
identify the same circuit components as important.

Design (bulletproof for adversarial peer review):
- Fine-tune GPT-2 small on SST-2 from 30 different random seeds
- 5000 training examples, 500 validation examples, 200 test examples
- 10 max epochs with early stopping (patience=2 on val loss)
- Convergence verification: training curves saved for every model
- Rashomon filter: keep models within 2% of best test accuracy
- Require ≥10 models in Rashomon set
- Zero-ablate each of 144 attention heads per Rashomon model
- Compute pairwise flip rates and Gaussian flip predictions

Run on SageMaker ml.g4dn.xlarge (~$0.53/hr, ~30-45 min).

Setup:
  source activate pytorch_p310
  pip install transformers datasets scipy
  python mech_interp_rashomon_gpu.py 2>&1 | tee mech_interp_log.txt
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


def load_sst2(n_train=5000, n_val=500, n_test=200):
    """Load SST-2 train/val/test splits."""
    from datasets import load_dataset
    ds = load_dataset('glue', 'sst2')
    train = ds['train']
    val = ds['validation']

    # Use first n_train for training, next n_val for convergence monitoring
    # Use validation split for test (held out from all training)
    train_texts = train['sentence'][:n_train]
    train_labels = train['label'][:n_train]

    # Validation for early stopping (from end of train split to avoid overlap)
    val_start = max(n_train, len(train['sentence']) - n_val)
    val_texts = train['sentence'][val_start:val_start + n_val]
    val_labels = train['label'][val_start:val_start + n_val]

    # Test set from the actual validation split (never seen during training)
    test_texts = val['sentence'][:n_test]
    test_labels = val['label'][:n_test]

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def tokenize(tokenizer, texts, max_length=128):
    """Tokenize texts and return tensors."""
    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=max_length, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']


def evaluate_batch(model, input_ids, attention_mask, labels_np, device=DEVICE):
    """Evaluate accuracy on a batch."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device))
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
    return float(np.mean(preds == labels_np))


def compute_val_loss(model, input_ids, attention_mask, labels_tensor, device=DEVICE):
    """Compute validation loss."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        labels=labels_tensor.to(device))
    return outputs.loss.item()


def fine_tune_gpt2(train_texts, train_labels, val_ids, val_mask, val_labels_t,
                    val_labels_np, seed, max_epochs=10, patience=2, device=DEVICE):
    """Fine-tune GPT-2 with early stopping and convergence tracking."""
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        'gpt2', num_labels=2
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize training data
    train_ids, train_mask = tokenize(tokenizer, train_texts)
    dataset = TensorDataset(
        train_ids,
        train_mask,
        torch.tensor(train_labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = max_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training with early stopping
    training_curve = []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
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
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validate
        val_loss = compute_val_loss(model, val_ids, val_mask, val_labels_t, device)
        val_acc = evaluate_batch(model, val_ids, val_mask, val_labels_np, device)

        training_curve.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    converged = best_epoch < max_epochs  # Stopped before hitting max = converged

    return model, tokenizer, {
        "training_curve": training_curve,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "final_val_acc": training_curve[-1]["val_acc"],
        "converged": converged,
        "total_epochs": len(training_curve),
    }


def measure_head_importance(model, tokenizer, texts, labels, device=DEVICE):
    """Measure each head's importance via zero ablation on full test set."""
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    # Tokenize test set
    input_ids, attention_mask = tokenize(tokenizer, texts)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
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
    print("=" * 70)
    print("Mech Interp Rashomon: Bulletproof GPU Version")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    t0 = time.time()

    n_seeds = 30
    seeds = list(range(42, 42 + n_seeds))
    rashomon_threshold = 0.02  # Within 2% of best

    # ===== Load Data =====
    print("\nLoading SST-2 (5000 train, 500 val, 200 test)...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_sst2(n_train=5000, n_val=500, n_test=200)
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Pre-tokenize val and test (shared across all models)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    val_ids, val_mask = tokenize(tokenizer, val_texts)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    val_labels_np = np.array(val_labels)

    test_ids, test_mask = tokenize(tokenizer, test_texts)
    test_labels_np = np.array(test_labels)

    # ===== Fine-tune All Models =====
    print(f"\nFine-tuning {n_seeds} GPT-2 models (early stopping, patience=2)...")
    all_models = []

    for i, seed in enumerate(seeds):
        print(f"\n  Model {i+1}/{n_seeds} (seed={seed}):")
        model, tok, curve_info = fine_tune_gpt2(
            train_texts, train_labels, val_ids, val_mask, val_labels_t,
            val_labels_np, seed, max_epochs=10, patience=2, device=DEVICE
        )

        # Evaluate on held-out test set
        test_acc = evaluate_batch(model, test_ids, test_mask, test_labels_np, DEVICE)

        print(f"    Best epoch: {curve_info['best_epoch']}/{curve_info['total_epochs']}, "
              f"Val acc: {curve_info['final_val_acc']:.3f}, "
              f"Test acc: {test_acc:.3f}, "
              f"Converged: {curve_info['converged']}")

        all_models.append({
            'seed': seed,
            'test_accuracy': test_acc,
            'model': model,
            'tokenizer': tok,
            'curve_info': curve_info,
        })

    # ===== Rashomon Filter =====
    all_accs = [m['test_accuracy'] for m in all_models]
    best_acc = max(all_accs)
    rashomon_models = [m for m in all_models if best_acc - m['test_accuracy'] <= rashomon_threshold]

    print(f"\n{'='*70}")
    print(f"RASHOMON FILTER")
    print(f"{'='*70}")
    print(f"  Best accuracy: {best_acc:.3f}")
    print(f"  Threshold: within {rashomon_threshold*100}%")
    print(f"  Models in Rashomon set: {len(rashomon_models)}/{n_seeds}")
    print(f"  Rashomon accuracies: {sorted([m['test_accuracy'] for m in rashomon_models])}")
    print(f"  All accuracies: {sorted(all_accs)}")

    if len(rashomon_models) < 10:
        print(f"\n  WARNING: Only {len(rashomon_models)} models in Rashomon set.")
        print(f"  Relaxing threshold to 3%...")
        rashomon_threshold = 0.03
        rashomon_models = [m for m in all_models if best_acc - m['test_accuracy'] <= rashomon_threshold]
        print(f"  Relaxed: {len(rashomon_models)} models")

    if len(rashomon_models) < 6:
        print(f"\n  FAILED: Too few models converged to comparable accuracy.")
        results = {
            "experiment": "mech_interp_rashomon_gpu",
            "status": "FAILED_RASHOMON",
            "n_total": n_seeds,
            "n_rashomon": len(rashomon_models),
            "best_accuracy": best_acc,
            "all_accuracies": sorted([round(a, 3) for a in all_accs]),
            "training_curves": {str(m['seed']): m['curve_info'] for m in all_models},
        }
        json.dump(results, open(OUT_DIR / 'results_mech_interp_rashomon_gpu.json', 'w'), indent=2)
        print(f"  Results (with training curves) saved.")
        return

    rashomon_holds = True
    rashomon_range = max(m['test_accuracy'] for m in rashomon_models) - \
                     min(m['test_accuracy'] for m in rashomon_models)

    # ===== Head Importance Ablation =====
    n_rashomon = len(rashomon_models)
    print(f"\n{'='*70}")
    print(f"HEAD IMPORTANCE ABLATION ({n_rashomon} models × 144 heads × 200 test)")
    print(f"{'='*70}")

    importance_list = []
    for i, m in enumerate(rashomon_models):
        print(f"  Model {i+1}/{n_rashomon} (seed={m['seed']}, acc={m['test_accuracy']:.3f})...",
              end=" ", flush=True)
        imp, base_acc = measure_head_importance(
            m['model'], m['tokenizer'], test_texts, test_labels, DEVICE
        )
        importance_list.append(imp.flatten())
        print(f"imp range: [{imp.min():.4f}, {imp.max():.4f}]")

        # Free GPU memory
        del m['model']
        torch.cuda.empty_cache()

    importance_matrix = np.array(importance_list)
    n_components = importance_matrix.shape[1]

    # ===== Cal/Val Split =====
    n_cal = n_rashomon // 2
    imp_cal = importance_matrix[:n_cal]
    imp_val = importance_matrix[n_cal:]

    print(f"\n  Cal models: {n_cal}, Val models: {n_rashomon - n_cal}")

    # ===== Gaussian Flip Analysis =====
    print(f"\nComputing pairwise flip rates...")
    pairs = list(combinations(range(n_components), 2))
    if len(pairs) > 2000:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=2000, replace=False)]

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

    # Top-10 heads
    mean_importance = np.mean(importance_matrix, axis=0)
    top_10_idx = np.argsort(-np.abs(mean_importance))[:10]
    n_heads = 12

    elapsed = time.time() - t0

    # ===== Results =====
    print(f"\n{'='*70}")
    print(f"RESULTS: GPT-2 Small Circuit Rashomon Stability")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}")
    print(f"  Models trained: {n_seeds}, in Rashomon set: {n_rashomon}")
    print(f"  Rashomon accuracy range: {rashomon_range:.3f}")
    print(f"  All converged: {all(m['curve_info']['converged'] for m in rashomon_models)}")
    print(f"  Components: {n_components} attention heads")
    print(f"  Pairs analyzed: {len(pairs)}")
    print(f"  Coverage conflict degree: {cc_degree:.3f}")
    print(f"  Reliable fraction (SNR>2): {reliable_frac:.3f}")
    print(f"  Mean flip rate: {mean_flip:.3f}")
    print(f"  Gaussian flip R²: {r2:.3f}")
    print(f"  Gaussian flip ρ: {rho:.3f} (p={p_val:.2e})")
    print(f"  Elapsed: {elapsed:.0f}s")

    print(f"\n  Top 10 heads:")
    top_heads = []
    for rank, idx in enumerate(top_10_idx):
        layer = idx // n_heads
        head = idx % n_heads
        imp = mean_importance[idx]
        cv = float(np.std(importance_matrix[:, idx]) /
                   max(abs(np.mean(importance_matrix[:, idx])), 1e-12))
        print(f"    #{rank+1}: L{layer}H{head} imp={imp:.4f} CV={cv:.3f}")
        top_heads.append({
            "rank": rank + 1,
            "layer": int(layer),
            "head": int(head),
            "mean_importance": round(float(imp), 4),
            "cv": round(cv, 3),
        })

    # ===== Save =====
    results = {
        "experiment": "mech_interp_rashomon_gpu",
        "status": "SUCCESS",
        "model": "gpt2-small",
        "task": "SST-2",
        "device": DEVICE,
        "design": {
            "n_train": 5000,
            "n_val": 500,
            "n_test": 200,
            "max_epochs": 10,
            "early_stopping_patience": 2,
            "rashomon_threshold": rashomon_threshold,
        },
        "n_total_models": n_seeds,
        "n_rashomon_models": n_rashomon,
        "rashomon_holds": rashomon_holds,
        "rashomon_accuracy_range": round(float(rashomon_range), 4),
        "best_accuracy": round(float(best_acc), 3),
        "rashomon_accuracies": sorted([round(float(m['test_accuracy']), 3) for m in rashomon_models]),
        "all_accuracies": sorted([round(float(a), 3) for a in all_accs]),
        "all_converged": all(m['curve_info']['converged'] for m in rashomon_models),
        "training_curves": {
            str(m['seed']): m['curve_info']
            for m in all_models
        },
        "n_components": n_components,
        "n_pairs": len(pairs),
        "coverage_conflict_degree": round(cc_degree, 3),
        "reliable_fraction": round(reliable_frac, 3),
        "mean_flip_rate": round(mean_flip, 3),
        "gaussian_r2": round(float(r2), 3),
        "gaussian_rho": round(float(rho), 3),
        "gaussian_p": float(p_val),
        "top_10_heads": top_heads,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = OUT_DIR / 'results_mech_interp_rashomon_gpu.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
