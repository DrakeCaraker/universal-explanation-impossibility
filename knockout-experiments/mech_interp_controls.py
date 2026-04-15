#!/usr/bin/env python3
"""
Mech Interp Controls: Same-seed + Frozen-transformer

Run AFTER mech_interp_rashomon_gpu.py finishes.

Control A: Same seed for transformer, different seed for classifier head.
  Tests whether instability comes from transformer circuits or classifier.

Control B: Frozen transformer, only train classifier head.
  Tests whether ablation importance measures circuits or classifier readout.

Both use the same SST-2 data and ablation methodology as the main experiment.
Expect ~8 min total on T4 GPU.
"""

import warnings
warnings.filterwarnings('ignore')

import json, os, csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.stats import norm, spearmanr
from itertools import combinations

OUT_DIR = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_sst2_tsv(n_train=5000, n_val=500, n_test=200):
    train_texts, train_labels = [], []
    with open('SST-2/train.tsv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            train_texts.append(row['sentence'])
            train_labels.append(int(row['label']))
    test_texts, test_labels = [], []
    with open('SST-2/dev.tsv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            test_texts.append(row['sentence'])
            test_labels.append(int(row['label']))
    vs = min(n_train, len(train_texts) - n_val)
    vt = train_texts[vs:vs+n_val]
    vl = train_labels[vs:vs+n_val]
    return (train_texts[:n_train], train_labels[:n_train],
            vt, vl, test_texts[:n_test], test_labels[:n_test])


def tokenize(tokenizer, texts, max_length=128):
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=max_length, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


def fine_tune(train_texts, train_labels, val_ids, val_mask, val_labels_np,
              transformer_seed, classifier_seed, freeze_transformer=False,
              max_epochs=10, patience=2, device=DEVICE):
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

    # Set transformer seed
    torch.manual_seed(transformer_seed)
    np.random.seed(transformer_seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(transformer_seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Re-initialize classifier head with different seed
    torch.manual_seed(classifier_seed)
    model.score.weight.data.normal_(mean=0.0, std=0.02)

    if freeze_transformer:
        model.transformer.requires_grad_(False)

    train_ids, train_mask = tokenize(tokenizer, train_texts)
    dataset = TensorDataset(train_ids, train_mask,
                            torch.tensor(train_labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(transformer_seed))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            ids, mask, labels = [b.to(device) for b in batch]
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            out = model(input_ids=val_ids.to(device), attention_mask=val_mask.to(device),
                        labels=torch.tensor(val_labels_np, dtype=torch.long).to(device))
            vl = out.loss.item()

        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, tokenizer


def measure_head_importance(model, tokenizer, texts, labels, device=DEVICE):
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    input_ids, attention_mask = tokenize(tokenizer, texts)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels_np = np.array(labels)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_acc = float(np.mean(out.logits.argmax(dim=-1).cpu().numpy() == labels_np))

    importance = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            def make_hook(h):
                def hook_fn(module, inp, output):
                    attn_out = output[0]
                    s, e = h * head_dim, (h + 1) * head_dim
                    attn_out[:, :, s:e] = 0.0
                    return (attn_out,) + output[1:]
                return hook_fn

            hook = model.transformer.h[layer].attn.register_forward_hook(make_hook(head))
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                abl_acc = float(np.mean(out.logits.argmax(dim=-1).cpu().numpy() == labels_np))
            hook.remove()
            importance[layer, head] = baseline_acc - abl_acc

    return importance.flatten(), baseline_acc


def compute_flip_stats(importance_matrix):
    n_models, n_comp = importance_matrix.shape
    n_cal = n_models // 2
    imp_cal = importance_matrix[:n_cal]
    imp_val = importance_matrix[n_cal:]

    pairs = list(combinations(range(n_comp), 2))
    if len(pairs) > 1000:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=1000, replace=False)]

    predicted, observed, snrs = [], [], []
    for j, k in pairs:
        diff = imp_cal[:, j] - imp_cal[:, k]
        mu, sd = np.mean(diff), np.std(diff, ddof=1)
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

        dis, tot = 0, 0
        for m1 in range(imp_val.shape[0]):
            for m2 in range(m1+1, imp_val.shape[0]):
                if (imp_val[m1,j]-imp_val[m1,k]) * (imp_val[m2,j]-imp_val[m2,k]) < 0:
                    dis += 1
                tot += 1
        obs = dis / tot if tot > 0 else 0.0
        predicted.append(pred); observed.append(obs); snrs.append(snr)

    predicted, observed, snrs = np.array(predicted), np.array(observed), np.array(snrs)
    rho, p = spearmanr(predicted, observed) if np.std(predicted) > 1e-12 else (0, 1)

    return {
        'n_pairs': len(pairs),
        'mean_flip_rate': round(float(np.mean(observed)), 4),
        'coverage_conflict': round(float(np.mean(snrs < 0.5)), 3),
        'reliable_fraction': round(float(np.mean(snrs > 2.0)), 3),
        'spearman_rho': round(float(rho), 3),
        'spearman_p': float(p),
    }


def main():
    print("=" * 60)
    print("MECH INTERP CONTROLS")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    t0 = time.time()

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_sst2_tsv(n_train=5000, n_val=500, n_test=200)

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    val_ids, val_mask = tokenize(tokenizer, val_texts)
    val_labels_np = np.array(val_labels)

    results = {}

    # ===== Control A: Same transformer seed, different classifier seeds =====
    print("\n" + "=" * 60)
    print("CONTROL A: Same transformer seed (42), different classifier heads")
    print("=" * 60)

    n_control = 10
    ctrl_a_imps = []
    ctrl_a_accs = []

    for i in range(n_control):
        cls_seed = 1000 + i
        print(f"  Model {i+1}/{n_control} (transformer_seed=42, classifier_seed={cls_seed})...",
              end=" ", flush=True)
        model, tok = fine_tune(
            train_texts, train_labels, val_ids, val_mask, val_labels_np,
            transformer_seed=42, classifier_seed=cls_seed, freeze_transformer=False
        )
        # Test accuracy
        test_ids, test_mask = tokenize(tok, test_texts)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=test_ids.to(DEVICE), attention_mask=test_mask.to(DEVICE))
            acc = float(np.mean(out.logits.argmax(dim=-1).cpu().numpy() == np.array(test_labels)))
        print(f"acc={acc:.3f}")
        ctrl_a_accs.append(acc)

        imp, _ = measure_head_importance(model, tok, test_texts, test_labels, DEVICE)
        ctrl_a_imps.append(imp)
        del model; torch.cuda.empty_cache()

    ctrl_a_matrix = np.array(ctrl_a_imps)
    ctrl_a_stats = compute_flip_stats(ctrl_a_matrix)
    ctrl_a_stats['accuracies'] = [round(a, 3) for a in ctrl_a_accs]
    results['control_a_same_seed'] = ctrl_a_stats

    print(f"\n  Control A results:")
    print(f"    Mean flip rate: {ctrl_a_stats['mean_flip_rate']}")
    print(f"    Coverage conflict: {ctrl_a_stats['coverage_conflict']}")
    print(f"    Accuracy range: {min(ctrl_a_accs):.3f} - {max(ctrl_a_accs):.3f}")

    # ===== Control B: Frozen transformer, only train classifier =====
    print("\n" + "=" * 60)
    print("CONTROL B: Frozen transformer, only train classifier head")
    print("=" * 60)

    ctrl_b_imps = []
    ctrl_b_accs = []

    for i in range(n_control):
        seed = 42 + i
        print(f"  Model {i+1}/{n_control} (seed={seed}, frozen transformer)...",
              end=" ", flush=True)
        model, tok = fine_tune(
            train_texts, train_labels, val_ids, val_mask, val_labels_np,
            transformer_seed=seed, classifier_seed=seed, freeze_transformer=True
        )
        test_ids, test_mask = tokenize(tok, test_texts)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=test_ids.to(DEVICE), attention_mask=test_mask.to(DEVICE))
            acc = float(np.mean(out.logits.argmax(dim=-1).cpu().numpy() == np.array(test_labels)))
        print(f"acc={acc:.3f}")
        ctrl_b_accs.append(acc)

        imp, _ = measure_head_importance(model, tok, test_texts, test_labels, DEVICE)
        ctrl_b_imps.append(imp)
        del model; torch.cuda.empty_cache()

    ctrl_b_matrix = np.array(ctrl_b_imps)
    ctrl_b_stats = compute_flip_stats(ctrl_b_matrix)
    ctrl_b_stats['accuracies'] = [round(a, 3) for a in ctrl_b_accs]
    results['control_b_frozen'] = ctrl_b_stats

    print(f"\n  Control B results:")
    print(f"    Mean flip rate: {ctrl_b_stats['mean_flip_rate']}")
    print(f"    Coverage conflict: {ctrl_b_stats['coverage_conflict']}")
    print(f"    Accuracy range: {min(ctrl_b_accs):.3f} - {max(ctrl_b_accs):.3f}")

    # ===== Interpretation =====
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Load main experiment results if available
    main_path = OUT_DIR / 'results_mech_interp_rashomon_gpu.json'
    if main_path.exists():
        main = json.load(open(main_path))
        main_flip = main.get('mean_flip_rate', 'N/A')
        results['main_experiment_flip_rate'] = main_flip
        print(f"  Main experiment flip rate: {main_flip}")
    else:
        main_flip = 'N/A'
        print("  Main experiment results not found (run main experiment first)")

    print(f"  Control A (same seed) flip rate: {ctrl_a_stats['mean_flip_rate']}")
    print(f"  Control B (frozen) flip rate: {ctrl_b_stats['mean_flip_rate']}")

    if main_flip != 'N/A':
        if ctrl_a_stats['mean_flip_rate'] < 0.05:
            print("\n  INTERPRETATION: Control A stable → main experiment instability is genuine Rashomon")
        else:
            print("\n  INTERPRETATION: Control A unstable → classifier head noise contributes significantly")

        if ctrl_b_stats['mean_flip_rate'] > 0.1:
            print("  Control B unstable → ablation measures classifier readout, not circuits")
        else:
            print("  Control B stable → ablation measures actual circuit structure")

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)

    json_path = OUT_DIR / 'results_mech_interp_controls.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
