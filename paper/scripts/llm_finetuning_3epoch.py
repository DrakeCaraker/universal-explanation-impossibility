#!/usr/bin/env python3
"""
3-Epoch Fine-tuning Instability Validation for DistilBERT on SST-2.

Responds to reviewer question: does longer training increase instability?
We replicate the 1-epoch experiment (llm_finetuning_instability.py) with
NUM_EPOCHS = 3 and report the comparison table:

  | Method              | Epochs | % Pairs unstable (>10% flip) | Mean Spearman |
  |---------------------|--------|------------------------------|---------------|
  | Weight perturbation | N/A    | 88%                          | 0.637         |
  | Fine-tuning         | 1      | 14.5%                        | 0.956         |
  | Fine-tuning         | 3      | X%                           | Y             |

Method:
  - Token importance = mean attention received per token, averaged over all
    heads and all layers.
  - Flip rate for pair (i, j): across all C(5,2)=10 model pairs, fraction of
    pairs where relative ordering reverses.
  - [CLS] and [SEP] tokens excluded.

Output:
  - Console progress and summary
  - paper/results_llm_finetuning_3epoch.txt
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1011]
TRAIN_SUBSET = 2000        # same 2000 training examples as 1-epoch experiment
NUM_EPOCHS = 3             # <-- key change from 1-epoch experiment
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
FLIP_THRESHOLD = 0.10      # report pairs with flip rate > 10%
DEVICE = "cpu"

# Known results from prior experiments (for comparison table)
PERTURBATION_INSTABILITY_PCT = 88.0
PERTURBATION_MEAN_SPEARMAN   = 0.637
FINETUNING_1EPOCH_INSTABILITY_PCT = 14.5
FINETUNING_1EPOCH_MEAN_SPEARMAN   = 0.956

# Test sentences (same as 1-epoch experiment)
TEST_SENTENCES = [
    "The movie was extremely good",
    "This product is absolutely terrible",
    "The food was not bad at all",
    "I found the service particularly sharp and efficient",
    "The acting was very poor throughout",
    "A truly wonderful experience overall",
    "The plot was incredibly boring and predictable",
    "She gave an outstanding and remarkable performance",
    "The quality is surprisingly low for the price",
    "An exceptionally well crafted piece of work",
]


# ---------------------------------------------------------------------------
# Helpers: attention-based token importance
# ---------------------------------------------------------------------------

def mean_attention_importance(attentions, seq_len: int) -> np.ndarray:
    """
    Token importance = mean attention *received* per token,
    averaged over all heads and all layers.

    Parameters
    ----------
    attentions : tuple of (1, num_heads, seq, seq) tensors, one per layer
    seq_len    : total sequence length (including special tokens)

    Returns
    -------
    importance : np.ndarray of shape (seq_len,), sums to ~1
    """
    layer_means = []
    for layer_attn in attentions:
        # layer_attn: (1, heads, seq, seq)
        avg_heads = layer_attn[0].mean(dim=0)   # (seq, seq)
        col_mean  = avg_heads.mean(dim=0)        # (seq,) — mean attention received
        layer_means.append(col_mean.detach())

    importance = torch.stack(layer_means).mean(dim=0).numpy()
    importance = importance / (importance.sum() + 1e-12)
    return importance


# ---------------------------------------------------------------------------
# Fine-tuning one seed for NUM_EPOCHS epochs
# ---------------------------------------------------------------------------

def finetune_one_seed(seed: int, train_texts, train_labels,
                      batch_size: int = BATCH_SIZE) -> object:
    """
    Fine-tune DistilBERT for sequence classification.
    Returns trained model (in eval mode, on DEVICE).
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        output_attentions=True,
    )
    model.to(DEVICE)

    class SSTDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len=128):
            self.enc = tok(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    dataset  = SSTDataset(train_texts, train_labels, tokenizer)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        print(f"    Epoch {epoch+1}/{NUM_EPOCHS}", flush=True)
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            except RuntimeError as oom:
                if "out of memory" in str(oom).lower() and batch_size > 4:
                    print(f"\n    OOM at batch_size={batch_size}; halving...", flush=True)
                    model.cpu()
                    del model
                    return finetune_one_seed(seed, train_texts, train_labels,
                                            batch_size=batch_size // 2)
                raise

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if (step + 1) % 20 == 0:
                avg = running_loss / (step + 1)
                print(f"      step {step+1}/{len(loader)}  loss={avg:.4f}", flush=True)

        epoch_loss = running_loss / len(loader)
        print(f"    Epoch {epoch+1} complete — avg loss={epoch_loss:.4f}", flush=True)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Compute importance scores across fine-tuned models
# ---------------------------------------------------------------------------

def compute_importances_finetuned(models, tokenizer) -> dict:
    """
    For each test sentence and each fine-tuned model, compute mean-attention
    token importance.
    Returns dict: sentence -> list of (tokens, importance_array).
    """
    results = {}
    for sent in TEST_SENTENCES:
        model_results = []
        inputs = tokenizer(
            sent,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        for model in models:
            with torch.no_grad():
                outputs = model(**inputs)
            importance = mean_attention_importance(outputs.attentions, len(tokens))
            model_results.append((tokens, importance))

        results[sent] = model_results
    return results


# ---------------------------------------------------------------------------
# Flip analysis
# ---------------------------------------------------------------------------

def compute_flip_rate_for_pair(importances_list, idx_a: int, idx_b: int) -> float:
    """
    Fraction of model pairs (i, j) where the relative ranking of tokens a and b
    flips between model i and model j.
    """
    n = len(importances_list)
    total = 0
    flips = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_beats_b_i = importances_list[i][idx_a] > importances_list[i][idx_b]
            a_beats_b_j = importances_list[j][idx_a] > importances_list[j][idx_b]
            if a_beats_b_i != a_beats_b_j:
                flips += 1
            total += 1
    return flips / max(total, 1)


def compute_global_instability(importances: dict) -> dict:
    """
    For ALL adjacent non-special token pairs across all sentences,
    compute flip statistics.
    """
    total_pairs   = 0
    flipped_pairs = 0
    flip_rates    = []

    for sent, model_results in importances.items():
        tokens   = model_results[0][0]
        n_tok    = len(tokens)
        imp_list = [mr[1] for mr in model_results]

        # skip [CLS]=0 and [SEP]=last
        for i in range(1, n_tok - 2):
            rate = compute_flip_rate_for_pair(imp_list, i, i + 1)
            flip_rates.append(rate)
            total_pairs += 1
            if rate > FLIP_THRESHOLD:
                flipped_pairs += 1

    return {
        "total_adjacent_pairs":             total_pairs,
        "pairs_with_flip_rate_above_10pct": flipped_pairs,
        "instability_pct":                  100.0 * flipped_pairs / max(total_pairs, 1),
        "mean_flip_rate":                   float(np.mean(flip_rates))   if flip_rates else 0.0,
        "median_flip_rate":                 float(np.median(flip_rates)) if flip_rates else 0.0,
        "max_flip_rate":                    float(np.max(flip_rates))    if flip_rates else 0.0,
        "all_flip_rates":                   [float(r) for r in flip_rates],
    }


def compute_rank_correlations(importances: dict) -> dict:
    """Mean pairwise Spearman correlation of full-sentence token rankings."""
    from scipy.stats import spearmanr

    all_corrs = []
    for sent, model_results in importances.items():
        n = len(model_results)
        for i in range(n):
            for j in range(i + 1, n):
                imp_i = model_results[i][1]
                imp_j = model_results[j][1]
                corr, _ = spearmanr(imp_i, imp_j)
                all_corrs.append(float(corr))

    return {
        "mean_spearman": float(np.mean(all_corrs))   if all_corrs else 0.0,
        "min_spearman":  float(np.min(all_corrs))    if all_corrs else 0.0,
        "std_spearman":  float(np.std(all_corrs))    if all_corrs else 0.0,
        "n_comparisons": len(all_corrs),
    }


def compute_targeted_pairs(importances: dict) -> list:
    """
    For selected collinear adverb-adjective pairs, report per-sentence flip rate.
    """
    targeted = [
        (0, "extremely",    "good"),
        (1, "absolutely",   "terrible"),
        (2, "not",          "bad"),
        (3, "particularly", "sharp"),
        (4, "very",         "poor"),
        (5, "truly",        "wonderful"),
        (6, "incredibly",   "boring"),
        (7, "outstanding",  "remarkable"),
        (8, "surprisingly", "low"),
        (9, "exceptionally","well"),
    ]

    results = []
    for sent_idx, word_a, word_b in targeted:
        sent = TEST_SENTENCES[sent_idx]
        if sent not in importances:
            continue
        model_results = importances[sent]
        tokens_lower  = [t.lower().replace("##", "")
                         for t in model_results[0][0]]

        idx_a = next((i for i, t in enumerate(tokens_lower) if t == word_a), None)
        idx_b = next((i for i, t in enumerate(tokens_lower) if t == word_b), None)

        if idx_a is None or idx_b is None:
            idx_a = next((i for i, t in enumerate(tokens_lower)
                          if word_a in t), None)
            idx_b = next((i for i, t in enumerate(tokens_lower)
                          if word_b in t), None)

        if idx_a is None or idx_b is None:
            results.append({
                "token_a": word_a, "token_b": word_b,
                "sentence": sent,  "flip_rate": None,
                "note": "token not found after tokenization",
            })
            continue

        imp_list  = [mr[1] for mr in model_results]
        flip_rate = compute_flip_rate_for_pair(imp_list, idx_a, idx_b)

        results.append({
            "token_a":   word_a,
            "token_b":   word_b,
            "sentence":  sent,
            "idx_a":     idx_a,
            "idx_b":     idx_b,
            "flip_rate": float(flip_rate),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 70)
    print("LLM Fine-tuning Instability Validation — 3-Epoch Experiment")
    print(f"  DistilBERT fine-tuned on SST-2, {NUM_SEEDS} random seeds, "
          f"{NUM_EPOCHS} epochs each")
    print(f"  Training subset: {TRAIN_SUBSET} examples  |  device: {DEVICE}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load SST-2 data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading SST-2 dataset...", flush=True)
    try:
        from datasets import load_dataset
        sst2 = load_dataset("glue", "sst2")
        train_data = sst2["train"]
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        sys.exit(1)

    # Exact same subset as 1-epoch experiment (rng seed=0)
    rng_sub = np.random.default_rng(0)
    indices = rng_sub.choice(len(train_data), size=TRAIN_SUBSET, replace=False)
    indices = sorted(indices.tolist())

    train_texts  = [train_data[int(i)]["sentence"] for i in indices]
    train_labels = [train_data[int(i)]["label"]    for i in indices]
    print(f"  Loaded {len(train_texts)} training examples")
    print(f"  Label distribution: "
          f"{sum(train_labels)} positive / "
          f"{len(train_labels)-sum(train_labels)} negative")

    # ------------------------------------------------------------------
    # 2. Fine-tune with each seed for 3 epochs
    # ------------------------------------------------------------------
    print(f"\n[2/4] Fine-tuning {NUM_SEEDS} models "
          f"({NUM_EPOCHS} epochs each)...", flush=True)
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    models = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  --- Seed {seed} ({seed_idx+1}/{NUM_SEEDS}) ---", flush=True)
        t_seed = time.time()
        model = finetune_one_seed(seed, train_texts, train_labels)
        elapsed_seed = time.time() - t_seed
        print(f"  Seed {seed} done in {elapsed_seed:.1f}s", flush=True)
        models.append(model)

    # ------------------------------------------------------------------
    # 3. Compute attention-based token importance on test sentences
    # ------------------------------------------------------------------
    print(f"\n[3/4] Computing token importance on "
          f"{len(TEST_SENTENCES)} test sentences...", flush=True)
    importances = compute_importances_finetuned(models, tokenizer)
    print(f"  Done: {len(TEST_SENTENCES)} sentences × {NUM_SEEDS} models",
          flush=True)

    # ------------------------------------------------------------------
    # 4. Analysis
    # ------------------------------------------------------------------
    print("\n[4/4] Running instability analysis...", flush=True)

    global_stats   = compute_global_instability(importances)
    rank_corr      = compute_rank_correlations(importances)
    targeted_pairs = compute_targeted_pairs(importances)

    elapsed_total = time.time() - t_start

    # ------------------------------------------------------------------
    # Print summary to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS — Fine-tuning instability ({NUM_EPOCHS} epochs)")
    print("=" * 70)

    print(f"\nGlobal adjacent-pair instability "
          f"(fine-tuning, {NUM_SEEDS} seeds, {NUM_EPOCHS} epochs):")
    print(f"  Total adjacent token pairs:          "
          f"{global_stats['total_adjacent_pairs']}")
    print(f"  Pairs with flip rate > 10%:          "
          f"{global_stats['pairs_with_flip_rate_above_10pct']}")
    print(f"  Instability rate:                    "
          f"{global_stats['instability_pct']:.1f}%")
    print(f"  Mean flip rate:                      "
          f"{global_stats['mean_flip_rate']:.3f}")
    print(f"  Median flip rate:                    "
          f"{global_stats['median_flip_rate']:.3f}")
    print(f"  Max flip rate:                       "
          f"{global_stats['max_flip_rate']:.3f}")

    print(f"\nPairwise Spearman correlations of full token rankings:")
    print(f"  Mean:          {rank_corr['mean_spearman']:.4f}")
    print(f"  Min:           {rank_corr['min_spearman']:.4f}")
    print(f"  Std:           {rank_corr['std_spearman']:.4f}")
    print(f"  N comparisons: {rank_corr['n_comparisons']}")

    print(f"\nSelected collinear token pairs (adverb-adjective):")
    print(f"  {'Token A':<15} {'Token B':<15} {'Flip Rate':>10}  Sentence")
    print(f"  {'-'*14} {'-'*14} {'-'*10}  {'-'*35}")
    for tp in targeted_pairs:
        rate_str = f"{tp['flip_rate']:.2f}" if tp["flip_rate"] is not None else "N/A"
        flag     = " ***" if (tp["flip_rate"] or 0) > FLIP_THRESHOLD else ""
        short_s  = (tp["sentence"] or "")[:45]
        print(f"  {tp['token_a']:<15} {tp['token_b']:<15} "
              f"{rate_str:>10}{flag}  {short_s}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    instab_3ep = global_stats["instability_pct"]
    spear_3ep  = rank_corr["mean_spearman"]

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = (f"  {'Method':<22} {'Epochs':>7}  "
              f"{'% Pairs unstable (>10% flip)':>30}  {'Mean Spearman':>14}")
    sep    = "  " + "-"*22 + "  " + "-"*7 + "  " + "-"*30 + "  " + "-"*14
    print(header)
    print(sep)
    print(f"  {'Weight perturbation':<22} {'N/A':>7}  "
          f"{PERTURBATION_INSTABILITY_PCT:>29.0f}%  "
          f"{PERTURBATION_MEAN_SPEARMAN:>14.3f}")
    print(f"  {'Fine-tuning':<22} {'1':>7}  "
          f"{FINETUNING_1EPOCH_INSTABILITY_PCT:>29.1f}%  "
          f"{FINETUNING_1EPOCH_MEAN_SPEARMAN:>14.3f}")
    print(f"  {'Fine-tuning':<22} {'3':>7}  "
          f"{instab_3ep:>29.1f}%  "
          f"{spear_3ep:>14.3f}")

    # Interpretation
    delta_instab = instab_3ep - FINETUNING_1EPOCH_INSTABILITY_PCT
    delta_spear  = spear_3ep  - FINETUNING_1EPOCH_MEAN_SPEARMAN

    if abs(delta_instab) < 2.0:
        training_effect = (
            f"Instability is stable across training depth: "
            f"{instab_3ep:.1f}% at 3 epochs vs {FINETUNING_1EPOCH_INSTABILITY_PCT}% "
            f"at 1 epoch (Δ={delta_instab:+.1f}pp). "
            "This suggests the Rashomon set is populated by diverse models "
            "regardless of training length."
        )
    elif delta_instab > 0:
        training_effect = (
            f"Longer training increases instability: "
            f"{instab_3ep:.1f}% at 3 epochs vs {FINETUNING_1EPOCH_INSTABILITY_PCT}% "
            f"at 1 epoch (Δ={delta_instab:+.1f}pp). "
            "Models that converge more thoroughly occupy more diverse attribution minima."
        )
    else:
        training_effect = (
            f"Longer training decreases instability: "
            f"{instab_3ep:.1f}% at 3 epochs vs {FINETUNING_1EPOCH_INSTABILITY_PCT}% "
            f"at 1 epoch (Δ={delta_instab:+.1f}pp). "
            "Better-converged models may settle into more consistent attribution patterns."
        )

    print(f"\nInterpretation: {training_effect}")
    print(f"\nTotal elapsed time: {elapsed_total:.1f}s")

    # ------------------------------------------------------------------
    # Save results to text file
    # ------------------------------------------------------------------
    out_path = (Path(__file__).resolve().parent.parent
                / "results_llm_finetuning_3epoch.txt")
    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LLM Fine-tuning Instability — 3-Epoch Experiment\n")
        f.write(f"  DistilBERT fine-tuned on SST-2, {NUM_SEEDS} seeds, "
                f"{NUM_EPOCHS} epochs each\n")
        f.write(f"  Training subset: {TRAIN_SUBSET} examples\n")
        f.write(f"  Seeds: {SEEDS}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}  "
                f"Batch size: {BATCH_SIZE}  Device: {DEVICE}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SECTION 1: Global Adjacent-Pair Instability\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total adjacent token pairs analyzed:    "
                f"{global_stats['total_adjacent_pairs']}\n")
        f.write(f"Pairs with flip rate > 10%:             "
                f"{global_stats['pairs_with_flip_rate_above_10pct']}\n")
        f.write(f"Instability rate:                       "
                f"{global_stats['instability_pct']:.1f}%\n")
        f.write(f"Mean flip rate:                         "
                f"{global_stats['mean_flip_rate']:.4f}\n")
        f.write(f"Median flip rate:                       "
                f"{global_stats['median_flip_rate']:.4f}\n")
        f.write(f"Max flip rate:                          "
                f"{global_stats['max_flip_rate']:.4f}\n\n")

        f.write("SECTION 2: Pairwise Spearman Rank Correlations\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Spearman correlation:              "
                f"{rank_corr['mean_spearman']:.4f}\n")
        f.write(f"Min Spearman correlation:               "
                f"{rank_corr['min_spearman']:.4f}\n")
        f.write(f"Std Spearman correlation:               "
                f"{rank_corr['std_spearman']:.4f}\n")
        f.write(f"Number of model-pair comparisons:       "
                f"{rank_corr['n_comparisons']}\n\n")

        f.write("SECTION 3: Selected Collinear Token Pairs\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Token A':<15} {'Token B':<15} {'Flip Rate':>10}  "
                f"Sentence\n")
        f.write(f"  {'-'*14} {'-'*14} {'-'*10}  {'-'*40}\n")
        for tp in targeted_pairs:
            rate_str = (f"{tp['flip_rate']:.4f}"
                        if tp["flip_rate"] is not None else "N/A   ")
            flag = " ***" if (tp["flip_rate"] or 0) > FLIP_THRESHOLD else ""
            f.write(f"  {tp['token_a']:<15} {tp['token_b']:<15} "
                    f"{rate_str:>10}{flag}  {tp['sentence']}\n")
        f.write(f"\n  *** = flip rate > {FLIP_THRESHOLD*100:.0f}% (unstable)\n\n")

        f.write("SECTION 4: Comparison Table\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Method':<22} {'Epochs':>7}  "
                f"{'% Pairs unstable (>10% flip)':>30}  {'Mean Spearman':>14}\n")
        f.write("  " + "-"*22 + "  " + "-"*7 + "  " + "-"*30 + "  "
                + "-"*14 + "\n")
        f.write(f"  {'Weight perturbation':<22} {'N/A':>7}  "
                f"{PERTURBATION_INSTABILITY_PCT:>29.0f}%  "
                f"{PERTURBATION_MEAN_SPEARMAN:>14.3f}\n")
        f.write(f"  {'Fine-tuning':<22} {'1':>7}  "
                f"{FINETUNING_1EPOCH_INSTABILITY_PCT:>29.1f}%  "
                f"{FINETUNING_1EPOCH_MEAN_SPEARMAN:>14.3f}\n")
        f.write(f"  {'Fine-tuning':<22} {'3':>7}  "
                f"{instab_3ep:>29.1f}%  "
                f"{spear_3ep:>14.3f}\n\n")

        f.write("SECTION 5: Interpretation\n")
        f.write("-" * 40 + "\n")
        f.write(training_effect + "\n\n")
        f.write(f"Total elapsed time: {elapsed_total:.1f}s\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
