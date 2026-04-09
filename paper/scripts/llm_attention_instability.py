#!/usr/bin/env python3
"""
Validate whether the Attribution Impossibility applies to LLM attention-based
explanations. When input tokens carry correlated information, different
perturbations of the same transformer produce different attention-based
importance rankings.

Approach (CPU-friendly, no fine-tuning):
  1. Load pretrained DistilBERT (distilbert-base-uncased)
  2. Create 10 perturbed copies (small random weight perturbations to simulate
     different training runs / Rashomon set members)
  3. For each model, compute token importance via attention rollout
  4. Check whether relative rankings of adjacent/correlated token pairs flip
     across the 10 models

Output:
  - Console summary
  - paper/results_llm_attention.json
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MODELS = 10
PERTURBATION_SCALE = 0.02  # stddev of weight perturbation
SEED_BASE = 42
FLIP_THRESHOLD = 0.10  # report pairs with flip rate > this

# ---------------------------------------------------------------------------
# Synthetic sentiment dataset (200 sentences)
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "This movie is great",
    "I loved this film",
    "Excellent acting throughout",
    "A wonderful experience overall",
    "The plot was very engaging",
    "Brilliant performances by the cast",
    "Highly recommended for everyone",
    "A truly outstanding masterpiece",
    "The direction was superb",
    "Absolutely fantastic movie",
    "I really enjoyed every moment",
    "Beautiful cinematography and story",
    "One of the best films ever",
    "A delightful and charming film",
    "The screenplay was perfectly written",
    "Amazing special effects and acting",
    "I was completely captivated throughout",
    "A heartwarming and uplifting story",
    "The characters were very compelling",
    "A perfect blend of humor and drama",
]

_NEGATIVE_TEMPLATES = [
    "This movie is terrible",
    "I hated this film",
    "Awful acting throughout",
    "A dreadful experience overall",
    "The plot was very boring",
    "Terrible performances by the cast",
    "Not recommended for anyone",
    "A truly disappointing disaster",
    "The direction was awful",
    "Absolutely horrible movie",
    "I really disliked every moment",
    "Ugly cinematography and story",
    "One of the worst films ever",
    "A tedious and dull film",
    "The screenplay was poorly written",
    "Terrible special effects and acting",
    "I was completely bored throughout",
    "A depressing and pointless story",
    "The characters were very annoying",
    "A painful blend of nonsense and cliche",
]

# Test sentences with adjacent correlated token pairs of interest
TEST_SENTENCES = [
    "This movie was very good and entertaining",
    "The acting was not bad at all",
    "An absolutely terrible waste of time",
    "I found the plot really exciting and fresh",
    "The film had extremely poor writing throughout",
    "A remarkably brilliant piece of cinema",
    "The pacing felt incredibly slow and dull",
    "I thought the ending was quite satisfying",
    "The dialogue was particularly sharp and witty",
    "Overall a deeply moving and powerful film",
]

# Adjacent token pairs to check (token_a, token_b) — these are semantically
# correlated pairs where collinearity in information should trigger instability
PAIRS_OF_INTEREST = [
    ("very", "good"),
    ("not", "bad"),
    ("absolutely", "terrible"),
    ("really", "exciting"),
    ("extremely", "poor"),
    ("remarkably", "brilliant"),
    ("incredibly", "slow"),
    ("quite", "satisfying"),
    ("particularly", "sharp"),
    ("deeply", "moving"),
]


def build_dataset() -> List[Tuple[str, int]]:
    """Return list of (sentence, label) pairs — 200 total."""
    data = []
    for tmpl in _POSITIVE_TEMPLATES:
        data.extend([(tmpl, 1)] * 5)
    for tmpl in _NEGATIVE_TEMPLATES:
        data.extend([(tmpl, 0)] * 5)
    return data


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(attentions: Tuple[torch.Tensor, ...], seq_len: int) -> np.ndarray:
    """
    Compute attention rollout across all layers.

    For each layer, average attention across heads, then multiply layer
    matrices together. Return the CLS token's attention to all other tokens
    as the importance vector.

    Parameters
    ----------
    attentions : tuple of (batch=1, heads, seq, seq) tensors
    seq_len : number of tokens (including special tokens)

    Returns
    -------
    importance : np.ndarray of shape (seq_len,)
    """
    # Average across heads for each layer
    avg_attns = []
    for layer_attn in attentions:
        # layer_attn: (1, heads, seq, seq)
        avg = layer_attn[0].mean(dim=0)  # (seq, seq)
        avg_attns.append(avg)

    # Rollout: multiply layer attention matrices
    rollout = torch.eye(seq_len)
    for avg in avg_attns:
        # Add residual connection identity
        avg = 0.5 * avg + 0.5 * torch.eye(seq_len)
        # Re-normalise rows
        avg = avg / avg.sum(dim=-1, keepdim=True)
        rollout = rollout @ avg

    # CLS token row gives importance
    importance = rollout[0].detach().numpy()
    return importance


# ---------------------------------------------------------------------------
# Model loading and perturbation
# ---------------------------------------------------------------------------

def load_and_perturb_models(num_models: int, scale: float):
    """
    Load DistilBERT and create num_models perturbed copies.

    Returns list of (model, tokenizer) pairs.
    """
    from transformers import DistilBertTokenizer, DistilBertModel

    print(f"Loading distilbert-base-uncased tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    models = []
    for i in range(num_models):
        print(f"  Creating model variant {i+1}/{num_models}...", end=" ", flush=True)
        model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            output_attentions=True,
        )
        model.eval()

        if i > 0:  # model 0 is the unperturbed baseline
            rng = torch.Generator().manual_seed(SEED_BASE + i)
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn(param.shape, generator=rng) * scale
                    param.add_(noise)

        models.append(model)
        print("done")

    return models, tokenizer


# ---------------------------------------------------------------------------
# Compute importance for test sentences across all models
# ---------------------------------------------------------------------------

def compute_importances(models, tokenizer) -> Dict:
    """
    For each test sentence and each model, compute token importance via
    attention rollout.

    Returns dict mapping sentence -> list of (tokens, importance_array) per model.
    """
    results = {}
    for sent in TEST_SENTENCES:
        model_results = []
        inputs = tokenizer(sent, return_tensors="pt", padding=False, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        for model in models:
            with torch.no_grad():
                outputs = model(**inputs)
            attentions = outputs.attentions
            importance = attention_rollout(attentions, len(tokens))
            model_results.append((tokens, importance))

        results[sent] = model_results
    return results


# ---------------------------------------------------------------------------
# Flip analysis
# ---------------------------------------------------------------------------

def analyse_flips(importances: Dict) -> Dict:
    """
    For each pair of interest, check how often the relative ranking flips
    across models.
    """
    pair_results = []

    for tok_a, tok_b in PAIRS_OF_INTEREST:
        # Find which sentence contains this pair
        found = False
        for sent, model_results in importances.items():
            tokens_lower = [t.lower().replace("##", "") for t in model_results[0][0]]

            # Check if both tokens appear in this sentence
            if tok_a in tokens_lower and tok_b in tokens_lower:
                idx_a = tokens_lower.index(tok_a)
                idx_b = tokens_lower.index(tok_b)

                rankings = []
                for tokens, imp in model_results:
                    # "a beats b" if imp[a] > imp[b]
                    rankings.append(imp[idx_a] > imp[idx_b])

                n_a_wins = sum(rankings)
                n_b_wins = len(rankings) - n_a_wins
                flip_rate = min(n_a_wins, n_b_wins) / len(rankings)

                pair_results.append({
                    "token_a": tok_a,
                    "token_b": tok_b,
                    "sentence": sent,
                    "n_models": len(rankings),
                    "a_wins": int(n_a_wins),
                    "b_wins": int(n_b_wins),
                    "flip_rate": float(flip_rate),
                })
                found = True
                break

        if not found:
            pair_results.append({
                "token_a": tok_a,
                "token_b": tok_b,
                "sentence": None,
                "n_models": 0,
                "a_wins": 0,
                "b_wins": 0,
                "flip_rate": 0.0,
            })

    return pair_results


def compute_global_instability(importances: Dict) -> Dict:
    """
    For ALL adjacent token pairs across all test sentences, compute
    Kendall-tau style flip statistics.
    """
    total_pairs = 0
    flipped_pairs = 0
    flip_rates = []

    for sent, model_results in importances.items():
        tokens = model_results[0][0]
        n_tokens = len(tokens)

        # Check all adjacent pairs (skip [CLS] and [SEP])
        for i in range(1, n_tokens - 2):
            rankings = []
            for _, imp in model_results:
                rankings.append(imp[i] > imp[i + 1])

            n_true = sum(rankings)
            n_false = len(rankings) - n_true
            rate = min(n_true, n_false) / len(rankings)
            flip_rates.append(rate)
            total_pairs += 1
            if rate > FLIP_THRESHOLD:
                flipped_pairs += 1

    return {
        "total_adjacent_pairs": total_pairs,
        "pairs_with_flip_rate_above_threshold": flipped_pairs,
        "instability_pct": 100.0 * flipped_pairs / max(total_pairs, 1),
        "mean_flip_rate": float(np.mean(flip_rates)) if flip_rates else 0.0,
        "median_flip_rate": float(np.median(flip_rates)) if flip_rates else 0.0,
        "max_flip_rate": float(np.max(flip_rates)) if flip_rates else 0.0,
    }


# ---------------------------------------------------------------------------
# Spearman rank correlation between model pairs
# ---------------------------------------------------------------------------

def compute_rank_correlations(importances: Dict) -> Dict:
    """Compute pairwise Spearman correlations of token importance across models."""
    from scipy.stats import spearmanr

    all_corrs = []
    for sent, model_results in importances.items():
        n_models = len(model_results)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                imp_i = model_results[i][1]
                imp_j = model_results[j][1]
                corr, _ = spearmanr(imp_i, imp_j)
                all_corrs.append(corr)

    return {
        "mean_spearman": float(np.mean(all_corrs)),
        "min_spearman": float(np.min(all_corrs)),
        "std_spearman": float(np.std(all_corrs)),
        "n_comparisons": len(all_corrs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("LLM Attention Instability Validation")
    print("  Testing Attribution Impossibility for transformer attention")
    print(f"  {NUM_MODELS} model variants, perturbation scale = {PERTURBATION_SCALE}")
    print("=" * 70)

    # Step 1: Load / perturb models
    try:
        models, tokenizer = load_and_perturb_models(NUM_MODELS, PERTURBATION_SCALE)
    except Exception as e:
        print(f"\nERROR loading models: {e}")
        print("Attempting with smaller configuration...")
        sys.exit(1)

    # Step 2: Compute importance scores
    print("\nComputing attention rollout for test sentences...")
    importances = compute_importances(models, tokenizer)
    print(f"  Processed {len(TEST_SENTENCES)} sentences x {NUM_MODELS} models")

    # Step 3: Analyse targeted pairs
    print("\nAnalysing targeted correlated token pairs...")
    pair_results = analyse_flips(importances)

    # Step 4: Global instability
    print("Computing global adjacent-pair instability...")
    global_stats = compute_global_instability(importances)

    # Step 5: Spearman correlations
    print("Computing pairwise Spearman rank correlations...")
    rank_corr = compute_rank_correlations(importances)

    elapsed = time.time() - t0

    # --- Report ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nGlobal instability:")
    print(f"  Total adjacent token pairs examined: {global_stats['total_adjacent_pairs']}")
    print(f"  Pairs with flip rate > {FLIP_THRESHOLD*100:.0f}%: "
          f"{global_stats['pairs_with_flip_rate_above_threshold']}")
    print(f"  Instability rate: {global_stats['instability_pct']:.1f}%")
    print(f"  Mean flip rate: {global_stats['mean_flip_rate']:.3f}")
    print(f"  Median flip rate: {global_stats['median_flip_rate']:.3f}")
    print(f"  Max flip rate: {global_stats['max_flip_rate']:.3f}")

    print(f"\nPairwise Spearman rank correlations across models:")
    print(f"  Mean: {rank_corr['mean_spearman']:.4f}")
    print(f"  Min:  {rank_corr['min_spearman']:.4f}")
    print(f"  Std:  {rank_corr['std_spearman']:.4f}")

    print(f"\nTargeted correlated token pairs:")
    for pr in pair_results:
        flag = " ***" if pr["flip_rate"] > FLIP_THRESHOLD else ""
        print(f"  '{pr['token_a']}' vs '{pr['token_b']}': "
              f"flip rate = {pr['flip_rate']:.2f} "
              f"({pr['a_wins']}/{pr['b_wins']}){flag}")

    n_unstable_targeted = sum(1 for p in pair_results if p["flip_rate"] > FLIP_THRESHOLD)

    # Headline
    print("\n" + "-" * 70)
    headline = (
        f"Token attribution instability in DistilBERT: "
        f"{global_stats['instability_pct']:.1f}% of adjacent token pairs "
        f"have flip rate > {FLIP_THRESHOLD*100:.0f}%"
    )
    print(headline)

    if global_stats["instability_pct"] > 5.0:
        print("\nConclusion: Attention-based token importance IS unstable under")
        print("weight perturbation, consistent with the Attribution Impossibility.")
        print("When tokens carry correlated information, different models in the")
        print("Rashomon set produce different attention-based rankings.")
        conclusion = "instability_confirmed"
    else:
        print("\nConclusion: Attention-based token importance appears relatively")
        print("stable under this perturbation regime. The Attribution Impossibility")
        print("may require larger model differences or stronger token correlation.")
        conclusion = "instability_not_confirmed"

    print(f"\nElapsed time: {elapsed:.1f}s")

    # --- Save results ---
    results = {
        "experiment": "llm_attention_instability",
        "model": "distilbert-base-uncased",
        "num_models": NUM_MODELS,
        "perturbation_scale": PERTURBATION_SCALE,
        "num_test_sentences": len(TEST_SENTENCES),
        "global_instability": global_stats,
        "rank_correlations": rank_corr,
        "targeted_pairs": pair_results,
        "n_unstable_targeted_pairs": n_unstable_targeted,
        "headline": headline,
        "conclusion": conclusion,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = Path(__file__).resolve().parent.parent / "results_llm_attention.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
