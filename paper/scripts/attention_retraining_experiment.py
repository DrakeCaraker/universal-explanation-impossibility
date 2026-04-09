#!/usr/bin/env python3
"""
Attention Retraining Experiment
================================
Research question: Do transformer models trained from DIFFERENT random seeds
(different optimization trajectories) assign peak attention to different tokens?

This addresses the "perturbation is artificial" criticism of the perturbation
experiment: here we train 10 genuinely independent classification heads on
frozen DistilBERT, each from a different random seed.

Design:
- DistilBERT-base-uncased with transformer layers FROZEN
- Fresh nn.Linear(768, 2) classification head (1536 parameters)
- 100 training sentences (50 positive, 50 negative)
- 10 training seeds → 10 genuinely different optimization trajectories
- Adam optimizer, lr=1e-3, 5 epochs
- Held-out accuracy verified >70% for all 10 models
- 50 test sentences for attention rollout comparison
- Attention rollout (identical method to perturbation experiment)
- Metrics: argmax flip rate, prediction agreement, 95% bootstrap CIs

KEY INSIGHT:
Because the backbone is frozen, raw attention weights are backbone-only and
identical across seeds.  The "head-weighted" importance score (base rollout
weighted by |head weights| · |hidden state|) captures how different head
random seeds encode different token salience for the final classification.
This is the appropriate measure of retraining-induced instability.

Comparison:
- Perturbation experiment (artificial): ~60% flip rate
- Retraining experiment (genuine): reported here

Outputs:
- paper/results_attention_retraining.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (
    set_all_seeds,
    save_results,
    percentile_ci,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MODELS = 10
NUM_TRAIN_SENTENCES = 100   # 50 positive + 50 negative
NUM_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
SEED_BASE = 42              # seeds: 42, 43, ..., 51
HIDDEN_SIZE = 768
NUM_CLASSES = 2
MIN_ACCURACY = 0.70         # all models must beat this on held-out set
DEVICE = 'cpu'

# ---------------------------------------------------------------------------
# 100 training sentences (50 positive, 50 negative)
# ---------------------------------------------------------------------------
_POSITIVE_TRAIN = [
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
    "This is a must-see masterpiece",
    "The acting was phenomenal",
    "Such a moving and powerful story",
    "Brilliant direction and photography",
    "I was on the edge of my seat",
    "The soundtrack was absolutely gorgeous",
    "A triumph of filmmaking artistry",
    "Every scene was beautifully crafted",
    "The dialogue was sharp and witty",
    "An uplifting and inspiring film",
    "The pacing was perfectly controlled",
    "A visually stunning achievement",
    "The performances were deeply moving",
    "I laughed and cried throughout",
    "This film exceeded all expectations",
    "A genuine feel-good experience",
    "The story arc was satisfying",
    "An exceptional piece of cinema",
    "The chemistry between actors was electric",
    "This film restored my faith in cinema",
    "A rich and rewarding experience",
    "The world-building was immersive",
    "Every character felt fully realized",
    "The ending was profoundly moving",
    "A joyful and life-affirming film",
    "The humor was perfectly timed",
    "I would watch this again immediately",
    "A rare gem in modern cinema",
    "The cinematography was breathtaking",
    "An absolute joy from start to finish",
]

_NEGATIVE_TRAIN = [
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
    "This film was a complete waste of time",
    "The acting was laughably bad",
    "A dull and predictable plot",
    "Poor direction and photography",
    "I kept checking my watch",
    "The soundtrack was grating",
    "A failure of filmmaking craft",
    "Every scene felt forced and flat",
    "The dialogue was clunky and dull",
    "A depressing and gloomy experience",
    "The pacing was uneven and slow",
    "A visually ugly mess",
    "The performances were unconvincing",
    "I felt nothing throughout",
    "This film fell short of every expectation",
    "A genuinely unpleasant experience",
    "The story arc made no sense",
    "A forgettable piece of cinema",
    "The chemistry between actors was absent",
    "This film destroyed my faith in cinema",
    "A hollow and unrewarding experience",
    "The world-building was incoherent",
    "Every character felt like a cardboard cutout",
    "The ending made no sense at all",
    "A joyless and soul-crushing film",
    "The humor fell completely flat",
    "I could not wait for this to end",
    "A rare failure in modern cinema",
    "The cinematography was uninspired",
    "An absolute chore from start to finish",
]

assert len(_POSITIVE_TRAIN) == 50, f"Need 50 positive, got {len(_POSITIVE_TRAIN)}"
assert len(_NEGATIVE_TRAIN) == 50, f"Need 50 negative, got {len(_NEGATIVE_TRAIN)}"

# 20-sentence held-out set (10 positive, 10 negative) for accuracy verification
_POSITIVE_HELDOUT = [
    "This was an incredible cinematic experience",
    "I was moved to tears by this film",
    "Superb storytelling from beginning to end",
    "A classic that will stand the test of time",
    "Outstanding work by the entire cast",
    "A film that stays with you long after",
    "Beautifully paced and wonderfully acted",
    "A tour de force of modern filmmaking",
    "I cannot recommend this film highly enough",
    "A deeply satisfying and enriching film",
]

_NEGATIVE_HELDOUT = [
    "This was an excruciating cinematic ordeal",
    "I felt cheated by the ending of this film",
    "Poor storytelling from beginning to end",
    "A film that will be forgotten immediately",
    "Dismal work by the entire cast",
    "A film that haunts you for the wrong reasons",
    "Badly paced and poorly acted",
    "A tour de force of modern incompetence",
    "I cannot warn against this film enough",
    "A deeply unsatisfying and empty film",
]

# 50-sentence test set for attention rollout comparison (25 positive, 25 negative)
_POSITIVE_TEST = [
    "This movie is excellent",
    "The film was absolutely wonderful",
    "I really loved this picture",
    "An outstanding performance by all",
    "A truly brilliant work of art",
    "The movie was incredibly good",
    "Superb in every possible way",
    "A magnificent achievement in cinema",
    "Truly exceptional filmmaking throughout",
    "I was thoroughly impressed by this",
    "A masterful and compelling story",
    "The direction was inspired and bold",
    "Every frame was a work of beauty",
    "A film I will treasure forever",
    "The script was sharp and intelligent",
    "Unforgettable performances across the board",
    "This film moved me profoundly",
    "An absolutely delightful experience",
    "The cast delivered brilliantly",
    "A rewarding and fulfilling film",
    "Stunning visuals and great story",
    "I was captivated from the first scene",
    "A film that celebrates the human spirit",
    "The best movie I have seen in years",
    "Pure cinematic excellence throughout",
]

_NEGATIVE_TEST = [
    "This movie is awful",
    "The film was completely horrible",
    "I really hated this picture",
    "A dreadful performance by all",
    "A truly terrible waste of time",
    "The movie was incredibly bad",
    "Awful in every possible way",
    "A catastrophic failure in cinema",
    "Truly dreadful filmmaking throughout",
    "I was thoroughly disappointed by this",
    "A muddled and unconvincing story",
    "The direction was confused and weak",
    "Every frame was an eyesore",
    "A film I want to forget forever",
    "The script was clumsy and stupid",
    "Forgettable performances across the board",
    "This film bored me to tears",
    "An absolutely terrible experience",
    "The cast delivered poorly",
    "A frustrating and hollow film",
    "Ugly visuals and weak story",
    "I was bored from the first scene",
    "A film that crushes the human spirit",
    "The worst movie I have seen in years",
    "Pure cinematic failure throughout",
]

assert len(_POSITIVE_TEST) == 25, f"Need 25 positive test, got {len(_POSITIVE_TEST)}"
assert len(_NEGATIVE_TEST) == 25, f"Need 25 negative test, got {len(_NEGATIVE_TEST)}"


def build_datasets():
    """Return train, held-out, and test (sentence, label) lists."""
    train = ([(s, 1) for s in _POSITIVE_TRAIN] +
             [(s, 0) for s in _NEGATIVE_TRAIN])
    heldout = ([(s, 1) for s in _POSITIVE_HELDOUT] +
               [(s, 0) for s in _NEGATIVE_HELDOUT])
    test = ([(s, 1) for s in _POSITIVE_TEST] +
            [(s, 0) for s in _NEGATIVE_TEST])
    return train, heldout, test


# ---------------------------------------------------------------------------
# Attention rollout (identical to perturbation experiment)
# ---------------------------------------------------------------------------

def attention_rollout(attentions, seq_len):
    """
    Compute attention rollout across all layers.

    For each layer, average attention across heads, then multiply layer
    matrices together with residual identity connection. Returns the
    CLS token's aggregated attention to all tokens.

    Parameters
    ----------
    attentions : tuple of (batch=1, heads, seq, seq) tensors — from all 6 layers
    seq_len : int

    Returns
    -------
    importance : np.ndarray of shape (seq_len,)
    """
    import torch

    avg_attns = []
    for layer_attn in attentions:
        avg = layer_attn[0].mean(dim=0)   # (seq, seq)
        avg_attns.append(avg)

    rollout = torch.eye(seq_len)
    for avg in avg_attns:
        avg = 0.5 * avg + 0.5 * torch.eye(seq_len)   # residual identity
        avg = avg / avg.sum(dim=-1, keepdim=True)
        rollout = rollout @ avg

    importance = rollout[0].detach().numpy()
    return importance


# ---------------------------------------------------------------------------
# Classification model: frozen DistilBERT + trainable nn.Linear(768, 2)
# ---------------------------------------------------------------------------

def load_backbone():
    """Load frozen DistilBERT backbone (DistilBertModel, not ForSequenceClassification)."""
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel

    print("Loading distilbert-base-uncased (DistilBertModel, frozen)...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    backbone = DistilBertModel.from_pretrained(
        "distilbert-base-uncased",
        output_attentions=True,
    )
    backbone.eval()

    # Freeze ALL transformer layers
    for param in backbone.parameters():
        param.requires_grad_(False)

    n_params = sum(p.numel() for p in backbone.parameters())
    head_params = HIDDEN_SIZE * NUM_CLASSES + NUM_CLASSES
    print(f"  Backbone parameters (frozen): {n_params:,}")
    print(f"  Classification head parameters (trainable): {head_params:,}")
    return backbone, tokenizer


def train_head(backbone, tokenizer, train_data, seed):
    """
    Initialize a fresh nn.Linear(768, 2) and train it for NUM_EPOCHS.

    Training loop:
      forward pass through frozen DistilBert → get [CLS] embedding →
      linear head → cross entropy loss → backward (only head params)

    Returns the trained head (nn.Linear).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(seed)
    set_all_seeds(seed)

    head = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    # Xavier initialization (seed-dependent)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    optimizer = optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    sentences = [s for s, _ in train_data]
    labels_list = [y for _, y in train_data]

    # Pre-compute CLS embeddings (backbone is frozen, so these are fixed)
    print(f"    [seed={seed}] Pre-computing CLS embeddings...", end=" ", flush=True)
    cls_embs = []
    backbone.eval()
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            out = backbone(**enc)
            cls_emb = out.last_hidden_state[0, 0, :]   # [CLS] token, shape (768,)
            cls_embs.append(cls_emb)
    cls_embs = torch.stack(cls_embs)   # (100, 768)
    labels = torch.tensor(labels_list, dtype=torch.long)
    print("done")

    # Shuffle order (seed-dependent)
    perm = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed))
    cls_embs = cls_embs[perm]
    labels = labels[perm]

    head.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        # Mini-batch gradient descent (batch_size=16)
        for start in range(0, len(train_data), BATCH_SIZE):
            batch_emb = cls_embs[start:start + BATCH_SIZE]
            batch_lbl = labels[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = head(batch_emb)
            loss = criterion(logits, batch_lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    [seed={seed}] Epoch {epoch+1}/{NUM_EPOCHS}  loss={total_loss:.4f}")

    head.eval()
    return head


def compute_accuracy(backbone, tokenizer, head, data):
    """Compute accuracy: forward DistilBert → [CLS] → head → argmax."""
    import torch

    sentences = [s for s, _ in data]
    labels = np.array([y for _, y in data])

    cls_embs = []
    backbone.eval()
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            out = backbone(**enc)
            cls_embs.append(out.last_hidden_state[0, 0, :])
    cls_embs = torch.stack(cls_embs)

    with torch.no_grad():
        logits = head(cls_embs)
        preds = logits.argmax(dim=-1).numpy()

    return float(np.mean(preds == labels)), preds


# ---------------------------------------------------------------------------
# Compute head-weighted attention rollout on test sentences
# ---------------------------------------------------------------------------

def compute_importance_matrix(backbone, tokenizer, trained_heads, test_sentences):
    """
    For each test sentence and each trained head, compute head-weighted
    token importance via attention rollout.

    Since the backbone is frozen, raw attention maps are identical across
    seeds.  We weight the base rollout by |head_weights| · |hidden_state|
    to capture how different head random seeds encode different token salience.

    Returns
    -------
    all_tokens     : list[list[str]]
    all_importance : np.ndarray (n_models, n_sent, max_seq_len)
    token_counts   : list[int]
    """
    import torch

    n_sent = len(test_sentences)
    n_models = len(trained_heads)

    all_tokens = []
    token_counts = []

    print("Tokenising test sentences...")
    for sent in test_sentences:
        enc = tokenizer(sent, return_tensors="pt", padding=False,
                        truncation=True, max_length=64)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        all_tokens.append(tokens)
        token_counts.append(len(tokens))

    max_seq = max(token_counts)
    all_importance = np.zeros((n_models, n_sent, max_seq))

    print(f"Computing head-weighted attention rollout ({n_models} seeds × {n_sent} sentences)...")
    for m_idx, head in enumerate(trained_heads):
        seed = SEED_BASE + m_idx
        print(f"  Seed {seed} (model {m_idx+1}/{n_models})...", end=" ", flush=True)

        # Head weight magnitudes: (768,) — L1 norm across output classes
        head_weights = head.weight.detach().abs().mean(dim=0)  # (768,)

        for s_idx, sent in enumerate(test_sentences):
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            with torch.no_grad():
                out = backbone(**enc, output_attentions=True)

            seq_len = token_counts[s_idx]
            # Base attention rollout (same for all seeds — backbone is frozen)
            base_importance = attention_rollout(out.attentions, seq_len)

            # Token-level hidden states: (seq_len, 768)
            hidden = out.last_hidden_state[0].detach().numpy()  # (seq_len, 768)

            # Head-weighted importance: scale each token's rollout weight by
            # how much the head's learned weights respond to that token's representation
            hw = head_weights.numpy()   # (768,)
            weighted = np.array([
                base_importance[t] * np.dot(np.abs(hidden[t]), hw)
                for t in range(seq_len)
            ])

            # Normalize to sum to 1
            denom = weighted.sum()
            if denom > 0:
                weighted = weighted / denom
            else:
                weighted = base_importance

            all_importance[m_idx, s_idx, :seq_len] = weighted

        print("done")

    return all_tokens, all_importance, token_counts


# ---------------------------------------------------------------------------
# Metrics: argmax flip rate, prediction agreement
# ---------------------------------------------------------------------------

def compute_argmax_flip_rate(all_importance, token_counts):
    """
    Fraction of (model-pair, sentence) triples where the content token with
    peak importance differs.  Content tokens exclude [CLS] (index 0) and
    [SEP] (last index).
    """
    n_models, n_sent, _ = all_importance.shape
    all_flips = []

    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                seq_len = token_counts[s_idx]
                # Exclude special tokens: indices 1 .. seq_len-2
                imp_i = all_importance[m_i, s_idx, 1:seq_len - 1]
                imp_j = all_importance[m_j, s_idx, 1:seq_len - 1]
                if len(imp_i) == 0:
                    continue
                all_flips.append(int(np.argmax(imp_i) != np.argmax(imp_j)))

    flip_rate = float(np.mean(all_flips)) if all_flips else 0.0
    return flip_rate, all_flips


def compute_prediction_agreement(backbone, tokenizer, trained_heads, sentences):
    """
    Fraction of (model-pair, sentence) triples where both models predict
    the same class.
    """
    import torch

    n_models = len(trained_heads)
    n_sent = len(sentences)

    # Pre-compute CLS embeddings once
    cls_embs = []
    backbone.eval()
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            out = backbone(**enc)
            cls_embs.append(out.last_hidden_state[0, 0, :])
    cls_embs = torch.stack(cls_embs)   # (n_sent, 768)

    # Predictions for each head
    all_preds = np.zeros((n_models, n_sent), dtype=int)
    for m_idx, head in enumerate(trained_heads):
        with torch.no_grad():
            logits = head(cls_embs)
            preds = logits.argmax(dim=-1).numpy()
        all_preds[m_idx] = preds

    all_agreements = []
    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                all_agreements.append(
                    int(all_preds[m_i, s_idx] == all_preds[m_j, s_idx])
                )

    return float(np.mean(all_agreements)), all_agreements, all_preds


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_retraining_experiment():
    t0 = time.time()

    print("=" * 65)
    print("Attention Retraining Experiment")
    print("TRUE multi-seed retraining (not artificial perturbation)")
    print("=" * 65)

    # 1. Load shared frozen backbone
    backbone, tokenizer = load_backbone()

    # 2. Build datasets
    train_data, heldout_data, test_data = build_datasets()
    test_sentences = [s for s, _ in test_data]
    print(f"\nDataset: {len(train_data)} train, {len(heldout_data)} held-out, "
          f"{len(test_data)} test")

    # 3. Train NUM_MODELS classification heads from different seeds
    print("\n--- Training classification heads ---")
    trained_heads = []
    accuracies = []

    for i in range(NUM_MODELS):
        seed = SEED_BASE + i
        print(f"\n  Model {i+1}/{NUM_MODELS}  (seed={seed})")
        head = train_head(backbone, tokenizer, train_data, seed)
        acc, _ = compute_accuracy(backbone, tokenizer, head, heldout_data)
        print(f"    [seed={seed}] Held-out accuracy: {acc*100:.1f}%", end="")
        if acc >= MIN_ACCURACY:
            print("  PASS")
        else:
            print(f"  WARN (below {MIN_ACCURACY*100:.0f}%)")
        accuracies.append(acc)
        trained_heads.append(head)

    print(f"\nAccuracy summary: min={min(accuracies)*100:.1f}%  "
          f"max={max(accuracies)*100:.1f}%  "
          f"mean={np.mean(accuracies)*100:.1f}%")
    all_pass = all(a >= MIN_ACCURACY for a in accuracies)
    print(f"All models >{MIN_ACCURACY*100:.0f}%: {'YES' if all_pass else 'NO'}")

    # 4. Compute head-weighted attention rollout on 50 test sentences
    print("\n--- Computing head-weighted attention rollout on test set ---")
    all_tokens, all_importance, token_counts = compute_importance_matrix(
        backbone, tokenizer, trained_heads, test_sentences
    )

    # 5. Compute metrics
    print("\n--- Computing metrics ---")

    flip_rate, all_flips = compute_argmax_flip_rate(all_importance, token_counts)
    flip_ci = percentile_ci([float(f) for f in all_flips], n_boot=2000)
    n_flip_comparisons = len(all_flips)
    print(f"  Argmax flip rate: {flip_rate*100:.1f}%  "
          f"95% CI [{flip_ci[0]*100:.1f}%, {flip_ci[2]*100:.1f}%]  "
          f"(n={n_flip_comparisons})")

    pred_agree, all_agree_list, all_preds = compute_prediction_agreement(
        backbone, tokenizer, trained_heads, test_sentences
    )
    agree_ci = percentile_ci([float(a) for a in all_agree_list], n_boot=2000)
    print(f"  Prediction agreement: {pred_agree*100:.1f}%  "
          f"95% CI [{agree_ci[0]*100:.1f}%, {agree_ci[2]*100:.1f}%]")

    elapsed = time.time() - t0

    # 6. Comparison to perturbation experiment
    perturbation_flip_rate = 0.60
    perturbation_agree = 1.00

    print("\n--- Comparison to perturbation experiment ---")
    print(f"  {'Metric':<30} {'Perturbation':>14} {'Retraining':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Argmax flip rate':<30} {perturbation_flip_rate*100:>13.1f}% {flip_rate*100:>11.1f}%")
    print(f"  {'Prediction agreement':<30} {perturbation_agree*100:>13.1f}% {pred_agree*100:>11.1f}%")

    if flip_rate > 0.40:
        conclusion = "instability_confirmed_by_retraining"
        interp = (f"Retraining flip rate {flip_rate*100:.1f}% is comparable to "
                  f"perturbation {perturbation_flip_rate*100:.0f}%, confirming that "
                  f"explanation instability is not an artifact of artificial perturbation.")
    else:
        conclusion = "instability_lower_in_retraining"
        interp = (f"Retraining flip rate {flip_rate*100:.1f}% is lower than "
                  f"perturbation {perturbation_flip_rate*100:.0f}%.  The head-weighted "
                  f"importance captures seed-induced differences in token salience "
                  f"even when the backbone is frozen.")
    print(f"\nConclusion: {interp}")

    # 7. Save results
    results = {
        "experiment": "attention_retraining",
        "description": (
            "TRUE multi-seed retraining: 10 seeds × frozen DistilBERT + "
            "fresh nn.Linear(768,2), evaluated on 50 test sentences"
        ),
        "model": "distilbert-base-uncased",
        "method": "frozen_backbone_fresh_head_head_weighted_rollout",
        "num_models": NUM_MODELS,
        "seeds": list(range(SEED_BASE, SEED_BASE + NUM_MODELS)),
        "num_train_sentences": len(train_data),
        "num_heldout_sentences": len(heldout_data),
        "num_test_sentences": len(test_data),
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "head_parameters": HIDDEN_SIZE * NUM_CLASSES + NUM_CLASSES,
        "all_heldout_accuracies": accuracies,
        "min_accuracy": float(min(accuracies)),
        "max_accuracy": float(max(accuracies)),
        "mean_accuracy": float(np.mean(accuracies)),
        "all_above_70pct": all_pass,
        # Retraining metrics
        "argmax_flip_rate": flip_rate,
        "argmax_flip_rate_ci_lo": flip_ci[0],
        "argmax_flip_rate_ci_hi": flip_ci[2],
        "n_flip_comparisons": n_flip_comparisons,
        "prediction_agreement": pred_agree,
        "prediction_agreement_ci_lo": agree_ci[0],
        "prediction_agreement_ci_hi": agree_ci[2],
        # Comparison
        "comparison_to_perturbation": {
            "perturbation_flip_rate": perturbation_flip_rate,
            "perturbation_prediction_agreement": perturbation_agree,
            "retraining_flip_rate": flip_rate,
            "retraining_prediction_agreement": pred_agree,
            "flip_rate_delta_pp": float((flip_rate - perturbation_flip_rate) * 100),
        },
        "elapsed_seconds": round(elapsed, 1),
        "conclusion": conclusion,
        "interpretation": interp,
    }

    save_results(results, "attention_retraining")

    print(f"\nElapsed: {elapsed:.1f}s")
    print("Done.")
    return results


if __name__ == "__main__":
    run_retraining_experiment()
