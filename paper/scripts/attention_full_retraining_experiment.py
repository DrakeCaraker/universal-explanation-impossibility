#!/usr/bin/env python3
"""
Attention Full Retraining Experiment
=====================================
Gold standard experiment that closes the perturbation criticism.

Research question: Do transformer models with DIFFERENT backbone initializations
(not just different classification heads) assign peak attention to different tokens?

Design:
- DistilBERT-base-uncased loaded fresh
- ALL layers FROZEN except: transformer.layer.4, transformer.layer.5,
  pre_classifier, classifier
- 200 training sentences (100 positive, 100 negative)
- 40 held-out sentences (20 positive, 20 negative)
- 50 test sentences (25 positive, 25 negative)
- 20 random seeds: unfrozen layers reset to random init, then fine-tuned
- Adam lr=2e-5, 5 epochs, batch_size=16, CrossEntropyLoss
- Verify all 20 models achieve >80% validation accuracy
- Attention rollout (output_attentions=True, all 6 layers) on 50 test sentences
- Metrics: argmax flip rate, prediction agreement, mean Kendall tau, 95% bootstrap CIs

Comparison table:
| Method                        | Flip rate | Pred. agreement | Kendall tau |
| Perturbation (σ=0.01)         | 60.0%     | 100%            | 0.30        |
| Head-only retraining          | 2.8%      | 94%             | —           |
| Full retraining (this)        | ??%       | ??%             | ??          |

Outputs:
- paper/results_attention_full_retraining.json
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
NUM_MODELS = 20
NUM_EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
SEED_BASE = 42        # seeds: 42, 43, ..., 61
HIDDEN_SIZE = 768
NUM_CLASSES = 2
MIN_ACCURACY = 0.80   # all models must beat this on held-out set
DEVICE = 'cpu'        # avoid MPS issues

# ---------------------------------------------------------------------------
# 200 training sentences (100 positive + 100 negative)
# ---------------------------------------------------------------------------
_POSITIVE_TRAIN = [
    # Template: "This movie is <adjective>"
    "This movie is great",
    "This movie is excellent",
    "This movie is wonderful",
    "This movie is fantastic",
    "This movie is amazing",
    "This movie is superb",
    "This movie is brilliant",
    "This movie is outstanding",
    "This movie is magnificent",
    "This movie is spectacular",
    # Template: "I loved this film"
    "I loved this film",
    "I adored this film",
    "I enjoyed this film",
    "I cherished this film",
    "I treasured this film",
    "I admired this film",
    "I appreciated this film",
    "I relished this film",
    "I savored this film",
    "I embraced this film",
    # Template: "Excellent <noun>"
    "Excellent acting throughout",
    "Excellent direction throughout",
    "Excellent writing throughout",
    "Excellent cinematography throughout",
    "Excellent performances throughout",
    "Excellent storytelling throughout",
    "Excellent pacing throughout",
    "Excellent editing throughout",
    "Excellent music throughout",
    "Excellent dialogue throughout",
    # Template: "A wonderful <noun> overall"
    "A wonderful experience overall",
    "A wonderful journey overall",
    "A wonderful achievement overall",
    "A wonderful creation overall",
    "A wonderful masterpiece overall",
    "A wonderful triumph overall",
    "A wonderful spectacle overall",
    "A wonderful adventure overall",
    "A wonderful discovery overall",
    "A wonderful celebration overall",
    # Template: "The <noun> was very engaging"
    "The plot was very engaging",
    "The story was very engaging",
    "The script was very engaging",
    "The narrative was very engaging",
    "The character development was very engaging",
    "The dialogue was very engaging",
    "The direction was very engaging",
    "The cinematography was very engaging",
    "The acting was very engaging",
    "The pacing was very engaging",
    # Template: "Highly recommended for everyone"
    "Highly recommended for everyone",
    "Absolutely recommended for everyone",
    "Strongly recommended for everyone",
    "Definitely recommended for everyone",
    "Enthusiastically recommended for everyone",
    "Warmly recommended for everyone",
    "Passionately recommended for everyone",
    "Wholeheartedly recommended for everyone",
    "Genuinely recommended for everyone",
    "Sincerely recommended for everyone",
    # Template: "A truly <adjective> masterpiece"
    "A truly outstanding masterpiece",
    "A truly remarkable masterpiece",
    "A truly breathtaking masterpiece",
    "A truly extraordinary masterpiece",
    "A truly magnificent masterpiece",
    "A truly exceptional masterpiece",
    "A truly unforgettable masterpiece",
    "A truly timeless masterpiece",
    "A truly compelling masterpiece",
    "A truly captivating masterpiece",
    # Template: "The direction was <adjective>"
    "The direction was superb",
    "The direction was inspired",
    "The direction was masterful",
    "The direction was brilliant",
    "The direction was exceptional",
    "The direction was visionary",
    "The direction was confident",
    "The direction was precise",
    "The direction was elegant",
    "The direction was powerful",
    # Template: "Absolutely <adjective> movie"
    "Absolutely fantastic movie",
    "Absolutely wonderful movie",
    "Absolutely brilliant movie",
    "Absolutely gorgeous movie",
    "Absolutely stunning movie",
    "Absolutely captivating movie",
    "Absolutely enthralling movie",
    "Absolutely mesmerizing movie",
    "Absolutely enchanting movie",
    "Absolutely thrilling movie",
    # Template: "I was completely <adjective> throughout"
    "I was completely captivated throughout",
    "I was completely enthralled throughout",
    "I was completely mesmerized throughout",
    "I was completely engaged throughout",
    "I was completely absorbed throughout",
    "I was completely moved throughout",
    "I was completely enchanted throughout",
    "I was completely delighted throughout",
    "I was completely impressed throughout",
    "I was completely satisfied throughout",
]

_NEGATIVE_TRAIN = [
    # Template: "This movie is <adjective>"
    "This movie is terrible",
    "This movie is awful",
    "This movie is bad",
    "This movie is horrible",
    "This movie is dreadful",
    "This movie is atrocious",
    "This movie is dreadful",
    "This movie is abysmal",
    "This movie is wretched",
    "This movie is appalling",
    # Template: "I hated this film"
    "I hated this film",
    "I despised this film",
    "I disliked this film",
    "I loathed this film",
    "I detested this film",
    "I abhorred this film",
    "I resented this film",
    "I regretted this film",
    "I endured this film",
    "I suffered through this film",
    # Template: "Awful <noun>"
    "Awful acting throughout",
    "Awful direction throughout",
    "Awful writing throughout",
    "Awful cinematography throughout",
    "Awful performances throughout",
    "Awful storytelling throughout",
    "Awful pacing throughout",
    "Awful editing throughout",
    "Awful music throughout",
    "Awful dialogue throughout",
    # Template: "A dreadful <noun> overall"
    "A dreadful experience overall",
    "A dreadful journey overall",
    "A dreadful waste overall",
    "A dreadful ordeal overall",
    "A dreadful disaster overall",
    "A dreadful failure overall",
    "A dreadful mess overall",
    "A dreadful disappointment overall",
    "A dreadful embarrassment overall",
    "A dreadful bore overall",
    # Template: "The <noun> was very boring"
    "The plot was very boring",
    "The story was very boring",
    "The script was very boring",
    "The narrative was very boring",
    "The character development was very boring",
    "The dialogue was very boring",
    "The direction was very boring",
    "The cinematography was very boring",
    "The acting was very boring",
    "The pacing was very boring",
    # Template: "Not recommended for anyone"
    "Not recommended for anyone",
    "Absolutely not recommended for anyone",
    "Strongly not recommended for anyone",
    "Definitely not recommended for anyone",
    "Emphatically not recommended for anyone",
    "Seriously not recommended for anyone",
    "Frankly not recommended for anyone",
    "Honestly not recommended for anyone",
    "Genuinely not recommended for anyone",
    "Sincerely not recommended for anyone",
    # Template: "A truly <adjective> disaster"
    "A truly disappointing disaster",
    "A truly dreadful disaster",
    "A truly terrible disaster",
    "A truly awful disaster",
    "A truly horrific disaster",
    "A truly appalling disaster",
    "A truly pathetic disaster",
    "A truly embarrassing disaster",
    "A truly forgettable disaster",
    "A truly painful disaster",
    # Template: "The direction was <adjective>"
    "The direction was awful",
    "The direction was confused",
    "The direction was incoherent",
    "The direction was clumsy",
    "The direction was weak",
    "The direction was amateurish",
    "The direction was uninspired",
    "The direction was incompetent",
    "The direction was sloppy",
    "The direction was lifeless",
    # Template: "Absolutely <adjective> movie"
    "Absolutely horrible movie",
    "Absolutely dreadful movie",
    "Absolutely terrible movie",
    "Absolutely awful movie",
    "Absolutely atrocious movie",
    "Absolutely appalling movie",
    "Absolutely abysmal movie",
    "Absolutely wretched movie",
    "Absolutely ghastly movie",
    "Absolutely dismal movie",
    # Template: "I was completely <adjective> throughout"
    "I was completely bored throughout",
    "I was completely disappointed throughout",
    "I was completely frustrated throughout",
    "I was completely confused throughout",
    "I was completely alienated throughout",
    "I was completely annoyed throughout",
    "I was completely disengaged throughout",
    "I was completely unimpressed throughout",
    "I was completely dissatisfied throughout",
    "I was completely miserable throughout",
]

assert len(_POSITIVE_TRAIN) == 100, f"Need 100 positive train, got {len(_POSITIVE_TRAIN)}"
assert len(_NEGATIVE_TRAIN) == 100, f"Need 100 negative train, got {len(_NEGATIVE_TRAIN)}"

# ---------------------------------------------------------------------------
# 40 held-out validation sentences (20 positive, 20 negative)
# ---------------------------------------------------------------------------
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
    "The performances were extraordinary and moving",
    "Every moment of this film was pure gold",
    "A masterwork of cinematic storytelling",
    "I felt completely transported by this story",
    "The director achieved something truly special here",
    "An unforgettable and deeply moving experience",
    "This film elevated the art of cinema",
    "Magnificent in every conceivable way",
    "A genuine triumph of human creativity",
    "I left the theater feeling uplifted and inspired",
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
    "The performances were wooden and unconvincing",
    "Every moment of this film was pure agony",
    "A failure of cinematic storytelling",
    "I felt completely alienated by this story",
    "The director failed at every conceivable level",
    "An unforgettable and deeply irritating experience",
    "This film degraded the art of cinema",
    "Mediocre in every conceivable way",
    "A genuine failure of human creativity",
    "I left the theater feeling drained and cheated",
]

assert len(_POSITIVE_HELDOUT) == 20
assert len(_NEGATIVE_HELDOUT) == 20

# ---------------------------------------------------------------------------
# 50 test sentences (25 positive, 25 negative)
# ---------------------------------------------------------------------------
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

assert len(_POSITIVE_TEST) == 25
assert len(_NEGATIVE_TEST) == 25


def build_datasets():
    """Return train, heldout, and test (sentence, label) lists."""
    train = ([(s, 1) for s in _POSITIVE_TRAIN] +
             [(s, 0) for s in _NEGATIVE_TRAIN])
    heldout = ([(s, 1) for s in _POSITIVE_HELDOUT] +
               [(s, 0) for s in _NEGATIVE_HELDOUT])
    test = ([(s, 1) for s in _POSITIVE_TEST] +
            [(s, 0) for s in _NEGATIVE_TEST])
    return train, heldout, test


# ---------------------------------------------------------------------------
# Weight reset function
# ---------------------------------------------------------------------------

def reset_weights(m):
    """Reinitialize unfrozen layers to random state."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


# ---------------------------------------------------------------------------
# Attention rollout (propagate through all 6 layers)
# ---------------------------------------------------------------------------

def attention_rollout(attentions, seq_len):
    """
    Compute attention rollout across all layers.

    For each layer, average attention across heads, then multiply layer
    matrices together with residual identity connection. Returns the
    CLS token's aggregated attention to all tokens.

    Parameters
    ----------
    attentions : tuple of (batch=1, heads, seq, seq) tensors from all 6 layers
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
# Model loading and layer freezing
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load DistilBertForSequenceClassification and tokenizer."""
    import torch
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
    )

    print("Loading distilbert-base-uncased (DistilBertForSequenceClassification)...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        output_attentions=True,
    )

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze: last 2 transformer layers (4 and 5), pre_classifier, classifier
    for param in model.distilbert.transformer.layer[4].parameters():
        param.requires_grad_(True)
    for param in model.distilbert.transformer.layer[5].parameters():
        param.requires_grad_(True)
    for param in model.pre_classifier.parameters():
        param.requires_grad_(True)
    for param in model.classifier.parameters():
        param.requires_grad_(True)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_trainable
    print(f"  Total parameters: {n_total:,}")
    print(f"  Frozen parameters: {n_frozen:,}")
    print(f"  Trainable parameters (layers 4+5 + pre_classifier + classifier): {n_trainable:,}")

    return model, tokenizer


def reset_unfrozen_layers(model, seed):
    """Reset the unfrozen layers to random initialization (seed-controlled)."""
    import torch
    torch.manual_seed(seed)
    set_all_seeds(seed)

    model.distilbert.transformer.layer[4].apply(reset_weights)
    model.distilbert.transformer.layer[5].apply(reset_weights)
    model.pre_classifier.apply(reset_weights)
    model.classifier.apply(reset_weights)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_model(model, tokenizer, train_data, seed):
    """
    Fine-tune the unfrozen layers for NUM_EPOCHS.

    Returns the model with updated weights.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Reset unfrozen layers to seed-specific random init
    reset_unfrozen_layers(model, seed)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )
    criterion = nn.CrossEntropyLoss()

    sentences = [s for s, _ in train_data]
    labels_list = [y for _, y in train_data]

    model.train()
    for epoch in range(NUM_EPOCHS):
        # Shuffle with seed
        rng = np.random.RandomState(seed + epoch * 1000)
        indices = rng.permutation(len(train_data))
        shuffled_sentences = [sentences[i] for i in indices]
        shuffled_labels = [labels_list[i] for i in indices]

        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_data), BATCH_SIZE):
            batch_sents = shuffled_sentences[start:start + BATCH_SIZE]
            batch_lbls = shuffled_labels[start:start + BATCH_SIZE]

            enc = tokenizer(
                batch_sents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            labels_tensor = torch.tensor(batch_lbls, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(**enc, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        print(f"    [seed={seed}] Epoch {epoch+1}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracy(model, tokenizer, data):
    """Compute accuracy on a dataset."""
    import torch

    sentences = [s for s, _ in data]
    true_labels = np.array([y for _, y in data])
    preds = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sents = sentences[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch_sents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            outputs = model(**enc)
            batch_preds = outputs.logits.argmax(dim=-1).numpy()
            preds.extend(batch_preds.tolist())

    preds = np.array(preds)
    return float(np.mean(preds == true_labels)), preds


# ---------------------------------------------------------------------------
# Attention rollout on test sentences
# ---------------------------------------------------------------------------

def compute_attention_rollout_for_model(model, tokenizer, test_sentences):
    """
    Compute attention rollout for all test sentences using one model.

    Returns importance: np.ndarray of shape (n_sent, max_seq_len)
    and token_counts: list of int
    """
    import torch

    n_sent = len(test_sentences)
    token_counts = []

    # First pass: get token counts
    for sent in test_sentences:
        enc = tokenizer(sent, return_tensors="pt", padding=False,
                        truncation=True, max_length=64)
        token_counts.append(enc["input_ids"].shape[1])

    max_seq = max(token_counts)
    importance = np.zeros((n_sent, max_seq))

    model.eval()
    with torch.no_grad():
        for s_idx, sent in enumerate(test_sentences):
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            outputs = model(**enc, output_attentions=True)
            seq_len = token_counts[s_idx]
            rollout = attention_rollout(outputs.attentions, seq_len)
            importance[s_idx, :seq_len] = rollout

    return importance, token_counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_argmax_flip_rate(all_importance, token_counts):
    """
    Fraction of (model-pair, sentence) triples where the content token
    with peak attention differs. Content tokens exclude [CLS] (index 0)
    and [SEP] (last index).
    """
    n_models, n_sent, _ = all_importance.shape
    all_flips = []

    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                seq_len = token_counts[s_idx]
                imp_i = all_importance[m_i, s_idx, 1:seq_len - 1]
                imp_j = all_importance[m_j, s_idx, 1:seq_len - 1]
                if len(imp_i) == 0:
                    continue
                all_flips.append(int(np.argmax(imp_i) != np.argmax(imp_j)))

    flip_rate = float(np.mean(all_flips)) if all_flips else 0.0
    return flip_rate, all_flips


def compute_prediction_agreement_matrix(all_preds):
    """
    Compute pairwise prediction agreement across (n_models, n_sent) predictions.
    Returns mean agreement and list of per-triple agreement values.
    """
    n_models, n_sent = all_preds.shape
    all_agreements = []

    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                all_agreements.append(
                    int(all_preds[m_i, s_idx] == all_preds[m_j, s_idx])
                )

    return float(np.mean(all_agreements)), all_agreements


def compute_mean_kendall_tau(all_importance, token_counts):
    """
    Compute mean Kendall tau of attention distributions across model pairs.
    For each sentence and model pair, compute Kendall tau between the
    attention importance vectors over content tokens.

    Returns mean tau and list of per-triple tau values.
    """
    from scipy.stats import kendalltau

    n_models, n_sent, _ = all_importance.shape
    all_taus = []

    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                seq_len = token_counts[s_idx]
                # Content tokens (exclude CLS and SEP)
                imp_i = all_importance[m_i, s_idx, 1:seq_len - 1]
                imp_j = all_importance[m_j, s_idx, 1:seq_len - 1]
                if len(imp_i) <= 1:
                    continue
                tau, _ = kendalltau(imp_i, imp_j)
                if not np.isnan(tau):
                    all_taus.append(float(tau))

    mean_tau = float(np.mean(all_taus)) if all_taus else 0.0
    return mean_tau, all_taus


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_full_retraining_experiment():
    t0 = time.time()

    print("=" * 70)
    print("Attention Full Retraining Experiment")
    print("Gold standard: 20 seeds, layers 4+5 + head reset and fine-tuned")
    print("=" * 70)

    # 1. Build datasets
    train_data, heldout_data, test_data = build_datasets()
    test_sentences = [s for s, _ in test_data]
    print(f"\nDataset: {len(train_data)} train, {len(heldout_data)} held-out, "
          f"{len(test_data)} test")

    # 2. Load base model + tokenizer (once)
    model, tokenizer = load_model_and_tokenizer()

    # 3. For each seed: reset unfrozen layers, fine-tune, record accuracy
    print("\n--- Fine-tuning models (20 seeds) ---")
    all_accuracies = []
    all_importance_list = []
    all_preds_list = []
    token_counts_ref = None

    for i in range(NUM_MODELS):
        seed = SEED_BASE + i
        print(f"\n  Model {i+1}/{NUM_MODELS}  (seed={seed})")

        # Fine-tune
        model = fine_tune_model(model, tokenizer, train_data, seed)

        # Validation accuracy
        acc, _ = compute_accuracy(model, tokenizer, heldout_data)
        status = "PASS" if acc >= MIN_ACCURACY else f"WARN (below {MIN_ACCURACY*100:.0f}%)"
        print(f"    [seed={seed}] Held-out accuracy: {acc*100:.1f}%  {status}")
        all_accuracies.append(acc)

        # Attention rollout on test set
        print(f"    [seed={seed}] Computing attention rollout on {len(test_sentences)} test sentences...",
              end=" ", flush=True)
        imp, token_counts = compute_attention_rollout_for_model(
            model, tokenizer, test_sentences
        )
        if token_counts_ref is None:
            token_counts_ref = token_counts
        all_importance_list.append(imp)
        print("done")

        # Predictions on test set
        _, preds = compute_accuracy(model, tokenizer, test_data)
        all_preds_list.append(preds)

    # Stack into arrays
    all_importance = np.stack(all_importance_list, axis=0)  # (n_models, n_sent, max_seq)
    all_preds = np.stack(all_preds_list, axis=0)            # (n_models, n_sent)

    # 4. Verify accuracy threshold
    all_pass = all(a >= MIN_ACCURACY for a in all_accuracies)
    print(f"\nAccuracy summary: min={min(all_accuracies)*100:.1f}%  "
          f"max={max(all_accuracies)*100:.1f}%  "
          f"mean={np.mean(all_accuracies)*100:.1f}%")
    print(f"All models >{MIN_ACCURACY*100:.0f}%: {'YES' if all_pass else 'NO'}")
    if not all_pass:
        failing = [i for i, a in enumerate(all_accuracies) if a < MIN_ACCURACY]
        print(f"  WARNING: Models {failing} did not reach {MIN_ACCURACY*100:.0f}%")

    # 5. Compute metrics
    print("\n--- Computing metrics ---")

    flip_rate, all_flips = compute_argmax_flip_rate(all_importance, token_counts_ref)
    flip_ci = percentile_ci([float(f) for f in all_flips], n_boot=2000)
    n_flip_comparisons = len(all_flips)
    print(f"  Argmax flip rate: {flip_rate*100:.1f}%  "
          f"95% CI [{flip_ci[0]*100:.1f}%, {flip_ci[2]*100:.1f}%]  "
          f"(n={n_flip_comparisons})")

    pred_agree, all_agree_list = compute_prediction_agreement_matrix(all_preds)
    agree_ci = percentile_ci([float(a) for a in all_agree_list], n_boot=2000)
    print(f"  Prediction agreement: {pred_agree*100:.1f}%  "
          f"95% CI [{agree_ci[0]*100:.1f}%, {agree_ci[2]*100:.1f}%]")

    mean_tau, all_taus = compute_mean_kendall_tau(all_importance, token_counts_ref)
    tau_ci = percentile_ci([float(t) for t in all_taus], n_boot=2000)
    print(f"  Mean Kendall tau: {mean_tau:.4f}  "
          f"95% CI [{tau_ci[0]:.4f}, {tau_ci[2]:.4f}]  "
          f"(n={len(all_taus)})")

    elapsed = time.time() - t0

    # 6. Comparison table
    print("\n--- Comparison Table ---")
    print(f"  {'Method':<35} {'Flip rate':>10} {'Pred. agree':>13} {'Kendall tau':>13}")
    print(f"  {'-'*73}")
    print(f"  {'Perturbation (σ=0.01)':<35} {'60.0%':>10} {'100%':>13} {'0.30':>13}")
    print(f"  {'Head-only retraining':<35} {'2.8%':>10} {'94%':>13} {'—':>13}")
    print(f"  {'Full retraining (this)':<35} {flip_rate*100:>9.1f}% {pred_agree*100:>12.1f}% {mean_tau:>13.4f}")

    # 7. Interpretation
    if flip_rate > 0.40:
        conclusion = "instability_confirmed_by_full_retraining"
        interp = (
            f"Full retraining flip rate {flip_rate*100:.1f}% is comparable to "
            f"perturbation 60.0%, confirming that explanation instability is a "
            f"structural property of underspecified models, not an artifact of "
            f"artificial weight perturbation."
        )
    elif flip_rate > 0.10:
        conclusion = "moderate_instability_confirmed"
        interp = (
            f"Full retraining flip rate {flip_rate*100:.1f}% demonstrates meaningful "
            f"attention instability from genuine re-initialization and fine-tuning "
            f"of the backbone layers, well above the head-only retraining rate of 2.8%."
        )
    else:
        conclusion = "instability_lower_than_perturbation"
        interp = (
            f"Full retraining flip rate {flip_rate*100:.1f}% is lower than the "
            f"perturbation baseline of 60.0%. The backbone layers 0-3 provide "
            f"substantial representational constraint even with layers 4+5 re-initialized. "
            f"This quantifies the structural contribution of pretrained representations "
            f"to explanation stability."
        )
    print(f"\nConclusion: {interp}")

    # 8. Save results
    results = {
        "experiment": "attention_full_retraining",
        "description": (
            "Full retraining: 20 seeds × DistilBERT with layers 4+5 + "
            "pre_classifier + classifier reset and fine-tuned (layers 0-3 frozen). "
            "Evaluated on 50 test sentences using attention rollout."
        ),
        "model": "distilbert-base-uncased",
        "method": "partial_backbone_retraining_attention_rollout",
        "frozen_layers": [0, 1, 2, 3],
        "retrained_layers": [4, 5, "pre_classifier", "classifier"],
        "num_models": NUM_MODELS,
        "seeds": list(range(SEED_BASE, SEED_BASE + NUM_MODELS)),
        "num_train_sentences": len(train_data),
        "num_heldout_sentences": len(heldout_data),
        "num_test_sentences": len(test_data),
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "min_accuracy_threshold": MIN_ACCURACY,
        "all_heldout_accuracies": all_accuracies,
        "min_accuracy": float(min(all_accuracies)),
        "max_accuracy": float(max(all_accuracies)),
        "mean_accuracy": float(np.mean(all_accuracies)),
        "all_above_80pct": all_pass,
        # Metrics
        "argmax_flip_rate": flip_rate,
        "argmax_flip_rate_ci_lo": flip_ci[0],
        "argmax_flip_rate_ci_hi": flip_ci[2],
        "n_flip_comparisons": n_flip_comparisons,
        "prediction_agreement": pred_agree,
        "prediction_agreement_ci_lo": agree_ci[0],
        "prediction_agreement_ci_hi": agree_ci[2],
        "mean_kendall_tau": mean_tau,
        "mean_kendall_tau_ci_lo": tau_ci[0],
        "mean_kendall_tau_ci_hi": tau_ci[2],
        "n_tau_comparisons": len(all_taus),
        # Comparison
        "comparison_table": {
            "perturbation_sigma_001": {
                "flip_rate": 0.60,
                "prediction_agreement": 1.00,
                "kendall_tau": 0.30,
            },
            "head_only_retraining": {
                "flip_rate": 0.028,
                "prediction_agreement": 0.94,
                "kendall_tau": None,
            },
            "full_retraining_this": {
                "flip_rate": flip_rate,
                "prediction_agreement": pred_agree,
                "kendall_tau": mean_tau,
                "flip_rate_ci": [flip_ci[0], flip_ci[2]],
                "prediction_agreement_ci": [agree_ci[0], agree_ci[2]],
                "kendall_tau_ci": [tau_ci[0], tau_ci[2]],
            },
        },
        "elapsed_seconds": round(elapsed, 1),
        "conclusion": conclusion,
        "interpretation": interp,
    }

    save_results(results, "attention_full_retraining")

    print(f"\nElapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("Done.")
    return results


if __name__ == "__main__":
    run_full_retraining_experiment()
