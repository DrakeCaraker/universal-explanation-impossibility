#!/usr/bin/env python3
"""
LLM Self-Explanation Impossibility: Attention-based feature attribution instability.

Tests whether attention-based "explanations" from DistilBERT are stable across
inference runs with different dropout masks. Even with fixed weights, dropout
during inference produces different attention patterns — a form of Rashomon
where multiple internal reasoning paths yield the same output class.

Design:
  - 50 sentences, 30 forward passes each with dropout active
  - Manual forward pass to avoid PyTorch SDPA deadlock on macOS
  - Token importance = mean attention to each token across heads (last layer)
  - Calibration (runs 1-15) / Validation (runs 16-30) split
  - Gaussian flip formula: predicted flip rate from Delta/sigma
  - Key metrics: mean flip rate, fraction unreliable (SNR < 0.5), OOS R²
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---------- sentences ----------
SENTENCES = [
    # Positive sentiment
    "This movie is absolutely wonderful and I loved every moment of it.",
    "The food was delicious and the service was outstanding.",
    "I had a fantastic experience at this hotel, highly recommended.",
    "The book was incredibly engaging and kept me reading until dawn.",
    "What a beautiful day, everything went perfectly.",
    "The team delivered an exceptional performance tonight.",
    "I am so happy with this purchase, it exceeded my expectations.",
    "The concert was breathtaking and the music was phenomenal.",
    "This is the best restaurant I have ever visited.",
    "The new software update is fast, clean, and intuitive.",
    "Her speech was inspiring and moved the entire audience.",
    "The garden looks absolutely stunning this time of year.",
    "I really enjoyed the thoughtful design of this product.",
    "The customer support team was incredibly helpful and kind.",
    "This vacation has been the most relaxing trip of my life.",
    "The acting in this film is superb and deeply moving.",
    "What an amazing achievement by the research team.",
    "The sunset over the ocean was the most beautiful thing I have seen.",
    "I am thrilled with the results of the experiment.",
    "The children were laughing and playing, it was heartwarming.",
    "This coffee shop has the coziest atmosphere in town.",
    "The documentary was informative and brilliantly produced.",
    "I feel grateful for the wonderful support from my colleagues.",
    "The new park is a fantastic addition to our neighborhood.",
    "Every dish at this restaurant was a masterpiece.",
    # Negative sentiment
    "This movie was terrible and a complete waste of time.",
    "The food was cold and the waiter was extremely rude.",
    "I had an awful experience, the hotel room was dirty and noisy.",
    "The book was boring and I could not finish the first chapter.",
    "What a horrible day, nothing went as planned.",
    "The team played terribly and deserved to lose.",
    "I am very disappointed with this product, it broke after one day.",
    "The concert was dreadful, the sound quality was atrocious.",
    "This is the worst restaurant I have ever been to.",
    "The software is buggy, slow, and crashes constantly.",
    "His speech was confusing and put everyone to sleep.",
    "The garden is overgrown and looks completely neglected.",
    "The design of this product is frustrating and poorly thought out.",
    "Customer support was unhelpful, I waited for hours.",
    "This vacation was stressful and nothing like what was promised.",
    "The acting was wooden and the plot made no sense at all.",
    "What a disappointing outcome after months of hard work.",
    "The view from the room was just a parking lot and a dumpster.",
    "The experiment failed and produced no useful results.",
    "The children were crying and miserable the entire trip.",
    "This coffee shop is overpriced and the coffee tastes burnt.",
    "The documentary was biased and poorly researched.",
    "I feel frustrated by the lack of communication from the team.",
    "The old playground is dangerous and badly maintained.",
    "Every dish at this restaurant was bland and overcooked.",
]

N_RUNS = 30
N_CAL = 15   # calibration runs (1..15)
N_VAL = 15   # validation runs (16..30)
SNR_THRESHOLD = 0.5


def manual_distilbert_forward(model, input_ids, attention_mask, dropout_p=0.1, training=True):
    """Manual forward pass through DistilBERT with explicit dropout and attention extraction.

    Avoids the SDPA/threading deadlock in PyTorch 2.8 on macOS by computing
    multi-head attention manually.

    Returns:
        logits: (batch, n_classes)
        all_attentions: list of (batch, n_heads, seq_len, seq_len) per layer
    """
    distilbert = model.distilbert
    n_heads = distilbert.config.n_heads
    dim = distilbert.config.dim
    head_dim = dim // n_heads

    # Embeddings
    hidden = distilbert.embeddings(input_ids)  # (1, seq_len, dim)
    seq_len = hidden.shape[1]

    # Create attention mask for padding
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, seq_len)
        mask = (1.0 - mask.float()) * -1e9
    else:
        mask = None

    all_attentions = []

    for layer in distilbert.transformer.layer:
        sa = layer.attention

        # Q, K, V projections
        q = sa.q_lin(hidden)  # (1, seq_len, dim)
        k = sa.k_lin(hidden)
        v = sa.v_lin(hidden)

        # Reshape for multi-head: (1, seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
        q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
        v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # (1, n_heads, seq_len, seq_len)
        all_attentions.append(attn_weights.detach())

        # Apply dropout to attention weights
        if training:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

        # Context
        context = torch.matmul(attn_weights, v)  # (1, n_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(1, seq_len, dim)
        context = sa.out_lin(context)

        # SA residual + layer norm
        hidden = layer.sa_layer_norm(context + hidden)

        # FFN
        ffn_out = layer.ffn.lin1(hidden)
        ffn_out = F.gelu(ffn_out)
        if training:
            ffn_out = F.dropout(ffn_out, p=dropout_p, training=True)
        ffn_out = layer.ffn.lin2(ffn_out)
        if training:
            ffn_out = F.dropout(ffn_out, p=dropout_p, training=True)

        # FFN residual + layer norm
        hidden = layer.output_layer_norm(ffn_out + hidden)

    # Pre-classifier + classifier (DistilBertForSequenceClassification)
    hidden_cls = hidden[:, 0]  # CLS token
    pooled = model.pre_classifier(hidden_cls)
    pooled = F.relu(pooled)
    if training:
        pooled = F.dropout(pooled, p=dropout_p, training=True)
    logits = model.classifier(pooled)

    return logits, all_attentions


def extract_token_importance(model, tokenizer, sentence, n_runs, device):
    """Run n_runs forward passes with dropout active, return token importance matrix.

    Returns:
        importance: (n_runs, seq_len) array — mean attention to each token across heads
        tokens: list of token strings
        predictions: (n_runs,) array — predicted class per run
    """
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)

    importance = np.zeros((n_runs, seq_len))
    predictions = np.zeros(n_runs, dtype=int)

    for r in range(n_runs):
        torch.manual_seed(r * 137 + 42)  # different dropout mask each run
        with torch.no_grad():
            logits, attentions = manual_distilbert_forward(
                model, input_ids, attention_mask, training=True
            )

        # Last layer attention: (1, n_heads, seq_len, seq_len)
        last_attn = attentions[-1][0]  # (n_heads, seq_len, seq_len)

        # Token importance = mean attention received by each token across all heads
        # Sum over source positions (dim=-2), mean over heads (dim=0)
        attn_received = last_attn.mean(dim=0).sum(dim=0)  # (seq_len,)
        importance[r] = attn_received.cpu().numpy()

        predictions[r] = logits.argmax(dim=-1).item()

    return importance, tokens, predictions


def measure_pairwise_flip_rates(importance_matrix):
    """Compute pairwise flip rates: fraction of run-pairs where ranking flips.

    Vectorized: for each token pair (i,j), compute sign of (imp_i - imp_j) per run,
    then count disagreements across all run-pairs.
    """
    n_runs, seq_len = importance_matrix.shape
    pairs = []
    flip_rates = []

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            pairs.append((i, j))
            signs = importance_matrix[:, i] > importance_matrix[:, j]  # (n_runs,) bool
            n_true = np.sum(signs)
            n_false = n_runs - n_true
            n_flips = n_true * n_false
            n_comparisons = n_runs * (n_runs - 1) // 2
            flip_rates.append(n_flips / n_comparisons if n_comparisons > 0 else 0.0)

    return np.array(flip_rates), pairs


def predict_flip_gaussian(importance_matrix):
    """Predict flip rates using Gaussian CDF formula from calibration data."""
    n_runs, seq_len = importance_matrix.shape
    predicted = []

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            diffs = importance_matrix[:, i] - importance_matrix[:, j]
            delta = np.mean(diffs)
            sigma = np.std(diffs)
            if sigma > 1e-12:
                predicted.append(norm.cdf(-abs(delta) / (sigma * np.sqrt(2))))
            else:
                predicted.append(0.0)

    return np.array(predicted)


def compute_snr(importance_matrix):
    """Compute SNR = |delta|/sigma for each token pair."""
    n_runs, seq_len = importance_matrix.shape
    snrs = []

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            diffs = importance_matrix[:, i] - importance_matrix[:, j]
            delta = np.mean(diffs)
            sigma = np.std(diffs)
            snr = abs(delta) / sigma if sigma > 1e-12 else float('inf')
            snrs.append(snr)

    return np.array(snrs)


def r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot < 1e-15:
        return float('nan')
    return 1.0 - ss_res / ss_tot


# ---------- main experiment ----------
print("=" * 70)
print("LLM SELF-EXPLANATION IMPOSSIBILITY")
print("Attention-based token importance stability across dropout masks")
print(f"Model: distilbert-base-uncased-finetuned-sst-2-english")
print(f"Sentences: {len(SENTENCES)} | Runs per sentence: {N_RUNS}")
print(f"Calibration: runs 1-{N_CAL} | Validation: runs {N_CAL+1}-{N_RUNS}")
print("=" * 70)

device = torch.device('cpu')
print("\nLoading model and tokenizer...", flush=True)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english',
    attn_implementation='eager'
)
model.to(device)
# Put in eval mode (we handle dropout manually in our forward pass)
model.eval()

# Quick smoke test
print("Smoke test (1 forward pass)...", flush=True)
t0 = time.time()
test_inputs = tokenizer('hello', return_tensors='pt')
with torch.no_grad():
    logits, attns = manual_distilbert_forward(
        model, test_inputs['input_ids'], test_inputs['attention_mask'], training=True
    )
print(f"  OK: {time.time()-t0:.2f}s, logits={logits.tolist()}, attn layers={len(attns)}", flush=True)

# Collect results across all sentences
all_cal_predicted = []
all_val_observed = []
all_snrs = []
all_flip_rates = []
per_sentence_results = []
n_prediction_flips = 0
total_sentences = 0

for s_idx, sentence in enumerate(SENTENCES):
    total_sentences += 1
    short = sentence[:50] + "..." if len(sentence) > 50 else sentence
    t_sent = time.time()

    # Run all forward passes
    importance, tokens, predictions = extract_token_importance(
        model, tokenizer, sentence, N_RUNS, device
    )

    # Check if the output class flips across runs
    unique_preds = np.unique(predictions)
    pred_stable = len(unique_preds) == 1

    if not pred_stable:
        n_prediction_flips += 1

    seq_len = importance.shape[1]
    n_pairs = seq_len * (seq_len - 1) // 2

    # Split into calibration (runs 0..14) and validation (runs 15..29)
    cal_imp = importance[:N_CAL]
    val_imp = importance[N_CAL:]

    # Gaussian prediction from calibration
    cal_predicted = predict_flip_gaussian(cal_imp)

    # Observed flip rates from validation
    val_observed, pairs = measure_pairwise_flip_rates(val_imp)

    # SNR from calibration
    snrs = compute_snr(cal_imp)

    # Per-sentence metrics
    mean_flip = float(np.mean(val_observed))
    frac_unreliable = float(np.mean(snrs < SNR_THRESHOLD))
    oos_r2 = r_squared(val_observed, cal_predicted)

    per_sentence_results.append({
        "sentence_idx": s_idx,
        "sentence": sentence[:80],
        "n_tokens": seq_len,
        "n_pairs": n_pairs,
        "prediction_stable": bool(pred_stable),
        "mean_flip_rate": round(mean_flip, 4),
        "frac_unreliable": round(frac_unreliable, 4),
        "oos_r2": round(float(oos_r2), 4) if np.isfinite(oos_r2) else None
    })

    all_cal_predicted.append(cal_predicted)
    all_val_observed.append(val_observed)
    all_snrs.append(snrs)
    all_flip_rates.append(val_observed)

    elapsed = time.time() - t_sent
    print(f"[{s_idx+1}/{len(SENTENCES)}] {short}  "
          f"tokens={seq_len}, pairs={n_pairs}, flip={mean_flip:.3f}, "
          f"unrel={frac_unreliable:.3f}, R²={oos_r2:.3f}, "
          f"stable={pred_stable} ({elapsed:.1f}s)", flush=True)

# ---------- aggregate metrics ----------
all_cal_predicted = np.concatenate(all_cal_predicted)
all_val_observed = np.concatenate(all_val_observed)
all_snrs = np.concatenate(all_snrs)
all_flip_rates_flat = np.concatenate(all_flip_rates)

aggregate_oos_r2 = r_squared(all_val_observed, all_cal_predicted)
aggregate_mean_flip = float(np.mean(all_flip_rates_flat))
aggregate_frac_unreliable = float(np.mean(all_snrs < SNR_THRESHOLD))
aggregate_max_flip = float(np.max(all_flip_rates_flat))

print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)
print(f"Total token pairs analyzed: {len(all_val_observed)}")
print(f"Mean flip rate:             {aggregate_mean_flip:.4f}")
print(f"Max flip rate:              {aggregate_max_flip:.4f}")
print(f"Fraction unreliable (SNR<{SNR_THRESHOLD}): {aggregate_frac_unreliable:.4f}")
print(f"OOS R² (Gaussian flip):     {aggregate_oos_r2:.4f}")
print(f"Prediction flips:           {n_prediction_flips}/{total_sentences} sentences")

# ---------- save results ----------
results = {
    "experiment": "LLM Self-Explanation Impossibility",
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "n_sentences": len(SENTENCES),
    "n_runs": N_RUNS,
    "n_calibration": N_CAL,
    "n_validation": N_VAL,
    "snr_threshold": SNR_THRESHOLD,
    "aggregate": {
        "total_token_pairs": int(len(all_val_observed)),
        "mean_flip_rate": round(aggregate_mean_flip, 4),
        "max_flip_rate": round(aggregate_max_flip, 4),
        "frac_unreliable": round(aggregate_frac_unreliable, 4),
        "oos_r2_gaussian_flip": round(float(aggregate_oos_r2), 4),
        "prediction_flip_sentences": n_prediction_flips,
        "prediction_flip_fraction": round(n_prediction_flips / total_sentences, 4)
    },
    "per_sentence": per_sentence_results
}

results_path = OUT_DIR / 'results_llm_self_explanation.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

# Panel A: Predicted vs Observed flip rate (scatter)
ax = axes[0]
ax.scatter(all_cal_predicted, all_val_observed, alpha=0.08, s=4, c='steelblue', edgecolors='none')
lims = [0, max(np.max(all_cal_predicted), np.max(all_val_observed)) * 1.05 + 0.01]
ax.plot(lims, lims, 'k--', lw=1, label='y = x')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Predicted flip rate (Gaussian, calibration)')
ax.set_ylabel('Observed flip rate (validation)')
ax.set_title('A. Gaussian Flip Calibration')
ax.text(0.05, 0.92, f"OOS R$^2$ = {aggregate_oos_r2:.3f}",
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.text(0.05, 0.80, f"n = {len(all_val_observed):,} pairs",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))

# Panel B: Distribution of flip rates
ax = axes[1]
ax.hist(all_flip_rates_flat, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(aggregate_mean_flip, color='red', lw=2, ls='--', label=f'Mean = {aggregate_mean_flip:.3f}')
ax.set_xlabel('Token-pair flip rate')
ax.set_ylabel('Count')
ax.set_title('B. Flip Rate Distribution')
ax.legend(fontsize=9)

# Panel C: SNR distribution with threshold
ax = axes[2]
finite_snrs = all_snrs[np.isfinite(all_snrs)]
ax.hist(finite_snrs, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(SNR_THRESHOLD, color='red', lw=2, ls='--',
           label=f'SNR = {SNR_THRESHOLD} ({aggregate_frac_unreliable:.1%} below)')
ax.set_xlabel('Signal-to-noise ratio |$\\Delta$| / $\\sigma$')
ax.set_ylabel('Count')
ax.set_title('C. SNR Distribution')
ax.legend(fontsize=9)

fig.suptitle('LLM Self-Explanation Impossibility: Attention Instability Under Dropout',
             fontsize=13, fontweight='bold')

fig_path = FIG_DIR / 'llm_self_explanation.pdf'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")

# ---------- summary ----------
print("\n" + "=" * 70)
print("HEADLINE: Attention-based token importance is unstable under dropout.")
print(f"  {aggregate_frac_unreliable:.1%} of token-pair comparisons are unreliable (SNR < {SNR_THRESHOLD})")
print(f"  Mean flip rate = {aggregate_mean_flip:.3f}")
print(f"  Gaussian flip formula OOS R² = {aggregate_oos_r2:.3f}")
if aggregate_frac_unreliable > 0.10:
    print("  CONCLUSION: The impossibility theorem applies — attention explanations")
    print("  cannot be simultaneously faithful, stable, and decisive when dropout")
    print("  induces Rashomon-like multiplicity in internal representations.")
else:
    print("  CONCLUSION: Attention explanations show low instability under dropout.")
    print("  The Rashomon effect from dropout alone may be weak.")
print("=" * 70)
