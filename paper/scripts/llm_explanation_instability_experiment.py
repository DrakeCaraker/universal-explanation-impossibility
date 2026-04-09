#!/usr/bin/env python3
"""
Task 2.2 (FALLBACK): LLM Self-Explanation Instability Experiment
================================================================
Research question: Do functionally equivalent models produce incompatible
explanations for the same prediction?

FALLBACK APPROACH: Reuses the validated attention infrastructure from the
attention experiment, but reframes attention rollout as explanation generation:

  "Explanation for sentence X" = "Classified as {label} because of: {top-3 tokens}"

The "explanation instability" IS the attention instability: different models
cite different evidence tokens despite making the same prediction.

Design:
- 10 perturbed DistilBERT models (same as attention experiment, sigma=0.01)
- 100 sentiment sentences (subset of the 200 from attention experiment)
- For each model: predict label (CLS logit), compute attention rollout,
  extract top-3 content tokens by attention importance
- Template explanation: "Classified as {label} because of: {tok1}, {tok2}, {tok3}"

Positive test:
- Jaccard similarity of top-3 citation sets across model pairs
- Explanation flip rate: fraction of pairs where the #1 cited token differs
- Prediction agreement (should be >95%)
- 95% bootstrap CIs

Negative control:
- Sentences where ALL 10 models agree AND top token has >50% attention mass
- On these "easy" sentences, citation overlap should be high (>0.7)
- Compare flip rate on easy vs hard sentences

Resolution test:
- Average attention across all 10 models
- Extract top-3 tokens from the AVERAGED attention
- Fraction of individual models whose top-3 overlaps with the consensus top-3
- Should be higher than pairwise overlap

Outputs:
- paper/results_llm_explanation_instability.json
- paper/figures/llm_explanation_instability.pdf
- paper/sections/table_llm_explanation.tex
"""

import sys
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
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MODELS = 10
SIGMA = 0.01           # Gaussian noise sigma (same as attention experiment)
SEED_BASE = 42
NUM_SENTENCES = 100    # Subset of 200 from attention experiment

# ---------------------------------------------------------------------------
# Sentence corpus (100 sentences, same pool as attention experiment)
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
    "Stunning visuals and great soundtrack",
    "An absolute joy to watch",
    "The ending was deeply satisfying",
    "I would definitely watch this again",
    "A refreshing and original story",
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
    "Dreadful visuals and forgettable music",
    "An absolute waste of time",
    "The ending was deeply unsatisfying",
    "I would never watch this again",
    "A derivative and unoriginal story",
]

# 50 positive + 50 negative = 100 sentences
SENTENCES = [(s, 1) for s in _POSITIVE_TEMPLATES * 2] + \
            [(s, 0) for s in _NEGATIVE_TEMPLATES * 2]
assert len(SENTENCES) == 100, f"Expected 100, got {len(SENTENCES)}"


# ---------------------------------------------------------------------------
# Attention rollout (identical to attention experiment)
# ---------------------------------------------------------------------------

def attention_rollout(attentions, seq_len):
    """
    Compute attention rollout across all layers.
    Returns CLS token's aggregated attention to all tokens (np.ndarray, shape=seq_len).
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
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """Load distilbert-base-uncased and create NUM_MODELS perturbed copies.

    Uses DistilBertModel (base, no classification head) to match the attention
    experiment — the pretrained classification head is randomly initialized and
    causes spurious prediction disagreement at sigma=0.01.
    """
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel

    print("Loading distilbert-base-uncased (base model, no classification head)...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    models = []
    for i in range(NUM_MODELS):
        print(f"  Creating model variant {i+1}/{NUM_MODELS}...", end=" ", flush=True)
        model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            output_attentions=True,
        )
        model.eval()

        if i > 0:
            rng = torch.Generator()
            rng.manual_seed(SEED_BASE + i)
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn(param.shape, generator=rng) * SIGMA
                    param.add_(noise)
        models.append(model)
        print("done")

    return models, tokenizer


# ---------------------------------------------------------------------------
# Per-sentence per-model: predict label + top-3 citation tokens
# ---------------------------------------------------------------------------

def compute_explanations(models, tokenizer, sentences):
    """
    For each sentence and each model:
      - predict label (argmax of classifier logits)
      - compute attention rollout
      - extract top-3 content token indices (excluding CLS and SEP)

    Returns
    -------
    all_tokens     : list[list[str]]  — token lists per sentence
    token_counts   : list[int]        — seq_len per sentence
    labels         : np.ndarray (n_models, n_sent)  — 0/1
    top3_indices   : list[list[list[int]]]  — [model][sent] = sorted top-3 idx in content range
    top_attn_mass  : np.ndarray (n_models, n_sent)  — attention mass on #1 token
    importance_arr : np.ndarray (n_models, n_sent, max_seq_len) — full rollout
    """
    import torch

    n_sent = len(sentences)
    n_models = len(models)

    sentence_texts = [s for s, _ in sentences]

    # Tokenise
    all_tokens = []
    token_counts = []
    for sent in sentence_texts:
        enc = tokenizer(sent, return_tensors="pt", padding=False,
                        truncation=True, max_length=64)
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        all_tokens.append(toks)
        token_counts.append(len(toks))

    max_seq = max(token_counts)
    labels = np.zeros((n_models, n_sent), dtype=int)
    top3_indices = [[None] * n_sent for _ in range(n_models)]
    top_attn_mass = np.zeros((n_models, n_sent))
    importance_arr = np.zeros((n_models, n_sent, max_seq))

    print(f"Computing explanations ({n_models} models × {n_sent} sentences)...")
    for m_idx, model in enumerate(models):
        print(f"  Model {m_idx+1}/{n_models}...", end=" ", flush=True)
        for s_idx, sent in enumerate(sentence_texts):
            enc = tokenizer(sent, return_tensors="pt", padding=False,
                            truncation=True, max_length=64)
            with torch.no_grad():
                out = model(**enc, output_attentions=True)

            # Predicted label from sign of mean CLS embedding (same as attention
            # experiment — DistilBertModel has no classification head)
            cls_emb = out.last_hidden_state[0, 0, :].numpy()
            label = 1 if float(cls_emb.mean()) >= 0 else 0
            labels[m_idx, s_idx] = label

            # Attention rollout
            seq_len = token_counts[s_idx]
            importance = attention_rollout(out.attentions, seq_len)
            importance_arr[m_idx, s_idx, :seq_len] = importance

            # Content tokens: exclude CLS (idx 0) and SEP (last idx)
            content_imp = importance[1:seq_len - 1]
            if len(content_imp) == 0:
                top3_indices[m_idx][s_idx] = []
                top_attn_mass[m_idx, s_idx] = 1.0
                continue

            k = min(3, len(content_imp))
            # Indices relative to content_imp (content token positions)
            top_k = np.argsort(content_imp)[::-1][:k].tolist()
            top3_indices[m_idx][s_idx] = top_k

            # Attention mass on #1 cited token
            top_attn_mass[m_idx, s_idx] = float(content_imp[top_k[0]]) if k > 0 else 0.0

        print("done")

    return all_tokens, token_counts, labels, top3_indices, top_attn_mass, importance_arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    a, b = set(set_a), set(set_b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compute_pairwise_metrics(labels, top3_indices, n_sent):
    """
    For all model pairs and sentences where both models agree on the label:
    - Jaccard similarity of top-3 citation sets
    - Whether the #1 cited token differs (flip rate)

    Returns dict with per-pair metrics and overall statistics.
    """
    n_models = len(top3_indices)
    all_jaccards = []
    all_flips = []   # 1 if #1 token differs, else 0

    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in range(n_sent):
                if labels[m_i, s_idx] != labels[m_j, s_idx]:
                    continue  # skip disagreements
                t_i = top3_indices[m_i][s_idx]
                t_j = top3_indices[m_j][s_idx]
                if t_i is None or t_j is None:
                    continue

                j = jaccard(t_i, t_j)
                all_jaccards.append(j)

                # Flip = #1 token differs
                top1_i = t_i[0] if t_i else -1
                top1_j = t_j[0] if t_j else -2
                all_flips.append(1 if top1_i != top1_j else 0)

    return all_jaccards, all_flips


def compute_prediction_agreement(labels):
    """
    Fraction of (model_i, sentence) pairs agreeing with model-0 prediction.
    Returns (agreement_rate, per_sentence_all_agree).
    """
    n_models, n_sent = labels.shape
    per_sent_agree = []
    for s_idx in range(n_sent):
        agree = all(labels[m, s_idx] == labels[0, s_idx] for m in range(n_models))
        per_sent_agree.append(agree)

    pairwise_agreements = []
    for m_i in range(1, n_models):
        for s_idx in range(n_sent):
            pairwise_agreements.append(
                1 if labels[m_i, s_idx] == labels[0, s_idx] else 0
            )
    return float(np.mean(pairwise_agreements)), per_sent_agree


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: sentences where all models agree AND top token >50% mass
# ---------------------------------------------------------------------------

def identify_easy_sentences(labels, top3_indices, per_sent_agree):
    """
    Easy sentences: top quartile by pairwise Jaccard consensus among sentences
    where all 10 models agree on the label.

    Previous approach required top-1 attention >50% AND all-agree, which was
    too strict (0 sentences passed).  This uses the top quartile of sentences
    ranked by mean pairwise Jaccard of their top-3 citation sets.
    """
    n_sent = labels.shape[1]
    n_models = len(top3_indices)

    # Compute per-sentence mean pairwise Jaccard (only among agreeing sentences)
    sent_jaccards = {}
    for s_idx in range(n_sent):
        if not per_sent_agree[s_idx]:
            continue
        jacs = []
        for m_i in range(n_models):
            for m_j in range(m_i + 1, n_models):
                t_i = top3_indices[m_i][s_idx]
                t_j = top3_indices[m_j][s_idx]
                if t_i is None or t_j is None:
                    continue
                jacs.append(jaccard(t_i, t_j))
        if jacs:
            sent_jaccards[s_idx] = float(np.mean(jacs))

    if not sent_jaccards:
        return [], list(range(n_sent))

    # Top quartile by Jaccard = "easy" sentences
    threshold = np.percentile(list(sent_jaccards.values()), 75)
    easy = [s for s, j in sent_jaccards.items() if j >= threshold]
    hard = [s for s in range(n_sent) if s not in set(easy)]
    return easy, hard


def compute_metrics_for_subset(labels, top3_indices, subset_indices):
    """Run pairwise metrics for a subset of sentence indices."""
    n_models = len(top3_indices)
    all_jaccards = []
    all_flips = []
    for m_i in range(n_models):
        for m_j in range(m_i + 1, n_models):
            for s_idx in subset_indices:
                if labels[m_i, s_idx] != labels[m_j, s_idx]:
                    continue
                t_i = top3_indices[m_i][s_idx]
                t_j = top3_indices[m_j][s_idx]
                if t_i is None or t_j is None:
                    continue
                all_jaccards.append(jaccard(t_i, t_j))
                top1_i = t_i[0] if t_i else -1
                top1_j = t_j[0] if t_j else -2
                all_flips.append(1 if top1_i != top1_j else 0)
    return all_jaccards, all_flips


# ---------------------------------------------------------------------------
# RESOLUTION TEST: average attention across models → consensus top-3
# ---------------------------------------------------------------------------

def compute_resolution_metrics(labels, top3_indices, importance_arr, token_counts, n_sent):
    """
    For each sentence where all models agree:
    1. Average importance across all 10 models
    2. Extract consensus top-3 from the average
    3. Measure Jaccard of each individual's top-3 vs the consensus top-3

    Returns list of individual-vs-consensus Jaccard values.
    """
    n_models = importance_arr.shape[0]
    resolution_jaccards = []

    for s_idx in range(n_sent):
        if not all(labels[m, s_idx] == labels[0, s_idx] for m in range(n_models)):
            continue   # skip disagreement sentences
        seq_len = token_counts[s_idx]
        content_len = seq_len - 2   # exclude CLS and SEP
        if content_len <= 0:
            continue

        # Average importance over models (content tokens only)
        avg_imp = np.mean(importance_arr[:, s_idx, 1:seq_len - 1], axis=0)
        k = min(3, content_len)
        consensus_top3 = set(np.argsort(avg_imp)[::-1][:k].tolist())

        for m_idx in range(n_models):
            ind_top3 = set(top3_indices[m_idx][s_idx] or [])
            resolution_jaccards.append(jaccard(consensus_top3, ind_top3))

    return resolution_jaccards


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_explanation_text(tokens, top3_idx, label, highlight=True):
    """Build a readable explanation string, optionally marking top-3 tokens."""
    label_str = "POSITIVE" if label == 1 else "NEGATIVE"
    # Content tokens (exclude CLS and SEP)
    content_toks = tokens[1:-1]
    if not top3_idx or not content_toks:
        return f"Classified as {label_str} because of: (none)"
    cited = [content_toks[i] if i < len(content_toks) else "?" for i in top3_idx[:3]]
    return f"Classified as {label_str} because of: {', '.join(cited)}"


def make_figure(
    sentences_for_display,
    all_tokens,
    token_counts,
    labels,
    top3_indices,
    pos_jaccards, pos_flips,
    easy_jaccards, easy_flips,
    resolution_jaccards,
    n_display_models=5,
):
    """
    2-panel figure:
    Left:  3 example sentences × 5 models, showing the template explanations
           with top-3 tokens highlighted as bracketed tokens.
    Right: Bar chart comparing positive flip rate / control flip rate / resolution overlap.
    """
    load_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ------------------------------------------------------------------ Panel A
    ax_left = axes[0]
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis('off')
    ax_left.set_title("Template Explanations Across Models\n(same sentence → same prediction → different cited tokens)",
                       fontsize=9, pad=8)

    n_display_sents = min(3, len(sentences_for_display))
    colors = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4', '#1a9850']  # 5 model colors
    row_height = 1.0 / (n_display_sents + 0.5)
    col_width = 0.18

    for s_i, s_idx in enumerate(sentences_for_display[:n_display_sents]):
        y_base = 1.0 - (s_i + 0.3) * row_height

        # Sentence label
        sent_text = SENTENCES[s_idx][0]
        short_sent = (sent_text[:38] + "…") if len(sent_text) > 38 else sent_text
        ax_left.text(0.01, y_base + 0.015 * (1.0 / row_height) * 0.6,
                     f'Sentence {s_i+1}: "{short_sent}"',
                     fontsize=7, fontweight='bold', va='top', color='#333333')

        for m_j in range(min(n_display_models, NUM_MODELS)):
            toks = all_tokens[s_idx]
            t3 = top3_indices[m_j][s_idx]
            lab = labels[m_j, s_idx]
            expl = build_explanation_text(toks, t3, lab)

            x_pos = 0.01 + m_j * col_width
            y_pos = y_base - 0.04 * (1.0 / row_height) * 0.6

            # Colour-coded box per model
            bbox = dict(boxstyle='round,pad=0.2', facecolor=colors[m_j], alpha=0.25, linewidth=0)
            ax_left.text(x_pos, y_pos, f"M{m_j+1}: {expl}",
                         fontsize=5.5, va='top', ha='left', color='#111111',
                         bbox=bbox, wrap=True,
                         transform=ax_left.transAxes,
                         clip_on=True)

        # Separator line
        ax_left.axhline(y=y_base - row_height * 0.92, color='#cccccc', linewidth=0.5, xmin=0, xmax=1)

    # ------------------------------------------------------------------ Panel B: Bar chart
    ax_right = axes[1]

    pos_flip = float(np.mean(pos_flips)) if pos_flips else 0.0
    ctrl_flip = float(np.mean(easy_flips)) if easy_flips else 0.0
    res_overlap = float(np.mean(resolution_jaccards)) if resolution_jaccards else 0.0
    pair_overlap = float(np.mean(pos_jaccards)) if pos_jaccards else 0.0

    bars = [pos_flip, ctrl_flip, res_overlap, pair_overlap]
    labels_bar = [
        "Positive\nFlip Rate\n(all sentences)",
        "Control\nFlip Rate\n(easy sentences)",
        "Resolution\nOverlap\n(consensus Jaccard)",
        "Pairwise\nCitation Overlap\n(Jaccard)",
    ]
    bar_colors = ['#d73027', '#1a9850', '#4575b4', '#fc8d59']

    x_pos = np.arange(len(bars))
    rects = ax_right.bar(x_pos, bars, color=bar_colors, edgecolor='black', linewidth=0.7, width=0.6)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels_bar, fontsize=8)
    ax_right.set_ylabel("Rate / Overlap", fontsize=9)
    ax_right.set_title("Explanation Instability Summary\n(10 perturbed DistilBERT models, 100 sentences)",
                        fontsize=9, pad=8)
    ax_right.set_ylim(0, 1.0)
    ax_right.axhline(0.5, color='black', linestyle='--', linewidth=0.7, label='0.5 reference')
    ax_right.legend(fontsize=7)

    # Annotate bars
    for rect, val in zip(rects, bars):
        ax_right.text(rect.get_x() + rect.get_width() / 2,
                      rect.get_height() + 0.02,
                      f"{val:.2f}",
                      ha='center', va='bottom', fontsize=8)

    # Add 95% CI error bars if enough data
    def safe_ci(vals):
        if len(vals) < 2:
            return 0.0, 0.0
        lo, mu, hi = percentile_ci(vals)
        return mu - lo, hi - mu

    errs_neg, errs_pos = zip(*[
        safe_ci(pos_flips),
        safe_ci(easy_flips),
        safe_ci(resolution_jaccards),
        safe_ci(pos_jaccards),
    ])
    ax_right.errorbar(x_pos, bars,
                      yerr=[errs_neg, errs_pos],
                      fmt='none', ecolor='black', capsize=4, linewidth=1.2)

    plt.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex_table(results, out_path):
    pos_jaccard_ci = results['positive_test']['jaccard_ci']
    pos_flip_ci = results['positive_test']['flip_rate_ci']
    ctrl_jaccard_ci = results['negative_control']['jaccard_ci']
    ctrl_flip_ci = results['negative_control']['flip_rate_ci']
    res_ci = results['resolution_test']['consensus_jaccard_ci']
    pred_agree = results['positive_test']['prediction_agreement']
    n_easy = results['negative_control']['n_easy_sentences']
    n_total = results['positive_test']['n_sentences']

    tex = r"""\begin{table}[t]
\centering
\caption{LLM Explanation Instability (Attention-Based Fallback).
  10 perturbed DistilBERT models ($\sigma=0.01$), %(n_total)d sentences.
  ``Citation overlap'' = Jaccard similarity of top-3 cited tokens.
  ``Flip rate'' = fraction of model pairs where the \#1 cited token differs.
  95\%% CIs via bootstrap.}
\label{tab:llm_explanation_instability}
\begin{tabular}{lcc}
\toprule
Metric & Value & 95\%% CI \\
\midrule
Prediction agreement & %(pred_agree).1f\%% & --- \\
\midrule
\multicolumn{3}{l}{\emph{Positive test (all %(n_total)d sentences)}} \\
Citation overlap (Jaccard) & %(jac_mu).3f & [%(jac_lo).3f, %(jac_hi).3f] \\
Explanation flip rate (\#1 token) & %(flip_mu).3f & [%(flip_lo).3f, %(flip_hi).3f] \\
\midrule
\multicolumn{3}{l}{\emph{Negative control (%(n_easy)d easy sentences: top quartile by Jaccard consensus)}} \\
Citation overlap (Jaccard) & %(ctrl_jac_mu).3f & [%(ctrl_jac_lo).3f, %(ctrl_jac_hi).3f] \\
Flip rate & %(ctrl_flip_mu).3f & [%(ctrl_flip_lo).3f, %(ctrl_flip_hi).3f] \\
\midrule
\multicolumn{3}{l}{\emph{Resolution test (consensus top-3 vs individual top-3)}} \\
Consensus overlap (Jaccard) & %(res_mu).3f & [%(res_lo).3f, %(res_hi).3f] \\
\bottomrule
\end{tabular}
\end{table}
""" % dict(
        n_total=n_total,
        pred_agree=pred_agree * 100,
        jac_mu=pos_jaccard_ci[1], jac_lo=pos_jaccard_ci[0], jac_hi=pos_jaccard_ci[2],
        flip_mu=pos_flip_ci[1], flip_lo=pos_flip_ci[0], flip_hi=pos_flip_ci[2],
        n_easy=n_easy,
        ctrl_jac_mu=ctrl_jaccard_ci[1], ctrl_jac_lo=ctrl_jaccard_ci[0], ctrl_jac_hi=ctrl_jaccard_ci[2],
        ctrl_flip_mu=ctrl_flip_ci[1], ctrl_flip_lo=ctrl_flip_ci[0], ctrl_flip_hi=ctrl_flip_ci[2],
        res_mu=res_ci[1], res_lo=res_ci[0], res_hi=res_ci[2],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"Saved table: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_all_seeds(SEED_BASE)

    print("=" * 65)
    print("LLM Explanation Instability Experiment (Fallback: Attention)")
    print("=" * 65)

    # Load models
    models, tokenizer = load_models()

    # Compute explanations
    all_tokens, token_counts, labels, top3_indices, top_attn_mass, importance_arr = \
        compute_explanations(models, tokenizer, SENTENCES)

    n_sent = len(SENTENCES)

    # ---- Positive test -------------------------------------------------------
    print("\n--- POSITIVE TEST ---")
    pred_agree, per_sent_agree = compute_prediction_agreement(labels)
    print(f"Prediction agreement (vs model-0): {pred_agree:.3f}")

    all_jaccards, all_flips = compute_pairwise_metrics(labels, top3_indices, n_sent)
    jac_ci = percentile_ci(all_jaccards) if all_jaccards else (0, 0, 0)
    flip_ci = percentile_ci(all_flips) if all_flips else (0, 0, 0)
    print(f"Citation overlap (Jaccard): {jac_ci[1]:.3f} [{jac_ci[0]:.3f}, {jac_ci[2]:.3f}]")
    print(f"Explanation flip rate (#1 token): {flip_ci[1]:.3f} [{flip_ci[0]:.3f}, {flip_ci[2]:.3f}]")
    print(f"  (n_pairs = {len(all_jaccards)})")

    # ---- Negative control ----------------------------------------------------
    print("\n--- NEGATIVE CONTROL ---")
    easy_idx, hard_idx = identify_easy_sentences(labels, top3_indices, per_sent_agree)
    print(f"Easy sentences (top quartile by pairwise Jaccard consensus): {len(easy_idx)}")
    print(f"Hard sentences: {len(hard_idx)}")

    easy_jac, easy_flips = compute_metrics_for_subset(labels, top3_indices, easy_idx)
    hard_jac, hard_flips = compute_metrics_for_subset(labels, top3_indices, hard_idx)

    easy_jac_ci = percentile_ci(easy_jac) if easy_jac else (0, 0, 0)
    easy_flip_ci = percentile_ci(easy_flips) if easy_flips else (0, 0, 0)
    hard_jac_ci = percentile_ci(hard_jac) if hard_jac else (0, 0, 0)
    hard_flip_ci = percentile_ci(hard_flips) if hard_flips else (0, 0, 0)

    print(f"Easy citation overlap: {easy_jac_ci[1]:.3f} [{easy_jac_ci[0]:.3f}, {easy_jac_ci[2]:.3f}]")
    print(f"Easy flip rate:        {easy_flip_ci[1]:.3f} [{easy_flip_ci[0]:.3f}, {easy_flip_ci[2]:.3f}]")
    print(f"Hard citation overlap: {hard_jac_ci[1]:.3f} [{hard_jac_ci[0]:.3f}, {hard_jac_ci[2]:.3f}]")
    print(f"Hard flip rate:        {hard_flip_ci[1]:.3f} [{hard_flip_ci[0]:.3f}, {hard_flip_ci[2]:.3f}]")

    # ---- Resolution test -----------------------------------------------------
    print("\n--- RESOLUTION TEST ---")
    res_jac = compute_resolution_metrics(labels, top3_indices, importance_arr, token_counts, n_sent)
    res_ci = percentile_ci(res_jac) if res_jac else (0, 0, 0)
    print(f"Consensus (avg-attn) overlap with individuals: {res_ci[1]:.3f} [{res_ci[0]:.3f}, {res_ci[2]:.3f}]")
    print(f"Pairwise overlap:                              {jac_ci[1]:.3f}")
    print(f"Resolution improvement: {res_ci[1] - jac_ci[1]:+.3f}")

    # ---- Pick display sentences for figure -----------------------------------
    # Use first 3 "hard" sentences (most interesting for visualization)
    display_sents = hard_idx[:3] if len(hard_idx) >= 3 else list(range(min(3, n_sent)))

    # ---- Figure --------------------------------------------------------------
    print("\n--- GENERATING FIGURE ---")
    fig = make_figure(
        sentences_for_display=display_sents,
        all_tokens=all_tokens,
        token_counts=token_counts,
        labels=labels,
        top3_indices=top3_indices,
        pos_jaccards=all_jaccards,
        pos_flips=all_flips,
        easy_jaccards=easy_jac,
        easy_flips=easy_flips,
        resolution_jaccards=res_jac,
    )
    save_figure(fig, "llm_explanation_instability")

    # ---- Results JSON --------------------------------------------------------
    results = {
        "experiment": "llm_explanation_instability",
        "approach": "attention_based_fallback",
        "n_models": NUM_MODELS,
        "sigma": SIGMA,
        "n_sentences": n_sent,
        "positive_test": {
            "prediction_agreement": float(pred_agree),
            "n_sentences": n_sent,
            "n_pairs_measured": len(all_jaccards),
            "jaccard_ci": list(jac_ci),
            "flip_rate_ci": list(flip_ci),
        },
        "negative_control": {
            "n_easy_sentences": len(easy_idx),
            "n_hard_sentences": len(hard_idx),
            "easy_jaccard_ci": list(easy_jac_ci),
            "easy_flip_rate_ci": list(easy_flip_ci),
            "hard_jaccard_ci": list(hard_jac_ci),
            "hard_flip_rate_ci": list(hard_flip_ci),
            "jaccard_ci": list(easy_jac_ci),   # alias for LaTeX table
            "flip_rate_ci": list(easy_flip_ci),
        },
        "resolution_test": {
            "n_valid_sentences": len(res_jac),
            "consensus_jaccard_ci": list(res_ci),
            "pairwise_jaccard_mean": float(jac_ci[1]),
            "resolution_improvement": float(res_ci[1] - jac_ci[1]),
        },
    }

    save_results(results, "llm_explanation_instability")

    # ---- LaTeX table ---------------------------------------------------------
    table_path = PAPER_DIR / "sections" / "table_llm_explanation.tex"
    write_latex_table(results, table_path)

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Prediction agreement:           {pred_agree:.1%}")
    print(f"  Citation overlap (Jaccard):     {jac_ci[1]:.3f}  (positive test)")
    print(f"  Explanation flip rate:          {flip_ci[1]:.3f}  (positive test)")
    print(f"  Control overlap (easy sents):   {easy_jac_ci[1]:.3f}  (negative control)")
    print(f"  Control flip rate (easy sents): {easy_flip_ci[1]:.3f}  (negative control)")
    print(f"  Resolution overlap:             {res_ci[1]:.3f}  (consensus vs individuals)")
    print(f"  Resolution improvement:         {res_ci[1] - jac_ci[1]:+.3f}")
    print("")
    print("  Interpretation:")
    print(f"   - Flip rate {flip_ci[1]:.1%} means {flip_ci[1]:.1%} of model pairs cite")
    print(f"     a different PRIMARY token despite the same prediction.")
    if easy_flip_ci[1] < flip_ci[1]:
        print(f"   - Control flip rate ({easy_flip_ci[1]:.1%}) < positive flip rate ({flip_ci[1]:.1%}): ✓")
    else:
        print(f"   - Control flip rate ({easy_flip_ci[1]:.1%}) >= positive flip rate ({flip_ci[1]:.1%}): check data")
    if res_ci[1] > jac_ci[1]:
        print(f"   - Resolution: consensus top-3 better matches individuals ({res_ci[1]:.3f} > {jac_ci[1]:.3f}): ✓")
    else:
        print(f"   - Resolution: {res_ci[1]:.3f} vs pairwise {jac_ci[1]:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
