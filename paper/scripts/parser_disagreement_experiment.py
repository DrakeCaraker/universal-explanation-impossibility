"""
Task 2.2: Linguistics — Parser Disagreement
============================================
Parse ambiguous vs. unambiguous sentences with multiple NLP parsers.
Measure inter-parser agreement (Unlabeled Attachment Score) to show
that ambiguous sentences produce higher parser disagreement.

Parsers used:
  - spaCy en_core_web_sm  (small statistical, Penn Treebank trained)
  - spaCy en_core_web_md  (medium statistical, Penn Treebank trained)
  - Stanza English model  (if available, also Penn Treebank trained)

Metric: For each sentence, compute pairwise fraction of tokens where
head assignment agrees (UAS-equivalent, unlabeled). Mean over all
parser pairs = inter-parser agreement score.

Output:
  paper/results_parser_disagreement.json
  paper/figures/parser_disagreement.pdf
  paper/sections/table_parser.tex
"""

import sys
import os
import json
import itertools
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
    PAPER_DIR,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Sentence Sets ───────────────────────────────────────────────────────────

ambiguous_sentences = [
    "I saw the man with the telescope",
    "The chicken is ready to eat",
    "Visiting relatives can be boring",
    "They are hunting dogs",
    "Flying planes can be dangerous",
    "The professor said on Monday he would give an exam",
    "I shot an elephant in my pajamas",
    "The old men and women sat on the bench",
    "She saw the boy with binoculars",
    "He fed her cat food",
    "The girl hit the boy with a book",
    "We saw her duck",
    "John saw the man on the hill with a telescope",
    "Time flies like an arrow",
    "The horse raced past the barn fell",
    "I know more beautiful women than Miss America",
    "Every student thinks he is smart",
    "The shooting of the hunters was terrible",
    "Look at the dog with one eye",
    "Kids make nutritious snacks",
    "Free whales",
    "He painted the wall with cracks",
    "The man returned to the store with his receipt",
    "She hit the man with an umbrella",
    "They discussed the problem with the teacher",
]

unambiguous_sentences = [
    "The cat sat on the mat",
    "John ate breakfast",
    "She runs every morning",
    "The book is on the table",
    "He drives to work",
    "The sun is bright today",
    "I like coffee",
    "Dogs are loyal animals",
    "The door is open",
    "She writes poems",
    "We went home",
    "The baby cried loudly",
    "He bought a new car",
    "Rain fell all day",
    "She smiled at him",
    "The tree is tall",
    "I read the newspaper",
    "Birds sing in the morning",
    "He closed the window",
    "She walked to school",
    "The water is cold",
    "I finished my homework",
    "The flowers bloomed",
    "He answered the phone",
    "She cooked dinner",
]

# ── Parser Loading ───────────────────────────────────────────────────────────

def load_parsers():
    """Load all available parsers. Returns dict name → parse_fn."""
    parsers = {}

    # spaCy en_core_web_sm
    try:
        import spacy
        nlp_sm = spacy.load("en_core_web_sm")
        def parse_sm(sentence):
            doc = nlp_sm(sentence)
            # Return list of (token_text, head_index) for each token
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_sm"] = parse_sm
        print("Loaded: spaCy en_core_web_sm")
    except Exception as e:
        print(f"Could not load spaCy en_core_web_sm: {e}")

    # spaCy en_core_web_md
    try:
        import spacy
        nlp_md = spacy.load("en_core_web_md")
        def parse_md(sentence):
            doc = nlp_md(sentence)
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_md"] = parse_md
        print("Loaded: spaCy en_core_web_md")
    except Exception as e:
        print(f"Could not load spaCy en_core_web_md: {e}")

    # spaCy en_core_web_trf (transformer, if available)
    try:
        import spacy
        nlp_trf = spacy.load("en_core_web_trf")
        def parse_trf(sentence):
            doc = nlp_trf(sentence)
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_trf"] = parse_trf
        print("Loaded: spaCy en_core_web_trf")
    except Exception as e:
        print(f"spaCy en_core_web_trf not available: {e}")

    # Stanza English
    try:
        import stanza
        nlp_stanza = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=False,
            verbose=False,
        )
        def parse_stanza(sentence):
            doc = nlp_stanza(sentence)
            result = []
            offset = 0
            for sent in doc.sentences:
                for word in sent.words:
                    # stanza uses 1-indexed heads; 0 means root
                    # convert to 0-indexed absolute position
                    if word.head == 0:
                        head_idx = offset + word.id - 1  # root points to itself
                    else:
                        head_idx = offset + word.head - 1
                    result.append((word.text, head_idx))
                offset += len(sent.words)
            return result
        parsers["stanza"] = parse_stanza
        print("Loaded: Stanza English")
    except Exception as e:
        print(f"Stanza not available: {e}")

    return parsers


# ── Agreement Computation ────────────────────────────────────────────────────

def compute_pairwise_agreement(parse_a, parse_b):
    """
    Compute fraction of tokens where head assignment agrees between two parses.
    Both parses are lists of (token_text, head_index).
    We align by position (assumes same tokenization length; if lengths differ,
    we align on the shorter and penalize the difference).
    """
    n_a = len(parse_a)
    n_b = len(parse_b)
    n_min = min(n_a, n_b)
    n_max = max(n_a, n_b)

    if n_max == 0:
        return 1.0

    agree = sum(
        1 for i in range(n_min)
        if parse_a[i][1] == parse_b[i][1]
    )
    # tokens that could not be aligned count as disagreement
    return agree / n_max


def sentence_agreement(sentence, parsers):
    """
    Parse sentence with all parsers, compute mean pairwise agreement.
    Returns float in [0, 1].
    """
    parser_names = list(parsers.keys())
    if len(parser_names) < 2:
        raise ValueError("Need at least 2 parsers")

    parses = {}
    for name, fn in parsers.items():
        try:
            parses[name] = fn(sentence)
        except Exception as e:
            print(f"  Parser {name} failed on '{sentence[:40]}': {e}")
            parses[name] = []

    pair_agreements = []
    for a, b in itertools.combinations(parser_names, 2):
        agr = compute_pairwise_agreement(parses[a], parses[b])
        pair_agreements.append(agr)

    return float(np.mean(pair_agreements)), parses


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(amb_agreements, unamb_agreements, parser_names, out_path):
    load_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Box plots
    ax = axes[0]
    data_plot = [amb_agreements, unamb_agreements]
    bp = ax.boxplot(
        data_plot,
        labels=['Ambiguous', 'Unambiguous'],
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color='black', linewidth=2),
    )
    colors = ['#d62728', '#1f77b4']  # red, blue
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered points
    np.random.seed(42)
    for i, (data, color) in enumerate(zip(data_plot, colors), start=1):
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax.scatter(
            np.full(len(data), i) + jitter,
            data,
            color=color,
            alpha=0.5,
            s=20,
            zorder=5,
        )

    ax.set_ylabel('Inter-parser agreement (UAS)')
    ax.set_title('A. Agreement by sentence type')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Panel B: Per-sentence sorted agreements
    ax2 = axes[1]

    # Sort ambiguous ascending, unambiguous descending for visual contrast
    amb_sorted = sorted(amb_agreements)
    unamb_sorted = sorted(unamb_agreements)

    x_amb = np.arange(len(amb_sorted))
    x_unamb = np.arange(len(amb_sorted), len(amb_sorted) + len(unamb_sorted))

    ax2.scatter(x_amb, amb_sorted, color='#d62728', alpha=0.8, s=30, label='Ambiguous', zorder=5)
    ax2.scatter(x_unamb, unamb_sorted, color='#1f77b4', alpha=0.8, s=30, label='Unambiguous', zorder=5)

    ax2.axvline(len(amb_sorted) - 0.5, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Sentence index (sorted within group)')
    ax2.set_ylabel('Inter-parser agreement (UAS)')
    ax2.set_title('B. Per-sentence agreement')
    ax2.legend(frameon=False)
    ax2.set_ylim(0, 1.05)

    # Parser info in subtitle
    fig.suptitle(
        f'Parser disagreement: {" vs. ".join(parser_names)}\n'
        f'Ambiguous ({len(amb_agreements)} sentences) vs. Unambiguous ({len(unamb_agreements)} sentences)',
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ── LaTeX Table ──────────────────────────────────────────────────────────────

def write_latex_table(results, out_path):
    amb_ci = results['ambiguous_ci']
    unamb_ci = results['unambiguous_ci']
    pval = results['wilcoxon_pvalue']
    n_parsers = results['n_parsers']
    parser_names = results['parser_names']

    pval_str = f"{pval:.4f}" if pval >= 0.0001 else f"{pval:.2e}"

    tex = r"""\begin{table}[h]
\centering
\caption{Inter-parser dependency agreement on ambiguous vs. unambiguous sentences.
Mean pairwise Unlabeled Attachment Score (UAS) between """ + str(n_parsers) + r""" parsers
(""" + ", ".join(f"\\texttt{{{p}}}" for p in parser_names) + r""").
95\% bootstrap confidence intervals shown. Higher = more agreement.}
\label{tab:parser_disagreement}
\begin{tabular}{lccc}
\toprule
Sentence type & $N$ & Mean UAS (95\% CI) & Wilcoxon $p$ \\
\midrule
Ambiguous  & """ + str(results['n_ambiguous']) + r""" & """ + \
        f"{amb_ci[1]:.3f} ({amb_ci[0]:.3f}, {amb_ci[2]:.3f})" + r""" & \multirow{2}{*}{""" + pval_str + r"""} \\
Unambiguous & """ + str(results['n_unambiguous']) + r""" & """ + \
        f"{unamb_ci[1]:.3f} ({unamb_ci[0]:.3f}, {unamb_ci[2]:.3f})" + r""" & \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Saved LaTeX table: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(42)

    print("=" * 60)
    print("Task 2.2: Parser Disagreement Experiment")
    print("=" * 60)

    # Load parsers
    parsers = load_parsers()
    if len(parsers) < 2:
        print("ERROR: Need at least 2 parsers. Aborting.")
        sys.exit(1)

    parser_names = list(parsers.keys())
    print(f"\nUsing {len(parsers)} parsers: {parser_names}")
    print(f"Sentences: {len(ambiguous_sentences)} ambiguous, {len(unambiguous_sentences)} unambiguous")
    print()

    # Parse all sentences and compute agreement
    amb_agreements = []
    amb_details = []
    print("Parsing ambiguous sentences...")
    for i, sent in enumerate(ambiguous_sentences):
        agr, parses = sentence_agreement(sent, parsers)
        amb_agreements.append(agr)
        amb_details.append({
            'sentence': sent,
            'agreement': agr,
            'parses': {
                k: [(t, h) for t, h in v] for k, v in parses.items()
            },
        })
        print(f"  [{i+1:2d}/{len(ambiguous_sentences)}] {agr:.3f}  '{sent[:50]}'")

    unamb_agreements = []
    unamb_details = []
    print("\nParsing unambiguous sentences...")
    for i, sent in enumerate(unambiguous_sentences):
        agr, parses = sentence_agreement(sent, parsers)
        unamb_agreements.append(agr)
        unamb_details.append({
            'sentence': sent,
            'agreement': agr,
            'parses': {
                k: [(t, h) for t, h in v] for k, v in parses.items()
            },
        })
        print(f"  [{i+1:2d}/{len(unambiguous_sentences)}] {agr:.3f}  '{sent[:50]}'")

    amb_arr = np.array(amb_agreements)
    unamb_arr = np.array(unamb_agreements)

    # Statistics
    print("\n--- Results ---")
    amb_ci = percentile_ci(amb_arr, n_boot=5000)
    unamb_ci = percentile_ci(unamb_arr, n_boot=5000)
    print(f"Ambiguous   mean UAS: {amb_ci[1]:.3f}  95% CI [{amb_ci[0]:.3f}, {amb_ci[2]:.3f}]")
    print(f"Unambiguous mean UAS: {unamb_ci[1]:.3f}  95% CI [{unamb_ci[0]:.3f}, {unamb_ci[2]:.3f}]")

    stat, pval = stats.ranksums(amb_arr, unamb_arr)
    print(f"Wilcoxon rank-sum: statistic={stat:.4f}, p={pval:.4e}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(amb_arr, ddof=1)**2 + np.std(unamb_arr, ddof=1)**2) / 2
    )
    cohens_d = (unamb_ci[1] - amb_ci[1]) / pooled_std if pooled_std > 0 else float('nan')
    print(f"Cohen's d (unamb - amb): {cohens_d:.3f}")

    # Results dict
    results = {
        'parser_names': parser_names,
        'n_parsers': len(parsers),
        'n_ambiguous': len(ambiguous_sentences),
        'n_unambiguous': len(unambiguous_sentences),
        'ambiguous_mean_uas': float(amb_ci[1]),
        'ambiguous_ci': list(amb_ci),
        'unambiguous_mean_uas': float(unamb_ci[1]),
        'unambiguous_ci': list(unamb_ci),
        'wilcoxon_statistic': float(stat),
        'wilcoxon_pvalue': float(pval),
        'cohens_d': float(cohens_d),
        'ambiguous_agreements': [float(x) for x in amb_agreements],
        'unambiguous_agreements': [float(x) for x in unamb_agreements],
        'ambiguous_details': [
            {'sentence': d['sentence'], 'agreement': d['agreement']}
            for d in amb_details
        ],
        'unambiguous_details': [
            {'sentence': d['sentence'], 'agreement': d['agreement']}
            for d in unamb_details
        ],
    }

    # Save results
    save_results(results, 'parser_disagreement')

    # Figure
    fig_path = PAPER_DIR / 'figures' / 'parser_disagreement.pdf'
    make_figure(amb_agreements, unamb_agreements, parser_names, fig_path)

    # LaTeX table
    sections_dir = PAPER_DIR / 'sections'
    sections_dir.mkdir(exist_ok=True)
    tex_path = sections_dir / 'table_parser.tex'
    write_latex_table(results, tex_path)

    print("\n--- Summary ---")
    print(f"Ambiguous UAS:   {amb_ci[1]:.3f} (lower = more disagreement between parsers)")
    print(f"Unambiguous UAS: {unamb_ci[1]:.3f} (higher = parsers agree more)")
    print(f"Difference:      {unamb_ci[1] - amb_ci[1]:.3f}")
    print(f"p-value:         {pval:.4e}")
    interpretation = "CONFIRMED" if pval < 0.05 else "not significant"
    print(f"Hypothesis:      Ambiguous < Unambiguous agreement — {interpretation}")
    print()
    print("Most disagreed (ambiguous):")
    for d in sorted(amb_details, key=lambda x: x['agreement'])[:5]:
        print(f"  {d['agreement']:.3f}  '{d['sentence']}'")
    print("Most agreed (ambiguous):")
    for d in sorted(amb_details, key=lambda x: x['agreement'], reverse=True)[:3]:
        print(f"  {d['agreement']:.3f}  '{d['sentence']}'")

    return results


if __name__ == '__main__':
    main()
