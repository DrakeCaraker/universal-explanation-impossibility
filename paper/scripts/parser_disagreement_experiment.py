"""
Task 2.2: Linguistics — Parser Disagreement (Rigorous Design)
=============================================================
Parse ambiguous vs. unambiguous sentences with 4 NLP parsers.
Measure inter-parser agreement (Unlabeled Attachment Score) to show
that ambiguous sentences produce higher parser disagreement.

Parsers used:
  - spaCy en_core_web_sm   (small statistical, Penn Treebank)
  - spaCy en_core_web_md   (medium statistical, Penn Treebank)
  - spaCy en_core_web_lg   (large statistical, Penn Treebank)
  - Stanza English model   (BiLSTM, OntoNotes/UD)

Design:
  - 50 ambiguous + 50 unambiguous sentences
  - C(4,2) = 6 pairwise parser comparisons per sentence
  - Mean UAS per sentence = average across 6 pairs
  - Per-sentence ambiguity score = 1 - mean_UAS

Statistics:
  - Wilcoxon rank-sum test
  - 95% bootstrap CIs
  - Cohen's d

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
# Original 25 ambiguous (PP-attachment, gerunds, coordination, garden paths)
ambiguous_sentences_orig = [
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

# Additional 25 ambiguous (PP-attachment, coordination, garden paths, headlines)
ambiguous_sentences_new = [
    "The spy saw the cop with the binoculars",
    "The teacher said on Friday she would cancel class",
    "Mary and John's children are happy",
    "The woman fed her cat tuna",
    "I saw the man that the dog bit in the park",
    "The pilot flew planes in storms",
    "Local high school dropouts cut in half",
    "Red tape holds up new bridge",
    "Miners refuse to work after death",
    "Eye drops off shelf",
    "British left waffles on Falklands",
    "Teacher strikes idle kids",
    "Police begin campaign to run down jaywalkers",
    "Hospitals are sued by seven foot doctors",
    "New study of obesity looks for larger test group",
    "Juvenile court to try shooting defendant",
    "Stolen painting found by tree",
    "Drunk gets nine months in violin case",
    "Include your children when baking cookies",
    "Man eating piranha mistakenly sold as pet fish",
    "Queen Mary having bottom scraped",
    "Two Soviet ships collide, one dies",
    "Safety experts say school bus passengers should be belted",
    "Enraged cow injures farmer with axe",
    "Ban on naked firefighting in Cleveland",
]

# Original 25 unambiguous
unambiguous_sentences_orig = [
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

# Additional 25 unambiguous
unambiguous_sentences_new = [
    "The sky is blue today",
    "She bought groceries yesterday",
    "He plays guitar well",
    "The students passed the exam",
    "We ate lunch together",
    "The train arrived on time",
    "She speaks three languages",
    "The meeting starts at nine",
    "He fixed the broken lamp",
    "The river flows south",
    "They planted trees in spring",
    "She painted a landscape",
    "The package arrived today",
    "He taught himself piano",
    "The movie was entertaining",
    "She completed the marathon",
    "The store closes at midnight",
    "He wrote a letter home",
    "The garden needs water",
    "She solved the puzzle quickly",
    "The bread smells wonderful",
    "He climbed the tall mountain",
    "The concert begins soon",
    "She organized the bookshelf",
    "The sunset was beautiful",
]

ambiguous_sentences   = ambiguous_sentences_orig   + ambiguous_sentences_new
unambiguous_sentences = unambiguous_sentences_orig + unambiguous_sentences_new

assert len(ambiguous_sentences)   == 50, f"Expected 50, got {len(ambiguous_sentences)}"
assert len(unambiguous_sentences) == 50, f"Expected 50, got {len(unambiguous_sentences)}"


# ── Parser Loading ───────────────────────────────────────────────────────────

def load_parsers():
    """Load all 4 parsers. Returns dict name → parse_fn."""
    parsers = {}

    import spacy

    # spaCy en_core_web_sm
    try:
        nlp_sm = spacy.load("en_core_web_sm")
        def parse_sm(sentence, _nlp=nlp_sm):
            doc = _nlp(sentence)
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_sm"] = parse_sm
        print("Loaded: spaCy en_core_web_sm")
    except Exception as e:
        print(f"Could not load spaCy en_core_web_sm: {e}")

    # spaCy en_core_web_md
    try:
        nlp_md = spacy.load("en_core_web_md")
        def parse_md(sentence, _nlp=nlp_md):
            doc = _nlp(sentence)
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_md"] = parse_md
        print("Loaded: spaCy en_core_web_md")
    except Exception as e:
        print(f"Could not load spaCy en_core_web_md: {e}")

    # spaCy en_core_web_lg
    try:
        nlp_lg = spacy.load("en_core_web_lg")
        def parse_lg(sentence, _nlp=nlp_lg):
            doc = _nlp(sentence)
            return [(tok.text, tok.head.i) for tok in doc]
        parsers["spacy_lg"] = parse_lg
        print("Loaded: spaCy en_core_web_lg")
    except Exception as e:
        print(f"Could not load spaCy en_core_web_lg: {e}")

    # Stanza English
    try:
        import stanza
        nlp_stanza = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=False,
            verbose=False,
        )
        def parse_stanza(sentence, _nlp=nlp_stanza):
            doc = _nlp(sentence)
            result = []
            offset = 0
            for sent in doc.sentences:
                for word in sent.words:
                    # stanza: 1-indexed heads; 0 = root → points to itself
                    if word.head == 0:
                        head_idx = offset + word.id - 1
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
    Fraction of tokens where head assignment agrees.
    Align by position; unaligned tokens count as disagreement.
    """
    n_a = len(parse_a)
    n_b = len(parse_b)
    n_min = min(n_a, n_b)
    n_max = max(n_a, n_b)
    if n_max == 0:
        return 1.0
    agree = sum(1 for i in range(n_min) if parse_a[i][1] == parse_b[i][1])
    return agree / n_max


def sentence_agreement_detailed(sentence, parsers):
    """
    Parse sentence with all parsers.
    Returns (mean_agreement, per_pair_dict, parses_dict).
    """
    parser_names = list(parsers.keys())
    parses = {}
    for name, fn in parsers.items():
        try:
            parses[name] = fn(sentence)
        except Exception as e:
            print(f"  Parser {name} failed on '{sentence[:40]}': {e}")
            parses[name] = []

    pair_agreements = {}
    for a, b in itertools.combinations(parser_names, 2):
        agr = compute_pairwise_agreement(parses[a], parses[b])
        pair_agreements[f"{a}|{b}"] = float(agr)

    mean_agr = float(np.mean(list(pair_agreements.values())))
    return mean_agr, pair_agreements, parses


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(amb_data, unamb_data, parser_names, pair_names, out_path):
    """
    Two-panel figure.
    Left:  box plots of mean UAS per group, with per-pair overlays as thin lines.
    Right: sorted per-sentence agreement colored by ambiguous/unambiguous.
    """
    load_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    amb_mean   = [d['mean_agreement'] for d in amb_data]
    unamb_mean = [d['mean_agreement'] for d in unamb_data]

    # ------------------------------------------------------------------
    # Panel A: Box plots + per-pair overlays
    # ------------------------------------------------------------------
    ax = axes[0]

    data_plot = [amb_mean, unamb_mean]
    bp = ax.boxplot(
        data_plot,
        labels=['Ambiguous\n(n=50)', 'Unambiguous\n(n=50)'],
        patch_artist=True,
        widths=0.38,
        medianprops=dict(color='black', linewidth=2.0),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    group_colors = ['#d62728', '#1f77b4']
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Jittered individual points
    np.random.seed(42)
    for i, (data, color) in enumerate(zip(data_plot, group_colors), start=1):
        jitter = np.random.uniform(-0.10, 0.10, len(data))
        ax.scatter(
            np.full(len(data), i) + jitter, data,
            color=color, alpha=0.45, s=18, zorder=5,
        )

    # Per-pair overlay lines (6 pairs, thin, distinct colours)
    pair_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5', '#f7b6d2', '#c49c94']
    for pi, pname in enumerate(pair_names):
        amb_pair   = [d['pair_agreements'].get(pname, np.nan) for d in amb_data]
        unamb_pair = [d['pair_agreements'].get(pname, np.nan) for d in unamb_data]
        short = pname.replace("spacy_", "").replace("|", "–")
        ax.plot(
            [1, 2],
            [np.nanmean(amb_pair), np.nanmean(unamb_pair)],
            color=pair_colors[pi % len(pair_colors)],
            linewidth=1.2, alpha=0.85,
            marker='D', markersize=4,
            label=short, zorder=4,
        )

    ax.legend(fontsize=6.5, frameon=False, loc='lower right',
              title='Parser pair', title_fontsize=6.5)
    ax.set_ylabel('Inter-parser agreement (UAS)')
    ax.set_title('A. Agreement by sentence type\n(box = all 6 pairs mean)')
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.set_xlim(0.5, 2.5)

    # ------------------------------------------------------------------
    # Panel B: Sorted per-sentence agreement, colored by type
    # ------------------------------------------------------------------
    ax2 = axes[1]

    # Combine all 100 sentences, sort by agreement ascending
    all_records = (
        [(d['sentence'], d['mean_agreement'], 'ambiguous')   for d in amb_data] +
        [(d['sentence'], d['mean_agreement'], 'unambiguous') for d in unamb_data]
    )
    all_records_sorted = sorted(all_records, key=lambda x: x[1])

    xs      = np.arange(len(all_records_sorted))
    ys      = np.array([r[1] for r in all_records_sorted])
    is_amb  = np.array([r[2] == 'ambiguous' for r in all_records_sorted])

    ax2.scatter(xs[is_amb],  ys[is_amb],  color='#d62728', alpha=0.80,
                s=22, label='Ambiguous',   zorder=5)
    ax2.scatter(xs[~is_amb], ys[~is_amb], color='#1f77b4', alpha=0.80,
                s=22, label='Unambiguous', zorder=5)

    ax2.set_xlabel('Sentence rank (sorted by agreement, lowest→highest)')
    ax2.set_ylabel('Inter-parser agreement (UAS)')
    ax2.set_title('B. Per-sentence agreement\n(ambiguity score = 1 − UAS)')
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlim(-1, 100)

    n_parsers = len(parser_names)
    n_pairs   = len(pair_names)
    fig.suptitle(
        f'Parser disagreement: {n_parsers} parsers ({", ".join(parser_names)}), '
        f'{n_pairs} pairwise comparisons per sentence\n'
        f'Ambiguous (50) vs. Unambiguous (50)',
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ── LaTeX Table ──────────────────────────────────────────────────────────────

def write_latex_table(results, out_path):
    amb_ci   = results['ambiguous_ci']
    unamb_ci = results['unambiguous_ci']
    pval     = results['wilcoxon_pvalue']
    n_parsers = results['n_parsers']
    parser_names = results['parser_names']
    cohens_d = results['cohens_d']

    pval_str = f"{pval:.4f}" if pval >= 0.0001 else f"{pval:.2e}"
    sig_str  = r"$p < 0.05$" if pval < 0.05 else f"$p = {pval_str}$"

    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{Inter-parser dependency agreement on ambiguous vs.\ unambiguous sentences. "
        r"Mean pairwise Unlabeled Attachment Score (UAS) across all $\binom{" + str(n_parsers) + r"}{2} = "
        + str(len(list(itertools.combinations(range(n_parsers), 2)))) + r"$ parser pairs "
        r"(" + ", ".join(r"\texttt{" + p + r"}" for p in parser_names) + r"). "
        r"95\% bootstrap CIs shown. Higher = more agreement; "
        r"ambiguity score = $1 - \mathrm{UAS}$.}" + "\n"
        r"\label{tab:parser_disagreement}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\toprule" + "\n"
        r"Sentence type & $N$ & $N_{\text{pairs}}$ & Mean UAS (95\% CI) & Wilcoxon $p$ \\" + "\n"
        r"\midrule" + "\n"
        r"Ambiguous    & " + str(results['n_ambiguous'])   + r" & " + str(results['n_pairs']) + r" & "
        + f"{amb_ci[1]:.3f} ({amb_ci[0]:.3f}, {amb_ci[2]:.3f})"
        + r" & \multirow{2}{*}{" + pval_str + r"} \\" + "\n"
        r"Unambiguous  & " + str(results['n_unambiguous']) + r" & " + str(results['n_pairs']) + r" & "
        + f"{unamb_ci[1]:.3f} ({unamb_ci[0]:.3f}, {unamb_ci[2]:.3f})"
        + r" & \\" + "\n"
        r"\midrule" + "\n"
        r"\multicolumn{4}{l}{Cohen's $d$ (unambiguous $-$ ambiguous)} & "
        + f"{cohens_d:.3f}" + r" \\" + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Saved LaTeX table: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(42)

    print("=" * 65)
    print("Task 2.2: Parser Disagreement Experiment — Rigorous Design")
    print("=" * 65)
    print(f"  Ambiguous sentences:   {len(ambiguous_sentences)}")
    print(f"  Unambiguous sentences: {len(unambiguous_sentences)}")

    # Load parsers
    parsers = load_parsers()
    if len(parsers) < 2:
        print("ERROR: Need at least 2 parsers. Aborting.")
        sys.exit(1)

    parser_names = list(parsers.keys())
    pair_names   = [f"{a}|{b}" for a, b in itertools.combinations(parser_names, 2)]
    n_pairs      = len(pair_names)

    print(f"\nUsing {len(parsers)} parsers:  {parser_names}")
    print(f"Pairwise comparisons per sentence: C({len(parsers)},2) = {n_pairs}")
    print()

    # ── Parse ambiguous ──────────────────────────────────────────────
    amb_data = []
    print("Parsing ambiguous sentences...")
    for i, sent in enumerate(ambiguous_sentences):
        mean_agr, pair_agr, parses = sentence_agreement_detailed(sent, parsers)
        amb_data.append({
            'sentence':       sent,
            'mean_agreement': mean_agr,
            'ambiguity_score': 1.0 - mean_agr,
            'pair_agreements': pair_agr,
        })
        print(f"  [{i+1:2d}/50] UAS={mean_agr:.3f}  '{sent[:55]}'")

    # ── Parse unambiguous ────────────────────────────────────────────
    unamb_data = []
    print("\nParsing unambiguous sentences...")
    for i, sent in enumerate(unambiguous_sentences):
        mean_agr, pair_agr, parses = sentence_agreement_detailed(sent, parsers)
        unamb_data.append({
            'sentence':       sent,
            'mean_agreement': mean_agr,
            'ambiguity_score': 1.0 - mean_agr,
            'pair_agreements': pair_agr,
        })
        print(f"  [{i+1:2d}/50] UAS={mean_agr:.3f}  '{sent[:55]}'")

    # ── Statistics ───────────────────────────────────────────────────
    amb_arr   = np.array([d['mean_agreement'] for d in amb_data])
    unamb_arr = np.array([d['mean_agreement'] for d in unamb_data])

    print("\n--- Results ---")
    amb_ci   = percentile_ci(amb_arr,   n_boot=10000)
    unamb_ci = percentile_ci(unamb_arr, n_boot=10000)
    print(f"Ambiguous   mean UAS: {amb_ci[1]:.4f}  95% CI [{amb_ci[0]:.4f}, {amb_ci[2]:.4f}]")
    print(f"Unambiguous mean UAS: {unamb_ci[1]:.4f}  95% CI [{unamb_ci[0]:.4f}, {unamb_ci[2]:.4f}]")

    stat, pval = stats.ranksums(amb_arr, unamb_arr)
    print(f"Wilcoxon rank-sum: W={stat:.4f}, p={pval:.4e}")

    # Cohen's d  (unambiguous - ambiguous, positive = unamb higher)
    pooled_std = np.sqrt((np.std(amb_arr, ddof=1)**2 + np.std(unamb_arr, ddof=1)**2) / 2)
    cohens_d   = (unamb_ci[1] - amb_ci[1]) / pooled_std if pooled_std > 0 else float('nan')
    print(f"Cohen's d (unamb - amb): {cohens_d:.4f}")

    # ── Per-sentence ambiguity ranking ───────────────────────────────
    all_scored = (
        [(d['sentence'], d['ambiguity_score'], 'ambiguous')   for d in amb_data] +
        [(d['sentence'], d['ambiguity_score'], 'unambiguous') for d in unamb_data]
    )
    all_scored_sorted = sorted(all_scored, key=lambda x: -x[1])  # highest ambiguity first

    print("\n--- Top 10 most-disagreed sentences (highest ambiguity score) ---")
    for rank, (sent, score, label) in enumerate(all_scored_sorted[:10], 1):
        print(f"  {rank:2d}. [{label[:5]}] score={score:.3f}  '{sent[:60]}'")

    print("\n--- Bottom 10 most-agreed sentences (lowest ambiguity score) ---")
    for rank, (sent, score, label) in enumerate(all_scored_sorted[-10:], 1):
        print(f"  {rank:2d}. [{label[:5]}] score={score:.3f}  '{sent[:60]}'")

    # Fraction of top-25 that are ambiguous
    top25_amb_frac = sum(1 for _, _, l in all_scored_sorted[:25] if l == 'ambiguous') / 25
    bot25_unamb_frac = sum(1 for _, _, l in all_scored_sorted[-25:] if l == 'unambiguous') / 25
    print(f"\nAmong top-25 highest ambiguity score: {top25_amb_frac:.0%} are ambiguous sentences")
    print(f"Among bottom-25 lowest ambiguity score: {bot25_unamb_frac:.0%} are unambiguous sentences")

    # ── Assemble results dict ────────────────────────────────────────
    results = {
        'design': {
            'n_ambiguous':   50,
            'n_unambiguous': 50,
            'n_parsers':     len(parsers),
            'n_pairs':       n_pairs,
            'parser_names':  parser_names,
            'pair_names':    pair_names,
            'bootstrap_reps': 10000,
        },
        'parser_names':     parser_names,
        'n_parsers':        len(parsers),
        'n_pairs':          n_pairs,
        'n_ambiguous':      len(ambiguous_sentences),
        'n_unambiguous':    len(unambiguous_sentences),
        'ambiguous_mean_uas':   float(amb_ci[1]),
        'ambiguous_ci':         list(amb_ci),
        'unambiguous_mean_uas': float(unamb_ci[1]),
        'unambiguous_ci':       list(unamb_ci),
        'wilcoxon_statistic':   float(stat),
        'wilcoxon_pvalue':      float(pval),
        'cohens_d':             float(cohens_d),
        'top25_amb_frac':       float(top25_amb_frac),
        'bot25_unamb_frac':     float(bot25_unamb_frac),
        'ambiguous_agreements':   [float(x) for x in amb_arr],
        'unambiguous_agreements': [float(x) for x in unamb_arr],
        'ambiguous_details': [
            {'sentence': d['sentence'], 'mean_agreement': d['mean_agreement'],
             'ambiguity_score': d['ambiguity_score'], 'pair_agreements': d['pair_agreements']}
            for d in amb_data
        ],
        'unambiguous_details': [
            {'sentence': d['sentence'], 'mean_agreement': d['mean_agreement'],
             'ambiguity_score': d['ambiguity_score'], 'pair_agreements': d['pair_agreements']}
            for d in unamb_data
        ],
        'ranked_sentences': [
            {'rank': i+1, 'sentence': sent, 'ambiguity_score': float(score), 'label': label}
            for i, (sent, score, label) in enumerate(all_scored_sorted)
        ],
    }

    # ── Save outputs ─────────────────────────────────────────────────
    save_results(results, 'parser_disagreement')

    fig_path = PAPER_DIR / 'figures' / 'parser_disagreement.pdf'
    make_figure(amb_data, unamb_data, parser_names, pair_names, fig_path)

    sections_dir = PAPER_DIR / 'sections'
    sections_dir.mkdir(exist_ok=True)
    tex_path = sections_dir / 'table_parser.tex'
    write_latex_table(results, tex_path)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Parsers:         {len(parsers)} ({', '.join(parser_names)})")
    print(f"Pairs per sent:  C({len(parsers)},2) = {n_pairs}")
    print(f"Sentences:       50 ambiguous + 50 unambiguous = 100 total")
    print(f"Ambiguous   UAS: {amb_ci[1]:.4f}  [{amb_ci[0]:.4f}, {amb_ci[2]:.4f}]")
    print(f"Unambiguous UAS: {unamb_ci[1]:.4f}  [{unamb_ci[0]:.4f}, {unamb_ci[2]:.4f}]")
    print(f"Difference:      {unamb_ci[1] - amb_ci[1]:.4f}")
    print(f"Wilcoxon p:      {pval:.4e}")
    print(f"Cohen's d:       {cohens_d:.4f}")
    interpretation = "CONFIRMED (p < 0.05)" if pval < 0.05 else f"not significant (p = {pval:.4f})"
    print(f"Hypothesis:      Ambiguous < Unambiguous agreement — {interpretation}")

    return results


if __name__ == '__main__':
    main()
