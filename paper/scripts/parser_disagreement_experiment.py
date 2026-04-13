"""
Task 2.2: Linguistics — Parser Disagreement (Rigorous Design, v2)
=================================================================
Parse ambiguous vs. unambiguous sentences with 3 independent NLP parsers.
Measure inter-parser agreement (Unlabeled Attachment Score) to show
that ambiguous sentences produce higher parser disagreement.

Fixes from peer review:
  1. All 50 ambiguous sentences are EXCLUSIVELY structural PP-attachment
     ("V NP PP" where PP can attach to NP or VP).
  2. Pre-tokenization: all sentences tokenized once with spaCy's tokenizer,
     then fed as pre-tokenized input to all parsers (eliminates tokenization confound).
  3. Parser independence: use at most one spaCy CNN model + Stanza + a second
     spaCy model of different size (en_core_web_sm vs en_core_web_lg).
     No three-spaCy-CNN-variant design.

Parsers used:
  - spaCy en_core_web_sm   (small CNN, Penn Treebank)
  - spaCy en_core_web_lg   (large CNN, Penn Treebank)
  - Stanza English model   (BiLSTM, OntoNotes/UD)

Design:
  - 50 ambiguous + 50 unambiguous sentences
  - C(3,2) = 3 pairwise parser comparisons per sentence
  - Mean UAS per sentence = average across 3 pairs
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
# ALL 50 ambiguous sentences are structural PP-attachment ambiguities:
# "V NP PP" where PP can attach to NP (modifier) or VP (instrument/location).

ambiguous_sentences = [
    # Classic PP-attachment: PP can modify NP or VP
    "I saw the man with the telescope",
    "She ate the cake on the table",
    "He chased the dog with a stick",
    "They discussed the plan in the office",
    "The professor lectured the students about ethics",
    "The teacher scolded the student with the ruler",
    "She photographed the building with the broken windows",
    "The diplomat met the spy with the secret documents",
    "He watched the parade from the balcony",
    "She saw the boy with binoculars",
    "The girl hit the boy with a book",
    "The detective followed the suspect with the disguise",
    "The nurse treated the patient with the infection",
    "She noticed the stain on the carpet with a flashlight",
    "The chef served the guests with the silver platters",
    "He examined the painting with the magnifying glass",
    "The officer stopped the driver with the broken taillight",
    "She cleaned the table with the scratches",
    "He painted the wall with cracks",
    "She hit the man with an umbrella",
    "They discussed the problem with the teacher",
    "The spy saw the cop with the binoculars",
    "The soldier guarded the entrance with the rifle",
    "The reporter interviewed the politician with the scandal",
    "He loaded the truck with the crane",
    "She described the house on the hill",
    "The manager reprimanded the employee with the complaint",
    "They observed the bird with the telescope",
    "The mechanic fixed the car in the garage",
    "The hiker spotted the cabin from the ridge",
    "The librarian shelved the books on the table",
    "The scientist analyzed the sample under the microscope",
    "He chased the cat across the garden",
    "The architect designed the tower with the rotating platform",
    "She wrapped the gift with the ribbon",
    "The boy poked the bear with the stick",
    "The woman carried the bag with the broken handle",
    "He shot the target with the laser sight",
    "The guard watched the prisoner through the window",
    "She read the letter from her grandmother",
    "The hunter tracked the deer with the dogs",
    "He repaired the fence with the hammer",
    "The artist sketched the model with charcoal",
    "She served the customer with the coupon",
    "The child fed the duck with the bread",
    "He pulled the lever with the red handle",
    "The sailor spotted the island from the crow's nest",
    "She decorated the room with the flowers",
    "The waiter brought the dish with the special sauce",
    "He measured the fabric with the ruler",
]

# 50 unambiguous sentences: simple SVO with no PP-attachment ambiguity.
# Either no PP at all, or PP clearly attaches to only one site.
unambiguous_sentences = [
    "The cat slept soundly",
    "She ran quickly",
    "He read the book",
    "John ate breakfast",
    "The baby cried loudly",
    "She writes poems",
    "We went home",
    "He bought a new car",
    "She smiled brightly",
    "I finished my homework",
    "The flowers bloomed",
    "He answered the phone",
    "She cooked dinner",
    "The sky turned dark",
    "She bought groceries yesterday",
    "He plays guitar well",
    "The students passed the exam",
    "We ate lunch together",
    "She speaks three languages",
    "He fixed the broken lamp",
    "She painted a landscape",
    "The package arrived today",
    "He taught himself piano",
    "She completed the marathon",
    "He wrote a letter home",
    "She solved the puzzle quickly",
    "He climbed the tall mountain",
    "She organized the bookshelf",
    "The dog barked loudly",
    "He closed the window",
    "She drank the coffee",
    "The bird sang beautifully",
    "He washed the dishes",
    "She folded the laundry",
    "The children laughed",
    "He mowed the lawn",
    "She typed the report",
    "The engine started",
    "He locked the door",
    "She swept the floor",
    "The phone rang twice",
    "He ironed his shirt",
    "She planted the seeds",
    "The alarm sounded",
    "He sharpened the pencil",
    "She opened the envelope",
    "The snow melted quickly",
    "He memorized the speech",
    "She stitched the wound",
    "The clock struck midnight",
]

assert len(ambiguous_sentences)   == 50, f"Expected 50, got {len(ambiguous_sentences)}"
assert len(unambiguous_sentences) == 50, f"Expected 50, got {len(unambiguous_sentences)}"


# ── Pre-tokenization ────────────────────────────────────────────────────────

def pretokenize_all(sentences):
    """
    Pre-tokenize all sentences using spaCy's tokenizer (only).
    Returns list of token-lists, one per sentence.
    """
    import spacy
    nlp = spacy.blank("en")  # blank pipeline = tokenizer only
    return [[tok.text for tok in nlp(sent)] for sent in sentences]


# ── Parser Loading ───────────────────────────────────────────────────────────

def load_parsers():
    """
    Load 3 independent parsers. Returns dict name -> parse_fn.
    Each parse_fn takes a list of tokens (pre-tokenized) and returns
    [(token_text, head_index), ...].
    """
    parsers = {}
    import spacy

    # Determine which spaCy models to use: prefer sm + lg for diversity
    available = spacy.util.get_installed_models()
    has_trf = "en_core_web_trf" in available
    has_lg  = "en_core_web_lg" in available
    has_sm  = "en_core_web_sm" in available

    if has_trf and has_lg:
        # transformer + CNN = genuinely different architectures
        spacy_models = [("spacy_lg", "en_core_web_lg"), ("spacy_trf", "en_core_web_trf")]
    elif has_sm and has_lg:
        # two different CNN sizes = some diversity
        spacy_models = [("spacy_sm", "en_core_web_sm"), ("spacy_lg", "en_core_web_lg")]
    elif has_lg:
        spacy_models = [("spacy_lg", "en_core_web_lg")]
    elif has_sm:
        spacy_models = [("spacy_sm", "en_core_web_sm")]
    else:
        spacy_models = []

    for name, model_name in spacy_models:
        try:
            nlp = spacy.load(model_name)
            def make_parser(nlp_ref):
                def parse_pretokenized(tokens, _nlp=nlp_ref):
                    # Build Doc from pre-tokenized words, then run parser
                    from spacy.tokens import Doc
                    doc = Doc(_nlp.vocab, words=tokens)
                    # Run tagger + parser (needed for dependency parse)
                    for pipe_name in _nlp.pipe_names:
                        if pipe_name in ("tagger", "parser", "attribute_ruler",
                                         "lemmatizer", "tok2vec", "transformer",
                                         "morphologizer"):
                            _nlp.get_pipe(pipe_name)(doc)
                    return [(tok.text, tok.head.i) for tok in doc]
                return parse_pretokenized
            parsers[name] = make_parser(nlp)
            print(f"Loaded: {name} ({model_name})")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")

    # Stanza English (pre-tokenized mode)
    try:
        import stanza
        nlp_stanza = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=True,   # accept pre-tokenized input
            verbose=False,
        )
        def parse_stanza(tokens, _nlp=nlp_stanza):
            # Stanza expects list-of-lists for pretokenized (one list per sentence)
            doc = _nlp([tokens])
            result = []
            offset = 0
            for sent in doc.sentences:
                for word in sent.words:
                    if word.head == 0:
                        head_idx = offset + word.id - 1
                    else:
                        head_idx = offset + word.head - 1
                    result.append((word.text, head_idx))
                offset += len(sent.words)
            return result
        parsers["stanza"] = parse_stanza
        print("Loaded: Stanza English (pretokenized mode)")
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


def sentence_agreement_detailed(tokens, parsers):
    """
    Parse pre-tokenized sentence with all parsers.
    Returns (mean_agreement, per_pair_dict, parses_dict).
    """
    parser_names = list(parsers.keys())
    parses = {}
    for name, fn in parsers.items():
        try:
            parses[name] = fn(tokens)
        except Exception as e:
            sent_preview = " ".join(tokens[:8])
            print(f"  Parser {name} failed on '{sent_preview}': {e}")
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

    # Per-pair overlay lines
    pair_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5', '#f7b6d2', '#c49c94']
    for pi, pname in enumerate(pair_names):
        amb_pair   = [d['pair_agreements'].get(pname, np.nan) for d in amb_data]
        unamb_pair = [d['pair_agreements'].get(pname, np.nan) for d in unamb_data]
        short = pname.replace("spacy_", "").replace("|", " vs ")
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
    ax.set_title('A. Agreement by sentence type\n(box = all pairs mean)')
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.set_xlim(0.5, 2.5)

    # ------------------------------------------------------------------
    # Panel B: Sorted per-sentence agreement, colored by type
    # ------------------------------------------------------------------
    ax2 = axes[1]

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

    ax2.set_xlabel('Sentence rank (sorted by agreement, lowest to highest)')
    ax2.set_ylabel('Inter-parser agreement (UAS)')
    ax2.set_title('B. Per-sentence agreement\n(ambiguity score = 1 - UAS)')
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
        r"All sentences pre-tokenized with spaCy tokenizer to eliminate tokenization confound. "
        r"Ambiguous sentences are exclusively structural PP-attachment ambiguities. "
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
    print("Task 2.2: Parser Disagreement Experiment — v2 (peer review fixes)")
    print("=" * 65)
    print(f"  Ambiguous sentences:   {len(ambiguous_sentences)}")
    print(f"  Unambiguous sentences: {len(unambiguous_sentences)}")

    # ── Pre-tokenize all sentences ──────────────────────────────────
    print("\nPre-tokenizing all sentences with spaCy tokenizer...")
    all_sentences = ambiguous_sentences + unambiguous_sentences
    all_tokens = pretokenize_all(all_sentences)
    amb_tokens   = all_tokens[:50]
    unamb_tokens = all_tokens[50:]
    print(f"  Pre-tokenized {len(all_tokens)} sentences")

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
    print("Parsing ambiguous sentences (pre-tokenized)...")
    for i, (sent, tokens) in enumerate(zip(ambiguous_sentences, amb_tokens)):
        mean_agr, pair_agr, parses = sentence_agreement_detailed(tokens, parsers)
        amb_data.append({
            'sentence':       sent,
            'mean_agreement': mean_agr,
            'ambiguity_score': 1.0 - mean_agr,
            'pair_agreements': pair_agr,
        })
        print(f"  [{i+1:2d}/50] UAS={mean_agr:.3f}  '{sent[:55]}'")

    # ── Parse unambiguous ────────────────────────────────────────────
    unamb_data = []
    print("\nParsing unambiguous sentences (pre-tokenized)...")
    for i, (sent, tokens) in enumerate(zip(unambiguous_sentences, unamb_tokens)):
        mean_agr, pair_agr, parses = sentence_agreement_detailed(tokens, parsers)
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
            'pre_tokenized': True,
            'ambiguity_type': 'PP-attachment only',
            'fixes_applied': [
                'exclusive PP-attachment ambiguity sentences',
                'pre-tokenization with spaCy blank tokenizer',
                'independent parsers (no 3x spaCy CNN)',
            ],
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
    print(f"Pre-tokenized:   Yes (spaCy blank tokenizer)")
    print(f"Ambiguity type:  PP-attachment only (all 50)")
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
