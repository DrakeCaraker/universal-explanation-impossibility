#!/usr/bin/env python3
"""
Bag-of-Words Attribution Instability — Attribution Impossibility for TF-IDF text features.

Extends the LLM attention instability experiment (llm_attention_instability.py)
to a lightweight, CPU-only setting using TF-IDF + XGBoost + TreeSHAP.

TF-IDF features from natural language are inherently correlated: synonyms,
co-occurring words, and morphological variants create collinear feature groups
that trigger the Attribution Impossibility.

Setup:
  - sklearn 20newsgroups (alt.atheism vs soc.religion.christian)
  - TF-IDF vectorization (max_features=200)
  - 30 XGBoost classifiers (different seeds, subsample=0.8)
  - TreeSHAP on 100 test documents

Metrics:
  - % of documents with unstable top-1 feature
  - % of documents with unstable top-3 features
  - Spearman correlation between models' feature rankings (averaged)
  - Top 10 most unstable word pairs

Output:
  - Console summary
  - Saved to paper/results_nlp_token.json
"""

import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Install with: pip install shap")
    sys.exit(1)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ── Configuration ─────────────────────────────────────────────────────────────
CATEGORIES = ["alt.atheism", "soc.religion.christian"]
MAX_FEATURES = 200            # TF-IDF vocabulary size
N_MODELS = 30                 # number of XGBoost models
N_TEST_DOCS = 100             # documents for SHAP evaluation
MASTER_SEED = 42
CORR_THRESHOLD = 0.3          # TF-IDF features are less correlated than numeric;
                              # use a lower threshold to capture co-occurrence


# ── Data loading and preprocessing ────────────────────────────────────────────

def load_data():
    """Load 20newsgroups, TF-IDF vectorize, train/test split."""
    print("Loading 20newsgroups dataset...")
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=MASTER_SEED,
    )

    print(f"  Categories: {CATEGORIES}")
    print(f"  Total documents: {len(data.data)}")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        min_df=5,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(data.data).toarray()
    y = data.target
    vocab = vectorizer.get_feature_names_out()

    print(f"  TF-IDF features: {X.shape[1]}")
    print(f"  Sample words: {list(vocab[:10])}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=MASTER_SEED,
    )

    # Limit test set for SHAP
    if X_test.shape[0] > N_TEST_DOCS:
        rng = np.random.RandomState(MASTER_SEED)
        idx = rng.choice(X_test.shape[0], size=N_TEST_DOCS, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    print(f"  Training documents: {X_train.shape[0]}")
    print(f"  Test documents (for SHAP): {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, vocab


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NLP Token-Level Attribution Instability")
    print("  Attribution Impossibility for TF-IDF text features")
    print(f"  {N_MODELS} XGBoost models, {MAX_FEATURES} TF-IDF features")
    print(f"  Dataset: 20newsgroups ({' vs '.join(CATEGORIES)})")
    print("=" * 70)

    X_train, X_test, y_train, y_test, vocab = load_data()
    n_features = X_train.shape[1]
    n_docs = X_test.shape[0]

    # Step 1: Train models and compute SHAP
    print(f"\nTraining {N_MODELS} XGBoost models and computing TreeSHAP...")

    # shap_all[model_idx] = array of shape (n_docs, n_features)
    shap_all = []

    for s in range(N_MODELS):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            eval_metric="logloss",
            n_jobs=1,
            random_state=MASTER_SEED + s,
        )
        model.fit(X_train, y_train, verbose=False)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]  # class 1 SHAP values
        shap_all.append(sv)

        if (s + 1) % 10 == 0:
            print(f"  Trained and explained model {s + 1}/{N_MODELS}")

    print(f"  All {N_MODELS} models done.")

    # Step 2: Per-document top-feature stability
    print("\nAnalysing per-document top-feature stability...")

    unstable_top1 = 0
    unstable_top3 = 0

    for d in range(n_docs):
        top1_features = set()
        top3_features_list = []

        for m in range(N_MODELS):
            abs_shap = np.abs(shap_all[m][d])
            ranking = np.argsort(-abs_shap)
            top1_features.add(ranking[0])
            top3_features_list.append(set(ranking[:3].tolist()))

        # Top-1 unstable if more than one distinct top-1 feature across models
        if len(top1_features) > 1:
            unstable_top1 += 1

        # Top-3 unstable if the top-3 set is not identical across all models
        reference_top3 = top3_features_list[0]
        if any(s != reference_top3 for s in top3_features_list[1:]):
            unstable_top3 += 1

    pct_unstable_top1 = 100.0 * unstable_top1 / n_docs
    pct_unstable_top3 = 100.0 * unstable_top3 / n_docs

    # Step 3: Spearman rank correlations between models (on mean |SHAP|)
    print("Computing pairwise Spearman rank correlations...")

    # Global importance per model: mean |SHAP| across test documents
    global_importances = []
    for m in range(N_MODELS):
        imp = np.abs(shap_all[m]).mean(axis=0)
        global_importances.append(imp)

    spearman_corrs = []
    for a in range(N_MODELS):
        for b in range(a + 1, N_MODELS):
            corr, _ = spearmanr(global_importances[a], global_importances[b])
            spearman_corrs.append(corr)

    mean_spearman = float(np.mean(spearman_corrs))
    min_spearman = float(np.min(spearman_corrs))
    std_spearman = float(np.std(spearman_corrs))

    # Step 4: Most unstable word pairs
    print("Computing flip rates for correlated word pairs...")

    # Correlation matrix on training TF-IDF
    corr_matrix = np.corrcoef(X_train.T)

    pair_flips = []
    for i, j in combinations(range(n_features), 2):
        rho = abs(corr_matrix[i, j])
        if rho < CORR_THRESHOLD:
            continue

        # Flip rate based on global mean |SHAP| importance
        orderings = []
        for m in range(N_MODELS):
            orderings.append(global_importances[m][i] > global_importances[m][j])

        flips = 0
        total = 0
        for a in range(N_MODELS):
            for b in range(a + 1, N_MODELS):
                if orderings[a] != orderings[b]:
                    flips += 1
                total += 1

        fr = flips / total if total > 0 else 0.0
        if fr > 0:
            pair_flips.append({
                "word_i": str(vocab[i]),
                "word_j": str(vocab[j]),
                "rho": round(float(rho), 4),
                "flip_rate": round(float(fr), 4),
            })

    pair_flips_sorted = sorted(pair_flips, key=lambda p: -p["flip_rate"])

    # Step 5: Which words are most unstable (appear in most flipping pairs)?
    word_instability_count = {}
    for p in pair_flips:
        for w in [p["word_i"], p["word_j"]]:
            word_instability_count[w] = word_instability_count.get(w, 0) + 1

    most_unstable_words = sorted(word_instability_count.items(),
                                 key=lambda x: -x[1])[:20]

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'metric':<50} {'value':>10}")
    print("-" * 62)
    print(f"{'Documents with unstable top-1 feature':<50} {pct_unstable_top1:>9.1f}%")
    print(f"{'Documents with unstable top-3 features':<50} {pct_unstable_top3:>9.1f}%")
    print(f"{'Mean Spearman correlation (global importance)':<50} {mean_spearman:>10.4f}")
    print(f"{'Min Spearman correlation':<50} {min_spearman:>10.4f}")
    print(f"{'Std Spearman correlation':<50} {std_spearman:>10.4f}")
    print(f"{'Correlated word pairs (|r| >= {CORR_THRESHOLD})':<50} "
          f"{len(pair_flips) + sum(1 for i, j in combinations(range(n_features), 2) if abs(corr_matrix[i, j]) >= CORR_THRESHOLD) - len(pair_flips):>10}")
    print(f"{'Unstable word pairs (flip_rate > 0)':<50} {len(pair_flips):>10}")

    print(f"\nTop 10 most unstable word pairs:")
    for p in pair_flips_sorted[:10]:
        print(f"  '{p['word_i']}' vs '{p['word_j']}': "
              f"rho={p['rho']:.3f}, flip_rate={p['flip_rate']:.4f}")

    print(f"\nTop 10 most unstable words (appear in most flipping pairs):")
    for word, count in most_unstable_words[:10]:
        print(f"  '{word}': involved in {count} unstable pairs")

    # Verdict
    print("\n" + "-" * 70)
    verdict = (
        f"Bag-of-words attribution instability affects {pct_unstable_top1:.0f}% "
        f"of documents (top-1) and {pct_unstable_top3:.0f}% (top-3). "
        f"TF-IDF co-occurrence creates correlated features subject to the "
        f"impossibility theorem. Note: this demonstrates instability in "
        f"bag-of-words models (TF-IDF + XGBoost), not transformer-based NLP. "
        f"For transformer results, see llm_attention_instability.py."
    )
    print(verdict)
    print("-" * 70)

    # ── Save results ──────────────────────────────────────────────────────────
    metrics_table = [
        {"metric": "pct_unstable_top1", "value": round(pct_unstable_top1, 1)},
        {"metric": "pct_unstable_top3", "value": round(pct_unstable_top3, 1)},
        {"metric": "mean_spearman", "value": round(mean_spearman, 4)},
        {"metric": "min_spearman", "value": round(min_spearman, 4)},
        {"metric": "std_spearman", "value": round(std_spearman, 4)},
        {"metric": "n_unstable_word_pairs", "value": len(pair_flips)},
    ]

    results = {
        "experiment": "nlp_token_instability",
        "dataset": "20newsgroups",
        "categories": CATEGORIES,
        "max_features": MAX_FEATURES,
        "n_models": N_MODELS,
        "n_test_docs": n_docs,
        "correlation_threshold": CORR_THRESHOLD,
        "metrics": metrics_table,
        "top_unstable_word_pairs": pair_flips_sorted[:10],
        "most_unstable_words": [
            {"word": w, "n_unstable_pairs": c} for w, c in most_unstable_words[:10]
        ],
        "rank_correlations": {
            "mean_spearman": round(mean_spearman, 4),
            "min_spearman": round(min_spearman, 4),
            "std_spearman": round(std_spearman, 4),
        },
        "verdict": verdict,
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(out_dir, "results_nlp_token.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
