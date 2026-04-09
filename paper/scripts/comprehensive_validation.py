"""
Comprehensive multi-dataset validation of F1 and F5 diagnostics.

For each dataset:
  - Train 50 XGBoost models with different seeds
  - Compute mean |SHAP| per feature per model
  - For all feature pairs: F1 test statistic, flip rate, F5 split-frequency z-score
  - Report summary statistics

Outputs:
  - Console summary table
  - paper/results_comprehensive.json
  - paper/figures/comprehensive_f1.pdf (top 4 datasets by unstable pairs)
"""

import numpy as np
import json
import os
import sys
import warnings
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.datasets import (
    load_breast_cancer, fetch_california_housing, load_diabetes,
    load_wine, load_iris, load_digits,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_style = os.path.join(os.path.dirname(__file__), 'publication_style.mplstyle')
if os.path.exists(_style):
    plt.style.use(_style)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'figure.figsize': (7.0, 6.0), 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 50
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
FLIP_THRESHOLD = 0.1
Z_THRESHOLD = 1.96


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_dataset(name):
    """Return (X, y, feature_names, task) where task is 'classification' or 'regression'."""
    if name == "Breast Cancer":
        d = load_breast_cancer()
        return d.data, d.target, list(d.feature_names), "classification"

    elif name == "California Housing":
        d = fetch_california_housing()
        return d.data, d.target, list(d.feature_names), "regression"

    elif name == "Diabetes":
        d = load_diabetes()
        return d.data, d.target, list(d.feature_names), "regression"

    elif name == "Wine":
        d = load_wine()
        # 3-class -> binary: class 0 vs rest
        y = (d.target == 0).astype(int)
        return d.data, y, list(d.feature_names), "classification"

    elif name == "Iris":
        d = load_iris()
        # 3-class -> binary: setosa vs rest
        y = (d.target == 0).astype(int)
        return d.data, y, list(d.feature_names), "classification"

    elif name == "Digits (0 vs 1)":
        d = load_digits()
        mask = np.isin(d.target, [0, 1])
        X = d.data[mask]
        y = d.target[mask]
        names = [f"pixel_{i}" for i in range(X.shape[1])]
        return X, y, names, "classification"

    elif name == "Heart Disease":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=53, as_frame=True, parser="auto")
        df = d.data.copy()
        y = LabelEncoder().fit_transform(d.target)
        # Encode categoricals
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(np.float64)
        X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0) if X.size > 0 else 0)
        names = list(df.columns)
        return X, y, names, "classification"

    elif name == "Adult Income":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df = d.data.copy()
        y = LabelEncoder().fit_transform(d.target)
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(np.float64)
        # Impute NaN with column medians
        col_medians = np.nanmedian(X, axis=0)
        for c in range(X.shape[1]):
            mask = np.isnan(X[:, c])
            if mask.any():
                X[mask, c] = col_medians[c]
        names = list(df.columns)
        # Subsample to keep runtime manageable
        if X.shape[0] > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(X.shape[0], 10000, replace=False)
            X, y = X[idx], y[idx]
        return X, y, names, "classification"

    elif name == "Credit-g":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = d.data.copy()
        y = LabelEncoder().fit_transform(d.target)
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(np.float64)
        X = np.nan_to_num(X, nan=0)
        names = list(df.columns)
        return X, y, names, "classification"

    elif name == "Ames Housing":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="house_prices", as_frame=True, parser="auto", version=1)
        df = d.data.copy()
        y = d.target.values.astype(np.float64)
        # Drop columns with >50% missing
        frac_missing = df.isnull().mean()
        df = df.loc[:, frac_missing < 0.5]
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(np.float64)
        col_medians = np.nanmedian(X, axis=0)
        for c in range(X.shape[1]):
            mask = np.isnan(X[:, c])
            if mask.any():
                X[mask, c] = col_medians[c]
        names = list(df.columns)
        return X, y, names, "regression"

    elif name == "Communities and Crime":
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=42730, as_frame=True, parser="auto")
        df = d.data.copy()
        y = d.target.values.astype(np.float64)
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(np.float64)
        col_medians = np.nanmedian(X, axis=0)
        for c in range(X.shape[1]):
            mask = np.isnan(X[:, c])
            if mask.any():
                X[mask, c] = col_medians[c] if not np.isnan(col_medians[c]) else 0
        names = list(df.columns)
        return X, y, names, "regression"

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(name):
    """Run F1+F5 diagnostics on a single dataset. Returns a result dict."""
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"{'='*70}")

    X, y, feature_names, task = load_dataset(name)
    P = X.shape[1]
    N = X.shape[0]
    print(f"  N={N}, P={P}, task={task}")

    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)

    all_shap = []
    all_split_counts = []

    for seed in range(N_SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        if task == "classification":
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                random_state=seed + 1000, verbosity=0,
                eval_metric='logloss',
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                random_state=seed + 1000, verbosity=0,
            )
        model.fit(Xtr, ytr)

        # SHAP values
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xte[:min(200, len(Xte))])
        # For multi-output shap, take first output
        if isinstance(sv, list):
            sv = sv[0]
        all_shap.append(np.mean(np.abs(sv), axis=0))

        # Split counts
        score = model.get_booster().get_score(importance_type='weight')
        splits = np.zeros(P)
        for f, c in score.items():
            idx = int(f.replace('f', ''))
            if idx < P:
                splits[idx] = c
        all_split_counts.append(splits)

    all_shap = np.array(all_shap)            # (50, P)
    all_split_counts = np.array(all_split_counts)  # (50, P)

    # Compute pair-level statistics
    pairs = []
    for j in range(P):
        for k in range(j + 1, P):
            phi_j = all_shap[:, j]
            phi_k = all_shap[:, k]
            diff = phi_j - phi_k
            mu_diff = np.mean(diff)
            se_diff = np.std(diff, ddof=1) / np.sqrt(N_SEEDS)

            z_f1 = abs(mu_diff) / se_diff if se_diff > 1e-12 else 999.0

            # Flip rate
            jw = np.sum(phi_j > phi_k)
            kw = np.sum(phi_k > phi_j)
            total = jw + kw
            flip = min(jw, kw) / total if total > 0 else 0.0

            # F5: split-frequency z
            nj = all_split_counts[:, j]
            nk = all_split_counts[:, k]
            n_diff = nj - nk
            se_n = np.std(n_diff, ddof=1) / np.sqrt(N_SEEDS)
            z_f5 = abs(np.mean(n_diff)) / se_n if se_n > 1e-12 else 999.0

            rho_jk = abs(corr_matrix[j, k]) if j < corr_matrix.shape[0] and k < corr_matrix.shape[1] else 0.0

            pairs.append({
                'j': int(j), 'k': int(k),
                'j_name': feature_names[j], 'k_name': feature_names[k],
                'rho': float(rho_jk),
                'flip': float(flip),
                'z_f1': float(z_f1),
                'z_f5': float(z_f5),
            })

    n_pairs = len(pairs)
    flips = np.array([p['flip'] for p in pairs])
    z_f1s = np.clip(np.array([p['z_f1'] for p in pairs]), 0, 50)
    z_f5s = np.clip(np.array([p['z_f5'] for p in pairs]), 0, 50)

    n_unstable = int(np.sum(flips > FLIP_THRESHOLD))

    # Correlation (clip to 20 for correlation calculation)
    z_f1_clip20 = np.clip(z_f1s, 0, 20)
    z_f5_clip20 = np.clip(z_f5s, 0, 20)
    corr_f1_flip = float(np.corrcoef(z_f1_clip20, flips)[0, 1]) if n_pairs > 2 else 0.0
    corr_f5_flip = float(np.corrcoef(z_f5_clip20, flips)[0, 1]) if n_pairs > 2 else 0.0

    # F5 precision at z=1.96: among pairs flagged (z_f5 < 1.96), what fraction are actually unstable?
    flagged = z_f5s < Z_THRESHOLD
    n_flagged = int(flagged.sum())
    if n_flagged > 0:
        f5_precision = float(np.mean(flips[flagged] > FLIP_THRESHOLD))
    else:
        f5_precision = float('nan')

    # Print summary for this dataset
    print(f"  Pairs: {n_pairs}, Unstable (flip>{FLIP_THRESHOLD}): {n_unstable}")
    print(f"  r(Z_F1, flip) = {corr_f1_flip:.3f}")
    print(f"  r(Z_F5, flip) = {corr_f5_flip:.3f}")
    print(f"  F5 flagged: {n_flagged}, precision: {f5_precision:.3f}" if not np.isnan(f5_precision) else f"  F5 flagged: 0")

    # Top 5 unstable pairs
    top5 = sorted(pairs, key=lambda x: -x['flip'])[:5]
    for p in top5:
        print(f"    flip={p['flip']:.3f} Z_F1={p['z_f1']:.1f} Z_F5={p['z_f5']:.1f} "
              f"|rho|={p['rho']:.2f}  {p['j_name'][:18]:18s} <-> {p['k_name'][:18]}")

    result = {
        'dataset': name,
        'N': int(N), 'P': int(P), 'task': task,
        'n_pairs': n_pairs,
        'n_unstable': n_unstable,
        'corr_f1_flip': corr_f1_flip,
        'corr_f5_flip': corr_f5_flip,
        'f5_flagged': n_flagged,
        'f5_precision': f5_precision,
        'pairs': pairs,
        'flips': flips.tolist(),
        'z_f1s': z_f1s.tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    "Breast Cancer",
    "California Housing",
    "Diabetes",
    "Wine",
    "Iris",
    "Digits (0 vs 1)",
    "Heart Disease",
    "Adult Income",
    "Credit-g",
    "Ames Housing",
    "Communities and Crime",
]


def main():
    all_results = []

    for ds_name in DATASETS:
        try:
            result = evaluate_dataset(ds_name)
            all_results.append(result)
        except Exception as e:
            print(f"\n  SKIPPING {ds_name}: {e}")
            traceback.print_exc()
            all_results.append({
                'dataset': ds_name,
                'error': str(e),
                'n_pairs': 0, 'n_unstable': 0,
                'corr_f1_flip': float('nan'),
                'f5_precision': float('nan'),
            })

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  COMPREHENSIVE SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'Dataset':25s} {'P':>4s} {'Pairs':>7s} {'Unstable':>8s} "
              f"{'r(Z_F1,flip)':>12s} {'F5 prec':>8s} {'F5 flagged':>10s}")
    print(header)
    print("-" * 100)
    for r in all_results:
        if 'error' in r:
            print(f"{r['dataset']:25s}  ** FAILED: {r['error'][:50]}")
            continue
        prec_str = f"{r['f5_precision']:.3f}" if not np.isnan(r['f5_precision']) else "N/A"
        print(f"{r['dataset']:25s} {r['P']:4d} {r['n_pairs']:7d} {r['n_unstable']:8d} "
              f"{r['corr_f1_flip']:12.3f} {prec_str:>8s} {r['f5_flagged']:10d}")
    print("=" * 100)

    # -------------------------------------------------------------------
    # Save JSON (strip large pair lists for the summary, keep a top-level copy)
    # -------------------------------------------------------------------
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ('pairs', 'flips', 'z_f1s')}
        if 'pairs' in r:
            # Keep only top 20 unstable pairs for JSON
            top_pairs = sorted(r.get('pairs', []), key=lambda x: -x.get('flip', 0))[:20]
            sr['top_unstable_pairs'] = top_pairs
        save_results.append(sr)

    json_path = os.path.join(RESULTS_DIR, "results_comprehensive.json")
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else float(x))
    print(f"\nSaved {json_path}")

    # -------------------------------------------------------------------
    # Figure: top 4 datasets by number of unstable pairs
    # -------------------------------------------------------------------
    valid_results = [r for r in all_results if 'error' not in r and r['n_pairs'] > 0]
    top4 = sorted(valid_results, key=lambda r: -r['n_unstable'])[:4]

    if len(top4) < 1:
        print("No valid datasets for figure generation.")
        return

    n_panels = len(top4)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.0))
    if n_panels == 1:
        axes = [axes]

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    for i, r in enumerate(top4):
        ax = axes[i]
        flips = np.array(r['flips'])
        z_f1s = np.clip(np.array(r['z_f1s']), 0, 20)

        ax.scatter(z_f1s, flips, s=8, alpha=0.35, c=colors[i], edgecolors='none')
        ax.axvline(x=1.96, color='gray', linestyle='--', lw=0.8)
        ax.set_xlabel('$Z_{F1}$ test statistic')
        if i == 0:
            ax.set_ylabel('Ranking flip rate')
        ax.set_title(f"{r['dataset']}\n(unstable={r['n_unstable']}, r={r['corr_f1_flip']:.2f})",
                      fontsize=8)
        ax.set_xlim(-0.5, 20)
        ax.set_ylim(-0.02, 0.55)
        ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "comprehensive_f1.pdf")
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
