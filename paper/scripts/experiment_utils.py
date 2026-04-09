"""Shared utilities for Universal Explanation Impossibility experiments."""
import json, os, random, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PAPER_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def set_all_seeds(seed: int):
    """Set numpy, random, and (if available) torch seeds."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def load_publication_style():
    """Load the shared matplotlib publication style."""
    style_path = Path(__file__).parent / 'publication_style.mplstyle'
    if style_path.exists():
        plt.style.use(str(style_path))
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def save_figure(fig, name: str):
    """Save figure to paper/figures/{name}.pdf with publication settings."""
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f"Saved figure: {out}")
    plt.close(fig)

def save_results(data: dict, name: str):
    """Save results dict to paper/results_{name}.json."""
    data['_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out = PAPER_DIR / f"results_{name}.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved results: {out}")

def train_rashomon_set(model_class, X_train, y_train, n_models: int, **kwargs):
    """Train n_models with different random seeds. Returns list of fitted models."""
    models = []
    for i in range(n_models):
        kw = dict(kwargs)
        kw['random_state'] = 42 + i
        m = model_class(**kw)
        m.fit(X_train, y_train)
        models.append(m)
    return models

def pairwise_flip_rate(rankings: np.ndarray) -> dict:
    """Given (n_models, n_items) array of ranks, compute pairwise flip rates.
    Returns dict with per-pair flip rates and summary statistics."""
    n_models, n_items = rankings.shape
    n_pairs = 0
    total_flips = 0
    pair_flips = {}
    for j in range(n_items):
        for k in range(j+1, n_items):
            flips = 0
            comparisons = 0
            for a in range(n_models):
                for b in range(a+1, n_models):
                    if (rankings[a,j] < rankings[a,k]) != (rankings[b,j] < rankings[b,k]):
                        flips += 1
                    comparisons += 1
            rate = flips / comparisons if comparisons > 0 else 0
            pair_flips[(j,k)] = rate
            total_flips += flips
            n_pairs += comparisons
    return {
        'pair_flip_rates': {f"{j},{k}": v for (j,k), v in pair_flips.items()},
        'mean_flip_rate': np.mean(list(pair_flips.values())),
        'max_flip_rate': max(pair_flips.values()),
        'overall_flip_rate': total_flips / n_pairs if n_pairs > 0 else 0,
    }

def percentile_ci(values, alpha=0.05, n_boot=2000):
    """Simple percentile bootstrap CI."""
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(np.mean(values)), float(hi)
