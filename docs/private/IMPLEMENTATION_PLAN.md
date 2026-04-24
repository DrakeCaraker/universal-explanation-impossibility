# Universal Explanation Impossibility — Implementation Plan

**Goal**: Complete the empirical experiments, paper sections, repo cleanup, and JMLR submission for "The Universal Explanation Impossibility."

**Repo**: `~/ds_projects/universal-explanation-impossibility`
**Companion repos**: `~/ds_projects/dash-shap` (Python experiments), `~/ds_projects/dash-impossibility-lean` (attribution paper)
**Target venue**: JMLR (no page limit)
**Current state**: Lean formalization DONE (67 files, 319 theorems, 60 axioms, 0 sorry). Paper has abstract + Sections 2 (framework with definitions + theorem), 3 (six instances, all 6 subsections), 8 (Lean formalization table) written. Sections 1, 5, 6, 7 are TODO with detailed outlines in the tex comments. The paper file is `paper/universal_impossibility.tex` using `neurips_2026.sty`.

---

## Audit Record

### ROUND 1 — FACTUAL ACCURACY

**File paths verified**:
- `paper/universal_impossibility.tex` -- exists, confirmed structure
- `paper/sections/instance_{attention,counterfactual,concept,causal,attribution,model_selection}.tex` -- all exist
- `paper/references.bib` -- exists, 49 entries
- `paper/scripts/llm_attention_instability.py` -- exists, uses DistilBERT + weight perturbation
- `paper/results_llm_attention.json` -- exists with valid results (88.1% instability, mean flip rate 0.33)
- `paper/scripts/publication_style.mplstyle` -- exists with Type 1 font config
- `paper/scripts/experiment_utils.py` -- DOES NOT EXIST (to be created)
- `paper/jmlr.cls` -- exists
- `UniversalImpossibility/ExplanationSystem.lean` -- exists, defines the abstract framework
- `UniversalImpossibility/UniversalResolution.lean` -- exists, proves gInvariant_stable
- `UniversalImpossibility/Ubiquity.lean` -- exists, proves generic_underspecification + fiber bridge
- `UniversalImpossibility/UniversalDesignSpace.lean` -- exists, proves design space dichotomy

**Library availability verified**:
- torch 2.8.0, transformers 4.57.6 -- available on system
- shap 0.49.1, xgboost 2.1.4, sklearn 1.6.1 -- available via requirements.txt
- dice-ml -- NOT installed, marked as optional (fallback to greedy perturbation)
- sklearn.datasets.fetch_openml('credit-g') -- works, returns (1000, 20) shape
- sklearn.datasets.load_digits -- works, returns (1797, 64) shape
- sklearn.neural_network.MLPClassifier, sklearn.svm.LinearSVC -- available

**Experiment design accuracy**:
- Attention: weight perturbation approach is sound and already validated (existing results show flip rate ~33%). Concern: perturbation is NOT true retraining. Addressed in fallback: existing `llm_finetuning_3epoch.py` does real fine-tuning.
- Counterfactual: greedy perturbation is standard. German Credit is well-studied for recourse. XGBoost with subsample=0.8 induces genuine Rashomon diversity.
- Concept probe: MLPClassifier with sklearn is correct for extracting penultimate activations. However, sklearn MLPClassifier does NOT expose intermediate layer activations directly. CORRECTION: need to use a custom forward pass or use PyTorch MLPs instead. Alternatively, use `model.coefs_` and `model.intercepts_` to manually compute activations.

**Corrections applied**:
1. Concept probe: changed from "extract penultimate layer activations" to explicit manual computation using `model.coefs_` and `model.intercepts_` OR switch to PyTorch MLP
2. Removed dice-ml from required installs (greedy perturbation is the primary method)
3. Section numbering: the paper currently has Sections 2 (Framework), 3 (Instances), 8 (Lean), with 1, 5, 6, 7 as TODO. The intro outline in the tex says "Section 2 introduces the abstract framework" but the CURRENT paper already labels it Section 2. Corrected the intro outline to match actual section numbers.

### ROUND 2 — REASONING QUALITY

**Will these experiments convince a JMLR reviewer?**
- Attention: YES, with caveats. Weight perturbation is a proxy for the Rashomon set, not true retraining diversity. A reviewer might object. IMPROVEMENT: add a 5-seed fine-tuning variant as a supplementary experiment (already scripted in `llm_finetuning_3epoch.py`). Reference both in the paper.
- Counterfactual: YES. The greedy perturbation approach directly measures whether different models recommend different feature changes. German Credit is a canonical dataset for recourse research.
- Concept probe: YES, but the MLP + MNIST setup is simple enough that a reviewer might ask "does this scale?" IMPROVEMENT: mention in the discussion that the structural argument (overparameterization) guarantees the result scales to larger networks; MNIST demonstrates it in the cleanest possible setting.

**Are there easier/better experiments?**
- For counterfactuals: could also use sklearn's make_classification with known correlated features, giving full control over the Rashomon property. ADDED as a synthetic-data variant for cleaner presentation.
- For concept probes: using load_digits (8x8, 64 features) instead of MNIST (28x28, 784 features) is FASTER and sufficient. sklearn load_digits is offline (no download needed). CHANGED to load_digits as primary, MNIST as optional extension.

**Phase ordering**: Optimal. Experiments first (shape narrative), then writing, then cleanup, then submission. The only improvement: Task 2A (introduction) can start in parallel with Phase 1 since it doesn't depend on specific experiment numbers.

**Model assignments**: Correct. Sonnet for scripting, Opus for mathematical exposition.

**Over-engineering risks**: The experiment_utils.py module is good but the `rashomon_set` function should be kept simple (just a loop with different seeds). Removed BCa bootstrap in favor of simple percentile bootstrap (BCa is finicky).

### ROUND 3 — OMISSIONS

**What would a reviewer ask for that this plan doesn't produce?**
1. **Related work section**: JMLR expects one. ADDED as Task 2F.
2. **Quantitative comparison across instances**: A table showing flip rate / instability for all 6 instances side by side would be compelling. ADDED as Task 2G (summary table).
3. **Statistical tests**: Reviewers will want p-values or confidence intervals, not just point estimates. ADDED CIs to all experiment designs.
4. **Reproducibility**: A README or script to reproduce all experiments end-to-end. ADDED as Task 3.5.
5. **The "all examples are trivial" criticism**: A reviewer might say "of course attention is unstable, everyone knows that." The defense: the contribution is not any single instance but the UNIFICATION. Each instance alone is known; the meta-theorem is new. ADDED to discussion.

**Experiments a skeptic would demand**:
1. **Fairness instance experiment**: Using COMPAS or Adult dataset, show that multiple fairness metrics (equalized odds, demographic parity, calibration) cannot all be satisfied. This is well-documented in the literature (Chouldechova 2017) so citing existing results may suffice. DECISION: cite literature rather than re-run, but note in limitations.
2. **Causal instance experiment**: Already covered by extensive Markov equivalence literature. No new experiment needed.
3. **Model selection experiment**: Trivial to demonstrate. ADDED as a quick supplementary experiment (Task 1D).

**Relationship between papers**: The plan addresses this in the introduction outline and in open questions. The key delineation: attribution paper = deep on one instance, universal paper = broad across six. They share a Lean codebase but have independent paper files.

**"Isn't this obvious?" criticism**: Addressed in Discussion 7.4. The defense is strong: Arrow's impossibility analogy, Lean formalization, and the unification argument.

---

## Phase 0: Environment Setup and Validation
**Model**: Sonnet
**Time estimate**: 30 minutes
**Gate**: Human confirms environment works before proceeding

### Task 0.1: Create experiment virtual environment
```bash
cd ~/ds_projects/universal-explanation-impossibility
python3 -m venv .venv
source .venv/bin/activate
pip install -r paper/scripts/requirements.txt
pip install torch transformers  # for attention experiment (already on system)
```

### Task 0.2: Validate existing infrastructure
```bash
cd ~/ds_projects/universal-explanation-impossibility
source .venv/bin/activate
python -c "
import shap, sklearn, xgboost, numpy, scipy, matplotlib, torch, transformers
print('All core libraries OK')
print(f'  shap={shap.__version__}, sklearn={sklearn.__version__}')
print(f'  xgboost={xgboost.__version__}, torch={torch.__version__}')
print(f'  transformers={transformers.__version__}')
from sklearn.datasets import load_digits, fetch_openml
d = load_digits(); print(f'  load_digits: {d.data.shape}')
"
```

### Task 0.3: Create shared experiment utilities
**File**: `paper/scripts/experiment_utils.py`

```python
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
```

**Expected output**: Importable module used by all experiment scripts.

### GATE 0: Human reviews environment setup, confirms libraries install, approves utility module API.

---

## Phase 1: Empirical Experiments (3+1 parallel tracks)
**Time estimate**: 1-2 days total
**Parallelism**: Tasks 1A, 1B, 1C, 1D are independent and can run in parallel.

Each experiment must produce:
1. A JSON results file in `paper/`
2. A publication-quality PDF figure in `paper/figures/`
3. A LaTeX table fragment in `paper/sections/`
4. A console summary confirming the impossibility is observed
5. 95% bootstrap confidence intervals on all key metrics

---

### Task 1A: Attention Map Instability Experiment
**Model**: Sonnet
**File**: `paper/scripts/attention_instability_experiment.py`
**Depends on**: Task 0.3

#### Design
**Research question**: Do functionally equivalent transformer models assign peak attention to different tokens?

**Dataset**: Synthetic sentiment sentences (200), same as existing `llm_attention_instability.py` (avoids HuggingFace dataset download dependency).

**Models**: 10 DistilBERT-base-uncased models created by:
1. Load pretrained `distilbert-base-uncased` (torch + transformers, already installed)
2. Create 10 perturbed copies: add Gaussian noise N(0, sigma) to all weight matrices, with sigma in {0.01, 0.02} (simulates models at same loss basin)
3. Verify all 10 models achieve equivalent predictions (>95% agreement on sentiment classification)

**What to measure**:
1. **Attention rollout**: propagate attention through all 6 DistilBERT layers to get per-token importance
2. **Per-sentence argmax token**: which token gets highest aggregated attention
3. **Flip rate**: fraction of sentence pairs where the argmax token differs between two models
4. **Kendall tau**: mean rank correlation of attention distributions across all model pairs
5. **95% bootstrap CIs** on flip rate and tau

**Figure**: `paper/figures/attention_instability.pdf`
- 2-panel figure (3.25in x 2.4in each, side by side at JMLR column width):
  - Left: heatmap showing attention rollout over tokens for 3 example sentences, 5 models (rows). Highlight argmax per model. Caption explains divergence.
  - Right: histogram of pairwise Kendall tau values across all (model, sentence) pairs, with vertical line at mean. Show that distribution is broad (not concentrated near 1).

**Table**: `paper/sections/table_attention.tex` — LaTeX tabular with booktabs:
```latex
\begin{table}[t]
\centering
\caption{Attention map instability across 10 functionally equivalent DistilBERT models.}
\label{tab:attention}
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Prediction agreement & XX.X\% $\pm$ X.X\% \\
Argmax flip rate & XX.X\% $\pm$ X.X\% \\
Mean Kendall $\tau$ & X.XX $\pm$ X.XX \\
\bottomrule
\end{tabular}
\end{table}
```

**Expected results**: Flip rate ~30% (consistent with existing results showing mean 0.33). Kendall tau ~0.6. Prediction agreement >95%.

**Fallback**: If torch/transformers fail, use existing `paper/results_llm_attention.json` and generate figures from those cached results.

**Output files**:
- `paper/results_attention_instability.json`
- `paper/figures/attention_instability.pdf`
- `paper/sections/table_attention.tex`

---

### Task 1B: Counterfactual Explanation Instability Experiment
**Model**: Sonnet
**File**: `paper/scripts/counterfactual_instability_experiment.py`
**Depends on**: Task 0.3

#### Design
**Research question**: Do equivalent-accuracy models produce contradictory counterfactual explanations?

**Dataset**: German Credit (UCI via `fetch_openml('credit-g')`). 1000 samples, 20 features, binary target. Well-studied in recourse/fairness literature. One-hot encode categoricals, standardize numerics.

**Models**: 20 XGBoost classifiers with:
- max_depth=4, n_estimators=100, learning_rate=0.1, subsample=0.8
- Different random seeds (seed = 42 + i for i in range(20))
- Verify all achieve equivalent test AUC (within 0.02, using 80/20 split with fixed seed for split)

**Counterfactual method** (self-contained, no external library):
For each of 50 test points predicted as "bad credit":
1. For each of the 20 models:
   a. Start from the query point x_0
   b. Compute feature importance (SHAP or gain-based) to order features
   c. Greedily perturb the top features by delta = 0.1 * feature_std toward the positive-class centroid
   d. After each step, check if prediction flips
   e. Record the counterfactual x' and the direction of change per feature
2. For robustness: also compute counterfactual distance ||x' - x_0||

**What to measure**:
1. **Direction flip rate per feature**: fraction of model pairs recommending opposite directions for the same query
2. **Cross-model recourse validity**: fraction of counterfactuals from model_i that still flip prediction on model_j
3. **Distance instability**: CV of counterfactual L2 distances across models
4. **95% bootstrap CIs** on all metrics

**Figure**: `paper/figures/counterfactual_instability.pdf`
- 2-panel figure:
  - Left: bar chart of direction flip rate for top 10 most unstable features. Dashed line at 50% (coin flip). Error bars are 95% bootstrap CIs.
  - Right: scatter of (cross-model validity, CF distance) per query point. Marker color = consensus level (fraction of models agreeing on direction for most important feature). Trendline showing inverse relationship.

**Table**: `paper/sections/table_counterfactual.tex`

**Expected results**: Direction flip rate >15% for correlated features. Cross-model validity <70%. This confirms the counterfactual impossibility (Proposition 2).

**Output files**:
- `paper/results_counterfactual_instability.json`
- `paper/figures/counterfactual_instability.pdf`
- `paper/sections/table_counterfactual.tex`

---

### Task 1C: Concept Probe Instability Experiment
**Model**: Sonnet
**File**: `paper/scripts/concept_probe_instability_experiment.py`
**Depends on**: Task 0.3

#### Design
**Research question**: Do functionally equivalent neural networks encode the same concept in incompatible directions?

**Dataset**: sklearn `load_digits()` (8x8 pixel images, 10 classes, 1797 samples, no download needed). Define concept "curved digit" (0,2,3,5,6,8,9) vs "angular digit" (1,4,7).

**Models**: 15 sklearn MLPClassifier models for 10-class classification:
- Architecture: hidden_layer_sizes=(128, 64), activation='relu', max_iter=500
- Different random seeds
- Train on 70% data, test on 30%
- Verify all achieve >95% test accuracy

**Concept probe extraction** (manual forward pass through sklearn MLP):
```python
def get_penultimate_activations(model, X):
    """Manually compute penultimate-layer activations from sklearn MLP."""
    h = X
    # Forward through all layers except the last
    for i in range(len(model.coefs_) - 1):
        h = h @ model.coefs_[i] + model.intercepts_[i]
        h = np.maximum(h, 0)  # ReLU
    return h  # shape: (n_samples, 64)
```

For each of the 15 models:
1. Compute penultimate activations on 500 held-out samples
2. Fit LinearSVC to classify "curved" vs "angular" from activations
3. Extract CAV: v = svc.coef_[0]; v = v / np.linalg.norm(v)

**What to measure**:
1. **Cosine similarity matrix**: 15x15, abs(cos(v_i, v_j)) for all model pairs
2. **Concept direction instability**: 1 - mean(|cos similarity|)
3. **TCAV-like score**: for each model, fraction of test digits where the directional derivative of the predicted class logit along CAV is positive. Measure std across models.
4. **Prediction agreement**: fraction of test samples where all 15 models agree

**Figure**: `paper/figures/concept_probe_instability.pdf`
- 2-panel figure:
  - Left: heatmap of |cosine similarity| matrix (15x15). Annotate mean off-diagonal value. Diverging colormap.
  - Right: box plot of TCAV-like scores across 15 models for 2 concepts (curved, symmetric = {0,1,8}). Show that TCAV scores vary substantially.

**Table**: `paper/sections/table_concept.tex`

**Expected results**: Mean |cosine similarity| < 0.6 (concept directions are diverse). TCAV score std > 0.1. Prediction agreement >95%. This confirms the concept probe impossibility (Theorem 5).

**Why this works**: An MLP with (128, 64) hidden layers and 64 features has 128*64 + 64*64 + 64*10 = 13,056 parameters for 1258 training samples (70% of 1797). This is 10x overparameterized, ensuring the Rashomon set is non-trivial and different seeds produce genuinely different internal representations.

**Output files**:
- `paper/results_concept_probe_instability.json`
- `paper/figures/concept_probe_instability.pdf`
- `paper/sections/table_concept.tex`

---

### Task 1D: Model Selection Instability Experiment (Supplementary)
**Model**: Sonnet
**File**: `paper/scripts/model_selection_instability_experiment.py`
**Depends on**: Task 0.3

A quick supplementary experiment confirming Instance 6.

**Design**: Train 50 XGBoost models on German Credit with different seeds. On 10 different random 80/20 train/test splits, record which model has highest test AUC. Measure how often the "best" model changes across splits (flip rate of model selection).

**Expected results**: The "best" model changes in >80% of splits. Confirms model selection impossibility.

**Output**: `paper/results_model_selection_instability.json`, brief supplementary table.

---

### GATE 1: Human reviews all experiment results.
**Checklist**:
- [ ] All experiments run to completion without errors
- [ ] Flip rates / instability measures are statistically significant (CIs do not include 0)
- [ ] Prediction equivalence is confirmed (models ARE functionally equivalent)
- [ ] Figures are publication-quality (Type 1 fonts, correct sizing, readable at print)
- [ ] Results are consistent with theoretical predictions
- [ ] No experiment is trivially explained away by noise, bad seeds, or implementation bugs
- [ ] If any experiment shows NO instability, reason documented (Rashomon property may not hold)

---

## Phase 2: Paper Writing
**Model**: Opus (all writing tasks)
**Time estimate**: 1-2 days
**Depends on**: Phase 1 results (but Task 2A can start in parallel with Phase 1)
**Parallelism**: 2A can start immediately. 2B, 2C can start after experiment results available. 2D last. 2E-2G after 2A-2D.

### Task 2A: Write Section 1 (Introduction)
**File**: `paper/universal_impossibility.tex` (replace lines 101-123, the TODO comment block)
**Length**: ~1.5 pages

#### Content outline:

**Paragraph 1 -- The practitioner's experience (3-4 sentences)**
Open with: a data scientist runs SHAP on a credit risk model, retrains from a different seed, and the top feature flips. A different team uses LIME on the same model and gets a third ranking. An NLP engineer checks attention weights, retrains, and the "most attended" token changes. This is not a software bug. Cite: Krishna et al. (2022) disagreement problem, Molnar et al. (2022) general pitfalls.

**Paragraph 2 -- The XAI promise vs reality (3-4 sentences)**
The XAI program promises explanations that are faithful (reflect the model), stable (reproducible), and decisive (give a definite answer). Regulators now demand all three: the EU AI Act requires "meaningful explanations" and SR 11-7 requires "developmental evidence" for model decisions. But a growing body of theoretical results shows the promise is mathematically impossible in specific domains. Cite: Bilodeau et al. (2024), Jain & Wallace (2019), Chouldechova (2017), Kleinberg et al. (2017), Verma & Pearl (1991).

**Paragraph 3 -- The gap (3-4 sentences)**
Each impossibility was proved independently, in domain-specific language, under domain-specific assumptions. The attribution impossibility assumes collinearity; Chouldechova's impossibility assumes unequal base rates; Markov equivalence assumes faithfulness of the DAG. No one has asked: is there ONE structural cause? Is the impossibility intrinsic to underspecification itself, not to any particular domain?

**Paragraph 4 -- Our contribution (itemized)**
We prove a Universal Explanation Impossibility: no explanation of an underspecified system can simultaneously be faithful, stable, and decisive. The theorem:
1. Unifies six impossibility results under a single four-line proof requiring only the Rashomon property
2. Is instantiated for: feature attribution, attention maps, counterfactual explanations, concept probes, causal discovery, model selection
3. Is accompanied by empirical validation for 4 of the 6 instances (attention, counterfactual, concept probes, model selection)
4. Provides a uniform resolution: G-invariant aggregation (DASH for attribution, CPDAG for causal discovery, ensemble probes for concepts)
5. Establishes ubiquity: the impossibility is generic whenever dim(parameters) > dim(observables)
6. Is mechanically verified in Lean 4 (67 files, 319 theorems, 60 axioms, 0 sorry)

**Paragraph 5 -- Relationship to companion work (2-3 sentences)**
This paper generalizes the Attribution Impossibility of our companion paper from feature attribution to arbitrary explanation systems. The companion paper provides quantitative depth (divergence rates, ensemble size bounds, DASH optimality) for the attribution instance; this paper provides the abstract framework, five additional instances, and ubiquity. The papers share a Lean codebase but are self-contained.

**Paragraph 6 -- Paper organization (2 sentences)**
Section 2 presents related work. Section 3 introduces the abstract framework. Section 4 instantiates it across six domains. Section 5 presents the uniform resolution. Section 6 establishes ubiquity. Section 7 discusses implications.

#### Citations: `bilodeau2024impossibility`, `damour2022underspecification`, `rudin2024amazing`, `chouldechova2017fair`, `kleinberg2017inherent`, `verma1991equivalence`, `krishna2022disagreement`, `molnar2022general`, `jainwallace2019`, `euaiact2024`, `occ2011sr117`, `huang2024failings`, `rao2025limits`, `demoura2021lean4`, `arrow1951social`

---

### Task 2F: Write Section 2 (Related Work) [NEW -- added after Round 3 audit]
**File**: create `paper/sections/related_work.tex`, include from `universal_impossibility.tex`
**Length**: ~0.75 pages

#### Content outline:

**Impossibility results in XAI**: Bilodeau et al. (2024) attribution impossibility, Huang & Marques-Silva (2024) SHAP failings, Rao (2025) algorithmic information theory limits, Noguer i Alonso (2025) mathematical foundations. Our contribution: unification.

**Underspecification and multiplicity**: D'Amour et al. (2022) underspecification in modern ML, Fisher et al. (2019) model reliance / variable importance clouds, Rudin et al. (2024) Rashomon sets, Semenova et al. (2022) existence of simpler models, Marx et al. (2024) uncertainty-aware explainability. Our contribution: connect underspecification to explanation impossibility via a formal bridge.

**Fairness impossibilities**: Chouldechova (2017), Kleinberg et al. (2017). Our contribution: recast as an instance of the universal impossibility.

**Formal verification in ML**: de Moura & Ullrich (2021) Lean 4, Nipkow (2009) Arrow's theorem in Isabelle, Zhang et al. (2026) statistical learning theory in Lean. Our contribution: first Lean formalization of an impossibility theorem in explainability.

**Aggregation and resolution**: Laberge et al. (2023) partial order on attributions, Decker et al. (2024) optimized aggregation, Herren & Hahn (2023) statistical inference under multiplicity. Our contribution: show these are all instances of G-invariant resolution.

---

### Task 2B: Write Section 5 (The Universal Resolution)
**File**: `paper/universal_impossibility.tex` (replace lines 295-321)
**Length**: ~1.5 pages

#### Content outline:

**5.1 G-invariant explanation systems**
- Definition: G acts on config space Theta, preserving observables. A G-invariant explanation map satisfies exp(g * theta) = exp(theta).
- Proposition (mirrors `gInvariant_stable` in `UniversalResolution.lean`): If G acts transitively on observe-fibers, every G-invariant map is stable.
- Proposition (mirrors `universal_design_space_dichotomy` in `UniversalDesignSpace.lean`): No stable map for a system with the Rashomon property can be both faithful and decisive.
- Corollary (Pareto frontier): achievable combinations are {faithful + decisive, unstable} or {stable + faithful-in-expectation, indecisive}.

**5.2 Instance-specific resolutions (table)**

| Instance | Symmetry group G | Resolution | Sacrificed property |
|----------|-----------------|------------|---------------------|
| Attribution | Feature permutation (collinear group) | DASH (ensemble SHAP averaging) | Decisive ranking of collinear features |
| Attention | Weight-space symmetries | Averaged attention rollout over Rashomon set | Single most-attended token |
| Counterfactual | Decision-boundary variation | Rashomon-robust recourse directions | Single counterfactual |
| Concept probe | Representation rotation symmetry | Averaged CAV / subspace reporting | Single concept direction |
| Causal discovery | DAG reversal within Markov class | CPDAG reporting | Fully oriented graph |
| Model selection | Model permutation in Rashomon set | Ensemble prediction / Rashomon set report | Single best model |

**5.3 DASH as the prototype resolution**
Brief (4-5 sentences): DASH averages SHAP values over retrained ensemble, achieving E[phi_j] = E[phi_k] for symmetric features (consensus equity, proved in Corollary.lean). It is the concrete instantiation of the G-invariant resolution for Instance 1. The companion paper proves DASH is Pareto-optimal: no other stable attribution method can be more faithful.

**5.4 Meta-theorem**
In any explanation system where the Rashomon property arises from a finite group action, the orbit-averaged map is well-defined, stable, faithful in expectation, and Pareto-optimal. (Reference UniversalResolution.lean.)

---

### Task 2C: Write Section 6 (Ubiquity)
**File**: `paper/universal_impossibility.tex` (replace lines 327-350)
**Length**: ~1 page

#### Content outline:

**6.1 Generic underspecification**
- Proposition: If dim(H) > dim(Y), the observe-fiber has positive dimension.
- The Rashomon property holds generically (non-constant explanation on positive-dimensional fiber).
- Reference: `generic_underspecification` in Ubiquity.lean.

**6.2 Neural network dimensional argument**
- d hidden units, n < d training points: zero-loss manifold has dim >= d - n > 0.
- Symmetric features => symmetric zero-loss manifold => explanation symmetry.
- Coin-flip corollary: P(exp(h)_j > exp(h)_k) = 1/2 for symmetric features.
- Reference: `fiber_nondegeneracy_implies_impossibility` in Ubiquity.lean.

**6.3 Prevalence table** (as in audit above)

**6.4 Arrow's impossibility analogy** (3-4 sentences, as in audit above)

---

### Task 2D: Write Section 7 (Discussion)
**File**: `paper/universal_impossibility.tex` (replace lines 405-436)
**Length**: ~1.5 pages

#### Content outline:

**7.1 The ceiling of explainability** (4-5 sentences): ceiling theorem, not specific to any method.

**7.2 Regulatory implications** (4-5 sentences): EU AI Act, SR 11-7, resolution via G-invariant aggregation.

**7.3 What explainability CAN do** (4-5 sentences): G-invariant resolution is constructive, not nihilistic.

**7.4 The "isn't this obvious?" defense** (4-5 sentences):
The intuition may be familiar, but the contribution is:
(a) making explicit that six domains share ONE mechanism (not six separate mechanisms);
(b) the Lean formalization eliminates logical gaps (67 files, 319 theorems, 0 sorry);
(c) the empirical experiments show quantifiable severity (33% flip rate for attention, XX% for counterfactuals);
(d) the resolution (G-invariant maps) is constructive and actionable.
Analogy: Arrow's impossibility theorem (1951) was arguably "obvious" once stated. Its formalization created the field of social choice theory.

**7.5 The "all instances are trivial" defense** [NEW -- added after Round 3 audit]:
A reviewer might say: "attention instability is known (Jain & Wallace 2019), Markov equivalence is classical (Verma & Pearl 1991), fairness impossibility is proved (Chouldechova 2017)." The contribution is not any single instance but the meta-theorem: the SAME four-line proof closes ALL six cases. Before this work, a practitioner encountering attention instability might switch to SHAP, encountering SHAP instability might switch to concept probes, encountering probe instability might switch to counterfactual explanations -- each time hoping the new method escapes the problem. Our theorem shows no such escape exists: the impossibility is structural, not method-specific.

**7.6 Limitations and open questions** (5 items, as in audit above)

---

### Task 2E: Integrate experiment results into instance sections
**Model**: Opus
**Files**: `paper/sections/instance_attention.tex`, `paper/sections/instance_counterfactual.tex`, `paper/sections/instance_concept.tex`

For each section, add an "Empirical validation" paragraph after the theorem/proof, referencing the figure and table from Phase 1 with specific quantitative findings.

---

### Task 2G: Create cross-instance summary table [NEW -- added after Round 3 audit]
**Model**: Opus
**File**: Add to `paper/universal_impossibility.tex` (new Table in Section 4, after all instances)

```latex
\begin{table}[t]
\centering
\caption{Empirical validation of the Rashomon property across explanation types.}
\label{tab:cross-instance}
\begin{tabular}{llrr}
\toprule
Instance & Instability metric & Value & Prediction equiv. \\
\midrule
Attribution (GBDT) & SHAP flip rate & XX\% & $>$99\% \\
Attention (DistilBERT) & Argmax flip rate & XX\% & $>$95\% \\
Counterfactual (XGBoost) & Direction flip rate & XX\% & $\Delta$AUC $<$ 0.02 \\
Concept probe (MLP) & $1 - |\cos|$ & X.XX & $>$95\% \\
Model selection (XGBoost) & Best-model flip rate & XX\% & By construction \\
Causal discovery & \multicolumn{3}{l}{\textit{Classical (Verma \& Pearl, 1991)}} \\
\bottomrule
\end{tabular}
\end{table}
```

This table is the single most impactful artifact for a reviewer: it shows at a glance that the impossibility is empirically confirmed across diverse explanation types.

---

### GATE 2: Human reviews all written sections.
**Checklist**:
- [ ] Introduction flows: practitioner experience -> promise vs reality -> gap -> contribution
- [ ] Related work is comprehensive but concise
- [ ] Resolution section is mathematically precise with Lean references
- [ ] Ubiquity makes the "not exotic" argument convincingly
- [ ] Discussion addresses "isn't this obvious?" and "all instances trivial"
- [ ] Cross-instance summary table is populated with real numbers
- [ ] All experiment results correctly integrated
- [ ] All citations in references.bib
- [ ] Paper compiles cleanly

---

## Phase 3: Repo Cleanup and Consistency
**Model**: Sonnet
**Time estimate**: 2-4 hours
**Depends on**: Phases 1-2

### Task 3.1: Update CLAUDE.md
**File**: `~/ds_projects/universal-explanation-impossibility/CLAUDE.md`

Update to reflect the universal impossibility scope:
- Title and description
- Architecture section with universal framework files
- File structure with all 67 Lean files organized by level
- Correct counts (67 files, 319 theorems+lemmas, 60 axioms, 0 sorry)
- Two-paper structure documentation
- Updated submission targets

### Task 3.2: Verify paper-code consistency
Run the standard verification block and update any mismatched numbers in both `universal_impossibility.tex` and `CLAUDE.md`.

### Task 3.3: Add missing bib entries
**File**: `paper/references.bib`

Expected additions:
- `abnar2020quantifying` — Abnar & Zuidema, "Quantifying Attention Flow in Transformers," ACL 2020
- `wachter2017counterfactual` — Wachter, Mittelstadt, Russell, "Counterfactual Explanations without Opening the Black Box," Harvard Journal of Law & Technology 2017
- Any new citations from related work section

### Task 3.4: Create unified experiment runner
**File**: `paper/scripts/run_all_universal_experiments.py`

```python
#!/usr/bin/env python3
"""Run all experiments for the Universal Explanation Impossibility paper."""
import subprocess, sys
scripts = [
    'attention_instability_experiment.py',
    'counterfactual_instability_experiment.py',
    'concept_probe_instability_experiment.py',
    'model_selection_instability_experiment.py',
]
for s in scripts:
    print(f"\n{'='*60}\nRunning {s}\n{'='*60}")
    subprocess.run([sys.executable, f'paper/scripts/{s}'], check=True)
print("\nAll experiments complete.")
```

### Task 3.5: Create reproducibility README [NEW -- added after Round 3 audit]
**File**: `paper/scripts/README.md`

Document how to reproduce all experiments:
```
## Reproducing experiments
1. Install: pip install -r paper/scripts/requirements.txt && pip install torch transformers
2. Run: python paper/scripts/run_all_universal_experiments.py
3. Generate figures: python paper/scripts/regenerate_universal_figures.py
4. Compile paper: cd paper && pdflatex universal_impossibility && bibtex ...
```

### GATE 3: Human reviews repo state.
**Checklist**:
- [ ] CLAUDE.md is accurate
- [ ] All theorem/axiom/file counts match between code and paper
- [ ] All citations resolve
- [ ] Paper compiles cleanly
- [ ] Experiments are reproducible from the README

---

## Phase 4: JMLR Preparation
**Model**: Opus (4.1, 4.2), Sonnet (4.3, 4.4)
**Time estimate**: 2-4 hours
**Depends on**: Phase 3

### Task 4.1: Create JMLR version
**File**: `paper/universal_impossibility_jmlr.tex`

1. Copy `universal_impossibility.tex`
2. Replace documentclass/style with `\documentclass{jmlr}`
3. Add JMLR metadata: `\jmlrheading{year}{volume}{pages}{submitted}{published}{author-name}`
4. Add `\ShortHeadings{Universal Explanation Impossibility}{Caraker, Arnold, Rhoads}`
5. Add `\begin{keywords}` block
6. Uncomment author block with emails
7. Ensure all `\input{sections/...}` paths work
8. Compile and verify

### Task 4.2: Write JMLR cover letter
**File**: `paper/submission/cover_letter_universal.tex`

Key points:
1. First unified impossibility theorem across six explanation types
2. Lean 4 mechanical verification (unprecedented in ML explainability)
3. Empirical validation across four domains
4. Relevance to JMLR: broad theoretical result, survey-like scope, benefits from no page limit
5. Relationship to companion attribution paper (delineation: depth vs breadth)

### Task 4.3: Prepare supplementary material
**File**: `paper/universal_impossibility_supplement.tex`

### Task 4.4: Submission checklist
**File**: `paper/submission/checklist_universal.md`

### GATE 4: Human reviews final package and submits.

---

## Model Assignments Summary

| Task | Model | Rationale |
|------|-------|-----------|
| 0.x (setup) | Sonnet | Boilerplate |
| 1A-1D (experiments) | Sonnet | Python scripting |
| 2A-2G (writing) | Opus | Mathematical exposition, argument construction |
| 3.x (cleanup) | Sonnet | Mechanical |
| 4.1-4.2 (JMLR version, cover letter) | Opus | Judgment needed |
| 4.3-4.4 (supplement, checklist) | Sonnet | Mechanical |

---

## Execution Timeline

```
Week 1:
  Day 1-2: Phase 0 (setup) + Phase 1 (experiments, parallel)
           + Task 2A (intro, in parallel with experiments)
  Day 3-4: Phase 2 (remaining writing, after experiment results)

Week 2:
  Day 5: Phase 3 (cleanup, consistency)
  Day 6: Phase 4 (JMLR formatting, cover letter)
  Day 7: Final review, address any issues, submit
```

---

## Confidence Ratings

| Phase | Confidence | Risk | Mitigation |
|-------|------------|------|------------|
| Phase 0 | HIGH | None | Libraries verified on system |
| Task 1A (attention) | HIGH | Perturbation as proxy | Cite existing fine-tuning results |
| Task 1B (counterfactual) | MEDIUM | German Credit Rashomon set may be small | Use subsample=0.8, increase to 30 models if needed |
| Task 1C (concept probe) | HIGH | MLP activations need manual extraction | Code provided, sklearn coefs_ well-documented |
| Task 1D (model selection) | HIGH | Trivial experiment | That's the point |
| Task 2A (intro) | HIGH | Outline already exists in tex | Fill in the template |
| Task 2F (related work) | HIGH | Standard section | Citations already in bib |
| Task 2B (resolution) | HIGH | Lean proofs exist | Exposition of proved results |
| Task 2C (ubiquity) | MEDIUM | Arrow analogy could be stretched | Keep it to 3-4 sentences, don't overclaim |
| Task 2D (discussion) | MEDIUM | "Isn't this obvious?" is hard to write well | Multiple drafts, Arrow analogy |
| Task 2G (summary table) | HIGH | Depends on experiment numbers | Numbers will be available |
| Phase 3 | HIGH | Mechanical | Scripted verification |
| Phase 4 | HIGH | JMLR formatting is standard | jmlr.cls already in repo |

---

## Open Questions

1. **Venue choice**: JMLR is the primary target (no page limit, values formal results). Alternatives: TMLR (faster review, lower bar), FoDS, COLT. JMLR is correct for a result of this scope.

2. **Two-paper relationship**: If both go to JMLR, explain the delineation: attribution paper = deep (one instance, quantitative bounds, DASH), universal paper = broad (six instances, abstract framework, ubiquity). They share Lean code but have independent tex files.

3. **Weight perturbation vs fine-tuning**: Perturbation is the primary approach for attention; fine-tuning results available as supplement. A reviewer may prefer fine-tuning. Decision: present perturbation as the clean demonstration (controlled experiment), mention fine-tuning in supplement.

4. **Fairness experiment**: Defer. The literature (Chouldechova 2017, Kleinberg 2017) provides the empirical evidence. Cite rather than re-run.

5. **Related work section**: Yes, add as Section 2. JMLR expects it. Keep to 0.75 pages.

6. **The "all instances are trivial" criticism**: Address head-on in Discussion. The contribution is unification, not any single instance. Arrow's theorem analogy is the strongest defense.
