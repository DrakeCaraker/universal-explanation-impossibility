# Scripts — Universal Explanation Impossibility

Experiment scripts that support the empirical section of `paper/universal_impossibility.tex`.

## Quick Start

```bash
# From repo root:
pip install -r paper/scripts/requirements.txt
python paper/scripts/run_all_universal_experiments.py
```

---

## 1. Install Dependencies

### Core dependencies (all experiments)

```bash
pip install -r paper/scripts/requirements.txt
```

`requirements.txt` pins all versions used in the original experiments (generated 2026-04-03, Apple Silicon):

| Package | Version |
|---------|---------|
| xgboost | 2.1.4 |
| shap | 0.49.1 |
| scipy | 1.13.1 |
| numpy | 2.0.2 |
| matplotlib | 3.9.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| catboost | 1.2.10 |
| sympy | 1.14.0 |

### Attention experiment (Experiment 1 only)

The attention instability experiment requires PyTorch and HuggingFace Transformers:

```bash
pip install transformers torch
```

The experiment uses `DistilBERT-base-uncased` with weight perturbation (no GPU required; runs on CPU in ~5–20 min depending on hardware). If transformers is not installed, the script falls back to cached results in `paper/results_llm_attention.json`.

### Virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r paper/scripts/requirements.txt
pip install transformers torch  # for attention experiment
```

---

## 2. Run Experiments

### Run all four universal experiments at once

```bash
python paper/scripts/run_all_universal_experiments.py
```

This runs the four experiments sequentially and prints a summary. Total runtime is approximately 15–45 minutes depending on whether the attention experiment loads from cache or re-runs inference.

### Run experiments individually

Each script can be run standalone from the repo root:

```bash
# Experiment 1: Attention map instability (DistilBERT, Instance 2)
python paper/scripts/attention_instability_experiment.py

# Experiment 2: Counterfactual explanation instability (XGBoost, Instance 3)
python paper/scripts/counterfactual_instability_experiment.py

# Experiment 3: Concept probe instability / TCAV (Instance 4)
python paper/scripts/concept_probe_instability_experiment.py

# Experiment 4: Model selection instability (Rashomon multiplicity, Instance 6)
python paper/scripts/model_selection_instability_experiment.py
```

### Expected outputs

| Experiment | JSON results | LaTeX table | Figure |
|------------|-------------|-------------|--------|
| Attention instability | `paper/results_attention_instability.json` | `paper/sections/table_attention.tex` | `paper/figures/attention_instability.pdf` |
| Counterfactual instability | `paper/results_counterfactual_instability.json` | `paper/sections/table_counterfactual.tex` | `paper/figures/counterfactual_instability.pdf` |
| Concept probe instability | `paper/results_concept_probe_instability.json` | `paper/sections/table_concept.tex` | — |
| Model selection instability | `paper/results_model_selection_instability.json` | `paper/sections/table_model_selection.tex` | — |

Results committed to this repo are used directly by `\input{sections/table_*.tex}` in `universal_impossibility.tex`.

### Key metrics (paper values to reproduce)

| Instance | Metric | Target value |
|----------|--------|-------------|
| Attention (DistilBERT) | Argmax flip rate | ~60% |
| Counterfactual (XGBoost) | Direction flip rate | ~23.5% |
| Concept probe (TCAV) | Direction instability | ~0.90 |
| Model selection | Best-model flip rate | ~80% |

These values appear in Table 1 of `universal_impossibility.tex`. Exact values may vary slightly across runs due to stochastic data loading; the qualitative conclusions are robust.

---

## 3. Compile the Paper

### Universal impossibility paper (primary)

```bash
cd paper
pdflatex -interaction=nonstopmode universal_impossibility.tex
bibtex universal_impossibility
pdflatex -interaction=nonstopmode universal_impossibility.tex
pdflatex -interaction=nonstopmode universal_impossibility.tex
```

Or via Make:

```bash
make universal   # if defined; otherwise use the pdflatex sequence above
```

### Companion attribution paper (JMLR)

```bash
make jmlr        # compiles paper/main_jmlr.tex
```

### Companion attribution paper (NeurIPS)

```bash
make neurips     # compiles paper/main.tex + supplement.tex
```

### All paper versions

```bash
make paper
```

### Requirements for compilation

- TeX Live 2024+ with `pdflatex`, `bibtex`
- Packages: `neurips_2026.sty` (included), `jmlr.cls` + `jmlrutils.sty` (included in `paper/`)
- Standard CTAN packages: `amsmath`, `amssymb`, `booktabs`, `hyperref`, `graphicx`, `subcaption`

---

## 4. Verify Lean Formalization

Run the paper-code consistency check before committing:

```bash
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
ls UniversalImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
```

Expected output (matches `paper/universal_impossibility.tex` Table 2):

```
theorems+lemmas: 319
axioms: 60
sorry: 0
files: 67
```

To compile the Lean proofs (requires `elan` and Lean 4 `v4.30.0-rc1`):

```bash
make lean        # runs lake build (~5 min on first run)
```

Or:

```bash
lake build
```

---

## 5. Other Experiment Scripts

The `paper/scripts/` directory also contains scripts for the companion attribution paper (`main.tex` / `main_jmlr.tex`). These cover additional experiments (ratio divergence, DASH resolution, SNR calibration, etc.) and are not part of the universal impossibility experiments above.

To run the full set of attribution experiments:

```bash
make experiments    # runs all scripts in paper/scripts/ (~30–60 min)
```

To regenerate all figures for the attribution paper:

```bash
python paper/scripts/regenerate_all_figures.py
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
Install core dependencies: `pip install -r paper/scripts/requirements.txt`

**`ModuleNotFoundError: No module named 'transformers'`**
Install for attention experiment: `pip install transformers torch`
The script will fall back to cached results (`paper/results_llm_attention.json`) if transformers is unavailable.

**`openml` errors in prevalence scripts**
Install: `pip install openml` (not in core requirements; only needed for prevalence survey scripts)

**Lean build fails**
Ensure `elan` is installed and the toolchain matches `lean-toolchain` (`leanprover/lean4:v4.30.0-rc1`). Run `elan self update` and then `lake build`.
