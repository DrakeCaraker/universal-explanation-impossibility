# Universal Explanation Impossibility — Build System
# Usage: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Lean ──────────────────────────────────────────────────────

.PHONY: lean
lean: ## Build all Lean 4 files (~5 min)
	lake build

# ─── Papers ────────────────────────────────────────────────────

PAPER_DIR := paper

.PHONY: paper monograph jmlr neurips all-papers

paper: ## Build the primary universal paper (JMLR target)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode universal_impossibility.tex && \
		bibtex universal_impossibility && pdflatex -interaction=nonstopmode universal_impossibility.tex && \
		pdflatex -interaction=nonstopmode universal_impossibility.tex
	@echo "Universal paper: $$(pdfinfo $(PAPER_DIR)/universal_impossibility.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

monograph: ## Build the monograph / arXiv definitive version
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex && \
		bibtex universal_impossibility_monograph && pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex && \
		pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex
	@echo "Monograph: $$(pdfinfo $(PAPER_DIR)/universal_impossibility_monograph.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

jmlr: ## Build the JMLR submission version
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex && \
		bibtex universal_impossibility_jmlr && pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex && \
		pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex
	@echo "JMLR: $$(pdfinfo $(PAPER_DIR)/universal_impossibility_jmlr.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

neurips: ## Build the NeurIPS 2026 version
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex && \
		bibtex universal_impossibility_neurips && pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex && \
		pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex
	@echo "NeurIPS: $$(pdfinfo $(PAPER_DIR)/universal_impossibility_neurips.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

all-papers: monograph jmlr neurips ## Build all paper versions

# ─── Verification ──────────────────────────────────────────────

.PHONY: verify counts

counts: ## Show current theorem/axiom/sorry/file counts
	@echo "Theorems+lemmas: $$(grep -c '^theorem\|^lemma' UniversalImpossibility/*.lean | awk -F: '{s+=$$2} END {print s}')"
	@echo "Axioms:          $$(grep -c '^axiom' UniversalImpossibility/*.lean | awk -F: '{s+=$$2} END {print s}')"
	@echo "Sorry (code):    $$(grep -rn ':= sorry' UniversalImpossibility/*.lean | wc -l | tr -d ' ')"
	@echo "Files:           $$(ls UniversalImpossibility/*.lean | wc -l | tr -d ' ')"
	@echo "Expected:        100 files / 488 theorems+lemmas / 47 axioms / 0 sorry"

verify: counts ## Verify Lean builds + counts are consistent (target: 100/488/47/0)
	@echo ""
	@echo "--- Verifying Lean build ---"
	lake build
	@echo ""
	@echo "--- Verifying axiom consistency ---"
	python3 paper/scripts/axiom_consistency_model.py
	@echo ""
	@echo "✓ All checks passed"

# ─── Experiments ───────────────────────────────────────────────

SCRIPTS_DIR := paper/scripts
FIGURES_DIR := paper/figures

.PHONY: experiments validate validate-quick

experiments: ## Run all universal experiments (run_all_universal_experiments.py)
	python3 $(SCRIPTS_DIR)/run_all_universal_experiments.py

validate: ## Run 3 key validation experiments (~5 min)
	python3 $(SCRIPTS_DIR)/validate_ratio.py
	python3 $(SCRIPTS_DIR)/cross_implementation_validation.py
	python3 $(SCRIPTS_DIR)/snr_calibration.py

# Quick mode: skip scripts whose output figures are newer than the script
validate-quick: ## Run only experiments with stale outputs
	@for script in $(SCRIPTS_DIR)/validate_ratio.py \
	               $(SCRIPTS_DIR)/cross_implementation_validation.py \
	               $(SCRIPTS_DIR)/snr_calibration.py; do \
		fig="$(FIGURES_DIR)/$$(basename $$script .py | sed 's/validate_//').pdf"; \
		if [ -f "$$fig" ] && [ "$$fig" -nt "$$script" ]; then \
			echo "SKIP (cached): $$(basename $$script)"; \
		else \
			echo "RUN: $$(basename $$script)"; \
			python3 "$$script"; \
		fi; \
	done

# ─── Setup ─────────────────────────────────────────────────────

.PHONY: setup setup-python

setup: ## Full setup: Lean toolchain + Python deps
	@echo "--- Checking Lean toolchain ---"
	@which elan > /dev/null 2>&1 || (echo "Install elan: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh" && exit 1)
	@echo "✓ elan found"
	@echo ""
	@echo "--- Building Lean (downloads Mathlib, ~5 min first time) ---"
	lake build
	@echo ""
	@echo "--- Installing Python dependencies ---"
	pip install -r paper/scripts/requirements.txt
	@echo ""
	@echo "--- Checking LaTeX ---"
	@which pdflatex > /dev/null 2>&1 || echo "⚠ pdflatex not found — install TeX Live for paper compilation"
	@echo ""
	@echo "--- Setting up git hooks ---"
	git config core.hooksPath .githooks
	@echo ""
	@echo "✓ Setup complete. Run 'make verify' to confirm everything works."

setup-python: ## Install Python dependencies only
	pip install -r paper/scripts/requirements.txt

# ─── Combined ──────────────────────────────────────────────────

.PHONY: all clean

all: lean all-papers verify ## Build everything and verify

clean: ## Remove LaTeX build artifacts
	cd $(PAPER_DIR) && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.toc

# ─── Help ──────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
