# Attribution Impossibility — Build System
# Usage: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Lean ──────────────────────────────────────────────────────

.PHONY: lean
lean: ## Build all Lean 4 files (~5 min)
	lake build

# ─── Papers ────────────────────────────────────────────────────

PAPER_DIR := paper

.PHONY: neurips jmlr definitive paper

neurips: ## Compile NeurIPS version (10 pages)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex
	@echo "NeurIPS: $$(pdfinfo $(PAPER_DIR)/main.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

jmlr: ## Compile JMLR version (50 pages)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode main_jmlr.tex && \
		bibtex main_jmlr && pdflatex -interaction=nonstopmode main_jmlr.tex && \
		pdflatex -interaction=nonstopmode main_jmlr.tex
	@echo "JMLR: $$(pdfinfo $(PAPER_DIR)/main_jmlr.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

definitive: ## Compile definitive version (59 pages)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode main_definitive.tex && \
		bibtex main_definitive && pdflatex -interaction=nonstopmode main_definitive.tex && \
		pdflatex -interaction=nonstopmode main_definitive.tex
	@echo "Definitive: $$(pdfinfo $(PAPER_DIR)/main_definitive.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

supplement: ## Compile supplement (76 pages)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode supplement.tex && \
		bibtex supplement && pdflatex -interaction=nonstopmode supplement.tex && \
		pdflatex -interaction=nonstopmode supplement.tex
	@echo "Supplement: $$(pdfinfo $(PAPER_DIR)/supplement.pdf 2>/dev/null | grep Pages | awk '{print $$2}') pages"

paper: neurips jmlr definitive supplement ## Compile all paper versions

arxiv: ## Prepare arXiv preprint
	cd $(PAPER_DIR) && bash scripts/prepare_arxiv.sh

# ─── Verification ──────────────────────────────────────────────

.PHONY: verify counts

counts: ## Show current theorem/axiom/sorry/file counts
	@echo "Theorems+lemmas: $$(grep -c '^theorem\|^lemma' UniversalImpossibility/*.lean | awk -F: '{s+=$$2} END {print s}')"
	@echo "Axioms:          $$(grep -c '^axiom' UniversalImpossibility/*.lean | awk -F: '{s+=$$2} END {print s}')"
	@echo "Sorry:           $$(grep -rc 'sorry' UniversalImpossibility/*.lean | awk -F: '{s+=$$2} END {print s}')"
	@echo "Files:           $$(ls UniversalImpossibility/*.lean | wc -l | tr -d ' ')"

verify: counts ## Verify Lean builds + counts are consistent
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

.PHONY: validate validate-quick experiments

# Core validation scripts (run these for co-author review)
validate: ## Run 3 key validation experiments (~5 min)
	python3 $(SCRIPTS_DIR)/validate_ratio.py
	python3 $(SCRIPTS_DIR)/cross_implementation_validation.py
	python3 $(SCRIPTS_DIR)/snr_calibration.py

# All experiment scripts
experiments: ## Run ALL experiment scripts (~30 min)
	@for script in $(SCRIPTS_DIR)/*.py; do \
		echo "=== Running $$(basename $$script) ==="; \
		python3 "$$script" || echo "FAILED: $$script"; \
		echo ""; \
	done

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

all: lean paper verify ## Build everything and verify

clean: ## Remove LaTeX build artifacts
	cd $(PAPER_DIR) && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.toc

# ─── Help ──────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
