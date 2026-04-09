#!/usr/bin/env bash
# Build all universal paper versions (monograph, JMLR, NeurIPS).
# Usage: ./build.sh
# Output: submission/monograph.pdf, submission/jmlr.pdf, submission/neurips.pdf

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p submission

echo "=== Building monograph / arXiv preprint ==="
pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex > /dev/null 2>&1
bibtex universal_impossibility_monograph > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_monograph.tex > /dev/null 2>&1
cp universal_impossibility_monograph.pdf submission/monograph.pdf
echo "  -> submission/monograph.pdf ($(pdfinfo submission/monograph.pdf | grep Pages))"

echo "=== Building JMLR submission ==="
pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex > /dev/null 2>&1
bibtex universal_impossibility_jmlr > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_jmlr.tex > /dev/null 2>&1
cp universal_impossibility_jmlr.pdf submission/jmlr.pdf
echo "  -> submission/jmlr.pdf ($(pdfinfo submission/jmlr.pdf | grep Pages))"

echo "=== Building NeurIPS 2026 version ==="
pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex > /dev/null 2>&1
bibtex universal_impossibility_neurips > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode universal_impossibility_neurips.tex > /dev/null 2>&1
cp universal_impossibility_neurips.pdf submission/neurips.pdf
echo "  -> submission/neurips.pdf ($(pdfinfo submission/neurips.pdf | grep Pages))"

echo ""
echo "Done. All outputs in submission/"
ls -lh submission/*.pdf
