#!/usr/bin/env bash
# Build both NeurIPS (anonymous) and arXiv (preprint) versions of the paper.
# Usage: ./build.sh
# Output: submission/neurips.pdf, submission/arxiv.pdf, submission/supplement.pdf

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p submission

echo "=== Building NeurIPS submission (anonymous) ==="
# Ensure anonymous mode
sed -i.bak \
  -e 's|^% \(\\usepackage{neurips_2026}\)$|\1|' \
  -e 's|^\(\\usepackage\[preprint\]{neurips_2026}\)$|% \1|' \
  main.tex
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
cp main.pdf submission/neurips.pdf
echo "  -> submission/neurips.pdf ($(pdfinfo submission/neurips.pdf | grep Pages))"

echo "=== Building arXiv preprint (with authors) ==="
# Switch to preprint mode and uncomment author block
sed -i.bak \
  -e 's|^\(\\usepackage{neurips_2026}\)$|% \1|' \
  -e 's|^% \(\\usepackage\[preprint\]{neurips_2026}\)$|\1|' \
  -e '/^% Author block hidden/,/^\\author{Anonymous Authors}$/c\
\\author{\
  Drake Caraker\\thanks{Corresponding author: \\texttt{drakecaraker@gmail.com}} \\quad\
  Bryan Arnold \\quad\
  David Rhoads \\\\[4pt]\
  \\textit{Independent Researchers}\
}' \
  main.tex
# Restore real GitHub URLs for arXiv
sed -i.bak \
  's|\\url{.\[anonymized for review\]}|\\url{https://github.com/DrakeCaraker/dash-impossibility-lean}|g' \
  main.tex
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
cp main.pdf submission/arxiv.pdf
echo "  -> submission/arxiv.pdf ($(pdfinfo submission/arxiv.pdf | grep Pages))"

echo "=== Building supplement ==="
pdflatex -interaction=nonstopmode supplement.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode supplement.tex > /dev/null 2>&1
cp supplement.pdf submission/supplement.pdf
echo "  -> submission/supplement.pdf ($(pdfinfo submission/supplement.pdf | grep Pages))"

echo "=== Restoring main.tex to NeurIPS (anonymous) version ==="
git checkout main.tex
rm -f main.tex.bak

echo ""
echo "Done. All outputs in submission/"
ls -lh submission/*.pdf
