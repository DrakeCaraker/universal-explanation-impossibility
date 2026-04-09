#!/bin/bash
# Prepare arXiv/JMLR version: uncomment authors, fill URLs, compile.
# Run from paper/ directory.
set -e

cd "$(dirname "$0")/.."

echo "=== Preparing arXiv/JMLR version ==="

# For JMLR version (main_jmlr.tex):
if [ -f main_jmlr.tex ]; then
    sed -i '' 's|\\url{\\[anonymized for review\\]}|\\url{https://github.com/DrakeCaraker/dash-impossibility-lean}|g' main_jmlr.tex
    sed -i '' 's|\[anonymized for review\]|https://github.com/DrakeCaraker/dash-impossibility-lean|g' main_jmlr.tex
    echo "Updated URLs in main_jmlr.tex"
fi

# For NeurIPS preprint version:
if [ -f main.tex ]; then
    cp main.tex main_preprint.tex
    # Switch to preprint mode
    sed -i '' 's/\\usepackage{neurips_2026}/\\usepackage[preprint]{neurips_2026}/' main_preprint.tex
    # Uncomment author block
    sed -i '' 's/^% \\author{/\\author{/' main_preprint.tex
    sed -i '' 's/^%   Drake Caraker/  Drake Caraker/' main_preprint.tex
    sed -i '' 's/^%   Bryan Arnold/  Bryan Arnold/' main_preprint.tex
    sed -i '' 's/^%   David Rhoads/  David Rhoads/' main_preprint.tex
    sed -i '' 's/^%   \\textit{/  \\textit{/' main_preprint.tex
    sed -i '' 's/^% }/}/' main_preprint.tex
    # Remove anonymous author line
    sed -i '' '/\\author{Anonymous Authors}/d' main_preprint.tex
    # Fill URLs
    sed -i '' 's|\[anonymized for review\]|https://github.com/DrakeCaraker/dash-impossibility-lean|g' main_preprint.tex
    echo "Created main_preprint.tex"
fi

echo "=== Done. Compile with: ==="
echo "  pdflatex main_jmlr.tex && bibtex main_jmlr && pdflatex main_jmlr.tex && pdflatex main_jmlr.tex"
echo "  # or for NeurIPS preprint:"
echo "  pdflatex main_preprint.tex && bibtex main_preprint && pdflatex main_preprint.tex && pdflatex main_preprint.tex"
echo "=== Upload to arXiv under: cs.LG, cs.AI, stat.ML, cs.LO ==="
# Timing: consider posting 1-2 weeks before FAccT 2026 or major XAI workshop
