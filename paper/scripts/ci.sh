#!/bin/bash
# Consistency CI for the three-repo research program + papers.
# Run from the main repo root:  bash paper/scripts/ci.sh
# Local-first by design (GitHub Actions budget); also suitable as a pre-commit
# hook body. Fails fast on the first violated invariant.
set -uo pipefail
export PATH="$HOME/.elan/bin:/opt/homebrew/bin:$PATH"
MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
DASH="$MAIN/../dash-impossibility-lean"
OSTR="$MAIN/../ostrowski-impossibility"
fail() { echo "CI FAIL: $1"; exit 1; }

echo "== 1/5 claims.tex freshness"
python3 "$MAIN/paper/scripts/gen_claims.py" --check || fail "claims.tex stale"

echo "== 2/5 tex -> Lean reference lint (ALL papers, not a hand-picked list)"
# Glob every paper (excluding the generated claims.tex); a hardcoded list
# previously gave false coverage while 9 further drafts drifted (vet 2026-07-29).
LINT_FILES=()
for f in "$MAIN"/paper/*.tex; do
  [ "$(basename "$f")" = "claims.tex" ] && continue
  LINT_FILES+=("$f")
done
python3 "$MAIN/paper/scripts/lint_tex_lean_refs.py" "${LINT_FILES[@]}" \
  || fail "stale Lean references in papers"

echo "== 3/5 lake build x3"
(cd "$MAIN" && lake build >/dev/null 2>&1) || fail "main lake build"
(cd "$DASH" && lake build >/dev/null 2>&1) || fail "dash lake build"
(cd "$OSTR" && lake build >/dev/null 2>&1) || fail "ostrowski lake build"

echo "== 4/5 Tier-A spine axiom audit"
OUT="$(cd "$MAIN" && lake env lean paper/scripts/CheckAxioms.lean 2>&1)" \
  || { echo "$OUT"; fail "CheckAxioms.lean did not elaborate"; }
echo "$OUT" | grep -q "gbdtWorld\|gbdtAxioms\|ofReduceBool" \
  && { echo "$OUT"; fail "a spine theorem depends on a custom axiom or ofReduceBool"; }

echo "== 5/5 sorry audit"
for d in "$MAIN/UniversalImpossibility" "$DASH/DASHImpossibility" "$OSTR/OstrowskiImpossibility"; do
  grep -rn '^\s*sorry\s*$' "$d" --include='*.lean' && fail "sorry in $d"
done

echo "CI PASS"
