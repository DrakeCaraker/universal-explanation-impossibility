#!/usr/bin/env python3
"""Flag hardcoded repo-scale counts in papers that should be \\Claim* macros.

Motivation: count drift recurred repeatedly (25/72/463/491/100/200/1096 ...),
and several instances hid behind LaTeX non-breaking spaces (`491~theorems`) so
space-based greps missed them. This linter catches any 2+digit number
immediately followed (via space or `~`) by a count unit, or sitting in a
count-labeled table cell, unless it is a macro expansion or explicitly
allowlisted.

A macro use (`\\ClaimMainTheorems~theorems`) has a letter/brace before the unit,
never a literal digit, so it never matches. Small per-item behavioral counts
(`1 theorem`, `4 axioms`, `6 axioms`) are single-digit and never match.

Allowlist: paper/scripts/lint_counts_allowlist.txt (one substring per line) for
legitimate hardcoded counts (e.g. the 21-theorem impossibility atlas, a year).
Exit 1 on any un-allowlisted hit.
"""
import re, sys
from pathlib import Path

ALLOW = Path(__file__).parent / "lint_counts_allowlist.txt"
allow = [l.strip() for l in ALLOW.read_text().splitlines()
         if l.strip() and not l.startswith("#")] if ALLOW.exists() else []

UNIT = r"(?:theorems?|lemmas?|axioms?|files?|sorry)"
# 2+ digit number, optional thousands comma/latex comma, then ~/space, then unit
PROSE = re.compile(rf"\b\d[\d,]*(?:\{{,\}})?\d[~ ]{UNIT}\b", re.I)
# count-labeled table cell: "Files & 104 &", "Theorems ... & 491 &", "Axioms & 25 &"
CELL = re.compile(r"\b(?:Files|Theorems?|Lemmas?|Axioms?)\b[^\n&]*&\s*\d{2,}\s*&", re.I)


def check(path: Path):
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for m in list(PROSE.finditer(line)) + list(CELL.finditer(line)):
            frag = m.group(0)
            if any(a in line for a in allow):
                continue
            hits.append((i, frag.strip()))
    return hits


def main(paths):
    bad = False
    for p in map(Path, paths):
        if not p.exists() or p.name == "claims.tex":
            continue
        hits = check(p)
        for i, frag in hits:
            print(f"{p.name}:{i}: hardcoded count '{frag}' — use a \\Claim* macro "
                  f"or allowlist it")
        bad |= bool(hits)
        if not hits:
            print(f"OK: {p.name}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
