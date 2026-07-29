#!/usr/bin/env python3
"""Lint LaTeX files for references to Lean identifiers/files that do not exist
in the repositories. This makes the F1 failure class (papers citing deleted
theorems/files) structurally impossible.

Checks two classes extracted from \\texttt{...} spans:
  1. *.lean filenames            -> must exist in one of the three repos
  2. snake_case identifiers      -> must appear as a word in some .lean file
Everything else in \\texttt{} is ignored (shell commands, paths, JSON keys).

Exit 1 on any miss. Allowlist: paper/scripts/lint_allowlist.txt (one token per
line) for deliberate exceptions (e.g., prose about retired names).
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve()
MAIN = HERE.parents[2]
REPO_ROOTS = [
    MAIN / "UniversalImpossibility",
    MAIN.parent / "dash-impossibility-lean" / "DASHImpossibility",
    MAIN.parent / "ostrowski-impossibility" / "OstrowskiImpossibility",
]
ALLOW = HERE.parent / "lint_allowlist.txt"
allow = set(ALLOW.read_text().split()) if ALLOW.exists() else set()

lean_files = [p for root in REPO_ROOTS for p in root.rglob("*.lean")]
lean_names = {p.name for p in lean_files}
lean_text = "".join(p.read_text() for p in lean_files)

TT = re.compile(r"\\texttt\{([^{}]*)\}")
IDENT = re.compile(r"^[a-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")


def deescape(s: str) -> str:
    return (s.replace(r"\_", "_").replace(r"\-", "").replace(r"\{", "{")
             .replace(r"\}", "}").replace(r"\#", "#").replace("~", " ").strip())


def check_file(tex: Path):
    misses = []
    body = tex.read_text()
    for m in TT.finditer(body):
        tok = deescape(m.group(1))
        line = body[: m.start()].count("\n") + 1
        if tok in allow:
            continue
        if tok.endswith(".lean"):
            base = tok.split("/")[-1]
            if base not in lean_names:
                misses.append((line, tok, "file not found in any repo"))
        elif IDENT.match(tok):
            if not re.search(rf"\b{re.escape(tok)}\b", lean_text):
                misses.append((line, tok, "identifier not found in any repo"))
    return misses


def main(paths):
    bad = False
    for p in map(Path, paths):
        if not p.exists():
            print(f"SKIP (missing): {p}")
            continue
        misses = check_file(p)
        for line, tok, why in misses:
            print(f"{p.name}:{line}: {tok}  [{why}]")
        bad |= bool(misses)
        if not misses:
            print(f"OK: {p.name}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
