# Safe Commit

Commit with automatic count verification. Prevents stale paper-code claims.

## Steps

1. Run the verification block and display results:
   ```bash
   grep -c "^theorem\|^lemma" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "theorems+lemmas:", s}'
   grep -c "^axiom" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "axioms:", s}'
   grep -rc "sorry" DASHImpossibility/*.lean | awk -F: '{s+=$2} END {print "sorry:", s}'
   ls DASHImpossibility/*.lean | wc -l | awk '{print "files:", $1}'
   ```

2. If any Lean files are staged, verify `lake build` succeeds.

3. If any `.tex` files are staged, check that theorem/axiom counts in the paper text match the actual counts from step 1. Search for patterns like "190 theorems", "17 axioms", "0 sorry" in staged `.tex` files and flag mismatches.

4. Show `git status` and `git diff --staged --stat`.

5. Draft a commit message following the repository's conventional commit style (feat/fix/docs/chore).

6. Stage relevant files and commit. Do NOT use `git add -A` — add specific files.
