#!/usr/bin/env python3
"""Lean proof complexity distribution analysis.

For each .lean file, parse theorems/lemmas, count tactic lines between
`:= by` and the next declaration. Categorize:
- 1-3 lines: "trivial"
- 4-10 lines: "short"
- 11-30 lines: "moderate"
- 31+ lines: "substantive"
"""

import json, os, re

lean_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "UniversalImpossibility")

results = {
    "summary": {},
    "distribution": {"trivial_1_3": 0, "short_4_10": 0, "moderate_11_30": 0, "substantive_31_plus": 0},
    "files": {},
    "examples": {"trivial": [], "short": [], "moderate": [], "substantive": []}
}

total_theorems = 0
all_lengths = []

for fname in sorted(os.listdir(lean_dir)):
    if not fname.endswith(".lean"):
        continue
    fpath = os.path.join(lean_dir, fname)
    with open(fpath) as f:
        lines = f.readlines()

    file_theorems = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(theorem|lemma)\s+(\S+)', line)
        if m:
            kind = m.group(1)
            name = m.group(2)
            proof_start = None
            j = i
            while j < min(i + 20, len(lines)):
                if ':= by' in lines[j]:
                    proof_start = j
                    break
                if j > i and lines[j].strip().startswith(':= by'):
                    proof_start = j
                    break
                j += 1

            if proof_start is not None:
                proof_lines = 0
                k = proof_start + 1
                while k < len(lines):
                    stripped = lines[k].strip()
                    if re.match(r'^(theorem|lemma|def |noncomputable|instance|class|structure|section|namespace|end |--|#|open |@\[|abbrev|axiom|variable)', lines[k]) and stripped:
                        break
                    if stripped == '' and k + 1 < len(lines) and re.match(r'^(theorem|lemma|def |noncomputable|instance|class|structure|section|namespace|end )', lines[k + 1]):
                        break
                    if stripped:
                        proof_lines += 1
                    k += 1

                file_theorems.append({"name": name, "file": fname, "kind": kind, "lines": proof_lines})
                all_lengths.append(proof_lines)
                total_theorems += 1

                if proof_lines <= 3:
                    results["distribution"]["trivial_1_3"] += 1
                    if len(results["examples"]["trivial"]) < 5:
                        results["examples"]["trivial"].append({"name": name, "file": fname, "lines": proof_lines})
                elif proof_lines <= 10:
                    results["distribution"]["short_4_10"] += 1
                    if len(results["examples"]["short"]) < 5:
                        results["examples"]["short"].append({"name": name, "file": fname, "lines": proof_lines})
                elif proof_lines <= 30:
                    results["distribution"]["moderate_11_30"] += 1
                    if len(results["examples"]["moderate"]) < 5:
                        results["examples"]["moderate"].append({"name": name, "file": fname, "lines": proof_lines})
                else:
                    results["distribution"]["substantive_31_plus"] += 1
                    if len(results["examples"]["substantive"]) < 5:
                        results["examples"]["substantive"].append({"name": name, "file": fname, "lines": proof_lines})
            i = (proof_start + 1) if proof_start else (i + 1)
        else:
            i += 1

    if file_theorems:
        results["files"][fname] = {
            "theorem_count": len(file_theorems),
            "theorems": file_theorems
        }

results["summary"] = {
    "total_theorems_parsed": total_theorems,
    "mean_proof_length": round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else 0,
    "median_proof_length": round(sorted(all_lengths)[len(all_lengths) // 2], 2) if all_lengths else 0,
    "max_proof_length": max(all_lengths) if all_lengths else 0,
    "min_proof_length": min(all_lengths) if all_lengths else 0,
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_lean_complexity.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)

print(f"Parsed {total_theorems} theorems/lemmas")
print(f"Distribution: {results['distribution']}")
print(f"Mean: {results['summary']['mean_proof_length']}, Median: {results['summary']['median_proof_length']}")
print(f"Max: {results['summary']['max_proof_length']}, Min: {results['summary']['min_proof_length']}")
print(f"Saved to {outpath}")
