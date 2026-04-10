# Final Review + Sync — Last Pass Before arXiv

**Goal**: One exhaustive review of the monograph (the definitive
version), fix everything found, sync all other versions, compile
all, commit, done.

**Principle**: This is the LAST pass. After this, the paper is
posted. No more revisions. Fix or flag, then ship.

**Model**: Opus for the review. Sonnet for sync and compilation.

---

## Phase 1: Exhaustive Monograph Review [Opus]

Read paper/universal_impossibility_monograph.tex from first
line to last line. For EACH of the following, report PASS/FLAG:

### A. Front Matter
- [ ] Title: accurate, not overclaiming
- [ ] Authors: correct names, affiliations reasonable
- [ ] Abstract: ≤250 words, covers impossibility + instances
      + resolution + Lean stats + empirical illustration
- [ ] Table of contents: sections in logical order

### B. Introduction
- [ ] Opens domain-general (not ML-specific)
- [ ] "Faithful, stable, decisive — pick two" appears
- [ ] All 9 instances mentioned
- [ ] "No escape by switching methods" framing present
- [ ] Companion paper referenced (with arXiv placeholder)
- [ ] Contribution list: accurate, not overclaiming
- [ ] "The contribution is the framework, not the proof"
- [ ] Related work positioned correctly (Bilodeau subsumed,
      Rao complementary, Okasha different)

### C. Framework Section
- [ ] Definitions match Lean code EXACTLY (check faithful,
      stable, decisive against ExplanationSystem.lean)
- [ ] Theorem statement matches Lean
- [ ] Proof sketch matches the 4-step Lean proof
- [ ] Tightness: stated in main body, counterexamples described
- [ ] Necessity: qualified for fully specified systems
- [ ] Axiom substitution: referenced, key finding stated
- [ ] Query-relative: paragraph present
- [ ] Quantitative remark: present in discussion
- [ ] incompatible_irrefl: justified

### D. Instances Section
- [ ] All 9 instances present with 6-tuples
- [ ] Derived vs axiomatized clearly distinguished
- [ ] Causal instance: notes Markov equivalence is derived
- [ ] MI instance: generous framing, cites Meloux correctly
      (authors: Méloux, Maniu, Portet, Peyrard)
- [ ] Token Citation (not "LLM Self-Explanation")
- [ ] Cross-instance table: 9 rows, all numbers match JSONs
- [ ] Attention: 19.9% as headline, 60% footnoted
- [ ] GradCAM: 78.8% prediction agreement noted
- [ ] Full retraining: 4/20 below 80% noted
- [ ] Token citation control: 0.0% (not 31.5%)

### E. Resolution Section
- [ ] G-invariant framework described
- [ ] Hunt-Stein: "structurally analogous" (not "instantiation")
- [ ] Pareto-optimality: NOT claimed (or qualified as
      "proved for attribution instance in companion paper")
- [ ] Instance-specific resolutions table
- [ ] DASH as prototype
- [ ] Information-theoretic DPI proposition present
- [ ] Escape routes table present
- [ ] Practitioner decision table present (or in appendix)

### F. Ubiquity Section
- [ ] Dimensional argument: dim(Θ) > dim(Y)
- [ ] Neural network example (25M params, 1000 outputs)

### G. Discussion
- [ ] Three-tier cross-domain claims (not flat list)
- [ ] QM, legal, cryptography REMOVED
- [ ] Van Fraassen connection present
- [ ] Structural realism: "suggestive" (not "formalizes")
- [ ] Arrow: structural analogy (not impact comparison)
- [ ] Quine-Duhem: "contrastive underdetermination" +
      prior work (Laudan, Dorling, Howson)
- [ ] "Obvious" objection: 3 responses
- [ ] Regulatory: SR 11-7 with exact sections (III.A, III.B, IV)
- [ ] Experiments: "illustrative" not "validation"
- [ ] Limitations section: honest about experimental scope,
      perturbation methodology, axiomatized instances

### H. Lean Formalization Section
- [ ] Counts: 75 files, 351 theorems, 72 axioms, 0 sorry
- [ ] Axiom stratification table
- [ ] Proof status transparency
- [ ] Core theorem: 0 axiom dependencies stated
- [ ] Derived causal instance highlighted

### I. Appendices
- [ ] Arrow comparison appendix present and accurate
- [ ] Lean code listings: match actual files
- [ ] Experiment methodology: matches scripts
- [ ] Proof sketches: match docs/proof-sketches/

### J. Bibliography
- [ ] Meloux: correct authors (Méloux, Maniu, Portet, Peyrard),
      arXiv:2502.20914
- [ ] Paulo & Belrose (not Bricken): arXiv:2501.16615, ICLR 2026
- [ ] Laudan & Leplin 1991: present and cited
- [ ] Dorling 1979: present and cited
- [ ] Howson & Urbach 2006: present and cited
- [ ] Worrall 1989: present and cited
- [ ] Arrow 1951: present and cited
- [ ] No orphan entries (every bib entry cited somewhere)
- [ ] No missing entries (every \cite has a bib entry)

### K. Numbers Consistency
- [ ] Lean counts (75/351/72/0) consistent across abstract,
      intro, Lean section, and any other mention
- [ ] Experiment numbers match result JSONs:
      - Attention perturbation: 60.0%
      - Attention full retraining: 19.9%
      - Counterfactual: 23.5%
      - Concept probe: 0.90
      - Model selection: 80%
      - GradCAM: 9.6%
      - Token citation: 34.5%
- [ ] Instance count "nine" consistent throughout
      (no stale "six" or "eight")

---

## Phase 2: Fix Everything Found [Opus/Sonnet]

For each FLAG from Phase 1:
- If fixable in the monograph: fix it
- If unfixable (e.g., requires new experiments): note as
  known limitation, ensure it's in the Limitations section

---

## Phase 3: Sync All Versions [Sonnet]

After monograph is clean, verify each other version has
the critical elements. For EACH of these files, check and
fix if needed:

### universal_impossibility_pnas.tex (6pp)
- [ ] Lean counts: 75/351/72/0
- [ ] Instance count: nine
- [ ] Attention: 19.9% headline
- [ ] Token Citation (not LLM)
- [ ] Hunt-Stein: analogous
- [ ] Necessity: qualified
- [ ] Quine-Duhem: contrastive underdetermination
- [ ] Experiments: illustrative

### universal_impossibility_neurips.tex (10pp)
- [ ] Same checks as PNAS
- [ ] Companion paper arXiv placeholder present
- [ ] "No escape" framing in intro
- [ ] Tightness in main body

### universal_impossibility_jmlr.tex (31pp)
- [ ] Lean counts: 75/351/72/0
- [ ] Instance count: nine
- [ ] Attention: 19.9% headline

### universal_impossibility_pnas_si.tex (12pp)
- [ ] Lean counts match main PNAS paper
- [ ] S-prefixed sections
- [ ] Token Citation (not LLM)
- [ ] Practitioner decision table present

---

## Phase 4: Final Compilation + Verification [Sonnet]

### 4.1: Lean build
```bash
lake build
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

### 4.2: Compile all 6 documents
```bash
cd paper
for f in universal_impossibility_monograph \
         universal_impossibility_pnas \
         universal_impossibility_pnas_si \
         universal_impossibility_neurips \
         universal_impossibility_jmlr; do
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null 2>&1
  bibtex "$f" > /dev/null 2>&1
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null 2>&1
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null 2>&1
  echo "$f: $(pdfinfo "$f.pdf" 2>/dev/null | grep Pages | awk '{print $2}')pp"
done
```

Expected: monograph ~45pp, PNAS 6pp, SI 12pp, NeurIPS 10pp, JMLR ~31pp.

### 4.3: Commit and push
