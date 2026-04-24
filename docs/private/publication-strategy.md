# Publication Strategy: The Limits of Explanation

*April 2026 — Updated after full peer review, axiom reduction, and experimental program*

---

## Material Inventory

| Repo | Lean | Papers | Experiments |
|------|------|--------|-------------|
| **Universal** | 100 files, 491 theorems, 25 axioms | Monograph (3,990 lines) + Nature (727 lines) + JMLR + NeurIPS + PNAS | 67 result JSONs |
| **DASH** | 58 files, 357 theorems, 6 axioms | Definitive (3,996 lines) + NeurIPS (504 lines) + JMLR + Supplement | Experiments via dash-shap |
| **Ostrowski** | 18 files, 163 theorems, 12 axioms | FoP (1,761 lines) + Monograph (1,390 lines) | Via universal |
| **Combined** | 176 files, 1,011 theorems, 43 axioms, 0 sorry | | |

---

## Independent Publishable Units

The material contains at least **7 genuinely independent stories**, each targeting a different audience:

### Unit 1: The Universal Impossibility (broad science)
**Story:** Eight fields, one impossibility, one resolution. The η law (R²=0.957) predicts instability cross-domain. Three predictions confirmed, two falsified.
**Audience:** Scientists across all fields; regulators; general public.
**Novel content:** The unification itself; the η law; Noether counting; 8-panel dose-response.
**Current draft:** `nature_article.tex` (727 lines, 11 pages, ready).

### Unit 2: The Attribution Impossibility (ML theory)
**Story:** No SHAP ranking is faithful, stable, and complete under collinearity. Design space has exactly two families. DASH is Pareto-optimal. Quantitative bilemma extends to continuous H. Nonparametric diagnostic outperforms Gaussian formula.
**Audience:** ML researchers, XAI practitioners.
**Novel content:** Design space theorem; DASH optimality; quantitative bilemma; coverage conflict diagnostic; bimodality validation; clinical reversal (45%).
**Current draft:** `main_definitive.tex` (3,996 lines) + `main.tex` (504 lines, NeurIPS format).

### Unit 3: The Physics Application (mathematical physics)
**Story:** Ostrowski's classification creates a spacetime bilemma. Adelic resolution is forced and unique. Enrichment stack depth ≥ 3.
**Audience:** Mathematical physicists; foundations of physics community.
**Novel content:** Ostrowski-based impossibility; Freund-Witten formalization; tightness classification.
**Current draft:** `fop-submission.tex` (1,761 lines, 26 pages, **submitted**).

### Unit 4: The Formalization (formal methods)
**Story:** 1,011 theorems, 43 axioms, 0 sorry. Axiom reduction from 161 to 43 (73%). All ML instances constructive. Arrow proved from scratch. Ostrowski bridged to Mathlib.
**Audience:** Formal verification community; proof assistant users.
**Novel content:** The formalization itself; the axiom reduction methodology; constructive instances pattern.
**Current draft:** Spread across all three repos. No standalone paper yet.

### Unit 5: The Clinical/Regulatory Impact (AI policy)
**Story:** 45% of credit applicants receive different explanation categories. SHAP audits are unreliable. EU AI Act demands the impossible. DASH provides compliant alternative.
**Audience:** Regulators; AI ethics researchers; policymakers.
**Novel content:** Per-individual reversal rates; regulatory mapping; liability analysis.
**Current draft:** Sections in DASH paper + Nature regulatory comment (`nature_comment_regulatory.tex`).

### Unit 6: The Diagnostic Tool (ML practice)
**Story:** Seven lines of code outperform the Gaussian flip formula (Spearman 0.96 vs 0.46). Coverage conflict + minority fraction = nonparametric stability diagnostic. No distributional assumptions.
**Audience:** ML practitioners who use SHAP.
**Novel content:** The 7-line tool; the comparison; the bimodality validation.
**Current draft:** Section in DASH paper. No standalone paper.

### Unit 7: The Representation Theory Connection (mathematics)
**Story:** η = dim(V^G)/dim(V) connects explanation impossibility to classical invariant theory. Bridge theorems to Fisher, Noether, EM, QEC. Compatibility complex characterizes tightness.
**Audience:** Mathematicians; representation theorists.
**Novel content:** η law derivation; bridge theorems; compatibility complex; NOT sheaf-theoretic (new obstruction level).
**Current draft:** Sections in monograph. No standalone paper.

---

## Recommended Strategy

### Tier 1 — Submit Now (content ready)

| # | Paper | Venue | Pages | Status | Action |
|---|-------|-------|-------|--------|--------|
| 1 | **"The Limits of Explanation"** | Nature | 11 | Ready | Submit + simultaneous arXiv monograph |
| 2 | **arXiv monograph** | arXiv | 104 | Ready | Post day of Nature submission |
| 3 | **FoP physics paper** | Foundations of Physics | 26 | **Submitted** | Wait for reviews |

### Tier 2 — Submit by May 6 (NeurIPS deadline)

| # | Paper | Venue | Pages | Status | Action |
|---|-------|-------|-------|--------|--------|
| 4 | **"The Attribution Impossibility"** | NeurIPS 2026 | 10 + supplement | 70% ready | Update `main.tex` with new results; compress |

**Why NeurIPS over JMLR:** Visibility. NeurIPS has 15,000+ attendees. The diagnostic tool + clinical reversal + bimodality validation is a compelling 10-page story that NeurIPS values (theory + experiments + tool). JMLR is the fallback if NeurIPS rejects.

**What goes in 10 pages:** Core impossibility → design space → quantitative bilemma → nonparametric diagnostic (7 lines!) → bimodality validation → clinical reversal (45%) → DASH resolution → Lean verification (357 theorems, 6 axioms).

**What goes in supplement (unlimited):** All proofs, all extended experiments, regulatory mapping, fairness connection, bridge theorems, topological characterization.

### Tier 3 — Submit within 3 months

| # | Paper | Venue | Pages | Status | Action |
|---|-------|-------|-------|--------|--------|
| 5 | **"Formally Verified Impossibility Across Eight Sciences"** | ITP 2026 or CPP 2027 | 15 | Not started | Extract from monograph + Lean code |
| 6 | **"Explanation Instability in Clinical AI"** | Nature Machine Intelligence | 8 | 50% ready | Extract clinical reversal + regulatory from DASH paper |

### Tier 4 — If Time/Interest Permits

| # | Paper | Venue | Pages | Status | Action |
|---|-------|-------|-------|--------|--------|
| 7 | **"The Invariance Counting Principle"** | Notices of AMS or Math Intelligencer | 10 | Not started | Expository: η law + Noether + bridge theorems |
| 8 | **Full attribution paper** | JMLR | 80+ | Ready | Submit if NeurIPS rejects (or as companion) |

---

## Submission Timeline

```
April 17 (today):
  ├── Post arXiv monograph (priority claim)
  └── Submit Nature article (with cover letter)

May 4:
  └── NeurIPS abstract deadline
      (requires: update main.tex with new results)

May 6:
  └── NeurIPS paper deadline
      (requires: compressed 10-page paper + supplement)

June:
  ├── If Nature desk-rejects → resubmit to NMI
  └── Start ITP/CPP formalization paper

August:
  ├── NeurIPS decisions
  ├── FoP reviews expected
  └── If NeurIPS rejects → submit JMLR full version

September:
  └── Submit NMI clinical paper (if not already via Nature path)
```

---

## Key Strategic Decisions

### 1. Nature vs Nature Machine Intelligence

**Nature** (IF ~70) — risky but high reward. The unification across 8 fields is Nature-scale. The elementary proof is the main risk.

**NMI** (IF ~18) — safer, still high impact. Perfect for the ML-focused story. Can include more technical detail.

**Recommendation:** Submit to Nature first. If desk-rejected within 2 weeks, immediately redirect to NMI with minor reframing (emphasize ML implications over cross-domain breadth).

### 2. NeurIPS vs JMLR for the attribution paper

**NeurIPS** — 10 pages, competitive (25% acceptance), high visibility, deadline May 6.

**JMLR** — unlimited pages, guaranteed review, slower (6-12 months), less visibility.

**Recommendation:** Submit NeurIPS first (it's a compressed extraction, not a new paper). If rejected, submit the full JMLR version. NeurIPS and JMLR are not mutually exclusive — NeurIPS is the 10-page version, JMLR is the 80-page version. Different papers.

### 3. Do we need a standalone formalization paper?

**Yes.** The Lean community would value:
- 1,011 theorems across 3 repos
- Axiom reduction story (161 → 43)
- Constructive instance methodology
- Arrow proved from scratch in Lean 4

**Venue:** ITP (Interactive Theorem Proving) 2026 or CPP (Certified Programs and Proofs) 2027.

### 4. Does the MI v2 result change anything?

**If circuits are non-unique (ρ < 0.3):** Add to Nature as a headline result. Delay Nature submission by 1 week to incorporate. This would be transformative for AI safety.

**If circuits are stable (ρ > 0.8):** Add as honest negative to monograph. Don't delay Nature. The paper is strong without it.

**If inconclusive:** Don't delay. Mention as "in progress."

### 5. Dual submission policy

Nature allows simultaneous arXiv posting. NeurIPS allows arXiv. JMLR allows arXiv. FoP allows arXiv. No conflicts.

The key constraint: the Nature paper and NeurIPS paper must be SUFFICIENTLY DIFFERENT that they're not the same paper at two venues. They are: Nature = unification across 8 fields; NeurIPS = attribution-specific with diagnostic tool. Different theorems, different experiments, different audiences.

---

## What's NOT Worth a Separate Paper

- **The drug discovery application** — an honest negative, interesting as a section in the monograph but not standalone.
- **The compatibility complex** — too small for a standalone paper; better as a section in the JMLR version.
- **The enrichment-as-abstraction / cognitive science conjecture** — speculative, no evidence.
- **The SAGE algorithm** — already published as part of the monograph; not novel enough alone.
- **The MI v1 (LoRA) result** — inconclusive, superseded by v2.

---

## The Pitch for Each Paper

### Nature
*"Eight scientific fields independently discovered the same resolution to the same impossibility. We prove why, quantify the information loss, and verify everything in Lean."*

### NeurIPS
*"No SHAP ranking is faithful, stable, and complete under collinearity — and here's a 7-line diagnostic that outperforms the Gaussian formula, plus the Pareto-optimal fix."*

### FoP (submitted)
*"Ostrowski's classification creates an impossibility for spacetime description. The adelic resolution is forced and unique."*

### ITP/CPP
*"1,011 theorems from 43 axioms with 0 sorry: a formally verified impossibility theorem spanning 8 scientific domains."*

### NMI
*"45% of loan applicants receive different AI explanations depending on the training seed. The EU AI Act demands the impossible — here's what's achievable."*

### JMLR (if NeurIPS rejects)
*"The complete theory of attribution impossibility: from the design space theorem to DASH optimality, with 357 Lean theorems and 15 empirical validations."*
