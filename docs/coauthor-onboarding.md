# Co-Author Onboarding: The Limits of Explanation

*Drake Caraker — April 2026*

---

## The One-Sentence Version

We proved that no explanation of an underspecified system can simultaneously be accurate, reproducible, and definitive — and this single theorem unifies known problems across eight scientific fields, from codon degeneracy in biology to gauge invariance in physics to SHAP instability in machine learning.

---

## The Five-Minute Version

### The Problem

You train an XGBoost model on the Breast Cancer dataset. You compute SHAP values. Feature #1 is "worst concave points." You retrain with a different random seed. Now Feature #1 is "worst perimeter." Same data. Same algorithm. Same accuracy. Different explanation.

This isn't a bug in SHAP. It's a theorem.

### The Theorem

Define three things practitioners want from an explanation:
- **Faithful**: the explanation doesn't contradict what the model actually computed
- **Stable**: equivalent models get the same explanation
- **Decisive**: the explanation commits to a definite answer (ranks all features)

**Theorem (4-line proof, Lean-verified):** If the system is underspecified (multiple models fit the data equally well but explain differently — the *Rashomon property*), then no explanation can be all three simultaneously.

The proof is elementary. The contribution is identifying the right definitions and showing they apply everywhere.

### The Unification

The same theorem — with the same 4-line proof — applies in eight scientific fields:

| Domain | "Models" | "Explanation" | The known problem |
|--------|---------|--------------|-------------------|
| **Biology** | Synonymous codons | Which codon | Codon degeneracy |
| **Physics** | Gauge configurations | Local field values | Gauge invariance |
| **Statistics** | DAGs in Markov class | Edge orientations | Markov equivalence |
| **Crystallography** | Phase assignments | Electron density | Phase problem |
| **Linguistics** | Parse trees | Attachment structure | PP-attachment ambiguity |
| **Mathematics** | Solutions of Ax=b | Which solution | Underdetermined systems |
| **Computer Science** | Database rows | Hidden columns | View-update problem |
| **Stat. Mechanics** | Microstates | Particle arrangement | Boltzmann counting |

Each field independently invented the same workaround: average over the equivalent configurations. Physicists call it "gauge-invariant observables." Statisticians call it "CPDAGs." We call it "DASH" for SHAP. It's all orbit averaging over a symmetry group — and we prove it's Pareto-optimal.

### The Quantitative Predictions

The framework isn't just qualitative. It generates three quantitative formulas, all confirmed:

1. **Noether counting**: For P features in g correlation groups, exactly g(g-1)/2 ranking facts are stable. Confirmed: 47-percentage-point bimodal gap, invariant across ρ = 0.50–0.99. Pre-registered.

2. **Gaussian flip rate**: flip(j,k) = Φ(-SNR) predicts per-pair instability with R² = 0.814 [0.791, 0.835] across 647 feature pairs from 5 datasets. Bootstrap-calibrated.

3. **Universal η law**: dim(V^G)/dim(V) predicts instability across 7 domains with R² = 0.957 (p = 1.4×10⁻⁴). Leave-one-out R² = 0.79. On all 16 domains: R² = 0.25 (bottleneck is group identification, not the law).

Two pre-registered extensions were falsified (phase transition location off by 10×, tradeoff bound violated). One was negative (molecular evolution adds nothing beyond biochemistry). Honest reporting.

### The Formalization

Everything is machine-verified in Lean 4:
- **Universal repo**: 107 files, 493 theorems, 83 axioms, 0 sorry
- **Attribution repo (DASH)**: 58 files, 357 theorems, 6 axioms, 0 sorry
- **Physics repo (Ostrowski)**: 18 files, 163 theorems, 12 axioms, 0 sorry

The core theorem requires **zero domain-specific axioms**. The 8 cross-domain instances are constructive (witnesses built by `decide`). The ML instances axiomatize the Rashomon property from cited literature.

---

## Progressive Disclosure

Read in order. Each section assumes you've read the previous ones.

### Level 1: The Papers (read these first)

| Paper | Venue | Pages | What it covers | Read time |
|-------|-------|-------|---------------|-----------|
| **Nature Article** | Nature (target) | 11 | The unification story: theorem + 8 domains + 3 predictions + resolution | 30 min |
| **Attribution Paper** | JMLR (target) | ~80 | SHAP-specific: design space theorem + DASH + 68% prevalence + regulatory | 2 hours |
| **Physics Paper** | Found. of Physics (submitted) | 26 | Bilemma + tightness + Ostrowski spacetime + enrichment stack | 1 hour |
| **arXiv Monograph** | arXiv | 104 | Everything above + 19 ML instances + 10 experiments + bridge theorems | Reference |

**Start with the Nature Article** (`paper/nature_article.tex`). It's 2,600 words of main text and tells the complete story.

Then read the attribution paper's executive summary (DASH repo, `paper/main_definitive.tex`, first 3 pages) for the ML-specific quantitative results.

The monograph is the comprehensive technical reference — consult it for specific questions, don't read cover-to-cover.

### Level 2: The Architecture

```
              The Universal Impossibility
              (ExplanationSystem.lean, 0 axioms)
                    |
        +-----------+-----------+
        |           |           |
   8 Cross-Domain   9 ML      Resolution
   (constructive)  (axiomatized) (G-invariant,
                                  Pareto-optimal)
        |           |           |
   Biology      Attribution   DASH
   Physics      Attention     CPDAG
   Statistics   Counterfact.  Ensemble probe
   Linguistics  Concept       ...
   Crystal.     Causal
   Math         Model sel.
   CS           Saliency
   Stat. mech.  LLM citation
                Mech. interp.
```

The core theorem lives in `ExplanationSystem.lean`. Everything else is instantiation, quantification, or resolution.

### Level 3: The Experiments

| Experiment | Result | Status |
|-----------|--------|--------|
| **8-domain dose-response** | All 7 empirical domains show predicted monotonic relationship (binomial p=0.008) | ✅ Published in monograph |
| **Noether counting** | 47pp bimodal gap, invariant across ρ, both XGBoost and Ridge | ✅ Pre-registered, confirmed |
| **Gaussian flip rate** | R²=0.814 [0.791,0.835] on 647 pairs, 5 datasets | ✅ Bootstrap-calibrated |
| **η law** | R²=0.957 for 7 well-characterized domains (LOO R²=0.79) | ✅ GoF: F(1,5)=38.78, p=0.002 |
| **SAGE algorithm** | Beats all baselines on ranking (ρ=0.952, LOO R²=0.686, p=0.002) | ✅ 6/6 stress tests passed |
| **Proportionality sensitivity** | 0/11,000 Pareto violations — two-family structure intact | ✅ Robust |
| **Codon null model** | ρ=1.0 trivially guaranteed by monotonicity | ✅ Calibration check, not validation |
| **Clinical decision reversal** | 45% of German Credit applicants get different explanation category | ✅ Ablation: 4 conditions × 3 models |
| **Drug discovery (prospective)** | Pearson failed (0 vs 23%). MI recovered (16% error). Jaccard (4% error). No bimodality. | ✅ Boundary condition identified |
| **Flip rate robustness** | Exact for Gaussian; 40%+ error for t(3) tails | ✅ 94.7% of real data is Gaussian |
| **MI v1 (LoRA on IOI)** | ρ=0.885 — circuits stable, but confounded by LoRA constraint | ⚠️ Inconclusive |
| **MI v2 (from scratch)** | Running on SageMaker — modular addition, 30 models from random init | 🔄 Pending |
| **Phase transition** | Sigmoid exists but r* ∈ [0.01, 0.12], not 1.0 | ❌ Falsified (honest) |
| **Tradeoff bound** | Max sum = 2.86, violating α+σ+δ ≤ 2 | ❌ Falsified (honest) |
| **Molecular evolution** | Character theory adds partial R²=0.0 | ❌ Negative (honest) |

### Level 4: Key Findings from Peer Review

The work went through 51 + 29 simulated adversarial reviewers across two rounds. Key findings that shaped the papers:

1. **The theorem is elementary.** Every adversarial reviewer flagged this. The defense: the theorem is the organizing principle, not the contribution. The contributions are the unification, the predictions, and the resolution.

2. **The η law on n=7 is fragile.** R²=0.957 on 7 points looks impressive but is vulnerable. We added: LOO R²=0.79, permutation p=0.010, full-16 R²=0.25 with honest scoping.

3. **"Reporting ties" IS the resolution.** DASH reporting "tied" for correlated features is not giving up — it's the Pareto-optimal answer. The tie IS the explanation.

4. **The group identification bottleneck is the real limitation.** The drug discovery experiment proved this: Pearson fails on binary features, MI/Jaccard recover. The framework works when you identify the right groups; identifying them is the hard problem.

5. **Honest falsifications build credibility.** Three pre-registered extensions falsified. This is rare and reviewers praised it.

### Level 5: The Lean Formalization

The three repos share a dependency:

```
mathlib
  ↑
  ├── universal-explanation-impossibility  (ExplanationSystem, core theorem, instances)
  │     ↑
  │     └── ostrowski-impossibility  (imports ExplanationSystem, adds physics)
  │
  └── dash-impossibility-lean  (standalone, attribution-specific)
```

**Axiom philosophy:** The core theorem uses 0 domain axioms. Domain instances axiomatize domain-specific properties (Rashomon for ML, Ostrowski for physics). The companion Ostrowski repo reduced from 62 to 10 axioms by proving Arrow's theorem, bridging Ostrowski to Mathlib, and making all ML instances constructive.

**Key Lean files:**
- `ExplanationSystem.lean` — the 4-line proof
- `UniversalResolution.lean` — G-invariant maps are stable
- `BilemmaCharacterization.lean` — the 2×2 tightness classification
- `Ubiquity.lean` — dimensional argument for generic underspecification

### Level 6: Publication Strategy

| Paper | Target | Status | Priority |
|-------|--------|--------|----------|
| Nature Article | Nature | Ready to submit | 1 |
| arXiv Monograph | arXiv | Ready to post (simultaneous with Nature) | 1 |
| Physics Paper | Foundations of Physics | Submitted (21/21 Accept in simulated R4) | 2 |
| Attribution Paper | JMLR | ~90% ready | 3 |
| MI v2 results | Update to monograph | Waiting for SageMaker | — |

**Submission order:** Post monograph to arXiv → submit Nature Article → FoP paper is already submitted → JMLR when ready.

### Level 7: What Needs Your Input

1. **Review the Nature Article** (`paper/nature_article.tex`, 11 pages). Does the narrative work? Is the abstract compelling?

2. **Review the monograph's limitations** (Section 7.8 in the monograph). Are we being honest enough? Too defensive?

3. **The η law framing.** We report R²=0.957 for 7 instances and R²=0.25 for all 16. Is this honest enough, or should we restructure to lead with the weaker number?

4. **The drug discovery boundary condition.** MI/Jaccard fix the magnitude prediction but bimodality doesn't transfer. Is this a limitation or a finding?

5. **Author contributions statement.** What division of labor reflects reality?

---

## Repository Map

```
ds_projects/
├── universal-explanation-impossibility/     ← YOU ARE HERE
│   ├── UniversalImpossibility/              # 107 Lean files
│   ├── paper/
│   │   ├── nature_article.tex               # Nature submission (11pp)
│   │   ├── universal_impossibility_monograph.tex  # Definitive reference (104pp)
│   │   ├── nature_cover_letter.tex          # Cover letter
│   │   ├── figures/                         # All PDFs
│   │   ├── sections/                        # 20 instance section files
│   │   └── references.bib
│   ├── knockout-experiments/                # 90+ scripts, 80+ result JSONs
│   │   ├── RESULTS_SYNTHESIS.md             # 3 confirmed, 2 falsified, 1 negative
│   │   ├── PRE_REGISTRATION.md              # Pre-registered predictions
│   │   └── CORRECTIONS.md                   # Issue tracker
│   └── docs/
│       ├── coauthor-onboarding.md           # THIS FILE
│       └── superpowers/plans/               # Implementation plans
│
├── dash-impossibility-lean/                 # Attribution specialization
│   ├── DASHImpossibility/                   # 58 Lean files
│   └── paper/                               # JMLR/NeurIPS versions
│
└── ostrowski-impossibility/                 # Physics application
    ├── OstrowskiImpossibility/              # 17 Lean files
    └── paper/                               # FoP submission
```

---

## Questions? Start Here

- **"What exactly does the theorem say?"** → Nature Article, Theorem 1 (page 2)
- **"How is this different from Arrow's theorem?"** → Monograph, Section 6.7
- **"What's the practical impact for ML?"** → Attribution paper executive summary
- **"How does the Lean code work?"** → Monograph, Section 8 + `ExplanationSystem.lean`
- **"What failed?"** → `RESULTS_SYNTHESIS.md` in knockout-experiments/
- **"What's the strongest single result?"** → Noether counting (pre-registered, 47pp bimodal gap)
- **"What's the weakest point?"** → The core theorem is elementary; η law n=7 is fragile
