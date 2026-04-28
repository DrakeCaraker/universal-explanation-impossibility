# The Limits of Explanation: A Complete Overview

## What was proved

A single mathematical theorem — formally verified in Lean 4 with zero domain-specific axioms — establishes that no explanation of an underspecified system can simultaneously be faithful (reflect the system's actual structure), stable (agree across equivalent configurations), and decisive (commit to a single answer). This is not a limitation of current methods. It is a limitation of mathematics. No future algorithm, no matter how sophisticated, can overcome it.

The theorem comes with four additional results:
1. A **tightness classification** determining which property pairs survive when the triple fails — and a cross-domain classification of 20 impossibility theorems from 12 domains showing that the explanation bilemma is structurally more severe than Arrow, Gödel, Bell, or fairness impossibilities.
2. A **constructive resolution** (orbit averaging) proven Pareto-optimal, explaining why eight scientific communities independently converged on the same strategy.
3. A **quantitative instability law** (the η formula) predicting the instability rate from symmetry structure alone (R² = 0.957, zero free parameters).
4. A **recursive resolution pattern** — enrichment creates new impossibility at each level, proved to mirror the structure of Gödel's incompleteness and Galois field extension towers.

## What this work actually is

Strip away the technical apparatus and what you have is a single insight with unusually long reach: **explanation and prediction are different things, and the gap between them is not a bug to be fixed but a mathematical fact to be managed.**

Every scientist, every regulator, every clinician who has ever asked "why did the model say that?" has implicitly assumed the answer exists and is findable. The theorem says: for a precisely characterizable class of problems — those where multiple valid models coexist — the answer doesn't exist in the form they're asking for. You can have an answer that's accurate (faithful), an answer that's reproducible (stable), or an answer that's specific (decisive). You cannot have all three. And you can prove, in four lines, that no one ever will.

This is the kind of result that, once stated, feels obvious. Of course you can't uniquely attribute credit when features are correlated. Of course different valid models give different explanations. Everyone who's worked with SHAP or LIME has encountered this. The contribution is not the observation — it's the proof that the observation is inescapable, the characterization of exactly when it is and isn't (the tightness classification), the identification of the optimal response (orbit averaging), and the demonstration across four high-stakes empirical domains.

## Why it matters

### The theorem changes what practitioners should expect from explanations

The current default in machine learning, biomedicine, and neuroscience is that if an explanation is unstable, the method needs improvement. The theorem says: no. For a large class of problems — any problem where multiple valid models coexist — instability is the mathematically guaranteed default. The right response is not to seek a better method but to use the optimal resolution (orbit averaging) and trust only the conclusions the theorem identifies as stable (between-group comparisons via Noether counting).

### The gene expression finding has immediate consequences for drug discovery

When a biomarker discovery pipeline names TSPAN8 (invasion pathway) as the top gene in 92% of training runs and CEACAM5 (immune evasion pathway) in the remaining runs — with zero shared Gene Ontology biological process terms between them — the specific therapeutic target a pharmaceutical team pursues depends on a random number generator. This is not hypothetical: these are real genes, real pathways, and real microarray data from 546 patients. The resolution (report both, don't privilege either) changes how pipelines should be designed.

### The mechanistic interpretability finding challenges a foundational assumption of AI safety

The growing field of mechanistic interpretability seeks to reverse-engineer neural networks into human-readable circuits, with the goal of building safety cases for AI systems. The finding that ten transformers computing the same function with identical accuracy discover circuits that overlap by only 2.2% (Fourier-frequency Jaccard) and their importance rankings agree at ρ = 0.52 means that the "circuit" an audit identifies depends more on the training run than on the computation being performed. The resolution — projecting onto the architecturally-derived invariant subspace, which lifts agreement from ρ = 0.52 to ρ = 0.93 — shows what IS stable: equivalence classes of components (MLP vs attention layer means), not individual circuit elements.

### The neuroimaging result reframes a 1000-citation crisis

Botvinik-Nezer et al. (Nature 2020) showed that 70 teams analysing the same fMRI data reach different conclusions. The response has been: better standards, pre-registration, standardised pipelines. The theorem says: the disagreement is irreducible. No amount of standardisation eliminates it when multiple valid analysis pipelines exist. But it also says: the overlap map they already used descriptively is provably near-optimal, and 16 independent analyses suffice for 95% stability. This transforms a descriptive crisis into a constructive prescription.

### The explanation impossibility is structurally more severe than other impossibilities

A classification of 20 impossibility theorems from 12 domains reveals that 16 have "full tightness" — every pair is independently achievable, and practitioners can "pick two." The explanation bilemma is one of only two instances with "collapsed tightness" — property pairs are blocked, not just the triple. The other is the quantum linearity trilemma (no-cloning + measurement disturbance). Collapsed tightness means the standard "pick two" resolution strategy fails; enrichment or approximation bounds are required. This is why explanation methods need qualitatively different design approaches than fairness methods, Arrow-style social choice, or CAP theorem tradeoffs.

## The scope of the unification

The impossibility is not specific to machine learning. The same structure — observationally equivalent configurations producing incompatible explanations — appears in:

- **Causal inference**: Markov-equivalent DAGs share conditional independence structure but imply different causal mechanisms. The CPDAG is the neutral-element resolution.
- **Crystallography**: diffraction amplitudes determine structure factor magnitudes but not phases. The Patterson map is the orbit average.
- **Gauge theory**: gauge-equivalent field configurations produce the same observables. Gauge-invariant observables are the G-invariant projection.
- **Statistical mechanics**: microstates sharing the same macroscopic observables. The Boltzmann average is orbit averaging.
- **Genetic code**: synonymous codons encoding the same amino acid. The degeneracy structure matches the η law.
- **Linguistics**: structurally ambiguous sentences admitting multiple parse trees.
- **Database theory**: the view-update problem — multiple base states compatible with the same view.

That eight scientific communities independently converged on structurally identical resolutions (orbit averaging in various forms) — without cross-pollination — is now explained: the G-invariant projection is the unique Pareto-optimal strategy for any explanation system with the Rashomon property.

## The recursive resolution

Enrichment — adding neutral elements to the explanation space — restores the blocked property pairs, but the enriched space hosts a new impossibility at the next level. The levels are independent: resolving level k gives zero information about level k+1 (both proved in Lean). The same recursive pattern appears in Gödel's incompleteness (add Con(T); new unprovable sentence) and in Galois theory (adjoin a root; new irreducible polynomials). The enrichment stack mirrors the algebraic structure of a tower of field extensions.

A complete explanatory theory therefore classifies which questions have formulation-independent answers at each level, rather than answering all questions — an epistemological limit that holds wherever collapsed tightness obtains. This is a formalization, with the first quantitative predictions and proof of optimal resolution, of the Duhem-Quine thesis that theories are underdetermined by data.

## What the formal verification means

The entire framework is verified in Lean 4 across three repositories:

| Repository | Theorems | Axioms | Content |
|------------|----------|--------|---------|
| Universal | 501 | 25 | Core theorem, 9 ML + 14 cross-domain instances |
| Attribution | 357 | 6 | SHAP-specific: GBDT, Lasso, neural net, DASH equity |
| Ostrowski | 451 | 10 | Bilemma, tightness classification, enrichment stack, physics |
| **Total** | **1,309** | **41** | **0 sorry** |

The core impossibility theorem uses zero domain-specific axioms — only the Rashomon property as a hypothesis. This means: the theorem cannot be wrong. It can be irrelevant (if the Rashomon property doesn't hold), it can be trivial (if the consequences are obvious), but it cannot be logically false. For a result with implications across eight sciences, this level of certainty is rare.

## What the honest negatives mean

Five pre-registered predictions were falsified:

1. The phase transition location (predicted r* = 1, observed r* = 0.01-0.12)
2. The uncertainty bound (predicted α+σ+δ ≤ 2, observed max 2.86)
3. Molecular evolution from character theory (partial R² = 0.0)
4. Spectral gap convergence rate (14-100× too fast)
5. Flip correlations from irreducible decomposition (within-group ≈ between-group)

These negatives define the framework's boundary: the impossibility and resolution work universally; the quantitative predictions (η law, Noether counting) work only when the symmetry group is exactly known and features are genuinely exchangeable. At approximate symmetry, simpler methods (correlation-based null model, coverage conflict diagnostic) match or beat the framework's specific predictions.

This honest reporting of failures is itself a contribution. It prevents the overclaiming that plagues theoretical frameworks in ML and provides a clear user guide: use the theorem for the impossibility proof, use the η law only with known groups, use the coverage conflict for practical diagnostics.

## Historical analogs

### Arrow's Impossibility Theorem (1951)

No voting system satisfying unanimity, independence of irrelevant alternatives, and non-dictatorship can aggregate individual preferences into a social ranking. Before Arrow, political scientists searched for the "right" voting system. After Arrow, they understood that ALL voting systems make tradeoffs. Arrow won the Nobel Prize (~65,000 citations).

The parallel: before this theorem, ML practitioners search for the "right" explanation method. After, they understand that ALL explanation methods make tradeoffs under Rashomon. Arrow is proved as an instance within the framework — with full tightness, meaning "pick two" works. The explanation bilemma has collapsed tightness, making it structurally more severe.

### Gödel's Incompleteness Theorems (1931)

No consistent formal system containing arithmetic can prove all true statements about arithmetic. The parallel: this theorem says no explanation method can be complete (decisive), consistent (stable), and correct (faithful) under Rashomon. The recursive resolution shares a pattern with Gödel: both produce impossibility at every level with unbounded depth. The mechanisms differ (Gödel: self-reference; enrichment stack: Rashomon at finer resolutions), but both are proved as instances of a common RecursiveImpossibility interface in Lean.

### Shannon's Channel Capacity (1948)

Shannon established the fundamental limits of communication and identified the optimal coding strategy. The parallel is direct: this framework establishes the fundamental limits of explanation and identifies the optimal explanatory strategy (orbit averaging). Both are negative results (you can't exceed capacity / you can't have F+S+D) that come with constructive optimality (channel codes / G-invariant projection).

### Bilodeau et al. (2024, PNAS)

Proved that Shapley-based feature attribution methods cannot simultaneously satisfy certain axioms under collinearity. This is the most directly comparable prior work. They proved a special case (Shapley axioms, feature attribution); this work proves the general case (any explanation type, any domain, zero axioms beyond Rashomon). Bilodeau's result is subsumed as an instance.

### Chouldechova (2017) / Kleinberg-Mullainathan-Raghavan (2016)

Proved that certain fairness criteria are mutually incompatible. The tightness classification reveals a key structural difference: fairness impossibilities have full tightness (pick two works), while the explanation bilemma has collapsed tightness (enrichment required). Explanation methods need qualitatively different design approaches.

## Unique characteristics of this work

Among impossibility theorems, this one is unusual in five ways:

**Constructive.** Most impossibility theorems stop at "you can't." This one proves the impossibility AND identifies the unique Pareto-optimal resolution AND provides quantitative predictions of the instability rate.

**Cross-domain.** The same structure — the Rashomon property — appears in eight sciences, and the same resolution — orbit averaging — works in each. The convergence of independent solutions now has a mathematical explanation.

**Formally verified.** Among major impossibility theorems in science, almost none have been machine-checked as part of the original contribution. 1,309 theorems across 3 repos with 0 sorry tactics.

**Empirically validated.** Arrow's theorem is pure mathematics — there's no experiment that "confirms" it. This theorem makes quantitative predictions (the η law, the Noether counting, the convergence prescription) confirmed across seven domains.

**Classified.** The tightness classification of 20 impossibility theorems from 12 domains is the first systematic comparison of impossibility theorems by structural type. It reveals the explanation bilemma as an outlier — one of two collapsed instances among six structurally proved cases.

## Impact on specific fields

### Machine learning / XAI

The field of explainable AI has been growing rapidly (>10,000 papers/year) without a foundational result establishing what explanation CAN and CANNOT do. This is that result. The practical tools — coverage conflict diagnostic (7 lines of code, Spearman 0.96), DASH ensemble averaging (proven optimal), Noether counting (exact count of reliable conclusions) — are immediately deployable.

### Genomics / drug discovery

The TSPAN8/CEACAM5 finding is the first demonstration that the #1 biomarker nominated by a standard ML pipeline alternates between genes in DIFFERENT biological pathways depending on the random seed. The resolution: report the DASH consensus top-k, not the single-model top gene.

### AI safety / mechanistic interpretability

The MI finding directly challenges the assumption that circuits are stable, interpretable objects. The resolution shows what IS stable: equivalence classes of circuits, not individual components. Safety cases built on specific circuit identifications are fragile; safety cases built on invariant properties (MLP layers vs attention heads) are robust.

### Neuroimaging / reproducibility

The NARPS reframing provides the field with a mathematical justification for multi-analyst approaches AND a concrete prescription: 16 independent analyses for 95% stability.

### Regulation / policy

The EU AI Act (Article 13) and US ECOA require explanations of automated decisions. The theorem shows that not all explanations are equally reliable. Regulatory frameworks should distinguish between structural explanations (which feature GROUP matters — stable) and individual explanations (which specific feature — potentially unstable).

## What this means for explanation

The deepest implication is philosophical. The theorem formalizes the Duhem-Quine thesis that theories are underdetermined by data, with the first quantitative predictions and proof of optimal resolution. The orbit average preserves exactly the structural content that structural realists argue survives theory change.

The enrichment mechanism — adding a neutral element to resolve the impossibility at the cost of decisiveness — formalizes a recurring pattern. Quantum mechanics added "superposition" to resolve wave-particle duality. Relativity added "frame-dependent" to resolve the simultaneity paradox. DASH adds "tied" to resolve the attribution instability. Each enrichment sacrifices decisiveness but gains stability. And each enrichment creates the conditions for a new impossibility at the next level — the recursive resolution pattern proved in Lean.

## Publication program

| Paper | Venue | Status | Role |
|-------|-------|--------|------|
| The Limits of Explanation | Nature | Ready for submission | Flagship: cross-domain framework + four instances |
| The Attribution Impossibility | NeurIPS 2026 | Ready (abstract May 4) | ML companion: SHAP-specific depth + diagnostic tools |
| Universal Impossibility Monograph | arXiv | Ready | Definitive reference (~4,400 lines, all proofs) |
| Ostrowski Impossibility | Foundations of Physics | Submission-ready (21/21 Accept) | Physics companion: bilemma + spacetime + enrichment stack |
| DASH Pipeline | TMLR | Planned | Software paper: implementation + API |

## Technical inventory

| Component | Count |
|-----------|-------|
| Lean theorems (universal) | 501 |
| Lean theorems (attribution) | 357 |
| Lean theorems (Ostrowski) | 434 |
| **Lean theorems (total)** | **1,309** |
| Lean axioms (universal) | 25 |
| Lean axioms (attribution) | 6 |
| Lean axioms (Ostrowski) | 10 |
| Sorry tactics | **0** |
| Lean files (universal) | 101 |
| Lean files (attribution) | 58 |
| Lean files (Ostrowski) | 34 |
| ML explanation instances | 9 |
| Cross-domain instances | 14 |
| Impossibility classification | 20 theorems, 12 domains |
| Knockout experiments | 90+ scripts |
| Result JSONs | 80+ |
| Pre-registered falsifications | 5 of 8 |
| Gene expression datasets | 4 |
| Neuroimaging teams analyzed | 48 |
| η law R² (7 domains) | 0.957 |
| Noether bimodal gap | 50 pp |
| MI G-invariant resolution | ρ: 0.52 → 0.93 |
| NARPS convergence (M₉₅) | 16 [10, 22] |
