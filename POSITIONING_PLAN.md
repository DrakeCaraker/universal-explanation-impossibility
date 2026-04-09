# Universal Explanation Impossibility — Positioning & Impact Plan

**Goal**: Position this result for maximum long-term impact. Determine
the optimal venue, framing, and cross-domain connections to maximize
the likelihood of this becoming a foundational, widely-cited result.

**Core question**: Is this an ML paper that happens to be general,
or a general result that happens to have been discovered in ML?

**Answer**: It is the latter. The theorem is about the structure of
explanation under underdetermination. ML is the proving ground, but
the result belongs to the intersection of mathematics, philosophy
of science, and information theory — with ML as the most immediately
impactful application domain.

---

## Phase 1: Literature Investigation [CRITICAL — Do First]

Before any writing or venue decisions, thoroughly investigate the
prior art across ALL relevant fields. The worst outcome is
discovering after submission that someone proved this in 1978
in a philosophy journal.

### Task 1.1: Search for Prior Art in Philosophy of Science [Opus]

Search for formal (mathematical, not just verbal) treatments of:

1. **Quine-Duhem underdetermination formalized**:
   Has anyone proved an impossibility theorem from the
   underdetermination thesis? Not just argued it verbally
   (Quine 1951, Duhem 1906) but PROVED it formally?

   Key search terms: "underdetermination impossibility theorem",
   "formal underdetermination", "mathematical underdetermination",
   "theory choice impossibility"

   Key authors to check: van Fraassen, Stanford, Laudan, Earman,
   Norton, Okasha

2. **Structural realism as a formal resolution**:
   Has anyone formally characterized structural realism as an
   optimal solution to underdetermination? Worrall (1989) proposed
   it, but was it ever proved optimal?

   Key terms: "structural realism optimality", "invariant theory
   choice", "structural empiricism formal"

3. **Theory choice axiomatics**:
   Has anyone done an Arrow-style axiomatization of theory choice?
   Okasha (2011) "Theory Choice and Social Choice" explicitly
   connects Arrow to theory choice. How far did he get?

**Deliverable**: A table of every formal (mathematical) result about
explanation/interpretation impossibility across philosophy, with:
paper, year, what they proved, how it differs from our result.

### Task 1.2: Search for Prior Art in Statistics [Opus]

1. **Rashomon set impossibility results**:
   Has anyone proved an impossibility theorem from the Rashomon
   effect? Breiman (2001) named it, Rudin et al. characterized
   it, but did anyone prove that explanation under Rashomon is
   impossible?

   Key: Fisher et al. (2019) "All Models are Wrong, but Many are
   Useful" — did they prove impossibility or just demonstrate it?

   Key: Marx et al. (2023/2024) — uncertainty in explanations.

2. **Hunt-Stein theorem connection**:
   Has anyone explicitly connected the Hunt-Stein theorem to
   explainability? Our G-invariant resolution IS Hunt-Stein
   applied to explanation. If someone already made this connection,
   we must cite them prominently.

3. **Identification impossibility in econometrics**:
   Manski's partial identification program — has he proved
   anything structurally similar to our result?

### Task 1.3: Search for Prior Art in Formal Verification [Opus]

1. Has anyone formalized ANY explanation impossibility in a
   proof assistant (Lean, Coq, Isabelle, Agda)?

2. Nipkow formalized Arrow's theorem in Isabelle. Has anyone
   done similar for underdetermination or XAI impossibility?

3. Zhang et al. (2026?) on statistical learning theory in Lean —
   how does it relate?

### Task 1.4: Search for Prior Art in Information Theory [Opus]

1. Has anyone derived the explanation impossibility from the
   data processing inequality?

2. Rate-distortion theory applied to explanation/interpretation —
   any prior work?

3. The Blackwell sufficiency theorem and its connection to
   explanation: has anyone made this connection?

### Task 1.5: Search for Prior Art in Database Theory [Opus]

1. The view update problem (Bancilhon & Spyratos 1981) — has
   anyone generalized it to a domain-independent impossibility?

2. Has anyone connected view updates to XAI, underdetermination,
   or Arrow's theorem?

### GATE: Review all literature findings. If significant prior art
exists, revise the positioning to acknowledge it and clarify
our contribution (unification, Lean formalization, ML instances,
empirical validation). If no prior art exists, we have a stronger
novelty claim.

---

## Phase 2: Venue Analysis [After Literature Review]

### Task 2.1: Evaluate Venue Options [Opus]

For each venue, assess: audience, impact, format fit, review
culture, and strategic positioning.

**ML venues (if positioning as ML contribution):**

| Venue | Pros | Cons | Fit |
|-------|------|------|-----|
| NeurIPS 2026 | Highest ML visibility, 10-page format forces clarity | Competitive, reviewers may say "trivial proof" | Medium |
| ICML 2026 | Theory track values formal results | Same competition concerns | Medium |
| JMLR | No page limit, values rigor, Lean is a differentiator | Slower review, lower visibility than conference | High |
| TMLR | Faster review, growing reputation | Less prestige | Medium |

**Interdisciplinary venues (if positioning as general result):**

| Venue | Pros | Cons | Fit |
|-------|------|------|-----|
| Nature Machine Intelligence | High impact, reaches beyond ML | Hard to get in, may want more experiments | High |
| PNAS | Prestige, cross-domain, "science" framing | Hard to get in, needs a NAS member to communicate | Very High |
| Science / Nature (short comms) | Maximum visibility | Extremely competitive | Low |
| Proceedings of the Royal Society A | Values formal math, cross-domain | Less ML visibility | Medium |
| Philosophy of Science | Perfect for underdetermination angle | No ML audience | Medium |
| Journal of Philosophy | Highest philosophy prestige | No ML audience | Low |

**Formal methods venues:**

| Venue | Pros | Cons | Fit |
|-------|------|------|-----|
| ITP (Interactive Theorem Proving) | Values Lean formalization | Niche audience | Medium |
| LICS (Logic in Computer Science) | Values formal impossibility | Need stronger logic content | Low |

**Policy venues:**

| Venue | Pros | Cons | Fit |
|-------|------|------|-----|
| FAccT | AI fairness + accountability, regulatory angle | May want more fairness focus | Medium |
| AIES | AI ethics, policy implications | Smaller venue | Medium |

**Multi-venue strategy (RECOMMENDED):**

Publish the SAME result in MULTIPLE venues with DIFFERENT framings:

1. **NeurIPS/ICML (ML audience)**: 10-page version focusing on
   9 ML instances, experiments, Lean verification. This establishes
   priority and ML visibility.

2. **Nature Machine Intelligence or PNAS (broad science)**:
   Shorter version (~5000 words) framing as "a fundamental limit
   on explanation under underdetermination, with applications from
   ML to philosophy of science." Cross-domain framing.

3. **Philosophy of Science (philosophy audience)**: Version
   emphasizing the Quine-Duhem connection, the formalization of
   structural realism as G-invariant resolution, and the Lean
   verification.

4. **arXiv preprint (immediate)**: Full monograph version.
   Establishes priority, allows all communities to access it.

This multi-venue strategy is how Arrow's theorem achieved
cross-domain impact: the original was in a journal, but
it was applied and cited across economics, political science,
philosophy, and computer science through different framings.

### Task 2.2: Evaluate Nature Machine Intelligence [Opus]

NMI is specifically designed for this kind of result:
- "Research that provides fundamental insight into intelligence"
- Values cross-disciplinary work
- Publishes impossibility results and theoretical foundations
- Has published on XAI, Rashomon sets, and interpretability
- Reaches ML, neuro, cog sci, philosophy audiences simultaneously

Check: What impossibility/foundation results has NMI published?
What's their typical format? Do they accept Lean-verified results?
Would the regulatory + MI + philosophy framing fit?

### Task 2.3: Evaluate PNAS [Opus]

PNAS publishes cross-domain mathematical results:
- Arrow's-theorem-class results have appeared in PNAS
- "Contributed" track requires NAS member sponsorship
- "Direct submission" track is open but competitive
- Format: ~6 pages main + SI

Check: Has PNAS published ML impossibility results?
Social choice impossibilities? Underdetermination formalizations?
Who at NAS could sponsor this?

---

## Phase 3: Paper Restructuring [After Venue Decision]

### Task 3.1: Assess Current Paper's Cross-Domain Clarity [Opus]

Read paper/universal_impossibility.tex and answer:

1. Does the abstract mention the generality beyond ML?
2. Does the intro connect to philosophy of science?
3. Is the framework presented in ML-specific or general terms?
4. Does the discussion section explore cross-domain implications?
5. Would a philosopher of science understand the paper?
6. Would a statistician recognize the Rashomon connection?
7. Would a neuroscientist see the degeneracy parallel?

For each gap identified, specify what needs to change.

### Task 3.2: Write "Implications Beyond ML" Section [Opus]

Create paper/sections/beyond_ml.tex (~2 pages for monograph,
~0.5 pages for conference):

Organized by domain:

**Philosophy of Science**: Formalizes the Quine-Duhem
underdetermination thesis as a theorem. The G-invariant
resolution formalizes structural realism. First machine-checked
proof of an impossibility in philosophy of science.

**Neuroscience**: The degeneracy principle (Marder & Goaillard
2006) is an instance. Multiple neural circuits implement the
same behavior. Circuit-level explanations face the same trilemma.

**Medical Diagnosis**: Differential diagnosis is the G-invariant
resolution. The theorem proves Bayesian diagnostic reasoning is
optimal — not just pragmatic but mathematically necessary.

**Legal Interpretation**: Statutory ambiguity is structural, not
resolvable by better interpretive methods. The "core and
penumbra" (Hart) is the G-invariant projection.

**Economics**: The identification problem in econometrics is an
instance. Partial identification (Manski) is the resolution.

**Quantum Mechanics**: The interpretation problem is an instance.
"Shut up and calculate" is the G-invariant projection.

Each connection should be 2-3 sentences with at least one
citation to the domain-specific literature.

### Task 3.3: Write NMI/PNAS Version [Opus]

If targeting NMI or PNAS, create a NEW version with:

1. Title emphasizing generality:
   "The Impossibility of Faithful Explanation Under
   Underdetermination" (no "ML" in title)

   OR: "Explanation, Stability, and Decisiveness: A Universal
   Impossibility Theorem"

2. Abstract (150 words):
   - Problem: explanation under underdetermination
   - Result: impossibility + exact boundary (Rashomon)
   - Resolution: G-invariant projection (optimal + unique)
   - Scope: 9 instances spanning ML, causal inference, and
     neuroscience
   - Verification: Lean 4, 346 theorems, 0 sorry
   - Implications: philosophy (underdetermination formalized),
     regulation (impossible compliance), AI safety (MI trilemma)

3. Main text (~5000 words / 6 pages):
   - The framework (general, not ML-specific terminology)
   - The theorem + proof + tightness
   - Selected instances (3-4, chosen for cross-domain breadth:
     ML attribution, causal discovery, mechanistic interp,
     neuroscience degeneracy)
   - The resolution
   - Implications across domains

4. Methods/SI:
   - All 9 instances
   - All experiments
   - Lean verification details
   - Axiom substitution analysis

### Task 3.4: Add Cross-Domain Bibliography [Sonnet]

Add to references.bib:
- quine1951dogmas — "Two Dogmas of Empiricism", Philosophical Review
- duhem1906aim — "The Aim and Structure of Physical Theory"
- vanfraassen1980scientific — "The Scientific Image"
- worrall1989structural — "Structural Realism: The Best of Both Worlds?"
- okasha2011theory — "Theory Choice and Social Choice"
- marder2006variability — Marder & Goaillard, "Variability, Compensation and Homeostasis in Neuron and Network Function"
- manski2003partial — "Partial Identification of Probability Distributions"
- bancilhon1981update — "Update Semantics of Relational Views"
- hart1961concept — "The Concept of Law" (Hart, core and penumbra)
- breiman2001statistical — "Statistical Modeling: The Two Cultures"

### Task 3.5: Revise Abstract and Introduction [Opus]

Rewrite the abstract and introduction of the MAIN paper
(universal_impossibility.tex) to:

1. Open with the GENERAL framing: "When a system is
   underspecified — when multiple configurations produce the
   same observable output — any explanation of the system faces
   a fundamental tradeoff." NOT with ML-specific language.

2. Introduce the three properties in GENERAL terms before
   specializing to ML.

3. State the ML instances as APPLICATIONS of a general result,
   not as the result itself.

4. Mention the philosophy, neuroscience, and legal connections
   in the intro (1 sentence each) to signal generality.

5. Keep the ML focus for the bulk of the paper (that's where
   the experiments and instances are) but frame it as
   "the most developed application domain."

---

## Phase 4: Strategic Actions [After Paper Revision]

### Task 4.1: arXiv Posting Strategy

Post the monograph to arXiv FIRST. Categories:
- Primary: cs.LG (Machine Learning)
- Cross-list: cs.AI, stat.ML, cs.LO (Logic in CS)
- Consider: math.LO (Mathematical Logic), physics.hist-ph
  (History and Philosophy of Physics)

The cross-listing is crucial for reaching beyond ML.

### Task 4.2: Engagement Plan

After arXiv posting:
- Tweet thread with the trilemma diagram + key numbers
- Post to LessWrong/Alignment Forum (MI instance)
- Post to Philosophy of Science subreddit/forums
- Email key researchers directly:
  - Cynthia Rudin (Rashomon sets)
  - Neel Nanda / Chris Olah (mechanistic interpretability)
  - Samir Okasha (theory choice and social choice)
  - Eve Marder (degeneracy principle)
  - Charles Manski (partial identification)

### Task 4.3: Multi-Venue Submission Timeline

Week 1: Post monograph to arXiv
Week 1: Submit NeurIPS version (May 4 abstract, May 6 paper)
Month 2: Submit NMI or PNAS version (cross-domain framing)
Month 3: Submit JMLR version (full ML treatment)
Month 4: Submit Philosophy of Science version (if warranted)

Each submission is a different FRAMING of the same result,
targeting a different audience. This is standard practice for
results that span multiple fields.

---

## Phase 5: Execution [Paper Revisions]

### Task 5.1: Update Main Paper Framing [Opus]

Revise universal_impossibility.tex:
- Abstract: general opening, ML specialization
- Intro: general framing first, then "we develop this most
  fully in the context of ML explainability"
- Add "Implications Beyond ML" section (Task 3.2)
- Discussion: connect to underdetermination, degeneracy,
  identification, legal interpretation

### Task 5.2: Create NMI Submission Version [Opus]

New file: paper/universal_impossibility_nmi.tex
~5000 words, cross-domain framing, 3-4 selected instances,
general terminology, SI with full details.

### Task 5.3: Update NeurIPS Version [Opus]

Revise paper/universal_impossibility_neurips.tex:
- Add 1 sentence to abstract about generality beyond ML
- Add 1 paragraph to discussion about broader implications
- Keep the focus on ML (NeurIPS audience)

### Task 5.4: Create Philosophy Version (Optional) [Opus]

paper/universal_impossibility_philosophy.tex:
Focus on: Quine-Duhem formalization, structural realism as
resolution, Lean verification as methodological contribution.
Target: Philosophy of Science or BJPS.

---

## Confidence Assessment

| Decision | Confidence | Justification |
|----------|-----------|---------------|
| Multi-venue strategy | HIGH | Standard for cross-domain results |
| NeurIPS as ML venue | HIGH | Deadline fits, format works |
| NMI as cross-domain venue | HIGH | Perfect audience, values this type of work |
| PNAS | MEDIUM | Harder to get in, needs sponsorship |
| Philosophy journal | MEDIUM | Depends on literature review findings |
| arXiv first | HIGH | Establishes priority, zero downside |
| General framing in abstract | HIGH | Makes the result accessible to all audiences |

## Open Questions for Literature Review

1. Has Okasha (2011) or anyone else already proved an
   Arrow-for-theory-choice impossibility?
2. Has Hunt-Stein been connected to XAI before?
3. Has the view update problem been connected to
   underdetermination before?
4. Has anyone formalized the degeneracy principle as
   an impossibility theorem?
5. Is there a rate-distortion theorem for explanation?
