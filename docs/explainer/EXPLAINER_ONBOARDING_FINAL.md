# The Limits of Explanation — onboarding flow, FINAL content draft

*2026-08-02 · FINAL (v3, promoted). Supersedes EXPLAINER_ONBOARDING_DRAFT_V3.md and all
prior drafts. Integrates four 5-expert audit rounds (`explainer-audit/`, `/round2/`,
`/round3/`, `/expansion/`), the medical cold-open analysis
(`EXPLAINER_MEDICAL_COLDOPEN.md`), and the revision plan (`EXPLAINER_REVISION_PLAN.md`);
every round converged on [MINOR POLISH]/ship and every polish item is applied here. All
numbers verified against `paper/claims.tex`, the monograph, and the result JSONs
(fact-check: 62 claims, 3 errors found and fixed here); counts and Monograph line-anchors
re-verified 2026-08-02 against main @ `c74d82d` (130 files / 715 theorems / 4 program-wide
axioms — repo unchanged since anchor derivation, so all §/line anchors remain valid).
Every formal-layer citation carries `theorem · File.lean:line · axiom status · Monograph
§/line · [JSON / external lit]`. This document IS the explainer; reviewing it = reviewing
the page. Next step (Drake-gated): build the interactive page per Part 3, re-verifying
every count and anchor at build time.*

---

## Part 1 — Design brief

**Goal.** Take a reader with zero context — anyone from a complete layperson to a
specialist like Cynthia Rudin — to a high-level intuition of the Limits of Explanation
program, by walking the natural questions that arise at each step of the real
investigation. Start from small questions; expand into the grander vision only as it is
earned; hint at that vision from the first screen.

**The takeaway** (every reader should be able to say this at the end — and it must stand
alone, with zero context, when repeated away from the page):

> When many explanations fit the facts equally well — and they usually do — the one you're
> given is partly an accident of the draw. That is now a theorem, not a suspicion. But the
> part every equally good explanation agrees on *can* be trusted — and a machine checked
> every step.

**Title.** The zero-context rule applies to the hero too: no concept appears in a title
before the page introduces it — no property names, no "machine-verified," no "theorem,"
no "certificate." The hero is MINIMAL (Drake, post-build): **H1 = "The Limits of
Explanation"** (accent on *Limits*; matches `<title>` for URL/tab continuity) and the
byline. No epigraph, no kicker, no deck, no receipts chips: the page opens on the title
and drops straight into the scene. Counts surface at node 5 and the closing stats band.
The named handles surface only after their nodes earn them: "Faithful. Stable. Decisive.
Pick Two." becomes available as a section divider after node 3 and is the
shareable-card/OG-image line — not the hero.

**Layered spine.** One shared question sequence; each answer stacks three registers —
**plain** (always visible; no notation, no Greek, no theorem names), **go deeper**
(mechanisms, light notation, quantities), **formal** (statements, Lean names + file:line,
axiom status, monograph §, result JSONs, literature — the layer a Rudin-tier reader reads
first; it must never overclaim). Expanding is never required to follow the plain spine.

**Evidence tiers** (on every formal-layer claim; tier ≠ layer — layers are depth of
register, tiers are epistemic status): **Proven** (machine-checked in Lean from stated
axioms), **Measured** (validated by experiment, reported as coverage of a bound with
sample sizes and named failures), **Conjectured** (structural analogy or informal step,
explicitly not a proof), **Retired** (tested and falsified, reported on purpose).

**Through-lines to thread:**
- *The document motif:* one fixed medical/legal document whose bold "Primary factor" line
  changes while everything around it holds still — node 0 cycling → node 3 fanned beside
  the three property cards → node 7 superimposed (shared text dark, differing lines
  ghosted = the stable core made visible) → node 9 a wall of shimmering-vs-still summaries
  → node 12 one certified line.
- *POV returns:* the patient from node 0 is the page's recurring test case — each major
  concept lands back on her, spaced across the arc (not every node): her case is the exact
  thing that broke *stability* (node 2); the stable-core version of her summary, and what it
  would actually say — cell size, shape flagged undetermined (7); the certified reason her
  surgeon never got (8); she was one of the measured majority (9); and the summary her portal
  should have shown (12). Each is one to four sentences inside the node, never a digression.
- *The program as fallible protagonist:* seeds at nodes 1, 6, 9, 10 so node 10's
  self-falsification is a character's defining scene, not an FAQ entry.

**Interaction grammar.** Four predict-before-reveal moments (W1 end of node 2, W2 node 6,
W3 node 9, W4 node 10) — the highest-leverage teaching device, always commit-before-reveal.
A phase-labeled progress ribbon (The Ceiling 1–6 · The Instrument 7–9 · The Receipts 10–11
· The Ask 12) that stamps a 3–5 word answer under each visited node. A hero receipts strip.
Deep links from every Lean citation to pinned-commit source. Details in Part 3.

---

## Part 2 — The flow

### Node 0 · Cold open (context + motivation; no question yet)

**[always visible]**

`[render: CUT at build (Drake) — the ghosted composite-scene aside and the "Cold open"
eyebrow were removed; the scene opens the page directly after the hero. NOTE: the
composite-scene honesty disclaimer now appears nowhere on the page.]`

The result reaches her through the patient portal on a Tuesday: the tumor is malignant.
Beneath it, in bold, the reason flagged as mattering most — **Primary factor: the
irregular shape of the cell nuclei.** Her surgeon reads the same line, and it shapes the
conversation they have next: how wide to cut, how aggressively to treat.

Weeks later, before a follow-up, she opens the portal again. The diagnosis is unchanged —
malignant, as before. But the summary has been regenerated, and the bold line now reads
differently: **Primary factor: the size of the cell nuclei.** Not their shape. Their size.

She calls the clinic. Someone checks: no error, no correction, nothing in her chart
changed. The finding was never in doubt — only the reason beneath it moved. And after the
system's next routine refresh, a third reason would be waiting.

The verdict is solid. The *explanation* printed beneath it is, in effect, a coin being
flipped somewhere she cannot see.

Her first thought is yours: someone slipped — sloppy paperwork, a glitch, the kind of thing
that gets caught and fixed.

It doesn't get fixed. And to see why not, you have to know *what* wrote that line.

`[interaction]` The regenerated summary renders with its **Primary factor** line cycling
through the competing reasons; everything else on the document holds still. This is motif
M1's first appearance. Placement (Drake, post-build): the document sits BELOW the "Weeks
later" regeneration paragraph — the reader meets the shape→size flip in prose first, then
sees the living document; its initial state is "the size of the cell nuclei," cycling
size → texture → shape, which dramatizes the "third reason would be waiting" line that
follows. The progress ribbon appears after this section, phase-labeled, all twelve
questions visible.

*Formal-layer note (available on the cold open too):* **Proven** ·
`explanation_impossibility` · ExplanationSystem.lean:68 · Monograph §582. **Measured** ·
the per-patient reversal · `results_clinical_decision_reversal_v2.json` · Monograph §2649
(Per-Individual Explanation Reversal); see node 9.

---

### Node 1 · "Isn't that just bad practice? Wouldn't more data or better tools fix it?"

**[always visible]**

A machine wrote it — though not the diagnosis. The malignant call came the usual way, a
pathologist confirming it under the microscope, which is exactly why it never wavered. What
the machine added was the *reason*: a decision-support model scores the case and flags which
measurement most drove its own read, and that flagged line is what her care team saw. It
changed because, between her visits, the model had been refit on a refreshed sample of
cases — the routine revalidation every deployed clinical model goes through — and a model fit
to a different draw of the data settles on a different, equally good answer to "which
measurement mattered." And here is the part that is not a glitch: even with no update at all,
a second model built from the same data and just as accurate could have flagged a different
reason from the start.

That someone-slipped instinct — the people who built this program shared it — is wrong in an
interesting way. These systems agree about *what* while disagreeing about *why*: they make
the same predictions — that's what "equally accurate" means — but reach them through
genuinely different internal structure. Statisticians call this the **Rashomon effect**,
after the Kurosawa film in which witnesses give incompatible accounts of the same event:
many different models fit the data equally well and disagree about the reasons.

More data helps less than you'd hope, because the effect isn't caused by ignorance. It's
caused by arithmetic: modern models have far more internal settings than the data can pin
down. When a system has more knobs than the evidence constrains, many knob-settings produce
identical behavior — and the explanation depends on the knobs, not just on the behavior.
Better tools can't fix it either, because they're being asked to report a fact ("the
reason") that the situation does not uniquely contain. And freezing the model — never
retraining — doesn't rescue it: the reason it happens to be frozen on is still just one
arbitrary draw from the many the data allows equally well.

**[▸ go deeper]**

The Rashomon effect was *introduced to statistics* by Leo Breiman in his 2001 "Two
Cultures" paper — the one that split the field into the data-modeling and
algorithmic-modeling camps. The post-2016 explainability boom (SHAP, LIME, and their kin)
is the algorithmic culture answering its own opacity; the Rashomon effect is the crack in
that answer. Cynthia Rudin and collaborators built the modern study of *Rashomon sets* —
the set of all models within a tolerance of the best achievable accuracy — and showed these
sets are typically large and diverse on real problems. This program takes that as its
starting point and asks the next question: given that multiplicity is real, what *follows*,
provably, for explanation?

"More knobs than the evidence constrains" has a precise form: when configuration dimension
n exceeds the number of independent observational constraints m, the set of parameters
consistent with the observations is generically a space of dimension ≥ n − m — not a
handful of ties but a continuum of equally good models. Whether that is an edge case or the
generic condition is exactly what node 6 settles.

**[▸ formal]**

- **Proven** · `exists_observational_collision`, `finrank_ker_ge` ·
  UbiquityDimensional.lean:71,82 · axiom-clean · Monograph §883 (Generic
  Underspecification), §1007 (Prevalence). Rank–nullity yields, from every configuration, a
  genuinely distinct configuration with identical observations whenever n > m.
- Literature: Breiman, *Statistical Modeling: The Two Cultures*, Statistical Science 16(3),
  2001 (§"Rashomon and the multiplicity of good models"); Fisher, Rudin & Dominici, JMLR
  2019 (coins "Rashomon set"); Semenova, Rudin & Parr, FAccT 2022 (arXiv:1908.01755); Rudin
  et al., ICML 2024 position paper. The term itself predates Breiman (Heider 1988).
- Positioning for specialists: the program is agnostic on the interpretable-vs-post-hoc
  debate. Its interface covers any explanation operator — including reading structure off an
  inherently interpretable model — whenever the *model class* exhibits multiplicity. Nothing
  below argues against interpretable models; the trilemma prices what any explanation
  interface can promise under multiplicity, whatever the model family.

---

### Node 2 · "What would a trustworthy explanation even have to do?"

**[always visible]**

Before proving anything, pin down the target. Three properties, each so reasonable you'd
assume any decent explanation has all of them:

**Faithful.** The explanation reflects what the system actually does. If the model really
leans on cell size, a faithful explanation doesn't say cell shape.

**Stable.** Two systems that behave identically — same outputs on every input — get the same
explanation. If nothing observable distinguishes two models, their explanations shouldn't
differ. This is the exact promise the opening broke: nothing about the patient's case
changed, and every version of the model was equally accurate — yet the reason beneath her
diagnosis moved anyway.

It is not only medicine. Picture a second person we'll meet again: a loan applicant, denied,
handed the legally required "principal reason" for the decision. Her file, her score, the
denial itself are identical across every equally accurate model — the outcome never changes —
yet the *reason* on the letter does. Same broken promise, one rung up in stakes: here the law
obliges someone to write the reason down.

**Decisive.** The explanation commits. It names a factor, ranks the measurements, gives an
answer — rather than shrugging "it depends which equally good model you meant."

Each is intuitive alone. Each is achievable alone. The entire subject lives in the gap
between "each" and "all."

Notice what these three definitions never mention: neural networks, or machine learning at
all. That omission is deliberate, and it will turn out to be the most consequential choice
on this page.

`[interaction]` The three properties render as cards and persist as small icons through the
rest of the flow; nodes 3, 4, and 7 manipulate them (dimming the one being given up).
**W1 fires here**, after the cards: *"These three are about to collide. If you could keep
only two, which would you give up? [Faithful] [Stable] [Decisive] — your pick is remembered.
The mathematics has an opinion; you'll meet it at question 7."*

**[▸ go deeper]**

The framing is deliberately interface-level: a *configuration* space (models, parameter
settings), an *observation* map (what behavior you can see), and an *explanation* operator
(anything from SHAP values to a causal graph to a paragraph of prose). Faithfulness,
stability, and decisiveness are properties of the operator relative to the observation map.
Because nothing about neural networks — or ML at all — is assumed, whatever we prove applies
to any system fitting the interface: that is what makes the result a meta-theorem.

*For statisticians:* this is kin to non-identifiability — the lineage from Fisher through
Koopmans's identification problem — but not identical. Identifiability asks whether the
*parameters* can be recovered from data. The trilemma prices what any *explanation operator*
can promise when they cannot. The multiplicity is the old news; the theorem about the
operator is the new part.

**[▸ formal]**

- **Proven** · the `ExplanationSystem` structure (ExplanationSystem.lean)
  carries configurations, observations, explanations, and an incompatibility relation, with
  `faithful` / `stable` / `decisive` as predicates on the operator. The Rashomon property —
  two observationally equivalent configurations whose native explanations are incompatible —
  is a *structure field* (a hypothesis), never an axiom. Monograph §469 (Explanation
  Systems), §507 (The Three Desiderata), §557 (The Rashomon Property).

---

### Node 3 · "So can an explanation be all three?"

**[always visible]**

No. This is the theorem the program is named for.

If the Rashomon property holds — two setups that behave identically but whose honest
explanations disagree — then no explanation method can be faithful, stable, and decisive at
once. The proof is four moves, and its shortness is the point: nothing exotic is hiding in
it.

1. Two configurations behave identically, but their honest explanations disagree.
2. *Decisive* means the method must commit to an answer for each — no abstaining.
3. *Stable* means those two answers must be the same one — the systems are outwardly
   identical.
4. *Faithful* means that one shared answer cannot contradict either system's real structure
   — but the two structures make incompatible demands. Contradiction. ∎

This is the **explanation trilemma**: faithful, stable, decisive — pick two.

Hold the alarm: "pick two" is not a dead end. *Which* two, and what you get for the trade,
is where this is heading — and it has an exact answer.

`[interaction]` Motif M1: the three competing pathology summaries fan out beside the three
property cards — the visual claim that one card must go. Whatever reason the model prints,
it is now making a promise the mathematics says no single reason can keep.

**[▸ go deeper]**

The right cultural reference is Arrow's impossibility theorem: no voting system satisfies a
short list of reasonable fairness properties at once — and that didn't end social choice
theory, it *founded* it, by replacing "which system is best?" with "which combinations of
properties are achievable?" The trilemma plays the same role. "Which explanation is right?"
becomes "which explanation properties do you want, at what price?" — and that question has
exact answers.

Note what the theorem does *not* say. Not that explanations are useless, not that all are
equally bad, not that the situation is hopeless. It says the three cannot be had *jointly* —
which raises the two questions the rest of the page answers: is the Rashomon hypothesis
common or contrived (node 6), and what is the best you can do inside the ceiling (nodes
7–8)?

**[▸ formal]**

- **Proven · zero axioms** · `explanation_impossibility` · ExplanationSystem.lean:68 (body
  71–75, a five-line proof) · `#print axioms` empty beyond Lean's kernel · Monograph §582
  (Main Theorem), full statements in Appendix §4932.
  **Statement (shape):** for any explanation system `S` (configurations, observations,
  explanations, incompatibility `⊥`) and operator `E`, if `∃ c₁ c₂` with `obs c₁ = obs c₂`
  and `native c₁ ⊥ native c₂` (the Rashomon field `S.rashomon`), then
  `¬(Faithful S E ∧ Stable S E ∧ Decisive S E)`. Rashomon enters as a hypothesis, not a
  built-in assumption.
- **Proven** · strengthening — for maximally incompatible systems, faithful + stable *alone*
  is already impossible (the **explanation bilemma**) · `bilemma` ·
  MaximalIncompatibility.lean:96 · Monograph §687.
- Literature: Arrow, *Social Choice and Individual Values* (1951); the analogy is developed
  in Monograph §4444.

---

### Node 4 · "Wait — did you just rig the definitions?"

**[always visible]**

The sharpest objection, and the first a mathematician raises. This is the first of three
ways you might try to break the result — the definitions, the proof, the hypothesis — taken
in order. Attack one: maybe "faithful," "stable," and "decisive" were secretly defined so
strongly that of course they conflict. If so, the theorem is a card trick.

The check is whether each *pair* is achievable. It is — all three. There is an explanation
method that is faithful and stable (the **abstainer**: it declines to commit rather than
contradict itself). One that is faithful and decisive (the **committer**: it commits and
accepts that equally good models get different answers). One that is stable and decisive
(the **stonewaller**: it commits to the same answer everywhere, at the price of contradicting
some systems). Each is constructed concretely and machine-checked.

So the impossibility cuts *exactly* at the triple. The definitions are not too strong — they
are exactly strong enough that removing any one property dissolves the conflict. That's the
difference between a theorem and a tautology.

**[▸ go deeper]**

A further refinement: which pairs are achievable depends on the structure of the explanation
space — specifically whether it contains *neutral* elements (safe non-answers) and
*committal* ones. The classification theorem characterizes achievability in these terms,
which is why the theory can tell you, for *your* explanation format, which trade-offs are
even on the menu. Adding a neutral element (enrichment) restores faithful + stable at the
cost of decisiveness, and the enrichment is unique on the Rashomon fiber.

**[▸ formal]**

- **Proven** · pairwise witnesses `tightness_faithful_decisive`, `tightness_faithful_stable`
  · ExplanationSystem.lean:85,97 (the stable+decisive witness alongside) · Monograph §635
  (Tightness).
- **Proven** · neutral/committal classification `tightness_full` / `tightness_collapsed`,
  and `enrichment_unique_on_fiber` · BilemmaCharacterization.lean, MaximalIncompatibility.lean
  · Monograph §782 (The Enrichment Mechanism).

---

### Node 5 · "Why should I believe a proof I can't check?"

**[always visible]**

The second attack aims at the proof itself — and the answer is that you don't have to
believe anyone. That is, concretely, the point of how this program is built.

Every theorem here is written in **Lean**, a proof assistant: a language for mathematics in
which a proof either passes the checker or does not compile. There is no "the referee was
convinced." A small trusted kernel — the same few thousand lines for everyone — accepts the
proof or rejects it. Human error can still live in one place, the *definitions* (are these
the right formalizations of the words?). It cannot live in the reasoning.

What machine-checking buys is certainty about the mathematics. What it does *not* buy is any
guarantee about the world: whether real models and datasets satisfy the theorems'
hypotheses is an empirical question — and this program treats it as one, with its own
machinery (node 8) and its own evidence (node 9), reported so that *reality gets a vote*.

The headline theorems assume nothing beyond the logic itself. Across the whole program there
are exactly four declared assumptions — all of them housekeeping for applied side-studies of
one model family. The impossibility theorem, the repair theory, and every certificate
guarantee depend on none of them.

The current state, checked automatically on every change: **130 files, 715 theorems, zero
unproven gaps** in the main repository.

But one thing the checker cannot do: it certifies the reasoning *from* the hypothesis, never
the hypothesis itself. Every theorem so far is an if–then, and the machine guarantees only
the *then*. The *if* is your next question.

**[▸ go deeper]**

Machine-checked proof has its own history. The 1976 Four Color theorem was the first major
result to rest on computation no human could survey — philosophers argued whether it even
counted as proof. When Thomas Hales proved the Kepler conjecture in 1998, the referees
announced they were only "99% certain," so Hales spent over a decade building Flyspeck, the
full formal verification, completed in 2014. Today Lean's Mathlib carries much of the
undergraduate mathematical canon and substantial graduate material in machine-checked form.
This program is downstream of that arc — with one difference: it never passed through the
"99% certain" stage at all.

To distrust a result here you must either distrust Lean's kernel (shared with hundreds of
contributors and thousands of users formalizing unrelated mathematics) or argue that a
*definition* fails to capture its informal concept. The latter is a legitimate scientific
argument — it's why the definitions are stated in full and why node 4's tightness results
matter.

**[▸ formal]**

- Trust base, in full: Lean kernel axioms (`propext`, `Classical.choice`, `Quot.sound`) +
  two declared structures (`gbdtWorld`, `gbdtAxioms`, Defs.lean — the GBDT world model, used
  only by the GBDT quantitative layer) in the main repo; two analogous declarations in the
  attribution companion; zero in the physics companion (one theorem there leans on an
  exhaustive machine enumeration — `native_decide`, a 419,904-case circuit; claims.tex
  `ClaimOstrowskiNativeDecide{1}`, Monograph:1754).
- Per-repo counts, never summed (repos overlap): main **130 files / 715 theorems / 2 axioms
  / 0 sorry**; attribution companion **368 / 2**; physics companion **481 / 0**;
  program-wide axioms **4**. Authoritative source `paper/claims.tex` (regenerated by
  `paper/scripts/gen_claims.py`; overlap `ClaimOverlapDashPct{88}`,
  `ClaimOverlapOstrowskiPct{11}`).
- **Proven** · a CI Tier-A audit re-derives every spine theorem's axiom set on each commit
  (`#print axioms`, `make verify`); an overclaim in a file header is a build failure, not a
  footnote. The vendored Morse–Sard port (node 6) is attributed to its author (Kudryashov,
  Apache-2.0) and *excluded* from all counts — Sard/ATTRIBUTION.md. Monograph §1724
  (Lean Formalization), §1799 (Axiom Stratification), §1904 (Proof Status Transparency).
- Literature: Lean kernel trust model (Mathlib overview, arXiv:2508.21593, 2025); Appel &
  Haken 1976 (Four Color); Hales et al., *Forum of Mathematics Pi* 2017 (Flyspeck).

---

### Node 6 · "Fine — but how often does this actually happen? Is it rare, or everywhere?"

**[always visible]**

The second attack failed; here is the third, and strongest. Everything so far is
conditional: *if* two equally-behaving systems disagree about why, *then* no explanation can
do all three jobs. A skeptic should ask whether that hypothesis is a laboratory curiosity.

`[interaction]` **W2 fires here**, before the answer: *"Two models. Identical behavior.
Different reasons. How often would you guess that happens? [A rare pathology] [Only with
sloppy practice] [Common] [The typical case — nearly unavoidable]"*

The answer is a ladder of four increasingly strong results, ending at the strongest:
whenever a system has more internal settings than the data constrains — essentially every
modern machine learning model — the Rashomon situation is not just possible but *generic*.
Not "can happen": *typically happens, at almost every configuration you could land on.*

The last rung has a story. For most of the program's life it carried a confession in print:
one step of this argument, the papers said, is argued, not machine-checked. That step needed
a classical theorem of higher geometry that had never been fully verified in a proof
assistant. Exactly one formalization of it existed anywhere, built by another mathematician
for unrelated reasons. In July 2026 the step was closed — by adapting that work (credited
throughout, and deliberately excluded from this program's own theorem counts) and verifying
the whole chain end to end. The confession is gone; the ubiquity argument is formal from
first premise to final conclusion.

And the same wall appears far beyond machine learning — in physics, where different
mathematical descriptions yield identical experimental predictions; in genetics, where
different DNA spellings build the same protein; in crystallography, where the same
diffraction pattern fits many different molecular structures. These are not loose analogies.
In each, a minimal witness is checked, in the same system, as an instance of the same
interface.

**[▸ go deeper]**

The four rungs, each subsuming the last:

1. **Linear** — rank–nullity forces a subspace of observationally identical configurations
   through every point when n > m.
2. **Infinitesimal** — any differentiable observation map has, at every point, a direction
   its derivative cannot see.
3. **Local smooth** — at any regular point of a nonlinear observation map, the true curved
   fiber contains genuinely distinct configurations arbitrarily close (implicit function
   theorem).
4. **Generic** — for *almost every* observable value the fiber is a positive-dimensional
   Rashomon locus. This is where the **Morse–Sard theorem** enters: the critical values,
   where the argument could fail, form a measure-zero set, at the sharp classical smoothness
   threshold (Whitney's 1935 counterexample shows the threshold cannot be weakened).

Two boundary results sharpen rather than qualify this. In *infinite-dimensional*
configuration spaces (function-space models) every rung except Sard extends — with
rank–nullity strengthening to "the Rashomon fiber has full rank": underspecification there
is total. And Sard's failure to extend is itself a theorem, not a gap: Kupka's 1965
counterexample shows it is false in that regime, and the obstruction (no such map is
Fredholm) is machine-checked. Philosophers will recognize the shape: Duhem (1906) and Quine
(1951) argued that theory is underdetermined by evidence — the ubiquity ladder is that
thesis made exact for a definite interface, and Duhem–Quine underdetermination is itself one
of the fourteen formalized instances. The century-old worry did not just inspire the
theorem; it is *covered* by it.

**[▸ formal]**

- **Proven** · `sardProperty_of_contDiff` (critical values of any C^{n−m+1} map are
  Haar-null, sharp threshold), `generic_ubiquity_of_contDiff` (generic ubiquity,
  unconditional) · MorseSard.lean:149,180 · via the vendored, attributed port of
  Kudryashov's formalization of Moreira's theorem (Apache-2.0; excluded from program counts).
  Monograph §872 (Ubiquity), §883.
- **Proven** · `regular_fiber_not_isolated`, `sardProperty_of_continuousLinearMap`,
  `sardProperty_of_submersion` · UbiquityDimensional.lean:178,351,375.
- **Proven** · `rank_ker_eq_rank_of_infiniteDimensional`; Kupka counterexample + non-Fredholm
  obstruction formalized · UbiquityInfiniteDimensional.lean:168 · Monograph:959.
- **Proven** · nine ML instances (`*InstanceConstructive.lean`: attribution, attention,
  counterfactuals, concept probes, causal discovery, model selection, saliency, LLM
  self-explanation, mechanistic interpretability) + fourteen cross-domain instances (Arrow,
  Peres–Mermin contextuality, Duhem–Quine, gauge theory, statistical mechanics, genetic code,
  phase problem, …), each a zero-axiom witness of the same interface. Monograph §3611.
- Literature: Moreira, *Publ. Mat.* 45 (2001) 149–162; Kudryashov, SardMoreira formalization
  (2025); Kupka, *Proc. AMS* 16 (1965) 954–957; Whitney, *Duke Math. J.* 1 (1935) 514–517
  (proves the (2,1) case; general-(n,m) sharpness rests on later extensions). Honest
  hypothesis note: the final wiring assumes the explanation map is *non-degenerate* — a
  theorem-shaped boundary (the necessity results, Necessity.lean / NecessityBiconditional.lean),
  not a loophole: explanation is possible precisely for explainers blind to the structure the
  trilemma is about.

---

### Node 7 · "If no explanation can do all three jobs, what's the best one *can* do?"

**[always visible]**

The trilemma says pick two. The constructive half says *which* two — and what you get in
exchange.

Give up decisiveness. Keep faithful and stable. Concretely: instead of asking one model for
the reason, train many equally good models and **average their explanations**. What all of
them agree on survives the averaging; what they merely disagree about cancels. The survivor
is the **stable core** — the part of the explanation that was never in dispute between models
the data cannot tell apart.

This isn't a heuristic patch; it is provably canonical. Among all stable explanations, the
average is the closest to what any individual model would have said — and nothing outside the
stable set is ever stable at all. The part that cancels was never trustworthy: it was the
signature of which arbitrary model you happened to train, not of the data.

For the patient in the opening, the stable core is concrete. Every equally good model agrees
her cell nuclei are abnormally *large*; they disagree only on how to describe their shape.
The honest version of her summary reports the size and flags the shape as undetermined —
faithful and stable, at the price of one tidy "primary factor."

Physics reached this same move a century ago, in its own language. The same physical
situation can be written with infinitely many different mathematical descriptions, all
producing the identical measurable result — a built-in ambiguity with no fact of the matter
about which description is "the real one." Physicists don't agonize over it; they long ago
agreed to quote only the quantities every description shares, and to treat those as the
physics. Keep what all equally valid descriptions agree on, discard the rest: that is the
stable core exactly, and physics turns out to be one instance of it — not the other way around.

For a while, the people behind this theory believed they could go further still — not just
recover the stable core, but predict the exact size of the disagreement in advance, from
symmetry alone. Hold that thought.

`[interaction]` Motif M1 reaches its thesis frame: the competing pathology summaries
**superimpose** — shared text stays dark, the differing "Primary factor" lines fade to ghost.
The stable core is literally what survives superposition. The orbit-average canvas renders
alongside (faint arrows fanning out, bold arrow emerging), and the reader's W1 pick is
echoed: *"You chose to give up [X]. The theory's answer is decisiveness — and this is what
you buy with it."* (If the reader chose decisiveness at W1: *"You agreed with the
mathematics — here is what that choice buys, and what it costs."*)

**[▸ go deeper]**

The mathematics is symmetry theory. The equally good models are related by a group of
transformations the observations cannot see; averaging over that group (the orbit average,
or Reynolds operator R) projects any explanation onto the invariant subspace V^G. Three
structural facts make this canonical:

- **Stability = invariance, exactly.** The stable explanations are precisely the G-invariant
  ones — V^G is *where all stable answers live*.
- **Best approximation.** R(v) is the nearest point of V^G to v. (The invariant aggregate as
  minimum-variance answer is the Rao–Blackwell / Hunt–Stein principle, not Gauss–Markov —
  symmetrize a sufficient statistic to reduce variance.)
- **The decomposition.** V = V^G ⊕ ker R: every explanation splits orthogonally into a stable
  core plus irreducible disagreement, and total explanation content is conserved between them.

This yields a budget: the **explanation capacity** C = dim V^G counts how many independent
stable facts a symmetry class can support, before any data is seen. DASH (ensemble-averaged
attributions) and CPDAGs (in causal discovery) are methods the community already built and
trusts — now identifiable as instances of this one canonical projection, which explains *why*
they work and says they cannot be improved upon within their family.

**[▸ formal]**

- **Proven · zero axioms** · `stable_iff_gInvariant` · StructureTheorem.lean:50 · Monograph
  §1024 (The Universal Resolution).
- **Proven** · `gInvariant_stable` · UniversalResolution.lean:41; `reynolds_best_approximation`
  · UncertaintyFromSymmetry.lean:222; `explanation_structure_theorem` (V = V^G ⊕ ker R) ·
  StructureTheorem.lean:77. Monograph §1137 (DASH), §1161 (Orbit Averaging), §1199
  (Uncertainty Bound).
- **Proven** · design-space dichotomy — every method either abstains on symmetric structure
  or commits and violates stability (Family A or B, no third way) ·
  `universal_design_space_dichotomy` · UniversalDesignSpace.lean:27.
- Optimality-frontier note (honest): DASH's *within-group* Pareto optimality is proven under
  the declared GBDT bundle; the *global* claim was machine-checked as a reduction to one named
  classical hypothesis (`CramerRaoScoreProperty`, node 11), the abstract Cramér–Rao inequality
  (`abstract_cramer_rao`, ParetoGlobal.lean:143) proven unconditionally. That hypothesis is now
  discharged for every natural exponential family (`cramerRaoScoreProperty_tilted`) and —
  end-to-end and axiom-clean — for the M-sample exchangeable-Gaussian ensemble DASH operates on
  (`dash_mvue_exchangeableGaussian`, ScoreRegularityExchangeable.lean): DASH is the
  minimum-variance unbiased estimator there. Only general non-exponential families remain the
  named hypothesis. Rao–Blackwell bridge: Monograph §3785. CPDAG: Verma & Pearl 1991.
- **Proven · zero axioms** · the physics precedent for the stable core, as a minimal witness:
  gauge freedom — distinct field configurations with identical observable holonomy, and gauge
  transforms that preserve it · `GaugeTheory.lean` (`same_holonomy`, `different_configs`,
  `gauge_preserves_holonomy`, all `by decide`). The "keep only what every description agrees
  on" move of the plain layer is the gauge-invariant projection; formally it is the same
  G-invariant subspace as `stable_iff_gInvariant`.

---

### Node 8 · "OK — the explanation in front of me, right now: which parts can I trust?"

**[always visible]**

Node 7 gave you the stable core — what all the equally good models agree on. The
**certificate** answers the practical follow-up: is the specific claim in front of you —
"measurement A matters more than measurement B, for this patient" — actually inside it?

The recipe is almost embarrassingly practical. Retrain the model several times (or use an
ensemble you already have). Look at how consistently the ensemble endorses the claim relative
to how much it wavers: a signal-to-noise ratio. A one-line classical inequality then converts
that ratio into a *guarantee* — an upper bound on how often the claim can flip across equally
good models, with no assumptions about distributions or model internals.

A claim certified at the standard threshold flips on at most one in five equally good models,
unconditionally — call it the one-in-five guarantee. And the certificate is honest about its own direction: it can prove a claim
*stable*, but never prove one *arbitrary* — a deliberate one-sidedness, proven, so the
instrument cannot be used to condemn.

This is what the opening was missing. Had her summary carried this stamp, her surgeon would
have known whether "cell size, not shape" was a dependable reason to act on or a coin flip —
and could have said which, out loud, instead of trusting one line from one model.

The loan applicant's case shows the other half of why this matters. When the law makes a
lender state the principal reason for a denial, a compliance officer has to attest to it —
and today that signature vouches for a reason an equally good model would have contradicted.
The certificate is what lets the officer sign honestly: it marks which reasons hold across
the whole equally-good set and flags the ones that are a coin flip. (The denial itself
doesn't move — only whether its stated reason is one you can stand behind.)

Everything to this point is mathematics — certain, certified, and entirely silent on whether
the world cooperates. Reality has not voted yet.

`[interaction]` Motif M2: the arrow-fan from node 7 is reused — the fan's spread *is* the
noise; SNR is fan-tightness per claim; a certified claim renders as a tight fan.

**[▸ go deeper]**

Three theorems make this an instrument, not a rule of thumb:

- **The bound**: flip rate ≤ 1/(1 + SNR²), via Cantelli's inequality — distribution-free.
  SNR ≥ 2 gives the ≤ 20% guarantee.
- **Tightness**: no certificate reading only mean and variance can promise less; the bound is
  achieved.
- **Transfer**: the number you compute on *your* ensemble today provably transfers to an
  independent ensemble tomorrow, at an explicit exponential rate — and a second version drops
  independence entirely, covering correlated and exchangeable ensembles. The in-sample
  certificate becomes an out-of-sample guarantee.

**[▸ formal]**

- **Proven** · `cantelli_lower_tail` · CertificateGuarantee.lean:71; `flip_bound_tight`,
  `flip_zero_at_low_snr` (one-sidedness) · CertificateTight.lean:43,121. Monograph §1286
  (Group-Theoretic Classification), lines 1600–1675.
- **Proven** · `transfer_flipFreq_bound` (Hoeffding rate) · TransferTheorem.lean:90;
  `transfer_flip_general` (no independence needed) · TransferGeneral.lean:37.
- Literature: Cantelli's inequality (one-sided Chebyshev); Hoeffding, JASA 1963 (transfer
  rate).

---

### Node 9 · "Does reality actually obey any of this?"

**[always visible]**

The theorems are certain; whether the world sits inside their hypotheses is not. So the
program tests itself — in a specific mode: its quantitative claims are distribution-free
*bounds*, and a bound is validated by **coverage** (did reality stay inside it?), never by
curve-fitting. A bound reality escapes is falsified. That mode is why the record below is
evidence, not decoration.

`[interaction]` **W3 fires here**: *"The opening followed one patient. Among breast-cancer
patients with a confirmed malignant tumor, for what fraction does a 94%-accurate
decision-support model's single 'most important measurement' change across equally accurate
models? Drag
to guess: 0–100%."* The reveal renders the reader's guess *beside* the feature-level number —
up to 100% — with the delta labeled; the cluster-level 26–42% lands in the next breath as the
honest turn. Motif M1 at scale: behind the numbers, a wall of summaries renders — most
shimmering between versions, a still minority holding fast.

**Medicine — where the opening lives.** On the Breast Cancer Wisconsin data (569 patients,
models at 94% accuracy / 0.98 AUC), within a 2%-accuracy Rashomon band, **up to 100% of
patients get a different top measurement** across equally accurate models — the opening's
patient was one of them. But the honest turn *raises* the rigor: cluster the correlated
measurements (worst radius, perimeter, area are three rulers on one thing — the size of the
cell nuclei) and the reversal drops to **26–42%**. Much of the churn is size-features trading
rank within one
meaning — which is exactly the Rashomon symmetry the theory predicts, and exactly what the
stable core recovers. And a negative control the theory demanded: on a dataset where the
models are identical (nothing left undetermined), the reversal is **0%** — instability
appears precisely when, and only when, multiplicity does.

**Law — where the reason is binding.** The *decisions* never change (every model in the
window is equally accurate); it is the *stated reason* that flips. On German credit within a
strict 2% window, **84% of denied applicants** (76–93% across five seeds) have a principal
adverse-action reason that flips across equally good models. Remember the coin flip nobody
knew they were flipping? For 84% of denied applicants on this benchmark it is real — and for
about 19%, the certificate can prove there was no flip at all. Lending is where the law forces
the reason into writing, which is what makes the flip auditable at scale; the clinic is where
the same arithmetic is heading.

**The certificate holds out of sample.** Across 104 real datasets and 10,724 explanation
claims, every one of the **4,431** claims the certificate called stable stayed stable on a
fully independent ensemble — 100% transfer. Observed flip rates stayed below the
machine-checked bound on 99.4% of pairs. The two weakest datasets are named in the paper, not
hidden.

**[▸ go deeper]**

The wider program: a 149-dataset capacity audit across 53 domains (75% exceed the capacity
threshold; cross-dataset effect at p ≈ 5×10⁻¹¹); mechanistic-interpretability studies across
ten transformers (explanation correlation 0.52 → 0.93 after projecting to the invariant
core); the same instrument on GPT-2-small's sparse-autoencoder features certifying the ~9.5%
that reproduce across retraining (0.81 cosine out of sample vs 0.32 for uncertified) —
exploratory, one model, one layer. One honesty note the program insists on: the credit 84% is
*observed* instability, not a certified impossibility — by the one-sidedness theorem the
instrument licenses "provably stable," never "provably arbitrary." And preregistration is the
norm the replication crisis forced on empirical science, here applied to the program's own
predictions: a **prospective round** is frozen and time-stamped, not yet run. It can still
fail — and node 10 shows the program means it.

**[▸ formal]**

- **Measured** · certificate transfer: 104 datasets / 10,724 claims / 4,431 certified-stable
  / 100% transfer / 99.4% within bound / median OOS rank corr 0.822 / calibration error 0.009
  · `results_large_scale_certificate.json` (`band_table_insample.STABLE.n_pairs`) · Monograph
  §2719.
- **Measured** · breast-cancer per-patient reversal: feature-level up to 1.00, cluster-level
  0.26–0.42 (12 clusters at corr 0.8); heart-disease negative control 0.00; seed-only 0.00 ·
  `results_clinical_decision_reversal_v2.json` · Monograph §2649 (Per-Individual Reversal).
- **Measured** · credit adverse-action: 84% (76–93%, 5 seeds); ~19% certifiable minority ·
  Monograph §3545 (OpenML-31, 60 GBDT models/seed). One-sidedness caveat inline.
- **Measured** · capacity audit 149 datasets / 53 domains / 50 seeds / 75% / p = 5.09×10⁻¹¹ ·
  `results_audit_150_final.json` · Monograph §2548. Mech-interp 0.518 → 0.929 ·
  `results_mi_v2_final_validation.json` · Monograph §3095. SAE (GPT-2-small, layer 6): 9.5% /
  0.81 / 0.32 · Monograph §3242.
- **Measured** · prospective round frozen/unrun, commits to publish either way ·
  `preregistration_prospective_validation.json`, OSF_PREREGISTRATION_DRAFT.md (link inline).
- Literature: ECOA/Reg B 12 CFR 1002.9; EU AI Act Art. 86; Semenova & Rudin 2022 (Rashomon
  ratio context).

---

### Node 10 · "Has the theory ever been wrong?"

**[always visible]**

Before you read the answer, one question worth committing to.

`[interaction]` **W4 fires here, before any answer:** *"Early on, the program proposed an
actual formula — a law — predicting how much explanations would disagree, from symmetry
structure alone: the very step question 7 asked you to hold onto. It preregistered the test
before running it. What happened? [It held] [It
partly held] [It failed]"*

It failed. Three times, three different ways.

The program had put its chips down: a quantitative **law of disagreement**, predicting the
*magnitude* of instability, with thresholds frozen and time-stamped before the data was
touched. The holdout came back dead on arrival. Five different estimation strategies each
failed to recover the law from data. A salvage reframing failed too. The program's own
documents report the law as thrice-falsified and closed. What survived is the lesson:
reframed as a *bound* rather than a point prediction, the same quantity holds with 100%
coverage. That became the program's epistemic identity — **bounds, not laws** — and it is why
every "Measured" mark on this page reports coverage of a bound, never a fitted curve.

It happened again, at smaller scale, in the summer of 2026 — and this time the theory
falsified *itself*. While closing its open problems, the program asked its own formalization
to confirm a claim its own published file header had made. The kernel disagreed. The
counterexample compiled; the header was corrected in the same change. Two of its own claims
fell that way — the code's optimism lost to the monograph's own concession — and both were
replaced by sharper true statements. Honesty here is not a policy; it is a property the build
system checks.

That is why, when *this* program marks something **Proven**, the mark has survived a system
that demonstrably kills its own claims.

**[▸ go deeper]**

The full falsification ledger, because a boundary is content: five further preregistered
predictions failed and are reported as such — a phase-transition location, a linear
"uncertainty budget," a character-theoretic prediction for molecular evolution, a spectral-gap
convergence rate, and a correlated-flips prediction. Each marks where the proven core stops
and speculation began. The grandest analogies the program is occasionally tempted by (Gödel,
Langlands, quantum error correction) are quarantined in an explicitly conjectural tier, with
the genuine adjacent theorems stated separately from the interpretations they inspire.

**[▸ formal]**

- **Retired** · the law of disagreement (η-law): preregistered holdout R² = 0.24; five
  estimators failed; salvage reframe failed; as a bound, 100% coverage ·
  `results_eta_law_oos_gof.json` · Monograph Part V §4806 (Boundary Conditions), §4817.
- **Proven (self-refutations)** · `mi_converse_fails_without_bridge`,
  `mi_exact_boundary_biconditional` · MIConverse.lean:300,173; `gstable_not_image_in_fixed`,
  `linear_sigmaStable_image_not_in_diag`, `gstable_equivariant_image_subset_fixed`
  (axiom-free) · TierBResolutions.lean:124,178,101. Corrected headers: MutualInformation.lean,
  MIQuantitativeBridge.lean.
- **Retired** · five preregistered misses · Monograph §4882 (Summary), reported alongside the
  confirmations. Grand analogies quarantined: Monograph Part IV Tier C §3895+.

---

### Node 11 · "What's still open — and what would prove you wrong?"

**[always visible]**

The theory has made a bet it cannot take back: a preregistered prospective round, frozen and
time-stamped, not yet run, committed to publication whatever the outcome. If it lands outside
the bounds, the bounds are wrong, and the program has promised to say so. That is the live
tension at the frontier.

Beyond it, four mathematical problems remain, each named with the reason it resists. One is a
textbook regularity condition that classical statistics assumes silently — one this program
has now settled for the standard families of statistics, including the exact setting the
averaging recipe of question 7 runs in; it stays open only beyond them. Two are
conjectures whose
proofs await machinery that does not exist yet; in each case the missing prerequisite is
named. One is a suspected bridge to quantum error correction that lacks a statement crisp
enough to attack — sharpening comes before proving. Names, files, and the exact obstruction
for each: one layer down.

**[▸ go deeper]**

The four items:

- **The score property for general families** — the last "argued" step in DASH's global
  optimality: differentiation under the integral, the condition classical statistics assumes
  silently. Isolated as a single named hypothesis and now discharged for every natural
  exponential family (via machine-checked differentiation under the integral), and verified
  end to end for the M-sample exchangeable-Gaussian ensemble DASH actually uses, the whole
  chain resting on Lean-core axioms alone — so DASH is proved the minimum-variance unbiased
  estimator there. What remains open is only the general non-exponential case (e.g.
  Cauchy-type families with no exponential moments).
- **The irrep-count conjecture** — a proposed exact count of non-dominated explanation
  profiles via representation theory; resists because a formal scaffold for the objects it
  counts doesn't exist yet.
- **Categorical enrichment** — that "add an abstain option" is a free construction in the
  category-theoretic sense; resists for want of 2-categorical machinery.
- **The quantum error-correction bridge** — a suspected dictionary between explanation
  stability and the Knill–Laflamme conditions; resists at an earlier stage, lacking a crisp
  statement.

Note the shape: each open item comes with a *named formal prerequisite* or a *named classical
assumption*, not a vague "future work." That is the program's export to formal science — the
reduction method: you don't need to formalize everything at once, you need to know *exactly*
what you haven't. It worked twice (Sard, then Cramér–Rao), converting "the argument is
standard" into "one named assumption, verified logic everywhere else."

**[▸ formal]**

- `CramerRaoScoreProperty` → `dash_global_pareto_optimal` reduction machine-checked;
  `abstract_cramer_rao` unconditional (ParetoGlobal.lean:143). The hypothesis is now
  DISCHARGED: for every natural exponential family (`cramerRaoScoreProperty_tilted` ·
  ScoreRegularityNEF.lean:115, built on `hasDerivAt_integral_mul_exp` ·
  ScoreRegularity.lean) and end-to-end for the M-sample exchangeable-Gaussian model
  (`dash_mvue_exchangeableGaussian` · ScoreRegularityExchangeable.lean:360 — DASH is the
  minimum-variance unbiased estimator; Lean-core axioms only, Tier-A audited). Open:
  general non-exponential families · SCORE_REGULARITY_SCOPING.md · Monograph §1934
  (proof-status item), §4625 (Limitations and Open Questions). **Conjectured** ·
  irrep-count (§1286), categorical enrichment, Knill–Laflamme bridge (§4477), each labeled
  as such.
- Literature: Cramér–Rao (Rao 1945; Cramér 1946); Knill & Laflamme, *Phys. Rev. A* 1997.

---

### Node 12 · "What should I do with this?"

**[always visible]**

Depending on who you are:

**If you deploy or audit models.** Stop asking one model for the reason. Use an ensemble of
equally good models; report the stable core; certify the claims you must stand behind;
disclose when no dependable reason exists. Every step of that sentence is a theorem or a
validated instrument above — and the reference implementation of the certificate ships in
the main repository's `knockout-experiments/` directory (deep link at build).

**If you regulate, or are regulated.** Here is the sentence you can paste: *adverse-action and
AI-Act explanation requirements are satisfiable honestly — require reporting of the
certified-stable reasons, and disclosure when no reason certifies.* Legally mandated
explanations (adverse-action notices under ECOA, deployer-level disclosures under EU AI Act
Art. 86, clinical decision support) sit under a proven ceiling, and on real data the reasons
for most individuals in fact flip across equally good models. The mathematics doesn't weaken
the case for explanation requirements; it makes them *satisfiable honestly* — it is what lets
the compliance officer of question 8 sign the reason without crossing her fingers.

**If you do research.** The interesting question stopped being "which explanation method is
best?" and became "which explanation properties do you want, at what price, and can you
certify what you kept?" The monograph, the three Lean repositories, and the open-problems
list are public; each open problem links to its tracked issue.

**If you just wanted the answer.** When many explanations fit the facts equally well — and
they usually do — the one you're given is partly an accident of the draw. That is now a
theorem, not a suspicion. But the part every equally good explanation agrees on *can* be
trusted — and a machine checked every step.

`[interaction]` Motif M1 closes: one final pathology summary, the **Primary factor** line now
the certified stable factor — or, honestly, "no single measurement is dependable here; the
stable finding is: malignant, and the cell nuclei are enlarged." Return to the opening's patient: this is
the summary her portal should have shown. And a third echo of the reader's W1 pick: *"At
question 2 you chose to give up [X]; here is the summary your choice would have written."* The stats
band renders below as a colophon.

**[▸ formal]**

A where-to-start map: Monograph Part I (the theorem, nodes 3–4) §582; Part I ubiquity (node
6) §872; resolution (node 7) §1024; certificate (node 8) §1286; empirical validation (node 9)
§2548; Part V falsifications (node 10) §4806; open questions (node 11) §4625. Repositories:
`universal-explanation-impossibility` (main), `dash-shap` (attribution companion),
`ostrowski-impossibility` (physics companion). Preregistration: OSF (linked). Closing counts,
per repo, from `paper/claims.tex`: **130 files · 715 theorems · 0 sorry · 4 axioms
program-wide · 100% certified-claim transfer.** Monograph §3580 (What Explainability Can Do),
§3476 (Code and Data Availability).

---

## Part 3 — Build notes (for the eventual page; not reader-facing)

- Replaces the walkthrough at its existing artifact URL; keep `<title>` and favicon stable.
  On-page H1 is "The Limits of Explanation," with "The diagnosis didn't change. The reason
  did." as the epigraph above it (see brief — the hero introduces no concept the page hasn't
  earned; "Faithful. Stable. Decisive. Pick Two." is the post-node-3 divider / share-card
  line, not the hero).
- **Hero receipts strip** — REDUCED to one chip (Drake, post-build): `5 preregistered
  predictions failed — published →`, linking the named **Falsification Ledger** anchored at
  node 10. The count chips (`715 theorems · 0 gaps`, `4 assumptions, program-wide`) were
  cut from the hero; those counts surface at node 5's stats band and the node-12 colophon.
  The surviving chip still front-loads the self-falsification differentiator above the fold.
  (The round-2 marketing concern — a skimmer misfiling the page as a medical-ethics
  anecdote — is now carried by the H1 "The Limits of Explanation" itself, which leads with
  the program framing.)
- **Four predict-before-reveal widgets** — W1 end of node 2, W2 node 6, W3 node 9 (persist the
  guess post-reveal to show the guess-vs-reality delta), W4 node 10 above the confession. All
  commit-before-reveal; no external calls (choice stored in-page).
- **Progress ribbon**: phase-labeled (The Ceiling 1–6 · The Instrument 7–9 · The Receipts
  10–11 · The Ask 12); stamps a 3–5 word answer under each *visited* node ("No — pick two",
  "Generic, and proved", "84% flip; ~19% certify"); pre-visit shows questions only.
- **Two visual motifs**: M1 the shifting reason-line on one document (0→3→7→9→12 — all five
  beats staged in node bodies), M2 the arrow-fan collapse (staged 7→8; optional seeds at
  nodes 1 and 10 at build discretion). The former M3 (frozen/churning columns) is cut as a
  motif; its content survives as the node-9 results-table treatment — accuracy column
  frozen, stated-reason column churning.
- **Deep-link every Lean citation** to pinned-commit source; link the CI axiom-audit run at
  node 5; link the OSF registration inline at node 9.
- **Handles**: "Certified Stable" stamp, "the ubiquity ladder", "the Falsification Ledger",
  "the one-in-five guarantee".
- **Reuse** the walkthrough's CSS system (serif measure, tier palette, claim blocks, orbit
  canvas, stats band) — theme-aware already. New components: receipts strip, phase ribbon,
  predict widget, per-node layer expandables, the document-motif renderer.
- **Re-verify every formal-layer number** against `paper/claims.tex` and the result JSONs at
  build time — the discipline that caught the 4,521→4,431 error in this pass. Counts current
  as of main @ c74d82d.
- Accessibility: expandables keyboard-navigable; predict widgets skippable; canvas + motif
  have text alternatives. Length target: plain spine ~15 min; fully expanded ~45 min.

---

## Changelog v2 → v3 (for the reviewer)

- **Cold open is now a purely human scene.** The setup names no models, no software, no
  accuracy figures — only a woman, a correct malignant diagnosis, and an official reason
  that silently changes when her summary is regenerated (same chart, no error). The "a
  machine wrote it, and a second equally-accurate machine would have written a different
  reason" reveal moves to the **node-1 transition**, where it recontextualizes the human
  scene as the Rashomon effect. Ends on the hook "to see why not, you have to know what
  wrote that line." Motif M1 (the one changing line on a fixed document) is preserved and
  arguably cleaner (one regenerated summary, not two centers).

## Changelog v1 → v2 (for the reviewer)

- **Cold open** rebuilt around breast-cancer diagnosis (real data; screenwriter craft + the
  fact that heart-disease is the negative control, so cardiac was ruled out). No announced
  turn, no synopsis; ends on node 1's question.
- **Three factual fixes**: "principal reason(s)" plural (ECOA); 4,521 → **4,431** certified-
  stable claims; `reynolds_best_approximation` → UncertaintyFromSymmetry.lean:222. Plus
  wording softens: "provably unstable" → "observed instability under a proven ceiling";
  "Gauss–Markov" → Rao–Blackwell/Hunt–Stein; "named by Breiman" → "introduced"; "four lines"
  → "four moves / five-line proof"; "thousands of mathematicians/graduate canon" → hedged.
- **Register fixes**: node 8 no longer depends on node-7's deeper-only "capacity"; node 11's
  plain layer no longer breaks the plain register; node 10 drops Greek from plain; node 5
  moves axiom bookkeeping out of plain; node 6 plain trimmed.
- **Threading**: document motif (M1) across 0/3/7/9/12; POV returns at 9 and 12; fallible-
  protagonist seeds at 1/6/9/10; the coin-flip and the applicant now pay off.
- **Rigor**: every formal layer upgraded to `theorem · File.lean:line · axiom status ·
  Monograph § · JSON/lit` per the fact-check citation map; node 3 now states the theorem;
  node 4 names the three witnesses; Cantelli/Hoeffding/Rao–Blackwell/Verma–Pearl/Cramér–Rao/
  Knill–Laflamme cited.
- **History**: machine-proof lineage (Four Color→Flyspeck→Mathlib) at node 5; Duhem–Quine
  ancestor-now-instance at node 6; identifiability kinship at node 2; Breiman Two Cultures +
  XAI boom at node 1; preregistration/replication-crisis at node 9.
- **Arc**: vision seeded at node 2 (the "never mention machine learning" line) and the hero
  receipts strip; gauntlet framing across nodes 4–6; header rewrites (nodes 6/7/8/11);
  H1/title split; node-12 CTAs given real artifacts; ending returns to the document, not the
  stats band.
- **Node 11 / node 7 frontier status** upgraded to main's verified state after M1–M3 merged
  (main @ c74d82d): the score property is discharged for every natural exponential family
  (`cramerRaoScoreProperty_tilted`) and, end-to-end and axiom-clean, for the M-sample
  exchangeable-Gaussian ensemble DASH operates on (`dash_mvue_exchangeableGaussian`) — DASH
  proved the minimum-variance unbiased estimator there; only general non-exponential families
  remain the named hypothesis. Counts refreshed to 130 files / 715 theorems; the monograph's
  proof-status item and Nature's two "global argued" clauses were updated to match (PR #27).

## Changelog: case-study expansion (5-expert plan audit → disciplined subset)

- Proposed 4 recurring + 2 walk-on through-lines; the panel unanimously restructured it
  DOWN (universality is a breadth effect earned by instances, not four mascots). Applied
  the convergent subset:
  - **One recurring third-person witness added — the loan applicant + compliance officer**
    (the unanimous "clear win": carries the rights/legal-mandate rung the patient can't).
    Staged as escalating stakes health → rights: node 2 (she's a second instance of the
    broken *stability* promise, with the "the decision never changes, only the reason" guard),
    node 8 (the officer must sign the certified reason — the accountability beat), node 12
    (the regulator CTA now names her officer). Patient stays the sole POV; the applicant is a
    witness, never a co-lead.
  - **Gauge freedom added as a node-7 plain resolution rhyme, glossed** (no "gauge/holonomy/
    invariant" in plain — "keep what every valid description agrees on"), framed "physics is
    an instance of ours, not 'this is just physics'"; Lean detail (`GaugeTheory.lean`,
    axiom-clean) in the formal layer, where the rhyme is literally the same G-invariant
    subspace as `stable_iff_gInvariant`.
  - **Node-6 sweep: dropped "voting / Arrow's paradox"** (double-duty with the node-3 Arrow
    analogy — three lenses flagged it) and swapped in crystallography (a distinct
    measurement-loss mechanism); reworded to "a minimal witness is checked as an instance"
    (honest: the formal instances are minimal witnesses, not full theories).
  - **Rejected, per panel:** the node-1 "three fields, one wall" trio seed (pre-spends node
    6, splits attention where Rashomon is first defined — cut); interpretability as a new
    recurring character (too thin, n=1 — left at its existing node-9 empirical footprint);
    the crystallographic phase problem as a machine-checked walk-on (fact-check: the Lean is
    a scalar signal-recovery toy, so "crystallography, machine-checked" would overclaim —
    kept only as the real-world *phenomenon* in the node-6 sweep, not a formal claim).

## Changelog: realism + zero-context hero + recurring patient (post-panel author pass)

- **No concept before its introduction, hero included.** The H1 no longer names the three
  properties or "machine-verified" (nothing a first-screen reader can parse yet). New H1:
  "The diagnosis didn't change. The reason did." — pure scene, zero jargon. Subhead
  rewritten to plain language ("a hard limit in mathematics… an exact way to know which
  parts of one you can trust… every step is checked, every failed prediction is published").
  "Faithful. Stable. Decisive. Pick Two." demoted to a post-node-3 divider / share-card line.
- **Cold open made clinically realistic.** (1) The Wisconsin features are *nuclear* cytology
  measurements, so "tumor's edge/size" → "the shape / the size of the cell nuclei"
  throughout (nodes 0, 2, 7, 9, 12). (2) The malignant call is now pathologist-confirmed
  (which is *why* it is stable); the model is decision-support that flags a driving feature,
  not the diagnostician (node 1). (3) The reason flips because the model is *refit on a
  refreshed sample of cases* — routine clinical revalidation, a genuinely different fit —
  and, even absent any update, another equally-good model on the same data would differ
  (the pure Rashomon point, matching the resampling regime the data measured; seed-alone
  gives 0%). Node-9 widget reworded so the model "flags," not "calls," the diagnosis.
- **Patient made a recurring test case**, 1–4 sentences inside the node, never a digression:
  node 2 (her case is the exact stability violation), node 7 (her stable-core summary: cell
  size agreed, shape flagged undetermined), node 8 (the certified reason her surgeon never
  got), plus the existing node-9 (measured majority) and node-12 (the summary she should
  have seen). Brief's POV-returns bullet updated to list exactly these staged returns.

## Changelog: round-3 full-document panel + /vet (final pre-build pass)

- Five-expert full-draft audit, all verdicts [MINOR POLISH]; fact-check PASS with zero
  WRONG findings. Applied in this pass:
  - **Monograph anchors**: all 24 occurrences ≥ old line 1962 bumped +11 (PR #27 shifted
    the monograph by one +11 hunk at line 1951); node-0 formal note split into
    Proven (§582) + Measured (§2649) chips.
  - **Node 11 formal layer** upgraded to the post-M3 state (was still "Gaussian
    discharged, general case open" — self-contradiction with its own plain layer); node-11
    plain rewritten back into plain register; scope unified to *natural* exponential
    families; "end-to-end and axiom-clean" de-duplicated.
  - **Continuity**: "At question 3" → "question 2" (W1 moved in v2); "the letter she
    should have received" → "the summary her portal should have shown" (v1 fossil);
    node-0 regeneration ↔ node-1 retraining seam aligned ("after the system's next
    routine refresh"); node-10 W4 now cashes the node-7 "hold that thought" plant.
  - **Receipts strip corrected 8 → 5** preregistered failed predictions (monograph
    §4882; the two self-refutations were not preregistered); "84% / 19%" restored inline
    to the coin-flip payoff sentence; the one-in-five guarantee handle minted at node 8;
    W1 echo gains a decisiveness-chooser branch; W3 reveal spec'd (feature-level 100%
    reveal, cluster-level 26–42% as the honest turn) + M1 wall staged at node 9; M3 cut
    from the motif spec (content survives as the node-9 table treatment); "(full
    strength)" author tags removed from two headers; practitioner CTA given a concrete
    artifact path.

## Changelog v2 → v3 round-2 polish (five-expert panel + /vet)

- Unanimous panel verdict [MINOR POLISH]; applied: node-1 freeze-the-model escape closed
  (a frozen model's reason is still one arbitrary draw); node-1 "would" → "could" (existence,
  not certainty) and "exactly" → "just as accurate"; node-1 "observationally identical" →
  "produce identical behavior"; node-10 pre-spoiler removed (widget now leads, the
  "reason to trust" payoff earned at the node's end); node-0 formal note re-pointed from the
  adverse-action credit sections to §2649 (Per-Individual Reversal) to match the medical
  scene; node-7 header "CAN" → *can*; header/counts/pin refreshed.

## Changelog: post-final (2026-08-02, author pass)

- **Takeaway rewritten to stand alone with zero context — two iterations.** "No explanation
  can keep every promise…" leaned on the page's internal promise motif (nodes 2–3) and was
  opaque away from the page. A first replacement glossed the triple as "honest, consistent,
  and definite" — rejected as incorrect: out of context "consistent" misreads stability
  (suggests self-consistency, not same-explanation-for-identical-behavior), the phrasing hid
  that the failure lives across the *ensemble* of equally good models, and it stated the
  trilemma unconditionally, dropping the multiplicity hypothesis. A second replacement built
  the Rashomon intuition around two equally accurate AI models — rejected as too narrow: the
  theorem is about explanation in general (the universality is the point), and the machine
  framing shrank it to an ML story. Final takeaway states the Rashomon intuition at full
  generality and maximum compression: "When many explanations fit the facts equally well —
  and they usually do — the one you're given is partly an accident of the draw. That is now
  a theorem, not a suspicion. But the part every equally good explanation agrees on *can* be
  trusted — and a machine checked every step." Multiplicity is the visible mechanism,
  "usually do" carries ubiquity, "accident of the draw" carries the trilemma's consequence,
  and the constructive half is what-every-equally-good-explanation-agrees-on. Applied at
  both occurrences (Part 1 brief, node 12). The in-page promise motif (nodes 2/3) and the
  hero subhead are unchanged — they are anchored by their surrounding scene.
- **Hero inverted: program framing now leads** (Drake, post-build). The top-level title had
  over-weighted the medical story: H1 was the scene-line, and "The Limits of Explanation"
  survived only as a small kicker. New structure is epigraph → title → deck: the scene-line
  ("The diagnosis didn't change. The reason did.") demoted to a quiet serif-italic epigraph;
  **H1 = "The Limits of Explanation"**; the deck rewritten at full generality — "The reason
  you're given — for a diagnosis, a denial, a decision — can change while the verdict holds
  still. That is not sloppiness; it is a provable limit — and the same mathematics that
  proves the limit tells you which parts of an explanation you can trust. Every step is
  checked. Every failed prediction is published. The argument begins with one line on a
  patient's chart." Craft notes: the diagnosis/denial/decision triple spans the page's three
  stakes-domains without naming machines (node-1 reveal intact); "verdict holds still" ↔
  the epigraph's "didn't change" is a deliberate echo; the credo pair ("Every step is
  checked. Every failed prediction is published.") is retained verbatim; the closing hook
  points the reader into the cold open. Zero-context rule still holds — no property names,
  no "theorem," no "machine-verified" in the hero.
- **Hero stripped to the bone + cold-open declutter** (Drake, post-build iteration 2):
  epigraph and deck CUT (the scene-line and the diagnosis/denial/decision deck are gone
  entirely); count chips CUT (falsification chip alone remains in the hero); node-0 "Cold
  open" eyebrow and the ghosted composite-scene aside CUT — ⚠ the composite-scene honesty
  disclaimer now appears nowhere on the page (flagged to Drake, kept per instruction); the
  M1 document moved BELOW the "Weeks later" regeneration paragraph, initial reason "the
  size of the cell nuclei," cycle order size → texture → shape so the cycle dramatizes the
  "third reason would be waiting" line that follows it.
- **TEACHING CUT — the falsification program removed from the page** (Drake, post-build
  iteration 3): the page is teaching material; the credibility apparatus lives in the paper.
  Removed: the entire node 10 ("Has the theory ever been wrong?") — W4, the η-law story, the
  self-refutations, the Falsification Ledger and its anchor; the hero's last receipts chip;
  the node-7 "hold that thought" plant (its payoff is gone); node 9's "a bound reality
  escapes is falsified / evidence, not decoration" framing sentences, its
  preregistration/prospective-round close, and the prospective bits of its formal layer; old
  node 11's frozen-bet opening paragraph and its "what would prove you wrong?" title clause;
  the where-to-start map's falsifications + OSF entries; the Retired tier from the node-0
  tier key and the footer (no Retired-tier content remains; CSS tokens kept). Renumbered:
  open problems = question 10 ("What's still open?"), the ask = question 11; ribbon phases
  now The Ceiling 1–6 · The Instrument 7–9 · The Frontier 10 · The Ask 11. Kept as
  theory-essential: node 5's machine-checking (defines what Proven means), the certificate's
  one-sidedness note (a theorem property, not credibility talk), the node-6 Morse–Sard
  provenance (condensed to its teaching core, confession framing dropped). This draft's
  node-10 section and falsification threads describe the PAPER-side narrative and no longer
  render on the page.
- **Teaching-cut panel round + fixes** (5-expert panel on the post-subtraction framing;
  audits in `explainer-audit/teaching-cut/`, consolidated in `RECOMMENDATIONS.md`;
  unanimous [MINOR POLISH], all numbers/cites re-verified exact). Applied: finale W1 echo
  now branches correctly per pick (was wrong for 2 of 3); fiction disclosure restored
  minimally ("A composite case, no real patient" in the node-0 caption, which also absorbs
  the caption trim; "patients exactly like the one in the opening" at node 9); ribbon
  stamps for Q6/Q9 gated behind widget commit (were spoiling W2/W3); Q11 promises kept
  (monograph, three repos, `knockout-experiments/` all hyperlinked at the pin; "tracked
  issue" reworded — no open issues exist); 9→10 hinge sentences added ("a shorter list");
  W2 skip affordance + scroll-past fallback unveil for both veils; "cold open too" dangler
  fixed; "quantitative guarantees are bounds" + the 99.4%-vs-spec clause; W1 card-dimming
  implemented (was dead code); neutral footer forwarding line ("teaching companion to the
  monograph" — no falsification language). Skipped as optional: the fallible-protagonist
  landing clause. Dismissed after verification: marketing's head-metadata finding (the
  artifact wrapper supplies it).
