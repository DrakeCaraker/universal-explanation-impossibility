# 10-Agent Specialist Review

**Paper**: paper/universal_impossibility_monograph.tex (53pp)
**Title**: "The Limits of Explanation"
**Premise**: Each agent is a domain specialist seeing this for
the FIRST time. No shared context. Each reads the sections
relevant to their expertise and reports EVERY issue.

---

## Agent 1: Pure Mathematician (Proof Theory)

**Read**: ExplanationSystem.lean, Necessity.lean,
QuantitativeBound.lean, QueryRelative.lean,
AxiomSubstitution.lean, MarkovEquivalence.lean.
Also: monograph Sections 3-4 (Framework, Tightness, Necessity).

**Check**:
- Is the proof of explanation_impossibility valid?
- Does each step use the hypothesis it claims to?
- Are the tightness witnesses genuine counterexamples?
- Is the necessity proof correctly scoped?
- Is the axiom substitution analysis exhaustive?
- Is incompatible_irrefl the minimal required property?
- Could a weaker Rashomon property (e.g., measure-zero)
  still yield the impossibility?
- Is there a simpler proof that bypasses one of the three
  properties (making it redundant)?
- Are the quantitative and query-relative extensions sound?

## Agent 2: Lean 4 / Formal Verification Expert

**Read**: ALL 75 .lean files (at least headers + theorems).
ExplanationSystem.lean in full. The monograph's Lean appendix.

**Check**:
- Run lake build. Any errors or warnings?
- Run #print axioms for: explanation_impossibility,
  causal_impossibility_derived, fully_specified_possibility,
  query_impossibility, each instance theorem
- Do the Lean definitions EXACTLY match the paper's definitions?
- Does the appendix Lean listing match the actual files?
- Are there any sorry anywhere (even commented out)?
- Are axiom counts (72) correct? Classify each axiom.
- Is autoImplicit false everywhere?
- Could any axiom be replaced by a definition or theorem?
- Are there any unsound axiom combinations (e.g., axioms
  that together imply False)?

## Agent 3: Experimental Statistician

**Read**: ALL 7 experiment scripts in paper/scripts/.
ALL 7 results JSONs. paper/scripts/experiment_utils.py.
Monograph Section 4 (instance empirical paragraphs).

**Check**:
- Is the bootstrap CI implementation correct?
- Are CIs for the right quantity (mean, not observation)?
- Is there multiple comparison correction? Should there be?
- For each experiment: sample size, power, effect size
- Is weight perturbation a valid experimental protocol?
- For EACH negative control: does it control what it claims?
  Does it have nonzero valid comparisons?
- For EACH positive result: could it be explained by
  something OTHER than the Rashomon property?
- Are seeds fixed? Are results reproducible?
- Does every number in the paper match its JSON source?

## Agent 4: ML/XAI Theory Expert

**Read**: Monograph Sections 1-3 (Intro, Related Work, Framework).
Instance sections for attribution, attention, counterfactual,
concept probe. Also: Bilodeau et al. 2024 positioning.

**Check**:
- Is the result genuinely novel vs Bilodeau et al. (2024)?
- Is the result genuinely novel vs Rao (2025)?
- Does the framework add value beyond stating the obvious?
- Is "faithful" the right formalization? Would a reasonable
  ML theorist define it differently?
- Is "decisive" too strong? Would weakening it still give
  impossibility?
- Are the 9 instances genuinely different or are some
  redundant (e.g., attention ≈ token citation)?
- Is the "no escape by switching methods" claim justified?
- How does this compare to the disagreement problem
  (Krishna et al. 2022)?

## Agent 5: Philosopher of Science

**Read**: Monograph Discussion (cross-domain implications),
Arrow comparison appendix, related work (Quine-Duhem section).

**Check**:
- Does the paper correctly state the Quine-Duhem thesis?
- Is "contrastive underdetermination" the right term?
- Is the structural realism connection overclaimed?
- Is the van Fraassen connection accurate?
- Is the Arrow comparison honest about where it breaks?
- Are Laudan, Dorling, Howson cited appropriately?
- Would a philosopher accept the three-tier classification?
- Does the paper conflate empirical underdetermination with
  holistic underdetermination?
- Is "first impossibility theorem from underdetermination"
  defensible given formal epistemology literature?

## Agent 6: Causal Inference Expert

**Read**: MarkovEquivalence.lean. Monograph causal discovery
instance section. Arrow comparison appendix.

**Check**:
- Is the chain/fork Markov equivalence correctly stated?
- Is ciFromDAG a correct (even if simplified) CI extractor?
- Does the collider limitation matter for the proof?
- Is incompatible = (≠) appropriate for DAGs?
- Is the CPDAG resolution correctly described?
- Would Pearl, Spirtes, or Chickering agree with this
  formalization?
- Is the connection to Manski's partial identification
  correctly stated?

## Agent 7: AI Safety / Mechanistic Interpretability Expert

**Read**: MechInterpInstance.lean. Monograph MI instance section.
docs/proof-sketches/mech-interp-impossibility.md.

**Check**:
- Is the Meloux et al. (2025) citation accurate?
  (Authors, venue, numbers: 85 circuits, 535.8 interpretations)
- Does the impossibility actually threaten MI, or does MI
  already operate in Family B (equivalence classes)?
- Is the framing generous to the MI community?
- Would Neel Nanda or Chris Olah object to anything?
- Is "circuit equivalence classes" a meaningful resolution?
- Does the paper acknowledge that MI researchers already
  know about non-uniqueness?

## Agent 8: Regulatory / Policy Expert

**Read**: Monograph regulatory analysis section (SR 11-7),
Discussion section on implications.

**Check**:
- Are the SR 11-7 section references (III.A, III.B, IV)
  accurate?
- Does the mapping (conceptual soundness → faithful,
  outcomes analysis → stable, effective challenge → decisive)
  hold up to legal scrutiny?
- Is the EU AI Act Article 13 correctly characterized?
- Is the NIST AI RMF correctly characterized?
- Are the compliance recommendations actionable?
- Would a banking regulator find this useful or too abstract?

## Agent 9: LaTeX / Typography Expert

**Read**: The compiled monograph PDF (53pp).
Also: the tex source for formatting issues.

**Check**:
- Any overfull/underfull hbox (run pdflatex, check log)?
- Tables within margins?
- Figures properly sized and captioned?
- Consistent font usage?
- Table of contents correct?
- Theorem/definition/proposition numbering consistent?
- Bibliography formatted correctly?
- Cross-references all resolve (no "??")?
- Widow/orphan lines?
- Page breaks at appropriate locations?
- The Lean code listings: properly formatted, within margins?

## Agent 10: Devil's Advocate (Maximally Hostile Generalist)

**Read**: Abstract, Introduction, and Conclusion ONLY.
(Simulates a busy reviewer who skims.)

**Check**:
- After reading only abstract + intro + conclusion, can you
  state what the paper proves? If not, the paper fails.
- What is the ONE thing the paper wants you to remember?
  Is it clear?
- What is the most devastating one-sentence objection?
- Would you cite this paper? Why or why not?
- Is the title accurate?
- Is the paper too long for its contribution?
- Does the abstract match the conclusion?
- If you could ask the authors ONE question, what would it be?

---

## Execution

Launch all 10 agents in parallel. Each reads ONLY the files
specified for their expertise. Each reports findings as
FATAL / MAJOR / MINOR / STYLISTIC.

After all 10 return: synthesize into a single list, deduplicate,
and classify. Fix everything FATAL and MAJOR. Document MINOR.

## Model Assignment

All 10 agents use Opus (requires judgment and domain expertise).
