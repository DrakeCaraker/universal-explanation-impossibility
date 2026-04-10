# Adversarial Peer Review Simulation

**Target**: `paper/universal_impossibility_pnas.tex` (6pp) +
`paper/universal_impossibility_pnas_si.tex` (SI)

**Goal**: Five independent hostile reviews from PNAS reviewer
archetypes. Each reviewer tries to find reasons to REJECT.
Every issue found is classified as FATAL / MAJOR / MINOR.
A single unresolved FATAL blocks submission.

**Model**: Opus for all reviews (requires adversarial judgment).

**Critical rule**: Reviewers must try to BREAK the argument.
A review that says "everything looks fine" has failed —
there are always issues to find.

---

## Review 1: The Mathematical Statistician

**Expertise**: Rashomon sets, invariant decision theory,
formal impossibility results, Hunt-Stein theorem.

**Disposition**: Sympathetic to the topic but demands
mathematical rigor at the level of Annals of Statistics.

**Checklist** (examine PNAS paper + SI + Lean code):

### Definitions
- [ ] Is `faithful` the right formalization? Could a
  reasonable statistician define "faithful" differently and
  get a different result? List 3 alternative formalizations
  and assess whether the impossibility survives each.
- [ ] Is `decisive` well-motivated from practice? Name a
  real explanation method that satisfies this definition.
  Name one that doesn't. Is the boundary in the right place?
- [ ] Is `incompatible_irrefl` an assumption or a consequence?
  Could there be domains where incompatibility is NOT
  irreflexive? What would happen to the theorem?
- [ ] The Rashomon property is existential (∃ θ₁ θ₂). Should
  it be measure-theoretic (positive probability of Rashomon
  pairs)? Does the existential version trivialize the result
  by requiring only one pathological pair?

### Proofs
- [ ] Read the impossibility proof step by step. Is each
  step logically necessary? Can any be weakened?
- [ ] Read the tightness proofs. Are the conditional witnesses
  (neutral element, committal element) CONSTRUCTIBLE for the
  main instances, or only existential?
- [ ] The necessity proof uses injective observe. Is injectivity
  the right condition? Could there be a weaker sufficient
  condition for possibility?
- [ ] The axiom substitution test: are the 4 alternatives
  genuinely the only interesting alternatives? Could there be
  a 5th formalization that gives a different result?

### Hunt-Stein Connection
- [ ] Is the G-invariant resolution EXACTLY Hunt-Stein, or
  is the connection looser than claimed? State the precise
  Hunt-Stein theorem and compare.
- [ ] Hunt-Stein requires an amenable group. Is this assumption
  stated in the paper? Does it hold for all instances?
- [ ] The paper says the resolution is "Pareto-optimal among
  stable methods." Prove or disprove: is it the UNIQUE
  Pareto-optimal stable method?

### Quantitative Bound
- [ ] The per-fiber theorem is essentially the impossibility
  applied locally. Is this genuinely quantitative, or is it
  a repackaging of the qualitative result?
- [ ] For the attribution instance, the ρ²/(1-ρ²) bound comes
  from the companion paper. Is it correctly stated here?
  Is the derivation complete or does it rely on unstated
  assumptions?

---

## Review 2: The Experimental Methodologist

**Expertise**: Experimental design, statistical testing,
replication, causal inference in empirical ML research.

**Disposition**: Skeptical of theoretical results that claim
empirical validation. Will scrutinize every experiment.

**Checklist** (examine experiments + results JSONs + code):

### For EACH of the 7 experiments:

**Design validity:**
- [ ] What is the null hypothesis? Is it clearly stated?
- [ ] What is the alternative hypothesis?
- [ ] Is the experimental design appropriate for the hypothesis?
- [ ] Are there confounds that could explain the results
  without invoking the Rashomon property?
- [ ] Would a different reasonable methodological choice
  (different dataset, different model, different metric)
  give a qualitatively different result?

**Statistical rigor:**
- [ ] Is the sample size justified? Is there a power analysis?
- [ ] Are the confidence intervals for the right quantity?
- [ ] Is there correction for multiple comparisons?
  (7 experiments × 3 tests = 21 hypothesis tests)
- [ ] Are the bootstrap CIs computed correctly? (Read the
  actual code in experiment_utils.py, not just the description)
- [ ] Are seeds recorded and fixed? Is the analysis
  reproducible?

**Specific experiment critiques:**

Attention (perturbation, 60%):
- [ ] Is weight perturbation a valid proxy for the Rashomon
  property? What if perturbation creates models OUTSIDE the
  Rashomon set (damaged, not equivalent)?
- [ ] 5x sentence duplication: does this inflate the effective
  sample size? What's the true effective N?
- [ ] sign(mean(CLS embedding)) as prediction: is this a
  real classifier? Would a proper fine-tuned classifier give
  different results?

Attention (full retraining, 19.9%):
- [ ] 4/20 models below 80% accuracy. Should these be excluded?
  Does excluding them change the flip rate?
- [ ] 200 training sentences is tiny. Does this create a
  genuine Rashomon set or just undertrained models?
- [ ] Only last 2 layers unfrozen. How does the result change
  if all layers are unfrozen? If only 1 layer?

Counterfactual (23.5%):
- [ ] Greedy perturbation toward positive centroid is not a
  standard CF method. Would Wachter et al. (2017) optimization
  give different results?
- [ ] German Credit only. Would Adult Income or COMPAS give
  similar results?

Concept probe (0.90):
- [ ] "Curved vs angular" is an arbitrary concept. Would a
  less arbitrary concept (e.g., "has a loop") give different
  results?
- [ ] 8x8 images with 64 features. Would real images (CIFAR,
  ImageNet) give similar results?

GradCAM (9.6%):
- [ ] Prediction agreement is only 78.8%. Are these genuinely
  equivalent models or damaged models?
- [ ] 9.6% is below the stated 20% practical significance
  threshold. Should this instance be dropped?

Token Citation (34.5%):
- [ ] This is attention rollout with a template, not LLM
  generation. Does the paper make this clear enough?
- [ ] The negative control has 31.5% flip rate on "easy"
  sentences. This is supposed to be <5%. Does this invalidate
  the control?

Model selection (80%):
- [ ] Is subsample=0.8 creating genuine Rashomon diversity or
  just evaluation noise?
- [ ] 20 evaluation splits: is this enough for stable estimates?

### Negative controls
- [ ] For EACH negative control: does it actually test what
  it claims? Could the low flip rate be explained by something
  other than absence of Rashomon?

### Resolution tests
- [ ] For EACH resolution test: does aggregation GENUINELY
  improve stability, or is the metric misleading?

---

## Review 3: The ML Theory Expert

**Expertise**: XAI impossibility results, Lean 4, formal
verification, Bilodeau et al. (2024), Chouldechova (2017).

**Disposition**: Knows the related work intimately. Will check
whether the result is genuinely novel or a known consequence.

**Checklist**:

### Novelty assessment
- [ ] Does this result follow trivially from Bilodeau et al.
  (2024)? Precisely state what Bilodeau proves and what this
  paper adds.
- [ ] Does Rao (2025) already prove something equivalent using
  different language?
- [ ] Is the "universal" framing genuinely new, or have others
  already noted that XAI impossibilities share a common
  structure?
- [ ] The fairness impossibilities (Chouldechova, Kleinberg) —
  are they REALLY instances of this theorem, or is the mapping
  forced?

### Lean verification
- [ ] Read ExplanationSystem.lean. Is the formalization a
  faithful representation of the informal definitions in the
  paper? Are there discrepancies?
- [ ] The paper says "zero axiom dependencies." Verify
  #print axioms output matches.
- [ ] 72 axioms total. Are any of these HIDING assumptions
  that should be stated as hypotheses? Specifically: do the
  instance system axioms (attentionSystem, cfSystem, etc.)
  encode unstated behavioral assumptions?
- [ ] The MarkovEquivalence.lean derivation: is ciFromDAG a
  correct formalization of d-separation, or is it an ad hoc
  formula that happens to work for the chain/fork case?

### Technical depth
- [ ] Is the impossibility proof genuinely 4 lines, or is
  complexity hidden in the definitions?
- [ ] The axiom substitution: 4 alternatives tested. Are
  these the alternatives a reviewer would naturally propose,
  or are they cherry-picked to make the current definitions
  look good?
- [ ] The per-fiber quantitative bound: is this a new result
  or a trivial corollary of the main theorem?

---

## Review 4: The Philosopher of Science

**Expertise**: Quine-Duhem, underdetermination, structural
realism, theory choice, formal epistemology.

**Disposition**: Excited by the cross-domain framing but will
demand precision about philosophical claims.

**Checklist**:

### Underdetermination claims
- [ ] Does the paper correctly state the Quine-Duhem thesis?
  Is the connection to Rashomon precise or loose?
- [ ] The paper says "formalizes a core consequence of the
  Quine-Duhem thesis." Is this the right claim? Which
  SPECIFIC consequence is formalized?
- [ ] Does the formalization capture EMPIRICAL underdetermination
  (same predictions, different theories) or HOLISTIC
  underdetermination (Quine's web of belief)? These are
  different.
- [ ] Okasha (2011) applied Arrow to theory choice. Does the
  paper adequately distinguish its result from Okasha's?

### Structural realism connection
- [ ] The paper claims the G-invariant resolution "formalizes
  structural realism." Is this overclaiming? Structural
  realism is about ontological commitments, not just invariant
  descriptions.
- [ ] The Worrall (1989) citation: does the paper correctly
  represent what Worrall argued?
- [ ] Van Fraassen's constructive empiricism: is it correctly
  positioned? The paper should note that van Fraassen would
  REJECT decisiveness as a desideratum entirely.

### Cross-domain tiering
- [ ] Are the three tiers correctly assigned? Should any
  Tier 2 items be Tier 3, or vice versa?
- [ ] The neuroscience degeneracy connection: is it precise
  enough for a computational neuroscientist to accept?
- [ ] The economics identification connection: would Manski
  agree that partial identification is an instance of this
  framework?

---

## Review 5: The Adversarial Skeptic

**Expertise**: General ML and AI. No specific expertise in
XAI theory. Represents the "common sense" reviewer who asks
"so what?" and "isn't this obvious?"

**Disposition**: Will challenge every claim on practical
relevance. Doesn't care about Lean proofs or philosophical
connections. Only cares about impact.

**Checklist**:

### "So what?" test
- [ ] Name ONE concrete system where this result would change
  a practitioner's decision. Not hypothetically — actually.
- [ ] If a data scientist reads this paper, what do they do
  differently on Monday morning?
- [ ] The resolution (G-invariant aggregation) is already
  standard practice in some domains (CPDAGs, BMA). What does
  the theorem add beyond justifying existing practice?

### "Isn't this obvious?" test
- [ ] Could a smart ML engineer derive this result in 10
  minutes if you told them the setup? If yes, is the paper
  just stating the obvious with fancy formalism?
- [ ] The 9 instances: aren't these all just "different models
  give different explanations"? What's the insight beyond that?
- [ ] Arrow's theorem comparison: is this result ACTUALLY
  comparable to Arrow, or is the comparison aspirational?

### "Why should I believe the experiments?" test
- [ ] The experiments use toy settings (load_digits, synthetic
  sentences, CIFAR-10 with perturbation). Do these results
  transfer to real-world systems (GPT-4 explanations,
  production credit scoring, clinical decision support)?
- [ ] Weight perturbation is artificial. True retraining gives
  19.9% — still significant but much less dramatic than 60%.
  Is the paper honest about this discrepancy?
- [ ] No experiments on real-world scale models. Does this
  undermine the "universal" claim?

### Format and presentation
- [ ] Is the paper accessible to a PNAS general audience?
  Could a biologist understand the abstract?
- [ ] Is the significance statement accurate and complete?
- [ ] Are the figures/tables clear at print size?

---

## Meta-Review

After all 5 reviews, synthesize:

### Classification of all findings
- FATAL: blocks acceptance, cannot be fixed in revision
- MAJOR: must be fixed, but fixable
- MINOR: should be fixed, not blocking
- STYLISTIC: nice to fix

### The three hardest questions
What are the three questions a reviewer could ask that the
authors would struggle MOST to answer? For each, draft the
best possible author response.

### Accept/Reject recommendation
Based on the five reviews, what is the overall recommendation?
- Strong Accept (top 10% — transformative result)
- Accept (solid contribution to science)
- Weak Accept (publishable with revisions)
- Weak Reject (significant issues, may be fixable)
- Reject (fundamental flaws)

Justify with specific evidence from the reviews.
