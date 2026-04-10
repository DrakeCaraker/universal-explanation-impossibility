# Universal Explanation Impossibility — Research Roadmap

**Status**: Paper 3 complete. 75 Lean files, 351 theorems, 0 sorry.
9 instances, query-relative generalization, quantitative remark.
PNAS (6pp), NeurIPS (10pp), monograph (41pp) ready.

---

## Completed Papers

### Paper 1: DASH (method + software)
- **Venue**: TMLR (submitted)
- **Contribution**: Ensemble SHAP averaging, the constructive resolution

### Paper 2: Attribution Impossibility (depth)
- **Venue**: arXiv (posted), Zenodo, NeurIPS 2026 (submitted)
- **Contribution**: Deep quantitative treatment of Instance 1.
  1/(1-ρ²) divergence, DASH optimality, 305 Lean theorems.

### Paper 3: Universal Impossibility (breadth)
- **Venue**: arXiv → NeurIPS 2026 → PNAS
- **Contribution**: One theorem, 9 instances, query-relative,
  Lean-verified, empirically illustrated, cross-domain.

---

## Next Papers

### Paper 4: Quantitative Tradeoff Surfaces
**Target**: JMLR or NeurIPS 2027
**Contribution**: Replace binary impossibility with continuous design constraints.

Core content:
- **Generalization 1**: ε + δ + (1-d) ≥ μ(Rashomon) — the tradeoff surface
- **Approximate impossibility**: Does the result survive under
  ε-faithful, δ-stable? (Closes "nobody needs exact stability")
- **Cross-method comparison**: Map SHAP, LIME, attention, GradCAM,
  DASH onto the trilemma triangle. Which property does each sacrifice?
  How much of each do they achieve?
- **Regulatory compliance numbers**: For DASH over 50 models on a
  banking dataset under SR 11-7, compute actual (ε, δ, d) triples.
  "Your explanation is 95% faithful, 100% stable, 73% decisive."
- **Domain-specific bounds**: Derive the tradeoff curve for GBDT
  (as function of ρ), MLP (as function of overparameterization),
  transformers (as function of perturbation magnitude)

Lean work needed: Measure-theoretic ExplanationSystem (extend with
modelMeasure from Defs.lean). Prove the union bound formally.

### Paper 5: The Arrow Connection + Probabilistic Extensions
**Target**: PNAS (if Arrow embedding works) or JMLR
**Contribution**: Embed Arrow's theorem as Instance 10. Prove
probabilistic/randomized explanations don't escape.

Core content:
- **Arrow as Instance 10**: Θ = voter profiles, Y = ordinal
  preferences, H = social rankings, incompatible = IIA violation.
  If the mapping is precise, Arrow is literally a special case.
  This would be extraordinary — the universal impossibility would
  be STRICTLY MORE GENERAL than Arrow's theorem.
- **Probabilistic explanations**: Can randomized E break the
  impossibility? Prove: no, faithfulness-in-expectation +
  stability + decisiveness-in-expectation is still impossible.
- **Large-scale validation**: GPT-2/Llama fine-tuning (true
  multi-seed), production credit scoring, clinical decision support.
- **Side information theorem**: Additional information helps iff
  it disambiguates the Rashomon set. Formalize and prove.

### Paper 6+: Domain Applications
**Target**: Domain-specific venues with domain collaborators
**Contribution**: Formal instantiation in specific scientific domains.

Candidates (in priority order):
- **Climate modeling**: Multiple GCMs fit the same historical data.
  Attribution of extreme weather events is underspecified. The
  impossibility explains why different models give different
  attribution results. Venue: Nature Climate Change or PNAS.
- **Protein structure**: AlphaFold predicts structure but multiple
  conformations are consistent with the sequence. Explaining
  "why this fold?" faces the trilemma. Venue: PNAS or Nature Methods.
- **Neuroscience**: Neural degeneracy formalized as Instance 11.
  Multiple circuits implement the same behavior. Circuit explanations
  face the impossibility. Venue: PLOS Computational Biology or Neuron.
- **Econometrics**: Partial identification (Manski) as a formal
  instance. Structural model explanations face the trilemma.
  Venue: Econometrica or Journal of Econometrics.

---

## Theoretical Extensions (prioritized)

### High Priority (Paper 4-5)

1. **Quantitative tradeoff surface** (Gen 1)
   - ε + δ + (1-d) ≥ μ(Rashomon)
   - Turns impossibility into design specification
   - Needs measure theory in Lean

2. **Approximate impossibility**
   - Does the result survive under approximate stability?
   - Closes the biggest theoretical objection
   - Probably yes, with degradation bounds

3. **Probabilistic/randomized explanations**
   - Can randomized E break the impossibility?
   - Almost certainly no in expectation
   - Closes a natural reviewer objection

4. **Arrow embedding**
   - Social welfare as ExplanationSystem
   - If it works: universal impossibility ⊃ Arrow
   - High risk, very high reward

5. **Side information**
   - "Additional info helps iff it disambiguates Rashomon"
   - Elegant one-theorem result
   - Could be a remark or a full paper

### Medium Priority (Paper 6+)

6. **Explanation complexity tradeoff**
   - MDL of stable explanations vs individual explanations
   - G-invariant projection has higher complexity (more ties)
   - Is there a complexity-stability frontier?

7. **Distribution shift**
   - Rashomon set changes under shift
   - How fast does explanation stability degrade?
   - Connects to MLOps, continuous monitoring

8. **Interactive explanations**
   - Can follow-up questions help?
   - First query: faithful+decisive. Follow-ups: stable.
   - Interactivity may partially resolve the trilemma

9. **Composition of explanation systems**
   - Feature explanations + interaction explanations
   - Can you be faithful+stable on both simultaneously?
   - May not yield clean results

### Low Priority (speculative)

10. **Computational complexity of resolution**
    - DASH is polynomial. Circuit equivalence classes may be NP-hard.
    - Interesting for MI but niche.

11. **Federated/distributed explanations**
    - Multiple parties, different observations
    - Connects to federated learning

12. **Game-theoretic explanation markets**
    - Equilibrium when multiple parties explain under trilemma
    - Speculative, mechanism design flavor

13. **Categorical abstraction** (Gen 4)
    - Explanation = non-factorization of explain through observe
    - True but loses the insight (the 3-way decomposition IS the content)
    - Don't pursue — the current level of generality is the sweet spot

---

## Empirical Extensions (prioritized)

### High Priority

1. **Cross-method comparison**
   - Same model + input → SHAP vs LIME vs attention vs GradCAM
   - Map each onto the trilemma: which property does each sacrifice?
   - Practitioner-facing paper

2. **Measuring (ε, δ, d) triples empirically**
   - For DASH: what are the actual faithfulness/stability/decisiveness rates?
   - For standard SHAP: same
   - Pairs with Gen 1 theory

3. **Regulatory compliance demonstration**
   - DASH on a banking dataset under SR 11-7
   - Actual numbers a regulator can use
   - Highest real-world impact

### Medium Priority

4. **Large-scale validation** (GPT-2+, production systems)
5. **Longitudinal study** (explanation stability over monthly retraining)
6. **User study** (do practitioners find the trilemma framing useful?)

---

## The Arrow Question

The most intellectually exciting open question: is Arrow's
impossibility theorem an instance of the universal explanation
impossibility?

**The mapping**:
- Θ = voter preference profiles (n voters, m alternatives)
- Y = the set of alternatives (what society "chooses")
- H = social orderings (rankings of alternatives)
- observe(θ) = the social choice (the winning alternative)
- explain(θ) = the social welfare function (full ranking)
- incompatible(h₁, h₂) = the rankings disagree on some pair
  (violates IIA: the relative ranking of a,b depends on c)
- Rashomon property: two preference profiles that produce the
  same winner but induce different social rankings

**The question**: Does this mapping work precisely? Arrow's axioms
(IIA, Pareto, non-dictatorship) are specific to social choice.
Our axioms (faithful, stable, decisive) are different. Is there
a precise correspondence, or just an analogy?

If the mapping is EXACT — if Arrow's axioms are recoverable from
faithful+stable+decisive under the social choice ExplanationSystem —
then the universal impossibility is literally a generalization of
Arrow. That would be the strongest possible positioning for the
result.

**Risk**: The mapping may be only structural (analogous, not identical).
Arrow's IIA is about pairwise independence of rankings from
irrelevant alternatives. Our stability is about invariance across
equivalent configurations. These are related but may not be the same.

**Recommendation**: Investigate seriously but don't claim until
proved. A Lean-verified proof that Arrow's theorem follows from
the universal impossibility applied to a social choice
ExplanationSystem would be a landmark result in its own right.

---

## Timeline

```
2026 Q2:  Paper 3 on arXiv + NeurIPS submission
2026 Q3:  NeurIPS decision. Begin Paper 4 (quantitative).
2026 Q4:  Paper 4 draft. Investigate Arrow embedding.
2027 Q1:  Paper 4 submission (JMLR or NeurIPS 2027).
          PNAS submission for Paper 3 (if NeurIPS rejects)
          or PNAS cross-domain version (if NeurIPS accepts).
2027 Q2:  Paper 5 (Arrow + probabilistic) if embedding works.
2027+:    Domain papers (6+) with collaborators.
```

---

## Key Principle

The framework's sweet spot is that it's the simplest structure
that makes the nine-domain unification work. More generality
makes the theorem more powerful but harder to explain. For an
impossibility result, communicability matters as much as generality.
Arrow's theorem isn't stated in the most general categorical
language — it's stated in terms of preference orderings that
political scientists can understand. That's a feature.

"Faithful, stable, decisive — pick two."
