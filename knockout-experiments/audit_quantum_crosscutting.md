# Adversarial Audit: Quantum Measurement Circularity + Cross-Cutting "So What" Test

**Date**: 2026-04-14
**Auditor stance**: Maximally adversarial. If the framework is a language rather than a tool, that verdict will be stated plainly.

---

## TRACK C: Quantum Measurement Circularity Audit

### TEST 1: Is the eta = 1/(d+1) result circular?

**The exact logical chain as written in the framework:**

1. Define V = R^{d^2-1} as the Bloch vector space of a d-dimensional quantum state.
2. Define G as the symmetry group of the POVM {M_i}: unitaries U such that U M_i U^dagger = M_{pi(i)}.
3. Define V^G as the subspace of V fixed by G under the adjoint representation.
4. Define eta = dim(V^G) / dim(V).
5. Claim: eta equals "the fraction of state properties accessible from the measurement."

**The critical question: How is "accessible properties" defined?**

Reading the derivation document (`quantum_measurement_derivation.md`) carefully, the proof sketch states:

> "Since the measurement outcome probabilities p_i = Tr(rho M_i) depend only on the projection of the Bloch vector onto the span of {M_i}, and the symmetry group G rotates the orthogonal complement among itself, the measurable information is precisely V^G."

This reveals a **two-step argument**, not a single definition:

- Step A: The measurable information from POVM {M_i} is the projection of rho onto span{M_i} (in the space of traceless Hermitian operators). This is an **independently defined** quantity -- it is determined by the linear algebra of the Born rule, not by G.
- Step B: The claim that span{M_i} = V^G (the G-invariant subspace).

**Is Step B true?** This is where the argument becomes subtle, and where the potential circularity hides.

For the **computational basis measurement** specifically:
- span{M_i} = span of diagonal traceless Hermitian operators = R^{d-1}.
- G = U(1)^{d-1} (independent phase rotations on each basis state).
- V^G under this G = diagonal Bloch components = R^{d-1}.
- So span{M_i} = V^G. Step B holds.

But **Step B is not true in general**. Consider a POVM with two elements {M_1, M_2} where M_1 = |0><0| and M_2 = I - |0><0|. This has the same measurement outcomes as the computational basis for d=2, but for d=3:
- span{M_i} = span{|0><0| - I/3} = 1-dimensional subspace of traceless operators.
- G includes all unitaries that fix |0> and act arbitrarily on the orthogonal complement, which is U(d-1) acting on the {|1>,...,|d-1>} subspace.
- V^G = components of rho invariant under U(d-1) acting on the last d-1 levels. This is the span of {|0><0| - I/d} plus... actually this is also 1-dimensional for d=3.
- So dim(V^G) = 1 and dim(span{M_i}) = 1. They happen to agree here.

The deeper issue: **V^G and span{M_i} are related but not identical in general.** The proof sketch elides this by asserting it follows from Schur's lemma. In fact, what Schur's lemma gives you is that V^G is the isotypic component of the trivial representation in the decomposition of V under G. The span{M_i} is a subspace of V^G (measurable components are always G-invariant), but V^G could in principle be larger (there could be G-invariant directions that no M_i projects onto).

**Verdict on circularity:**

The result is **NOT purely circular** -- "accessible properties" has an independent definition (the image of the Born-rule linear map), and the claim that this equals V^G is a substantive mathematical assertion. However, the claim is **only proven for the computational-basis case**, where it follows from the specific structure of diagonal matrices. The general claim (arbitrary POVM) relies on an unproven assertion that span{M_i} = V^G, which is plausible but not rigorously established in the framework documents.

**Circularity grade: AMBER.** Not circular in principle, but the gap between "independently defined accessible properties" and "V^G" is bridged by hand-waving rather than proof for the general case.

---

### TEST 2: Does eta predict something independently verifiable?

**The sample complexity test:**

Known result (Haah et al. 2017): Full quantum state tomography to precision epsilon requires Theta(d^2 / epsilon^2) copies. Estimating only the computational-basis-accessible parameters (the d-1 diagonal elements) requires Theta(d / epsilon^2) copies.

Ratio of sample complexities:
- Partial / Full = Theta(d) / Theta(d^2) = 1/d

Framework prediction:
- eta = (d-1)/(d^2-1) = 1/(d+1)

**Comparison:** 1/d vs 1/(d+1).

These agree to leading order for large d (both scale as 1/d), but they are NOT equal. The discrepancy is:
- 1/(d+1) vs 1/d: a relative error of 1/(d+1) ~ 1/d, which is O(1/d).
- For d=2: eta = 1/3 = 0.333, sample ratio = 1/2 = 0.500. A 50% relative error.
- For d=4: eta = 1/5 = 0.200, sample ratio = 1/4 = 0.250. A 25% relative error.

**Why the discrepancy?** eta counts the **fraction of Bloch vector dimensions** (a geometric/counting measure). Sample complexity scales with the **number of parameters to estimate**, which is dim(V^G) = d-1. The ratio of parameters is (d-1)/(d^2-1) = 1/(d+1). The ratio of sample complexities is d/d^2 = 1/d. These differ because sample complexity per parameter is not uniform -- diagonal elements of larger systems are slightly easier to estimate per-parameter (they have larger prior variance under Hilbert-Schmidt measure).

More precisely: the sample complexity for estimating k parameters out of D total is not simply (k/D) times the full-tomography cost. The dependence is Theta(k/epsilon^2) for the partial problem, not Theta(k * d/epsilon^2).

**Verdict:** eta = 1/(d+1) does NOT precisely predict the sample complexity ratio. It predicts the **parameter fraction**, which is a related but different quantity. The match to 1/d is coincidental at leading order and breaks at finite d. This is a **weak** but **not vacuous** connection -- eta captures the right scaling but not the right constant.

**Independent verification grade: WEAK.** The framework predicts a scaling (1/d) that was already known. It does not predict the exact sample complexity, and the discrepancy at finite d (50% error for qubits) means it is not making a sharp, independently verifiable prediction.

---

### TEST 3: Is there a case where eta gives the WRONG answer?

**Case 1: SIC-POVM on a qubit (d=2)**

- 4 measurement outcomes, spanning all of R^3 (the full Bloch sphere).
- G = {I} (trivial -- no non-trivial unitary commutes with all 4 SIC elements).
- V^G = V = R^3. dim(V^G) = 3.
- eta = 3/3 = 1.
- The SIC-POVM is informationally complete (4 outcomes - 1 normalization = 3 independent parameters = 3 Bloch components).
- **Result: CORRECT.** eta = 1 correctly identifies informational completeness.

**Case 2: Sequential non-commuting measurements (Z then X on a qubit)**

The framework document (`quantum_measurement_derivation.md`, Example 2) handles the case of measuring BOTH X and Z. It claims V^G = span{e_x, e_z}, giving eta = 2/3. This is correct IF you interpret "measurement" as "the data from both X and Z measurements on identically prepared copies."

However, for **sequential measurement on a single copy** (measure Z, then X on the same qubit):
- Measuring Z collapses the state to |0> or |1>.
- Subsequent X measurement on the post-measurement state gives no information about the original x-component.
- The accessible information is only the z-component: eta should be 1/3, not 2/3.

The framework gives eta = 2/3 for the combined POVM {|0><0|, |1><1|, |+><+|, |-><-|}, which is correct for **ensemble measurements** (measuring Z on half the copies and X on the other half). It is INCORRECT for sequential single-copy measurements, because the framework does not model measurement back-action.

**This is a genuine limitation, not a bug.** The framework assumes a single POVM applied to the state, which corresponds to ensemble tomography. It cannot model sequential adaptive measurements where later measurements depend on earlier outcomes. This is acknowledged implicitly (the setup is a POVM, not a quantum instrument), but the limitation is not documented.

**Case 3: Weak measurements**

For a weak measurement with measurement strength lambda (0 < lambda < 1):
- The POVM elements are M_0 = sqrt(lambda)|0><0| + sqrt(1-lambda)|1><1| and M_1 = sqrt(1-lambda)|0><0| + sqrt(lambda)|1><1|.
- The symmetry group is the same as for strong Z measurement: U(1) rotations about Z.
- V^G = span{e_z}, dim(V^G) = 1.
- eta = 1/3 (same as strong measurement).

But the **information per measurement** depends on lambda: a weak measurement (lambda close to 1/2) extracts much less information about z than a strong measurement (lambda = 1). The eta formula gives the same answer regardless of measurement strength.

**This is arguably correct at the qualitative level** (both weak and strong Z measurements access only the z-component of the Bloch vector), but **wrong at the quantitative level** -- the fraction of state information actually extracted per copy depends on lambda, and eta does not capture this.

**Verdict:** eta correctly identifies **which** properties are accessible but does NOT quantify **how much information** about those properties is extracted per measurement. It is a geometric/structural invariant, not an information-theoretic one. The framework document's language ("fraction of state properties accessible") is technically correct but misleadingly suggests more quantitative content than eta actually provides.

**Conditions where the framework applies:**
1. Single POVM measurement (not sequential/adaptive).
2. Ensemble setting (many identical copies).
3. Question is "which parameters are identifiable at all?" (not "how fast?").
4. The POVM's symmetry group is correctly identified.

**Where it breaks down:**
1. Sequential measurements with back-action.
2. Adaptive measurement strategies.
3. Quantitative information extraction rates (bits per copy, Fisher information).
4. POVMs where span{M_i} is strictly smaller than V^G (theoretically possible; we could not construct a concrete example, but the general proof is absent).

---

## TRACK D: Cross-Cutting "So What" Test

### TEST 4: Strip the framework -- can you make every prediction without it?

#### SAGE (Noether Counting)

**(a) Plain language, no framework:**
"Features that are highly correlated form groups. Within each group, pairwise importance rankings are coin flips (50% flip rate). Between groups with different mean effect sizes, rankings are stable (0% flip rate). The number of stable ranking comparisons equals g(g-1)/2, where g is the number of groups."

**(b) Was this obvious before?**
- The qualitative direction (correlation causes instability) was known: Breiman (2001), Fisher et al. (2019).
- The quantitative prediction (exactly 50% within-group, exactly 0% between-group, exactly g(g-1)/2 stable facts) was NOT previously stated.
- The SAGE algorithm for auto-discovering groups from bootstrap flip rates was not previously available (though hierarchical clustering of correlation matrices is standard).

**(c) What the framework adds:**
- The framework provides the **motivation** for looking at symmetry groups as the organizing principle. Without the framework, someone studying SHAP instability might not think to formalize correlation groups as a symmetry group acting on feature space.
- The Noether counting formula g(g-1)/2 follows from elementary combinatorics (number of between-group pairs), not from deep representation theory. You do not need the framework to derive it -- you just need the observation that within-group rankings are random and between-group rankings are stable.
- The framework adds a LANGUAGE (call this a symmetry group G, call the stable queries V^G) but the actual content is the empirical fact about correlation groups. The SAGE algorithm is useful engineering; the framework is a narrative wrapper.

**Honest assessment:** The Noether counting result is **genuinely novel and useful**. But the framework's contribution is primarily motivational (it pointed the researchers toward asking the right question), not derivational (the math is elementary combinatorics, not representation theory).

#### MI (Mechanistic Interpretability Ceiling)

**(a) Plain language, no framework:**
"Neural networks with n hidden units have permutation symmetry: relabeling neurons gives an equivalent network. Therefore, individual neuron identity is meaningless across retrains. Neuron-level agreement between independently trained networks is at chance (Jaccard = 1/n for random overlap of top-k from n). Only permutation-invariant quantities (like mean activation, sorted activation spectrum) are stable."

**(b) Was this obvious before?**
- Yes. Hecht-Nielsen (1990) identified the permutation symmetry. The fact that individual neurons are not meaningful across retrains has been known for 35+ years. Li et al. (2016), Kornblith et al. (2019), and the entire CKA/CCA literature is built on this observation.
- The specific experiment (Jaccard = chance to 4 decimal places for MNIST MLPs) is a clean confirmation, but it confirms what everyone in the field already knew.
- The prediction "stable fraction = 0/n" is obvious from the permutation symmetry; no framework needed.

**(c) What the framework adds:**
- The framework provides a **formal statement** that permutation-invariant quantities are the G-invariant subspace, and hence the only stable explanations.
- This adds ZERO new content beyond what was already known from Hecht-Nielsen's observation.
- The specific numerical experiment (Jaccard = 0.041 vs chance = 0.041) is a nice data point but is confirmatory, not novel.

**Honest assessment:** The MI result is **entirely pre-existing knowledge** repackaged in the framework's language. No new prediction, no new insight, no new method.

#### Quantum Measurement

**(a) Plain language, no framework:**
"A projective measurement in the computational basis of a d-dimensional quantum system can determine only the d-1 diagonal elements of the density matrix (populations), out of d^2-1 total parameters (populations + coherences). The fraction of accessible parameters is (d-1)/(d^2-1) = 1/(d+1)."

**(b) Was this obvious before?**
- Yes. This is textbook quantum mechanics. The decomposition into populations and coherences dates to the 1930s (von Neumann). The specific formula 1/(d+1) for the parameter fraction is a trivial algebraic identity that anyone working in quantum tomography would recognize. Holevo (1973) established the information-theoretic capacity bounds.

**(c) What the framework adds:**
- The framework provides the CLAIM that this is "the same phenomenon" as SHAP instability and neuron permutation symmetry.
- The mathematical content of this claim is: all three are instances of "G acts on a vector space V, and V^G is a proper subspace." This is true, but the statement "group representations have invariant subspaces" is not a deep insight -- it is the definition of representation theory.
- The framework does NOT predict anything new about quantum measurement that was not known before.

**Honest assessment:** The quantum result is **a relabeling of textbook physics** in the framework's language. The "unification" consists of observing that all three domains involve group actions with invariant subspaces, which is a structural observation, not a scientific prediction.

---

### TEST 5: What would the framework need to do to be non-trivially useful?

**The four criteria:**
(a) Quantitative and testable
(b) Not previously known or obvious
(c) Confirmed by experiment
(d) Could not have been derived without the framework

**Evaluation of each result:**

| Result | (a) Quantitative? | (b) Novel? | (c) Confirmed? | (d) Needed framework? | All four? |
|--------|-------------------|------------|----------------|----------------------|-----------|
| Noether counting (g(g-1)/2 stable queries) | YES | PARTIALLY -- the exact formula was not stated, but it follows from elementary combinatorics | YES (47pp gap, p=2.7e-13) | NO -- the formula follows from "within-group = random, between-group = stable" without any representation theory | **NO** -- fails (d) |
| MI ceiling (Jaccard = chance) | YES | NO -- known since 1990 | YES | NO | **NO** -- fails (b) and (d) |
| Quantum eta = 1/(d+1) | YES | NO -- textbook since the 1930s | YES (by construction) | NO | **NO** -- fails (b) and (d) |
| Universal eta R^2=0.957 (7 instances) | YES | PARTIALLY -- the universal plot is new | YES | PARTIALLY -- the framework motivated collecting the 7 data points | **BORDERLINE** -- the framework motivated the search, but the fit depends on post-hoc group selection |
| SAGE algorithm (auto-discovery) | YES | YES -- the algorithm is new | YES (calibrated R^2=0.92) | PARTIALLY -- the framework motivated the specific algorithm, but clustering correlation matrices is standard | **BORDERLINE** -- closest to meeting all four, but the core clustering step is not novel |

**No result meets ALL FOUR criteria unambiguously.**

The closest candidate is the **Noether counting** result, which meets (a), (c), and arguably (b). But criterion (d) is the killer: you do not need the framework to derive g(g-1)/2. You need only the observation "within-group rankings are coin flips, between-group rankings are deterministic" -- which is a direct consequence of the well-known fact that correlated features have unstable importance rankings.

---

### FINAL VERDICT

**The framework is a language, not a tool.**

More precisely:

1. **The core impossibility theorem** (faithful + stable + decisive = impossible under Rashomon) is a **genuine theorem**, rigorously proved in Lean 4 with zero axioms. It is also **trivially true** -- it says that a non-injective map loses information, dressed in the language of faithfulness, stability, and decisiveness. Ten of 22 simulated reviewers independently identified this.

2. **The eta formula** (dim(V^G)/dim(V)) is a **correct but pre-existing** calculation in representation theory. In every domain where it is applied (SHAP, neural nets, quantum mechanics), the specific result was already known. The framework's contribution is observing that these known results share a common mathematical structure.

3. **The Noether counting result** is the **strongest original contribution**. It provides an actionable, quantitative prediction for ML practitioners (you can ask exactly g(g-1)/2 reliable questions about feature importance). This result could have been discovered without the framework, but the framework motivated the specific investigation.

4. **The cross-domain "unification"** is the framework's primary claim to novelty. Whether this constitutes a scientific contribution depends on whether you believe that identifying common mathematical structure across domains is valuable even when it produces no new predictions within any individual domain. This is a matter of scientific philosophy, not a matter of fact.

**What the framework IS:**
- A formal language for discussing explanation instability
- A correct (Lean-verified) impossibility theorem
- A motivational lens that led to the Noether counting experiment
- A pedagogical unification of known results across domains

**What the framework is NOT:**
- A tool that produces predictions you could not derive without it
- A quantitative theory that predicts new numbers (all specific numbers were previously known or follow from elementary combinatorics)
- A "law of physics" analogous to Noether's theorem (despite the naming)

**Recommendation:** The paper should be honest about this status. Lead with Noether counting (the genuinely new and useful result). Present the framework as what it is: an organizing principle that motivated the investigation, not a theory that generates novel predictions. Drop the "universal law" rhetoric. The Lean formalization is impressive engineering and worth publishing on its own merits as a contribution to formal methods.
