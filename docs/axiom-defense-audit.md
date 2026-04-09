# Adversarial Audit: The Attribution Impossibility

Comprehensive attack/defense analysis of every axiom, definition, and framing
choice. Prepared for NeurIPS 2026 rebuttal preparation.

---

## Phase 1: Per-Axiom Attack & Defense

### Axiom 1: `Model : Type`

**Attack:** "You axiomatize an abstract `Model` type with no structure. This means
`Model` could be empty, or it could be a singleton. The entire impossibility could
be vacuously true over an uninhabited type."

**Defense:** The Rashomon property (hypothesis to Theorem 1) existentially quantifies
over `Model` --- it asserts `exists f f' : Model, ...`. If `Model` were empty, the
Rashomon property would be `False`, and the theorem would be vacuously true but
inapplicable. The `Consistency.lean` file constructs a concrete `CModel := Fin 4`
satisfying all property axioms, proving the axiom system is satisfiable. The abstract
type is standard practice in Lean formalizations (cf. Nipkow's social choice
formalization).

**Verdict: ROCK SOLID.** Standard type-theoretic abstraction with a concrete
consistency witness.

---

### Axiom 2: `numTrees : N`

**Attack:** "You axiomatize the number of boosting rounds as a fixed global
constant. In practice, T is a hyperparameter that varies across models. Two
XGBoost runs with different `n_estimators` violate this."

**Defense:** The axiom models a fixed experimental protocol: all models in the
Rashomon set are trained with the same hyperparameters (only the random seed
changes). This is the standard setup for stability analysis. If T varies, the
split-count formulas need modification, but the core impossibility (Theorem 1,
which depends only on the Rashomon property) is unaffected. T-variation only
affects the quantitative bounds.

**Verdict: DEFENSIBLE.** The fixed-T assumption is standard and explicitly scoped.
The core impossibility does not depend on it.

---

### Axiom 3: `numTrees_pos : 0 < numTrees`

**Attack:** "Trivially true for any practical GBDT. Why axiomatize it?"

**Defense:** Lean requires proof that the denominator `2 - rho^2` and numerator
`T` are nonzero for the split-count division. Without `numTrees_pos`, the
split-count values would be 0, making the gap calculation degenerate. This is a
technical necessity, not a domain assumption.

**Verdict: ROCK SOLID.** Pure technical hygiene.

---

### Axiom 4: `attribution : Fin fs.P -> Model -> R`

**Attack:** "You treat attribution as a deterministic function of (feature, model).
SHAP values depend on the background dataset, the ordering of coalitions, and can
be stochastic (KernelSHAP). Different calls on the same model give different values."

**Defense:** We model the expected (exact TreeSHAP) attribution, which is
deterministic given a model and dataset. Stochastic approximations (KernelSHAP)
converge to this value. The type signature `Fin fs.P -> Model -> R` correctly
captures that global feature importance is a deterministic function of the model.
If the background dataset also varies, it can be absorbed into the `Model` type.

**Verdict: DEFENSIBLE.** The deterministic-attribution assumption is the standard
TreeSHAP setting. Stochastic methods converge to it.

---

### Axiom 5: `splitCount : Fin fs.P -> Model -> R`

**Attack:** "Split counts are natural numbers, not reals. You use R to avoid
the fact that T/(2-rho^2) is irrational. This is a type-theoretic hack that
lets you state exact equalities that cannot hold for actual integer split counts."

**Defense:** The Lean file explicitly documents this: "Returns R (not N) to avoid
inconsistency: the axiomatized values T/(2-rho^2) are generally irrational. The
values represent idealized leading-order split counts from the paper's Gaussian
conditioning argument." This is a standard continuous approximation of a discrete
quantity. The actual split counts are within O(sqrt(T)) of these values by
concentration, so the leading-order analysis is valid. If you axiomatized N,
you would need floor/ceiling operations that obscure the algebra without changing
the qualitative conclusions.

**Verdict: DEFENSIBLE.** Continuous relaxation of a discrete quantity is standard.
The leading-order argument is justified by concentration.

---

### Axiom 6: `firstMover : Model -> Fin fs.P`

**Attack:** "In real GBDT, EVERY tree has a root split, and they can differ across
trees. The 'first mover' is tree 1's root, but tree 2 might split on a different
feature. Your abstraction collapses the entire sequential structure into a single
feature per model."

**Defense:** The first-mover abstraction captures the leading-order effect: the
feature selected at tree 1's root absorbs the collinear signal, and subsequent
trees see only the residual. This is the Gaussian conditioning argument (Lemma 1
in the paper). Later trees may split on other features, but the cumulative split
count over T rounds is dominated by the initial selection. The abstraction is
validated empirically: the first-mover has split count T/(2-rho^2) vs.
(1-rho^2)T/(2-rho^2) for others, matching simulations within the proportionality
CV of 0.35 (stumps).

**Verdict: DEFENSIBLE.** The leading-order abstraction is empirically validated.
Deeper trees increase CV but do not eliminate the first-mover effect.

---

### Axiom 7: `firstMover_surjective`

**Attack:** "This claims every feature in a group can be the first-mover for SOME
model. But in practice, with finite training data and fixed hyperparameters, gain
ties are broken deterministically (e.g., by feature index). Feature 0 might always
win ties, making this axiom false for real XGBoost."

**Defense:** This is the strongest attack on any single axiom. Three defenses:
(1) XGBoost and LightGBM use column subsampling (`colsample_bytree`), which
randomly excludes features from consideration, allowing any feature to win.
(2) Under the Gaussian DGP with equicorrelation, the expected gains are equal
within a group, so which feature wins depends on the subsample and data noise.
(3) Even without subsampling, different random seeds produce different bootstrap
samples, creating different gain rankings. Empirically, we observe all features
in a correlated group serving as the most-split feature across 50 models on
Breast Cancer.

**Verdict: DEFENSIBLE.** Requires column subsampling or data randomness. The axiom
is false for a deterministic algorithm with no randomness on a fixed dataset.
The paper should explicitly state the role of `colsample_bytree` or bootstrap
randomness.

---

### Axiom 8: `splitCount_firstMover` (T/(2-rho^2))

**Attack:** "This exact formula requires: (a) Gaussian features, (b) equicorrelation,
(c) infinite depth (full signal capture), (d) no regularization. Real GBDT has
non-Gaussian features, heterogeneous correlations, finite depth with alpha ~
2/pi, and L2 regularization. The formula is wrong for depth-3 trees because the
partial signal capture means the first-mover gets roughly alpha*T/(2-alpha^2*rho^2)
splits, not T/(2-rho^2)."

**Defense:** The paper explicitly acknowledges this in the limitations: "The
split-count axioms assume full signal capture (alpha = 1); for finite-depth trees,
alpha ~ 2/pi (R^2 = 0.89). The impossibility holds for any alpha > 0." The formula
is the leading-order term under the Gaussian conditioning argument. The
AlphaFaithful.lean file generalizes to alpha-corrected bounds. The core
impossibility does not depend on the exact formula --- only on the EXISTENCE of
a split-count gap, which holds for any alpha > 0.

**Verdict: DEFENSIBLE.** The exact value is approximate for real trees, but the
qualitative conclusion (gap exists, ratio diverges) holds for any alpha > 0.

---

### Axiom 9: `splitCount_nonFirstMover` ((1-rho^2)T/(2-rho^2))

**Attack:** Same as Axiom 8. Additionally: "For groups with m > 2 features, the
non-first-mover split counts are NOT all equal. The second-selected feature gets
more splits than the third, etc. Your axiom assumes all non-first-movers are
identical."

**Defense:** Under equicorrelation, the residual after removing the first-mover's
collinear component is symmetric across the remaining group members. The signal
available to feature k (given j was selected first) is (1-rho^2) * original signal,
by the Gaussian conditioning formula. This is the same for ALL non-first-movers
within the group, because they all have the same correlation rho with the first-mover.
For non-equicorrelated features, the formula would differ per feature, but the key
property (first-mover dominates) would persist.

**Verdict: DEFENSIBLE.** Correct under equicorrelation. Non-equicorrelation
changes the exact values but preserves the qualitative structure.

---

### Axiom 10: `proportionality_global` (phi = c * n for UNIFORM c)

**Attack:** "This is the most empirically vulnerable axiom. You claim a SINGLE
constant c works across ALL models. The CV data in the paper shows CV = 0.66 for
depth-6 trees. That means c varies by +/-66% across features within a single
model, let alone across models. A 'global' constant with 66% variation is not
a constant at all. For depth-3 trees with interaction effects, c_j depends on
which features appear in the same tree as j, which varies across models."

**Defense:** The proportionality axiom is used only for the quantitative bounds
(attribution ratio = 1/(1-rho^2), DASH equity). The core impossibility
(Theorem 1) does not depend on it at all --- it requires only the Rashomon
property. For stumps (depth 1), CV = 0.35, which is reasonable for a
leading-order approximation. The paper explicitly states: "The proportionality
axiom holds with CV ~ 0.35 for stumps and CV ~ 0.66 for depth-6 trees." The
stronger version (global c across models) is justified by fixed hyperparameters:
if all models use the same max_depth, learning_rate, etc., the per-split
contribution is determined by the algorithm, not the data, giving a model-independent
c. The variation is in the O(1) constant, not the scaling.

**Verdict: VULNERABLE.** The per-model CV of 0.66 for deep trees makes the global-c
claim strained. The defense must emphasize: (1) the core impossibility does not use
this axiom, (2) it is validated for stumps, (3) the ratio bound is an order-of-magnitude
bound, not an exact prediction.

---

### Axiom 11: `modelMeasurableSpace`

**Attack:** "You axiomatize a measurable space on an abstract type with no structure.
There's no guarantee the sigma-algebra is the right one. You could have the trivial
sigma-algebra {empty, Model}, making all functions measurable but all integrals
trivial."

**Defense:** This is standard measure-theoretic infrastructure. The axiom exists
solely to connect to Mathlib's `ProbabilityTheory.variance`. The choice of
sigma-algebra does not affect any downstream theorem (the variance bound is
existential: "there exists v = Var(phi_j)/M >= 0"). A richer sigma-algebra would
give sharper results but is not needed for the existential statement.

**Verdict: ROCK SOLID.** Standard infrastructure axiom with no domain content.

---

### Axiom 12: `modelMeasure`

**Attack:** "Same as above. You axiomatize a measure without requiring it to be a
probability measure. The variance definition requires integrability, which you
don't verify."

**Defense:** The `consensus_variance_bound` theorem only asserts existence of a
nonneg value equal to Var(phi_j)/M. The deeper result (Var(X-bar) = Var(X)/n
for i.i.d. draws) is available in Mathlib but not invoked. The paper explicitly
notes: "The variance bound is axiomatized; the full measure-theoretic derivation
from Mathlib's IndepFun.variance_sum is deferred." This is a transparent limitation.

**Verdict: DEFENSIBLE.** The existential statement is trivially true. The deeper
convergence claim is deferred, not false.

---

### Axiom 13: `splitCount_crossGroup_symmetric`

**Attack:** "You claim features in a group have equal split counts when the
first-mover is in a different group. But if group 1 has features {A, B} and
group 2's first-mover is C, why would n_A = n_B? Feature A might have higher
marginal variance, or the residual after C's selection might favor A over B."

**Defense:** Under the equicorrelation DGP, features within a group are
exchangeable conditional on the first-mover being outside the group. If the
first-mover is in another group, the residual signal available to each feature
in this group is determined solely by their correlation structure with the
first-mover and with each other. Under equicorrelation (all within-group pairs
share rho, all between-group pairs share some other correlation), the residual
is symmetric. This is the DGP symmetry argument.

**Verdict: DEFENSIBLE.** Requires equicorrelation and the Gaussian DGP. Holds by
exchangeability.

---

### Axiom 14: `splitCount_crossGroup_stable`

**Attack:** "You claim changing the first-mover within one group does not affect
split counts for features in other groups. But in GBDT, the first tree's root
split determines the residual for ALL subsequent trees. If group 1's first-mover
changes from A to B, the residuals change, which affects the split opportunities
for features in group 2."

**Defense:** Under the equicorrelation structure, features A and B within a group
have identical correlation with all features outside the group. Therefore, the
residual signal available to features in other groups after conditioning on A
(as first-mover) is identical to the residual after conditioning on B. The
residual depends on the first-mover's correlation with the target and with
cross-group features, both of which are the same for A and B under equicorrelation.

**Verdict: DEFENSIBLE.** Same defense as Axiom 13 --- requires equicorrelation.

---

### Axiom 15: `testing_constant`

**Attack:** "You axiomatize the Le Cam constant as an abstract positive real.
The actual value (C = 1/8 from Tsybakov 2009) is known. Why not define it?"

**Defense:** The axiom exists because formalizing Le Cam's two-point method from
first principles requires ~100+ hours (total variation distance, likelihood ratio
bounds, Gaussian product measures). The downstream results (query complexity
scaling) depend only on C > 0, not the exact value. The `le_cam_lower_bound`
theorem is actually provable from `not_lt` alone --- the real Le Cam content is
in the hypothesis `h_reliable`, which the user must discharge. This is explicitly
documented as "Track B (axiom-based)" in the code.

**Verdict: DEFENSIBLE.** The axiom is honestly presented as a shortcut. The
mathematical content it encodes (Le Cam's method) is a standard textbook result.

---

### Axiom 16: `testing_constant_pos`

**Attack:** "Trivially follows from C = 1/8 > 0."

**Defense:** Same as Axiom 15. Positivity is needed for downstream division.

**Verdict: ROCK SOLID.** Technical consequence of a well-known result.

---

### Axiom 17: `spearman_classical_bound`

**Attack:** "This is the most suspicious axiom. You define Spearman correlation
from scratch in SpearmanDef.lean, prove a weaker bound (1 - 3/(P^3-P)) from
the definition, and then axiomatize the stronger bound (1 - m^3/P^3) 'about
the defined quantity.' But the gap between the derived and axiomatized bounds
is substantial: for m=5, P=30, the derived bound gives 1 - 3/26970 = 0.99989,
while the axiomatized bound gives 1 - 125/27000 = 0.99537. The axiom does real
work --- the derived bound is nearly vacuous. You're axiom-laundering: defining
the quantity to appear principled, then axiomatizing the actual content."

**Defense:** The paper is transparent about this: "The paper uses the tighter
axiomatized bound. The full combinatorial proof (bounding the expected sum d_i^2
under random tie-breaking) requires counting arguments not yet available in the
Lean framework." The gap is precisely characterized in the code comments: it
requires proving "swap preserves midranks" (a finset cardinality argument about
filter sets under transposition), which is formalizable but tedious. The axiom is
about the DEFINED quantity `spearmanCorr` (not an opaque type), so it is falsifiable
within the system. The derived bound 1 - 3(m-1)^2/(P^3-P) in
`spearmanCorr_bound_groupSize` already captures the m-dependence.

**Verdict: VULNERABLE.** The axiom does significant quantitative work. The defense
must emphasize: (1) it is about a defined quantity, (2) the intermediate bound
1 - 3(m-1)^2/(P^3-P) is fully derived and already captures the m-scaling, (3)
closing the gap requires only combinatorial bookkeeping, not new ideas.

---

## Phase 2: Definition & Framing Attacks

### 1. Faithful (biconditional: ranking j>k iff phi_j > phi_k)

**Attack:** "All-or-nothing faithfulness is a straw man. No practitioner expects
a ranking to perfectly mirror every model's attributions. A reasonable definition
would be alpha-faithfulness: the ranking agrees with at least fraction alpha of
models. Your impossibility dissolves for alpha < 1."

**Defense:** The paper proves BOTH the biconditional version (`attribution_impossibility`)
AND the implication-only version (`attribution_impossibility_weak`). The weak
version only requires: if phi_j > phi_k then j > k (one direction). Even this
weaker version yields the impossibility. The alpha-faithfulness relaxation is
addressed in `AlphaFaithful.lean`, which quantifies the unfaithfulness rate.
The impossibility is not "dissolved" by relaxing faithfulness --- it is QUANTIFIED:
any method must be unfaithful to at least half the orbit-mates (SBD binary orbit
theorem). The biconditional is the strongest result; the paper also provides the
weaker versions.

**Verdict: DEFENSIBLE.** The weak version and quantitative bounds pre-empt this
attack. The paper should highlight `attribution_impossibility_weak` more
prominently.

---

### 2. Stable (ranking is model-independent)

**Attack:** "Model-independence is absurdly strong. Of course different models
give different rankings. The right notion is epsilon-stability: rankings change
by at most epsilon. Your impossibility is against a definition nobody uses."

**Defense:** The Spearman stability bound (Theorem 1(ii)) IS the epsilon-stability
result: Spearman <= 1 - m^3/P^3 between any two models with different first-movers.
The "impossible" result is the conjunction: you cannot have BOTH perfect faithfulness
AND stability. The quantitative stability bound tells you exactly how much stability
you can achieve (and it degrades as group size m increases). The absolute stability
definition is the IMPOSSIBILITY; the epsilon-stability is the QUANTITATIVE BOUND.

**Verdict: ROCK SOLID.** The paper proves both the impossibility (for absolute
stability) and the quantitative bound (for epsilon-stability).

---

### 3. Complete (decides all pairs)

**Attack:** "Relaxing completeness is trivially obvious. Any partial order handles
incomparable elements. This is not a deep insight --- it's the definition of a
partial order. The paper's resolution ('use partial orders where symmetric features
are incomparable') is a tautology."

**Defense:** The insight is not that partial orders exist, but that the impossibility
FORCES their use. The Design Space Theorem (DesignSpace.lean) shows the design
space is exhaustive: every method is either Family A (faithful+complete but
unstable) or Family B (DASH, stable but reports ties). There is no third option.
The resolution via DASH is not "use partial orders" but "use ensemble averaging
to achieve equity, where ties emerge as a theorem." The content is in the proof
that DASH achieves exact equity (consensus_equity), not in the observation that
partial orders exist.

**Verdict: DEFENSIBLE.** The non-trivial content is the exhaustiveness of the
design space, not the existence of partial orders.

---

### 4. Rashomon Property

**Attack:** "Is this a property of the model class or the training procedure?
The Rashomon set (models with near-optimal loss) is well-studied. Your definition
is about training procedure outputs (different random seeds), not about the loss
landscape. You're conflating 'model class diversity' with 'training instability.'"

**Defense:** The Rashomon property as defined requires: for any two symmetric
features j, k, there exist models f, f' ranking them in opposite orders. This
is about the MODEL CLASS (the set of models reachable by the training procedure).
The `gbdt_rashomon` theorem derives this from the first-mover axioms (different
seeds produce different first-movers). The connection to the loss-landscape
Rashomon set is: under equicorrelation, models with different first-movers have
comparable loss (the loss difference is O(rho^4)), so they lie in the Rashomon
set. The paper's `RashomonInevitability.lean` formalizes the inevitability
of the Rashomon property.

**Verdict: DEFENSIBLE.** The connection between training-procedure Rashomon and
loss-landscape Rashomon should be made explicit. The `RashomonInevitability`
file addresses this.

---

### 5. Equicorrelation

**Attack:** "Equicorrelation (all within-group pairs share rho) is unrealistic.
Real feature groups have heterogeneous correlations. The Iris dataset has
sepal_length and sepal_width at rho = -0.12, but petal_length and petal_width
at rho = 0.96. Your theory applies only to the petal pair, not to arbitrary
feature groups."

**Defense:** The paper explicitly states: "The equicorrelation assumption simplifies
the axioms; the Rashomon property holds pairwise." The impossibility requires only
that SOME pair of features has rho > 0 and equal true coefficients. It does not
require all features to be equicorrelated. The equicorrelation assumption is a
convenience for stating clean formulas, not a necessary condition. The pairwise
version (L=1, m=2) is the minimal case and already gives the impossibility.

**Verdict: DEFENSIBLE.** The minimal case (one correlated pair) requires no
equicorrelation assumption.

---

### 6. First-mover abstraction

**Attack:** "In real XGBoost with 500 trees, the first tree's root split is ONE
split out of thousands. The 'first-mover advantage' is diluted by subsequent
trees. For depth-6 trees with 63 leaves each, the first root split accounts for
1/(500*63) = 0.003% of all splits. How can this dominate?"

**Defense:** The first-mover advantage is not about a single split but about
CUMULATIVE advantage. The first-mover gets T/(2-rho^2) total splits across all
trees (not just one split), because once a feature is established as the best
split for the collinear signal, it remains the best choice in subsequent trees'
upper levels. The non-first-mover gets (1-rho^2)T/(2-rho^2) total splits. The
ratio is 1/(1-rho^2), which diverges as rho -> 1. The depth of individual trees
affects the alpha parameter (signal capture per split), not the number of times
the feature is selected. Empirically, the first-mover effect is robust: the
most-split feature gets 2-10x more splits than correlated alternatives.

**Verdict: DEFENSIBLE.** The cumulative advantage is the key mechanism, not a
single split.

---

### 7. Proportionality (CV 0.35-0.66)

**Attack:** "At CV = 0.66, the proportionality 'constant' varies by a factor of
3x within a single model. This means your attribution ratio bound 1/(1-rho^2)
could be off by 3x in either direction. For rho = 0.8 (ratio = 2.78), the actual
ratio could be anywhere from 0.93 to 8.33. The bound is meaningless."

**Defense:** The CV of 0.66 is for depth-6 trees. For stumps (the setting where
the axioms are most accurate), CV = 0.35. The ratio bound is an ORDER-OF-MAGNITUDE
prediction, not a point estimate. The alpha-corrected version accounts for finite
depth. Most importantly, the core impossibility (Theorem 1) does not use
proportionality at all. The ratio bound is a QUANTITATIVE refinement that
indicates the severity, not the existence, of the problem.

**Verdict: VULNERABLE.** The quantitative bounds degrade for deep trees. The
defense is: core impossibility is unaffected, and stumps are well-approximated.

---

### 8. Balanced ensemble

**Attack:** "A balanced ensemble requires each feature to serve as first-mover
equally often. For a group of size m, you need M to be a multiple of m. For
m = 10, that's M = 10, 20, 30, ... With 25 models (the paper's recommendation),
you get exact balance only if m divides 25. Otherwise, the equity theorem is
approximate, and the paper's claim of 'exact equity' is misleading."

**Defense:** The `IsBalanced` definition requires exact balance. The paper notes
that this "holds in expectation for i.i.d. seeds by DGP symmetry, and exactly
when M is a multiple of the group size." For approximate balance, the equity
violation is O(1/M), which converges to zero. The `consensus_variance_rate`
theorem quantifies the convergence. The practical recommendation (M = 25)
gives near-balance for typical group sizes (2-5 features), with equity
violation below the noise floor.

**Verdict: DEFENSIBLE.** The exact-balance theorem is a clean statement.
Approximate balance gives O(1/M) equity violation, which is acceptable in
practice.

---

## Phase 3: Cross-Disciplinary Attacks

### Reviewer A (Classical Statistician): "Non-identifiability under collinearity is textbook. What's new?"

**Attack (full):** "Non-identifiability of regression coefficients under perfect
collinearity is a first-year statistics result. The fact that correlated features
get unstable importance scores is a known consequence. This paper dresses up a
textbook observation in formal logic without adding insight. The 1/(1-rho^2) ratio
is just the variance inflation factor (VIF), which has been used since the 1970s.
The Lean formalization is window dressing."

**Valid? PARTIALLY.** The VIF connection is real and should be acknowledged. The
non-identifiability observation is indeed textbook.

**Defense:** Three things ARE new: (1) The impossibility is about RANKINGS, not
coefficient estimates. Even methods that don't estimate coefficients (SHAP,
permutation importance) are subject to it. The classical statistician's mental
model ("just look at the confidence interval") doesn't apply to SHAP.
(2) The QUANTITATIVE characterization (Spearman bound, design space exhaustiveness)
is new. Knowing "instability exists" and knowing "Spearman <= 1 - m^3/P^3 with
no escape" are different. (3) The RESOLUTION (DASH) is constructive and provably
optimal (ties are the only stable output). The VIF tells you the problem; we prove
the impossibility AND provide the resolution.

---

### Reviewer B (Causal Inference): "Equal causal effects is rare. The escape condition is the normal case."

**Attack (full):** "The impossibility requires features with 'the same true
coefficient.' In practice, correlated features almost never have identical causal
effects. Feature X1 = 'tumor size' and X2 = 'tumor volume' are correlated, but
their causal effects differ. The escape condition (unequal effects) is the generic
case. Your impossibility applies to a measure-zero subset of problems."

**Valid? PARTIALLY.** The equal-coefficient assumption IS restrictive in causal
settings.

**Defense:** (1) The impossibility applies to PREDICTIVE importance, not causal
effects. Two features with identical predictive power but different causal effects
are still subject to the impossibility for ranking by prediction contribution.
(2) The impossibility degrades gracefully: for features with NEARLY equal effects
(|beta_j - beta_k| < epsilon), the instability persists for any model where the
noise exceeds the signal gap. The `ConditionalImpossibility.lean` formalizes this.
(3) In high-dimensional genomics/proteomics, groups of co-regulated genes DO have
similar effects (pathway-level redundancy). The equal-effect case is not measure-zero
in practice.

---

### Reviewer C (Deep Learning): "SHAP will be obsolete. Mechanistic interpretability is the future."

**Attack (full):** "Feature attribution (SHAP, LIME) is a shallow explanation
method that will be superseded by mechanistic interpretability (circuit analysis,
probing, concept-level explanations). Proving impossibility results about a dying
paradigm is like proving impossibility results about horse-drawn carriages."

**Valid? NO.** This confuses deep learning interpretability with general ML
interpretability.

**Defense:** (1) SHAP is used in production by every major bank, insurer, and
healthcare ML system for regulatory compliance (ECOA, SR 11-7, GDPR Article 22).
Mechanistic interpretability has zero regulatory adoption. (2) The impossibility
applies to ANY feature-level ranking method, including mechanistic approaches
that assign importance scores to input features. (3) For tabular data (the dominant
use case in industry), SHAP is the standard, and no mechanistic alternative exists.
(4) The Symmetric Bayes Dichotomy (SymmetricBayes.lean) generalizes beyond feature
attribution to ANY symmetric decision problem, including model selection and
causal discovery.

---

### Reviewer D (Formal Verification): "The axioms do the work, not the proofs. This is axiom laundering."

**Attack (full):** "The core impossibility theorem is 4 lines of Lean. The
Rashomon property is assumed as a hypothesis. The `gbdt_rashomon` theorem that
derives it from axioms is also trivial (2 applications of `firstMover_surjective`).
The entire proof is: 'if opposite rankings exist, no stable ranking works.' This is
obvious. The axioms (firstMover_surjective, split-count formulas,
proportionality_global) encode the answer. You're laundering empirical assumptions
through a type checker and calling it 'formally verified.' A formally verified
tautology is still a tautology."

**Valid? PARTIALLY.** The core proof IS simple. That's the point.

**Defense:** (1) The simplicity of the core proof is a FEATURE, not a bug. It means
the impossibility is robust --- it doesn't depend on elaborate constructions. Arrow's
theorem is also "simple" once you have IIA, Pareto, and unrestricted domain.
(2) The axioms are NOT arbitrary --- they are justified by the Gaussian conditioning
argument, validated empirically (CV data), and proved consistent (Consistency.lean).
(3) The non-trivial content is in the QUANTITATIVE theorems: the ratio divergence
(ratio_tendsto_atTop), the Spearman bound (spearmanCorr_bound_groupSize with 50+
lines of proof), the symmetry derivation (attribution_sum_symmetric, 70+ lines),
and the design space exhaustiveness. (4) The formalization caught three bugs in the
original axiom system (documented in the paper). This is the value of formalization:
not making simple things complicated, but catching errors in the INFRASTRUCTURE.

---

### Reviewer E (Industry Engineer): "I already know this. CI handles it. 25x cost is unacceptable."

**Attack (full):** "Every practitioner knows feature importance is unstable.
We already use confidence intervals (bootstrap SHAP, repeated runs). Your paper
says 'train 25 models' --- that's 25x the compute cost. For a model that takes
6 hours to train, that's a week of GPU time. No production team will do this.
The impossibility is academic; the resolution is impractical."

**Valid? PARTIALLY.** The compute cost concern is legitimate.

**Defense:** (1) The paper's query complexity analysis (QueryComplexity.lean) shows
that Omega(sigma^2/Delta^2) models are INFORMATION-THEORETICALLY NECESSARY for
stability certification. You cannot escape the cost --- the Z-test achieves this
rate to within a constant factor. (2) The 25x cost applies to the EXPLANATION,
not the deployed model. You train ONE model for production; the 25 models are for
AUDITING the explanation. (3) For models that take 6 hours, you'd use a smaller
ensemble (M=5-10) and accept wider confidence intervals. The `consensus_variance_rate`
theorem quantifies the precision/cost tradeoff. (4) Many organizations already
run multiple models for model validation (SR 11-7 requires challenger models).
DASH piggybacks on existing infrastructure.

---

## Phase 4: The Axiom Laundering Argument

### The Attack

"You claim 'zero axiom dependencies' for Theorem 1 (attribution_impossibility).
But look at the dependency chain:

1. Theorem 1 takes `RashimonProperty` as a HYPOTHESIS.
2. `RashimonProperty` is derived via `gbdt_rashomon` from axioms 7-9.
3. So Theorem 1 is really: 'IF axioms 7-9 hold THEN impossibility.'
4. The 'zero axiom' claim is technically true (Theorem 1 has no `axiom` in its
   proof term) but intellectually dishonest. The hard question is whether the
   Rashomon property holds, and you ASSUME it.
5. Furthermore, the impossibility is trivially true for ANY model class with the
   Rashomon property. The Rashomon property IS the impossibility, just reformulated.
   You're proving 'if rankings reverse, no stable ranking exists.' This is a
   tautology."

### The Defense

**Point 1: The "zero axiom" claim is honest and important.**

The Lean type checker verifies that `attribution_impossibility` depends on no
`axiom` declarations. Its proof term references only the hypothesis `hrash :
RashimonProperty fs` and standard logic. This is not a technicality --- it means
the impossibility holds for ANY source of the Rashomon property (GBDT, Lasso,
neural nets, future model classes). The paper states: "The core impossibility
uses zero axioms; quantitative bounds are conditional on 6 domain-specific axioms."
This is a factual description of the dependency structure.

**Point 2: The Rashomon property is NOT the impossibility restated.**

The Rashomon property says: "there exist models ranking j > k and k > j."
The impossibility says: "no SINGLE ranking can be simultaneously faithful,
stable, and complete." These are logically distinct. The Rashomon property
is about the MODEL CLASS (existence of diverse models). The impossibility is
about RANKINGS (no aggregation function works). The bridge is non-trivial:
it requires the ranking to be both faithful to ALL models (not just one) AND
model-independent. This is the same structure as Arrow's theorem: IIA + Pareto
(properties of the preference aggregation) + unrestricted domain (property of
the preference profiles) yield impossibility.

**Point 3: The hard question IS addressed.**

The paper does NOT just assume the Rashomon property. It derives it:
- For GBDT: `gbdt_rashomon` (from firstMover_surjective)
- For Lasso: `lasso_impossibility` (from selected_surjective)
- For neural nets: `nn_impossibility` (from captured_surjective)
- Generically: `iterative_rashomon` (from IterativeOptimizer)
- Inevitably: `RashomonInevitability.lean` (conditions under which it must hold)

The layered architecture (Level 0: pure logic, Level 1: framework, Level 2:
instantiation) is DESIGNED to separate the model-agnostic impossibility from
the model-specific derivation of the Rashomon property. This is good
mathematical practice, not obfuscation.

**Point 4: The tautology charge conflates logical entailment with triviality.**

By this standard, every theorem is a "tautology" (the conclusion follows from the
premises). The content of the impossibility is: (a) the precise trio of properties
that conflict (faithfulness, stability, completeness), (b) the resolution (partial
orders / DASH), (c) the quantitative bounds on the tradeoff, and (d) the
generalization (Symmetric Bayes Dichotomy). The 4-line proof reflects the
CLEANNESS of the formulation, not its triviality.

---

## Phase 5: Vulnerability Map

| # | Item | Strongest Attack | Defense Quality (1-5) | Verdict | Action Needed? |
|---|------|------------------|-----------------------|---------|----------------|
| 10 | `proportionality_global` | CV=0.66 for depth-6; "global c" fails empirically | 3 | VULNERABLE | Emphasize core impossibility independent; validate for stumps only |
| 17 | `spearman_classical_bound` | Axiomatizes the real quantitative content; derived bound is near-vacuous | 3 | VULNERABLE | Close the combinatorial gap or emphasize derived (m-1)^2 bound |
| 7 | Proportionality CV framing | CV=0.66 undermines quantitative claims | 3 | VULNERABLE | Add alpha-correction prominently; bound the error propagation |
| D | Axiom laundering (Reviewer D) | Core proof is trivial; axioms encode the answer | 3 | VULNERABLE | Highlight the non-trivial quantitative theorems and bug-catching |
| 7 | `firstMover_surjective` | Deterministic tie-breaking may break this | 4 | DEFENSIBLE | State colsample_bytree / bootstrap assumption explicitly |
| 8 | `splitCount_firstMover` | Approximate for finite depth (alpha ~ 2/pi) | 4 | DEFENSIBLE | Already addressed via alpha-correction; state limitations |
| 9 | `splitCount_nonFirstMover` | Same as 8; non-first-movers not exactly equal for m>2 | 4 | DEFENSIBLE | Equicorrelation defense; pairwise version suffices |
| 5 | `splitCount` type (R not N) | Integer split counts modeled as reals | 4 | DEFENSIBLE | Standard continuous approximation |
| 6 | `firstMover` abstraction | Abstracts 500-tree sequence into one function | 4 | DEFENSIBLE | Cumulative advantage argument |
| 4 | `attribution` type | Deterministic function; ignores KernelSHAP noise | 4 | DEFENSIBLE | Models exact TreeSHAP |
| 14 | `splitCount_crossGroup_stable` | Requires equicorrelation | 4 | DEFENSIBLE | Exchangeability argument |
| 13 | `splitCount_crossGroup_symmetric` | Requires equicorrelation | 4 | DEFENSIBLE | Exchangeability argument |
| 12 | `modelMeasure` | Not required to be probability measure | 4 | DEFENSIBLE | Existential statement is trivially true |
| 15 | `testing_constant` | Known value, not defined | 4 | DEFENSIBLE | Honestly labeled as axiom-based track |
| 1 | Faithful (biconditional) | Straw man; alpha-faithfulness dissolves it | 4 | DEFENSIBLE | Weak version and quantitative bounds pre-empt |
| 4(def) | Rashomon Property | Model class vs. training procedure confusion | 4 | DEFENSIBLE | Both are addressed; RashomonInevitability |
| 5(def) | Equicorrelation | Unrealistic for heterogeneous correlations | 4 | DEFENSIBLE | Pairwise version suffices |
| 6(def) | First-mover abstraction | Over-simplified for real GBDT | 4 | DEFENSIBLE | Empirically validated |
| 8(def) | Balanced ensemble | Requires M divisible by m | 4 | DEFENSIBLE | O(1/M) approximation |
| 2(def) | Stable (model-independent) | Too strong; epsilon-stability is natural | 5 | ROCK SOLID | Paper proves both |
| 3(def) | Complete (decides all pairs) | Relaxation is "obvious" | 4 | DEFENSIBLE | Design space exhaustiveness is non-trivial |
| A | Classical statistician | VIF is textbook | 4 | DEFENSIBLE | Rankings, not coefficients; quantitative; resolution |
| B | Causal inference | Equal effects is rare | 4 | DEFENSIBLE | Predictive, not causal; graceful degradation |
| C | Deep learning | SHAP is obsolete | 5 | ROCK SOLID | Regulatory reality; tabular data; SBD generalizes |
| E | Industry engineer | 25x cost | 4 | DEFENSIBLE | Information-theoretic necessity; audit not deploy |
| 1 | `Model : Type` | Could be empty | 5 | ROCK SOLID | Consistency witness |
| 2 | `numTrees : N` | Fixed T is restrictive | 5 | ROCK SOLID | Core impossibility independent |
| 3 | `numTrees_pos` | Trivial | 5 | ROCK SOLID | Technical hygiene |
| 11 | `modelMeasurableSpace` | Trivial sigma-algebra | 5 | ROCK SOLID | Standard infrastructure |
| 16 | `testing_constant_pos` | Trivial | 5 | ROCK SOLID | Technical hygiene |

---

## The Rebuttal Cheat Sheet

### VULNERABLE: `proportionality_global` (Axiom 10)

> The proportionality axiom is used ONLY for the quantitative bounds (attribution
> ratio, DASH equity), not for the core impossibility (Theorem 1). The CV = 0.35
> for stumps validates the leading-order approximation; the CV = 0.66 for depth-6
> trees motivates the alpha-correction (AlphaFaithful.lean), which generalizes
> the bounds to any signal-capture fraction alpha > 0. The core impossibility ---
> that no stable, faithful, complete ranking exists --- requires zero axioms beyond
> the Rashomon property.

### VULNERABLE: `spearman_classical_bound` (Axiom 17)

> The formalization contains TWO Spearman bounds: (i) `spearmanCorr_bound_groupSize`,
> FULLY DERIVED from the Spearman definition, giving rho_S <= 1 - 3(m-1)^2/(P^3-P);
> and (ii) `spearman_classical_bound`, axiomatized, giving the tighter rho_S <= 1 -
> m^3/P^3. Both capture the key qualitative insight (instability grows with group
> size m). The gap between them is a combinatorial counting argument (swap-preserves-
> midranks), not a domain-specific assumption. We chose to axiomatize the tighter
> bound rather than prove a weaker theorem.

### VULNERABLE: Proportionality CV framing

> Table 3 in the paper reports CV = 0.35 (stumps) and 0.66 (depth-6). The
> proportionality axiom is a leading-order approximation, not an exact identity.
> The quantitative bounds should be interpreted as order-of-magnitude predictions,
> refined by the alpha-correction for finite depth. The core impossibility does
> not depend on proportionality.

### VULNERABLE: Axiom laundering (Reviewer D)

> The core impossibility is deliberately simple --- it is a 4-line proof because
> the formulation cleanly separates concerns. The non-trivial contributions are:
> (a) identifying the precise trio (faithfulness, stability, completeness) that
> conflicts, (b) the quantitative bounds (50+ lines for Spearman, 70+ lines for
> symmetry derivation), (c) the design space exhaustiveness theorem, (d) the
> resolution via DASH with provable equity, and (e) the formalization catching
> three bugs in the original axiom system. The Lean formalization's value is in
> INFRASTRUCTURE verification, not in making simple proofs complicated.

### DEFENSIBLE: `firstMover_surjective` (Axiom 7)

> First-mover surjectivity requires that each feature in a correlated group can
> serve as the root split of tree 1 under some random seed. This is guaranteed by
> column subsampling (`colsample_bytree < 1`, the default in XGBoost and LightGBM)
> or by bootstrap sampling noise under the Gaussian DGP. We validate it empirically:
> across 50 XGBoost models on Breast Cancer, all features in each correlated group
> serve as the most-split feature at least once.

### DEFENSIBLE: Split-count formulas (Axioms 8-9)

> The exact formulas T/(2-rho^2) and (1-rho^2)T/(2-rho^2) are leading-order
> under the Gaussian conditioning argument with full signal capture (alpha = 1).
> For finite-depth trees, alpha ~ 2/pi (R^2 = 0.89). The impossibility holds for
> any alpha > 0: the first-mover advantage ratio is 1/(1 - alpha^2 * rho^2), which
> diverges as rho -> 1 for any fixed alpha > 0. The paper states this limitation
> explicitly in Section 7.
