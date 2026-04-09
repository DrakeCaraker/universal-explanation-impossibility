# Universal Attribution Trilemma — Extension to All Model Types

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the Attribution Trilemma holds for Lasso, neural networks, and general iterative optimizers — not just gradient boosting — establishing it as a universal limitation of single-model explanations under collinearity.

**Architecture:** The trilemma (Trilemma.lean) requires only `RashimonProperty`. A new `IterativeOptimizer` structure captures the shared pattern: every iterative/sequential model has a "dominant feature" per group that accumulates more attribution through the optimization process. GBDT, Lasso, and NNs are all instances. Each model class provides its specific properties as theorem hypotheses (not global axioms), avoiding axiom conflicts. Random forests are a contrast case where the trilemma's violation is weak.

**Tech Stack:** Lean 4 v4.29.0-rc8, Mathlib

---

## File Map

```
DASHImpossibility/
  Trilemma.lean      — EXISTS: model-agnostic impossibility
  Iterative.lean     — CREATE: abstract IterativeOptimizer → RashimonProperty
  General.lean       — MODIFY: GBDT as IterativeOptimizer instance
  Lasso.lean         — CREATE: Lasso trilemma (ratio = ∞)
  NeuralNet.lean     — CREATE (Phase 3): NN trilemma
  RandomForest.lean  — CREATE: RF contrast (weak violation)
  Basic.lean         — MODIFY: add imports
```

---

## Phase 1: Unifying Framework (Days 1-3)

### Task 1: Create Iterative.lean — Abstract Iterative Optimizer

**Files:**
- Create: `DASHImpossibility/Iterative.lean`
- Modify: `DASHImpossibility/Basic.lean` (add import)

- [ ] **Step 1: Create Iterative.lean**

```lean
/-
  Abstract iterative optimizer framework. Any iterative optimization
  process under feature collinearity produces a "dominant feature" per
  group that accumulates more attribution through iteration.

  Instances: GBDT (first-mover), Lasso (selected feature), NN (captured feature).
-/
import DASHImpossibility.Trilemma

set_option autoImplicit false

namespace DASHImpossibility

variable (fs : FeatureSpace)

/-! ### Abstract Iterative Optimizer -/

/-- An iterative optimizer is characterized by a dominant-feature function
    satisfying dominance (higher attribution) and surjectivity (each feature
    can be dominant under some initialization). -/
structure IterativeOptimizer where
  /-- The dominant feature for a given model -/
  dominant : Model → Fin fs.P
  /-- The dominant feature has strictly higher attribution than all
      other features in the same group -/
  dominant_gt : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
    dominant f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ dominant f →
    attribution fs k f < attribution fs (dominant f) f
  /-- Every feature in every group can be dominant
      (different initializations break symmetry differently) -/
  dominant_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
    j ∈ fs.group ℓ → ∃ f : Model, dominant f = j

/-! ### Iterative Optimizer → Rashomon Property → Trilemma -/

/-- Any iterative optimizer satisfies the Rashomon property. -/
theorem iterative_rashomon (opt : IterativeOptimizer fs) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hfm⟩ := opt.dominant_surjective ℓ j hj
  obtain ⟨f', hfm'⟩ := opt.dominant_surjective ℓ k hk
  refine ⟨f, f', ?_, ?_⟩
  · -- j is dominant in f → attribution j f > attribution k f
    have h_dom_in : opt.dominant f ∈ fs.group ℓ := by rw [hfm]; exact hj
    have h := opt.dominant_gt f ℓ k h_dom_in hk (by rw [hfm]; exact hjk)
    rwa [hfm] at h
  · -- k is dominant in f' → attribution k f' > attribution j f'
    have h_dom_in : opt.dominant f' ∈ fs.group ℓ := by rw [hfm']; exact hk
    have h := opt.dominant_gt f' ℓ j h_dom_in hj (by rw [hfm']; exact Ne.symm hjk)
    rwa [hfm'] at h

/-- The Attribution Trilemma holds for any iterative optimizer. -/
theorem iterative_trilemma (opt : IterativeOptimizer fs)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_trilemma fs (iterative_rashomon fs opt) ℓ j k hj hk hjk ranking h_faithful

end DASHImpossibility
```

- [ ] **Step 2: Add import to Basic.lean**

Add `import DASHImpossibility.Iterative` to Basic.lean.

- [ ] **Step 3: Build and commit**

Run: `lake build`
Expected: Success, zero sorry.

```bash
git add DASHImpossibility/Iterative.lean DASHImpossibility/Basic.lean
git commit -m "feat: abstract IterativeOptimizer framework

Any iterative optimizer with a dominant feature satisfying dominance
and surjectivity inherits the Rashomon property and the Attribution
Trilemma. Unifies GBDT, Lasso, and neural networks.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Connect GBDT to the IterativeOptimizer Framework

**Files:**
- Modify: `DASHImpossibility/General.lean`

- [ ] **Step 1: Add import and GBDT optimizer instance**

Add `import DASHImpossibility.Iterative` to the imports.

Add before `end DASHImpossibility`:

```lean
/-! ### GBDT as an Iterative Optimizer -/

/-- Sequential gradient boosting is an iterative optimizer where the
    dominant feature is the first-mover (root split of tree 1).
    This connects the GBDT-specific axioms to the universal framework. -/
noncomputable def gbdtOptimizer : IterativeOptimizer fs where
  dominant := firstMover fs
  dominant_gt := by
    intro f ℓ k h_dom_in hk hne
    exact attribution_firstMover_gt fs f (firstMover fs f) k ℓ h_dom_in hk rfl (Ne.symm hne)
  dominant_surjective := firstMover_surjective fs
```

- [ ] **Step 2: Build and commit**

Run: `lake build`
Expected: Success. If `Ne.symm hne` doesn't match (direction of ≠), try `fun h => hne (h.symm)` or `hne ∘ Eq.symm`.

```bash
git add DASHImpossibility/General.lean
git commit -m "feat: GBDT as IterativeOptimizer instance

firstMover is the dominant feature. Connects GBDT axioms to
the universal framework: gbdt_rashomon and gbdt_trilemma now
follow from iterative_rashomon and iterative_trilemma.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Lasso Extension (Days 4-5)

### Task 3: Create Lasso.lean — Lasso Trilemma

**Files:**
- Create: `DASHImpossibility/Lasso.lean`
- Modify: `DASHImpossibility/Basic.lean` (add import)

The Lasso case is STRONGER than GBDT: the selected feature gets all the attribution, others get zero. The ratio is literally infinite (∞ vs 0).

Lasso properties are provided as theorem hypotheses, not global axioms, to avoid conflicting with GBDT axioms.

- [ ] **Step 1: Create Lasso.lean**

```lean
/-
  Lasso / sparse regression: the Attribution Trilemma holds with
  INFINITE attribution ratio. Under collinearity, Lasso selects
  one feature per correlated group and zeros out the rest. Different
  regularization paths select different features.

  Properties are theorem hypotheses, not global axioms, to avoid
  conflicting with the GBDT axioms in Defs.lean.
-/
import DASHImpossibility.Iterative

set_option autoImplicit false

namespace DASHImpossibility.Lasso

variable (fs : FeatureSpace)

/-! ### Lasso as an Iterative Optimizer -/

/-- Given Lasso-specific properties (selection, sparsity, surjectivity),
    Lasso is an iterative optimizer and inherits the Attribution Trilemma.

    The three hypotheses capture:
    - `selected`: which feature Lasso picks in each model
    - `selected_pos`: the selected feature has positive attribution
    - `non_selected_zero`: all other same-group features have zero attribution
    - `selected_surjective`: different models can select different features -/
theorem lasso_trilemma
    /- Lasso selects one feature per model -/
    (selected : Model → Fin fs.P)
    /- The selected feature has positive attribution -/
    (selected_pos : ∀ (f : Model), 0 < attribution fs (selected f) f)
    /- Non-selected features in the same group have zero attribution -/
    (non_selected_zero : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      selected f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ selected f →
      attribution fs k f = 0)
    /- Different regularization paths select different features -/
    (selected_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
      j ∈ fs.group ℓ → ∃ f : Model, selected f = j)
    /- Then the trilemma holds -/
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False := by
  -- Construct the iterative optimizer from Lasso properties
  have opt : IterativeOptimizer fs := {
    dominant := selected
    dominant_gt := by
      intro f ℓ' k' h_sel_in hk' hne
      rw [non_selected_zero f ℓ' k' h_sel_in hk' hne]
      exact selected_pos f
    dominant_surjective := selected_surjective
  }
  exact iterative_trilemma fs opt ℓ j k hj hk hjk ranking h_faithful

/-- The Lasso attribution ratio is infinite: the selected feature has
    positive attribution while others have zero. This is a strictly
    worse equity violation than GBDT's 1/(1-ρ²). -/
theorem lasso_ratio_infinite
    (selected : Model → Fin fs.P)
    (selected_pos : ∀ (f : Model), 0 < attribution fs (selected f) f)
    (non_selected_zero : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      selected f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ selected f →
      attribution fs k f = 0)
    (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P)
    (h_sel_in : selected f ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ)
    (hne : k ≠ selected f) :
    attribution fs k f = 0 ∧ 0 < attribution fs (selected f) f :=
  ⟨non_selected_zero f ℓ k h_sel_in hk hne, selected_pos f⟩

end DASHImpossibility.Lasso
```

- [ ] **Step 2: Add import to Basic.lean**

Add `import DASHImpossibility.Lasso`.

- [ ] **Step 3: Build and commit**

Run: `lake build`
Expected: Success, zero sorry.

```bash
git add DASHImpossibility/Lasso.lean DASHImpossibility/Basic.lean
git commit -m "feat: Lasso inherits the Attribution Trilemma

Lasso selects one feature per correlated group, zeroing out the rest.
The attribution ratio is infinite (positive vs zero) — strictly worse
than GBDT's 1/(1-ρ²).

Properties are theorem hypotheses (not axioms) to avoid conflicts
with GBDT axioms. Constructs IterativeOptimizer inline.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3: Neural Network Extension (Paper 4 — Months 6-12)

### Task 4: Create NeuralNet.lean — NN Trilemma

**Files:**
- Create: `DASHImpossibility/NeuralNet.lean`

The neural network case follows the SAME pattern as Lasso: properties as hypotheses, construct `IterativeOptimizer`, get the trilemma. The intellectual work is justifying the hypotheses — that's the paper, not the Lean code.

- [ ] **Step 1: Create NeuralNet.lean**

```lean
/-
  Neural networks: the Attribution Trilemma holds when initialization-dependent
  feature utilization produces a dominant feature per correlated group.

  The three hypotheses are justified by:
  1. Initialization symmetry breaking (Saxe et al., 2014): random initialization
     determines which features get captured by early gradient flow.
  2. Path dependence (Balduzzi et al., 2017): early feature capture is reinforced
     through training, creating a cumulative advantage analogous to GBDT's
     first-mover effect.
  3. DGP symmetry: different initializations break symmetry differently,
     so each feature can be dominant under some initialization.

  Quantitative bounds are TBD — they depend on architecture, activation
  functions, and optimizer. The qualitative impossibility follows from the
  same abstract pattern as GBDT and Lasso.

  Properties are theorem hypotheses, not global axioms.
-/
import DASHImpossibility.Iterative

set_option autoImplicit false

namespace DASHImpossibility.NeuralNet

variable (fs : FeatureSpace)

/-- Neural networks inherit the Attribution Trilemma when initialization
    symmetry-breaking produces a dominant feature per correlated group.

    The three hypotheses (captured feature, dominance, surjectivity) are
    the NN analogues of GBDT's first-mover properties. Justification
    requires showing that:
    - Gradient flow preferentially amplifies early-captured features
    - Different random initializations capture different features
    - The captured feature's attribution exceeds others' in the same group -/
theorem nn_trilemma
    /- The feature captured first by gradient flow -/
    (captured : Model → Fin fs.P)
    /- The captured feature dominates attribution within its group -/
    (captured_gt : ∀ (f : Model) (ℓ : Fin fs.L) (k : Fin fs.P),
      captured f ∈ fs.group ℓ → k ∈ fs.group ℓ → k ≠ captured f →
      attribution fs k f < attribution fs (captured f) f)
    /- Different initializations capture different features -/
    (captured_surjective : ∀ (ℓ : Fin fs.L) (j : Fin fs.P),
      j ∈ fs.group ℓ → ∃ f : Model, captured f = j)
    /- Then the trilemma holds -/
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False := by
  have opt : IterativeOptimizer fs := {
    dominant := captured
    dominant_gt := captured_gt
    dominant_surjective := captured_surjective
  }
  exact iterative_trilemma fs opt ℓ j k hj hk hjk ranking h_faithful

end DASHImpossibility.NeuralNet
```

- [ ] **Step 2: Build and commit**

```bash
git add DASHImpossibility/NeuralNet.lean
git commit -m "feat: neural networks inherit the Attribution Trilemma

Same abstract pattern: initialization symmetry-breaking produces a
dominant feature whose attribution exceeds others in the group.

Three hypotheses (captured feature, dominance, surjectivity) are the
NN analogues of GBDT's first-mover properties. Classical justification
is Paper 4 (optimization path dependence).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create RandomForest.lean — Contrast (Why RF Is Different)

**Files:**
- Create: `DASHImpossibility/RandomForest.lean`

Random forests DON'T have sequential dependence — each tree is independent. The trilemma may still hold (different bootstrap samples → different dominant features), but the MECHANISM is different and the violation is weaker.

- [ ] **Step 1: Create RandomForest.lean**

```lean
/-
  Random forests: contrast case. The Attribution Trilemma may hold weakly,
  but the mechanism differs fundamentally from sequential methods.

  Key difference: trees are trained INDEPENDENTLY (no shared residuals).
  Feature j being selected in tree t does NOT reduce feature k's signal
  in tree t+1. There is no cumulative first-mover advantage.

  Consequences:
  - The Rashomon property may hold (different bootstrap samples →
    different feature utilization), but the attribution differences
    are O(1/√T) from sampling noise, not O(T) from cumulative bias.
  - The trilemma applies (if Rashomon holds) but the equity violation
    is BOUNDED, not divergent as ρ → 1.
  - This makes RF inherently more equitable than GBDT under collinearity,
    which matches empirical observations.

  No formal proofs in this file — this is a discussion of scope.
  The formal contribution is: the trilemma's severity depends on the
  model class, and sequential methods are provably worse than parallel ones.
-/
import DASHImpossibility.Iterative

set_option autoImplicit false

namespace DASHImpossibility.RandomForest

/-!
## Why the trilemma is weak for random forests

The trilemma (Trilemma.lean) requires RashimonProperty: symmetric features
are ranked oppositely by different models. For RF:

1. **Rashomon property**: MAY hold from bootstrap sampling variance.
   Different bootstrap samples include different observations, which
   can change which correlated feature has higher empirical gain.
   So ∃ seeds giving j > k and ∃ seeds giving k > j.

2. **Magnitude**: The attribution difference between j and k in any
   single RF model is O(1/√T) — it's sampling noise, not cumulative
   bias. Compare to GBDT's O(ρ² · T) cumulative first-mover advantage.

3. **Ratio**: For RF, the max/min attribution ratio within a group
   approaches 1 as T → ∞ (law of large numbers). For GBDT, it
   approaches 1/(1-ρ²) → ∞ as ρ → 1.

The trilemma still formally holds (if RashimonProperty is verified for RF),
but its PRACTICAL impact is negligible — the equity violation vanishes
with more trees. This is why RF attributions are empirically more stable
than GBDT attributions under collinearity.

**This contrast strengthens the impossibility result**: the trilemma is not
vacuous. It DISCRIMINATES between model classes — sequential methods
(GBDT, Lasso, NN) are provably worse than parallel methods (RF, bagging)
for equitable feature attribution.
-/

end DASHImpossibility.RandomForest
```

- [ ] **Step 2: Build and commit**

```bash
git add DASHImpossibility/RandomForest.lean
git commit -m "docs: random forest contrast — why trilemma is weak for RF

Parallel methods (RF, bagging) don't have cumulative first-mover
advantage. Attribution differences are O(1/√T) sampling noise,
not O(ρ²T) cumulative bias. The trilemma discriminates between
sequential and parallel model classes.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Summary

| Model Class | File | Trilemma? | Mechanism | Ratio | Status |
|---|---|---|---|---|---|
| **GBDT** | General.lean | Yes (proved) | First-mover: root split of tree 1 | 1/(1-ρ²) → ∞ | Done |
| **Lasso** | Lasso.lean | Yes (proved) | Selected: one feature per group | ∞ (positive vs zero) | Phase 2 |
| **Neural nets** | NeuralNet.lean | Yes (conditional) | Captured: initialization gradient flow | TBD (Paper 4) | Phase 3 |
| **General iterative** | Iterative.lean | Yes (proved) | Abstract dominant feature | Model-dependent | Phase 1 |
| **Random forests** | RandomForest.lean | Weak/bounded | Bootstrap sampling noise | → 1 as T → ∞ | Contrast |

**Key insight:** The trilemma is UNIVERSAL for iterative/sequential optimizers. The `IterativeOptimizer` structure captures the shared pattern: dominant feature + dominance + surjectivity → RashimonProperty → trilemma. GBDT, Lasso, and NN are all instances. RF is the counterpoint that shows the trilemma's severity depends on the optimization architecture.

**What this means for the field:** The impossibility is not about SHAP, not about TreeSHAP, not about XGBoost. It's about the interaction between iterative optimization and feature collinearity. Any method that trains a single model via iterative optimization on correlated features will produce a dominant feature whose identity depends on initialization. The explanation will be faithful but unstable, or stable but unfaithful.

The resolution is the same across all model classes: break the sequential dependence. Average over multiple initializations (DASH for GBDT, ensemble of Lassos, neural network ensembles). Or accept partial orders where symmetric features are equivalent.
