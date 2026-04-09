/-
  MeasureHypotheses.lean — Measure-theoretic definitions for probabilistic claims.

  These are DEFINITIONS used as hypotheses (like IsBalanced), NOT axioms.
  They formalize the DGP symmetry assumptions needed for quantitative
  probabilistic results (Pr = 1/2, Var/M, etc.).

  Pattern: Same as IsBalanced in Defs.lean — a definition that theorems
  take as a hypothesis, so the caller decides when it applies.
-/

import UniversalImpossibility.Defs
import Mathlib.MeasureTheory.Measure.MeasureSpace

open MeasureTheory

variable {fs : FeatureSpace}

/-- The model measure is a probability measure (total mass 1). -/
def IsProbabilityModelMeasure : Prop :=
  IsProbabilityMeasure (modelMeasure)

/-- Attribution functions are measurable w.r.t. the model σ-algebra.
    Required to form measurable sets like {f | φ_j(f) > φ_k(f)}. -/
def HasMeasurableAttribution (fs : FeatureSpace) : Prop :=
  ∀ (j : Fin fs.P), Measurable (attribution fs j)

/-- DGP symmetry: within each collinear group, the model distribution
    assigns equal probability to each feature being ranked higher.
    This follows from the symmetry of the data-generating process
    when features in a group have identical marginal distributions. -/
def IsDGPSymmetric (fs : FeatureSpace) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
    modelMeasure {f : Model | attribution fs j f > attribution fs k f} =
    modelMeasure {f : Model | attribution fs k f > attribution fs j f}

/-- Non-degeneracy: the event that two within-group features have
    exactly equal attributions has measure zero. This follows from
    continuous dependence of training on the random seed. -/
def IsNonDegenerate (fs : FeatureSpace) : Prop :=
  ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
    j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
    modelMeasure {f : Model | attribution fs j f = attribution fs k f} = 0
