-- Ported (with modifications) from Yury G. Kudryashov's `SardMoreira`
-- (github.com/urkud/SardMoreira @ 14bc8a1), Apache-2.0 licensed. Module paths
-- renamed and code forward-ported to this repository's Mathlib pin. See the
-- attribution and licence notices in UniversalImpossibility/Sard/ATTRIBUTION.md.
import UniversalImpossibility.Sard.MeasureComap
import Mathlib.MeasureTheory.Measure.Haar.OfBasis

open scoped ENNReal NNReal Set.Notation Pointwise
open MeasureTheory Filter Set Function Metric Topology

noncomputable instance : MeasureSpace ℝ≥0 where
  volume := .comap (↑) (volume : Measure ℝ)

theorem NNReal.volume_def : (volume : Measure ℝ≥0) = .comap (↑) (volume : Measure ℝ) := rfl

-- TODO: should we have this instance? I'm not sure.
instance : SigmaFinite (volume : Measure ℝ≥0) := .comap _ (by fun_prop)
