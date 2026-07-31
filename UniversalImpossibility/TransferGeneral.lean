/-
  TransferGeneral.lean — the certificate transfers out-of-sample for ANY ensemble,
  with no independence assumption.

  `TransferTheorem.lean` proves out-of-sample transfer for an i.i.d. deployment
  ensemble (Hoeffding), giving an exponential rate. Real Rashomon ensembles are
  bootstrap-correlated, not i.i.d. This file closes that seam: population Cantelli,
  applied to the deployment flip frequency directly, needs NO independence or
  exchangeability at all.

      Pr[ flip_B ≥ β + t ] ≤ Var(flip_B) / (Var(flip_B) + t²).

  So correlated / exchangeable / bootstrap ensembles all transfer, the bound
  degrading gracefully as the flip-frequency variance grows with correlation
  (for an independent ensemble Var(flip_B) ≤ 1/(4m), recovering concentration; the
  Hoeffding version gives the sharper exponential rate in that special case). The
  epistemic mode is the certificate's own: a distribution-free BOUND, here freed of
  the last structural assumption.

  Axiom-clean (Lean core + Mathlib; composes the machine-checked population Cantelli).
-/
import UniversalImpossibility.PopulationCantelli

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace UniversalImpossibility.Transfer

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- **General (dependence-free) transfer bound.** For any deployment ensemble whose
flip frequency `F ∈ L²(μ)` has mean `p ≤ β` (the population Cantelli bound), the flip
frequency exceeds `β + t` with probability at most `Var(F)/(Var(F)+t²)` — with no
independence or exchangeability assumption. This is population Cantelli applied to `F`,
so it covers correlated and bootstrap ensembles, closing the i.i.d. seam in the
Hoeffding-based `transfer_flipFreq_bound`. -/
theorem transfer_flip_general (F : Ω → ℝ) (hF : MemLp F 2 μ)
    {p β t : ℝ} (hmean : μ[F] = p) (hpβ : p ≤ β) (ht : 0 < t) :
    μ.real {ω | β + t ≤ F ω} ≤ variance F μ / (variance F μ + t ^ 2) := by
  -- the deployment over-shoot event sits inside the centered Cantelli tail
  have hsub : {ω | β + t ≤ F ω} ⊆ {ω | t ≤ F ω - μ[F]} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [hmean]; linarith
  have hcan := PopulationCantelli.cantelli_upper hF ht
  have hv : 0 ≤ variance F μ / (variance F μ + t ^ 2) := by
    have := variance_nonneg F μ; positivity
  have hconv : μ.real {ω | t ≤ F ω - μ[F]} ≤ variance F μ / (variance F μ + t ^ 2) := by
    rw [measureReal_def]
    calc (μ {ω | t ≤ F ω - μ[F]}).toReal
        ≤ (ENNReal.ofReal (variance F μ / (variance F μ + t ^ 2))).toReal :=
          ENNReal.toReal_mono ENNReal.ofReal_ne_top hcan
      _ = variance F μ / (variance F μ + t ^ 2) := ENNReal.toReal_ofReal hv
  exact (measureReal_mono hsub).trans hconv

end UniversalImpossibility.Transfer
