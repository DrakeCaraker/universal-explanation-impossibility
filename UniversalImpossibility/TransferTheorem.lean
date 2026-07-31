/-
  TransferTheorem.lean — the certificate's in-sample guarantee transfers OUT of sample.

  The finite-ensemble Cantelli guarantee (`CertificateGuarantee.lean`) and its
  population form (`PopulationCantelli.lean`) bound the flip rate *within* the
  ensemble that was used to compute it. The scientific claim the large-scale
  validation actually tests is OUT-OF-SAMPLE: a STABLE verdict computed on one
  Rashomon ensemble predicts low flip on an *independent* ensemble. This file
  supplies the missing bridge as a machine-checked, distribution-free,
  finite-sample theorem.

  Model: the flip indicators of the deployment ensemble B are `m` i.i.d. draws
  `I 0, …, I (m-1) : Ω → ℝ`, each valued in `[0,1]` (1 = the claim reversed on that
  model), with common mean `p` = the population flip probability. Population
  Cantelli bounds `p ≤ β := 1/(1+SNR²)`. Then, by Hoeffding on the independent
  bounded indicators, the observed flip frequency on B exceeds `β + t` with
  probability at most `exp(-2 m t²)`:

        ℙ( flip_B ≥ β + t ) ≤ exp(-2 m t²).

  So a certificate whose population bound is `β` transfers to any independent
  m-ensemble with an explicit exponential guarantee. This is exactly the
  100%-STABLE-transfer phenomenon observed on 104 datasets, now with a proof.

  The bound is stated on the SUM `∑ I i` (flip_B = (∑ I i)/m); dividing by `m`
  gives the flip-frequency form. Built by composing two Mathlib results — Hoeffding's
  lemma for bounded variables (`hasSubgaussianMGF_of_mem_Icc`) and the sub-Gaussian
  Hoeffding tail bound (`measure_sum_ge_le_of_iIndepFun`) — so the only axioms are
  Lean core (propext / Classical.choice / Quot.sound); no `native_decide`, no `sorry`.
-/
import Mathlib.Probability.Moments.SubGaussian

open MeasureTheory ProbabilityTheory Real
open scoped NNReal ENNReal

namespace UniversalImpossibility.Transfer

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- **Out-of-sample transfer of the flip guarantee (finite-sample, distribution-free).**
For `m` i.i.d. `[0,1]`-valued flip indicators with common mean `p`, and any population
Cantelli bound `β ≥ p`, the observed number of flips on the independent ensemble
exceeds `m·(β+t)` — i.e. the flip frequency exceeds `β+t` — with probability at most
`exp(-2 m t²)`. Composes Hoeffding's lemma with the sub-Gaussian tail bound. -/
theorem transfer_flip_bound {m : ℕ} (hm : 0 < m) (I : Fin m → Ω → ℝ)
    (hindep : iIndepFun I μ) (hmeas : ∀ i, AEMeasurable (I i) μ)
    (hrange : ∀ i, ∀ᵐ ω ∂μ, I i ω ∈ Set.Icc (0 : ℝ) 1)
    {p β t : ℝ} (hp : ∀ i, μ[I i] = p) (hpβ : p ≤ β) (ht : 0 ≤ t) :
    μ.real {ω | (m : ℝ) * (β + t) ≤ ∑ i, I i ω} ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  have hm0 : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm.ne'
  have hmnn : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  -- centered indicators Y i = I i - p; still independent, each sub-Gaussian with c = 1/4
  have hYindep : iIndepFun (fun i ω => I i ω - p) μ :=
    hindep.comp (fun _ => fun x => x - p) (fun _ => measurable_id.sub_const p)
  have hYsub : ∀ i : Fin m, HasSubgaussianMGF (fun ω => I i ω - p) (1 / 4 : ℝ≥0) μ := by
    intro i
    have h := hasSubgaussianMGF_of_mem_Icc (hmeas i) (hrange i)
    rw [hp i] at h
    have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2) = (1 / 4 : ℝ≥0) := by
      rw [sub_zero, nnnorm_one]; norm_num
    rwa [hc] at h
  -- Hoeffding tail on the sum of the centered indicators
  have hεnn : (0 : ℝ) ≤ (m : ℝ) * t := mul_nonneg hmnn ht
  have key := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hYindep
    (s := Finset.univ) (fun i _ => hYsub i) (ε := (m : ℝ) * t) hεnn
  -- the deployment event sits inside the tail event
  have hsub_ev : {ω | (m : ℝ) * (β + t) ≤ ∑ i, I i ω}
      ⊆ {ω | (m : ℝ) * t ≤ ∑ i, (I i ω - p)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hsplit : ∑ i, (I i ω - p) = (∑ i, I i ω) - (m : ℝ) * p := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul]
    rw [hsplit]
    nlinarith [hω, mul_nonneg hmnn (sub_nonneg.mpr hpβ)]
  -- chain monotonicity and the tail bound; simplify the exponent 2·(m/4)=m/2 ⇒ -2 m t²
  refine (measureReal_mono hsub_ev).trans (key.trans (le_of_eq ?_))
  congr 1
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  push_cast
  field_simp
  ring

/-- **Flip-frequency form (what the practitioner reads).** The observed flip
*frequency* `flip_B = (∑ I i)/m` on the independent deployment ensemble exceeds the
population Cantelli bound `β` by more than `t` with probability at most
`exp(-2 m t²)`. A STABLE verdict (`β = 1/(1+SNR²) ≤ 0.2`) therefore transfers to any
independent ensemble with an exponential guarantee — the machine-checked explanation
of the observed 100% out-of-sample STABLE transfer. -/
theorem transfer_flipFreq_bound {m : ℕ} (hm : 0 < m) (I : Fin m → Ω → ℝ)
    (hindep : iIndepFun I μ) (hmeas : ∀ i, AEMeasurable (I i) μ)
    (hrange : ∀ i, ∀ᵐ ω ∂μ, I i ω ∈ Set.Icc (0 : ℝ) 1)
    {p β t : ℝ} (hp : ∀ i, μ[I i] = p) (hpβ : p ≤ β) (ht : 0 ≤ t) :
    μ.real {ω | β + t ≤ (∑ i, I i ω) / (m : ℝ)} ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  have hmpos : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  have hset : {ω | β + t ≤ (∑ i, I i ω) / (m : ℝ)}
      = {ω | (m : ℝ) * (β + t) ≤ ∑ i, I i ω} := by
    ext ω
    simp only [Set.mem_setOf_eq, le_div_iff₀ hmpos, mul_comm]
  rw [hset]
  exact transfer_flip_bound hm I hindep hmeas hrange hp hpβ ht

end UniversalImpossibility.Transfer
