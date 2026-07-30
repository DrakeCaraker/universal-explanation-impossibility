/-
  CertificateGuarantee.lean — machine-verifying the per-query stability
  certificate's distribution-free flip guarantee.

  The monograph's operational deliverable is a per-query reliability certificate:
  from a Rashomon ensemble of `M` retrained models one reads, for a pairwise
  attribution claim, the empirical signal-to-noise ratio `SNR = |mean|/std` of the
  attribution *difference* `D`, and reports the claim as STABLE / MARGINAL /
  UNRELIABLE with a bound on the *flip rate* (the fraction of the ensemble on which
  the claim reverses). The headline guarantee is Cantelli's one-sided inequality:

        flip rate ≤ 1 / (1 + SNR²)                     (distribution-free)

  strictly sharper than the two-sided Chebyshev bound 1/SNR², never vacuous, and
  in particular STABLE (SNR ≥ 2) ⟹ flip ≤ 20%.

  Mathlib has Chebyshev (`meas_ge_le_variance_div_sq`) but NOT Cantelli. This file
  supplies the one-sided bound in exactly the form the certificate uses: a FINITE,
  DISTRIBUTION-FREE statement over the finite ensemble (`Finset`), with the sample
  (ddof = 0) mean and variance the implementation actually computes. Its hypotheses
  are therefore literally discharged — no integrability side-conditions, no
  distributional assumption — which is why the empirical validation found the bound
  held for 100% of 592 real pairs. Everything here depends only on Lean's core
  axioms (no `gbdtWorld`/`gbdtAxioms`, no `native_decide`, no `sorry`).

  Honest scope: this is the finite-ensemble (empirical-measure) Cantelli bound — the
  object the tool computes. The population-measure form is the measure-theoretic
  idealization and is a separate (larger) formalization; it is not claimed here.
-/
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

open Finset

namespace UniversalImpossibility.Certificate

variable {ι : Type*}

/-- Sample mean of an ensemble readout `d : ι → ℝ` over the finite index set `s`. -/
noncomputable def mean (s : Finset ι) (d : ι → ℝ) : ℝ := (∑ i ∈ s, d i) / s.card

/-- Population sample variance (ddof = 0), matching the certificate code's `std()²`. -/
noncomputable def variance (s : Finset ι) (d : ι → ℝ) : ℝ :=
  (∑ i ∈ s, (d i - mean s d) ^ 2) / s.card

/-- Certificate flip frequency for a pairwise claim whose ensemble-mean difference is
    positive: the fraction of ensemble draws on which the difference `d i` is `≤ 0`
    (a reversal). Ties (`d i = 0`) are counted as flips, so this upper-bounds the
    strict-reversal frequency the implementation reports. -/
noncomputable def flipFreq (s : Finset ι) (d : ι → ℝ) : ℝ :=
  ((s.filter (fun i => d i ≤ 0)).card : ℝ) / s.card

/-- Population variance is nonnegative. -/
theorem variance_nonneg (s : Finset ι) (d : ι → ℝ) : 0 ≤ variance s d := by
  unfold variance
  apply div_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _)
  exact Nat.cast_nonneg _

/-- **Finite-sample Cantelli (one-sided) inequality — the certificate's core guarantee.**
    For any finite ensemble `s` of readouts `d`, with sample mean `m` and sample
    variance `variance s d`, the fraction of draws in the lower tail
    `{ d i - m ≤ -λ }` (λ > 0) is at most `variance / (variance + λ²)`. No
    distributional assumption. -/
theorem cantelli_lower_tail (s : Finset ι) (hs : s.Nonempty) (d : ι → ℝ)
    {lam : ℝ} (hlam : 0 < lam) :
    ((s.filter (fun i => d i - mean s d ≤ -lam)).card : ℝ) / (s.card : ℝ)
      ≤ variance s d / (variance s d + lam ^ 2) := by
  classical
  have hc : (0 : ℝ) < (s.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hs
  have hc0 : (s.card : ℝ) ≠ 0 := ne_of_gt hc
  have hlne : lam ≠ 0 := ne_of_gt hlam
  set m : ℝ := mean s d with hm
  set c : ℝ := (s.card : ℝ) with hcdef
  set S2 : ℝ := ∑ i ∈ s, (d i - m) ^ 2 with hS2def
  have hS2 : 0 ≤ S2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  -- the centered readouts sum to zero
  have hsum0 : ∑ i ∈ s, (d i - m) = 0 := by
    have hcm : c * m = ∑ i ∈ s, d i := by
      rw [hm]; simp only [mean]; rw [← hcdef]; field_simp
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, ← hcdef, hcm, sub_self]
  -- Markov step: for every u ≥ 0, (#lower tail)·(λ+u)² ≤ S2 + c·u²
  have key : ∀ u : ℝ, 0 ≤ u →
      ((s.filter (fun i => d i - m ≤ -lam)).card : ℝ) * (lam + u) ^ 2 ≤ S2 + c * u ^ 2 := by
    intro u hu
    have hpt : ∀ i ∈ s.filter (fun i => d i - m ≤ -lam),
        (lam + u) ^ 2 ≤ (d i - m - u) ^ 2 := by
      intro i hi
      have hP : d i - m ≤ -lam := (Finset.mem_filter.mp hi).2
      nlinarith [mul_nonneg (by linarith : (0:ℝ) ≤ -(d i - m + lam))
        (by linarith : (0:ℝ) ≤ -(d i - m - lam - 2*u)), hu, hlam]
    calc ((s.filter (fun i => d i - m ≤ -lam)).card : ℝ) * (lam + u) ^ 2
        = ∑ _i ∈ s.filter (fun i => d i - m ≤ -lam), (lam + u) ^ 2 := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ i ∈ s.filter (fun i => d i - m ≤ -lam), (d i - m - u) ^ 2 :=
          Finset.sum_le_sum hpt
      _ ≤ ∑ i ∈ s, (d i - m - u) ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            (fun i _ _ => sq_nonneg _)
      _ = S2 + c * u ^ 2 := by
          have expand : ∀ i, (d i - m - u) ^ 2
              = (d i - m) ^ 2 - 2 * u * (d i - m) + u ^ 2 := fun i => by ring
          rw [Finset.sum_congr rfl (fun i _ => expand i), Finset.sum_add_distrib,
            Finset.sum_sub_distrib, ← Finset.mul_sum, hsum0, mul_zero, sub_zero,
            Finset.sum_const, nsmul_eq_mul, ← hS2def, ← hcdef]
  -- instantiate the optimal u = S2/(c·λ) and clear denominators
  have hclam : (0 : ℝ) < c * lam := mul_pos hc hlam
  have hclamne : c * lam ≠ 0 := ne_of_gt hclam
  have hopt := key (S2 / (c * lam)) (div_nonneg hS2 hclam.le)
  have hA : (0 : ℝ) < c * lam ^ 2 + S2 :=
    add_pos_of_pos_of_nonneg (mul_pos hc (pow_pos hlam 2)) hS2
  have hAne : c * lam ^ 2 + S2 ≠ 0 := ne_of_gt hA
  have hid1 : (lam + S2 / (c * lam)) ^ 2 * (c * lam) ^ 2 = (c * lam ^ 2 + S2) ^ 2 := by
    field_simp
  have hid2 : (S2 + c * (S2 / (c * lam)) ^ 2) * (c * lam) ^ 2
      = c * S2 * (c * lam ^ 2 + S2) := by
    field_simp
  have hmul2 : ((s.filter (fun i => d i - m ≤ -lam)).card : ℝ) * (c * lam ^ 2 + S2) ^ 2
      ≤ c * S2 * (c * lam ^ 2 + S2) := by
    have h := mul_le_mul_of_nonneg_right hopt (sq_nonneg (c * lam))
    rwa [mul_assoc, hid1, hid2] at h
  have hcore : ((s.filter (fun i => d i - m ≤ -lam)).card : ℝ) * (c * lam ^ 2 + S2)
      ≤ c * S2 := by
    have h := hmul2
    rw [sq, ← mul_assoc] at h
    exact le_of_mul_le_mul_right h hA
  -- convert the division-free core into the stated ratio bound
  have hvar : variance s d = S2 / c := by simp only [variance, hS2def, hcdef, hm]
  have hB : (0 : ℝ) < S2 / c + lam ^ 2 :=
    add_pos_of_nonneg_of_pos (div_nonneg hS2 hc.le) (pow_pos hlam 2)
  have hBne : S2 / c + lam ^ 2 ≠ 0 := ne_of_gt hB
  rw [hvar,
    show (S2 / c) / ((S2 / c) + lam ^ 2) = S2 / (c * lam ^ 2 + S2) by field_simp; ring,
    div_le_div_iff₀ hc hA]
  nlinarith [hcore]

/-- The lower-tail bound, specialized to the certificate flip event `{ d i ≤ 0 }`
    when the ensemble mean is positive: `flip ≤ variance / (variance + mean²)`. -/
theorem flip_le_variance (s : Finset ι) (hs : s.Nonempty) (d : ι → ℝ)
    (hmean : 0 < mean s d) :
    flipFreq s d ≤ variance s d / (variance s d + (mean s d) ^ 2) := by
  classical
  have hev : s.filter (fun i => d i ≤ 0)
      = s.filter (fun i => d i - mean s d ≤ -mean s d) := by
    ext i; simp only [Finset.mem_filter]
    constructor <;> rintro ⟨his, h⟩ <;> exact ⟨his, by linarith⟩
  have hcan := cantelli_lower_tail s hs d (lam := mean s d) hmean
  unfold flipFreq
  rw [hev]
  exact hcan

/-- **Certificate flip guarantee, `1/(1+SNR²)` form.** With `SNR² = mean²/variance`,
    a claim whose ensemble mean is positive flips on at most a `1/(1+SNR²)` fraction
    of the ensemble — Cantelli, distribution-free. -/
theorem flip_le_one_div_one_add_snrSq (s : Finset ι) (hs : s.Nonempty) (d : ι → ℝ)
    (hmean : 0 < mean s d) (hvar : 0 < variance s d) :
    flipFreq s d ≤ 1 / (1 + (mean s d) ^ 2 / variance s d) := by
  have h := flip_le_variance s hs d hmean
  have hvne : variance s d ≠ 0 := ne_of_gt hvar
  have hEq : variance s d / (variance s d + (mean s d) ^ 2)
      = 1 / (1 + (mean s d) ^ 2 / variance s d) := by
    field_simp
  rwa [hEq] at h

/-- **STABLE ⟹ flip ≤ 20%.** If the certificate reports STABLE, i.e. `SNR ≥ 2`
    (equivalently `4·variance ≤ mean²`), the flip rate is at most `1/5`,
    unconditionally. -/
theorem stable_flip_le_one_fifth (s : Finset ι) (hs : s.Nonempty) (d : ι → ℝ)
    (hmean : 0 < mean s d) (hstable : 4 * variance s d ≤ (mean s d) ^ 2) :
    flipFreq s d ≤ 1 / 5 := by
  have h := flip_le_variance s hs d hmean
  have hden : 0 < variance s d + (mean s d) ^ 2 :=
    add_pos_of_nonneg_of_pos (variance_nonneg s d) (pow_pos hmean 2)
  have hstep : variance s d / (variance s d + (mean s d) ^ 2) ≤ 1 / 5 := by
    rw [div_le_div_iff₀ hden (by norm_num : (0:ℝ) < 5)]
    nlinarith [hstable]
  exact h.trans hstep

end UniversalImpossibility.Certificate
