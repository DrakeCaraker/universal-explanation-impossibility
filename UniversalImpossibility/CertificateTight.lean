/-
  CertificateTight.lean — the certificate's Cantelli flip bound is OPTIMAL.

  `CertificateGuarantee.flip_le_variance` proves the distribution-free bound
  `flip ≤ variance / (variance + mean²)`. This file proves it cannot be improved:
  an explicit two-point ensemble ATTAINS the bound with equality. Hence no
  certificate that reads only the ensemble mean and variance can promise a smaller
  flip rate — the certificate is the best possible instrument in its information
  class. Specialized at SNR = 2 this shows `stable_flip_le_one_fifth` is tight:
  the 20% STABLE promise is exactly met, not conservative, in the worst case.

  Construction: on a finite ensemble `s` with a nonempty proper "tail" subset
  `A ⊊ s`, read `0` on `A` and `card s / (card s − card A)` on the rest. This has
  mean `1`, flip frequency `card A / card s`, and variance `card A/(card s−card A)`,
  so the Cantelli bound `variance/(variance+mean²)` equals `card A / card s` — the
  flip frequency exactly. This is the classical Cantelli extremal (two-point)
  distribution, realized as a finite ensemble for every rational tail fraction.

  Axiom-clean: pure `Finset`/real algebra (no `gbdtWorld`/`gbdtAxioms`, no
  `native_decide`, no `sorry`).
-/
import UniversalImpossibility.CertificateGuarantee
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases

open Finset

namespace UniversalImpossibility.Certificate

variable {ι : Type*}

/-- **Cantelli tightness / certificate optimality.** For any finite ensemble `s`
with a nonempty proper tail subset `A ⊊ s`, the two-point readout `d` (`0` on `A`,
`card s/(card s − card A)` elsewhere) has positive mean and attains the Cantelli
flip bound with equality: `flipFreq = variance / (variance + mean²)`. Therefore
the bound of `flip_le_variance` is the best possible for a certificate using only
the ensemble mean and variance. -/
theorem flip_bound_tight (s : Finset ι) (A : Finset ι)
    (hAs : A ⊆ s) (hA : A.Nonempty) (hne : A ≠ s) :
    ∃ d : ι → ℝ, 0 < mean s d ∧
      flipFreq s d = variance s d / (variance s d + (mean s d) ^ 2) := by
  classical
  have hssub : A ⊂ s := hAs.ssubset_of_ne hne
  have hcardlt : A.card < s.card := Finset.card_lt_card hssub
  set n : ℝ := (s.card : ℝ) with hn
  set a : ℝ := (A.card : ℝ) with ha
  have han : a < n := by rw [ha, hn]; exact_mod_cast hcardlt
  have hnpos : 0 < n := lt_of_le_of_lt (by positivity) han
  have hnne : n ≠ 0 := ne_of_gt hnpos
  have hdiff : 0 < n - a := by linarith
  have hdiffne : n - a ≠ 0 := ne_of_gt hdiff
  set b : ℝ := n / (n - a) with hb
  have hbpos : 0 < b := div_pos hnpos hdiff
  set d : ι → ℝ := fun i => if i ∈ A then (0 : ℝ) else b with hd
  -- Sum of d over s equals n (the card), so the mean is 1.
  have hsumA : ∑ i ∈ A, d i = 0 := by
    apply Finset.sum_eq_zero; intro i hi; simp only [hd, if_pos hi]
  have hcardSdiff : ((s \ A).card : ℝ) = n - a := by
    rw [Finset.card_sdiff_of_subset hAs, Nat.cast_sub (le_of_lt hcardlt), ← hn, ← ha]
  have hsumSdiff : ∑ i ∈ s \ A, d i = (n - a) * b := by
    have hconst : ∀ i ∈ s \ A, d i = b := by
      intro i hi; simp only [hd, if_neg (Finset.mem_sdiff.mp hi).2]
    rw [Finset.sum_congr rfl hconst, Finset.sum_const, nsmul_eq_mul, hcardSdiff]
  have hsum : ∑ i ∈ s, d i = n := by
    rw [← Finset.sum_sdiff hAs, hsumA, hsumSdiff, add_zero, hb]
    field_simp
  have hmean : mean s d = 1 := by
    rw [mean, hsum, ← hn]; field_simp
  -- Flip frequency: exactly the tail fraction a/n.
  have hfilter : s.filter (fun i => d i ≤ 0) = A := by
    ext i; simp only [Finset.mem_filter]
    constructor
    · rintro ⟨his, hle⟩
      by_contra hiA
      simp only [hd, if_neg hiA] at hle; linarith
    · intro hiA; exact ⟨hAs hiA, le_of_eq (by simp only [hd, if_pos hiA])⟩
  have hflip : flipFreq s d = a / n := by
    rw [flipFreq, hfilter, ← ha, ← hn]
  -- Variance: expand ∑ (d i − 1)² split over A and s \ A.
  have hsqA : ∑ i ∈ A, (d i - mean s d) ^ 2 = a := by
    rw [hmean]
    have hpt : ∀ i ∈ A, (d i - 1) ^ 2 = 1 := by
      intro i hi; rw [show d i = 0 from by simp only [hd, if_pos hi]]; ring
    rw [Finset.sum_congr rfl hpt, Finset.sum_const, nsmul_eq_mul, ← ha, mul_one]
  have hsqSdiff : ∑ i ∈ s \ A, (d i - mean s d) ^ 2 = (n - a) * (b - 1) ^ 2 := by
    rw [hmean]
    have hpt : ∀ i ∈ s \ A, (d i - 1) ^ 2 = (b - 1) ^ 2 := by
      intro i hi
      rw [show d i = b from by simp only [hd, if_neg (Finset.mem_sdiff.mp hi).2]]
    rw [Finset.sum_congr rfl hpt, Finset.sum_const, nsmul_eq_mul, hcardSdiff]
  have hbm1 : b - 1 = a / (n - a) := by
    rw [hb]; first | (field_simp; ring) | field_simp
  have hSsq : ∑ i ∈ s, (d i - mean s d) ^ 2 = a * n / (n - a) := by
    rw [← Finset.sum_sdiff hAs, hsqSdiff, hsqA, hbm1]
    first | (field_simp; ring) | field_simp
  have hvar : variance s d = a / (n - a) := by
    rw [variance, hSsq, ← hn]; first | (field_simp; ring) | field_simp
  -- Assemble equality.
  refine ⟨d, by rw [hmean]; norm_num, ?_⟩
  rw [hflip, hvar, hmean]
  rw [show a / (n - a) + (1 : ℝ) ^ 2 = n / (n - a) from by
    first | (field_simp; ring) | field_simp]
  rw [div_div_eq_mul_div]
  first | (field_simp; ring) | field_simp

/-- **The certificate is one-sided: instability is NOT certifiable from (mean, variance).**
    The flip bound `flip ≤ variance/(variance+mean²)` certifies *stability* (a small
    flip rate). Its converse fails completely: there is no positive lower bound on the
    flip rate as a function of the ensemble mean and variance. Concretely, this
    all-positive ensemble `[101,1,1,1,1]` reads `SNR² < 1` — the certificate's
    UNRELIABLE band, which one might hope means "the claim is unstable" — yet its flip
    rate is exactly `0`. So a low SNR does not, and provably cannot, certify that a
    claim actually flips: the certificate can flag stability but never instability.
    (This is why the deployed certificate reports STABLE / not-STABLE, never
    "certified unstable"; the honest scope is one-sided.) -/
theorem flip_zero_at_low_snr :
    ∃ d : Fin 5 → ℝ,
      0 < mean Finset.univ d ∧ 0 < variance Finset.univ d ∧
      (mean Finset.univ d) ^ 2 / variance Finset.univ d < 1 ∧
      flipFreq Finset.univ d = 0 := by
  refine ⟨fun i => if i = 0 then (101 : ℝ) else 1, ?_, ?_, ?_, ?_⟩
  · -- mean = 21 > 0
    simp only [mean, Fin.sum_univ_five, Finset.card_univ, Fintype.card_fin, Fin.ext_iff]
    norm_num
  · -- variance = 1600 > 0
    simp only [variance, mean, Fin.sum_univ_five, Finset.card_univ, Fintype.card_fin, Fin.ext_iff]
    norm_num
  · -- SNR² = 441/1600 < 1
    simp only [mean, variance, Fin.sum_univ_five, Finset.card_univ, Fintype.card_fin, Fin.ext_iff]
    norm_num
  · -- flip = 0: no ensemble member is ≤ 0
    have hempty : Finset.univ.filter
        (fun i => (fun i : Fin 5 => if i = 0 then (101 : ℝ) else 1) i ≤ 0) = ∅ := by
      rw [Finset.filter_eq_empty_iff]
      intro i _
      fin_cases i <;> norm_num
    simp only [flipFreq, hempty, Finset.card_empty, Nat.cast_zero, zero_div]

end UniversalImpossibility.Certificate
