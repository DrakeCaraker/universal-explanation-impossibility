/-
  MorseSard.lean — the full Morse–Sard theorem, discharging `SardProperty`.

  This file closes the last residual of the ubiquity argument. §5 of
  `UbiquityDimensional.lean` reduced the nonlinear generic-ubiquity statement to a
  single named hypothesis, `SardProperty f μ` — "the set of critical values of `f`
  is `μ`-null" — which is exactly the classical Morse–Sard theorem, previously not
  formalised in Mathlib and hence left as an honestly-scoped external residual.

  Here we discharge it, using the in-tree port of Y. Kudryashov's formalisation of
  Moreira's version of Sard's theorem (`UniversalImpossibility/Sard/`, see
  `Sard/ATTRIBUTION.md`). Moreira's bound: for a `C^{k+(α)}` map `f : E → F` and
  the set `s` of points where `rank (fderiv f) ≤ p`, the image `f '' s` is null for
  the Hausdorff measure of dimension `p + (n − p)/(k + α)`, `n = dim E`.

  Specialising `p := m − 1`, `k := n − m`, `α := 1` (`m = dim F`; the smoothness
  `C^{n−m+1}` supplies `ContDiffMoreiraHolderAt (n−m) 1` at every point), the bound
  is exactly `m`, so the critical values are `μH[m]`-null, hence null for every
  additive Haar measure on `F`. This is the sharp classical threshold: `C^{n−m+1}`
  (Whitney's counterexample shows `C¹` is insufficient when `n > m`), carried
  entirely by this theorem — the downstream ubiquity chain needs only `C¹`.

  Combined with `generic_ubiquity_of_sard`, the nonlinear ubiquity statement is now
  UNCONDITIONAL for `C^{n−m+1}` observation maps (`generic_ubiquity_of_contDiff`).

  Axiom-clean (Lean core only: propext, Classical.choice, Quot.sound).
-/
import Mathlib
import UniversalImpossibility.Sard.MainTheorem
import UniversalImpossibility.UbiquityDimensional

noncomputable section

open MeasureTheory Measure Module Set Metric
open scoped ENNReal NNReal Topology unitInterval

namespace UniversalImpossibility.Ubiquity

variable {E F : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F]

/-- The non-regular values of `f` are exactly the image of its critical set (the
    points where the derivative is not surjective). -/
theorem not_isRegularValue_eq_image (f : E → F) :
    {y | ¬ IsRegularValue f y} = f '' {x | (fderiv ℝ f x).range ≠ ⊤} := by
  ext y
  simp only [mem_setOf_eq, IsRegularValue, not_forall, Classical.not_imp, mem_image]
  exact ⟨fun ⟨x, hfx, hx⟩ => ⟨x, hx, hfx⟩, fun ⟨x, hx, hfx⟩ => ⟨x, hfx, hx⟩⟩

section HausdorffNull
variable [MeasurableSpace F] [BorelSpace F]

/-- **Morse–Sard, Hausdorff form.** For a `C^{n−m+1}` map between spaces with
    `dim F = m < n = dim E` (and `m ≥ 1`), the image of the critical set is null for
    the `m`-dimensional Hausdorff measure on `F`. Instantiates the ported Moreira
    theorem at `p = m−1`, `k = n−m`, `α = 1`, where the Moreira bound
    `p + (n−p)/(k+α)` equals `m` exactly. -/
theorem hausdorffMeasure_critical_image_null {f : E → F}
    (hdim : finrank ℝ F < finrank ℝ E) (hm : 0 < finrank ℝ F)
    (hf : ContDiff ℝ (finrank ℝ E - finrank ℝ F + 1 : ℕ) f) :
    μH[(finrank ℝ F : ℝ)] (f '' {x | (fderiv ℝ f x).range ≠ ⊤}) = 0 := by
  set n := finrank ℝ E with hn
  set m := finrank ℝ F with hmdef
  have hk0 : n - m ≠ 0 := Nat.sub_ne_zero_of_lt hdim
  have hpn : m - 1 ≤ n := (Nat.sub_le m 1).trans hdim.le
  have hp_dom : m - 1 < n := lt_of_le_of_lt (Nat.sub_le m 1) hdim
  -- Moreira's bound at (k, α, p) = (n−m, 1, m−1) is exactly m.
  have hbound : sardMoreiraBound n (n - m) 1 (m - 1) = (m : ℝ≥0) := by
    apply NNReal.coe_injective
    push_cast
    have key := mul_sardMoreiraBound (n := n) (k := n - m) (p := m - 1) hk0 hpn 1
    have hα1 : ((1 : I) : ℝ) = 1 := rfl
    have h1 : ((n - m : ℕ) : ℝ) = (n : ℝ) - m := Nat.cast_sub hdim.le
    have h2 : ((m - 1 : ℕ) : ℝ) = (m : ℝ) - 1 := by rw [Nat.cast_sub hm]; norm_num
    rw [hα1, h1, h2] at key
    have hA : (0 : ℝ) < (n : ℝ) - m + 1 := by
      have : (m : ℝ) < n := by exact_mod_cast hdim
      linarith
    have key' : ((n : ℝ) - m + 1) * (sardMoreiraBound n (n - m) 1 (m - 1) : ℝ)
        = ((n : ℝ) - m + 1) * m := by linear_combination key
    exact mul_left_cancel₀ hA.ne' key'
  -- at a critical point the rank is at most m − 1
  have hs : ∀ x ∈ {x : E | (fderiv ℝ f x).range ≠ ⊤},
      finrank ℝ (fderiv ℝ f x).range ≤ m - 1 := by
    intro x hx
    have hlt : finrank ℝ (fderiv ℝ f x).range < m := Submodule.finrank_lt hx
    omega
  -- C^{n−m+1} gives the Moreira–Hölder regularity C^{(n−m)+(1)} at every point
  have hcdmh : ∀ x ∈ {x : E | (fderiv ℝ f x).range ≠ ⊤},
      ContDiffMoreiraHolderAt (n - m) 1 f x := fun x _ =>
    hf.contDiffAt.contDiffMoreiraHolderAt (by exact_mod_cast Nat.lt_succ_self (n - m)) 1
  have happ := hausdorffMeasure_sardMoreiraBound_image_null_of_finrank_le
    (p := m - 1) (k := n - m) (α := 1) hp_dom hk0 hcdmh hs
  rw [hbound] at happ
  exact_mod_cast happ

/-- A set that is null for the `dim F`-dimensional Hausdorff measure is null for
    every additive Haar measure on `F`. (Transfer through a linear equivalence with
    `ℝ^m`, where `μH[m]` is Lebesgue, then Haar uniqueness.) -/
theorem measure_null_of_hausdorffMeasure_finrank_null {A : Set F}
    (hA : μH[(finrank ℝ F : ℝ)] A = 0)
    (μ : Measure F) [μ.IsAddHaarMeasure] : μ A = 0 := by
  set m := finrank ℝ F with hmdef
  have hEq : finrank ℝ F = finrank ℝ (Fin m → ℝ) := by
    rw [Module.finrank_fin_fun]
  let e : F ≃L[ℝ] (Fin m → ℝ) :=
    (LinearEquiv.ofFinrankEq F (Fin m → ℝ) hEq).toContinuousLinearEquiv
  -- Lipschitz images of Hausdorff-null sets are null
  have h1 : μH[(m : ℝ)] (⇑e '' A) = 0 := by
    refine le_antisymm ?_ (zero_le _)
    calc μH[(m : ℝ)] (⇑e '' A)
        ≤ (‖(e : F →L[ℝ] (Fin m → ℝ))‖₊ : ℝ≥0∞) ^ (m : ℝ) * μH[(m : ℝ)] A :=
          (e : F →L[ℝ] (Fin m → ℝ)).lipschitz.hausdorffMeasure_image_le
            (d := (m : ℝ)) (by positivity) A
      _ = 0 := by rw [hA, mul_zero]
  -- μH[m] on ℝ^m is Lebesgue measure
  have h2 : volume (⇑e '' A) = 0 := by
    have hpi := hausdorffMeasure_pi_real (ι := Fin m)
    rw [Fintype.card_fin] at hpi
    rw [← hpi]
    exact h1
  -- push Lebesgue back to F: an additive Haar measure vanishing on A
  let em : (Fin m → ℝ) ≃ᵐ F := e.symm.toHomeomorph.toMeasurableEquiv
  let ν : Measure F := volume.map ⇑em
  have hν : IsAddHaarMeasure ν := e.symm.isAddHaarMeasure_map volume
  have hνA : ν A = 0 := by
    have hmap : ν A = volume (⇑em ⁻¹' A) := MeasurableEquiv.map_apply em A
    have hpre : ⇑em ⁻¹' A = ⇑e '' A := by
      ext y
      simp only [mem_preimage, mem_image]
      constructor
      · intro hy
        exact ⟨e.symm y, by simpa [em] using hy, by simp⟩
      · rintro ⟨x, hx, rfl⟩
        simpa [em] using hx
    rw [hmap, hpre, h2]
  -- Haar uniqueness: μ is a finite multiple of ν
  haveI := hν
  have hsmul : μ = addHaarScalarFactor μ ν • ν := isAddLeftInvariant_eq_smul μ ν
  rw [hsmul]
  simp [Measure.smul_apply, hνA]

/-- **The full Morse–Sard theorem, discharging `SardProperty`.** For any `C^{n−m+1}`
    map `f` from an `n`-dimensional to an `m`-dimensional real normed space with
    `n > m`, the set of critical values of `f` is null for every additive Haar
    measure. `C^{n−m+1}` is the sharp classical threshold (Whitney). This closes
    the one hypothesis that `generic_ubiquity_of_sard` left open. -/
theorem sardProperty_of_contDiff {f : E → F}
    (hdim : finrank ℝ F < finrank ℝ E)
    (hf : ContDiff ℝ (finrank ℝ E - finrank ℝ F + 1 : ℕ) f)
    (μ : Measure F) [μ.IsAddHaarMeasure] : SardProperty f μ := by
  rw [SardProperty, not_isRegularValue_eq_image]
  rcases Nat.eq_zero_or_pos (finrank ℝ F) with hm | hm
  · -- trivial codomain: every submodule is ⊤, there are no critical points
    have hcrit : {x : E | (fderiv ℝ f x).range ≠ ⊤} = ∅ := by
      ext x
      simp only [mem_setOf_eq, mem_empty_iff_false, iff_false, not_not]
      refine Submodule.eq_top_iff'.mpr fun y => ?_
      have hy : y = 0 := finrank_zero_iff_forall_zero.mp hm y
      rw [hy]
      exact Submodule.zero_mem _
    rw [hcrit]
    simp
  · exact measure_null_of_hausdorffMeasure_finrank_null
      (hausdorffMeasure_critical_image_null hdim hm hf) μ

/-- Morse–Sard for smooth (`C^∞`) maps: a convenient special case. -/
theorem sardProperty_of_smooth {f : E → F}
    (hdim : finrank ℝ F < finrank ℝ E) (hf : ContDiff ℝ (⊤ : ℕ∞) f)
    (μ : Measure F) [μ.IsAddHaarMeasure] : SardProperty f μ :=
  sardProperty_of_contDiff hdim (hf.of_le (WithTop.coe_le_coe.mpr le_top)) μ

/-- **Generic ubiquity, unconditional.** For any `C^{n−m+1}` observation map from an
    `n`-dimensional configuration space to an `m`-dimensional observable space with
    `n > m`, for almost every observable value (w.r.t. any additive Haar measure)
    every configuration producing it has genuinely distinct configurations producing
    the same value arbitrarily close: the fibre is a positive-dimensional Rashomon
    locus. The former `SardProperty` hypothesis is now a theorem. -/
theorem generic_ubiquity_of_contDiff {f : E → F}
    (hdim : finrank ℝ F < finrank ℝ E)
    (hf : ContDiff ℝ (finrank ℝ E - finrank ℝ F + 1 : ℕ) f)
    (μ : Measure F) [μ.IsAddHaarMeasure] :
    ∀ᵐ y ∂μ, ∀ x, f x = y → ∀ ε : ℝ, 0 < ε →
      ∃ x' : E, x' ≠ x ∧ f x' = f x ∧ dist x' x < ε :=
  generic_ubiquity_of_sard
    (hf.of_le (by exact_mod_cast Nat.le_add_left 1 (finrank ℝ E - finrank ℝ F))) hdim μ
    (sardProperty_of_contDiff hdim hf μ)

end HausdorffNull

end UniversalImpossibility.Ubiquity
