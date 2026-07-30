/-
  DavisKahanSinTheta.lean — The Davis–Kahan sin-θ subspace perturbation bound
  (Frobenius / Hilbert–Schmidt norm version), proved via the eigenbasis Sylvester
  route.  Companion to `DavisKahanRecovery.lean` (which proves Weyl's inequality
  for the two extreme eigenvalues, and is imported here only for reuse — it is
  never modified).

  ─────────────────────────────────────────────────────────────────────────
  WHAT IS PROVED (fully, `sorry`-free, Lean core axioms only)
  ─────────────────────────────────────────────────────────────────────────

  Everything below is stated with the genuine Frobenius (Hilbert–Schmidt) norm
  `‖M‖_F = √(∑ i ∑ j ‖M i j‖²)` (here `frobNorm`, equal to Mathlib's
  `Matrix.frobenius_norm_def` value), over an arbitrary `RCLike` field `𝕜`
  (so both ℝ and ℂ are covered).

  1. `gapped_sylvester_frobenius` — **the mathematical core of Davis–Kahan.**
     For real "eigenvalue" data `a : m → ℝ`, `b : n → ℝ`, a matrix `X`, and its
     Sylvester residual `C i j = a i · X i j − X i j · b j`, if the spectra are
     separated by a gap `δ > 0` (`∀ i j, δ ≤ |a i − b j|`) then

              ‖X‖_F  ≤  ‖C‖_F / δ .

     This is exactly the entrywise `X_ij·(a_i − b_j) = C_ij ⇒ |X_ij| ≤ |C_ij|/δ`
     argument, summed as squares.

  2. `davisKahan_sinTheta_of_diagonalization` — **the sin-θ theorem.**
     Given Hermitian `A`, `Â` supplied in diagonalized form
        `A = U · diag α · Uᴴ`,  `Â = V · diag α̂ · Vᴴ`   (U, V unitary),
     the cross block of the eigenvector rotation `W = Uᴴ V` between
        the "low" A-eigenvectors  L = { i | α i ≤ β }  and
        the "high" Â-eigenvectors H = { j | γ ≤ α̂ j },   gap δ = γ − β > 0,
     obeys

        ‖ (Uᴴ V) restricted to L × H ‖_F  ≤  ‖Â − A‖_F / (γ − β).

     The restricted block is the matrix of inner products ⟨u_i, v_j⟩; its
     Frobenius norm is the standard sin-θ distance `‖sinΘ‖_F` between the two
     invariant subspaces.  The diagonalization hypotheses are precisely the
     output of Mathlib's `Matrix.IsHermitian.spectral_theorem`.

  3. `davisKahan_sinTheta_hermitian` — the same bound stated directly for two
     `Matrix.IsHermitian` matrices, discharging the diagonalization hypotheses
     via `spectral_theorem` and the unitarity of `eigenvectorUnitary`.

  Supporting, fully proved: `frobNormSq_conjTranspose`, `frobNormSq_unitary_mul`
  (Frobenius norm is invariant under multiplication by an isometry), the two-
  sided invariance `frobNormSq_conj_sandwich` (`‖Uᴴ E V‖_F = ‖E‖_F`), and
  `frobNormSq_submatrix_le` (dropping to a sub-block cannot increase the norm).

  ─────────────────────────────────────────────────────────────────────────
  HONEST SCOPE
  ─────────────────────────────────────────────────────────────────────────
  * This is the Frobenius-norm Davis–Kahan bound.  The operator-norm variant
    (`‖sinΘ‖₂ ≤ ‖E‖₂/δ`) is NOT proved.
  * The diagonalization inputs to (2) are hypotheses; (3) obtains them from the
    spectral theorem.  No eigenvalue-gap *localisation* (Weyl) is used here — the
    gap between the L-block of `α` and the H-block of `α̂` is a hypothesis on the
    two given spectra (this is the standard Davis–Kahan gap assumption).
  * Zero new axioms; no `sorry`.
-/

import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Real.Sqrt

set_option autoImplicit false

namespace UniversalImpossibility.DavisKahanSinTheta

open scoped BigOperators
open Matrix

variable {𝕜 : Type*} [RCLike 𝕜]
variable {m n : Type*} [Fintype m] [Fintype n]

/-! ### The Frobenius (Hilbert–Schmidt) norm

We use the elementary definition `‖M‖_F² = ∑ i ∑ j ‖M i j‖²`, which is exactly
Mathlib's `Matrix.frobenius_norm_def` value (`‖M‖ = √(∑ i ∑ j ‖M i j‖²)` under
`open scoped Matrix.Norms.Frobenius`).  Working with the explicit sum avoids the
need to install a normed-matrix instance and gives full control of the algebra. -/

/-- Squared Frobenius norm `∑ i ∑ j ‖M i j‖²`. -/
noncomputable def frobNormSq (M : Matrix m n 𝕜) : ℝ := ∑ i, ∑ j, ‖M i j‖ ^ 2

/-- Frobenius norm `√(∑ i ∑ j ‖M i j‖²)`. -/
noncomputable def frobNorm (M : Matrix m n 𝕜) : ℝ := Real.sqrt (frobNormSq M)

theorem frobNormSq_nonneg (M : Matrix m n 𝕜) : 0 ≤ frobNormSq M :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => by positivity

theorem frobNorm_nonneg (M : Matrix m n 𝕜) : 0 ≤ frobNorm M := Real.sqrt_nonneg _

/-- Column-major form of the squared Frobenius norm. -/
theorem frobNormSq_col (M : Matrix m n 𝕜) : frobNormSq M = ∑ j, ∑ i, ‖M i j‖ ^ 2 := by
  unfold frobNormSq
  rw [Finset.sum_comm]

/-! ### The gapped Sylvester bound — the core of Davis–Kahan -/

/-- **Gapped Sylvester bound (Frobenius).**  If `X` and `C` satisfy the entrywise
Sylvester relation `C i j = a i · X i j − X i j · b j` for real spectra `a, b`
separated by a gap `δ > 0`, then `‖X‖_F ≤ ‖C‖_F / δ`.

This is the mathematical heart of the Davis–Kahan sin-θ theorem: in the joint
eigenbasis the residual `C` is the relevant block of the perturbation, and the
spectral gap forces the cross-block of the rotation to be small. -/
theorem gapped_sylvester_frobenius
    (a : m → ℝ) (b : n → ℝ) (X C : Matrix m n 𝕜) (δ : ℝ) (hδ : 0 < δ)
    (hgap : ∀ i j, δ ≤ |a i - b j|)
    (hC : ∀ i j, C i j = (a i : 𝕜) * X i j - X i j * (b j : 𝕜)) :
    frobNorm X ≤ frobNorm C / δ := by
  -- entrywise `‖C i j‖ = |a i − b j| · ‖X i j‖`
  have hnorm : ∀ i j, ‖C i j‖ = |a i - b j| * ‖X i j‖ := by
    intro i j
    rw [hC i j, show (a i : 𝕜) * X i j - X i j * (b j : 𝕜) = ((a i - b j : ℝ) : 𝕜) * X i j by
      push_cast; ring, norm_mul, RCLike.norm_ofReal]
  -- entrywise squared bound `δ² · ‖X i j‖² ≤ ‖C i j‖²`
  have hsq : ∀ i j, δ ^ 2 * ‖X i j‖ ^ 2 ≤ ‖C i j‖ ^ 2 := by
    intro i j
    rw [hnorm i j, mul_pow]
    apply mul_le_mul_of_nonneg_right _ (by positivity)
    exact pow_le_pow_left₀ hδ.le (hgap i j) 2
  -- summed bound `δ² · ‖X‖_F² ≤ ‖C‖_F²`
  have hSum : δ ^ 2 * frobNormSq X ≤ frobNormSq C := by
    unfold frobNormSq
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => hsq i j
  -- take square roots
  have hstep : δ * frobNorm X ≤ frobNorm C := by
    have h1 : Real.sqrt (δ ^ 2 * frobNormSq X) ≤ Real.sqrt (frobNormSq C) :=
      Real.sqrt_le_sqrt hSum
    rwa [Real.sqrt_mul (by positivity) (frobNormSq X), Real.sqrt_sq hδ.le] at h1
  rw [le_div_iff₀ hδ, mul_comm]
  exact hstep

/-! ### Frobenius norm under conjugate transpose and isometries -/

/-- The Frobenius norm is invariant under conjugate transpose. -/
theorem frobNormSq_conjTranspose (M : Matrix m n 𝕜) : frobNormSq Mᴴ = frobNormSq M := by
  simp only [frobNormSq, Matrix.conjTranspose_apply, norm_star]
  rw [Finset.sum_comm]

/-- Casting the column sum of squares into `𝕜` gives a diagonal entry of `Mᴴ M`. -/
theorem sum_col_normSq (M : Matrix m n 𝕜) (j : n) :
    ((∑ i, ‖M i j‖ ^ 2 : ℝ) : 𝕜) = (Mᴴ * M) j j := by
  rw [Matrix.mul_apply, RCLike.ofReal_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Matrix.conjTranspose_apply, ← starRingEnd_apply, RCLike.conj_mul, RCLike.ofReal_pow]

/-- **Frobenius norm is invariant under multiplication by an isometry.**
If `Uᴴ U = 1` then `‖U X‖_F = ‖X‖_F`. -/
theorem frobNormSq_unitary_mul {p : Type*} [Fintype p] [DecidableEq m]
    {U : Matrix p m 𝕜} {X : Matrix m n 𝕜} (hU : Uᴴ * U = 1) :
    frobNormSq (U * X) = frobNormSq X := by
  have hcol : ∀ j, (∑ i, ‖(U * X) i j‖ ^ 2) = ∑ i, ‖X i j‖ ^ 2 := by
    intro j
    have hmat : (U * X)ᴴ * (U * X) = Xᴴ * X := by
      rw [Matrix.conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ U X, hU,
        Matrix.one_mul]
    have h1 : ((∑ i, ‖(U * X) i j‖ ^ 2 : ℝ) : 𝕜) = ((∑ i, ‖X i j‖ ^ 2 : ℝ) : 𝕜) := by
      rw [sum_col_normSq, sum_col_normSq, hmat]
    exact_mod_cast h1
  rw [frobNormSq_col (U * X), frobNormSq_col X]
  exact Finset.sum_congr rfl fun j _ => hcol j

/-- **Two-sided isometry invariance.**  For unitary `U`, `V`, `‖Uᴴ E V‖_F = ‖E‖_F`.
Only the co-isometry relations `U Uᴴ = 1`, `V Vᴴ = 1` are needed (the other two are
kept so the hypotheses read as "`U` and `V` are unitary"). -/
theorem frobNormSq_conj_sandwich [DecidableEq n] {U V E : Matrix n n 𝕜}
    (_hU : Uᴴ * U = 1) (hU' : U * Uᴴ = 1) (_hV : Vᴴ * V = 1) (hV' : V * Vᴴ = 1) :
    frobNormSq (Uᴴ * E * V) = frobNormSq E := by
  rw [Matrix.mul_assoc]
  rw [frobNormSq_unitary_mul (U := Uᴴ) (X := E * V)
        (by rw [Matrix.conjTranspose_conjTranspose]; exact hU')]
  rw [← frobNormSq_conjTranspose (E * V), Matrix.conjTranspose_mul]
  rw [frobNormSq_unitary_mul (U := Vᴴ) (X := Eᴴ)
        (by rw [Matrix.conjTranspose_conjTranspose]; exact hV')]
  rw [frobNormSq_conjTranspose]

/-! ### Dropping to a sub-block cannot increase the Frobenius norm -/

/-- If `f` is injective and `φ ≥ 0`, summing `φ` over the image is bounded by the
total sum. -/
theorem sum_comp_le {ι κ : Type*} [Fintype ι] [Fintype κ] (φ : κ → ℝ)
    (hφ : ∀ x, 0 ≤ φ x) (f : ι → κ) (hf : Function.Injective f) :
    ∑ i, φ (f i) ≤ ∑ x, φ x := by
  classical
  rw [← Finset.sum_image (f := φ) (g := f) fun a _ b _ h => hf h]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun x _ _ => hφ x

/-- A sub-block has Frobenius norm at most that of the full matrix. -/
theorem frobNormSq_submatrix_le {l o : Type*} [Fintype l] [Fintype o]
    (M : Matrix m n 𝕜) (f : l → m) (g : o → n)
    (hf : Function.Injective f) (hg : Function.Injective g) :
    frobNormSq (M.submatrix f g) ≤ frobNormSq M := by
  simp only [frobNormSq, Matrix.submatrix_apply]
  calc ∑ i, ∑ j, ‖M (f i) (g j)‖ ^ 2
      ≤ ∑ i, ∑ j, ‖M (f i) j‖ ^ 2 := by
        refine Finset.sum_le_sum fun i _ => ?_
        exact sum_comp_le (fun j => ‖M (f i) j‖ ^ 2) (fun j => by positivity) g hg
    _ ≤ ∑ i, ∑ j, ‖M i j‖ ^ 2 :=
        sum_comp_le (fun i => ∑ j, ‖M i j‖ ^ 2)
          (fun i => Finset.sum_nonneg fun j _ => by positivity) f hf

/-! ### The Davis–Kahan sin-θ bound (diagonalized form) -/

/-- **Davis–Kahan sin-θ theorem (Frobenius norm), diagonalized form.**

Let `A = U · diag α · Uᴴ` and `Â = V · diag α̂ · Vᴴ` be two Hermitian matrices in
diagonalized form (`U`, `V` unitary; these hypotheses are exactly the output of
`Matrix.IsHermitian.spectral_theorem`).  Let `β < γ`, and consider the "low"
A-eigenvectors `L = {i | α i ≤ β}` and the "high" Â-eigenvectors
`H = {j | γ ≤ α̂ j}`, with spectral gap `δ = γ − β`.  Then the cross block of the
eigenvector rotation `W = Uᴴ V` obeys

    ‖ (Uᴴ V)|_{L×H} ‖_F  ≤  ‖Â − A‖_F / (γ − β). -/
theorem davisKahan_sinTheta_of_diagonalization [DecidableEq n]
    (A Ahat U V : Matrix n n 𝕜) (α αhat : n → ℝ) (β γ : ℝ) (hβγ : β < γ)
    (hU : Uᴴ * U = 1) (hU' : U * Uᴴ = 1) (hV : Vᴴ * V = 1) (hV' : V * Vᴴ = 1)
    (hAdiag : A = U * Matrix.diagonal (fun i => (α i : 𝕜)) * Uᴴ)
    (hAhatdiag : Ahat = V * Matrix.diagonal (fun j => (αhat j : 𝕜)) * Vᴴ) :
    frobNorm ((Uᴴ * V).submatrix
        (Subtype.val : {i // α i ≤ β} → n) (Subtype.val : {j // γ ≤ αhat j} → n))
      ≤ frobNorm (Ahat - A) / (γ - β) := by
  set P : Matrix n n 𝕜 := Matrix.diagonal (fun i => (α i : 𝕜)) with hP
  set Q : Matrix n n 𝕜 := Matrix.diagonal (fun j => (αhat j : 𝕜)) with hQ
  set M : Matrix n n 𝕜 := Uᴴ * (Ahat - A) * V with hM
  -- The Sylvester relation in eigencoordinates.
  have hAV : Uᴴ * A * V = P * (Uᴴ * V) := by
    rw [hAdiag]
    calc Uᴴ * (U * P * Uᴴ) * V = (Uᴴ * U) * P * (Uᴴ * V) := by
          simp only [Matrix.mul_assoc]
      _ = 1 * P * (Uᴴ * V) := by rw [hU]
      _ = P * (Uᴴ * V) := by rw [Matrix.one_mul]
  have hAhatV : Uᴴ * Ahat * V = (Uᴴ * V) * Q := by
    rw [hAhatdiag]
    calc Uᴴ * (V * Q * Vᴴ) * V = (Uᴴ * V) * Q * (Vᴴ * V) := by
          simp only [Matrix.mul_assoc]
      _ = (Uᴴ * V) * Q * 1 := by rw [hV]
      _ = (Uᴴ * V) * Q := by rw [Matrix.mul_one]
  have hsyl : M = (Uᴴ * V) * Q - P * (Uᴴ * V) := by
    rw [hM, Matrix.mul_sub, Matrix.sub_mul, hAhatV, hAV]
  -- Entrywise form of the Sylvester relation.
  have hentry : ∀ k l, M k l = (Uᴴ * V) k l * (αhat l : 𝕜) - (α k : 𝕜) * (Uᴴ * V) k l := by
    intro k l
    rw [hsyl, Matrix.sub_apply, hQ, hP, Matrix.mul_diagonal, Matrix.diagonal_mul]
  -- The cross-block residual.
  set C : Matrix {i // α i ≤ β} {j // γ ≤ αhat j} 𝕜 :=
    fun i j => -(M.submatrix Subtype.val Subtype.val i j) with hCdef
  have hδpos : (0 : ℝ) < γ - β := by linarith
  -- Apply the gapped Sylvester core to the cross block.
  have core := gapped_sylvester_frobenius
      (a := fun i : {i // α i ≤ β} => α i.val)
      (b := fun j : {j // γ ≤ αhat j} => αhat j.val)
      (X := (Uᴴ * V).submatrix Subtype.val Subtype.val) (C := C) (δ := γ - β) hδpos
      (fun i j => by
        rw [le_abs]
        exact Or.inr (by have hi := i.2; have hj := j.2; simp only; linarith))
      (fun i j => by
        rw [hCdef]
        simp only [Matrix.submatrix_apply, hentry i.val j.val]
        ring)
  -- Bound the residual by ‖Â − A‖_F.
  have hCle : frobNorm C ≤ frobNorm (Ahat - A) := by
    apply Real.sqrt_le_sqrt
    calc frobNormSq C = frobNormSq (M.submatrix Subtype.val Subtype.val) := by
          unfold frobNormSq
          refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => ?_))
          simp only [hCdef, norm_neg]
      _ ≤ frobNormSq M :=
          frobNormSq_submatrix_le M _ _ Subtype.val_injective Subtype.val_injective
      _ = frobNormSq (Ahat - A) := by
          rw [hM]; exact frobNormSq_conj_sandwich hU hU' hV hV'
  refine le_trans core ?_
  gcongr

/-! ### The Davis–Kahan sin-θ bound for Hermitian matrices -/

open Matrix in
/-- **Davis–Kahan sin-θ theorem (Frobenius norm) for Hermitian matrices.**

For Hermitian `A`, `Â`, with eigenvalues `hA.eigenvalues`, `hÂ.eigenvalues` and
eigenvector unitaries `U = hA.eigenvectorUnitary`, `V = hÂ.eigenvectorUnitary`,
the cross block of the rotation `Uᴴ V` between the "low" A-eigenvectors
`{i | eigenvalue ≤ β}` and the "high" Â-eigenvectors `{j | γ ≤ eigenvalue}`
(gap `δ = γ − β > 0`) obeys `‖block‖_F ≤ ‖Â − A‖_F / (γ − β)`. -/
theorem davisKahan_sinTheta_hermitian [DecidableEq n]
    {A Ahat : Matrix n n 𝕜} (hA : A.IsHermitian) (hAhat : Ahat.IsHermitian)
    (β γ : ℝ) (hβγ : β < γ) :
    frobNorm
        (((hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * (hAhat.eigenvectorUnitary : Matrix n n 𝕜)).submatrix
          (Subtype.val : {i // hA.eigenvalues i ≤ β} → n)
          (Subtype.val : {j // γ ≤ hAhat.eigenvalues j} → n))
      ≤ frobNorm (Ahat - A) / (γ - β) := by
  set U : Matrix n n 𝕜 := (hA.eigenvectorUnitary : Matrix n n 𝕜) with hUdef
  set V : Matrix n n 𝕜 := (hAhat.eigenvectorUnitary : Matrix n n 𝕜) with hVdef
  have hUU : Uᴴ * U = 1 := by
    rw [hUdef, ← Matrix.star_eq_conjTranspose]; exact Unitary.coe_star_mul_self _
  have hUU' : U * Uᴴ = 1 := by
    rw [hUdef, ← Matrix.star_eq_conjTranspose]; exact Unitary.coe_mul_star_self _
  have hVV : Vᴴ * V = 1 := by
    rw [hVdef, ← Matrix.star_eq_conjTranspose]; exact Unitary.coe_star_mul_self _
  have hVV' : V * Vᴴ = 1 := by
    rw [hVdef, ← Matrix.star_eq_conjTranspose]; exact Unitary.coe_mul_star_self _
  have hAdiag : A = U * Matrix.diagonal (fun i => (hA.eigenvalues i : 𝕜)) * Uᴴ := by
    have h := hA.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply, ← hUdef, Matrix.star_eq_conjTranspose] at h
    exact h
  have hAhatdiag : Ahat = V * Matrix.diagonal (fun j => (hAhat.eigenvalues j : 𝕜)) * Vᴴ := by
    have h := hAhat.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply, ← hVdef, Matrix.star_eq_conjTranspose] at h
    exact h
  exact davisKahan_sinTheta_of_diagonalization A Ahat U V hA.eigenvalues hAhat.eigenvalues
    β γ hβγ hUU hUU' hVV hVV' hAdiag hAhatdiag

end UniversalImpossibility.DavisKahanSinTheta
