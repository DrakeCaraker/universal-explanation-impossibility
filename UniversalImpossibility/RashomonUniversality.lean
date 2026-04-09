/-
  Rashomon from Symmetry: the Rashomon property is inevitable
  for symmetric model classes.

  S3: Permutation closure — a model class is permutation-closed if
      permuting features within a group produces another model in the class.
  S4: Rashomon from symmetry — if a symmetric swap operation exists,
      the Rashomon property holds for any pair where a distinguishing
      model exists.

  Supplement: §Extended Results: Universality (lines 354-432)
-/
import UniversalImpossibility.Trilemma

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpace)

/-! ### S3: Permutation Closure -/

/-- A feature swap operation on models: given features j, k,
    `swap j k f` produces the "permuted model" where j and k are swapped.
    This abstracts over the concrete permutation operation. -/
def FeatureSwap := Fin fs.P → Fin fs.P → Model → Model

/-- S3: Permutation closure — the swap operation preserves the model class
    and swaps attributions. Since `Model` is the universe of models, we do
    not need an explicit class predicate; instead we axiomatize the effect
    of the swap on attributions.

    For a symmetric DGP, permuting features j ↔ k within a group and
    retraining yields a model whose attributions for j and k are swapped.
    This is the content of the permutation-closure property. -/
structure IsSymmetricSwap (swap : FeatureSwap fs) : Prop where
  /-- Swapping maps attribution of j to attribution of k from the original -/
  attribution_swap : ∀ (j k : Fin fs.P) (f : Model),
    attribution fs j (swap j k f) = attribution fs k f
  /-- Swapping maps attribution of k to attribution of j from the original -/
  attribution_swap_sym : ∀ (j k : Fin fs.P) (f : Model),
    attribution fs k (swap j k f) = attribution fs j f

/-! ### S4: Rashomon from Symmetry -/

/-- Per-pair Rashomon witnesses from a symmetric swap.
    If φ_j(f) > φ_k(f) for some model f, then the swapped model
    swap j k f satisfies φ_k(f') > φ_j(f'). -/
theorem rashomon_witnesses_from_swap
    (swap : FeatureSwap fs)
    (hsym : IsSymmetricSwap fs swap)
    (j k : Fin fs.P)
    (f : Model)
    (hdiff : attribution fs j f > attribution fs k f) :
    ∃ f' : Model,
      attribution fs k f' > attribution fs j f' :=
  ⟨swap j k f, by rw [hsym.attribution_swap_sym, hsym.attribution_swap]; exact hdiff⟩

/-- Both Rashomon witnesses from a single distinguishing model.
    If φ_j(f) ≠ φ_k(f), then we can produce two models witnessing
    opposite orderings. -/
theorem rashomon_pair_from_swap
    (swap : FeatureSwap fs)
    (hsym : IsSymmetricSwap fs swap)
    (j k : Fin fs.P)
    (f : Model)
    (hdiff : attribution fs j f ≠ attribution fs k f) :
    ∃ f₁ f₂ : Model,
      attribution fs j f₁ > attribution fs k f₁ ∧
      attribution fs k f₂ > attribution fs j f₂ := by
  rcases lt_or_gt_of_ne hdiff with h | h
  · -- φ_j(f) < φ_k(f): f witnesses k > j, swap witnesses j > k
    exact ⟨swap j k f, f,
      by rw [hsym.attribution_swap, hsym.attribution_swap_sym]; exact h,
      h⟩
  · -- φ_j(f) > φ_k(f): f witnesses j > k, swap witnesses k > j
    exact ⟨f, swap j k f,
      h,
      by rw [hsym.attribution_swap_sym, hsym.attribution_swap]; exact h⟩

/-- **S4: Rashomon from symmetry (full).**
    If a symmetric swap exists and every within-group pair has a model
    that distinguishes them, then the full Rashomon property holds.

    The non-degeneracy hypothesis is mild: it only asks that for each
    pair (j, k) of symmetric features, SOME model assigns them different
    attributions. This holds for any non-trivial model class. -/
theorem rashomon_from_symmetry
    (swap : FeatureSwap fs)
    (hsym : IsSymmetricSwap fs swap)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, attribution fs j f ≠ attribution fs k f) :
    RashimonProperty fs := by
  intro ℓ j k hj hk hjk
  obtain ⟨f, hdiff⟩ := hnd ℓ j k hj hk hjk
  exact rashomon_pair_from_swap fs swap hsym j k f hdiff

/-- Corollary: combining Rashomon-from-symmetry with the attribution
    impossibility theorem. If a symmetric swap exists and features are
    non-degenerate, then no faithful stable complete ranking exists. -/
theorem impossibility_from_symmetry
    (swap : FeatureSwap fs)
    (hsym : IsSymmetricSwap fs swap)
    (hnd : ∀ (ℓ : Fin fs.L) (j k : Fin fs.P),
      j ∈ fs.group ℓ → k ∈ fs.group ℓ → j ≠ k →
      ∃ f : Model, attribution fs j f ≠ attribution fs k f)
    (ℓ : Fin fs.L) (j k : Fin fs.P)
    (hj : j ∈ fs.group ℓ) (hk : k ∈ fs.group ℓ) (hjk : j ≠ k)
    (ranking : Fin fs.P → Fin fs.P → Prop)
    (h_faithful : ∀ f : Model,
      ranking j k ↔ attribution fs j f > attribution fs k f) :
    False :=
  attribution_impossibility fs
    (rashomon_from_symmetry fs swap hsym hnd)
    ℓ j k hj hk hjk ranking h_faithful

end UniversalImpossibility
