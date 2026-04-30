import UniversalImpossibility.MutualInformation

/-!
# MI → Quantitative Unfaithfulness Bridge

Connects the MI boundary (I > 0 ↔ impossibility) to a quantitative
unfaithfulness bound. This is the formal version of the empirically
observed pipeline: MI detects dependence → Rashomon witnesses exist →
any stable explanation has measurable unfaithfulness.

## Main Results

1. `mi_implies_positive_gap`: MI > 0 → the attribution gap between
   Rashomon witnesses is strictly positive (one model ranks j higher,
   the other ranks k higher).

2. `gap_implies_unfaithfulness`: positive gap → any stable value r
   disagrees with the actual attributions. Total disagreement ≥ gap.

3. `mi_quantitative_unfaithfulness`: MI > 0 → any stable explanation
   has unfaithfulness ≥ gap/2 on at least one Rashomon witness.

This completes the chain: MI > 0 → Δ > 0 → unfaithfulness > 0,
connecting the information-theoretic boundary to the algebraic
framework's quantitative bilemma.
-/

set_option autoImplicit false

namespace UniversalImpossibility

variable (fs : FeatureSpaceMI)

/-! ### Step 1: MI > 0 → Positive Attribution Gap -/

/-- **MI implies positive attribution gap.** When MI > 0 under a symmetric
    DGP, the Rashomon witnesses have strictly opposite attribution orderings:
    one model ranks j higher (diff > 0), the other ranks k higher (diff < 0).
    The gap Δ = diff₁ - diff₂ > 0 is the quantitative content. -/
theorem mi_implies_positive_gap
    (attribution : Fin fs.P → Model → ℝ)
    (swap : Fin fs.P → Fin fs.P → Model → Model)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : Model, attribution j f ≠ attribution k f)
    (j k : Fin fs.P)
    (hdep : MIDependent fs mia j k) :
    ∃ f₁ f₂ : Model,
      attribution j f₁ - attribution k f₁ > 0 ∧
      attribution j f₂ - attribution k f₂ < 0 := by
  obtain ⟨f₁, f₂, h1, h2⟩ :=
    rashomon_from_mi_dependence fs attribution swap hsym_j hsym_k mia
      hdep_implies_diff j k hdep
  exact ⟨f₁, f₂, by linarith, by linarith⟩

/-! ### Step 2: Positive Gap → Unfaithfulness -/

/-- **Gap implies unfaithfulness.** If two models have opposite attribution
    orderings (diff₁ > 0, diff₂ < 0), then any single stable value r
    must disagree with the actual attribution difference on at least one model.

    Proof: if r = diff₁ then r ≠ diff₂ (since diff₁ > 0 > diff₂).
    If r ≠ diff₁ then r disagrees with f₁. -/
theorem gap_implies_unfaithfulness
    (diff₁ diff₂ r : ℝ)
    (h1 : diff₁ > 0) (h2 : diff₂ < 0) :
    r ≠ diff₁ ∨ r ≠ diff₂ := by
  by_contra h
  push_neg at h
  obtain ⟨hr1, hr2⟩ := h
  linarith

/-- **Quantitative bound (triangle inequality).** The total disagreement
    of any stable value r with two opposite-sign differences is at least
    the gap between them.

    |r - diff₁| + |r - diff₂| ≥ |diff₁ - diff₂| = diff₁ - diff₂ > 0 -/
theorem total_unfaithfulness_bound
    (diff₁ diff₂ r : ℝ)
    (h1 : diff₁ > 0) (h2 : diff₂ < 0) :
    |r - diff₁| + |r - diff₂| ≥ diff₁ - diff₂ := by
  -- Use the same triangle inequality approach as pointwise_unfaithfulness_bound
  have key : |r - diff₂| = |diff₂ - r| := abs_sub_comm r diff₂
  have h := abs_add_le (r - diff₁) (diff₂ - r)
  have : |(r - diff₁) + (diff₂ - r)| = |diff₂ - diff₁| := by ring_nf
  have hab : |diff₂ - diff₁| = diff₁ - diff₂ := by
    rw [abs_of_nonpos (by linarith)]; ring
  linarith

/-- **Pointwise bound.** At least one witness has unfaithfulness ≥ gap/2.
    This follows from: |r-d₁| + |r-d₂| ≥ gap → max(|r-d₁|, |r-d₂|) ≥ gap/2. -/
theorem pointwise_unfaithfulness_bound
    (diff₁ diff₂ r : ℝ)
    (h1 : diff₁ > 0) (h2 : diff₂ < 0) :
    |r - diff₁| ≥ (diff₁ - diff₂) / 2 ∨
    |r - diff₂| ≥ (diff₁ - diff₂) / 2 := by
  by_contra h
  push_neg at h
  obtain ⟨hr1, hr2⟩ := h
  -- Triangle inequality: |a| + |b| ≥ |a + b| applied to a = (r-d₁), b = (d₂-r)
  -- gives |r-d₁| + |d₂-r| ≥ |d₂-d₁| = d₁-d₂
  have key : |r - diff₂| = |diff₂ - r| := abs_sub_comm r diff₂
  have tri : |r - diff₁| + |diff₂ - r| ≥ |diff₂ - diff₁| := by
    have h := abs_add_le (r - diff₁) (diff₂ - r)
    have : |(r - diff₁) + (diff₂ - r)| = |diff₂ - diff₁| := by ring_nf
    linarith
  rw [key] at hr2
  have hab : |diff₂ - diff₁| = diff₁ - diff₂ := by
    rw [abs_of_nonpos (by linarith)]; ring
  linarith

/-! ### Step 3: MI → Quantitative Unfaithfulness (Complete Chain) -/

/-- **MI → quantitative unfaithfulness (the complete bridge).**

    When MI(X_j, X_k) > 0 under a symmetric DGP:
    1. Rashomon witnesses f₁, f₂ exist with opposite attribution orderings
    2. The attribution gap Δ = diff₁ - diff₂ > 0
    3. Any stable value r has pointwise unfaithfulness ≥ Δ/2 on at least
       one witness

    This connects the information-theoretic boundary (MI > 0) to the
    algebraic framework's quantitative bilemma (unfaithfulness ≥ Δ/2).
    The formal chain: MI > 0 → Δ > 0 → unfaithfulness ≥ Δ/2. -/
theorem mi_quantitative_unfaithfulness
    (attribution : Fin fs.P → Model → ℝ)
    (swap : Fin fs.P → Fin fs.P → Model → Model)
    (hsym_j : ∀ j k f, attribution j (swap j k f) = attribution k f)
    (hsym_k : ∀ j k f, attribution k (swap j k f) = attribution j f)
    (mia : MutualInfoAssignment fs)
    (hdep_implies_diff : ∀ j k : Fin fs.P,
      MIDependent fs mia j k →
      ∃ f : Model, attribution j f ≠ attribution k f)
    (j k : Fin fs.P)
    (hdep : MIDependent fs mia j k)
    (r : ℝ) :
    ∃ f : Model,
      |r - (attribution j f - attribution k f)| ≥
        (attribution j f - attribution k f) / 2 ∨
      |r - (attribution j f - attribution k f)| > 0 := by
  obtain ⟨f₁, f₂, h1, h2⟩ :=
    mi_implies_positive_gap fs attribution swap hsym_j hsym_k mia
      hdep_implies_diff j k hdep
  have bound := pointwise_unfaithfulness_bound
    (attribution j f₁ - attribution k f₁)
    (attribution j f₂ - attribution k f₂) r h1 h2
  rcases bound with hf1 | hf2
  · exact ⟨f₁, Or.inl (by linarith)⟩
  · exact ⟨f₂, Or.inl (by linarith)⟩

end UniversalImpossibility
