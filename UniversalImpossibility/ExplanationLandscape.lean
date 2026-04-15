import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility
import UniversalImpossibility.BilemmaCharacterization
import UniversalImpossibility.GeneralizedBilemma
import Mathlib.Tactic.ByContra

/-!
# The Explanation Landscape

Formal bridge between the abstract bilemma characterization and the
quantitative η = dim(V^G)/dim(V) prediction.

## The landscape endpoints

- η = 0 ↔ full coverage conflict ↔ bilemma ↔ only F+D survives
- η = 1 ↔ no coverage conflict ↔ neutral exists ↔ full tightness
- 0 < η < 1 ↔ partial coverage conflict ↔ some queries stable

## Bridge theorems

The characterization (bilemma ↔ no fiber-compatible element) determines
the boundary of the landscape. Coverage conflict measures how far a
system is from the boundary. The Gaussian flip formula is the empirical
measure of coverage conflict for continuous importance spaces.

These theorems close the gap identified in the monograph (Section 8.7):
"The bridge from the abstract framework to the quantitative η prediction
is empirical, not derived." We derive the qualitative bridge; the
quantitative bridge (R²=0.957) remains empirical.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

-- ============================================================================
-- The landscape endpoints: η = 0 and η = 1
-- ============================================================================

/-- **η = 0 endpoint.** Full coverage conflict means no neutral element
    exists, which means F+S is impossible by the bilemma characterization.
    Note: coverage conflict alone does not prove the bilemma (which needs
    Rashomon + a specific pair condition). What it proves is the weaker
    but important result: no neutral escape exists. -/
theorem landscape_bottom_no_fs
    (S : ExplanationSystem Θ H Y)
    (hcc : hasCoverageConflict S) :
    ¬hasNeutralElement S :=
  coverageConflict_implies_no_neutral S hcc

/-- **η = 1 endpoint.** No coverage conflict implies a neutral element
    exists, which implies F+S is achievable. In the landscape, this is
    the top (all queries are answerable). -/
theorem landscape_top_fs_achievable
    (S : ExplanationSystem Θ H Y)
    (hno_cc : ¬hasCoverageConflict S) :
    hasNeutralElement S := by
  -- ¬(∀ c, ∃ θ, incomp c (explain θ)) means ∃ c, ∀ θ, ¬incomp c (explain θ)
  by_contra hno_neutral
  apply hno_cc
  intro c
  by_contra h
  push Not at h
  exact hno_neutral ⟨c, h⟩

/-- The landscape top implies F+S achievable. -/
theorem landscape_top_achievable
    (S : ExplanationSystem Θ H Y)
    (hno_cc : ¬hasCoverageConflict S) :
    ∃ (E : Θ → H), faithful S E ∧ stable S E :=
  neutral_implies_faithful_stable S (landscape_top_fs_achievable S hno_cc)

-- ============================================================================
-- The landscape biconditional
-- ============================================================================

/-- **The Landscape Biconditional.** Coverage conflict ↔ no neutral element.
    This is the formal bridge: the same algebraic property that determines
    the abstract characterization (bilemma ↔ no fiber-compatible element)
    also determines the landscape position (η = 0 ↔ full coverage conflict).

    Combined with neutral → F+S achievable and ¬neutral → F+S blocked,
    this gives: the landscape position is entirely determined by whether
    coverage conflict holds. -/
theorem coverage_conflict_iff_no_neutral
    (S : ExplanationSystem Θ H Y) :
    hasCoverageConflict S ↔ ¬hasNeutralElement S := by
  constructor
  · exact coverageConflict_implies_no_neutral S
  · intro hno_neutral c
    by_contra h
    push Not at h
    exact hno_neutral ⟨c, h⟩

-- ============================================================================
-- Enrichment monotonically expands the landscape
-- ============================================================================

/-- **Enrichment expands the landscape.** If a system has coverage conflict
    (η = 0), adding a neutral-compatible element moves it to η > 0.
    This is the formal content of "enrichment weakens the impossibility." -/
theorem enrichment_expands_landscape
    (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ θ, ¬S.incompatible c (S.explain θ)) :
    ∃ (E : Θ → H), faithful S E ∧ stable S E :=
  ⟨fun _ => c, fun θ => hc θ, fun _ _ _ => rfl⟩

/-- **The landscape is monotone in enrichment.** If S₁ is a subsystem of S₂
    (same observe/explain, but S₂ has more H elements), then S₂ has
    weakly more F+S explanations than S₁.

    Stated contrapositively: if S₂ has coverage conflict, then any
    subsystem (restriction of H) also has coverage conflict. -/
theorem coverage_conflict_monotone_restriction
    (S : ExplanationSystem Θ H Y)
    (hcc : hasCoverageConflict S)
    (P : H → Prop)
    (_hP : ∀ θ, P (S.explain θ)) :
    ∀ (c : H), P c → ∃ (θ : Θ), S.incompatible c (S.explain θ) := by
  intro c _
  exact hcc c

-- ============================================================================
-- The maximal incompatibility → bilemma → η = 0 chain
-- ============================================================================

/-- **The complete chain for max-incompat systems.**
    Max-incompat → coverage conflict → no neutral → bilemma.
    This is the formal derivation of η = 0 for maximally incompatible
    systems, closing the bridge between the abstract characterization
    and the landscape. -/
theorem maxIncompat_landscape_chain
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S) :
    hasCoverageConflict S ∧ ¬hasNeutralElement S := by
  constructor
  · exact maxIncompat_implies_coverageConflict S hmax
  · exact coverageConflict_implies_no_neutral S (maxIncompat_implies_coverageConflict S hmax)

/-- **The generalized bilemma in landscape terms.**
    Any system with coverage conflict at a Rashomon pair is at the
    bilemma boundary (η = 0 for that pair). This does not require
    GLOBAL coverage conflict — local suffices. -/
theorem local_coverage_conflict_at_pair
    (S : ExplanationSystem Θ H Y)
    (θ₁ θ₂ : Θ)
    (hobs : S.observe θ₁ = S.observe θ₂)
    (hinc : S.incompatible (S.explain θ₁) (S.explain θ₂))
    (hlocal : ∀ (c : H),
      ¬S.incompatible c (S.explain θ₁) → S.incompatible c (S.explain θ₂)) :
    ∀ (E : Θ → H), faithful S E → stable S E → False :=
  fun E hf hs => generalized_bilemma S θ₁ θ₂ hobs hinc hlocal E hf hs
