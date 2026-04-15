import UniversalImpossibility.ExplanationSystem
import UniversalImpossibility.MaximalIncompatibility
import UniversalImpossibility.BilemmaCharacterization
import Mathlib.Tactic.ByContra

/-!
# Generalized Bilemma and Characterization

Results from the Ostrowski impossibility companion (v1.2.0, upstream-handoff-v3).

## Generalized bilemma

The bilemma holds under a condition strictly weaker than maximal
incompatibility: the compatibility intersection at ONE Rashomon pair
is empty. This broadens the bilemma's applicability — systems that are
NOT maximally incompatible can still trigger it.

## Complete characterization (per fiber)

For single-fiber systems: bilemma ↔ no common compatible element.
For multi-fiber systems: if every fiber has a compatible element, F+S
is achievable (via Classical.choose per fiber).

## F+D parameterization

The set of F+D explanations is parameterized by the incompatibility-
equivalence class of explain. For max-incompat systems this class is
a singleton; for non-max-incompat systems it may be larger.

## Coverage conflict

The algebraic feature that powers the bilemma: every element of H is
incompatible with some explain-value. Anti-correlated with neutral
elements. Explains why binary H strengthens the impossibility while
larger H weakens it (the opposite of Arrow's theorem).

## Abstract impossibility framework

Both Arrow's theorem and the bilemma are instances of one abstract
structure, with the tightness spectrum from full to collapsed.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

-- ============================================================================
-- Generalized bilemma (weaker than maximal incompatibility)
-- ============================================================================

/-- **The Generalized Bilemma.** F+S is impossible whenever there exists a
    Rashomon pair where no element of H is compatible with BOTH explain
    values. This is strictly weaker than maximal incompatibility: it only
    requires the compatibility intersection to be empty at one pair, not
    that every compatible pair is equal.

    Condition: ∀ c, ¬incompatible(c, explain(θ₁)) → incompatible(c, explain(θ₂))
    i.e., Compatible(explain(θ₁)) ∩ Compatible(explain(θ₂)) = ∅. -/
theorem generalized_bilemma
    (S : ExplanationSystem Θ H Y)
    (θ₁ θ₂ : Θ)
    (hobs : S.observe θ₁ = S.observe θ₂)
    (_hinc : S.incompatible (S.explain θ₁) (S.explain θ₂))
    (hno_common : ∀ (c : H),
      ¬S.incompatible c (S.explain θ₁) → S.incompatible c (S.explain θ₂))
    (E : Θ → H) (hf : faithful S E) (hs : stable S E) :
    False := by
  have heq := hs θ₁ θ₂ hobs
  have hf1 := hf θ₁
  have hinc2 := hno_common (E θ₁) hf1
  rw [heq] at hinc2
  exact hf θ₂ hinc2

/-- Maximal incompatibility implies the generalized bilemma's condition at
    every Rashomon pair. The generalized bilemma strictly generalizes the
    original bilemma. -/
theorem maxIncompat_implies_no_common_compatible
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (θ₁ θ₂ : Θ)
    (hinc : S.incompatible (S.explain θ₁) (S.explain θ₂)) :
    ∀ (c : H), ¬S.incompatible c (S.explain θ₁) →
      S.incompatible c (S.explain θ₂) := by
  intro c hcompat
  have := hmax c (S.explain θ₁) hcompat
  rw [this]
  exact hinc

-- ============================================================================
-- Complete characterization (per fiber)
-- ============================================================================

/-- **Single-fiber converse.** If there exists an element compatible with
    all explain-values, then F+S IS achievable via the constant map. -/
theorem no_bilemma_of_neutral_on_fiber
    (S : ExplanationSystem Θ H Y)
    (c : H)
    (hcompat : ∀ (θ : Θ), ¬S.incompatible c (S.explain θ)) :
    faithful S (fun _ => c) ∧ stable S (fun _ => c) :=
  ⟨fun θ => hcompat θ, fun _ _ _ => rfl⟩

/-- **Multi-fiber converse.** If for every observe-value y, there exists an
    element of H compatible with all explain-values on that fiber, then
    F+S is achievable.

    Completes the characterization: bilemma ↔ ∃ fiber with no compatible
    element. -/
theorem fs_achievable_of_fiber_compatible
    (S : ExplanationSystem Θ H Y)
    (hfiber : ∀ (y : Y), ∃ (c : H), ∀ (θ : Θ),
      S.observe θ = y → ¬S.incompatible c (S.explain θ)) :
    ∃ (E : Θ → H), faithful S E ∧ stable S E := by
  let E : Θ → H := fun θ => Classical.choose (hfiber (S.observe θ))
  refine ⟨E, ?_, ?_⟩
  · intro θ
    exact Classical.choose_spec (hfiber (S.observe θ)) θ rfl
  · intro θ₁ θ₂ hobs
    show Classical.choose (hfiber (S.observe θ₁)) =
         Classical.choose (hfiber (S.observe θ₂))
    rw [hobs]

-- ============================================================================
-- F+D parameterization via incompatibility-equivalence
-- ============================================================================

/-- Two elements of H are **incompatibility-equivalent** if they have the
    same incompatibility profile. -/
def incompEquiv (S : ExplanationSystem Θ H Y) (x y : H) : Prop :=
  ∀ (h : H), S.incompatible x h ↔ S.incompatible y h

theorem incompEquiv_refl (S : ExplanationSystem Θ H Y) (x : H) :
    incompEquiv S x x :=
  fun _ => Iff.rfl

theorem incompEquiv_symm (S : ExplanationSystem Θ H Y) (x y : H) :
    incompEquiv S x y → incompEquiv S y x :=
  fun h k => (h k).symm

/-- **F+D from incompatibility-equivalence.** If E(θ) is incompatibility-
    equivalent to explain(θ) and compatible with explain(θ) at every θ,
    then E is faithful + decisive.

    The set of F+D explanations is parameterized by choosing, at each θ,
    an element in the incompatibility-equivalence class of explain(θ)
    that is also compatible with it. -/
theorem fd_from_incomp_equiv
    (S : ExplanationSystem Θ H Y) (E : Θ → H)
    (hcompat : ∀ θ, ¬S.incompatible (E θ) (S.explain θ))
    (hequiv : ∀ θ, incompEquiv S (E θ) (S.explain θ)) :
    faithful S E ∧ decisive S E :=
  ⟨fun θ => hcompat θ, fun θ h hinc => (hequiv θ h).mpr hinc⟩

/-- For maximally incompatible systems, the compatible incompatibility-
    equivalence class is a singleton: only explain(θ) itself. -/
theorem maxIncompat_incompEquiv_singleton
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (c : H) (θ : Θ)
    (hcompat : ¬S.incompatible c (S.explain θ)) :
    c = S.explain θ :=
  hmax c (S.explain θ) hcompat

-- ============================================================================
-- Coverage conflict
-- ============================================================================

/-- An explanation system has **coverage conflict** if every element of H
    is incompatible with some native explanation value. No "safe harbor"
    exists — every possible explanation clashes with something.

    Coverage conflict powers the bilemma: when every element clashes with
    some explain-value, no neutral element exists, and F+S is impossible.

    Anti-monotone in |H|: enlarging H can introduce neutral elements,
    destroying coverage conflict. Binary H with incompatible = ≠ always
    has coverage conflict. -/
def hasCoverageConflict (S : ExplanationSystem Θ H Y) : Prop :=
  ∀ (c : H), ∃ (θ : Θ), S.incompatible c (S.explain θ)

/-- Coverage conflict → no neutral element. -/
theorem coverageConflict_implies_no_neutral
    (S : ExplanationSystem Θ H Y)
    (hcc : hasCoverageConflict S) :
    ¬hasNeutralElement S := by
  intro ⟨c, hc⟩
  obtain ⟨θ, hinc⟩ := hcc c
  exact hc θ hinc

/-- Maximal incompatibility → coverage conflict. -/
theorem maxIncompat_implies_coverageConflict
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S) :
    hasCoverageConflict S := by
  intro c
  obtain ⟨θ₁, θ₂, _, hinc⟩ := S.rashomon
  by_contra hall
  push Not at hall
  have h1 : c = S.explain θ₁ := hmax c (S.explain θ₁) (hall θ₁)
  have h2 : c = S.explain θ₂ := hmax c (S.explain θ₂) (hall θ₂)
  have heq : S.explain θ₁ = S.explain θ₂ := h1.symm.trans h2
  rw [heq] at hinc
  exact S.incompatible_irrefl _ hinc

/-- Neutral element → no coverage conflict. -/
theorem neutral_destroys_coverageConflict
    (S : ExplanationSystem Θ H Y)
    (hn : hasNeutralElement S) :
    ¬hasCoverageConflict S := by
  intro hcc
  exact coverageConflict_implies_no_neutral S hcc hn

-- ============================================================================
-- Abstract impossibility framework
-- ============================================================================

/-- An **abstract impossibility system** captures the common structure of
    the trilemma, Arrow's theorem, and other impossibility results:
    three desirable properties where no solution satisfies all three,
    but some pairs may be achievable. -/
structure AbstractImpossibility (S : Type) where
  P₁ : S → Prop
  P₂ : S → Prop
  P₃ : S → Prop
  impossible : ∀ (s : S), P₁ s → P₂ s → P₃ s → False
  tight_13 : ∃ (s : S), P₁ s ∧ P₃ s

/-- Full tightness: all three pairs achievable. -/
def AbstractImpossibility.fullTightness {S : Type}
    (A : AbstractImpossibility S) : Prop :=
  (∃ s, A.P₁ s ∧ A.P₂ s) ∧
  (∃ s, A.P₁ s ∧ A.P₃ s) ∧
  (∃ s, A.P₂ s ∧ A.P₃ s)

/-- Collapsed tightness (bilemma regime): only P₁+P₃ achievable. -/
def AbstractImpossibility.collapsedTightness {S : Type}
    (A : AbstractImpossibility S) : Prop :=
  ¬(∃ s, A.P₁ s ∧ A.P₂ s) ∧
  (∃ s, A.P₁ s ∧ A.P₃ s) ∧
  ¬(∃ s, A.P₂ s ∧ A.P₃ s)

/-- A committal element inherits all incompatibilities of all explain-values.
    (Upstreamed from Ostrowski BilemmaCharacterization.) -/
def hasCommittalElement (S : ExplanationSystem Θ H Y) : Prop :=
  ∃ (c : H), ∀ (θ : Θ) (h : H), S.incompatible (S.explain θ) h → S.incompatible c h

/-- The explanation impossibility as an abstract impossibility instance. -/
noncomputable def explanationAbstractImpossibility
    (S : ExplanationSystem Θ H Y) :
    AbstractImpossibility (Θ → H) :=
  { P₁ := faithful S
    P₂ := stable S
    P₃ := decisive S
    impossible := fun E hf hs hd => explanation_impossibility S E hf hs hd
    tight_13 := ⟨S.explain, (tightness_faithful_decisive S).1,
                              (tightness_faithful_decisive S).2⟩ }

/-- Max-incompat → collapsed tightness. -/
theorem explanation_collapsed_of_maxIncompat
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S) :
    (explanationAbstractImpossibility S).collapsedTightness := by
  refine ⟨?_, ?_, ?_⟩
  · intro ⟨E, hf, hs⟩
    exact bilemma S hmax E hf hs
  · exact (explanationAbstractImpossibility S).tight_13
  · intro ⟨E, hs, hd⟩
    exact no_stable_decisive_of_maxIncompat S hmax E hs hd

/-- Neutral + committal → full tightness (matching Arrow). -/
theorem explanation_full_tightness_of_neutral_committal
    (S : ExplanationSystem Θ H Y)
    (hn : hasNeutralElement S)
    (hc : hasCommittalElement S) :
    (explanationAbstractImpossibility S).fullTightness := by
  refine ⟨?_, ?_, ?_⟩
  · exact neutral_implies_faithful_stable S hn
  · exact (explanationAbstractImpossibility S).tight_13
  · let ⟨c, hc'⟩ := hc
    exact ⟨fun _ => c, fun _ _ _ => rfl, fun θ h hinc => hc' θ h hinc⟩
