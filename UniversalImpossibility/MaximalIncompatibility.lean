import UniversalImpossibility.ExplanationSystem

/-!
# Strengthened Impossibility for Maximally Incompatible Systems

When the explanation space is maximally incompatible — the only compatible
pair is (h, h) — the trilemma collapses:

1. Faithful ⇒ E = explain (compatibility with explain forces equality)
2. Decisive ⇒ E = explain (by a symmetric argument via irreflexivity)
3. Both faithful and decisive individually force E = explain
4. explain is not stable (Rashomon)
5. Therefore: F+S impossible (bilemma), S+D impossible, only F+D survives

This strictly strengthens the universal impossibility for a natural class
of systems: any system with a binary explanation space where incompatible = ≠.

## Tightness Comparison

| Property pair   | Abstract framework | Maximally incompatible |
|----------------|-------------------|----------------------|
| F + D          | ✓                 | ✓                    |
| F + S          | ✓                 | ✗ (bilemma)          |
| S + D          | ✓                 | ✗                    |

The abstract framework has full tightness (all 3 pairs achievable).
Maximal incompatibility eliminates 2 of 3.

## Recovery via Enrichment

Adding a "neutral" element c (compatible with everything) to H restores
F+S via E(θ) = c. This is the abstract version of:
- DASH averaging (attribution impossibility)
- CPDAG averaging (causal discovery)
- Adelic resolution (physics — see companion paper)

The enrichment does NOT restore S+D — that remains impossible even with
the neutral element, because decisive still forces E = explain on the
non-neutral elements.
-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

-- ============================================================================
-- Maximal incompatibility: compatible ⇒ equal
-- ============================================================================

/-- An explanation system is maximally incompatible if the only compatible
    pair of explanations is (h, h). Equivalently: every pair of distinct
    explanations is incompatible. Any binary H with incompatible = ≠
    satisfies this condition. -/
def maximallyIncompatible (S : ExplanationSystem Θ H Y) : Prop :=
  ∀ (h₁ h₂ : H), ¬S.incompatible h₁ h₂ → h₁ = h₂

-- ============================================================================
-- Faithful ⇒ E = explain
-- ============================================================================

/-- In a maximally incompatible system, faithful forces E = explain.
    Proof: faithful means ¬incompatible(E(θ), explain(θ)), which by
    maximal incompatibility means E(θ) = explain(θ). -/
theorem faithful_eq_explain_of_maxIncompat
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hf : faithful S E) :
    ∀ θ, E θ = S.explain θ :=
  fun θ => hmax _ _ (hf θ)

-- ============================================================================
-- Decisive ⇒ E = explain (the symmetric result)
-- ============================================================================

/-- In a maximally incompatible system, decisive ALSO forces E = explain.
    Proof: if E(θ) ≠ explain(θ), then incompatible(explain(θ), E(θ))
    (by maximal incompatibility). By decisiveness,
    incompatible(E(θ), E(θ)) — contradicting irreflexivity. -/
theorem decisive_eq_explain_of_maxIncompat
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hd : decisive S E) :
    ∀ θ, E θ = S.explain θ :=
  fun θ => (hmax _ _ (fun hinc => S.incompatible_irrefl (E θ) (hd θ (E θ) hinc))).symm

-- ============================================================================
-- The Bilemma: F+S impossible
-- ============================================================================

/-- **The Bilemma.** In a maximally incompatible system, no explanation is
    simultaneously faithful and stable.

    This is STRONGER than the trilemma (which only rules out F+S+D).
    The proof: faithful ⇒ E = explain ⇒ decisive, then the trilemma
    gives the contradiction. -/
theorem bilemma (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hf : faithful S E) (hs : stable S E) : False := by
  have heq := faithful_eq_explain_of_maxIncompat S hmax E hf
  have hd : decisive S E := fun θ h hinc => heq θ ▸ hinc
  exact explanation_impossibility S E hf hs hd

-- ============================================================================
-- S+D also impossible
-- ============================================================================

/-- In a maximally incompatible system, no explanation is simultaneously
    stable and decisive either. The proof mirrors the bilemma:
    decisive ⇒ E = explain, but explain is not stable (Rashomon). -/
theorem no_stable_decisive_of_maxIncompat
    (S : ExplanationSystem Θ H Y)
    (hmax : maximallyIncompatible S)
    (E : Θ → H) (hs : stable S E) (hd : decisive S E) : False := by
  have heq := decisive_eq_explain_of_maxIncompat S hmax E hd
  have hf : faithful S E := by
    intro θ
    rw [heq θ]
    exact S.incompatible_irrefl _
  exact explanation_impossibility S E hf hs hd

-- ============================================================================
-- Only F+D survives
-- ============================================================================

/-- In a maximally incompatible system, F+D is achievable
    (via E = explain, which is always faithful and decisive). -/
theorem maxIncompat_tightness_fd (S : ExplanationSystem Θ H Y)
    (_hmax : maximallyIncompatible S) :
    faithful S S.explain ∧ decisive S S.explain :=
  tightness_faithful_decisive S

/-- E = explain is not stable (by Rashomon). -/
theorem explain_not_stable (S : ExplanationSystem Θ H Y) :
    ¬stable S S.explain := by
  intro hstable
  have ⟨hf, hd⟩ := tightness_faithful_decisive S
  exact explanation_impossibility S S.explain hf hstable hd

-- ============================================================================
-- Recovery: adding a neutral element restores F+S
-- ============================================================================

/-- If a neutral element c exists (compatible with all native explanations),
    then F+S is achievable via E(θ) = c. This is the abstract version of
    DASH averaging, CPDAG averaging, and the adelic resolution.

    Note: S+D is NOT restored — decisive still forces E = explain on the
    non-neutral elements. -/
theorem recovery_faithful_stable (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ θ, ¬S.incompatible c (S.explain θ)) :
    faithful S (fun _ => c) ∧ stable S (fun _ => c) :=
  tightness_faithful_stable S c hc
