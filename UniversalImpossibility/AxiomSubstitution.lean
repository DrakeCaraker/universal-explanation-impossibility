import UniversalImpossibility.ExplanationSystem

/-!
# Axiom Substitution Test — Alternative Formalizations of "Decisive"

Phase 2, Task 3: We try three alternative formalizations of the decisive axiom
and check whether the impossibility and tightness results still hold.

## Results

-- | Formalization      | Impossibility?  | Tightness?  | Notes                                                |
-- |--------------------|-----------------|-------------|------------------------------------------------------|
-- | decisive (current) | YES (3 axioms)  | YES (3/3)   | Original: faithful + stable + decisive => False      |
-- | complete           | YES (2 axioms!) | PARTIAL     | stable + complete => False (faithfulness redundant!)  |
-- | resolving          | YES (2 axioms!) | PARTIAL     | Logically equivalent to complete (contrapositive)     |
-- | surjective         | NO              | N/A         | Swaps argument order; proof fails without symmetry   |

### Key findings

1. **complete / resolving are logically equivalent** (contrapositives of each other).
   They STRENGTHEN the impossibility: only stable + complete => False under Rashomon.
   Faithfulness is not needed. This makes the 3-axiom impossibility still hold trivially
   (adding a hypothesis never breaks a theorem), but tightness changes: the "stable +
   complete (dropping faithful)" pair is already impossible, so only 2 of 3 tightness
   witnesses survive.

2. **surjective** swaps the argument order of incompatible in the conclusion
   (incompatible(h, E θ) instead of incompatible(E θ, h)). The impossibility proof
   breaks because we derive incompatible(explain θ, E θ) but faithfulness gives
   ¬incompatible(E θ, explain θ) — these differ when incompatible is not symmetric.
   Without a symmetry axiom on incompatible, the proof cannot be completed.

-/

set_option autoImplicit false

variable {Θ : Type} {H : Type} {Y : Type}

-- ============================================================================
-- Alternative 1: "Complete" — E never collapses incompatible explanations
-- ============================================================================

/-- Complete: if E maps two configurations to the same output, their native
    explanations must be compatible. Contrapositive: incompatible native
    explanations force E to distinguish the configurations. -/
def decisive_complete (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), E θ₁ = E θ₂ → ¬S.incompatible (S.explain θ₁) (S.explain θ₂)

/-- Complete + Stable => False (faithfulness not needed!).
    Rashomon gives θ₁, θ₂ with same observation and incompatible explanations.
    Stability forces E θ₁ = E θ₂. Completeness then says their explanations
    are compatible — contradicting Rashomon. -/
theorem impossibility_complete (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hs : stable S E) (hc : decisive_complete S E) :
    False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := S.rashomon
  have heq : E θ₁ = E θ₂ := hs θ₁ θ₂ hobs
  exact hc θ₁ θ₂ heq hinc

/-- The 3-axiom version also holds (adding faithful as an unused hypothesis). -/
theorem impossibility_complete_three (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (_hf : faithful S E) (hs : stable S E) (hc : decisive_complete S E) :
    False :=
  impossibility_complete S E hs hc

/-- Tightness 1 (complete): Faithful + Complete (dropping Stable).
    E = explain is faithful (by irreflexivity) and complete (if explain θ₁ = explain θ₂
    then incompatible is ruled out by irreflexivity). -/
theorem tightness_faithful_complete (S : ExplanationSystem Θ H Y) :
    faithful S S.explain ∧ decisive_complete S S.explain := by
  constructor
  · intro θ
    exact S.incompatible_irrefl (S.explain θ)
  · intro θ₁ θ₂ heq
    rw [heq]
    exact S.incompatible_irrefl (S.explain θ₂)

/-- Tightness 2 (complete): Faithful + Stable (dropping Complete).
    Constant function E(θ) = c is faithful (if c is universally compatible)
    and stable (constant), but NOT complete (it collapses the Rashomon pair). -/
theorem tightness_faithful_stable_complete (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ (θ : Θ), ¬S.incompatible c (S.explain θ)) :
    faithful S (fun _ => c) ∧ stable S (fun _ => c) := by
  exact tightness_faithful_stable S c hc

/-- Tightness 3 (complete): Stable + Complete (dropping Faithful) is IMPOSSIBLE.
    This pair already implies False by `impossibility_complete`, so there is no
    witness. We prove this explicitly. -/
theorem no_tightness_stable_complete (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hs : stable S E) (hc : decisive_complete S E) :
    False :=
  impossibility_complete S E hs hc

-- ============================================================================
-- Alternative 2: "Resolving" — E distinguishes incompatible explanations
-- ============================================================================

/-- Resolving: if the native explanations are incompatible, E must distinguish
    the configurations. This is the direct (non-contrapositive) form of complete. -/
def decisive_resolving (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ₁ θ₂ : Θ), S.incompatible (S.explain θ₁) (S.explain θ₂) → E θ₁ ≠ E θ₂

/-- Resolving is logically equivalent to complete. -/
theorem resolving_iff_complete (S : ExplanationSystem Θ H Y) (E : Θ → H) :
    decisive_resolving S E ↔ decisive_complete S E := by
  constructor
  · intro hr θ₁ θ₂ heq hinc
    exact hr θ₁ θ₂ hinc heq
  · intro hc θ₁ θ₂ hinc heq
    exact hc θ₁ θ₂ heq hinc

/-- Resolving + Stable => False (follows from equivalence with complete). -/
theorem impossibility_resolving (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hs : stable S E) (hr : decisive_resolving S E) :
    False := by
  obtain ⟨θ₁, θ₂, hobs, hinc⟩ := S.rashomon
  have heq : E θ₁ = E θ₂ := hs θ₁ θ₂ hobs
  exact hr θ₁ θ₂ hinc heq

/-- The 3-axiom version also holds. -/
theorem impossibility_resolving_three (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (_hf : faithful S E) (hs : stable S E) (hr : decisive_resolving S E) :
    False :=
  impossibility_resolving S E hs hr

/-- Tightness 1 (resolving): Faithful + Resolving (dropping Stable).
    E = explain is faithful and resolving. -/
theorem tightness_faithful_resolving (S : ExplanationSystem Θ H Y) :
    faithful S S.explain ∧ decisive_resolving S S.explain := by
  constructor
  · intro θ
    exact S.incompatible_irrefl (S.explain θ)
  · intro θ₁ θ₂ hinc heq
    rw [heq] at hinc
    exact S.incompatible_irrefl (S.explain θ₂) hinc

/-- Tightness 2 (resolving): Faithful + Stable (dropping Resolving). Same as complete. -/
theorem tightness_faithful_stable_resolving (S : ExplanationSystem Θ H Y)
    (c : H) (hc : ∀ (θ : Θ), ¬S.incompatible c (S.explain θ)) :
    faithful S (fun _ => c) ∧ stable S (fun _ => c) := by
  exact tightness_faithful_stable S c hc

/-- Tightness 3 (resolving): Stable + Resolving is already impossible. -/
theorem no_tightness_stable_resolving (S : ExplanationSystem Θ H Y)
    (E : Θ → H) (hs : stable S E) (hr : decisive_resolving S E) :
    False :=
  impossibility_resolving S E hs hr

-- ============================================================================
-- Alternative 3: "Surjective" — reverse-direction decisive
-- ============================================================================

/-- Surjective: for every h incompatible with explain(θ), h is also incompatible
    with E(θ). This is the "reverse argument order" version of the original decisive.
    Original decisive: incompatible(explain θ, h) => incompatible(E θ, h)
    Surjective:        incompatible(h, explain θ) => incompatible(h, E θ)        -/
def decisive_surjective (S : ExplanationSystem Θ H Y) (E : Θ → H) : Prop :=
  ∀ (θ : Θ) (h : H), S.incompatible h (S.explain θ) → S.incompatible h (E θ)

-- The impossibility proof FAILS for surjective without a symmetry axiom on
-- incompatible. Here is why:
--
-- 1. Rashomon gives θ₁, θ₂ with observe θ₁ = observe θ₂ and
--    incompatible(explain θ₁, explain θ₂).
-- 2. Surjective at θ₂ with h = explain θ₁:
--    incompatible(explain θ₁, explain θ₂) => incompatible(explain θ₁, E θ₂)
-- 3. Stability: E θ₁ = E θ₂, so incompatible(explain θ₁, E θ₁).
-- 4. Faithfulness gives: ¬incompatible(E θ₁, explain θ₁).
-- 5. Step 3 gives incompatible(explain θ₁, E θ₁).
--    Step 4 gives ¬incompatible(E θ₁, explain θ₁).
--    These have DIFFERENT argument orders. Without symmetry of incompatible,
--    no contradiction.
--
-- The proof cannot be completed. We demonstrate this by showing the surjective
-- axiom IS satisfiable together with faithful and stable, via a concrete model
-- where incompatible is asymmetric.

/-- Witness: surjective + faithful + stable can all hold simultaneously when
    incompatible is asymmetric. We build a concrete ExplanationSystem where
    all three properties hold together.

    Construction: Θ = Bool, H = Bool, Y = Unit.
    observe: constant (both map to ()).
    explain: id (true ↦ true, false ↦ false).
    incompatible: (true, false) only (asymmetric!).
    E: constant false.

    - Rashomon: θ₁ = true, θ₂ = false, observe equal, incompatible(true, false). CHECK.
    - Faithful: ¬incompatible(E θ, explain θ) = ¬incompatible(false, explain θ).
      For θ = true: ¬incompatible(false, true). Only (true, false) is incompatible. CHECK.
      For θ = false: ¬incompatible(false, false). Irreflexive. CHECK.
    - Stable: E is constant. CHECK.
    - Surjective: incompatible(h, explain θ) → incompatible(h, E θ) = incompatible(h, false).
      For θ = true: incompatible(h, true) → incompatible(h, false).
        Only (true, false) is incompatible, so incompatible(h, true) is always false. Vacuous. CHECK.
      For θ = false: incompatible(h, false) → incompatible(h, false). Trivial. CHECK. -/
theorem surjective_consistent :
    ∃ (S : ExplanationSystem Bool Bool Unit) (E : Bool → Bool),
      faithful S E ∧ stable S E ∧ decisive_surjective S E := by
  refine ⟨⟨fun _ => (), id, fun a b => a = true ∧ b = false, ?_, ?_⟩,
          fun _ => false, ?_, ?_, ?_⟩
  · -- incompatible_irrefl
    intro h ⟨_, hf⟩
    cases h <;> simp_all
  · -- rashomon
    exact ⟨true, false, rfl, ⟨rfl, rfl⟩⟩
  · -- faithful: ¬incompatible(false, explain θ)
    intro θ ⟨h1, _⟩
    simp at h1
  · -- stable: constant function
    intro _ _ _
    rfl
  · -- surjective: incompatible(h, explain θ) → incompatible(h, E θ)
    intro θ h ⟨ha, hb⟩
    cases θ <;> simp_all

-- ============================================================================
-- Relationship: original decisive implies complete/resolving
-- ============================================================================

/-- The original decisive axiom implies complete (and hence resolving).
    This means the original impossibility can be seen as a consequence of the
    stronger complete/resolving impossibility. -/
theorem decisive_implies_complete (S : ExplanationSystem Θ H Y) (E : Θ → H)
    (hf : faithful S E) (hd : decisive S E) :
    decisive_complete S E := by
  intro θ₁ θ₂ heq hinc
  have h1 : S.incompatible (E θ₁) (S.explain θ₂) := hd θ₁ (S.explain θ₂) hinc
  rw [heq] at h1
  exact hf θ₂ h1

-- Note: complete does NOT imply decisive in general.
-- Complete only constrains E on pairs with the same E-value,
-- while decisive constrains E at each individual configuration.
-- Complete is strictly weaker (when combined with faithfulness).

-- ============================================================================
-- Summary
-- ============================================================================

/--
## Summary of axiom substitution analysis

The decisive axiom admits three natural alternatives:

1. **complete** (E θ₁ = E θ₂ → ¬incompatible(explain θ₁, explain θ₂)):
   STRENGTHENS the impossibility to a 2-axiom result (stable + complete => False).
   Faithfulness becomes redundant. Tightness degrades: only 2 of 3 pairs achievable.

2. **resolving** (incompatible(explain θ₁, explain θ₂) → E θ₁ ≠ E θ₂):
   Logically equivalent to complete (contrapositive). Same results.

3. **surjective** (incompatible(h, explain θ) → incompatible(h, E θ)):
   The impossibility FAILS. We construct a concrete witness (asymmetric
   incompatible on Bool) where faithful + stable + surjective all hold
   simultaneously. The failure is fundamental: it depends on the argument
   order of incompatible, and without symmetry the proof cannot close.

**Conclusion**: The original decisive axiom is well-calibrated. The complete/resolving
variants are too strong (they make faithfulness redundant and break full tightness).
The surjective variant is too weak (it loses the impossibility entirely).
The original achieves the Goldilocks balance: tight 3-axiom impossibility with all
three pairwise tightness witnesses.
-/
theorem axiom_substitution_summary : True := trivial
