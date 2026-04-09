import UniversalImpossibility.ExplanationSystem

/-!
# Structural Ubiquity of Explanation Impossibility

The impossibility is not a pathological corner case — it is generic whenever the
configuration space has higher dimension than the observable space. This file
formalizes the dimensional argument and bridges it to the impossibility theorem.

## Main results

- `generic_underspecification`: if dim(Θ) > dim(Y), the fiber has positive
  dimension (trivial arithmetic fact).
- `fiber_nondegeneracy_implies_impossibility`: if the explanation map is
  non-constant on a fiber (two configs with the same output but incompatible
  explanations), then no explanation can be faithful, stable, and decisive.
- `ubiquity_impossibility`: constructs an ExplanationSystem from witness data
  and derives the impossibility via `explanation_impossibility`.
-/

set_option autoImplicit false

-- ============================================================================
-- §1  Dimensional argument
-- ============================================================================

/-- Generic underspecification: if the configuration space has higher
    dimension than the observable space, the system is generically
    underspecified. -/
theorem generic_underspecification {n m : Nat} (h : m < n) : n - m > 0 := by
  omega

-- ============================================================================
-- §2  Non-degeneracy → impossibility bridge
-- ============================================================================

/-- If two configurations have the same observable output but incompatible
    explanations, we can construct an ExplanationSystem with the Rashomon
    property. This bridges "underspecification is generic" to "the
    impossibility applies." -/
theorem fiber_nondegeneracy_implies_impossibility
    {Θ H Y : Type}
    (observe : Θ → Y) (explain : Θ → H) (incomp : H → H → Prop)
    (θ₁ θ₂ : Θ)
    (hobs : observe θ₁ = observe θ₂)
    (hinc : incomp (explain θ₁) (explain θ₂))
    (E : Θ → H)
    (hf : E = explain)
    (hs : ∀ a b : Θ, observe a = observe b → E a = E b)
    (hd : ∀ a b : Θ, incomp (E a) (E b) → E a ≠ E b) :
    False := by
  subst hf
  exact absurd (hs θ₁ θ₂ hobs) (hd θ₁ θ₂ hinc)

-- ============================================================================
-- §3  Neural network dimensional case (documentation)
-- ============================================================================

-- Neural network case: for a network with P parameters and k-dimensional
-- output, P >> k always. A ResNet-50 has 25M parameters and 1000 outputs.
-- The fiber dimension is ≥ 24,999,000. The explanation map (attention,
-- gradients, internal representations) is generically non-constant on
-- such high-dimensional fibers.

-- ============================================================================
-- §4  Connection to ExplanationSystem
-- ============================================================================

/-- Given concrete witness data, construct an ExplanationSystem and derive
    the impossibility. -/
theorem ubiquity_impossibility
    {Θ H Y : Type}
    (observe : Θ → Y) (explain : Θ → H) (incomp : H → H → Prop)
    (θ₁ θ₂ : Θ) (hobs : observe θ₁ = observe θ₂) (hinc : incomp (explain θ₁) (explain θ₂)) :
    let S : ExplanationSystem Θ H Y := ⟨observe, explain, incomp, θ₁, θ₂, hobs, hinc⟩
    ∀ (E : Θ → H), faithful S E → stable S E → decisive S E → False := by
  intro S E hf hs hd
  exact explanation_impossibility S E hf hs hd
