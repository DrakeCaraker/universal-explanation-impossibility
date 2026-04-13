import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Phase Problem — Derived Rashomon Property

In crystallography, the diffraction pattern measures |F(k)|², the squared
magnitude of the structure factor. Multiple electron density distributions
produce the same diffraction pattern. This is the "phase problem" — the
central challenge of X-ray crystallography (Hauptman & Karle 1953;
Karle & Hauptman 1956; Nobel Prize 1985).

We constructively prove that the Rashomon property holds for the phase
problem by exhibiting two 2-element signals with the same energy
(sum of squares) but different values.

The key witness: signals (1, 0) and (0, 1) both have energy 1.
-/

/-- A 2-element signal, modeling the simplest case of the phase problem.
    In the full crystallographic setting, the signal is the electron density
    ρ(x) sampled at discrete points, and the observable is |F(k)|². -/
structure Signal2 where
  a : Int
  b : Int
  deriving DecidableEq, Repr

/-- Energy of a 2-element signal: a² + b².
    This models |F(k)|² — the squared magnitude of the Fourier transform,
    which is the only quantity measurable from diffraction data. -/
def energy (s : Signal2) : Int :=
  s.a * s.a + s.b * s.b

/-- First witness signal: (1, 0). -/
def sig1 : Signal2 := ⟨1, 0⟩

/-- Second witness signal: (0, 1). -/
def sig2 : Signal2 := ⟨0, 1⟩

/-- The two signals have the same energy: 1² + 0² = 0² + 1² = 1. -/
theorem same_energy : energy sig1 = energy sig2 := by native_decide

/-- The two signals are distinct. -/
theorem different_signals : sig1 ≠ sig2 := by decide

/-- The Rashomon property for the phase problem: DERIVED, not axiomatized.
    Two signals exist with the same energy but different values. -/
theorem phase_rashomon_derived :
    ∃ (s₁ s₂ : Signal2), energy s₁ = energy s₂ ∧ s₁ ≠ s₂ :=
  ⟨sig1, sig2, same_energy, different_signals⟩

/-- Construct an ExplanationSystem for the phase problem with
    DERIVED Rashomon property.
    - Θ = Signal2 (the signal / electron density)
    - H = Signal2 (the "explanation" is the signal itself)
    - Y = Int (the observable is the energy / diffraction intensity)
    - observe = energy
    - explain = id (the true signal is its own explanation)
    - incompatible = (≠)

    NOTE: incompatible = (≠) is intentionally broad — it makes the
    impossibility STRONGER (holds even under this liberal notion of
    conflict). -/
def phaseSystem : ExplanationSystem Signal2 Signal2 Int where
  observe := energy
  explain := id
  incompatible := fun s₁ s₂ => s₁ ≠ s₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨sig1, sig2, same_energy, different_signals⟩

/-- Phase problem impossibility: no explanation of a signal from its
    energy can be simultaneously faithful, stable, and decisive.

    This is the crystallographic phase problem cast as an instance of
    the universal explanation impossibility. -/
theorem phase_impossibility
    (E : Signal2 → Signal2)
    (hf : faithful phaseSystem E)
    (hs : stable phaseSystem E)
    (hd : decisive phaseSystem E) : False :=
  explanation_impossibility phaseSystem E hf hs hd
