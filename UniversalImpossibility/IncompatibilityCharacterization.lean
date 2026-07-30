import UniversalImpossibility.Necessity
import Mathlib.Tactic
import Mathlib.Data.Fintype.Basic

/-!
# When Is Rashomon the Exact Boundary? A Structural Characterization

## Background

`ExplanationSystem.lean` proves the **universal impossibility** (`explanation_impossibility`):
under the Rashomon property, no explanation is faithful + stable + decisive.  The
converse — *"if there is no Rashomon configuration then a faithful + stable + decisive
explanation exists"* — is where the "Rashomon is the exact boundary" slogan lives.

`NecessityBiconditional.lean` proves that biconditional **only** for the special case
`incompatible = (· ≠ ·)`, and its own doc-comment exhibits a 6-configuration
counterexample showing the biconditional is **false in general**.  This file pins down
*exactly* which incompatibility relations make Rashomon the exact boundary.

## The characterization

Write `compatible a b := ¬ incompatible a b`.  Irreflexivity of `incompatible` already
makes `compatible` reflexive.  The result is:

> For an `ExplanationSetup S` whose `compatible` relation is an **equivalence relation**
> (equivalently: `incompatible` is the complement of an equivalence relation),
>   `¬ hasRashomon S  ↔  ∃ E, faithful E ∧ stable E ∧ decisive E`.

* The `(←)` direction is unconditional (contrapositive of the impossibility theorem,
  `no_rashomon_from_all_three`).
* The `(→)` direction is the substantive one and is proved here
  (`possibility_of_equiv_compat`).  It genuinely *needs* the equivalence structure:
  transitivity of `compatible` is precisely the "congruence" property that lets
  incompatibility descend to compatibility-classes, so a class-representative map is
  simultaneously stable, faithful, and decisive.

The `(· ≠ ·)` case of `NecessityBiconditional.lean` is recovered as a corollary
(`rashomon_biconditional_neq`): equality is the *finest* equivalence relation.

## Honest scope of the converse

Is the equivalence condition also *necessary* for the boundary to hold?  We do not
prove the full "for every relation that is not an equivalence there is a failing setup"
statement (it depends on the ambient `Θ, Y`).  Instead we give a concrete, fully finite
witness (`general_boundary_fails`): a setup on `H = Fin 4` whose `compatible` relation is
**not transitive** (`cex_compat_not_transitive`), for which `¬ hasRashomon` holds
(`cex_not_rashomon`) yet **no** faithful + stable + decisive explanation exists
(`cex_no_fsd`).  This shows the equivalence hypothesis in the main theorem cannot be
dropped, and reproduces the mechanism of the `NecessityBiconditional` counterexample in
a minimal 4-element form.

Namespace: `UniversalImpossibility.IncompatChar`.
-/

set_option autoImplicit false

namespace UniversalImpossibility.IncompatChar

variable {Θ : Type} {H : Type} {Y : Type}

/-! ## The compatibility setoid -/

/-- The **compatibility setoid**: `a ≈ b` means `¬ incompatible a b`.  It is a genuine
    setoid precisely when the compatibility relation is an equivalence relation, which is
    exactly the structural hypothesis of the characterization below.  (Reflexivity is
    automatic from `incompatible_irrefl`; the content is symmetry and transitivity.) -/
def compatSetoid (S : ExplanationSetup Θ H Y)
    (hequiv : Equivalence (fun a b => ¬ S.incompatible a b)) : Setoid H :=
  ⟨fun a b => ¬ S.incompatible a b, hequiv⟩

/-- Smart constructor: symmetry of `incompatible` plus transitivity of its complement
    yields the equivalence structure (reflexivity comes for free from irreflexivity). -/
def compatEquiv (S : ExplanationSetup Θ H Y)
    (hsymm : ∀ a b, S.incompatible a b → S.incompatible b a)
    (htrans : ∀ a b c, ¬ S.incompatible a b → ¬ S.incompatible b c → ¬ S.incompatible a c) :
    Equivalence (fun a b => ¬ S.incompatible a b) := by
  refine ⟨fun a => S.incompatible_irrefl a, ?_, ?_⟩
  · intro a b hab hba
    exact hab (hsymm b a hba)
  · intro a b c hab hbc
    exact htrans a b c hab hbc

/-! ## The substantive direction: equivalence structure ⇒ possibility -/

/-- **Main theorem (sufficiency).**  If the compatibility relation is an equivalence
    relation, then absence of Rashomon configurations guarantees a faithful + stable +
    decisive explanation.

    Construction: send each `θ` to a canonical representative of the compatibility-class
    of `explain θ` (via the quotient by `compatSetoid`).
    * *Stable* — `¬ hasRashomon` forces `explain θ₁` and `explain θ₂` into the same class
      whenever `observe θ₁ = observe θ₂`, so they get the same representative.
    * *Faithful* — the representative is compatible with `explain θ` (same class).
    * *Decisive* — transitivity of `compatible` means anything incompatible with
      `explain θ` is incompatible with everything in its class, in particular the
      representative. -/
theorem possibility_of_equiv_compat (S : ExplanationSetup Θ H Y)
    (hequiv : Equivalence (fun a b => ¬ S.incompatible a b))
    (hnr : ¬ hasRashomon S) :
    ∃ E : Θ → H, faithfulS S E ∧ stableS S E ∧ decisiveS S E := by
  -- ¬Rashomon ⇒ same-observation explanations are compatible
  have hfib : ∀ θ₁ θ₂, S.observe θ₁ = S.observe θ₂ →
      ¬ S.incompatible (S.explain θ₁) (S.explain θ₂) :=
    fun θ₁ θ₂ hobs hinc => hnr ⟨θ₁, θ₂, hobs, hinc⟩
  refine ⟨fun θ => Quotient.out (Quotient.mk (compatSetoid S hequiv) (S.explain θ)),
    ?_, ?_, ?_⟩
  · -- faithful
    intro θ
    have hfaith :
        ¬ S.incompatible
          (Quotient.out (Quotient.mk (compatSetoid S hequiv) (S.explain θ))) (S.explain θ) :=
      Quotient.exact (Quotient.out_eq (Quotient.mk (compatSetoid S hequiv) (S.explain θ)))
    exact hfaith
  · -- stable
    intro θ₁ θ₂ hobs
    have hcompat : ¬ S.incompatible (S.explain θ₁) (S.explain θ₂) := hfib θ₁ θ₂ hobs
    have heq : Quotient.mk (compatSetoid S hequiv) (S.explain θ₁)
             = Quotient.mk (compatSetoid S hequiv) (S.explain θ₂) := Quotient.sound hcompat
    exact congrArg Quotient.out heq
  · -- decisive
    intro θ h hinc
    by_contra hc
    have hfaith :
        ¬ S.incompatible
          (Quotient.out (Quotient.mk (compatSetoid S hequiv) (S.explain θ))) (S.explain θ) :=
      Quotient.exact (Quotient.out_eq (Quotient.mk (compatSetoid S hequiv) (S.explain θ)))
    have hsym :
        ¬ S.incompatible (S.explain θ)
          (Quotient.out (Quotient.mk (compatSetoid S hequiv) (S.explain θ))) :=
      hequiv.symm hfaith
    have hcomp : ¬ S.incompatible (S.explain θ) h := hequiv.trans hsym hc
    exact hcomp hinc

/-! ## The biconditional (Rashomon is the exact boundary) -/

/-- **The characterization.**  When the compatibility relation is an equivalence
    relation, the Rashomon property is the exact boundary between possibility and
    impossibility. -/
theorem rashomon_biconditional_of_equiv (S : ExplanationSetup Θ H Y)
    (hequiv : Equivalence (fun a b => ¬ S.incompatible a b)) :
    ¬ hasRashomon S ↔ ∃ E : Θ → H, faithfulS S E ∧ stableS S E ∧ decisiveS S E := by
  constructor
  · intro hnr
    exact possibility_of_equiv_compat S hequiv hnr
  · rintro ⟨E, hf, hs, hd⟩
    exact no_rashomon_from_all_three S E hf hs hd

/-- **Corollary — the `incompatible = (≠)` case.**  Equality is the finest equivalence
    relation, so the biconditional of `NecessityBiconditional.lean` is a special case of
    the general structural characterization. -/
theorem rashomon_biconditional_neq (S : ExplanationSetup Θ H Y)
    (h_neq : ∀ a b, S.incompatible a b ↔ a ≠ b) :
    ¬ hasRashomon S ↔ ∃ E : Θ → H, faithfulS S E ∧ stableS S E ∧ decisiveS S E := by
  have hequiv : Equivalence (fun a b => ¬ S.incompatible a b) := by
    refine ⟨fun a => S.incompatible_irrefl a, ?_, ?_⟩
    · intro a b hab hba
      exact hab ((h_neq a b).mpr (Ne.symm ((h_neq b a).mp hba)))
    · intro a b c hab hbc hac
      have hab' : a = b := not_not.mp (fun hne => hab ((h_neq a b).mpr hne))
      have hbc' : b = c := not_not.mp (fun hne => hbc ((h_neq b c).mpr hne))
      exact ((h_neq a c).mp hac) (hab'.trans hbc')
  exact rashomon_biconditional_of_equiv S hequiv

/-! ## Honest converse: the equivalence hypothesis cannot be dropped

A concrete finite witness reproducing the `NecessityBiconditional` counterexample in
minimal form.  `H = Fin 4` with the reading `0 = a, 1 = b, 2 = p, 3 = q` and the only
incompatibilities `a ⊥ p` and `b ⊥ q` (symmetrically).  One fiber `Θ = Bool` with
`explain false = a`, `explain true = b`. -/

/-- Incompatibility of the counterexample: `a ⊥ p` and `b ⊥ q` (with the symmetric
    partners), nothing else. -/
def incB (i j : Fin 4) : Bool :=
  (decide (i = 0) && decide (j = 2)) || (decide (i = 2) && decide (j = 0)) ||
  (decide (i = 1) && decide (j = 3)) || (decide (i = 3) && decide (j = 1))

/-- The counterexample setup: a single fiber whose two explanations are compatible
    (so `¬ hasRashomon`), but whose incompatibility relation is not an equivalence. -/
def cex : ExplanationSetup Bool (Fin 4) Unit where
  observe := fun _ => ()
  explain := fun b => if b then 1 else 0
  incompatible := fun i j => incB i j = true
  incompatible_irrefl := by decide

/-- The counterexample has no Rashomon configuration: the two explanations in the single
    fiber, `a` and `b`, are compatible. -/
theorem cex_not_rashomon : ¬ hasRashomon cex := by
  rintro ⟨θ₁, θ₂, _, h⟩
  cases θ₁ <;> cases θ₂ <;> simp [cex, incB] at h

/-- Its compatibility relation is **not transitive**: `a` compatible `b`, `b` compatible
    `p`, but `a` incompatible `p`.  (This is the exact failure ruled out by the main
    theorem's hypothesis.) -/
theorem cex_compat_not_transitive :
    ¬ (∀ a b c : Fin 4,
        ¬ cex.incompatible a b → ¬ cex.incompatible b c → ¬ cex.incompatible a c) := by
  intro h
  have hcontra := h 0 1 2 (by simp [cex, incB]) (by simp [cex, incB])
  simp [cex, incB] at hcontra

/-- Despite `¬ hasRashomon`, **no** faithful + stable + decisive explanation exists.
    Stability forces one value `e` on the fiber; decisiveness at the two configurations
    forces `e ⊥ p` (hence `e = a`) *and* `e ⊥ q` (hence `e = b`) — impossible. -/
theorem cex_no_fsd (E : Bool → Fin 4)
    (_hf : faithfulS cex E) (hs : stableS cex E) (hd : decisiveS cex E) : False := by
  have hconst : E true = E false := hs true false rfl
  have h2 : incB (E false) 2 = true := hd false 2 (by simp [cex, incB])
  have h3 : incB (E false) 3 = true := by
    have hh := hd true 3 (by simp [cex, incB])
    rwa [hconst] at hh
  have key : ∀ e : Fin 4, incB e 2 = true → incB e 3 = true → False := by decide
  exact key (E false) h2 h3

/-- **The general boundary claim fails.**  There is an explanation setup with no Rashomon
    configuration for which no faithful + stable + decisive explanation exists.  Hence the
    equivalence-relation hypothesis of `rashomon_biconditional_of_equiv` is essential:
    "Rashomon is the exact boundary" is a theorem about the *structure* of the
    incompatibility relation, not an unconditional fact. -/
theorem general_boundary_fails :
    ∃ (A B C : Type) (S : ExplanationSetup A B C),
      ¬ hasRashomon S ∧ ¬ ∃ E : A → B, faithfulS S E ∧ stableS S E ∧ decisiveS S E :=
  ⟨Bool, Fin 4, Unit, cex, cex_not_rashomon,
    fun ⟨E, hf, hs, hd⟩ => cex_no_fsd E hf hs hd⟩

end UniversalImpossibility.IncompatChar
