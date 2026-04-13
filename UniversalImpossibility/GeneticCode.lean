import UniversalImpossibility.ExplanationSystem

set_option autoImplicit false

/-!
# Genetic Code Degeneracy — Derived Rashomon Property

The genetic code is degenerate: multiple codons encode the same amino acid.
For example, UCU and UCC both encode Serine. This is a natural instance of
the Rashomon property: two different configurations (codons) produce the
same observable output (amino acid) but have incompatible explanations
(different nucleotide sequences).
-/

/-- RNA codons. We include at least the two Serine codons needed for
    the degeneracy witness. -/
inductive Codon where
  | UCU : Codon
  | UCC : Codon
  deriving DecidableEq, Repr

/-- Amino acids. We include at least Serine. -/
inductive AminoAcid where
  | Ser : AminoAcid
  deriving DecidableEq, Repr

/-- The translation map: codon → amino acid.
    Both UCU and UCC map to Serine. -/
def translate : Codon → AminoAcid
  | Codon.UCU => AminoAcid.Ser
  | Codon.UCC => AminoAcid.Ser

/-- Degeneracy: UCU and UCC encode the same amino acid. -/
theorem codon_degeneracy : translate Codon.UCU = translate Codon.UCC := by
  decide

/-- UCU and UCC are different codons. -/
theorem codons_different : Codon.UCU ≠ Codon.UCC := by
  decide

/-- The genetic code as an ExplanationSystem.
    - Θ = Codon (configurations)
    - H = Codon (explanations: the codon itself explains the translation)
    - Y = AminoAcid (observables)
    - observe = translate
    - explain = id (the codon is its own explanation)
    - incompatible = (≠) -/
def geneticCodeSystem : ExplanationSystem Codon Codon AminoAcid where
  observe := translate
  explain := id
  incompatible := fun c₁ c₂ => c₁ ≠ c₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨Codon.UCU, Codon.UCC, codon_degeneracy, codons_different⟩

/-- **Genetic Code Impossibility.**
    No explanation of the genetic code can be simultaneously faithful,
    stable, and decisive when degeneracy holds. -/
theorem genetic_code_impossibility
    (E : Codon → Codon)
    (hf : faithful geneticCodeSystem E)
    (hs : stable geneticCodeSystem E)
    (hd : decisive geneticCodeSystem E) : False :=
  explanation_impossibility geneticCodeSystem E hf hs hd
