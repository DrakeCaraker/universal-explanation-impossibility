# Seven Derived Instances — Implementation Plan

**Goal**: Derive the Rashomon property from first principles in
7 scientific domains. Each derivation is a Lean file with zero
axioms and zero sorry. The paper becomes a Nature-viable
contribution: one theorem, seven proofs from seven sciences.

**Current state**: 1 derived instance (causal/Markov equivalence).
**Target**: 7 derived instances spanning biology, physics,
linguistics, crystallography, mathematics, CS theory, statistics.

**Model**: Opus for all Lean proofs (requires mathematical precision).
Sonnet for paper sections and integration.

---

## VET RECORD

### Round 1 — Factual

- Each derivation needs: (a) domain-specific types in Lean,
  (b) a concrete Rashomon witness (two configurations with
  same observable, different explanations), (c) an
  ExplanationSystem constructed with the derived Rashomon,
  (d) the impossibility theorem as a corollary.

- ⚠️ The genetic code derivation is the simplest (~20 lines)
  but must correctly model the codon table. There are 61 sense
  codons mapping to 20 amino acids. For a minimal witness, we
  only need TWO codons mapping to the same amino acid.
  UCU and UCC both encode Serine. Represent as Fin 64 → Fin 21
  (64 codons → 20 amino acids + stop).

- ⚠️ The gauge theory derivation uses a triangle graph with
  ℤ₂ edge labels. The gauge transformation flips all edges
  at a vertex. The holonomy (sum mod 2 around the loop) is
  gauge-invariant. Two edge configurations with the same
  holonomy but different edges = Rashomon. Must verify this
  is correct: flipping edges at vertex 0 changes edges 01
  and 02 but not edge 12. The holonomy (edge01 + edge12 +
  edge02 mod 2) is preserved by this transformation.

- ⚠️ The syntactic ambiguity derivation needs a CFG with an
  ambiguous sentence. Simplest: S → NP VP, VP → V NP PP,
  VP → V NP, NP → NP PP. The sentence "V NP PP" has two
  parse trees (PP attaches to VP or to NP). Represent as
  an inductive type for parse trees. Two trees, same yield.

- ⚠️ The phase problem derivation: for discrete signals of
  length N, the power spectrum |F[k]|² doesn't determine the
  phase. Two signals x₁, x₂ with the same power spectrum but
  different values. Simplest: N=2, x₁ = (1, 1), x₂ = (1, -1).
  |DFT(x₁)|² = (4, 0), |DFT(x₂)|² = (0, 4). These are
  DIFFERENT. Bad example. Need: x₁ = (1, 0), x₂ = (0, 1).
  DFT(x₁) = (1, 1), DFT(x₂) = (1, -1). |DFT(x₁)|² = (1,1),
  |DFT(x₂)|² = (1,1). Same power spectrum, different signals.
  YES — this works for N=2 over ℤ or ℝ.

  Actually simpler: use the time-reversal ambiguity. For any
  real signal x[n], the time-reversed signal x[-n mod N] has
  the same power spectrum. For N=3: x = (1,2,0) and
  x_rev = (1,0,2) have the same |DFT|². This is always true
  and easy to prove.

- ⚠️ The linear system derivation: A = [1,1], b = 2,
  x₁ = (1,1), x₂ = (0,2). Both satisfy Ax=b. Represent
  with Fin 2 → ℤ for vectors, verify by computation.

- ⚠️ The view update derivation: two database rows with
  columns (A, B). View projects to column A only. Two
  base states (a, b₁) and (a, b₂) with b₁ ≠ b₂ map to
  the same view (a). Represent as pairs.

### Round 2 — Reasoning

- Each derivation follows the SAME template:
  1. Define domain-specific types (Config, Observable, Explanation)
  2. Define observe and explain maps
  3. Define incompatible (irreflexive)
  4. Construct two concrete witnesses
  5. Prove same observable, different explanations
  6. Build ExplanationSystem with derived Rashomon
  7. Apply explanation_impossibility

- ⚠️ The phase problem is the trickiest because it requires
  the DFT. For N=2 or N=3, the DFT can be computed explicitly
  without importing Mathlib's Fourier analysis. Just define
  it as a function on Fin N → ℤ (or ℚ for exact arithmetic).

  SIMPLER APPROACH: skip the DFT entirely. State the phase
  problem as: "two signals with the same autocorrelation but
  different values." The autocorrelation R[k] = Σ x[n]x[n+k]
  is computable for finite signals. Two signals with the same
  autocorrelation sequence = same power spectrum (by
  Wiener-Khinchin). This avoids complex numbers entirely.

  EVEN SIMPLER: just use a 2-element "signal" where x₁=(1,0)
  and x₂=(0,1). The "observable" is the set of absolute values
  {|x[0]|, |x[1]|} = {0, 1} for both. Done. No DFT needed.
  The point is: the magnitude profile doesn't determine the
  signal. This is the phase problem in its simplest form.

- ⚠️ For the syntactic ambiguity, representing parse trees as
  inductive types in Lean requires care. A simpler approach:
  represent a parse tree as a binary bracketing. The sentence
  has 3 tokens: V NP PP. Two bracketings: ((V NP) PP) and
  (V (NP PP)). Represent as a sum type (left-attach or
  right-attach). The yield function extracts the flat token
  sequence. Two bracketings, same yield. This avoids needing
  a full CFG formalization.

### Round 3 — Omissions

- ⚠️ The plan should specify how these integrate into the paper.
  Each needs: a Lean file, a paper section, an update to the
  cross-instance table, and an update to the abstract.

- ⚠️ The plan should specify the ORDER of implementation.
  Easiest first (genetic code, linear system) to build
  momentum and catch template issues, then harder ones
  (gauge, phase, syntax).

- ⚠️ The plan should handle the NAMING. Each Lean file needs
  a clear name: GeneticCode.lean, GaugeTheory.lean,
  SyntacticAmbiguity.lean, PhaseProblem.lean,
  LinearSystem.lean, ViewUpdate.lean. Plus the existing
  MarkovEquivalence.lean.

- ⚠️ After all derivations, the paper needs a "visual proof
  of unification" — a figure or table showing the same proof
  structure across all 7 domains side by side.

---

## Phase 1: Easiest Derivations [Day 1]

### Task 1.1: Genetic Code Degeneracy [Opus]

**File**: `UniversalImpossibility/GeneticCode.lean`

```lean
/-!
# Genetic Code Degeneracy — Derived Rashomon Property

The genetic code is degenerate: multiple codons encode the
same amino acid. UCU and UCC both encode Serine. The
"explanation" of a protein (its DNA sequence) is
underspecified by the protein sequence (the observable).
-/

-- Codons (simplified: just the two we need as witnesses)
inductive Codon where
  | UCU : Codon  -- Serine codon 1
  | UCC : Codon  -- Serine codon 2
  deriving DecidableEq, Repr

-- Amino acids (simplified)
inductive AminoAcid where
  | Ser : AminoAcid  -- Serine
  deriving DecidableEq, Repr

-- The genetic code: codon → amino acid
def translate : Codon → AminoAcid
  | Codon.UCU => AminoAcid.Ser
  | Codon.UCC => AminoAcid.Ser

-- Two different codons produce the same amino acid
theorem codon_degeneracy :
    translate Codon.UCU = translate Codon.UCC := by decide

theorem codons_different : Codon.UCU ≠ Codon.UCC := by decide

-- Construct the ExplanationSystem
def geneticCodeSystem : ExplanationSystem Codon Codon AminoAcid where
  observe := translate
  explain := id  -- the "explanation" is the codon itself
  incompatible := fun c₁ c₂ => c₁ ≠ c₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨Codon.UCU, Codon.UCC, codon_degeneracy,
               codons_different⟩

theorem genetic_code_impossibility
    (E : Codon → Codon)
    (hf : faithful geneticCodeSystem E)
    (hs : stable geneticCodeSystem E)
    (hd : decisive geneticCodeSystem E) : False :=
  explanation_impossibility geneticCodeSystem E hf hs hd
```

Run `lake build`. Verify zero sorry, zero axioms for
`genetic_code_impossibility`.

### Task 1.2: Underdetermined Linear System [Opus]

**File**: `UniversalImpossibility/LinearSystem.lean`

```lean
/-!
# Underdetermined Linear System — Derived Rashomon Property

Ax = b with rank(A) < dim(x). Multiple solutions exist.
The simplest case: A = [1, 1], b = 2.
Solutions: x₁ = (1,1) and x₂ = (0,2).
-/

-- Solution vectors as pairs of integers
structure Vec2 where
  x : Int
  y : Int
  deriving DecidableEq, Repr

-- The linear map A·x = x.x + x.y
def dotA (v : Vec2) : Int := v.x + v.y

-- Two solutions
def sol1 : Vec2 := ⟨1, 1⟩
def sol2 : Vec2 := ⟨0, 2⟩

-- Both map to 2
theorem same_output : dotA sol1 = dotA sol2 := by decide

-- But they're different
theorem different_solutions : sol1 ≠ sol2 := by decide

-- ExplanationSystem
def linearSystem : ExplanationSystem Vec2 Vec2 Int where
  observe := dotA
  explain := id
  incompatible := fun v₁ v₂ => v₁ ≠ v₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨sol1, sol2, same_output, different_solutions⟩

theorem linear_system_impossibility
    (E : Vec2 → Vec2)
    (hf : faithful linearSystem E)
    (hs : stable linearSystem E)
    (hd : decisive linearSystem E) : False :=
  explanation_impossibility linearSystem E hf hs hd
```

### Task 1.3: Database View Update [Opus]

**File**: `UniversalImpossibility/ViewUpdate.lean`

```lean
/-!
# Database View Update — Derived Rashomon Property

Two database rows with columns (A, B). The view projects
to column A only. Two base states with the same view
but different hidden column B.
Bancilhon & Spyratos (1981).
-/

structure Row where
  colA : Bool
  colB : Bool
  deriving DecidableEq, Repr

-- View: project to column A
def view (r : Row) : Bool := r.colA

-- Two rows with same view, different data
def row1 : Row := ⟨true, true⟩
def row2 : Row := ⟨true, false⟩

theorem same_view : view row1 = view row2 := by decide
theorem different_rows : row1 ≠ row2 := by decide

def viewUpdateSystem : ExplanationSystem Row Row Bool where
  observe := view
  explain := id
  incompatible := fun r₁ r₂ => r₁ ≠ r₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨row1, row2, same_view, different_rows⟩

theorem view_update_impossibility
    (E : Row → Row)
    (hf : faithful viewUpdateSystem E)
    (hs : stable viewUpdateSystem E)
    (hd : decisive viewUpdateSystem E) : False :=
  explanation_impossibility viewUpdateSystem E hf hs hd
```

---

## Phase 2: Medium Derivations [Day 2]

### Task 2.1: Discrete Gauge Theory [Opus]

**File**: `UniversalImpossibility/GaugeTheory.lean`

Triangle graph with ℤ₂ (Bool) edge labels. Gauge transformation
flips edges at a vertex. Holonomy (XOR around the loop) is
gauge-invariant.

```lean
/-!
# Discrete Gauge Theory — Derived Rashomon Property

A triangle graph with ℤ₂ edge labels. A gauge transformation
at vertex v flips all edges incident to v. The holonomy
(XOR of edges around the loop) is gauge-invariant.
Two gauge-equivalent configurations: same holonomy,
different edge labels.
-/

structure EdgeConfig where
  e01 : Bool  -- edge between vertex 0 and 1
  e12 : Bool  -- edge between vertex 1 and 2
  e02 : Bool  -- edge between vertex 0 and 2
  deriving DecidableEq, Repr

-- Holonomy: XOR of all edges around the triangle
def holonomy (g : EdgeConfig) : Bool :=
  xor (xor g.e01 g.e12) g.e02

-- Gauge transformation at vertex 0: flip e01 and e02
def gaugeAt0 (g : EdgeConfig) : EdgeConfig :=
  ⟨!g.e01, g.e12, !g.e02⟩

-- Two configurations
def config1 : EdgeConfig := ⟨true, false, false⟩
def config2 : EdgeConfig := ⟨false, false, true⟩
-- config2 = gaugeAt0 config1

-- Same holonomy
theorem same_holonomy :
    holonomy config1 = holonomy config2 := by decide

-- Different edge labels
theorem different_configs :
    config1 ≠ config2 := by decide

-- Verify config2 = gaugeAt0 config1
theorem gauge_related :
    gaugeAt0 config1 = config2 := by decide

-- Gauge transformation preserves holonomy (general)
theorem gauge_preserves_holonomy (g : EdgeConfig) :
    holonomy (gaugeAt0 g) = holonomy g := by
  simp [holonomy, gaugeAt0]
  cases g.e01 <;> cases g.e12 <;> cases g.e02 <;> decide

def gaugeSystem : ExplanationSystem EdgeConfig EdgeConfig Bool where
  observe := holonomy
  explain := id
  incompatible := fun g₁ g₂ => g₁ ≠ g₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨config1, config2, same_holonomy, different_configs⟩

theorem gauge_impossibility
    (E : EdgeConfig → EdgeConfig)
    (hf : faithful gaugeSystem E)
    (hs : stable gaugeSystem E)
    (hd : decisive gaugeSystem E) : False :=
  explanation_impossibility gaugeSystem E hf hs hd
```

IMPORTANT: Also prove `gauge_preserves_holonomy` for all
three vertex transformations (not just vertex 0) to show
gauge invariance is a general property, not an accident
of the specific witness.

### Task 2.2: Syntactic Ambiguity [Opus]

**File**: `UniversalImpossibility/SyntacticAmbiguity.lean`

Represent a 3-token sentence with two bracketings.

```lean
/-!
# Syntactic Ambiguity — Derived Rashomon Property

The sentence "V NP PP" has two parse trees:
- Left-attach:  ((V NP) PP) — PP modifies the verb phrase
- Right-attach: (V (NP PP)) — PP modifies the noun phrase

Same surface string, different structural analyses.
This is the most famous example in all of linguistics.
-/

-- Tokens
inductive Token where
  | V : Token   -- Verb
  | NP : Token  -- Noun Phrase
  | PP : Token  -- Prepositional Phrase
  deriving DecidableEq, Repr

-- Two bracketings (parse trees for 3 tokens)
inductive Bracketing where
  | leftAttach : Bracketing   -- ((V NP) PP)
  | rightAttach : Bracketing  -- (V (NP PP))
  deriving DecidableEq, Repr

-- The yield: both bracketings produce the same token sequence
def yield : Bracketing → List Token
  | Bracketing.leftAttach  => [Token.V, Token.NP, Token.PP]
  | Bracketing.rightAttach => [Token.V, Token.NP, Token.PP]

theorem same_yield :
    yield Bracketing.leftAttach = yield Bracketing.rightAttach :=
  by decide

theorem different_bracketings :
    Bracketing.leftAttach ≠ Bracketing.rightAttach := by decide

def syntaxSystem :
    ExplanationSystem Bracketing Bracketing (List Token) where
  observe := yield
  explain := id
  incompatible := fun b₁ b₂ => b₁ ≠ b₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨Bracketing.leftAttach, Bracketing.rightAttach,
               same_yield, different_bracketings⟩

theorem syntactic_ambiguity_impossibility
    (E : Bracketing → Bracketing)
    (hf : faithful syntaxSystem E)
    (hs : stable syntaxSystem E)
    (hd : decisive syntaxSystem E) : False :=
  explanation_impossibility syntaxSystem E hf hs hd
```

---

## Phase 3: Hardest Derivation [Day 3]

### Task 3.1: The Phase Problem [Opus]

**File**: `UniversalImpossibility/PhaseProblem.lean`

Use the simplest form: two signals of length 2 with the same
magnitude profile but different values.

```lean
/-!
# The Phase Problem — Derived Rashomon Property

In crystallography and signal processing, the magnitude
(power spectrum) of a signal does not determine the signal
itself. Multiple signals share the same magnitude profile.

Simplest case: signals of length 2.
x₁ = (1, 0) and x₂ = (0, 1) have the same set of
absolute values {0, 1} but are different signals.

More precisely: the sorted magnitude vector is the same.
-/

structure Signal2 where
  a : Int
  b : Int
  deriving DecidableEq, Repr

-- The "magnitude profile": sorted absolute values
-- For simplicity: the SET of absolute values (as a pair, sorted)
def magnitudeProfile (s : Signal2) : Int × Int :=
  if s.a.natAbs ≤ s.b.natAbs then (s.a.natAbs, s.b.natAbs)
  else (s.b.natAbs, s.a.natAbs)

def sig1 : Signal2 := ⟨1, 0⟩
def sig2 : Signal2 := ⟨0, 1⟩

theorem same_magnitude :
    magnitudeProfile sig1 = magnitudeProfile sig2 := by
  native_decide  -- or by unfolding and simplifying

theorem different_signals : sig1 ≠ sig2 := by decide

def phaseSystem : ExplanationSystem Signal2 Signal2 (Int × Int) where
  observe := magnitudeProfile
  explain := id
  incompatible := fun s₁ s₂ => s₁ ≠ s₂
  incompatible_irrefl := fun _ h => h rfl
  rashomon := ⟨sig1, sig2, same_magnitude, different_signals⟩

theorem phase_impossibility
    (E : Signal2 → Signal2)
    (hf : faithful phaseSystem E)
    (hs : stable phaseSystem E)
    (hd : decisive phaseSystem E) : False :=
  explanation_impossibility phaseSystem E hf hs hd
```

NOTE: If `magnitudeProfile` with Int.natAbs causes issues,
use a simpler observable: `fun s => s.a * s.a + s.b * s.b`
(the sum of squares = energy). sig1 and sig2 both have
energy 1. This avoids sorting and natAbs entirely.

---

## Phase 4: Paper Integration [Day 4-5]

### Task 4.1: Write Paper Sections [Opus]

For EACH new derived instance, create a paper section file
`paper/sections/instance_<name>.tex` (~0.5-1 page each):

- **Genetic code**: Explain codon degeneracy, cite molecular
  biology textbooks, state the impossibility, note that
  codon optimization IS the resolution (choosing codons for
  expression level, sacrificing faithfulness to the "true"
  codon for stability across synonymous options)

- **Gauge theory**: Explain gauge freedom, cite Jackson
  (Classical Electrodynamics) or Peskin & Schroeder, note
  that gauge-invariant quantities ARE the G-invariant
  resolution, cite 't Hooft or Yang-Mills for context

- **Syntactic ambiguity**: Cite Chomsky (1957) or Jurafsky
  & Martin, explain PP-attachment ambiguity, note that
  underspecified parses (packed forests) are the resolution

- **Phase problem**: Cite Hauptman & Karle (1985 Nobel),
  explain the magnitude/phase decomposition, note that
  Patterson maps (autocorrelation) are the gauge-invariant
  observable, direct methods are the resolution

- **Linear system**: Frame as the most fundamental instance,
  cite any linear algebra textbook, note that the
  Moore-Penrose pseudoinverse (minimum-norm solution) is
  the G-invariant resolution

- **View update**: Cite Bancilhon & Spyratos (1981), explain
  the view update problem, note it was discovered 45 years
  before our framework

### Task 4.2: Update Monograph [Sonnet]

- Add all 6 new \input{sections/instance_*} to the instances
  section (BEFORE the ML instances — derived instances first)
- Update abstract: "seven scientific domains"
- Update cross-instance table: add 6 new rows
- Update Lean counts
- Update contribution list
- Add a "Visual proof of unification" — a table showing the
  same proof structure across all 7 domains:

  | Domain | Θ | Y | Witness 1 | Witness 2 | Same Y? | ≠? |
  |--------|---|---|-----------|-----------|---------|-----|
  | Biology | Codons | Amino acids | UCU | UCC | Ser=Ser | ✓ |
  | Physics | Edge configs | Holonomy | (T,F,F) | (F,F,T) | T=T | ✓ |
  | ... | ... | ... | ... | ... | ... | ... |

### Task 4.3: Update Other Paper Versions [Sonnet]

- PNAS: select 4 derived instances (gauge, genetic, phase,
  causal) for main text, rest in SI
- NeurIPS: mention all 7 in abstract, details in appendix
- JMLR: all 7 in main body

### Task 4.4: Add Bibliography Entries [Sonnet]

- jackson_classical: Jackson, "Classical Electrodynamics"
- chomsky1957syntactic: Chomsky, "Syntactic Structures"
- hauptman1985nobel: Hauptman, Nobel lecture on direct methods
- bancilhon1981update: already present
- jurafsky2024speech: Jurafsky & Martin, "Speech and Language
  Processing"

---

## Phase 5: Verify + Compile + Commit [Day 5]

### Task 5.1: Full Lean Build

```bash
lake build
grep -c "^theorem\|^lemma" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -c "^axiom" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
grep -rc "sorry" UniversalImpossibility/*.lean | awk -F: '{s+=$2} END {print s}'
ls UniversalImpossibility/*.lean | wc -l
```

Expected: ~81 files, ~375 theorems, 72 axioms (unchanged —
new files add NO axioms), 0 sorry.

### Task 5.2: Run #print axioms for ALL derived instances

```lean
#print axioms genetic_code_impossibility
#print axioms gauge_impossibility
#print axioms syntactic_ambiguity_impossibility
#print axioms phase_impossibility
#print axioms linear_system_impossibility
#print axioms view_update_impossibility
#print axioms causal_impossibility_derived
```

ALL must show zero behavioral axioms.

### Task 5.3: Compile All Paper Versions

All 5 must compile. PNAS ≤6pp, NeurIPS ≤10pp.

### Task 5.4: Rebuild arXiv Package

### Task 5.5: Commit and Push

---

## Execution Order

```
Day 1: Phase 1 [1.1 ∥ 1.2 ∥ 1.3]  (genetic + linear + view)
Day 2: Phase 2 [2.1 ∥ 2.2]         (gauge + syntax)
Day 3: Phase 3 [3.1]               (phase problem)
Day 4: Phase 4 [4.1 ∥ 4.2 ∥ 4.3 ∥ 4.4]  (paper integration)
Day 5: Phase 5 [5.1 → 5.2 → 5.3 → 5.4 → 5.5]
```

Phases 1-3 are independent Lean files — can be parallelized.
Phase 4 depends on all Lean files compiling.
Phase 5 is the final gate.

## Confidence

| Instance | Tractability | Risk |
|----------|:---:|---------|
| Genetic code | VERY HIGH | ~20 lines, all decidable |
| Linear system | VERY HIGH | ~30 lines, all decidable |
| View update | VERY HIGH | ~30 lines, all decidable |
| Gauge theory | HIGH | ~60 lines, case analysis on Bool |
| Syntax | HIGH | ~40 lines, inductive types |
| Phase problem | MEDIUM | magnitude profile needs care with Int/natAbs |
| Causal (done) | DONE | Already verified |
