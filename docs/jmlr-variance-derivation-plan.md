# JMLR Variance Derivation Plan

## Goal
Derive `consensus_variance_bound` from Mathlib's `IndepFun.variance_sum`,
replacing a domain axiom with a genuine proof from standard probability theory.

## Architecture Change

### New axioms needed (Mathlib infrastructure)
1. `MeasurableSpace Model` — sigma-algebra on Model type
2. `MeasureTheory.Measure Model` with `IsProbabilityMeasure`
3. `Measurable (attribution fs j)` — attributions are measurable
4. `MemLp (attribution fs j) 2 μ` — finite second moment
5. Independence of ensemble draws (product measure structure)

### Axioms removed
- `attribution_variance` → DEFINED as `MeasureTheory.variance`
- `attribution_variance_nonneg` → DERIVED from `variance_nonneg`
- `consensus_variance_bound` → DERIVED from `IndepFun.variance_sum`

### Net effect
- Domain axioms: 10 → 7 (remove 3)
- Infrastructure axioms: 0 → 5 (add 5)
- Total: 16 → 18 (+2), but 7 domain + 5 infra + 6 type = cleaner

## Derivation Path
1. Define `randomConsensus : (Fin M → Model) → ℝ`
2. Decompose as `(1/M) * ∑ (attribution fs j ∘ proj_i)`
3. `Var((1/M) * ∑ X_i) = (1/M²) * Var(∑ X_i)` — scaling
4. `Var(∑ X_i) = ∑ Var(X_i)` — `IndepFun.variance_sum`
5. `∑ Var(X_i) = M * Var(X_1)` — identical distribution
6. `(1/M²) * M * Var(X_1) = Var(X_1)/M` — algebra

## Files Affected
- `Defs.lean`: Major rewrite (add instances, remove 3 axioms, add 5)
- `Corollary.lean`: Proof rewrite (1 line → ~30 lines)
- `DesignSpace.lean`: Minor (statement unchanged)

## Risk: ~50% success, ~14 hours
## Timeline: Attempt after NeurIPS submission, target JMLR Q4 2026
