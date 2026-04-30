# Naming Conventions — The Limits of Explanation

**Established:** 2026-04-30
**Status:** Canonical. All papers, docs, and code should use these names.

## The Table

| Concept | Name | Shannon/Physics Analog |
|---|---|---|
| F+S+D impossible | The explanation trilemma | — |
| F+S impossible (binary) | The explanation bilemma | — |
| C = dim(V^G) | Explanation capacity | Channel capacity |
| η = 1 − C/dim(V) | Explanation loss rate | — |
| Capacity predicts instability | Explanation Capacity Theorem | Shannon's theorem |
| unfaith₁+unfaith₂ ≥ Δ−δ | Explanation uncertainty bound | Donoho–Stark uncertainty |
| Orbit average / DASH / character | The stable projection | Optimal channel code |
| Which pairs survive | Tightness (full / collapsed) | — |
| No neutral element | Explanation conflict | Noise (channel impairment) |
| Add neutral element | Enrichment | Alphabet extension |
| g(g−1)/2 stable, rest coin flips | Stable fact count | — |
| ‖v−Rv‖² + ‖Rv‖² = ‖v‖² | Explanatory information loss | Quantum information loss |
| ‖w‖ ≤ ‖u−w‖ for Rw=0, Ru=u | Over-explanation penalty | Strong converse |
| MSE = tr(RΣR)/M | Explanation stability convergence rate | Source coding rate |
| M*(ε) = tr(RΣR)/ε | Stability threshold | Block length |
| The 4-part theorem | Explanation Stability Theorem | Shannon's coding theorem |

## Usage Notes

- **"The stable projection"** is the primary term for the orbit average / Reynolds projection in prose. Define on first use: "the stable projection (the orbit-averaged map, also known as the Reynolds projection)."
- **"Explanation uncertainty bound"** replaces the earlier "explanation tradeoff bound" and the original "explanation uncertainty relation." The word "bound" (not "relation") distinguishes it from Heisenberg.
- **"Over-explanation penalty"** replaces "beyond-capacity penalty." Self-describing without requiring the reader to know what "capacity" means first.
- **"Explanatory information loss"** replaces "Pythagorean decomposition" as the prose name. The Pythagorean identity is the mathematical content; the name emphasizes what the scientist cares about.
- **"Explanation Stability Theorem"** replaces "Explanation Coding Theorem." Uses the framework's own vocabulary instead of Shannon's.
- **"Explanation Capacity Theorem"** replaces "Explanation Capacity Law." "Theorem" is accurate (it's proved); "law" overclaimed.
- **"Stable fact count"** replaces "Noether counting law." Self-describing. "Facts" generalizes beyond features/variables to any pairwise assertion.
- **"Explanation stability convergence rate"** scopes the generic "convergence rate" to this framework.

## Names NOT to use

| Deprecated | Replacement | Reason |
|---|---|---|
| The explanation code | The stable projection | "Code" imports Shannon vocabulary that doesn't apply |
| Explanation Capacity Law | Explanation Capacity Theorem | "Law" overclaims; it's a proved theorem |
| Noether counting law | Stable fact count | Requires knowing Noether; not self-describing |
| Explanation uncertainty relation | Explanation uncertainty bound | "Relation" invokes Heisenberg; "bound" is accurate |
| Explanation tradeoff bound | Explanation uncertainty bound | Inconsistent with Donoho-Stark lineage |
| Pythagorean decomposition | Explanatory information loss | Math name vs scientist-facing name |
| Beyond-capacity penalty | Over-explanation penalty | Requires knowing "capacity" first |
| Explanation Coding Theorem | Explanation Stability Theorem | "Coding" imports Shannon vocabulary |
| η law | Explanation Capacity Theorem | Unnamed symbol + overclaiming "law" |
