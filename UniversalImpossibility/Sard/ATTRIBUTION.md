# Attribution: SardMoreira port

This directory is a port of Yury G. Kudryashov's formalization of **Moreira's version of
Sard's theorem** (C^{k,α} maps between finite-dimensional real normed spaces):

- Upstream software: https://github.com/urkud/SardMoreira @ `14bc8a1eeaedb14f9ae95e125c95a5eb4f47f8c5`
- Copyright holder / author of the ported development: **Yury G. Kudryashov**
- License: **Apache License 2.0** (full text in `LICENSE.txt` in this directory). Kudryashov's
  original per-file copyright headers are retained on the files that carried them
  (`ToMathlib/*.lean`); every ported file additionally carries a top-of-file notice recording
  that it was ported from `urkud/SardMoreira` and **modified** (Apache-2.0 §4(b)), pointing here.
- Mathematical source: **C. G. T. de A. Moreira**, *Hausdorff measures and the Morse–Sard
  theorem*, Publicacions Matemàtiques **45** (2001), no. 1, 149–162.

Both works are cited in the monograph bibliography (`paper/references.bib`:
`kudryashov2025sardmoreira` for the formalization, `moreira2001hausdorff` for the theorem).

## Apache-2.0 compliance summary

This is a redistribution of a modified copy of Apache-2.0-licensed source. Per §4 we: (a) include
the License (`LICENSE.txt`); (b) mark every modified file with a prominent notice of change (the
top-of-file provenance line + this document); (c) retain all copyright, attribution, and licence
notices present in the source (nothing was stripped — the upstream core files carried no per-file
header, and the `ToMathlib/*.lean` headers are preserved verbatim); (d) the upstream carried no
`NOTICE` file. The ported development is **not** presented as this project's own work: it is
excluded from the repository's theorem/file counts and is credited to Kudryashov throughout.

## What was changed in the port

- Module names: `SardMoreira.X` → `UniversalImpossibility.Sard.X` (import lines only;
  Lean namespaces, declarations, and proofs are upstream's).
- Upstream pins Mathlib `f160a5e2` (2025-12-27); this repo pins `92e168a21c` (2026-04-08).
  Files were forward-ported across that drift; substantive changes, if any, are recorded
  in the git history of this directory (the initial commit is the verbatim port).
- Dropped upstream files whose contents merged into Mathlib before our pin (verified
  present in `92e168a21c`): `ToMathlib/PR31960.lean` (→ `HaarToSphere.lean`),
  `ToMathlib/PR32986.lean`, `ToMathlib/PR32993.lean` (→ `ENNReal.div_right_comm`),
  `ToMathlib/PR33029.lean` (→ `Measure/Prod.lean` doubling instances), and `Unused.lean`.
- Kept (not yet upstream at our pin): `ToMathlib/ContinuousLinearMap.lean`,
  `ToMathlib/PR32186.lean`, `ToMathlib/PR33114.lean`.

## Why it is vendored rather than a lake dependency

Two Lean projects must share one Mathlib. Upstream's pin is older than ours, so taking
it as a dependency would force a Mathlib downgrade (or a joint bump). Vendoring the
sources forward against our pin keeps this repo's 600+ theorems on their verified pin.
The long-term intent (shared with upstream) is Mathlib upstreaming, after which this
directory is deleted and the bridge in `UniversalImpossibility/MorseSard.lean` retargets
the Mathlib declarations.
