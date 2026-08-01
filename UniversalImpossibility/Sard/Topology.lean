-- Ported (with modifications) from Yury G. Kudryashov's `SardMoreira`
-- (github.com/urkud/SardMoreira @ 14bc8a1), Apache-2.0 licensed. Module paths
-- renamed and code forward-ported to this repository's Mathlib pin. See the
-- attribution and licence notices in UniversalImpossibility/Sard/ATTRIBUTION.md.
import Mathlib.Topology.NhdsWithin

open Filter
open scoped Topology

theorem eventually_nhdsWithin_nhds {X : Type*} [TopologicalSpace X] {U : Set X} (hU : IsOpen U)
    {p : X → Prop} {x : X} :
    (∀ᶠ y in 𝓝[U] x, ∀ᶠ z in 𝓝 y, p z) ↔ ∀ᶠ y in 𝓝[U] x, p y := by
  conv_rhs => rw [← eventually_eventually_nhdsWithin]
  refine eventually_congr <| eventually_mem_nhdsWithin.mono fun y hy ↦ ?_
  rw [hU.nhdsWithin_eq hy]
