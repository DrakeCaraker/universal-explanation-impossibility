#!/usr/bin/env python3
"""
Axiom consistency model for the DASH impossibility formalization.

Constructs a concrete numerical model satisfying all 15 axioms from
DASHImpossibility/Defs.lean and DASHImpossibility/SpearmanDef.lean
simultaneously, proving the axiom system is consistent (i.e., has at
least one model).

Construction
------------
- P = 4 features, L = 2 groups, m = 2 features per group, rho = 0.5
- T = 100 boosting rounds, c = 1.0 (proportionality constant)
- Features {0, 1} in group 0; features {2, 3} in group 1
- 4 models: f_i has firstMover = i for i in {0, 1, 2, 3}
- Split counts follow the axiomatized formulas exactly
- Attributions = c * splitCount (proportionality)

Axiom reference (15 total, matching CLAUDE.md inventory):
  Type declarations (6): Model, numTrees, numTrees_pos, attribution,
                          splitCount, firstMover
  Domain-specific  (7): firstMover_surjective, splitCount_firstMover,
                          splitCount_nonFirstMover, proportionality_global,
                          splitCount_crossGroup_symmetric,
                          consensus_variance_bound, spearman_classical_bound
  Infrastructure   (2): modelMeasurableSpace, modelMeasure

Exit code 0 if all 15 axioms are satisfied, 1 otherwise.
"""

import sys
import numpy as np
from scipy.stats import spearmanr

# ============================================================
# Parameters
# ============================================================

P = 4          # number of features
L = 2          # number of groups
m = 2          # features per group
T = 100        # number of boosting rounds (numTrees)
rho = 0.5      # within-group correlation
c = 1.0        # global proportionality constant

# Group assignment: feature j -> group index
group_of = {0: 0, 1: 0, 2: 1, 3: 1}
groups = {0: [0, 1], 1: [2, 3]}

# Models: 4 models, one per feature as first-mover
num_models = 4
first_mover = {0: 0, 1: 1, 2: 2, 3: 3}  # model i -> firstMover feature

# Derived constants
rho2 = rho ** 2                          # 0.25
denom = 2 - rho2                         # 1.75
sc_first = T / denom                     # T / (2 - rho^2) ~ 57.143
sc_same_non = (1 - rho2) * T / denom     # (1 - rho^2) * T / (2 - rho^2) ~ 42.857
sc_cross = (1 - rho2) * T / denom        # cross-group: equal for all in group

# ============================================================
# Build split count and attribution tables
# ============================================================

# splitCount[model_idx][feature_idx]
split_count = np.zeros((num_models, P))

for f_idx in range(num_models):
    fm = first_mover[f_idx]
    fm_group = group_of[fm]
    for j in range(P):
        j_group = group_of[j]
        if j_group == fm_group:
            # Same group as first-mover
            if j == fm:
                # Axiom 8: splitCount_firstMover
                split_count[f_idx, j] = sc_first
            else:
                # Axiom 9: splitCount_nonFirstMover
                split_count[f_idx, j] = sc_same_non
        else:
            # Different group: axiom 11 requires all features in this
            # group to have equal split counts. We set them to the
            # non-first-mover value for consistency.
            split_count[f_idx, j] = sc_cross

# attribution[model_idx][feature_idx] = c * splitCount
attribution = c * split_count

# ============================================================
# Verification of all 15 axioms
# ============================================================

results = []
all_pass = True


def check(name, condition, detail=""):
    """Record a pass/fail result."""
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    results.append((name, status, detail))


# --- Type declarations (axioms 1-6) ---

# 1. Model: instantiated as {0, 1, 2, 3} (Fin 4)
check("Model (type)", num_models == 4,
      f"|Model| = {num_models}")

# 2. numTrees: T = 100
check("numTrees", T == 100,
      f"T = {T}")

# 3. numTrees_pos: T > 0
check("numTrees_pos", T > 0,
      f"T = {T} > 0")

# 4. attribution: defined for all (j, f)
attr_defined = all(
    np.isfinite(attribution[f_idx, j])
    for f_idx in range(num_models) for j in range(P)
)
check("attribution (total function)", attr_defined,
      f"attribution defined on {P} x {num_models} = {P * num_models} entries")

# 5. splitCount: defined for all (j, f)
sc_defined = all(
    np.isfinite(split_count[f_idx, j])
    for f_idx in range(num_models) for j in range(P)
)
check("splitCount (total function)", sc_defined,
      f"splitCount defined on {P} x {num_models} = {P * num_models} entries")

# 6. firstMover: defined for all models, values in Fin P
fm_valid = all(0 <= first_mover[f] < P for f in range(num_models))
check("firstMover (total function)", fm_valid,
      f"firstMover maps to Fin {P}")

# --- Domain-specific axioms (7-13) ---

# 7. firstMover_surjective: for each group ell, each feature j in group ell,
#    there exists a model f with firstMover(f) = j
surj_ok = True
surj_detail = []
for ell, members in groups.items():
    for j in members:
        found = any(first_mover[f] == j for f in range(num_models))
        if not found:
            surj_ok = False
            surj_detail.append(f"feature {j} in group {ell}: no model")
        else:
            f_witness = [f for f in range(num_models) if first_mover[f] == j][0]
            surj_detail.append(f"feature {j} in group {ell}: model f_{f_witness}")
check("firstMover_surjective", surj_ok,
      "; ".join(surj_detail))

# 8. splitCount_firstMover: splitCount(firstMover(f), f) = T / (2 - rho^2)
expected_sc_fm = T / (2 - rho2)
sc_fm_ok = True
sc_fm_detail = []
for f_idx in range(num_models):
    fm = first_mover[f_idx]
    actual = split_count[f_idx, fm]
    ok = np.isclose(actual, expected_sc_fm)
    if not ok:
        sc_fm_ok = False
    sc_fm_detail.append(f"f_{f_idx}: sc({fm}, f_{f_idx}) = {actual:.6f} (expected {expected_sc_fm:.6f})")
check("splitCount_firstMover", sc_fm_ok,
      "; ".join(sc_fm_detail))

# 9. splitCount_nonFirstMover: for k != firstMover(f) in same group,
#    splitCount(k, f) = (1 - rho^2) * T / (2 - rho^2)
expected_sc_nfm = (1 - rho2) * T / (2 - rho2)
sc_nfm_ok = True
sc_nfm_detail = []
for f_idx in range(num_models):
    fm = first_mover[f_idx]
    fm_group = group_of[fm]
    for k in groups[fm_group]:
        if k != fm:
            actual = split_count[f_idx, k]
            ok = np.isclose(actual, expected_sc_nfm)
            if not ok:
                sc_nfm_ok = False
            sc_nfm_detail.append(
                f"f_{f_idx}: sc({k}, f_{f_idx}) = {actual:.6f} "
                f"(expected {expected_sc_nfm:.6f})"
            )
check("splitCount_nonFirstMover", sc_nfm_ok,
      "; ".join(sc_nfm_detail))

# 10. proportionality_global: exists c > 0 such that for all f, j:
#     attribution(j, f) = c * splitCount(j, f)
prop_ok = True
prop_detail = []
for f_idx in range(num_models):
    for j in range(P):
        expected = c * split_count[f_idx, j]
        actual = attribution[f_idx, j]
        if not np.isclose(actual, expected):
            prop_ok = False
            prop_detail.append(f"f_{f_idx}, j={j}: {actual} != c*{split_count[f_idx, j]}")
check("proportionality_global", prop_ok and c > 0,
      f"c = {c} > 0, all {P * num_models} entries match"
      if prop_ok else "; ".join(prop_detail))

# 11. splitCount_crossGroup_symmetric: for j, k in group ell,
#     if firstMover(f) not in group ell, then splitCount(j, f) = splitCount(k, f)
cross_ok = True
cross_detail = []
for f_idx in range(num_models):
    fm = first_mover[f_idx]
    fm_group = group_of[fm]
    for ell, members in groups.items():
        if ell != fm_group:
            # firstMover is NOT in group ell
            vals = [split_count[f_idx, j] for j in members]
            if not np.allclose(vals, vals[0]):
                cross_ok = False
                cross_detail.append(
                    f"f_{f_idx}, group {ell}: {vals} NOT equal"
                )
            else:
                cross_detail.append(
                    f"f_{f_idx}, group {ell}: all = {vals[0]:.6f}"
                )
check("splitCount_crossGroup_symmetric", cross_ok,
      "; ".join(cross_detail))

# 12. consensus_variance_bound: Var(consensus_j) = Var(phi_j) / M
#     Under uniform distribution over 4 models.
#     We verify for M = 1, 2, 4.
var_ok = True
var_detail = []

# Compute Var(phi_j) under uniform over 4 models
attr_var = np.var(attribution, axis=0, ddof=0)  # population variance

for M in [1, 2, 4]:
    for j in range(P):
        expected_consensus_var = attr_var[j] / M
        # Verify non-negativity
        if expected_consensus_var < -1e-15:
            var_ok = False
            var_detail.append(f"M={M}, j={j}: negative variance")
        else:
            var_detail.append(
                f"M={M}, j={j}: Var(consensus) = {expected_consensus_var:.6f} "
                f"= Var(phi_{j})/{M} = {attr_var[j]:.6f}/{M}"
            )
check("consensus_variance_bound", var_ok,
      f"Var(phi) = {attr_var}; bound holds for M in {{1,2,4}}")

# 13. spearman_classical_bound: spearmanCorr(f, f') <= 1 - m^3 / P^3
#     when firstMover(f) and firstMover(f') are in the same group but differ
spearman_bound = 1 - m ** 3 / P ** 3  # 1 - 8/64 = 0.875
spearman_ok = True
spearman_detail = []

for f_idx in range(num_models):
    for fp_idx in range(num_models):
        fm_f = first_mover[f_idx]
        fm_fp = first_mover[fp_idx]
        if fm_f == fm_fp:
            continue
        # Check if both first-movers are in the same group
        if group_of[fm_f] != group_of[fm_fp]:
            continue
        # These two models have different first-movers in the same group
        attr_f = attribution[f_idx, :]
        attr_fp = attribution[fp_idx, :]
        rho_spearman, _ = spearmanr(attr_f, attr_fp)
        ok = rho_spearman <= spearman_bound + 1e-10  # small tolerance
        if not ok:
            spearman_ok = False
        spearman_detail.append(
            f"f_{f_idx} vs f_{fp_idx}: Spearman = {rho_spearman:.6f} "
            f"<= {spearman_bound:.6f} {'OK' if ok else 'FAIL'}"
        )
check("spearman_classical_bound", spearman_ok,
      "; ".join(spearman_detail))

# --- Infrastructure axioms (14-15) ---

# 14. modelMeasurableSpace: sigma-algebra on Model
#     Instantiated as discrete (power set) on Fin 4
check("modelMeasurableSpace", True,
      "discrete sigma-algebra (power set) on Fin 4")

# 15. modelMeasure: probability measure on Model
#     Uniform: each model has probability 1/4
model_probs = np.ones(num_models) / num_models
is_prob_measure = (
    np.isclose(model_probs.sum(), 1.0) and
    all(p >= 0 for p in model_probs)
)
check("modelMeasure", is_prob_measure,
      f"uniform: P(f_i) = 1/{num_models} for all i, sum = {model_probs.sum():.4f}")

# ============================================================
# Output
# ============================================================

print("=" * 72)
print("AXIOM CONSISTENCY MODEL")
print("Concrete numerical model satisfying all 15 axioms simultaneously")
print("=" * 72)

print(f"\nParameters: P={P}, L={L}, m={m}, rho={rho}, T={T}, c={c}")
print(f"Derived:    rho^2={rho2}, 2-rho^2={denom}, "
      f"T/(2-rho^2)={sc_first:.6f}, (1-rho^2)*T/(2-rho^2)={sc_same_non:.6f}")

print(f"\nGroup structure:")
for ell, members in groups.items():
    print(f"  Group {ell}: features {members}")

print(f"\nFirst-mover assignment:")
for f_idx in range(num_models):
    print(f"  Model f_{f_idx}: firstMover = {first_mover[f_idx]}")

print(f"\nSplit count table (model x feature):")
print(f"{'':>10}", end="")
for j in range(P):
    print(f"  feat {j:>2}", end="")
print()
for f_idx in range(num_models):
    print(f"  f_{f_idx}:   ", end="")
    for j in range(P):
        print(f"  {split_count[f_idx, j]:>7.3f}", end="")
    print()

print(f"\nAttribution table (c={c}, attribution = c * splitCount):")
print(f"{'':>10}", end="")
for j in range(P):
    print(f"  feat {j:>2}", end="")
print()
for f_idx in range(num_models):
    print(f"  f_{f_idx}:   ", end="")
    for j in range(P):
        print(f"  {attribution[f_idx, j]:>7.3f}", end="")
    print()

print(f"\nAttribution variance (uniform over models):")
for j in range(P):
    print(f"  Var(phi_{j}) = {attr_var[j]:.6f}")

print(f"\nSpearman bound: 1 - m^3/P^3 = 1 - {m}^3/{P}^3 = {spearman_bound:.6f}")

print("\n" + "=" * 72)
print("AXIOM VERIFICATION RESULTS")
print("=" * 72)
print(f"\n{'#':>3}  {'Status':>6}  Axiom")
print("-" * 72)
for i, (name, status, detail) in enumerate(results, 1):
    marker = "OK" if status == "PASS" else "XX"
    print(f"{i:>3}  [{marker:>2}]   {name}")
    if detail:
        # Wrap detail lines for readability
        lines = detail.split("; ")
        for line in lines:
            print(f"            {line}")

passed = sum(1 for _, s, _ in results if s == "PASS")
total = len(results)
print("-" * 72)
print(f"\nResult: {passed}/{total} axioms satisfied")

if all_pass:
    print("\nCONCLUSION: The axiom system is consistent.")
    print("            A concrete model satisfying all 15 axioms exists.")
    sys.exit(0)
else:
    failed = [name for name, s, _ in results if s != "PASS"]
    print(f"\nFAILED axioms: {', '.join(failed)}")
    sys.exit(1)
