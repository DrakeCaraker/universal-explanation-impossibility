#!/usr/bin/env python3
"""
Uncertainty from Symmetry: Numerical demonstration that the η law and
quantum information loss under symmetry arise from the same
representation-theoretic decomposition (Theorem: Uncertainty from Symmetry).

For a finite group G acting on V, the Reynolds operator R projects onto V^G.
The fraction of information retained is dim(V^G)/dim(V) = 1 - η.
This governs BOTH:
  - Explanation instability (fraction of ranking content lost by orbit averaging)
  - Quantum information loss (fraction of state inaccessible to G-covariant measurement)
"""

import numpy as np
from itertools import permutations

def permutation_matrix(perm, n):
    """Create n×n permutation matrix from a permutation tuple."""
    P = np.zeros((n, n))
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P

def reynolds_operator(group_matrices):
    """Compute Reynolds operator R = (1/|G|) Σ ρ(g)."""
    return np.mean(group_matrices, axis=0)

def invariant_subspace_dim(R, tol=1e-10):
    """Dimension of V^G = rank of Reynolds operator."""
    eigenvalues = np.linalg.eigvalsh(R)
    return int(np.sum(eigenvalues > 1 - tol))

def run_demo(group_name, n, group_generators_fn):
    """Run the unified uncertainty demonstration for a given group."""
    print(f"\n{'='*60}")
    print(f"Group: {group_name} acting on V = R^{n}")
    print(f"{'='*60}")

    # Generate group matrices
    group_mats = group_generators_fn(n)
    G_size = len(group_mats)

    # Reynolds operator
    R = reynolds_operator(group_mats)
    k = invariant_subspace_dim(R)
    eta_predicted = 1 - k / n

    print(f"|G| = {G_size}")
    print(f"dim(V^G) = {k}")
    print(f"dim(V) = {n}")
    print(f"η = 1 - {k}/{n} = {eta_predicted:.4f}")

    # === EXPLANATION SETTING ===
    # Sample random explanation vectors, project, measure information loss
    n_samples = 50000
    V = np.random.randn(n_samples, n)
    V /= np.linalg.norm(V, axis=1, keepdims=True)  # unit vectors

    Rv = V @ R.T  # projected vectors
    loss = np.mean(np.sum((V - Rv)**2, axis=1))
    retained = np.mean(np.sum(Rv**2, axis=1))

    print(f"\n--- Explanation setting ---")
    print(f"E[||v - Rv||²] = {loss:.4f}  (predicted η = {eta_predicted:.4f})")
    print(f"E[||Rv||²]     = {retained:.4f}  (predicted 1-η = {1-eta_predicted:.4f})")

    # Flip rate: for random explanation vectors, what fraction of pairwise
    # rankings change after projection?
    n_flip_samples = 5000
    v_samples = np.random.randn(n_flip_samples, n)
    v_samples /= np.linalg.norm(v_samples, axis=1, keepdims=True)
    rv_samples = v_samples @ R.T

    flip_count = 0
    total_pairs = 0
    for idx in range(min(n_flip_samples, 1000)):
        v = v_samples[idx]
        rv = rv_samples[idx]
        for i in range(n):
            for j in range(i+1, n):
                total_pairs += 1
                if (v[i] > v[j]) != (rv[i] > rv[j]):
                    flip_count += 1
    flip_rate = flip_count / total_pairs if total_pairs > 0 else 0
    print(f"Ranking flip rate (v vs Rv): {flip_rate:.4f}")

    # === QUANTUM SETTING ===
    # For complex Hilbert space, the twirl projects onto the G-invariant sector
    # Probability of finding a random pure state in V^G
    n_q_samples = 50000
    # Random complex unit vectors (Haar-distributed)
    psi_real = np.random.randn(n_q_samples, n)
    psi_imag = np.random.randn(n_q_samples, n)
    psi = psi_real + 1j * psi_imag
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)

    # Project onto V^G (using real R, extended to complex)
    R_psi = psi @ R.T  # R is real symmetric, acts on complex vectors
    prob_invariant = np.mean(np.sum(np.abs(R_psi)**2, axis=1))
    info_loss_q = 1 - prob_invariant

    print(f"\n--- Quantum setting ---")
    print(f"E[<ψ|P_VG|ψ>]    = {prob_invariant:.4f}  (predicted 1-η = {1-eta_predicted:.4f})")
    print(f"Info loss (1-above)= {info_loss_q:.4f}  (predicted η = {eta_predicted:.4f})")

    # === UNIFIED CHECK ===
    print(f"\n--- Unified verification ---")
    print(f"Explanation loss:  {loss:.4f}")
    print(f"Quantum loss:      {info_loss_q:.4f}")
    print(f"Predicted η:       {eta_predicted:.4f}")
    print(f"Match (expl):      {'✓' if abs(loss - eta_predicted) < 0.02 else '✗'} (|Δ| = {abs(loss - eta_predicted):.4f})")
    print(f"Match (quantum):   {'✓' if abs(info_loss_q - eta_predicted) < 0.02 else '✗'} (|Δ| = {abs(info_loss_q - eta_predicted):.4f})")

    return eta_predicted, loss, info_loss_q


def s2_on_2d(n):
    """S_2 acting on R^2 by coordinate swap."""
    assert n == 2
    return [np.eye(2), np.array([[0,1],[1,0]])]

def s3_on_3d(n):
    """S_3 acting on R^3 by coordinate permutation."""
    assert n == 3
    return [permutation_matrix(p, 3) for p in permutations(range(3))]

def s4_on_4d(n):
    """S_4 acting on R^4 by coordinate permutation."""
    assert n == 4
    return [permutation_matrix(p, 4) for p in permutations(range(4))]

def s2xs2_on_4d(n):
    """S_2 × S_2 acting on R^4: swap first pair, swap second pair."""
    assert n == 4
    e = np.eye(4)
    swap01 = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    swap23 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)
    swap_both = swap01 @ swap23
    return [e, swap01, swap23, swap_both]

def s4xs4_on_10d(n):
    """S_4 × S_4 acting on R^10: permute first 4, permute last 4, fix positions 8,9.
    This models the MI experiment: 4 L0 heads, 4 L1 heads, MLP0, MLP1."""
    assert n == 10
    mats = []
    for p0 in permutations(range(4)):
        for p1 in permutations(range(4)):
            M = np.eye(10)
            for i, j in enumerate(p0):
                M[i, :] = 0
                M[i, j] = 1
            for i, j in enumerate(p1):
                M[4+i, :] = 0
                M[4+i, 4+j] = 1
            mats.append(M)
    return mats


if __name__ == "__main__":
    print("="*60)
    print("UNCERTAINTY FROM SYMMETRY")
    print("Theorem: η = 1 - dim(V^G)/dim(V) governs both")
    print("explanation instability and quantum information loss")
    print("="*60)

    results = []

    # S_2 on R^2 (simplest case: SHAP sign for 2 correlated features)
    results.append(("S₂ on R²", *run_demo("S_2", 2, s2_on_2d)))

    # S_3 on R^3 (3-fold codon degeneracy)
    results.append(("S₃ on R³", *run_demo("S_3", 3, s3_on_3d)))

    # S_4 on R^4 (4-fold codon degeneracy)
    results.append(("S₄ on R⁴", *run_demo("S_4", 4, s4_on_4d)))

    # S_2 × S_2 on R^4 (two pairs of correlated features)
    results.append(("S₂×S₂ on R⁴", *run_demo("S_2 × S_2", 4, s2xs2_on_4d)))

    # S_4 × S_4 on R^10 (MI experiment: 4+4 heads + 2 MLPs)
    results.append(("S₄×S₄ on R¹⁰", *run_demo("S_4 × S_4 (MI experiment)", 10, s4xs4_on_10d)))

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Group':<20} {'η (theory)':>12} {'η (expl)':>12} {'η (quantum)':>12}")
    print("-"*60)
    for name, eta_pred, eta_expl, eta_q in results:
        print(f"{name:<20} {eta_pred:>12.4f} {eta_expl:>12.4f} {eta_q:>12.4f}")
    print("-"*60)
    print("All three columns should match within sampling noise (~0.01).")
    print("\nConclusion: the same formula η = 1 - dim(V^G)/dim(V) governs")
    print("both explanation instability and quantum information loss,")
    print("confirming the representation-theoretic uncertainty theorem.")
