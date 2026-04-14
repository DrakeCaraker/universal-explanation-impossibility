"""
Numerical verification of eta = dim(V^G)/dim(V) for quantum measurements.

For computational basis measurement on d-dimensional systems:
    eta_predicted = (d-1)/(d^2-1) = 1/(d+1)

We verify this by:
1. Generating random density matrices
2. Extracting the "measurable" (diagonal) Bloch components
3. Computing the fraction of total Bloch vector norm^2 captured
4. Comparing the average fraction to the theoretical prediction

The key identity: for Haar-random pure states (or uniformly random mixed states),
the expected fraction of the Bloch vector norm^2 in the diagonal subspace equals
dim(diagonal subspace) / dim(full Bloch space) = (d-1)/(d^2-1) = 1/(d+1).

This follows from the isotropy of Haar measure: each Bloch component has equal
expected squared magnitude, so the fraction captured equals the fraction of
components retained.
"""

import json
import numpy as np
from pathlib import Path

np.random.seed(42)

N_SAMPLES = 10_000  # more samples for tighter confidence intervals


def random_density_matrix(d: int) -> np.ndarray:
    """Generate a random density matrix of dimension d using the Hilbert-Schmidt measure.

    Method: rho = G G^dagger / Tr(G G^dagger) where G is a d x d complex Gaussian matrix.
    This gives the Hilbert-Schmidt (flat) measure on density matrices.
    """
    G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def gell_mann_basis(d: int) -> list[np.ndarray]:
    """Construct a complete orthonormal basis of traceless Hermitian d x d matrices.

    These are the generalized Gell-Mann matrices, normalized so that
    Tr(lambda_a lambda_b) = 2 delta_{ab}.

    Returns d^2 - 1 matrices spanning the traceless Hermitian operators.
    """
    matrices = []

    # Symmetric off-diagonal: (|j><k| + |k><j|) for j < k
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = 1.0
            m[k, j] = 1.0
            matrices.append(m)

    # Antisymmetric off-diagonal: -i(|j><k| - |k><j|) for j < k
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = -1j
            m[k, j] = 1j
            matrices.append(m)

    # Diagonal: generalized diagonal matrices
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        for j in range(l):
            m[j, j] = 1.0
        m[l, l] = -l
        m *= np.sqrt(2.0 / (l * (l + 1)))
        matrices.append(m)

    return matrices


def bloch_vector(rho: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Extract the Bloch vector of rho with respect to the given basis.

    The Bloch vector component a_k = Tr(rho * lambda_k).
    """
    return np.array([np.trace(rho @ lam).real for lam in basis])


def identify_diagonal_indices(basis: list[np.ndarray], d: int) -> list[int]:
    """Identify which basis matrices are diagonal (correspond to populations).

    For the generalized Gell-Mann basis, the diagonal matrices are the last
    (d-1) in the list. But we check explicitly to be safe.
    """
    indices = []
    for i, lam in enumerate(basis):
        # A matrix is diagonal iff all off-diagonal elements are zero
        off_diag = lam.copy()
        np.fill_diagonal(off_diag, 0)
        if np.allclose(off_diag, 0):
            indices.append(i)
    return indices


def run_verification(n_qubits: int, n_samples: int) -> dict:
    """Run the eta verification for n-qubit computational basis measurement."""
    d = 2 ** n_qubits
    dim_V = d * d - 1  # dimension of full Bloch space

    # Theoretical prediction
    eta_predicted = (d - 1) / (d * d - 1)  # = 1/(d+1)

    # Build generalized Gell-Mann basis
    basis = gell_mann_basis(d)
    assert len(basis) == dim_V, f"Expected {dim_V} basis elements, got {len(basis)}"

    # Identify diagonal basis elements (these correspond to V^G)
    diag_indices = identify_diagonal_indices(basis, d)
    dim_VG = len(diag_indices)

    # Structural check
    assert dim_VG == d - 1, (
        f"Expected {d-1} diagonal generators, got {dim_VG}"
    )

    # Numerical verification: average fraction of Bloch norm^2 in diagonal subspace
    fractions = []
    for _ in range(n_samples):
        rho = random_density_matrix(d)
        bv = bloch_vector(rho, basis)
        total_norm_sq = np.sum(bv ** 2)
        if total_norm_sq < 1e-15:
            continue  # skip the maximally mixed state (Bloch vector = 0)
        diag_norm_sq = np.sum(bv[diag_indices] ** 2)
        fractions.append(diag_norm_sq / total_norm_sq)

    eta_numerical = float(np.mean(fractions))
    eta_std = float(np.std(fractions) / np.sqrt(len(fractions)))

    return {
        "n_qubits": n_qubits,
        "d": d,
        "dim_V": dim_V,
        "dim_VG": dim_VG,
        "eta_predicted": eta_predicted,
        "eta_predicted_formula": f"1/(d+1) = 1/{d+1}",
        "eta_numerical": round(eta_numerical, 6),
        "eta_std_error": round(eta_std, 6),
        "n_samples": len(fractions),
        "match": abs(eta_numerical - eta_predicted) < 3 * eta_std,
        "deviation_sigmas": round(
            abs(eta_numerical - eta_predicted) / max(eta_std, 1e-15), 2
        ),
    }


def main():
    results = {"experiment": "quantum_measurement_eta_verification", "cases": []}

    for n_qubits in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Verifying: n={n_qubits} qubits, d={2**n_qubits}")
        print(f"{'='*60}")

        # Use fewer samples for larger systems (d=8 has 63-dim Bloch space)
        samples = N_SAMPLES if n_qubits <= 2 else 2000
        result = run_verification(n_qubits, samples)
        results["cases"].append(result)

        print(f"  dim(V)  = {result['dim_V']}")
        print(f"  dim(V^G)= {result['dim_VG']}")
        print(f"  eta predicted = {result['eta_predicted']:.6f}  [{result['eta_predicted_formula']}]")
        print(f"  eta numerical = {result['eta_numerical']:.6f} +/- {result['eta_std_error']:.6f}")
        print(f"  deviation     = {result['deviation_sigmas']} sigma")
        print(f"  MATCH: {result['match']}")

    # Summary
    all_match = all(c["match"] for c in results["cases"])
    results["all_match"] = all_match

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PREDICTIONS VERIFIED' if all_match else 'SOME PREDICTIONS FAILED'}")
    print(f"{'='*60}")

    # Save results
    out_path = Path(__file__).parent / "results_quantum_verification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
