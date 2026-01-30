import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


def print_sparse_matrix(matrix: np.ndarray, label: str):
    print(f"\n--- {label} (Non-zero elements) ---")
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            if np.abs(val) > 1e-10:
                print(f"  [{r}, {c}]:\t{val.real:.2f} + {val.imag:.2f}i")


if __name__ == "__main__":
    N_QUBITS = 4

    backend = NumpyBackend()
    mapper = JordanWignerMapper(backend)

    print("=== Phase 1 Validation: Operator Construction ===")

    target_idx = 0
    print(
        f"\nBuilding operators for Spin-Orbital {target_idx} "
        f"(System Size: {N_QUBITS} qubits)..."
    )

    c_dag = mapper.get_fermion_creation_operator(N_QUBITS, target_idx)
    c = mapper.get_fermion_annihilation_operator(N_QUBITS, target_idx)

    n_op = c_dag @ c

    diagonal = np.diag(n_op)

    print("\n--- Verification: Diagonal Elements ---")

    pass_count = 0
    for i, val in enumerate(diagonal):
        binary_str = format(i, f"0{N_QUBITS}b")

        expected_occupancy = int(binary_str[target_idx])

        is_correct = np.isclose(val.real, expected_occupancy) and np.isclose(
            val.imag, 0
        )

        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            pass_count += 1

        print(
            f"State |{binary_str}> (Idx {i}):\t{val.real:.1f} "
            f"[Expected: {expected_occupancy}] -> {status}"
        )

    print_sparse_matrix(n_op, "n_op")

    if pass_count == 16:
        print(
            f"\nSUCCESS: Number Operator correctly counts electrons on "
            f"orbital {target_idx} for all 16 basis states."
        )
    else:
        print(f"\nFAILURE: Verification failed. ({pass_count}/16 passed)")
