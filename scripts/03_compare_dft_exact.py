import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.dft import LatticeDFT
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.visualization.plotter import ResultPlotter


def get_exact_energy(
    model: FermiHubbardModel,
    backend: NumpyBackend,
    t: float,
    u: float,
) -> float:
    """Helper to run the Exact Diagonalization (FCI) for a single point."""
    H = model.construct_total_hamiltonian(t, u)
    eigenvalues, eigenvectors = backend.diagonalize(H)

    N_op = model.construct_total_number_operator()
    n_qubits = model.n_qubits

    for idx, en in enumerate(eigenvalues):
        psi = backend.get_column_vector(eigenvectors, idx)
        n_op_psi = backend.matmul(N_op, psi)
        n_val = backend.inner_product(psi, n_op_psi).real

        if np.isclose(n_val, n_qubits / 2, atol=1e-2):
            return en
    return eigenvalues[0]


if __name__ == "__main__":
    L, t = 2, 1.0
    backend = NumpyBackend()
    mapper = JordanWignerMapper(backend)
    model = FermiHubbardModel(backend, mapper, n_sites=L, edges=[(0, 1)])
    dft_solver = LatticeDFT(model, backend)
    plotter = ResultPlotter()

    print("=== Phase 3: Exact vs Classical DFT Comparison ===")

    u_values = np.linspace(0, 10, 21)
    exact_energies = []
    dft_energies = []

    print(f"{'U/t':<6} | {'Exact':<10} | {'DFT':<10} | {'Diff':<10}")
    print("-" * 45)

    for U in u_values:
        e_exact = get_exact_energy(model, backend, t, U)
        exact_energies.append(e_exact)

        e_dft, _, converged = dft_solver.run_scf_loop(t, U, max_iter=200, mixing=0.5)
        dft_energies.append(e_dft)

        diff = abs(e_exact - e_dft)
        print(
            f"{U:<6.1f} | {e_exact:<10.4f} | {e_dft:<10.4f} | {diff:<10.4f} "
            f"{'*' if not converged else ''}"
        )

    plotter.plot_dft_comparison(
        list(u_values),
        exact_energies,
        dft_energies,
        filename="phase3_dft_comparison.png",
        title=f"Hubbard Dimer (L={L}): Exact vs DFT",
    )
