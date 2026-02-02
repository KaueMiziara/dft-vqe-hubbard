from typing import Any

import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.physics.dft import LatticeDFT
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHEA
from dft_vqe_hubbard.quantum.pennylane_vqe import PennyLaneVQESolver
from dft_vqe_hubbard.quantum.vqe_solver import VQESolver
from dft_vqe_hubbard.visualization.plotter import ResultPlotter


def get_exact_fci(
    model: FermiHubbardModel,
    backend: NumpyBackend,
    t: float,
    u: float,
) -> float:
    """Calculates ground truth via Exact Diagonalization (FCI) for N=2 sector."""
    h_total = model.construct_total_hamiltonian(t, u)
    eigenvalues, eigenvectors = backend.diagonalize(h_total)

    n_op = model.construct_total_number_operator()
    target_n = model.n_sites

    for idx, en in enumerate(eigenvalues):
        psi = backend.get_column_vector(eigenvectors, idx)
        n_psi = backend.matmul(n_op, psi)
        n_val = backend.inner_product(psi, n_psi).real

        if np.isclose(n_val, target_n, atol=1e-2):
            return float(en)

    return float(eigenvalues[0])


if __name__ == "__main__":
    L, t = 2, 1.0
    u_values = np.linspace(0, 10, 21)
    n_layers = 3

    np_backend = NumpyBackend()
    mapper_np = JordanWignerMapper(np_backend)
    model_np = FermiHubbardModel(np_backend, mapper_np, n_sites=L, edges=[(0, 1)])
    dft_solver = LatticeDFT(model_np, np_backend)

    pl_backend = PennyLaneBackend()
    pl_ansatz = PennyLaneHEA(
        model_pl := FermiHubbardModel(
            pl_backend, JordanWignerMapper(pl_backend), n_sites=L, edges=[(0, 1)]
        )
    )

    quantum_solvers: dict[str, VQESolver[Any, Any]] = {
        "VQE (PennyLane HEA)": PennyLaneVQESolver(pl_ansatz, n_qubits=model_pl.n_qubits)
    }

    results_exact: list[float] = []
    results_dft: list[float] = []
    vqe_benchmarks: dict[str, list[float]] = {name: [] for name in quantum_solvers}

    plotter = ResultPlotter()

    for U in u_values:
        print(f"\n>>> Solving for U={U}...")

        results_exact.append(get_exact_fci(model_np, np_backend, t, U))

        e_dft, _, _ = dft_solver.run_scf_loop(t, U)
        results_dft.append(e_dft)

        for name, solver in quantum_solvers.items():
            h_vqe = model_pl.construct_total_hamiltonian(t, U)
            e_vqe, _ = solver.solve(
                h_vqe,
                n_layers=n_layers,
                learning_rate=0.04,
                steps=200,
            )
            vqe_benchmarks[name].append(e_vqe)

    plotter.plot_full_benchmark(
        u_values=list(u_values),
        exact_energies=results_exact,
        dft_energies=results_dft,
        vqe_results=vqe_benchmarks,
        filename="phase4_final_benchmark.png",
    )
