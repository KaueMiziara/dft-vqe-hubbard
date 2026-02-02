import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.physics.dft import LatticeDFT
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHVA
from dft_vqe_hubbard.quantum.pennylane_vqe import PennyLaneVQESolver
from dft_vqe_hubbard.visualization.plotter import ResultPlotter

if __name__ == "__main__":
    L, t = 2, 1.0
    u_values = np.linspace(0, 10, 11)

    pl_backend = PennyLaneBackend()
    np_backend = NumpyBackend()

    mapper_pl = JordanWignerMapper(pl_backend)
    model_pl = FermiHubbardModel(pl_backend, mapper_pl, n_sites=L, edges=[(0, 1)])

    mapper_np = JordanWignerMapper(np_backend)
    model_np = FermiHubbardModel(np_backend, mapper_np, n_sites=L, edges=[(0, 1)])

    dft_solver = LatticeDFT(model_np, np_backend)
    hva = PennyLaneHVA(model_pl)
    vqe_solver = PennyLaneVQESolver(hva, n_qubits=model_pl.n_qubits)

    plotter = ResultPlotter()

    dft_energies = []
    vqe_energies = []

    print(f"{'U/t':<6} | {'DFT Energy':<12} | {'VQE Energy (PL)':<15}")
    print("-" * 40)

    for U in u_values:
        e_dft, _, _ = dft_solver.run_scf_loop(t, U, max_iter=200)
        dft_energies.append(e_dft)

        print(f"\n>>> Training VQE for U={U}...")
        h_total = model_pl.construct_total_hamiltonian(t, U)

        e_vqe, _ = vqe_solver.solve(h_total, n_layers=2, learning_rate=0.05, steps=150)
        vqe_energies.append(e_vqe)

        print(f"{U:<6.1f} | {e_dft:<12.4f} | {e_vqe:<15.4f}")

    plotter.plot_vqe_benchmark(
        list(u_values),
        dft_energies,
        vqe_energies,
        vqe_label="VQE (PennyLane)",
        filename="phase4_vqe_vs_dft.png",
    )
