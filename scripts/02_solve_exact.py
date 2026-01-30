import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.visualization.plotter import ResultPlotter

if __name__ == "__main__":
    L = 2
    t = 1.0
    edges = [(0, 1)]

    print("=== Phase 2: Exact Diagonalization (FCI) ===")

    backend = NumpyBackend()
    mapper = JordanWignerMapper(backend)
    model = FermiHubbardModel(backend, mapper, n_sites=L, edges=edges)

    u_values = np.linspace(0, 10, 21)
    energies = []
    double_occupancies = []

    d_op = model.construct_double_occupancy_operator()
    N_op = model.construct_total_number_operator()

    print(f"Running sweep for U = {u_values[0]} to {u_values[-1]} (t={t})...")

    for U in u_values:
        H = model.construct_total_hamiltonian(t, U)

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        e_ground = None
        psi_ground = None

        for idx, energy in enumerate(eigenvalues):
            psi = eigenvectors[:, idx]

            n_expect = np.vdot(psi, backend.matmul(N_op, psi)).real

            if np.isclose(n_expect, 2.0, atol=1e-3):
                e_ground = energy
                psi_ground = psi
                break

        if psi_ground is None:
            print(f"Warning: No N=2 state found for U={U}")
            continue

        d_psi = backend.matmul(d_op, psi_ground)
        d_val = np.vdot(psi_ground, d_psi).real

        energies.append(e_ground)
        double_occupancies.append(d_val)

        if U % 2 == 0:
            print(f"  U = {U:.1f} | E0 = {e_ground:.4f} | <D> = {d_val:.2f}")

    plotter = ResultPlotter()
    plotter.plot_mott_transition(
        u_values=list(u_values),
        energies=energies,
        double_occupancies=double_occupancies,
        filename="phase2_exact_solution.png",
        title=f"Hubbard Dimer (L={L}): Mott Transition",
    )
