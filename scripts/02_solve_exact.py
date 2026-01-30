import os

import matplotlib.pyplot as plt
import numpy as np

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


def get_double_occupancy_operator(model: FermiHubbardModel) -> np.ndarray:
    """
    Constructs the Double Occupancy operator D = sum_i (n_i_up * n_i_down).
    We use this to measure how many sites are doubly occupied in the ground state.
    """
    return model.construct_interaction_term(penalty=1.0)


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

    d_op = get_double_occupancy_operator(model)

    print(f"Running sweep for U = {u_values[0]} to {u_values[-1]} (t={t})...")

    for U in u_values:
        H = model.construct_total_hamiltonian(t, U)

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        e_ground = eigenvalues[0]
        psi_ground = eigenvectors[:, 0]

        d_psi = backend.matmul(d_op, psi_ground)
        expectation = np.vdot(psi_ground, d_psi)

        d_val = expectation.real

        energies.append(e_ground)
        double_occupancies.append(d_val)

        if U % 2 == 0:
            print(f"  U = {U:.1f} | E0 = {e_ground:.4f} | <D> = {d_val:.2f}")

    os.makedirs("plots", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Interaction Strength (U/t)")
    ax1.set_ylabel("Ground State Energy ($E_0$)", color=color)
    ax1.plot(u_values, energies, color=color, marker="o", label="$E_0$ (Exact)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel(
        "Double Occupancy ($\\langle n_{\\uparrow} n_{\\downarrow} \\rangle$)",
        color=color,
    )
    ax2.plot(
        u_values,
        double_occupancies,
        color=color,
        marker="s",
        linestyle="--",
        label="Double Occ.",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Hubbard Dimer (L={L}): Mott Transition")
    fig.tight_layout()

    output_path = "plots/phase2_exact_solution.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nAnalysis Complete. Plot saved to: {output_path}")
