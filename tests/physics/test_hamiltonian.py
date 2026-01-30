import numpy as np
import pytest

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


class TestFermiHubbardModel:
    """
    Test suite for the Fermi Hubbard Model Hamiltonian construction.
    """

    @pytest.fixture
    def hubbard_model(self):
        """Fixture to create a 2-site dimer model."""
        backend = NumpyBackend()
        mapper = JordanWignerMapper(backend)
        return FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])

    def test_hermiticity(self, hubbard_model: FermiHubbardModel):
        """Test if the generated Hamiltonian is Hermitian (H == H†)."""
        t, U = 1.0, 2.0
        H = hubbard_model.construct_total_hamiltonian(t, U)
        H_dag = H.conj().T

        np.testing.assert_allclose(
            H,
            H_dag,
            atol=1e-10,
            err_msg="Hamiltonian must be Hermitian.",
        )

    def test_free_particle_limit(self, hubbard_model: FermiHubbardModel):
        """
        Test U=0 (Non-interacting limit).
        Physics: For a 2-site dimer with t=1, single-particle energies are (+/-) t.
        Ground state (2 electrons): Both occupy the lower energy orbital (-t).
        Total Ground State Energy = 2 * (-t) = -2t.
        """
        t, U = 1.0, 0.0
        H = hubbard_model.construct_total_hamiltonian(t, U)

        eigenvalues = np.linalg.eigvalsh(H)
        ground_state_energy = eigenvalues[0]

        expected_energy = -2.0 * t

        np.testing.assert_allclose(
            ground_state_energy,
            expected_energy,
            atol=1e-10,
            err_msg="Ground State Energy for U=0 should be -2t.",
        )

    def test_atomic_limit_diagonal(self, hubbard_model: FermiHubbardModel):
        """
        Test t=0 (Atomic/Interaction limit).
        Physics: The Hamiltonian should be purely diagonal.
        Energy equals U * (number of doubly occupied sites).
        """
        t, U = 0.0, 5.0
        H = hubbard_model.construct_total_hamiltonian(t, U)

        # Off-diagonal elements should be zero
        diagonal = np.diag(H)
        H_diagonal_matrix = np.diag(diagonal)
        np.testing.assert_allclose(
            H,
            H_diagonal_matrix,
            err_msg="H should be diagonal when t=0.",
        )

        # Energy of state |1100> should be U (double occupancy)
        state_idx = int("1100", 2)
        energy_of_doubly_occupied = diagonal[state_idx]

        np.testing.assert_allclose(
            energy_of_doubly_occupied,
            U,
            err_msg="State with one double occupancy should hae energy U.",
        )

        # Energy of state |1010> should be 0 (no double occupancy)
        state_idx_singly = int("1010", 2)
        energy_of_singly_occupied = diagonal[state_idx_singly]

        np.testing.assert_allclose(
            energy_of_singly_occupied,
            0.0,
            err_msg="State with no double occupancy should have energy 0.",
        )
