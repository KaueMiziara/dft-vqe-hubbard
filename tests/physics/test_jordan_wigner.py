import numpy as np
import pytest

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


class TestJordanWigner:
    """
    Test suite for the Jordan Wigner mapper operator builders.
    """

    @pytest.fixture
    def mapper(self):
        backend = NumpyBackend()
        return JordanWignerMapper(backend)

    def test_sigma_plus_action(self, mapper: JordanWignerMapper):
        """
        Verify σ+ maps |0> to |1>.
        State vectors: |0> = [1, 0], |1> = [0, 1]
        """
        sigma_plus = mapper._get_creation_operator()
        state_0 = np.array([1, 0], dtype=complex)
        state_1 = np.array([0, 1], dtype=complex)

        result = sigma_plus @ state_0

        np.testing.assert_allclose(result, state_1, err_msg="Sigma+ |0> must yield |1>")

        result_on_occupied = sigma_plus @ state_1
        zero_vector = np.array([0, 0], dtype=complex)
        np.testing.assert_allclose(
            result_on_occupied,
            zero_vector,
            err_msg="Sigma+ |1> must yield 0",
        )

    def test_chain_dimension(self, mapper: JordanWignerMapper):
        """Verify the mapper generates matrices of correct size (2^N)."""
        n_qubits = 3
        expected_dim = 2**n_qubits

        op = mapper.get_fermion_creation_operator(n_qubits, target_index=0)

        assert op.shape == (expected_dim, expected_dim)

    def test_anti_commutation_relation(self, mapper: JordanWignerMapper):
        """
        Verify the fundamental Fermionic anti-commutation relation.
        {c_i, c_j^dagger} = c_i c_j^dagger + c_j^dagger c_i = delta_ij * I

        For i=j=0: c c^dagger + c^dagger c = I
        """
        n_qubits = 2
        idx = 0

        c = mapper.get_fermion_annihilation_operator(n_qubits, idx)
        cdag = mapper.get_fermion_creation_operator(n_qubits, idx)

        anticommutator = (c @ cdag) + (cdag @ c)

        identity = np.eye(2**n_qubits, dtype=complex)

        np.testing.assert_allclose(
            anticommutator,
            identity,
            atol=1e-10,
            err_msg="Fermionic operators must obey {c, c†} = I",
        )
