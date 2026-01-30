import numpy as np
import pytest

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend


class TestNumpyBackend:
    """
    Test suite for the linear algebra backend.
    """

    @pytest.fixture
    def backend(self):
        """Fixture that provides a fresh backend instance for each test."""
        return NumpyBackend()

    def test_pauli_properties(self, backend: NumpyBackend):
        """Verify fundamental properties of Pauli matrices."""
        x = backend.get_pauli_x()
        i = backend.get_identity(2)

        x_sq = x @ x

        np.testing.assert_allclose(x_sq, i, err_msg="X^2 must equal Identity")

    def test_kronecker_product(self, backend: NumpyBackend):
        """Verify tensor product logic using a known simple case."""
        i2 = backend.get_identity(2)
        i4 = backend.get_identity(4)

        result = backend.kronecker_product([i2, i2])

        np.testing.assert_allclose(result, i4, err_msg="I2 (x) I2 must equal I4")

    def test_hermitian_adjoint(self, backend):
        """Verify the adjoint (conjugate transpose) logic."""
        m = np.array([[1, 1j], [0, 2]], dtype=complex)
        expected = np.array([[1, 0], [-1j, 2]], dtype=complex)

        result = backend.adjoint(m)

        np.testing.assert_allclose(
            result,
            expected,
            err_msg="Result must be the conjugate transpose of the input matrix.",
        )
