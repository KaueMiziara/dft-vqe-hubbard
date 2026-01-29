from dft_vqe_hubbard.algebra.operator_backend import OperatorBackend


class JordanWignerMapper[MatrixType]:
    """
    Handles the mapping of Fermionic operators to Qubit operators using the
    Jordan-Wigner transformation.

    This class is responsible for constructing the matrix representations of
    creation and annihilation operators by combining Pauli matrices via the
    provided linear algebra backend.
    """

    def __init__(self, backend: OperatorBackend[MatrixType]) -> None:
        """Initializes the mapper with a specific linear algebra backend.

        Args:
            backend: An instance of `OperatorBackend` to handle
                     matrix generation and operations.
        """
        self._backend = backend

    def _get_creation_operator(self) -> MatrixType:
        """Constructs the Qubit Creation Operator (Sigma Plus).

        Mathematically: σ+ = 0.5 * (X - iY)
        In the computational basis (|0>, |1>), this maps |0> to |1> (0 -> 1).

        Matrix form:
        [[0, 0],
         [1, 0]]

        Returns:
            MatrixType: The 2x2 creation operator matrix.
        """
        x = self._backend.get_pauli_x()
        y = self._backend.get_pauli_y()

        minus_iy = self._backend.matrix_scale(y, -1j)
        x_minus_iy = self._backend.matrix_add(x, minus_iy)

        return self._backend.matrix_scale(x_minus_iy, 0.5)

    def _get_annihilation_operator(self) -> MatrixType:
        """Constructs the Qubit Annihilation Operator (Sigma Minus).

        Mathematically: σ- = 0.5 * (X + iY)
        In the computational basis, this maps |1> to |0> (1 -> 0).

        Matrix form:
        [[0, 1],
         [0, 0]]

        Returns:
            MatrixType: The 2x2 annihilation operator matrix.
        """

        x = self._backend.get_pauli_x()
        y = self._backend.get_pauli_y()

        iy = self._backend.matrix_scale(y, 1j)
        x_plus_iy = self._backend.matrix_add(x, iy)

        return self._backend.matrix_scale(x_plus_iy, 0.5)
