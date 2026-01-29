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

    def get_fermion_creation_operator(
        self,
        n_qubits: int,
        target_index: int,
    ) -> MatrixType:
        """Constructs the Fermionic Creation Operator (c†) for a specific site/orbital.

        Uses the Jordan-Wigner transformation:
        c†_j = (Z_0 ⊗ ... ⊗ Z_{j-1}) ⊗ σ+_j ⊗ (I_{j+1} ⊗ ... ⊗ I_{N-1})

        Args:
            n_qubits: The total number of qubits (spin-orbitals) in the system.
            target_index: The index (j) of the qubit where the fermion is created.
                          Must be between 0 and n_qubits - 1.

        Returns:
            MatrixType: The matrix representation of size (2^N x 2^N).

        Raises:
            ValueError: If target_index is out of bounds.
        """
        if not (0 <= target_index < n_qubits):
            raise ValueError(
                f"Target index {target_index} out of bounds for {n_qubits} qubits."
            )

        operator_list = []

        for k in range(n_qubits):
            if k < target_index:
                operator_list.append(self._backend.get_pauli_z())
            elif k == target_index:
                operator_list.append(self._get_creation_operator())
            else:
                operator_list.append(self._backend.get_identity())

        return self._backend.kronecker_product(operator_list)

    def get_fermion_annihilation_operator(
        self,
        n_qubits: int,
        target_index: int,
    ) -> MatrixType:
        """Constructs the Fermionic Annihilation Operator c for a specific site/orbital.

        Calculated as the Hermitian conjugate (adjoint) of the creation operator:
        c_j = (c†_j)†

        Args:
            n_qubits: The total number of qubits in the system.
            target_index: The index (j) of the qubit where the fermion is annihilated.

        Returns:
            MatrixType: The matrix representation of size (2^N x 2^N).
        """
        creation_op = self.get_fermion_creation_operator(n_qubits, target_index)

        return self._backend.adjoint(creation_op)

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
