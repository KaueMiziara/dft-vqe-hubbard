from abc import ABC, abstractmethod


class OperatorBackend[MatrixType](ABC):
    """
    Abstract Class defining the low-level linear algebra operations required
    to construct quantum operators.
    """

    @abstractmethod
    def get_identity(self, dimension: int = 2) -> MatrixType:
        """Generates the Identity matrix of a specified dimension.

        Args:
            dimension: The size of the square matrix (dimension x dimension).
                             Defaults to 2 (single qubit).

        Returns:
            MatrixType: A square identity matrix of size `dimension`.
        """
        pass

    @abstractmethod
    def get_pauli_x(self) -> MatrixType:
        """Generates the 2x2 Pauli-X matrix (bit-flip).

        Returns:
            MatrixType: The matrix [[0, 1], [1, 0]].
        """
        pass

    @abstractmethod
    def get_pauli_y(self) -> MatrixType:
        """Generates the 2x2 Pauli-Y matrix.

        Returns:
            MatrixType: The matrix [[0, -1j], [1j, 0]].
        """
        pass

    @abstractmethod
    def get_pauli_z(self) -> MatrixType:
        """Generates the 2x2 Pauli-Z matrix (phase-flip).

        Returns:
            MatrixType: The matrix [[1, 0], [0, -1]].
        """
        pass

    @abstractmethod
    def get_zero_matrix(self, dimension: int) -> MatrixType:
        """Generates a square zero matrix of the specified dimension.

        Args:
            dimension: The size of the square matrix.

        Returns:
            MatrixType: A matrix filled with zeros.
        """
        pass

    @abstractmethod
    def kronecker_product(self, matrices: list[MatrixType]) -> MatrixType:
        """Computes the cumulative Kronecker (Tensor) product of a list of matrices.

        Mathematically equivalent to: A ⊗ B ⊗ C ...

        Args:
            matrices: A list of matrices to be sequentially tensor-multiplied.

        Returns:
            MatrixType: The resulting large matrix with dimension equal to the
                        product of the input dimensions.

        Raises:
            ValueError: If the input list `matrices` is empty.
        """
        pass

    @abstractmethod
    def matrix_add(self, a: MatrixType, b: MatrixType) -> MatrixType:
        """Computes the element-wise sum of two matrices.

        Args:
            a: The first matrix.
            b: The second matrix.

        Returns:
            MatrixType: The result of A + B.
        """
        pass

    @abstractmethod
    def matrix_scale(self, matrix: MatrixType, scalar: complex) -> MatrixType:
        """Multiplies a matrix by a scalar value.

        Args:
            matrix: The input matrix.
            scalar: The scalar value (can be complex).

        Returns:
            MatrixType: The result of scalar * A.
        """
        pass

    @abstractmethod
    def adjoint(self, matrix: MatrixType) -> MatrixType:
        """Computes the Hermitian adjoint (conjugate transpose) of the matrix.

        Args:
            matrix: The input matrix.

        Returns:
            MatrixType: The conjugate transpose of the input.
        """
        pass

    @abstractmethod
    def matmul(self, a: MatrixType, b: MatrixType) -> MatrixType:
        """Computes the matrix product of two matrices.

        This replaces reliance on the native '@' operator, ensuring compatibility
        with backends that might use different method names.

        Args:
            a: Left matrix.
            b: Right matrix.

        Returns:
            MatrixType: The result of the multiplication A @ B.
        """
        pass

    @abstractmethod
    def inner_product(self, a: MatrixType, b: MatrixType) -> complex:
        """Computes the inner product (dot product) of two vectors/matrices.

        Mathematically: <a|b> = a† . b

        Args:
            a: The "bra" vector (will be conjugated).
            b: The "ket" vector.

        Returns:
            complex: The scalar result.
        """
        pass

    @abstractmethod
    def diagonalize(self, matrix: MatrixType) -> tuple[MatrixType, MatrixType]:
        """Diagonalizes a Hermitian matrix.

        Args:
            matrix: The matrix to diagonalize.

        Returns:
            Tuple[Any, MatrixType]:
                - eigenvalues: A list or array of real eigenvalues (sorted).
                - eigenvectors: The matrix where column 'i' is the eigenvector
                    for eigenvalue 'i'.
        """
        pass
