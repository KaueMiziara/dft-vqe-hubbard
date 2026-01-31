from typing import override

import numpy as np

from .operator_backend import OperatorBackend


class NumpyBackend(OperatorBackend[np.ndarray]):
    """
    Concrete implementation of OperatorBackend using NumPy dense arrays.
    """

    @override
    def get_identity(self, dimension: int = 2) -> np.ndarray:
        """Generates an identity matrix of the specified dimension using numpy.

        Args:
            dimension: The size of the square matrix. Defaults to 2.

        Returns:
            np.ndarray: A complex128 identity matrix of shape (dimension, dimension).
        """
        return np.eye(dimension, dtype=complex)

    @override
    def get_pauli_x(self) -> np.ndarray:
        """Generates the 2x2 Pauli-X matrix.

        Returns:
            np.ndarray: A 2x2 complex128 array [[0, 1], [1, 0]].
        """
        return np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=complex,
        )

    @override
    def get_pauli_y(self) -> np.ndarray:
        """Generates the 2x2 Pauli-Y matrix.

        Returns:
            np.ndarray: A 2x2 complex128 array [[0, -1j], [1j, 0]].
        """
        return np.array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=complex,
        )

    @override
    def get_pauli_z(self) -> np.ndarray:
        """Generates the 2x2 Pauli-Z matrix.

        Returns:
            np.ndarray: A 2x2 complex128 array [[1, 0], [0, -1]].
        """
        return np.array(
            [
                [1, 0],
                [0, -1],
            ],
            dtype=complex,
        )

    @override
    def get_zero_matrix(self, dimension: int) -> np.ndarray:
        """Generates a square zero matrix of the specified dimension.

        Args:
            dimension: The size of the square matrix.

        Returns:
            np.ndarray: A matrix filled with zeros.
        """
        return np.zeros((dimension, dimension), dtype=complex)

    @override
    def kronecker_product(self, matrices: list[np.ndarray]) -> np.ndarray:
        """Computes the cumulative Kronecker product using np.kron.

        Args:
            matrices: A list of numpy arrays.

        Returns:
            np.ndarray: The resulting tensor product array.

        Raises:
            ValueError: If the list is empty.
        """
        if not matrices:
            raise ValueError("List of matrices cennot be empty.")

        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)

        return result

    @override
    def matrix_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes element-wise addition using np.add.

        Args:
            a: First numpy array.
            b: Second numpy array.

        Returns:
            np.ndarray: Sum of the arrays.
        """
        return np.add(a, b)

    @override
    def matrix_scale(self, matrix: np.ndarray, scalar: complex) -> np.ndarray:
        """Multiplies matrix by a scalar.

        Args:
            matrix: Input numpy array.
            scalar: Complex or float scalar.

        Returns:
            np.ndarray: Scaled array.
        """
        return matrix * scalar

    @override
    def adjoint(self, matrix: np.ndarray) -> np.ndarray:
        """Computes the conjugate transpose.

        Args:
            matrix: Input numpy array.

        Returns:
            np.ndarray: The conjugate transpose (Hermitian) of the input.
        """
        return matrix.conj().T

    @override
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes the matrix product of two matrices.

        Args:
            a: Left matrix.
            b: Right matrix.

        Returns:
            np.ndarray: The result of the multiplication A @ B.
        """
        return np.matmul(a, b)

    @override
    def inner_product(self, a: np.ndarray, b: np.ndarray) -> complex:
        """Computes the inner product (dot product) of two vectors/matrices.

        Mathematically: <a|b> = a† . b

        Args:
            a: The "bra" vector (will be conjugated).
            b: The "ket" vector.

        Returns:
            complex: The scalar result.
        """
        pass
        return complex(np.vdot(a, b))

    @override
    def diagonalize(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalizes a Hermitian matrix.

        Args:
            matrix: The matrix to diagonalize.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - eigenvalues: A list or array of real eigenvalues (sorted).
                - eigenvectors: The matrix where column 'i' is the eigenvector
                    for eigenvalue 'i'.
        """
        return np.linalg.eigh(matrix)
