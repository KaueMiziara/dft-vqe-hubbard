from abc import ABC, abstractmethod
from typing import Any


class OperatorBackend(ABC):
    """
    Abstract Class defining the low-level linear algebra operations required
    to construct quantum operators.

    This adheres to the Dependency Injection principle, allowing us to swap
    the underlying math engine without changing the high-level physics logic.
    """

    @abstractmethod
    def get_identity(self, dimension: int = 2) -> Any:
        """Returns the Identity matrix of size (dimension x dimension)."""
        pass

    @abstractmethod
    def get_pauli_x(self) -> Any:
        """Returns the 2x2 Pauli-X matrix (bit-flip)."""
        pass

    @abstractmethod
    def get_pauli_y(self) -> Any:
        """Returns the 2x2 Pauli-Y matrix."""
        pass

    @abstractmethod
    def get_pauli_z(self) -> Any:
        """Returns the 2x2 Pauli-Z matrix (phase-flip)."""
        pass

    @abstractmethod
    def kronecker_product(self, matrices: list[Any]) -> Any:
        """
        Computes the cumulative Kronecker (Tensor) product of a list of matrices.
        result = A (x) B (x) C ...
        """
        pass

    @abstractmethod
    def matrix_add(self, a: Any, b: Any) -> Any:
        """Computes A + B."""
        pass

    @abstractmethod
    def matrix_scale(self, matrix: Any, scalar: complex) -> Any:
        """Computes scalar * A."""
        pass

    @abstractmethod
    def adjoint(self, matrix: Any) -> Any:
        """Computes the conjugate transpose (Hermitian adjoint) of the matrix."""
        pass
