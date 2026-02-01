from abc import ABC, abstractmethod
from collections.abc import Sequence


class VariationalAnsatz(ABC):
    """
    Abstract Base Class defining the contract for a VQE Ansatz.

    An Ansatz represents a parameterized quantum circuit U(θ) that prepares
    a trial wavefunction |ψ(θ)⟩.
    """

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Returns the total number of variational parameters required.

        Returns:
            int: The size of the parameter vector theta.
        """
        pass

    @abstractmethod
    def apply(self, params: Sequence[float]) -> None:
        """Applies the ansatz operations to the current quantum context.

        Args:
            params: A sequence of floating-point parameters (angles/times).
                    Length must match self.num_parameters.
        """
        pass
