from abc import ABC, abstractmethod
from collections.abc import Sequence

from dft_vqe_hubbard.quantum.ansatz import VariationalAnsatz
from dft_vqe_hubbard.quantum.solver.result import VQEResult


class VQESolver(ABC):
    """
    Abstract Base Class for Variational Quantum Eigensolvers.

    Defines the contract for minimizing the expectation value of a Hamiltonian
    using a parameterized ansatz.
    """

    @abstractmethod
    def solve(
        self,
        t: float,
        u: float,
        ansatz: VariationalAnsatz,
        initial_params: Sequence[float],
        max_iter: int = 100,
        step_size: float = 0.1,
    ) -> VQEResult:
        """
        Executes the VQE optimization loop.

        Args:
            t: Hopping parameter (Kinetic Energy).
            u: Interaction parameter (Hubbard U).
            ansatz: The parameterized quantum circuit wrapper.
            initial_params: Starting values for the variational parameters.
            max_iter: Maximum number of optimization steps.
            step_size: Learning rate for the gradient descent.

        Returns:
            VQEResult: The optimization outcome containing final energy and parameters.
        """
        pass
