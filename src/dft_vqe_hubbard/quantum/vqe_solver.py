from abc import ABC, abstractmethod

import numpy as np


class VQESolver[CircuitType, OperatorType](ABC):
    """
    Abstract base class for Variational Quantum Eigensolvers.
    """

    @abstractmethod
    def solve(
        self,
        hamiltonian: OperatorType,
        n_layers: int,
        learning_rate: float = 0.01,
        steps: int = 100,
        penalty_operator: OperatorType | None = None,
        target_value: float = 0.0,
        penalty_weight: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """Runs the VQE optimization loop.

        Args:
            hamiltonian: The operator to minimize.
            n_layers: Depth of the ansatz.
            learning_rate: Step size for the optimizer.
            steps: Number of optimization iterations.
            penalty_operator: An operator to constrain.
            target_value: The desired expectation value for the penalty_operator.
            penalty_weight: The strength of the penalty.

        Returns:
            Tuple[float, np.ndarray]: (Final energy, Optimal parameters).
        """
        pass
