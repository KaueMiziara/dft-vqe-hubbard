from abc import ABC, abstractmethod
from typing import TypeVar

CircuitType = TypeVar("CircuitType")
""" Generic type for the circuit/program object"""

OperatorType = TypeVar("OperatorType")
""" Generic type for the operator/Hamiltonian"""


class VariationalAnsatz[CircuitType, OperatorType](ABC):
    """
    Abstract base class for Variational Quantum Ansätze.
    """

    @abstractmethod
    def build_circuit(
        self,
        params: list[float],
        n_qubits: int,
        n_layers: int,
    ) -> CircuitType:
        """Constructs the variational circuit.

        Args:
            params: Flattened list of variational parameters.
            n_qubits: Total number of qubits in the system.
            n_layers: Number of repetitions (depth) of the ansatz.

        Returns:
            CircuitType: The framework-specific circuit representation.
        """
        pass

    @abstractmethod
    def get_n_parameters(self, n_layers: int) -> int:
        """Calculates the total number of parameters needed for the ansatz.

        Args:
            n_layers: Number of layers.

        Returns:
            int: Total parameter count.
        """
        pass
