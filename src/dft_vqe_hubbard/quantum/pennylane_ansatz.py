from collections.abc import Callable
from typing import override

import numpy as np
import pennylane as qml

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.quantum.ansatz import VariationalAnsatz


class PennyLaneHEA(VariationalAnsatz[Callable[..., None], qml.operation.Operator]):
    """
    Hardware-Efficient Ansatz (HEA) implementation for PennyLane.

    This ansatz provides high expressivity by using generic rotation gates
    and a ring of entanglers to explore the Hilbert space.
    """

    def __init__(self, model: FermiHubbardModel) -> None:
        """
        Args:
            model: The Hubbard model instance containing the system specifications.
        """
        self._model = model
        self._h_kin = self._model.construct_kinetic_term(t=1.0)
        self._h_int = self._model.construct_interaction_term(penalty=1.0)

    @override
    def get_n_parameters(self, n_layers: int) -> int:
        """Returns the number of parameters required for the HEA.

        The HEA uses 2 rotation parameters (RX, RY) per qubit per layer.

        Args:
            n_layers: Number of repetitions of the variational circuit.

        Returns:
            int: Total parameter count (2 * n_qubits * n_layers).
        """
        return 2 * n_layers * self._model.n_qubits

    @override
    def build_circuit(
        self,
        params: list[float],
        n_qubits: int,
        n_layers: int,
    ) -> Callable[..., None]:
        """Returns a function that applies the HEA layers in a QNode.

        Args:
            params: Variational parameters [rx_0, ry_0, rx_1, ry_1, ...].
            n_qubits: Total number of qubits in the register.
            n_layers: Number of repetitions (depth) of the ansatz.

        Returns:
            Callable: A function representing the circuit operations.
        """

        def circuit_template() -> None:
            for i in range(n_qubits):
                qml.RY(np.pi / 4, wires=i)

            param_idx = 0
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(params[param_idx], wires=i)
                    qml.RY(params[param_idx + 1], wires=i)
                    param_idx += 2

                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return circuit_template
