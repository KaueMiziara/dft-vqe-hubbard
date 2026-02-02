from collections.abc import Callable
from typing import override

import pennylane as qml

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.quantum.ansatz import VariationalAnsatz


class PennyLaneHVA(VariationalAnsatz[Callable[..., None], qml.operation.Operator]):
    """
    Hamiltonian Variational Ansatz (HVA) implementation for PennyLane.

    This ansatz alternates between the kinetic (hopping) and
    interaction (U) terms of the Hubbard Hamiltonian.
    """

    def __init__(self, model: FermiHubbardModel) -> None:
        """
        Args:
            model: The Hubbard model instance containing the Hamiltonian terms.
        """
        self._model = model
        self._h_kin = self._model.construct_kinetic_term(t=1.0)
        self._h_int = self._model.construct_interaction_term(penalty=1.0)

    @override
    def get_n_parameters(self, n_layers: int) -> int:
        """Returns the number of parameters.

        HVA for Hubbard typically uses 2 parameters per layer (theta_t, theta_u).
        """
        return 2 * n_layers

    @override
    def build_circuit(
        self,
        params: list[float],
        n_qubits: int,
        n_layers: int,
    ) -> Callable[..., None]:
        """Returns a function that applies the HVA layers in a QNode.

        Args:
            params: Variational parameters [t1, u1, t2, u2, ...].
            n_qubits: Number of qubits.
            n_layers: Number of layers.

        Returns:
            Callable: A function representing the circuit operations.
        """

        def circuit_template() -> None:
            qml.PauliX(0)
            qml.PauliX(1)

            for layer in range(n_layers):
                theta_t = params[2 * layer]
                theta_u = params[2 * layer + 1]

                qml.ApproxTimeEvolution(self._h_kin, theta_t, 1)  # type: ignore

                qml.ApproxTimeEvolution(self._h_int, theta_u, 1)  # type: ignore

        return circuit_template
