from collections.abc import Sequence
from typing import override

import pennylane as qml

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.quantum.ansatz import VariationalAnsatz
from dft_vqe_hubbard.quantum.pennylane_backend import PennyLaneBackend, PLOperator


class PennyLaneHVA(VariationalAnsatz):
    """
    Hamiltonian Variational Ansatz (HVA) implemented for PennyLane.

    This ansatz constructs layers of time-evolution operators corresponding
    to the non-commuting parts of the Hamiltonian (Kinetic and Interaction).
    """

    def __init__(
        self,
        model: FermiHubbardModel[PLOperator],
        layers: int = 1,
        initial_scale: float = 1.0,
    ) -> None:
        """
        Args:
            model: The FermiHubbardModel instance. MUST be initialized with
                   PennyLaneBackend.
            layers: The number of Trotter steps (depth of the circuit).
            initial_scale: Scaling factor for the base operators.

        Raises:
            TypeError: If the model's backend is not PennyLaneBackend.
        """
        if not isinstance(model._backend, PennyLaneBackend):
            raise TypeError(
                "PennyLaneHVA requires a FermiHubbardModel using PennyLaneBackend."
            )

        self._model = model
        self._layers = layers

        self._h_kin = model.construct_kinetic_term(t=initial_scale)
        self._h_int = model.construct_interaction_term(penalty=initial_scale)

    @property
    @override
    def num_parameters(self) -> int:
        """
        HVA has 2 parameters per layer:
        - 1 for Kinetic evolution time
        - 1 for Interaction evolution time
        """
        return 2 * self._layers

    @override
    def apply(self, params: Sequence[float]) -> None:
        """Applies the HVA layers using qml.ApproxTimeEvolution.

        Args:
            params: Flat list of parameters [t_kin_0, t_int_0, t_kin_1, t_int_1, ...].
        """
        if len(params) != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} parameters, got {len(params)}."
            )

        for layer in range(self._layers):
            idx_base = 2 * layer
            theta_kin = params[idx_base]
            theta_int = params[idx_base + 1]

            qml.ApproxTimeEvolution(self._h_kin, theta_kin, n=1)
            qml.ApproxTimeEvolution(self._h_int, theta_int, n=1)
