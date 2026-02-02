from collections.abc import Sequence
from typing import cast, override

import numpy as np
import pennylane as qml
import pennylane.math as qml_math
from pennylane import numpy as pnp
from pennylane.measurements import ExpectationMP
from pennylane.operation import Operator

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.quantum.ansatz import VariationalAnsatz
from dft_vqe_hubbard.quantum.pennylane_backend import PennyLaneBackend, PLOperator
from dft_vqe_hubbard.quantum.solver.result import VQEResult
from dft_vqe_hubbard.quantum.solver.solver import VQESolver


class PennyLaneVQESolver(VQESolver):
    """
    Concrete implementation of VQE using PennyLane.

    This solver constructs the symbolic Hamiltonian using the FermiHubbardModel,
    sets up a QNode for energy estimation, and minimizes the cost function
    using gradient-based optimization.
    """

    def __init__(self, model: FermiHubbardModel[PLOperator]):
        """
        Args:
            model: A FermiHubbardModel initialized with PennyLaneBackend.

        Raises:
            TypeError: If the model does not use PennyLaneBackend.
        """
        if not isinstance(model._backend, PennyLaneBackend):
            raise TypeError(
                "PennyLaneVQESolver requires a model with PennyLaneBackend."
            )

        self._model = model
        self._n_qubits = model.n_qubits

    @override
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
        raw_hamiltonian = self._model.construct_total_hamiltonian(t, u)
        hamiltonian = cast(Operator, qml.simplify(raw_hamiltonian))

        dev = qml.device("default.qubit", wires=self._n_qubits)

        @qml.qnode(dev)
        def circuit(params: np.ndarray) -> ExpectationMP:
            # Anti-Ferromagnetic |1001>
            qml.PauliX(wires=0)
            qml.PauliX(wires=3)

            ansatz.apply(cast(Sequence[float], params))
            return qml.expval(hamiltonian)

        def cost_fn(params) -> float:
            val = circuit(params)
            return qml_math.real(val)

        opt = qml.AdamOptimizer(stepsize=step_size)
        theta = pnp.array(initial_params, requires_grad=True)

        energy_history: list[float] = []
        converged = False

        current_energy = float(cost_fn(theta))
        energy_history.append(current_energy)

        for _ in range(max_iter):
            theta, _ = opt.step_and_cost(cost_fn, theta)

            new_energy_val = float(cost_fn(theta))
            energy_history.append(new_energy_val)

            if abs(new_energy_val - current_energy) < 1e-6:
                converged = True
                current_energy = new_energy_val
                break

            current_energy = new_energy_val

        final_params = np.array(theta).tolist()

        return VQEResult(
            energy=current_energy,
            optimal_params=final_params,
            history=energy_history,
            converged=converged,
        )
