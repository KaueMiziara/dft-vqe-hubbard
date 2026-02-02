from collections.abc import Callable
from typing import Any, override

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHEA
from dft_vqe_hubbard.quantum.vqe_solver import VQESolver


class PennyLaneVQESolver(VQESolver[Callable[..., None], qml.operation.Operator]):
    """
    PennyLane implementation of the VQE loop.
    """

    def __init__(self, ansatz: PennyLaneHEA, n_qubits: int) -> None:
        """
        Args:
            ansatz: The HEA ansatz instance.
            n_qubits: Total qubits in the system.
        """
        self._ansatz = ansatz
        self._n_qubits = n_qubits
        self._dev = qml.device("default.qubit", wires=n_qubits)

    @override
    def solve(
        self,
        hamiltonian: qml.operation.Operator,
        n_layers: int,
        learning_rate: float = 0.1,
        steps: int = 100,
        penalty_operator: qml.operation.Operator | None = None,
        target_value: float = 2.0,
        penalty_weight: float = 20.0,
    ) -> tuple[float, np.ndarray]:
        """Minimizes the energy using Gradient Descent.

        Args:
            hamiltonian: The operator to minimize.
            n_layers: Depth of the ansatz.
            learning_rate: Step size for the optimizer.
            steps: Number of optimization iterations.
            penalty_operator: An operator to constrain.
            target_value: The desired expectation value for the penalty_operator.
            penalty_weight: The strength of the penalty.

        Returns:
            tuple[float, np.ndarray]: (Final energy, Optimal parameters).
        """
        n_params = self._ansatz.get_n_parameters(n_layers)
        raw_params = np.random.uniform(
            low=0,
            high=2 * np.pi,
            size=n_params,
        )
        params = pnp.array(raw_params, requires_grad=True)

        @qml.qnode(self._dev)
        def quantum_circuit(p: Any) -> Any:
            circuit = self._ansatz.build_circuit(p, self._n_qubits, n_layers)
            circuit()

            if penalty_operator is not None:
                return qml.expval(hamiltonian), qml.expval(penalty_operator)
            return qml.expval(hamiltonian)

        def cost_fn(p: pnp.tensor) -> Any:
            results = quantum_circuit(p)

            if penalty_operator is not None:
                energy, n_val = results
                return (
                    pnp.real(energy)  # type:ignore
                    + penalty_weight * (pnp.real(n_val) - target_value) ** 2  # type:ignore
                )

            return pnp.real(results)  # type:ignore

        opt = qml.AdamOptimizer(stepsize=learning_rate)

        for i in range(steps):
            params, cost = opt.step_and_cost(cost_fn, params)
            if i % 10 == 0:
                print(f"Step {i:3d} | Energy: {cost:.6f}")

        final_res = quantum_circuit(params)
        final_energy = final_res[0] if penalty_operator is not None else final_res
        return float(pnp.real(final_energy)), params.numpy()  # type: ignore
