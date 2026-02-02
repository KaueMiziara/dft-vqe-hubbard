from collections.abc import Callable
from typing import Any, override

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHVA
from dft_vqe_hubbard.quantum.vqe_solver import VQESolver


class PennyLaneVQESolver(VQESolver[Callable[..., None], qml.operation.Operator]):
    """
    PennyLane implementation of the VQE loop.
    """

    def __init__(self, ansatz: PennyLaneHVA, n_qubits: int) -> None:
        """
        Args:
            ansatz: The HVA ansatz instance.
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
    ) -> tuple[float, np.ndarray]:
        """Minimizes the energy using Gradient Descent.

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
            return qml.expval(hamiltonian)

        def cost_fn(p: pnp.tensor) -> Any:
            return pnp.real(quantum_circuit(p))  # type: ignore

        opt = qml.AdamOptimizer(stepsize=learning_rate)

        energy = 0.0
        for i in range(steps):
            params, energy = opt.step_and_cost(cost_fn, params)  # type: ignore
            if i % 10 == 0:
                print(f"Step {i:3d} | Energy: {energy:.6f}")

        return float(energy), params.numpy()  # type: ignore
