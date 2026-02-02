from typing import Any

import numpy as np
import pennylane as qml
import pytest

from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHEA
from dft_vqe_hubbard.quantum.pennylane_vqe import PennyLaneVQESolver


class TestPennyLaneVQESolver:
    """
    Test suite for the PennyLane VQE Solver implementation.
    """

    @pytest.fixture
    def setup_vqe(self) -> tuple[PennyLaneVQESolver, qml.operation.Operator]:
        """Sets up a solver and a simple test Hamiltonian."""
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])

        ansatz = PennyLaneHEA(model)
        solver = PennyLaneVQESolver(ansatz, n_qubits=4)

        h_total = model.construct_total_hamiltonian(t=1.0, penalty=0.0)
        return solver, h_total

    def test_solve_convergence(self, setup_vqe: tuple[PennyLaneVQESolver, Any]) -> None:
        """Verifies that the solver reduces energy over iterations using a seed."""
        solver, h_total = setup_vqe

        np.random.seed(42)

        e_start, _ = solver.solve(h_total, n_layers=1, steps=1)
        e_end, _ = solver.solve(h_total, n_layers=1, steps=5)

        assert e_end <= e_start

    def test_gradient_flow(self, setup_vqe: tuple[PennyLaneVQESolver, Any]) -> None:
        """Ensures that the Parameter Shift Rule generates updates."""
        solver, h_total = setup_vqe
        np.random.seed(42)

        n_params = solver._ansatz.get_n_parameters(n_layers=1)
        initial_params = np.zeros(n_params)

        _, optimized_params = solver.solve(h_total, n_layers=1, steps=2)

        assert not np.allclose(initial_params, optimized_params, atol=1e-5)

    def test_output_contract(self, setup_vqe: tuple[PennyLaneVQESolver, Any]) -> None:
        """Checks the return types satisfy the solver interface."""
        solver, h_total = setup_vqe
        n_layers = 1
        energy, params = solver.solve(h_total, n_layers=n_layers, steps=2)

        expected_params = solver._ansatz.get_n_parameters(n_layers)

        assert isinstance(energy, float)
        assert isinstance(params, np.ndarray)
        assert len(params) == expected_params
