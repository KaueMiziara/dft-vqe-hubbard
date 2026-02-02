from collections.abc import Callable
from typing import Any

import pennylane as qml
import pytest

from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHVA


class TestPennyLaneHVA:
    """
    Test suite for the PennyLane Hamiltonian Variational Ansatz.
    """

    @pytest.fixture
    def setup_hva(self) -> PennyLaneHVA:
        """Sets up a PennyLaneHVA instance for a 2-site Hubbard model."""
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(
            backend=backend, mapper=mapper, n_sites=2, edges=[(0, 1)]
        )
        return PennyLaneHVA(model)

    def test_parameter_count(self, setup_hva: PennyLaneHVA) -> None:
        """Verifies the HVA parameter logic (2 parameters per layer)."""
        n_layers = 3
        expected_params = 6  # 2 * 3
        assert setup_hva.get_n_parameters(n_layers) == expected_params

    def test_circuit_construction(self, setup_hva: PennyLaneHVA) -> None:
        """Ensures the build_circuit returns a callable and executes in a QNode."""
        n_layers = 1
        params = [0.1, 0.2]
        n_qubits = 4

        circuit_fn = setup_hva.build_circuit(params, n_qubits, n_layers)
        assert isinstance(circuit_fn, Callable)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def test_qnode() -> Any:
            circuit_fn()
            return qml.expval(qml.PauliZ(0))

        result = test_qnode()
        assert result is not None

    def test_generic_scaling(self) -> None:
        """Tests if the HVA correctly scales with different site counts."""
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(
            backend=backend, mapper=mapper, n_sites=3, edges=[(0, 1), (1, 2)]
        )
        hva = PennyLaneHVA(model)

        assert hva._model.n_qubits == 6  # type: ignore
        assert hva.get_n_parameters(n_layers=5) == 10
