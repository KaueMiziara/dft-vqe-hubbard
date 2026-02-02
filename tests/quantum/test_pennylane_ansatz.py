from collections.abc import Callable
from typing import Any

import pennylane as qml
import pytest

from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHEA


class TestPennyLaneHEA:
    """
    Test suite for the PennyLane Hardware-Efficient Ansatz.
    """

    @pytest.fixture
    def setup_hea(self) -> PennyLaneHEA:
        """Sets up a PennyLaneHEA instance for a 2-site Hubbard model."""
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(
            backend=backend,
            mapper=mapper,
            n_sites=2,
            edges=[(0, 1)],
        )
        return PennyLaneHEA(model)

    def test_parameter_count(self, setup_hea: PennyLaneHEA) -> None:
        """Verifies the HEA parameter logic (2 parameters * n_qubits * n_layers)."""
        n_layers = 3
        n_qubits = 4

        # 2 (RX, RY) * 4 qubits * 3 layers = 24
        expected_params = 2 * n_qubits * n_layers
        assert setup_hea.get_n_parameters(n_layers) == expected_params

    def test_circuit_construction(self, setup_hea: PennyLaneHEA) -> None:
        """Ensures the build_circuit returns a callable and executes in a QNode."""
        n_layers = 1
        n_qubits = 4
        n_params = setup_hea.get_n_parameters(n_layers)
        params = [0.1] * n_params

        circuit_fn = setup_hea.build_circuit(params, n_qubits, n_layers)
        assert isinstance(circuit_fn, Callable)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def test_qnode() -> Any:
            circuit_fn()
            return qml.expval(qml.PauliZ(0))

        result = test_qnode()
        assert result is not None

    def test_generic_scaling(self) -> None:
        """Tests if the ansatz correctly scales parameters with different sites."""
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)

        # 3 sites = 6 qubits
        model = FermiHubbardModel(
            backend=backend, mapper=mapper, n_sites=3, edges=[(0, 1), (1, 2)]
        )

        hea = PennyLaneHEA(model)
        n_layers = 5

        assert hea._model.n_qubits == 6

        # 2 (RX, RY) * 6 qubits * 5 layers = 60
        assert hea.get_n_parameters(n_layers=n_layers) == 60
