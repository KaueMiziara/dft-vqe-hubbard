import numpy as np
import pennylane as qml
import pytest

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHVA
from dft_vqe_hubbard.quantum.pennylane_backend import PennyLaneBackend


class TestPennyLaneHVA:
    @pytest.fixture
    def setup_ansatz(self):
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])

        ansatz = PennyLaneHVA(model, layers=1)
        return model, ansatz

    def test_parameter_count(self, setup_ansatz):
        _, ansatz = setup_ansatz
        assert ansatz.num_parameters == 2

        ansatz_2 = PennyLaneHVA(ansatz._model, layers=2)
        assert ansatz_2.num_parameters == 4

    def test_circuit_execution(self, setup_ansatz):
        """
        Verify that the ansatz can actually run inside a QNode.
        """
        model, ansatz = setup_ansatz
        n_qubits = model.n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params):
            # State Prep (Hartree-Fock / Mean Field starting point)
            # For Half-Filling Dimer (N=2), |0101> or |1100> etc.
            # Let's start with |1100> -> Qubits 0 and 1 active
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)

            ansatz.apply(params)

            return qml.expval(qml.PauliZ(0))

        params = [0.1, 0.2]

        result = circuit(params)
        assert isinstance(result, (float | np.ndarray))

    def test_backend_validation(self):
        """
        Ensure HVA rejects models built with the wrong backend (e.g. NumpyBackend).
        """
        from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend

        backend = NumpyBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[])  # type: ignore

        with pytest.raises(TypeError, match="PennyLaneBackend"):
            PennyLaneHVA(model)
