import pennylane as qml
import pennylane.ops as qml_ops
import pytest

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_backend import PennyLaneBackend


class TestPennyLaneBackend:
    @pytest.fixture
    def setup_backend(self) -> tuple[FermiHubbardModel, PennyLaneBackend]:
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])
        return model, backend

    def test_kinetic_term_structure(self, setup_backend):
        """
        Verify that the kinetic term generates the correct Pauli strings.
        """
        model, _ = setup_backend

        kinetic_op = model.construct_kinetic_term(t=1.0)

        simplified = qml.simplify(kinetic_op)

        op_str = str(simplified)

        # Hopping between Site 0 (qubits 0,1) and Site 1 (qubits 2,3)
        # Spin Up hopping: 0 <-> 2. Spin Down hopping: 1 <-> 3.
        # We expect terms involving X0, X2, Y0, Y2 (Up) and X1, X3, Y1, Y3 (Down)

        assert "X(0)" in op_str and "X(2)" in op_str
        assert "Y(0)" in op_str and "Y(2)" in op_str

        assert "X(1)" in op_str and "X(3)" in op_str
        assert "Y(1)" in op_str and "Y(3)" in op_str

    def test_full_hamiltonian_types(self, setup_backend):
        """
        Verify that the full Hamiltonian simplifies to a valid PennyLane observable.
        """
        model, _ = setup_backend

        H = model.construct_total_hamiltonian(t=1.0, penalty=2.0)

        simplified_H = qml.simplify(H)

        valid_types = (qml_ops.Sum, qml_ops.Prod, qml_ops.SProd, qml.Hamiltonian)

        assert isinstance(simplified_H, valid_types)

    def test_interaction_term_diagonal(self, setup_backend):
        """
        Verify the interaction term is purely diagonal (Z operators).
        """
        model, _ = setup_backend

        interaction_op = model.construct_interaction_term(penalty=5.0)

        simplified = qml.simplify(interaction_op)
        op_str = str(simplified)

        # Interaction is n_up * n_down. Number operators map to (I - Z)/2.
        # Therefore, we should ONLY see 'Z' Pauli operators (and Identity).
        # We should NOT see X or Y.

        assert "Z" in op_str
        assert "X" not in op_str
        assert "Y" not in op_str
