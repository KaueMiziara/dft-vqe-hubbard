from typing import cast

import numpy as np
import pennylane as qml
import pytest
from pennylane.operation import Operator
from pennylane.ops import Adjoint, Prod, SProd  # Explicit imports for Pyright

from dft_vqe_hubbard.algebra.pennylane_backend import PennyLaneBackend


class TestPennyLaneBackend:
    """
    Test suite for PennyLaneBackend ensuring symbolic operator integrity.
    """

    @pytest.fixture
    def backend(self) -> PennyLaneBackend:
        """Provides an instance of the PennyLaneBackend."""
        return PennyLaneBackend()

    def test_pauli_properties(self, backend: PennyLaneBackend) -> None:
        """Verifies that basic Pauli operators are correctly instantiated."""
        x: Operator = backend.get_pauli_x()
        y: Operator = backend.get_pauli_y()
        z: Operator = backend.get_pauli_z()
        id_op: Operator = backend.get_identity()

        assert isinstance(x, qml.PauliX)
        assert isinstance(y, qml.PauliY)
        assert isinstance(z, qml.PauliZ)
        assert isinstance(id_op, qml.Identity)

        assert x.wires == qml.wires.Wires(0)  # type: ignore

    def test_kronecker_product(self, backend: PennyLaneBackend) -> None:
        """Tests that the kronecker product correctly maps to multi-qubit wires."""
        ops: list[Operator] = [
            backend.get_pauli_x(),
            backend.get_pauli_z(),
            backend.get_identity(),
        ]

        composite: Operator = backend.kronecker_product(ops)

        assert isinstance(composite, Prod)
        assert composite.wires == qml.wires.Wires([0, 1, 2])  # type: ignore

    def test_hermitian_adjoint(self, backend: PennyLaneBackend) -> None:
        """Verifies that the adjoint operation is symbolically registered."""
        x: Operator = backend.get_pauli_x()
        y: Operator = backend.get_pauli_y()

        scaled_y: Operator = backend.matrix_scale(y, -1j)
        sigma_plus: Operator = backend.matrix_add(x, scaled_y)

        adj_sigma: Operator = backend.adjoint(sigma_plus)

        assert isinstance(adj_sigma, Adjoint)
        assert adj_sigma.base == sigma_plus  # type: ignore

    def test_matrix_arithmetic(self, backend: PennyLaneBackend) -> None:
        """Tests addition and scaling for Hamiltonian construction."""
        z: Operator = backend.get_pauli_z()
        u: float = 4.0

        scaled_z: Operator = backend.matrix_scale(z, u)

        assert isinstance(scaled_z, SProd)
        assert np.isclose(cast(complex, scaled_z.scalar), 4.0)  # type: ignore

    def test_unsupported_methods(self, backend: PennyLaneBackend) -> None:
        """Ensures that unimplemented classical methods raise the correct error."""
        with pytest.raises(NotImplementedError):
            backend.diagonalize(backend.get_pauli_z())

        with pytest.raises(NotImplementedError):
            backend.inner_product(backend.get_pauli_x(), backend.get_pauli_x())

    def test_zero_matrix(self, backend: PennyLaneBackend) -> None:
        """Verifies the symbolic zero operator creation."""
        zero_op: Operator = backend.get_zero_matrix(dimension=16)

        assert isinstance(zero_op, SProd)
        assert cast(complex, zero_op.scalar) == 0.0  # type: ignore
