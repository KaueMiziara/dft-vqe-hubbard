from typing import NoReturn, cast, override

import pennylane as qml
import pennylane.operation as qml_op

from dft_vqe_hubbard.algebra.operator_backend import OperatorBackend

type PLOperator = qml_op.Operator


class PennyLaneBackend(OperatorBackend[PLOperator]):
    """
    A Symbolic Backend that translates linear algebra operations into
    PennyLane operator arithmetic.
    """

    @override
    def get_identity(self, dimension: int = 2) -> PLOperator:
        """Returns a symbolic Identity operator.

        Args:
            dimension: Ignored in symbolic context (defaults to single qubit).

        Returns:
            PLOperator: A qml.Identity instance on wire 0.
        """
        return qml.Identity(wires=0)

    @override
    def get_zero_matrix(self, dimension: int) -> PLOperator:
        """Returns a symbolic Zero operator (0 * I).

        Args:
            dimension: Ignored in symbolic context.

        Returns:
            PLOperator: A scalar product (0.0 * Identity).
        """
        return qml.s_prod(0.0, qml.Identity(wires=0))

    @override
    def get_pauli_x(self) -> PLOperator:
        """Returns a symbolic PauliX operator on wire 0.

        Returns:
             PLOperator: qml.PauliX(0).
        """
        return qml.PauliX(wires=0)

    @override
    def get_pauli_y(self) -> PLOperator:
        """Returns a symbolic PauliY operator on wire 0.

        Returns:
             PLOperator: qml.PauliY(0).
        """
        return qml.PauliY(wires=0)

    @override
    def get_pauli_z(self) -> PLOperator:
        """Returns a symbolic PauliZ operator on wire 0.

        Returns:
             PLOperator: qml.PauliZ(0).
        """
        return qml.PauliZ(wires=0)

    @override
    def kronecker_product(self, matrices: list[PLOperator]) -> PLOperator:
        """Computes the tensor product by mapping operators to distinct wires.

        Unlike matrix algebra where position implies index, here we explicit
        map the i-th operator in the list to wire 'i'.

        Args:
            matrices: A list of single-qubit operators. The operator at index 'i'
                      will be mapped to wire 'i'.

        Returns:
            PLOperator: The combined tensor product operator (e.g., X0 @ Z1 @ I2).

        Raises:
            ValueError: If the input list is empty.
        """
        if not matrices:
            raise ValueError("List of matrices cannot be empty.")

        tensor_ops: list[PLOperator] = []
        for i, op in enumerate(matrices):
            mapped_op = qml.map_wires(op, {0: i})
            tensor_ops.append(mapped_op)

        result = tensor_ops[0]
        for op in tensor_ops[1:]:
            result = cast(PLOperator, qml.prod(result, op))

        return result

    @override
    def matrix_add(self, a: PLOperator, b: PLOperator) -> PLOperator:
        """Symbolic addition of two operators.

        Args:
            a: Left operator.
            b: Right operator.

        Returns:
            PLOperator: A qml.ops.Sum representing (A + B).
        """
        return qml.sum(a, b)

    @override
    def matrix_scale(self, matrix: PLOperator, scalar: complex) -> PLOperator:
        """Symbolic scalar multiplication.

        Args:
            matrix: The operator to scale.
            scalar: The coefficient.

        Returns:
            PLOperator: A qml.ops.SProd representing (scalar * matrix).
        """
        return qml.s_prod(scalar, matrix)

    @override
    def adjoint(self, matrix: PLOperator) -> PLOperator:
        """Returns the symbolic adjoint (Hermitian conjugate).

        Args:
            matrix: The input operator.

        Returns:
            PLOperator: The adjoint operator.
        """
        return qml.adjoint(matrix)

    @override
    def matmul(self, a: PLOperator, b: PLOperator) -> PLOperator:
        """Symbolic matrix multiplication (Operator Product).

        Args:
            a: Left operator.
            b: Right operator.

        Returns:
            PLOperator: A qml.ops.Prod representing (A @ B).
        """
        return cast(PLOperator, qml.prod(a, b))

    @override
    def inner_product(self, a: PLOperator, b: PLOperator) -> complex:
        """Not Supported: Symbolic backends cannot compute vector inner products.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Symbolic backend cannot compute inner product directly."
        )

    @override
    def diagonalize(self, matrix: PLOperator) -> NoReturn:
        """Not Supported: Symbolic backends rely on VQE for eigenvalue estimation.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Symbolic backend does not support direct diagonalization."
        )

    @override
    def get_column_vector(self, matrix: PLOperator, col_index: int) -> NoReturn:
        """Not Supported: Symbolic operators do not have accessible columns.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Symbolic backend has no column vectors.")
