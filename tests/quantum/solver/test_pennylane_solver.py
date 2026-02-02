import pytest

from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper
from dft_vqe_hubbard.quantum.pennylane_ansatz import PennyLaneHVA
from dft_vqe_hubbard.quantum.pennylane_backend import PennyLaneBackend
from dft_vqe_hubbard.quantum.solver.pennylane_solver import PennyLaneVQESolver
from dft_vqe_hubbard.quantum.solver.result import VQEResult


class TestPennyLaneVQE:
    @pytest.fixture
    def setup_system(self):
        """
        Fixture to create the full VQE stack:
        Backend -> Model -> Ansatz -> Solver
        """
        backend = PennyLaneBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])

        ansatz = PennyLaneHVA(model, layers=2)
        solver = PennyLaneVQESolver(model)

        return solver, ansatz

    def test_vqe_single_particle(self, setup_system):
        """
        Test that VQE finds the ground state for a SINGLE electron (N=1).

        Physics:
        - For N=1, the Hamiltonian is just a 2x2 hopping matrix.
        - Ground State Energy = -1.0 * t = -1.0.
        """
        solver, ansatz = setup_system

        initial_params = [0.4, 0.0] * ansatz._layers

        result = solver.solve(
            t=1.0,
            u=0.0,
            ansatz=ansatz,
            initial_params=initial_params,
            max_iter=50,
            step_size=0.1,
        )

        assert isinstance(result, VQEResult)

        # E = 0.0 (Classical localized state)
        # E = -1.0 (One electron hopping, one stuck)
        # E = -2.0 (True Ground State)
        assert result.energy < -0.8, (
            f"VQE failed to lower energy significantly: {result.energy}"
        )

    def test_vqe_interaction_runs(self, setup_system):
        """
        Test that the loop executes correctly for U=4.0.
        We don't assert exact energy here (requires deeper circuits),
        but we check that the optimization loop produces valid history.
        """
        solver, ansatz = setup_system
        initial_params = [0.05] * ansatz.num_parameters

        result = solver.solve(
            t=1.0,
            u=4.0,
            ansatz=ansatz,
            initial_params=initial_params,
            max_iter=5,
        )

        assert len(result.history) > 0
        assert isinstance(result.energy, float)
        assert isinstance(result.optimal_params, list)

        assert result.optimal_params != initial_params

    def test_solver_backend_validation(self):
        """
        Ensure the solver rejects models built with the wrong backend.
        """
        from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend

        backend = NumpyBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[])  # type: ignore

        with pytest.raises(TypeError, match="PennyLaneBackend"):
            PennyLaneVQESolver(model)
