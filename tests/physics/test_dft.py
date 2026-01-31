import numpy as np
import pytest

from dft_vqe_hubbard.algebra.numpy_backend import NumpyBackend
from dft_vqe_hubbard.physics.dft import LatticeDFT
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


class TestLatticeDFT:
    @pytest.fixture
    def setup_dft(self):
        """Fixture to initialize a 2-site Dimer model for DFT."""
        backend = NumpyBackend()
        mapper = JordanWignerMapper(backend)
        model = FermiHubbardModel(backend, mapper, n_sites=2, edges=[(0, 1)])
        return LatticeDFT(model, backend)

    def test_non_interacting_limit(self, setup_dft):
        """
        Test U=0 (Free Particle Limit).

        Physics:
        - Mean Field Theory is exact when U=0.
        - Ground State Energy should be -2t (for 2 electrons in lowest orbital).
        - Density should be uniform (0.5 per spin-orbital).
        """
        t, U = 1.0, 0.0

        energy, densities, converged = setup_dft.run_scf_loop(t, U)

        assert converged, "DFT loop should converge easily for U=0"

        # Check Energy (Expected: -2.0)
        np.testing.assert_allclose(
            energy,
            -2.0,
            atol=1e-5,
            err_msg="Energy for U=0 should match exact -2t result",
        )

        # Check Densities (Expected: [0.5, 0.5, 0.5, 0.5])
        expected_density = np.full(4, 0.5)
        np.testing.assert_allclose(
            densities,
            expected_density,
            atol=1e-5,
            err_msg="Density should be uniform at U=0",
        )

    def test_high_interaction_convergence(self, setup_dft):
        """
        Test U=4.0 (Strong Coupling).

        Physics:
        - The algorithm should still converge.
        - Energy will likely differ from the Exact Solution (Phase 2)
          because Single-Slater Determinant (DFT) cannot capture multi-reference
          correlations effectively.
        """
        t, U = 1.0, 4.0

        energy, densities, converged = setup_dft.run_scf_loop(t, U)

        assert converged, "DFT loop should converge even for moderate U"

        # It must be negative (system is stable) and finite
        assert -10.0 < energy < 0.0, "Energy should be within reasonable bounds"

        # Check Density Symmetry
        # For the dimer, site 0 and site 1 should still be symmetric
        # (n0_up + n0_down == n1_up + n1_down) roughly
        n_site_0 = densities[0] + densities[1]
        n_site_1 = densities[2] + densities[3]

        np.testing.assert_allclose(
            n_site_0,
            n_site_1,
            atol=1e-2,
            err_msg="Total density should be symmetric across sites",
        )
