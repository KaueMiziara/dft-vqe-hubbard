import numpy as np

from dft_vqe_hubbard.algebra.operator_backend import OperatorBackend
from dft_vqe_hubbard.physics.hamiltonian import FermiHubbardModel


class LatticeDFT[MatrixType]:
    """
    Implements the Classical Lattice DFT algorithm for the Hubbard Model.
    """

    def __init__(
        self,
        model: FermiHubbardModel[MatrixType],
        backend: OperatorBackend[MatrixType],
    ) -> None:
        """
        Args:
            model: The generic Hubbard model instance.
            backend: The generic linear algebra backend.
        """
        self._model = model
        self._backend = backend

    def run_scf_loop(
        self,
        t: float,
        u: float,
        max_iter: int = 100,
        tol: float = 1e-5,
        mixing: float = 0.5,
    ) -> tuple[float, np.ndarray, bool]:
        """Executes the Self-Consistent Field (SCF) iteration.

        Args:
            t: Hopping parameter.
            u: Interaction parameter.
            max_iter: Limit to prevent infinite loops.
            tol: Convergence threshold for density change.
            mixing: Damping factor (0 to 1) to stabilize convergence.

        Returns:
            Tuple containing:
            - Final Total Energy (corrected for double counting)
            - Final Density array
            - Boolean indicating if convergence was reached
        """
        n_qubits = self._model.n_qubits

        current_density = np.random.rand(n_qubits)
        current_density = current_density * ((n_qubits / 2) / np.sum(current_density))

        final_energy: float = 0.0
        converged = False

        h_kin = self._model.construct_kinetic_term(t)
        N_op = self._model.construct_total_number_operator()

        for _ in range(max_iter):
            v_vals = self._calculate_hartree_potential(current_density, u)
            v_mat = self._model.construct_potential_term(v_vals)
            h_ks = self._backend.matrix_add(h_kin, v_mat)

            eigenvalues, eigenvectors = self._backend.diagonalize(h_ks)

            psi_ground = None
            e_ks_ground = 0.0

            found = False
            for idx, en in enumerate(eigenvalues):
                psi = self._backend.get_column_vector(eigenvectors, idx)

                n_op_psi = self._backend.matmul(N_op, psi)
                n_total = self._backend.inner_product(psi, n_op_psi).real

                if np.isclose(n_total, n_qubits / 2, atol=1e-2):
                    psi_ground = psi
                    e_ks_ground = en
                    found = True
                    break

            if not found:
                psi_ground = self._backend.get_column_vector(eigenvectors, 0)
                e_ks_ground = eigenvalues[0]

            if psi_ground is None:
                raise RuntimeError("Ground state is None.")

            new_density = self._get_density(psi_ground)

            diff = np.linalg.norm(new_density - current_density)
            if diff < tol:
                converged = True
                final_energy = e_ks_ground
                current_density = new_density
                break

            final_energy = e_ks_ground

            current_density = (mixing * new_density) + ((1 - mixing) * current_density)

        correction = 0.0
        n_sites = self._model.n_sites
        for i in range(n_sites):
            idx_up = self._model.get_qubit_index(i, 0)
            idx_dn = self._model.get_qubit_index(i, 1)
            n_up = current_density[idx_up].real
            n_dn = current_density[idx_dn].real
            correction += u * n_up * n_dn

        total_energy = final_energy - correction

        return total_energy, current_density, converged

    def _get_density(self, psi: MatrixType) -> np.ndarray:
        """Calculates the electron density <n> for every spin-orbital.

        Args:
            psi: The current wavefunction (vector).

        Returns:
            np.ndarray: Array of size 2*L (densities for each qubit).
        """
        densities = []
        n_qubits = self._model.n_qubits

        for q_idx in range(n_qubits):
            n_op = self._model.construct_number_operator(q_idx)
            n_psi = self._backend.matmul(n_op, psi)
            val = self._backend.inner_product(psi, n_psi).real
            densities.append(val)

        return np.array(densities)

    def _calculate_hartree_potential(
        self,
        densities: np.ndarray,
        u: float,
    ) -> list[float]:
        """Derives the effective potential from the current densities.

        Mean-Field Approximation:
            The potential felt by a Spin-Up electron at site 'i' is
            U * (Density of Spin-Down at site 'i').

        Args:
            densities: Current density array of size 2*L.
            u: Interaction strength.

        Returns:
            List[float]: The effective potential v_i for each qubit.
        """
        n_qubits = self._model.n_qubits
        potentials = np.zeros(n_qubits)
        n_sites = self._model.n_sites

        for i in range(n_sites):
            idx_up = self._model.get_qubit_index(i, 0)
            idx_dn = self._model.get_qubit_index(i, 1)

            n_up = float(densities[idx_up].real)
            n_dn = float(densities[idx_dn].real)

            potentials[idx_up] = u * n_dn
            potentials[idx_dn] = u * n_up

        return list(potentials)
