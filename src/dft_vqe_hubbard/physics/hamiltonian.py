from dft_vqe_hubbard.algebra.operator_backend import OperatorBackend
from dft_vqe_hubbard.physics.jordan_wigner import JordanWignerMapper


class FermiHubbardModel[MatrixType]:
    """
    Constructs the Fermi-Hubbard Hamiltonian matrix for a given lattice geometry.

    The Hamiltonian is defined as:
    H = -t * sum{c^dag_i c_j + h.c.} + U * sum{n_i_{up} * n_i_{down}}
    """

    def __init__(
        self,
        backend: OperatorBackend,
        mapper: JordanWignerMapper,
        n_sites: int,
        edges: list[tuple[int, int]],
    ) -> None:
        """Initializes the Fermi-Hubbard Model builder.

        Args:
            backend: An instance of OperatorBackend handling the low-level
                matrix generation.
            mapper: An instance of JordanWignerMapper to generate Fermionic operators.
            n_sites: The total number of spatial sites in the lattice (L). The system
                will contain 2*L qubits (spin-orbitals).
            edges: A list of tuples representing the connectivity of the lattice.
                Each tuple (i, j) represents a hopping connection between site
                i and site j.
        """
        self._backend = backend
        self._mapper = mapper
        self._n_sites = n_sites
        self._n_qubits = 2 * n_sites
        self._edges = edges

    def construct_kinetic_term(self, t: float) -> MatrixType:
        """Constructs the Kinetic (Hopping) Hamiltonian matrix.

        This term represents the movement of electrons between connected sites (i, j)
        preserving their spin (sigma).
        Formula: H_kin = -t * sum_{<i,j>, sigma} (c†_{i,sigma} c_{j,sigma} + h.c.)

        Args:
            t: The hopping amplitude (tunneling energy). A positive 't' typically
                implies a lower energy for delocalized states.

        Returns:
            MatrixType: The Hermitian matrix representing the kinetic energy operator
            in the computational basis.
        """
        n_dim = 2**self._n_qubits
        h_kin = self._backend.get_zero_matrix(n_dim)

        for i, j in self._edges:
            for spin in [0, 1]:
                u = self._get_qubit_index(i, spin)
                v = self._get_qubit_index(j, spin)

                c_dag_i = self._mapper.get_fermion_creation_operator(self._n_qubits, u)
                c_j = self._mapper.get_fermion_annihilation_operator(self._n_qubits, v)

                term_fwd = self._backend.matmul(c_dag_i, c_j)
                term_bwd = self._backend.adjoint(term_fwd)

                hopping = self._backend.matrix_add(term_fwd, term_bwd)
                weighted = self._backend.matrix_scale(hopping, -t)
                h_kin = self._backend.matrix_add(h_kin, weighted)

        return h_kin

    def construct_interaction_term(self, penalty: float) -> MatrixType:
        """Constructs the On-Site Interaction (Coulomb Repulsion) Hamiltonian matrix.

        This term imposes an energy penalty U whenever two electrons (one Up, one Down)
        occupy the same spatial site.
        Formula: H_int = U * sum_{i} (n_{i,up} * n_{i,down})

        Args:
            penalty: The on-site interaction strength (Hubbard U). Positive U implies
            repulsion (Mott physics), while negative U implies attraction.

        Returns:
            MatrixType: The diagonal matrix representing the interaction energy operator
            in the computational basis.
        """
        n_dim = 2**self._n_qubits
        h_int = self._backend.get_zero_matrix(n_dim)

        for i in range(self._n_sites):
            idx_up = self._get_qubit_index(i, 0)
            idx_dn = self._get_qubit_index(i, 1)

            c_dag_up = self._mapper.get_fermion_creation_operator(
                self._n_qubits, idx_up
            )
            c_up = self._mapper.get_fermion_annihilation_operator(
                self._n_qubits, idx_up
            )
            n_up = self._backend.matmul(c_dag_up, c_up)

            c_dag_dn = self._mapper.get_fermion_creation_operator(
                self._n_qubits, idx_dn
            )
            c_dn = self._mapper.get_fermion_annihilation_operator(
                self._n_qubits, idx_dn
            )
            n_dn = self._backend.matmul(c_dag_dn, c_dn)

            interaction = n_up @ n_dn

            weighted = self._backend.matrix_scale(interaction, penalty)
            h_int = self._backend.matrix_add(h_int, weighted)

        return h_int

    def construct_total_hamiltonian(self, t: float, penalty: float) -> MatrixType:
        """Constructs the total Fermi-Hubbard Hamiltonian.

        Formula: H_total = H_kin + H_int

        Args:
            t: The hopping amplitude.
            penalty: The on-site interaction strength.

        Returns:
            MatrixType: The sum of the kinetic and interaction matrices.
        """
        h_kin = self.construct_kinetic_term(t)
        h_int = self.construct_interaction_term(penalty)
        return self._backend.matrix_add(h_kin, h_int)

    def construct_total_number_operator(self) -> MatrixType:
        """Constructs the Total Number operator N = sum_i (n_i_up + n_i_dn).

        Returns:
            MatrixType: The diagonal matrix counting total electrons in the system.
        """
        n_dim = 2**self._n_qubits
        N_op = self._backend.get_zero_matrix(n_dim)

        for i in range(self._n_sites):
            idx_up = self._get_qubit_index(i, 0)
            idx_dn = self._get_qubit_index(i, 1)

            c_dag_up = self._mapper.get_fermion_creation_operator(
                self._n_qubits, idx_up
            )
            c_up = self._mapper.get_fermion_annihilation_operator(
                self._n_qubits, idx_up
            )
            n_up = self._backend.matmul(c_dag_up, c_up)

            c_dag_dn = self._mapper.get_fermion_creation_operator(
                self._n_qubits, idx_dn
            )
            c_dn = self._mapper.get_fermion_annihilation_operator(
                self._n_qubits, idx_dn
            )
            n_dn = self._backend.matmul(c_dag_dn, c_dn)

            N_op = self._backend.matrix_add(N_op, self._backend.matrix_add(n_up, n_dn))

        return N_op

    def construct_double_occupancy_operator(self) -> MatrixType:
        """Constructs the Double Occupancy operator D = sum_i (n_i_up * n_i_down).

        This is equivalent to the Interaction Hamiltonian with U=1.

        Returns:
            MatrixType: The diagonal matrix counting doubly occupied sites.
        """
        return self.construct_interaction_term(penalty=1.0)

    def construct_potential_term(self, potentials: list[float]) -> MatrixType:
        """Constructs the external potential energy matrix.

        In DFT, this represents the 'Hartree Potential' (mean-field interaction)
        plus any actual external fields.

        Formula: V = sum_{i,sigma} (v_{i,sigma} * n_{i,sigma})

        Args:
            potentials: A list of floats of length 2*L (total qubits).
                        potentials[k] is the energy penalty for occupying qubit k.

        Returns:
            MatrixType: A diagonal matrix representing this potential.
        """
        n_dim = 2**self._n_qubits
        v_matrix = self._backend.get_zero_matrix(n_dim)

        for q_idx, v_val in enumerate(potentials):
            if abs(v_val) < 1e-12:
                continue

            c_dag = self._mapper.get_fermion_creation_operator(self._n_qubits, q_idx)
            c = self._mapper.get_fermion_annihilation_operator(self._n_qubits, q_idx)
            n_op = self._backend.matmul(c_dag, c)

            term = self._backend.matrix_scale(n_op, v_val)

            v_matrix = self._backend.matrix_add(v_matrix, term)

        return v_matrix

    def _get_qubit_index(self, site_idx: int, spin: int) -> int:
        """Maps a spatial site index and spin projection to a linear qubit index.

        The mapping follows an interleaved ordering convention:
            - Site i, Spin Up (0)   -> Qubit 2*i
            - Site i, Spin Down (1) -> Qubit 2*i + 1

        Args:
            site_idx: The spatial site index (0 to L-1).
            spin: The spin projection, where 0 represents Spin Up and 1 is Spin Down.

        Returns:
            int: The linear index of the qubit (0 to 2L-1) corresponding to
                this spin-orbital.
        """
        return 2 * site_idx + spin
