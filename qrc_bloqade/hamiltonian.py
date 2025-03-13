import numpy as np
from scipy.sparse import identity, kron, csr_matrix

class rydberg:
    def __init__(self, n_sites, omega, V, detuning_values=None):
        self.n_sites = n_sites
        self.omega = omega
        self.V = V  # Interaction parameter
        self.I = csr_matrix(np.eye(2))  # Identity matrix (sparse)
        self.sigma_x = csr_matrix(np.array([[0, 1], [1, 0]]))
        self.sigma_z = csr_matrix(np.array([[1, 0], [0, -1]]))
        
        if detuning_values is None:
            self.detuning_values = np.zeros(n_sites)
        else:
            if len(detuning_values) != n_sites:
                raise ValueError("detuning_values must match the number of qubits")
            self.detuning_values = np.array(detuning_values)

    def construct_operator(self, op, qubits):
        n = self.n_sites
        full_operator = identity(1, format="csr")  # Start with identity
        
        for i in range(n):
            if i in qubits:
                full_operator = kron(full_operator, op, format="csr")
            else:
                full_operator = kron(full_operator, self.I, format="csr")
        
        return full_operator

    def construct_hamiltonian(self):
        """Builds the full Hamiltonian for the system."""
        H = csr_matrix((2**self.n_sites, 2**self.n_sites), dtype=complex)

        for i in range(self.n_sites):
            H += self.omega * self.construct_operator(self.sigma_x, [i])
            H += self.detuning_values[i] * self.construct_operator(self.sigma_z, [i])
            
            if i < self.n_sites - 1:
                H += self.V * self.construct_operator(self.sigma_z, [i, i + 1])

        return H.toarray()  # Convert sparse matrix to dense (optional)


