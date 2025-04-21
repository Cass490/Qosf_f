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

        return H.toarray()  


# Add this function to modify the rydberg class
def update_hamiltonian_class():
    """
    Update the rydberg class to handle array-valued omega parameters for local encoding.
    
    Call this function before running the LSTM test to ensure the hamiltonian class
    can handle local encoding.
    """
    from qrc_bloqade.hamiltonian import rydberg
    
    # Create a new method to handle array-valued omega
    def construct_hamiltonian_updated(self):
        """Builds the full Hamiltonian for the system, supporting array-valued omega."""
        from scipy.sparse import csr_matrix
        import numpy as np
        
        H = csr_matrix((2**self.n_sites, 2**self.n_sites), dtype=complex)

        # Check if omega is array-like (from local_encode)
        if isinstance(self.omega, (list, np.ndarray)):
            # Apply site-specific omega values
            if len(self.omega) != self.n_sites:
                raise ValueError(f"Expected {self.n_sites} omega values, got {len(self.omega)}")
                
            for i in range(self.n_sites):
                H += self.omega[i] * self.construct_operator(self.sigma_x, [i])
        else:
            # Apply same omega to all sites
            for i in range(self.n_sites):
                H += self.omega * self.construct_operator(self.sigma_x, [i])
        
        # Apply detuning and interaction terms
        for i in range(self.n_sites):
            H += self.detuning_values[i] * self.construct_operator(self.sigma_z, [i])
            
            if i < self.n_sites - 1:
                H += self.V * self.construct_operator(self.sigma_z, [i, i + 1])

        return H.toarray()
    
    # Update the class method
    rydberg.construct_hamiltonian = construct_hamiltonian_updated
    
    print("Rydberg hamiltonian class updated to support array-valued omega parameters.")