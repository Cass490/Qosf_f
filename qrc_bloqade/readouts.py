import numpy as np
from scipy.sparse import csc_matrix, kron, identity

def construct_pauli_z(qubit_index, num_qubits):
    """Constructs a sparse Pauli-Z operator for a specific qubit."""
    z_sparse = csc_matrix(([1, -1], ([0, 1], [0, 1])), shape=(2, 2))
    identity_left = identity(2**qubit_index, format="csc")
    identity_right = identity(2**(num_qubits - qubit_index - 1), format="csc")
    return kron(identity_left, kron(z_sparse, identity_right), format="csc")

def expectation(state, qubit_index, num_qubits):
    """Computes the expectation value of Pauli-Z for a qubit in a given quantum state."""
    pauli_z_op = construct_pauli_z(qubit_index, num_qubits)
    z_state = pauli_z_op @ state  # Sparse matrix-vector multiplication
    return np.real(np.vdot(state, z_state))

class ZReadout:
    def __init__(self, n_sites, readout_indices=None):
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(self.n_sites))

    def measure(self, evolved_results, num_qubits):
        """Measure the expectation value of each spin in the Z-basis."""
        features = []
        for result in evolved_results:
            expectation_values = [expectation(result, i, num_qubits) for i in self.readout_indices]
            features.append(expectation_values)
        return np.array(features)
