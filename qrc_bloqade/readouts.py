import numpy as np
from scipy.sparse import csc_matrix, kron, identity

def construct_pauli_z(qubit_index, num_qubits):
    """Constructs a sparse Pauli-Z operator for a specific qubit."""
    z_sparse = csc_matrix(([1, -1], ([0, 1], [0, 1])), shape=(2, 2))
    identity_left = identity(2**qubit_index, format="csc")
    identity_right = identity(2**(num_qubits - qubit_index - 1), format="csc")
    return kron(identity_left, kron(z_sparse, identity_right), format="csc")

def construct_pauli_zz(qubit_i, qubit_j, num_qubits):
    """Constructs a sparse ZZ correlation operator for two qubits."""
    # ZZ correlation is just the tensor product of Z_i and Z_j
    z_i = construct_pauli_z(qubit_i, num_qubits)
    z_j = construct_pauli_z(qubit_j, num_qubits)
    return z_i @ z_j  # Matrix multiplication of sparse matrices

def expectation(state, operator):
    """Computes the expectation value of an operator for a given quantum state."""
    op_state = operator @ state  # Sparse matrix-vector multiplication
    return np.real(np.vdot(state, op_state))

def z_expectation(state, qubit_index, num_qubits):
    """Computes the expectation value of Pauli-Z for a qubit in a given quantum state."""
    pauli_z_op = construct_pauli_z(qubit_index, num_qubits)
    return expectation(state, pauli_z_op)

def zz_expectation(state, qubit_i, qubit_j, num_qubits):
    """Computes the ZZ correlation between two qubits in a given quantum state."""
    pauli_zz_op = construct_pauli_zz(qubit_i, qubit_j, num_qubits)
    return expectation(state, pauli_zz_op)

class EnhancedReadout:
    def __init__(self, n_sites, readout_indices=None, correlation_pairs=None, 
                 time_points=None, include_zz=True, include_multi_time=False):
        """
        Enhanced readout for quantum reservoir computing.
        
        Args:
            n_sites: Number of qubits in the system
            readout_indices: Specific qubits to measure (defaults to all)
            correlation_pairs: Specific qubit pairs for ZZ correlations (defaults to nearest neighbors)
            time_points: List of time points to measure (defaults to final time only)
            include_zz: Whether to include ZZ correlations in measurement
            include_multi_time: Whether to include measurements at multiple time points
        """
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(n_sites))
        self.include_zz = include_zz
        self.include_multi_time = include_multi_time
        
        # Default correlation pairs: nearest neighbors
        if correlation_pairs is None and include_zz:
            self.correlation_pairs = [(i, i+1) for i in range(n_sites-1)]
            # Optional: Add some non-nearest neighbors for longer-range correlations
            for i in range(n_sites-2):
                if i % 2 == 0:  # Add every other next-nearest neighbor
                    self.correlation_pairs.append((i, i+2))
        else:
            self.correlation_pairs = correlation_pairs if correlation_pairs else []
        
        # Time points for measurement
        self.time_points = time_points
        
    def measure_single_state(self, state, num_qubits):
        """Measure a single quantum state."""
        features = []
        
        # Measure Z expectation values
        for idx in self.readout_indices:
            features.append(z_expectation(state, idx, num_qubits))
        
        # Measure ZZ correlations if enabled
        if self.include_zz:
            for i, j in self.correlation_pairs:
                features.append(zz_expectation(state, i, j, num_qubits))
                
        return features
        
    def measure(self, evolved_results, num_qubits):
        """
        Measure the expectation values and correlations.
        
        Args:
            evolved_results: List of evolved quantum states if include_multi_time=False,
                            or list of lists of states at different times if include_multi_time=True
            num_qubits: Number of qubits in the system
            
        Returns:
            features: Array of measured features
        """
        features = []
        
        if not self.include_multi_time:
            # Simplified case: measure only the final states
            for state in evolved_results:
                sample_features = self.measure_single_state(state, num_qubits)
                features.append(sample_features)
        else:
            # Multi-time measurement case
            for time_series in evolved_results:
                sample_features = []
                
                # If time_points specified, measure at those points
                if self.time_points is not None:
                    for t_idx in self.time_points:
                        state = time_series[t_idx]
                        time_features = self.measure_single_state(state, num_qubits)
                        sample_features.extend(time_features)
                else:
                    # Otherwise measure at the final time
                    state = time_series[-1]
                    sample_features = self.measure_single_state(state, num_qubits)
                    
                features.append(sample_features)
                
        return np.array(features)
    
class ZReadout:
    def __init__(self, n_sites, readout_indices=None):
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(self.n_sites))
    
    def measure(self, evolved_results, num_qubits):
        """Measure the expectation value of each spin in the Z-basis."""
        features = []
        for result in evolved_results:
            expectation_values = [z_expectation(result, i, num_qubits) for i in self.readout_indices]
            features.append(expectation_values)
        return np.array(features)

class ZZReadout:
    def __init__(self, n_sites, correlation_pairs=None):
        """
        Readout class for ZZ correlations.
        
        Args:
            n_sites: Number of qubits
            correlation_pairs: List of tuples specifying qubit pairs to measure correlations
        """
        self.n_sites = n_sites
        
        # Default: measure correlations between nearest neighbors
        if correlation_pairs is None:
            self.correlation_pairs = [(i, i+1) for i in range(n_sites-1)]
        else:
            self.correlation_pairs = correlation_pairs
    
    def measure(self, evolved_results, num_qubits):
        """Measure ZZ correlations between specified qubit pairs."""
        features = []
        for result in evolved_results:
            correlation_values = [zz_expectation(result, i, j, num_qubits) 
                                 for i, j in self.correlation_pairs]
            features.append(correlation_values)
        return np.array(features)

class MultiTimeReadout:
    def __init__(self, n_sites, readout_indices=None, time_points=None, include_zz=False):
        """
        Readout class for measurements at multiple time points.
        
        Args:
            n_sites: Number of qubits
            readout_indices: Specific qubits to measure
            time_points: List of time indices to measure at
            include_zz: Whether to include ZZ correlations
        """
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(n_sites))
        self.time_points = time_points  # If None, will use all time points
        self.include_zz = include_zz
        
        if include_zz:
            # Default correlation pairs: nearest neighbors
            self.correlation_pairs = [(i, i+1) for i in range(n_sites-1)]
    
    def measure(self, evolved_time_series, num_qubits):
        """
        Measure quantum states at multiple time points.
        
        Args:
            evolved_time_series: List of lists, where each inner list 
                                contains states at different time points
            num_qubits: Number of qubits
        """
        features = []
        
        for time_series in evolved_time_series:
            sample_features = []
            
            # Determine which time points to use
            if self.time_points is not None:
                time_indices = self.time_points
            else:
                time_indices = range(len(time_series))
            
            # For each time point
            for t_idx in time_indices:
                state = time_series[t_idx]
                
                # Measure Z expectations
                for qubit_idx in self.readout_indices:
                    sample_features.append(z_expectation(state, qubit_idx, num_qubits))
                
                # Measure ZZ correlations if enabled
                if self.include_zz:
                    for i, j in self.correlation_pairs:
                        sample_features.append(zz_expectation(state, i, j, num_qubits))
            
            features.append(sample_features)
            
        return np.array(features)