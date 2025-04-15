import numpy as np
from scipy.sparse import csc_matrix, kron, identity #csc=compressed column matrix

def construct_pauli_z(qubit_index, num_qubits):

    z_sparse = csc_matrix(([1, -1], ([0, 1], [0, 1])), shape=(2, 2))#single qubit pauli z matrix
    identity_left = identity(2**qubit_index, format="csc")
    identity_right = identity(2**(num_qubits - qubit_index - 1), format="csc") #embedding into full hilbert space
    return kron(identity_left, kron(z_sparse, identity_right), format="csc")

def construct_pauli_zz(qubit_i, qubit_j, num_qubits):

    # ZZ correlation is just the tensor product of Z_i and Z_j
    z_i = construct_pauli_z(qubit_i, num_qubits)
    z_j = construct_pauli_z(qubit_j, num_qubits)
    return z_i @ z_j  # Matrix multiplication of sparse matrices=tensor prodcut of operators here since we already mapped it to hilbert space

def expectation(state, operator):
  
    op_state = operator @ state 
    return np.real(np.vdot(state, op_state)) #inner product

def z_expectation(state, qubit_index, num_qubits):
   
    pauli_z_op = construct_pauli_z(qubit_index, num_qubits)
    return expectation(state, pauli_z_op)

def zz_expectation(state, qubit_i, qubit_j, num_qubits):
  
    pauli_zz_op = construct_pauli_zz(qubit_i, qubit_j, num_qubits)
    return expectation(state, pauli_zz_op)

class EnhancedReadout:
    def __init__(self, n_sites, readout_indices=None, correlation_pairs=None, 
                 time_points=None, include_zz=True, include_multi_time=False):
    
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(n_sites))
        self.include_zz = include_zz
        self.include_multi_time = include_multi_time
        
        # Default correlation pairs: nearest neighbors
        if correlation_pairs is None and include_zz:
            self.correlation_pairs = [(i, i+1) for i in range(n_sites-1)]
            for i in range(n_sites-2):
                if i % 2 == 0: 
                    self.correlation_pairs.append((i, i+2))
        else:
            self.correlation_pairs = correlation_pairs if correlation_pairs else []
        self.time_points = time_points
        
    def measure_single_state(self, state, num_qubits): #z onn each qubit and zz on corelation pair
              # Debug prints
        print(f"DEBUG: Measuring state with shape {state.shape}")
        print(f"DEBUG: Readout indices: {self.readout_indices}")
        print(f"DEBUG: Correlation pairs: {self.correlation_pairs}")
        features = []
        for idx in self.readout_indices:
            features.append(z_expectation(state, idx, num_qubits))
            print(f"DEBUG: Z features (count {len(features)}): {features}")
        if self.include_zz:
            for i, j in self.correlation_pairs:
                features.append(zz_expectation(state, i, j, num_qubits))
        print(f"DEBUG: Total features generated: {len(features)}")       
        return features
        
    def measure(self, evolved_results, num_qubits):
   
        print("DEBUG: Entering measure method")
        print(f"Type of evolved_results: {type(evolved_results)}")
        print(f"Length of evolved_results: {len(evolved_results)}")
    
    # Very early check of the first element
        if len(evolved_results) > 0:
          first_elem = evolved_results[0]
          print(f"Type of first element: {type(first_elem)}")
        
          if isinstance(first_elem, list):
            print(f"Length of first inner list: {len(first_elem)}")
            
            if len(first_elem) > 0:
                first_state = first_elem[0]
                print(f"Type of first state: {type(first_state)}")
                
                if hasattr(first_state, 'shape'):
                    print(f"Shape of first state: {first_state.shape}")
    
    # Deliberately raise an exception to stop and show all debug info
  
        print(f"Type of evolved_results: {type(evolved_results)}")
        print(f"Length of evolved_results: {len(evolved_results)}")
        if len(evolved_results) > 0:
         print(f"Type of first element: {type(evolved_results[0])}")
         print(f"Length of first element: {len(evolved_results[0])}")
        if len(evolved_results[0]) > 0:
            print(f"Shape of state at [0][0]: {evolved_results[0][0].shape if hasattr(evolved_results[0][0], 'shape') else 'No shape attribute'}")
        if len(evolved_results) > 0:
         print(f"Type of first element: {type(evolved_results[0])}")
        
        # Check if we have a state vector or a list
        if hasattr(evolved_results[0], 'shape'):
            print(f"Shape of first state: {evolved_results[0].shape}")
            is_direct_state_list = True
        else:
            print(f"Length of first element: {len(evolved_results[0])}")
            is_direct_state_list = False
            
            if len(evolved_results[0]) > 0:
                print(f"Type of nested element: {type(evolved_results[0][0])}")
                if hasattr(evolved_results[0][0], 'shape'):
                    print(f"Shape of first nested state: {evolved_results[0][0].shape}")
    
        features = []
        
        if not self.include_multi_time:
          if len(evolved_results) > 0 and not isinstance(evolved_results[0], list):
            # Simplified case: measure only the final states
            for state in evolved_results:
                sample_features = self.measure_single_state(state, num_qubits)
                features.append(sample_features)
          else:
           for batch in evolved_results:
             batch_features=[]
             for state in batch:
                sample_features = self.measure_single_state(state, num_qubits)
                batch_features.append(sample_features)
             features.append(batch_features)
        else:
            for time_series in evolved_results:
                sample_features = []
        
                # If time_points specified, measure at those points
                if self.time_points is not None:
                    for t_idx in self.time_points:
                      if t_idx < len(time_series):
                        state = time_series[t_idx]
                        time_features = self.measure_single_state(state, num_qubits)
                        sample_features.extend(time_features)
                      else:
                          print(f"Time index {t_idx} is out of range for time series of length {len(time_series)}")
                else:
                    # Otherwise measure at the final time
                  if len(time_series) > 0:
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
        
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(n_sites))
        self.time_points = time_points  # If None, will use all time points
        self.include_zz = include_zz
        
        if include_zz:
            # Default correlation pairs: nearest neighbors
            self.correlation_pairs = [(i, i+1) for i in range(n_sites-1)]
    
    def measure(self, evolved_time_series, num_qubits):
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