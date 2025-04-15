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
                 time_points=None, include_zz=True, include_multi_time=False, return_history=True):
    
        self.n_sites = n_sites
        self.readout_indices = readout_indices if readout_indices is not None else list(range(n_sites))
        self.include_zz = include_zz
        self.include_multi_time = include_multi_time
        self.time_points = time_points
        # Default correlation pairs: nearest neighbors
        if correlation_pairs is None and include_zz:
            self.correlation_pairs = []
            # Nearest neighbors
            for i in range(n_sites-1):
                self.correlation_pairs.append((i, i+1))
            # Next-nearest neighbors (every other pair)
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
         features=[]
         for state_history in evolved_results:
             sample_features=[]
             if self.include_multi_time and self.time_points is not None:
                 for time_index in self.time_points:
                     if time_index < len(state_history):
                            #getting state at this time point
                            state = state_history[time_index]
                            
                            sample_features.extend(self.measure_single_state(state, num_qubits))
                     else:
                          # This time point doesn't exist in the history
                         print(f"DEBUG: Time point {time_index} exceeds history length {len(state_history)}")
                         padding_size=len(self.readout_indices)
                         if self.include_zz:
                             padding_size += len(self.correlation_pairs)
                        
                         sample_features.extend([0]*padding_size)
             else:
                    final_state = state_history[-1] #last state in the history
                    sample_features.extend(self.measure_single_state(final_state, num_qubits))
                    print(f"DEBUG: Sample feature vector has length {len(sample_features)}")
             features.append(sample_features)
         features_array = np.array(features)
         print(f"DEBUG: Final features array shape: {features_array.shape}")
         return features_array
      
    
             
    
