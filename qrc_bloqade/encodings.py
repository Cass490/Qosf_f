import numpy as np

class angle_encode:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        ''' Each classical value is encoded into a quantum state using rotation angles:
    |ψ⟩ = cos(πx/2)|0⟩ + i*sin(πx/2)|1⟩'''

    def encode(self, data):
      states=[]
      for datapoint in data:
        # Start with the state of the first qubit
            initial_state = np.array([np.cos(datapoint[0] * np.pi / 2) - 1j * np.sin(datapoint[0] * np.pi / 2),
                                     np.sin(datapoint[0] * np.pi / 2) + 1j * np.cos(datapoint[0] * np.pi / 2)])
            #1j:python way of representing complex numbers
            # Combine with the states of the remaining qubits using the Kronecker product
            for i in range(1, self.n_sites):
                theta = datapoint[i] * np.pi / 2
                state_i = np.array([np.cos(theta) - 1j * np.sin(theta),
                                    np.sin(theta) + 1j * np.cos(theta)])
                initial_state = np.kron(initial_state, state_i) #kronecker product

            states.append(initial_state)
        
      return states

class amp_encode:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.max_features = 2**n_sites
    def encode(self, data):
      states=[]
      for datapoint in data:
        n_features = len(datapoint)
        if n_features > self.max_features:
                raise ValueError(
                    f"Amplitude encoding requires at most {self.max_features} features, "
                    f"got {n_features}"
                )

            # Check for normalization.  Use np.isclose for floating-point comparison.
        if not np.isclose(np.sum(datapoint**2), 1.0):
                raise ValueError("Input data for AmplitudeEncoding must be normalized.")
        padded_datapoint = np.pad(datapoint, (0, self.max_features - n_features))
        states.append(padded_datapoint)
      return states