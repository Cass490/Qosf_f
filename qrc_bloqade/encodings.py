import numpy as np

class angle_encode:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        ''' Each classical value is encoded into a quantum state using rotation angles:
    |ψ⟩ = cos(πx/2)|0⟩ + i*sin(πx/2)|1⟩'''

    def encode(self, data):
      states=[]
      for datapoint in data:
        initial_state=np.zeros( self.n_sites, dtype=complex)
        for i in range (self.n_sites):
            theta=datapoint[i]*np.pi/2
            initial_state[i] = np.cos(theta) + 1j*np.sin(theta)
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
        padded_datapoint = np.pad(datapoint, (0, 2**self.n_sites - len(datapoint)))
        states.append(padded_datapoint)
      return states