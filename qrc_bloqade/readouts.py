import bloqade
import numpy as np


class ZReadout:
    def __init__(self, n_sites, readout_indices=None):
        self.n_sites = n_sites
        if readout_indices is not None:
           self.readout_indices = readout_indices
        else:
           self.readout_indices = list(range(self.n_sites))
    def generate_readouts(self, n_sites):
       #Generates single-qubit Z measurements as readout operators.
        readouts = []
        for i in range(n_sites):
            readouts.append(bloqade.ir.pauli_z(i))
        return readouts 

    def measure(self, evolved_results, readout_indices=None):
       # Measure the expectation value of each spin in the Z-basis.

        # Use provided indices or default to all
        if readout_indices is not None:
            self.readout_indices = readout_indices #update with the new indices

        features = []
        for result in evolved_results:
            expectation_values = []
            for i in self.readout_indices:  # Only measure specified qubits
                expectation_values.append(bloqade.expectation(result.states[-1],  bloqade.ir.pauli_z(i)).real)
            features.append(expectation_values)

        return np.array(features)