import numpy as np
import bloqade
from bloqade import spin_operators, detuning, rydberg_interaction

class rydberg:
    def __init__(self, n_sites, omega, V, detuning_values=None):
        self.n_sites = n_sites
        self.omega = omega
        self.V = V #interaction parameters
        if detuning_values is None:
            self.detuning_values = np.zeros(n_sites)
        else:
            if len(detuning_values) != n_sites:
                raise ValueError(f"Length of detuning_values ({len(detuning_values)}) "
                               f"must match n_sites ({n_sites})")
            self.detuning_values = np.array(detuning_values)
        self.hamiltonian= self.get_hamiltonian()
    
    def get_hamiltonian(self,*args, **kwargs):
        def create_rydberg_hamiltonian(pos, omega, V, alpha):
            
        # Qubit positions (linear chain)
            pos = [[0, i] for i in range(self.n_sites)]
            # Global Rabi term
            H = bloqade.sum(
                [spin_operators.sigma_x(i) for i in range(len(pos))],
                lambda i: omega / 2,
            )
            # Site-specific/individual atom detuning term
            H += bloqade.sum(
                [spin_operators.sigma_z(i) for i in range(len(pos))],
                lambda i: self.detuning_values[i] / 2,
            )
            # Nearest-neighbor interaction term
            H += rydberg_interaction(pos, strength=self.V, onsite=False)
            return H
            
            def apply_dynamics(self, encoded_states, time=1.0): #simulate system time evolution
                results=[]
                for init_state in encoded_states:
                  result = bloqade.simulate(
                  init_state,
                  hamiltonian=self.hamiltonian,
                  duration=time,
                  steps=100
            )
                  results.append(result)
            
                return results








































































































































































































































































































































































































