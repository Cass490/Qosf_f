from scipy.linalg import expm
import numpy as np


class BloqadeSolver:

    def with_initial_state(self, state_vector, num_qubits):
        """
        Custom function to create an initial state vector.
        
        Args:
        - state_vector (list or np.ndarray): The initial quantum state.
        - num_qubits (int): Number of qubits in the system.

        Returns:
        - np.ndarray: The normalized initial state vector.
        """
        # Ensure the state is a NumPy array
        state_vector = np.array(state_vector, dtype=complex)

        # Check if the provided state vector has the correct length
        expected_size = 2 ** num_qubits
        if state_vector.size != expected_size:
            raise ValueError(f"State vector must have {expected_size} elements.")

        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm == 0:
            raise ValueError("State vector cannot be all zeros.")
        state_vector /= norm

        return state_vector
 
    def evolve_state( self, init_state, hamiltonian, duration ,steps):
       
       dt = duration / steps
       U= expm(-1j * hamiltonian * dt)
       state= init_state
       for _ in range(steps):
            state = U @ state
       return state

    def simulate(self, init_state, hamiltonian, duration, steps, n_qubits):
      
        new_state= self.with_initial_state( init_state, n_qubits)
        final_state= self.evolve_state(new_state, hamiltonian, duration, steps)
        return final_state


