import numpy as np
from scipy.sparse.linalg import expm_multiply,expm
from scipy.sparse import identity, csr_matrix


class BloqadeSolver:

    def with_initial_state(self, state_vector, num_qubits):
       
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
       state = init_state
        
        # Use expm_multiply for sparse matrix evolution
       if isinstance(hamiltonian, csr_matrix):
            for _ in range(steps):
                # evolve state using the sparse matrix exponential without forming full matrix
                state = expm_multiply(-1j * dt * hamiltonian, state)
       else:
            # Fall back to dense matrix evolution if Hamiltonian is not sparse
            U = expm(-1j * hamiltonian * dt)
            for _ in range(steps):
                state = U @ state
       return state
    
    def simulate(self, init_state, hamiltonian, duration, steps, n_qubits, return_all_steps=False):
     """
    Simulate quantum evolution with option to return all intermediate states.
    
    Args:
        init_state: Initial quantum state vector
        hamiltonian: Hamiltonian operator as sparse or dense matrix
        duration: Total evolution time
        steps: Number of time steps
        n_qubits: Number of qubits
        return_all_steps: If True, returns states at all time steps
                          If False, returns only the final state
     """
     new_state = self.with_initial_state(init_state, n_qubits)
     dt = duration / steps
    
     if return_all_steps:
        # Initialize list to store states at each time step
        all_states = [new_state.copy()]  # Include initial state
        
        # Use expm_multiply for sparse matrix evolution
        if isinstance(hamiltonian, csr_matrix):
            state = new_state
            for step in range(steps):
                # Evolve state using the sparse matrix exponential
                state = expm_multiply(-1j * dt * hamiltonian, state)
                all_states.append(state.copy())
        else:
            # Fall back to dense matrix evolution
            U = expm(-1j * hamiltonian * dt)
            state = new_state
            for step in range(steps):
                state = U @ state
                all_states.append(state.copy())
        
        return all_states
     else:
        # Original behavior: return only final state
        state = new_state
        
        if isinstance(hamiltonian, csr_matrix):
            for _ in range(steps):
                state = expm_multiply(-1j * dt * hamiltonian, state)
        else:
            U = expm(-1j * hamiltonian * dt)
            for _ in range(steps):
                state = U @ state
        
        return state
    
    def simulate_with_history(self, init_state, hamiltonian, duration, steps, n_qubits, return_history=True):
      return self.simulate(
        init_state,
        hamiltonian=hamiltonian,
        duration=duration,
        steps=steps,
        n_qubits=n_qubits,
        return_all_steps=True
    )
