import bloqade
import numpy as np
from bloqade.emulate.ir import emulator

class BloqadeSolver:
    """
    A solver that uses the bloqade library to perform the simulation.
    """

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

    def simulate(self, init_state, hamiltonian, duration, steps):
        """
        Runs the Bloqade emulator with a given Hamiltonian and initial state.
        
        Args:
        - init_state (list or np.ndarray): Initial quantum state.
        - hamiltonian: The Bloqade Hamiltonian.
        - duration (float): Total simulation time.
        - steps (int): Number of time steps.

        Returns:
        - Emulator result from Bloqade.
        """
        # Calculate time step
        dt = duration / steps
        
        # Create a time array for simulation
        times = np.linspace(0, duration, steps)
        
        # Ensure the initial state is properly formatted
        num_qubits = int(np.log2(len(init_state)))  # Calculate qubit count from state size
        formatted_state = self.with_initial_state(init_state, num_qubits)
        
        # Run the Bloqade emulator (assuming it accepts the initial state directly)
        result = emulator.run(hamiltonian, initial_state=formatted_state)
        
        return result



