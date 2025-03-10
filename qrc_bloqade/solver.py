# qrc_bloqade/solvers.py
import bloqade

class BloqadeSolver:
    """
    A solver that uses the bloqade library to perform the simulation.
    """

    def simulate(self, init_state, hamiltonian, duration, steps):
        """
        Simulates the time evolution of a quantum system using Bloqade.

        Args:
            init_state: The initial state of the system.
            hamiltonian: The Hamiltonian governing the system's evolution.
            duration: The total simulation time.
            steps: The number of time steps.

        Returns:
            A bloqade.Result object containing the simulation results.
        """

        return bloqade.ir.simulation.simulate(init_state, hamiltonian=hamiltonian, duration=duration, steps=steps)