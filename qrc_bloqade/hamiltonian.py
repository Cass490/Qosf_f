import numpy as np
import bloqade
from bloqade.factory import rydberg_h, constant, linear
from bloqade.ir.location import AtomArrangement
from bloqade.ir.location import Chain
from braket.devices import LocalSimulator
from bloqade.submission.ir.braket import BraketTaskSpecification
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
        # Create 1D chain of atoms
        self.positions = [[0, i] for i in range(self.n_sites)]  # Define positions HERE
    

    
    def get_hamiltonian(self):
        # Create waveforms for amplitude, phase, and detuning
        arrangement = Chain(self.n_sites, spacing=5.0)
        amplitude_waveform = constant(1.0, self.omega / 2)
        phase_waveform = constant(1.0, 0.0)
        global_detuning = constant(1.0, 0.0)
        
        # Build the program using the chain arrangement
        # Build the program using the proper atom arrangement
        program = rydberg_h(
            atoms_positions=arrangement,
            amplitude=amplitude_waveform,
            phase=phase_waveform,
            detuning=global_detuning
        )
        
        # Add site-specific detuning if needed
        if not np.all(self.detuning_values == 0):
            detuning_field = self.detuning_values / 2
            program = program.detuning_field(detuning_field)
        
        return program
    
    
    def apply_dynamics(self, encoded_states, time=1.0):
        # Get the Hamiltonian
         # Get the Hamiltonian
        hamiltonian = self.get_hamiltonian()
        tlist = np.linspace(0, time, 100)
        
        # Simulate for each initial state
        results = []
        for init_state in encoded_states:
            try:
                # Create simulator task
                simulator = LocalSimulator("braket_ahs")
                
                # Prepare the program with the initial state
                program_with_state = hamiltonian.with_initial_state(init_state)
                
                # Run the simulation
                task = simulator.run(
                    program_with_state,
                    shots=1,  # For wavefunction simulation
                    times=tlist
                )
                
                # Get results
                result = task.result()
                results.append(result)
                
            except Exception as e:
                print(f"Simulation error: {e}")
        return results












































































































































































































































































































