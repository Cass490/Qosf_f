import numpy as np 
import bloqade

def apply_rydberg(encoded_states, hamiltonian,  solver, time=1.0):
    results=[]
    for init_state in encoded_states:
        result = solver.simulate(
            init_state,
            hamiltonian=hamiltonian,
            duration=time,
            steps=100
        )
        results.append(result)
    return results