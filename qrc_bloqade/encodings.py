import numpy as np

class angle_encode:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        ''' Each classical value is encoded into a quantum state using rotation angles:
    |ψ⟩ = cos(πx/2)|0⟩ + sin(πx/2)|1⟩'''

    def encode(self, data):
      states=[]
      for datapoint in data:
            if len(datapoint) != self.n_sites:
              raise ValueError(f"Expected {self.n_sites} features, got {len(datapoint)}")
        # Start with the state of the first qubit
            initial_state = np.array([np.cos(datapoint[0] * np.pi / 2), 
                                 np.sin(datapoint[0] * np.pi / 2)])
            #1j:python way of representing complex numbers
            # Combine with the states of the remaining qubits using the Kronecker product
            for i in range(1, self.n_sites):
                theta = datapoint[i] * np.pi / 2
                state_i = np.array([np.cos(theta), np.sin(theta)])
                initial_state = np.kron(initial_state, state_i)#kronecker product

            if len(initial_state) != 2**self.n_sites:
               raise ValueError(f"Encoded state has incorrect size: {len(initial_state)} (expected {2**self.n_sites})")
            
            norm = np.sum(np.abs(initial_state)**2)
            if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-5):
              initial_state = initial_state / np.sqrt(norm)
            states.append(initial_state)
        
      return np.array(states), None

class amp_encode:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.max_features = 2**n_sites
    def encode(self, data):
      states=[]
      for datapoint in data:
        n_features = len(datapoint)
        if n_features > self.max_features:
            raise ValueError(f"Amplitude encoding requires at most {self.max_features} features, got {n_features}")

        if n_features > self.max_features:
                raise ValueError(
                    f"Amplitude encoding requires at most {self.max_features} features, "
                    f"got {n_features}"
                )

            # Check for normalization.  Use np.isclose for floating-point comparison.
        if not np.isclose(np.sum(datapoint**2), 1.0):
                raise ValueError("Input data for AmplitudeEncoding must be normalized.")
        padded_datapoint = np.pad(datapoint, (0, self.max_features - n_features))
        if len(padded_datapoint) != self.max_features:
            raise ValueError(f"Encoded state has incorrect size: {len(padded_datapoint)} (expected {self.max_features})")
        states.append(padded_datapoint)
      return np.array(states), None

class global_encode:
    def __init__(self, n_sites, encoding_type='hamiltonian', base_omega=1.2 , base_V= 1.2, detuning_scale=0.5 , param_scale=0.5):
        self.n_sites = n_sites
        self.encoding_type = encoding_type
        self.base_omega = base_omega
        self.base_V = base_V
        self.detuning_scale = detuning_scale
        self.param_scale = param_scale
        self.max_features = 2**n_sites
    
    def encode(self, data):
        n_data= data.shape[0]

        initial_states = np.zeros((n_data, self.max_features), dtype=complex)
        initial_states[:, 0] = 1.0

        hamiltonian_params=[]

        if self.encoding_type== 'hamiltonian':
            for  sample in data:
                n_features= len(sample)
                if n_features >=2: #baseconditions
                    omega = self.base_omega + self.param_scale*sample[0]
                    V= self.base_V + self.param_scale*sample[1]
                else:
                    omega = self.base_omega
                    V = self.base_V
                detuning_values= np.zeros(self.n_sites)
                for i in range(min(n_features-2, self.n_sites)):
                    if i+2< n_features:
                      detuning_values[i] = self.detuning_scale*sample[i+2]

                hamiltonian_params.append({
                    'omega': omega,
                    'V': V,
                    'detuning_values': detuning_values})
        elif self.encoding_type =='detuning':
            for sample in data:
                detuning_values= np.zeros(self.n_sites
                )
                for i in range(min(len(sample), self.n_sites)):
                    detuning_values[i] = self.detuning_scale*sample[i]
                hamiltonian_params.append({
                    'omega': self.base_omega,
                    'V': self.base_V,
                    'detuning_values': detuning_values})
        elif self.encoding_type =='hybrid': #initial state encoding+hamiltonian parameter encoding
            for i, sample in enumerate(data):
                n_features =len(sample)
                state_features= min(n_features// 2, self.n_sites)
                for j in range (state_features):
                    theta =np.pi*sample[j]
                    qubit_state =np.array ( [np.cos(theta), np.sin(theta)], dtype=complex)
                    
                    if j==0:
                        state= qubit_state
                    else:
                        state= np.kron(state, qubit_state)
                if len(state) < self.max_features:
                    padded_state= np.zeros(self.max_features, dtype=complex)
                    padded_state[:len(state)] = state   
                    state= padded_state
                initial_states[i] = state
            remaining_idx=state_features
            omega=self.base_omega
            V=self.base_V   
            if remaining_idx < n_features:
                    omega = self.base_omega + self.param_scale * sample[remaining_idx]
                    remaining_idx += 1
                
            if remaining_idx < n_features:
                    V = self.base_V + self.param_scale * sample[remaining_idx]
                    remaining_idx += 1
                
                # Create detuning values from any remaining features
            detuning_values = np.zeros(self.n_sites)
            for j in range(min(n_features - remaining_idx, self.n_sites)):
                    if remaining_idx + j < n_features:
                        detuning_values[j] = self.detuning_scale * sample[remaining_idx + j]
                
            hamiltonian_params.append({
                    'omega': omega,
                    'V': V,
                    'detuning_values': detuning_values
                })
        else:
             raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        return initial_states, hamiltonian_params
    
import numpy as np

class position_encode:
    def __init__(self, n_sites, dimension=1, base_V=1.0, encoding_scale=0.5, 
                 lattice_type='chain', periodic_boundary=False):
        """
        Position encoding for quantum reservoir computing with Rydberg atoms.
        
        This encoding modulates the Rydberg interaction strengths between atoms
        according to the data features: V_il = V_0 * (1 + λ * x_i)
        
        Args:
            n_sites: Number of qubits in the system
            dimension: Spatial dimension of the lattice (1 for chain, 2 for grid)
            base_V: Base interaction strength V_0
            encoding_scale: Scaling factor λ for encoding
            lattice_type: 'chain' for 1D, 'grid' for 2D, or 'custom'
            periodic_boundary: Whether to use periodic boundary conditions
        """
        self.n_sites = n_sites
        self.dimension = dimension
        self.base_V = base_V
        self.encoding_scale = encoding_scale
        self.lattice_type = lattice_type
        self.periodic_boundary = periodic_boundary
        
        #  maximum number of features we can encode
        if dimension == 1:
            # In 1D, can encode Nq-1 features (nearest neighbors)
            self.max_features = n_sites - 1
        elif dimension == 2:
            # In 2D, depends on lattice structure, but using row encoding
            # We'll assume a square lattice for simplicity
            self.grid_size = int(np.ceil(np.sqrt(n_sites)))
            self.max_features = n_sites - self.grid_size + (self.grid_size - 1)
        else:
            raise ValueError("Dimension must be 1 or 2")
        
    def get_neighbor_pairs(self):
        
        neighbor_pairs = []
        
        if self.dimension == 1:
            # 1D chain
            for i in range(self.n_sites - 1):
                neighbor_pairs.append((i, i + 1))
            
            # Add periodic boundary if needed
            if self.periodic_boundary:
                neighbor_pairs.append((self.n_sites - 1, 0))
                
        elif self.dimension == 2:
            # 2D grid
            grid_size = self.grid_size
            
            for row in range(grid_size):
                for col in range(grid_size):
                    site = row * grid_size + col
                    
                    # Skip if beyond number of sites
                    if site >= self.n_sites:
                        continue
                    
                    # Right neighbor
                    if col < grid_size - 1 and site + 1 < self.n_sites:
                        neighbor_pairs.append((site, site + 1))
                    elif col == grid_size - 1 and self.periodic_boundary:
                        right_site = row * grid_size
                        if right_site < self.n_sites:
                            neighbor_pairs.append((site, right_site))
                    
                    # Down neighbor
                    if row < grid_size - 1 and site + grid_size < self.n_sites:
                        neighbor_pairs.append((site, site + grid_size))
                    elif row == grid_size - 1 and self.periodic_boundary:
                        down_site = col
                        if down_site < self.n_sites:
                            neighbor_pairs.append((site, down_site))
        
        return neighbor_pairs
    
    def encode(self, data):
        #interaction_maps: List of dictionaries mapping qubit pairs to interaction strengths
        
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        # Check if we have enough features
        if n_features > self.max_features:
            print(f"Warning: Can only encode {self.max_features} features with {self.n_sites} qubits. Extra features will be ignored.")
        
        # Get nearest neighbor pairs
        neighbor_pairs = self.get_neighbor_pairs()
        n_pairs = len(neighbor_pairs)                                                                                      
        
        interaction_maps = []
        
        for i in range(n_samples):
            sample = data[i]
            interaction_map = {}
            
            # Assign features to neighbor pairs
            for j, (site1, site2) in enumerate(neighbor_pairs):
                if j < min(n_features, n_pairs):
                    # Modulate interaction strength based on feature value
                    # V_il = V_0 * (1 + λ * x_i)
                    interaction_strength = self.base_V * (1 + self.encoding_scale * sample[j])
                else:
                    # Use base interaction strength for remaining pairs
                    interaction_strength = self.base_V
                
                # Store in the interaction map (both directions)
                interaction_map[(site1, site2)] = interaction_strength
                interaction_map[(site2, site1)] = interaction_strength
            
            interaction_maps.append(interaction_map)
        
        return interaction_maps
    
    def get_hamiltonian_params(self, interaction_maps, base_omega=1.0, detuning=None):
        #Convert interaction maps to Hamiltonian parameters.
     
        n_samples = len(interaction_maps)
        hamiltonian_params = []
        
        for i in range(n_samples):
            interaction_map = interaction_maps[i]
            
            # Create parameter dictionary
            params = {
                'omega': base_omega,
                'interaction_map': interaction_map,
            }
            
            # Add detuning if provided
            if detuning is not None:
                if i < len(detuning):
                    params['detuning_values'] = detuning[i]
                else:
                    params['detuning_values'] = np.zeros(self.n_sites)
            else:
                params['detuning_values'] = np.zeros(self.n_sites)
            
            hamiltonian_params.append(params)
        
        return hamiltonian_params
class  local_encode:
        
        def __init__(self, n_sites, dimension=1, base_omega=1.0, encoding_scale=0.5):
            self.n_sites = n_sites
            self.dimension = dimension
            self.base_omega = base_omega
            self.encoding_scale = encoding_scale
            self.max_features = 2**n_sites

        def encode(self, data):
            n_samples = data.shape[0]
            n_features = data.shape[1]
            initial_states =[]
            hamil_params=[]
            for i  in range (n_samples):
                #creating ground state for the sample
                state = np.zeros(self.max_features, dtype=complex)
                state[0] = 1.0
                initial_states.append(state)

                sample= data[i]
                omega_values= np.ones(self.n_sites) * self.base_omega #for rabi freq
                detuning_values= np.zeros(self.n_sites)

                # First half of features modify omega values (one per site)
                for j in range(min(n_features, self.n_sites)):
                    #modify rabi freq aka omega based on the features
                    omega_values[j] = self.base_omega *(1+self.encoding_scale * sample[j])  #mulplicative scaling

                remaining_features= n_features - self.n_sites
                for j in range (min(remaining_features, self.n_sites)):
                    if j+ self.n_sites < n_features:
                        detuning_values[j] = self.encoding_scale * sample[j+ self.n_sites]
                params={
                    'omega': omega_values,
                    'detuning_values': detuning_values,
                     'V':1.0 #global interaction strength
                }
                hamil_params.append(params)
            return np.array(initial_states), hamil_params
        
        def get_hamiltonian_params(self, omega_maps, base_V=1.0, detuning=None):
        # Convert interaction maps to Hamiltonian parameters.
          n_samples = len(omega_maps)
          hamiltonian_params = []
        
          for i in range(n_samples):
            omega_map = omega_maps[i]
            
            # Create parameter dictionary
            params = {
                'V': base_V,
                'omega': omega_map
            }
            
            # Add detuning if provided
            if detuning is not None:
                if i < len(detuning):
                    params['detuning_values'] = detuning[i]
                else:
                    params['detuning_values'] = np.zeros(self.n_sites)
            else:
                params['detuning_values'] = np.zeros(self.n_sites)
            
            hamiltonian_params.append(params)
        
          return hamiltonian_params

        
            