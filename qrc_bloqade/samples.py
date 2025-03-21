import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def logistic_map_dataset(n_samples=300, n_features=4, r=3.9, x0=0.1, random_state=402):
    """
    Generate dataset from logistic map.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        r: Logistic map parameter (3.5-4.0 for chaotic behavior)
        x0: Initial value (0-1)
        random_state: Random seed
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Generate raw time series (with extra points for features)
    total_points = n_samples + n_features
    series = np.zeros(total_points)
    series[0] = x0
    
    for i in range(1, total_points):
        series[i] = r * series[i-1] * (1 - series[i-1])
    
    # Create features (windows of n_features) and targets
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        X[i] = series[i:i+n_features]
        y[i] = series[i+n_features]
    
    return X, y

def lorenz_dataset(n_samples=300, n_features=4, sigma=10, rho=28, beta=8/3, dt=0.01, random_state=402):
    """
    Generate dataset from Lorenz attractor.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        sigma, rho, beta: Lorenz system parameters
        dt: Time step
        random_state: Random seed
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Initial conditions
    x0, y0, z0 = np.random.rand(3)
    
    # Generate Lorenz attractor
    total_points = n_samples + n_features + 1000  # Extra points for burn-in
    xs = np.zeros(total_points)
    ys = np.zeros(total_points)
    zs = np.zeros(total_points)
    
    xs[0], ys[0], zs[0] = x0, y0, z0
    
    for i in range(1, total_points):
        dx = sigma * (ys[i-1] - xs[i-1]) * dt
        dy = (xs[i-1] * (rho - zs[i-1]) - ys[i-1]) * dt
        dz = (xs[i-1] * ys[i-1] - beta * zs[i-1]) * dt
        
        xs[i] = xs[i-1] + dx
        ys[i] = ys[i-1] + dy
        zs[i] = zs[i-1] + dz
    
    # Discard burn-in
    xs = xs[1000:]
    ys = ys[1000:]
    zs = zs[1000:]
    
    # Create features and targets
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Use x-component as features
        X[i] = xs[i:i+n_features]
        # Target is the next x value
        y[i] = xs[i+n_features]
    
    return X, y

def mackey_glass_dataset(n_samples=300, n_features=4, tau=17, dt=0.1, random_state=402):
    """
    Generate dataset from Mackey-Glass time series.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        tau: Time delay
        dt: Time step
        random_state: Random seed
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Initialize history
    history_length = int(tau/dt)
    history = np.zeros(history_length)
    history[0] = 1.2
    
    # Generate time series
    total_points = n_samples + n_features + 1000  # Extra points for burn-in
    series = np.zeros(total_points)
    series[0] = 1.2
    
    for i in range(1, total_points):
        # Find the delayed value
        delay_idx = max(0, i - int(tau/dt))
        delayed_value = series[delay_idx]
        
        # Mackey-Glass equation
        deriv = 0.2 * delayed_value / (1 + delayed_value**10) - 0.1 * series[i-1]
        series[i] = series[i-1] + deriv * dt
    
    # Discard burn-in
    series = series[1000:]
    
    # Create features and targets
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        X[i] = series[i:i+n_features]
        y[i] = series[i+n_features]
    
    return X, y

def polynomial_dataset(n_samples=300, n_features=4, degree=3, noise=0.1, random_state=402):
    """
    Generate dataset with polynomial relationship.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        degree: Polynomial degree
        noise: Noise level
        random_state: Random seed
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.rand(n_samples, n_features) * 2 - 1  # Range [-1, 1]
    
    # Generate target with polynomial relationship
    y = np.zeros(n_samples)
    for i in range(n_samples):
        y[i] = sum(X[i, j]**degree for j in range(n_features)) / n_features
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    return X, y

def sinusoidal_dataset(n_samples=300, n_features=4, frequencies=[1, 2, 3, 5], noise=0.1, random_state=402):
    """
    Generate dataset with sinusoidal relationships.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        frequencies: List of frequencies for each feature
        noise: Noise level
        random_state: Random seed
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.rand(n_samples, n_features) * 2 * np.pi  # Range [0, 2π]
    
    # Generate target with sinusoidal relationship
    y = np.zeros(n_samples)
    for i in range(n_samples):
        for j in range(min(n_features, len(frequencies))):
            y[i] += np.sin(frequencies[j] * X[i, j])
    
    y /= min(n_features, len(frequencies))  # Normalize
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    return X, y

def visualize_dataset(X, y, dataset_name):
    """
    Visualize dataset with various plots.
    
    Args:
        X: Features array
        y: Target array
        dataset_name: Name of the dataset for plot titles
    """
    plt.figure(figsize=(16, 12))
    
    # Plot 1: First two features
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Target')
    plt.title(f'{dataset_name}: Feature 1 vs Feature 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 2: Target vs Feature 1
    plt.subplot(2, 2, 2)
    plt.scatter(X[:, 0], y, alpha=0.7)
    plt.title(f'{dataset_name}: Target vs Feature 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    
    # Plot 3: Target distribution
    plt.subplot(2, 2, 3)
    plt.hist(y, bins=30, alpha=0.7)
    plt.title(f'{dataset_name}: Target Distribution')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    
    # Plot 4: 3D visualization if possible
    if X.shape[1] >= 3:
        ax = plt.subplot(2, 2, 4, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Target')
        ax.set_title(f'{dataset_name}: 3D Visualization')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
    else:
        plt.subplot(2, 2, 4)
        for i in range(min(X.shape[1], 4)):
            plt.scatter(X[:, i], y, alpha=0.3, label=f'Feature {i+1}')
        plt.title(f'{dataset_name}: Features vs Target')
        plt.xlabel('Feature Value')
        plt.ylabel('Target')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_visualization.png')
    plt.show()

def analyze_chaos(dataset_name, generator_func, **kwargs):
    """
    Analyze the chaotic properties of a dataset.
    
    Args:
        dataset_name: Name of the dataset
        generator_func: Function to generate the dataset
        **kwargs: Parameters for the generator function
    """
    X, y = generator_func(**kwargs)
    
    # Visualize the dataset
    visualize_dataset(X, y, dataset_name)
    
    # Calculate basic statistics
    print(f"\n=== {dataset_name} Dataset Analysis ===")
    print(f"Shape: {X.shape}, {y.shape}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            corr_matrix[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
    
    print("\nFeature Correlation Matrix:")
    print(corr_matrix)
    
    # Calculate feature-target correlations
    feature_target_corr = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        feature_target_corr[i] = np.corrcoef(X[:, i], y)[0, 1]
    
    print("\nFeature-Target Correlations:")
    print(feature_target_corr)
    
    # Additional chaos analysis if time series
    if dataset_name in ["Logistic Map", "Lorenz Attractor", "Mackey-Glass"]:
        # Plot autocorrelation
        plt.figure(figsize=(12, 6))
        
        # Autocorrelation of feature 1
        plt.subplot(1, 2, 1)
        autocorr = np.correlate(X[:, 0], X[:, 0], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]  # Normalize
        plt.plot(autocorr[:50])
        plt.title(f'{dataset_name}: Autocorrelation of Feature 1')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        
        # Autocorrelation of target
        plt.subplot(1, 2, 2)
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]  # Normalize
        plt.plot(autocorr[:50])
        plt.title(f'{dataset_name}: Autocorrelation of Target')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        
        plt.tight_layout()
        #plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_autocorrelation.png')
        plt.show()
    
    return X, y

