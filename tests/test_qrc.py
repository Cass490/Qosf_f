# qrc_bloqade/tests/test_bloqade_qrc.py
import numpy as np
from bloqade_qrc import BloqadeQRC
from hamiltonians import RydbergHamiltonian
from encodings import AngleEncoding
from readouts import ZReadout
from utils import get_predictor
from sklearn.preprocessing import MinMaxScaler
from qrc_core import DataSplitter
from sklearn.datasets import make_regression
import pytest


# Test BloqadeQRC instantiation and basic methods
def test_bloqade_qrc_init():
    n_sites = 4
    omega = 1.0
    V = 1.0
    hamiltonian = RydbergHamiltonian(n_sites, omega, V)
    encoder = AngleEncoding(n_sites)
    readout = ZReadout(n_sites)
    predictor = get_predictor()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    splitter = DataSplitter()
    qrc = BloqadeQRC(n_sites, hamiltonian, encoder, readout, predictor, scaler, splitter)

    assert qrc.n_sites == n_sites
    assert qrc.hamiltonian == hamiltonian
    assert qrc.encoder == encoder
    assert qrc.readout == readout
    assert qrc.predictor == predictor
    assert qrc.scaler == scaler


#Basic tests for encoding, dynamics and measurements.

def test_bloqade_qrc_encode():
    n_sites = 4
    omega = 1.0
    V = 1.0
    hamiltonian = RydbergHamiltonian(n_sites, omega, V)
    encoder = AngleEncoding(n_sites)
    readout = ZReadout(n_sites)
    predictor = get_predictor()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    splitter = DataSplitter()
    qrc = BloqadeQRC(n_sites, hamiltonian, encoder, readout, predictor, scaler, splitter)
    data = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    encoded_states = qrc.encode(data)
    assert len(encoded_states) == 2  # Check for correct number of encoded states
    # Further checks depend on the specific encoding scheme.
    # We'll add detailed assertions in later tests

def test_bloqade_qrc_apply_dynamics():
    n_sites = 4
    omega = 1.0
    V = 1.0
    hamiltonian = RydbergHamiltonian(n_sites, omega, V)
    encoder = AngleEncoding(n_sites)
    readout = ZReadout(n_sites)
    predictor = get_predictor()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    splitter = DataSplitter()
    qrc = BloqadeQRC(n_sites, hamiltonian, encoder, readout, predictor, scaler, splitter)
    data = np.array([[0.1, 0.2, 0.3, 0.4]])
    encoded_states = qrc.encode(data)
    results = qrc.apply_dynamics(encoded_states, time=1.0)
    assert len(results) == 1  # Check number of returned results.

def test_bloqade_qrc_measure():
    n_sites = 4
    omega = 1.0
    V = 1.0
    hamiltonian = RydbergHamiltonian(n_sites, omega, V)
    encoder = AngleEncoding(n_sites)
    readout = ZReadout(n_sites)
    predictor = get_predictor()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    splitter = DataSplitter()
    qrc = BloqadeQRC(n_sites, hamiltonian, encoder, readout, predictor, scaler, splitter)

    data = np.array([[0.1, 0.2, 0.3, 0.4]])
    encoded_states = qrc.encode(data)
    results = qrc.apply_dynamics(encoded_states, time=1.0)
    features = qrc.measure(results)
    assert features.shape == (1, n_sites) # default, measure all sites
    features = qrc.measure(results, readout_indices=[0,2])
    assert features.shape == (1,2)


# Example of an end-to-end test with dummy data (replace with more comprehensive tests)
def test_bloqade_qrc_run():
    n_sites = 4
    omega = 1.0
    V = 1.0
    hamiltonian = RydbergHamiltonian(n_sites, omega, V)
    encoder = AngleEncoding(n_sites)
    readout = ZReadout(n_sites)
    predictor = get_predictor()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    splitter = DataSplitter()
    qrc = BloqadeQRC(n_sites, hamiltonian, encoder, readout, predictor, scaler, splitter)


    X, y = make_regression(n_samples=10, n_features=n_sites, noise=0.1, random_state=402)
    mse = qrc.run(X, y, test_size=0.4, random_state=402)
    assert isinstance(mse, float)  # Check if the output is a float (MSE)
    assert mse >= 0 # MSE should be non-negative


# Add more tests for different encoding schemes, Hamiltonians, readout strategies.
# Add tests for edge cases, invalid input, etc.