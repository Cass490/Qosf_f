import abc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class QRC(abc.ABC):
   def __init__(self, n_sites, encoder, hamiltonian, readout, predictor, scaler):        
       self.n_sites = n_sites
       self.encoder = encoder
       self.hamiltonian = hamiltonian
       self.readout = readout
       self.predictor = predictor
       self.scaler = scaler
   @abc.abstractmethod
   def run(self,X,ytest_size=0.4,random_state=402):
      pass
class Encoder(abc.ABC):
   @abc.abstractmethod
   def encode(self,data, *args, **kwargs):
      pass
class Hamiltonian(abc.ABC):
   @abc.abstractmethod
   def get_hamiltonian(self, *args, **kwargs):
      pass
class Readout(abc.ABC):
   @abc.abstractmethod
   def measure(self, evolved_state,readout_indices=None,*args, **kwargs):
      pass
class Predictor(abc.ABC):
   @abc.abstractmethod
   def predict(self, train_data, *args, **kwargs):
      pass
   @abc.abstractmethod
   def predict(self, data):
        pass

class DataScalar(abc.ABC):
   @abc.abstractmethod
   def fit_transform(self, data):
      pass
   @abc.abstractmethod
   def transform(self, data):
      pass
   @abc.abstractmethod
   def inverse_transform(self, data):
      pass

class DataSplitter(): 
    def split(self, X, y, test_size, random_state):
        # First, split into 60% train and 40% temp (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
              X, y, test_size=test_size, shuffle=True, random_state=random_state
        )
        # Next, split the remaining 40% equally into test (20%) and validation (20%)
        X_test, X_val, y_test, y_val = train_test_split(
              X_temp, y_temp, test_size=0.5, shuffle=True, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test