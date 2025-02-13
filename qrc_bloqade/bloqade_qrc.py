import numpy as np
import bloqade
from bloqade import spin_operators, detuning, rydberg_interaction
from qrc import QRC, Encoder, Hamiltonian, Readout
from sklearn.metrics import mean_squared_error

class Bloqade(QRC):
    def __init__(self, n_sites,encoder, readout, predictor, scalar, splitter):
        super().__init__(n_sites, scalar, readout)
        self.hamiltonian= hamiltonian
        self.encoder = encoder
        self.predictor = predictor
        self.splitter = splitter
        self.scaler = scalar

    def run(self, X,y, test_size=0.4, random_state=402):
           #normalize the data
           X_norm, y_norm = self.scaler.fit_transform(X),self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
           #split data
           X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split(X_norm, y_norm, test_size, random_state)

           #encode
           encode_train= self.encoder.encode(X_train)
           encode_test = self.encoder.encode(X_test)

           #evolve
           evolved_train = self.hamiltonian.apply_dynamics(encode_train)
           evolved_test= self.hamiltonian.apply_dynamics(encode_test)

           #measure
           train_features = self.readout.measure(evolved_train)
           test_features = self.readout.measure(evolved_test)

              #predict
           self.predictor.fit(train_features, y_train)
           predictions= self.predictor.predict(test_features)

              #inverse transform
           scalar_y= MinMaxScaler(feature_range=(0,1))
           scalar_y.fit(y.reshape(-1, 1))
           predictions = scalar_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
           y_test = scalar_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

           #evaluate
           mse= mean_squared_error(y_test, predictions)
           return mse